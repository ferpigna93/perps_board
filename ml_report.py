#!/usr/bin/env python3
"""
ML Probability Signal — standalone report generator.

Trains two calibrated XGBoost models on extended historical data and writes
an HTML report with:
  1. ML signal chart  — price + P(up)/P(dn) series + high-confidence markers
  2. Feature importance — which inputs drive each model's decisions

Usage:
    python ml_report.py [SYMBOL] [OPTIONS]

Examples:
    python ml_report.py ZECUSDT
    python ml_report.py BTCUSDT --window 12 --threshold 3 --candles 6000
    python ml_report.py ETHUSDT --signal-threshold 0.65 --open
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser

from rich import box
from rich.console import Console
from rich.table import Table

import binance_client as bc
import charts as ch
import config
import ml_signal as ml

console = Console()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ML Probability Signal — standalone report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("symbol", nargs="?", default=config.SYMBOL,
                   help=f"Binance symbol, e.g. BTCUSDT (default: {config.SYMBOL})")
    p.add_argument("--window", type=int, default=ml.DEFAULT_WINDOW_H,
                   help="Forecast horizon in hours (default: %(default)s)")
    p.add_argument("--threshold", type=float, default=None,
                   help=(
                       "Target move %% for the main signal labels. "
                       f"Defaults to --dist-max when --dist is used, "
                       f"otherwise {ml.DEFAULT_THRESHOLD_PCT}%%"
                   ))
    p.add_argument("--candles", type=int, default=ml.DEFAULT_N_CANDLES_1H,
                   help="1h candles of training history (default: %(default)s)")
    p.add_argument("--signal-threshold", type=float, default=0.75,
                   help="Probability level above which markers appear on chart "
                        "(default: %(default)s)")
    p.add_argument("--top-features", type=int, default=20,
                   help="Features to show in importance chart (default: %(default)s)")

    dist = p.add_argument_group("Probability distribution (--dist)")
    dist.add_argument("--dist", action="store_true",
                      help="Compute P(reach level) across a range of threshold levels")
    dist.add_argument("--dist-min", type=float, default=1.0,
                      help="Minimum threshold %% for distribution sweep (default: %(default)s)")
    dist.add_argument("--dist-max", type=float, default=8.0,
                      help="Maximum threshold %% for distribution sweep (default: %(default)s)")
    dist.add_argument("--dist-points", type=int, default=8,
                      help="Number of threshold levels to evaluate (default: %(default)s)")

    p.add_argument("--open", action="store_true",
                   help="Open HTML report in the default browser when done")
    return p.parse_args()


# ── Terminal output ───────────────────────────────────────────────────────────

def print_signal_table(current: dict, metrics: dict,
                        window_h: int, threshold_pct: float) -> None:
    t = Table(
        title=f"ML Signal — P(±{threshold_pct}% in {window_h}h)",
        box=box.ROUNDED, header_style="bold magenta",
    )
    t.add_column("Metric", style="bold", min_width=30)
    t.add_column("Value",  justify="right", min_width=14)

    def _prob(v: float) -> str:
        if v >= 0.60:
            return f"[bold red]{v:.1%}[/bold red]"
        if v >= 0.40:
            return f"[yellow]{v:.1%}[/yellow]"
        return f"[dim]{v:.1%}[/dim]"

    t.add_row(f"P(+{threshold_pct}% in {window_h}h)", _prob(current["p_up"]))
    t.add_row(f"P(−{threshold_pct}% in {window_h}h)", _prob(current["p_dn"]))
    bias = current.get("bias", 0.0)
    bias_str = (f"[green]{bias:+.1f}[/green]" if bias >= 0
                else f"[red]{bias:+.1f}[/red]")
    t.add_row("Directional bias  100×(P(up)−P(dn))", bias_str)
    t.add_row("As-of timestamp",        str(current["timestamp"]))
    if metrics.get("auc_up") is not None:
        t.add_row("AUC-ROC (up model)",    f"{metrics['auc_up']:.3f}")
    if metrics.get("auc_dn") is not None:
        t.add_row("AUC-ROC (down model)",  f"{metrics['auc_dn']:.3f}")
    t.add_row("Brier score (up)",       f"{metrics.get('brier_up', '—')}")
    t.add_row("Brier score (down)",     f"{metrics.get('brier_dn', '—')}")
    t.add_row("Training samples",       str(metrics.get("n_train", "—")))
    t.add_row("Test samples",           str(metrics.get("n_test",  "—")))
    t.add_row("Positive events — up %", f"{metrics.get('pos_up_pct', '—')}%")
    t.add_row("Positive events — dn %", f"{metrics.get('pos_dn_pct', '—')}%")
    console.print(t)


def print_distribution_table(dist_df, current_price: float, window_h: int) -> None:
    t = Table(
        title=f"Probability Distribution — window {window_h}h",
        box=box.ROUNDED, header_style="bold cyan",
    )
    t.add_column("Threshold", justify="right", min_width=10)
    t.add_column("P(up)",     justify="right", min_width=10)
    t.add_column("Price (+)", justify="right", min_width=12)
    t.add_column("P(dn)",     justify="right", min_width=10)
    t.add_column("Price (−)", justify="right", min_width=12)

    def _prob(v: float) -> str:
        if v >= 0.60:
            return f"[bold yellow]{v:.1%}[/bold yellow]"
        if v >= 0.40:
            return f"[white]{v:.1%}[/white]"
        return f"[dim]{v:.1%}[/dim]"

    for _, row in dist_df.iterrows():
        thr   = row["threshold"]
        p_up  = row["p_up"]
        p_dn  = row["p_dn"]
        up_px = current_price * (1 + thr / 100)
        dn_px = current_price * (1 - thr / 100)
        t.add_row(
            f"±{thr:.2f}%",
            _prob(p_up),
            f"[green]{up_px:,.4f}[/green]",
            _prob(p_dn),
            f"[red]{dn_px:,.4f}[/red]",
        )
    console.print(t)


def print_top_features(importances: dict, top_n: int = 10) -> None:
    for direction, color in [("up", "green"), ("dn", "red")]:
        imp = importances.get(direction, {})
        top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
        t = Table(
            title=f"Top {top_n} features — [{color}]{direction}[/{color}] model",
            box=box.SIMPLE, header_style=f"bold {color}",
        )
        t.add_column("Feature", style="bold", min_width=28)
        t.add_column("Importance (gain)", justify="right", min_width=18)
        for feat, score in top:
            t.add_row(feat, f"{score:.5f}")
        console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    symbol = args.symbol.upper().replace(".P", "")

    # Resolve threshold: when --dist is active and --threshold was not set
    # explicitly, align the main signal with --dist-max so both use the same
    # data regime (avoids "0 positive examples" errors on low-volatility assets).
    if args.threshold is None:
        args.threshold = args.dist_max if args.dist else ml.DEFAULT_THRESHOLD_PCT

    console.rule(f"[bold cyan] ML REPORT — {symbol} [/bold cyan]")
    console.print(
        f"\n[cyan]Window:[/cyan] [bold white]{args.window}h[/bold white]  "
        f"[cyan]Threshold:[/cyan] [bold white]±{args.threshold}%[/bold white]  "
        f"[cyan]Candles:[/cyan] [bold white]{args.candles}[/bold white]\n"
    )

    # 1 ── Fetch supporting data ────────────────────────────────────────────────
    with console.status("[bold green]Fetching OI history …"):
        try:
            df_oi = bc.get_open_interest_history(symbol, period="1h", limit=500)
        except RuntimeError as exc:
            console.print(f"[yellow]WARN: OI fetch failed ({exc}) — continuing without OI features.[/yellow]")
            df_oi = None

    with console.status("[bold green]Fetching funding rate history …"):
        try:
            df_funding = bc.get_funding_rate_history(symbol, limit=100)
        except RuntimeError as exc:
            console.print(f"[yellow]WARN: Funding fetch failed ({exc}) — continuing without funding features.[/yellow]")
            df_funding = None

    # 2 ── Run ML pipeline ──────────────────────────────────────────────────────
    with console.status(
        f"[bold green]Fetching {args.candles} 1h candles + 4h data, "
        "building features and training models …"
    ):
        try:
            result = ml.run_ml_pipeline(
                symbol        = symbol,
                window_h      = args.window,
                threshold_pct = args.threshold,
                n_candles_1h  = args.candles,
                df_oi         = df_oi,
                df_funding    = df_funding,
            )
        except Exception as exc:
            console.print(f"\n[bold red]ERROR:[/bold red] {exc}")
            sys.exit(1)

    # 3 ── Terminal output ──────────────────────────────────────────────────────
    console.print()
    print_signal_table(result["current"], result["metrics"],
                       args.window, args.threshold)
    console.print()
    print_top_features(result["signal"].feature_importance(), top_n=10)

    # 4 ── Probability distribution (optional) ────────────────────────────────
    dist_result = None
    if args.dist:
        console.print(
            f"\n[cyan]Distribution:[/cyan] [bold white]{args.dist_points} levels, "
            f"{args.dist_min}%–{args.dist_max}%[/bold white]\n"
        )
        with console.status(
            f"[bold green]Sweeping {args.dist_points} threshold levels "
            f"({args.dist_min}%–{args.dist_max}%) — "
            f"building {args.dist_points * 2} models …"
        ):
            try:
                dist_result = ml.run_probability_distribution(
                    symbol       = symbol,
                    window_h     = args.window,
                    dist_min     = args.dist_min,
                    dist_max     = args.dist_max,
                    n_points     = args.dist_points,
                    n_candles_1h = args.candles,
                    df_oi        = df_oi,
                    df_funding   = df_funding,
                    df_1h        = result["df_1h"],
                    df_4h        = result["df_4h"],
                )
            except Exception as exc:
                console.print(f"[yellow]WARN: distribution sweep failed ({exc}) — skipping.[/yellow]")

        if dist_result is not None and not dist_result["dist_df"].empty:
            console.print()
            print_distribution_table(
                dist_result["dist_df"], dist_result["current_price"], args.window,
            )

    # 5 ── Generate HTML report ─────────────────────────────────────────────────
    with console.status("[bold green]Rendering charts …"):
        fig_signal = ch.plot_ml_signal(
            result["proba_df"], result["df_1h"], symbol,
            args.window, args.threshold, result["current"],
            signal_threshold=args.signal_threshold,
        )
        importances = result["signal"].feature_importance()
        fig_importance = ch.plot_ml_feature_importance(
            importances, symbol, args.window, args.threshold,
            top_n=args.top_features,
        )

        entries = [
            ("ml_signal",     "ML probability signal", fig_signal),
            ("ml_importance", "Feature importance",    fig_importance),
        ]
        if dist_result is not None and not dist_result["dist_df"].empty:
            fig_dist = ch.plot_probability_distribution(
                dist_result["dist_df"], symbol,
                args.window, dist_result["current_price"],
            )
            entries.append(("ml_distribution", "Probability distribution", fig_dist))

        os.makedirs(config.CHART_DIR, exist_ok=True)
        out_path = os.path.join(
            config.CHART_DIR,
            f"{symbol}_ml_report_w{args.window}_t{args.threshold}.html",
        )
        dist_note = (
            f" · Distribution: {args.dist_points} levels "
            f"{args.dist_min}%–{args.dist_max}%"
            if args.dist else ""
        )
        ch._write_combined_html(
            out_path,
            f"{symbol} — ML Signal Report  (±{args.threshold}% in {args.window}h)",
            entries,
            subtitle=(
                f"Trained on {result['metrics']['n_train']} samples · "
                f"AUC up={result['metrics'].get('auc_up', '—')}  "
                f"dn={result['metrics'].get('auc_dn', '—')}  · "
                f"Signal threshold: P > {args.signal_threshold:.0%}"
                f"{dist_note}"
            ),
        )

    console.print(f"\n[bold green]Report saved:[/bold green] "
                  f"[link=file://{out_path}]{out_path}[/link]")

    if args.open:
        webbrowser.open(f"file://{out_path}")

    console.rule("[bold cyan] Done [/bold cyan]")


if __name__ == "__main__":
    main()
