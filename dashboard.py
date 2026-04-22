#!/usr/bin/env python3
"""
Perps Board — Crypto Perpetual Futures Analysis Dashboard.

Usage:
    python dashboard.py [SYMBOL] [OPTIONS]

Examples:
    python dashboard.py                          # ZECUSDT, 1h, 500 candles
    python dashboard.py BTCUSDT --interval 4h
    python dashboard.py ETHUSDT --interval 1d --limit 365
    python dashboard.py SOLUSDT --no-charts      # terminal only, no HTML files
    python dashboard.py ZECUSDT --open           # save charts and open in browser
"""
from __future__ import annotations

import argparse
import sys
import webbrowser

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import binance_client as bc
import charts as ch
import config
import indicators as ind
import liquidations as liq
import market_metrics as mm
import ml_signal as ml

console = Console()


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crypto Perpetual Futures Analysis Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("symbol", nargs="?", default=config.SYMBOL,
                   help=f"Binance symbol, e.g. BTCUSDT (default: {config.SYMBOL})")
    p.add_argument("--interval", default=config.INTERVAL,
                   help="Candle interval: 1m 5m 15m 1h 4h 1d (default: %(default)s)")
    p.add_argument("--limit", type=int, default=config.LIMIT,
                   help="Number of candles, max 1500 (default: %(default)s)")
    p.add_argument("--oi-period", default=config.OI_PERIOD,
                   help="Period for OI / L/S data: 5m 15m 30m 1h 4h 6h 12h 1d "
                        "(default: %(default)s)")
    p.add_argument("--no-charts", action="store_true",
                   help="Skip HTML chart generation")
    p.add_argument("--open", action="store_true",
                   help="Open generated charts in the default browser")
    p.add_argument("--no-liq", action="store_true",
                   help="Skip liquidation heatmaps")
    p.add_argument("--ml", action="store_true",
                   help="Run ML probability signal (XGBoost + calibration)")
    p.add_argument("--ml-window", type=int, default=ml.DEFAULT_WINDOW_H,
                   help="Forecast horizon in hours (default: %(default)s)")
    p.add_argument("--ml-threshold", type=float, default=ml.DEFAULT_THRESHOLD_PCT,
                   help="Target move %% (default: %(default)s)")
    p.add_argument("--ml-candles", type=int, default=ml.DEFAULT_N_CANDLES_1H,
                   help="1h candles of history for ML training (default: %(default)s)")
    return p.parse_args()


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_all(symbol: str, interval: str, limit: int, oi_period: str) -> dict:
    """Fetch all required data from Binance public endpoints."""
    console.print(
        f"\n[cyan]Symbol:[/cyan] [bold white]{symbol}[/bold white]  "
        f"[cyan]Interval:[/cyan] [bold white]{interval}[/bold white]  "
        f"[cyan]Candles:[/cyan] [bold white]{limit}[/bold white]\n"
    )

    steps = [
        ("Futures OHLCV",           lambda: bc.get_futures_klines(symbol, interval, limit)),
        ("Funding rate history",    lambda: bc.get_funding_rate_history(symbol, limit=100)),
        ("Premium index",           lambda: bc.get_premium_index(symbol)),
        ("Open interest (current)", lambda: bc.get_open_interest_current(symbol)),
        ("Open interest (history)", lambda: bc.get_open_interest_history(
            symbol, period=oi_period, limit=min(limit, 500))),
        ("Global L/S ratio",        lambda: bc.get_global_ls_ratio(
            symbol, period=oi_period, limit=min(limit, 500))),
        ("Top-trader account L/S",  lambda: bc.get_top_trader_account_ratio(
            symbol, period=oi_period, limit=min(limit, 500))),
        ("Top-trader position L/S", lambda: bc.get_top_trader_position_ratio(
            symbol, period=oi_period, limit=min(limit, 500))),
        ("Taker buy/sell ratio",    lambda: bc.get_taker_ls_ratio(
            symbol, period=oi_period, limit=min(limit, 500))),
    ]

    results = []
    for label, fn in steps:
        with console.status(f"[bold green]Fetching {label} …"):
            results.append(fn())

    keys = ["df_fut", "df_funding", "premium", "oi_current",
            "df_oi", "df_global", "df_top_acc", "df_top_pos", "df_taker"]
    data = dict(zip(keys, results))

    # Spot klines are optional — futures-only symbols (e.g. RAVEUSDT) have no spot market
    with console.status("[bold green]Fetching Spot OHLCV …"):
        try:
            data["df_spot"] = bc.get_spot_klines(symbol, interval, limit)
        except RuntimeError:
            console.print(
                f"[yellow]INFO: No spot market for {symbol} — "
                "CVD / net money flow will be skipped.[/yellow]"
            )
            data["df_spot"] = None

    return data


# ── Terminal tables ───────────────────────────────────────────────────────────

def _yn(v: bool) -> str:
    return "[green]YES[/green]" if v else "[red]NO[/red]"


def _rate_color(v: float) -> str:
    if v < 0:
        return f"[green]{v:+.4f}%[/green]"
    if v > 0.03:
        return f"[red]{v:+.4f}%[/red]"
    return f"[yellow]{v:+.4f}%[/yellow]"


def _pct_color(v) -> str:
    if v is None:
        return "—"
    return f"[green]{v:+.2f}%[/green]" if v >= 0 else f"[red]{v:+.2f}%[/red]"


def _ratio_color(v) -> str:
    if v is None:
        return "—"
    return f"[green]{v:.4f}[/green]" if v > 1 else f"[red]{v:.4f}[/red]"


def _flow_color(v) -> str:
    if v is None:
        return "—"
    return f"[green]${v:+,.0f}[/green]" if v >= 0 else f"[red]${v:+,.0f}[/red]"


def print_ta_table(signals: dict, symbol: str, interval: str) -> None:
    t = Table(
        title=f"Technical Indicators — {symbol} ({interval})",
        box=box.ROUNDED, header_style="bold cyan",
        show_lines=False,
    )
    t.add_column("Indicator",   style="bold", min_width=20)
    t.add_column("Value",       justify="right", min_width=12)
    t.add_column("Signal",      justify="center", min_width=14)

    def rsi_sig(v):
        if v > 70: return "[red]OVERBOUGHT[/red]"
        if v < 30: return "[green]OVERSOLD[/green]"
        return "[yellow]NEUTRAL[/yellow]"

    price = signals["price"]
    t.add_row("Price",           f"{price:,.5f}",                        "—")
    t.add_row("Above EMA 9",     "",                                      _yn(signals["above_ema9"]))
    t.add_row("Above EMA 21",    "",                                      _yn(signals["above_ema21"]))
    t.add_row("Above EMA 50",    "",                                      _yn(signals["above_ema50"]))
    t.add_row("Above EMA 200",   "",                                      _yn(signals["above_ema200"]))
    t.add_row("RSI (14)",        f"{signals['rsi']:.2f}",                rsi_sig(signals["rsi"]))
    t.add_row("MACD Bull",       "",                                      _yn(signals["macd_bull"]))
    t.add_row("MACD Histogram",  f"{signals['macd_hist']:+.5f}",         "—")
    t.add_row("BB %B",           f"{signals['bb_pct_b']:.4f}",           "—")
    t.add_row("BB Width",        f"{signals['bb_width']:.4f}",           "—")
    t.add_row("ATR (14)",        f"{signals['atr']:.5f}",                "—")
    t.add_row(
        "ADX (14)",
        f"{signals['adx']:.2f}",
        "[green]TRENDING[/green]" if signals["adx"] > 25 else "[yellow]RANGING[/yellow]",
    )
    t.add_row("ADX Bias",        signals["adx_bias"].upper(),            "—")
    t.add_row("Stoch RSI K",     f"{signals['stoch_k']:.4f}",           "—")
    t.add_row("Stoch RSI D",     f"{signals['stoch_d']:.4f}",           "—")
    console.print(t)


def print_funding_table(fs: dict) -> None:
    t = Table(title="Funding Rate", box=box.ROUNDED, header_style="bold magenta")
    t.add_column("Metric", style="bold", min_width=22)
    t.add_column("Value", justify="right", min_width=16)

    t.add_row("Current Rate",         _rate_color(fs["current_rate_pct"]))
    t.add_row("Next Funding Time",     fs["next_funding_time"])
    t.add_row("Mark Price",            f"{fs['mark_price']:,.4f}")
    t.add_row("Index Price",           f"{fs['index_price']:,.4f}")
    t.add_row("Basis (mark − index)",  f"{fs['basis_pct']:+.4f}%")
    t.add_row("7-day Avg Rate",        _rate_color(fs["7d_avg_rate_pct"]))
    t.add_row("30-day Avg Rate",       _rate_color(fs["30d_avg_rate_pct"]))
    t.add_row("Annualised Rate",       f"{fs['annualized_rate_pct']:+.2f}%")
    t.add_row("% Positive Payments",   f"{fs['positive_pct']:.1f}%")
    t.add_row("Max Rate Observed",     _rate_color(fs["max_rate_pct"]))
    t.add_row("Min Rate Observed",     _rate_color(fs["min_rate_pct"]))
    console.print(t)


def print_oi_table(ois: dict) -> None:
    t = Table(title="Open Interest", box=box.ROUNDED, header_style="bold yellow")
    t.add_column("Metric", style="bold", min_width=22)
    t.add_column("Value", justify="right", min_width=16)

    t.add_row("Latest OI (USDT)",    f"${ois['latest_oi_usdt']:>14,.0f}")
    t.add_row("OI Mean (USDT)",      f"${ois['oi_mean_usdt']:>14,.0f}")
    t.add_row("OI Max  (USDT)",      f"${ois['oi_max_usdt']:>14,.0f}")
    t.add_row("OI Min  (USDT)",      f"${ois['oi_min_usdt']:>14,.0f}")
    t.add_row("24h OI Change",       _pct_color(ois["24h_oi_change_pct"]))
    t.add_row("7d  OI Change",       _pct_color(ois["7d_oi_change_pct"]))
    t.add_row("30d OI Change",       _pct_color(ois["30d_oi_change_pct"]))
    console.print(t)


def print_ls_table(lss: dict) -> None:
    t = Table(title="Long / Short Ratios", box=box.ROUNDED, header_style="bold blue")
    t.add_column("Metric", style="bold", min_width=26)
    t.add_column("Value", justify="right", min_width=14)

    t.add_row("Global L/S Ratio (retail)",     _ratio_color(lss["global_ls_ratio"]))
    t.add_row("  → Long  %",                   f"{lss['global_long_pct']:.2f}%"
                                                if lss["global_long_pct"] else "—")
    t.add_row("  → Short %",                   f"{lss['global_short_pct']:.2f}%"
                                                if lss["global_short_pct"] else "—")
    t.add_row("Top Traders — Account L/S",     _ratio_color(lss["top_acc_ls_ratio"]))
    t.add_row("Top Traders — Position L/S",    _ratio_color(lss["top_pos_ls_ratio"]))
    t.add_row("Taker Buy/Sell Volume Ratio",   _ratio_color(lss["taker_buy_sell_ratio"]))
    t.add_row("Taker Buy Volume  (24h)",        f"{lss['taker_buy_vol_24h']:,.2f}"
                                                if lss["taker_buy_vol_24h"] else "—")
    t.add_row("Taker Sell Volume (24h)",        f"{lss['taker_sell_vol_24h']:,.2f}"
                                                if lss["taker_sell_vol_24h"] else "—")
    t.add_row("Composite Bias",                f"[bold]{lss['bias_score']}[/bold]")
    console.print(t)


def print_flow_table(sfs: dict) -> None:
    t = Table(title="Spot Money Flow & CVD", box=box.ROUNDED, header_style="bold green")
    t.add_column("Metric", style="bold", min_width=26)
    t.add_column("Value", justify="right", min_width=16)

    t.add_row("Net Flow (last closed candle)", _flow_color(sfs["net_flow_last_candle_usdt"]))
    t.add_row("Net Flow (24h)",                _flow_color(sfs["net_flow_24h_usdt"]))
    t.add_row("Net Flow (7d)",                 _flow_color(sfs["net_flow_7d_usdt"]))
    t.add_row("CVD (current)",                 f"{sfs['cvd_current']:,.4f}")
    t.add_row("CVD (24h change)",              f"{sfs['cvd_24h_change']:+,.4f}"
                                                if sfs["cvd_24h_change"] else "—")
    t.add_row("CVD Trend",
              f"[green]{sfs['cvd_trend'].upper()}[/green]"
              if sfs["cvd_trend"] == "increasing"
              else f"[red]{sfs['cvd_trend'].upper()}[/red]")
    t.add_row("Taker Buy % (24h)",             f"{sfs['buy_pct_24h']:.2f}%"
                                                if sfs["buy_pct_24h"] else "—")
    console.print(t)


def print_ml_table(current: dict, metrics: dict,
                   window_h: int, threshold_pct: float) -> None:
    t = Table(
        title=f"ML Signal — P(±{threshold_pct}% in {window_h}h)",
        box=box.ROUNDED, header_style="bold magenta",
    )
    t.add_column("Metric", style="bold", min_width=28)
    t.add_column("Value", justify="right", min_width=14)

    def _prob(v: float) -> str:
        if v >= 0.60:
            return f"[bold red]{v:.1%}[/bold red]"
        if v >= 0.40:
            return f"[yellow]{v:.1%}[/yellow]"
        return f"[dim]{v:.1%}[/dim]"

    t.add_row(f"P(+{threshold_pct}% in {window_h}h)",  _prob(current["p_up"]))
    t.add_row(f"P(−{threshold_pct}% in {window_h}h)",  _prob(current["p_dn"]))
    bias = current.get("bias", 0.0)
    bias_str = (f"[green]{bias:+.3f}[/green]" if bias >= 0
                else f"[red]{bias:+.3f}[/red]")
    t.add_row("Directional bias  0.5×(P(up)−P(dn))",  bias_str)
    t.add_row("As-of timestamp",                        str(current["timestamp"]))
    if metrics.get("auc_up") is not None:
        t.add_row("AUC-ROC (up model)",    f"{metrics['auc_up']:.3f}")
    if metrics.get("auc_dn") is not None:
        t.add_row("AUC-ROC (down model)",  f"{metrics['auc_dn']:.3f}")
    t.add_row("Brier score (up)",          f"{metrics.get('brier_up', '—')}")
    t.add_row("Brier score (down)",        f"{metrics.get('brier_dn', '—')}")
    t.add_row("Training samples",          str(metrics.get("n_train", "—")))
    t.add_row("Test samples",              str(metrics.get("n_test",  "—")))
    t.add_row("Positive events — up %",    f"{metrics.get('pos_up_pct', '—')}%")
    t.add_row("Positive events — dn %",    f"{metrics.get('pos_dn_pct', '—')}%")
    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def print_liq_table(stats: dict) -> None:
    t = Table(title="Liquidations — inferred from OI drops", box=box.ROUNDED,
              header_style="bold red")
    t.add_column("Metric", style="bold", min_width=26)
    t.add_column("Value", justify="right", min_width=16)

    def _usdt(v):
        return f"${v:>14,.0f}" if v else "—"

    t.add_row("Total Liquidated (USDT)",  _usdt(stats.get("total_usdt")))
    t.add_row("  → Longs liquidated",     _usdt(stats.get("long_usdt")))
    t.add_row("  → Shorts liquidated",    _usdt(stats.get("short_usdt")))
    t.add_row("Long liq %",
              f"{stats['long_pct']:.1f}%" if stats.get("long_pct") is not None else "—")
    t.add_row("Number of events",         str(stats.get("n_events", 0)))
    t.add_row("Largest single event",     _usdt(stats.get("largest_event_usdt")))
    console.print(t)


def main() -> None:
    args   = parse_args()
    symbol = args.symbol.upper().replace(".P", "")   # strip TradingView suffix if present

    console.rule(f"[bold cyan] PERPS BOARD — {symbol} [/bold cyan]")

    # 1 ─ Fetch ------------------------------------------------------------------
    try:
        data = fetch_all(symbol, args.interval, args.limit, args.oi_period)
    except Exception as exc:
        console.print(f"\n[bold red]ERROR fetching data:[/bold red] {exc}")
        sys.exit(1)

    # 2 ─ Compute indicators -----------------------------------------------------
    with console.status("[bold green]Computing technical indicators …"):
        df_ta   = ind.add_all_indicators(data["df_fut"])
        signals = ind.latest_signal_summary(df_ta)

    # 3 ─ Compute market metrics -------------------------------------------------
    with console.status("[bold green]Computing market metrics …"):
        fs        = mm.funding_summary(data["df_funding"], data["premium"])
        df_oi_enr = mm.enrich_oi(data["df_oi"], data["df_fut"])
        ois       = mm.oi_summary(data["df_oi"])
        lss       = mm.ls_summary(data["df_global"], data["df_top_acc"],
                                  data["df_top_pos"], data["df_taker"])
        if data["df_spot"] is not None:
            df_flow = mm.add_spot_flow_metrics(data["df_spot"])
            sfs     = mm.spot_flow_summary(df_flow)
        else:
            df_flow = None
            sfs     = None

    # 4 ─ Liquidation data -------------------------------------------------------
    df_orders   = None
    liq_stats   = {}
    hist_hmap   = {}
    liq_snap    = None
    est_hmap    = {}

    if not args.no_liq:
        with console.status("[bold green]Inferring liquidations from OI + OHLCV …"):
            try:
                df_orders = liq.infer_liquidations_from_oi(
                    data["df_fut"], data["df_oi"]
                )
                liq_stats = liq.liq_stats(df_orders)
            except Exception as exc:
                console.print(f"[yellow]WARN: Liquidation inference failed: {exc}[/yellow]")
                df_orders = None

        with console.status("[bold green]Building liquidation heatmaps …"):
            oi_usdt = float(data["df_oi"]["sumOpenInterestValue"].iloc[-1])

            if df_orders is not None and not df_orders.empty:
                hist_hmap = liq.build_historical_heatmap(
                    df_orders, data["df_fut"], n_price_bins=120, time_bucket="4h"
                )

            liq_snap = liq.estimate_liq_map(
                data["df_fut"], oi_usdt, price_range=0.35, n_bins=300
            )
            est_hmap = liq.build_estimated_heatmap_over_time(
                data["df_fut"], data["df_oi"], price_range=0.30,
                n_price_bins=150, window=48,
            )

    # 5 ─ ML signal (optional) ---------------------------------------------------
    ml_result: dict | None = None
    if args.ml:
        with console.status(
            f"[bold green]Training ML signal "
            f"(window={args.ml_window}h, threshold={args.ml_threshold}%) …"
        ):
            try:
                ml_result = ml.run_ml_pipeline(
                    symbol        = symbol,
                    window_h      = args.ml_window,
                    threshold_pct = args.ml_threshold,
                    n_candles_1h  = args.ml_candles,
                    df_oi         = data["df_oi"],
                    df_funding    = data["df_funding"],
                )
            except Exception as exc:
                console.print(f"[yellow]WARN: ML pipeline failed: {exc}[/yellow]")
                ml_result = None

    # 6 ─ Print terminal dashboard -----------------------------------------------
    console.print()
    print_ta_table(signals, symbol, args.interval)
    console.print()
    print_funding_table(fs)
    console.print()
    print_oi_table(ois)
    console.print()
    print_ls_table(lss)
    if sfs is not None:
        console.print()
        print_flow_table(sfs)
    if liq_stats:
        console.print()
        print_liq_table(liq_stats)
    if ml_result is not None:
        console.print()
        print_ml_table(
            ml_result["current"], ml_result["metrics"],
            args.ml_window, args.ml_threshold,
        )

    # 7 ─ Generate charts --------------------------------------------------------
    if not args.no_charts:
        with console.status("[bold green]Generating interactive HTML charts …"):
            recent = df_ta.tail(200)
            fig_price_ta = ch.plot_price_ta(recent, symbol)
            fig_oi       = ch.plot_open_interest(df_oi_enr, data["df_fut"], symbol)
            fig_funding  = ch.plot_funding_rate(data["df_funding"], symbol)
            fig_ls       = ch.plot_ls_ratios(
                data["df_global"], data["df_top_acc"], data["df_top_pos"],
                data["df_taker"], data["df_fut"], symbol,
            )
            fig_flow = ch.plot_spot_flow(df_flow, symbol) if df_flow is not None else None

            fig_liq_hist = None
            fig_liq_est  = None
            if not args.no_liq:
                fig_liq_hist = ch.plot_liquidation_historical(hist_hmap, symbol)
                if liq_snap is not None and est_hmap:
                    fig_liq_est = ch.plot_liquidation_estimated(
                        liq_snap, est_hmap, symbol
                    )

            fig_ml = None
            if ml_result is not None:
                fig_ml = ch.plot_ml_signal(
                    ml_result["proba_df"], ml_result["df_1h"], symbol,
                    args.ml_window, args.ml_threshold, ml_result["current"],
                )

            paths = ch.save_all_charts(
                symbol, fig_price_ta, fig_oi, fig_funding, fig_ls, fig_flow,
                fig_liq_hist, fig_liq_est, fig_ml,
            )

        console.print("\n[bold green]Charts saved:[/bold green]")
        for path in paths:
            console.print(f"  [link=file://{path}]{path}[/link]")

        if args.open:
            for path in paths:
                webbrowser.open(f"file://{path}")

    console.rule("[bold cyan] Analysis complete [/bold cyan]")


if __name__ == "__main__":
    main()
