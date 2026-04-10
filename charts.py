"""
Interactive Plotly charts for the Perps Board.
Each function returns a plotly Figure; call save_all_charts() to write HTML files.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CHART_DIR


def _mkdir() -> None:
    os.makedirs(CHART_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candle_colors(df: pd.DataFrame):
    return ["#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df["close"], df["open"])]


# ── 1. Price + Technical Indicators ───────────────────────────────────────────

def plot_price_ta(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Four-panel chart:
      Row 1 — Candlestick + EMAs + Bollinger Bands
      Row 2 — RSI (14)
      Row 3 — MACD histogram + lines
      Row 4 — Volume bars
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.18, 0.17],
        vertical_spacing=0.025,
        subplot_titles=["Price · EMAs · Bollinger Bands", "RSI (14)",
                        "MACD (12/26/9)", "Volume"],
    )

    # ── Row 1 — Candlestick ───────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    ema_styles = {
        9:   ("#FFD700", 1.2),
        21:  ("#FF8C00", 1.2),
        50:  ("#00BFFF", 1.4),
        200: ("#EE82EE", 1.6),
    }
    for period, (color, width) in ema_styles.items():
        col_name = f"ema_{period}"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=f"EMA {period}",
                line=dict(color=color, width=width),
            ), row=1, col=1)

    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"], name="BB Upper",
            line=dict(color="rgba(120,120,255,0.7)", width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"], name="BB Lower",
            line=dict(color="rgba(120,120,255,0.7)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(120,120,255,0.07)",
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_mid"], name="BB Mid",
            line=dict(color="rgba(120,120,255,0.45)", width=1),
            showlegend=False,
        ), row=1, col=1)

    if "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["vwap"], name="VWAP",
            line=dict(color="#FF69B4", width=1.2, dash="dash"),
        ), row=1, col=1)

    # ── Row 2 — RSI ───────────────────────────────────────────────────────────
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"], name="RSI",
            line=dict(color="#FF6B35", width=1.5),
        ), row=2, col=1)
        for level, color in [(70, "rgba(239,83,80,0.4)"), (30, "rgba(38,166,154,0.4)"),
                              (50, "rgba(255,255,255,0.15)")]:
            fig.add_hline(y=level, line_dash="dash", line_color=color,
                          line_width=0.8, row=2, col=1)

    # ── Row 3 — MACD ─────────────────────────────────────────────────────────
    if "macd_hist" in df.columns:
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["macd_hist"], name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd"], name="MACD",
            line=dict(color="#2196F3", width=1.5),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_signal"], name="Signal",
            line=dict(color="#FF9800", width=1.5),
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                      line_width=0.8, row=3, col=1)

    # ── Row 4 — Volume ────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=_candle_colors(df),
        showlegend=False,
    ), row=4, col=1)

    fig.update_layout(
        title=f"{symbol} — Price & Technical Indicators",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=950,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font_size=11),
        margin=dict(l=60, r=20, t=70, b=40),
    )
    fig.update_yaxes(row=2, col=1, range=[0, 100])
    return fig


# ── 2. Open Interest ──────────────────────────────────────────────────────────

def plot_open_interest(df_oi: pd.DataFrame, df_price: pd.DataFrame,
                       symbol: str) -> go.Figure:
    """Price vs Open Interest with 30-period MA and period-over-period % change."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.45, 0.20],
        vertical_spacing=0.03,
        subplot_titles=["Futures Price", "Open Interest (USDT)", "OI Change %"],
    )

    fig.add_trace(go.Scatter(
        x=df_price.index, y=df_price["close"], name="Price",
        line=dict(color="#26a69a", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_oi.index, y=df_oi["sumOpenInterestValue"],
        name="OI (USDT)", line=dict(color="#FF9800", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.08)",
    ), row=2, col=1)

    if "oi_ma30" in df_oi.columns:
        fig.add_trace(go.Scatter(
            x=df_oi.index, y=df_oi["oi_ma30"], name="OI MA30",
            line=dict(color="#FF5722", width=1.2, dash="dot"),
        ), row=2, col=1)

    if "oi_change_pct" in df_oi.columns:
        oi_chg = df_oi["oi_change_pct"]
        chg_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in oi_chg.fillna(0)]
        fig.add_trace(go.Bar(
            x=df_oi.index, y=oi_chg, name="OI Δ%",
            marker_color=chg_colors, showlegend=False,
        ), row=3, col=1)

    fig.update_layout(
        title=f"{symbol} — Open Interest",
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1),
    )
    return fig


# ── 3. Funding Rate ───────────────────────────────────────────────────────────

def plot_funding_rate(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Bar chart of every historical funding-rate payment."""
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["fundingRate"]]
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df.index,
        y=df["fundingRate"] * 100,
        name="Funding Rate (%)",
        marker_color=colors,
    ))

    # 7-day rolling average overlay
    rolling_avg = df["fundingRate"].rolling(21).mean() * 100
    fig.add_trace(go.Scatter(
        x=df.index, y=rolling_avg, name="7d Rolling Avg",
        line=dict(color="#FF9800", width=2),
    ))

    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.3)",
                  line_width=0.8)
    fig.add_hline(y=0.01, line_dash="dash", line_color="rgba(239,83,80,0.4)",
                  line_width=0.8, annotation_text="0.01% (neutral+)", annotation_position="right")
    fig.add_hline(y=-0.01, line_dash="dash", line_color="rgba(38,166,154,0.4)",
                  line_width=0.8, annotation_text="-0.01% (neutral−)", annotation_position="right")

    fig.update_layout(
        title=f"{symbol} — Funding Rate History (every 8 h)",
        yaxis_title="Rate (%)",
        template="plotly_dark",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1),
    )
    return fig


# ── 4. Long/Short Ratios ──────────────────────────────────────────────────────

def plot_ls_ratios(
    df_global: pd.DataFrame,
    df_top_acc: pd.DataFrame,
    df_top_pos: pd.DataFrame,
    df_taker: pd.DataFrame,
    df_price: pd.DataFrame,
    symbol: str,
) -> go.Figure:
    """
    Three-panel chart:
      Row 1 — Futures price
      Row 2 — Global / top-trader account / position L/S ratios
      Row 3 — Taker buy/sell volume ratio (aggressor-side)
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.30, 0.40, 0.30],
        vertical_spacing=0.04,
        subplot_titles=[
            "Futures Price",
            "Long/Short Ratios",
            "Taker Buy/Sell Volume Ratio",
        ],
    )

    fig.add_trace(go.Scatter(
        x=df_price.index, y=df_price["close"], name="Price",
        line=dict(color="#26a69a", width=1.5),
    ), row=1, col=1)

    ls_datasets = [
        (df_global,  "Global L/S (retail)",    "#FF9800"),
        (df_top_acc, "Top Traders (account)", "#2196F3"),
        (df_top_pos, "Top Traders (position)", "#9C27B0"),
    ]
    for df_ls, name, color in ls_datasets:
        if not df_ls.empty:
            fig.add_trace(go.Scatter(
                x=df_ls.index, y=df_ls["longShortRatio"], name=name,
                line=dict(color=color, width=1.5),
            ), row=2, col=1)

    fig.add_hline(y=1, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  line_width=0.8, row=2, col=1)

    if not df_taker.empty:
        taker_colors = ["#26a69a" if v >= 1 else "#ef5350"
                        for v in df_taker["buySellRatio"]]
        fig.add_trace(go.Bar(
            x=df_taker.index, y=df_taker["buySellRatio"],
            name="Taker B/S Ratio", marker_color=taker_colors,
            showlegend=False,
        ), row=3, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                      line_width=0.8, row=3, col=1)

    fig.update_layout(
        title=f"{symbol} — Long/Short Ratios",
        template="plotly_dark",
        height=750,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1),
    )
    return fig


# ── 5. Spot Money Flow & CVD ──────────────────────────────────────────────────

def plot_spot_flow(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Three-panel chart:
      Row 1 — Spot price
      Row 2 — Cumulative Volume Delta (CVD)
      Row 3 — Per-candle net money flow (USDT) + 24-period rolling sum
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.30, 0.35, 0.35],
        vertical_spacing=0.04,
        subplot_titles=[
            "Spot Price",
            "Cumulative Volume Delta (CVD)",
            "Net Money Flow per Candle (USDT)",
        ],
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="Spot Price",
        line=dict(color="#26a69a", width=1.5),
    ), row=1, col=1)

    if "cvd" in df.columns:
        cvd_color = "#26a69a" if df["cvd"].iloc[-1] >= 0 else "#ef5350"
        cvd_fill  = "rgba(38,166,154,0.1)" if cvd_color == "#26a69a" else "rgba(239,83,80,0.1)"
        fig.add_trace(go.Scatter(
            x=df.index, y=df["cvd"], name="CVD",
            line=dict(color=cvd_color, width=1.5),
            fill="tozeroy", fillcolor=cvd_fill,
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                      line_width=0.8, row=2, col=1)

    if "net_flow_usdt" in df.columns:
        nf_colors = ["#26a69a" if v >= 0 else "#ef5350"
                     for v in df["net_flow_usdt"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["net_flow_usdt"], name="Net Flow",
            marker_color=nf_colors, showlegend=False,
        ), row=3, col=1)

        if "net_flow_24p" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["net_flow_24p"], name="24-period Net Flow",
                line=dict(color="#2196F3", width=1.8),
            ), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                      line_width=0.8, row=3, col=1)

    fig.update_layout(
        title=f"{symbol} — Spot Money Flow & CVD",
        template="plotly_dark",
        height=750,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1),
    )
    return fig


# ── Save all charts to disk ───────────────────────────────────────────────────

def save_all_charts(
    symbol: str,
    fig_price_ta: go.Figure,
    fig_oi: go.Figure,
    fig_funding: go.Figure,
    fig_ls: go.Figure,
    fig_flow: go.Figure,
) -> list[str]:
    """Write all figures as self-contained HTML files; return the list of paths."""
    _mkdir()
    mapping = {
        "1_price_ta":       fig_price_ta,
        "2_open_interest":  fig_oi,
        "3_funding_rate":   fig_funding,
        "4_ls_ratios":      fig_ls,
        "5_spot_flow":      fig_flow,
    }
    paths = []
    for name, fig in mapping.items():
        path = os.path.join(CHART_DIR, f"{symbol}_{name}.html")
        fig.write_html(path, include_plotlyjs="cdn")
        paths.append(path)
    return paths
