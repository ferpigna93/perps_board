"""
Interactive Plotly charts for the Perps Board.
Each function returns a plotly Figure; call save_all_charts() to write HTML files
(individual pages plus one scrollable page with every chart stacked).
"""
from __future__ import annotations

import os

import numpy as np
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


# ── 6. Historical Liquidation Heatmap ────────────────────────────────────────

def plot_liquidation_historical(heatmap_data: dict, symbol: str) -> go.Figure:
    """
    2-D heatmap of actual forced-liquidation orders (from Binance allForceOrders).

    Top panel    — price line
    Middle panel — long  liquidation volume heatmap (SELL orders = longs liquidated)
    Bottom panel — short liquidation volume heatmap (BUY  orders = shorts liquidated)

    Color intensity = USD value of liquidations in that (time, price) cell.
    Hot zones indicate where cascading liquidations have historically clustered.
    """
    if not heatmap_data:
        fig = go.Figure()
        fig.update_layout(title=f"{symbol} — Historical Liquidations (no data)",
                          template="plotly_dark")
        return fig

    x  = heatmap_data["x_labels"]
    y  = heatmap_data["y_prices"]
    pl = heatmap_data["price_line"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.25, 0.375, 0.375],
        vertical_spacing=0.03,
        subplot_titles=[
            "Price",
            "Long Liquidations (SELL — price fell through level)",
            "Short Liquidations (BUY — price rose through level)",
        ],
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=x, y=pl, name="Price",
        line=dict(color="#ffffff", width=1.5),
    ), row=1, col=1)

    # Long liquidations heatmap (reds — danger when price falls)
    z_long = heatmap_data["z_long"]
    z_long_log = np.log1p(z_long)
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=z_long_log,
        name="Long Liq",
        colorscale=[
            [0.0,  "rgba(0,0,0,0)"],
            [0.15, "rgba(255,200,0,0.3)"],
            [0.5,  "rgba(255,100,0,0.7)"],
            [1.0,  "rgba(255,0,0,1.0)"],
        ],
        showscale=False,
        hovertemplate="Time: %{x}<br>Price: %{y:.4f}<br>Liq Vol (log): %{z:.2f}<extra>Longs</extra>",
    ), row=2, col=1)
    # Price overlay on long heatmap
    fig.add_trace(go.Scatter(
        x=x, y=pl, name="Price", line=dict(color="white", width=1),
        showlegend=False,
    ), row=2, col=1)

    # Short liquidations heatmap (blues — danger when price rises)
    z_short = heatmap_data["z_short"]
    z_short_log = np.log1p(z_short)
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=z_short_log,
        name="Short Liq",
        colorscale=[
            [0.0,  "rgba(0,0,0,0)"],
            [0.15, "rgba(0,200,255,0.3)"],
            [0.5,  "rgba(0,100,255,0.7)"],
            [1.0,  "rgba(100,0,255,1.0)"],
        ],
        showscale=False,
        hovertemplate="Time: %{x}<br>Price: %{y:.4f}<br>Liq Vol (log): %{z:.2f}<extra>Shorts</extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=pl, name="Price", line=dict(color="white", width=1),
        showlegend=False,
    ), row=3, col=1)

    fig.update_layout(
        title=f"{symbol} — Historical Liquidation Heatmap",
        template="plotly_dark",
        height=850,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=70, r=20, t=70, b=40),
    )
    return fig


# ── 7. Estimated Future Liquidation Heatmap ───────────────────────────────────

def plot_liquidation_estimated(
    liq_snapshot: pd.DataFrame,
    heatmap_data: dict,
    symbol: str,
) -> go.Figure:
    """
    Two-panel chart estimating where future liquidation cascades could occur.

    Left-side panel (snapshot):
      Horizontal bar chart centred on the current price.
      • Orange/red bars below current price → long liquidations if price falls here
      • Blue bars above current price       → short liquidations if price rises here

    Right-side panel (time-evolution heatmap):
      Rolling estimated liquidation density (price × time).
      Shows how the liquidation landscape has shifted as positions were built up.
      White line = actual price.
    """
    if liq_snapshot.empty or not heatmap_data:
        fig = go.Figure()
        fig.update_layout(title=f"{symbol} — Estimated Liquidations (no data)",
                          template="plotly_dark")
        return fig

    current_price = float(heatmap_data["current_price"])

    # ── Left: snapshot bar chart ─────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.30, 0.70],
        subplot_titles=[
            f"Liquidation Snapshot @ {current_price:,.4f}",
            "Rolling Estimated Liquidation Density (price × time)",
        ],
        horizontal_spacing=0.04,
    )

    snap = liq_snapshot.copy()
    longs_mask  = snap["price"] <= current_price
    shorts_mask = snap["price"] >  current_price

    # Long liq bars (below current price)
    fig.add_trace(go.Bar(
        x=snap.loc[longs_mask, "liq_long"],
        y=snap.loc[longs_mask, "price"],
        orientation="h",
        name="Long Liq (price falls)",
        marker_color="rgba(255,100,30,0.75)",
        hovertemplate="Price: %{y:.4f}<br>Liq Vol: $%{x:,.0f}<extra>Longs</extra>",
    ), row=1, col=1)

    # Short liq bars (above current price)
    fig.add_trace(go.Bar(
        x=snap.loc[shorts_mask, "liq_short"],
        y=snap.loc[shorts_mask, "price"],
        orientation="h",
        name="Short Liq (price rises)",
        marker_color="rgba(30,130,255,0.75)",
        hovertemplate="Price: %{y:.4f}<br>Liq Vol: $%{x:,.0f}<extra>Shorts</extra>",
    ), row=1, col=1)

    # Current price horizontal line
    fig.add_hline(
        y=current_price, line_dash="dash",
        line_color="rgba(255,255,255,0.8)", line_width=1.5,
        annotation_text=f" {current_price:,.4f}",
        annotation_position="right",
        row=1, col=1,
    )

    # ── Right: rolling heatmap ────────────────────────────────────────────────
    x_lbls  = heatmap_data["x_labels"]
    y_prc   = heatmap_data["y_prices"]
    z_mat   = heatmap_data["z"]
    z_log   = np.log1p(z_mat)
    pl      = heatmap_data["price_line"]

    fig.add_trace(go.Heatmap(
        x=x_lbls, y=y_prc, z=z_log,
        colorscale=[
            [0.0,  "rgba(0,0,30,0)"],
            [0.10, "rgba(0,0,80,0.4)"],
            [0.35, "rgba(255,200,0,0.6)"],
            [0.65, "rgba(255,80,0,0.85)"],
            [1.0,  "rgba(255,0,0,1.0)"],
        ],
        showscale=True,
        colorbar=dict(title="Liq Density<br>(log scale)", thickness=12, len=0.8),
        hovertemplate="Time: %{x}<br>Price: %{y:.4f}<br>Density: %{z:.2f}<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x_lbls, y=pl, name="Price",
        line=dict(color="white", width=1.5),
        showlegend=False,
    ), row=1, col=2)

    # Mark current price on the heatmap too
    fig.add_hline(
        y=current_price, line_dash="dash",
        line_color="rgba(255,255,255,0.5)", line_width=1,
        row=1, col=2,
    )

    fig.update_layout(
        title=f"{symbol} — Estimated Future Liquidation Heatmap",
        template="plotly_dark",
        height=750,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=70, r=20, t=70, b=40),
    )
    fig.update_xaxes(title_text="Estimated Liq Volume (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    return fig


# ── 8. ML Probability Signal ─────────────────────────────────────────────────

def plot_ml_signal(
    proba_df: pd.DataFrame,
    df_price: pd.DataFrame,
    symbol: str,
    window_h: int,
    threshold_pct: float,
    current: dict,
) -> go.Figure:
    """
    Two-panel chart:
      Row 1 — Futures price (trimmed to the proba_df date range)
      Row 2 — Calibrated P(up) and P(dn) probability series

    Parameters
    ----------
    proba_df      : DataFrame(p_up, p_dn) from MLSignal.backtest_series()
    df_price      : futures OHLCV DataFrame
    current       : dict with p_up, p_dn, timestamp (last closed candle)
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.65],
        vertical_spacing=0.04,
        subplot_titles=[
            "Futures Price",
            f"ML Signal — P(±{threshold_pct}% in {window_h}h)",
        ],
    )

    # ── Row 1: price ─────────────────────────────────────────────────────────
    mask = (df_price.index >= proba_df.index[0]) & (df_price.index <= proba_df.index[-1])
    price_slice = df_price["close"].loc[mask]
    fig.add_trace(go.Scatter(
        x=price_slice.index, y=price_slice, name="Price",
        line=dict(color="#26a69a", width=1.5),
    ), row=1, col=1)

    # ── Row 2: probabilities ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=proba_df.index, y=proba_df["p_up"],
        name=f"P(+{threshold_pct}%)",
        line=dict(color="#26a69a", width=2.0),
        fill="tozeroy", fillcolor="rgba(38,166,154,0.12)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=proba_df.index, y=proba_df["p_dn"],
        name=f"P(−{threshold_pct}%)",
        line=dict(color="#ef5350", width=2.0),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.12)",
    ), row=2, col=1)

    fig.add_hline(y=0.5, line_dash="dash",
                  line_color="rgba(255,255,255,0.30)", line_width=1.0,
                  row=2, col=1)
    fig.add_hline(y=0.25, line_dash="dot",
                  line_color="rgba(255,255,255,0.12)", line_width=0.8,
                  row=2, col=1)
    fig.add_hline(y=0.75, line_dash="dot",
                  line_color="rgba(255,255,255,0.12)", line_width=0.8,
                  row=2, col=1)

    # Vertical marker at the current-signal timestamp
    ts = current.get("timestamp")
    if ts is not None:
        fig.add_vline(
            x=ts, line_dash="dot",
            line_color="rgba(255,255,0,0.55)", line_width=1.5,
            annotation_text=" now",
            annotation_font_color="rgba(255,255,0,0.8)",
            annotation_position="top right",
        )

    p_up = current.get("p_up", 0.0)
    p_dn = current.get("p_dn", 0.0)
    title_suffix = f"  |  current: P(up) = {p_up:.1%}   P(dn) = {p_dn:.1%}"

    fig.update_layout(
        title=f"{symbol} — ML Probability Signal{title_suffix}",
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font_size=11),
        margin=dict(l=60, r=20, t=70, b=40),
    )
    fig.update_yaxes(range=[0, 1], row=2, col=1, tickformat=".0%")
    return fig


# ── Save all charts to disk ───────────────────────────────────────────────────

def _chart_entries(
    fig_price_ta: go.Figure,
    fig_oi: go.Figure,
    fig_funding: go.Figure,
    fig_ls: go.Figure,
    fig_flow: go.Figure,
    fig_liq_hist: go.Figure | None = None,
    fig_liq_est: go.Figure | None = None,
    fig_ml: go.Figure | None = None,
) -> list[tuple[str, str, go.Figure]]:
    """Ordered (file_key, section_label, figure) for non-None charts."""
    rows: list[tuple[str, str, go.Figure | None]] = [
        ("1_price_ta",       "Price & technical indicators", fig_price_ta),
        ("2_open_interest",  "Open interest",                fig_oi),
        ("3_funding_rate",   "Funding rate",                 fig_funding),
        ("4_ls_ratios",      "Long / short ratios",          fig_ls),
        ("5_spot_flow",      "Spot flow",                    fig_flow),
        ("6_liq_historical", "Historical liquidations",      fig_liq_hist),
        ("7_liq_estimated",  "Estimated liquidations",       fig_liq_est),
        ("8_ml_signal",      "ML probability signal",        fig_ml),
    ]
    return [(k, label, f) for k, label, f in rows if f is not None]


def _write_combined_html(symbol: str, entries: list[tuple[str, str, go.Figure]]) -> str:
    """
    One HTML page: load plotly.js from CDN only once (same URL as write_html uses — not the
    pip package version), then each figure below the previous (vertical scroll, no overlap).
    """
    blocks: list[str] = []
    for i, (key, label, fig) in enumerate(entries):
        # First fragment must include plotlyjs: the CDN path uses the bundled plotly.js
        # semver (e.g. 2.35.2), not the Python package version — a wrong URL 404s and charts
        # render as empty black boxes.
        fragment = fig.to_html(
            include_plotlyjs="cdn" if i == 0 else False,
            full_html=False,
            config={"displayModeBar": True, "responsive": True},
        )
        blocks.append(
            f'<section class="chart-block" id="chart-{key}" '
            f'aria-label="{label}">\n{fragment}\n</section>'
        )
    body = "\n".join(blocks)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{symbol} — Perps Board (all charts)</title>
  <style>
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: #0d0d0d;
      color: #e8e8e8;
      font-family: system-ui, -apple-system, sans-serif;
    }}
    .page-header {{
      padding: 1rem 1.25rem 0.5rem;
      border-bottom: 1px solid #2a2a2a;
    }}
    .page-header h1 {{
      margin: 0;
      font-size: 1.15rem;
      font-weight: 600;
    }}
    .page-header p {{
      margin: 0.35rem 0 0;
      font-size: 0.85rem;
      opacity: 0.75;
    }}
    .chart-block {{
      width: 100%;
      box-sizing: border-box;
      padding: 0.75rem 0.5rem 2rem;
      border-bottom: 1px solid #252525;
    }}
    .chart-block:last-child {{ border-bottom: none; }}
    .chart-block .plotly-graph-div {{
      width: 100% !important;
      max-width: 100%;
    }}
  </style>
</head>
<body>
  <header class="page-header">
    <h1>{symbol} — all charts</h1>
    <p>Scroll to view each chart in order (same content as the individual HTML files).</p>
  </header>
{body}
</body>
</html>
"""
    path = os.path.join(CHART_DIR, f"{symbol}_all_charts.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def save_all_charts(
    symbol: str,
    fig_price_ta: go.Figure,
    fig_oi: go.Figure,
    fig_funding: go.Figure,
    fig_ls: go.Figure,
    fig_flow: go.Figure | None = None,
    fig_liq_hist: go.Figure | None = None,
    fig_liq_est: go.Figure | None = None,
    fig_ml: go.Figure | None = None,
) -> list[str]:
    """Write all charts as a single scrollable HTML file; return [path]."""
    entries = _chart_entries(
        fig_price_ta, fig_oi, fig_funding, fig_ls, fig_flow,
        fig_liq_hist, fig_liq_est, fig_ml,
    )
    _mkdir()
    if not entries:
        return []
    return [_write_combined_html(symbol, entries)]
