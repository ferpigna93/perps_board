"""
Liquidation analysis for perpetual futures.

Two complementary tools:

1. HISTORICAL heatmap
   Source: /fapi/v1/allForceOrders (Binance public endpoint)
   How:    bucket each actual forced-liquidation order by (time, price),
           producing a 2-D matrix of liq volume that is plotted as a heatmap
           overlaid on the price chart.

2. ESTIMATED future heatmap
   Source: recent OHLCV + current open-interest (no private data needed)
   How:    use recent candles as a proxy for the entry-price distribution
           (weighted by volume), spread that OI across a realistic leverage
           distribution, compute long/short liquidation prices, accumulate
           into a price grid, and plot the density as a horizontal bar chart
           (snapshot) + a rolling 2-D heatmap (how the landscape evolves).

Interpretation:
  - Dense historical clusters  → price levels where cascades happened before
  - Dense estimated clusters   → "danger zones": if price reaches these levels
                                 a cascade of forced closings may accelerate
                                 the move (self-reinforcing liquidation spiral)
  - SELL-side liq (longs)      → accumulates BELOW the current price
  - BUY-side  liq (shorts)     → accumulates ABOVE the current price
"""
from __future__ import annotations

import time as _time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# ── Leverage distribution (fraction of OI assumed at each leverage tier) ──────
# Based on Binance user-statistics studies and community research.
# Adjust if you have better data for a specific asset.
LEVERAGE_DIST: dict[int, float] = {
    2:   0.04,
    3:   0.05,
    5:   0.13,
    10:  0.28,   # most popular tier
    15:  0.10,
    20:  0.18,
    25:  0.05,
    50:  0.11,
    100: 0.06,
}
# Maintenance margin rate (simplified; actual varies by notional tier)
_MMR = 0.005


# ── Historical liquidation data ───────────────────────────────────────────────

def fetch_force_orders(symbol: str, days: int = 7,
                       limit_per_call: int = 1000) -> pd.DataFrame:
    """
    Pull historical forced-liquidation orders from Binance, going back `days`
    days.  Paginates in 1-day windows (Binance rejects windows > 24 h on this
    endpoint) and walks backwards from now to `days` ago.

    Returns a DataFrame with columns:
        time (index, UTC), side, price, avg_price, qty, value_usdt
        side == 'SELL' → long liquidated  (price fell through their level)
        side == 'BUY'  → short liquidated (price rose through their level)
    """
    import binance_client as bc

    _DAY_MS    = int(24 * 3600 * 1000)          # 1 day in ms — safe window size
    _BUFFER_MS = 3_000                           # 3 s behind "now" to avoid clock-skew 400s

    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000) - _BUFFER_MS
    start_ms = end_ms - int(days * _DAY_MS)

    all_rows: list[dict] = []
    cursor = end_ms

    while cursor > start_ms:
        window_start = max(start_ms, cursor - _DAY_MS)
        try:
            raw = bc._get(
                bc.FUTURES_BASE, "/fapi/v1/allForceOrders",
                {"symbol": symbol, "startTime": window_start,
                 "endTime": cursor, "limit": limit_per_call},
                retries=2,
            )
        except RuntimeError:
            # No data for this window or API error — move on
            cursor = window_start - 1
            _time.sleep(0.1)
            continue

        if raw:
            all_rows.extend(raw)
        cursor = window_start - 1
        _time.sleep(0.15)   # ~6 calls/s — stay well within rate limits

    if not all_rows:
        return pd.DataFrame(columns=["side", "price", "avg_price", "qty", "value_usdt"])

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates()
    df["time"]      = pd.to_datetime(df["time"].astype(int), unit="ms", utc=True)
    df["price"]     = df["price"].astype(float)
    df["avg_price"] = df["avgPrice"].astype(float)
    df["qty"]       = df["origQty"].astype(float)
    df["value_usdt"] = df["qty"] * df["avg_price"]
    df = df.set_index("time").sort_index()
    return df[["side", "price", "avg_price", "qty", "value_usdt"]]


def build_historical_heatmap(
    df_orders: pd.DataFrame,
    df_price: pd.DataFrame,
    n_price_bins: int = 120,
    time_bucket: str = "4h",
) -> dict:
    """
    Aggregate historical liquidations into a 2-D (price × time) matrix.

    Returns a dict with keys:
        z_long   — 2-D array  [n_price_bins × n_time_bins], long liq volume
        z_short  — same for short liquidations
        x_labels — list of time-bin labels (str)
        y_prices — array of price-bin centres
        price_line — price series resampled to the same time buckets
    """
    if df_orders.empty:
        return {}

    # Price range: expand slightly beyond actual liquidation prices
    p_min = df_orders["price"].min() * 0.99
    p_max = df_orders["price"].max() * 1.01
    price_edges = np.linspace(p_min, p_max, n_price_bins + 1)
    price_centres = (price_edges[:-1] + price_edges[1:]) / 2

    # Time buckets
    df_orders["time_bucket"] = df_orders.index.floor(time_bucket)
    time_bins = sorted(df_orders["time_bucket"].unique())
    t_labels  = [str(t)[:16] for t in time_bins]

    z_long  = np.zeros((n_price_bins, len(time_bins)))
    z_short = np.zeros((n_price_bins, len(time_bins)))

    for j, tb in enumerate(time_bins):
        window = df_orders[df_orders["time_bucket"] == tb]
        for _, row in window.iterrows():
            bin_idx = np.searchsorted(price_edges, row["price"], side="right") - 1
            bin_idx = np.clip(bin_idx, 0, n_price_bins - 1)
            if row["side"] == "SELL":          # long liquidated
                z_long[bin_idx, j] += row["value_usdt"]
            else:                              # short liquidated
                z_short[bin_idx, j] += row["value_usdt"]

    # Price line aligned to the same buckets
    price_resampled = (
        df_price["close"]
        .resample(time_bucket)
        .last()
        .reindex(time_bins, method="nearest")
        .values
    )

    return dict(
        z_long=z_long, z_short=z_short,
        x_labels=t_labels, y_prices=price_centres,
        price_line=price_resampled,
    )


def liq_stats(df_orders: pd.DataFrame) -> dict:
    """Summary statistics for the liquidation history DataFrame."""
    if df_orders.empty:
        return {"total_usdt": 0, "long_usdt": 0, "short_usdt": 0,
                "n_events": 0, "largest_event_usdt": 0}

    longs  = df_orders[df_orders["side"] == "SELL"]["value_usdt"]
    shorts = df_orders[df_orders["side"] == "BUY"]["value_usdt"]
    return {
        "total_usdt":         round(df_orders["value_usdt"].sum(), 0),
        "long_usdt":          round(longs.sum(), 0),
        "short_usdt":         round(shorts.sum(), 0),
        "n_events":           len(df_orders),
        "largest_event_usdt": round(df_orders["value_usdt"].max(), 0),
        "long_pct":           round(longs.sum() / df_orders["value_usdt"].sum() * 100, 1)
                              if df_orders["value_usdt"].sum() else 0,
    }


# ── Estimated future liquidation map ─────────────────────────────────────────

def estimate_liq_map(
    df_klines: pd.DataFrame,
    oi_usdt: float,
    price_range: float = 0.35,
    n_bins: int = 300,
    long_short_split: float = 0.5,
    sigma_pct: float = 0.003,
) -> pd.DataFrame:
    """
    Estimate the liquidation density across price levels given the current
    open interest and the recent entry-price distribution.

    Parameters
    ----------
    df_klines       : futures OHLCV DataFrame (close + volume required)
    oi_usdt         : total open interest in USDT (scalar)
    price_range     : ± fraction around current price to model (default 35 %)
    n_bins          : number of price levels in the output grid
    long_short_split: fraction of OI assumed to be long (default 0.5 = 50/50)
    sigma_pct       : Gaussian smoothing width as fraction of price (default 0.3 %)

    Returns
    -------
    DataFrame with columns: price, liq_long, liq_short, liq_total
    liq_long  → USDT that would be liquidated if price DROPS to this level
    liq_short → USDT that would be liquidated if price RISES to this level
    """
    current_price = float(df_klines["close"].iloc[-1])
    p_min = current_price * (1 - price_range)
    p_max = current_price * (1 + price_range)
    price_grid = np.linspace(p_min, p_max, n_bins)

    liq_long  = np.zeros(n_bins)
    liq_short = np.zeros(n_bins)

    total_vol = df_klines["volume"].sum()
    if total_vol == 0:
        total_vol = 1.0

    oi_long  = oi_usdt * long_short_split
    oi_short = oi_usdt * (1 - long_short_split)

    sigma = current_price * sigma_pct   # Gaussian spread

    for _, row in df_klines.iterrows():
        entry  = float(row["close"])
        weight = float(row["volume"]) / total_vol   # OI weight proportional to volume

        for leverage, lev_frac in LEVERAGE_DIST.items():
            oi_tier_long  = oi_long  * weight * lev_frac
            oi_tier_short = oi_short * weight * lev_frac

            # Simplified liquidation price (accounting for maintenance margin)
            liq_price_long  = entry * (1 - (1 / leverage) + _MMR)
            liq_price_short = entry * (1 + (1 / leverage) - _MMR)

            # Only add if liquidation level falls within our price grid
            if p_min <= liq_price_long <= p_max:
                kernel = np.exp(-0.5 * ((price_grid - liq_price_long) / sigma) ** 2)
                liq_long  += oi_tier_long  * kernel / (kernel.sum() or 1)

            if p_min <= liq_price_short <= p_max:
                kernel = np.exp(-0.5 * ((price_grid - liq_price_short) / sigma) ** 2)
                liq_short += oi_tier_short * kernel / (kernel.sum() or 1)

    return pd.DataFrame({
        "price":     price_grid,
        "liq_long":  liq_long,
        "liq_short": liq_short,
        "liq_total": liq_long + liq_short,
    })


def build_estimated_heatmap_over_time(
    df_klines: pd.DataFrame,
    df_oi: pd.DataFrame,
    price_range: float = 0.30,
    n_price_bins: int = 150,
    window: int = 48,
) -> dict:
    """
    Build a rolling estimated liquidation heatmap (price × time).

    For each time step t, estimate the liquidation map using the `window`
    candles before t as the entry-price proxy and the OI value at t.

    Returns dict with keys:
        z        — 2-D array [n_price_bins × n_time_steps]
        x_labels — time labels
        y_prices — price bin centres
        price_line — close prices at each time step
    """
    # Align OI to klines index
    oi_series = df_oi["sumOpenInterestValue"].reindex(
        df_klines.index, method="nearest"
    ).ffill().fillna(df_oi["sumOpenInterestValue"].mean())

    # Use only the last min(500, len) candles to keep computation tractable
    df_k = df_klines.tail(min(500, len(df_klines))).copy()
    oi_s = oi_series.reindex(df_k.index, method="nearest").ffill()

    current_price = float(df_k["close"].iloc[-1])
    p_min = current_price * (1 - price_range)
    p_max = current_price * (1 + price_range)
    price_edges   = np.linspace(p_min, p_max, n_price_bins + 1)
    price_centres = (price_edges[:-1] + price_edges[1:]) / 2

    n_time = len(df_k)
    z = np.zeros((n_price_bins, n_time))

    total_vol_full = df_k["volume"].sum() or 1.0

    for t_idx in range(window, n_time):
        win  = df_k.iloc[t_idx - window: t_idx]
        oi_t = float(oi_s.iloc[t_idx])
        w_vol = win["volume"].sum() or 1.0

        for _, row in win.iterrows():
            entry  = float(row["close"])
            weight = float(row["volume"]) / w_vol
            for leverage, lev_frac in LEVERAGE_DIST.items():
                oi_tier = oi_t * weight * lev_frac * 0.5  # 50/50 long-short

                for liq_price in [
                    entry * (1 - (1 / leverage) + _MMR),   # long liq
                    entry * (1 + (1 / leverage) - _MMR),   # short liq
                ]:
                    if p_min <= liq_price <= p_max:
                        bin_idx = np.searchsorted(price_edges, liq_price, side="right") - 1
                        bin_idx = int(np.clip(bin_idx, 0, n_price_bins - 1))
                        z[bin_idx, t_idx] += oi_tier

    x_labels = [str(idx)[:16] for idx in df_k.index]

    return dict(
        z=z, x_labels=x_labels, y_prices=price_centres,
        price_line=df_k["close"].values,
        current_price=current_price,
    )
