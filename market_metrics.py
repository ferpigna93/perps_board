"""
Futures-specific market metrics:
  - Funding rate statistics
  - Open Interest analysis
  - Long/Short ratio summaries (global, top-trader account, top-trader position)
  - Spot Cumulative Volume Delta (CVD) and net money flow
"""
from __future__ import annotations

import datetime

import pandas as pd


# ── Funding Rate ───────────────────────────────────────────────────────────────

def funding_summary(df_funding: pd.DataFrame, premium: dict) -> dict:
    """
    Return key funding-rate statistics.

    Funding is paid every 8 h → 3 payments/day → annualised ≈ rate × 3 × 365.
    Positive funding → longs pay shorts (market is bullish/overheated).
    Negative funding → shorts pay longs (market is bearish/oversold).
    """
    rates = df_funding["fundingRate"]
    ann_factor = 3 * 365  # payments per year

    last_rate   = float(premium.get("lastFundingRate", 0))
    mark_price  = float(premium.get("markPrice", 0))
    index_price = float(premium.get("indexPrice", 0))
    basis_pct   = ((mark_price - index_price) / index_price * 100
                   if index_price else 0)

    next_funding_ts = premium.get("nextFundingTime")
    next_funding_dt = (
        datetime.datetime.fromtimestamp(int(next_funding_ts) / 1000,
                                        tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if next_funding_ts else "—"
    )

    return {
        "current_rate_pct":    round(last_rate * 100, 4),
        "next_funding_time":   next_funding_dt,
        "mark_price":          mark_price,
        "index_price":         index_price,
        "basis_pct":           round(basis_pct, 4),
        "7d_avg_rate_pct":     round(rates.tail(21).mean() * 100, 4),   # 21 × 8 h = 7 d
        "30d_avg_rate_pct":    round(rates.tail(90).mean() * 100, 4),
        "annualized_rate_pct": round(rates.tail(21).mean() * ann_factor * 100, 2),
        "positive_pct":        round((rates > 0).mean() * 100, 1),
        "max_rate_pct":        round(rates.max() * 100, 4),
        "min_rate_pct":        round(rates.min() * 100, 4),
    }


# ── Open Interest ──────────────────────────────────────────────────────────────

def oi_summary(df_oi: pd.DataFrame) -> dict:
    """Return key OI statistics derived from the historical OI DataFrame."""
    oi_val = df_oi["sumOpenInterestValue"]
    n = len(oi_val)

    def _change(periods):
        if n > periods:
            return round((oi_val.iloc[-1] / oi_val.iloc[-periods - 1] - 1) * 100, 2)
        return None

    return {
        "latest_oi_usdt":     round(oi_val.iloc[-1], 2),
        "24h_oi_change_pct":  _change(24),
        "7d_oi_change_pct":   _change(168),
        "30d_oi_change_pct":  _change(720),
        "oi_max_usdt":        round(oi_val.max(), 2),
        "oi_min_usdt":        round(oi_val.min(), 2),
        "oi_mean_usdt":       round(oi_val.mean(), 2),
    }


def enrich_oi(df_oi: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
    """
    Merge OI history with price candles and add derived columns:
      oi_change_pct   — period-over-period OI change
      oi_ma30         — 30-period moving average of OI (USDT)
      price_oi_corr   — rolling 30-period correlation between price and OI
    """
    merged = df_oi.copy()
    if not df_price.empty:
        merged = merged.join(df_price[["close"]], how="left")
        merged["close"] = merged["close"].ffill()

    merged["oi_change_pct"] = merged["sumOpenInterestValue"].pct_change() * 100
    merged["oi_ma30"]       = merged["sumOpenInterestValue"].rolling(30).mean()
    if "close" in merged.columns:
        merged["price_oi_corr"] = (
            merged["close"]
            .rolling(30)
            .corr(merged["sumOpenInterestValue"])
        )
    return merged


# ── Long/Short Ratios ──────────────────────────────────────────────────────────

def ls_summary(
    df_global: pd.DataFrame,
    df_top_acc: pd.DataFrame,
    df_top_pos: pd.DataFrame,
    df_taker: pd.DataFrame,
) -> dict:
    """
    Aggregate the most recent snapshot from all L/S-ratio sources.

    Interpretation:
      global_ls_ratio  > 1 → more retail accounts are long
      top_acc_ls_ratio > 1 → top traders are net long by account
      top_pos_ls_ratio > 1 → top traders hold larger long positions
      taker_ratio      > 1 → buy-side aggression dominates
    """
    def last(df: pd.DataFrame, col: str):
        try:
            return round(float(df[col].iloc[-1]), 4)
        except Exception:
            return None

    gl  = last(df_global,  "longShortRatio")
    ta_ = last(df_top_acc, "longShortRatio")
    tp  = last(df_top_pos, "longShortRatio")
    tk  = last(df_taker,   "buySellRatio")

    result = {
        # Ratios
        "global_ls_ratio":      gl,
        "global_long_pct":      round(last(df_global, "longAccount") * 100, 2)
                                if last(df_global, "longAccount") else None,
        "global_short_pct":     round(last(df_global, "shortAccount") * 100, 2)
                                if last(df_global, "shortAccount") else None,
        "top_acc_ls_ratio":     ta_,
        "top_pos_ls_ratio":     tp,
        "taker_buy_sell_ratio": tk,
        # 24-h taker volume totals
        "taker_buy_vol_24h":    round(df_taker["buyVol"].tail(24).sum(), 2)
                                if not df_taker.empty else None,
        "taker_sell_vol_24h":   round(df_taker["sellVol"].tail(24).sum(), 2)
                                if not df_taker.empty else None,
    }

    # Composite bias signal
    bulls = sum(1 for v in [gl, ta_, tp, tk] if v is not None and v > 1)
    bears = sum(1 for v in [gl, ta_, tp, tk] if v is not None and v < 1)
    result["bias_score"] = f"{bulls}/4 bullish, {bears}/4 bearish"

    return result


# ── Spot CVD & Net Money Flow ─────────────────────────────────────────────────

def add_spot_flow_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Cumulative Volume Delta (CVD) and net money flow from spot klines.

    Binance klines include taker_buy_base and taker_buy_quote so we can
    separate aggressive buyers from aggressive sellers each candle:

        taker_sell_base  = volume - taker_buy_base
        delta (base)     = taker_buy_base  - taker_sell_base
        net_flow (USDT)  = taker_buy_quote - taker_sell_quote

    CVD = cumulative(delta) — rising CVD confirms bullish price action.
    When price rises but CVD falls, buying pressure is weakening (divergence).
    """
    df = df.copy()
    df["taker_sell_base"]   = df["volume"]       - df["taker_buy_base"]
    df["taker_sell_quote"]  = df["quote_volume"] - df["taker_buy_quote"]

    # Candle-level delta
    df["delta"]         = df["taker_buy_base"]  - df["taker_sell_base"]
    df["net_flow_usdt"] = df["taker_buy_quote"] - df["taker_sell_quote"]

    # Cumulative series
    df["cvd"]               = df["delta"].cumsum()
    df["net_flow_cum_usdt"] = df["net_flow_usdt"].cumsum()

    # Rolling 24-period net flow (≈ 24 h on 1-h candles)
    df["net_flow_24p"] = df["net_flow_usdt"].rolling(24).sum()

    # Buy ratio per candle
    df["buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, float("nan"))

    return df


def spot_flow_summary(df: pd.DataFrame) -> dict:
    """Summarise the most recent spot money-flow metrics."""
    def _fmt(val):
        try:
            return round(float(val), 2)
        except Exception:
            return None

    buy_24  = df["taker_buy_base"].tail(24).sum()
    vol_24  = df["volume"].tail(24).sum()
    buy_pct = round(buy_24 / vol_24 * 100, 2) if vol_24 else None

    cvd_now  = df["cvd"].iloc[-1]
    cvd_prev = df["cvd"].iloc[-25] if len(df) > 25 else df["cvd"].iloc[0]

    return {
        "net_flow_last_candle_usdt": _fmt(df["net_flow_usdt"].iloc[-2]),
        "net_flow_24h_usdt":         _fmt(df["net_flow_usdt"].tail(24).sum()),
        "net_flow_7d_usdt":          _fmt(df["net_flow_usdt"].tail(168).sum()),
        "cvd_current":               _fmt(cvd_now),
        "cvd_24h_change":            _fmt(cvd_now - cvd_prev),
        "cvd_trend":                 "increasing" if cvd_now > cvd_prev else "decreasing",
        "buy_pct_24h":               buy_pct,
    }
