"""
Technical indicators computed on top of a standard OHLCV DataFrame.
Depends on the `ta` library (pip install ta).

Each add_* function returns the same DataFrame with new columns appended.
Call add_all_indicators() to compute everything in one shot.
"""
from __future__ import annotations

import math

import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume

from config import (
    ADX_PERIOD, ATR_PERIOD, BB_PERIOD, BB_STD,
    EMA_PERIODS, MACD_FAST, MACD_SIGNAL, MACD_SLOW,
    RSI_PERIOD, STOCH_RSI_D, STOCH_RSI_K,
)


# ── Individual indicator functions ─────────────────────────────────────────────

def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Exponential Moving Averages for each period in EMA_PERIODS."""
    for p in EMA_PERIODS:
        df[f"ema_{p}"] = ta.trend.EMAIndicator(
            df["close"], window=p, fillna=False
        ).ema_indicator()
    return df


def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger Bands (BB_PERIOD, BB_STD).
    Added columns: bb_upper, bb_mid, bb_lower, bb_width, bb_pct_b.
    """
    bb = ta.volatility.BollingerBands(
        df["close"], window=BB_PERIOD, window_dev=BB_STD, fillna=False
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()   # (upper − lower) / mid  ×100
    df["bb_pct_b"] = bb.bollinger_pband()   # 0 = at lower, 1 = at upper
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Relative Strength Index."""
    df["rsi"] = ta.momentum.RSIIndicator(
        df["close"], window=RSI_PERIOD, fillna=False
    ).rsi()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.
    Added columns: macd, macd_signal, macd_hist.
    """
    macd = ta.trend.MACD(
        df["close"],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL,
        fillna=False,
    )
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Average True Range — measures volatility in price units."""
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=ATR_PERIOD, fillna=False
    ).average_true_range()
    return df


def add_stoch_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stochastic RSI — momentum within the RSI range.
    Added columns: stoch_rsi_k, stoch_rsi_d.
    """
    srsi = ta.momentum.StochRSIIndicator(
        df["close"],
        window=RSI_PERIOD,
        smooth1=STOCH_RSI_K,
        smooth2=STOCH_RSI_D,
        fillna=False,
    )
    df["stoch_rsi_k"] = srsi.stochrsi_k()
    df["stoch_rsi_d"] = srsi.stochrsi_d()
    return df


def add_adx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average Directional Index — trend strength (0–100).
    Added columns: adx, adx_pos (+DI), adx_neg (−DI).
    ADX > 25 indicates a trending market.
    """
    adx = ta.trend.ADXIndicator(
        df["high"], df["low"], df["close"], window=ADX_PERIOD, fillna=False
    )
    df["adx"]     = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume-Weighted Average Price.
    Note: `ta` computes a running VWAP from the start of the DataFrame.
    For an intraday session VWAP, filter the DataFrame to a single session first.
    """
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
        df["high"], df["low"], df["close"], df["volume"], fillna=False
    ).volume_weighted_average_price()
    return df


# ── Composite ──────────────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator to a copy of df and return the enriched DataFrame."""
    df = df.copy()
    df = add_emas(df)
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_stoch_rsi(df)
    df = add_adx(df)
    df = add_vwap(df)
    return df


# ── Signal summary ─────────────────────────────────────────────────────────────

def _safe(value, default=float("nan")):
    try:
        v = float(value)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def latest_signal_summary(df: pd.DataFrame) -> dict:
    """
    Derive simple directional signals from the last *completed* candle (iloc[-2]).
    Using -2 avoids relying on a candle that is still forming.
    """
    r = df.iloc[-2]
    close = _safe(r["close"])

    return {
        # Price
        "price": close,
        # Trend — EMA alignment
        "above_ema9":   close > _safe(r.get("ema_9")),
        "above_ema21":  close > _safe(r.get("ema_21")),
        "above_ema50":  close > _safe(r.get("ema_50")),
        "above_ema200": close > _safe(r.get("ema_200")),
        # Momentum
        "rsi":       round(_safe(r.get("rsi"), 50), 2),
        "macd_bull": _safe(r.get("macd"), 0) > _safe(r.get("macd_signal"), 0),
        "macd_hist": round(_safe(r.get("macd_hist"), 0), 5),
        # Volatility
        "bb_pct_b": round(_safe(r.get("bb_pct_b"), 0.5), 4),
        "bb_width":  round(_safe(r.get("bb_width"), 0), 4),
        "atr":       round(_safe(r.get("atr"), 0), 5),
        # Trend strength
        "adx":      round(_safe(r.get("adx"), 0), 2),
        "adx_bias": "bullish" if _safe(r.get("adx_pos"), 0) > _safe(r.get("adx_neg"), 0)
                    else "bearish",
        # Stochastic RSI
        "stoch_k": round(_safe(r.get("stoch_rsi_k"), 0.5), 4),
        "stoch_d": round(_safe(r.get("stoch_rsi_d"), 0.5), 4),
    }
