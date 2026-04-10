"""
Binance public API client — perpetual futures (USDT-M) + spot.
No API key is required for any of these endpoints.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from config import FUTURES_BASE, SPOT_BASE


# ── Low-level request helper ───────────────────────────────────────────────────

def _get(base: str, path: str, params: dict, retries: int = 3) -> list | dict:
    url = f"{base}{path}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Request failed for {url}: {exc}") from exc
            wait = 2 ** attempt
            print(f"[WARN] {exc} — retrying in {wait}s …")
            time.sleep(wait)


def _to_df_klines(raw: list) -> pd.DataFrame:
    """Parse a Binance klines response into a typed DataFrame."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    numeric = ["open", "high", "low", "close", "volume",
               "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df[numeric] = df[numeric].astype(float)
    return df.drop(columns=["close_time", "trades", "_ignore"])


# ── Futures endpoints ──────────────────────────────────────────────────────────

def get_futures_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """OHLCV candles from USDT-M perpetual futures."""
    raw = _get(FUTURES_BASE, "/fapi/v1/klines",
               {"symbol": symbol, "interval": interval, "limit": limit})
    return _to_df_klines(raw)


def get_premium_index(symbol: str) -> dict:
    """
    Current mark price, index price, last funding rate, and next funding time.
    Response keys: symbol, markPrice, indexPrice, estimatedSettlePrice,
                   lastFundingRate, interestRate, nextFundingTime, time.
    """
    return _get(FUTURES_BASE, "/fapi/v1/premiumIndex", {"symbol": symbol})


def get_funding_rate_history(symbol: str, limit: int = 100) -> pd.DataFrame:
    """Historical funding rates (one record every 8 h)."""
    raw = _get(FUTURES_BASE, "/fapi/v1/fundingRate",
               {"symbol": symbol, "limit": limit})
    df = pd.DataFrame(raw)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df.set_index("fundingTime", inplace=True)
    return df[["fundingRate"]]


def get_open_interest_current(symbol: str) -> dict:
    """Snapshot of current open interest (in contracts)."""
    return _get(FUTURES_BASE, "/fapi/v1/openInterest", {"symbol": symbol})


def get_open_interest_history(symbol: str, period: str = "1h",
                              limit: int = 500) -> pd.DataFrame:
    """
    Historical open interest.
    Columns: sumOpenInterest (contracts), sumOpenInterestValue (USDT).
    """
    raw = _get(FUTURES_BASE, "/futures/data/openInterestHist",
               {"symbol": symbol, "period": period, "limit": limit})
    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df["sumOpenInterest"]      = df["sumOpenInterest"].astype(float)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
    return df[["sumOpenInterest", "sumOpenInterestValue"]]


def get_global_ls_ratio(symbol: str, period: str = "1h",
                        limit: int = 500) -> pd.DataFrame:
    """Long/Short account ratio for ALL traders (retail)."""
    raw = _get(FUTURES_BASE, "/futures/data/globalLongShortAccountRatio",
               {"symbol": symbol, "period": period, "limit": limit})
    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df[["longAccount", "shortAccount", "longShortRatio"]] = \
        df[["longAccount", "shortAccount", "longShortRatio"]].astype(float)
    return df[["longAccount", "shortAccount", "longShortRatio"]]


def get_top_trader_account_ratio(symbol: str, period: str = "1h",
                                 limit: int = 500) -> pd.DataFrame:
    """Long/Short ratio by ACCOUNT for top-tier traders."""
    raw = _get(FUTURES_BASE, "/futures/data/topLongShortAccountRatio",
               {"symbol": symbol, "period": period, "limit": limit})
    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df[["longAccount", "shortAccount", "longShortRatio"]] = \
        df[["longAccount", "shortAccount", "longShortRatio"]].astype(float)
    return df[["longAccount", "shortAccount", "longShortRatio"]]


def get_top_trader_position_ratio(symbol: str, period: str = "1h",
                                  limit: int = 500) -> pd.DataFrame:
    """Long/Short ratio by POSITION SIZE for top-tier traders."""
    raw = _get(FUTURES_BASE, "/futures/data/topLongShortPositionRatio",
               {"symbol": symbol, "period": period, "limit": limit})
    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df[["longAccount", "shortAccount", "longShortRatio"]] = \
        df[["longAccount", "shortAccount", "longShortRatio"]].astype(float)
    return df[["longAccount", "shortAccount", "longShortRatio"]]


def get_taker_ls_ratio(symbol: str, period: str = "1h",
                       limit: int = 500) -> pd.DataFrame:
    """
    Taker buy/sell volume ratio — reveals who is the aggressor each period.
    Columns: buyVol, sellVol, buySellRatio.
    """
    raw = _get(FUTURES_BASE, "/futures/data/takerlongshortRatio",
               {"symbol": symbol, "period": period, "limit": limit})
    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df[["buyVol", "sellVol", "buySellRatio"]] = \
        df[["buyVol", "sellVol", "buySellRatio"]].astype(float)
    return df[["buyVol", "sellVol", "buySellRatio"]]


# ── Spot endpoints ─────────────────────────────────────────────────────────────

def get_spot_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Spot OHLCV candles including taker buy volumes.
    taker_buy_base  — contracts bought by market takers (base asset)
    taker_buy_quote — USD value bought by market takers
    These are used to compute CVD and net money flow.
    """
    raw = _get(SPOT_BASE, "/api/v3/klines",
               {"symbol": symbol, "interval": interval, "limit": limit})
    return _to_df_klines(raw)
