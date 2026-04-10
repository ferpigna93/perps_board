"""
Perps Board — central configuration.
Edit this file to change symbol, timeframe, or indicator parameters.
"""
from __future__ import annotations

# ── Symbol & timeframe ─────────────────────────────────────────────────────────
SYMBOL    = "ZECUSDT"   # Binance symbol (no ".P" suffix needed for the API)
INTERVAL  = "1h"        # Candle interval: 1m 5m 15m 1h 4h 1d …
LIMIT     = 500         # Number of candles to fetch (max 1 500 for klines)
OI_PERIOD = "1h"        # Period for OI / L/S-ratio history: 5m 15m 30m 1h 4h 6h 12h 1d
OI_LIMIT  = 500         # Number of OI / L/S-ratio data points

# ── Binance API base URLs (public endpoints — no key required) ─────────────────
FUTURES_BASE = "https://fapi.binance.com"
SPOT_BASE    = "https://api.binance.com"

# ── Technical indicator parameters ────────────────────────────────────────────
EMA_PERIODS  = [9, 21, 50, 200]
BB_PERIOD    = 20
BB_STD       = 2.0
RSI_PERIOD   = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
ATR_PERIOD   = 14
STOCH_RSI_K  = 3
STOCH_RSI_D  = 3
ADX_PERIOD   = 14

# ── Output ─────────────────────────────────────────────────────────────────────
CHART_DIR = "charts"    # Directory where interactive HTML charts are written
