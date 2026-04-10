# perps_board

A Python toolkit for studying perpetual futures pairs (e.g. `ZECUSDT.P`) on Binance.
Uses only **public endpoints** — no API key required.

## Features

| Category | What's included |
|---|---|
| **Technical Indicators** | EMA 9/21/50/200, Bollinger Bands, RSI, MACD, ATR, Stochastic RSI, ADX, VWAP |
| **Funding Rate** | Current rate, next payment time, 7-day/30-day averages, annualised rate, basis |
| **Open Interest** | Current OI (USDT), 24h/7d/30d change, MA30, price–OI correlation |
| **Long/Short Ratios** | Global retail, top-trader by account, top-trader by position, taker buy/sell ratio |
| **Spot Money Flow** | Net flow per candle (USDT), 24h/7d totals, Cumulative Volume Delta (CVD) |
| **Charts** | 5 interactive HTML charts (Plotly dark theme) |
| **Terminal** | Rich-formatted tables with colour-coded signals |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Default: ZECUSDT, 1-hour candles, 500 periods
python dashboard.py

# Custom symbol and timeframe
python dashboard.py BTCUSDT --interval 4h
python dashboard.py ETHUSDT --interval 1d --limit 365

# Terminal only — skip chart files
python dashboard.py SOLUSDT --no-charts

# Save charts and open them in the browser
python dashboard.py ZECUSDT --open

# TradingView ".P" suffix is automatically stripped
python dashboard.py ZECUSDT.P --interval 1h
```

## File layout

```
perps_board/
├── config.py          # Symbol, timeframe, indicator parameters
├── binance_client.py  # Binance public API wrapper (futures + spot)
├── indicators.py      # Technical indicators (wraps the `ta` library)
├── market_metrics.py  # OI, funding, L/S ratios, CVD, net money flow
├── charts.py          # Interactive Plotly charts
├── dashboard.py       # Main entry point (terminal + HTML output)
└── charts/            # Generated HTML files (created on first run)
```

## Output charts

| File | Content |
|---|---|
| `{SYMBOL}_1_price_ta.html` | Candlestick + EMAs + BB + RSI + MACD + Volume |
| `{SYMBOL}_2_open_interest.html` | Price vs OI (USDT) + MA30 + OI change % |
| `{SYMBOL}_3_funding_rate.html` | Per-payment funding rate + 7-day rolling avg |
| `{SYMBOL}_4_ls_ratios.html` | Global / top-trader account / position L/S + taker ratio |
| `{SYMBOL}_5_spot_flow.html` | Spot price + CVD + per-candle net money flow |

## Key concepts

**Funding Rate** — charged every 8 hours between longs and shorts. Positive → longs pay shorts (market is overheated bullish). Negative → shorts pay longs.

**Open Interest** — total value of all outstanding contracts. Rising OI + rising price confirms a strong trend. Rising OI + falling price signals aggressive short-selling.

**Long/Short Ratio** — three separate sources are shown:
- *Global* — ratio of all retail accounts
- *Top-trader account* — whether whales are net long or short
- *Top-trader position* — size-weighted version of the above

**Cumulative Volume Delta (CVD)** — running total of `taker_buy − taker_sell` on the spot market. CVD diverging from price is an early sign of exhaustion.

**Net Money Flow** — dollar value of taker buys minus taker sells per candle. Positive means more aggressive buying pressure.
