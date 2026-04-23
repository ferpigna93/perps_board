# perps_board

A Python toolkit for studying perpetual futures pairs (e.g. `ZECUSDT.P`) on Binance.
Uses only **public endpoints** — no API key required.

## Features

| Category | What's included |
|---|---|
| **Technical Indicators** | EMA 9/21/50/200, Bollinger Bands, RSI, MACD, ATR, Stochastic RSI, ADX, VWAP |
| **Funding Rate** | Current rate, next payment time, 7d/30d averages, annualised rate, basis |
| **Open Interest** | Current OI (USDT), 24h/7d/30d change, MA30 |
| **Long/Short Ratios** | Global retail, top-trader by account, top-trader by position, taker buy/sell ratio |
| **Spot Money Flow** | Net flow per candle (USDT), 24h/7d totals, Cumulative Volume Delta (CVD) |
| **Liquidation Heatmaps** | Historical (inferred from OI drops) + estimated future (leverage distribution model) |
| **ML Signal** | Calibrated XGBoost models predicting P(±X% in Nh) + directional bias index |
| **Charts** | All charts combined in a single scrollable HTML (Plotly dark theme) |
| **Terminal** | Rich-formatted tables with colour-coded signals |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Main dashboard

```bash
# Default: ZECUSDT, 1h candles, 500 periods
python dashboard.py

# Custom symbol and timeframe
python dashboard.py BTCUSDT --interval 4h
python dashboard.py ETHUSDT --interval 1d --limit 365

# Terminal only — skip chart generation
python dashboard.py SOLUSDT --no-charts

# Save charts and open in browser
python dashboard.py ZECUSDT --open

# Skip liquidation heatmaps (faster)
python dashboard.py ZECUSDT --no-liq

# Include ML signal (adds ~1–2 min for data fetch + training)
python dashboard.py ZECUSDT --ml
python dashboard.py ZECUSDT --ml --ml-window 12 --ml-threshold 5
python dashboard.py ZECUSDT --ml --ml-candles 6000

# TradingView ".P" suffix is automatically stripped
python dashboard.py ZECUSDT.P
```

### Standalone ML report

Runs the full ML pipeline independently and generates a focused HTML report
with the probability signal chart and feature importance chart.

```bash
# Default parameters (24h window, ±10% threshold, 3000 candles)
python ml_report.py ZECUSDT

# Custom configuration
python ml_report.py BTCUSDT --window 12 --threshold 3 --candles 6000

# Adjust the high-confidence marker threshold on the chart
python ml_report.py ZECUSDT --signal-threshold 0.65

# Open the report in the browser immediately
python ml_report.py ZECUSDT --open
```

## File layout

```
perps_board/
├── config.py          # Symbol, timeframe, indicator parameters
├── binance_client.py  # Binance public API wrapper (futures + spot, with pagination)
├── indicators.py      # Technical indicators (wraps the `ta` library)
├── market_metrics.py  # OI, funding, L/S ratios, CVD, net money flow
├── liquidations.py    # Historical inference + estimated future liquidation heatmaps
├── ml_signal.py       # ML pipeline: features, labels, XGBoost, calibration
├── charts.py          # Interactive Plotly charts + combined HTML writer
├── dashboard.py       # Main entry point (terminal + HTML output)
├── ml_report.py       # Standalone ML report generator
└── charts/            # Generated HTML files (created on first run)
```

## Dashboard CLI flags

| Flag | Default | Description |
|---|---|---|
| `symbol` | `ZECUSDT` | Binance symbol (TradingView `.P` suffix is stripped) |
| `--interval` | `1h` | Candle interval: `1m` `5m` `15m` `1h` `4h` `1d` |
| `--limit` | `500` | Number of candles (max 1500 per Binance call) |
| `--oi-period` | `1h` | Period for OI and L/S ratio data |
| `--no-charts` | — | Skip HTML chart generation |
| `--open` | — | Open the combined HTML in the browser |
| `--no-liq` | — | Skip liquidation heatmaps |
| `--ml` | — | Run the ML probability signal |
| `--ml-window` | `24` | Forecast horizon in hours |
| `--ml-threshold` | `10.0` | Target move % for labels |
| `--ml-candles` | `3000` | 1h candles of history for ML training |

## ML signal — how it works

The ML module trains two calibrated XGBoost classifiers:

- **P(up)** — probability that price rises ≥ `threshold%` at any point within the next `window` hours
- **P(dn)** — probability that price drops ≥ `threshold%` at any point within the next `window` hours
- **Bias index** — `100 × (P(up) − P(dn))`, range `[−100, +100]`

### Labels

Built with a forward-rolling window — no look-ahead bias:

```
y_up[t] = 1  if  max(high[t+1 … t+W])  ≥  close[t] × (1 + threshold%)
y_dn[t] = 1  if  min(low [t+1 … t+W])  ≤  close[t] × (1 − threshold%)
```

`y_up = 1` means the price **touched** the target at any point in the window, not necessarily at close.

### Feature set (~32 features)

| Group | Features |
|---|---|
| **1h technical** | RSI, BB %B, BB width, ATR%, ADX, ADX+/−, MACD hist%, Stoch RSI K/D, EMA distances (9/21/50/200), volume ratio, returns (1h/4h/24h) |
| **4h technical** | Same set on the 4h timeframe, aligned with `merge_asof` (no look-ahead) |
| **Market structure** | OI change % (1h, 24h), funding rate, 7d avg funding rate |
| **Liquidation pressure** | `liq_below`: fraction of OI (longs) with forced-close price in `[close×(1−thr), close]` — how much long OI would cascade if price drops by `threshold%`. `liq_above`: same for shorts on the upside. |

### Training

- **Split**: strict temporal 70% train / 15% calibration / 15% test — no random shuffle
- **Model**: XGBoost with `scale_pos_weight` to handle class imbalance
- **Calibration**: IsotonicRegression on the held-out calibration set so output scores are true probabilities
- **Evaluation**: AUC-ROC and Brier score on the held-out test set

### ML report charts

The `ml_report.py` HTML output contains two charts:

1. **Signal chart** — full evaluation period price with dotted vertical lines marking candles where P(up) or P(dn) exceeded the signal threshold (default 75%); probability time series; directional bias index bar chart
2. **Feature importance** — XGBoost gain scores for the top N features in each model, shown side by side for the up and down models

## Liquidation heatmaps

### Historical (inferred from OI + OHLCV)

Binance's `/fapi/v1/allForceOrders` endpoint was permanently retired. Liquidation events are instead inferred from two signals that are always available:

- A significant OI drop (> 0.3%) on a candle → positions were force-closed
- Candle direction → which side was hit (`close < open` → longs liquidated; `close > open` → shorts)

### Estimated future

Models where liquidation cascades could occur if price moves from the current level, using:
- Recent candles as a volume-weighted entry-price distribution
- A realistic leverage distribution (2×–100×, calibrated to Binance user data)
- Simplified liquidation price formula per leverage tier

## Key concepts

**Funding Rate** — charged every 8 hours between longs and shorts. Positive → longs pay shorts (market overheated bullish). Negative → shorts pay longs.

**Open Interest** — total value of all outstanding contracts. Rising OI + rising price confirms trend. Rising OI + falling price signals aggressive short-selling.

**Long/Short Ratios** — three separate sources:
- *Global* — ratio of all retail accounts
- *Top-trader account* — whether whales are net long or short
- *Top-trader position* — size-weighted version of the above

**Cumulative Volume Delta (CVD)** — running total of `taker_buy − taker_sell` on spot. CVD diverging from price is an early sign of exhaustion.

**Liquidation pressure features** — `liq_below` and `liq_above` represent what fraction of estimated open interest has its forced-close price within the `threshold%` range of the current price. High values indicate a cluster of vulnerable positions that would amplify a move in that direction (self-reinforcing cascade risk).

**Directional bias index** — `100 × (P(up) − P(dn))`. Ranges from −100 (maximum bearish) to +100 (maximum bullish). Reference levels ±25 mark moderate conviction.
