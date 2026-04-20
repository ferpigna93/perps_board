"""
ML probability signal for perpetual futures.

Trains two calibrated XGBoost classifiers on historical features:
  P(price rises  > threshold% within the next window_h hours)
  P(price drops  > threshold% within the next window_h hours)

Feature set spans two timeframes (1 h and 4 h) plus market-structure data
(funding rate, open interest). Labels are built without look-ahead bias
using a vectorised forward-rolling window.

Training uses a strict time-ordered split:
  70 % training  → XGBoost fit
  15 % calibration → IsotonicRegression probability calibration
  15 % test      → AUC-ROC / Brier score evaluation

Usage
-----
    result = run_ml_pipeline(
        symbol        = "ZECUSDT",
        window_h      = 24,
        threshold_pct = 10.0,
        n_candles_1h  = 3000,
        df_oi         = data["df_oi"],
        df_funding    = data["df_funding"],
    )
    print(result["current"])   # {"p_up": 0.18, "p_dn": 0.42, ...}
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

import binance_client as bc
import indicators as ind

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_WINDOW_H      = 24
DEFAULT_THRESHOLD_PCT = 10.0
DEFAULT_N_CANDLES_1H  = 3_000
MIN_POSITIVE_SAMPLES  = 30     # refuse to train if fewer positive labels

# ── XGBoost — conservative to limit overfitting on small datasets ─────────────
_XGB = dict(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.03,
    subsample        = 0.8,
    colsample_bytree = 0.7,
    min_child_weight = 5,
    gamma            = 1.0,
    reg_lambda       = 2.0,
    eval_metric      = "logloss",
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)


# ── Feature engineering ───────────────────────────────────────────────────────

def _engineer(df_ind: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Derive ML features from an already-indicator-enriched OHLCV DataFrame.
    All distance features are expressed as a fraction of price so they are
    scale-invariant across different assets and time periods.
    """
    close = df_ind["close"]
    feat  = pd.DataFrame(index=df_ind.index)

    feat[f"rsi{suffix}"]          = df_ind.get("rsi")
    feat[f"bb_pctb{suffix}"]      = df_ind.get("bb_pct_b")
    feat[f"bb_width{suffix}"]     = df_ind.get("bb_width")
    feat[f"atr_pct{suffix}"]      = df_ind.get("atr") / close
    feat[f"adx{suffix}"]          = df_ind.get("adx")
    feat[f"adx_pos{suffix}"]      = df_ind.get("adx_pos")
    feat[f"adx_neg{suffix}"]      = df_ind.get("adx_neg")
    feat[f"macd_hist_pct{suffix}"]= df_ind.get("macd_hist") / close
    feat[f"stoch_k{suffix}"]      = df_ind.get("stoch_rsi_k")
    feat[f"stoch_d{suffix}"]      = df_ind.get("stoch_rsi_d")

    for p in [9, 21, 50, 200]:
        col = f"ema_{p}"
        if col in df_ind.columns:
            feat[f"ema{p}_dist{suffix}"] = (close - df_ind[col]) / close

    vol_ma = df_ind["volume"].rolling(20).mean()
    feat[f"vol_ratio{suffix}"]    = df_ind["volume"] / vol_ma.replace(0, np.nan)
    feat[f"ret_1{suffix}"]        = close.pct_change(1)
    feat[f"ret_4{suffix}"]        = close.pct_change(4)

    return feat


def build_feature_matrix(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_oi: pd.DataFrame | None = None,
    df_funding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix aligned to 1 h candle timestamps.

    1 h features — RSI, EMA distances, BB, ATR, ADX, MACD, Stoch RSI,
                   volume ratio, short-term returns.
    4 h features — same set, aligned with pd.merge_asof (backward fill).
    Market data  — funding rate (current + 7d avg), OI change % (1 h, 24 h).
    """

    def _tz(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.tz is None:
            return df.tz_localize("UTC")
        return df.tz_convert("UTC")

    df_1h = _tz(df_1h)
    df_4h = _tz(df_4h)

    # ── 1 h features ──────────────────────────────────────────────────────────
    ind_1h = ind.add_all_indicators(df_1h)
    feat   = _engineer(ind_1h, "_1h")

    # additional 1 h returns
    feat["ret_24_1h"] = df_1h["close"].pct_change(24)

    # ── 4 h features (aligned to 1 h) ─────────────────────────────────────────
    ind_4h   = ind.add_all_indicators(df_4h)
    feat_4h  = _engineer(ind_4h, "_4h")
    feat_4h["ret_1_4h_extra"] = df_4h["close"].pct_change(1)   # 1-candle return on 4h

    merged = pd.merge_asof(
        feat.reset_index(),
        feat_4h.reset_index(),
        on="open_time",
        direction="backward",
        tolerance=pd.Timedelta("4h"),
    ).set_index("open_time")

    # ── Market structure ───────────────────────────────────────────────────────
    if df_oi is not None and not df_oi.empty:
        oi = _tz(df_oi)["sumOpenInterestValue"]
        # bfill() fills leading NaN (from pct_change warmup) so rows older than
        # the OI coverage window are not silently dropped by dropna() later.
        merged["oi_chg_1h"]  = oi.pct_change(1).reindex(merged.index, method="nearest").bfill().ffill()
        merged["oi_chg_24h"] = oi.pct_change(24).reindex(merged.index, method="nearest").bfill().ffill()

    if df_funding is not None and not df_funding.empty:
        fr = _tz(df_funding)["fundingRate"]
        merged["funding_rate"]   = fr.reindex(merged.index, method="nearest").bfill().ffill()
        merged["funding_7d_avg"] = (
            fr.rolling(21).mean()
              .reindex(merged.index, method="nearest")
              .bfill().ffill()
        )

    # ── Clean ─────────────────────────────────────────────────────────────────
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    return merged


# ── Label construction ────────────────────────────────────────────────────────

def build_labels(
    df_1h: pd.DataFrame,
    window_h: int,
    threshold_pct: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Build forward-looking binary labels without look-ahead bias.

    For candle at time t:
      y_up[t] = 1  if  max(high[t+1 … t+window_h])  > close[t] × (1 + thr)
      y_dn[t] = 1  if  min(low [t+1 … t+window_h])  < close[t] × (1 − thr)

    Vectorised trick: high.shift(-window_h).rolling(window_h).max()
      • shift(-W) at index i → high[i + W]
      • rolling(W).max() at index i → max(high[i+1], …, high[i+W])  ✓
    The last W entries become NaN automatically (the shifted series contains NaN).
    """
    close = df_1h["close"]
    fh    = df_1h["high"].shift(-window_h).rolling(window_h).max()
    fl    = df_1h["low"].shift(-window_h).rolling(window_h).min()

    thr_up = close * (1 + threshold_pct / 100)
    thr_dn = close * (1 - threshold_pct / 100)

    y_up = (fh >= thr_up).astype(float)
    y_dn = (fl <= thr_dn).astype(float)

    # Rows where future window is incomplete must not be used as labels
    y_up[fh.isna()] = np.nan
    y_dn[fl.isna()] = np.nan

    return y_up, y_dn


# ── Model ──────────────────────────────────────────────────────────────────────

class MLSignal:
    """
    Two calibrated XGBoost classifiers: one for up-moves, one for down-moves.

    Probability calibration is done post-hoc with IsotonicRegression on a
    held-out calibration split so that output scores are true probabilities.
    """

    def __init__(self, window_h: int = DEFAULT_WINDOW_H,
                 threshold_pct: float = DEFAULT_THRESHOLD_PCT) -> None:
        self.window_h      = window_h
        self.threshold_pct = threshold_pct
        self.feature_cols: list[str] = []
        self.metrics:      dict      = {}
        self._model_up:    XGBClassifier | None     = None
        self._calib_up:    IsotonicRegression | None = None
        self._model_dn:    XGBClassifier | None     = None
        self._calib_dn:    IsotonicRegression | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y_up: pd.Series,
            y_dn: pd.Series) -> "MLSignal":
        """
        Time-ordered split: 70 % train / 15 % calibrate / 15 % test.
        Raises ValueError when too few positive examples exist.
        """
        for label, y in [("up", y_up), ("down", y_dn)]:
            n_pos = int(y.sum())
            if n_pos < MIN_POSITIVE_SAMPLES:
                raise ValueError(
                    f"Only {n_pos} positive '{label}' examples "
                    f"(need ≥ {MIN_POSITIVE_SAMPLES}). "
                    f"Try --ml-threshold lower than {self.threshold_pct}% "
                    f"or --ml-candles for more history."
                )

        self.feature_cols = list(X.columns)
        n       = len(X)
        n_train = int(n * 0.70)
        n_cal   = int(n * 0.15)

        slices = {
            "train": slice(0, n_train),
            "cal":   slice(n_train, n_train + n_cal),
            "test":  slice(n_train + n_cal, None),
        }

        self._model_up, self._calib_up = self._fit_one(
            X.iloc[slices["train"]], y_up.iloc[slices["train"]],
            X.iloc[slices["cal"]],   y_up.iloc[slices["cal"]],
        )
        self._model_dn, self._calib_dn = self._fit_one(
            X.iloc[slices["train"]], y_dn.iloc[slices["train"]],
            X.iloc[slices["cal"]],   y_dn.iloc[slices["cal"]],
        )

        # Evaluate on the held-out test split
        X_test   = X.iloc[slices["test"]]
        p_up_t   = self._apply(self._model_up, self._calib_up, X_test)
        p_dn_t   = self._apply(self._model_dn, self._calib_dn, X_test)
        y_up_t   = y_up.iloc[slices["test"]]
        y_dn_t   = y_dn.iloc[slices["test"]]

        def _safe_auc(y_true, y_pred):
            try:
                return round(float(roc_auc_score(y_true, y_pred)), 3)
            except Exception:
                return None

        self.metrics = {
            "auc_up":     _safe_auc(y_up_t, p_up_t),
            "auc_dn":     _safe_auc(y_dn_t, p_dn_t),
            "brier_up":   round(float(brier_score_loss(y_up_t, p_up_t)), 4),
            "brier_dn":   round(float(brier_score_loss(y_dn_t, p_dn_t)), 4),
            "n_total":    n,
            "n_train":    n_train,
            "n_test":     len(X_test),
            "pos_up_pct": round(float(y_up.mean() * 100), 1),
            "pos_dn_pct": round(float(y_dn.mean() * 100), 1),
        }
        return self

    def _fit_one(
        self,
        X_tr: pd.DataFrame, y_tr: pd.Series,
        X_cal: pd.DataFrame, y_cal: pd.Series,
    ) -> tuple[XGBClassifier, IsotonicRegression]:
        """Fit XGBoost then calibrate probabilities with isotonic regression."""
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        spw   = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0

        model = XGBClassifier(**{**_XGB, "scale_pos_weight": spw})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)

        raw_cal   = model.predict_proba(X_cal)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_cal, y_cal.values)
        return model, calibrator

    # ── Inference ─────────────────────────────────────────────────────────────

    def _apply(self, model: XGBClassifier, calib: IsotonicRegression,
               X: pd.DataFrame) -> np.ndarray:
        raw = model.predict_proba(X[self.feature_cols])[:, 1]
        return calib.predict(raw)

    def predict_proba(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return calibrated P(up) and P(down) for every row in X."""
        return {
            "up": self._apply(self._model_up, self._calib_up, X),
            "dn": self._apply(self._model_dn, self._calib_dn, X),
        }

    def current_signal(self, X: pd.DataFrame) -> dict:
        """
        Probability estimate for the last *closed* candle (iloc[-2]).
        We skip iloc[-1] to avoid a still-forming candle, consistent with
        how latest_signal_summary() works in indicators.py.
        """
        row    = X.iloc[[-2]]
        probas = self.predict_proba(row)
        return {
            "p_up":      round(float(probas["up"][0]), 4),
            "p_dn":      round(float(probas["dn"][0]), 4),
            "timestamp": X.index[-2],
        }

    def backtest_series(self, X: pd.DataFrame) -> pd.DataFrame:
        """Historical probability time series (for charting)."""
        p = self.predict_proba(X)
        return pd.DataFrame({"p_up": p["up"], "p_dn": p["dn"]}, index=X.index)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_ml_pipeline(
    symbol:        str,
    window_h:      int   = DEFAULT_WINDOW_H,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    n_candles_1h:  int   = DEFAULT_N_CANDLES_1H,
    df_oi:         pd.DataFrame | None = None,
    df_funding:    pd.DataFrame | None = None,
) -> dict:
    """
    End-to-end ML pipeline:
      1. Fetch extended 1 h and 4 h history
      2. Build feature matrix
      3. Build forward-looking labels
      4. Align, drop NaN, verify sample counts
      5. Train MLSignal (XGBoost + isotonic calibration)
      6. Return current probabilities + backtest series + metrics

    Returns
    -------
    dict with keys:
        signal       — trained MLSignal instance
        current      — {"p_up": float, "p_dn": float, "timestamp": ...}
        metrics      — training / evaluation stats
        proba_df     — pd.DataFrame(p_up, p_dn) over training history
        feature_cols — list of feature column names
    """
    # 1 ── Fetch history ────────────────────────────────────────────────────────
    df_1h = bc.get_futures_klines_extended(symbol, "1h", n_candles_1h)
    df_4h = bc.get_futures_klines(symbol, "4h", limit=1_000)

    if df_1h.empty or len(df_1h) < 200:
        raise ValueError(
            f"Insufficient 1h history for {symbol} "
            f"(got {len(df_1h)} candles, need ≥ 200)."
        )

    # 2 ── Feature matrix ───────────────────────────────────────────────────────
    X_raw = build_feature_matrix(df_1h, df_4h, df_oi, df_funding)

    # 3 ── Labels ───────────────────────────────────────────────────────────────
    y_up, y_dn = build_labels(df_1h, window_h, threshold_pct)

    # 4 ── Align & clean ────────────────────────────────────────────────────────
    combined = (
        X_raw
        .join(y_up.rename("y_up"))
        .join(y_dn.rename("y_dn"))
        .dropna()
    )
    if len(combined) < 100:
        raise ValueError(
            f"Only {len(combined)} clean samples after dropping NaN. "
            "Use --ml-candles for more history or lower --ml-threshold."
        )

    X   = combined[X_raw.columns]
    y_u = combined["y_up"].astype(int)
    y_d = combined["y_dn"].astype(int)

    # 5 ── Train ────────────────────────────────────────────────────────────────
    signal = MLSignal(window_h, threshold_pct)
    signal.fit(X, y_u, y_d)

    # 6 ── Return ───────────────────────────────────────────────────────────────
    proba_df = signal.backtest_series(X)
    current  = signal.current_signal(X)

    return {
        "signal":       signal,
        "current":      current,
        "metrics":      signal.metrics,
        "proba_df":     proba_df,
        "feature_cols": signal.feature_cols,
    }
