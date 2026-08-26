"""Round 4 machine-learning pipeline: purged walk-forward prediction, causally mapped to weights.

Everything here is built so that the value used on day t is a deterministic function of data
up to the close of day t-1, which is what the tournament's forward-leakage probe tests:

- Features are lagged one day (they reuse and extend the causal library in model/strategy.py).
- The label for a sample dated s is the log return over the following H days, so it is only
  fully known H days later. A model retrained on date r may therefore train only on samples
  with s + H < r ("purged" walk-forward). Retraining happens every 90 days on an expanding
  window; predictions between retraining dates come from the most recent legal model.
- The raw prediction stream is standardised causally: each day's prediction is scaled by the
  expanding mean and standard deviation of the predictions made before it.
- The standardised prediction becomes a bounded multiplier through the same remaining-budget
  pacing allocator used by every other candidate. scikit-learn only (tournament-legal).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from model.strategy import MIN_WEIGHT, PRICE_COL, Params, allocate, construct_features

RETRAIN_EVERY = 90        # days between walk-forward refits
MIN_TRAIN_LABELS = 730    # first fit only after this many fully closed labels
HALVINGS = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])


@dataclass(frozen=True)
class MLConfig:
    model: str = "ridge"      # "ridge" or "hgbr"
    horizon: int = 30         # label horizon H in days
    a_ml: float = 1.0         # multiplier strength on the standardised prediction
    clip_ml: float = 3.0
    m_min: float = 0.25
    m_max: float = 5.0


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The registered feature set, every column lagged at least one day."""
    base = construct_features(df, Params())
    p = df[PRICE_COL].astype(float)
    lag = p.shift(1)
    X = pd.DataFrame(index=df.index)
    X["ma_gap"] = base["f_ma_gap"]
    X["drawdown"] = base["f_drawdown"]
    X["netflow_z"] = base["f_netflow_z"]
    X["mvrv"] = df["CapMVRVCur"].astype(float).shift(1) if "CapMVRVCur" in df.columns else np.nan
    X["ret_7"] = lag / lag.shift(7) - 1
    X["ret_30"] = lag / lag.shift(30) - 1
    X["ret_90"] = lag / lag.shift(90) - 1
    r = np.log(lag).diff()
    X["vol_30"] = r.rolling(30, min_periods=15).std() * np.sqrt(365)
    for col, name in (("AdrActCnt", "adr_mom"), ("FeeTotNtv", "fee_mom"), ("HashRate", "hash_mom")):
        if col in df.columns:
            s = df[col].astype(float).shift(1)
            X[name] = s.rolling(30, min_periods=20).mean() / s.rolling(365, min_periods=180).mean() - 1
        else:
            X[name] = np.nan
    X["roi_1yr"] = df["ROI1yr"].astype(float).shift(1) if "ROI1yr" in df.columns else lag / lag.shift(365) - 1
    dsh = np.array([(d - HALVINGS[HALVINGS <= d].max()).days if (HALVINGS <= d).any() else np.nan for d in df.index], dtype=float)
    X["cycle_pos"] = np.cos(2 * np.pi * dsh / 1461.0)
    return X


def _make_model(kind: str):
    if kind == "ridge":
        return Ridge(alpha=10.0)
    if kind == "hgbr":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
    raise ValueError(kind)


def walk_forward_predictions(df: pd.DataFrame, cfg: MLConfig) -> pd.Series:
    """Out-of-sample prediction for every day, from models trained only on fully closed labels."""
    X = feature_matrix(df)
    p = df[PRICE_COL].astype(float)
    y = np.log(p.shift(-cfg.horizon) / p)          # label for sample date s closes at s + H
    valid_x = X.notna().all(axis=1)
    n = len(df)
    idx = df.index
    preds = np.full(n, np.nan)
    model, scaler = None, None
    positions = np.arange(n)
    label_closed = positions + cfg.horizon         # position at which sample's label is known
    y_arr = y.to_numpy(float)
    X_arr = X.to_numpy(float)
    vx = valid_x.to_numpy()
    for start in range(0, n, RETRAIN_EVERY):
        r = start                                   # retraining position
        train_mask = vx & (label_closed < r) & np.isfinite(y_arr)
        if train_mask.sum() >= MIN_TRAIN_LABELS:
            Xt, yt = X_arr[train_mask], y_arr[train_mask]
            if cfg.model == "ridge":
                scaler = StandardScaler().fit(Xt)
                model = _make_model(cfg.model).fit(scaler.transform(Xt), yt)
            else:
                scaler = None
                model = _make_model(cfg.model).fit(Xt, yt)
        if model is not None:
            seg = slice(start, min(start + RETRAIN_EVERY, n))
            Xs = X_arr[seg]
            ok = vx[seg]
            if ok.any():
                Xin = Xs[ok]
                out = model.predict(scaler.transform(Xin) if scaler is not None else Xin)
                block = np.full(seg.stop - seg.start, np.nan)
                block[ok] = out
                preds[seg] = block
    return pd.Series(preds, index=idx, name="ml_pred")


def causal_standardise(pred: pd.Series, clip: float) -> pd.Series:
    """Each day's prediction scaled by the expanding mean and std of PRIOR predictions only."""
    past = pred.shift(1)
    mu = past.expanding(min_periods=90).mean()
    sd = past.expanding(min_periods=90).std()
    z = (pred - mu) / sd
    return z.clip(-clip, clip)


def build_ml_features(df: pd.DataFrame, cfg: MLConfig) -> pd.DataFrame:
    """Feature frame for the allocator: base columns plus the causal ML signal f_ml_z."""
    feats = construct_features(df, Params())
    feats["f_ml_z"] = causal_standardise(walk_forward_predictions(df, cfg), cfg.clip_ml)
    return feats


def make_ml_strategy(cfg: MLConfig):
    """Strategy function with the engine's signature. Recomputes the full causal pipeline from
    whatever dataframe it receives, so the leakage probe exercises the training loop itself."""
    def fn(df_window: pd.DataFrame) -> pd.Series:
        if df_window.empty:
            return pd.Series(dtype=float)
        if "f_ml_z" in df_window.columns:
            f = df_window
        else:
            f = build_ml_features(df_window, cfg)
        z = f["f_ml_z"].to_numpy(float)
        n = len(f)
        w = np.empty(n)
        remaining = 1.0
        floor = MIN_WEIGHT * 1.01
        for i in range(n):
            days_left = n - i
            if days_left == 1:
                w[i] = remaining
                break
            zi = z[i] if np.isfinite(z[i]) else 0.0
            m = float(np.exp(cfg.a_ml * zi))
            m = min(max(m, cfg.m_min), cfg.m_max)
            pace = remaining / days_left
            cap = remaining - (days_left - 1) * floor
            wi = min(max(pace * m, floor), cap)
            w[i] = wi
            remaining -= wi
        return pd.Series(w, index=df_window.index)
    fn.cfg = cfg
    return fn
