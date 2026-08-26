"""Machine-learning pipeline for rounds 4 to 9: purged walk-forward prediction, causally mapped
to weights, with selectable learner kinds, allocation shapes and feature matrices.

Causality, the property the tournament probe tests, is preserved by construction everywhere:

- Features are lagged one day and computed from the dataframe the strategy receives, so when
  the probe masks future rows, everything downstream is rebuilt from the masked data.
- The label for a sample dated s closes at s + H, so a model refit on date r trains only on
  samples with s + H < r ("purged" walk-forward, refit every 90 days, expanding window).
- The prediction stream is standardised against the expanding mean and deviation of past
  predictions only.
- Weights come from one of two allocation shapes, both of which keep every weight above the
  floor and each window summing to one by construction.

scikit-learn only: every learner here exists in the tournament's pinned environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from model.strategy import MIN_WEIGHT, PRICE_COL, Params, construct_features

RETRAIN_EVERY = 90        # days between walk-forward refits
MIN_TRAIN_LABELS = 730    # first fit only after this many fully closed labels
HALVINGS = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])
SCALED_KINDS = {"ridge", "enet", "krr", "mlp", "predavg", "stack"}


@dataclass(frozen=True)
class MLConfig:
    model: str = "ridge"      # ridge | hgbr | rf | enet | krr | mlp | gbr_q25 | gbclf | predavg | stack
    horizon: int = 30         # label horizon H in days
    a_ml: float = 1.0         # multiplier strength on the standardised prediction
    clip_ml: float = 3.0
    m_min: float = 0.25
    m_max: float = 5.0
    features: tuple = ()      # empty tuple = all columns of the feature matrix; else the named subset
    shape: str = "pace"       # "pace" (remaining-budget pacing) or "tailpay" (boost paid by the tail)
    b_quad: float = 0.0       # convexity: log m = a z + b z |z|
    hgbr_depth: int = 3
    hgbr_lr: float = 0.05
    matrix: str = "v1"        # "v1" (round 4 set) or "v2" (round 9 expansion)


def fetch_fear_greed(path: str = "data/fear_greed.csv") -> pd.Series:
    """Fear and Greed index from the alternative.me open API, cached to a committed CSV.
    Refreshes when online; falls back to the cache when not."""
    try:
        import requests
        r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=60)
        r.raise_for_status()
        rows = r.json()["data"]
        s = pd.Series({pd.to_datetime(int(x["timestamp"]), unit="s").normalize(): float(x["value"]) for x in rows})
        s = s.sort_index()
        s.rename("fear_greed").to_csv(path)
    except Exception:
        if not os.path.exists(path):
            raise
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.name = "fear_greed"
    return s


def feature_matrix(df: pd.DataFrame, matrix: str = "v1") -> pd.DataFrame:
    """The registered feature sets. v1 is the round-4 matrix; v2 adds the round-9 expansion.
    Every column uses data up to the previous day only."""
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

    def momentum(col, short, long):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        s = df[col].astype(float).shift(1)
        return s.rolling(short, min_periods=max(5, short * 2 // 3)).mean() / s.rolling(long, min_periods=long // 2).mean() - 1

    X["adr_mom"] = momentum("AdrActCnt", 30, 365)
    X["fee_mom"] = momentum("FeeTotNtv", 30, 365)
    X["hash_mom"] = momentum("HashRate", 30, 365)
    X["roi_1yr"] = df["ROI1yr"].astype(float).shift(1) if "ROI1yr" in df.columns else lag / lag.shift(365) - 1
    dsh = np.array([(d - HALVINGS[HALVINGS <= d].max()).days if (HALVINGS <= d).any() else np.nan for d in df.index], dtype=float)
    X["cycle_pos"] = np.cos(2 * np.pi * dsh / 1461.0)

    if matrix == "v2":
        X["vol_usd_mom"] = momentum("volume_reported_spot_usd_1d", 30, 365)
        X["tx_mom"] = momentum("TxCnt", 30, 365)
        X["hash_mom_60"] = momentum("HashRate", 60, 365)
        X["fee_mom_60"] = momentum("FeeTotNtv", 60, 365)
        if "SplyExNtv" in df.columns and "SplyCur" in df.columns:
            share = (df["SplyExNtv"] / df["SplyCur"]).astype(float).shift(1)
            X["exch_share_chg"] = share - share.shift(90)
        else:
            X["exch_share_chg"] = np.nan
        if "fear_greed" in df.columns:
            fg = df["fear_greed"].astype(float).shift(1)
            mu = fg.rolling(365, min_periods=180).mean()
            sd = fg.rolling(365, min_periods=180).std()
            X["fg_z"] = (fg - mu) / sd
        else:
            X["fg_z"] = np.nan
    return X


def _make_model(kind: str, cfg: "MLConfig" = None):
    if kind == "ridge":
        return Ridge(alpha=10.0)
    if kind == "hgbr":
        d = cfg.hgbr_depth if cfg else 3
        lr = cfg.hgbr_lr if cfg else 0.05
        return HistGradientBoostingRegressor(max_depth=d, learning_rate=lr, max_iter=200, random_state=0)
    if kind == "rf":
        return RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1)
    if kind == "enet":
        return ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)
    if kind == "krr":
        return KernelRidge(kernel="rbf", alpha=1.0, gamma=0.1)
    if kind == "mlp":
        return MLPRegressor(hidden_layer_sizes=(64, 32), random_state=0, max_iter=400, early_stopping=False)
    if kind == "gbr_q25":
        return GradientBoostingRegressor(loss="quantile", alpha=0.25, max_depth=3, n_estimators=200,
                                         learning_rate=0.05, random_state=0)
    if kind == "gbclf":
        return GradientBoostingClassifier(max_depth=3, n_estimators=200, learning_rate=0.05, random_state=0)
    if kind == "stack":
        return StackingRegressor(
            estimators=[("ridge", Ridge(alpha=10.0)),
                        ("hgbr", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0))],
            final_estimator=Ridge(alpha=1.0), cv=TimeSeriesSplit(3))
    raise ValueError(kind)


def walk_forward_predictions(df: pd.DataFrame, cfg: MLConfig) -> pd.Series:
    """Out-of-sample prediction for every day, from models trained only on fully closed labels."""
    X = feature_matrix(df, cfg.matrix)
    if cfg.features:
        X = X[list(cfg.features)]
    p = df[PRICE_COL].astype(float)
    y = np.log(p.shift(-cfg.horizon) / p)          # label for sample date s closes at s + H
    if cfg.model == "gbclf":
        y = (y > 0).astype(float).where(y.notna())
    valid_x = X.notna().all(axis=1)
    n = len(df)
    preds = np.full(n, np.nan)
    fitted, scaler = None, None
    label_closed = np.arange(n) + cfg.horizon
    y_arr = y.to_numpy(float)
    X_arr = X.to_numpy(float)
    vx = valid_x.to_numpy()
    kinds = ["ridge", "hgbr"] if cfg.model == "predavg" else [cfg.model]
    for start in range(0, n, RETRAIN_EVERY):
        train_mask = vx & (label_closed < start) & np.isfinite(y_arr)
        if train_mask.sum() >= MIN_TRAIN_LABELS:
            Xt, yt = X_arr[train_mask], y_arr[train_mask]
            scaler = StandardScaler().fit(Xt) if cfg.model in SCALED_KINDS else None
            Xin = scaler.transform(Xt) if scaler is not None else Xt
            fitted = [(k, _make_model(k, cfg).fit(Xin, yt)) for k in kinds]
        if fitted is not None:
            seg = slice(start, min(start + RETRAIN_EVERY, n))
            ok = vx[seg]
            if ok.any():
                Xs = X_arr[seg][ok]
                Xin = scaler.transform(Xs) if scaler is not None else Xs
                outs = []
                for k, m in fitted:
                    outs.append(m.predict_proba(Xin)[:, 1] if k == "gbclf" else m.predict(Xin))
                out = np.mean(outs, axis=0)
                block = np.full(seg.stop - seg.start, np.nan)
                block[ok] = out
                preds[seg] = block
    return pd.Series(preds, index=df.index, name="ml_pred")


def causal_standardise(pred: pd.Series, clip: float) -> pd.Series:
    """Each day's prediction scaled by the expanding mean and std of PRIOR predictions only."""
    past = pred.shift(1)
    mu = past.expanding(min_periods=90).mean()
    sd = past.expanding(min_periods=90).std()
    return ((pred - mu) / sd).clip(-clip, clip)


def build_ml_features(df: pd.DataFrame, cfg: MLConfig) -> pd.DataFrame:
    """Feature frame for the allocator: base columns plus the causal ML signal f_ml_z."""
    feats = construct_features(df, Params())
    feats["f_ml_z"] = causal_standardise(walk_forward_predictions(df, cfg), cfg.clip_ml)
    return feats


def _multiplier(z: float, cfg: MLConfig) -> float:
    log_m = cfg.a_ml * z + cfg.b_quad * z * abs(z)
    return min(max(float(np.exp(log_m)), cfg.m_min), cfg.m_max)


def _allocate_pace(z: np.ndarray, cfg: MLConfig) -> np.ndarray:
    """Remaining-budget pacing: each day spends the even pace of what is left, scaled by m."""
    n = len(z)
    w = np.empty(n)
    remaining = 1.0
    floor = MIN_WEIGHT * 1.01
    for i in range(n):
        days_left = n - i
        if days_left == 1:
            w[i] = remaining
            break
        m = _multiplier(z[i] if np.isfinite(z[i]) else 0.0, cfg)
        pace = remaining / days_left
        cap = remaining - (days_left - 1) * floor
        wi = min(max(pace * m, floor), cap)
        w[i] = wi
        remaining -= wi
    return w


def _allocate_tailpay(z: np.ndarray, cfg: MLConfig) -> np.ndarray:
    """Reference-style shape: start uniform; a boosted day is paid for by equally reducing the
    days in the window's second half that have not yet been reached. Only future days are ever
    reduced, so the shape is causal; if the tail cannot fund a boost, the boost is skipped."""
    n = len(z)
    base = 1.0 / n
    w = np.full(n, base)
    floor = MIN_WEIGHT * 1.01
    tail_start = n // 2
    for i in range(n):
        m = _multiplier(z[i] if np.isfinite(z[i]) else 0.0, cfg)
        if m <= 1.0:
            continue
        desired = w[i] * m
        excess = desired - w[i]
        payers = np.arange(max(tail_start, i + 1), n)
        if payers.size == 0:
            continue
        cut = excess / payers.size
        if np.all(w[payers] - cut >= floor):
            w[i] = desired
            w[payers] -= cut
    # every boost adds exactly what the payers give up, so the sum is 1 by construction;
    # floating-point drift (~1e-15) is absorbed by the last day rather than renormalised,
    # because dividing by a window sum is the forward-looking pattern this project bans
    w[-1] += 1.0 - w.sum()
    return w


def make_ml_strategy(cfg: MLConfig):
    """Strategy function with the engine's signature. Recomputes the full causal pipeline from
    whatever dataframe it receives, so the leakage probe exercises the training loop itself."""
    def fn(df_window: pd.DataFrame) -> pd.Series:
        if df_window.empty:
            return pd.Series(dtype=float)
        f = df_window if "f_ml_z" in df_window.columns else build_ml_features(df_window, cfg)
        z = f["f_ml_z"].to_numpy(float)
        w = _allocate_pace(z, cfg) if cfg.shape == "pace" else _allocate_tailpay(z, cfg)
        return pd.Series(w, index=df_window.index)
    fn.cfg = cfg
    return fn
