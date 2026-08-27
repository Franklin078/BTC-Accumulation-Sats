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
    matrix: str = "v1"        # "v1" (round 4 set), "v2" (round 9 expansion) or "v3" (round 10)
    horizons: tuple = ()      # non-empty: a committee; each horizon's forecast is causally
                              # standardised, then the z-scores are averaged with equal weights
    sample_halflife: float = 0.0  # >0: training samples decay with age, half-life in days
    a_pos: float = 0.0        # asymmetric multiplier: response to positive signals
    a_neg: float = 0.0        # response to negative signals (both fall back to a_ml when 0)
    blocks: tuple = ()        # non-empty: a block committee ((family, (feature, ...)), ...);
                              # one model per family, z-scores combined with no fitted weights
    combine: str = "avg"      # block combination: "avg" (equal-weight mean of finite family
                              # z-scores) or "gate" (the mean passes through only on days when
                              # at least three families have finite signals sharing its sign)
    cond_features: tuple = () # non-empty: a conditioner model on these features adds f_cond_z
                              # to the frame; it scales how hard the engine tilts, never which way
    cond_gain: float = 0.0    # g in a_eff = a x (1 + g tanh(z_cond)); 0 disables conditioning
    cond_depth: int = 2       # tree depth of the conditioner model only; separate from
                              # hgbr_depth so conditioning an engine never alters the engine
    phase_depth: float = 0.0  # >0: the response scales by (1 + phase_depth) during the
                              # registered phase of the halving cycle; a pure calendar rule,
                              # trained on nothing
    phase_def: str = "late"   # "late": days beyond 730 after the most recent halving (the
                              # historically cheap half); "mid": days 366 to 1095, where the
                              # cycle feature is negative (contains both historical extremes)
    phase_target: str = "buy" # what the phase scales: "buy" (a_pos only), "slopes" (both
                              # slopes), or "ceiling" (m_max)


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


def _col(df: pd.DataFrame, name: str):
    """Resolve a CoinMetrics column under either naming convention: the live CSV uses bare
    names; the frozen tournament parquet suffixes them with _coinmetrics."""
    if name in df.columns:
        return df[name]
    if f"{name}_coinmetrics" in df.columns:
        return df[f"{name}_coinmetrics"]
    return None


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
    mv = _col(df, "CapMVRVCur")
    X["mvrv"] = mv.astype(float).shift(1) if mv is not None else np.nan
    X["ret_7"] = lag / lag.shift(7) - 1
    X["ret_30"] = lag / lag.shift(30) - 1
    X["ret_90"] = lag / lag.shift(90) - 1
    r = np.log(lag).diff()
    X["vol_30"] = r.rolling(30, min_periods=15).std() * np.sqrt(365)

    def momentum(col, short, long):
        series = _col(df, col)
        if series is None:
            return pd.Series(np.nan, index=df.index)
        s = series.astype(float).shift(1)
        return s.rolling(short, min_periods=max(5, short * 2 // 3)).mean() / s.rolling(long, min_periods=long // 2).mean() - 1

    X["adr_mom"] = momentum("AdrActCnt", 30, 365)
    X["fee_mom"] = momentum("FeeTotNtv", 30, 365)
    X["hash_mom"] = momentum("HashRate", 30, 365)
    roi = _col(df, "ROI1yr")
    X["roi_1yr"] = roi.astype(float).shift(1) if roi is not None else lag / lag.shift(365) - 1
    dsh = np.array([(d - HALVINGS[HALVINGS <= d].max()).days if (HALVINGS <= d).any() else np.nan for d in df.index], dtype=float)
    X["cycle_pos"] = np.cos(2 * np.pi * dsh / 1461.0)

    if matrix == "v3":
        X["vol_usd_mom"] = momentum("volume_reported_spot_usd_1d", 30, 365)
        X["tx_mom"] = momentum("TxCnt", 30, 365)
        sply_ex3, sply3 = _col(df, "SplyExNtv"), _col(df, "SplyCur")
        if sply_ex3 is not None and sply3 is not None:
            share3 = (sply_ex3 / sply3).astype(float).shift(1)
            X["exch_share_chg"] = share3 - share3.shift(90)
        else:
            X["exch_share_chg"] = np.nan

    if matrix == "v2":
        X["vol_usd_mom"] = momentum("volume_reported_spot_usd_1d", 30, 365)
        X["tx_mom"] = momentum("TxCnt", 30, 365)
        X["hash_mom_60"] = momentum("HashRate", 60, 365)
        X["fee_mom_60"] = momentum("FeeTotNtv", 60, 365)
        sply_ex, sply = _col(df, "SplyExNtv"), _col(df, "SplyCur")
        if sply_ex is not None and sply is not None:
            share = (sply_ex / sply).astype(float).shift(1)
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
            final_estimator=Ridge(alpha=1.0), cv=3)  # contiguous unshuffled folds; all samples
            # in a training set have labels closed before the refit date, so internal folds
            # cannot leak future information regardless of their arrangement
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
    if n > MIN_TRAIN_LABELS + RETRAIN_EVERY and vx.sum() == 0:
        raise ValueError("no valid feature rows on a frame long enough to train: the input "
                         "dataframe is probably missing the CoinMetrics columns this model "
                         "needs (check column naming); refusing to degenerate to uniform weights")
    kinds = ["ridge", "hgbr"] if cfg.model == "predavg" else [cfg.model]
    for start in range(0, n, RETRAIN_EVERY):
        train_mask = vx & (label_closed < start) & np.isfinite(y_arr)
        if train_mask.sum() >= MIN_TRAIN_LABELS:
            Xt, yt = X_arr[train_mask], y_arr[train_mask]
            scaler = StandardScaler().fit(Xt) if cfg.model in SCALED_KINDS else None
            Xin = scaler.transform(Xt) if scaler is not None else Xt
            fit_kw = {}
            if cfg.sample_halflife > 0:
                age = start - np.flatnonzero(train_mask)   # days since each sample's date
                fit_kw["sample_weight"] = 0.5 ** (age / cfg.sample_halflife)
            fitted = [(k, _make_model(k, cfg).fit(Xin, yt, **fit_kw)) for k in kinds]
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


def _committee_signal(df: pd.DataFrame, cfg: MLConfig) -> pd.Series:
    """The causal ML signal under any of the three signal structures.

    Plain: one walk-forward forecast, causally standardised. Horizons committee: each horizon's
    forecast standardised on its own history, then averaged with fixed equal weights. Block
    committee: one model per feature family, each family's forecast standardised on its own
    history, combined by the equal-weight mean of the finite family z-scores, optionally gated
    to zero on days when fewer than three families share the mean's sign. No combination weight
    is ever fitted, so none of these structures adds selection on top of the registered grid."""
    from dataclasses import replace
    if cfg.blocks:
        zs = [causal_standardise(
                  walk_forward_predictions(df, replace(cfg, features=tuple(f), blocks=(), horizons=())),
                  cfg.clip_ml).rename(name)
              for name, f in cfg.blocks]
        Z = pd.concat(zs, axis=1)
        avg = Z.mean(axis=1)  # pandas mean skips NaN: the mean of the finite family signals
        if cfg.combine == "gate":
            pos = (Z > 0).sum(axis=1)
            neg = (Z < 0).sum(axis=1)
            agree = ((avg > 0) & (pos >= 3)) | ((avg < 0) & (neg >= 3))
            avg = avg.where(agree, 0.0)
        elif cfg.combine != "avg":
            raise ValueError(f"unknown block combination {cfg.combine!r}")
        return avg.clip(-cfg.clip_ml, cfg.clip_ml)
    if cfg.horizons:
        zs = [causal_standardise(walk_forward_predictions(df, replace(cfg, horizon=h, horizons=())), cfg.clip_ml)
              for h in cfg.horizons]
        return (sum(zs) / len(zs)).clip(-cfg.clip_ml, cfg.clip_ml)
    return causal_standardise(walk_forward_predictions(df, cfg), cfg.clip_ml)


def build_ml_features(df: pd.DataFrame, cfg: MLConfig) -> pd.DataFrame:
    """Feature frame for the allocator: base columns plus the causal ML signal f_ml_z, and,
    for a conditioned configuration, the conditioner signal f_cond_z built the same causal way
    from its own feature set."""
    from dataclasses import replace
    feats = construct_features(df, Params())
    feats["f_ml_z"] = _committee_signal(df, cfg)
    if cfg.cond_features:
        # the conditioner takes its own depth (cond_depth), never the engine's hgbr_depth:
        # the two are registered separately and conditioning must leave the engine untouched
        cond_cfg = replace(cfg, features=tuple(cfg.cond_features), cond_features=(),
                           blocks=(), horizons=(), hgbr_depth=cfg.cond_depth)
        feats["f_cond_z"] = causal_standardise(walk_forward_predictions(df, cond_cfg), cfg.clip_ml)
    if cfg.phase_depth > 0:
        # the phase indicator is a pure function of each row's calendar date: 1.0 inside the
        # registered cycle phase, 0.0 outside it and before the first halving; the depth is
        # applied in the allocator, so both grid depths share one frame
        dsh = np.array([(d - HALVINGS[HALVINGS <= d].max()).days if (HALVINGS <= d).any() else np.nan
                        for d in df.index], dtype=float)
        with np.errstate(invalid="ignore"):
            if cfg.phase_def == "late":
                ind = dsh > 730.5
            elif cfg.phase_def == "mid":
                ind = (dsh > 365.25) & (dsh < 1095.75)
            else:
                raise ValueError(f"unknown phase definition {cfg.phase_def!r}")
        feats["f_phase"] = np.where(np.isfinite(dsh) & ind, 1.0, 0.0)
    return feats


def _multiplier(z: float, cfg: MLConfig, zc: float = 0.0, acc: float = 0.0) -> float:
    if cfg.a_pos > 0 or cfg.a_neg > 0:
        a = (cfg.a_pos if z >= 0 else cfg.a_neg) or cfg.a_ml
    else:
        a = cfg.a_ml
    if cfg.cond_gain > 0:
        # the conditioner scales the strength of the tilt, bounded by tanh, and never its
        # direction; with the gain at most 0.5 the effective strength stays within half to
        # one-and-a-half times the base, and the ceiling and floor still apply after it
        a = a * (1.0 + cfg.cond_gain * float(np.tanh(zc)))
    m_max = cfg.m_max
    if cfg.phase_depth > 0 and acc > 0.0:
        # calendar-phase scaling: acc is the day's 0/1 phase indicator from the frame; the
        # scaled quantity is the registered target and nothing else
        ph = 1.0 + cfg.phase_depth * acc
        if cfg.phase_target == "slopes":
            a = a * ph
        elif cfg.phase_target == "buy":
            if z >= 0:
                a = a * ph
        elif cfg.phase_target == "ceiling":
            m_max = m_max * ph
        else:
            raise ValueError(f"unknown phase target {cfg.phase_target!r}")
    log_m = a * z + cfg.b_quad * z * abs(z)
    return min(max(float(np.exp(log_m)), cfg.m_min), m_max)


def _allocate_pace(z: np.ndarray, cfg: MLConfig, zc: np.ndarray = None,
                   acc: np.ndarray = None) -> np.ndarray:
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
        zci = zc[i] if zc is not None and np.isfinite(zc[i]) else 0.0
        acci = acc[i] if acc is not None and np.isfinite(acc[i]) else 0.0
        m = _multiplier(z[i] if np.isfinite(z[i]) else 0.0, cfg, zci, acci)
        pace = remaining / days_left
        cap = remaining - (days_left - 1) * floor
        wi = min(max(pace * m, floor), cap)
        w[i] = wi
        remaining -= wi
    return w


def _allocate_tailpay(z: np.ndarray, cfg: MLConfig, zc: np.ndarray = None,
                      acc: np.ndarray = None) -> np.ndarray:
    """Reference-style shape: start uniform; a boosted day is paid for by equally reducing the
    days in the window's second half that have not yet been reached. Only future days are ever
    reduced, so the shape is causal; if the tail cannot fund a boost, the boost is skipped."""
    n = len(z)
    base = 1.0 / n
    w = np.full(n, base)
    floor = MIN_WEIGHT * 1.01
    tail_start = n // 2
    for i in range(n):
        zci = zc[i] if zc is not None and np.isfinite(zc[i]) else 0.0
        acci = acc[i] if acc is not None and np.isfinite(acc[i]) else 0.0
        m = _multiplier(z[i] if np.isfinite(z[i]) else 0.0, cfg, zci, acci)
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
        zc = None
        if cfg.cond_features:
            if "f_cond_z" not in f.columns:
                # a conditioned strategy scored against a frame without its conditioner column
                # would silently fall back to plain v5; refuse, the same way the harness refuses
                # frames that lack f_ml_z (decision log D35)
                raise ValueError("conditioned strategy given a frame without f_cond_z; build "
                                 "the frame with build_ml_features under this configuration")
            zc = f["f_cond_z"].to_numpy(float)
        acc = None
        if cfg.phase_depth > 0:
            if "f_phase" not in f.columns:
                # a phase-modulated strategy scored against a frame without its phase column
                # would silently fall back to the unmodulated model; refuse, as with f_cond_z
                raise ValueError("phase-modulated strategy given a frame without f_phase; "
                                 "build the frame with build_ml_features under this configuration")
            acc = f["f_phase"].to_numpy(float)
        w = (_allocate_pace(z, cfg, zc, acc) if cfg.shape == "pace"
             else _allocate_tailpay(z, cfg, zc, acc))
        return pd.Series(w, index=df_window.index)
    fn.cfg = cfg
    return fn
