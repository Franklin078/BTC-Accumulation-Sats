"""Dynamic DCA model: causal "dip-pacing" allocation.

Design rules (these are the rules the tournament validator enforces, restated as code invariants):

1. The weight for day t is decided from information available at the close of day t-1.
   Every feature is lagged one day. Nothing after t is ever read.
2. Weights live in the clipped simplex: w_t >= MIN_WEIGHT and sum(w) = 1 over the window.
   This holds by construction through remaining-budget pacing, not by dividing by a
   window sum after the fact (which would be forward-looking).
3. The model is a deterministic function of a handful of interpretable features.

How allocation works
--------------------
Start each 12-month window with the full budget (1.0). On day t the "pace" is the budget
left divided by the days left. The model scales that pace by a multiplier m_t >= 0 that
rises when Bitcoin looks cheap relative to its own recent history and falls when it looks
expensive. The last day of the window takes whatever budget is left, so the window always
spends exactly 1.0. Because the multiplier is bounded, the budget can never be exhausted
early and no day can fall below MIN_WEIGHT.

Features (all lagged one day; "p" is PriceUSD_coinmetrics)
----------------------------------------------------------
ma_gap   : p / MA_L(p) - 1                 distance from the long moving average
drawdown : p / max(p, last 365 days) - 1   depth of the current drawdown from the 1-year high
win_rel  : p / mean(p since window start) - 1   price relative to the window's own running average
mvrv_z   : z-score of CapMVRVCur over 365 days (on-chain valuation), optional

log m_t = -a_ma * (ma_gap - ma0) - a_dd * (drawdown + dd0) - a_win * win_rel - a_mvrv * mvrv_z
(dd0 and ma0 are centering offsets: the drawdown term is neutral at depth dd0, so the model
holds back when the drawdown is shallower than dd0 and leans in when it is deeper)
m_t is clipped to [m_min, m_max].
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"
MIN_WEIGHT = 1e-5


@dataclass(frozen=True)
class Params:
    ma_len: int = 200
    a_ma: float = 2.0
    a_dd: float = 2.0
    a_win: float = 2.0
    a_mvrv: float = 0.0
    dd0: float = 0.0         # drawdown depth at which the drawdown term is neutral
    ma0: float = 0.0         # ma_gap at which the MA term is neutral
    m_min: float = 0.25
    m_max: float = 4.0
    clip_ma: float = 0.6     # |ma_gap| clipped here
    clip_dd: float = 0.8     # |drawdown| clipped here
    clip_win: float = 0.5    # |win_rel| clipped here
    clip_mvrv: float = 3.0   # |mvrv term| clipped here
    mvrv_mode: str = "level" # "level": (MVRV - mvrv0); "z": 365-day z-score of MVRV
    mvrv0: float = 1.8       # neutral MVRV level for mvrv_mode == "level"
    a_r7: float = 0.0        # weight on the trailing 7-day return (negative return -> buy more)
    clip_r7: float = 0.3
    ma_asym: bool = False    # True: the MA term only acts when price is below the MA (like the official reference)
    bias: float = 0.0        # constant added to log m (a mild front-loading prior when > 0)
    bias_above: float | None = None  # if set: bias used when price is above the long MA (uptrend); `bias` then applies below it
    win_mode: str = "mean"   # "mean": relative to the window's running mean; "start": relative to the window's first price
    same_day: bool = False   # False: features use the close of t-1 (strict lag). True: close of t,
                             # the convention of the official 2025 tournament reference strategy.

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Params()


# ---------------------------------------------------------------- features
def construct_features(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    """Global (history-aware) features, every one lagged by one day.

    Safe to call on the full history or on a window slice; a slice simply has less history
    for the rolling statistics (min_periods handles the warm-up).
    """
    out = pd.DataFrame(index=df.index)
    p = df[PRICE_COL].astype(float)
    out[PRICE_COL] = p
    lag = p if params.same_day else p.shift(1)
    ma = lag.rolling(params.ma_len, min_periods=max(10, params.ma_len // 4)).mean()
    out["f_ma_gap"] = (lag / ma - 1.0)
    hi = lag.rolling(365, min_periods=30).max()
    out["f_drawdown"] = (lag / hi - 1.0)
    if MVRV_COL in df.columns:
        mv = df[MVRV_COL].astype(float).shift(1)
        mu = mv.rolling(365, min_periods=90).mean()
        sd = mv.rolling(365, min_periods=90).std()
        out["f_mvrv_z"] = (mv - mu) / sd if params.mvrv_mode == "z" else (mv - params.mvrv0)
    else:
        out["f_mvrv_z"] = np.nan
    out["f_r7"] = lag / lag.shift(7) - 1.0
    out["f_lag_price"] = lag
    return out


FEATURE_COLS = ["f_ma_gap", "f_drawdown", "f_mvrv_z", "f_r7", "f_lag_price"]


# ---------------------------------------------------------------- allocation core
def allocate(ma_gap: np.ndarray, drawdown: np.ndarray, mvrv_z: np.ndarray, r7: np.ndarray, lag_price: np.ndarray,
             params: Params, min_weight: float = MIN_WEIGHT) -> np.ndarray:
    """Remaining-budget pacing. Pure numpy, sequential by necessity (day t depends on t-1)."""
    n = len(ma_gap)
    w = np.empty(n)
    if n == 0:
        return w
    # The validator tests "w < MIN_WEIGHT" strictly, so the floor carries a 1 per cent safety margin
    # against floating-point dust on the last day's remainder.
    min_weight = min_weight * 1.01
    remaining = 1.0
    run_sum = 0.0   # running sum of lagged prices inside the window
    run_cnt = 0
    first_price = np.nan
    for i in range(n):
        days_left = n - i
        if days_left == 1:
            w[i] = remaining
            break
        # within-window relative price, from lagged prices only
        lp = lag_price[i]
        if run_cnt > 0 and np.isfinite(lp):
            ref = (run_sum / run_cnt) if params.win_mode == "mean" else first_price
            win_rel = lp / ref - 1.0
        else:
            win_rel = 0.0
        if run_cnt == 0 and np.isfinite(lp):
            first_price = lp
        if np.isfinite(lp):
            run_sum += lp
            run_cnt += 1
        g = ma_gap[i] if np.isfinite(ma_gap[i]) else 0.0
        d = drawdown[i] if np.isfinite(drawdown[i]) else 0.0
        z = mvrv_z[i] if np.isfinite(mvrv_z[i]) else 0.0
        g = min(max(g, -params.clip_ma), params.clip_ma)
        if params.ma_asym and g > params.ma0:
            g = params.ma0
        d = min(max(d, -params.clip_dd), params.clip_dd)
        r = min(max(win_rel, -params.clip_win), params.clip_win)
        z = min(max(z, -params.clip_mvrv), params.clip_mvrv)
        q = r7[i] if np.isfinite(r7[i]) else 0.0
        q = min(max(q, -params.clip_r7), params.clip_r7)
        b = params.bias
        if params.bias_above is not None and g > 0:
            b = params.bias_above
        log_m = b - (params.a_ma * (g - params.ma0) + params.a_dd * (d + params.dd0) + params.a_win * r + params.a_mvrv * z + params.a_r7 * q)
        m = np.exp(log_m)
        m = min(max(m, params.m_min), params.m_max)
        pace = remaining / days_left
        cap = remaining - (days_left - 1) * min_weight
        wi = min(max(pace * m, min_weight), cap)
        w[i] = wi
        remaining -= wi
    return w


def compute_weights(df_window: pd.DataFrame, params: Params = DEFAULT) -> pd.Series:
    """Template-compatible entry point: 12-month slice in, weights out (sum to 1, >= MIN_WEIGHT)."""
    if df_window.empty:
        return pd.Series(dtype=float)
    if all(c in df_window.columns for c in FEATURE_COLS):
        f = df_window
    else:
        f = construct_features(df_window, params)
    w = allocate(f["f_ma_gap"].to_numpy(float), f["f_drawdown"].to_numpy(float),
                 f["f_mvrv_z"].to_numpy(float), f["f_r7"].to_numpy(float), f["f_lag_price"].to_numpy(float), params)
    return pd.Series(w, index=df_window.index)


def make_strategy(params: Params):
    """Return a function with the signature the Trilemma engine expects."""
    def _fn(df_window: pd.DataFrame) -> pd.Series:
        return compute_weights(df_window, params)
    _fn.params = params
    return _fn


# ---------------------------------------------------------------- fast evaluation for model selection
def window_starts(index: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """Exactly the upstream window generator: daily starts, 1-year offset, end within range."""
    offset = pd.DateOffset(years=1)
    starts = pd.date_range(pd.to_datetime(start), pd.to_datetime(end) - offset, freq="D")
    return pd.DatetimeIndex([s for s in starts if s + offset <= pd.to_datetime(end)])


def fast_spd_table(features: pd.DataFrame, params: Params, start: str, end: str) -> pd.DataFrame:
    """Same arithmetic as template.prelude_template.compute_cycle_spd, on precomputed arrays.

    Used only for the pre-registered model-selection grid. Every reported number is
    recomputed afterwards with the upstream function itself.
    """
    idx = features.index
    price = features[PRICE_COL].to_numpy(float)
    g = features["f_ma_gap"].to_numpy(float)
    d = features["f_drawdown"].to_numpy(float)
    z = features["f_mvrv_z"].to_numpy(float)
    q = features["f_r7"].to_numpy(float)
    lp = features["f_lag_price"].to_numpy(float)
    pos = {t: i for i, t in enumerate(idx)}
    offset = pd.DateOffset(years=1)
    rows = []
    for ws in window_starts(idx, start, end):
        we = ws + offset
        a = pos[ws]; b = pos[we] + 1  # inclusive slice like df.loc[ws:we]
        pr = price[a:b]
        w = allocate(g[a:b], d[a:b], z[a:b], q[a:b], lp[a:b], params)
        inv = 1e8 / pr
        mn, mx = inv.min(), inv.max(); span = mx - mn
        uni = inv.mean(); dyn = (w * inv).sum()
        rows.append((ws, (uni - mn) / span * 100, (dyn - mn) / span * 100))
    out = pd.DataFrame(rows, columns=["window_start", "uniform_percentile", "dynamic_percentile"]).set_index("window_start")
    return out
