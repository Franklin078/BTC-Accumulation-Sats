import numpy as np, pandas as pd
MIN_WEIGHT = 1e-5
# Dynamic DCA by causal dip-pacing. Franklin Kipkorir, UTS MDataScQF x Trilemma Foundation, 2026.
# Only numpy and pandas (already imported in the Prelude). Every feature is lagged one day.
# Weights come from remaining-budget pacing, so w_t >= MIN_WEIGHT and sum(w) = 1 hold by construction.

P_MA_LEN = 200
P_A_MA = 1.0
P_A_DD = 0.5
P_A_MVRV = 1.0
P_BIAS = 0.2
P_MVRV0 = 1.8
P_M_MIN = 0.25
P_M_MAX = 5.0
P_CLIP_MA = 0.6
P_CLIP_DD = 0.8
P_CLIP_MVRV = 3.0
P_MA_ASYM = True


def construct_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price, the long moving average, and three lagged features (no look-ahead)."""
    out = pd.DataFrame(index=df.index)
    p = df['PriceUSD_coinmetrics'].astype(float)
    lag = p.shift(1)
    ma = lag.rolling(P_MA_LEN, min_periods=max(10, P_MA_LEN // 4)).mean()
    out['PriceUSD_coinmetrics'] = p
    out['ma'] = ma
    out['f_ma_gap'] = lag / ma - 1.0
    out['f_drawdown'] = lag / lag.rolling(365, min_periods=30).max() - 1.0
    mv_col = 'CapMVRVCur_coinmetrics' if 'CapMVRVCur_coinmetrics' in df.columns else ('CapMVRVCur' if 'CapMVRVCur' in df.columns else None)
    out['f_mvrv'] = (df[mv_col].astype(float).shift(1) - P_MVRV0) if mv_col else np.nan
    return out


def compute_weights(df_window: pd.DataFrame) -> pd.Series:
    """Remaining-budget pacing with a bounded multiplier. Day t uses data to the close of t-1."""
    if df_window.empty:
        return pd.Series(dtype=float)
    f = df_window if all(c in df_window.columns for c in ('f_ma_gap', 'f_drawdown', 'f_mvrv')) else construct_features(df_window)
    g_arr = f['f_ma_gap'].to_numpy(float)
    d_arr = f['f_drawdown'].to_numpy(float)
    z_arr = f['f_mvrv'].to_numpy(float)
    n = len(f)
    w = np.empty(n)
    floor = MIN_WEIGHT * 1.01  # the validator tests "w < MIN_WEIGHT" strictly; keep a margin above it
    remaining = 1.0
    for i in range(n):
        days_left = n - i
        if days_left == 1:
            w[i] = remaining
            break
        g = g_arr[i] if np.isfinite(g_arr[i]) else 0.0
        d = d_arr[i] if np.isfinite(d_arr[i]) else 0.0
        z = z_arr[i] if np.isfinite(z_arr[i]) else 0.0
        g = min(max(g, -P_CLIP_MA), P_CLIP_MA)
        if P_MA_ASYM and g > 0.0:
            g = 0.0
        d = min(max(d, -P_CLIP_DD), P_CLIP_DD)
        z = min(max(z, -P_CLIP_MVRV), P_CLIP_MVRV)
        log_m = P_BIAS - (P_A_MA * g + P_A_DD * d + P_A_MVRV * z)
        m = min(max(np.exp(log_m), P_M_MIN), P_M_MAX)
        pace = remaining / days_left
        cap = remaining - (days_left - 1) * floor
        wi = min(max(pace * m, floor), cap)
        w[i] = wi
        remaining -= wi
    return pd.Series(w, index=df_window.index)
