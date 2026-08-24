"""Part 2 of the notebook build: the tournament 3-cell submission and the educational notebook.
Imported by build_notebooks.py; do not run directly."""
import json
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P = json.load(open("model/final_params.json"))
RES = json.load(open("output/results_summary.json")) if os.path.exists("output/results_summary.json") else {}

TOURNAMENT_CELL = '''# Dynamic DCA by causal dip-pacing. Franklin Kipkorir, UTS MDataScQF x Trilemma Foundation, 2026.
# Only numpy and pandas (already imported in the Prelude). Every feature is lagged one day.
# Weights come from remaining-budget pacing, so w_t >= MIN_WEIGHT and sum(w) = 1 hold by construction.

P_MA_LEN = {ma_len}
P_A_MA = {a_ma}
P_A_DD = {a_dd}
P_A_MVRV = {a_mvrv}
P_BIAS = {bias}
P_MVRV0 = {mvrv0}
P_M_MIN = {m_min}
P_M_MAX = {m_max}
P_CLIP_MA = {clip_ma}
P_CLIP_DD = {clip_dd}
P_CLIP_MVRV = {clip_mvrv}
P_MA_ASYM = {ma_asym}


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
'''


def build_tournament_notebook():
    src = nbf.read("tournament_2025/model_development_template.ipynb", as_version=4)
    code_cells = [i for i, c in enumerate(src.cells) if c.cell_type == "code"]
    assert len(code_cells) == 3, "template must have exactly 3 code cells"
    cell = TOURNAMENT_CELL.format(**{k: P[k] for k in ["ma_len", "a_ma", "a_dd", "a_mvrv", "bias", "mvrv0", "m_min", "m_max", "clip_ma", "clip_dd", "clip_mvrv", "ma_asym"]})
    src.cells[code_cells[1]].source = cell
    for i in code_cells:
        src.cells[i].outputs = []
        src.cells[i].execution_count = None
    nbf.write(src, "tournament_2025/btc_accumulation_model.ipynb")
    print("wrote tournament_2025/btc_accumulation_model.ipynb")
    open("tournament_2025/model_cell.py", "w", encoding="utf-8").write("import numpy as np, pandas as pd\nMIN_WEIGHT = 1e-5\n" + cell)


def fmt(model, regime, key):
    for r in RES.get("results", []):
        if r["model"] == model and r["regime"] == regime:
            v = r[key]
            return f"{v:.2f}" if isinstance(v, float) else str(v)
    return "n/a"


FM = "Final model"
UB = "Upstream 2026 baseline (200-MA)"
TR = "Tournament 2025 reference"
UN = "Uniform DCA"


def educational_cells():
    """Draft outline. The completed version, with final numbers from the concluded study, is built at the end."""
    draft = """> **Draft.** This notebook is completed at the end of the study, when the final model and its
> results are settled. The structure below is the Trilemma educational notebook outline; numbered
> placeholders are filled from the concluded results only.

## 1. Executive summary

The thirty-second version: uniform dollar cost averaging buys the same amount every day. This
project builds a rule that keeps that discipline but varies the amount using only information
available at the time, and scores it with the Trilemma Foundation's own engine against uniform
DCA over rolling 12-month windows.

[Final Model Score, win rate and recency-weighted percentile for each scoring configuration,
with the uniform-DCA baseline and the tournament reference implementations beside them.]

[Strengths and limitations, written from the concluded results.]

## 2. Exploratory data analysis highlights

[Selected charts from `notebooks/02_eda.ipynb`: price history with halvings, drawdown from the
one-year high, return distribution and volatility clustering, where uniform DCA lands inside a
window, and the on-chain valuation series. Each chart with one sentence on what it contributed
to the model.]

Full analysis: [`notebooks/02_eda.ipynb`](../notebooks/02_eda.ipynb). Features and the lag test:
[`notebooks/03_features.ipynb`](../notebooks/03_features.ipynb).

## 3. Model explanation

[The final rule: features, signs, bounds, and the pacing allocation, in plain language.]

[Why it beats uniform DCA; why it should generalise; known failure modes.]

[Future improvements.]

Validation: [`notebooks/05_validation.ipynb`](../notebooks/05_validation.ipynb). Selection
protocol: [`notebooks/04_model_selection.ipynb`](../notebooks/04_model_selection.ipynb). Results:
[`notebooks/06_results.ipynb`](../notebooks/06_results.ipynb). Tournament-format notebook:
[`tournament_2025/btc_accumulation_model.ipynb`](../tournament_2025/btc_accumulation_model.ipynb).
"""
    return [("md", draft)]
