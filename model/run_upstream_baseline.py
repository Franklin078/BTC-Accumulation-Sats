"""Score the unmodified Trilemma capstone baseline (200-day MA, template/model_development_template.py)
under regimes A, B and C on data/Coin Metrics/coinmetrics_btc.csv."""
import json, time, logging
import pandas as pd
from template.model_development_template import precompute_features, compute_window_weights
from model.regimes import REGIMES, evaluate, load_btc, configure_logging

configure_logging(); logging.getLogger().setLevel(logging.WARNING)
df = load_btc()
feats = precompute_features(df)

def upstream_fn(df_window: pd.DataFrame) -> pd.Series:
    if df_window.empty: return pd.Series(dtype=float)
    s, e = df_window.index.min(), df_window.index.max()
    return compute_window_weights(feats, s, e, e)

out = {}
for k in ["A", "B", "C"]:
    t0 = time.time(); spd, summ = evaluate(df, upstream_fn, feats, REGIMES[k]); summ["seconds"] = round(time.time() - t0, 1)
    out[k] = summ
    print(f"Regime {k} {REGIMES[k].label}: windows={summ['windows']} win={summ['win_rate']:.2f}% RW={summ['rw_spd_pct']:.2f}% SCORE={summ['score']:.2f}%  (uniform RW {summ['uniform_rw_spd_pct']:.2f}%) [{summ['seconds']}s]")
json.dump(out, open("output/upstream_baseline_scores.json", "w"), indent=2)
