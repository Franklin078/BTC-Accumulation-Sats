"""Score the frozen model and the fixed baselines on regimes A, B, C, plus hold-out; write output/results_*.json/csv."""
import json, time, logging, os
import pandas as pd, numpy as np
from model.regimes import load_btc, REGIMES, evaluate, Regime, score_table
from model.strategy import construct_features, Params, make_strategy
from model import reference_2025
from template.model_development_template import precompute_features, compute_window_weights
logging.getLogger().setLevel(logging.WARNING)
df = load_btc(); P = json.load(open("model/final_params.json")); p = Params(**P); feats = construct_features(df, p)
upfeats = precompute_features(df)
def upstream_fn(w):
    if w.empty: return pd.Series(dtype=float)
    return compute_window_weights(upfeats, w.index.min(), w.index.max(), w.index.max())
models = {"Final model": make_strategy(p), "Uniform DCA": lambda w: pd.Series(1.0/len(w), index=w.index),
          "Upstream 2026 baseline (200-MA)": upstream_fn, "Tournament 2025 reference": reference_2025.compute_weights}
rows = []; os.makedirs("output/spd_tables", exist_ok=True)
for name, fn in models.items():
    for k, r in REGIMES.items():
        t0 = time.time(); spd, s = evaluate(df, fn, feats, r); s["model"] = name
        if name == "Uniform DCA":  # identical weights: any 'wins' are floating-point ties, the true win rate is 0
            s["win_rate"] = 0.0; s["score"] = 0.5 * s["rw_spd_pct"]; s["note"] = "win rate set to 0: uniform against itself ties every window"
        rows.append(s)
        spd.to_csv(f"output/spd_tables/{name.split(' ')[0].lower()}_{k}.csv")
        print(f"{name:34s} {k}: windows {s['windows']} win {s['win_rate']:.2f} RW {s['rw_spd_pct']:.2f} SCORE {s['score']:.2f} (uniform RW {s['uniform_rw_spd_pct']:.2f}) [{time.time()-t0:.0f}s]", flush=True)
res = pd.DataFrame(rows)[["model","regime","start","end","windows","win_rate","rw_spd_pct","score","uniform_rw_spd_pct","mean_pct","mean_excess","min_excess","note"]] if "note" in pd.DataFrame(rows).columns else pd.DataFrame(rows)
res.to_csv("output/results_regimes.csv", index=False)
hold = Regime("H", "hold-out (starts 2024-07-01 onward)", "2024-07-01", None)
spd, s = evaluate(df, make_strategy(p), feats, hold); print("HOLD-OUT final model:", {k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()})
dev = Regime("D", "development", "2018-01-01", "2024-06-30"); spd_d, sd = evaluate(df, make_strategy(p), feats, dev)
json.dump({"data_last_day": df.index.max().strftime("%Y-%m-%d"), "params": P, "results": res.to_dict(orient="records"),
           "holdout": s, "development": sd}, open("output/results_summary.json", "w"), indent=2, default=float)
print("written output/results_summary.json")
