"""Robustness battery for the three frozen finalists (candidates v5, v6, v7), as reporting
only: nothing here selects anything. Three components:

1. Regime tables: each finalist scored on regimes A, B and C, window-level SPD tables saved.
2. Leave-one-year-of-starts-out: win rate and score on regime B with each calendar year of
   window starts excluded in turn, showing how much any single year carries.
3. One-at-a-time parameter sensitivity on the development windows: each allocator parameter
   perturbed alone around the registered value, everything else fixed, reported as the change
   in the development selection metric.

The strict-lag versus same-day comparison is structural for the learner track (every feature
is lagged one day by construction), so that axis remains the candidate v1 record.

Reproduce:  python -m model.robustness
Outputs:    output/robustness_finalists.json, output/loyo_finalists.csv,
            output/sensitivity_finalists.csv, output/spd_tables/robust_v{5,6,7}_{A,B,C}.csv
"""
import json
import logging

import numpy as np
import pandas as pd

from model.ml import MLConfig, build_ml_features, make_ml_strategy
from model.regimes import REGIMES, evaluate, load_btc
from model.roundutil import DEV

FINALISTS = {
    "v5": dict(a_ml=2.0, a_pos=0.0, a_neg=0.0, m_max=12.0),
    "v6": dict(a_ml=2.0, a_pos=3.0, a_neg=1.0, m_max=12.0),
    "v7": dict(a_ml=2.0, a_pos=5.0, a_neg=1.0, m_max=16.0),
}
# one-at-a-time perturbations per finalist: parameter -> values to try alone
SENSITIVITY = {
    "v5": {"a_ml": [1.5, 2.5], "m_max": [8.0, 16.0]},
    "v6": {"a_pos": [2.0, 4.0], "a_neg": [0.5, 1.5], "m_max": [8.0, 16.0]},
    "v7": {"a_pos": [4.0, 6.0], "a_neg": [0.5, 1.5], "m_max": [12.0, 20.0]},
}


def base_cfg(over: dict) -> MLConfig:
    """The shared engine (candidate v4's signal) under one finalist's allocator settings."""
    v5 = json.load(open("model/candidates.json"))["Candidate v5 (v4 signal, ceiling 12, round 6)"]
    return MLConfig(model=str(v5["model"]), horizon=int(v5["horizon"]),
                    features=tuple(v5["features"]), shape=str(v5["shape"]),
                    b_quad=float(v5["b_quad"]), m_min=0.25, **over)


def main():
    logging.getLogger().setLevel(logging.WARNING)
    import sklearn
    df = load_btc()
    # all finalists share one prediction frame: they differ only in the allocator
    feats = build_ml_features(df, base_cfg(FINALISTS["v5"]))

    result = {"sklearn_version": sklearn.__version__,
              "data_last_day": df.index.max().strftime("%Y-%m-%d"), "regimes": {}}
    loyo_rows, sens_rows = [], []
    for tag, over in FINALISTS.items():
        fn = make_ml_strategy(base_cfg(over))
        result["regimes"][tag] = {}
        for k, regime in REGIMES.items():
            spd, s = evaluate(df, fn, feats, regime)
            spd.to_csv(f"output/spd_tables/robust_{tag}_{k}.csv")
            result["regimes"][tag][k] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                         for kk, v in s.items()}
            print(f"{tag} regime {k}: win {s['win_rate']:.2f} RW {s['rw_spd_pct']:.2f} "
                  f"score {s['score']:.2f}", flush=True)
            if k == "B":
                dyn = spd["dynamic_percentile"].to_numpy()
                uni = spd["uniform_percentile"].to_numpy()
                years = pd.to_datetime([w.split(" → ")[0] for w in spd.index]).year
                loyo_rows.append({"model": tag, "excluded_year": "none",
                                  "windows": len(spd), "win_rate": float((dyn > uni).mean() * 100)})
                for y in sorted(set(years)):
                    keep = years != y
                    loyo_rows.append({"model": tag, "excluded_year": int(y),
                                      "windows": int(keep.sum()),
                                      "win_rate": float((dyn[keep] > uni[keep]).mean() * 100)})
        # sensitivity on the development windows, one parameter at a time
        spd, s = evaluate(df, fn, feats, DEV)
        base_sel = 0.5 * s["win_rate"] + 0.5 * s["mean_pct"]
        sens_rows.append({"model": tag, "parameter": "base", "value": "registered",
                          "dev_selection": base_sel, "delta": 0.0})
        for param, values in SENSITIVITY[tag].items():
            for v in values:
                fn_p = make_ml_strategy(base_cfg({**over, param: v}))
                spd, sp = evaluate(df, fn_p, feats, DEV)
                sel = 0.5 * sp["win_rate"] + 0.5 * sp["mean_pct"]
                sens_rows.append({"model": tag, "parameter": param, "value": v,
                                  "dev_selection": sel, "delta": sel - base_sel})
                print(f"{tag} sensitivity {param}={v}: dev {sel:.2f} ({sel - base_sel:+.2f})",
                      flush=True)

    pd.DataFrame(loyo_rows).to_csv("output/loyo_finalists.csv", index=False)
    pd.DataFrame(sens_rows).to_csv("output/sensitivity_finalists.csv", index=False)
    json.dump(result, open("output/robustness_finalists.json", "w"), indent=2, default=float)
    print("robustness battery written")


if __name__ == "__main__":
    main()
