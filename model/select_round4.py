"""Round 4: machine-learning candidates, as registered (D21, 26 August 2026).

Grid, single pass: model {ridge, hgbr} x horizon {30, 90} x a_ml {0.5, 1.0, 2.0} = 12
configurations. Protocol unchanged; baseline to beat: candidate v3's development selection
metric 63.09. Winner, if any, proceeds to hold-out, full regimes, the tournament forward-
leakage probe (which retrains the walk-forward inside every masked pass) and constraint checks.

Reproduce:  python -m model.select_round4
Outputs:    output/selection_grid_round4.csv, output/selection_result_round4.json,
            output/round4_report.txt
"""
import itertools
import json
import logging
import time

import numpy as np
import pandas as pd

from model.ml import MLConfig, build_ml_features, make_ml_strategy
from model.regimes import REGIMES, Regime, constraint_check, evaluate, forward_leakage_probe, load_btc, score_table
from model.strategy import PRICE_COL

DEV = Regime("D", "development (starts 2018-01-01 to 2023-06-30)", "2018-01-01", "2024-06-30")
HOLDOUT = Regime("H", "hold-out (starts 2024-07-01 onward)", "2024-07-01", None)
V3_DEV_SELECTION_METRIC = 63.09

GRID = dict(model=["ridge", "hgbr"], horizon=[30, 90], a_ml=[0.5, 1.0, 2.0])


def main():
    logging.getLogger().setLevel(logging.WARNING)
    df = load_btc()
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    rows, t0 = [], time.time()
    feats_cache = {}
    for i, vals in enumerate(combos, 1):
        kw = dict(zip(keys, vals))
        cfg = MLConfig(**kw)
        fkey = (cfg.model, cfg.horizon)
        if fkey not in feats_cache:
            feats_cache[fkey] = build_ml_features(df, cfg)
        feats = feats_cache[fkey]
        fn = make_ml_strategy(cfg)
        spd, s = evaluate(df, fn, feats, DEV)
        s["selection_metric"] = 0.5 * s["win_rate"] + 0.5 * s["mean_pct"]
        s.update(kw)
        rows.append(s)
        print(f"[{i:2d}/{len(combos)}] {kw} -> win {s['win_rate']:.2f} meanpct {s['mean_pct']:.2f} "
              f"sel {s['selection_metric']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    grid = pd.DataFrame(rows).sort_values("selection_metric", ascending=False)
    grid.to_csv("output/selection_grid_round4.csv", index=False)

    best = grid.iloc[0]
    chosen = {k: (best[k] if k == "model" else float(best[k])) for k in keys}
    beats = float(best["selection_metric"]) > V3_DEV_SELECTION_METRIC
    lines = [f"ROUND 4: machine-learning grid, {len(grid)} configurations",
             f"best: {chosen} -> dev selection {best['selection_metric']:.2f} "
             f"(win {best['win_rate']:.2f}, mean pct {best['mean_pct']:.2f})",
             f"baseline (candidate v3): {V3_DEV_SELECTION_METRIC:.2f} -> "
             + ("BEATS v3" if beats else "DOES NOT BEAT v3 (negative result; v3 stands)")]
    result = {"registration": "decision log D21, 2026-08-26", "chosen": chosen,
              "dev_metrics": {k: float(best[k]) for k in ["win_rate", "mean_pct", "rw_spd_pct", "selection_metric"]},
              "beats_v3": bool(beats), "data_last_day": df.index.max().strftime("%Y-%m-%d")}

    cfg = MLConfig(model=str(chosen["model"]), horizon=int(chosen["horizon"]), a_ml=float(chosen["a_ml"]))
    fn = make_ml_strategy(cfg)
    feats = feats_cache[(cfg.model, cfg.horizon)]
    spd, sh = evaluate(df, fn, feats, HOLDOUT)
    lines.append(f"best config hold-out: win {sh['win_rate']:.2f}  RW {sh['rw_spd_pct']:.2f}  score {sh['score']:.2f} "
                 f"(uniform RW {sh['uniform_rw_spd_pct']:.2f})")
    result["holdout"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in sh.items()}
    for k, r in REGIMES.items():
        spd, s = evaluate(df, fn, feats, r)
        lines.append(f"best config regime {k}: win {s['win_rate']:.2f}  RW {s['rw_spd_pct']:.2f}  score {s['score']:.2f}")
        result[f"regime_{k}"] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v) for kk, v in s.items()}

    print("running the tournament forward-leakage probe (retrains inside every masked pass; slow)...", flush=True)
    pr = forward_leakage_probe(df, fn, "2018-01-01", REGIMES["B"].resolve_end(df))
    cc = constraint_check(df, fn, "2018-01-01", REGIMES["B"].resolve_end(df))
    lines.append(f"gates on the best config: probe {'PASS' if pr['passed'] else 'FAIL'} "
                 f"({len(pr['failures'])}/{pr['probes']} failures), constraints "
                 f"{'PASS' if cc['passed'] else 'FAIL'} ({cc['windows']} windows)")
    result["probe"] = pr
    result["constraints_passed"] = bool(cc["passed"])

    json.dump(result, open("output/selection_result_round4.json", "w"), indent=2, default=float)
    report = "\n".join(lines)
    open("output/round4_report.txt", "w", encoding="utf-8").write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
