"""Round 3b: refinement of the round 2 winner's neighbourhood, as registered (D17, 26 August 2026).

Grid: a_mvrv {0.75, 1.0, 1.25, 1.5} x a_flow {0.075, 0.15, 0.225, 0.3} x mvrv0 {1.6, 1.8, 2.0},
all other terms zero, m_max 5, m_min 0.25. 48 configurations, single pass, every cell saved.
Baseline to beat: candidate v2's development selection metric 61.31.

Reproduce:  python -m model.select_round3b
Outputs:    output/selection_grid_round3b.csv, output/selection_result_round3b.json,
            output/round3b_report.txt
"""
import itertools
import json
import logging
import time

import numpy as np
import pandas as pd

from model.regimes import REGIMES, Regime, constraint_check, evaluate, forward_leakage_probe, load_btc, score_table
from model.strategy import Params, construct_features, fast_spd_table, make_strategy

DEV = ("2018-01-01", "2024-06-30")
HOLDOUT = Regime("H", "hold-out (starts 2024-07-01 onward)", "2024-07-01", None)
V2_DEV_SELECTION_METRIC = 61.31

GRID = dict(a_mvrv=[0.75, 1.0, 1.25, 1.5], a_flow=[0.075, 0.15, 0.225, 0.3], mvrv0=[1.6, 1.8, 2.0])
FIXED = dict(ma_len=200, ma_asym=True, a_ma=0.0, a_dd=0.0, a_win=0.0, bias=0.0,
             m_min=0.25, m_max=5.0, mvrv_mode="level")


def main():
    logging.getLogger().setLevel(logging.WARNING)
    df = load_btc()
    feats_by_centre = {c: construct_features(df, Params(**FIXED, mvrv0=c)) for c in GRID["mvrv0"]}
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    rows, t0 = [], time.time()
    for i, vals in enumerate(combos, 1):
        kw = dict(zip(keys, vals))
        p = Params(**FIXED, **kw)
        s = score_table(fast_spd_table(feats_by_centre[kw["mvrv0"]], p, *DEV))
        s["selection_metric"] = 0.5 * s["win_rate"] + 0.5 * s["mean_pct"]
        s.update(kw)
        rows.append(s)
        print(f"[{i:2d}/{len(combos)}] {kw} -> sel {s['selection_metric']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    grid = pd.DataFrame(rows).sort_values("selection_metric", ascending=False)
    grid.to_csv("output/selection_grid_round3b.csv", index=False)

    best = grid.iloc[0]
    chosen = {k: float(best[k]) for k in keys}
    beats = float(best["selection_metric"]) > V2_DEV_SELECTION_METRIC
    lines = [f"ROUND 3B: refinement grid, {len(grid)} configurations",
             f"best: {chosen} -> dev selection {best['selection_metric']:.2f}",
             f"baseline (candidate v2): {V2_DEV_SELECTION_METRIC:.2f} -> "
             + ("BEATS v2" if beats else "DOES NOT BEAT v2 (negative result; v2 stands unrefined)")]
    result = {"registration": "decision log D17, 2026-08-26", "chosen": chosen,
              "dev_metrics": {k: float(best[k]) for k in ["win_rate", "mean_pct", "rw_spd_pct", "selection_metric"]},
              "beats_v2": bool(beats), "data_last_day": df.index.max().strftime("%Y-%m-%d")}

    if beats:
        p = Params(**FIXED, **chosen)
        fn = make_strategy(p)
        feats = feats_by_centre[chosen["mvrv0"]]
        spd, sh = evaluate(df, fn, feats, HOLDOUT)
        lines.append(f"winner hold-out: win {sh['win_rate']:.2f}  RW {sh['rw_spd_pct']:.2f}  score {sh['score']:.2f}")
        result["holdout"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in sh.items()}
        for k, r in REGIMES.items():
            spd, s = evaluate(df, fn, feats, r)
            lines.append(f"winner regime {k}: win {s['win_rate']:.2f}  RW {s['rw_spd_pct']:.2f}  score {s['score']:.2f}")
            result[f"regime_{k}"] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v) for kk, v in s.items()}
        pr = {k: forward_leakage_probe(df, fn, r.start, r.resolve_end(df)) for k, r in REGIMES.items()}
        cc = {k: constraint_check(df, fn, r.start, r.resolve_end(df)) for k, r in REGIMES.items()}
        result["gates_passed"] = bool(all(pr[k]["passed"] and cc[k]["passed"] for k in REGIMES))
        lines.append("gates: probe " + ("PASS" if all(pr[k]["passed"] for k in REGIMES) else "FAIL")
                     + ", constraints " + ("PASS" if all(cc[k]["passed"] for k in REGIMES) else "FAIL"))

    json.dump(result, open("output/selection_result_round3b.json", "w"), indent=2, default=float)
    report = "\n".join(lines)
    open("output/round3b_report.txt", "w", encoding="utf-8").write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
