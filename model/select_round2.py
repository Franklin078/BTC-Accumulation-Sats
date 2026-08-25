"""Round 2 model selection, as registered in the decision log on 26 August 2026 (D14).

Signals: drawdown (365-day high), MVRV level minus 1.8, exchange netflow z-score (new),
with the asymmetric 200-day MA term kept as a grid candidate. All features lagged one day.

Protocol (fixed): development window starts 2018-01-01 to 2023-06-30; 12-month embargo;
hold-out starts 2024-07-01 onward, untouched by selection; selection metric
0.5 * win rate + 0.5 * unweighted mean SPD percentile on development windows; tie-break
fewer active terms, then smaller m_max; a single pass over the grid, every cell saved.
Baseline to beat: candidate v1's development selection metric 58.45. Only the selected
configuration proceeds to hold-out and full-regime scoring, run by this same script.

Reproduce:  python -m model.select_round2
Outputs:    output/selection_grid_round2.csv, output/selection_result_round2.json,
            output/round2_report.txt, ledger rows in private/LEDGER_v2.csv
"""
import itertools
import json
import logging
import time

import numpy as np
import pandas as pd

from model.regimes import REGIMES, Regime, constraint_check, evaluate, forward_leakage_probe, load_btc, score_table
from model.strategy import Params, construct_features, fast_spd_table, make_strategy

DEV = ("2018-01-01", "2024-06-30")          # scores window starts 2018-01-01 .. 2023-06-30
HOLDOUT = Regime("H", "hold-out (starts 2024-07-01 onward)", "2024-07-01", None)
V1_DEV_SELECTION_METRIC = 58.45             # candidate v1 on the same metric (round 1 grid)

GRID = dict(
    a_ma=[0.0, 1.0],
    a_dd=[0.0, 0.5, 1.0],
    a_mvrv=[0.5, 1.0, 2.0],
    a_flow=[0.0, 0.15, 0.3],
    bias=[0.0, 0.2],
    m_max=[3.0, 5.0],
)
FIXED = dict(ma_len=200, ma_asym=True, a_win=0.0, m_min=0.25, mvrv_mode="level", mvrv0=1.8)


def main():
    logging.getLogger().setLevel(logging.WARNING)
    df = load_btc()
    feats = construct_features(df, Params(**FIXED))
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    rows, t0 = [], time.time()
    for i, vals in enumerate(combos, 1):
        kw = dict(zip(keys, vals))
        p = Params(**FIXED, **kw)
        s = score_table(fast_spd_table(feats, p, *DEV))
        s["selection_metric"] = 0.5 * s["win_rate"] + 0.5 * s["mean_pct"]
        s["active_terms"] = sum(1 for k in ("a_ma", "a_dd", "a_mvrv", "a_flow", "bias") if kw[k] > 0)
        s.update(kw)
        rows.append(s)
        print(f"[{i:3d}/{len(combos)}] {kw} -> win {s['win_rate']:.2f} meanpct {s['mean_pct']:.2f} "
              f"sel {s['selection_metric']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    grid = pd.DataFrame(rows).sort_values(["selection_metric", "active_terms", "m_max"],
                                          ascending=[False, True, True])
    grid.to_csv("output/selection_grid_round2.csv", index=False)

    best = grid.iloc[0]
    chosen = {k: float(best[k]) for k in keys}
    beats_v1 = float(best["selection_metric"]) > V1_DEV_SELECTION_METRIC
    result = {
        "registration": "decision log D14, 2026-08-26",
        "chosen": chosen,
        "dev_metrics": {k: float(best[k]) for k in ["win_rate", "mean_pct", "rw_spd_pct", "selection_metric"]},
        "v1_dev_selection_metric": V1_DEV_SELECTION_METRIC,
        "beats_v1_on_selection_metric": bool(beats_v1),
        "data_last_day": df.index.max().strftime("%Y-%m-%d"),
    }

    lines = [f"ROUND 2 GRID COMPLETE: {len(grid)} configurations, single pass",
             f"winner: {chosen}",
             f"development: win {best['win_rate']:.2f}  mean pct {best['mean_pct']:.2f}  selection {best['selection_metric']:.2f}",
             f"candidate v1 on the same metric: {V1_DEV_SELECTION_METRIC:.2f}  ->  round 2 "
             + ("BEATS v1" if beats_v1 else "DOES NOT BEAT v1 (negative result; v1 stands)")]

    p = Params(**FIXED, **chosen)
    fn = make_strategy(p)
    spd_h, sh = evaluate(df, fn, feats, HOLDOUT)
    lines.append(f"hold-out (untouched, starts 2024-07-01+): windows {sh['windows']}  win {sh['win_rate']:.2f}  "
                 f"RW {sh['rw_spd_pct']:.2f}  score {sh['score']:.2f}  (uniform RW {sh['uniform_rw_spd_pct']:.2f})")
    result["holdout"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in sh.items()}
    regime_rows = []
    for k, r in REGIMES.items():
        spd, s = evaluate(df, fn, feats, r)
        lines.append(f"regime {k}: windows {s['windows']}  win {s['win_rate']:.2f}  RW {s['rw_spd_pct']:.2f}  score {s['score']:.2f}")
        s["model"] = "Round 2 winner"
        regime_rows.append(s)
    pr = {k: forward_leakage_probe(df, fn, r.start, r.resolve_end(df)) for k, r in REGIMES.items()}
    cc = {k: constraint_check(df, fn, r.start, r.resolve_end(df)) for k, r in REGIMES.items()}
    gates = all(pr[k]["passed"] and cc[k]["passed"] for k in REGIMES)
    lines.append("gates: probe " + ("PASS" if all(pr[k]["passed"] for k in REGIMES) else "FAIL")
                 + ", constraints " + ("PASS" if all(cc[k]["passed"] for k in REGIMES) else "FAIL"))
    result["gates_passed"] = bool(gates)
    json.dump(result, open("output/selection_result_round2.json", "w"), indent=2, default=float)

    import datetime
    import os
    os.makedirs("private", exist_ok=True)
    led = pd.DataFrame(regime_rows)
    led.insert(0, "date", datetime.date.today().isoformat())
    led.insert(1, "commit", "round2")
    led.insert(2, "data_last_day", result["data_last_day"])
    led["params"] = json.dumps({**FIXED, **chosen})
    path = "private/LEDGER_v2.csv"
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    pd.concat([old, led]).to_csv(path, index=False)

    report = "\n".join(lines)
    open("output/round2_report.txt", "w", encoding="utf-8").write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
