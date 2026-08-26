"""Shared runner for registered rounds: evaluate a list of named configurations on the
development windows, pick the best by the fixed selection metric, score it on the hold-out
and the regimes, run the gates, and write the round's grid, result and report files.

The protocol pieces (development span, embargo, hold-out, selection metric, single pass)
are constants here so every round applies exactly the same rules.
"""
import json
import logging
import os
import time

import numpy as np
import pandas as pd

from model.regimes import REGIMES, Regime, constraint_check, evaluate, forward_leakage_probe, load_btc

DEV = Regime("D", "development (starts 2018-01-01 to 2023-06-30)", "2018-01-01", "2024-06-30")
HOLDOUT = Regime("H", "hold-out (starts 2024-07-01 onward)", "2024-07-01", None)


def run_round(round_name: str, registration: str, baseline_name: str, baseline_value: float,
              configs: list, out_prefix: str, df=None, probe_winner: bool = True) -> dict:
    """configs: list of dicts {"label": str, "fn": strategy_function, "feats": DataFrame, ...meta}."""
    logging.getLogger().setLevel(logging.WARNING)
    if df is None:
        df = load_btc()
    import hashlib
    fingerprint = hashlib.sha256(("|".join(sorted(c["label"] for c in configs))
                                  + "|" + df.index.max().strftime("%Y-%m-%d")).encode()).hexdigest()[:16]
    if os.path.exists(f"output/{out_prefix}_result.json"):
        prev = json.load(open(f"output/{out_prefix}_result.json"))
        if prev.get("fingerprint") == fingerprint:
            prev["beats_baseline"] = bool(prev["dev_metrics"]["selection_metric"] > baseline_value)
            print(f"{round_name}: cached result matches fingerprint {fingerprint}; reusing "
                  f"(beat flag recomputed against current baseline {baseline_value:.2f})")
            return prev
        print(f"{round_name}: cached result is stale (code, configs or data changed); recomputing")
    rows, t0 = [], time.time()
    by_label = {}
    for i, c in enumerate(configs, 1):
        spd, s = evaluate(df, c["fn"], c["feats"], DEV)
        s["selection_metric"] = 0.5 * s["win_rate"] + 0.5 * s["mean_pct"]
        s["label"] = c["label"]
        for k, v in c.items():
            if k not in ("fn", "feats", "label"):
                s[k] = v
        rows.append(s)
        by_label[c["label"]] = c
        print(f"[{i:2d}/{len(configs)}] {c['label']} -> sel {s['selection_metric']:.2f} "
              f"(win {s['win_rate']:.2f}, meanpct {s['mean_pct']:.2f}) ({time.time()-t0:.0f}s)", flush=True)
    grid = pd.DataFrame(rows).sort_values("selection_metric", ascending=False)
    grid.to_csv(f"output/{out_prefix}_grid.csv", index=False)

    best = grid.iloc[0]
    beats = float(best["selection_metric"]) > baseline_value
    lines = [f"{round_name}: {len(grid)} configurations, single pass",
             f"best: {best['label']} -> dev selection {best['selection_metric']:.2f}",
             f"baseline ({baseline_name}): {baseline_value:.2f} -> "
             + ("BEATS the baseline" if beats else "DOES NOT BEAT the baseline (negative result)")]
    result = {"registration": registration, "best_label": str(best["label"]),
              "dev_metrics": {k: float(best[k]) for k in ["win_rate", "mean_pct", "rw_spd_pct", "selection_metric"]},
              "baseline_name": baseline_name, "baseline_value": baseline_value, "beats_baseline": bool(beats),
              "data_last_day": df.index.max().strftime("%Y-%m-%d"), "fingerprint": fingerprint}

    winner = by_label[str(best["label"])]
    spd, sh = evaluate(df, winner["fn"], winner["feats"], HOLDOUT)
    lines.append(f"best config hold-out: win {sh['win_rate']:.2f}  RW {sh['rw_spd_pct']:.2f}  score {sh['score']:.2f}")
    result["holdout"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in sh.items()}
    for k, r in REGIMES.items():
        spd, s = evaluate(df, winner["fn"], winner["feats"], r)
        lines.append(f"best config regime {k}: win {s['win_rate']:.2f}  RW {s['rw_spd_pct']:.2f}  score {s['score']:.2f}")
        result[f"regime_{k}"] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v) for kk, v in s.items()}
    if probe_winner:
        print("running the tournament forward-leakage probe on the winner...", flush=True)
        pr = forward_leakage_probe(df, winner["fn"], "2018-01-01", REGIMES["B"].resolve_end(df))
        cc = constraint_check(df, winner["fn"], "2018-01-01", REGIMES["B"].resolve_end(df))
        lines.append(f"gates: probe {'PASS' if pr['passed'] else 'FAIL'} ({len(pr['failures'])}/{pr['probes']}), "
                     f"constraints {'PASS' if cc['passed'] else 'FAIL'}")
        result["probe_passed"] = bool(pr["passed"])
        result["constraints_passed"] = bool(cc["passed"])

    json.dump(result, open(f"output/{out_prefix}_result.json", "w"), indent=2, default=float)
    report = "\n".join(lines)
    open(f"output/{out_prefix}_report.txt", "w", encoding="utf-8").write(report)
    print("\n" + report)
    return result
