"""Round 3a: convex blends of candidate v1 and candidate v2, as registered (D16, 26 August 2026).

w = alpha * v1 + (1 - alpha) * v2, alpha in {0.25, 0.50, 0.75}, blended per window. A convex
mixture of valid weight vectors is itself valid (each weight stays above the floor and the sum
stays 1), so no new constraint machinery is needed. Every score comes from the upstream engine.

Baseline to beat: candidate v2's development selection metric 61.31.

Reproduce:  python -m model.select_round3a
Outputs:    output/round3a_report.txt, output/selection_result_round3a.json
"""
import json
import logging

import numpy as np
import pandas as pd

from model.regimes import REGIMES, Regime, constraint_check, evaluate, forward_leakage_probe, load_btc
from model.strategy import Params, construct_features, make_strategy

DEV = Regime("D", "development (starts 2018-01-01 to 2023-06-30)", "2018-01-01", "2024-06-30")
HOLDOUT = Regime("H", "hold-out (starts 2024-07-01 onward)", "2024-07-01", None)
V2_DEV_SELECTION_METRIC = 61.31
ALPHAS = [0.25, 0.50, 0.75]


def load_candidate_params():
    v1 = Params(**json.load(open("model/final_params.json")))
    r2 = json.load(open("output/selection_result_round2.json"))
    fixed = dict(ma_len=200, ma_asym=True, a_win=0.0, m_min=0.25, mvrv_mode="level", mvrv0=1.8)
    v2 = Params(**fixed, **{k: float(v) for k, v in r2["chosen"].items()})
    return v1, v2


def make_blend(f1, f2, alpha):
    def fn(df_window):
        w1 = f1(df_window)
        w2 = f2(df_window)
        return alpha * w1 + (1.0 - alpha) * w2
    return fn


def main():
    logging.getLogger().setLevel(logging.WARNING)
    df = load_btc()
    v1p, v2p = load_candidate_params()
    feats = construct_features(df, v1p)  # both candidates share the same feature definitions
    f1, f2 = make_strategy(v1p), make_strategy(v2p)

    lines = [f"ROUND 3A: blends of v1 and v2, alphas {ALPHAS}"]
    rows = []
    for a in ALPHAS:
        fn = make_blend(f1, f2, a)
        spd, s = evaluate(df, fn, feats, DEV)
        s["alpha_v1"] = a
        s["selection_metric"] = 0.5 * s["win_rate"] + 0.5 * s["mean_pct"]
        rows.append(s)
        lines.append(f"alpha {a:.2f}: dev win {s['win_rate']:.2f}  mean pct {s['mean_pct']:.2f}  "
                     f"selection {s['selection_metric']:.2f}")
        print(lines[-1], flush=True)
    table = pd.DataFrame(rows).sort_values("selection_metric", ascending=False)
    best = table.iloc[0]
    beats = float(best["selection_metric"]) > V2_DEV_SELECTION_METRIC
    lines.append(f"baseline (candidate v2): {V2_DEV_SELECTION_METRIC:.2f} -> best blend "
                 + ("BEATS v2" if beats else "DOES NOT BEAT v2 (negative result; v2 stands)"))

    result = {"registration": "decision log D16, 2026-08-26",
              "alphas": ALPHAS,
              "table": table.to_dict(orient="records"),
              "beats_v2": bool(beats),
              "data_last_day": df.index.max().strftime("%Y-%m-%d")}

    if beats:
        a = float(best["alpha_v1"])
        fn = make_blend(f1, f2, a)
        spd, sh = evaluate(df, fn, feats, HOLDOUT)
        lines.append(f"winner alpha {a:.2f} hold-out: win {sh['win_rate']:.2f}  RW {sh['rw_spd_pct']:.2f}  "
                     f"score {sh['score']:.2f} (uniform RW {sh['uniform_rw_spd_pct']:.2f})")
        result["holdout"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in sh.items()}
        for k, r in REGIMES.items():
            spd, s = evaluate(df, fn, feats, r)
            lines.append(f"winner regime {k}: win {s['win_rate']:.2f}  RW {s['rw_spd_pct']:.2f}  score {s['score']:.2f}")
            result[f"regime_{k}"] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v) for kk, v in s.items()}
        pr = {k: forward_leakage_probe(df, fn, r.start, r.resolve_end(df)) for k, r in REGIMES.items()}
        cc = {k: constraint_check(df, fn, r.start, r.resolve_end(df)) for k, r in REGIMES.items()}
        gates = all(pr[k]["passed"] and cc[k]["passed"] for k in REGIMES)
        lines.append("gates: probe " + ("PASS" if all(pr[k]["passed"] for k in REGIMES) else "FAIL")
                     + ", constraints " + ("PASS" if all(cc[k]["passed"] for k in REGIMES) else "FAIL"))
        result["gates_passed"] = bool(gates)

    json.dump(result, open("output/selection_result_round3a.json", "w"), indent=2, default=float)
    report = "\n".join(lines)
    open("output/round3a_report.txt", "w", encoding="utf-8").write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
