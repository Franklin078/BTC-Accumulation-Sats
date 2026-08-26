"""Round 5: feature-prioritised machine learning, as registered (D23, 26 August 2026).

Step 1 measures feature importance on the development period only, two ways: the absolute
Spearman rank correlation of each lagged feature with the 90-day forward log return, and
permutation importance from a gradient-boosting model trained on the first 70 per cent of
the development period and evaluated on the last 30 per cent behind a 90-day purge gap.
Features are ranked by the average of the two rank positions and the table is published.

Step 2 takes the top 3 and top 5 features as the candidate sets. Step 3 runs the registered
grid: model {ridge, hgbr} x feature set {top 3, top 5} x a_ml {1.0, 2.0}, horizon 90, with
the same purged walk-forward, causal standardisation and pacing allocator as round 4.
Baseline to beat: candidate v3's development selection metric 63.09. Step 4 scores the
development-selected best configuration on the hold-out and all regimes and runs the
tournament probe, whatever its development result. Step 5 writes a descriptive ranking of
every named model across the regimes (reporting, not selection).

Reproduce:  python -m model.select_round5
Outputs:    output/feature_importance_round5.csv, output/selection_grid_round5.csv,
            output/selection_result_round5.json, output/model_ranking.csv,
            output/round5_report.txt
"""
import itertools
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from model.ml import MLConfig, _make_model, build_ml_features, feature_matrix, make_ml_strategy
from model.regimes import REGIMES, Regime, constraint_check, evaluate, forward_leakage_probe, load_btc
from model.strategy import PRICE_COL

DEV = Regime("D", "development (starts 2018-01-01 to 2023-06-30)", "2018-01-01", "2024-06-30")
HOLDOUT = Regime("H", "hold-out (starts 2024-07-01 onward)", "2024-07-01", None)
V3_DEV_SELECTION_METRIC = 63.09
HORIZON = 90
GRID = dict(model=["ridge", "hgbr"], top_k=[3, 5], a_ml=[1.0, 2.0])
DEV_DAYS = ("2018-01-01", "2023-06-30")   # feature-importance sample: development window starts


def importance_table(df: pd.DataFrame) -> pd.DataFrame:
    X = feature_matrix(df).loc[DEV_DAYS[0]:DEV_DAYS[1]]
    p = df[PRICE_COL].astype(float)
    y = np.log(p.shift(-HORIZON) / p).loc[X.index]
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    ic = X.corrwith(y, method="spearman").abs().rename("abs_spearman_ic")
    n = len(X)
    split = int(n * 0.70)
    purge = HORIZON
    Xtr, ytr = X.iloc[:split - purge], y.iloc[:split - purge]
    Xva, yva = X.iloc[split:], y.iloc[split:]
    m = _make_model("hgbr").fit(Xtr, ytr)
    pi = permutation_importance(m, Xva, yva, n_repeats=10, random_state=0)
    perm = pd.Series(pi.importances_mean, index=X.columns, name="permutation_importance")
    t = pd.concat([ic, perm], axis=1)
    t["rank_ic"] = t["abs_spearman_ic"].rank(ascending=False)
    t["rank_perm"] = t["permutation_importance"].rank(ascending=False)
    t["avg_rank"] = (t["rank_ic"] + t["rank_perm"]) / 2
    return t.sort_values("avg_rank")


def main():
    logging.getLogger().setLevel(logging.WARNING)
    df = load_btc()
    imp = importance_table(df)
    imp.to_csv("output/feature_importance_round5.csv")
    ranked = list(imp.index)
    lines = ["ROUND 5 STEP 1: feature importance on the development period",
             imp.round(4).to_string(),
             f"priority order: {ranked}"]
    print("\n".join(lines), flush=True)

    rows, t0, cache = [], time.time(), {}
    combos = list(itertools.product(*GRID.values()))
    for i, (mdl, k, a) in enumerate(combos, 1):
        feats_key = (mdl, k)
        cfg = MLConfig(model=mdl, horizon=HORIZON, a_ml=a, features=tuple(ranked[:k]))
        if feats_key not in cache:
            cache[feats_key] = build_ml_features(df, cfg)
        fn = make_ml_strategy(cfg)
        spd, s = evaluate(df, fn, cache[feats_key], DEV)
        s["selection_metric"] = 0.5 * s["win_rate"] + 0.5 * s["mean_pct"]
        s.update(model=mdl, top_k=k, a_ml=a, features=";".join(ranked[:k]))
        rows.append(s)
        print(f"[{i}/{len(combos)}] {mdl} top{k} a={a} -> sel {s['selection_metric']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    grid = pd.DataFrame(rows).sort_values("selection_metric", ascending=False)
    grid.to_csv("output/selection_grid_round5.csv", index=False)

    best = grid.iloc[0]
    chosen = dict(model=str(best["model"]), top_k=int(best["top_k"]), a_ml=float(best["a_ml"]),
                  features=str(best["features"]))
    beats = float(best["selection_metric"]) > V3_DEV_SELECTION_METRIC
    lines += ["", f"ROUND 5 GRID: {len(grid)} configurations",
              f"best: {chosen} -> dev selection {best['selection_metric']:.2f}",
              f"baseline (candidate v3): {V3_DEV_SELECTION_METRIC:.2f} -> "
              + ("BEATS v3" if beats else "DOES NOT BEAT v3 (negative result; v3 stands)")]
    result = {"registration": "decision log D23, 2026-08-26", "importance": imp.to_dict(),
              "priority_order": ranked, "chosen": chosen, "beats_v3": bool(beats),
              "dev_metrics": {k: float(best[k]) for k in ["win_rate", "mean_pct", "rw_spd_pct", "selection_metric"]},
              "data_last_day": df.index.max().strftime("%Y-%m-%d")}

    cfg = MLConfig(model=chosen["model"], horizon=HORIZON, a_ml=chosen["a_ml"],
                   features=tuple(chosen["features"].split(";")))
    fn = make_ml_strategy(cfg)
    feats = cache[(chosen["model"], chosen["top_k"])]
    r5_scores = {}
    spd, sh = evaluate(df, fn, feats, HOLDOUT)
    lines.append(f"best config hold-out: win {sh['win_rate']:.2f}  RW {sh['rw_spd_pct']:.2f}  score {sh['score']:.2f}")
    result["holdout"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in sh.items()}
    for k, r in REGIMES.items():
        spd, s = evaluate(df, fn, feats, r)
        r5_scores[k] = s["score"]
        lines.append(f"best config regime {k}: win {s['win_rate']:.2f}  RW {s['rw_spd_pct']:.2f}  score {s['score']:.2f}")
        result[f"regime_{k}"] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v) for kk, v in s.items()}
    print("running the tournament forward-leakage probe (retrains under masking; slow)...", flush=True)
    pr = forward_leakage_probe(df, fn, "2018-01-01", REGIMES["B"].resolve_end(df))
    cc = constraint_check(df, fn, "2018-01-01", REGIMES["B"].resolve_end(df))
    lines.append(f"gates: probe {'PASS' if pr['passed'] else 'FAIL'} ({len(pr['failures'])}/{pr['probes']}), "
                 f"constraints {'PASS' if cc['passed'] else 'FAIL'}")
    result["probe_passed"] = bool(pr["passed"]); result["constraints_passed"] = bool(cc["passed"])

    # Step 5: descriptive ranking of all named models across regimes
    reg = pd.read_csv("output/results_regimes.csv")
    r4 = json.load(open("output/selection_result_round4.json"))
    rank_rows = []
    for m in reg.model.unique():
        sub = reg[reg.model == m]
        rank_rows.append({"model": m, **{f"score_{k}": float(sub[sub.regime == k]["score"].iloc[0]) for k in "ABC"}})
    rank_rows.append({"model": "Round 4 ML best (ridge, H90)", **{f"score_{k}": float(r4[f"regime_{k}"]["score"]) for k in "ABC"}})
    rank_rows.append({"model": f"Round 5 ML best ({chosen['model']}, top {chosen['top_k']})",
                      **{f"score_{k}": float(r5_scores[k]) for k in "ABC"}})
    ranking = pd.DataFrame(rank_rows)
    ranking["mean_ABC"] = ranking[["score_A", "score_B", "score_C"]].mean(axis=1)
    ranking = ranking.sort_values("mean_ABC", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    ranking.to_csv("output/model_ranking.csv")
    lines += ["", "STEP 5: descriptive ranking of all models (mean score across regimes A, B, C)",
              ranking.round(2).to_string()]

    json.dump(result, open("output/selection_result_round5.json", "w"), indent=2, default=float)
    report = "\n".join(lines)
    open("output/round5_report.txt", "w", encoding="utf-8").write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
