"""Rounds 6 to 9, as registered (D25 to D28, 26 August 2026), run in sequence under the
closure rule D29: modelling ends when round 9 completes; the final model is the candidate
with the best development selection metric across all rounds; the hold-out is reported
beside it and never used to choose.

Round 6  allocator shape on candidate v4's signal (pace vs tailpay, m_max, convexity)
Round 7  ensembles (v4 + v3 weight blends; prediction averaging; stacking)
Round 8  ML variants (quantile and classification targets; other model classes; hgbr refinement)
Round 9  feature expansion (matrix v2 including exchange-supply share and Fear and Greed),
         with the registered importance procedure re-run and published

Reproduce:  python -m model.select_rounds6to9
Outputs:    output/round{6,7,8,9}_{grid.csv,result.json,report.txt},
            output/feature_importance_round9.csv, output/final_adjudication.json
"""
import json

import numpy as np
import pandas as pd

from model.candidates import load_candidates
from model.ml import MLConfig, build_ml_features, fetch_fear_greed, make_ml_strategy
from model.roundutil import run_round
from model.regimes import load_btc

# candidate v4's configuration and development metric come from the committed round-5 result,
# never from constants that could drift if round 5 were ever re-run
_r5 = json.load(open("output/selection_result_round5.json"))
V4 = dict(model=str(_r5["chosen"]["model"]), horizon=90, a_ml=float(_r5["chosen"]["a_ml"]),
          features=tuple(str(_r5["chosen"]["features"]).split(";")))
V4_DEV = float(_r5["dev_metrics"]["selection_metric"])


def ml_config_entry(label, feats_cache, df, **kw):
    cfg = MLConfig(**{**V4, **kw})
    key = (cfg.model, cfg.horizon, cfg.features, cfg.matrix, cfg.hgbr_depth, cfg.hgbr_lr)
    if key not in feats_cache:
        feats_cache[key] = build_ml_features(df, cfg)
    return {"label": label, "fn": make_ml_strategy(cfg), "feats": feats_cache[key],
            **{k: (";".join(v) if isinstance(v, tuple) else v) for k, v in kw.items()}}


def main():
    df = load_btc()
    cache = {}
    standing_name, standing_value = "candidate v4", V4_DEV

    # ---------------- round 6: allocator shape on v4's signal ----------------
    configs = []
    for shape in ("pace", "tailpay"):
        for m_max in (5.0, 8.0, 12.0):
            for b in (0.0, 0.3):
                configs.append(ml_config_entry(f"shape={shape} m_max={m_max} b={b}", cache, df,
                                               shape=shape, m_max=m_max, b_quad=b))
    r6 = run_round("ROUND 6 (allocator shape)", "D25", standing_name, standing_value, configs, "round6", df=df)
    if r6["beats_baseline"]:
        standing_name, standing_value = f"round 6 winner ({r6['best_label']})", r6["dev_metrics"]["selection_metric"]

    # ---------------- round 7: ensembles ----------------
    cands = load_candidates()
    f_v4 = cands["Candidate v4 (feature-prioritised ML, round 5)"]
    f_v3 = cands["Candidate v3 (refined MVRV + netflow, round 3b)"]
    base_feats = cache[next(k for k in cache if k[0] == "hgbr" and k[3] == "v1" and k[4] == 3)]
    # each leg must see the feature frame built with ITS OWN parameters (candidate v3 centres
    # MVRV at 1.6, not the default 1.8); the blend slices each leg's frame by the window's
    # dates, so neither leg consumes features computed under the other's configuration
    from model.strategy import construct_features as _cf, Params as _P
    v3_params = _P(**json.load(open("model/candidates.json"))["Candidate v3 (refined MVRV + netflow, round 3b)"]["params"])
    feats_v3 = _cf(df, v3_params)
    configs = []
    for a in (0.25, 0.50, 0.75):
        def blend(dfw, _a=a):
            idx = dfw.index
            return _a * f_v4(base_feats.loc[idx]) + (1 - _a) * f_v3(feats_v3.loc[idx])
        configs.append({"label": f"weight blend alpha_v4={a}", "fn": blend, "feats": base_feats})
    for a_ml in (1.0, 2.0):
        configs.append(ml_config_entry(f"prediction average a={a_ml}", cache, df, model="predavg", a_ml=a_ml))
        configs.append(ml_config_entry(f"stacking a={a_ml}", cache, df, model="stack", a_ml=a_ml))
    r7 = run_round("ROUND 7 (ensembles)", "D26", standing_name, standing_value, configs, "round7", df=df)
    if r7["beats_baseline"]:
        standing_name, standing_value = f"round 7 winner ({r7['best_label']})", r7["dev_metrics"]["selection_metric"]

    # ---------------- round 8: ML variants ----------------
    configs = []
    for a_ml in (1.0, 2.0):
        configs.append(ml_config_entry(f"quantile-25 target a={a_ml}", cache, df, model="gbr_q25", a_ml=a_ml))
        configs.append(ml_config_entry(f"classifier target a={a_ml}", cache, df, model="gbclf", a_ml=a_ml))
    for kind in ("rf", "enet", "mlp"):
        configs.append(ml_config_entry(f"class={kind} a=2.0", cache, df, model=kind))
    configs.append(ml_config_entry("class=krr top5 a=2.0", cache, df, model="krr",
                                   features=("hash_mom", "fee_mom", "cycle_pos", "adr_mom", "roi_1yr")))
    for d in (2, 3, 4):
        for lr in (0.03, 0.05, 0.1):
            if (d, lr) != (3, 0.05):
                configs.append(ml_config_entry(f"hgbr depth={d} lr={lr}", cache, df, hgbr_depth=d, hgbr_lr=lr))
    r8 = run_round("ROUND 8 (ML variants)", "D27", standing_name, standing_value, configs, "round8", df=df)
    if r8["beats_baseline"]:
        standing_name, standing_value = f"round 8 winner ({r8['best_label']})", r8["dev_metrics"]["selection_metric"]

    # ---------------- round 9: feature expansion ----------------
    fg = fetch_fear_greed()
    df2 = df.join(fg, how="left")
    from model.ml import feature_matrix
    X2 = feature_matrix(df2, "v2").loc["2018-01-01":"2023-06-30"]
    p = df2["PriceUSD_coinmetrics"].astype(float)
    y = np.log(p.shift(-90) / p).loc[X2.index]
    ok = X2.notna().all(axis=1) & y.notna()
    X2, y = X2[ok], y[ok]
    from sklearn.inspection import permutation_importance
    from model.ml import _make_model
    ic = X2.corrwith(y, method="spearman").abs().rename("abs_spearman_ic")
    split = int(len(X2) * 0.70)
    m = _make_model("hgbr").fit(X2.iloc[:split - 90], y.iloc[:split - 90])
    pi = permutation_importance(m, X2.iloc[split:], y.iloc[split:], n_repeats=10, random_state=0)
    imp = pd.concat([ic, pd.Series(pi.importances_mean, index=X2.columns, name="permutation_importance")], axis=1)
    imp["rank_ic"] = imp["abs_spearman_ic"].rank(ascending=False)
    imp["rank_perm"] = imp["permutation_importance"].rank(ascending=False)
    imp["avg_rank"] = (imp["rank_ic"] + imp["rank_perm"]) / 2
    imp = imp.sort_values(["avg_rank", "abs_spearman_ic"], ascending=[True, False], kind="stable")
    imp.to_csv("output/feature_importance_round9.csv")
    ranked = list(imp.index)
    print("round 9 priority order:", ranked, flush=True)
    configs = []
    for mdl in ("ridge", "hgbr"):
        for k in (3, 5):
            for a_ml in (1.0, 2.0):
                configs.append(ml_config_entry(f"{mdl} v2-top{k} a={a_ml}", cache, df2, model=mdl,
                                               matrix="v2", features=tuple(ranked[:k]), a_ml=a_ml))
    r9 = run_round("ROUND 9 (feature expansion)", "D28", standing_name, standing_value, configs, "round9", df=df2)
    if r9["beats_baseline"]:
        standing_name, standing_value = f"round 9 winner ({r9['best_label']})", r9["dev_metrics"]["selection_metric"]

    # ---------------- closure: adjudication under D29 ----------------
    adjudication = {
        "rule": "D29: best development selection metric across all registered rounds; hold-out reported, never used to choose",
        "standing_best": standing_name, "standing_value": standing_value,
        "round_results": {"round6": r6, "round7": r7, "round8": r8, "round9": r9},
    }
    json.dump(adjudication, open("output/final_adjudication.json", "w"), indent=2, default=float)
    print(f"\nCLOSURE (D29): standing best after all rounds = {standing_name} at {standing_value:.2f}")


if __name__ == "__main__":
    main()
