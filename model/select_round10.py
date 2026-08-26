"""Round 10: the synthesis model, as registered (D36, 27 August 2026). Modelling was reopened
by the researcher for exactly this one round; closure re-arms when it completes.

Every component is motivated by an earlier round's finding:
- feature matrix v3 = the round-4 matrix plus three always-available additions (spot-volume
  momentum, transaction-count momentum, exchange-supply-share 90-day change), with no
  late-starting series (round 9's failure mode); the registered importance procedure re-runs
  on v3 and is published;
- a two-horizon HistGradientBoosting committee (30-day timing plus 90-day direction), each
  forecast causally standardised then averaged with fixed equal weights (never combined before);
- recency-weighted training samples (exponential half-life), matching the metric's own
  recency emphasis (untried lever);
- asymmetric multiplier response over the pace allocator with ceiling 12 (round 6's finding).

Grid, single pass, 18 configurations: feature set {top 4, top 5 of v3} x half-life
{730, 1460, none} x (a_pos, a_neg) {(2,1), (2,2), (3,1)}. Baseline: candidate v5's 73.12.

Reproduce:  python -m model.select_round10
Outputs:    output/feature_importance_round10.csv, output/round10_{grid.csv,result.json,report.txt}
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from model.ml import MLConfig, _make_model, build_ml_features, feature_matrix, make_ml_strategy
from model.regimes import load_btc
from model.roundutil import run_round
from model.strategy import PRICE_COL

V5_DEV = 73.12
DEV_DAYS = ("2018-01-01", "2023-06-30")
FIXED = dict(model="hgbr", horizons=(30, 90), horizon=90, m_max=12.0, m_min=0.25, matrix="v3")
GRID = dict(top_k=[4, 5], sample_halflife=[730.0, 1460.0, 0.0], asym=[(2.0, 1.0), (2.0, 2.0), (3.0, 1.0)])


def importance_v3(df: pd.DataFrame) -> pd.DataFrame:
    """The registered importance procedure (D23) on matrix v3, development period only."""
    X = feature_matrix(df, "v3").loc[DEV_DAYS[0]:DEV_DAYS[1]]
    p = df[PRICE_COL].astype(float)
    y = np.log(p.shift(-90) / p).loc[X.index]
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    ic = X.corrwith(y, method="spearman").abs().rename("abs_spearman_ic")
    split = int(len(X) * 0.70)
    m = _make_model("hgbr").fit(X.iloc[:split - 90], y.iloc[:split - 90])
    pi = permutation_importance(m, X.iloc[split:], y.iloc[split:], n_repeats=10, random_state=0)
    t = pd.concat([ic, pd.Series(pi.importances_mean, index=X.columns, name="permutation_importance")], axis=1)
    t["rank_ic"] = t["abs_spearman_ic"].rank(ascending=False)
    t["rank_perm"] = t["permutation_importance"].rank(ascending=False)
    t["avg_rank"] = (t["rank_ic"] + t["rank_perm"]) / 2
    return t.sort_values(["avg_rank", "abs_spearman_ic"], ascending=[True, False], kind="stable")


def main():
    logging.getLogger().setLevel(logging.WARNING)
    df = load_btc()
    imp = importance_v3(df)
    imp.to_csv("output/feature_importance_round10.csv")
    ranked = list(imp.index)
    print("v3 importance order:", ranked, flush=True)

    cache, configs = {}, []
    for k in GRID["top_k"]:
        feats_key = ("committee", tuple(ranked[:k]))
        for hl in GRID["sample_halflife"]:
            for a_pos, a_neg in GRID["asym"]:
                cfg = MLConfig(**FIXED, features=tuple(ranked[:k]),
                               sample_halflife=hl, a_pos=a_pos, a_neg=a_neg, a_ml=a_pos)
                key = (feats_key, hl)
                if key not in cache:
                    cache[key] = build_ml_features(df, cfg)
                configs.append({"label": f"top{k} hl={int(hl) if hl else 'none'} a+={a_pos} a-={a_neg}",
                                "fn": make_ml_strategy(cfg), "feats": cache[key],
                                "top_k": k, "halflife": hl, "a_pos": a_pos, "a_neg": a_neg,
                                "features": ";".join(ranked[:k])})
    run_round("ROUND 10 (synthesis committee)", "D36", "candidate v5", V5_DEV, configs, "round10", df=df)


if __name__ == "__main__":
    main()
