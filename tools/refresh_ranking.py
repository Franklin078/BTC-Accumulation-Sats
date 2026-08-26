"""Rebuild output/model_ranking.csv from committed sources only: results_regimes.csv for every
registry candidate and baseline (written by notebook 06), and the round result files for the
per-round best configurations that did not become candidates. Descriptive reporting; the final
model is chosen by the closure rule, never from this table.

Reproduce:  python tools/refresh_ranking.py
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

rows = []
reg = pd.read_csv("output/results_regimes.csv")
for m in reg.model.unique():
    sub = reg[reg.model == m]
    if set(sub.regime) >= {"A", "B", "C"}:
        rows.append({"model": m, **{f"score_{k}": float(sub[sub.regime == k]["score"].iloc[0]) for k in "ABC"}})

extras = [("Round 4 best (all-feature ridge, not a candidate)", "output/selection_result_round4.json"),
          ("Round 7 best (v4/v3 blend 0.75, negative round)", "output/round7_result.json"),
          ("Round 8 best (hgbr depth 2, negative round)", "output/round8_result.json"),
          ("Round 9 best (expanded features, negative round)", "output/round9_result.json"),
          ("Round 10 best (synthesis committee, negative round)", "output/round10_result.json")]
for label, path in extras:
    if os.path.exists(path):
        r = json.load(open(path))
        rows.append({"model": label, **{f"score_{k}": float(r[f"regime_{k}"]["score"]) for k in "ABC"}})

ranking = pd.DataFrame(rows).drop_duplicates(subset="model")
ranking["mean_ABC"] = ranking[["score_A", "score_B", "score_C"]].mean(axis=1)
ranking = ranking.sort_values("mean_ABC", ascending=False).reset_index(drop=True)
ranking.index = ranking.index + 1
ranking.to_csv("output/model_ranking.csv")
print(ranking.round(2).to_string())
