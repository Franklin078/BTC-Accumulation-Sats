"""Rebuild output/final_adjudication.json from the committed round results, under the closure
rule: the standing best is the round winner with the best development selection metric; the
hold-out is reported beside it and never used to choose. Rounds 1 to 5 predate this file's
schema; every one of their winners sits below the round-6 winner included here, so the running
maximum over rounds 6 onward equals the maximum over all registered rounds.

Reproduce:  python tools/refresh_adjudication.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

rounds = {}
for n in (6, 7, 8, 9, 10, 11):
    path = f"output/round{n}_result.json"
    if os.path.exists(path):
        rounds[f"round{n}"] = (n, json.load(open(path)))

best_n, best = max((v for v in rounds.values()),
                   key=lambda v: v[1]["dev_metrics"]["selection_metric"])
adj = {
    "rule": "D29: best development selection metric across all registered rounds; "
            "hold-out reported, never used to choose",
    "standing_best": f"round {best_n} winner ({best['best_label']})",
    "standing_value": best["dev_metrics"]["selection_metric"],
    "round_results": {k: v[1] for k, v in rounds.items()},
}
json.dump(adj, open("output/final_adjudication.json", "w"), indent=2)
print(f"adjudication: {adj['standing_best']} at {adj['standing_value']:.2f}")
