"""Candidate registry: the named models produced by the registered selection rounds.

`build_registry()` assembles model/candidates.json from the committed round outputs
(final_params.json for v1, selection_result_round2.json for v2, and the round 3 results
if their winners beat the standing candidate). `load_candidates()` turns the registry
into named strategy functions for scoring. Notebooks 04 and 06 use these, so the
comparison tables always reflect exactly what the rounds produced.

Reproduce:  python -m model.candidates   (rebuilds candidates.json)
"""
import json
import os

from model.strategy import Params, make_strategy

FIXED_V2 = dict(ma_len=200, ma_asym=True, a_win=0.0, m_min=0.25, mvrv_mode="level", mvrv0=1.8)
REG_PATH = "model/candidates.json"


def build_registry() -> dict:
    reg = {}
    reg["Candidate v1 (drawdown-led, round 1)"] = {
        "type": "params", "params": json.load(open("model/final_params.json"))}
    r2 = json.load(open("output/selection_result_round2.json"))
    v2_params = {**FIXED_V2, **{k: float(v) for k, v in r2["chosen"].items()}}
    reg["Candidate v2 (MVRV + netflow, round 2)"] = {"type": "params", "params": v2_params}
    if os.path.exists("output/selection_result_round3a.json"):
        r3a = json.load(open("output/selection_result_round3a.json"))
        if r3a.get("beats_v2"):
            a = float(max(r3a["table"], key=lambda r: r["selection_metric"])["alpha_v1"])
            reg[f"Round 3a blend (alpha_v1 = {a:.2f})"] = {
                "type": "blend", "alpha_v1": a,
                "v1_params": reg["Candidate v1 (drawdown-led, round 1)"]["params"],
                "v2_params": v2_params}
    if os.path.exists("output/selection_result_round3b.json"):
        r3b = json.load(open("output/selection_result_round3b.json"))
        if r3b.get("beats_v2"):
            p = {**FIXED_V2, "a_ma": 0.0, "a_dd": 0.0, "bias": 0.0, "m_max": 5.0,
                 **{k: float(v) for k, v in r3b["chosen"].items()}}
            reg["Candidate v3 (refined MVRV + netflow, round 3b)"] = {"type": "params", "params": p}
    json.dump(reg, open(REG_PATH, "w"), indent=2)
    return reg


def load_candidates() -> dict:
    """Return {name: strategy_function} for every registered candidate."""
    reg = json.load(open(REG_PATH))
    out = {}
    for name, spec in reg.items():
        if spec["type"] == "params":
            out[name] = make_strategy(Params(**spec["params"]))
        elif spec["type"] == "blend":
            f1 = make_strategy(Params(**spec["v1_params"]))
            f2 = make_strategy(Params(**spec["v2_params"]))
            a = float(spec["alpha_v1"])
            def fn(df_window, _f1=f1, _f2=f2, _a=a):
                return _a * _f1(df_window) + (1.0 - _a) * _f2(df_window)
            out[name] = fn
    return out


if __name__ == "__main__":
    reg = build_registry()
    print("registry written:", REG_PATH)
    for k in reg:
        print(" -", k)
