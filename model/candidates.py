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
    if os.path.exists("output/selection_result_round5.json"):
        r5 = json.load(open("output/selection_result_round5.json"))
        if r5.get("beats_v3"):
            ch = r5["chosen"]
            reg["Candidate v4 (feature-prioritised ML, round 5)"] = {
                "type": "ml", "model": str(ch["model"]), "horizon": 90,
                "a_ml": float(ch["a_ml"]), "features": str(ch["features"]).split(";")}
    if os.path.exists("output/round6_result.json"):
        r6 = json.load(open("output/round6_result.json"))
        if r6.get("beats_baseline"):
            import re as _re
            m = _re.fullmatch(r"shape=(\w+) m_max=([\d.]+) b=([\d.]+)", str(r6["best_label"]))
            if m is None:
                raise ValueError(f"unrecognised round-6 winner label: {r6['best_label']!r}")
            v4spec = reg["Candidate v4 (feature-prioritised ML, round 5)"]
            reg["Candidate v5 (v4 signal, ceiling 12, round 6)"] = {
                "type": "ml", "model": v4spec["model"], "horizon": v4spec["horizon"],
                "a_ml": v4spec["a_ml"], "features": v4spec["features"],
                "shape": m.group(1), "m_max": float(m.group(2)), "b_quad": float(m.group(3))}
    if os.path.exists("output/round11_result.json"):
        r11 = json.load(open("output/round11_result.json"))
        if r11.get("beats_baseline"):
            import re as _re
            m = _re.fullmatch(r"C: cond g=([\d.]+) a\+=([\d.]+) a-=([\d.]+)", str(r11["best_label"]))
            if m is None:
                raise ValueError(f"unrecognised round-11 winner label: {r11['best_label']!r}")
            if float(m.group(1)) != 0.0:
                raise ValueError("round-11 winner carries an active conditioner; the registry "
                                 "schema needs conditioner fields before it can hold this model")
            v5spec = reg["Candidate v5 (v4 signal, ceiling 12, round 6)"]
            # at gain 0 the conditioner is weight-inert by construction, so this winner's
            # weight function is exactly candidate v5's with the asymmetric response
            reg["Candidate v6 (v5 with asymmetric response, round 11)"] = {
                **v5spec, "a_pos": float(m.group(2)), "a_neg": float(m.group(3))}
    json.dump(reg, open(REG_PATH, "w"), indent=2)
    return reg


def load_candidates_with_feats(df) -> dict:
    """Return {name: (strategy_function, feature_frame)} where every candidate is paired with
    the feature frame built under ITS OWN configuration. Scoring an ML candidate against a
    frame that lacks its signal column silently degenerates it to uniform pacing, so any
    harness that scores candidates must use this, never a single shared frame."""
    from model.ml import MLConfig, build_ml_features
    from model.strategy import construct_features
    reg = json.load(open(REG_PATH))
    fns = load_candidates()
    out = {}
    for name, spec in reg.items():
        if spec["type"] == "ml":
            cfg = MLConfig(model=spec["model"], horizon=int(spec["horizon"]), a_ml=float(spec["a_ml"]),
                           features=tuple(spec["features"]), m_max=float(spec.get("m_max", 5.0)),
                           shape=str(spec.get("shape", "pace")), b_quad=float(spec.get("b_quad", 0.0)),
                           a_pos=float(spec.get("a_pos", 0.0)), a_neg=float(spec.get("a_neg", 0.0)))
            out[name] = (fns[name], build_ml_features(df, cfg))
        elif spec["type"] == "blend":
            out[name] = (fns[name], construct_features(df, Params(**spec["v1_params"])))
        else:
            out[name] = (fns[name], construct_features(df, Params(**spec["params"])))
    return out


def load_candidates() -> dict:
    """Return {name: strategy_function} for every registered candidate."""
    reg = json.load(open(REG_PATH))
    out = {}
    for name, spec in reg.items():
        if spec["type"] == "params":
            out[name] = make_strategy(Params(**spec["params"]))
        elif spec["type"] == "ml":
            from model.ml import MLConfig, make_ml_strategy
            cfg = MLConfig(model=spec["model"], horizon=int(spec["horizon"]), a_ml=float(spec["a_ml"]),
                           features=tuple(spec["features"]), m_max=float(spec.get("m_max", 5.0)),
                           shape=str(spec.get("shape", "pace")), b_quad=float(spec.get("b_quad", 0.0)),
                           a_pos=float(spec.get("a_pos", 0.0)), a_neg=float(spec.get("a_neg", 0.0)))
            out[name] = make_ml_strategy(cfg)
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
