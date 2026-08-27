"""Round 12: three arms in one registered round (D40, 27 August 2026). Modelling was reopened
by the researcher for exactly this round; closure re-arms when it completes.

All three arms build on the round-11 finding that the allocator's asymmetric response, not
wider features, is where the remaining performance lives:

Arm R  boundary refinement: the round-11 winner (a_pos 3, a_neg 1, ceiling 12) sat on the
       edge of its registered grid, exactly as round 2's winner did before the round-3b
       refinement produced candidate v3; slope and ceiling interact, so both are widened
       together. The cell (3, 1, 12) is candidate v6 itself and doubles as the harness anchor.
Arm T  two-horizon committee on the proven engine: the 30-plus-90-day committee has only ever
       been tested on broken or overwide feature sets (rounds 10 and 11); this is its first
       run on the three features that actually forecast.
Arm P  deterministic cycle-phase aggression: every learned modulator has failed, but the
       halving calendar is not learned; the response scales up during a registered phase of
       the cycle, with the phase definition, the scaled target and the depth all on the grid.
       The indicator is a pure function of the date, so it trains nothing and cannot leak.

Fixed everywhere: candidate v4's signal engine, pace allocator, floor multiplier 0.25,
convexity 0. One grid, 33 configurations, one winner across all arms, one hold-out reading.
Baseline to beat: candidate v6's development selection metric, read from the committed
round-11 result.

Reproduce:  python -m model.select_round12
Outputs:    output/round12_{grid.csv,result.json,report.txt}
"""
import json
import logging

import pandas as pd

from model.ml import MLConfig, build_ml_features, make_ml_strategy
from model.regimes import load_btc
from model.roundutil import run_round

V6_NAME = "Candidate v6 (v5 with asymmetric response, round 11)"


def v6_spec() -> tuple[dict, float]:
    """Candidate v6's configuration from the committed registry, and its development
    selection metric from the committed round-11 result, never from constants."""
    reg = json.load(open("model/candidates.json"))[V6_NAME]
    r11 = json.load(open("output/round11_result.json"))
    base = dict(model=str(reg["model"]), horizon=int(reg["horizon"]), a_ml=float(reg["a_ml"]),
                features=tuple(reg["features"]), shape=str(reg["shape"]),
                b_quad=float(reg["b_quad"]), m_min=0.25)
    if (float(reg["a_pos"]), float(reg["a_neg"]), float(reg["m_max"])) != (3.0, 1.0, 12.0):
        raise AssertionError(f"registry v6 {reg} does not match the registered anchor (3, 1, 12)")
    return base, float(r11["dev_metrics"]["selection_metric"])


def entry(label: str, arm: str, cache: dict, df: pd.DataFrame, **kw) -> dict:
    cfg = MLConfig(**kw)
    # the frame depends on what the models see and on the phase indicator's definition, never
    # on slopes, ceilings, depths or targets, so those grid axes share cached frames
    key = (cfg.matrix, cfg.features, cfg.horizon, cfg.horizons, cfg.hgbr_depth,
           bool(cfg.phase_depth), cfg.phase_def if cfg.phase_depth else "")
    if key not in cache:
        cache[key] = build_ml_features(df, cfg)
    meta = {k: (";".join(map(str, v)) if isinstance(v, tuple) else v)
            for k, v in kw.items()
            if k in ("m_max", "a_pos", "a_neg", "horizons", "phase_depth", "phase_def", "phase_target")}
    return {"label": label, "fn": make_ml_strategy(cfg), "feats": cache[key], "arm": arm, **meta}


def main():
    logging.getLogger().setLevel(logging.WARNING)
    df = load_btc()
    base, v6_dev = v6_spec()

    cache, configs = {}, []
    for a_pos in (3.0, 4.0, 5.0):
        for a_neg in (0.5, 1.0, 1.5):
            for m_max in (12.0, 16.0):
                configs.append(entry(f"R: a+={a_pos} a-={a_neg} mmax={m_max}", "R", cache, df,
                                     **base, m_max=m_max, a_pos=a_pos, a_neg=a_neg))
    for a_pos, a_neg in ((2.0, 1.0), (3.0, 1.0), (4.0, 1.0)):
        configs.append(entry(f"T: H30+90 top3 a+={a_pos} a-={a_neg}", "T", cache, df,
                             **base, horizons=(30, 90), m_max=12.0, a_pos=a_pos, a_neg=a_neg))
    for pdef in ("mid", "late"):
        for target in ("buy", "slopes", "ceiling"):
            for depth in (0.25, 0.5):
                configs.append(entry(f"P: {pdef} {target} depth={depth}", "P", cache, df,
                                     **base, m_max=12.0, a_pos=3.0, a_neg=1.0,
                                     phase_depth=depth, phase_def=pdef, phase_target=target))
    if len(configs) != 33 or len({c["label"] for c in configs}) != 33:
        raise AssertionError("the registered grid is 33 uniquely labelled configurations")

    run_round("ROUND 12 (three arms)", "D40", "candidate v6", v6_dev, configs, "round12", df=df)

    # the anchor cell is candidate v6 itself and must reproduce its development selection
    # metric at full precision (the target comes from the committed round-11 result and the
    # pipeline is deterministic); a deviation means the harness, not the market, produced
    # this round's numbers, so the verdict is stamped into the result file and a failure
    # aborts with a non-zero exit rather than scrolling past
    grid = pd.read_csv("output/round12_grid.csv")
    anchor = grid[grid["label"] == "R: a+=3.0 a-=1.0 mmax=12.0"]
    got = float(anchor["selection_metric"].iloc[0]) if len(anchor) == 1 else float("nan")
    ok = len(anchor) == 1 and abs(got - v6_dev) <= 1e-9 * max(abs(v6_dev), 1.0)
    result = json.load(open("output/round12_result.json"))
    result["anchor_check"] = {"label": "R: a+=3.0 a-=1.0 mmax=12.0", "value": got,
                              "target": v6_dev, "passed": bool(ok)}
    json.dump(result, open("output/round12_result.json", "w"), indent=2, default=float)
    print(f"anchor check: R (3,1,12) dev selection {got:.10f} vs candidate v6 {v6_dev:.10f} -> "
          + ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit("ANCHOR CHECK FAILED: harness fault; the round 12 result is invalid "
                         "and must not be adjudicated")


if __name__ == "__main__":
    main()
