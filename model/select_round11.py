"""Round 11: three arms in one registered round (D38, 27 August 2026). Modelling was reopened
by the researcher for exactly this round; closure re-arms when it completes.

Every arm is built from the failure record of the ten earlier rounds, and the three arms attack
the same question, how to use all sixteen always-available features, from different directions:

Arm A  in-model selection: the learner receives the whole v3 matrix with no selection step at
       all, because round 10's loss came from the external importance procedure excluding the
       best features, while trees select internally by split gain (round 4's dilution came
       from a linear model, not from width itself).
Arm B  block committee: the sixteen columns partitioned once into four economic families, one
       small model per family, combined with no fitted weights (equal-weight mean, or a
       three-of-four sign-agreement gate). Correlated features stay inside families, so the
       cross-feature importance leakage that broke round 10 cannot occur, and signals are
       blended rather than allocations, which is what diluted round 7.
Arm C  conditioned v5: candidate v5's engine untouched, with the thirteen non-engine features
       feeding a separate conditioner that scales how hard the engine tilts, never which way.
       Its worst case degrades toward v5, not toward uniform, and it gives round 10's
       asymmetric-response ordering its first test on a winning engine.

Fixed everywhere: pace allocator, ceiling 12, floor multiplier 0.25 (round 6), depth-limited
HistGradientBoosting (rounds 5 and 8), no recency weighting (round 10), asymmetry axis in every
arm. One grid, 27 configurations, one winner across all arms, one hold-out reading.

Anchor: the arm C cell g=0 with a_pos = a_neg = 2 is weight-identical to candidate v5 by
construction and must reproduce the development selection metric 73.12 (scikit-learn 1.9.0);
the script checks this after the grid and reports any deviation as a harness fault.

Reproduce:  python -m model.select_round11
Outputs:    output/round11_{grid.csv,result.json,report.txt}
"""
import json
import logging
import re

import pandas as pd

from model.ml import MLConfig, build_ml_features, feature_matrix, make_ml_strategy
from model.regimes import load_btc
from model.roundutil import run_round

V5_DEV = 73.12
ASYM = [(2.0, 1.0), (2.0, 2.0), (3.0, 1.0)]
ALLOC = dict(shape="pace", m_max=12.0, m_min=0.25, b_quad=0.0)   # candidate v5's allocator

# the registered partition of the sixteen v3 columns into four economic families; the script
# asserts below that it covers the matrix exactly once, so a column added to the matrix later
# cannot silently fall out of arm B
FAMILIES = (
    ("usage", ("hash_mom", "fee_mom", "tx_mom", "vol_usd_mom", "adr_mom")),
    ("valuation", ("mvrv", "roi_1yr", "ma_gap", "drawdown")),
    ("flows", ("netflow_z", "exch_share_chg")),
    ("price_calendar", ("ret_7", "ret_30", "ret_90", "vol_30", "cycle_pos")),
)


def engine_spec() -> dict:
    """Candidate v5's configuration, parsed from the committed round-5 and round-6 results,
    never from constants that could drift if a round were ever re-run."""
    r5 = json.load(open("output/selection_result_round5.json"))
    r6 = json.load(open("output/round6_result.json"))
    m = re.fullmatch(r"shape=(\w+) m_max=([\d.]+) b=([\d.]+)", str(r6["best_label"]))
    if m is None:
        raise ValueError(f"unrecognised round-6 winner label: {r6['best_label']!r}")
    return dict(model=str(r5["chosen"]["model"]), horizon=90, a_ml=float(r5["chosen"]["a_ml"]),
                features=tuple(str(r5["chosen"]["features"]).split(";")),
                shape=m.group(1), m_max=float(m.group(2)), b_quad=float(m.group(3)))


def entry(label: str, arm: str, cache: dict, df: pd.DataFrame, **kw) -> dict:
    cfg = MLConfig(**kw)
    # the frame depends only on what the models see, never on how the allocator responds,
    # so configurations differing only in asymmetry or gain share one cached frame
    key = (cfg.matrix, cfg.features, cfg.horizon, cfg.horizons, cfg.hgbr_depth,
           cfg.blocks, cfg.combine, cfg.cond_features, cfg.cond_depth)
    if key not in cache:
        cache[key] = build_ml_features(df, cfg)
    meta = {k: (";".join(map(str, v)) if isinstance(v, tuple) else v)
            for k, v in kw.items() if k in ("hgbr_depth", "combine", "cond_gain", "a_pos", "a_neg")}
    return {"label": label, "fn": make_ml_strategy(cfg), "feats": cache[key], "arm": arm, **meta}


def main():
    logging.getLogger().setLevel(logging.WARNING)
    df = load_btc()

    v3_cols = set(feature_matrix(df, "v3").columns)
    fam_cols = [c for _, f in FAMILIES for c in f]
    if len(fam_cols) != len(set(fam_cols)) or set(fam_cols) != v3_cols:
        raise AssertionError(f"family partition does not cover the v3 matrix exactly once: "
                             f"families {sorted(fam_cols)} vs matrix {sorted(v3_cols)}")

    eng = engine_spec()
    if eng["shape"] != "pace" or eng["m_max"] != ALLOC["m_max"] or eng["b_quad"] != ALLOC["b_quad"]:
        raise AssertionError(f"round-6 winner allocator {eng} does not match the registered "
                             f"fixed allocator {ALLOC}")
    # arm C runs the engine on the v3 matrix restricted to its three features; that restriction
    # must be value-identical to the v1 matrix the engine was registered on
    ef = list(eng["features"])
    a1, a3 = feature_matrix(df, "v1")[ef], feature_matrix(df, "v3")[ef]
    if not a1.equals(a3):
        raise AssertionError("v3 matrix restricted to the engine features differs from v1")
    cond_feats = tuple(sorted(v3_cols - set(eng["features"])))

    cache, configs = {}, []
    for hz_label, hz in (("H90", ()), ("H30+90", (30, 90))):
        for depth in (2, 3):
            for a_pos, a_neg in ASYM:
                configs.append(entry(
                    f"A: all-v3 {hz_label} depth={depth} a+={a_pos} a-={a_neg}", "A", cache, df,
                    model="hgbr", matrix="v3", features=(), horizon=90, horizons=hz,
                    hgbr_depth=depth, a_ml=2.0, a_pos=a_pos, a_neg=a_neg, **ALLOC))
    for combine in ("avg", "gate"):
        for a_pos, a_neg in ASYM:
            configs.append(entry(
                f"B: families {combine} a+={a_pos} a-={a_neg}", "B", cache, df,
                model="hgbr", matrix="v3", blocks=FAMILIES, combine=combine, horizon=90,
                hgbr_depth=2, a_ml=2.0, a_pos=a_pos, a_neg=a_neg, **ALLOC))
    for g in (0.0, 0.25, 0.5):
        for a_pos, a_neg in ASYM:
            configs.append(entry(
                f"C: cond g={g} a+={a_pos} a-={a_neg}", "C", cache, df,
                model=eng["model"], matrix="v3", features=eng["features"], horizon=eng["horizon"],
                a_ml=eng["a_ml"], cond_features=cond_feats, cond_gain=g, cond_depth=2,
                a_pos=a_pos, a_neg=a_neg, **ALLOC))

    run_round("ROUND 11 (three arms)", "D38", "candidate v5", V5_DEV, configs, "round11", df=df)

    # the anchor cell must reproduce candidate v5's development selection metric; a deviation
    # means the harness, not the market, produced this round's numbers
    grid = pd.read_csv("output/round11_grid.csv")
    anchor = grid[grid["label"] == "C: cond g=0.0 a+=2.0 a-=2.0"]
    if len(anchor) != 1:
        print("ANCHOR CHECK FAILED: anchor cell missing from the grid")
    else:
        got = float(anchor["selection_metric"].iloc[0])
        ok = abs(got - V5_DEV) < 0.02
        print(f"anchor check: C g=0 (2,2) dev selection {got:.4f} vs candidate v5 {V5_DEV} -> "
              + ("PASS" if ok else "FAIL, HARNESS FAULT, round invalid until explained"))


if __name__ == "__main__":
    main()
