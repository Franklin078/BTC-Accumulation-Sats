"""Cross-file consistency: the registry, the round results, the adjudication and the ranking
must agree with each other, so a number can never appear in two places with two values.

Run:  python -m pytest tests/test_consistency.py -q
"""
import json
import os
import re
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def _load(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    return json.load(open(path))


def test_registry_v4_matches_round5():
    reg = json.load(open("model/candidates.json"))
    r5 = _load("output/selection_result_round5.json")
    v4 = reg["Candidate v4 (feature-prioritised ML, round 5)"]
    assert v4["model"] == r5["chosen"]["model"]
    assert v4["features"] == str(r5["chosen"]["features"]).split(";")
    assert float(v4["a_ml"]) == float(r5["chosen"]["a_ml"])


def test_registry_v5_matches_round6_label():
    reg = json.load(open("model/candidates.json"))
    r6 = _load("output/round6_result.json")
    v5 = reg["Candidate v5 (v4 signal, ceiling 12, round 6)"]
    m = re.fullmatch(r"shape=(\w+) m_max=([\d.]+) b=([\d.]+)", str(r6["best_label"]))
    assert m, r6["best_label"]
    assert v5["shape"] == m.group(1)
    assert float(v5["m_max"]) == float(m.group(2))
    assert float(v5["b_quad"]) == float(m.group(3))


def test_adjudication_is_the_running_max():
    adj = _load("output/final_adjudication.json")
    vals = {name: r["dev_metrics"]["selection_metric"] for name, r in adj["round_results"].items()}
    assert abs(adj["standing_value"] - max(vals.values())) < 1e-9
    for name, r in adj["round_results"].items():
        assert r["dev_metrics"]["selection_metric"] <= adj["standing_value"] + 1e-9, name


def test_registry_v6_matches_round11_label():
    reg = json.load(open("model/candidates.json"))
    r11 = _load("output/round11_result.json")
    v6 = reg["Candidate v6 (v5 with asymmetric response, round 11)"]
    m = re.fullmatch(r"C: cond g=([\d.]+) a\+=([\d.]+) a-=([\d.]+)", str(r11["best_label"]))
    assert m, r11["best_label"]
    assert float(m.group(1)) == 0.0, "an active conditioner cannot be held by this registry entry"
    assert float(v6["a_pos"]) == float(m.group(2))
    assert float(v6["a_neg"]) == float(m.group(3))
    v5 = reg["Candidate v5 (v4 signal, ceiling 12, round 6)"]
    for k in ("model", "horizon", "a_ml", "features", "shape", "m_max", "b_quad"):
        assert v6[k] == v5[k], k


def test_registry_v7_matches_round12_label():
    reg = json.load(open("model/candidates.json"))
    r12 = _load("output/round12_result.json")
    v7 = reg["Candidate v7 (refined asymmetric response, round 12)"]
    m = re.fullmatch(r"R: a\+=([\d.]+) a-=([\d.]+) mmax=([\d.]+)", str(r12["best_label"]))
    assert m, r12["best_label"]
    assert float(v7["a_pos"]) == float(m.group(1))
    assert float(v7["a_neg"]) == float(m.group(2))
    assert float(v7["m_max"]) == float(m.group(3))
    v5 = reg["Candidate v5 (v4 signal, ceiling 12, round 6)"]
    for k in ("model", "horizon", "a_ml", "features", "shape", "b_quad"):
        assert v7[k] == v5[k], k


def test_round12_anchor_reproduced_v6():
    r12 = _load("output/round12_result.json")
    ac = r12.get("anchor_check")
    assert ac and ac.get("passed") is True, "round 12's anchor cell did not reproduce candidate v6"
    r11 = _load("output/round11_result.json")
    assert abs(float(ac["target"]) - float(r11["dev_metrics"]["selection_metric"])) < 1e-12


def test_round_results_carry_fingerprints():
    for n in (6, 7, 8, 9, 10, 11):
        r = _load(f"output/round{n}_result.json")
        assert r.get("fingerprint"), f"round {n} result has no fingerprint"


def test_ranking_rows_match_their_sources():
    if not os.path.exists("output/model_ranking.csv"):
        pytest.skip("ranking not built")
    ranking = pd.read_csv("output/model_ranking.csv", index_col=0).set_index("model")
    reg = pd.read_csv("output/results_regimes.csv")
    for m in reg.model.unique():
        if m in ranking.index:
            for k in "ABC":
                want = float(reg[(reg.model == m) & (reg.regime == k)]["score"].iloc[0])
                assert abs(ranking.loc[m, f"score_{k}"] - want) < 1e-6, (m, k)


def test_gates_passed_for_every_round_winner():
    for n in (6, 7, 8, 9, 10, 11):
        r = _load(f"output/round{n}_result.json")
        assert r.get("probe_passed") is True, f"round {n} winner failed or skipped the probe"
        assert r.get("constraints_passed") is True, f"round {n} winner failed constraints"
