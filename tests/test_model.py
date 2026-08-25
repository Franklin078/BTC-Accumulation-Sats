"""Tests that stand in for the tournament validator, plus reproduction of the published reference.

Run:  python -m pytest tests -q
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model.strategy import (MIN_WEIGHT, PRICE_COL, Params, allocate, compute_weights,  # noqa: E402
                            construct_features, fast_spd_table, make_strategy)
from model.regimes import REGIMES, evaluate, load_btc, score_table  # noqa: E402


def _random_inputs(n, seed):
    rng = np.random.default_rng(seed)
    price = np.exp(np.cumsum(rng.normal(0, 0.05, n))) * 1000
    ma_gap = rng.normal(0, 0.3, n)
    dd = -np.abs(rng.normal(0, 0.3, n))
    mv = rng.normal(0, 1, n)
    fl = rng.normal(0, 1, n)
    r7 = rng.normal(0, 0.1, n)
    return ma_gap, dd, mv, fl, r7, price


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("n", [2, 30, 366, 367])
def test_allocator_constraints(seed, n):
    g, d, z, fl, q, lp = _random_inputs(n, seed)
    for params in [Params(), Params(a_dd=8, m_max=10, bias=1.0, a_flow=0.3), Params(a_dd=0, a_ma=0, bias=-2.0, m_min=0.01)]:
        w = allocate(g, d, z, fl, q, lp, params)
        assert w.shape == (n,)
        assert np.all(w >= MIN_WEIGHT - 1e-15)
        assert np.isclose(w.sum(), 1.0, rtol=1e-5, atol=1e-8)


def test_allocator_is_causal():
    """Changing any future input must not change today's weight."""
    n = 366
    g, d, z, fl, q, lp = _random_inputs(n, 7)
    base = allocate(g, d, z, fl, q, lp, Params(a_flow=0.3))
    for t in [0, 10, 100, 200, 364]:
        g2, d2, z2, fl2, q2, lp2 = (x.copy() for x in (g, d, z, fl, q, lp))
        g2[t + 1:] += 5; d2[t + 1:] -= 0.5; z2[t + 1:] += 3; fl2[t + 1:] += 2; q2[t + 1:] -= 0.2; lp2[t + 1:] *= 3
        alt = allocate(g2, d2, z2, fl2, q2, lp2, Params(a_flow=0.3))
        assert np.allclose(alt[: t + 1], base[: t + 1], rtol=1e-12, atol=1e-15)


def test_features_are_lagged():
    """Setting rows after t to NaN must not change features on or before t."""
    df = load_btc().loc["2020-01-01":"2021-12-31"]
    feats = construct_features(df)
    t = pd.Timestamp("2021-03-15")
    masked = df.copy(); masked.loc[masked.index > t, :] = np.nan
    feats_m = construct_features(masked)
    cols = [c for c in feats.columns if c.startswith("f_")]
    a = feats.loc[:t, cols]; b = feats_m.loc[:t, cols]
    assert np.allclose(a.fillna(-999).to_numpy(), b.fillna(-999).to_numpy(), rtol=1e-12, atol=1e-12)


def test_compute_weights_window_contract():
    df = load_btc()
    w = compute_weights(df.loc["2024-01-01":"2025-01-01"])
    assert len(w) == 367 and (w >= MIN_WEIGHT).all() and np.isclose(w.sum(), 1.0)


def test_fast_path_equals_upstream_engine():
    df = load_btc(); feats = construct_features(df)
    p = Params()
    fast = fast_spd_table(feats, p, "2022-01-01", "2023-06-30")
    spd, _ = evaluate(df, make_strategy(p), feats, type(REGIMES["A"])("T", "test", "2022-01-01", "2023-06-30"))
    assert np.allclose(fast["dynamic_percentile"].to_numpy(), spd["dynamic_percentile"].to_numpy(), atol=1e-9)
    assert np.allclose(fast["uniform_percentile"].to_numpy(), spd["uniform_percentile"].to_numpy(), atol=1e-9)


def test_official_reference_reproduces_published_score():
    """The 2025 tournament template's own strategy must score 72.55 on the frozen parquet (regime C)."""
    from model import reference_2025
    pq = os.path.join(ROOT, "data", "official_2025", "stacking_sats_data.parquet")
    if not os.path.exists(pq):
        pytest.skip("frozen parquet not present")
    df = pd.read_parquet(pq)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    feats = construct_features(df)
    spd, s = evaluate(df, reference_2025.compute_weights, feats, REGIMES["C"])
    assert s["windows"] == 3075
    assert abs(s["score"] - 72.55) < 0.01
    assert abs(s["uniform_rw_spd_pct"] - 41.99) < 0.01
