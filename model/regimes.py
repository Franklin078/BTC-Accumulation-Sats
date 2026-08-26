"""Scoring regimes and a thin evaluation harness built on the unmodified Trilemma engine.

Nothing in this module re-implements scoring. Every number comes from
``template.prelude_template.compute_cycle_spd`` (Trilemma capstone template,
March 2026), which is byte-identical to upstream (see docs/UPSTREAM_HASHES.txt).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from template.prelude_template import compute_cycle_spd, load_data

PRICE_COL = "PriceUSD_coinmetrics"
RHO = 0.9

# Windows starting on or after this date are sequestered (decision log D32): they complete only
# after the modelling closure of 26 August 2026, no selection decision has ever seen them, and
# they are to be scored exactly once, for the manuscript's final out-of-sample table. evaluate()
# warns if a scoring run includes them so they cannot be consumed by accident.
SEQUESTERED_START = "2025-09-01"


@dataclass(frozen=True)
class Regime:
    key: str
    label: str
    start: str
    end: str | None  # None means "latest complete day in the data"

    def resolve_end(self, df: pd.DataFrame) -> str:
        if self.end is not None:
            return self.end
        return df.index.max().strftime("%Y-%m-%d")


# A: the 2026 Trilemma capstone template constants, exactly as shipped.
# B: the same start, extended to the latest complete day of CoinMetrics data.
# C: the frozen 2025 tournament configuration (for the 72.55 / 94.48 benchmarks).
REGIMES = {
    "A": Regime("A", "Capstone 2026 (2018-01-01 to 2025-12-31)", "2018-01-01", "2025-12-31"),
    "B": Regime("B", "Live-extended (2018-01-01 to latest day)", "2018-01-01", None),
    "C": Regime("C", "Tournament 2025 (2016-01-01 to 2025-06-01)", "2016-01-01", "2025-06-01"),
}


def exp_decay_average(values: np.ndarray, rho: float = RHO) -> float:
    """Recency-weighted mean with weights rho**(N-1-i), normalised to sum 1 (upstream definition)."""
    n = len(values)
    w = rho ** np.arange(n - 1, -1, -1)
    w = w / w.sum()
    return float((values * w).sum())


def score_table(spd: pd.DataFrame) -> dict:
    """Win rate (strict), exp-decay percentile and Final Model Score from an upstream SPD table."""
    dyn = spd["dynamic_percentile"].to_numpy()
    uni = spd["uniform_percentile"].to_numpy()
    win_rate = float((dyn > uni).mean() * 100)
    rw = exp_decay_average(dyn)
    return {
        "windows": int(len(spd)),
        "win_rate": win_rate,
        "rw_spd_pct": rw,
        "score": 0.5 * win_rate + 0.5 * rw,
        "mean_pct": float(dyn.mean()),
        "uniform_rw_spd_pct": exp_decay_average(uni),
        "mean_excess": float((dyn - uni).mean()),
        "min_excess": float((dyn - uni).min()),
    }


def evaluate(df: pd.DataFrame, strategy_fn, features_df: pd.DataFrame, regime: Regime) -> tuple[pd.DataFrame, dict]:
    """Run the upstream rolling-window SPD computation for one regime and summarise it."""
    end = regime.resolve_end(df)
    spd = compute_cycle_spd(df, strategy_fn, features_df=features_df, start_date=regime.start, end_date=end, validate_weights=True)
    starts = pd.to_datetime([w.split(" \u2192 ")[0] for w in spd.index])
    n_seq = int((starts >= pd.Timestamp(SEQUESTERED_START)).sum())
    if n_seq:
        logging.warning(f"evaluate(): {n_seq} sequestered windows (starts >= {SEQUESTERED_START}) "
                        f"included in this run; these are reserved for the one-time final reading (D32)")
    summary = score_table(spd)
    summary.update({"regime": regime.key, "start": regime.start, "end": end})
    return spd, summary


def load_btc() -> pd.DataFrame:
    """Upstream loader (data/Coin Metrics/coinmetrics_btc.csv) with the index guaranteed daily and complete."""
    df = load_data()
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = full.difference(df.index)
    if len(missing):
        raise ValueError(f"{len(missing)} missing days in price data, first {missing[0].date()}")
    return df


def forward_leakage_probe(df: pd.DataFrame, strategy_fn, start: str, end: str, n_probes: int = 50) -> dict:
    """The upstream probe, re-expressed so it returns a result instead of printing.

    For each probe date, every row after the probe is set to NaN and the weight on the
    probe date must be unchanged (rtol 1e-9, atol 1e-12). Identical logic to
    ``check_strategy_submission_ready`` in the tournament and capstone templates.
    """
    backtest_df = df.loc[start:end]
    full_weights = strategy_fn(df).reindex(backtest_df.index).fillna(0.0)
    step = max(len(backtest_df) // n_probes, 1)
    probes = backtest_df.index[::step]
    failures = []
    for probe in probes:
        masked = df.copy()
        masked.loc[masked.index > probe, :] = np.nan
        masked_wt = strategy_fn(masked).reindex(full_weights.index).fillna(0.0)
        if not np.isclose(masked_wt.loc[probe], full_weights.loc[probe], rtol=1e-9, atol=1e-12):
            failures.append((probe.date().isoformat(), float(abs(masked_wt.loc[probe] - full_weights.loc[probe]))))
    return {"probes": int(len(probes)), "failures": failures, "passed": len(failures) == 0}


def constraint_check(df: pd.DataFrame, strategy_fn, start: str, end: str, min_weight: float = 1e-5) -> dict:
    """Per-window: weights >= min_weight and sum to 1 (tournament tolerance rtol 1e-5, atol 1e-8)."""
    offset = pd.DateOffset(years=1)
    bad_min, bad_sum, n = [], [], 0
    for ws in pd.date_range(pd.to_datetime(start), pd.to_datetime(end) - offset, freq="D"):
        we = ws + offset
        w = strategy_fn(df.loc[ws:we])
        n += 1
        if (w < min_weight).any():
            bad_min.append(ws.date().isoformat())
        if not np.isclose(w.sum(), 1.0, rtol=1e-5, atol=1e-8):
            bad_sum.append(ws.date().isoformat())
    return {"windows": n, "below_min_weight": bad_min, "sum_not_one": bad_sum, "passed": not bad_min and not bad_sum}


def configure_logging() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
