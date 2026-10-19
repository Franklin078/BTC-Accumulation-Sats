"""Tables for the manuscript, derived from the committed results and the frozen data.

Nothing here selects, tunes or re-scores a model. Every file written to output/manuscript/ is
either read straight out of a committed result file or is arithmetic on the frozen CoinMetrics
series, the committed candidate parameters and the committed allocator, so each number quoted
in the manuscript can be traced to one file produced by this script. The index of files and
their provenance is written to output/manuscript/INDEX.md.

Run from the repository root, in the environment whose scikit-learn version matches the
committed learner results (1.9.0):

    py -3.14 tools/manuscript_support.py

The learner-dependent part (the final model's multiplier path) records the scikit-learn
version it ran under, as every learner number in this project does.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from model.regimes import load_btc, REGIMES, exp_decay_average  # noqa: E402
from model.strategy import construct_features, make_strategy, Params, MIN_WEIGHT  # noqa: E402
from model.ml import (MLConfig, feature_matrix, build_ml_features, _multiplier,  # noqa: E402
                      _allocate_pace, HALVINGS)
from model.roundutil import HOLDOUT  # noqa: E402
from model.regimes import evaluate  # noqa: E402

OUT = ROOT / "output" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

ARROW = " → "
INDEX: list[tuple[str, str]] = []


def save_json(name: str, obj, note: str) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf8")
    INDEX.append((name, note))


def save_csv(name: str, frame: pd.DataFrame, note: str, index: bool = False) -> None:
    frame.to_csv(OUT / name, index=index)
    INDEX.append((name, note))


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m-%d")
    raise TypeError(str(type(x)))


def window_starts_of(table: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime([s.split(ARROW)[0] for s in table.index])


def read_spd(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "output" / "spd_tables" / f"{name}.csv", index_col=0)


# ---------------------------------------------------------------- 1. data splits and periods
def splits(df_close: pd.DataFrame, data_close: str) -> dict:
    close = pd.Timestamp(data_close)
    last_start = close - pd.DateOffset(years=1)
    dev = pd.date_range("2018-01-01", "2023-06-30", freq="D")
    emb = pd.date_range("2023-07-01", "2024-06-30", freq="D")
    hold = pd.date_range("2024-07-01", last_start, freq="D")
    reading = json.loads((ROOT / "output" / "final_reading.json").read_text(encoding="utf8"))
    seq_n = int(next(iter(reading["candidates"].values()))["windows"])
    out = {
        "data_close": data_close,
        "window_convention": "a window starting on day s runs to the same calendar date one year "
                             "later, both ends inclusive (template/prelude_template.py, "
                             "WINDOW_OFFSET = pd.DateOffset(years=1)); 366 days unless the span "
                             "contains no 29 February",
        "development": {"first_start": "2018-01-01", "last_start": "2023-06-30", "windows": len(dev),
                        "used_for": "choosing between configurations (the development selection metric)"},
        "embargo": {"first_start": "2023-07-01", "last_start": "2024-06-30", "windows": len(emb),
                    "used_for": "nothing; separates development from hold-out because consecutive windows share up to 365 days"},
        "hold_out": {"first_start": "2024-07-01", "last_start": last_start.strftime("%Y-%m-%d"), "windows": len(hold),
                     "used_for": "scored once per round winner for the record, never used to choose"},
        "sequestered": {"first_start": "2025-09-01", "windows_read": seq_n, "read_on": reading["read_on"],
                        "data_last_day_at_reading": reading["data_last_day"],
                        "used_for": "one reporting table, read once"},
        "check_period_B_total": len(dev) + len(emb) + len(hold),
    }
    # Period bounds come from the committed scores (results_regimes.csv, data close 24 August 2026).
    # The spd_tables/final_*.csv, uniform_*.csv, tournament_*.csv and upstream_*.csv files were
    # written at the 22 August close of the first commit (2,791 windows on B) and "final" there
    # meant candidate v1; the robust_v5_*.csv tables are the final model's windows at the 24 August close.
    res = pd.read_csv(ROOT / "output" / "results_regimes.csv")
    periods = {}
    for key in ("A", "B", "C"):
        r = res[(res["model"].str.startswith("Candidate v5")) & (res["regime"] == key)].iloc[0]
        end = pd.Timestamp(r["end"])
        periods[key] = {"first_start": r["start"], "last_start": (end - pd.DateOffset(years=1)).strftime("%Y-%m-%d"),
                        "last_end": r["end"], "windows": int(r["windows"]), "label": REGIMES[key].label}
    out["evaluation_periods"] = periods
    out["note_on_committed_window_tables"] = ("output/spd_tables/final_*, uniform_*, tournament_* and upstream_* were written at the 22 August 2026 "
                                              "close (2,791 windows on B) and 'final' there is candidate v1; robust_v5_* are the final model at the "
                                              "24 August close; the benchmark tables at the 24 August close are regenerated by this script")
    assert out["check_period_B_total"] == periods["B"]["windows"], "split counts do not add up to period B"
    save_json("splits.json", out, "window counts and date bounds of the four data splits and the three evaluation periods")
    return out


# ---------------------------------------------------------------- 2. score decomposition
def decomposition() -> pd.DataFrame:
    res = pd.read_csv(ROOT / "output" / "results_regimes.csv")
    rank = pd.read_csv(ROOT / "output" / "model_ranking.csv", index_col=0)
    cat = {}
    for _, r in rank.iterrows():
        cat[r["model"]] = r["category"]
    rows = []
    for _, r in res.iterrows():
        rows.append({"model": r["model"], "category": cat.get(r["model"], "unclassified"), "period": r["regime"],
                     "windows": int(r["windows"]), "win_rate": r["win_rate"], "rw_pct": r["rw_spd_pct"],
                     "score": r["score"], "mean_pct": r["mean_pct"], "unweighted_score": 0.5 * r["win_rate"] + 0.5 * r["mean_pct"],
                     "uniform_rw_pct": r["uniform_rw_spd_pct"], "source": "output/results_regimes.csv"})
    # negative rounds whose result files carry the three periods
    for n in range(4, 13):
        f = ROOT / "output" / (f"round{n}_result.json" if n >= 6 else f"selection_result_round{n}.json")
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf8"))
        beats = next((v for k, v in d.items() if k.startswith("beats")), None)
        if beats is not False:
            continue
        label = d.get("best_label") or json.dumps(d.get("chosen", {}))
        for key in ("A", "B", "C"):
            blk = d.get(f"regime_{key}")
            if blk:
                rows.append({"model": f"Round {n} best ({label}), negative round",
                             "category": "unclassified", "period": key, "windows": int(blk["windows"]),
                             "win_rate": blk["win_rate"], "rw_pct": blk["rw_spd_pct"], "score": blk["score"],
                             "mean_pct": blk["mean_pct"], "unweighted_score": 0.5 * blk["win_rate"] + 0.5 * blk["mean_pct"],
                             "uniform_rw_pct": blk["uniform_rw_spd_pct"], "source": f.name})
    t = pd.DataFrame(rows)
    err = (0.5 * t["win_rate"] + 0.5 * t["rw_pct"] - t["score"]).abs().max()
    assert err < 1e-9, f"score is not half win rate plus half recency-weighted percentile (max error {err})"
    save_csv("decomposition.csv", t, "win rate, recency-weighted percentile and score for every model and period; score = 0.5 win + 0.5 RW checked to 1e-9")
    return t


# ---------------------------------------------------------------- 3. worked scoring example
def worked_scoring(df_close: pd.DataFrame, start: str = "2025-01-01") -> dict:
    s = pd.Timestamp(start)
    e = s + pd.DateOffset(years=1)
    price = df_close["PriceUSD_coinmetrics"].loc[s:e]
    spd = 1e8 / price
    uni = spd.mean()
    lo, hi = spd.min(), spd.max()
    pct = (uni - lo) / (hi - lo) * 100
    label = f"{s:%Y-%m-%d}{ARROW}{e:%Y-%m-%d}"
    committed = pd.read_csv(OUT / "spd_uniform_B.csv", index_col=0).loc[label]
    final = read_spd("robust_v5_B").loc[label]
    out = {
        "window": label, "days": int(len(price)),
        "note": "sats per dollar on day d = 1e8 / price_d; uniform DCA's window SPD is the plain mean of the daily values; "
                "the percentile places it between the window's worst and best single-day values",
        "uniform_spd": uni, "min_spd": lo, "min_spd_date": spd.idxmin().strftime("%Y-%m-%d"), "price_at_min_spd": price[spd.idxmin()],
        "max_spd": hi, "max_spd_date": spd.idxmax().strftime("%Y-%m-%d"), "price_at_max_spd": price[spd.idxmax()],
        "uniform_percentile": pct, "start_price": price.iloc[0], "end_price": price.iloc[-1],
        "check_against_committed_uniform_table": {"uniform_spd": committed["uniform_sats_per_dollar"], "uniform_percentile": committed["uniform_percentile"],
                                                   "max_abs_difference": float(max(abs(committed["uniform_sats_per_dollar"] - uni), abs(committed["uniform_percentile"] - pct)))},
        "final_model_on_this_window": {"dynamic_spd": final["dynamic_sats_per_dollar"], "dynamic_percentile": final["dynamic_percentile"],
                                        "excess_percentile": final["excess_percentile"]},
        "btc_per_million_dollars": {"uniform": uni * 1e6 / 1e8, "final_model": final["dynamic_sats_per_dollar"] * 1e6 / 1e8},
    }
    assert out["check_against_committed_uniform_table"]["max_abs_difference"] < 1e-6
    save_json("worked_scoring_2025.json", out, "the four-step scoring of the window starting 1 January 2025, checked against the uniform window table; the final model's row is from output/spd_tables/robust_v5_B.csv")
    daily = pd.DataFrame({"date": price.index.strftime("%Y-%m-%d"), "price_usd": price.values, "sats_per_dollar": spd.values})
    save_csv("worked_scoring_2025_daily.csv", daily, "daily prices and sats per dollar inside the window starting 1 January 2025")
    return out


# ---------------------------------------------------------------- 4. worked allocator example
def worked_allocator() -> dict:
    cfg = MLConfig(model="hgbr", horizon=90, a_ml=2.0, features=("hash_mom", "fee_mom", "cycle_pos"), shape="pace", m_max=12.0)
    floor = MIN_WEIGHT * 1.01

    def trace(mults: list[float], budget_dollars: float) -> pd.DataFrame:
        n = len(mults)
        z = np.array([np.log(m) / cfg.a_ml if m is not None else 0.0 for m in mults])
        w_fn = _allocate_pace(z, cfg)
        rows, remaining = [], 1.0
        for i in range(n):
            days_left = n - i
            if days_left == 1:
                rows.append({"day": i + 1, "budget_remaining": remaining, "days_left": 1, "even_pace": remaining, "multiplier": np.nan,
                             "raw_amount": remaining, "floor": floor, "cap": remaining, "rule": "last day takes the remainder",
                             "weight": remaining, "dollars": remaining * budget_dollars})
                break
            m = _multiplier(z[i], cfg)
            pace = remaining / days_left
            raw = pace * m
            cap = remaining - (days_left - 1) * floor
            w = min(max(raw, floor), cap)
            rule = "pace times multiplier" if w == raw else ("floor binds" if w == floor else "cap binds (reserve for later days)")
            rows.append({"day": i + 1, "budget_remaining": remaining, "days_left": days_left, "even_pace": pace, "multiplier": m,
                         "raw_amount": raw, "floor": floor, "cap": cap, "rule": rule, "weight": w, "dollars": w * budget_dollars})
            remaining -= w
        t = pd.DataFrame(rows)
        assert np.allclose(t["weight"].to_numpy(), w_fn), "trace disagrees with the committed allocator"
        assert abs(t["weight"].sum() - 1.0) < 1e-12
        return t

    five = trace([1.0, 2.0, 0.5, 12.0, None], 1000.0)
    save_csv("worked_allocator_five_day.csv", five, "five-day window with budget 1 through the committed pacing allocator (model/ml.py _allocate_pace), multipliers 1, 2, 0.5, 12 and the remainder day")

    one_day = []
    for m in (0.25, 1.0, 12.0):
        n = 365
        w1 = min(max(m / n, floor), 1 - (n - 1) * floor)
        r2 = 1 - w1
        one_day.append({"multiplier_day_1": m, "weight_day_1": w1, "budget_remaining_day_2": r2, "even_pace_day_2": r2 / (n - 1),
                        "even_pace_day_1": 1 / n})
    save_csv("worked_allocator_day_one.csv", pd.DataFrame(one_day), "day-one arithmetic for a 365-day window at the floor, neutral and ceiling multipliers")

    paths = {}
    n = 365
    for name, mults in (("neutral", [1.0] * n), ("ceiling_30_days", [12.0] * 30 + [1.0] * (n - 30)), ("floor_30_days", [0.25] * 30 + [1.0] * (n - 30))):
        z = np.log(np.array(mults)) / cfg.a_ml
        paths[name] = _allocate_pace(z, cfg)
    p = pd.DataFrame(paths)
    p.insert(0, "day", np.arange(1, n + 1))
    save_csv("weight_paths.csv", p, "daily weights over a 365-day window for three multiplier paths, from the committed allocator")

    resp = pd.DataFrame({"z": [-3, -2, -1, -0.5, 0, 0.5, 1, 1.24, 2, 3]})
    resp["multiplier"] = [_multiplier(z, cfg) for z in resp["z"]]
    resp["exp_2z_unbounded"] = np.exp(2 * resp["z"])
    save_csv("multiplier_response.csv", resp, "the final model's multiplier at selected standardised forecasts; ceiling binds above ln(12)/2 = 1.2425, floor below ln(0.25)/2 = -0.6931")
    return {"five_day_sum": float(five["weight"].sum()), "ceiling_binds_above_z": float(np.log(12) / 2), "floor_binds_below_z": float(np.log(0.25) / 2)}


# ---------------------------------------------------------------- 5. the round record
def round_record(df_close: pd.DataFrame) -> pd.DataFrame:
    reg = {}
    for line in (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf8").splitlines():
        m = re.match(r"\|\s*(\d+[ab]?)\s*\|\s*(D\d+)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]*)\|\s*([^|]*)\|", line)
        if m:
            reg[m.group(1)] = {"registration": m.group(2), "date": m.group(3), "baseline_text": m.group(4).strip(), "outcome_text": m.group(5).strip()}

    def grid_rows(name: str) -> int | None:
        f = ROOT / "output" / name
        return int(len(pd.read_csv(f))) if f.exists() else None

    sources = {
        "1": ("selection_result.json", "selection_grid.csv"), "2": ("selection_result_round2.json", "selection_grid_round2.csv"),
        "3a": ("selection_result_round3a.json", None), "3b": ("selection_result_round3b.json", "selection_grid_round3b.csv"),
        "4": ("selection_result_round4.json", "selection_grid_round4.csv"), "5": ("selection_result_round5.json", "selection_grid_round5.csv"),
    }
    for n in range(6, 13):
        sources[str(n)] = (f"round{n}_result.json", f"round{n}_grid.csv")

    rows = []
    for rnd, (res_name, grid_name) in sources.items():
        d = json.loads((ROOT / "output" / res_name).read_text(encoding="utf8"))
        row = {"round": rnd, "registration": reg.get(rnd, {}).get("registration"), "date": reg.get(rnd, {}).get("date"),
               "configurations": grid_rows(grid_name) if grid_name else (len(d.get("alphas", [])) or None),
               "outcome_text": reg.get(rnd, {}).get("outcome_text"), "baseline_text": reg.get(rnd, {}).get("baseline_text"),
               "source": res_name}
        if rnd == "3a":
            best = max(d["table"], key=lambda r: r["selection_metric"])
            row.update(best_label=f"blend alpha_v1={best['alpha_v1']}", dev_selection=best["selection_metric"], dev_win_rate=best["win_rate"],
                       dev_mean_pct=best["mean_pct"], dev_windows=best["windows"], beats_baseline=d["beats_v2"])
        else:
            dm = d["dev_metrics"]
            row.update(best_label=d.get("best_label") or json.dumps(d.get("chosen", {})), dev_selection=dm["selection_metric"], dev_win_rate=dm["win_rate"],
                       dev_mean_pct=dm["mean_pct"], dev_windows=d.get("dev_windows"),
                       beats_baseline=next((v for k, v in d.items() if k.startswith("beats")), None))
            h = d.get("holdout")
            if h:
                row.update(holdout_windows=h["windows"], holdout_score=h["score"], holdout_win_rate=h["win_rate"], holdout_rw_pct=h["rw_spd_pct"],
                           holdout_mean_pct=h["mean_pct"], holdout_unweighted=0.5 * h["win_rate"] + 0.5 * h["mean_pct"])
        row["baseline_value"] = d.get("baseline_value", d.get("v1_dev_selection_metric"))
        if row["baseline_value"] is None:
            # rounds 3a to 5 recorded the baseline only in the decision log ("v2, 61.31"; "v3, 63.09")
            m = re.search(r"(\d+\.\d+)", row.get("baseline_text") or "")
            row["baseline_value"] = float(m.group(1)) if m else None
        rows.append(row)
    t = pd.DataFrame(rows)

    # Round 1 predates the hold-out reporting in the result files; its hold-out reading is reproduced
    # here from the registered v1 parameters through the engine, exactly as later rounds did it.
    if pd.isna(t.loc[t["round"] == "1", "holdout_score"]).all():
        cands = json.loads((ROOT / "model" / "candidates.json").read_text(encoding="utf8"))
        p1 = Params(**cands["Candidate v1 (drawdown-led, round 1)"]["params"])
        feats = construct_features(df_close, p1)
        _, s = evaluate(df_close, make_strategy(p1), feats, HOLDOUT)
        idx = t.index[t["round"] == "1"][0]
        t.loc[idx, ["holdout_windows", "holdout_score", "holdout_win_rate", "holdout_rw_pct", "holdout_mean_pct"]] = [
            s["windows"], s["score"], s["win_rate"], s["rw_spd_pct"], s["mean_pct"]]
        t.loc[idx, "holdout_unweighted"] = 0.5 * s["win_rate"] + 0.5 * s["mean_pct"]
        t.loc[idx, "source"] = "selection_result.json; hold-out reproduced from the registered v1 parameters by this script"
    bl = pd.to_numeric(t["baseline_value"], errors="coerce")
    t["margin"] = t["dev_selection"] - bl
    candidates = {"1": "v1", "2": "v2", "3b": "v3", "5": "v4", "6": "v5", "11": "v6", "12": "v7"}
    t["candidate"] = t["round"].map(candidates)
    save_csv("round_record.csv", t, "the twelve registered rounds: registration, grid size, development metric, baseline, margin and the hold-out reading in all its parts")
    return t


def round6_summary() -> dict:
    g = pd.read_csv(ROOT / "output" / "round6_grid.csv").sort_values("selection_metric", ascending=False)
    best, second = g.iloc[0], g.iloc[1]
    v4_like = g[(g["shape"] == "pace") & (g["m_max"] == 5.0) & (g["b_quad"] == 0.0)].iloc[0]
    tail = g[g["shape"] == "tailpay"].iloc[0]
    out = {"best": best["label"], "best_metric": best["selection_metric"], "second": second["label"], "second_metric": second["selection_metric"],
           "margin_over_second": best["selection_metric"] - second["selection_metric"],
           "ceiling_5_same_shape": v4_like["label"], "ceiling_5_metric": v4_like["selection_metric"],
           "gain_from_ceiling_5_to_12": best["selection_metric"] - v4_like["selection_metric"],
           "best_tailpay": tail["label"], "best_tailpay_metric": tail["selection_metric"], "configurations": int(len(g))}
    save_json("round6_summary.json", out, "round 6's grid read for the ceiling effect: best, second best, the ceiling-5 cell and the best reference-shape cell")
    return out


# ---------------------------------------------------------------- 6. Chapter 3 measurements
def halving_cycles(df_close: pd.DataFrame) -> pd.DataFrame:
    # One cycle runs from a halving to the next. Its decline is the deepest drawdown from the
    # trailing 365-day high inside the cycle; its peak is the high that drawdown is measured from.
    p = df_close["PriceUSD_coinmetrics"].dropna()
    dd = p / p.rolling(365, min_periods=30).max() - 1
    rows = []
    halv = list(HALVINGS) + [p.index.max() + pd.Timedelta(days=1)]
    for k in range(len(HALVINGS)):
        h, nxt = halv[k], halv[k + 1]
        seg = dd.loc[h:nxt - pd.Timedelta(days=1)]
        trough_date = seg.idxmin()
        peak_date = p.loc[h:trough_date].idxmax()   # the cycle's high before its deepest trough
        rows.append({"halving": h.strftime("%Y-%m-%d"), "price_at_halving": p.loc[h], "peak_date": peak_date.strftime("%Y-%m-%d"),
                     "days_halving_to_peak": (peak_date - h).days, "peak_price": p.loc[peak_date], "trough_date": trough_date.strftime("%Y-%m-%d"),
                     "days_peak_to_trough": (trough_date - peak_date).days, "trough_price": p.loc[trough_date],
                     "peak_to_trough_pct": (p.loc[trough_date] / p.loc[peak_date] - 1) * 100, "complete": k + 1 < len(HALVINGS)})
    t = pd.DataFrame(rows)
    save_csv("halving_cycles.csv", t, "for each halving cycle: the deepest drawdown from the trailing 365-day high, the peak it fell from, and the days from the halving to that peak; the current cycle is incomplete")
    return t


def drawdowns(df_close: pd.DataFrame) -> pd.DataFrame:
    p = df_close["PriceUSD_coinmetrics"]
    dd = p / p.rolling(365, min_periods=30).max() - 1
    rows = []
    for label, a, b in (("2018 to 2019 decline", "2018-01-01", "2019-12-31"), ("2022 decline", "2022-01-01", "2022-12-31"), ("2025 to 2026 decline", "2025-01-01", df_close.index.max().strftime("%Y-%m-%d"))):
        seg = dd.loc[a:b]
        rows.append({"episode": label, "deepest_drawdown_pct": seg.min() * 100, "date": seg.idxmin().strftime("%Y-%m-%d")})
    shares = {f"share_of_days_deeper_than_{int(x*100)}_pct": float((dd.loc["2018-01-01":] < -x).mean()) for x in (0.2, 0.4, 0.6)}
    t = pd.DataFrame(rows)
    save_csv("drawdowns.csv", t, "deepest drawdown from the trailing 365-day high in each major decline, from the frozen series")
    save_json("drawdown_shares.json", shares, "share of days since 2018 spent more than 20, 40 and 60 per cent below the trailing 365-day high")
    return t


def volatility(df_close: pd.DataFrame) -> dict:
    p = df_close["PriceUSD_coinmetrics"].loc["2018-01-01":]
    r = np.log(p).diff().dropna()
    vol = r.rolling(30).std() * np.sqrt(365) * 100
    out = {"from": "2018-01-01", "to": df_close.index.max().strftime("%Y-%m-%d"), "daily_log_return_mean_pct": r.mean() * 100, "daily_log_return_sd_pct": r.std() * 100,
           "skewness": r.skew(), "excess_kurtosis": r.kurt(), "share_beyond_3_sd": float((r.abs() > 3 * r.std()).mean()),
           "share_beyond_3_sd_if_normal": 0.0027, "vol30_median_pct": vol.median(), "vol30_max_pct": vol.max(), "vol30_max_date": vol.idxmax().strftime("%Y-%m-%d"),
           "vol30_min_pct": vol.min(), "max_over_median": vol.max() / vol.median()}
    save_json("volatility.json", out, "return distribution and 30-day realised volatility statistics since 2018")
    return out


def spearman(df_close: pd.DataFrame) -> pd.DataFrame:
    committed = pd.read_csv(ROOT / "output" / "feature_importance_round5.csv", index_col=0)
    X_all = feature_matrix(df_close, "v1")
    p = df_close["PriceUSD_coinmetrics"].astype(float)
    y_all = np.log(p.shift(-90) / p)
    tried = {}
    for label, (a, b) in {"development scored range 2018-01-01 to 2024-06-30": ("2018-01-01", "2024-06-30"),
                          "development starts 2018-01-01 to 2023-06-30": ("2018-01-01", "2023-06-30"),
                          "whole frame": (None, None)}.items():
        X = X_all.loc[a:b] if a else X_all
        y = y_all.loc[X.index]
        ok = X.notna().all(axis=1) & y.notna()
        Xs, ys = X[ok], y[ok]
        ic = Xs.corrwith(ys, method="spearman")
        tried[label] = ic
        if abs(abs(ic["hash_mom"]) - committed.loc["hash_mom", "abs_spearman_ic"]) < 1e-9:
            t = pd.DataFrame({"feature": ic.index, "signed_spearman": ic.values, "abs_spearman": ic.abs().values})
            t = t.merge(committed[["abs_spearman_ic", "permutation_importance", "avg_rank"]], left_on="feature", right_index=True)
            t["n_days"] = int(ok.sum())
            t["first_day"] = Xs.index.min().strftime("%Y-%m-%d")
            t["last_day"] = Xs.index.max().strftime("%Y-%m-%d")
            t["sample"] = label
            t["target"] = "90-day forward logarithmic return"
            t = t.sort_values("abs_spearman", ascending=False)
            assert (t["abs_spearman"] - t["abs_spearman_ic"]).abs().max() < 1e-9
            save_csv("spearman_features.csv", t, "signed Spearman correlation of the thirteen round-4 features with the 90-day forward log return, reproducing round 5's committed absolute values exactly")
            return t
    raise RuntimeError("could not reproduce the committed Spearman values; tried " + ", ".join(tried))


def benchmark_tables(df_close: pd.DataFrame) -> None:
    """Window-level tables for uniform DCA, the 2025 reference and the 2026 template baseline on
    period B at the data close, through the unmodified engine; their scores must equal the
    committed results_regimes.csv rows."""
    from model import reference_2025
    from template.model_development_template import precompute_features, compute_window_weights
    res = pd.read_csv(ROOT / "output" / "results_regimes.csv")
    feats = construct_features(df_close, Params())
    upfeats = precompute_features(df_close)

    def upstream_fn(w):
        if w.empty:
            return pd.Series(dtype=float)
        return compute_window_weights(upfeats, w.index.min(), w.index.max(), w.index.max())

    models = {"uniform": ("Uniform DCA", lambda w: pd.Series(1.0 / len(w), index=w.index)),
              "reference": ("Tournament 2025 reference", reference_2025.compute_weights),
              "upstream": ("Upstream 2026 baseline (200-MA)", upstream_fn)}
    checks = {}
    for tag, (name, fn) in models.items():
        spd, s = evaluate(df_close, fn, feats, REGIMES["B"])
        committed = res[(res["model"] == name) & (res["regime"] == "B")].iloc[0]
        diff = max(abs(s["rw_spd_pct"] - committed["rw_spd_pct"]), abs(s["mean_pct"] - committed["mean_pct"]))
        if tag != "uniform":
            diff = max(diff, abs(s["win_rate"] - committed["win_rate"]), abs(s["score"] - committed["score"]))
        assert diff < 1e-6, f"{name}: regenerated period-B table disagrees with results_regimes.csv by {diff}"
        checks[name] = {"windows": int(s["windows"]), "max_abs_difference_to_committed": float(diff)}
        save_csv(f"spd_{tag}_B.csv", spd, f"window-level table of {name} on evaluation period B at the data close, regenerated through the engine and matched to results_regimes.csv", index=True)
    save_json("benchmark_tables_check.json", checks, "agreement of the regenerated benchmark window tables with the committed scores")


def mvrv_bands(df_close: pd.DataFrame) -> pd.DataFrame:
    uni = pd.read_csv(OUT / "spd_uniform_B.csv", index_col=0)
    starts = window_starts_of(uni)
    mv = df_close["CapMVRVCur"].astype(float).shift(1)   # value known at the start of the window's first day
    p = df_close["PriceUSD_coinmetrics"].astype(float)
    fwd = np.log(p.shift(-365) / p)
    frame = pd.DataFrame({"mvrv_lagged": mv.reindex(starts).values, "uniform_percentile": uni["uniform_percentile"].values,
                          "forward_365d_log_return": fwd.reindex(starts).values}, index=starts).dropna()
    bins = [0, 1, 1.5, 2, 3, np.inf]
    labels = ["below 1", "1 to 1.5", "1.5 to 2", "2 to 3", "above 3"]
    frame["band"] = pd.cut(frame["mvrv_lagged"], bins=bins, labels=labels, right=False)
    g = frame.groupby("band", observed=True).agg(window_starts=("uniform_percentile", "size"), mean_uniform_percentile=("uniform_percentile", "mean"),
                                                 mean_forward_365d_log_return=("forward_365d_log_return", "mean"),
                                                 share_forward_return_positive=("forward_365d_log_return", lambda s: float((s > 0).mean())))
    g = g.reset_index()
    g["first_start"], g["last_start"] = frame.index.min().strftime("%Y-%m-%d"), frame.index.max().strftime("%Y-%m-%d")
    save_csv("mvrv_bands.csv", g, "window starts in evaluation period B grouped by the MVRV level known at the start: uniform DCA's window percentile and the 365-day forward log return")
    return g


def split_profile(df_close: pd.DataFrame) -> pd.DataFrame:
    """Market context and the final model's record on each data split, from the committed
    window tables (final model: robust_v5_B; benchmarks: the regenerated period-B tables)."""
    def starts(t):
        return pd.to_datetime([s.split(ARROW)[0] for s in t.index])
    v5 = read_spd("robust_v5_B"); uni = pd.read_csv(OUT / "spd_uniform_B.csv", index_col=0); ref = pd.read_csv(OUT / "spd_reference_B.csv", index_col=0)
    for t in (v5, uni, ref):
        t["start"] = starts(t)
    p = df_close["PriceUSD_coinmetrics"]
    dd = p / p.rolling(365, min_periods=30).max() - 1
    rows = []
    for name, (a, b) in {"development": ("2018-01-01", "2023-06-30"), "embargo": ("2023-07-01", "2024-06-30"), "hold-out": ("2024-07-01", "2025-08-24")}.items():
        m = (v5["start"] >= a) & (v5["start"] <= b)
        rows.append({"split": name, "windows": int(m.sum()), "first_start": a, "last_start": b,
                     "price_at_first_start": p.loc[a], "price_at_last_start": p.loc[b],
                     "share_of_starts_more_than_30pct_below_365d_high": 100 * float((dd.reindex(v5.loc[m, "start"]).to_numpy() < -0.30).mean()),
                     "uniform_mean_percentile": uni.loc[m, "uniform_percentile"].mean(),
                     "final_model_win_rate": 100 * float((v5.loc[m, "dynamic_percentile"] > v5.loc[m, "uniform_percentile"]).mean()),
                     "final_model_mean_percentile": v5.loc[m, "dynamic_percentile"].mean(),
                     "reference_win_rate": 100 * float((ref.loc[m, "dynamic_percentile"] > ref.loc[m, "uniform_percentile"]).mean()),
                     "reference_mean_percentile": ref.loc[m, "dynamic_percentile"].mean()})
    t = pd.DataFrame(rows)
    save_csv("split_profile.csv", t, "market context (prices, drawdown share) and the final model's and the reference's record on the development, embargo and hold-out windows")
    return t


def recency() -> dict:
    out = {"rho": 0.9}
    for N in (60, 420, 2007, 2557, 2793, 3075):
        w = 0.9 ** np.arange(N - 1, -1, -1)
        w = w / w.sum()
        out[f"N_{N}"] = {"weight_most_recent": float(w[-1]), "share_last_60": float(w[-60:].sum()), "share_last_30": float(w[-30:].sum())}
    out["one_minus_0.9_pow_60"] = float(1 - 0.9 ** 60)
    save_json("recency_weights.json", out, "how much of the recency-weighted percentile the most recent windows carry, for each period size")
    return out


# ---------------------------------------------------------------- 7. the sequestered year in accumulation terms
def sequestered(df_full: pd.DataFrame) -> dict:
    t = read_spd("final_reading_v5")
    budget = 1_000_000.0
    rows = []
    for label, r in t.iterrows():
        s, e = (pd.Timestamp(x) for x in label.split(ARROW))
        days = len(df_full["PriceUSD_coinmetrics"].loc[s:e])
        rows.append({"window": label, "days": days, "daily_dollars": budget / days,
                     "uniform_spd": r["uniform_sats_per_dollar"], "model_spd": r["dynamic_sats_per_dollar"],
                     "uniform_btc": r["uniform_sats_per_dollar"] * budget / 1e8, "model_btc": r["dynamic_sats_per_dollar"] * budget / 1e8,
                     "extra_btc": (r["dynamic_sats_per_dollar"] - r["uniform_sats_per_dollar"]) * budget / 1e8,
                     "edge_pct": (r["dynamic_sats_per_dollar"] / r["uniform_sats_per_dollar"] - 1) * 100,
                     "uniform_percentile": r["uniform_percentile"], "model_percentile": r["dynamic_percentile"]})
    acc = pd.DataFrame(rows)
    first = acc.iloc[0]
    s, e = (pd.Timestamp(x) for x in first["window"].split(ARROW))
    price = df_full["PriceUSD_coinmetrics"].loc[s:e]
    out = {"budget_dollars": budget, "first_window": first.to_dict(),
           "first_window_prices": {"start_date": s.strftime("%Y-%m-%d"), "start_price": price.iloc[0], "max_price": price.max(), "max_price_date": price.idxmax().strftime("%Y-%m-%d"),
                                   "min_price": price.min(), "min_price_date": price.idxmin().strftime("%Y-%m-%d"), "end_price": price.iloc[-1],
                                   "uniform_effective_price_per_btc": budget / first["uniform_btc"], "model_effective_price_per_btc": budget / first["model_btc"]},
           "all_windows": {"count": int(len(acc)), "edge_pct_min": acc["edge_pct"].min(), "edge_pct_max": acc["edge_pct"].max(), "all_won": bool((acc["edge_pct"] > 0).all()),
                           "overlap_note": "the 14 windows start on consecutive days and share 352 to 365 of their 366 days; they are one episode, not fourteen"},
           "reading": json.loads((ROOT / "output" / "final_reading.json").read_text(encoding="utf8"))["candidates"]["Candidate v5 (v4 signal, ceiling 12, round 6)"]}
    save_csv("sequestered_accumulation.csv", acc, "the 14 sequestered windows in accumulation terms on a $1,000,000 budget, from the committed one-time reading table")
    save_json("sequestered_accumulation.json", out, "the first sequestered window in dollars, BTC and prices, from the committed reading and the frozen series")
    return out


# ---------------------------------------------------------------- 8. the final model's multiplier path (learner-dependent)
def v5_multipliers(df_close: pd.DataFrame, data_close: str) -> dict:
    import sklearn
    cfg5 = MLConfig(model="hgbr", horizon=90, a_ml=2.0, features=("hash_mom", "fee_mom", "cycle_pos"), shape="pace", m_max=12.0, b_quad=0.0)
    cfg4 = MLConfig(model="hgbr", horizon=90, a_ml=2.0, features=("hash_mom", "fee_mom", "cycle_pos"))
    feats = build_ml_features(df_close, cfg5)
    z = feats["f_ml_z"]
    m5 = z.apply(lambda v: _multiplier(v if np.isfinite(v) else 0.0, cfg5))
    m4 = z.apply(lambda v: _multiplier(v if np.isfinite(v) else 0.0, cfg4))
    series = pd.DataFrame({"date": z.index.strftime("%Y-%m-%d"), "z": z.values, "multiplier_v5": m5.values, "multiplier_v4": m4.values}).loc[lambda t: t["date"] >= "2018-01-01"]
    save_csv("v5_signal_daily.csv", series, "the final model's standardised forecast and multiplier for every day from 2018 (scikit-learn " + sklearn.__version__ + ")")

    last_start = pd.Timestamp(data_close) - pd.DateOffset(years=1)
    rows = []
    for s in pd.date_range("2024-07-01", last_start, freq="D"):
        e = s + pd.DateOffset(years=1)
        zz = z.loc[s:e].to_numpy(float)
        w5, w4 = _allocate_pace(zz, cfg5), _allocate_pace(zz, cfg4)
        rows.append({"window_start": s.strftime("%Y-%m-%d"), "max_multiplier": float(m5.loc[s:e].max()), "days_above_5": int((m5.loc[s:e] > 5).sum()),
                     "max_abs_weight_difference_v5_v4": float(np.abs(w5 - w4).max())})
    hold = pd.DataFrame(rows)
    period = z.loc["2018-01-01":data_close]
    dist = {"sklearn_version": sklearn.__version__, "days": int(period.notna().sum()),
            "share_at_ceiling": float((m5.loc["2018-01-01":data_close] >= 12.0 - 1e-12).mean()), "share_at_floor": float((m5.loc["2018-01-01":data_close] <= 0.25 + 1e-12).mean()),
            "share_above_5": float((m5.loc["2018-01-01":data_close] > 5).mean()), "share_neutral_band_0.8_1.25": float(((m5.loc["2018-01-01":data_close] > 0.8) & (m5.loc["2018-01-01":data_close] < 1.25)).mean()),
            "z_quantiles": {q: float(period.quantile(q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
            "holdout_windows": int(len(hold)), "holdout_max_multiplier": float(hold["max_multiplier"].max()), "holdout_windows_with_any_day_above_5": int((hold["days_above_5"] > 0).sum()),
            "holdout_max_abs_weight_difference_v5_v4": float(hold["max_abs_weight_difference_v5_v4"].max())}
    save_csv("v5_holdout_multipliers.csv", hold, "for each hold-out window: the largest multiplier the final model reached and whether its weights differ from candidate v4's")
    save_json("v5_multiplier_summary.json", dist, "distribution of the final model's multiplier over period B and the test of why rounds 5 and 6 share one hold-out score")
    return dist


def main() -> None:
    summary = json.loads((ROOT / "output" / "results_summary.json").read_text(encoding="utf8"))
    data_close = summary["data_last_day"]
    df_full = load_btc()
    df_close = df_full.loc[:data_close]
    print("data close", data_close, "| frame to", df_full.index.max().date())
    sp = splits(df_close, data_close)
    print("splits:", {k: sp[k]["windows"] for k in ("development", "embargo", "hold_out")}, "sequestered", sp["sequestered"]["windows_read"])
    decomposition()
    benchmark_tables(df_close)
    ws = worked_scoring(df_close)
    print("worked scoring 2025: days", ws["days"], "uniform pct %.4f" % ws["uniform_percentile"])
    wa = worked_allocator()
    print("allocator example sum", wa["five_day_sum"])
    rr = round_record(df_close)
    print(rr[["round", "dev_selection", "baseline_value", "margin", "holdout_score", "holdout_win_rate", "holdout_unweighted"]].round(2).to_string(index=False))
    round6_summary()
    halving_cycles(df_close)
    drawdowns(df_close)
    volatility(df_close)
    sf = spearman(df_close)
    print("spearman sample:", sf["sample"].iloc[0], "n", sf["n_days"].iloc[0])
    mvrv_bands(df_close)
    split_profile(df_close)
    recency()
    sequestered(df_full)
    dist = v5_multipliers(df_close, data_close)
    print("v5 multipliers: hold-out max %.3f, windows with any day above 5: %d, max |w5-w4| %.2e" % (
        dist["holdout_max_multiplier"], dist["holdout_windows_with_any_day_above_5"], dist["holdout_max_abs_weight_difference_v5_v4"]))
    lines = ["# Manuscript support files", "", "Produced by tools/manuscript_support.py from the committed results and the frozen data; "
             f"data close {data_close}. Nothing here changes a model or a score.", "", "| File | Content |", "|---|---|"]
    lines += [f"| {n} | {note} |" for n, note in INDEX]
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("wrote", len(INDEX), "files to output/manuscript/")


if __name__ == "__main__":
    main()
