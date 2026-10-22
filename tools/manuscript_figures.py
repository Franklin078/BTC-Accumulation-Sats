"""Print-friendly figures for the manuscript.

The repository's working charts use a dark theme. The manuscript is read on paper and beside
black text, so every figure it carries is drawn here on a white background from the same
committed data, with the quantity named in words in each title and axis label. Files go to
output/manuscript/ as PNG (200 dpi) and PDF (vector).

Run from the repository root after tools/manuscript_support.py:

    py -3.14 tools/manuscript_figures.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from model.regimes import load_btc  # noqa: E402
from model.ml import feature_matrix, HALVINGS  # noqa: E402
from model.strategy import construct_features, Params  # noqa: E402

OUT = ROOT / "output" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)
ARROW = " → "

# Okabe and Ito palette: distinguishable in print and for colour-blind readers.
C = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "pink": "#CC79A7", "yellow": "#E69F00",
     "sky": "#56B4E9", "black": "#000000", "grey": "#7F7F7F"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.edgecolor": "#444444",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
WIDTH = 6.3   # inches: the text width of an A4 page with 1-inch margins


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def mark_halvings(ax, label_first: bool = True) -> None:
    lo, hi = ax.get_xlim()
    first = True
    for h in HALVINGS:
        x = mdates.date2num(h)
        if lo <= x <= hi:
            ax.axvline(h, color=C["grey"], linestyle="--", linewidth=0.8, label="halving" if (first and label_first) else None)
            first = False
    ax.set_xlim(lo, hi)


def starts_of(table: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime([s.split(ARROW)[0] for s in table.index])


def spd(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "output" / "spd_tables" / f"{name}.csv", index_col=0)


# ---------------------------------------------------------------- Chapter 3
def fig_price_log(p: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    ax.plot(p.index, p, color=C["blue"], lw=0.9, label="closing price")
    ax.set_yscale("log")
    ax.set_title("Bitcoin closing price in US dollars, logarithmic scale")
    ax.set_ylabel("US dollars per bitcoin (log scale)")
    mark_halvings(ax)
    ax.legend(loc="lower right")
    save(fig, "fig_price_log")


def fig_drawdown(p: pd.Series) -> None:
    dd = (p / p.rolling(365, min_periods=30).max() - 1) * 100
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    ax.fill_between(dd.index, dd, 0, color=C["orange"], alpha=0.35, linewidth=0)
    ax.plot(dd.index, dd, color=C["orange"], lw=0.7)
    ax.set_title("Drawdown from the trailing 365-day high, per cent")
    ax.set_ylabel("per cent below the 365-day high")
    mark_halvings(ax)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    save(fig, "fig_drawdown")


def fig_returns_vol(p: pd.Series) -> None:
    r = np.log(p.loc["2018-01-01":]).diff().dropna() * 100
    vol = r.rolling(30).std() * np.sqrt(365)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].hist(r, bins=120, color=C["blue"])
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Daily log returns since 2018", fontsize=9.5)
    axes[0].set_xlabel("daily log return, per cent (log count)")
    axes[1].plot(vol.index, vol, color=C["green"], lw=0.8)
    axes[1].set_title("(b) 30-day realised volatility, annualised, per cent", fontsize=9.5)
    axes[1].set_xlabel("date")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    save(fig, "fig_returns_vol")


def fig_uniform_percentile() -> None:
    u = pd.read_csv(OUT / "spd_uniform_B.csv", index_col=0)
    st = starts_of(u)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    ax.plot(st, u["uniform_percentile"], color=C["pink"], lw=0.8, label="uniform DCA's window percentile")
    ax.axhline(50, color=C["grey"], ls=":", lw=0.8, label="midpoint, 50")
    ax.axhline(u["uniform_percentile"].mean(), color=C["black"], ls="-", lw=0.8, label=f"mean, {u['uniform_percentile'].mean():.1f}")
    ax.set_title("Uniform DCA's window percentile by window start, evaluation period B")
    ax.set_xlabel("window start date")
    ax.set_ylabel("percentile (0 worst, 100 best)")
    ax.set_ylim(0, 100)
    mark_halvings(ax)
    ax.legend(loc="upper right", ncol=2)
    save(fig, "fig_uniform_percentile")


def fig_mvrv(df: pd.DataFrame) -> None:
    mv = df["CapMVRVCur"].loc["2013-01-01":]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    ax.plot(mv.index, mv, color=C["yellow"], lw=0.8, label="MVRV ratio")
    ax.axhline(1, color=C["grey"], ls=":", lw=0.8, label="MVRV = 1 (market at its aggregate cost basis)")
    ax.axhline(3, color=C["grey"], ls="--", lw=0.8, label="MVRV = 3")
    ax.set_title("Market value to realised value (MVRV) ratio")
    ax.set_ylabel("ratio")
    mark_halvings(ax, label_first=False)
    ax.legend(loc="upper right")
    save(fig, "fig_mvrv")


# ---------------------------------------------------------------- Chapter 4
def fig_feature_hist(df: pd.DataFrame) -> None:
    X = feature_matrix(df, "v1").loc["2018-01-01":"2023-06-30"]
    base = construct_features(df, Params()).loc["2018-01-01":"2023-06-30"]
    panels = [
        ("hash_mom", X["hash_mom"], "(a) Hash-rate momentum\n30-day / 365-day mean, minus 1"),
        ("fee_mom", X["fee_mom"], "(b) Fee momentum\n30-day / 365-day mean, minus 1"),
        ("cycle_pos", X["cycle_pos"], "(c) Cycle position\ncos(2 pi d / 1461)"),
        ("ma_gap", X["ma_gap"], "(d) 200-day moving-average gap\nprice / 200-day mean, minus 1"),
        ("drawdown", X["drawdown"], "(e) Drawdown\nprice / 365-day high, minus 1"),
        ("mvrv_z", base["f_mvrv_z"].clip(-3, 3), "(f) MVRV z-score in [-3, 3]\n365-day rolling standardisation"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 5.0), gridspec_kw={"wspace": 0.45, "hspace": 0.55})
    cols = [C["blue"], C["blue"], C["blue"], C["orange"], C["orange"], C["orange"]]
    n_days = None
    for ax, (key, s, title), col in zip(axes.ravel(), panels, cols):
        s = s.dropna()
        n_days = len(s) if n_days is None else n_days
        ax.hist(s, bins=60, color=col)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("feature value, lagged one day", fontsize=8)
        ax.set_ylabel("number of days", fontsize=8)
        ax.tick_params(labelsize=7.5)
    fig.suptitle(f"Candidate features over the development period\n(window starts 1 January 2018 to 30 June 2023, {n_days:,} days)", fontsize=9.5)
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.09, right=0.98, wspace=0.5, hspace=0.6)
    fig.savefig(OUT / "fig_feature_hist.png", dpi=200); fig.savefig(OUT / "fig_feature_hist.pdf"); plt.close(fig); print("wrote fig_feature_hist")


def fig_weight_paths() -> None:
    w = pd.read_csv(OUT / "weight_paths.csv")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    ax.plot(w["day"], w["neutral"] * 100, color=C["black"], lw=1.0, label="multiplier 1 every day (uniform buying)")
    ax.plot(w["day"], w["ceiling_30_days"] * 100, color=C["orange"], lw=1.0, label="multiplier 12 for 30 days, then 1")
    ax.plot(w["day"], w["floor_30_days"] * 100, color=C["blue"], lw=1.0, label="multiplier 0.25 for 30 days, then 1")
    ax.set_title("Daily weight along a 365-day window for three multiplier paths")
    ax.set_xlabel("day of the window")
    ax.set_ylabel("share of budget, per cent (log)")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.3, 1, 3])
    ax.set_yticklabels(["0.1", "0.3", "1", "3"])
    ax.legend(loc="upper right")
    save(fig, "fig_weight_paths")


def fig_multiplier_response() -> None:
    z = np.linspace(-3, 3, 601)
    m = np.clip(np.exp(2 * z), 0.25, 12)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    ax.plot(z, np.exp(2 * z), color=C["grey"], ls=":", lw=0.9, label="exp(2 z), unbounded")
    ax.plot(z, m, color=C["blue"], lw=1.4, label="multiplier M = min(max(exp(2 z), 0.25), 12)")
    for zz in (-3, 0, 1, 3):
        mm = float(np.clip(np.exp(2 * zz), 0.25, 12))
        ax.plot([zz], [mm], "o", color=C["orange"], ms=5)
        ax.annotate(f"z = {zz}: M = {mm:.2f}", (zz, mm), textcoords="offset points", xytext=(6, 8) if zz < 2 else (-60, -16), fontsize=8)
    ax.axvline(np.log(12) / 2, color=C["grey"], lw=0.6, ls="--")
    ax.axvline(np.log(0.25) / 2, color=C["grey"], lw=0.6, ls="--")
    ax.set_ylim(0, 14)
    ax.set_title("The final model's multiplier as a function of the standardised forecast")
    ax.set_xlabel("standardised forecast z (clipped to the range -3 to 3)")
    ax.set_ylabel("multiplier M")
    ax.legend(loc="upper left")
    save(fig, "fig_multiplier_response")


def fig_walk_forward() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    ax.grid(False)
    ax.set_yticks([])
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    r = 1000
    tf = ax.get_xaxis_transform()
    # first refit
    ax.axvspan(0, r - 90, 0.62, 0.90, color=C["blue"], alpha=0.35)
    ax.text((r - 90) / 2, 0.76, "training set at refit r: every day s with s + 90 < r\n(its 90-day target has already closed)", ha="center", va="center", fontsize=8, transform=tf)
    ax.axvspan(r - 90, r, 0.62, 0.90, color=C["grey"], alpha=0.35)
    ax.text(r - 45, 0.92, "purge\n90 days", ha="center", va="bottom", fontsize=7.5, transform=tf)
    ax.axvspan(r, r + 90, 0.62, 0.90, color=C["orange"], alpha=0.45)
    ax.text(r + 45, 0.60, "forecasts for\ndays r to r + 89", ha="center", va="top", fontsize=7.5, transform=tf)
    ax.axvline(r, ymin=0.58, ymax=0.94, color=C["black"], lw=1.2)
    # second refit
    ax.axvspan(0, r, 0.12, 0.40, color=C["blue"], alpha=0.25)
    ax.axvspan(r, r + 90, 0.12, 0.40, color=C["grey"], alpha=0.35)
    ax.axvspan(r + 90, r + 180, 0.12, 0.40, color=C["orange"], alpha=0.35)
    ax.axvline(r + 90, ymin=0.08, ymax=0.44, color=C["black"], lw=1.2)
    ax.text(r / 2, 0.26, "next refit, r + 90: the training set has grown by 90 closed targets", ha="center", va="center", fontsize=8, transform=tf)
    ax.text(r + 135, 0.02, "forecasts for\nr + 90 to r + 179", ha="center", va="bottom", fontsize=7.5, transform=tf)
    ax.set_xlim(0, r + 190)
    ax.set_ylim(0, 1)
    ax.set_xlabel("calendar days from the start of the series (schematic, not to scale)")
    ax.set_title("Purged walk-forward training: what each refit may see", pad=14)
    save(fig, "fig_walk_forward")


def fig_splits_timeline() -> None:
    sp = json.loads((OUT / "splits.json").read_text(encoding="utf8"))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    ax.grid(False)
    rows = [("development set", sp["development"], C["blue"]), ("embargo", sp["embargo"], C["grey"]), ("hold-out set", sp["hold_out"], C["orange"]),
            ("sequestered set", {"first_start": "2025-09-01", "last_start": "2025-09-14", "windows": sp["sequestered"]["windows_read"]}, C["green"])]
    for i, (name, d, col) in enumerate(rows):
        a, b = pd.Timestamp(d["first_start"]), pd.Timestamp(d["last_start"])
        ax.barh(i, (b - a).days + 1, left=a, height=0.6, color=col, alpha=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{name}\n{d['windows']} window starts" for name, d, _ in rows], fontsize=8.5)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()
    ax.axvline(pd.Timestamp(sp["data_close"]), color=C["black"], lw=1, ls="--")
    ax.text(pd.Timestamp(sp["data_close"]) - pd.Timedelta(days=15), 3.45, f"data close\n{pd.Timestamp(sp['data_close']):%d %b %Y}", ha="right", va="bottom", fontsize=8)
    ax.set_xlim(pd.Timestamp("2017-10-01"), pd.Timestamp("2026-10-01"))
    ax.set_title("Window start dates of the four data splits")
    ax.set_xlabel("window start date")
    save(fig, "fig_splits_timeline")


def fig_scoring_example() -> None:
    d = pd.read_csv(OUT / "worked_scoring_2025_daily.csv", parse_dates=["date"])
    j = json.loads((OUT / "worked_scoring_2025.json").read_text(encoding="utf8"))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    ax.plot(d["date"], d["sats_per_dollar"], color=C["blue"], lw=1.0, label="sats per dollar each day (1e8 / price)")
    ax.axhline(j["uniform_spd"], color=C["black"], lw=1.0, label=f"uniform DCA: mean of the daily values, {j['uniform_spd']:.2f}")
    ax.axhline(j["max_spd"], color=C["green"], lw=0.9, ls="--", label=f"best single day, {j['max_spd']:.2f} ({pd.Timestamp(j['max_spd_date']):%d %b %Y})")
    ax.axhline(j["min_spd"], color=C["orange"], lw=0.9, ls="--", label=f"worst single day, {j['min_spd']:.2f} ({pd.Timestamp(j['min_spd_date']):%d %b %Y})")
    ax.set_title(f"Scoring the window 1 January 2025 to 1 January 2026 ({j['days']} days):\nuniform DCA's percentile is {j['uniform_percentile']:.2f}")
    ax.set_ylabel("satoshis per US dollar")
    ax.set_xlabel("date")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "fig_scoring_example")


# ---------------------------------------------------------------- Chapter 5 and 6
def fig_progression() -> None:
    rr = pd.read_csv(OUT / "round_record.csv", dtype={"round": str})
    cand = rr[rr["candidate"].notna()].copy()
    x = np.arange(len(cand))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    ax.axvspan(4.5, 6.5, color=C["orange"], alpha=0.10, lw=0)
    ax.text(6.0, 63, "rounds 11 and 12:\ndevelopment metric\nrising, hold-out\nscore falling", ha="center", va="center", fontsize=8.5, color=C["orange"])
    ax.plot(x, cand["dev_selection"], "-o", color=C["blue"], lw=1.4, label="development selection metric (used to select)")
    ax.plot(x, cand["holdout_score"], "-s", color=C["orange"], lw=1.4, label="hold-out score (recorded, never used to select)")
    for xi, (_, r) in zip(x, cand.iterrows()):
        ax.annotate(f"{r['dev_selection']:.2f}", (xi, r["dev_selection"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.annotate(f"{r['holdout_score']:.2f}", (xi, r["holdout_score"]), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8)
    v5 = cand.index[cand["candidate"] == "v5"][0]
    i5 = list(cand.index).index(v5)
    ax.plot([i5], [cand.loc[v5, "holdout_score"]], "o", ms=16, mfc="none", mec=C["black"], mew=1.5)
    ax.annotate("final model", (i5, cand.loc[v5, "holdout_score"]), textcoords="offset points", xytext=(-8, -26), ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nround {r}" for c, r in zip(cand["candidate"], cand["round"])], fontsize=8.5)
    ax.set_ylim(25, 85)
    ax.set_ylabel("score, 0 to 100")
    ax.set_title("The seven candidates in order of creation:\ndevelopment selection metric against hold-out score")
    ax.legend(loc="lower left")
    save(fig, "fig_progression")


def fig_dev_gradient() -> None:
    rr = pd.read_csv(OUT / "round_record.csv", dtype={"round": str})
    fin = rr[rr["round"].isin(["6", "11", "12"])].copy()
    fin["candidate"] = ["v5", "v6", "v7"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    for ax, col, title in ((axes[0], "holdout_unweighted", "(a) Like for like: hold-out value of the\nsame unweighted metric used in development"),
                           (axes[1], "holdout_score", "(b) Hold-out Final Model Score\n(recency-weighted percentile half)")):
        ax.plot(fin["dev_selection"], fin[col], "-o", color=C["orange"], lw=1.3)
        for _, r in fin.iterrows():
            ax.annotate(f"{r['candidate']}\nwin rate {r['holdout_win_rate']:.2f}", (r["dev_selection"], r[col]), textcoords="offset points", xytext=(6, -4), fontsize=8)
        ax.set_xlabel("development selection metric")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(72.8, 75.6)
    axes[0].set_ylabel("hold-out value")
    fig.suptitle("Across the three finalists the development metric rose\nwhile every hold-out measure fell", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "fig_dev_gradient.png", dpi=200); fig.savefig(OUT / "fig_dev_gradient.pdf"); plt.close(fig); print("wrote fig_dev_gradient")


def fig_percentile_by_window_B() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    tables = (("final model (candidate v5)", spd("robust_v5_B"), "dynamic_percentile", C["blue"]),
              ("2025 reference implementation", pd.read_csv(OUT / "spd_reference_B.csv", index_col=0), "dynamic_percentile", C["orange"]),
              ("uniform DCA", pd.read_csv(OUT / "spd_uniform_B.csv", index_col=0), "uniform_percentile", C["grey"]))
    for lab, t, col_name, col in tables:
        ax.plot(starts_of(t), t[col_name].rolling(7).mean(), lw=0.9, label=lab, color=col)
    ax.set_title("Window percentile by window start, period B (7-day moving average)")
    ax.set_xlabel("window start date")
    ax.set_ylabel("percentile (0 worst, 100 best)")
    ax.set_ylim(0, 100)
    mark_halvings(ax, label_first=False)
    ax.axvspan(pd.Timestamp("2024-07-01"), pd.Timestamp("2025-08-24"), color=C["green"], alpha=0.08, lw=0)
    ax.text(pd.Timestamp("2025-01-15"), 4, "hold-out starts", ha="center", fontsize=8, color=C["green"])
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    save(fig, "fig_percentile_by_window_B")


def fig_winrate_vs_rw() -> None:
    d = pd.read_csv(OUT / "decomposition.csv")
    keep = {"Candidate v5 (v4 signal, ceiling 12, round 6)": ("final model v5", C["blue"], "o"),
            "Tournament 2025 reference": ("2025 reference", C["orange"], "s"),
            "Upstream 2026 baseline (200-MA)": ("2026 template baseline", C["green"], "^"),
            "Candidate v1 (drawdown-led, round 1)": ("candidate v1 (rules)", C["pink"], "D")}
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    for model, (lab, col, mk) in keep.items():
        g = d[d["model"] == model]
        ax.scatter(g["rw_pct"], g["win_rate"], s=45, color=col, marker=mk, label=lab, zorder=3)
        for _, r in g.iterrows():
            dx = -11 if (model.startswith("Candidate v5") and r["period"] == "A") else 5
            ax.annotate(r["period"], (r["rw_pct"], r["win_rate"]), textcoords="offset points", xytext=(dx, 3), fontsize=8)
    for s in (50, 60, 70):
        xx = np.linspace(20, 80, 50)
        ax.plot(xx, 2 * s - xx, color=C["grey"], lw=0.6, ls=":")
        ax.annotate(f"score {s}", (78, 2 * s - 78), fontsize=7.5, color=C["grey"])
    ax.axhline(50, color=C["black"], lw=0.7)
    ax.set_xlabel("recency-weighted percentile")
    ax.set_ylabel("win rate against uniform DCA, per cent")
    ax.set_title("The two halves of the Final Model Score, periods A, B and C")
    ax.set_xlim(20, 80)
    ax.set_ylim(45, 95)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "fig_winrate_vs_rw")


def fig_v5_multiplier_path() -> None:
    s = pd.read_csv(OUT / "v5_signal_daily.csv", parse_dates=["date"])
    sp = json.loads((OUT / "splits.json").read_text(encoding="utf8"))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    ax.plot(s["date"], s["multiplier_v5"], color=C["blue"], lw=0.7, label="final model's daily multiplier")
    ax.axhline(1, color=C["black"], lw=0.7)
    ax.axhline(12, color=C["grey"], lw=0.7, ls="--", label="ceiling 12 and floor 0.25")
    ax.axhline(0.25, color=C["grey"], lw=0.7, ls="--")
    ax.axvspan(pd.Timestamp(sp["hold_out"]["first_start"]), pd.Timestamp(sp["data_close"]), color=C["orange"], alpha=0.10, lw=0, label="hold-out windows")
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1, 2, 5, 12])
    ax.set_yticklabels(["0.25", "0.5", "1", "2", "5", "12"])
    ax.set_title("The final model's daily multiplier, 2018 to the data close")
    ax.set_ylabel("multiplier (log scale)")
    ax.set_xlabel("date")
    mark_halvings(ax, label_first=False)
    ax.set_xlim(pd.Timestamp("2018-01-01"), pd.Timestamp(sp["data_close"]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8)
    save(fig, "fig_v5_multiplier_path")


def main() -> None:
    summary = json.loads((ROOT / "output" / "results_summary.json").read_text(encoding="utf8"))
    df = load_btc().loc[:summary["data_last_day"]]
    p = df["PriceUSD_coinmetrics"]
    fig_price_log(p)
    fig_drawdown(p)
    fig_returns_vol(p)
    fig_uniform_percentile()
    fig_mvrv(df)
    fig_feature_hist(df)
    fig_weight_paths()
    fig_multiplier_response()
    fig_walk_forward()
    fig_splits_timeline()
    fig_scoring_example()
    fig_progression()
    fig_dev_gradient()
    fig_percentile_by_window_B()
    fig_winrate_vs_rw()
    fig_v5_multiplier_path()


if __name__ == "__main__":
    main()
