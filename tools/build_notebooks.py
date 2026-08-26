"""Generate the project notebooks from source so that code lives in model/ and the notebooks stay short.

Run from the repository root:  python tools/build_notebooks.py
Produces notebooks/01..06, deliverables/btc_accumulation_model.ipynb and
tournament_2025/btc_accumulation_model.ipynb (the official template with only the model cell replaced).
"""
import json
import os
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PARAMS = json.load(open("model/final_params.json"))
RES = json.load(open("output/results_summary.json")) if os.path.exists("output/results_summary.json") else {}

VERSIONS = '''import sys, platform, numpy, pandas, matplotlib
print("Python", sys.version.split()[0], "| numpy", numpy.__version__, "| pandas", pandas.__version__, "| matplotlib", matplotlib.__version__, "|", platform.platform())'''

PATHFIX = '''# Works on a laptop and in Google Colab. On Colab it clones the private repository the first
# time, using the GH_TOKEN and GH_REPO values stored in Colab Secrets (the key icon on the left).
import os, sys, subprocess
def _find_root():
    d = os.getcwd()
    for c in (d, os.path.dirname(d), "/content/stacking-sats-uts"):
        if c and os.path.exists(os.path.join(c, "model", "strategy.py")):
            return c
    return None
ROOT = _find_root()
IN_COLAB = os.path.exists("/content")
if ROOT is None and IN_COLAB:
    from google.colab import userdata
    token = userdata.get("GH_TOKEN"); repo = userdata.get("GH_REPO")
    subprocess.run(["git", "clone", f"https://x-access-token:{token}@github.com/{repo}.git", "/content/stacking-sats-uts"], check=True, capture_output=True)
    ROOT = "/content/stacking-sats-uts"
assert ROOT, "repository not found: run this from the repository, or set GH_TOKEN and GH_REPO in Colab Secrets"
os.chdir(ROOT); sys.path.insert(0, ROOT)
print("repository root:", ROOT, "| Colab:", IN_COLAB)'''

STYLE = '''import matplotlib.pyplot as plt
plt.rcParams.update({"figure.facecolor": "#0D0D0D", "axes.facecolor": "#1A1A2E", "savefig.facecolor": "#0D0D0D",
    "axes.edgecolor": "#888", "axes.labelcolor": "#EEE", "xtick.color": "#DDD", "ytick.color": "#DDD", "text.color": "#EEE",
    "axes.titlesize": 15, "axes.labelsize": 12, "legend.fontsize": 11, "font.size": 12, "grid.color": "#333", "axes.grid": True})
PALETTE = ["#FFB000", "#00D4FF", "#FF5C8A", "#7CFC00", "#C084FC", "#FF8C42"]
HALVINGS = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]
def annotate_halvings(ax):
    import pandas as pd, matplotlib.dates as mdates
    lo, hi = ax.get_xlim()
    for h in HALVINGS:
        x = mdates.date2num(pd.Timestamp(h))
        if lo <= x <= hi:
            ax.axvline(pd.Timestamp(h), color="#888", linestyle="--", linewidth=1)
            ax.text(pd.Timestamp(h), ax.get_ylim()[1], " halving", rotation=90, va="top", color="#AAA", fontsize=9)
    ax.set_xlim(lo, hi)
os.makedirs("output/eda", exist_ok=True)'''


def nb(cells, title):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}")] + [
        nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


def write(n, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nbf.write(n, path)
    print("wrote", path)


P = PARAMS
p_repr = ", ".join(f"{k}={v!r}" for k, v in P.items())

# ------------------------------------------------------------------ 01 data
write(nb([
    ("md", "## What this notebook does\nIt downloads the daily Bitcoin series from the CoinMetrics community API, keeps only complete days, checks the series for gaps, and freezes it to `data/Coin Metrics/coinmetrics_btc.csv`, the file the Trilemma loader reads. Run it at the start of any scoring session. If the API is unreachable the frozen file is used and the date of the last refresh is printed."),
    ("code", PATHFIX), ("code", VERSIONS),
    ("md", "### Fetch\nThe community endpoint is free and needs no key. Metrics are requested in two batches because of URL length limits. Status columns are dropped. Today's row is partial and is removed."),
    ("code", '''import requests, time, pandas as pd
BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ["PriceUSD","CapMVRVCur","CapMrktCurUSD","AdrActCnt","AdrBalCnt","TxCnt","TxTfrCnt","HashRate","BlkCnt","FeeTotNtv","IssTotNtv",
           "IssTotUSD","SplyCur","FlowInExNtv","FlowOutExNtv","FlowInExUSD","FlowOutExUSD","ROI30d","ROI1yr","volume_reported_spot_usd_1d","SplyExNtv"]
OUT = "data/Coin Metrics/coinmetrics_btc.csv"
def fetch():
    frames = []
    for chunk in (METRICS[:11], METRICS[11:]):
        params = {"assets": "btc", "metrics": ",".join(chunk), "frequency": "1d", "start_time": "2010-07-18", "page_size": 10000}
        rows, url = [], BASE
        while True:
            r = requests.get(url, params=params if url == BASE else None, timeout=120)
            if r.status_code == 429: time.sleep(6); continue
            r.raise_for_status(); j = r.json(); rows += j["data"]; url = j.get("next_page_url")
            if not url: break
        d = pd.DataFrame(rows); d["time"] = pd.to_datetime(d["time"]).dt.tz_localize(None).dt.normalize()
        frames.append(d.set_index("time").drop(columns=["asset"]))
    df = pd.concat(frames, axis=1); df = df.loc[:, ~df.columns.duplicated()]
    df = df[[c for c in df.columns if "-status" not in c]]
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()
try:
    df = fetch(); df = df.loc[:df["PriceUSD"].last_valid_index()]
    df.reset_index().to_csv(OUT, index=False); print("refreshed from API; last complete day:", df.index.max().date())
except Exception as e:
    print("API unavailable (", e, "); using the frozen file")
    df = pd.read_csv(OUT, parse_dates=["time"]).set_index("time")
print(df.shape)'''),
    ("md", "### Integrity gate\nNo missing calendar days, no null or non-positive prices, no duplicate dates. The notebook stops here if any check fails."),
    ("code", '''full = pd.date_range(df.index.min(), df.index.max(), freq="D")
checks = {"missing_days": int(len(full.difference(df.index))), "null_price": int(df.PriceUSD.isna().sum()),
          "nonpositive_price": int((df.PriceUSD <= 0).sum()), "duplicate_dates": int(df.index.duplicated().sum())}
print(checks); assert not any(checks.values()), "integrity gate failed"
import hashlib; print("sha256 of frozen file:", hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16], "| rows", len(df), "| first", df.index.min().date(), "| last", df.index.max().date())'''),
    ("md", "### Cross-check against the frozen 2025 tournament dataset\nThe tournament parquet is the same CoinMetrics source. The two series must agree to machine precision on every overlapping day; if they do not, something changed upstream and the decision log needs an entry."),
    ("code", '''pq = "data/official_2025/stacking_sats_data.parquet"
if os.path.exists(pq):
    off = pd.read_parquet(pq); off.index = pd.to_datetime(off.index).tz_localize(None).normalize()
    j = off[["PriceUSD_coinmetrics"]].join(df[["PriceUSD"]], how="inner")
    print("overlap days:", len(j), "| max abs relative difference:", float((j.PriceUSD / j.PriceUSD_coinmetrics - 1).abs().max()))
else:
    print("frozen parquet not present; skipped")'''),
], "01 Data: the official Bitcoin series, refreshed to today"), "notebooks/01_data.ipynb")

# ------------------------------------------------------------------ 02 eda
write(nb([
    ("md", "## What this notebook does\nIt looks at the series the model will be scored on and asks one question: what does a 12-month accumulation window look like, and what decides where uniform DCA lands inside it? Every chart is saved to `output/eda/` and reused in the educational notebook and the manuscript."),
    ("code", PATHFIX), ("code", VERSIONS), ("code", STYLE),
    ("code", '''import numpy as np, pandas as pd
from model.regimes import load_btc
df = load_btc(); p = df["PriceUSD_coinmetrics"]
print(df.index.min().date(), "to", df.index.max().date(), "|", len(df), "days")'''),
    ("md", "### Price on a log axis with halvings\nThe supply schedule halves the block reward roughly every four years. Each halving has so far sat near the start of a rise and a later drawdown of 50 to 85 per cent."),
    ("code", '''fig, ax = plt.subplots(figsize=(13, 5)); ax.plot(p.index, p, color=PALETTE[0], lw=1.2); ax.set_yscale("log")
ax.set_title("Bitcoin price, USD (CoinMetrics PriceUSD), log scale"); ax.set_xlabel("Date"); ax.set_ylabel("USD per BTC (log)"); annotate_halvings(ax)
plt.tight_layout(); plt.savefig("output/eda/01_price_log.png", dpi=200); plt.show()'''),
    ("md", "### Drawdown from the trailing one-year high\nThis is the model's main input. The depth and the duration of drawdowns is what a dynamic accumulation rule can respond to without knowing the future."),
    ("code", '''dd = p / p.rolling(365, min_periods=30).max() - 1
fig, ax = plt.subplots(figsize=(13, 4)); ax.fill_between(dd.index, dd * 100, 0, color=PALETTE[2], alpha=0.6); ax.plot(dd.index, dd * 100, color=PALETTE[2], lw=0.8)
ax.set_title("Drawdown from the trailing 365-day high (%)"); ax.set_ylabel("%"); ax.set_xlabel("Date"); annotate_halvings(ax)
plt.tight_layout(); plt.savefig("output/eda/02_drawdown.png", dpi=200); plt.show()
print("share of days deeper than 20% / 40% / 60%:", [(dd < -x).mean().round(3) for x in (0.2, 0.4, 0.6)])'''),
    ("md", "### Daily returns: fat tails and volatility clustering\nReturns are far from normal and large moves cluster. This matters because it means percentile outcomes inside a window are driven by a few stretches of days, not by a steady drift."),
    ("code", '''r = np.log(p).diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(r * 100, bins=150, color=PALETTE[1]); axes[0].set_yscale("log"); axes[0].set_title("Daily log returns (%), log count"); axes[0].set_xlabel("%")
vol = r.rolling(30).std() * np.sqrt(365) * 100; axes[1].plot(vol.index, vol, color=PALETTE[3], lw=0.9); axes[1].set_title("30-day realised volatility, annualised (%)"); axes[1].set_xlabel("Date")
plt.tight_layout(); plt.savefig("output/eda/03_returns_vol.png", dpi=200); plt.show()
print("mean %.3f%%  sd %.2f%%  skew %.2f  excess kurtosis %.1f" % (r.mean()*100, r.std()*100, r.skew(), r.kurt()))'''),
    ("md", "### Where uniform DCA lands inside a window\nFor every 12-month window starting from 2018 the chart shows the SPD percentile of buying the same amount every day. This is the benchmark the model has to beat, window by window. It is computed with the Trilemma engine, not re-implemented."),
    ("code", '''from template.prelude_template import compute_cycle_spd
from model.strategy import construct_features
feats = construct_features(df)
uniform = lambda w: pd.Series(1.0 / len(w), index=w.index)
spd = compute_cycle_spd(df, uniform, features_df=feats, start_date="2018-01-01", end_date=df.index.max().strftime("%Y-%m-%d"))
starts = pd.to_datetime([s.split(" → ")[0] for s in spd.index])
fig, ax = plt.subplots(figsize=(13, 4)); ax.plot(starts, spd["uniform_percentile"], color=PALETTE[4], lw=1)
ax.axhline(50, color="#888", ls=":"); ax.set_title("Uniform DCA: SPD percentile by window start (12-month windows)"); ax.set_xlabel("Window start"); ax.set_ylabel("percentile (%)"); annotate_halvings(ax)
plt.tight_layout(); plt.savefig("output/eda/04_uniform_percentile.png", dpi=200); plt.show()
print("mean %.2f  median %.2f  min %.2f  max %.2f" % (spd.uniform_percentile.mean(), spd.uniform_percentile.median(), spd.uniform_percentile.min(), spd.uniform_percentile.max()))'''),
    ("md", "### Does the window rise or fall?\nThe share of windows where the end price is above the start price explains why front-loaded rules win often: the asset drifted up in most years. It also explains where they lose."),
    ("code", '''ends = starts + pd.DateOffset(years=1)
ratio = pd.Series(p.reindex(ends).to_numpy() / p.reindex(starts).to_numpy(), index=starts)
print("share of windows ending higher than they started: %.1f%%" % (100 * (ratio > 1).mean()))
by_year = ratio.groupby(ratio.index.year).apply(lambda s: 100 * (s > 1).mean()).round(1); print(by_year.to_string())'''),
    ("md", "### On-chain valuation: MVRV\nMVRV (market value over realised value) is the on-chain valuation ratio that CoinMetrics publishes daily. Tops in 2017 and 2021 sat above 3; the 2025 top only reached about 2.4, which is the same level as early 2024. That limits its use as a top signal and is one reason the model treats it as a secondary term."),
    ("code", '''mv = df["CapMVRVCur"]
fig, ax = plt.subplots(figsize=(13, 4)); ax.plot(mv.index, mv, color=PALETTE[5], lw=1); ax.axhline(1, color="#888", ls=":"); ax.axhline(3, color="#888", ls=":")
ax.set_title("MVRV (CoinMetrics CapMVRVCur)"); ax.set_xlabel("Date"); annotate_halvings(ax)
plt.tight_layout(); plt.savefig("output/eda/05_mvrv.png", dpi=200); plt.show()'''),
], "02 Exploratory analysis of the official series"), "notebooks/02_eda.ipynb")

# ------------------------------------------------------------------ 03 features
write(nb([
    ("md", "## What this notebook does\nIt shows the feature library the model uses, proves each feature is lagged (setting future rows to NaN does not change it), and looks at how each feature relates to outcomes on development windows only."),
    ("code", PATHFIX), ("code", VERSIONS), ("code", STYLE),
    ("code", '''import numpy as np, pandas as pd
from model.regimes import load_btc
from model.strategy import construct_features, Params, FEATURE_COLS
df = load_btc(); feats = construct_features(df, Params()); feats.tail()'''),
    ("md", "### Lag test\nFor a probe date t, every row after t is replaced by NaN and the features are rebuilt. The feature values on and before t must be identical. This is the same idea as the Trilemma forward-leakage probe, applied to the features themselves."),
    ("code", '''for probe in ["2019-06-01", "2021-11-10", "2024-03-14", "2026-02-01"]:
    t = pd.Timestamp(probe); masked = df.copy(); masked.loc[masked.index > t, :] = np.nan
    f2 = construct_features(masked, Params())
    cols = [c for c in feats.columns if c.startswith("f_")]
    same = np.allclose(feats.loc[:t, cols].fillna(-9).to_numpy(), f2.loc[:t, cols].fillna(-9).to_numpy())
    print(probe, "features unchanged up to probe:", same)'''),
    ("md", "### Feature distributions"),
    ("code", '''fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
for ax, c, col in zip(axes, ["f_ma_gap", "f_drawdown", "f_mvrv_z"], PALETTE):
    ax.hist(feats[c].dropna(), bins=80, color=col); ax.set_title(c)
plt.tight_layout(); plt.savefig("output/eda/06_feature_distributions.png", dpi=200); plt.show()'''),
    ("md", "### Relationship with the next 12 months (development period only)\nAverage forward 12-month price ratio by feature decile, using only windows that start before mid-2023, the development period fixed in `model/select.py`. Hold-out windows are never used for this kind of look."),
    ("code", '''p = df["PriceUSD_coinmetrics"]; fwd = p.shift(-365) / p - 1
dev = feats.loc["2018-01-01":"2023-06-30"].copy(); dev["fwd"] = fwd.loc[dev.index]
for c in ["f_ma_gap", "f_drawdown", "f_mvrv_z"]:
    dec = pd.qcut(dev[c], 10, labels=False, duplicates="drop")
    print(c, "| mean forward 12m return by decile (low to high):", dev.groupby(dec)["fwd"].mean().round(2).tolist())'''),
], "03 Causal features"), "notebooks/03_features.ipynb")

# ------------------------------------------------------------------ 04 selection
write(nb([
    ("md", "## What this notebook does\nIt presents every registered model-selection round: the registration text (fixed before each run), the saved grid, and the outcome. Nothing here tunes anything; it reads the committed round outputs so the record and the presentation cannot drift apart."),
    ("code", PATHFIX), ("code", VERSIONS),
    ("md", "### The protocol, common to every round\nDevelopment windows start 2018-01-01 to 2023-06-30; a 12-month embargo separates them from the hold-out (window starts from 2024-07-01), because consecutive windows share up to 364 days. The selection metric is 0.5 x win rate + 0.5 x unweighted mean SPD percentile on development windows; the recency-weighted term is excluded from selection because it rewards whichever market phase ends the development period. Each round's grid is written down and registered before it runs, runs once, and is saved in full."),
    ("md", "### Round 1: price-structure features"),
    ("code", '''import json, pandas as pd
print(open("model/select.py").read().split('"""')[1])
g1 = pd.read_csv("output/selection_grid.csv"); r1 = json.load(open("output/selection_result.json"))
print("configurations:", len(g1), "| chosen:", r1["chosen"]); g1.head(5)'''),
    ("md", "### Round 2: on-chain features join the menu"),
    ("code", '''print(open("model/select_round2.py").read().split('"""')[1])
g2 = pd.read_csv("output/selection_grid_round2.csv"); r2 = json.load(open("output/selection_result_round2.json"))
print("configurations:", len(g2), "| chosen:", r2["chosen"], "| beats v1:", r2["beats_v1_on_selection_metric"])
print("best cell without the netflow term:", round(g2[g2.a_flow==0].selection_metric.max(), 2),
      "| with it:", round(g2[g2.a_flow>0].selection_metric.max(), 2))
g2.head(5)'''),
    ("md", "### Round 3a: convex blends of the two candidates"),
    ("code", '''print(open("model/select_round3a.py").read().split('"""')[1])
print(open("output/round3a_report.txt").read())'''),
    ("md", "### Round 3b: refinement of the round 2 winner's neighbourhood"),
    ("code", '''print(open("model/select_round3b.py").read().split('"""')[1])
print(open("output/round3b_report.txt").read())'''),
    ("md", "### Candidate registry\nThe registry is assembled from the committed round outputs; each entry is a named, fully specified model."),
    ("code", '''from model.candidates import build_registry
reg = build_registry()
for name, spec in reg.items():
    print(name, "->", spec["type"])'''),
], "04 Registered model-selection rounds"), "notebooks/04_model_selection.ipynb")

# ------------------------------------------------------------------ 05 validation
write(nb([
    ("md", "## What this notebook does\nIt runs every validation gate from instruction set v2, Part F, on the frozen model and writes `output/validation_report.json`. If any gate fails, nothing from this model may be reported."),
    ("code", PATHFIX), ("code", VERSIONS),
    ("code", '''import json, pandas as pd, numpy as np
from model.regimes import load_btc, REGIMES, forward_leakage_probe, constraint_check
from model.strategy import construct_features, Params, make_strategy, fast_spd_table
from template.prelude_template import check_strategy_submission_ready, compute_cycle_spd
df = load_btc(); P = json.load(open("model/final_params.json")); p = Params(**P); feats = construct_features(df, p); fn = make_strategy(p)
report = {"params": P, "data_last_day": df.index.max().strftime("%Y-%m-%d")}'''),
    ("md", "### F1: the upstream submission check (regime A constants)"),
    ("code", '''check_strategy_submission_ready(df, fn)'''),
    ("md", "### F2 and F3: leakage probe and constraint check on all three regimes"),
    ("code", '''for k, r in REGIMES.items():
    end = r.resolve_end(df)
    pr = forward_leakage_probe(df, fn, r.start, end); cc = constraint_check(df, fn, r.start, end)
    report[f"probe_{k}"] = pr; report[f"constraints_{k}"] = {"windows": cc["windows"], "passed": cc["passed"], "below_min": len(cc["below_min_weight"]), "sum_not_one": len(cc["sum_not_one"])}
    print(k, "probe", "PASS" if pr["passed"] else "FAIL", f"({len(pr['failures'])}/{pr['probes']})", "| constraints", "PASS" if cc["passed"] else "FAIL", f"({cc['windows']} windows)")'''),
    ("md", "### F7: the fast evaluator equals the upstream engine"),
    ("code", '''fast = fast_spd_table(feats, p, "2018-01-01", "2025-12-31")
up = compute_cycle_spd(df, fn, features_df=feats, start_date="2018-01-01", end_date="2025-12-31")
d = float(np.abs(fast.dynamic_percentile.to_numpy() - up.dynamic_percentile.to_numpy()).max()); report["fast_path_max_diff"] = d; print("max difference:", d)
report["all_passed"] = all(report[f"probe_{k}"]["passed"] and report[f"constraints_{k}"]["passed"] for k in REGIMES) and d < 1e-9
json.dump(report, open("output/validation_report.json", "w"), indent=2); print("ALL GATES PASSED" if report["all_passed"] else "A GATE FAILED")'''),
], "05 Validation gates"), "notebooks/05_validation.ipynb")

# ------------------------------------------------------------------ 06 results
write(nb([
    ("md", "## What this notebook does\nIt scores the frozen model and the fixed baselines on the three regimes with the unmodified Trilemma engine, runs the robustness checks, writes the ledger rows, and draws the result charts. Every number shown here is the number that goes into the educational notebook and the manuscript."),
    ("code", PATHFIX), ("code", VERSIONS), ("code", STYLE),
    ("code", '''import json, time, pandas as pd, numpy as np
from model.regimes import load_btc, REGIMES, evaluate, Regime
from model.strategy import construct_features, Params, make_strategy
from model import reference_2025
from model.candidates import build_registry, load_candidates
from template.model_development_template import precompute_features, compute_window_weights
df = load_btc(); P = json.load(open("model/final_params.json")); p = Params(**P); feats = construct_features(df, p)
upfeats = precompute_features(df)
def upstream_fn(w):
    if w.empty: return pd.Series(dtype=float)
    return compute_window_weights(upfeats, w.index.min(), w.index.max(), w.index.max())
build_registry()
models = dict(load_candidates())
models.update({"Uniform DCA": lambda w: pd.Series(1.0/len(w), index=w.index),
               "Upstream 2026 baseline (200-MA)": upstream_fn, "Tournament 2025 reference": reference_2025.compute_weights})'''),
    ("md", "### The triples for every regime. Uniform DCA against itself ties every window, so its win rate is set to zero rather than left to floating-point noise."),
    ("code", '''rows = []; tables = {}
for name, fn in models.items():
    for k, r in REGIMES.items():
        spd, s = evaluate(df, fn, feats, r); s["model"] = name; tables[(name, k)] = spd
        if name == "Uniform DCA":  # identical weights tie every window; any "wins" are floating-point dust
            s["win_rate"] = 0.0; s["score"] = 0.5 * s["rw_spd_pct"]
        rows.append(s)
res = pd.DataFrame(rows)[["model", "regime", "start", "end", "windows", "win_rate", "rw_spd_pct", "score", "uniform_rw_spd_pct", "mean_pct", "mean_excess"]]
res.to_csv("output/results_regimes.csv", index=False); res.round(2)'''),
    ("md", "### Robustness: leave one year of window starts out, and one-at-a-time parameter sensitivity (regime A)"),
    ("code", '''cand_names = [m for m in models if m.startswith(("Candidate", "Round"))]
spdA = tables[(cand_names[0], "A")]; starts = pd.to_datetime([s.split(" → ")[0] for s in spdA.index])
win = (spdA.dynamic_percentile > spdA.uniform_percentile)
print("win rate by year of window start:"); print(win.groupby(starts.year).mean().mul(100).round(1).to_string())
sens = []
for key, vals in {"a_dd": [P["a_dd"]*0.5, P["a_dd"]*1.5], "a_mvrv": [0.0, P["a_mvrv"]*2 if P["a_mvrv"] else 0.5], "m_max": [2.0, 8.0], "bias": [P["bias"]-0.1, P["bias"]+0.1]}.items():
    for v in vals:
        q = Params(**{**P, key: v}); spd, s = evaluate(df, make_strategy(q), construct_features(df, q), REGIMES["A"]); sens.append({"param": key, "value": v, **{m: round(s[m], 2) for m in ["win_rate", "rw_spd_pct", "score"]}})
pd.DataFrame(sens)'''),
    ("md", "### Charts"),
    ("code", '''fig, ax = plt.subplots(figsize=(13, 4.5))
for (name, k), col in zip([(m, "B") for m in cand_names] + [("Uniform DCA", "B"), ("Tournament 2025 reference", "B")], PALETTE):
    t = tables[(name, k)]; st = pd.to_datetime([s.split(" → ")[0] for s in t.index]); ax.plot(st, t.dynamic_percentile.rolling(7).mean(), lw=1.2, label=name, color=col)
ax.set_title("SPD percentile by window start, regime B (7-day smoothed)"); ax.set_xlabel("Window start"); ax.set_ylabel("percentile (%)"); ax.legend(); annotate_halvings(ax)
plt.tight_layout(); plt.savefig("output/07_percentile_by_window_B.png", dpi=200); plt.show()
fig, ax = plt.subplots(figsize=(7, 6))
for (m, g), col in zip(res.groupby("model"), PALETTE):
    ax.scatter(g.rw_spd_pct, g.win_rate, s=80, color=col, label=m)
    for _, row in g.iterrows(): ax.annotate(row.regime, (row.rw_spd_pct, row.win_rate), textcoords="offset points", xytext=(5, 3), fontsize=9)
ax.set_xlabel("recency-weighted SPD percentile (%)"); ax.set_ylabel("win rate (%)"); ax.set_title("Win rate against recency-weighted percentile, regimes A, B, C"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig("output/08_winrate_vs_rw.png", dpi=200); plt.show()'''),
    ("md", "### Ledger rows. The ledger is a private working file (`private/`, not tracked by git)."),
    ("code", '''import subprocess, datetime
try: commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
except Exception: commit = "uncommitted"
led = res.copy(); led.insert(0, "date", datetime.date.today().isoformat()); led.insert(1, "commit", commit); led.insert(2, "data_last_day", df.index.max().strftime("%Y-%m-%d")); led["params"] = json.dumps(P)
os.makedirs("private", exist_ok=True)
path = "private/LEDGER_v2.csv"; old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
pd.concat([old, led]).to_csv(path, index=False); print("ledger rows:", len(old) + len(led))
json.dump({"data_last_day": df.index.max().strftime("%Y-%m-%d"), "results": res.to_dict(orient="records")}, open("output/results_summary.json", "w"), indent=2)'''),
], "06 Results across regimes"), "notebooks/06_results.ipynb")

# ------------------------------------------------------------------ 07 machine learning round
write(nb([
    ("md", "## What this notebook does\nRound 4 asks whether a machine-learning model, trained and deployed with no access to future information, beats the rule-based candidates. This notebook presents the registration, the mechanics that keep the learner causal, the saved grid and the outcome. Like notebook 04, it reads committed round outputs and tunes nothing."),
    ("code", PATHFIX), ("code", VERSIONS),
    ("md", "### How the learner is kept causal\nThree mechanisms, all tested by the tournament's own probe. First, every feature is lagged at least one day. Second, training is purged walk-forward: the model refits every 90 days, and a sample only enters training once its H-day label has fully closed before the refit date, so no label ever carries information from after the day it is used. Third, the prediction stream is standardised against the expanding mean and deviation of past predictions only, then mapped through the same remaining-budget pacing allocator as every other candidate. The probe re-runs this entire pipeline, training included, on masked data at 51 probe dates."),
    ("md", "### Registration and mechanics"),
    ("code", '''print(open("model/select_round4.py").read().split('"""')[1])'''),
    ("code", '''print(open("model/ml.py").read().split('"""')[1])'''),
    ("md", "### Grid and outcome"),
    ("code", '''import json, pandas as pd
g = pd.read_csv("output/selection_grid_round4.csv")
print(g[[c for c in ["model","horizon","a_ml","win_rate","mean_pct","selection_metric","rw_spd_pct"] if c in g.columns]].round(2).to_string())
print()
print(open("output/round4_report.txt").read())'''),
    ("md", "### Prediction sanity checks\nOut-of-sample predictive power is reported as the rank correlation between the walk-forward predictions and the realised forward returns, per year. Weak or unstable correlations with a strong backtest would be a warning sign; weak correlations with a weak backtest are simply an honest negative."),
    ("code", '''import numpy as np
from model.regimes import load_btc
from model.ml import MLConfig, walk_forward_predictions
r4 = json.load(open("output/selection_result_round4.json")); ch = r4["chosen"]
cfg = MLConfig(model=str(ch["model"]), horizon=int(ch["horizon"]), a_ml=float(ch["a_ml"]))
df = load_btc(); pred = walk_forward_predictions(df, cfg)
p = df["PriceUSD_coinmetrics"]; fwd = np.log(p.shift(-cfg.horizon)/p)
both = pd.concat([pred, fwd.rename("fwd")], axis=1).dropna()
ic = both.groupby(both.index.year).apply(lambda d: d["ml_pred"].corr(d["fwd"], method="spearman"))
print("rank information coefficient by year (prediction vs realised forward return):")
print(ic.round(3).to_string())
print("overall:", round(both["ml_pred"].corr(both["fwd"], method="spearman"), 4))'''),
], "07 Round 4: a causal machine-learning candidate"), "notebooks/07_ml.ipynb")

# ------------------------------------------------------------------ 08 feature-prioritised ML and ranking
write(nb([
    ("md", "## What this notebook does\nRound 5 asks a sharper version of round 4's question: if the learner is restricted to the most important features, measured under registered rules, does it close the gap to the rule-based candidates? The notebook presents the importance ranking, the priority sets, the grid, the outcome, and closes with a descriptive ranking of every model this project has produced."),
    ("code", PATHFIX), ("code", VERSIONS),
    ("md", "### Why prioritise features at all\nRound 4 handed the learners all thirteen features at once. With noisy targets, every marginal feature is another dimension in which the model can fit chance, so the standard remedy is to rank features on development data and keep only the top of the list. The ranking uses two measures that fail differently: a rank correlation with the 90-day forward return (linear-free, per-feature) and permutation importance from a boosted model on a purged development split (captures interactions). Features are ordered by their average rank across the two, and the whole table is published rather than just the survivors."),
    ("md", "### Registration"),
    ("code", '''print(open("model/select_round5.py").read().split(chr(34)*3)[1])'''),
    ("md", "### Step 1 and 2: the importance table and the priority order"),
    ("code", '''import json, pandas as pd
imp = pd.read_csv("output/feature_importance_round5.csv", index_col=0)
print(imp.round(4).to_string())
r5 = json.load(open("output/selection_result_round5.json"))
print()
print("priority order:", r5["priority_order"])'''),
    ("md", "### Step 3 and 4: the grid, the outcome, and the gates"),
    ("code", '''g = pd.read_csv("output/selection_grid_round5.csv")
print(g[[c for c in ["model","top_k","a_ml","features","win_rate","mean_pct","selection_metric","rw_spd_pct"] if c in g.columns]].round(2).to_string())
print()
print(open("output/round5_report.txt").read().split("STEP 5")[0])'''),
    ("md", "### Step 5: ranking every model\nThe ranking below is descriptive reporting across the three regimes, not a selection device: the final model choice follows the registered protocol (development metric, then the untouched hold-out), and choosing a model from this table would amount to selecting on the test sets. The table exists so that every model built in this project can be seen in one place, including the ones the protocol rejected."),
    ("code", '''rank = pd.read_csv("output/model_ranking.csv", index_col=0)
print(rank.round(2).to_string())'''),
], "08 Round 5: feature-prioritised machine learning, and the full ranking"), "notebooks/08_feature_ml_ranking.ipynb")

# ------------------------------------------------------------------ 09 rounds 6 to 9 and closure
write(nb([
    ("md", "## What this notebook does\nRounds 6 to 9 tried every remaining legitimate route to a better score: reshaping how predictions become weights, combining models, changing what the learner predicts and which learner does the predicting, and widening the feature set. All four rounds were registered before running (decision log entries D25 to D28), a closure rule (D29) fixed in advance that modelling ends after round 9, and the final model is whichever candidate holds the best development selection metric, with the hold-out reported beside it but never used to choose. This notebook walks through each round: the reasoning first, then the committed results."),
    ("md", "### Why this cell exists\nThe next cell finds the repository root (cloning it first on Colab) so that every later cell can read the committed round outputs. It changes nothing."),
    ("code", PATHFIX),
    ("md", "### Why this cell exists\nRecording the exact Python and library versions makes every number in this notebook attributable to a specific environment, which is part of what makes the study rerunnable."),
    ("code", VERSIONS),
    ("md", "## Round 6: reshaping the allocator\nThe reasoning: candidate v4's win rates were already near the practical ceiling (84 per cent), but its recency-weighted percentile trailed the 2025 tournament reference by 20 to 30 points on two regimes. That gap is not about prediction quality; it is about how strongly a prediction is allowed to concentrate the budget. So round 6 held v4's prediction stream completely fixed and varied only three shaping choices: the allocation mechanism (gentle remaining-budget pacing against the reference's aggressive boost-now-pay-from-the-tail shape), the multiplier ceiling (5, 8 or 12 times pace), and a convexity term that amplifies extreme signals more than mild ones. If the gap closes, shape was the bottleneck; if it does not, concentration without better prediction just adds variance."),
    ("md", "### Why this cell exists\nIt prints round 6's full grid (all twelve shape combinations with their development metrics) and the round report, straight from the committed files, so the conclusion can be checked against every configuration that lost as well as the one that won."),
    ("code", '''import json, pandas as pd
print(pd.read_csv("output/round6_grid.csv").round(2).to_string())
print()
print(open("output/round6_report.txt").read())'''),
    ("md", "## Round 7: ensembles\nThe reasoning: mixtures of valid models are always valid because the constraint set is convex, so ensembles are legal by construction; whether they help is an empirical question this project had only partly answered. Round 3a showed that blending the two rule candidates diluted both. Round 7 completes the picture with three ensemble families: weight-level blends of v4 and v3 (the two best and most different candidates), prediction-level averaging of the two learners before any weight is formed (the classic variance-reduction move), and stacking, where a small meta-model learns how to weigh the two learners' opinions. The expectation, stated in advance: prediction-level combination is the most likely to help, weight-level blending the least, because the win-rate term punishes softened tilts."),
    ("md", "### Why this cell exists\nIt shows all seven ensemble configurations and the round's outcome from the committed files."),
    ("code", '''print(pd.read_csv("output/round7_grid.csv").round(2).to_string())
print()
print(open("output/round7_report.txt").read())'''),
    ("md", "## Round 8: changing what and how the learner learns\nThe reasoning, in three parts. Target engineering: predicting the mean forward return treats a 5 per cent and a 50 per cent rally as different targets, though the right allocation response is similar; a lower-quartile forecast (what is the pessimistic case?) and a simple probability-of-gain classifier are targets that map more directly onto how much to buy. Model classes: random forests, elastic net, kernel ridge and a small neural network cover the main families scikit-learn offers, so that no obvious learner is left untried; the neural network was included at the researcher's direction with low expectations recorded in advance, because a few thousand effective samples is thin for one. Refinement: the same local-neighbourhood treatment that round 3b gave the rules, applied to v4's depth and learning rate, guarding against grid-coarseness luck."),
    ("md", "### Why this cell exists\nIt shows all seventeen configurations of round 8 and the outcome."),
    ("code", '''print(pd.read_csv("output/round8_grid.csv").round(2).to_string())
print()
print(open("output/round8_report.txt").read())'''),
    ("md", "## Round 9: widening what the model can see\nThe reasoning: round 5's importance table crowned usage signals (hash rate, fees) that nobody hand-picked in rounds 1 to 3, which raises the obvious question of what else the data holds that nothing has looked at. Round 9 adds, all causally lagged: spot-volume and transaction-count momentum, slower 60-day versions of the winning usage signals, the 90-day change in the share of supply sitting on exchanges (a stock version of the netflow idea), and the Fear and Greed sentiment index, an open external series whose inclusion revised the project's earlier no-external-data decision at the researcher's direction. The registered importance procedure was re-run on the widened matrix and published before any model used it, and the learners were then restricted to the new top of the list, exactly as round 5 did."),
    ("md", "### Why this cell exists\nIt shows the re-ranked importance table for the widened feature set, then round 9's grid and outcome. Comparing this table with round 5's shows whether the new features displaced the old top three or merely joined the tail."),
    ("code", '''print(pd.read_csv("output/feature_importance_round9.csv", index_col=0).round(4).to_string())
print()
print(pd.read_csv("output/round9_grid.csv").round(2).to_string())
print()
print(open("output/round9_report.txt").read())'''),
    ("md", "## Closure\nThe closure rule was fixed before any of these rounds ran: modelling stops here, and the final model is the candidate with the best development selection metric across every registered round, with its hold-out score reported beside it. The next cell prints that adjudication from the committed file. Whatever it names is the model the write-up is built around, subject to the researcher's confirmation recorded in the decision log."),
    ("md", "### Why this cell exists\nIt prints the adjudication record: the standing best model, its metric, and each round's contribution to the journey."),
    ("code", '''adj = json.load(open("output/final_adjudication.json"))
print("rule:", adj["rule"])
print("standing best:", adj["standing_best"], "at development selection", round(adj["standing_value"], 2))
for name, r in adj["round_results"].items():
    print(f"{name}: best {r['best_label']} -> {r['dev_metrics']['selection_metric']:.2f} "
          + ("(beat the baseline)" if r["beats_baseline"] else "(negative result)"))'''),
], "09 Rounds 6 to 9: shapes, ensembles, learner variants, wider features, and closure"), "notebooks/09_rounds6to9.ipynb")

print("notebooks 01 to 09 written")

# ------------------------------------------------------------------ part 2: tournament submission and educational notebook
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_notebooks_part2 as p2  # noqa: E402
p2.build_tournament_notebook()
write(nb(p2.educational_cells(), "Dynamic DCA by causal dip-pacing: a Bitcoin accumulation model that beats uniform DCA without looking ahead"),
      "deliverables/btc_accumulation_model.ipynb")
