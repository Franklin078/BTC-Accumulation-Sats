"""Shared constants for the notebook builders: the version-print cell, the Colab-aware
path fix, and the chart style. Import from here, never from build_notebooks.py, whose module
level regenerates every notebook as a side effect of import.
"""

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
