"""The one-time reading of the reserved final-reporting windows (starts on or after
2025-09-01), kept for the manuscript's final out-of-sample table. The final model is
candidate v5, confirmed by recorded decision (docs/DECISIONS.md and output/final_model.json);
this reading selects nothing. It reports the final model with the other two finalists beside
it, exactly once: the script refuses to run before any reserved window has completed and
refuses to run twice.

Reproduce:  python -m model.final_reading   (once, when the manuscript's final table is built)
Outputs:    output/final_reading.json, output/spd_tables/final_reading_{v5,v6,v7}.csv
"""
import json
import os

import numpy as np
import pandas as pd

from model.candidates import load_candidates_with_feats
from model.regimes import Regime, evaluate, load_btc

RESERVED = Regime("S", "Reserved final-reporting windows (starts 2025-09-01 onward)",
                  "2025-09-01", None)
FINAL_MODEL = "Candidate v5 (v4 signal, ceiling 12, round 6)"
FINALISTS = [FINAL_MODEL,
             "Candidate v6 (v5 with asymmetric response, round 11)",
             "Candidate v7 (refined asymmetric response, round 12)"]
OUT = "output/final_reading.json"


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSED: {OUT} already exists. The reserved windows are read "
                         f"exactly once and that reading stands. Delete nothing.")
    import sklearn
    df = load_btc()
    last_start = df.index.max() - pd.DateOffset(years=1)
    n_windows = (last_start - pd.Timestamp(RESERVED.start)).days + 1
    if n_windows < 1:
        raise SystemExit(f"REFUSED: no reserved window has completed yet (data ends "
                         f"{df.index.max().date()}). This table is read once, with the "
                         f"manuscript. Nothing has been read.")
    pairs = load_candidates_with_feats(df)
    result = {"purpose": "one-time final reporting table; the final model was confirmed by "
                         "recorded decision and this reading selects nothing",
              "final_model": FINAL_MODEL,
              "read_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
              "data_last_day": df.index.max().strftime("%Y-%m-%d"),
              "sklearn_version": sklearn.__version__, "candidates": {}}
    for name in FINALISTS:
        fn, feats = pairs[name]
        spd, s = evaluate(df, fn, feats, RESERVED)
        tag = name.split(" ")[1].lower()
        spd.to_csv(f"output/spd_tables/final_reading_{tag}.csv")
        result["candidates"][name] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                      for k, v in s.items()}
        print(f"{name}: windows {s['windows']}  win {s['win_rate']:.2f}  "
              f"RW {s['rw_spd_pct']:.2f}  score {s['score']:.2f}")
    json.dump(result, open(OUT, "w"), indent=2, default=float)
    print(f"\nfinal reporting table written; the final model is {FINAL_MODEL}")


if __name__ == "__main__":
    main()
