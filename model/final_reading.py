"""The one-time final reading of the sequestered windows, as registered before any of them
completed. The final model is whichever frozen finalist (candidates v5, v6, v7) holds the
highest Final Model Score (0.5 win rate + 0.5 recency-weighted percentile) on the windows
starting on or after 2025-09-01; ties break toward the earlier candidate. This reading is the
manuscript's final out-of-sample table and happens exactly once: the script refuses to run
before the registered date and refuses to run twice.

Reproduce:  python -m model.final_reading   (on or after 2026-10-15, once)
Outputs:    output/final_reading.json, output/spd_tables/final_reading_{v5,v6,v7}.csv
"""
import datetime
import json
import os

import numpy as np

from model.candidates import load_candidates_with_feats
from model.regimes import Regime, evaluate, load_btc

READING_OPENS = datetime.date(2026, 10, 15)
SEQUESTERED = Regime("S", "Sequestered final reading (starts 2025-09-01 onward)", "2025-09-01", None)
FINALISTS = ["Candidate v5 (v4 signal, ceiling 12, round 6)",
             "Candidate v6 (v5 with asymmetric response, round 11)",
             "Candidate v7 (refined asymmetric response, round 12)"]
OUT = "output/final_reading.json"


def main():
    today = datetime.date.today()
    if today < READING_OPENS:
        raise SystemExit(f"REFUSED: the registered reading date is {READING_OPENS}; today is "
                         f"{today}. The sequestered windows are read exactly once, on or after "
                         f"that date. Nothing has been read.")
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSED: {OUT} already exists. The sequestered windows are read "
                         f"exactly once; this reading has already happened and its result "
                         f"stands. Delete nothing.")
    import sklearn
    df = load_btc()
    pairs = load_candidates_with_feats(df)
    result = {"registered_rule": "highest Final Model Score on sequestered windows; ties to the "
                                 "earlier candidate; read once on or after 2026-10-15",
              "read_on": today.isoformat(), "data_last_day": df.index.max().strftime("%Y-%m-%d"),
              "sklearn_version": sklearn.__version__, "candidates": {}}
    best_name, best_score = None, -np.inf
    for name in FINALISTS:
        fn, feats = pairs[name]
        spd, s = evaluate(df, fn, feats, SEQUESTERED)
        tag = name.split(" ")[1].lower()  # v5 / v6 / v7
        spd.to_csv(f"output/spd_tables/final_reading_{tag}.csv")
        result["candidates"][name] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                      for k, v in s.items()}
        print(f"{name}: windows {s['windows']}  win {s['win_rate']:.2f}  "
              f"RW {s['rw_spd_pct']:.2f}  score {s['score']:.2f}")
        if s["score"] > best_score:  # strict >, so ties keep the earlier candidate
            best_name, best_score = name, s["score"]
    result["final_model"] = best_name
    result["final_score_sequestered"] = float(best_score)
    json.dump(result, open(OUT, "w"), indent=2, default=float)
    print(f"\nFINAL MODEL under the registered rule: {best_name} "
          f"(sequestered score {best_score:.2f})")


if __name__ == "__main__":
    main()
