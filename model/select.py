"""Pre-registered model selection.

Protocol (fixed before any of these numbers were seen):
  development windows : starts 2018-01-01 to 2023-06-30 (scored range 2018-01-01 to 2024-06-30)
  embargo             : starts 2023-07-01 to 2024-06-30 are used by neither side (windows overlap by up to a year)
  hold-out windows    : starts 2024-07-01 onward (regime A: to 2024-12-31; regime B: to the latest start)
  selection metric    : 0.5 * win rate + 0.5 * unweighted mean SPD percentile on development windows
                        (the rho = 0.9 term is deliberately excluded from selection because it is regime-specific)
  tie-break           : fewer active terms, then smaller m_max
  grid                : listed below, 72 configurations, each evaluated once
Every configuration and its development metrics are written to output/selection_grid.csv.
"""
import itertools, json, sys, time, logging
import pandas as pd
from model.regimes import load_btc, score_table, REGIMES
from model.strategy import construct_features, Params, fast_spd_table

logging.getLogger().setLevel(logging.WARNING)
DEV = ("2018-01-01", "2024-06-30")
GRID = dict(a_ma=[0.0, 1.0], a_dd=[0.5, 1.0, 2.0], bias=[0.0, 0.2], a_mvrv=[0.0, 0.5, 1.0], m_max=[3.0, 5.0])

def main():
    df = load_btc()
    feats = construct_features(df, Params())
    rows = []
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    t0 = time.time()
    for i, vals in enumerate(combos, 1):
        kw = dict(zip(keys, vals))
        p = Params(a_win=0.0, ma_asym=True, **kw)
        t = fast_spd_table(feats, p, *DEV)
        s = score_table(t)
        s["selection_metric"] = 0.5 * s["win_rate"] + 0.5 * s["mean_pct"]
        s["active_terms"] = int(kw["a_ma"] > 0) + int(kw["a_dd"] > 0) + int(kw["a_mvrv"] > 0) + int(kw["bias"] > 0)
        s.update(kw)
        rows.append(s)
        print(f"[{i:3d}/{len(combos)}] {kw} -> win {s['win_rate']:.2f} meanpct {s['mean_pct']:.2f} sel {s['selection_metric']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    grid = pd.DataFrame(rows).sort_values(["selection_metric", "active_terms", "m_max"], ascending=[False, True, True])
    grid.to_csv("output/selection_grid.csv", index=False)
    best = grid.iloc[0]
    chosen = {k: (float(best[k]) if k != "m_max" else float(best[k])) for k in keys}
    json.dump({"protocol": __doc__, "chosen": chosen, "dev_metrics": {k: float(best[k]) for k in ["win_rate", "mean_pct", "rw_spd_pct", "selection_metric"]}}, open("output/selection_result.json", "w"), indent=2)
    print("\nCHOSEN:", chosen, "\n", best.to_dict())

if __name__ == "__main__":
    main()
