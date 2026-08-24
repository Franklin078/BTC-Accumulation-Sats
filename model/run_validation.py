"""Validation gates F1, F2, F3, F7 on the frozen model; writes output/validation_report.json."""
import json, io, contextlib, logging, time
import numpy as np, pandas as pd
from model.regimes import load_btc, REGIMES, forward_leakage_probe, constraint_check
from model.strategy import construct_features, Params, make_strategy, fast_spd_table
from template.prelude_template import check_strategy_submission_ready, compute_cycle_spd
logging.getLogger().setLevel(logging.WARNING)
df = load_btc(); P = json.load(open("model/final_params.json")); p = Params(**P); feats = construct_features(df, p); fn = make_strategy(p)
report = {"params": P, "data_last_day": df.index.max().strftime("%Y-%m-%d")}
buf = io.StringIO()
with contextlib.redirect_stdout(buf): check_strategy_submission_ready(df, fn)
txt = buf.getvalue(); report["F1_upstream_check"] = {"ready": "Strategy is ready for submission" in txt, "tail": txt.strip().splitlines()[-4:]}
print("F1 upstream check:", "READY" if report["F1_upstream_check"]["ready"] else "NOT READY"); print("\n".join(txt.strip().splitlines()[-4:]))
for k, r in REGIMES.items():
    end = r.resolve_end(df); t0 = time.time()
    pr = forward_leakage_probe(df, fn, r.start, end); cc = constraint_check(df, fn, r.start, end)
    report[f"F2_probe_{k}"] = pr; report[f"F3_constraints_{k}"] = {"windows": cc["windows"], "passed": cc["passed"], "below_min": len(cc["below_min_weight"]), "sum_not_one": len(cc["sum_not_one"])}
    print(f"regime {k}: probe {'PASS' if pr['passed'] else 'FAIL'} ({len(pr['failures'])}/{pr['probes']} failures) | constraints {'PASS' if cc['passed'] else 'FAIL'} ({cc['windows']} windows, {len(cc['below_min_weight'])} below min, {len(cc['sum_not_one'])} sum!=1) [{time.time()-t0:.0f}s]", flush=True)
fast = fast_spd_table(feats, p, "2018-01-01", "2025-12-31"); up = compute_cycle_spd(df, fn, features_df=feats, start_date="2018-01-01", end_date="2025-12-31")
d = float(np.abs(fast.dynamic_percentile.to_numpy() - up.dynamic_percentile.to_numpy()).max()); report["F7_fast_path_max_diff"] = d; print("F7 fast-path max diff:", d)
report["all_passed"] = report["F1_upstream_check"]["ready"] and all(report[f"F2_probe_{k}"]["passed"] and report[f"F3_constraints_{k}"]["passed"] for k in REGIMES) and d < 1e-9
json.dump(report, open("output/validation_report.json", "w"), indent=2); print("ALL GATES PASSED" if report["all_passed"] else "A GATE FAILED")
