# Manuscript support files

Produced by tools/manuscript_support.py from the committed results and the frozen data; data close 2026-08-24. Nothing here changes a model or a score.

| File | Content |
|---|---|
| splits.json | window counts and date bounds of the four data splits and the three evaluation periods |
| decomposition.csv | win rate, recency-weighted percentile and score for every model and period; score = 0.5 win + 0.5 RW checked to 1e-9 |
| spd_uniform_B.csv | window-level table of Uniform DCA on evaluation period B at the data close, regenerated through the engine and matched to results_regimes.csv |
| spd_reference_B.csv | window-level table of Tournament 2025 reference on evaluation period B at the data close, regenerated through the engine and matched to results_regimes.csv |
| spd_upstream_B.csv | window-level table of Upstream 2026 baseline (200-MA) on evaluation period B at the data close, regenerated through the engine and matched to results_regimes.csv |
| benchmark_tables_check.json | agreement of the regenerated benchmark window tables with the committed scores |
| worked_scoring_2025.json | the four-step scoring of the window starting 1 January 2025, checked against the uniform window table; the final model's row is from output/spd_tables/robust_v5_B.csv |
| worked_scoring_2025_daily.csv | daily prices and sats per dollar inside the window starting 1 January 2025 |
| worked_allocator_five_day.csv | five-day window with budget 1 through the committed pacing allocator (model/ml.py _allocate_pace), multipliers 1, 2, 0.5, 12 and the remainder day |
| worked_allocator_day_one.csv | day-one arithmetic for a 365-day window at the floor, neutral and ceiling multipliers |
| weight_paths.csv | daily weights over a 365-day window for three multiplier paths, from the committed allocator |
| multiplier_response.csv | the final model's multiplier at selected standardised forecasts; ceiling binds above ln(12)/2 = 1.2425, floor below ln(0.25)/2 = -0.6931 |
| round_record.csv | the twelve registered rounds: registration, grid size, development metric, baseline, margin and the hold-out reading in all its parts |
| round6_summary.json | round 6's grid read for the ceiling effect: best, second best, the ceiling-5 cell and the best reference-shape cell |
| halving_cycles.csv | for each halving cycle: the deepest drawdown from the trailing 365-day high, the peak it fell from, and the days from the halving to that peak; the current cycle is incomplete |
| drawdowns.csv | deepest drawdown from the trailing 365-day high in each major decline, from the frozen series |
| drawdown_shares.json | share of days since 2018 spent more than 20, 40 and 60 per cent below the trailing 365-day high |
| volatility.json | return distribution and 30-day realised volatility statistics since 2018 |
| spearman_features.csv | signed Spearman correlation of the thirteen round-4 features with the 90-day forward log return, reproducing round 5's committed absolute values exactly |
| mvrv_bands.csv | window starts in evaluation period B grouped by the MVRV level known at the start: uniform DCA's window percentile and the 365-day forward log return |
| recency_weights.json | how much of the recency-weighted percentile the most recent windows carry, for each period size |
| sequestered_accumulation.csv | the 14 sequestered windows in accumulation terms on a $1,000,000 budget, from the committed one-time reading table |
| sequestered_accumulation.json | the first sequestered window in dollars, BTC and prices, from the committed reading and the frozen series |
| v5_signal_daily.csv | the final model's standardised forecast and multiplier for every day from 2018 (scikit-learn 1.9.0) |
| v5_holdout_multipliers.csv | for each hold-out window: the largest multiplier the final model reached and whether its weights differ from candidate v4's |
| v5_multiplier_summary.json | distribution of the final model's multiplier over period B and the test of why rounds 5 and 6 share one hold-out score |
