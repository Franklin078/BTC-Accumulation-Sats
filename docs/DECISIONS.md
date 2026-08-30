# Decision log

Methodological decisions for this project, in my own record. Dates are when the decision was made.

| # | Date | Decision | Reason |
|---|------|----------|--------|
| 1 | 2026-08-23 | The Trilemma engine and templates are used unmodified as the base code; all scoring goes through them | Comparability with the tournament's published reference implementations |
| 2 | 2026-08-23 | CoinMetrics community data is the only scoring data; refreshed each session to the last complete day | It is the tournament's source and matches the frozen 2025 dataset exactly on the overlap |
| 3 | 2026-08-23 | Three scoring configurations reported together: 2026 capstone constants, live to latest day, 2025 tournament span | Each has a different purpose; no single configuration should stand alone |
| 4 | 2026-08-23 | Features lagged one day; weights produced by remaining-budget pacing | Real-time executability and weight constraints hold by construction |
| 5 | 2026-08-23 | numpy and pandas only inside model code | Interpretability and compatibility with the pinned evaluation environment |
| 6 | 2026-08-23 | Model parameters selected on a pre-registered grid over an early development period with a 12-month embargo; the first registered round produced candidate v1, held fixed | Overlapping windows leak across naive splits; registration keeps reported numbers honest |
| 7 | 2026-08-24 | Repository name BTC-Accumulation-Sats; private until the study is finished; results published only at completion | My call on presentation and timing |
| 8 | 2026-08-24 | Further modelling proceeds in registered rounds, each written down before it is run; candidate v1 is a baseline for later rounds, not a final answer | The final model is decided at the end of the study, not at the start |
| 9 | 2026-08-25 | Supervision updated: the supervisory panel is Dr James Brown and Dr Scott Alexander (previously Dr Len Patrick Garces) | Change of supervisory panel |
| 10 | 2026-08-27 | Modelling reopened for one further registered round (round 11): three arms on the full always-available feature set, in-model selection, a block committee with unfitted combination, and a conditioned allocator around the standing model; one grid, one winner, closure re-arms on completion | The previous round's loss traced to unstable external feature selection under correlated features; each arm removes or contains that mechanism in a different way |
| 11 | 2026-08-27 | Round 11 outcome: the winner (the standing model with an asymmetric response, tilting harder into buy signals than caution signals) becomes candidate v6 on the development metric, with its weaker hold-out and regime numbers reported beside it, never selected on | The registered rule decides on the development metric alone; everything else is reported, and the tension between the two is a finding, not a secret |
| 12 | 2026-08-27 | Round 12 registered and run: a boundary refinement of the asymmetry, a two-horizon committee on the proven feature set, and a deterministic halving-calendar aggression rule; the winner becomes candidate v7 on the development metric while its hold-out collapses, and the refinement line is terminated permanently | Three generations of development gains bought progressively worse out-of-development performance; continuing would provably harvest noise (see docs/METHODOLOGY_NOTES.md, "The development gradient, measured") |
| 13 | 2026-08-30 | Final adjudication rule registered before the first sequestered window completed: the final model is whichever of the frozen candidates v5, v6 and v7 scores highest on the windows starting from 2025-09-01, read exactly once by model/final_reading.py on or after 2026-10-15; ties break toward the earlier candidate; the reading doubles as the final out-of-sample table | The development metric and the depleted hold-out point at different models; only data nobody has seen can adjudicate, and only if the rule is fixed before it is looked at |
| 14 | 2026-08-30 | Repository prepared for public release and made public ahead of the study's conclusion, revising the earlier private-until-finished timing: console log captures removed from output/ (the grids, results and reports are the canonical record), the status section rewritten to state the study's actual stage, and the pre-release checks passed (validation gates, licence and data licensing notes, no credentials or personal data) | Working in public is the project's stated ethos and lets the partner organisation see progress; the registered adjudication protects the final result from anything publication could bias |

## Round registrations

Each modelling round was registered in writing before it ran; the registration identifier is
recorded in each round's committed result file and referenced by the notebooks.

| Round | Registration | Date | Baseline to beat | Outcome |
|-------|--------------|------|------------------|---------|
| 1 | D7 | 2026-08-23 | upstream baseline | candidate v1 (dev 58.45) |
| 2 | D14 | 2026-08-26 | v1, 58.45 | candidate v2 (61.31) |
| 3a | D16 | 2026-08-26 | v2, 61.31 | negative (blends, 58.72) |
| 3b | D17 | 2026-08-26 | v2, 61.31 | candidate v3 (63.09) |
| 4 | D21 | 2026-08-26 | v3, 63.09 | negative (naive learner, 57.12) |
| 5 | D23 | 2026-08-26 | v3, 63.09 | candidate v4 (71.14) |
| 6 | D25 | 2026-08-26 | v4, 71.14 | candidate v5 (73.12) |
| 7 | D26 | 2026-08-26 | v5, 73.12 | negative (ensembles, 70.99) |
| 8 | D27 | 2026-08-26 | v5, 73.12 | negative (learner variants, 72.64) |
| 9 | D28 | 2026-08-26 | v5, 73.12 | negative (feature expansion, 69.41) |
| 10 | D36 | 2026-08-27 | v5, 73.12 | negative (synthesis committee, 53.46) |
| 11 | D38 | 2026-08-27 | v5, 73.12 | candidate v6 (73.41) |
| 12 | D40 | 2026-08-27 | v6, 73.41 | candidate v7 (74.80); refinement line terminated |
