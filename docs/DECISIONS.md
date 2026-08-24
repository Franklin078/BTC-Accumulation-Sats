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
| 9 | 2026-08-25 | Supervision updated: Dr James Brown and Dr Scott Alexander (Dr Len Patrick Garces has left UTS) | Change of supervisory panel |
