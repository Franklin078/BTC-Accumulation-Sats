# BTC Accumulation (Sats)

Research project on dynamic Bitcoin accumulation strategies. Franklin Kipkorir, Master of Data Science in Quantitative Finance, University of Technology Sydney, in partnership with the Trilemma Foundation (Stacking Sats initiative). Supervisors: Dr James Brown and Dr Scott Alexander. Trilemma project leads: Mohammad Ashkani and Matt Faltyn.

The question: uniform dollar cost averaging buys the same amount of Bitcoin every day. Can a rule that varies the daily amount, using only information available at the time, accumulate more satoshis per dollar over 12-month windows, judged by the Trilemma Foundation's published scoring framework?

## How this repository is organised

The Trilemma tournament code is the base and stays untouched; everything on top of it is my own work.

- `template/` and `tournament_2025/model_development_template.ipynb` are the Trilemma Foundation's scoring engine and submission template, reproduced byte for byte. Their SHA-256 hashes are recorded in `docs/UPSTREAM_HASHES.txt`. All scores in this project come from these files, unmodified, so that results are comparable with the tournament reference implementations.
- `model/` is my code: feature construction, the allocation rule, the evaluation harness that calls the Trilemma engine, and the model selection protocol.
- `notebooks/` are my analysis notebooks, numbered in running order.
- `tests/` verify the constraints, the absence of look-ahead, and that the frozen 2025 tournament template reproduces its published score on this machine before anything else is trusted.

## Method choices

| Choice | What I do | Why |
|---|---|---|
| Scoring | Every score comes from the unmodified Trilemma engine (`template/prelude_template.py`), run over three window configurations: the 2026 capstone constants (2018 to end of 2025), a live configuration extended to the latest complete day, and the 2025 tournament span (2016 to mid-2025). | A score is only meaningful next to the published reference numbers, and those exist per configuration. Reporting all three keeps any single configuration from flattering a model. |
| Data | CoinMetrics community daily series, refreshed to the last complete day by `notebooks/01_data.ipynb`, with `PriceUSD` as the only price used for scoring. An integrity gate (no missing days, no null prices, no duplicates) runs before any scoring session. | CoinMetrics is the tournament's own source. The refreshed series matches the frozen 2025 tournament dataset to machine precision on every overlapping day, which the data notebook verifies on each run. |
| Causality | Every feature is lagged one day: the weight for day t is fixed from information available at the close of day t-1. | A daily accumulation schedule has to be executable in real time. The one-day lag makes that property structural rather than something to argue about. |
| Weight constraints | Remaining-budget pacing: each day's weight is the even pace of the remaining budget scaled by a bounded multiplier, and the final day takes the remainder. | The tournament requires every weight to stay above a minimum and each window's weights to sum to one. Pacing satisfies both by construction, with no rescaling step after the fact. |
| Model form | Two tracks. The tournament-format track is a small number of named features with stated signs in a bounded multiplier, numpy and pandas only. The study's primary track adds one causally trained learner signal (scikit-learn, which the pinned environment provides), produced by purged walk-forward training and standardised against its own past only. | The tournament judges interpretability alongside score, and its 3-cell format forbids new imports, so the strict artefact stays simple. The learner track exists because the registered rounds showed usage and calendar signals out-forecast hand-set rules; its causality is enforced by the same probe. |
| Selection | Parameters are chosen on a pre-registered grid over an early development period, separated from later data by a 12-month embargo, and held fixed once chosen. Each round of modelling is registered before it is run; the grid files are kept with the results. | Rolling windows overlap by up to a year, so naive train and test splits leak. Registering the search before running it is what makes a reported number mean what it appears to mean. |
| Validation | Before any configuration is reported: the tournament's own submission check, its forward-leakage probe (future rows masked, weights must not change), per-window constraint checks, and the hash check on the submission notebook. | These are the checks the tournament's evaluation engine runs. Passing them locally is a precondition for calling anything a result. |

## Status

The study is complete. The final model is candidate v5 (a HistGradientBoosting learner on hash-rate momentum, fee momentum and cycle position, mapped to daily weights by a paced allocator with ceiling 12), confirmed by recorded decision on the out-of-development record and stated with its basis in `output/final_model.json` and the decision log. Selection ran as twelve pre-registered rounds with every grid, result and negative kept in `output/`; the two later candidates that beat v5 on the development metric, and the measured reasons they were not chosen, are reported beside it throughout. A reserved set of late evaluation windows is read once, with the manuscript, as the final out-of-sample table (`model/final_reading.py`); it reports and selects nothing.

## Results

Final Model Score (0.5 x win rate + 0.5 x recency-weighted sats-per-dollar percentile) per scoring configuration: A is the 2026 capstone span (2018 to end of 2025), B is live to the latest complete day, C is the 2025 tournament span (2016 to mid-2025). Ranked by the three-configuration mean; every number is reproduced by `notebooks/06_results.ipynb` and the window-level tables in `output/spd_tables/`.

| # | Model | Category | A | B | C | Mean |
|---|-------|----------|------|------|------|------|
| 1 | Candidate v5 (v4 signal, ceiling 12, round 6), the final model | machine learning | 60.62 | 75.57 | 60.32 | 65.50 |
| 2 | Candidate v4 (feature-prioritised ML, round 5) | machine learning | 60.03 | 75.04 | 59.63 | 64.90 |
| 3 | Round 7 best (v4/v3 blend 0.75, negative round) | blend (ML + rules) | 60.76 | 73.61 | 59.56 | 64.64 |
| 4 | Round 8 best (hgbr depth 2, negative round) | machine learning | 61.75 | 75.74 | 56.19 | 64.56 |
| 5 | Candidate v6 (v5 with asymmetric response, round 11) | machine learning | 58.66 | 71.91 | 63.04 | 64.54 |
| 6 | Round 9 best (expanded features, negative round) | machine learning | 61.74 | 73.58 | 57.44 | 64.26 |
| 7 | Candidate v7 (refined asymmetric response, round 12) | machine learning | 57.53 | 69.63 | 64.43 | 63.86 |
| 8 | **Tournament 2025 reference** | **benchmark (rules)** | **67.02** | **49.18** | **72.55** | **62.91** |
| 9 | Round 4 best (all-feature ridge, not a candidate) | machine learning | 59.05 | 69.52 | 60.06 | 62.88 |
| 10 | Candidate v1 (drawdown-led, round 1) | rules | 59.05 | 57.68 | 56.55 | 57.76 |
| 11 | Candidate v2 (MVRV + netflow, round 2) | rules | 52.89 | 58.75 | 51.82 | 54.49 |
| 12 | Candidate v3 (refined MVRV + netflow, round 3b) | rules | 53.70 | 61.22 | 48.38 | 54.44 |
| 13 | Round 10 best (synthesis committee, negative round) | machine learning | 48.14 | 60.62 | 53.69 | 54.15 |
| 14 | Upstream 2026 baseline (200-MA) | benchmark (rules) | 51.74 | 56.66 | 51.26 | 53.22 |
| 15 | Uniform DCA | benchmark (uniform) | 19.08 | 24.31 | 21.00 | 21.46 |

The final model beat uniform DCA in 85.7 per cent of the 2,791 live windows and clears the bolded Trilemma 2025 reference on the three-configuration mean, as do the six models above it. Candidates v6 and v7 beat v5 on the development selection metric and were not chosen; the measured reasons are in `docs/METHODOLOGY_NOTES.md` and `output/final_model.json`, and the losing rounds' full grids are kept in `output/`.

## Repository map

```
BTC-Accumulation-Sats/
  template/                  Trilemma capstone engine, unmodified (base code)
  tournament_2025/           Trilemma 2025 tournament template, hasher, pinned Python 3.10
                             requirements, and my 3-cell submission notebook
  model/
    strategy.py              features, the allocation rule, template-compatible entry points
    regimes.py               the three scoring configurations; calls the Trilemma engine;
                             leakage probe and constraint checks
    ml.py                    the causal learner pipeline (purged walk-forward, causal
                             standardisation, committee and allocator variants)
    candidates.py            the registry of candidates the registered rounds produced
    select*.py               one script per registered round, kept exactly as registered
    roundutil.py             the shared round runner (fixed protocol, fingerprinted cache)
    robustness.py            leave-one-year-out and parameter sensitivity for the finalists
    final_reading.py         the registered one-time reading of the sequestered windows
    reference_2025.py        the 2025 reference strategy, kept for benchmarking
  notebooks/                 01 data, 02 exploratory analysis, 03 features,
                             04 model selection, 05 validation, 06 results,
                             07 causal learner, 08 feature priority and ranking,
                             09 rounds 6 to 9, 10 rounds 10 to 12
  deliverables/              the educational notebook (completed as the study concludes)
  data/Coin Metrics/         frozen CoinMetrics CSV (provider terms apply)
  data/official_2025/        frozen 2025 tournament dataset (benchmark only)
  output/                    grids, results, reports and charts for every round, losers
                             included, plus the descriptive ranking and validation report
  docs/                      upstream hashes, decision log, environments, methodology notes,
                             exploratory findings, data licensing notes
  tests/                     constraint, causality, reproduction and consistency tests
  tools/                     notebook builder, ranking and adjudication refreshers, gates
```

The directory name on disk may differ from the repository name above; the repository is the
same either way.

## Reproduce

```bash
python -m venv venv && venv\Scripts\activate          # Windows; use source venv/bin/activate elsewhere
pip install -r requirements.txt
python -m pytest tests -q
jupyter lab                                            # run notebooks 01 to 10 in order
```

The tournament-format check uses a second environment built from `tournament_2025/requirements_tournament_py310.txt` on Python 3.10, in which `tournament_2025/btc_accumulation_model.ipynb` runs top to bottom and `hasher.py` confirms the boilerplate cells are identical to the template.

## Scope

This is a research artefact, not a trading system, and nothing in it is investment advice. Spreads and slippage are outside the tournament's scope and outside this project's. The Bitcoin price data belongs to CoinMetrics under its community terms; the frozen tournament dataset belongs to the Trilemma Foundation.

## Licence and attribution

Code is MIT licensed (see `LICENSE`). The files under `template/` and the tournament template are the Trilemma Foundation's, MIT licensed, reproduced unchanged with hashes recorded. The scoring framework, the metric and the problem statement are the Trilemma Foundation's; the model, the protocol, the analysis and the conclusions are mine.
