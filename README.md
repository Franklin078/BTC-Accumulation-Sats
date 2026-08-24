# BTC Accumulation (Sats)

Research project on dynamic Bitcoin accumulation strategies. Franklin Kipkorir, Master of Data Science in Quantitative Finance, University of Technology Sydney, in partnership with the Trilemma Foundation (Stacking Sats initiative). Supervisors: Dr Len Patrick Garces and [SECOND SUPERVISOR NAME]. Trilemma project leads: Mohammad Ashkani and Matt Faltyn.

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
| Model form | A small number of named features with stated signs, combined in a bounded multiplier. Only numpy and pandas inside model code. | The tournament judges interpretability alongside score, and the pinned evaluation environment is minimal. A rule that fits in four sentences is also a rule an examiner can interrogate. |
| Selection | Parameters are chosen on a pre-registered grid over an early development period, separated from later data by a 12-month embargo, and held fixed once chosen. Each round of modelling is registered before it is run; the grid files are kept with the results. | Rolling windows overlap by up to a year, so naive train and test splits leak. Registering the search before running it is what makes a reported number mean what it appears to mean. |
| Validation | Before any configuration is reported: the tournament's own submission check, its forward-leakage probe (future rows masked, weights must not change), per-window constraint checks, and the hash check on the submission notebook. | These are the checks the tournament's evaluation engine runs. Passing them locally is a precondition for calling anything a result. |

## Status

Work in progress. The study is ongoing and results will be published here when it concludes. Interim numbers produced along the way are working artefacts, not findings.

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
    select.py                the pre-registered selection protocol
    reference_2025.py        the 2025 reference strategy, kept for benchmarking
  notebooks/                 01 data, 02 exploratory analysis, 03 features,
                             04 model selection, 05 validation, 06 results
  deliverables/              the educational notebook (drafted; completed at study end)
  data/Coin Metrics/         frozen CoinMetrics CSV (provider terms apply)
  data/official_2025/        frozen 2025 tournament dataset (benchmark only)
  output/                    charts and working artefacts
  docs/                      upstream file hashes and the decision log
  tests/                     constraint, causality and reproduction tests
  tools/                     notebook builder
```

## Reproduce

```bash
python -m venv venv && venv\Scripts\activate          # Windows; use source venv/bin/activate elsewhere
pip install -r requirements.txt
python -m pytest tests -q
jupyter lab                                            # run notebooks 01 to 06 in order
```

The tournament-format check uses a second environment built from `tournament_2025/requirements_tournament_py310.txt` on Python 3.10, in which `tournament_2025/btc_accumulation_model.ipynb` runs top to bottom and `hasher.py` confirms the boilerplate cells are identical to the template.

## Scope

This is a research artefact, not a trading system, and nothing in it is investment advice. Spreads and slippage are outside the tournament's scope and outside this project's. The Bitcoin price data belongs to CoinMetrics under its community terms; the frozen tournament dataset belongs to the Trilemma Foundation.

## Licence and attribution

Code is MIT licensed (see `LICENSE`). The files under `template/` and the tournament template are the Trilemma Foundation's, MIT licensed, reproduced unchanged with hashes recorded. The scoring framework, the metric and the problem statement are the Trilemma Foundation's; the model, the protocol, the analysis and the conclusions are mine.
