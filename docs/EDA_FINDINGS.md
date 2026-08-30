# Exploratory findings

Numbered statements from the exploratory analysis (notebook 02), each with its supporting
chart in `output/eda/`. These feed Section 2 of the educational notebook and the data chapter
of the manuscript. Quantities cited here come from the committed artefacts named beside them.

1. Bitcoin's price history is a sequence of halving-anchored cycles: each of the four halvings
   (November 2012, July 2016, May 2020, April 2024) is followed by an advance, a peak in the
   following 12 to 18 months, and a drawdown into the cycle's second half
   (`01_price_log.png`).

2. Drawdowns from the trailing high are deep and long: the major bear phases of 2018 and 2022
   both exceeded 70 per cent from peak, and recovery spans sit in the cycle's second half,
   which is where accumulation at depressed prices happens (`02_drawdown.png`).

3. Daily returns are heavy-tailed with pronounced volatility clustering: quiet regimes are
   punctuated by episodes several times the median volatility, so any rule that responds
   linearly to raw returns inherits that noise (`03_returns_vol.png`).

4. Uniform dollar cost averaging is a weak percentile performer inside its own windows: over
   the live configuration's 2,791 rolling 12-month windows the uniform strategy's mean
   sats-per-dollar percentile is 38.6, well below the midpoint, because equal daily buying
   overweights expensive stretches of trending windows
   (`04_uniform_percentile.png`, `output/spd_tables/uniform_B.csv`). This gap is the whole
   opportunity the study pursues.

5. Valuation measures identify stretch but not timing: the MVRV ratio marks the zones where
   forward 12-month outcomes were historically favourable (low MVRV) and unfavourable (high),
   yet its day-to-day movement is slow, which suits it as a brake on buying intensity rather
   than as a forecaster (`05_mvrv.png`).

6. Usage and calendar signals carry the strongest association with forward 12-month returns
   on the development period: hash-rate momentum (absolute Spearman rank correlation 0.49),
   halving-cycle position (0.48) and fee momentum (0.40) rank first on the registered
   two-way importance procedure, ahead of every valuation and price-derived feature
   (`06_feature_distributions.png`, `output/feature_importance_round5.csv`).

7. Price-derived momentum features are close to uninformative at this horizon: the 90-day
   return's rank correlation with the forward 12-month return is 0.016 and the 200-day
   moving-average gap's is 0.0005, which is why hand-set trend rules underperform the usage
   signals throughout the registered rounds (`output/feature_importance_round5.csv`).

8. The scoring metric's recency weighting concentrates evaluation: with decay 0.9, the most
   recent 60 windows carry 99.8 per cent of the recency-weighted term, so a model's score is
   dominated by the current market phase, and the same model can score very differently across
   the three reported configurations for that reason alone (`model/regimes.py`,
   documented wherever scores are shown).
