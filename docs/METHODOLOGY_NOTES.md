# Methodology notes

Working notes on the properties and limits of this study's method. These feed the manuscript's
critical-analysis chapter; they are recorded here so the caveats live next to the code they
describe.

## Sequential rounds and what they cost

Model selection ran as nine registered rounds. Within each round the discipline is strict: the
grid is written down before it runs, runs once, and is saved in full, losers included. Across
rounds, however, each design was written knowing what the previous rounds found. That sequence
is a form of multiple testing that no single round's registration accounts for, and it is the
reason the development metric of the final model (73.12) should be read as the outcome of a
guided search, not as an unbiased estimate of skill. The honest quantities to weight most are
the untouched hold-out and, once available, the sequestered windows described below. With
hindsight, fewer and larger rounds would have bought the same coverage with a shorter forking
path; the manuscript says so.

## Hold-out depletion

The hold-out (window starts from 2024-07-01) was never used to select anything, but the winner
of every round was scored on it and the numbers were read: nine readings in total. A test set
read nine times still constrains, but it no longer certifies. This is stated wherever hold-out
numbers appear.

## The sequestered windows (one-time final reading)

Windows starting on or after 2025-09-01 complete only after the modelling closure of
26 August 2026, and no selection decision has ever seen any part of their outcomes scored.
They are sequestered: scored exactly once, at manuscript time, for the final out-of-sample
table, and never before. `model/regimes.py` warns if any scoring run touches them earlier.
This is the study's only remaining fully clean test, and it stays that way by not being looked
at.

## Environment sensitivity of the learner

The final model's numbers depend measurably on the scikit-learn version: development selection
73.12 and hold-out 57.38 under scikit-learn 1.9.0; 72.56 and 57.80 under the tournament's
pinned 1.4.2. The adjudication is unchanged in both environments, but every learner-based
number in this project is version-tagged, and the rule-based candidates, which reproduce to
the second decimal across environments, are reported alongside for exactly this reason.

## Tournament-format compliance and the dual track

The 2025 tournament's submission format allows edits to one model cell and forbids new import
statements there. Its environment contains scikit-learn, but the boilerplate imports it only as
`import sklearn`, which does not expose the learner classes, and adding `from sklearn... import`
would be a new import under the rule's plain reading (verified empirically in the pinned
environment). Consequently the strict 3-cell artefact carries the strongest compliant simple
model (candidate v1, which passes the tournament's own checks on the frozen tournament dataset),
while the learner-based final model is presented as the primary result of the capstone and the
manuscript, where no such format restriction exists. This split is deliberate protocol
compliance, not an omission.

## Configuration versus dataset

Regime C in this repository reproduces the 2025 tournament's *configuration* (window span and
scoring) on current CoinMetrics data. The tournament's own *dataset* is the frozen parquet under
`data/official_2025/`, and the tournament-format notebook scores against that. The two agree to
machine precision on overlapping days at the time of writing, but they are not the same object,
and numbers from one are never presented as the other.

## Cache integrity

Round results are cached with a fingerprint of the configuration list and the data day; a cached
result is reused only when the fingerprint matches, and its baseline comparison is recomputed
against the baseline of the current run. This closes the gap in which a result computed by an
earlier version of the code could silently survive a code change.
