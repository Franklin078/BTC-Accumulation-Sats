# Methodology notes

Working notes on the properties and limits of this study's method. These feed the manuscript's
critical-analysis chapter; they are recorded here so the caveats live next to the code they
describe.

## Sequential rounds and what they cost

Model selection ran as twelve registered rounds. Within each round the discipline is strict:
the grid is written down before it runs, runs once, and is saved in full, losers included.
Across rounds, however, each design was written knowing what the previous rounds found. That
sequence is a form of multiple testing that no single round's registration accounts for, and
it is the reason the best development metric on record (74.80, the round 12 winner) should be
read as the outcome of a guided search, not as an unbiased estimate of skill. The honest
quantities to weight most are the untouched sequestered windows described below. With
hindsight, fewer and larger rounds would have bought the same coverage with a shorter forking
path; the manuscript says so.

## The development gradient, measured

The last three generations of winners document what sequential refinement does to a selection
metric. Candidate v5 scored 73.12 on the development selection metric and 57.38 on the
hold-out; candidate v6 scored 73.41 and 52.12; candidate v7 scored 74.80 and 34.14, a hold-out
win rate of 9.76 per cent. Every point gained on the development metric after round 6 was paid
for out of the hold-out, the ordering inverts on every out-of-development measure, and the
pattern replicates exactly in the pinned tournament environment, so it is not a version
artefact. Two supporting observations: the round 12 winner sits on the boundary of its widened
grid in both directions, and one grid step away the metric falls by five points. The
refinement line was terminated on this evidence, and the study's conclusion is that the
development metric was exhausted as a selection signal, with its marginal gains
anti-correlated with generalisation. The three finalists are presented together wherever any
one of them appears.

## Hold-out depletion

The hold-out (window starts from 2024-07-01) was never used to select anything, but the winner
of every round was scored on it and the numbers were read: twelve readings in total. A test
set read twelve times still constrains, but it no longer certifies; the collapse documented
above is visible only because these readings were recorded rather than averaged away. This is
stated wherever hold-out numbers appear.

## The final model, and how it was chosen

The final model is candidate v5, confirmed by recorded decision on 30 August 2026. The basis
is disclosed in full: v5 is preferred on the out-of-development record (hold-out 57.38
against 52.12 and 34.14 for its successors; three-regime mean 65.50, first on the descriptive
ranking; stable under every measured perturbation) over the development-metric margins of
candidates v6 and v7, whose gains came with the measured losses documented above. This
weights reported hold-out readings, so it is stated as a deviation from the
development-metric closure rule rather than hidden behind it; the alternative, following that
rule to candidate v7, would have shipped the model the untouched data most distrusts. An
adjudication-by-reading rule registered on 30 August was superseded the same day by this
confirmation, before any of the windows it would have read had completed, so it decided
nothing.

## The reserved windows (one-time final reporting)

Windows starting on or after 2025-09-01 complete only from September 2026 onward, and no
selection decision has ever seen any part of their outcomes scored. They are reserved for
exactly one reading, made with the manuscript, as the final out-of-sample table reporting the
final model with the other finalists beside it. `model/final_reading.py` performs that
reading, refuses to run before any reserved window has completed, refuses to run twice, and
selects nothing; `model/regimes.py` warns if any other scoring run touches these windows.

## Environment sensitivity of the learner

Learner-based numbers depend measurably on the scikit-learn version. Development selection and
hold-out for the finalists: candidate v5, 73.12 and 57.38 under scikit-learn 1.9.0 against
72.56 and 57.80 under the tournament's pinned 1.4.2; candidate v6, 73.41 and 52.12 against
72.84 and 53.08; candidate v7, 74.80 and 34.14 against 74.03 and 35.61. The orderings, both
the development ordering and its hold-out inversion, are identical in both environments. Every
learner-based number in this project is version-tagged, and the rule-based candidates, which
reproduce to the second decimal across environments, are reported alongside for exactly this
reason.

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

## Anchor cells

Every reopened round after the tenth carried an anchor: one grid cell constructed to be
weight-identical to the standing model, whose score therefore had to reproduce the standing
number through the round's new code path. Round 11's anchor reproduced 73.12 to four decimal
places; round 12's reproduced 73.41 to ten, is compared at a relative tolerance of one part in
a billion, stamps its verdict into the round's result file, and aborts the run as invalid on
any deviation. An anchor does not prove a model is good; it proves the harness scored the old
model identically before it scored anything new, which is the difference between a finding and
an artefact.
