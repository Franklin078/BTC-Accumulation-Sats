"""Generate notebooks/10_rounds10to12.ipynb: the record of rounds 10 to 12, the development
gradient they exposed, the robustness battery for the finalists, and the registered final
reading. Run from the repository root:  python tools/build_notebook10.py
"""
import os
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, "tools")
from nbcommon import PATHFIX, VERSIONS  # noqa: E402

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("# 10. Rounds 10 to 12: the synthesis, the asymmetry, and the gradient\n\n"
   "This notebook presents the last three registered rounds and what they proved. Round 10 "
   "tested a synthesis committee on a widened feature set and lost, exposing the instability "
   "of permutation importance under correlated features. Round 11 tested three ways to use "
   "the full feature set and was won by the simplest cell in its grid, the standing model "
   "with an asymmetric response, which became candidate v6. Round 12 refined that asymmetry "
   "to the edge of a widened grid, produced candidate v7, and in doing so exposed the "
   "study's most important finding: the development metric had been exhausted, and its "
   "marginal gains now anti-correlate with generalisation. Every number here is read from "
   "the committed round artefacts under `output/`; nothing is recomputed.")
code(PATHFIX)
code(VERSIONS)

md("## Round 10: the synthesis committee\n\nEvery component was motivated by an earlier "
   "round's finding: a widened always-available feature matrix, a two-horizon forecast "
   "committee, recency-weighted training, and an asymmetric response. It lost decisively, "
   "and the mechanism matters: on the widened matrix the registered importance procedure "
   "turned against the standing model's own best features, because permutation importance "
   "punishes a feature whenever correlated additions let the probe model reroute around it. "
   "The single-pass rule reports this rather than repairing it.")
code("import json\nimport pandas as pd\nr10 = json.load(open('output/round10_result.json'))\n"
     "print('best:', r10['best_label'], '->', round(r10['dev_metrics']['selection_metric'], 2))\n"
     "print('baseline:', r10['baseline_name'], round(r10['baseline_value'], 2), "
     "'| beats:', r10['beats_baseline'])\n"
     "print('gates: probe', r10['probe_passed'], '| constraints', r10['constraints_passed'])\n"
     "imp = pd.read_csv('output/feature_importance_round10.csv', index_col=0)\n"
     "imp.round(4)")

md("## Round 11: three arms on the full feature set\n\nArm A gave one tree model all sixteen "
   "features with no selection step. Arm B partitioned the features into four economic "
   "families combined without fitted weights. Arm C kept the standing engine untouched and "
   "let the remaining features scale only how hard it tilts. One grid, one winner. The grid "
   "carried an anchor cell constructed to be weight-identical to candidate v5, which had to "
   "reproduce v5's development metric through the new code path before anything else "
   "counted. The winner was arm C with the conditioner switched off entirely: the engine "
   "with an asymmetric response, tilting three times as hard into buy signals as caution "
   "signals. It became candidate v6.")
code("r11 = json.load(open('output/round11_result.json'))\n"
     "g11 = pd.read_csv('output/round11_grid.csv')\n"
     "print('best:', r11['best_label'], '->', round(r11['dev_metrics']['selection_metric'], 2),\n"
     "      '| baseline', round(r11['baseline_value'], 2), '| beats:', r11['beats_baseline'])\n"
     "anchor = g11[g11.label == 'C: cond g=0.0 a+=2.0 a-=2.0']\n"
     "print('anchor cell reproduces v5:', round(float(anchor.selection_metric.iloc[0]), 4))\n"
     "g11.sort_values('selection_metric', ascending=False)"
     "[['label', 'selection_metric', 'win_rate', 'mean_pct']].head(10).round(2)")

md("## Round 12: the refinement, and what it cost\n\nThe round 11 winner sat on the edge of "
   "its grid, so round 12 widened the box in both directions, added the two-horizon "
   "committee on the proven features, and a deterministic halving-calendar aggression rule. "
   "Its anchor reproduced candidate v6's metric to ten decimal places at a relative "
   "tolerance of one part in a billion. The winner, at the widened boundary again, became "
   "candidate v7 on the development metric, and its hold-out collapsed.")
code("r12 = json.load(open('output/round12_result.json'))\n"
     "g12 = pd.read_csv('output/round12_grid.csv')\n"
     "print('best:', r12['best_label'], '->', round(r12['dev_metrics']['selection_metric'], 2),\n"
     "      '| baseline', round(r12['baseline_value'], 2), '| beats:', r12['beats_baseline'])\n"
     "print('anchor check:', r12['anchor_check'])\n"
     "print('gates: probe', r12['probe_passed'], '| constraints', r12['constraints_passed'])\n"
     "g12.sort_values('selection_metric', ascending=False)"
     "[['label', 'selection_metric', 'win_rate', 'mean_pct']].head(10).round(2)")

md("## The development gradient\n\nThree generations of winners, each better than the last "
   "on the metric that selected them, each worse on everything that did not. This table is "
   "the study's central critical finding: sequential refinement climbed the development "
   "metric into the ground, and the pre-registered protocol is what made the climb visible "
   "instead of publishable as progress. The pattern replicates exactly under the pinned "
   "tournament environment, so it is not a version artefact.")
code("r6 = json.load(open('output/round6_result.json'))\n"
     "rows = []\n"
     "for name, rr in [('candidate v5', r6), ('candidate v6', r11), ('candidate v7', r12)]:\n"
     "    rows.append({'model': name,\n"
     "                 'category': 'machine learning',\n"
     "                 'dev_selection': rr['dev_metrics']['selection_metric'],\n"
     "                 'holdout_score': rr['holdout']['score'],\n"
     "                 'holdout_win_rate': rr['holdout']['win_rate'],\n"
     "                 'regime_mean': (rr['regime_A']['score'] + rr['regime_B']['score'] + rr['regime_C']['score']) / 3})\n"
     "grad = pd.DataFrame(rows).round(2)\n"
     "grad")
code("import matplotlib.pyplot as plt\n"
     "fig, ax = plt.subplots(figsize=(8, 5))\n"
     "ax.plot(grad.dev_selection, grad.holdout_score, marker='o', color='#FFB000', linewidth=2)\n"
     "for _, r in grad.iterrows():\n"
     "    ax.annotate(r.model.replace('candidate ', ''), (r.dev_selection, r.holdout_score),\n"
     "                textcoords='offset points', xytext=(8, 8))\n"
     "ax.set_xlabel('development selection metric')\n"
     "ax.set_ylabel('hold-out score (reported, never selected on)')\n"
     "ax.set_title('Each development gain was paid for out of the hold-out')\n"
     "fig.savefig('output/10_dev_gradient.png', dpi=200, bbox_inches='tight')\n"
     "plt.show()")

md("## Robustness of the finalists\n\nLeave-one-year-of-starts-out win rates and "
   "one-at-a-time parameter sensitivity, computed as reporting for all three frozen "
   "finalists. Two facts stand out. First, no single early year carries any finalist: all "
   "three won every window starting in 2018, 2021, 2022 and 2023, and the drag sits in the "
   "2024 window starts for all of them. Second, candidates v5 and v6 are stable under every "
   "perturbation while candidate v7 sits on a cliff: one grid step down in its caution slope "
   "costs five and a half points, which corroborates the gradient's diagnosis.")
code("loyo = pd.read_csv('output/loyo_finalists.csv')\n"
     "loyo.pivot(index='excluded_year', columns='model', values='win_rate').round(2)")
code("sens = pd.read_csv('output/sensitivity_finalists.csv')\n"
     "sens.round(2)")

md("## The final model\n\nThe development metric and the depleted hold-out pointed at "
   "different models, and the choice was made by recorded decision: the final model is "
   "candidate v5, preferred on the record that was never selected on, its hold-out, its "
   "regime means and its stability under perturbation, and stated plainly as a deviation "
   "from the development-metric closure rule. Following that rule to candidate v7 would "
   "have shipped the model the untouched data most distrusts. A reserved set of late "
   "windows is read once, with the manuscript, as the final reporting table; it selects "
   "nothing, and this cell reports its status without touching anything.")
code("import os\nfrom model.final_reading import OUT, FINAL_MODEL\n"
     "fm = json.load(open('output/final_model.json'))\n"
     "print('final model:', fm['final_model'])\n"
     "print('confirmed on:', fm['confirmed_on'])\n"
     "print('basis:', fm['basis'][:180] + '...')\n"
     "print('reserved reporting table:',\n"
     "      'written' if os.path.exists(OUT) else 'not yet read; reserved for the manuscript')")

md("## Closure\n\nTwelve registered rounds, six negatives, seven candidates, every winner "
   "probe-verified, a selection metric caught in the act of failing, and a final model "
   "chosen on the record that was never selected on. Modelling is closed.")

n = nbf.v4.new_notebook()
n.cells = cells
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, "notebooks/10_rounds10to12.ipynb")
print("written notebooks/10_rounds10to12.ipynb")
