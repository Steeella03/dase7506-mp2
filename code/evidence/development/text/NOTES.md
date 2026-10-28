# Text module development handoff

These are development observations, not independent acceptance results. Only
the supplied train and validation inputs/references were used. No test
references or test scoring were accessed.

## Implementation and hypothesis

`features/text_features.py` produces 96 raw and 96 within-question centered
features. Groups cover formatting/lexical style, question overlap, numeric
answers, strict finite arithmetic equalities, and agreement/comparison among
the four supplied candidates. Numeric answers use the unchanged official
parser. No question IDs, positional lookup, pretrained weights, or external
data are used. Arithmetic expressions are interpreted through an AST whitelist
of numeric constants and +, -, *, /; Python eval is never used.

The main hypothesis is that consensus and relative candidate features improve
over judging each explanation in isolation. A secondary hypothesis is that
local arithmetic consistency adds evidence beyond style and answer features.

The learned model uses deterministic NumPy histogram gradient-boosted trees
with a listwise softmax cross entropy against the uniform distribution over
correct candidates. All-correct groups provide no ranking gradient. This
loss does not incorrectly label other correct candidates as negatives.
Trees have 32 histogram bins, learning rate .06, L2 regularization, and a
minimum leaf count. Threshold quantiles are fit on train only. Every ten
iterations, validation accuracy selects a frozen checkpoint. A linear model
uses train-fit standardization, clipping, and Adam with L2 .005.

## Initial results

All metrics below describe validation development observations. Hard validation
means at least two distinct non-None parsed final answers (355 of 798 questions).

| Method | Correct / 798 | Hard correct / 355 | Asset bytes |
|---|---:|---:|---:|
| Full features, depth 3 | 725 | 282 | 17,262 |
| Linear full features | 725 | 282 | 4,608 |
| Depth 4 search | 723 | 280 | 28,902 |
| Remove arithmetic feature group | 729 | 286 | 22,614 |
| Remove consensus feature group | 719 | 276 | 27,958 |

Recommended checkpoint: `runs/text/no_arithmetic/model.npz`, exposing 180
selected features and 110 depth-3 trees. The arithmetic-free model improved by
four validation questions over the complete model. This is evidence that this
limited arithmetic matcher was not useful in this experiment, not evidence
that arithmetic verification in general is useless. The consensus ablation
is weaker than full features by six questions; other features still contain
some indirect group-related information, so it is not a complete independent-
candidate experiment. Deeper trees increased train accuracy without improving
validation accuracy, consistent with overfitting under this setup.

For the two feature-group ablations versus full depth-3 trees, all other
settings match exactly: seed record 2026, maximum 140 trees, depth 3, learning
rate .06, L2=10, minimum leaf=50, 32 bins, and evaluation every 10 iterations.
Each run keeps its own highest full-validation checkpoint, with earliest
checkpoint retained on ties. Selected tree counts are 80 (full), 110
(no arithmetic), and 140 (no consensus). Thus model capacity at selection
differs through a shared early-stopping rule, not a different search budget.

`validation_failure_examples.json` contains five concrete failures of the
recommended C model, including question, candidate traces, parsed answers,
selection logits, votes and correct-candidate indices. These are C-model
failures, not claims about the final fused method.

One concrete failure asks about three 2-hour naps per week over 70 days. The
C model selected a candidate computing 3*2*70=420 instead of converting days
to weeks: (70/7)*3*2=60. Both answers receive two votes. Every displayed
multiplication in the selected trace is numerically correct; the error is a
missing unit conversion. This illustrates why checking local arithmetic or
counting agreement does not establish semantic correctness.

## Reproduce

Run from code/, after activating the project environment. Set OMP_NUM_THREADS,
OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 1 for the recorded training runs.

```
python -B training/train_text.py --trees 140 --depth 3 --output runs/text/full
python -B training/train_text.py --model linear --linear-steps 1000 --output runs/text/linear
python -B training/train_text.py --trees 220 --depth 4 --l2 15 --min-leaf 70 --output runs/text/depth4
python -B training/train_text.py --trees 140 --depth 3 --exclude arithmetic --output runs/text/no_arithmetic
python -B training/train_text.py --trees 140 --depth 3 --exclude consensus --output runs/text/no_consensus
```

Seed 2026 is recorded; these particular implementations are deterministic and
do not use random initialization or stochastic sampling. The first full run
took 51.14 s including shared feature extraction; subsequent cached-feature
runs took 2.31 s (linear), 76.14 s (depth 4), 33.81 s (no arithmetic), and
31.81 s (no consensus). Full feature extraction is cached for training only in
`runs/text/features.npz`; this cache must not be included as an inference asset.
All inference preprocessing is recomputed from bank inputs.

To obtain the provisional observations and score ranges, run
`python -B runs/text/analyze_validation.py`. Root should run the independent
official evaluator and resource measurement for any accepted method.

## Limits and remaining ideas

- The equality parser covers only simple finite numeric expressions and may
  miss LaTeX nesting, algebra, units and semantically incorrect formulas.
- Final-answer correctness is not proof that all reasoning steps are valid.
- Feature changes require deleting only the training cache
  `runs/text/features.npz` before retraining to recompute the schema.
- The full and arithmetic-ablated tree scores offer useful candidates for
  fusion with independently learned embedding scores; fusion is owned by D.
- Scores for invalid parsed answers are -1e6 sentinels; exclude these before
  computing calibration statistics, or apply a masked softmax.
