# DASE7506 MP2 - Compact Evidence Fusion

Student ID: **3036709224**. Public repository: https://github.com/Steeella03/dase7506-mp2

## Data setup for the GitHub distribution

All frozen model weights and the report are included. Retrieve the unchanged
official data from instructor commit `bb801f4f2e7629d270df765ab12c333b2dc616bb`.
After creating the environment described below, run from `code`:

~~~bat
.venv\Scripts\python.exe fetch_official_data.py
.venv\Scripts\python.exe check_data.py
~~~

On Linux/macOS use `.venv/bin/python`. Only the initial data download requires
internet; training and inference run offline. The downloader checks every
original SHA-256 hash and reuses matching local files. Alternatively copy the
original `code/data` directory from that upstream revision before checking data.
No extra datasets or pretrained weights are downloaded.

An offline selector for the supplied four-candidate mathematics bank.

- Frozen entry point: code/submission.py
- Complete checkpoint: code/artifacts/final
- Release identifier: mp2-751aa13c6e1c
- Frozen test accuracy: 384/467 = 82.22698072805139%
- Majority baseline: 323/467 = 69.16488222698072%
- Additional inference assets: 5,375,667 uncompressed bytes (5.127 MiB)
- Complete local test time: 2.150-2.187 seconds across three fresh runs

The release identifier is a local source/artifact fingerprint, not a Git
commit. Its full hash, source hashes, asset hashes and freeze time are in
code/evidence/frozen_release.json. A public repository address and submission issue
are added only after publication; none is invented here.

## Install and check data

Use Python 3.12 and numpy==2.5.3 (code/requirements.txt). Development used
Python 3.12.14, NumPy 2.5.3 and CPU only on Windows 11, with 12 logical CPUs
and 15.73 GiB RAM. No extra inference dependency, GPU, model download or API
key is required. Official data, parser and scorer remain unchanged.

On the development Windows machine, open code in File Explorer and
double-click setup_environment.cmd. It installs the required dependency and
runs the official data check, keeping the result window open. This route does
not require changing PowerShell script execution policy. Project-generated
environments, caches and temporary files use D:.

On another Windows machine with Python 3.12, open Command Prompt in code:

~~~bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe check_data.py
~~~

On Linux/macOS, use python3.12 -m venv .venv, then equivalent commands with
.venv/bin/python. All subsequent commands are run from code.

## Reproduce the frozen score without retraining

Use the shipped artifacts/final directory and the matching submitted source.
Set inference thread limits before starting Python:

~~~bat
set OMP_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set MKL_NUM_THREADS=4
.venv\Scripts\python.exe -B predict.py --method submission.py --checkpoint artifacts/final --split test --output runs/reproduction/test-predictions.json
.venv\Scripts\python.exe -B score_predictions.py --split test --predictions runs/reproduction/test-predictions.json --output runs/reproduction/test-score.json
~~~

The score is the independent scorer's accuracy field in percent. It should be
82.22698072805139 for the frozen bundle. Official records are
evidence/official/final-test/test-score.json and evidence/official/final-test/record.json.
Reproduction does not require training caches, saved validation logits or
rerunning training/search.

For complete local process timing and asset accounting:

~~~bat
.venv\Scripts\python.exe -B run_experiment.py --name reproduced-final-test --method submission.py --checkpoint artifacts/final --split test --final-test
~~~

The explicit --final-test flag prevents accidental test use during development.
The harness runs the unchanged prediction and scoring scripts as fresh
processes. Complete time includes startup, imports, asset/data loading,
preprocessing and every ensemble member. Local timing does not establish
eligibility on the instructor's Xeon Platinum 8457C reference CPU.

For an unseen audit bank without references, replace PATH_TO_AUDIT_INPUTS:

~~~bat
.venv\Scripts\python.exe -B predict.py --method submission.py --checkpoint artifacts/final --split audit --data-dir PATH_TO_AUDIT_INPUTS --output runs/reproduction/audit-predictions.json
~~~

The method consumes incoming bank text and arrays. It does not depend on
public-test IDs, sample counts, reference files, answer lookups or cached
predictions.

## Retrain every component from scratch

A frozen recipe is included at training/final_recipe.json. The builder
regenerates text features, trains the linear embedding model and all five MLP
seeds, trains the text trees, and assembles the fixed fusion configuration.
It does not read an existing checkpoint or retune fusion weights.

Choose an empty output directory; the builder refuses to reuse a nonempty
directory. For example:

~~~bat
.venv\Scripts\python.exe -B training/reproduce_final.py --output runs/reproduced-final-new
~~~

The rebuilt checkpoint is runs/reproduced-final-new/bundle. Evaluate it by
replacing artifacts/final with that path in the official commands above.
The builder forces one thread per numerical library and stores complete
training logs, the recipe and reconstruction hashes. Its original clean
reconstruction is recorded in evidence/retraining/reproduction_record.json:
80.3146579 seconds including fresh features, training processes and assembly.
All seven rebuilt weight files match the frozen SHA-256 hashes. Root independent reconstruction acceptance confirmed all seven weight hashes,
semantic configuration equality and identical choices on all 798 validation
questions; see evidence/reproduction_acceptance.json.

## Method and fixed settings

The model combines:

1. A shared linear scorer on the supplied 1024-dimensional embeddings.
2. Five shared 128-unit ReLU MLP scorers on each embedding concatenated with
   its difference from the within-question mean.
3. Histogram gradient-boosted trees on candidate-text and relative-group
   features, excluding the limited local-arithmetic feature group.

Training-fit means, scales, tree thresholds, weights and scalar settings are
all in the checkpoint. The embedding objective is positive-set listwise loss
plus 0.25 times binary cross entropy. Adam uses batch size 128, at most 70
epochs and patience 12. The linear model uses seed 17, learning rate 0.002
and L2 0.001. MLPs use seeds 29, 31, 37, 41 and 43, learning rate 0.001,
L2 0.005 and dropout 0.3. MLPs train on hard training groups. Validation
selects checkpoints; normalization and gradients fit training only.

The five MLP raw logits are averaged equally. No member is discarded based on
validation performance. Linear logits, averaged MLP logits and tree scores
are separately converted to masked softmax distributions at temperature 1,
then mixed with weights 0.125, 0.125 and 0.75. Unparseable final-answer
candidates are excluded. Extra group pooling and explicit vote weights are
zero. The tree recipe uses seed record 2026, depth 3, 32 histogram bins,
learning rate 0.06, L2 10, minimum leaf 50 and at most 140 trees; validation
retains 110 trees. This particular tree implementation is deterministic.

## Results, resources and supporting records

| Comparison | Correct / questions | Accuracy (%) |
|---|---|---|
| Majority, full validation | 688 / 798 | 86.2155 |
| Final fusion, full validation | 751 / 798 | 94.1103 |
| Majority, hard validation | 245 / 355 | 69.0141 |
| Final fusion, hard validation | 308 / 355 | 86.7606 |
| Majority, frozen test | 323 / 467 | 69.1649 |
| Final fusion, frozen test | 384 / 467 | 82.2270 |

The frozen-test gain is 61 questions, or 13.0621 percentage points. The
5,375,667-byte bundle is below 16 MiB. Complete local test timings are
2.1535209, 2.1495663 and 2.1866493 seconds, each including prediction and
independent scoring. Reference-machine eligibility remains the instructor's
decision.

Evidence locations:

- report/MP2_Report.pdf: report of at most 10 pages; report/report.md is source.
- evidence/official/final-test/: official frozen-test JSON, logs and external timing.
- evidence/official/final-validation/: corresponding validation evidence.
- evidence/official/baseline-majority-test/: frozen baseline comparison.
- evidence/frozen_release.json: source/artifact fingerprint, freeze time and hashes.
- evidence/acceptance.json: root acceptance record.
- evidence/reproduction_acceptance.json: independent reconstruction comparison.
- evidence/resource_summary.json: complete search counts and recorded costs.
- evidence/validation_analysis.json: full/hard comparisons, paired bootstrap,
  confidence diagnostics, risk/coverage and final-method validation failures.
- evidence/hardware.json: local CPU/RAM/OS and training-device record.
- evidence/development/embedding/*_training.json and evidence/development/text/*/training.json: exact recipes,
  seeds, selected checkpoints, histories and component training costs.
- evidence/development/text/NOTES.md: same-recipe feature ablations and limitations.
- evidence/development/fusion/reproduction_map.json and training/final_recipe.json:
  component recipes and frozen settings.
- evidence/retraining/reproduction_record.json: clean reconstruction log.

Training/search timers have different scopes from official inference timings.
The report states their scope rather than presenting their sum as an exact
end-to-end search wall time. Development-only feature caches and validation
score arrays are not inference assets or inputs.

## Data use and limitations

The disjoint supplied splits contain 4,755 training, 798 validation and 467
test questions, each with four candidates. Test references were used only
after freezing for final scoring/reproduction, never for training, feature
statistics, hyperparameter selection or answer tables. The freeze predeclared
only the final method and majority baseline for test comparison. No model
or setting was changed in response to test performance.

Training/validation include easier groups, whereas every test group has at
least two distinct valid parsed answers. We therefore also analyse the
355-question difficulty-matched hard validation subset. Comparing its 86.76%
with test 82.23% is more informative than comparing mixed validation 94.11%
with test. Both distribution differences and validation-selection optimism
may contribute. Bootstrap/confidence results on the reused validation bank
do not remove that optimism or guarantee calibrated correctness.

The scorer checks final numeric answers, not every reasoning step. Fluent
text, agreement and locally correct arithmetic can still miss a unit
conversion, confuse receipts with profit or interpret the question wrongly.

## AI assistance and reuse disclosure

Substantive AI assistance was used through ChatGPT/Codex and multiple agents.
Agents helped interpret requirements, implement selectors and training,
write experiment tools, analyse validation results, and draft the report and
README. A coordinating agent runs independent official evaluation and
integrates artifacts. This must not be represented as unaided student work.
The student is responsible for reviewing and understanding all implementation
and claims before submission.

The assignment supplies questions/references derived from GSM8K, candidate
solutions and fixed representations prepared with Qwen3-0.6B, and official
evaluation tools. Student code uses Python's standard library and NumPy.
No external dataset or additional pretrained weights are included. Keep
code/data/LICENSE.txt and code/data/NOTICE.txt when redistributing.

## Submission

Deadline: 31 October 2026, 23:59 Hong Kong/Beijing time (UTC+8). Use
https://xudongwu-0.github.io/courses/dase7506/mp2/ with the student's ID,
official full-test accuracy, publicly accessible source repository and matching
checkpoint/configuration link. Preparing a link is not submission: create the
generated GitHub issue. Use the same GitHub account and student ID for updates.
Publication and formal score submission follow the student's final review.

## References

1. Assignment guide: https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md
2. Running instructions: https://github.com/xudongwu-0/DASE7506-MP2/blob/main/code/README.md
3. GSM8K: https://github.com/openai/grade-school-math
4. Qwen3-0.6B attribution: https://huggingface.co/Qwen/Qwen3-0.6B


