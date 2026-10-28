# Embedding module handoff

Implementation and development experiments only. Root owns independent official
evaluation, resource measurement and final acceptance. No test references were
read, and no test scoring was run by this module.

## Files
- `methods/embedding.py`: `candidate_scores(bank, arrays, checkpoint)` and
  `predict(bank, arrays, checkpoint)`. Only supplied embeddings and trained
  numerical assets are used at inference. Candidate IDs/text are not looked up.
- `training/train_embedding.py`: NumPy shared linear/one-hidden-layer MLP
  training with Adam; BCE, positive-set listwise or hybrid loss. Means/scales
  fit training features only. Hidden activations use ReLU and training dropout.
- `runs/embedding/search_config.json`: nine additional explicit configurations.
- Each `<name>.npz` contains all inference arrays plus feature mode. Each
  `<name>_training.json` records seed, hyperparameters, epoch history, best epoch,
  actual training duration, NumPy version and thread settings.
- `<name>_validation_scores.npy` contains development-only scores for fusion.
  These arrays and training logs are NOT required inference assets.
- `development_metrics.json`: all 14 models on full validation (798) and hard
  validation (355, at least two distinct non-None parsed final answers).

## Exact training commands
From `code/`, use the existing virtual environment and one thread:

```powershell
. .\activate_project.ps1
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& $projectPython -B training\train_embedding.py --preset initial
& $projectPython -B training\train_embedding.py --config runs\embedding\search_config.json
& $projectPython -B runs\embedding\development_metrics.py
```

Initial five configurations select the best checkpoint using full validation
accuracy, then validation positive-set listwise loss. Additional configurations
explicitly select by hard validation accuracy, then full accuracy and loss.
Early stopping therefore uses validation; no validation data is used for
gradient fitting or feature normalization. `hard_training=true` filters only
training groups to at least two distinct parsed answers.

## Development observations (not independent acceptance)

| Checkpoint | Full validation % | Hard validation % | File bytes |
|---|---:|---:|---:|
| linear_bce.npz | 89.7243 | 77.4648 | 13,536 |
| linear_listwise.npz | 91.4787 | 81.6901 | 13,536 |
| mlp64_raw.npz | 91.3534 | 81.4085 | 272,568 |
| mlp64_relative.npz | 91.2281 | 81.4085 | 542,924 |
| mlp128_hard.npz | 90.8521 | 82.5352 | 1,067,724 |

First five fits totalled approximately 17.9 seconds; the nine subsequent fits
approximately 36.0 seconds, from the per-configuration recorded timers. Those
timers cover the training loop and validation monitoring, not interpreter
startup or data/feature loading. All fits used one OpenBLAS/OpenMP/MKL thread.

Suggested fusion candidates: `linear_listwise`, `mlp64_raw`, `mlp128_hard`.
Small nonlinear models did not consistently outperform the linear model.
Difficulty-matched validation improves the meaningful comparison: over half
the full validation bank is not in the >=2-distinct-valid-answer hard subset.
Use root official scoring and external wall-clock timing before final choice.

Inference checkpoint argument is the `.npz` path; a directory is also accepted
if it contains `model.npz`. Inference uses no cached predictions or labels and
supports arbitrary bank lengths/IDs with embeddings shaped `[N,4,1024]`.

## Raw scores versus validity postprocessing

`methods/embedding.py` and every saved validation score array return raw model
logits; they DO NOT mask candidates whose final answer cannot be parsed. The
separate `complementarity.py` diagnostic reports this distinction explicitly.
For `mlp128_hard`, masking invalid choices changes development full validation
from 90.8521% to 92.4812%, and hard validation from 82.5352% to 83.0986%.
The root/fusion module owns any actual postprocessing. Direct equal-logit means
of these models did not improve the best masked single model. Two-model errors
remain complementary, so scale calibration/answer aggregation may be useful.
Additional training has been paused at root's request.

## Root-approved follow-up experiments

Root later approved centered ridge at four predeclared normalized-Gram alpha
values (0.001, 0.01, 0.1, 1.0), plus a fixed five-seed MLP ensemble (29, 31, 37,
41, 43). No test data was used. Ridge training is `training/train_ridge.py`;
the seed configuration is `runs/embedding/seed_ensemble_config.json`.

Ridge best raw hard validation accuracy was 81.6901% (alpha 0.1); it did not
beat the MLP. Shared preprocessing was 1.324 seconds, and the four fits with
validation approximately 0.145 seconds combined.

All five MLP members are retained equally (arithmetic mean of raw logits),
without choosing/discarding lucky seeds. Seed29 is `mlp128_hard.npz`; others
are `mlp128_hard_s31.npz`, `_s37.npz`, `_s41.npz`, `_s43.npz`. New four-member
training took 23.656 seconds; all checkpoints total 5,338,620 bytes. Raw ensemble
validation is 90.8521% full / 82.5352% hard; invalid-masked development scores
are 92.7318% full / 83.6620% hard. The masked hard score improves over seed29
alone by two questions. Root still owns official verification and acceptance.

Exact commands (with the one-thread environment described above):
```
python -B training/train_ridge.py
python -B training/train_embedding.py --config runs/embedding/seed_ensemble_config.json
python -B runs/embedding/summarize_seed_ensemble.py
```

The whole ensemble is described by `seed_ensemble_manifest.json`; mean logits
for development fusion are `ensemble5_validation_logits.npy`. The fusion agent
received every model path and result. No further training is running.
