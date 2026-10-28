# MP2 implementation coordination

User goal: invest time to improve accuracy and prepare the complete assignment,
including report, reproducible code and matching inference assets. The root
agent coordinates and accepts the work. Implementation agents do not perform
independent acceptance or claim the complete task passed.

## Rules
- Preserve official data, common.py, predict.py and score_predictions.py.
- Train on train only; validation may guide development. Never read test
  references or evaluate test during model development.
- No external data, pretrained weights, online inference or answer lookup.
- New bank inputs and arbitrary question IDs/counts must work.
- Inference: CPU, four total threads, <=30 seconds prediction plus independent
  scoring including startup and loading. <=16 MiB uncompressed extra assets.
- Use the existing code/.venv Python (3.12.14) and numpy==2.5.3. Do not install
  dependencies, change global settings, publish anything or operate git.
- All new work, caches and temporary files stay on D:.
- Training agents use one BLAS/OpenMP thread initially while jobs are parallel.
- Report actual seeds, hyperparameters, runtimes and failures. Never invent
  results. Development metrics are provisional until root official scoring.

## File ownership
- Agent A: methods/baselines.py; analysis/profile_training.py; runs/baseline/.
- Agent B: methods/embedding.py; training/train_embedding.py; runs/embedding/.
- Agent C: features/text_features.py; methods/text.py;
  training/train_text.py; runs/text/.
- Agent D (later): methods/fusion.py; training/tune_fusion.py; runs/fusion/.
- Agent E: report/; README_STUDENT.md. Root supplies verified experiment records.
- Root: submission.py, training_utils.py, run_experiment.py, tests/, shared
  orchestration and final package. Do not edit another owner's files.

Every method exposes predict(bank, arrays, checkpoint). Learned methods should
also expose candidate_scores(bank, arrays, checkpoint), shape [N,4], to enable
fusion. checkpoint is pathlib.Path or None. Return Python int choices.
training_utils.load_supervised(split) supports train and validation only.

## Handoff
Send paths, exact commands, dependencies, model configurations/seeds, asset
sizes, provisional validation observations, remaining issues and next ideas.
Leave method/checkpoint pairs usable without retraining. No test scoring.
