"""Build exactly one fixed five-seed fusion variant for the development record."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.fusion import answer_groups, candidate_scores
from training.tune_fusion import save_bundle, evaluate
from training_utils import load_supervised

source = json.loads(Path('runs/fusion/second/search_summary.json').read_text(encoding='utf8'))['best']
settings = {k: source[k] for k in ('schema_version', 'pooling', 'pool_weight', 'vote_weight')}
components = [dict(source['components'][0])]
seeds = [29, 31, 37, 41, 43]
paths = ['runs/embedding/mlp128_hard.npz'] + [f'runs/embedding/mlp128_hard_s{seed}.npz' for seed in seeds[1:]]
ensemble = dict(source['components'][1])
ensemble.pop('checkpoint')
ensemble['checkpoints'] = [str(Path(p).resolve()) for p in paths]
components += [ensemble, dict(source['components'][-1])]
configuration = save_bundle(Path('runs/fusion/ensemble5'), components, settings)
bank, arrays, targets = load_supervised('validation')
answers, valid, matches = answer_groups(bank)
hard = np.asarray([len({a for a in row if a is not None}) >= 2 for row in answers])
metrics = evaluate(candidate_scores(bank, arrays, Path('runs/fusion/ensemble5')), targets, hard)
record = {'hypothesis': 'Fixed equal-logit seed ensemble reduces variance; no seed selection or scalar retuning',
          'mlp_seeds': seeds, 'weights': {'text': .75, 'linear': .125, 'mlp_ensemble': .125},
          'development_metrics': metrics, 'search_configurations': 1,
          'configuration': 'runs/fusion/ensemble5/config.json',
          'note': 'All five members retained, mean raw logits followed by masked softmax. No test access. Root official scoring pending.'}
Path('runs/fusion/ensemble5/development_record.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
print(json.dumps(record, indent=2))
