"""Describe a predeclared, equal-weight seed ensemble on validation only."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from common import parse_answer
from training_utils import load_supervised, selection_accuracy

directory = Path(__file__).parent
bank, _, y = load_supervised('validation')
valid = np.array([[parse_answer(c['text']) is not None for c in r['candidates']] for r in bank])
hard = np.array([len({a for c in r['candidates'] if (a := parse_answer(c['text'])) is not None}) >= 2 for r in bank])
names = ['mlp128_hard', 'mlp128_hard_s31', 'mlp128_hard_s37', 'mlp128_hard_s41', 'mlp128_hard_s43']
members = []
arrays = []
for name in names:
    arrays.append(np.load(directory / f'{name}_validation_scores.npy'))
    members.append({'seed': 29 if name == 'mlp128_hard' else int(name.rsplit('s',1)[1]),
                    'checkpoint':f'runs/embedding/{name}.npz', 'weight':.2})
scores = np.mean(arrays, axis=0)
masked_scores = np.where(valid, scores, -1e4)
result = {'hypothesis':'Equal-weight averaging reduces sensitivity to random training seed.',
          'selection':'All five predeclared seeds included, without member selection on validation.',
          'aggregation':'Arithmetic mean of raw logits; invalid-answer mask reported separately.',
          'members':members, 'validation_n':len(bank), 'hard_validation_n':int(hard.sum()),
          'raw_validation_accuracy':selection_accuracy(scores,y),
          'raw_hard_validation_accuracy':selection_accuracy(scores[hard],y[hard]),
          'masked_validation_accuracy':selection_accuracy(masked_scores,y),
          'masked_hard_validation_accuracy':selection_accuracy(masked_scores[hard],y[hard]),
          'checkpoint_bytes':sum((directory / f'{name}.npz').stat().st_size for name in names),
          'new_member_training_seconds':sum(json.loads((directory / f'{name}_training.json').read_text())['training_seconds'] for name in names[1:]),
          'commands':['python -B training/train_ridge.py', 'python -B training/train_embedding.py --config runs/embedding/seed_ensemble_config.json']}
(directory / 'seed_ensemble_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
np.save(directory / 'ensemble5_validation_logits.npy',scores)
print(json.dumps(result,indent=2))
