"""Summarize saved validation predictions; no test data is accessed."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from common import parse_answer
from training_utils import load_supervised, selection_accuracy

bank, _, targets = load_supervised('validation')
hard = np.array([len({a for c in row['candidates'] if (a := parse_answer(c['text'])) is not None}) >= 2 for row in bank])
directory = Path(__file__).parent
models = {}
for path in sorted(directory.glob('*_validation_scores.npy')):
    scores = np.load(path, allow_pickle=False)
    name = path.name.removesuffix('_validation_scores.npy')
    models[name] = {'validation_accuracy': selection_accuracy(scores, targets),
                    'hard_validation_accuracy': selection_accuracy(scores[hard], targets[hard]),
                    'hard_validation_n': int(hard.sum()),
                    'checkpoint': str(directory / f'{name}.npz')}
result = {'models': models, 'validation_n': len(bank), 'hard_validation_n': int(hard.sum())}
(directory / 'development_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
