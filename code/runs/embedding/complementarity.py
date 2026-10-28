"""Development diagnostics for three frozen models; validation inputs only."""
from itertools import combinations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from common import parse_answer
from training_utils import load_supervised, selection_accuracy

bank, _, y = load_supervised('validation')
valid = np.array([[parse_answer(c['text']) is not None for c in r['candidates']] for r in bank])
hard = np.array([len({a for c in r['candidates'] if (a := parse_answer(c['text'])) is not None}) >= 2 for r in bank])
directory = Path(__file__).parent
names = ['linear_listwise', 'mlp64_raw', 'mlp128_hard']
scores = {n: np.load(directory / f'{n}_validation_scores.npy') for n in names}
selected_correct = {n: y[np.arange(len(y)), s.argmax(axis=1)].astype(bool) for n, s in scores.items()}
result = {'hard_n': int(hard.sum()), 'raw_vs_invalid_mask': {}, 'pairwise': {}, 'simple_ensembles': {}}
for name, s in scores.items():
    masked = np.where(valid, s, -1e4)
    result['raw_vs_invalid_mask'][name] = {'raw_full': selection_accuracy(s,y), 'raw_hard': selection_accuracy(s[hard],y[hard]), 'masked_full': selection_accuracy(masked,y), 'masked_hard': selection_accuracy(masked[hard],y[hard]), 'raw_invalid_selections': int((~valid[np.arange(len(y)),s.argmax(axis=1)]).sum())}
for a, b in combinations(names, 2):
    ca, cb = selected_correct[a], selected_correct[b]
    result['pairwise'][f'{a}+{b}'] = {'hard_a_only_correct':int((hard & ca & ~cb).sum()), 'hard_b_only_correct':int((hard & ~ca & cb).sum()), 'hard_both_wrong':int((hard & ~ca & ~cb).sum()), 'hard_choice_disagreements':int((hard & (scores[a].argmax(axis=1) != scores[b].argmax(axis=1))).sum())}
for group in list(combinations(names,2)) + [tuple(names)]:
    s = sum(scores[n] for n in group) / len(group)
    masked = np.where(valid,s,-1e4)
    result['simple_ensembles']['+'.join(group)] = {'raw_full':selection_accuracy(s,y), 'raw_hard':selection_accuracy(s[hard],y[hard]), 'masked_full':selection_accuracy(masked,y), 'masked_hard':selection_accuracy(masked[hard],y[hard])}
(directory / 'complementarity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
