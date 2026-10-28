"""Shared supervision helpers. Test references are intentionally unavailable here."""
import numpy as np
from common import DATA, load_inputs, parse_answer, read_jsonl


def load_supervised(split):
    if split not in ('train', 'validation'):
        raise ValueError('Only train and validation references may be used for development')
    bank, arrays = load_inputs(split)
    references = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    if [r['id'] for r in references] != [r['id'] for r in bank]:
        raise ValueError('Reference order mismatch')
    targets = np.zeros((len(bank), 4), dtype=np.float32)
    for i, (row, reference) in enumerate(zip(bank, references)):
        answer = parse_answer(reference['answer'])
        targets[i] = [float(answer is not None and parse_answer(c['text']) == answer)
                      for c in row['candidates']]
    if not (targets.sum(axis=1) >= 1).all():
        raise ValueError('A supervised group has no correct candidate')
    return bank, arrays, targets


def selection_accuracy(scores, targets):
    return float(100 * targets[np.arange(len(targets)), np.argmax(scores, axis=1)].mean())


def softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)
