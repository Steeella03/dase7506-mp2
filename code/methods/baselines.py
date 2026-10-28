"""Small training-free selectors using only the supplied candidate text.

Configurations contain ``mode``. Majority counts parsed final numeric answers;
unparseable answers receive zero votes. Ties use the first candidate unless a
length-based tie break is explicitly requested.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from common import parse_answer


MODES = ('first', 'majority', 'shortest', 'longest', 'majority_shortest', 'shortest_valid')


def candidate_scores(bank, arrays, checkpoint=None):
    mode = 'majority'
    if checkpoint is not None:
        path = Path(checkpoint)
        if path.is_dir():
            path = path / 'config.json'
        mode = json.loads(path.read_text(encoding='utf-8'))['mode']
    if mode not in MODES:
        raise ValueError(f'Unknown baseline mode: {mode}')
    scores = np.zeros((len(bank), 4), dtype=np.float32)
    for i, row in enumerate(bank):
        texts = [candidate['text'] for candidate in row['candidates']]
        if len(texts) != 4:
            raise ValueError('Expected four candidates')
        lengths = np.asarray([len(text) for text in texts], dtype=np.float32)
        if mode == 'first':
            scores[i] = -np.arange(4)
        elif mode == 'shortest':
            scores[i] = -lengths
        elif mode == 'longest':
            scores[i] = lengths
        else:
            answers = [parse_answer(text) for text in texts]
            counts = Counter(answer for answer in answers if answer is not None)
            votes = np.asarray([counts[answer] if answer is not None else 0
                                for answer in answers], dtype=np.float32)
            if mode == 'majority':
                scores[i] = votes
            elif mode == 'majority_shortest':
                # The correction cannot outweigh a single vote.
                scores[i] = votes - 0.001 * lengths / max(1.0, float(lengths.max()))
            else:
                valid = np.asarray([answer is not None for answer in answers])
                scores[i] = valid.astype(np.float32) - 0.001 * lengths / max(1.0, float(lengths.max()))
    return scores


def predict(bank, arrays, checkpoint=None):
    return candidate_scores(bank, arrays, checkpoint).argmax(axis=1).tolist()
