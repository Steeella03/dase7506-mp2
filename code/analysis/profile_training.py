"""Describe training data and simple selection heuristics; no test access."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import parse_answer
from methods.baselines import MODES, candidate_scores
from training_utils import load_supervised, selection_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/baseline/training_profile.json'))
    args = parser.parse_args()
    bank, arrays, targets = load_supervised('train')
    lengths = np.asarray([[len(c['text']) for c in row['candidates']] for row in bank])
    answers = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    answer_patterns = Counter()
    for row in answers:
        counts = Counter(answer for answer in row if answer is not None)
        answer_patterns['+'.join(map(str, sorted(counts.values(), reverse=True))) or 'none'] += 1
    result = {
        'split': 'train',
        'questions': len(bank),
        'correct_candidate_count': dict(sorted(Counter(map(int, targets.sum(axis=1))).items())),
        'candidate_correct_percent_by_position': (100 * targets.mean(axis=0)).tolist(),
        'parsed_answer_count_patterns': dict(answer_patterns.most_common()),
        'unparseable_candidates': sum(answer is None for row in answers for answer in row),
        'text_characters': {
            'mean': float(lengths.mean()),
            'median': float(np.median(lengths)),
            'p10': float(np.quantile(lengths, .1)),
            'p90': float(np.quantile(lengths, .9)),
            'mean_correct': float(lengths[targets.astype(bool)].mean()),
            'mean_incorrect': float(lengths[~targets.astype(bool)].mean()),
        },
        'heuristic_training_accuracy': {},
        'note': 'Descriptive training-only observations; official validation and final acceptance belong to the root agent.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        config = args.output.parent / f'{mode}.json'
        config.write_text(json.dumps({'mode': mode}, indent=2) + '\n', encoding='utf-8')
        scores = candidate_scores(bank, arrays, config)
        result['heuristic_training_accuracy'][mode] = selection_accuracy(scores, targets)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
