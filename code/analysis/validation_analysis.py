"""Explain verified validation predictions without opening test references."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import parse_answer
from training_utils import load_supervised, softmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/validation_analysis.json'))
    args = parser.parse_args()
    bank, arrays, targets = load_supervised('validation')
    answers = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    hard = np.array([len({a for a in row if a is not None}) >= 2 for row in answers])
    summaries = []
    correct_masks = {}
    for record_path in sorted((ROOT / 'runs' / 'official').glob('*/record.json')):
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if record['split'] != 'validation':
            continue
        predictions = json.loads((record_path.parent / 'validation-predictions.json').read_text(encoding='utf-8'))
        if predictions['ids'] != [row['id'] for row in bank]:
            raise ValueError('Predictions are not aligned')
        selected = np.array(predictions['selected'])
        correct = targets[np.arange(len(bank)), selected].astype(bool)
        correct_masks[record['name']] = correct
        summaries.append({'name': record['name'], 'accuracy': 100 * float(correct.mean()),
                          'hard_accuracy': 100 * float(correct[hard].mean()),
                          'easy_accuracy': 100 * float(correct[~hard].mean()),
                          'hard_correct': int(correct[hard].sum()), 'hard_n': int(hard.sum()),
                          'n': len(bank), 'asset_bytes': record['asset_bytes'],
                          'complete_seconds': record['timings'][0]['complete_seconds']})
    result = {'split': 'validation', 'n': len(bank), 'hard_n': int(hard.sum()),
              'hard_definition': 'At least two distinct non-None official parsed answers',
              'methods': summaries,
              'uncertainty_note': 'Answer-group probability mass is a diagnostic, not a calibrated correctness probability.'}
    if args.method:
        spec = importlib.util.spec_from_file_location('analysis_method', args.method.resolve())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scores = module.candidate_scores(bank, arrays, args.checkpoint)
        selected = scores.argmax(axis=1)
        correct = targets[np.arange(len(bank)), selected].astype(bool)
        valid_mask = np.array([[a is not None for a in row] for row in answers])
        nonnegative_scores = np.where(valid_mask, scores, 0)
        if (nonnegative_scores >= 0).all() and np.allclose(nonnegative_scores.sum(axis=1), 1, atol=1e-6):
            probabilities = nonnegative_scores
            result['confidence_source'] = 'Already-normalized masked component softmax mixture'
        else:
            probabilities = softmax(scores)
            result['confidence_source'] = 'Softmax of candidate scores'
        confidence = np.array([sum(probabilities[i, j] for j in range(4)
                                    if answers[i][j] == answers[i][selected[i]]) for i in range(len(bank))])
        bins = []
        ece = 0.0
        for lo in np.linspace(0, .9, 10):
            mask = (confidence >= lo) & (confidence < lo + .1 + (1e-8 if lo > .89 else 0))
            n = int(mask.sum())
            if n:
                empirical = float(correct[mask].mean())
                average = float(confidence[mask].mean())
                ece += n / len(bank) * abs(empirical - average)
                bins.append({'lower': round(float(lo), 1), 'n': n,
                             'mean_confidence': average, 'accuracy': empirical})
        errors = []
        for i in np.flatnonzero(~correct)[:8]:
            errors.append({'validation_id': bank[i]['id'], 'question': bank[i]['question'],
                           'selected': int(selected[i]), 'candidate_answers': answers[i],
                           'correct_candidates': np.flatnonzero(targets[i]).tolist(),
                           'group_mass': float(confidence[i])})
        majority = correct_masks.get('baseline-majority')
        if majority is not None:
            result['paired_vs_majority'] = {'both_correct': int((correct & majority).sum()),
                                            'only_final_correct': int((correct & ~majority).sum()),
                                            'only_majority_correct': int((~correct & majority).sum()),
                                            'both_wrong': int((~correct & ~majority).sum())}
            rng = np.random.default_rng(2026)
            differences = correct.astype(float) - majority.astype(float)
            bootstrap = differences[rng.integers(0, len(bank), size=(2000, len(bank)))].mean(axis=1) * 100
            result['paired_vs_majority']['gain_percentage_points'] = 100 * float(differences.mean())
            result['paired_vs_majority']['paired_bootstrap_95_percent_interval'] = np.quantile(bootstrap, [.025, .975]).tolist()
        result['selected_method'] = str(args.method)
        result['confidence'] = {'ece': ece, 'brier_score': float(np.mean((confidence - correct) ** 2)),
                                'bins': bins}
        hard_bins = []
        hard_ece = 0.0
        for lo in np.linspace(0, .9, 10):
            mask = hard & (confidence >= lo) & (confidence < lo + .1 + (1e-8 if lo > .89 else 0))
            n = int(mask.sum())
            if n:
                empirical = float(correct[mask].mean())
                average = float(confidence[mask].mean())
                hard_ece += n / int(hard.sum()) * abs(empirical - average)
                hard_bins.append({'lower': round(float(lo), 1), 'n': n,
                                  'mean_confidence': average, 'accuracy': empirical})
        result['confidence_hard_subset'] = {'ece': hard_ece,
                                          'brier_score': float(np.mean((confidence[hard] - correct[hard]) ** 2)),
                                          'bins': hard_bins}
        order = np.argsort(-confidence, kind='stable')
        result['risk_coverage'] = []
        for fraction in (.25, .5, .75, 1.0):
            n = max(1, int(np.ceil(len(bank) * fraction)))
            result['risk_coverage'].append({'coverage': n / len(bank), 'n': n,
                                            'accuracy': 100 * float(correct[order[:n]].mean())})
        result['validation_error_examples'] = errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'n': len(bank), 'hard_n': int(hard.sum()), 'methods': summaries}, indent=2))


if __name__ == '__main__':
    main()
