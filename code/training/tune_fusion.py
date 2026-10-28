"""Small, explicit validation-only fusion search; no fitting or test access."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.fusion import answer_groups, fuse_scores
from training_utils import load_supervised, selection_accuracy


def evaluate(scores, targets, hard):
    choices = scores.argmax(axis=1)
    correct = targets[np.arange(len(targets)), choices]
    return {'accuracy': float(100 * correct.mean()),
            'correct': int(correct.sum()), 'n': len(targets),
            'hard_accuracy': float(100 * correct[hard].mean()),
            'hard_correct': int(correct[hard].sum()), 'hard_n': int(hard.sum())}


def save_bundle(destination, components, config):
    destination.mkdir(parents=True, exist_ok=True)
    models = destination / 'models'
    models.mkdir(exist_ok=True)
    resolved_components = []
    for index, component in enumerate(components):
        current = dict(component)
        if current.get('checkpoints'):
            copied = []
            for member_index, asset in enumerate(current['checkpoints']):
                source = Path(asset)
                name = f'{index}-{member_index}-{current["method"]}-{source.name}'
                shutil.copyfile(source, models / name)
                copied.append(f'models/{name}')
            current['checkpoints'] = copied
        else:
            source = Path(current['checkpoint'])
            name = f'{index}-{current["method"]}-{source.name}'
            shutil.copyfile(source, models / name)
            current['checkpoint'] = f'models/{name}'
        resolved_components.append(current)
    payload = {**config, 'components': resolved_components}
    (destination / 'config.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=Path, action='append', required=True)
    parser.add_argument('--embedding-ensemble', type=Path, nargs='+', help='One equal-raw-logit embedding component with all listed members')
    parser.add_argument('--text', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/fusion'))
    parser.add_argument('--fixed', action='store_true', help='Rebuild a frozen scalar configuration without a grid')
    parser.add_argument('--text-weight', type=float, default=0.75)
    parser.add_argument('--pool-weight', type=float, default=0.0)
    parser.add_argument('--vote-weight', type=float, default=0.0)
    args = parser.parse_args()
    started = time.perf_counter()
    bank, arrays, targets = load_supervised('validation')
    answers, valid, matches = answer_groups(bank)
    hard = np.asarray([len(set(a for a in row if a is not None)) >= 2 for row in answers])
    args.output.mkdir(parents=True, exist_ok=True)
    component_sources = [{'method': 'embedding', 'checkpoint': str(p.resolve()),
                          'weight': 1.0, 'transform': 'softmax', 'temperature': 1.0}
                         for p in args.embedding]
    if args.embedding_ensemble:
        component_sources.append({'method': 'embedding', 'checkpoints': [str(p.resolve()) for p in args.embedding_ensemble],
                                  'weight': 1.0, 'transform': 'softmax', 'temperature': 1.0})
    embedding_count = len(component_sources)
    if args.text:
        component_sources.append({'method': 'text', 'checkpoint': str(args.text.resolve()),
                                  'weight': 1.0, 'transform': 'softmax', 'temperature': 1.0})
    raw_scores = []
    for component in component_sources:
        module = importlib.import_module('methods.' + component['method'])
        if component.get('checkpoints'):
            raw_scores.append(np.mean([module.candidate_scores(bank, arrays, Path(path))
                                       for path in component['checkpoints']], axis=0))
        else:
            raw_scores.append(module.candidate_scores(bank, arrays, Path(component['checkpoint'])))
    choices = []
    # Predeclared short grid, with no fine-grained validation fitting.
    text_weights = ([args.text_weight] if args.fixed else [0.0, 0.25, 0.5, 0.75, 1.0]) if args.text else [0.0]
    if any(not 0 <= weight <= 1 for weight in text_weights):
        parser.error('--text-weight must lie in [0,1]')
    for text_weight in text_weights:
        selected = []
        selected_scores = []
        if text_weight < 1:
            for component, scores in zip(component_sources[:embedding_count], raw_scores[:embedding_count]):
                current = dict(component, weight=(1 - text_weight) / embedding_count)
                selected.append(current)
                selected_scores.append(scores)
        if text_weight > 0:
            selected.append(dict(component_sources[-1], weight=text_weight))
            selected_scores.append(raw_scores[-1])
        for pool_weight in ([args.pool_weight] if args.fixed else (0.0, 0.5, 1.0)):
            for vote_weight in ([args.vote_weight] if args.fixed else (0.0, 0.1, 0.25)):
                config = {'schema_version': 1, 'pooling': 'sum', 'pool_weight': pool_weight,
                          'vote_weight': vote_weight}
                scores = fuse_scores(selected_scores, selected, valid, matches, config)
                metric = evaluate(scores, targets, hard)
                choices.append({'text_weight': text_weight, **config, **metric,
                                'components': selected})
    # Prefer less aggregation and fewer components when validation counts tie.
    choices.sort(key=lambda x: (-x['hard_correct'], -x['correct'],
                               len(x['components']), x['pool_weight'], x['vote_weight'],
                               x['text_weight']))
    best = choices[0]
    config_keys = ('schema_version', 'pooling', 'pool_weight', 'vote_weight')
    best_config = {k: best[k] for k in config_keys}
    best_bundle = save_bundle(args.output / 'best', best['components'], best_config)
    variants = {'best': best_bundle}
    for name, changes in [('without_votes', {'vote_weight': 0.0}),
                          ('without_pooling', {'pool_weight': 0.0})]:
        variants[name] = save_bundle(args.output / name, best['components'], {**best_config, **changes})
    embed_components = [dict(c, weight=1.0 / embedding_count)
                        for c in component_sources[:embedding_count]]
    variants['embedding_only'] = save_bundle(args.output / 'embedding_only', embed_components,
                                             {'schema_version': 1, 'pooling': 'sum', 'pool_weight': 0., 'vote_weight': 0.})
    variants['embedding_vote'] = save_bundle(args.output / 'embedding_vote', embed_components,
                                             {'schema_version': 1, 'pooling': 'sum', 'pool_weight': 0., 'vote_weight': .1})
    if args.text:
        variants['text_only'] = save_bundle(args.output / 'text_only', [dict(component_sources[-1], weight=1.0)],
                                             {'schema_version': 1, 'pooling': 'sum', 'pool_weight': 0., 'vote_weight': 0.})
        variants['text_vote'] = save_bundle(args.output / 'text_vote', [dict(component_sources[-1], weight=1.0)],
                                             {'schema_version': 1, 'pooling': 'sum', 'pool_weight': 0., 'vote_weight': .1})
        variants['without_text'] = save_bundle(args.output / 'without_text', embed_components, best_config)
    summary = {'split': 'validation', 'search_kind': 'frozen_configuration' if args.fixed else 'predeclared_small_scalar_grid',
               'searched_configurations': len(choices), 'search_seconds': time.perf_counter() - started,
               'seed': None, 'note': 'No statistical model fitted here; component models trained on train only. Root performs official scoring and acceptance.',
               'n': len(bank), 'hard_n': int(hard.sum()), 'component_sources': component_sources,
               'best': best, 'search_results': choices, 'variants': list(variants)}
    (args.output / 'search_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('searched_configurations', 'search_seconds', 'n', 'hard_n', 'best')}, indent=2))


if __name__ == '__main__':
    main()
