"""Offline candidate fusion and answer-group aggregation.

All component scores are calculated from the incoming bank and embeddings.
The configuration contains only component asset paths and scalar settings.
"""
import importlib
import json
from pathlib import Path

import numpy as np

from common import parse_answer


def answer_groups(bank):
    answers = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    valid = np.asarray([[a is not None for a in row] for row in answers], dtype=bool)
    matches = np.asarray([[[a is not None and a == b for b in row] for a in row]
                          for row in answers], dtype=bool)
    return answers, valid, matches


def transform_scores(scores, valid, transform='softmax', temperature=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if transform == 'softmax':
        logits = np.where(valid, scores / temperature, -1e12)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(np.clip(logits, -745, 0)) * valid
        return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-30)
    if transform == 'sigmoid':
        return np.where(valid, 1.0 / (1.0 + np.exp(-np.clip(scores / temperature, -30, 30))), 0.0)
    if transform == 'raw':
        return np.where(valid, scores, 0.0)
    raise ValueError(f'Unknown score transform: {transform}')


def fuse_scores(component_scores, components, valid, matches, config):
    if not components:
        raise ValueError('At least one scoring component is required')
    weight_total = sum(float(c.get('weight', 1.0)) for c in components)
    if weight_total <= 0:
        raise ValueError('Component weights must sum to a positive value')
    mixed = np.zeros(valid.shape, dtype=np.float64)
    for scores, component in zip(component_scores, components):
        weight = float(component.get('weight', 1.0))
        if weight < 0:
            raise ValueError('Component weights must be nonnegative')
        mixed += weight * transform_scores(scores, valid, component.get('transform', 'softmax'),
                                            float(component.get('temperature', 1.0)))
    mixed /= weight_total
    pooling = config.get('pooling', 'sum')
    pool_weight = float(config.get('pool_weight', 0.0))
    if not 0 <= pool_weight <= 1:
        raise ValueError('pool_weight must lie in [0,1]')
    counts = matches.sum(axis=2)
    if pooling == 'sum':
        pooled = np.einsum('nij,nj->ni', matches, mixed)
    elif pooling == 'mean':
        pooled = np.einsum('nij,nj->ni', matches, mixed) / np.maximum(counts, 1)
    elif pooling == 'max':
        pooled = np.max(np.where(matches, mixed[:, None, :], -1e12), axis=2)
    else:
        raise ValueError(f'Unknown pooling: {pooling}')
    output = (1.0 - pool_weight) * mixed + pool_weight * pooled
    output += float(config.get('vote_weight', 0.0)) * (counts - 1.0) / 3.0
    return np.where(valid, output, -1e12)


def load_configuration(checkpoint):
    if checkpoint is None:
        raise ValueError('A fusion configuration or bundle directory is required')
    path = Path(checkpoint)
    if path.is_dir():
        path = path / 'config.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    return path, config


def candidate_scores(bank, arrays, checkpoint):
    config_path, config = load_configuration(checkpoint)
    if not bank:
        return np.empty((0, 4), dtype=np.float64)
    _, valid, matches = answer_groups(bank)
    outputs = []
    components = config['components']
    for component in components:
        method_name = component['method']
        if method_name not in ('embedding', 'text', 'baselines'):
            raise ValueError(f'Unsupported component method: {method_name}')
        module = importlib.import_module(f'methods.{method_name}')
        assets = component.get('checkpoints')
        if assets:
            members = [np.asarray(module.candidate_scores(bank, arrays, (config_path.parent / asset).resolve()))
                       for asset in assets]
            scores = np.mean(members, axis=0)
        else:
            asset = component.get('checkpoint')
            asset = None if asset is None else (config_path.parent / asset).resolve()
            scores = np.asarray(module.candidate_scores(bank, arrays, asset))
        if scores.shape != (len(bank), 4) or not np.isfinite(scores).all():
            raise ValueError('A component returned invalid scores')
        outputs.append(scores)
    return fuse_scores(outputs, components, valid, matches, config)


def predict(bank, arrays, checkpoint):
    return [int(i) for i in np.argmax(candidate_scores(bank, arrays, checkpoint), axis=1)]
