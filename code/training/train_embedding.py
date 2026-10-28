"""Reproducible NumPy-only embedding experiments; train/validation only."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from methods.embedding import feature_matrix
from training_utils import load_supervised, selection_accuracy, softmax
from common import parse_answer


def hard_groups(bank):
    return np.array([len({answer for c in row['candidates']
                          if (answer := parse_answer(c['text'])) is not None}) >= 2
                     for row in bank], dtype=bool)


def infer(x, parameters):
    if 'w1' in parameters:
        hidden = np.maximum(x @ parameters['w1'] + parameters['b1'], 0)
        if 'w2' in parameters:
            hidden = np.maximum(hidden @ parameters['w2'] + parameters['b2'], 0)
        return (hidden @ parameters['wo']).squeeze(-1) + parameters['bo'].item()
    return (x @ parameters['wo']).squeeze(-1) + parameters['bo'].item()


def train(train_x, train_y, val_x, val_y, config, val_hard):
    rng = np.random.default_rng(config['seed'])
    d = train_x.shape[-1]
    hidden_size = config['hidden']
    if hidden_size:
        parameters = {
            'w1': (rng.normal(size=(d, hidden_size)) * np.sqrt(2 / d)).astype(np.float32),
            'b1': np.zeros(hidden_size, np.float32),
            'wo': (rng.normal(size=(hidden_size, 1)) * np.sqrt(1 / hidden_size)).astype(np.float32),
            'bo': np.zeros(1, np.float32),
        }
    else:
        parameters = {'wo': np.zeros((d, 1), np.float32), 'bo': np.zeros(1, np.float32)}
    first = {k: np.zeros_like(v) for k, v in parameters.items()}
    second = {k: np.zeros_like(v) for k, v in parameters.items()}
    best = None
    best_metric = (-1, -1, -np.inf)
    history = []
    step = 0
    no_improvement = 0
    tick = time.perf_counter()
    batch_size = config['batch_size']
    for epoch in range(config['epochs']):
        order = rng.permutation(len(train_x))
        loss_sum = 0.0
        for begin in range(0, len(order), batch_size):
            indices = order[begin:begin + batch_size]
            xb, yb = train_x[indices], train_y[indices]
            flat = xb.reshape(-1, d)
            m = len(indices)
            if hidden_size:
                pre = flat @ parameters['w1'] + parameters['b1']
                activ = np.maximum(pre, 0)
                if config['dropout']:
                    mask = (rng.random(activ.shape) >= config['dropout']).astype(np.float32) / (1 - config['dropout'])
                else:
                    mask = np.ones_like(activ)
                hidden = activ * mask
                logits = (hidden @ parameters['wo']).reshape(m, 4) + parameters['bo'].item()
            else:
                logits = (flat @ parameters['wo']).reshape(m, 4) + parameters['bo'].item()
            if config['loss'] == 'bce':
                probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                delta = (probability - yb) / (m * 4)
                loss = np.mean(np.logaddexp(0, logits) - yb * logits)
            else:
                probability = softmax(logits)
                positive_probability = probability * yb
                positive_mass = positive_probability.sum(axis=1, keepdims=True)
                delta = (probability - positive_probability / np.maximum(positive_mass, 1e-9)) / m
                loss = -np.log(np.maximum(positive_mass, 1e-9)).mean()
                if config['loss'] == 'hybrid':
                    sigmoid = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                    delta += np.float32(.25) * (sigmoid - yb) / (m * 4)
                    loss += .25 * np.mean(np.logaddexp(0, logits) - yb * logits)
            loss_sum += float(loss) * m
            flat_delta = delta.reshape(-1, 1)
            if hidden_size:
                gradients = {'wo': hidden.T @ flat_delta, 'bo': flat_delta.sum(axis=0)}
                hidden_delta = (flat_delta @ parameters['wo'].T) * mask * (pre > 0)
                gradients['w1'] = flat.T @ hidden_delta
                gradients['b1'] = hidden_delta.sum(axis=0)
            else:
                gradients = {'wo': flat.T @ flat_delta, 'bo': flat_delta.sum(axis=0)}
            step += 1
            for name, value in parameters.items():
                gradient = gradients[name] + (config['l2'] * value if name.startswith('w') else 0)
                first[name] *= .9
                first[name] += .1 * gradient
                second[name] *= .999
                second[name] += .001 * gradient ** 2
                value -= config['lr'] * (first[name] / (1 - .9 ** step)) / (np.sqrt(second[name] / (1 - .999 ** step)) + 1e-8)
        val_scores = infer(val_x, parameters)
        val_acc = selection_accuracy(val_scores, val_y)
        hard_acc = selection_accuracy(val_scores[val_hard], val_y[val_hard])
        probabilities = softmax(val_scores)
        val_loss = float(-np.log(np.maximum((probabilities * val_y).sum(axis=1), 1e-9)).mean())
        record = {'epoch': epoch + 1, 'train_loss': loss_sum / len(train_x), 'validation_accuracy': val_acc, 'hard_validation_accuracy': hard_acc, 'hard_validation_n': int(val_hard.sum()), 'validation_listwise_loss': val_loss}
        history.append(record)
        metric = (hard_acc, val_acc, -val_loss) if config.get('selection') == 'hard' else (val_acc, -val_loss, hard_acc)
        if metric > best_metric:
            best_metric = metric
            best = {k: v.copy() for k, v in parameters.items()}
            best_epoch = epoch + 1
            best_record = record.copy()
            no_improvement = 0
        else:
            no_improvement += 1
        if (epoch + 1) % 10 == 0:
            print(json.dumps({'config': config['name'], **record}), flush=True)
        if no_improvement >= config['patience']:
            break
    return best, {'config': config, 'best_epoch': best_epoch, 'validation_accuracy': best_record['validation_accuracy'], 'hard_validation_accuracy': best_record['hard_validation_accuracy'], 'hard_validation_n': best_record['hard_validation_n'], 'validation_listwise_loss': best_record['validation_listwise_loss'], 'training_seconds': time.perf_counter() - tick, 'history': history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/embedding'))
    parser.add_argument('--preset', choices=['initial', 'extended'], default='initial')
    parser.add_argument('--name', help='Run only this configuration name')
    parser.add_argument('--config', type=Path, help='JSON configuration or list of configurations, overriding the preset')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_bank, train_arrays, train_targets = load_supervised('train')
    val_bank, val_arrays, val_targets = load_supervised('validation')
    val_hard = hard_groups(val_bank)
    train_hard = hard_groups(train_bank)
    common = {'batch_size': 128, 'epochs': 70, 'patience': 12, 'lr': .001, 'l2': .001, 'dropout': .1}
    configurations = [
        {'name': 'linear_bce', 'mode': 'raw', 'hidden': 0, 'loss': 'bce', 'seed': 17, 'lr': .003},
        {'name': 'linear_listwise', 'mode': 'raw', 'hidden': 0, 'loss': 'hybrid', 'seed': 17, 'lr': .002},
        {'name': 'mlp64_raw', 'mode': 'raw', 'hidden': 64, 'loss': 'hybrid', 'seed': 17},
        {'name': 'mlp64_relative', 'mode': 'relative', 'hidden': 64, 'loss': 'hybrid', 'seed': 17},
        {'name': 'mlp128_raw', 'mode': 'raw', 'hidden': 128, 'loss': 'hybrid', 'seed': 29},
    ]
    if args.preset == 'extended':
        configurations += [
            {'name': f'mlp{h}_{mode}_s{seed}', 'mode': mode, 'hidden': h, 'loss': 'hybrid', 'seed': seed}
            for h, mode, seed in [(64,'raw',41),(64,'relative',41),(128,'relative',29),(128,'raw',71),(256,'raw',41)]
        ]
    if args.config:
        configurations = json.loads(args.config.read_text(encoding='utf-8'))
        if isinstance(configurations, dict):
            configurations = [configurations]
    records = []
    for partial in configurations:
        if args.name and partial['name'] != args.name:
            continue
        config = {**common, **partial}
        print(json.dumps({'starting': config}), flush=True)
        train_x = feature_matrix(train_arrays['embeddings'], config['mode'])
        val_x = feature_matrix(val_arrays['embeddings'], config['mode'])
        fit_targets = train_targets
        if config.get('hard_training'):
            train_x = train_x[train_hard]
            fit_targets = train_targets[train_hard]
        mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
        scale = np.maximum(train_x.reshape(-1, train_x.shape[-1]).std(axis=0), .05)
        train_x = ((train_x - mean) / scale).astype(np.float32)
        val_x = ((val_x - mean) / scale).astype(np.float32)
        parameters, record = train(train_x, fit_targets, val_x, val_targets, config, val_hard)
        model_path = args.output / f"{config['name']}.npz"
        np.savez(model_path, mean=mean, scale=scale, feature_mode=np.array(config['mode']), **parameters)
        np.save(args.output / f"{config['name']}_validation_scores.npy", infer(val_x, parameters))
        record['checkpoint'] = str(model_path)
        record['asset_bytes'] = model_path.stat().st_size
        record['threads'] = {name: os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
        record['numpy_version'] = np.__version__
        (args.output / f"{config['name']}_training.json").write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        records.append(record)
        (args.output / 'summary.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({k: v for k, v in record.items() if k != 'history'}), flush=True)
        del train_x, val_x


if __name__ == '__main__':
    main()
