"""Train-only group-centered ridge candidate scorers; development on validation."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import parse_answer
from training_utils import load_supervised, selection_accuracy


def main():
    output = Path('runs/embedding')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _, train_arrays, train_y = load_supervised('train')
    val_bank, val_arrays, val_y = load_supervised('validation')
    train_x = train_arrays['embeddings']
    mean = train_x.reshape(-1, 1024).mean(axis=0)
    scale = np.maximum(train_x.reshape(-1, 1024).std(axis=0), .05)
    train_x = (train_x - mean) / scale
    train_x -= train_x.mean(axis=1, keepdims=True)
    val_x = (val_arrays['embeddings'] - mean) / scale
    centered_y = train_y - train_y.mean(axis=1, keepdims=True)
    flat_x = train_x.reshape(-1, 1024).astype(np.float64)
    flat_y = centered_y.reshape(-1).astype(np.float64)
    gram = flat_x.T @ flat_x / len(flat_x)
    right = flat_x.T @ flat_y / len(flat_x)
    hard = np.array([len({a for c in row['candidates'] if (a := parse_answer(c['text'])) is not None}) >= 2 for row in val_bank])
    preprocessing_seconds = time.perf_counter() - started
    records = []
    # These four values were fixed before any validation observation.
    for alpha in (.001, .01, .1, 1.):
        tick = time.perf_counter()
        w = np.linalg.solve(gram + alpha * np.eye(1024), right).astype(np.float32)
        score = val_x @ w
        score -= score.mean(axis=1, keepdims=True)
        name = 'ridge_centered_' + str(alpha).replace('.', 'p')
        path = output / f'{name}.npz'
        np.savez(path, mean=mean, scale=scale, feature_mode=np.array('raw'),
                 wo=w[:,None], bo=np.zeros(1,np.float32), center_groups=np.array(True))
        np.save(output / f'{name}_validation_scores.npy', score)
        record = {'name':name, 'alpha':alpha, 'seed':None, 'feature_mode':'raw',
                  'objective':'mean_squared_error_of_group_centered_targets_and_embeddings',
                  'validation_accuracy':selection_accuracy(score,val_y),
                  'hard_validation_accuracy':selection_accuracy(score[hard],val_y[hard]),
                  'hard_validation_n':int(hard.sum()), 'fit_and_validation_seconds':time.perf_counter()-tick,
                  'shared_preprocessing_seconds':preprocessing_seconds,
                  'checkpoint':str(path), 'asset_bytes':path.stat().st_size,
                  'numpy_version':np.__version__,
                  'threads':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                  'command':'python -B training/train_ridge.py'}
        (output / f'{name}_training.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        records.append(record)
        print(json.dumps(record),flush=True)
    (output / 'ridge_summary.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
