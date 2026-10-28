"""Compact shared candidate scorers using only supplied fixed embeddings."""
from pathlib import Path
import numpy as np


def feature_matrix(embeddings, mode='raw'):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 3 or embeddings.shape[1:] != (4, 1024):
        raise ValueError('Expected embeddings of shape [N,4,1024]')
    if mode == 'raw':
        return embeddings
    if mode == 'relative':
        return np.concatenate([embeddings, embeddings - embeddings.mean(axis=1, keepdims=True)], axis=2)
    if mode == 'normalized':
        norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
        return embeddings / np.maximum(norms, np.float32(1e-8))
    raise ValueError(f'Unknown embedding feature mode: {mode}')


def candidate_scores(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('An embedding model checkpoint is required')
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / 'model.npz'
    with np.load(checkpoint, allow_pickle=False) as model:
        mode = str(model['feature_mode'].item())
        x = feature_matrix(arrays['embeddings'], mode)
        x = (x - model['mean']) / model['scale']
        if 'w1' in model.files:
            hidden = np.maximum(x @ model['w1'] + model['b1'], np.float32(0))
            if 'w2' in model.files:
                hidden = np.maximum(hidden @ model['w2'] + model['b2'], np.float32(0))
            scores = (hidden @ model['wo']).squeeze(-1) + model['bo'].item()
        else:
            scores = (x @ model['wo']).squeeze(-1) + model['bo'].item()
        if 'center_groups' in model.files and bool(model['center_groups'].item()):
            scores = scores - scores.mean(axis=1, keepdims=True)
    if scores.shape != (len(bank), 4):
        raise ValueError('Embedding inputs are not aligned with the bank')
    return np.asarray(scores, dtype=np.float32)


def predict(bank, arrays, checkpoint):
    return candidate_scores(bank, arrays, checkpoint).argmax(axis=1).astype(int).tolist()
