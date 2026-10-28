"""Compact text-feature models trained only on the provided training split."""
from pathlib import Path
import numpy as np
from features.text_features import extract_features


def tree_scores(x, model):
    features=model['tree_features']; thresholds=model['tree_thresholds']; values=model['tree_values']
    output=np.full(len(x),float(model['base_score']),dtype=np.float32)
    ridx=np.arange(len(x))
    for f,t,v in zip(features,thresholds,values):
        nodes=np.zeros(len(x),dtype=np.int32)
        for _ in range(int(model['depth'])):
            col=f[nodes]
            active=col>=0
            branch=x[ridx,np.maximum(col,0)]>t[nodes]
            nodes=np.where(active,2*nodes+1+branch,nodes)
        output += float(model['learning_rate'])*v[nodes]
    return output


def candidate_scores(bank, arrays, checkpoint):
    if not bank: return np.empty((0,4),dtype=np.float32)
    if checkpoint is None: raise ValueError('A text-model checkpoint is required')
    path=Path(checkpoint)
    if path.is_dir(): path=path/'model.npz'
    with np.load(path,allow_pickle=False) as bundle:
        model={k:bundle[k] for k in bundle.files}
    x=extract_features(bank).reshape(len(bank)*4,-1)[:,model['feature_indices']]
    if str(model['model_type'])=='tree':
        scores=tree_scores(x,model)
    else:
        x=np.clip((x-model['mean'])/model['scale'],-8,8)
        scores=x@model['weights']+model['bias']
    scores=scores.reshape(len(bank),4)
    # Invalid numeric answers cannot earn a point under the fixed scorer.
    from common import parse_answer
    valid=np.asarray([[parse_answer(c['text']) is not None for c in row['candidates']] for row in bank])
    return np.where(valid,scores,-1e6)


def predict(bank, arrays, checkpoint):
    return [int(x) for x in np.argmax(candidate_scores(bank,arrays,checkpoint),axis=1)]
