"""Frozen MP2 selector. All predictions use incoming texts and embeddings."""
from pathlib import Path

from methods.fusion import candidate_scores as fusion_scores


def candidate_scores(bank, arrays, checkpoint=None):
    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parent / 'artifacts' / 'final'
    return fusion_scores(bank, arrays, checkpoint)


def predict(bank, arrays, checkpoint=None):
    """Return one Python integer in 0..3 for every incoming question."""
    return candidate_scores(bank, arrays, checkpoint).argmax(axis=1).tolist()
