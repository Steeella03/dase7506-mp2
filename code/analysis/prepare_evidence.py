"""Collect compact experiment evidence outside ignored scratch run directories."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main():
    evidence = ROOT / 'evidence'
    records = []
    for folder, target in [('embedding', 'embedding'), ('text', 'text'), ('fusion', 'fusion')]:
        for path in sorted((ROOT / 'runs' / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.md'):
                copy(path, evidence / 'development' / target / path.relative_to(ROOT / 'runs' / folder))
    for path in sorted((ROOT / 'runs' / 'official').rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.log'):
            copy(path, evidence / 'official' / path.relative_to(ROOT / 'runs' / 'official'))
    for name in ('hardware.json', 'acceptance.json', 'frozen_release.json',
                 'reproduction_acceptance.json', 'validation_analysis.json'):
        copy(ROOT / 'runs' / name, evidence / name)
    for name in ('reproduction_record.json', 'embedding-training.log', 'text-training.log'):
        copy(ROOT / 'runs/reproduced-final' / name, evidence / 'retraining' / name)
    embedding_seconds = 0.0
    embedding_models = 0
    ridge_seconds = 0.0
    ridge_models = 0
    ridge_preprocessing = 0.0
    for path in (ROOT / 'runs/embedding').glob('*_training.json'):
        item = json.loads(path.read_text(encoding='utf-8'))
        if 'training_seconds' in item:
            embedding_seconds += item['training_seconds']
            embedding_models += 1
        else:
            ridge_models += 1
            ridge_seconds += item.get('fit_and_validation_seconds', 0)
            ridge_preprocessing = max(ridge_preprocessing, item.get('shared_preprocessing_seconds', 0))
    text_seconds = sum(json.loads(p.read_text(encoding='utf-8'))['total_seconds']
                       for p in (ROOT / 'runs/text').glob('*/training.json'))
    fusion_seconds = sum(json.loads(p.read_text(encoding='utf-8'))['search_seconds']
                         for p in (ROOT / 'runs/fusion').glob('*/search_summary.json'))
    reproduction = json.loads((ROOT / 'runs/reproduced-final/reproduction_record.json').read_text())
    summary = {'embedding_training_configurations': embedding_models,
               'ridge_configurations': ridge_models, 'text_configurations': 5,
               'fusion_grid_configurations': 90, 'fixed_seed_ensemble_comparisons': 1,
               'embedding_training_seconds_summed': embedding_seconds,
               'ridge_fit_and_shared_preprocessing_seconds': ridge_seconds + ridge_preprocessing,
               'text_experiment_seconds_summed': text_seconds,
               'fusion_search_and_fixed_rebuild_seconds_summed': fusion_seconds,
               'fresh_final_training_seconds': reproduction['total_seconds'],
               'training_threads_per_process': 1, 'inference_threads_total': 4, 'gpu_used': False,
               'note': 'Sum of recorded process costs, not orchestration elapsed time. Some text experiments reuse deterministic input-only feature caches. Final reconstruction regenerates features and all weights.'}
    (evidence / 'resource_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
