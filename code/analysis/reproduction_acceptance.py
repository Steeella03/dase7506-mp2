"""Verify the frozen release and the independent training reconstruction."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((ROOT / 'runs/frozen_release.json').read_text(encoding='utf-8'))
    for name, expected in frozen['inference_sources_sha256'].items():
        assert sha(ROOT / name) == expected, f'Frozen source changed: {name}'
    for row in frozen['assets']:
        assert sha(ROOT / row['path']) == row['sha256'], f'Frozen asset changed: {row["path"]}'
    final = ROOT / 'artifacts/final'
    rebuilt = ROOT / 'runs/reproduced-final/bundle'
    assert json.loads((final / 'config.json').read_text()) == json.loads((rebuilt / 'config.json').read_text())
    models = []
    for path in sorted((final / 'models').glob('*.npz')):
        actual = rebuilt / 'models' / path.name
        assert sha(path) == sha(actual), f'Retraining differs: {path.name}'
        models.append({'name': path.name, 'sha256': sha(path), 'byte_identical': True})
    original_predictions = json.loads((ROOT / 'runs/official/final-validation/validation-predictions.json').read_text())
    rebuilt_predictions = json.loads((ROOT / 'runs/official/reproduced-validation/validation-predictions.json').read_text())
    assert original_predictions['selected'] == rebuilt_predictions['selected']
    assert original_predictions['ids'] == rebuilt_predictions['ids']
    training = json.loads((ROOT / 'runs/reproduced-final/reproduction_record.json').read_text())
    result = {'status': 'passed', 'source_version': frozen['source_version'],
              'frozen_sources_and_assets_unchanged': True, 'configuration_semantically_identical': True,
              'all_seven_weight_files_byte_identical': len(models) == 7,
              'models': models, 'all_validation_predictions_identical': True,
              'fresh_retraining_seconds': training['total_seconds'],
              'note': 'Configuration JSON field order differs; decoded settings and all weights match.'}
    (ROOT / 'runs/reproduction_acceptance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'models'}, indent=2))


if __name__ == '__main__':
    main()
