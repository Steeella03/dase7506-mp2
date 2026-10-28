"""Freeze inference code/assets and predeclare final scoring before reading test labels."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_experiment import asset_manifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT / 'runs' / 'frozen_release.json'
    if destination.exists():
        raise RuntimeError('A release is already frozen; do not silently overwrite it')
    package = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    immutable = {}
    for relative, expected in package['files'].items():
        if relative == 'code/submission.py':
            continue
        actual = digest(ROOT.parent / relative)
        if actual != expected:
            raise AssertionError(f'Official package file changed: {relative}')
        immutable[relative] = actual
    sources = {name: digest(ROOT / name) for name in
               ('submission.py', 'methods/fusion.py', 'methods/embedding.py',
                'methods/text.py', 'methods/baselines.py', 'features/text_features.py',
                'common.py', 'predict.py', 'score_predictions.py')}
    assets, size = asset_manifest(ROOT / 'artifacts' / 'final')
    fingerprint = hashlib.sha256(json.dumps({'sources': sources, 'assets': assets}, sort_keys=True).encode()).hexdigest()
    validation = json.loads((ROOT / 'runs' / 'official' / 'final-validation' / 'record.json').read_text(encoding='utf-8'))
    result = {'source_version': 'mp2-' + fingerprint[:12], 'fingerprint_sha256': fingerprint,
              'frozen_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'selection_data': 'train and validation only',
              'validation_correct': validation['correct'], 'validation_n': validation['n'],
              'validation_accuracy': validation['accuracy'], 'asset_bytes': size,
              'inference_sources_sha256': sources, 'assets': assets,
              'unchanged_official_files_sha256': immutable,
              'predeclared_final_test_methods': [
                  {'name': 'final-test', 'method': 'submission.py', 'checkpoint': 'artifacts/final'},
                  {'name': 'baseline-majority-test', 'method': 'methods/baselines.py', 'checkpoint': None}],
              'test_use_policy': 'Final scoring/reporting only; do not change the selected method or settings based on test results.'}
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('source_version', 'validation_accuracy', 'asset_bytes',
                                            'frozen_at_utc', 'test_use_policy')}, indent=2))


if __name__ == '__main__':
    main()
