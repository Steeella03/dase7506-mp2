"""Root acceptance checks for the frozen selector; no test-reference access."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import check_data, load_inputs, parse_answer, validate_choices
from run_experiment import asset_manifest
import submission


def main():
    checkpoint = ROOT / 'artifacts' / 'final'
    assets, count = asset_manifest(checkpoint)
    check_data()
    bank, arrays = load_inputs('validation')
    # Warm imports, then deny student inference any non-asset data reads.
    original = submission.predict(bank[:3], {'embeddings': arrays['embeddings'][:3]}, checkpoint)
    assert all(type(i) is int and 0 <= i < 4 for i in original)
    real_open = builtins.open
    real_path_open = Path.open
    reads = []

    def allowed_path(path):
        if isinstance(path, int):
            return
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(checkpoint.resolve()):
            raise AssertionError(f'Inference attempted a non-checkpoint read: {resolved}')
        reads.append(str(resolved.relative_to(checkpoint)))

    def guarded_open(path, mode='r', *args, **kwargs):
        if any(letter in mode for letter in 'wax+'):
            raise AssertionError('Inference attempted a file write')
        allowed_path(path)
        return real_open(path, mode, *args, **kwargs)

    def guarded_path_open(path, mode='r', *args, **kwargs):
        if any(letter in mode for letter in 'wax+'):
            raise AssertionError('Inference attempted a file write')
        allowed_path(path)
        return real_path_open(path, mode, *args, **kwargs)

    checks = []
    builtins.open, Path.open = guarded_open, guarded_path_open
    try:
        indices = [23, 0, 111, 5, 23, 31, 300]
        selected_bank = [copy.deepcopy(bank[i]) for i in indices]
        selected_arrays = {'embeddings': arrays['embeddings'][indices].copy()}
        before_bank = copy.deepcopy(selected_bank)
        before_arrays = selected_arrays['embeddings'].copy()
        batch = submission.predict(selected_bank, selected_arrays, checkpoint)
        for row in selected_bank:
            row['id'] = 'unseen-audit-id-' + str(len(row['question']))
        assert submission.predict(selected_bank, selected_arrays, checkpoint) == batch
        checks.append('Unseen and duplicate IDs do not affect predictions')
        singleton = [submission.predict([row], {'embeddings': selected_arrays['embeddings'][i:i+1]}, checkpoint)[0]
                     for i, row in enumerate(selected_bank)]
        assert singleton == batch
        checks.append('Variable bank sizes, repeated rows and singleton batches agree')
        reversed_choices = submission.predict(selected_bank[::-1],
                                               {'embeddings': selected_arrays['embeddings'][::-1]}, checkpoint)
        assert reversed_choices == batch[::-1]
        checks.append('Question-order equivariance')
        order = [2, 0, 3, 1]
        permuted_bank = copy.deepcopy(selected_bank)
        for row in permuted_bank:
            old = row['candidates']
            row['candidates'] = [dict(old[j], index=i) for i, j in enumerate(order)]
        permuted_arrays = {'embeddings': selected_arrays['embeddings'][:, order]}
        permuted_choices = submission.predict(permuted_bank, permuted_arrays, checkpoint)
        for row, choice, reordered, reordered_choice in zip(selected_bank, batch, permuted_bank, permuted_choices):
            assert parse_answer(row['candidates'][choice]['text']) == parse_answer(reordered['candidates'][reordered_choice]['text'])
        checks.append('Candidate permutation preserves the selected numeric answer on acceptance cases')
        assert submission.predict([], {'embeddings': np.empty((0,4,1024), np.float32)}, checkpoint) == []
        checks.append('Empty bank returns an empty choice list')
        assert np.array_equal(before_arrays, selected_arrays['embeddings'])
        for original_row, current_row in zip(before_bank, selected_bank):
            original_row['id'] = current_row['id']
            assert original_row == current_row
        checks.append('Inference does not mutate input texts or embeddings')
        checks.append('Inference reads only the declared checkpoint assets')
    finally:
        builtins.open, Path.open = real_open, real_path_open
    result = {'status': 'passed', 'checks': checks, 'asset_bytes_counted': count,
              'asset_limit_bytes': 16 * 2**20, 'assets': assets,
              'observed_asset_read_paths': sorted(set(reads)),
              'note': 'Functional local checks do not replace instructor audit-bank/reference-CPU evaluation.'}
    assert count <= 16 * 2**20
    out = ROOT / 'runs' / 'acceptance.json'
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
