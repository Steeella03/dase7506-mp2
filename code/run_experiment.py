"""Run official prediction and scoring in fresh processes, with external timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent


def asset_manifest(checkpoint):
    if checkpoint is None:
        return [], 0
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    paths = sorted(checkpoint.rglob('*')) if checkpoint.is_dir() else [checkpoint]
    records = []
    for path in paths:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        size = len(raw)
        if path.suffix == '.npz':
            with zipfile.ZipFile(path) as bundle:
                size = max(size, sum(info.file_size for info in bundle.infolist()))
        records.append({'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
                        'bytes_on_disk': len(raw), 'uncompressed_bytes_counted': size,
                        'sha256': hashlib.sha256(raw).hexdigest()})
    return records, sum(row['uncompressed_bytes_counted'] for row in records)


def execute(command, environment):
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=180)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f'Command failed: {command}\n{result.stdout}\n{result.stderr}')
    return elapsed, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--method', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--final-test', action='store_true')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    if args.split == 'test' and not args.final_test:
        parser.error('Freeze your method first, then explicitly pass --final-test')
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    run_dir = ROOT / 'runs' / 'official' / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '4'
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONNOUSERSITE='1')
    assets, size = asset_manifest(args.checkpoint.resolve() if args.checkpoint else None)
    if size > 16 * 2**20:
        raise ValueError(f'Inference assets exceed 16 MiB: {size}')
    timings = []
    score = None
    for attempt in range(args.repeat):
        predictions = run_dir / f'{args.split}-predictions.json'
        score_path = run_dir / f'{args.split}-score.json'
        predict_command = [sys.executable, '-B', str(ROOT / 'predict.py'),
                           '--method', str(args.method.resolve()), '--split', args.split,
                           '--output', str(predictions)]
        if args.checkpoint:
            predict_command += ['--checkpoint', str(args.checkpoint.resolve())]
        score_command = [sys.executable, '-B', str(ROOT / 'score_predictions.py'),
                         '--split', args.split, '--predictions', str(predictions),
                         '--output', str(score_path)]
        pred_time, pred_out, pred_err = execute(predict_command, env)
        score_time, score_out, score_err = execute(score_command, env)
        timings.append({'prediction_process_seconds': pred_time, 'scoring_process_seconds': score_time,
                        'complete_seconds': pred_time + score_time})
        (run_dir / f'process-{attempt}.log').write_text(pred_out + pred_err + score_out + score_err,
                                                     encoding='utf-8')
        current = json.loads(score_path.read_text(encoding='utf-8'))
        if score is not None and current['correct'] != score['correct']:
            raise ValueError('Repeated evaluation gave different accuracy')
        score = current
    record = {'name': args.name, 'method': str(args.method), 'checkpoint': str(args.checkpoint),
              'split': args.split, 'accuracy': score['accuracy'], 'correct': score['correct'],
              'n': score['n'], 'asset_bytes': size, 'assets': assets, 'timings': timings,
              'threads': 4, 'python': platform.python_version(), 'platform': platform.platform(),
              'cpu': platform.processor(), 'local_time_limit_pass': max(t['complete_seconds'] for t in timings) <= 30,
              'reference_cpu_eligibility': 'Requires instructor reference-machine evaluation',
              'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (run_dir / 'record.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    with (ROOT / 'runs' / 'experiments.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')
    print(json.dumps({k: record[k] for k in ('name', 'split', 'accuracy', 'correct', 'n',
                                           'asset_bytes', 'timings', 'local_time_limit_pass')}, indent=2))


if __name__ == '__main__':
    main()
