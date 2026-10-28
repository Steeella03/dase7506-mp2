"""Retrain every final component from supplied train/validation data.

This builder never opens an existing checkpoint. It writes fresh model assets,
then assembles the frozen inference configuration without retuning its weights.
Official scoring and independent acceptance are left to the root evaluator.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]


def run_training(command, working_directory, environment, log_path):
    started = time.perf_counter()
    process = subprocess.run(command, cwd=working_directory, env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace')
    log_path.write_text(process.stdout, encoding='utf-8')
    if process.returncode:
        raise RuntimeError(f'Training failed; see {log_path}\n{process.stdout[-2500:]}')
    return {'command': command, 'seconds': time.perf_counter() - started,
            'log': str(log_path), 'returncode': process.returncode}


def assemble_bundle(output, recipe):
    models = output / 'bundle' / 'models'
    models.mkdir(parents=True)
    fusion = recipe['fusion']
    component_settings = {'transform': fusion['transform'], 'temperature': fusion['temperature']}
    linear_name = fusion['linear_name'] + '.npz'
    linear_target = '0-embedding-' + linear_name
    shutil.copyfile(output / 'embedding' / linear_name, models / linear_target)
    components = [{'method': 'embedding', 'checkpoint': 'models/' + linear_target,
                   'weight': fusion['linear_weight'], **component_settings}]
    members = []
    for index, name in enumerate(fusion['mlp_names']):
        target = f'1-{index}-embedding-{name}.npz'
        shutil.copyfile(output / 'embedding' / f'{name}.npz', models / target)
        members.append('models/' + target)
    components.append({'method': 'embedding', 'checkpoints': members,
                       'weight': fusion['mlp_ensemble_weight'], **component_settings})
    text_target = '2-text-model.npz'
    shutil.copyfile(output / 'text' / 'model.npz', models / text_target)
    components.append({'method': 'text', 'checkpoint': 'models/' + text_target,
                       'weight': fusion['text_weight'], **component_settings})
    config = {k: fusion[k] for k in ('schema_version', 'pooling', 'pool_weight', 'vote_weight')}
    config['components'] = components
    (output / 'bundle' / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/reproduced-final'))
    parser.add_argument('--recipe', type=Path, default=Path(__file__).with_name('final_recipe.json'))
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error('Use the project Python 3.12 environment')
    recipe = json.loads(args.recipe.read_text(encoding='utf8'))
    output = args.output if args.output.is_absolute() else CODE / args.output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Choose an empty --output directory so no prior features or checkpoints can be reused')
    output.mkdir(parents=True, exist_ok=True)
    working = output / 'working'
    # train_text.py uses this relative cache. An isolated working directory
    # forces it to regenerate features from the supplied input data.
    (working / 'runs' / 'text').mkdir(parents=True)
    temporary = working / 'tmp'
    temporary.mkdir()
    environment = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '1'
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUTF8='1',
                       TEMP=str(temporary), TMP=str(temporary))
    config_path = output / 'embedding-configs.json'
    config_path.write_text(json.dumps(recipe['embedding_configs'], indent=2) + '\n', encoding='utf8')
    started = time.perf_counter()
    print('Training frozen linear and all five MLP seeds from scratch...', flush=True)
    records = [run_training([sys.executable, '-B', str(CODE / 'training' / 'train_embedding.py'),
                             '--config', str(config_path), '--output', str(output / 'embedding')],
                            working, environment, output / 'embedding-training.log')]
    text_command = [sys.executable, '-B', str(CODE / 'training' / 'train_text.py'),
                    '--output', str(output / 'text')]
    for name, value in recipe['text_arguments'].items():
        text_command += ['--' + name.replace('_', '-'), str(value)]
    print('Training frozen text model with newly computed features...', flush=True)
    records.append(run_training(text_command, working, environment, output / 'text-training.log'))
    assemble_bundle(output, recipe)
    assets = [{'path': str(path.relative_to(output)), 'bytes': path.stat().st_size,
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for path in sorted((output / 'bundle').rglob('*')) if path.is_file()]
    record = {'recipe': recipe, 'python': platform.python_version(), 'threads_per_library': 1,
              'total_seconds': time.perf_counter() - started, 'training_processes': records,
              'bundle_assets': assets, 'bundle': str(output / 'bundle'),
              'note': 'Fresh training only. Test references were never opened. No official scoring or acceptance performed by this script.'}
    (output / 'reproduction_record.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
    print(f'Fresh bundle written to {output / "bundle"}', flush=True)


if __name__ == '__main__':
    main()
