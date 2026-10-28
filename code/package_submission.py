"""Package the frozen MP2 project; final acceptance remains the coordinator's job.

Run from any directory with ``python /path/to/code/package_submission.py``.
The default outputs are project-root ``deliverables/MP2_submission.zip`` and
``deliverables/MP2_submission_manifest.json``. Existing outputs are never
overwritten; use ``--name MP2_submission_v2`` to preserve an earlier package.
No training, prediction, scoring, remote upload or artifact validation runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


ARCHIVE_PREFIX = 'MiniProject2'
EXCLUDED_DIRECTORIES = frozenset({
    '.venv', '.environment', '.git', '__pycache__', 'deliverables',
})
EXCLUDED_RUN_SUFFIXES = frozenset({'.npz', '.npy'})
CHUNK_BYTES = 1024 * 1024


def _has_redirect(path: Path) -> bool:
    """Reject symlinks and Windows junction/reparse points before reading."""
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, 'st_file_attributes', 0)
        & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    )


def _relative_file(path: Path, root: Path) -> str:
    # Traversal already rejects every reparse point. Avoid GetFinalPathNameByHandle,
    # which restricted Windows hosts may deny even for readable project files.
    resolved = path.absolute()
    if not resolved.is_relative_to(root):
        raise ValueError(f'Source resolves outside the project: {path.name}')
    relative = path.relative_to(root).as_posix()
    parts = PurePosixPath(relative).parts
    if not parts or any(part in ('', '.', '..') for part in parts):
        raise ValueError(f'Unsafe relative source path: {relative}')
    if PurePosixPath(relative).is_absolute() or '\\' in relative:
        raise ValueError(f'Unsafe archive source path: {relative}')
    return relative


def _raise_walk_error(error: OSError):
    raise error


def enumerate_sources(root: Path) -> list[tuple[str, Path]]:
    """Enumerate regular project files only, without following redirects."""
    result = []
    for directory, directory_names, filenames in os.walk(root, followlinks=False, onerror=_raise_walk_error):
        current = Path(directory)
        retained = []
        for name in sorted(directory_names):
            if name.casefold() in EXCLUDED_DIRECTORIES:
                continue
            child = current / name
            if _has_redirect(child):
                raise ValueError(f'Redirected source directory is not packaged: {child.relative_to(root)}')
            retained.append(name)
        directory_names[:] = retained
        for name in sorted(filenames):
            path = current / name
            relative = _relative_file(path, root)
            parts = PurePosixPath(relative).parts
            if any(part.casefold() in EXCLUDED_DIRECTORIES for part in parts):
                continue
            if 'runs' in [part.casefold() for part in parts[:-1]] and path.suffix.casefold() in EXCLUDED_RUN_SUFFIXES:
                continue
            if _has_redirect(path) or not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f'Nonregular source file is not packaged: {relative}')
            result.append((relative, path))
    return sorted(result, key=lambda item: item[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def package_project(root: Path, name: str = 'MP2_submission') -> dict:
    """Write one source ZIP and its external checksum manifest without overwrite."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Package name must contain only letters, digits, underscores and hyphens')
    if _has_redirect(root):
        raise ValueError('The project root must not be a redirected directory')
    root = root.absolute()
    if not root.is_dir():
        raise ValueError('Project root must be a directory')
    if os.name == 'nt' and root.drive.upper() == 'C:':
        raise ValueError('This project must be packaged from D:, not from the system C: drive')
    # Presence checks prevent an accidental pre-freeze archive. They do not
    # judge experiment correctness, report quality or checkpoint eligibility.
    required = [root / 'README_STUDENT.md', root / 'report', root / 'code/artifacts/final']
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f'Required packaging input is missing: {path.relative_to(root)}')

    output_directory = root / 'deliverables'
    if output_directory.exists() and (_has_redirect(output_directory) or not output_directory.is_dir()):
        raise ValueError('deliverables must be a normal directory inside the project')
    archive_path = output_directory / f'{name}.zip'
    manifest_path = output_directory / f'{name}_manifest.json'
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError('Package outputs already exist; use --name with a new version name')

    sources = enumerate_sources(root)
    if not sources:
        raise ValueError('The project has no source files to package')
    output_directory.mkdir(exist_ok=True)
    records = []
    created_files = []
    try:
        # Exclusive creation guarantees that a preexisting deliverable cannot
        # be replaced, even if another process creates it after the check.
        with archive_path.open('xb') as archive_stream:
            created_files.append(archive_path)
            with zipfile.ZipFile(archive_stream, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as archive:
                for relative, path in sources:
                    _relative_file(path, root)
                    if _has_redirect(path):
                        raise ValueError(f'Source became redirected: {relative}')
                    before = path.stat()
                    digest = hashlib.sha256()
                    size = 0
                    archive_name = f'{ARCHIVE_PREFIX}/{relative}'
                    # The public streaming API supplies a fixed ZIP timestamp,
                    # sorted names and bounded memory, making repeated frozen
                    # source packages reproducible on the same Python runtime.
                    with path.open('rb') as source, archive.open(archive_name, 'w', force_zip64=True) as destination:
                        for chunk in iter(lambda: source.read(CHUNK_BYTES), b''):
                            destination.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                        after.st_size, after.st_mtime_ns, after.st_ino
                    ) or size != before.st_size:
                        raise RuntimeError(f'Source changed during packaging: {relative}')
                    records.append({
                        'path': relative,
                        'archive_path': archive_name,
                        'size_bytes': size,
                        'sha256': digest.hexdigest(),
                    })

        manifest = {
            'schema_version': 1,
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'archive_prefix': ARCHIVE_PREFIX + '/',
            'archive_path': archive_path.relative_to(root).as_posix(),
            'archive_sha256': _sha256(archive_path),
            'archive_size_bytes': archive_path.stat().st_size,
            'source_file_count': len(records),
            'source_size_bytes': sum(record['size_bytes'] for record in records),
            'exclusions': {
                'directory_names': sorted(EXCLUDED_DIRECTORIES),
                'runs_file_suffixes': sorted(EXCLUDED_RUN_SUFFIXES),
                'redirected_paths': 'rejected',
            },
            'files': records,
        }
        with manifest_path.open('x', encoding='utf-8', newline='\n') as stream:
            created_files.append(manifest_path)
            stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        return manifest
    except BaseException:
        # Remove only outputs exclusively created by this invocation; never
        # touch a source file or a deliverable that existed before this run.
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='MP2_submission', help='New output basename; existing files are never overwritten')
    args = parser.parse_args()
    root = Path(__file__).absolute().parents[1]
    manifest = package_project(root, args.name)
    print(json.dumps({key: value for key, value in manifest.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
