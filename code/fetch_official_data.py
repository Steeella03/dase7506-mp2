"""Fetch only the original official course data, then verify all SHA-256 hashes.

Run once before offline training/evaluation. No weights or other datasets are
downloaded. Existing different files are left unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

UPSTREAM_REVISION = "bb801f4f2e7629d270df765ab12c333b2dc616bb"
UPSTREAM = "https://raw.githubusercontent.com/xudongwu-0/DASE7506-MP2/"

def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def main() -> None:
    root = Path(__file__).absolute().parent.parent
    manifest = json.loads((root / "code/PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    records = sorted((name, value) for name, value in manifest["files"].items()
                     if name.startswith("code/data/"))
    if len(records) != 12:
        raise ValueError("Expected twelve files in the original official data manifest")
    for name, expected in records:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe path in manifest")
        target = root.joinpath(*relative.parts)
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f"Existing data differs; left unchanged: {name}")
            print(f"Verified {name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        request = Request(UPSTREAM + UPSTREAM_REVISION + "/" + name,
                          headers={"User-Agent": "DASE7506-MP2-data-setup"})
        created = False
        try:
            with temporary.open("xb") as output:
                created = True
                with urlopen(request, timeout=120) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
            if digest(temporary) != expected:
                raise ValueError(f"Downloaded data failed SHA-256: {name}")
            if target.exists():
                raise FileExistsError(f"Data appeared during download: {name}")
            temporary.rename(target)
        finally:
            if created:
                temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified {name}")
    print("All 12 official data files verified. Training and evaluation can run offline.")

if __name__ == "__main__":
    main()
