#!/usr/bin/env python3
"""Download and verify the frozen adapter before offline inference."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


RELEASE_ROOT = (
    "https://github.com/jhparktime/qwen-math-final-2026/releases/download/"
    "r3-r2continue-v1"
)
ASSETS = {
    "adapter_config.json": "6b3c883bb8bbf11d2f557cdca0131aebb08cba71af55f787b7547d1013423e93",
    "adapter_model.safetensors": "3b13039776a5e77567d8a0e3b8425b762bae747d5d195cd82966a3a87597633f",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, default=Path("artifacts/r3_r2continue_adapter"))
    args = parser.parse_args()
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_sha256 in ASSETS.items():
        target = args.adapter_dir / name
        if target.exists() and sha256_file(target) == expected_sha256:
            print(f"Verified: {target}")
            continue
        temporary = target.with_name(target.name + ".partial")
        with urllib.request.urlopen(f"{RELEASE_ROOT}/{name}") as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        digest = sha256_file(temporary)
        if digest != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"{name} SHA256 mismatch: {digest}")
        temporary.replace(target)
        print(f"Downloaded and verified: {target}")


if __name__ == "__main__":
    main()
