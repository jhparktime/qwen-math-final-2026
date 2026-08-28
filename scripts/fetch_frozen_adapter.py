#!/usr/bin/env python3
"""Download and verify the frozen adapter before offline inference."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


RELEASE_URL = (
    "https://github.com/jhparktime/qwen-math-final-2026/releases/download/"
    "r2-pro4hint-v1/adapter_model.safetensors"
)
EXPECTED_SHA256 = "e4a22286b3b6a3108c0f2a374012601309abee6511b96b2a108749d432909f11"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, default=Path("artifacts/r2_pro4_hint_adapter"))
    args = parser.parse_args()
    target = args.adapter_dir / "adapter_model.safetensors"
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) == EXPECTED_SHA256:
        print(f"Adapter already verified: {target}")
        return
    temporary = target.with_suffix(".safetensors.partial")
    with urllib.request.urlopen(RELEASE_URL) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    digest = sha256_file(temporary)
    if digest != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Adapter SHA256 mismatch: {digest}")
    temporary.replace(target)
    print(f"Adapter downloaded and verified: {target}")


if __name__ == "__main__":
    main()
