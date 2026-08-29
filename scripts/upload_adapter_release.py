#!/usr/bin/env python3
"""Create/update the private GitHub Release containing the frozen adapter.

This packaging utility is not used by inference. It relies on an interactive
``gh auth login`` session, so no GitHub credential is stored in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


REPOSITORY = "jhparktime/qwen-math-final-2026"
DEFAULT_TAG = "r2-pro4hint-v1"
DEFAULT_EXPECTED_SHA256 = "e4a22286b3b6a3108c0f2a374012601309abee6511b96b2a108749d432909f11"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--title", default="Frozen R2 Pro4-hint adapter")
    parser.add_argument(
        "--expected-sha256",
        default=DEFAULT_EXPECTED_SHA256,
        help="Use an empty string to accept the locally computed digest when packaging a new adapter.",
    )
    args = parser.parse_args()
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI (gh) is required; install it and run gh auth login first")
    weight = args.adapter_dir / "adapter_model.safetensors"
    config = args.adapter_dir / "adapter_config.json"
    if not weight.exists() or not config.exists():
        raise FileNotFoundError("adapter_config.json and adapter_model.safetensors are both required")
    digest = sha256_file(weight)
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ValueError(f"Unexpected adapter SHA256: {digest}")
    check = subprocess.run(
        ["gh", "release", "view", args.tag, "--repo", REPOSITORY],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if check.returncode:
        run(
            [
                "gh", "release", "create", args.tag, "--repo", REPOSITORY,
                "--title", args.title, "--notes",
                f"Frozen LoRA adapter. SHA-256: {digest}",
            ]
        )
    run(
        [
            "gh", "release", "upload", args.tag, str(weight), str(config), "--repo", REPOSITORY,
            "--clobber",
        ]
    )
    run(["gh", "release", "view", args.tag, "--repo", REPOSITORY, "--json", "url,assets"])


if __name__ == "__main__":
    main()
