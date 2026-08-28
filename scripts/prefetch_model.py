#!/usr/bin/env python3
"""Cache the exact permitted base-model revision before offline inference."""

from huggingface_hub import snapshot_download


MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


if __name__ == "__main__":
    path = snapshot_download(repo_id=MODEL, revision=REVISION)
    print(f"CACHED model={MODEL} revision={REVISION} path={path}")
