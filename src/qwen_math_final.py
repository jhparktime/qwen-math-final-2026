"""Deterministic parsing, voting, PAL execution, and artifact helpers.

This module intentionally has no model or network dependency. Integer answers
are kept as strings so arbitrarily large values are never coerced to float.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


NUMBER_RE = re.compile(r"(?<!\d)-?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.0+)?")
CODE_RE = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
ALLOWED_IMPORTS = {
    "bisect", "cmath", "collections", "decimal", "fractions", "functools",
    "heapq", "itertools", "math", "operator", "statistics",
}
DENIED_CALLS = {
    "__import__", "breakpoint", "compile", "dir", "eval", "exec", "globals",
    "help", "input", "locals", "open", "vars",
}
DENIED_ATTRS = {
    "fork", "forkpty", "getenv", "kill", "killpg", "listdir", "popen",
    "putenv", "remove", "removedirs", "rmdir", "scandir", "spawn", "system",
    "unlink", "walk",
}


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(global_seed: int, prompt_version: str, identifier: str) -> int:
    payload = f"{global_seed}|{prompt_version}|{identifier}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def normalize_integer(value: object) -> str | None:
    text = str(value or "").strip().replace(",", "").replace("$", "")
    if not re.fullmatch(r"-?\d+(?:\.0+)?", text):
        return None
    sign = "-" if text.startswith("-") else ""
    digits = text.lstrip("-").split(".", 1)[0].lstrip("0") or "0"
    return (sign if digits != "0" else "") + digits


def last_boxed(text: str) -> str | None:
    index = text.rfind("\\boxed")
    if index < 0:
        return None
    opening = text.find("{", index)
    if opening < 0:
        return None
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : position]
    return None


def extract_integer(text: object) -> tuple[str | None, str]:
    """Apply the frozen answer precedence used by the champion."""
    content = str(text or "")
    boxed = last_boxed(content)
    parsed = normalize_integer(boxed) if boxed is not None else None
    if parsed is not None:
        return parsed, "boxed"
    tail = content.rsplit("</think>", 1)[-1]
    markers = re.findall(
        r"(?:final\s+answer|answer|정답)\s*(?:is|:|=)?\s*([^\n.;]*)",
        tail,
        re.IGNORECASE,
    )
    if markers:
        numbers = NUMBER_RE.findall(markers[-1])
        if numbers:
            parsed = normalize_integer(numbers[0].replace(" ", ""))
            if parsed is not None:
                return parsed, "final_marker"
    for number in reversed(NUMBER_RE.findall(tail)):
        parsed = normalize_integer(number.replace(" ", ""))
        if parsed is not None:
            return parsed, "last_integer_fallback"
    return None, "parse_failure"


def terminal_boxed_integer(text: object) -> str | None:
    """Return an integer only when the last box terminates the response."""
    content = str(text or "").rstrip()
    index = content.rfind("\\boxed")
    boxed = last_boxed(content)
    parsed = normalize_integer(boxed) if boxed is not None else None
    if index < 0 or parsed is None:
        return None
    opening = content.find("{", index)
    depth = 0
    closing = None
    for position in range(opening, len(content)):
        if content[position] == "{":
            depth += 1
        elif content[position] == "}":
            depth -= 1
            if depth == 0:
                closing = position
                break
    if closing is None:
        return None
    tail = content[closing + 1 :].strip()
    tail = re.sub(r"^(?:\\\)|\\\]|\$)+", "", tail).strip()
    return parsed if re.fullmatch(r"[.!]*", tail) else None


def vote_candidates(candidates: Iterable[Mapping[str, object]]) -> dict[str, object]:
    ordered_candidates = sorted(candidates, key=lambda item: int(item.get("sample_index", 0)))
    ordered = [normalize_integer(item.get("answer")) for item in ordered_candidates]
    valid = [answer for answer in ordered if answer is not None]
    if not valid:
        return {
            "answer": "0", "top_count": 0, "second_count": 0,
            "margin": 0, "tie": False, "valid_count": 0,
        }
    counts = Counter(valid)
    top_count = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == top_count}
    answer = next(answer for answer in valid if answer in tied)
    ranked = sorted(counts.values(), reverse=True)
    second = ranked[1] if len(ranked) > 1 else 0
    return {
        "answer": answer,
        "top_count": top_count,
        "second_count": second,
        "margin": top_count - second,
        "tie": len(tied) > 1,
        "valid_count": len(valid),
    }


def load_jsonl_map(path: str | Path) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    source = Path(path)
    if not source.exists():
        return latest
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("id") is not None:
                latest[str(row["id"])] = row
    return latest


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def extract_python(text: object) -> str | None:
    blocks = CODE_RE.findall(str(text or ""))
    return blocks[-1].strip() if blocks else None


def audit_python(source: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax:{exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in ALLOWED_IMPORTS for alias in node.names):
                return False, "import_denied"
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS:
                return False, "importfrom_denied"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in DENIED_CALLS:
                return False, f"call_denied:{node.func.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in DENIED_ATTRS:
                return False, f"attribute_denied:{node.attr}"
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            return False, "dunder_name_denied"
    return True, "ok"


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def run_python(source: str | None) -> dict[str, object]:
    if not source:
        return {"ok": False, "reason": "no_code", "answer": None}
    allowed, reason = audit_python(source)
    if not allowed:
        return {"ok": False, "reason": reason, "answer": None}
    with tempfile.TemporaryDirectory(prefix="pal_") as directory:
        path = Path(directory) / "solution.py"
        path.write_text(source, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                env={"PATH": os.environ.get("PATH", "")},
                capture_output=True,
                text=True,
                timeout=4,
                preexec_fn=_limits,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "timeout", "answer": None}
    lines = [line.strip() for line in result.stdout[-4096:].splitlines() if line.strip()]
    answer = normalize_integer(lines[0]) if len(lines) == 1 else None
    ok = result.returncode == 0 and answer is not None
    return {"ok": ok, "reason": "ok" if ok else f"exit_{result.returncode}", "answer": answer}
