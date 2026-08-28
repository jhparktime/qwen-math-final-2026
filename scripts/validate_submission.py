#!/usr/bin/env python3
"""Validate the exact final submission schema without numeric coercion."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header: {path}")
        return [
            {str(key).lstrip("\ufeff").strip(): value or "" for key, value in row.items()}
            for row in reader
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=2000)
    args = parser.parse_args()
    source = rows(args.input)
    submission = rows(args.submission)
    if len(source) != args.expected_rows or len(submission) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows; input={len(source)} submission={len(submission)}")
    if submission and set(submission[0]) != {"id", "answer"}:
        raise ValueError("Submission columns must be exactly id,answer")
    source_ids = [row.get("id", "") for row in source]
    submission_ids = [row.get("id", "") for row in submission]
    if source_ids != submission_ids:
        raise ValueError("Submission ID values or order differ from the official input")
    if len(set(submission_ids)) != len(submission_ids):
        raise ValueError("Duplicate IDs")
    bad = [row["id"] for row in submission if re.fullmatch(r"-?\d+", row.get("answer", "")) is None]
    if bad:
        raise ValueError(f"Non-integer answers, first IDs: {bad[:5]}")
    print(f"VALID rows={len(submission)} path={args.submission}")


if __name__ == "__main__":
    main()
