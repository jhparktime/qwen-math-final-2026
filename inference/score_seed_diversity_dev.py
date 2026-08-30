#!/usr/bin/env python3
"""Score an existing explicit 16-seed R3 dev candidate pool without loading a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.qwen_math_final import load_jsonl_map, sha256_file, vote_candidates

SEED_COUNTS = (1, 4, 8, 12, 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    if len(frame) != 1000 or not {"id", "question", "answer"}.issubset(frame.columns):
        raise ValueError("Expected the fixed 1,000-row labeled dev split")
    if frame.id.duplicated().any() or not frame.answer.str.fullmatch(r"-?\d+").all():
        raise ValueError("Invalid dev split")
    records = load_jsonl_map(args.raw)
    expected = set(frame.id)
    if set(records) != expected:
        missing = sorted(expected - set(records))
        raise RuntimeError(f"Candidate pool incomplete: {len(records)}/{len(expected)}; missing={len(missing)}")

    prediction_dir, report_dir = args.output_dir / "predictions", args.output_dir / "reports"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    sweep, predictions = [], {}
    for count in SEED_COUNTS:
        rows = []
        for item in frame.itertuples(index=False):
            candidates = records[str(item.id)].get("candidates", [])[:count]
            if len(candidates) != count:
                raise RuntimeError(f"{item.id} has fewer than {count} candidates")
            vote = vote_candidates(candidates)
            rows.append({"id": str(item.id), "answer": str(item.answer), "prediction": vote["answer"], "correct": vote["answer"] == str(item.answer), "top_count": vote["top_count"], "margin": vote["margin"], "tie": vote["tie"], "valid_count": vote["valid_count"], "unique_answers": len({value.get("answer") for value in candidates if value.get("answer") is not None}), "parse_failures": sum(value.get("answer") is None for value in candidates)})
        result = pd.DataFrame(rows)
        path = prediction_dir / f"dev_r3_explicit_seedcount_{count}.csv"
        result.to_csv(path, index=False, encoding="utf-8")
        predictions[count] = result
        sweep.append({"seed_count": count, "rows": len(result), "correct": int(result.correct.sum()), "exact_match": float(result.correct.mean()), "tie_rows": int(result.tie.sum()), "parse_failure_candidates": int(result.parse_failures.sum()), "mean_unique_answers": float(result.unique_answers.mean()), "prediction": str(path)})
    sweep_frame = pd.DataFrame(sweep)
    sweep_path = report_dir / "seed_count_sweep.csv"
    sweep_frame.to_csv(sweep_path, index=False, encoding="utf-8")
    one = predictions[1][["id", "correct", "prediction"]].rename(columns={"correct": "correct_1", "prediction": "prediction_1"})
    paired = predictions[16][["id", "correct", "prediction"]].merge(one, on="id")
    paired["changed_vs_1"] = paired.prediction != paired.prediction_1
    paired["fix_vs_1"] = ~paired.correct_1 & paired.correct
    paired["break_vs_1"] = paired.correct_1 & ~paired.correct
    paired_path = prediction_dir / "seed16_vs_seed1_pairs.csv"
    paired.to_csv(paired_path, index=False, encoding="utf-8")
    report = {"run_id": args.output_dir.name, "status": "completed", "objective": "Dev-only scoring of an existing explicit 16-seed R3 candidate pool at 1,4,8,12,16 prefix counts.", "input": str(args.input.resolve()), "input_sha256": sha256_file(args.input.resolve()), "raw": str(args.raw.resolve()), "raw_sha256": sha256_file(args.raw.resolve()), "sweep": sweep, "seed16_vs_seed1": {"changed_rows": int(paired.changed_vs_1.sum()), "fixes": int(paired.fix_vs_1.sum()), "breaks": int(paired.break_vs_1.sum())}, "leaderboard_rows_read": 0, "final_test_rows_read": 0, "external_api_calls": 0}
    report_path = report_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(sweep_frame.to_string(index=False))
    print("[REPORT]", report_path)


if __name__ == "__main__":
    main()
