#!/usr/bin/env python3
"""Conditional R2 rescue for capped, low-agreement R3 SC16 questions.

The script never reads an answer column during leaderboard inference.  On a
labeled development split it freezes one selection policy from a deterministic
policy/confirmation partition.  The same frozen policy can then be applied to
an unlabeled leaderboard or official test file.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import logging
import os
import re
import site
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.qwen_math_final import append_jsonl, extract_integer, load_jsonl_map, sha256_file, stable_seed, vote_candidates


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--r3-adapter", type=Path, required=True)
    parser.add_argument("--r2-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "leaderboard"), required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "final_inference.json")
    parser.add_argument("--frozen-policy", type=Path)
    parser.add_argument("--reuse-r3-raw", type=Path)
    return parser.parse_args()


def read_frame(path: Path, require_labels: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(name).lstrip("\ufeff").strip() for name in frame.columns]
    if not {"id", "question"}.issubset(frame.columns):
        raise ValueError(f"Expected id,question; got {list(frame.columns)}")
    if frame.id.duplicated().any() or frame.id.str.strip().eq("").any() or frame.question.str.strip().eq("").any():
        raise ValueError("Input contains duplicate/empty IDs or empty questions")
    has_labels = "answer" in frame.columns and frame.answer.str.fullmatch(r"-?\d+").all()
    if require_labels and not has_labels:
        raise ValueError("A labeled development CSV is required for policy selection")
    if not require_labels and "answer" in frame.columns and frame.answer.str.strip().ne("").any():
        raise ValueError("Refusing labeled input in leaderboard mode")
    keep = ["id", "question"] + (["answer"] if has_labels else [])
    return frame[keep].copy()


def preload_cuda13() -> None:
    runtime, nvrtc = [], []
    for directory in site.getsitepackages():
        runtime += glob.glob(str(Path(directory) / "nvidia" / "cu13" / "lib" / "libcudart.so.13*"))
        runtime += glob.glob(str(Path(directory) / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.13*"))
        nvrtc += glob.glob(str(Path(directory) / "nvidia" / "cu13" / "lib" / "libnvrtc.so.13*"))
        nvrtc += glob.glob(str(Path(directory) / "nvidia" / "cuda_nvrtc" / "lib" / "libnvrtc.so.13*"))
    runtime, nvrtc = sorted(set(runtime)), sorted(set(nvrtc))
    if not runtime or not nvrtc:
        raise RuntimeError("CUDA 13 libraries are missing; run the notebook setup cell in an A100 runtime")
    ctypes.CDLL(runtime[0], mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(nvrtc[0], mode=ctypes.RTLD_GLOBAL)
    libraries = sorted({str(Path(runtime[0]).parent), str(Path(nvrtc[0]).parent)})
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(libraries + ([previous] if previous else []))
    os.environ["LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]


def validate_adapter(path: Path, expected_base: str) -> Path:
    config = path / "adapter_config.json"
    weight = path / "adapter_model.safetensors"
    if not config.exists() or not weight.exists():
        raise FileNotFoundError(f"Missing adapter_config.json or adapter_model.safetensors under {path}")
    adapter_base = str(json.loads(config.read_text(encoding="utf-8")).get("base_model_name_or_path", "")).rstrip("/")
    if adapter_base != expected_base and not adapter_base.endswith("/Qwen2.5-3B-Instruct"):
        raise ValueError(f"Unexpected adapter base: {adapter_base}")
    return weight


def hit_cap(candidate: dict[str, object], limit: int) -> bool:
    return str(candidate.get("finish_reason", "")) == "length" or int(candidate.get("generated_tokens", 0) or 0) >= limit - 1


def policy_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = [{"name": "r3_only", "r2_top_min": 99, "r2_margin_min": 99}]
    for top in (4, 5, 6, 7):
        for margin in (1, 2, 3):
            rows.append({"name": f"r2_top{top}_margin{margin}", "r2_top_min": top, "r2_margin_min": margin})
    evaluated = []
    for spec in rows:
        use = (
            data.target
            & data.r2_answer.ne(data.r3_answer)
            & data.r2_top.ge(spec["r2_top_min"])
            & data.r2_margin.ge(spec["r2_margin_min"])
        )
        answer = data.r3_answer.where(~use, data.r2_answer)
        for partition in ("policy", "confirm"):
            part = data[data.partition.eq(partition)]
            part_answer = answer.loc[part.index]
            baseline = part.r3_answer.eq(part.answer)
            selected = part_answer.eq(part.answer)
            evaluated.append({
                **spec,
                "partition": partition,
                "rows": len(part),
                "baseline_em": float(baseline.mean()),
                "selected_em": float(selected.mean()),
                "delta": float(selected.mean() - baseline.mean()),
                "changes": int(use.loc[part.index].sum()),
                "fixes": int((~baseline & selected).sum()),
                "breaks": int((baseline & ~selected).sum()),
            })
    return pd.DataFrame(evaluated)


def apply_policy(data: pd.DataFrame, policy: dict[str, object]) -> pd.DataFrame:
    result = data.copy()
    use = (
        result.target
        & result.r2_answer.ne(result.r3_answer)
        & result.r2_top.ge(int(policy["r2_top_min"]))
        & result.r2_margin.ge(int(policy["r2_margin_min"]))
    )
    result["used_r2_rescue"] = use
    result["final_answer"] = result.r3_answer.where(~use, result.r2_answer)
    return result


def main() -> None:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = config["model"]
    generation = config["generation"]
    frame = read_frame(args.input.resolve(), require_labels=args.split == "dev")
    r3_weight = validate_adapter(args.r3_adapter.resolve(), model["id"])
    r2_weight = validate_adapter(args.r2_adapter.resolve(), model["id"])
    output = args.output_dir.resolve()
    candidate_dir, prediction_dir, report_dir, submission_dir = [output / name for name in ("candidates", "predictions", "reports", "submissions")]
    for directory in (candidate_dir, prediction_dir, report_dir, submission_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TOKENIZERS_PARALLELISM": "false"})
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA A100 runtime is required")
    preload_cuda13()
    for stream, descriptor in ((sys.stdout, 1), (sys.stderr, 2)):
        try:
            stream.fileno()
        except Exception:
            stream.fileno = lambda value=descriptor: value

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    for logger_name in ("vllm.v1.engine.input_processor", "vllm.compilation.jit_monitor"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"], local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    torch.backends.cuda.matmul.allow_tf32 = True
    llm = LLM(
        model=model["id"], revision=model["revision"], dtype="bfloat16", tensor_parallel_size=1,
        distributed_executor_backend="uni", gpu_memory_utilization=generation["gpu_memory_utilization"],
        max_model_len=generation["max_model_len"], max_num_seqs=generation["max_num_seqs"],
        max_num_batched_tokens=generation["max_num_batched_tokens"], enable_chunked_prefill=True,
        enable_prefix_caching=True, enable_lora=True, max_loras=2, max_cpu_loras=2, max_lora_rank=64,
        performance_mode="throughput", trust_remote_code=False,
    )
    r3_request, r2_request = LoRARequest("r3_cap_rescue", 1, str(args.r3_adapter.resolve())), LoRARequest("r2_cap_rescue", 2, str(args.r2_adapter.resolve()))

    def prompt(question: str) -> str:
        user = f"{question.strip()}\n\n{config['prompt']['user_suffix']}"
        return tokenizer.apply_chat_template([{"role": "system", "content": config["prompt"]["system"]}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

    def generate(stage_frame: pd.DataFrame, path: Path, label: str, n: int, request, seed_prefix: str) -> dict[str, dict[str, object]]:
        existing = load_jsonl_map(path)
        expected = set(stage_frame.id.astype(str))
        if not set(existing).issubset(expected):
            raise ValueError(f"Unexpected IDs in {path}")
        started = time.time()
        print(f"[{label}] target={len(stage_frame)} done={len(existing)} pending={len(expected)-len(existing)}", flush=True)
        for start in range(0, len(stage_frame), generation["prompt_chunk"]):
            batch = stage_frame.iloc[start:start + generation["prompt_chunk"]]
            batch = batch[~batch.id.astype(str).isin(existing)].copy()
            if batch.empty:
                continue
            prompts = [prompt(question) for question in batch.question]
            if max(len(tokenizer(item, add_special_tokens=False)["input_ids"]) for item in prompts) + generation["base_max_new_tokens"] > generation["max_model_len"]:
                raise ValueError("Prompt is too long for SC16/2048")
            params = [SamplingParams(n=n, temperature=generation["temperature"], top_p=generation["top_p"], max_tokens=generation["base_max_new_tokens"], seed=stable_seed(generation["seed"], seed_prefix, str(qid))) for qid in batch.id.astype(str)]
            outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=False)
            rows = []
            for (_, row), response in zip(batch.iterrows(), outputs):
                candidates = []
                for index, completion in enumerate(response.outputs):
                    answer, source = extract_integer(completion.text)
                    candidates.append({"sample_index": index, "answer": answer, "parse_source": source, "generated_tokens": len(completion.token_ids), "finish_reason": str(completion.finish_reason or ""), "raw_output": completion.text})
                rows.append({"id": str(row.id), "stage": label, "n": n, "adapter": str(request.lora_path), "candidates": candidates})
            append_jsonl(path, rows)
            existing.update({row["id"]: row for row in rows})
            rate = len(existing) / max(time.time() - started, 1e-9)
            print(f"[{label}] {len(existing)}/{len(expected)} eta={(len(expected)-len(existing))/max(rate, 1e-9)/60:.1f}m", flush=True)
        final = load_jsonl_map(path)
        if set(final) != expected:
            raise RuntimeError(f"Incomplete {label}: {len(final)}/{len(expected)}")
        return final

    r3_path = candidate_dir / f"{args.split}_r3_sc16.jsonl"
    if args.reuse_r3_raw:
        r3 = load_jsonl_map(args.reuse_r3_raw)
        if set(r3) != set(frame.id.astype(str)):
            raise ValueError("--reuse-r3-raw does not exactly match this input ID set")
        print(f"[R3 SC16] reusing {args.reuse_r3_raw}", flush=True)
    else:
        r3 = generate(frame, r3_path, "R3-SC16", 16, r3_request, "r3_sc16_cap_rescue_v1")

    rows = []
    for row in frame.itertuples(index=False):
        qid = str(row.id)
        vote = vote_candidates(r3[qid]["candidates"])
        capped = any(hit_cap(candidate, generation["base_max_new_tokens"]) for candidate in r3[qid]["candidates"])
        rows.append({"id": qid, "position": len(rows), "r3_answer": vote["answer"], "r3_top": vote["top_count"], "r3_margin": vote["margin"], "r3_cap": capped, "target": bool(capped and (int(vote["margin"]) <= 1 or int(vote["top_count"]) < 8))})
    decision = pd.DataFrame(rows)
    targets = frame[frame.id.astype(str).isin(set(decision.loc[decision.target, "id"]))].copy()
    r2_path = candidate_dir / f"{args.split}_r2_cap_rescue_sc8.jsonl"
    r2 = generate(targets, r2_path, "R2-RESCUE-SC8", 8, r2_request, "r2_cap_rescue_sc8_v1") if len(targets) else {}
    r2_votes = {qid: vote_candidates(record["candidates"]) for qid, record in r2.items()}
    decision["r2_answer"] = decision.id.map(lambda qid: r2_votes.get(qid, {}).get("answer", ""))
    decision["r2_top"] = decision.id.map(lambda qid: int(r2_votes.get(qid, {}).get("top_count", 0)))
    decision["r2_margin"] = decision.id.map(lambda qid: int(r2_votes.get(qid, {}).get("margin", 0)))

    if args.split == "dev":
        decision = decision.merge(frame[["id", "answer"]], on="id", validate="one_to_one")
        decision["partition"] = decision.id.map(lambda qid: "policy" if stable_seed(20260830, "r3_r2_cap_rescue_partition", qid) % 2 == 0 else "confirm")
        sweep = policy_rows(decision)
        sweep.to_csv(report_dir / "policy_sweep.csv", index=False)
        wide = sweep.pivot(index=["name", "r2_top_min", "r2_margin_min"], columns="partition", values=["baseline_em", "selected_em", "delta", "changes", "fixes", "breaks"]).reset_index()
        wide.columns = ["_".join(str(part) for part in col if part) for col in wide.columns]
        eligible = wide[(wide.delta_confirm >= 0) & (wide.fixes_confirm >= wide.breaks_confirm)]
        if eligible.empty:
            chosen = {"name": "r3_only", "r2_top_min": 99, "r2_margin_min": 99, "reason": "no rescue policy passed confirmation"}
        else:
            best = eligible.sort_values(["delta_confirm", "delta_policy", "breaks_confirm", "changes_confirm"], ascending=[False, False, True, True]).iloc[0]
            chosen = {"name": str(best["name"]), "r2_top_min": int(best.r2_top_min), "r2_margin_min": int(best.r2_margin_min), "reason": "max confirmation delta; then policy delta; fewer breaks/calls"}
        result = apply_policy(decision, chosen)
        metrics = {"rows": len(result), "target_rows": int(result.target.sum()), "r3_em": float(result.r3_answer.eq(result.answer).mean()), "selected_em": float(result.final_answer.eq(result.answer).mean()), "delta": float(result.final_answer.eq(result.answer).mean()-result.r3_answer.eq(result.answer).mean()), "changes": int(result.used_r2_rescue.sum()), "fixes": int((~result.r3_answer.eq(result.answer) & result.final_answer.eq(result.answer)).sum()), "breaks": int((result.r3_answer.eq(result.answer) & ~result.final_answer.eq(result.answer)).sum())}
        policy = {"policy_split_seed": 20260830, "target_rule": "R3 has an SC16 cap and (margin<=1 or top_count<8)", "selected_policy": chosen, "metrics": metrics, "r3_adapter_sha256": sha256_file(r3_weight), "r2_adapter_sha256": sha256_file(r2_weight)}
        (report_dir / "frozen_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
        result.to_csv(prediction_dir / "dev_r3_r2_cap_rescue.csv", index=False)
        print(json.dumps(policy, ensure_ascii=False, indent=2), flush=True)
    else:
        if args.frozen_policy is None or not args.frozen_policy.exists():
            raise ValueError("Leaderboard mode requires --frozen-policy from the dev run")
        frozen = json.loads(args.frozen_policy.read_text(encoding="utf-8"))
        result = apply_policy(decision, frozen["selected_policy"])
        submission = result[["id", "final_answer"]].rename(columns={"final_answer": "answer"})
        if submission.id.tolist() != frame.id.astype(str).tolist() or not submission.answer.str.fullmatch(r"-?\d+").all():
            raise RuntimeError("Submission schema/order validation failed")
        submission_path = submission_dir / "submission_r3cap_r2rescue.csv"
        submission.to_csv(submission_path, index=False)
        result.to_csv(prediction_dir / "leaderboard_r3_r2_cap_rescue.csv", index=False)
        report = {"split": "leaderboard", "rows": len(result), "target_rows": int(result.target.sum()), "r2_rescue_changes": int(result.used_r2_rescue.sum()), "policy": frozen["selected_policy"], "r3_adapter_sha256": sha256_file(r3_weight), "r2_adapter_sha256": sha256_file(r2_weight), "submission": str(submission_path), "submission_sha256": sha256_file(submission_path), "external_api_calls": 0, "answer_lookup": False}
        (report_dir / "leaderboard_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
