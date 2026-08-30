#!/usr/bin/env python3
"""Dev-only direct comparison: final-style vLLM n=16 vs explicit 16 seeds."""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import site
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.qwen_math_final import append_jsonl, extract_integer, load_jsonl_map, sha256_file, stable_seed, vote_candidates  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
PROMPT_VERSION = "r3_r2continue_original_boxed_v1"
SYSTEM_PROMPT = "You are a helpful assistant that solves math problems step by step."
USER_SUFFIX = "Solve this step by step, then give the final answer as a single integer inside \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--explicit-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=3)
    return parser.parse_args()


def read_dev(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    if len(frame) != 1000 or not {"id", "question", "answer"}.issubset(frame.columns):
        raise ValueError("Expected the fixed 1,000-row labeled dev split")
    if frame.id.duplicated().any() or not frame.answer.str.fullmatch(r"-?\d+").all():
        raise ValueError("Invalid dev split")
    return frame[["id", "question", "answer"]].copy()


def preload_cuda13() -> None:
    runtime, nvrtc = [], []
    for directory in set(site.getsitepackages() + [site.getusersitepackages()] + sys.path):
        if directory:
            root = Path(directory)
            runtime += glob.glob(str(root / "nvidia" / "**" / "libcudart.so.13*"), recursive=True)
            nvrtc += glob.glob(str(root / "nvidia" / "**" / "libnvrtc.so.13*"), recursive=True)
    runtime, nvrtc = sorted(set(runtime)), sorted(set(nvrtc))
    if not runtime or not nvrtc:
        raise RuntimeError("CUDA 13 libraries missing; run setup and restart once")
    ctypes.CDLL(runtime[0], mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(nvrtc[0], mode=ctypes.RTLD_GLOBAL)
    libraries = sorted({str(Path(runtime[0]).parent), str(Path(nvrtc[0]).parent)})
    old = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(libraries + ([old] if old else []))
    os.environ["LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]


def main() -> None:
    args = parse_args()
    frame = read_dev(args.input.resolve())
    explicit = load_jsonl_map(args.explicit_raw.resolve())
    expected = set(frame.id)
    if set(explicit) != expected:
        raise RuntimeError(f"Explicit pool incomplete: {len(explicit)}/{len(expected)}")
    for qid in expected:
        if len(explicit[qid].get("candidates", [])) != 16:
            raise RuntimeError(f"Explicit pool has wrong candidate count: {qid}")

    adapter = args.adapter.resolve()
    weight = adapter / "adapter_model.safetensors"
    if not weight.exists() or not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(adapter)
    output_dir = args.output_dir.resolve()
    candidate_dir, prediction_dir, report_dir = output_dir / "candidates", output_dir / "predictions", output_dir / "reports"
    for directory in (candidate_dir, prediction_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    raw_path = candidate_dir / "dev_r3_vllm_n16_seed3.jsonl"
    vllm_records = load_jsonl_map(raw_path)
    if not set(vllm_records).issubset(expected):
        raise ValueError(f"Unexpected IDs in {raw_path}")

    if set(vllm_records) != expected:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to generate missing final-style candidates")
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TOKENIZERS_PARALLELISM": "false"})
        preload_cuda13()
        for stream, descriptor in ((sys.stdout, 1), (sys.stderr, 2)):
            try:
                stream.fileno()
            except Exception:
                stream.fileno = lambda value=descriptor: value
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        torch.backends.cuda.matmul.allow_tf32 = True
        llm = LLM(model=BASE_MODEL, revision=MODEL_REVISION, dtype="bfloat16", tensor_parallel_size=1, distributed_executor_backend="uni", gpu_memory_utilization=.94, max_model_len=8192, max_num_seqs=256, max_num_batched_tokens=65536, enable_chunked_prefill=True, enable_prefix_caching=True, enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=2, performance_mode="throughput")
        request = LoRARequest("r3_final_style_n16", 1, str(adapter))

        def prompt(question: str) -> str:
            return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{question.strip()}\n\n{USER_SUFFIX}"}], tokenize=False, add_generation_prompt=True)

        pending = frame.loc[~frame.id.isin(vllm_records)].copy()
        print(f"[FINAL-STYLE vLLM n16] done={len(vllm_records)} pending={len(pending)}", flush=True)
        started = time.time()
        for start in range(0, len(pending), 16):
            batch = pending.iloc[start : start + 16]
            prompts = [prompt(question) for question in batch.question]
            parameters = [SamplingParams(n=16, temperature=1.0, top_p=.95, max_tokens=2048, seed=stable_seed(args.base_seed, PROMPT_VERSION, str(qid))) for qid in batch.id]
            responses = llm.generate(prompts, parameters, lora_request=request, use_tqdm=False)
            rows = []
            for (_, item), response in zip(batch.iterrows(), responses):
                request_seed = stable_seed(args.base_seed, PROMPT_VERSION, str(item.id))
                candidates = []
                for sample_index, output in enumerate(response.outputs):
                    answer, source = extract_integer(output.text)
                    candidates.append({"sample_index": sample_index, "request_seed": request_seed, "answer": answer, "parse_source": source, "generated_tokens": len(output.token_ids), "finish_reason": str(output.finish_reason or ""), "raw_output": output.text})
                if len(candidates) != 16:
                    raise RuntimeError(f"Unexpected n for {item.id}")
                rows.append({"id": str(item.id), "mode": "final_style_vllm_n16", "candidates": candidates})
            append_jsonl(raw_path, rows)
            vllm_records.update({row["id"]: row for row in rows})
            rate = len(vllm_records) / max(time.time() - started, 1e-9)
            print(f"[FINAL-STYLE vLLM n16] {len(vllm_records)}/{len(frame)} eta={(len(frame)-len(vllm_records))/max(rate,1e-9)/60:.1f}m", flush=True)
    vllm_records = load_jsonl_map(raw_path)
    if set(vllm_records) != expected:
        raise RuntimeError(f"vLLM baseline incomplete: {len(vllm_records)}/{len(expected)}")

    def score(label: str, records: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, dict[str, object]]:
        rows = []
        for item in frame.itertuples(index=False):
            choices = records[str(item.id)]["candidates"]
            vote = vote_candidates(choices)
            rows.append({"id": str(item.id), "answer": str(item.answer), "prediction": vote["answer"], "correct": vote["answer"] == str(item.answer), "top_count": vote["top_count"], "margin": vote["margin"], "tie": vote["tie"], "valid_count": vote["valid_count"], "parse_failures": sum(choice.get("answer") is None for choice in choices)})
        result = pd.DataFrame(rows)
        path = prediction_dir / f"{label}_dev_sc16.csv"
        result.to_csv(path, index=False, encoding="utf-8")
        return result, {"label": label, "rows": len(result), "correct": int(result.correct.sum()), "exact_match": float(result.correct.mean()), "tie_rows": int(result.tie.sum()), "parse_failure_candidates": int(result.parse_failures.sum()), "prediction": str(path)}

    vllm_result, vllm_metrics = score("final_style_vllm_n16_seed3", vllm_records)
    explicit_result, explicit_metrics = score("explicit16_unique_seed3", explicit)
    paired = vllm_result[["id", "correct", "prediction"]].merge(explicit_result[["id", "correct", "prediction"]], on="id", suffixes=("_vllm", "_explicit"))
    paired["changed"] = paired.prediction_vllm != paired.prediction_explicit
    paired["fix"] = ~paired.correct_vllm & paired.correct_explicit
    paired["break"] = paired.correct_vllm & ~paired.correct_explicit
    paired_path = prediction_dir / "vllm_n16_vs_explicit16_pairs.csv"
    paired.to_csv(paired_path, index=False, encoding="utf-8")
    report = {
        "run_id": output_dir.name,
        "status": "completed",
        "objective": "Direct dev-only comparison of final-style vLLM n=16 sampling and explicit unique 16-seed sampling for frozen R3.",
        "base_model": BASE_MODEL,
        "model_revision": MODEL_REVISION,
        "adapter": str(adapter),
        "adapter_weight_sha256": sha256_file(weight),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input.resolve()),
        "base_seed": args.base_seed,
        "final_style_vllm_n16": vllm_metrics,
        "explicit16_unique_seed": explicit_metrics,
        "delta_explicit_minus_vllm": explicit_metrics["exact_match"] - vllm_metrics["exact_match"],
        "changed_rows": int(paired["changed"].sum()),
        "fixes": int(paired["fix"].sum()),
        "breaks": int(paired["break"].sum()),
        "artifacts": {"vllm_raw": str(raw_path), "explicit_raw": str(args.explicit_raw.resolve()), "pairs": str(paired_path)},
        "leaderboard_rows_read": 0,
        "final_test_rows_read": 0,
        "external_api_calls": 0,
    }
    report_path = report_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("final_style_vllm_n16", "explicit16_unique_seed", "delta_explicit_minus_vllm", "changed_rows", "fixes", "breaks")}, ensure_ascii=False, indent=2))
    print("[REPORT]", report_path)


if __name__ == "__main__":
    main()
