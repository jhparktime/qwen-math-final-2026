#!/usr/bin/env python3
"""Dev-only R3 ablation: does a larger set of explicit sampling seeds help?"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import logging
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
SYSTEM_PROMPT = "You are a helpful assistant that solves math problems step by step."
USER_SUFFIX = "Solve this step by step, then give the final answer as a single integer inside \\boxed{}."
PROMPT_VERSION = "r3_r2continue_original_boxed_v1"
SEED_COUNTS = (1, 4, 8, 12, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Fixed labeled dev CSV only")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def read_dev(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    if not {"id", "question", "answer"}.issubset(frame.columns):
        raise ValueError(f"Expected id,question,answer; got {frame.columns.tolist()}")
    if len(frame) != 1000 or frame["id"].duplicated().any():
        raise ValueError(f"Expected 1,000 unique dev IDs; got {len(frame)}")
    if not frame["answer"].str.fullmatch(r"-?\d+").all():
        raise ValueError("Dev labels must be signed integers")
    return frame[["id", "question", "answer"]].copy()


def validate_adapter(path: Path) -> Path:
    config, weight = path / "adapter_config.json", path / "adapter_model.safetensors"
    if not config.exists() or not weight.exists():
        raise FileNotFoundError(path)
    if str(json.loads(config.read_text(encoding="utf-8")).get("base_model_name_or_path", "")).rstrip("/") != BASE_MODEL:
        raise ValueError("Unexpected adapter base model")
    return weight


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
    if not torch.cuda.is_available():
        raise RuntimeError("Use a fresh A100 runtime")
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

    class WarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not any(fragment in record.getMessage() for fragment in ("deprecated support for supporting different tokenizers", "Triton kernel JIT compilation"))

    for logger_name in ("vllm.v1.engine.input_processor", "vllm.compilation.jit_monitor"):
        logging.getLogger(logger_name).addFilter(WarningFilter())

    frame = read_dev(args.input.resolve())
    adapter, weight = args.adapter.resolve(), validate_adapter(args.adapter.resolve())
    output_dir = args.output_dir.resolve()
    candidate_dir, prediction_dir, report_dir = output_dir / "candidates", output_dir / "predictions", output_dir / "reports"
    for directory in (candidate_dir, prediction_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    torch.backends.cuda.matmul.allow_tf32 = True
    llm = LLM(model=BASE_MODEL, revision=MODEL_REVISION, dtype="bfloat16", tensor_parallel_size=1, distributed_executor_backend="uni", gpu_memory_utilization=.94, max_model_len=8192, max_num_seqs=256, max_num_batched_tokens=65536, enable_chunked_prefill=True, enable_prefix_caching=True, enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=2, performance_mode="throughput")
    lora_request = LoRARequest("r3_seed_sweep", 1, str(adapter))

    def prompt(question: str) -> str:
        return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{question.strip()}\n\n{USER_SUFFIX}"}], tokenize=False, add_generation_prompt=True)

    raw_path = candidate_dir / "dev_r3_explicit16_seed3_pool.jsonl"
    records = load_jsonl_map(raw_path)
    expected = set(frame.id)
    if not set(records).issubset(expected):
        raise ValueError(f"Unexpected IDs in {raw_path}")
    pending = frame.loc[~frame.id.isin(records)].copy()
    print(f"[EXPLICIT SEED POOL] done={len(records)} pending={len(pending)}", flush=True)
    started = time.time()
    for start in range(0, len(pending), 16):
        batch = pending.iloc[start : start + 16]
        prompts, parameters, metadata = [], [], []
        for item in batch.itertuples(index=False):
            text = prompt(str(item.question))
            for sample_index in range(16):
                sampling_seed = stable_seed(args.base_seed, PROMPT_VERSION, f"{item.id}|sample={sample_index}")
                prompts.append(text)
                parameters.append(SamplingParams(n=1, temperature=1.0, top_p=.95, max_tokens=args.max_new_tokens, seed=sampling_seed))
                metadata.append((str(item.id), sample_index, sampling_seed))
        outputs = llm.generate(prompts, parameters, lora_request=lora_request, use_tqdm=False)
        grouped: dict[str, list[dict[str, object]]] = {str(item.id): [] for item in batch.itertuples(index=False)}
        for (qid, sample_index, sampling_seed), response in zip(metadata, outputs):
            if len(response.outputs) != 1:
                raise RuntimeError(f"Unexpected output count for {qid}")
            output = response.outputs[0]
            answer, source = extract_integer(output.text)
            grouped[qid].append({"sample_index": sample_index, "sampling_seed": sampling_seed, "answer": answer, "parse_source": source, "generated_tokens": len(output.token_ids), "finish_reason": str(output.finish_reason or ""), "raw_output": output.text})
        rows = []
        for item in batch.itertuples(index=False):
            candidates = sorted(grouped[str(item.id)], key=lambda value: int(value["sample_index"]))
            if len(candidates) != 16 or len({int(value["sampling_seed"]) for value in candidates}) != 16:
                raise RuntimeError(f"Expected 16 unique seeds for {item.id}")
            rows.append({"id": str(item.id), "mode": "explicit_n1_unique_seed_pool", "base_seed": args.base_seed, "candidates": candidates})
        append_jsonl(raw_path, rows)
        records.update({row["id"]: row for row in rows})
        rate = len(records) / max(time.time() - started, 1e-9)
        print(f"[EXPLICIT SEED POOL] {len(records)}/{len(frame)} eta={(len(frame)-len(records))/max(rate,1e-9)/60:.1f}m", flush=True)
    records = load_jsonl_map(raw_path)
    if set(records) != expected:
        raise RuntimeError(f"Incomplete candidate pool: {len(records)}/{len(expected)}")

    sweep, predictions = [], []
    for count in SEED_COUNTS:
        rows = []
        for item in frame.itertuples(index=False):
            candidates = records[str(item.id)]["candidates"][:count]
            vote = vote_candidates(candidates)
            rows.append({"id": str(item.id), "answer": str(item.answer), "prediction": vote["answer"], "correct": vote["answer"] == str(item.answer), "top_count": vote["top_count"], "margin": vote["margin"], "tie": vote["tie"], "valid_count": vote["valid_count"], "unique_answers": len({value["answer"] for value in candidates if value["answer"] is not None}), "parse_failures": sum(value["answer"] is None for value in candidates)})
        result = pd.DataFrame(rows)
        path = prediction_dir / f"dev_r3_explicit_seedcount_{count}.csv"
        result.to_csv(path, index=False, encoding="utf-8")
        predictions[count] = result
        sweep.append({"seed_count": count, "rows": len(result), "correct": int(result.correct.sum()), "exact_match": float(result.correct.mean()), "tie_rows": int(result.tie.sum()), "parse_failure_candidates": int(result.parse_failures.sum()), "mean_unique_answers": float(result.unique_answers.mean()), "prediction": str(path)})
    sweep_frame = pd.DataFrame(sweep)
    sweep_path = report_dir / "seed_count_sweep.csv"
    sweep_frame.to_csv(sweep_path, index=False, encoding="utf-8")
    baseline = predictions[1][["id", "correct", "prediction"]].rename(columns={"correct": "correct_1", "prediction": "prediction_1"})
    paired = predictions[16][["id", "correct", "prediction"]].merge(baseline, on="id")
    paired["changed_vs_1"] = paired.prediction != paired.prediction_1
    paired["fix_vs_1"] = ~paired.correct_1 & paired.correct
    paired["break_vs_1"] = paired.correct_1 & ~paired.correct
    paired_path = prediction_dir / "seed16_vs_seed1_pairs.csv"
    paired.to_csv(paired_path, index=False, encoding="utf-8")
    report = {"run_id": output_dir.name, "status": "completed", "objective": "Dev-only sweep of 1,4,8,12,16 explicit per-sample seeds from one R3 SC16 candidate pool.", "base_model": BASE_MODEL, "model_revision": MODEL_REVISION, "adapter": str(adapter), "adapter_weight_sha256": sha256_file(weight), "input": str(args.input.resolve()), "input_sha256": sha256_file(args.input.resolve()), "base_seed": args.base_seed, "seed_derivation": "stable_seed(base_seed, prompt_version, id|sample=index), with prefix voting over the first K candidates", "sweep": sweep, "seed16_vs_seed1": {"changed_rows": int(paired.changed_vs_1.sum()), "fixes": int(paired.fix_vs_1.sum()), "breaks": int(paired.break_vs_1.sum())}, "artifacts": {"raw_pool": str(raw_path), "sweep": str(sweep_path), "pairs": str(paired_path)}, "leaderboard_rows_read": 0, "final_test_rows_read": 0, "external_api_calls": 0}
    report_path = report_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(sweep_frame.to_string(index=False))
    print("[REPORT]", report_path)


if __name__ == "__main__":
    main()
