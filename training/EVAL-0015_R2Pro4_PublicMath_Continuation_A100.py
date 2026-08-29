#!/usr/bin/env python3
"""Frozen tune-SC8 and winner dev-SC16 evaluation for RFT-0015.

The script evaluates only organizer-held labeled splits. It never reads public
leaderboard or private-test questions, and it uses identical prompt, seed,
parser, tie-break, output limit, and base revision for every adapter.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import logging
import os
import site
import sys
import time
import unicodedata
import re
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.qwen_math_final import append_jsonl, extract_integer, load_jsonl_map, sha256_file, stable_seed, vote_candidates

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
PARENT_RUN_ID = "RFT-0004B-r2-pro4-hint-lowdrift-lora"
CANDIDATE_RUN_ID = "RFT-0015-r2pro4-external14k-lowdrift-r16"
SPLIT_RUN_ID = "AUDIT-0002-clean-split-passN-20260821-215706"
SYSTEM_PROMPT = "You are a helpful assistant that solves math problems step by step."
USER_SUFFIX = "Solve this step by step, then give the final answer as a single integer inside \\boxed{}."
SEED = 20260903


def compact(text: object) -> str:
    return re.sub(r"[\s_-]+", "", unicodedata.normalize("NFC", str(text)).casefold())


def preload_cuda13() -> None:
    runtime, nvrtc = [], []
    for directory in site.getsitepackages():
        runtime += glob.glob(str(Path(directory) / "nvidia" / "cu13" / "lib" / "libcudart.so.13*"))
        runtime += glob.glob(str(Path(directory) / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.13*"))
        nvrtc += glob.glob(str(Path(directory) / "nvidia" / "cu13" / "lib" / "libnvrtc.so.13*"))
        nvrtc += glob.glob(str(Path(directory) / "nvidia" / "cuda_nvrtc" / "lib" / "libnvrtc.so.13*"))
    runtime, nvrtc = sorted(set(runtime)), sorted(set(nvrtc))
    if not runtime or not nvrtc:
        raise RuntimeError("CUDA 13 libraries missing. Run the notebook setup cell, then restart once.")
    ctypes.CDLL(runtime[0], mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(nvrtc[0], mode=ctypes.RTLD_GLOBAL)
    library_dirs = sorted({str(Path(runtime[0]).parent), str(Path(nvrtc[0]).parent)})
    os.environ["LD_LIBRARY_PATH"] = ":".join(library_dirs + ([os.environ["LD_LIBRARY_PATH"]] if os.environ.get("LD_LIBRARY_PATH") else []))
    os.environ["LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]


def locate_project() -> Path:
    from google.colab import drive
    mount = Path("/content/drive")
    if not (mount / "MyDrive").exists():
        drive.mount(str(mount))
    roots = [path for path in (mount / "MyDrive").iterdir() if path.is_dir() and compact(path.name) == compact("2026소중한챌린지")]
    if len(roots) != 1:
        raise RuntimeError(roots)
    return roots[0]


def read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [column.strip() for column in frame.columns]
    if not {"id", "question", "answer"}.issubset(frame.columns):
        raise ValueError(frame.columns.tolist())
    if frame.id.duplicated().any() or not frame.answer.str.fullmatch(r"-?\d+").all():
        raise ValueError("Invalid held split")
    return frame[["id", "question", "answer"]].copy()


def adapter_weight(path: Path) -> Path:
    weight = next((path / name for name in ("adapter_model.safetensors", "adapter_model.bin") if (path / name).exists()), None)
    if weight is None or not (path / "adapter_config.json").exists():
        raise FileNotFoundError(path)
    return weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="EVAL-0015-r2pro4-external14k-frozen")
    parser.add_argument("--tune-n", type=int, default=8)
    parser.add_argument("--dev-n", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use a fresh A100 GPU runtime")
    preload_cuda13()
    for stream, file_descriptor in ((sys.stdout, 1), (sys.stderr, 2)):
        try:
            stream.fileno()
        except Exception:
            stream.fileno = lambda fd=file_descriptor: fd

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    class VLLMWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not any(fragment in record.getMessage() for fragment in ("deprecated support for supporting different tokenizers", "Triton kernel JIT compilation"))
    for logger_name in ("vllm.v1.engine.input_processor", "vllm.compilation.jit_monitor"):
        logging.getLogger(logger_name).addFilter(VLLMWarningFilter())

    project_dir = locate_project()
    runs_dir = project_dir / "runs"
    split_dir = runs_dir / SPLIT_RUN_ID / "splits"
    tune = read_split(split_dir / "tune_v1.csv")
    dev = read_split(split_dir / "dev_v1.csv")
    if set(tune.id) & set(dev.id):
        raise ValueError("Held splits overlap")
    parent = runs_dir / PARENT_RUN_ID / "adapter_final"
    candidate = runs_dir / CANDIDATE_RUN_ID
    adapter_paths = {"parent_r2": parent}
    checkpoint_71 = candidate / "checkpoints" / "checkpoint-71"
    if checkpoint_71.exists():
        adapter_paths["external_ckpt71"] = checkpoint_71
    adapter_paths["external_final"] = candidate / "adapter_final"
    for path in adapter_paths.values():
        adapter_weight(path)

    exp_dir = runs_dir / args.run_id
    candidate_dir, prediction_dir, report_dir = exp_dir / "candidates", exp_dir / "predictions", exp_dir / "reports"
    for path in (candidate_dir, prediction_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    torch.backends.cuda.matmul.allow_tf32 = True
    llm = LLM(model=BASE_MODEL, revision=MODEL_REVISION, dtype="bfloat16", tensor_parallel_size=1, distributed_executor_backend="uni", gpu_memory_utilization=.94, max_model_len=8192, max_num_seqs=256, max_num_batched_tokens=65536, enable_chunked_prefill=True, enable_prefix_caching=True, enable_lora=True, max_lora_rank=64, max_loras=4, max_cpu_loras=4, performance_mode="throughput")
    lora_requests = {name: LoRARequest(name, index + 1, str(path)) for index, (name, path) in enumerate(adapter_paths.items())}

    def prompt(question: str) -> str:
        return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{question.strip()}\n\n{USER_SUFFIX}"}], tokenize=False, add_generation_prompt=True)

    def generate(frame: pd.DataFrame, label: str, n: int, split_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
        raw_path = candidate_dir / f"{split_name}_{label}_sc{n}.jsonl"
        existing = load_jsonl_map(raw_path)
        expected = set(frame.id)
        if not set(existing).issubset(expected):
            raise ValueError(f"Unexpected IDs in {raw_path}")
        pending = frame.loc[~frame.id.isin(existing)].copy()
        print(f"[{label} {split_name}] done={len(existing)} pending={len(pending)}")
        started = time.time()
        for start in range(0, len(pending), 16):
            batch = pending.iloc[start:start + 16]
            prompts = [prompt(question) for question in batch.question]
            parameters = [SamplingParams(n=n, temperature=1.0, top_p=.95, max_tokens=args.max_new_tokens, seed=stable_seed(SEED, "r2_pro4_original_boxed", identifier)) for identifier in batch.id]
            requests = llm.generate(prompts, parameters, lora_request=lora_requests[label], use_tqdm=False)
            rows = []
            for (_, item), request in zip(batch.iterrows(), requests):
                candidates = []
                for sample_index, output in enumerate(request.outputs):
                    answer, parse_source = extract_integer(output.text)
                    candidates.append({"sample_index": sample_index, "answer": answer, "parse_source": parse_source, "generated_tokens": len(output.token_ids), "finish_reason": str(output.finish_reason or ""), "raw_output": output.text})
                rows.append({"id": item.id, "label": label, "split": split_name, "n": n, "seed": stable_seed(SEED, "r2_pro4_original_boxed", item.id), "candidates": candidates})
            append_jsonl(raw_path, rows)
            existing.update({row["id"]: row for row in rows})
            rate = len(existing) / max(time.time() - started, 1e-9)
            print(f"[{label} {split_name}] {len(existing)}/{len(frame)} eta={(len(frame)-len(existing))/max(rate,1e-9)/60:.1f}m", flush=True)
        records = load_jsonl_map(raw_path)
        if set(records) != expected:
            raise RuntimeError(f"Incomplete generation: {len(records)}/{len(expected)}")
        rows = []
        for item in frame.itertuples(index=False):
            record = records[item.id]
            vote = vote_candidates(record["candidates"])
            rows.append({"id": item.id, "answer": item.answer, "prediction": vote["answer"], "correct": vote["answer"] == item.answer, "top_count": vote["top_count"], "margin": vote["margin"], "tie": vote["tie"], "valid_count": vote["valid_count"], "parse_failure_candidates": sum(candidate["answer"] is None for candidate in record["candidates"])})
        result = pd.DataFrame(rows)
        result.to_csv(prediction_dir / f"{split_name}_{label}_sc{n}.csv", index=False)
        metrics = {"label": label, "split": split_name, "n": n, "rows": len(result), "correct": int(result.correct.sum()), "exact_match": float(result.correct.mean()), "tie_rows": int(result.tie.sum()), "parse_failure_candidates": int(result.parse_failure_candidates.sum()), "boxed_or_fallback_rate": float(1 - result.parse_failure_candidates.sum() / (len(result) * n)), "raw": str(raw_path), "prediction": str(prediction_dir / f"{split_name}_{label}_sc{n}.csv")}
        print("[METRICS]", json.dumps(metrics, indent=2))
        return result, metrics

    tune_metrics = []
    for label in adapter_paths:
        _, metrics = generate(tune, label, args.tune_n, "tune")
        metrics["adapter"] = str(adapter_paths[label])
        metrics["adapter_sha256"] = sha256_file(adapter_weight(adapter_paths[label]))
        tune_metrics.append(metrics)
    sweep = pd.DataFrame(tune_metrics).sort_values(["exact_match", "parse_failure_candidates", "label"], ascending=[False, True, True]).reset_index(drop=True)
    sweep.to_csv(report_dir / "tune_sc8_sweep.csv", index=False)
    winner = str(sweep.iloc[0].label)
    _, dev_metrics = generate(dev, winner, args.dev_n, "dev")
    dev_metrics["adapter"] = str(adapter_paths[winner])
    dev_metrics["adapter_sha256"] = sha256_file(adapter_weight(adapter_paths[winner]))
    report = {"run_id": args.run_id, "status": "completed", "base_model": BASE_MODEL, "model_revision": MODEL_REVISION, "prompt": {"system": SYSTEM_PROMPT, "user_suffix": USER_SUFFIX}, "seed": SEED, "tune_sweep": tune_metrics, "winner_frozen_from_tune": winner, "dev_sc16": dev_metrics, "reference_parent_dev_sc16": 0.768, "promotion": bool(winner != "parent_r2" and dev_metrics["exact_match"] >= .775), "promotion_rule": "candidate must win frozen tune SC8 and reach >=0.775 fixed dev SC16; no leaderboard read"}
    (report_dir / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[WINNER]", winner)
    print("[REPORT]", report_dir / "experiment_report.json")


if __name__ == "__main__":
    main()
