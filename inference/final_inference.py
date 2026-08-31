#!/usr/bin/env python3
"""Offline final-test inference for Qwen2.5-3B-Instruct + frozen LoRA.

Pipeline: SC16/2048 -> PAL4 fallback on SC16 margin <= 1.
Every generation stage is append-only and resume-safe by problem id.
"""

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
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.qwen_math_final import (  # noqa: E402
    append_jsonl,
    extract_integer,
    extract_python,
    load_jsonl_map,
    run_python,
    sha256_file,
    stable_seed,
    terminal_boxed_integer,
    vote_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Official final test CSV")
    parser.add_argument("--adapter", type=Path, required=True, help="Frozen R3 LoRA directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "final_inference.json"
    )
    return parser.parse_args()


def read_input(path: Path, expected_rows: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    if not {"id", "question"}.issubset(frame.columns):
        raise ValueError(f"Input must contain id,question; got {list(frame.columns)}")
    if frame["id"].duplicated().any() or (frame["id"].str.strip() == "").any():
        raise ValueError("Input IDs must be non-empty and unique")
    if (frame["question"].str.strip() == "").any():
        raise ValueError("Every question must be non-empty")
    if "answer" in frame.columns and (frame["answer"].str.strip() != "").any():
        raise ValueError("Refusing labeled input: official inference reads questions only")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}")
    return frame[["id", "question"]].copy()


def validate_adapter(path: Path, base_model: str, expected_sha256: str) -> Path:
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    weights = next(
        (path / name for name in ("adapter_model.safetensors", "adapter_model.bin") if (path / name).exists()),
        None,
    )
    if weights is None:
        raise FileNotFoundError(f"No adapter weights under {path}")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter_base = str(adapter_config.get("base_model_name_or_path", "")).rstrip("/")
    if adapter_base != base_model and not adapter_base.endswith("/Qwen2.5-3B-Instruct"):
        raise ValueError(f"Unexpected adapter base: {adapter_base}")
    digest = sha256_file(weights)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"Adapter SHA256 mismatch: {digest}")
    return weights


def preload_cuda13() -> None:
    runtime: list[str] = []
    nvrtc: list[str] = []
    package_roots = set(site.getsitepackages() + [site.getusersitepackages()] + sys.path)
    for directory in package_roots:
        if not directory:
            continue
        root = Path(directory)
        runtime += glob.glob(str(root / "nvidia" / "cu13" / "lib" / "libcudart.so.13*"))
        runtime += glob.glob(str(root / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.13*"))
        # Colab's package root has changed between images; keep a bounded fallback under nvidia/.
        runtime += glob.glob(str(root / "nvidia" / "**" / "libcudart.so.13*"), recursive=True)
        nvrtc += glob.glob(str(root / "nvidia" / "cu13" / "lib" / "libnvrtc.so.13*"))
        nvrtc += glob.glob(str(root / "nvidia" / "**" / "libnvrtc.so.13*"), recursive=True)
        nvrtc += glob.glob(str(Path(directory) / "nvidia" / "cuda_nvrtc" / "lib" / "libnvrtc.so.13*"))
    runtime, nvrtc = sorted(set(runtime)), sorted(set(nvrtc))
    if not runtime or not nvrtc:
        raise RuntimeError(
            "CUDA 13 runtime libraries are missing; run Cell 1, restart the runtime, "
            "then run Cells 2–4."
        )
    ctypes.CDLL(runtime[0], mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(nvrtc[0], mode=ctypes.RTLD_GLOBAL)
    directories = sorted({str(Path(runtime[0]).parent), str(Path(nvrtc[0]).parent)})
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(directories + ([previous] if previous else []))
    os.environ["LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    generation_cfg = config["generation"]
    pal_cfg = config["pal"]

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    frame = read_input(args.input.resolve(), config.get("expected_rows"))
    adapter_path = args.adapter.resolve()
    adapter_weight = validate_adapter(
        adapter_path, model_cfg["id"], model_cfg["adapter_weight_sha256"]
    )
    output_dir = args.output_dir.resolve()
    candidate_dir = output_dir / "candidates"
    prediction_dir = output_dir / "predictions"
    report_dir = output_dir / "reports"
    submission_dir = output_dir / "submissions"
    for directory in (candidate_dir, prediction_dir, report_dir, submission_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    preload_cuda13()
    for stream, descriptor in ((sys.stdout, 1), (sys.stderr, 2)):
        try:
            stream.fileno()
        except Exception:
            stream.fileno = lambda value=descriptor: value

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    class KnownWarningFilter(logging.Filter):
        fragments = (
            "deprecated support for supporting different tokenizers for different LoRAs",
            "Triton kernel JIT compilation during inference",
        )

        def filter(self, record: logging.LogRecord) -> bool:
            return not any(fragment in record.getMessage() for fragment in self.fragments)

    warning_filter = KnownWarningFilter()
    for logger_name in ("vllm.v1.engine.input_processor", "vllm.compilation.jit_monitor"):
        logging.getLogger(logger_name).addFilter(warning_filter)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"], revision=model_cfg["revision"], use_fast=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    torch.backends.cuda.matmul.allow_tf32 = True
    llm = LLM(
        model=model_cfg["id"],
        revision=model_cfg["revision"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        distributed_executor_backend="uni",
        gpu_memory_utilization=generation_cfg["gpu_memory_utilization"],
        max_model_len=generation_cfg["max_model_len"],
        max_num_seqs=generation_cfg["max_num_seqs"],
        max_num_batched_tokens=generation_cfg["max_num_batched_tokens"],
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        max_cpu_loras=2,
        performance_mode="throughput",
        trust_remote_code=False,
    )
    lora_request = LoRARequest("r2_pro4hint_final", 1, str(adapter_path))

    def cot_prompt(question: str) -> str:
        content = f"{question.strip()}\n\n{config['prompt']['user_suffix']}"
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config["prompt"]["system"]},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def pal_prompt(question: str) -> str:
        content = f"Problem:\n{question.strip()}\n\n{pal_cfg['user_suffix']}"
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": pal_cfg["system"]},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    id_set = set(frame["id"].astype(str))

    def generate_stage(
        stage_frame: pd.DataFrame,
        raw_path: Path,
        stage: str,
        max_tokens: int,
        prompt_builder,
        prompt_version: str,
        n_samples: int,
        temperature: float,
        top_p: float,
        chunk_size: int,
        seed_value: int,
    ) -> dict[str, dict[str, object]]:
        existing = load_jsonl_map(raw_path)
        expected = set(stage_frame["id"].astype(str))
        if not set(existing).issubset(expected):
            raise ValueError(f"{raw_path} contains unexpected IDs")
        started = time.time()
        print(f"[{stage}] target={len(stage_frame)} done={len(existing)} pending={len(expected)-len(existing)}")
        for start in range(0, len(stage_frame), chunk_size):
            batch = stage_frame.iloc[start : start + chunk_size]
            missing = set(batch["id"].astype(str)) - set(existing)
            if not missing:
                continue
            prompts = [prompt_builder(str(question)) for question in batch["question"]]
            lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
            if max(lengths, default=0) + max_tokens > generation_cfg["max_model_len"]:
                raise ValueError(f"Prompt plus {max_tokens} output tokens exceeds max_model_len")
            params = [
                SamplingParams(
                    n=n_samples,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=stable_seed(seed_value, prompt_version, str(qid)),
                )
                for qid in batch["id"].astype(str)
            ]
            outputs = llm.generate(prompts, params, lora_request=lora_request, use_tqdm=False)
            append_rows = []
            for (_, row), request in zip(batch.iterrows(), outputs):
                qid = str(row["id"])
                if qid not in missing:
                    continue
                if len(request.outputs) != n_samples:
                    raise RuntimeError(f"Unexpected sample count for {qid}")
                candidates = []
                for sample_index, completion in enumerate(request.outputs):
                    answer, parse_source = extract_integer(completion.text)
                    candidates.append(
                        {
                            "sample_index": sample_index,
                            "answer": answer,
                            "parse_source": parse_source,
                            "terminal_boxed_answer": terminal_boxed_integer(completion.text),
                            "generated_tokens": len(completion.token_ids),
                            "finish_reason": str(completion.finish_reason or ""),
                            "hit_cap": str(completion.finish_reason or "") == "length"
                            or len(completion.token_ids) >= max_tokens - 1,
                            "raw_output": completion.text,
                        }
                    )
                append_rows.append(
                    {
                        "id": qid,
                        "stage": stage,
                        "model": model_cfg["id"],
                        "model_revision": model_cfg["revision"],
                        "prompt_version": prompt_version,
                        "sampling_seed": stable_seed(seed_value, prompt_version, qid),
                        "n": n_samples,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_tokens,
                        "candidates": candidates,
                    }
                )
            append_jsonl(raw_path, append_rows)
            existing.update({str(row["id"]): row for row in append_rows})
            rate = len(existing) / max(time.time() - started, 1e-9)
            eta = (len(expected) - len(existing)) / max(rate, 1e-9) / 60
            print(f"[{stage}] {len(existing)}/{len(expected)} eta={eta:.1f}m", flush=True)
        final = load_jsonl_map(raw_path)
        if set(final) != expected:
            raise RuntimeError(f"Incomplete {stage}: {len(final)}/{len(expected)}")
        return final

    base_path = candidate_dir / "final_test_sc16_2048.jsonl"
    base = generate_stage(
        frame,
        base_path,
        "SC16-2048",
        generation_cfg["base_max_new_tokens"],
        cot_prompt,
        config["prompt"]["version"],
        generation_cfg["n"],
        generation_cfg["temperature"],
        generation_cfg["top_p"],
        generation_cfg["prompt_chunk"],
        generation_cfg["seed"],
    )

    base_votes: dict[str, dict[str, object]] = {}
    for qid in frame["id"].astype(str):
        original = sorted(base[qid]["candidates"], key=lambda item: int(item["sample_index"]))
        base_votes[qid] = vote_candidates(original)

    pal_target_ids = [
        qid for qid in frame["id"].astype(str) if int(base_votes[qid]["margin"]) <= pal_cfg["trigger_margin_le"]
    ]
    pal_frame = frame[frame["id"].astype(str).isin(pal_target_ids)].copy()
    pal_path = candidate_dir / "final_test_marginle1_pal4.jsonl"
    pal_raw = generate_stage(
        pal_frame,
        pal_path,
        "PAL4",
        pal_cfg["max_new_tokens"],
        pal_prompt,
        pal_cfg["prompt_version"],
        pal_cfg["n"],
        pal_cfg["temperature"],
        pal_cfg["top_p"],
        pal_cfg["prompt_chunk"],
        pal_cfg["seed"],
    ) if len(pal_frame) else {}

    pal_results: dict[str, dict[str, object]] = {}
    for qid in pal_target_ids:
        executions = []
        for candidate in sorted(pal_raw[qid]["candidates"], key=lambda item: int(item["sample_index"])):
            result = run_python(extract_python(candidate.get("raw_output")))
            executions.append({"sample_index": candidate["sample_index"], **result})
        ordered = [item["answer"] for item in executions if item["ok"] and item["answer"] is not None]
        counts = Counter(ordered)
        top_count = max(counts.values()) if counts else 0
        tied = {answer for answer, count in counts.items() if count == top_count}
        answer = next((answer for answer in ordered if answer in tied), None)
        pal_results[qid] = {
            "answer": answer,
            "top_count": top_count,
            "valid_executions": len(ordered),
            "executions": executions,
        }

    decision_rows = []
    for row in frame.itertuples(index=False):
        qid = str(row.id)
        base_vote = base_votes[qid]
        pal = pal_results.get(qid, {"answer": None, "top_count": 0, "valid_executions": 0})
        use_pal = bool(
            int(base_vote["margin"]) <= pal_cfg["trigger_margin_le"]
            and pal["answer"] is not None
            and int(pal["top_count"]) >= pal_cfg["min_agreement"]
            and pal["answer"] != base_vote["answer"]
        )
        final_answer = str(pal["answer"] if use_pal else base_vote["answer"])
        decision_rows.append(
            {
                "id": qid,
                "base_answer": base_vote["answer"],
                "base_top_count": base_vote["top_count"],
                "base_margin": base_vote["margin"],
                "pal_answer": pal["answer"],
                "pal_top_count": pal["top_count"],
                "pal_valid_executions": pal["valid_executions"],
                "used_pal": use_pal,
                "final_answer": final_answer,
            }
        )

    decisions = pd.DataFrame(decision_rows)
    if decisions["id"].tolist() != frame["id"].astype(str).tolist():
        raise RuntimeError("Prediction ID order changed")
    if decisions["id"].duplicated().any():
        raise RuntimeError("Duplicate prediction IDs")
    if not decisions["final_answer"].str.fullmatch(r"-?\d+").all():
        raise RuntimeError("Non-integer final answer")
    decisions_path = prediction_dir / "final_test_vote_diagnostics.csv"
    decisions.to_csv(decisions_path, index=False, encoding="utf-8")
    submission = decisions[["id", "final_answer"]].rename(columns={"final_answer": "answer"})
    submission_path = submission_dir / "submission.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8")
    reread = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
    if not reread.equals(submission):
        raise RuntimeError("Submission round-trip mismatch")

    report = {
        "run_id": config["run_id"],
        "status": "completed",
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input.resolve()),
        "rows": len(frame),
        "base_model": model_cfg["id"],
        "base_model_revision": model_cfg["revision"],
        "adapter": str(adapter_path),
        "adapter_weight": str(adapter_weight),
        "adapter_weight_sha256": sha256_file(adapter_weight),
        "configuration": config,
        "diagnostics": {
            "pal_target_rows": len(pal_target_ids),
            "pal_answer_changes": int(decisions["used_pal"].sum()),
            "no_valid_vote_rows": int((decisions["base_top_count"] == 0).sum()),
        },
        "artifacts": {
            "base_raw": str(base_path),
            "pal_raw": str(pal_path),
            "vote_diagnostics": str(decisions_path),
            "submission": str(submission_path),
            "submission_sha256": sha256_file(submission_path),
        },
        "external_api_calls": 0,
        "answer_lookup": False,
        "test_labels_read": False,
    }
    report_path = report_dir / "final_inference_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["diagnostics"], ensure_ascii=False, indent=2))
    print("[SUBMISSION]", submission_path)
    print("[REPORT]", report_path)


if __name__ == "__main__":
    main()
