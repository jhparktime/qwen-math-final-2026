#!/usr/bin/env python3
"""Low-drift continuation of the frozen R2 Pro4-hint adapter on public math.

This program never reads the competition leaderboard/private test. It uses the
already-audited DATA-0004 public-math corpus and continues the existing R2 LoRA
weights: it does not merge models or create a new base checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_PROJECT", "deep-learning-challenge-2026")
os.environ.setdefault("WANDB_MODE", "offline")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
PARENT_RUN_ID = "RFT-0004B-r2-pro4-hint-lowdrift-lora"
EXTERNAL_RUN_ID = "DATA-0004-balanced20k-qwen-hint-rewrite"
SYSTEM_PROMPT = "You are a helpful assistant that solves math problems step by step."
USER_SUFFIX = "Solve this step by step, then give the final answer as a single integer inside \\boxed{}."
BOX_RE = re.compile(r"\\boxed\s*\{\s*(-?\d(?:[\d,]*\d)?)\s*\}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(text: object) -> str:
    return re.sub(r"[\s_-]+", "", unicodedata.normalize("NFC", str(text)).casefold())


def terminal_boxed(text: object) -> str | None:
    value = str(text)
    matches = list(BOX_RE.finditer(value))
    if not matches:
        return None
    match = matches[-1]
    tail = re.sub(r"^(?:\\\)|\\\]|\$)+", "", value[match.end():].strip())
    tail = re.sub(r"^[.!]+$", "", tail).strip()
    integer = match.group(1).replace(",", "")
    return integer if not tail and re.fullmatch(r"-?\d+", integer) else None


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [name.strip() for name in frame.columns]
    for required, alternatives in {"id": ("id", "external_id"), "solution": ("solution", "qwen_solution", "rewritten_solution", "response")}.items():
        if required not in frame.columns:
            source = next((name for name in alternatives if name in frame.columns), None)
            if source is None:
                raise ValueError(f"Missing {required}; found {frame.columns.tolist()}")
            frame = frame.rename(columns={source: required})
    required = {"id", "question", "answer", "solution"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing {required - set(frame.columns)}")
    if "source" not in frame.columns:
        frame["source"] = "public_math"
    frame["answer"] = frame["answer"].astype(str).str.strip().str.replace(",", "", regex=False)
    if not frame["answer"].str.fullmatch(r"-?\d+").all():
        raise ValueError("Non-integer answer found in supposedly verified public corpus")
    return frame


def locate_project() -> Path:
    from google.colab import drive
    mount = Path("/content/drive")
    if not (mount / "MyDrive").exists():
        drive.mount(str(mount))
    roots = [path for path in (mount / "MyDrive").iterdir() if path.is_dir() and compact(path.name) == compact("2026소중한챌린지")]
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one project root, found {roots}")
    return roots[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="RFT-0015-r2pro4-external14k-lowdrift-r16")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-rows", type=int, default=14170)
    parser.add_argument("--epochs", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=6)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use an A100-class BF16 runtime")
    set_seed(args.seed)

    project_dir = locate_project()
    runs_dir = project_dir / "runs"
    parent_adapter = runs_dir / PARENT_RUN_ID / "adapter_final"
    external_csv = runs_dir / EXTERNAL_RUN_ID / "data" / "balanced20k_qwen_hint_rewrite_verified.csv"
    external_report = runs_dir / EXTERNAL_RUN_ID / "reports" / "experiment_report.json"
    parent_weight = next((path for path in (parent_adapter / "adapter_model.safetensors", parent_adapter / "adapter_model.bin") if path.exists()), None)
    for path in (parent_adapter / "adapter_config.json", parent_weight, external_csv, external_report):
        if path is None or not path.exists():
            raise FileNotFoundError(path)

    experiment_dir = runs_dir / args.run_id
    checkpoint_dir = experiment_dir / "checkpoints"
    final_adapter = experiment_dir / "adapter_final"
    report_dir = experiment_dir / "reports"
    tensorboard_dir = experiment_dir / "logs" / "tensorboard"
    wandb_dir = experiment_dir / "logs" / "wandb"
    for path in (checkpoint_dir, report_dir, tensorboard_dir, wandb_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    print("[GPU]", torch.cuda.get_device_name(0))
    print("[PARENT]", parent_adapter, sha256_file(parent_weight))
    print("[EXTERNAL]", external_csv, sha256_file(external_csv))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REVISION, use_fast=True, token=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    external = canonicalize_columns(pd.read_csv(external_csv, dtype=str, keep_default_na=False))
    if {"leaderboard_answer", "test_answer"}.intersection(external.columns):
        raise ValueError("Unexpected evaluation answer field in external training corpus")
    external = external[external.apply(lambda row: terminal_boxed(row.solution) == row.answer, axis=1)].copy()
    external["_rank"] = external.apply(lambda row: hashlib.sha256(f"{args.seed}|public|{row.id}|{row.solution}".encode()).hexdigest(), axis=1)
    external = external.sort_values("_rank").drop_duplicates(["question", "solution"]).head(args.max_rows).drop(columns="_rank").reset_index(drop=True)
    if len(external) < 10_000:
        raise ValueError(f"Only {len(external)} strict verified public records remain")

    def user_text(question: str) -> str:
        return f"{question.strip()}\n\n{USER_SUFFIX}"

    def prompt(question: str) -> str:
        return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text(question)}], tokenize=False, add_generation_prompt=True)

    def full(question: str, solution: str) -> str:
        return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text(question)}, {"role": "assistant", "content": solution.strip()}], tokenize=False, add_generation_prompt=False)

    def tokenize_row(row: dict[str, str]) -> dict[str, object]:
        prefix = tokenizer(prompt(row["question"]), add_special_tokens=False)["input_ids"]
        encoded = tokenizer(full(row["question"], row["solution"]), add_special_tokens=False)
        if encoded["input_ids"][: len(prefix)] != prefix:
            raise ValueError("Chat-template prefix mismatch")
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": [-100] * len(prefix) + encoded["input_ids"][len(prefix):], "total_tokens": len(encoded["input_ids"])}

    # Python 3.13's multiprocessing + datasets/tblib can emit BufferError while
    # serializing tokenizer exceptions. One worker remains fast enough here and
    # avoids turning a preprocessing warning into a worker-pool failure.
    dataset = Dataset.from_pandas(external[["id", "question", "answer", "solution", "source"]], preserve_index=False).map(tokenize_row, num_proc=1, desc="Tokenizing verified public math")
    lengths = np.asarray(dataset["total_tokens"])
    eligible = dataset.select(np.flatnonzero(lengths <= args.max_seq_length).tolist())
    rejected = int((lengths > args.max_seq_length).sum())
    if len(eligible) < 10_000 or rejected / len(dataset) >= 0.03:
        raise ValueError({"eligible": len(eligible), "overlength": rejected})
    diagnostic_ids = sorted(set(eligible["id"]), key=lambda value: hashlib.sha256(f"{args.seed}|diagnostic|{value}".encode()).hexdigest())
    eval_ids = set(diagnostic_ids[: max(256, len(diagnostic_ids) // 20)])
    train_indices = [index for index, value in enumerate(eligible["id"]) if value not in eval_ids]
    eval_indices = [index for index, value in enumerate(eligible["id"]) if value in eval_ids]
    remove_columns = ["id", "question", "answer", "solution", "source", "total_tokens"]
    train_dataset = eligible.select(train_indices).remove_columns(remove_columns)
    eval_dataset = eligible.select(eval_indices).remove_columns(remove_columns)
    pd.DataFrame({"id": sorted(eval_ids)}).to_csv(experiment_dir / "external_diagnostic_eval_ids.csv", index=False)
    print({"raw": len(dataset), "eligible": len(eligible), "train": len(train_dataset), "eval": len(eval_dataset), "overlength": rejected})

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=MODEL_REVISION, torch_dtype=torch.bfloat16, device_map={"": 0}, token=False)
    model = PeftModel.from_pretrained(base, str(parent_adapter), is_trainable=True, adapter_name="default")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if not 0 < trainable / total < 0.02:
        raise ValueError((trainable, total))
    model.print_trainable_parameters()
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
    updates = max(1, math.ceil(len(train_dataset) / (args.micro_batch * args.grad_accum) * args.epochs))
    save_every = max(1, min(25, updates // 3 or 1))
    evaluation_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    training_kwargs: dict[str, object] = {"output_dir": str(checkpoint_dir), "logging_dir": str(tensorboard_dir), "num_train_epochs": args.epochs, "learning_rate": args.learning_rate, "per_device_train_batch_size": args.micro_batch, "per_device_eval_batch_size": 4, "gradient_accumulation_steps": args.grad_accum, "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False}, "optim": "paged_adamw_8bit", "lr_scheduler_type": "cosine", "warmup_ratio": 0.03, "weight_decay": 0.0, "max_grad_norm": 1.0, "bf16": True, "fp16": False, "tf32": True, "logging_steps": 5, "logging_first_step": True, "eval_steps": save_every, "save_steps": save_every, "save_strategy": "steps", "save_total_limit": 4, "load_best_model_at_end": False, "prediction_loss_only": True, "report_to": ["tensorboard", "wandb"], "run_name": args.run_id, "remove_unused_columns": False, "group_by_length": True, "dataloader_num_workers": 2, "seed": args.seed, "data_seed": args.seed}
    training_kwargs[evaluation_key] = "steps"
    trainer = Trainer(model=model, args=TrainingArguments(**training_kwargs), train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collator)
    resume = get_last_checkpoint(str(checkpoint_dir))
    print({"resume": resume, "updates": updates, "effective_batch": args.micro_batch * args.grad_accum})
    started = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    runtime_seconds = time.time() - started
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))

    report = {"run_id": args.run_id, "status": "completed", "objective": "Continue R2 Pro4-hint r16 only on strict verified public math", "base_model": BASE_MODEL, "model_revision": MODEL_REVISION, "parent_adapter": str(parent_adapter), "parent_weight_sha256": sha256_file(parent_weight), "public_data": str(external_csv), "public_data_sha256": sha256_file(external_csv), "external_data_report": str(external_report), "raw_public_rows": len(dataset), "eligible_public_rows": len(eligible), "train_rows": len(train_dataset), "external_diagnostic_eval_rows": len(eval_dataset), "overlength_rejected": rejected, "source_counts": external["source"].value_counts().to_dict(), "epochs": args.epochs, "learning_rate": args.learning_rate, "effective_batch": args.micro_batch * args.grad_accum, "max_seq_length": args.max_seq_length, "runtime_seconds": runtime_seconds, "wandb_mode": os.environ["WANDB_MODE"], "wandb_dir": str(wandb_dir), "train_metrics": result.metrics, "adapter_final": str(final_adapter), "promotion_rule": "Evaluate all saved checkpoints on frozen tune SC8, then one winner on fixed dev SC16. Do not promote on loss."}
    report_path = report_dir / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[ADAPTER]", final_adapter)
    print("[REPORT]", report_path)
    print("[NEXT] Run frozen tune SC8 before any dev or leaderboard inference.")


if __name__ == "__main__":
    main()
