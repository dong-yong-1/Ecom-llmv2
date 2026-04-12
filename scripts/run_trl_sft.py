#!/usr/bin/env python3
"""Run SFT for the e-commerce客服 project with TRL + LoRA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "trl_sft"
DEFAULT_MODEL_DIR = ROOT / "model" / "trl_sft_run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRL SFT training with LoRA.")
    parser.add_argument("--model-name-or-path", required=True, help="Base model path or HF repo.")
    parser.add_argument("--train-file", default=str(DEFAULT_DATA_DIR / "train.jsonl"), help="Train JSONL.")
    parser.add_argument("--eval-file", default=str(DEFAULT_DATA_DIR / "val.jsonl"), help="Eval JSONL.")
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR), help="Output dir.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Train epochs.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override total train steps, -1 means disabled.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2, help="Train batch size per device.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2, help="Eval batch size per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Grad accumulation.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps.")
    parser.add_argument("--eval-steps", type=int, default=50, help="Eval steps when eval strategy is steps.")
    parser.add_argument("--save-steps", type=int, default=50, help="Save steps when save strategy is steps.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Max checkpoints to keep.")
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--save-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="attn_mlp",
        help="LoRA target modules alias or comma-separated explicit module names.",
    )
    parser.add_argument("--report-to", default="none", help="Transformers report_to setting, e.g. none/tensorboard/wandb.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def resolve_target_modules(value: str) -> str | list[str]:
    aliases = {
        "attention_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attn_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp_only": ["gate_proj", "up_proj", "down_proj"],
        "attn_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "all_linear": "all-linear",
        "all-linear": "all-linear",
    }
    key = value.strip().lower()
    if key in aliases:
        return aliases[key]
    return [item.strip() for item in value.split(",") if item.strip()]


def load_local_json_dataset(train_file: str, eval_file: str | None) -> tuple[Any, Any | None]:
    data_files: dict[str, str] = {"train": train_file}
    if eval_file:
        data_files["validation"] = eval_file
    dataset = load_dataset("json", data_files=data_files)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation")
    return train_dataset, eval_dataset


def detect_device_backend() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def render_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"<|{role}|>\n{content}".strip())
    return "\n\n".join(parts)


def maybe_prepare_text_dataset(train_dataset: Any, eval_dataset: Any | None, tokenizer: Any) -> tuple[Any, Any | None, str | None, bool]:
    if getattr(tokenizer, "chat_template", None):
        return train_dataset, eval_dataset, None, True

    def add_text(row: dict[str, Any]) -> dict[str, str]:
        prompt = render_messages(row["prompt"])
        completion = render_messages(row["completion"])
        return {"text": f"{prompt}\n\n{completion}"}

    train_dataset = train_dataset.map(add_text, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(add_text, remove_columns=eval_dataset.column_names)
    return train_dataset, eval_dataset, "text", False


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_backend = detect_device_backend()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    train_dataset, eval_dataset = load_local_json_dataset(args.train_file, args.eval_file if args.eval_strategy != "no" else None)
    train_dataset, eval_dataset, dataset_text_field, completion_only_loss = maybe_prepare_text_dataset(
        train_dataset,
        eval_dataset,
        tokenizer,
    )

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=resolve_target_modules(args.target_modules),
        )

    report_to: list[str] | str
    if args.report_to == "none":
        report_to = []
    elif "," in args.report_to:
        report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
    else:
        report_to = [args.report_to]

    gradient_checkpointing = args.gradient_checkpointing and not args.no_gradient_checkpointing

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=gradient_checkpointing,
        report_to=report_to,
        seed=args.seed,
        dataset_text_field=dataset_text_field or "text",
        dataset_kwargs={"skip_prepare_dataset": False},
        completion_only_loss=completion_only_loss,
        assistant_only_loss=False,
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    metrics = dict(train_result.metrics)

    if eval_dataset is not None and args.eval_strategy != "no":
        eval_metrics = trainer.evaluate()
        metrics.update({f"final_{k}": v for k, v in eval_metrics.items()})

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "device_backend": device_backend,
        "train_file": str(Path(args.train_file).resolve()),
        "eval_file": str(Path(args.eval_file).resolve()) if args.eval_file else None,
        "output_dir": str(output_dir.resolve()),
        "uses_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "dataset_text_field": dataset_text_field,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r if args.use_lora else None,
        "lora_alpha": args.lora_alpha if args.use_lora else None,
        "target_modules": resolve_target_modules(args.target_modules) if args.use_lora else None,
        "metrics": metrics,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
