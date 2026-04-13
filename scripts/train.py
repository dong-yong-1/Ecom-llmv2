#!/usr/bin/env python3
"""Unified training entry for SFT and LoRA ablation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "golden_v1_train.jsonl"
DEFAULT_DATA_DIR = ROOT / "data" / "trl_sft"
DEFAULT_SINGLE_OUTPUT = ROOT / "model" / "trl_sft_run"
DEFAULT_ABLATION_OUTPUT = ROOT / "model" / "trl_lora_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified SFT training script.")
    parser.add_argument("--mode", choices=["single", "ablation"], default="single", help="Training mode.")
    parser.add_argument("--model-name-or-path", required=True, help="Base model path or HF repo.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT), help="Golden train JSONL path.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Prepared train/val data directory.")
    parser.add_argument("--train-file", default=None, help="Optional prepared train JSONL path.")
    parser.add_argument("--eval-file", default=None, help="Optional prepared eval JSONL path.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--output-dir", default=str(DEFAULT_SINGLE_OUTPUT), help="Single-run output dir.")
    parser.add_argument("--output-root", default=str(DEFAULT_ABLATION_OUTPUT), help="Ablation output root.")
    parser.add_argument("--ranks", default="8,16,32", help="Comma-separated LoRA ranks for ablation mode.")
    parser.add_argument(
        "--target-modules-grid",
        default="attention_only,attn_mlp,all_linear",
        help="Comma-separated target modules for ablation mode.",
    )
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
        help="LoRA target modules alias or comma-separated explicit module names for single mode.",
    )
    parser.add_argument("--report-to", default="none", help="Transformers report_to setting.")
    parser.add_argument("--run-name", default=None, help="Optional run name for trainer / wandb.")
    parser.add_argument("--wandb-project", default=None, help="Optional wandb project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional wandb entity/team.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def parse_input_files(value: str) -> list[Path]:
    files: list[Path] = []
    for part in value.split(","):
        candidate = Path(part.strip())
        if not part.strip():
            continue
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        files.append(candidate)
    if not files:
        raise ValueError("At least one input file is required.")
    return files


def stable_bucket(row_id: str) -> str:
    return hashlib.md5(row_id.encode("utf-8")).hexdigest()


def stratified_split(rows: list[dict[str, Any]], val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["scenario"], []).append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for _, items in sorted(groups.items()):
        items_sorted = sorted(items, key=lambda item: stable_bucket(item["id"]))
        if len(items_sorted) <= 1:
            train_rows.extend(items_sorted)
            continue
        val_count = max(1, int(round(len(items_sorted) * val_ratio)))
        if val_count >= len(items_sorted):
            val_count = len(items_sorted) - 1
        val_rows.extend(items_sorted[:val_count])
        train_rows.extend(items_sorted[val_count:])
    return train_rows, val_rows


def format_context(context: dict[str, Any]) -> str:
    if not context:
        return "无"
    lines = []
    for key in sorted(context):
        value = context[key]
        if value is None:
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines) if lines else "无"


def build_user_message(train_payload: dict[str, Any]) -> str:
    context = train_payload.get("context") or {}
    history = train_payload.get("history") or []
    lines = ["已知上下文：", format_context(context), "", "对话历史："]
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "客服"
        lines.append(f"{label}：{content}")
    lines.extend(["", "请输出当前轮客服回复。"])
    return "\n".join(lines)


def normalize_row(raw: dict[str, Any]) -> dict[str, Any]:
    row_id = raw.get("id")
    train_payload = raw.get("train_payload")
    meta = raw.get("annotation_meta")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("Each row must have a non-empty id")
    if not isinstance(train_payload, dict) or not isinstance(meta, dict):
        raise ValueError(f"Invalid schema row: {row_id}")

    system = (train_payload.get("system") or "").strip()
    target = (train_payload.get("target") or "").strip()
    if not system or not target:
        raise ValueError(f"Row {row_id} has empty system or target")

    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_message(train_payload)},
    ]
    completion_messages = [{"role": "assistant", "content": target}]

    return {
        "id": row_id,
        "prompt": prompt_messages,
        "completion": completion_messages,
        "scenario": meta.get("scenario", "未知"),
        "intent": meta.get("intent", "未知"),
        "decision_type": meta.get("decision_type", "unknown"),
        "should_transfer": bool(meta.get("should_transfer", False)),
        "prompt_text": prompt_messages[1]["content"],
        "target_text": target,
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset(input_spec: str, data_dir: Path, val_ratio: float) -> tuple[Path, Path, dict[str, Any]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    input_files = parse_input_files(input_spec)
    raw_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for input_file in input_files:
        current_rows = load_jsonl(input_file)
        for row in current_rows:
            row_id = row.get("id")
            if row_id in seen_ids:
                duplicate_ids.append(str(row_id))
                continue
            seen_ids.add(str(row_id))
            raw_rows.append(row)
    if duplicate_ids:
        raise ValueError(f"Duplicate row ids found across input files: {sorted(set(duplicate_ids))[:10]}")
    normalized_rows = [normalize_row(row) for row in raw_rows]
    train_rows, val_rows = stratified_split(normalized_rows, val_ratio)

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)

    summary = {
        "input": input_spec,
        "input_files": [str(path) for path in input_files],
        "output_dir": str(data_dir.resolve()),
        "num_total": len(normalized_rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "train_file": str(train_path.resolve()),
        "val_file": str(val_path.resolve()),
    }
    (data_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return train_path, val_path, summary


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
    return dataset["train"], dataset.get("validation")


def detect_device_backend() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def render_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"<|{m.get('role', 'user')}|>\n{m.get('content', '')}".strip() for m in messages)


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


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_training_artifacts(output_dir: Path, trainer: Any) -> dict[str, Any]:
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    (output_dir / "log_history.json").write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    train_points: list[dict[str, Any]] = []
    eval_points: list[dict[str, Any]] = []
    learning_rate_points: list[dict[str, Any]] = []
    for entry in log_history:
        step = entry.get("step")
        epoch = entry.get("epoch")
        if "loss" in entry:
            train_points.append({"step": step, "epoch": epoch, "loss": entry.get("loss")})
        if "eval_loss" in entry:
            eval_points.append({"step": step, "epoch": epoch, "eval_loss": entry.get("eval_loss")})
        if "learning_rate" in entry:
            learning_rate_points.append({"step": step, "epoch": epoch, "learning_rate": entry.get("learning_rate")})

    curves = {
        "train_loss": train_points,
        "eval_loss": eval_points,
        "learning_rate": learning_rate_points,
    }
    (output_dir / "curves.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_lines = ["step,epoch,train_loss,eval_loss,learning_rate"]
    merged_by_step: dict[Any, dict[str, Any]] = {}
    for point in train_points:
        merged_by_step.setdefault(point["step"], {}).update(point)
    for point in eval_points:
        merged_by_step.setdefault(point["step"], {}).update(point)
    for point in learning_rate_points:
        merged_by_step.setdefault(point["step"], {}).update(point)
    for step in sorted(merged_by_step, key=lambda value: (-1 if value is None else value)):
        row = merged_by_step[step]
        csv_lines.append(
            ",".join(
                [
                    str("" if row.get("step") is None else row.get("step")),
                    str("" if row.get("epoch") is None else row.get("epoch")),
                    str("" if row.get("loss") is None else row.get("loss")),
                    str("" if row.get("eval_loss") is None else row.get("eval_loss")),
                    str("" if row.get("learning_rate") is None else row.get("learning_rate")),
                ]
            )
        )
    (output_dir / "curves.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    plot_created = False
    try:
        import matplotlib.pyplot as plt  # type: ignore

        if train_points or eval_points:
            fig, ax = plt.subplots(figsize=(8, 5))
            if train_points:
                ax.plot(
                    [point["step"] for point in train_points],
                    [point["loss"] for point in train_points],
                    label="train_loss",
                    linewidth=2,
                )
            if eval_points:
                ax.plot(
                    [point["step"] for point in eval_points],
                    [point["eval_loss"] for point in eval_points],
                    label="eval_loss",
                    linewidth=2,
                )
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.set_title("Training Loss Curve")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "loss_curve.png", dpi=150)
            plt.close(fig)
            plot_created = True
    except Exception:
        plot_created = False

    return {
        "num_log_entries": len(log_history),
        "num_train_loss_points": len(train_points),
        "num_eval_loss_points": len(eval_points),
        "loss_curve_png_created": plot_created,
        "log_history_file": str((output_dir / "log_history.json").resolve()),
        "curves_json_file": str((output_dir / "curves.json").resolve()),
        "curves_csv_file": str((output_dir / "curves.csv").resolve()),
        "loss_curve_png_file": str((output_dir / "loss_curve.png").resolve()) if plot_created else None,
    }


def train_once(args: argparse.Namespace, *, train_file: Path, eval_file: Path | None, output_dir: Path, lora_r: int | None = None, target_modules: str | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_backend = detect_device_backend()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    train_dataset, eval_dataset = load_local_json_dataset(str(train_file), str(eval_file) if eval_file and args.eval_strategy != "no" else None)
    train_dataset, eval_dataset, dataset_text_field, completion_only_loss = maybe_prepare_text_dataset(train_dataset, eval_dataset, tokenizer)

    peft_config = None
    effective_target_modules = target_modules or args.target_modules
    effective_lora_r = lora_r if lora_r is not None else args.lora_r
    if args.use_lora:
        peft_config = LoraConfig(
            r=effective_lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=resolve_target_modules(effective_target_modules),
        )

    report_to: list[str] | str
    if args.report_to == "none":
        report_to = []
    elif "," in args.report_to:
        report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
    else:
        report_to = [args.report_to]

    run_name = args.run_name or output_dir.name
    if "wandb" in report_to:
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ.setdefault("WANDB_WATCH", "false")

    gradient_checkpointing = args.gradient_checkpointing and not args.no_gradient_checkpointing
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
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
    artifact_summary = save_training_artifacts(output_dir, trainer)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "device_backend": device_backend,
        "train_file": str(train_file.resolve()),
        "eval_file": str(eval_file.resolve()) if eval_file else None,
        "output_dir": str(output_dir.resolve()),
        "run_name": run_name,
        "uses_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "dataset_text_field": dataset_text_field,
        "use_lora": args.use_lora,
        "lora_r": effective_lora_r if args.use_lora else None,
        "lora_alpha": args.lora_alpha if args.use_lora else None,
        "target_modules": resolve_target_modules(effective_target_modules) if args.use_lora else None,
        "metrics": metrics,
        "artifacts": artifact_summary,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del trainer
    del model
    cleanup_memory()
    return summary


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_ablation(args: argparse.Namespace, *, train_file: Path, eval_file: Path) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ranks = [int(item) for item in parse_csv(args.ranks)]
    target_modules_grid = parse_csv(args.target_modules_grid)

    records: list[dict[str, Any]] = []
    for index, (rank, target) in enumerate([(r, t) for r in ranks for t in target_modules_grid], start=1):
        exp_name = f"r{rank}_{target}"
        exp_output_dir = output_root / exp_name
        print(f"[{index}/{len(ranks) * len(target_modules_grid)}] {exp_name}")
        try:
            summary = train_once(args, train_file=train_file, eval_file=eval_file, output_dir=exp_output_dir, lora_r=rank, target_modules=target)
            records.append({
                "name": exp_name,
                "rank": rank,
                "target_modules": target,
                "output_dir": str(exp_output_dir.resolve()),
                "status": "ok",
                "metrics": summary.get("metrics", {}),
            })
        except Exception as exc:  # noqa: BLE001
            records.append({
                "name": exp_name,
                "rank": rank,
                "target_modules": target,
                "output_dir": str(exp_output_dir.resolve()),
                "status": "failed",
                "error": str(exc),
            })
            print(f"[error] {exp_name}: {exc}")
            cleanup_memory()

    (output_root / "ablation_summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok_count = sum(1 for row in records if row["status"] == "ok")
    print(json.dumps({"mode": "ablation", "num_total": len(records), "num_ok": ok_count, "output_root": str(output_root.resolve())}, ensure_ascii=False, indent=2))
    return 0 if ok_count == len(records) else 1


def main() -> int:
    args = parse_args()
    input_spec = args.input_file
    data_dir = Path(args.data_dir)
    train_file = Path(args.train_file) if args.train_file else data_dir / "train.jsonl"
    eval_file = Path(args.eval_file) if args.eval_file else data_dir / "val.jsonl"

    if args.train_file is None or args.eval_file is None:
        prepared_train, prepared_eval, _ = prepare_dataset(input_spec, data_dir, args.val_ratio)
        if args.train_file is None:
            train_file = prepared_train
        if args.eval_file is None:
            eval_file = prepared_eval

    if args.mode == "single":
        train_once(args, train_file=train_file, eval_file=eval_file, output_dir=Path(args.output_dir))
        return 0
    return run_ablation(args, train_file=train_file, eval_file=eval_file)


if __name__ == "__main__":
    raise SystemExit(main())
