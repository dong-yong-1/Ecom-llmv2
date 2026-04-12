#!/usr/bin/env python3
"""Generate eval predictions for baseline or fine-tuned chat models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_SET = ROOT / "data" / "golden_v1_eval.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions on the golden eval set.")
    parser.add_argument("--model-name-or-path", required=True, help="Base model path or HF repo.")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET), help="Eval set JSONL path.")
    parser.add_argument("--output", required=True, help="Output predictions JSONL path.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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


def build_prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    payload = row["train_payload"]
    system = (payload.get("system") or "").strip()
    context = payload.get("context") or {}
    history = payload.get("history") or []
    lines = ["已知上下文：", format_context(context), "", "对话历史："]
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "客服"
        lines.append(f"{label}：{content}")
    lines.extend(["", "请输出当前轮客服回复。"])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def main() -> int:
    args = parse_args()
    eval_rows = load_jsonl(Path(args.eval_set))
    if args.limit:
        eval_rows = eval_rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(eval_rows, start=1):
            messages = build_prompt_messages(row)
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

            inputs = tokenizer(prompt_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            do_sample = args.temperature > 0
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)
            generated = output_ids[0][inputs["input_ids"].shape[1] :]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            record = {
                "id": row["id"],
                "prediction": prediction,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(eval_rows)}] {row['id']}")
    print(f"Saved predictions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
