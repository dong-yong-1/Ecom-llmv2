#!/usr/bin/env python3
"""Convert schema_v1 golden data into TRL SFT prompt-completion JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "golden_v1_train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "trl_sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert schema_v1 golden data into TRL prompt-completion JSONL files."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input schema_v1 JSONL path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio per scenario.")
    parser.add_argument("--train-name", default="train.jsonl", help="Train JSONL filename.")
    parser.add_argument("--val-name", default="val.jsonl", help="Validation JSONL filename.")
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


def stable_bucket(row_id: str) -> str:
    return hashlib.md5(row_id.encode("utf-8")).hexdigest()


def stratified_split(rows: list[dict[str, Any]], val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["scenario"], []).append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for scenario, items in sorted(groups.items()):
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


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    raw_rows = load_jsonl(input_path)
    normalized_rows = [normalize_row(row) for row in raw_rows]
    train_rows, val_rows = stratified_split(normalized_rows, args.val_ratio)

    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)

    summary = {
        "input": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_total": len(normalized_rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "train_file": str(train_path.resolve()),
        "val_file": str(val_path.resolve()),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
