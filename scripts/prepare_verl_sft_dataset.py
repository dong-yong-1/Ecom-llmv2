#!/usr/bin/env python3
"""Convert schema_v1 golden data into verl SFT parquet files.

verl 0.7.x defaults to `MultiTurnSFTDataset`, which expects a parquet column
like `messages`. This script converts the project's schema_v1 JSONL into that
shape, while also keeping a flattened `question/answer` view for debugging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "golden_v1_train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "verl_sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert schema_v1 golden data into verl SFT parquet files."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input schema_v1 JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for train/val parquet files.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio per scenario group.",
    )
    parser.add_argument(
        "--train-name",
        default="train.parquet",
        help="Train parquet filename.",
    )
    parser.add_argument(
        "--val-name",
        default="val.parquet",
        help="Validation parquet filename.",
    )
    parser.add_argument(
        "--jsonl-preview",
        action="store_true",
        help="Also write train/val JSONL previews next to parquet files.",
    )
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


def format_history(history: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "客服"
        lines.append(f"{label}：{content}")
    return "\n".join(lines)


def build_question(train_payload: dict[str, Any]) -> str:
    system = (train_payload.get("system") or "").strip()
    context = train_payload.get("context") or {}
    history = train_payload.get("history") or []
    return (
        f"系统指令：\n{system}\n\n"
        f"上下文：\n{format_context(context)}\n\n"
        f"对话历史：\n{format_history(history)}\n\n"
        "请输出当前轮客服回复："
    )


def build_messages(train_payload: dict[str, Any]) -> list[dict[str, str]]:
    system = (train_payload.get("system") or "").strip()
    history = train_payload.get("history") or []
    target = (train_payload.get("target") or "").strip()

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})
    if target:
        messages.append({"role": "assistant", "content": target})
    return messages


def stable_bucket(row_id: str) -> str:
    digest = hashlib.md5(row_id.encode("utf-8")).hexdigest()
    return digest


def stratified_split(rows: list[dict[str, Any]], val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scenario = row["scenario"]
        groups.setdefault(scenario, []).append(row)

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


def normalize_row(raw: dict[str, Any]) -> dict[str, Any]:
    train_payload = raw.get("train_payload")
    meta = raw.get("annotation_meta")
    if not isinstance(train_payload, dict) or not isinstance(meta, dict):
        raise ValueError(f"Invalid schema row: {raw.get('id')}")
    row_id = raw.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("Each row must have a non-empty id")
    question = build_question(train_payload)
    answer = (train_payload.get("target") or "").strip()
    if not answer:
        raise ValueError(f"Row {row_id} has empty target")
    context = train_payload.get("context") or {}
    return {
        "id": row_id,
        "messages": build_messages(train_payload),
        "question": question,
        "answer": answer,
        "scenario": meta.get("scenario", "未知"),
        "intent": meta.get("intent", "未知"),
        "decision_type": meta.get("decision_type", "unknown"),
        "should_transfer": bool(meta.get("should_transfer", False)),
        "context_json": json.dumps(context, ensure_ascii=False, sort_keys=True),
    }


def get_parquet_writer() -> tuple[str, Any]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        def write(rows: list[dict[str, Any]], path: Path) -> None:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, path)

        return "pyarrow", write
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        def write(rows: list[dict[str, Any]], path: Path) -> None:
            df = pd.DataFrame(rows)
            df.to_parquet(path, index=False)

        return "pandas", write
    except Exception:
        pass

    raise RuntimeError(
        "No parquet writer found. Please install `pyarrow` or `pandas` before preparing verl data."
    )


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

    writer_name, write_parquet = get_parquet_writer()
    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    write_parquet(train_rows, train_path)
    write_parquet(val_rows, val_path)

    if args.jsonl_preview:
        write_jsonl(train_rows, train_path.with_suffix(".jsonl"))
        write_jsonl(val_rows, val_path.with_suffix(".jsonl"))

    summary = {
        "input": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "writer": writer_name,
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
