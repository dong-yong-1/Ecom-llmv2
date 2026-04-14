#!/usr/bin/env python3
"""Generate focused schema_v1 training data with DeepSeek and merge it safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_TRAIN = ROOT / "data" / "golden_v1_train.jsonl"
DEFAULT_SEED_FILE = ROOT / "data" / "golden_v1_train_focus_fix_v1.jsonl"
DEFAULT_RUN_DIR = ROOT / "data" / "augment" / "focus_fix_v1"
DEFAULT_GENERATED_OUTPUT = DEFAULT_RUN_DIR / "generated.jsonl"
DEFAULT_MERGED_OUTPUT = DEFAULT_RUN_DIR / "merged.jsonl"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_RUN_DIR / "summary.json"
DEFAULT_MODEL = "deepseek-chat"

TARGET_SCENARIOS = {"催发货", "取消订单", "修改地址", "退货条件判断", "查询物流进度", "退款流程说明"}
SCENARIO_PREFIX = {
    "催发货": "urge",
    "取消订单": "cancel",
    "修改地址": "change",
    "退货条件判断": "return",
    "查询物流进度": "logistics",
    "退款流程说明": "refund",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate focused training data with DeepSeek.")
    parser.add_argument("--base-train-file", default=str(DEFAULT_BASE_TRAIN), help="Original golden train JSONL.")
    parser.add_argument("--seed-file", default=str(DEFAULT_SEED_FILE), help="Focused seed JSONL used as examples.")
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Directory for one augmentation iteration. generated/merged/summary will be stored here by default.",
    )
    parser.add_argument(
        "--generated-output",
        default=None,
        help="Generated-only JSONL output. Default: <run-dir>/generated.jsonl",
    )
    parser.add_argument(
        "--merged-output",
        default=None,
        help="Merged JSONL output containing base + generated rows. Default: <run-dir>/merged.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Summary JSON output. Default: <run-dir>/summary.json",
    )
    parser.add_argument(
        "--scenarios",
        default="催发货,取消订单,修改地址,退货条件判断,查询物流进度,退款流程说明",
        help="Comma-separated scenarios to augment.",
    )
    parser.add_argument("--max-seeds", type=int, default=12, help="Max seed rows to use.")
    parser.add_argument("--variants-per-seed", type=int, default=1, help="New rows to generate per seed.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek chat model.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Generation temperature.")
    parser.add_argument("--sleep-seconds", type=float, default=0.8, help="Sleep between API calls.")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="HTTP timeout.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for one sample.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned generation workload.")
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


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


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_signature(row: dict[str, Any]) -> str:
    payload = row["train_payload"]
    normalized = {
        "context": payload.get("context", {}),
        "history": payload.get("history", []),
        "target": payload.get("target", ""),
        "scenario": row["annotation_meta"].get("scenario"),
    }
    return hashlib.md5(json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def next_row_id(scenario: str, used_ids: set[str], source_row: dict[str, Any], variant_index: int) -> str:
    prefix = SCENARIO_PREFIX.get(scenario, "aug")
    digest = hashlib.md5(
        json.dumps(
            {
                "scenario": scenario,
                "history": source_row["train_payload"].get("history", []),
                "context": source_row["train_payload"].get("context", {}),
                "variant_index": variant_index,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    candidate = f"aug_{prefix}_{digest}"
    suffix = 1
    while candidate in used_ids:
        candidate = f"aug_{prefix}_{digest}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def build_policy_block() -> str:
    return (
        "你必须严格遵守以下业务规则：\n"
        "1. 取消订单：未发货可取消；已发货不可取消，必须引导用户收到货后申请退货退款；状态未知先追问。\n"
        "2. 催发货：只能查询、解释、反馈催促，不能承诺发货时间、不能保证已加急；信息不足时先追问订单号或手机号尾号。\n"
        "3. 修改地址：未发货可修改地址；已发货或已揽收不可直接修改，必须引导用户自行联系承运物流尝试协调。\n"
        "4. 退货条件判断：要优先围绕是否签收、是否拆封、是否有质量问题来判断；拆封后无质量问题不支持售后，拆封后有质量问题可走质量问题售后。\n"
        "5. 查询物流进度：运输中只能说明当前物流状态并建议关注物流更新，不能编造具体到货时间，也不能承诺主动通知。\n"
        "6. 退款流程说明：信息不足时优先追问是否签收、是否有质量问题；不要编造已签收、质量问题成立等状态。\n"
        "7. 不编造订单状态、物流状态、处理结果、售后结果，也不要增加规则里没有的流程。\n"
        "8. 输出必须是 schema_v1 的单条 JSON 对象；可以写短多轮对话历史，但 target 只能是当前轮客服回复。\n"
        "9. 优先生成能修复 badcase 的 hardcase：边界条件、口语化表达、焦虑情绪、追问后承接。\n"
    )


def build_prompt(seed_row: dict[str, Any], variant_index: int) -> str:
    scenario = seed_row["annotation_meta"]["scenario"]
    payload = seed_row["train_payload"]
    prompt = {
        "task": "请基于下面的 seed 样本，生成 1 条新的高质量 schema_v1 训练样本。",
        "hard_requirements": [
            "新样本与 seed 语义接近，但措辞、对话内容、上下文字段组合要有变化。",
            "优先写成 2-4 轮短多轮对话，尤其是存在追问或承接时。",
            "只输出 1 个 JSON 对象，不要 markdown，不要解释。",
            "id 可先随意占位，后续会被脚本覆盖。",
            "version 固定为 schema_v1。",
            "scenario 只能是原场景，不要改场景。",
            "target 必须符合业务规则，不能编造。",
        ],
        "focus": {
            "scenario": scenario,
            "variant_index": variant_index,
            "seed_sample": seed_row,
        },
        "output_schema_hint": {
            "id": "placeholder_id",
            "version": "schema_v1",
            "train_payload": {
                "system": "沿用 seed 的系统指令风格",
                "context": "保留 schema_v1 的上下文字段结构",
                "history": "可以是多轮历史",
                "target": "当前轮客服回复",
            },
            "annotation_meta": {
                "scenario": scenario,
                "intent": "与样本匹配的中文意图",
                "decision_type": "direct_answer / ask_followup / transfer_human",
                "should_transfer": False,
                "missing_slots": [],
                "policy_basis": ["与规则匹配的中文依据"],
                "risk_flags": [],
                "quality_check": {
                    "is_consistent_with_context": True,
                    "contains_forbidden_promise": False,
                    "needs_revision": False,
                },
            },
        },
    }
    return build_policy_block() + "\n" + json.dumps(prompt, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output does not contain a JSON object.")
    return json.loads(text[start : end + 1])


def validate_generated_row(row: dict[str, Any]) -> None:
    if row.get("version") != "schema_v1":
        raise ValueError("version must be schema_v1")
    train_payload = row.get("train_payload")
    meta = row.get("annotation_meta")
    if not isinstance(train_payload, dict) or not isinstance(meta, dict):
        raise ValueError("Missing train_payload or annotation_meta")
    scenario = meta.get("scenario")
    if scenario not in TARGET_SCENARIOS:
        raise ValueError(f"Unsupported scenario: {scenario}")
    context = train_payload.get("context") or {}
    history = train_payload.get("history") or []
    target = (train_payload.get("target") or "").strip()
    if not target or not isinstance(history, list):
        raise ValueError("Missing target or history")

    if scenario == "催发货":
        shipping_status = context.get("shipping_status")
        if shipping_status == "未知":
            if "订单号" not in target and "手机号尾号" not in target and "手机尾号" not in target:
                raise ValueError("Unknown urge case should ask for order info")
            if "看到您的订单还未发货" in target or "这边看到您的订单还未发货" in target:
                raise ValueError("Unknown urge case cannot fabricate shipping status")
        if "今天" in target and "发" in target and "承诺" not in target:
            raise ValueError("Urge shipment reply looks like a time promise")

    if scenario == "取消订单":
        if context.get("shipping_status") == "已发货":
            if "取消" in target and ("可以" in target or "可" in target) and "无法" not in target and "不支持" not in target:
                raise ValueError("Shipped cancel case cannot allow direct cancellation")
            if "退货退款" not in target:
                raise ValueError("Shipped cancel case should guide return/refund")

    if scenario == "修改地址":
        if context.get("shipping_status") == "已发货":
            if "联系物流" not in target and "承运物流" not in target and "物流公司" not in target:
                raise ValueError("Shipped address change case should guide contacting logistics")
            if "联系卖家" in target or "联系寄件人" in target:
                raise ValueError("Shipped address change case should not guide seller/sender")

    if scenario == "退货条件判断":
        is_signed = context.get("is_signed")
        is_opened = context.get("is_opened")
        has_quality_issue = context.get("has_quality_issue")
        if "未知" in {is_signed, is_opened, has_quality_issue}:
            required = ["签收", "拆封", "质量"]
            if not all(keyword in target for keyword in required):
                raise ValueError("Return judgment with unknown slots should ask signed/opened/quality issue")
        if is_opened == "是" and has_quality_issue == "否" and ("不支持" not in target and "不能" not in target):
            raise ValueError("Opened and no-quality case should not support return")
        if is_opened == "是" and has_quality_issue == "是" and "质量问题" not in target:
            raise ValueError("Opened and quality issue case should guide quality-issue aftersales")

    if scenario == "查询物流进度":
        if context.get("logistics_status") == "运输中":
            forbidden_time_words = ["2-3天", "3天内", "明天", "今天内", "今晚", "小时内", "预计"]
            if any(word in target for word in forbidden_time_words):
                raise ValueError("Transit logistics case should not promise a delivery time")

    if scenario == "退款流程说明":
        is_signed = context.get("is_signed")
        has_quality_issue = context.get("has_quality_issue")
        if "未知" in {is_signed, has_quality_issue}:
            required = ["签收", "质量"]
            if not all(keyword in target for keyword in required):
                raise ValueError("Refund guidance with unknown slots should ask signed/quality issue")


def make_chat_request(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的数据合成助手，擅长生成符合 schema_v1 的中文电商客服训练样本。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def generate_rows(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    if not api_key and not args.dry_run:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in environment or .env")

    run_dir = Path(args.run_dir)
    generated_output = Path(args.generated_output) if args.generated_output else run_dir / "generated.jsonl"
    merged_output = Path(args.merged_output) if args.merged_output else run_dir / "merged.jsonl"
    summary_output = Path(args.summary_output) if args.summary_output else run_dir / "summary.json"

    base_rows = load_jsonl(Path(args.base_train_file))
    seed_rows = load_jsonl(Path(args.seed_file))
    selected_scenarios = {item.strip() for item in args.scenarios.split(",") if item.strip()}
    selected_seed_rows = [row for row in seed_rows if row["annotation_meta"]["scenario"] in selected_scenarios]
    selected_seed_rows = selected_seed_rows[: args.max_seeds]

    planned = len(selected_seed_rows) * args.variants_per_seed
    print(f"[info] selected_seeds={len(selected_seed_rows)} planned_generations={planned}")
    if args.dry_run:
        for row in selected_seed_rows:
            print(f"- {row['id']} | {row['annotation_meta']['scenario']}")
        return 0

    used_ids = {row["id"] for row in base_rows}
    seen_signatures = {sample_signature(row) for row in base_rows}
    generated_rows: list[dict[str, Any]] = []

    for seed_row in selected_seed_rows:
        scenario = seed_row["annotation_meta"]["scenario"]
        for variant_index in range(args.variants_per_seed):
            prompt = build_prompt(seed_row, variant_index)
            last_error: Exception | None = None
            for attempt in range(1, args.max_retries + 2):
                try:
                    response = make_chat_request(
                        api_key=api_key,
                        base_url=base_url,
                        model=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        timeout_seconds=args.timeout_seconds,
                    )
                    content = response["choices"][0]["message"]["content"]
                    row = extract_json_object(content)
                    row["id"] = next_row_id(scenario, used_ids, row, variant_index)
                    validate_generated_row(row)
                    signature = sample_signature(row)
                    if signature in seen_signatures:
                        raise ValueError("Duplicate sample signature")
                    seen_signatures.add(signature)
                    generated_rows.append(row)
                    print(f"[ok] {seed_row['id']} -> {row['id']}")
                    break
                except (KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
                    last_error = exc
                    print(f"[warn] seed={seed_row['id']} attempt={attempt} error={exc}", file=sys.stderr)
                    if attempt > args.max_retries:
                        print(f"[skip] seed={seed_row['id']} variant={variant_index}", file=sys.stderr)
                        break
                    time.sleep(args.sleep_seconds)
            time.sleep(args.sleep_seconds)
            if last_error and len(generated_rows) == 0:
                pass

    merged_rows = base_rows + generated_rows
    write_jsonl(generated_rows, generated_output)
    write_jsonl(merged_rows, merged_output)

    summary = {
        "base_train_file": str(Path(args.base_train_file).resolve()),
        "seed_file": str(Path(args.seed_file).resolve()),
        "run_dir": str(run_dir.resolve()),
        "generated_output": str(generated_output.resolve()),
        "merged_output": str(merged_output.resolve()),
        "generated_count": len(generated_rows),
        "merged_count": len(merged_rows),
        "scenarios": sorted(selected_scenarios),
        "model": args.model,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    return generate_rows(args)


if __name__ == "__main__":
    raise SystemExit(main())
