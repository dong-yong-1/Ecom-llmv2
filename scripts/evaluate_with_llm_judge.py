#!/usr/bin/env python3
"""
Evaluate model responses on the e-commerce golden eval set with an LLM judge.

The script combines:
1. local hard-rule checks for obvious business-rule violations
2. LLM-as-a-judge scoring for response quality and rule alignment

Only Python stdlib is used.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "doc"
ENV_PATH = ROOT / ".env"
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
SCORE_FIELDS = [
    "rule_correctness",
    "groundedness",
    "missing_info_handling",
    "transfer_decision",
    "actionability",
    "tone_control",
    "overall_quality",
]
FORBIDDEN_PROMISES = [
    "今天一定发",
    "明天就能到",
    "已经帮您修改成功",
    "已经为您取消成功",
    "已经退款成功",
    "仓库今天一定会处理",
    "我帮您操作好了",
    "已经加急处理完成",
]
ORDER_INFO_HINTS = ["订单号", "手机号尾号", "订单信息"]
TRANSFER_HINTS = ["人工", "人工客服"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate responses with LLM judge + hard-rule checks."
    )
    parser.add_argument(
        "--eval-set",
        default=str(ROOT / "data" / "golden_v1_eval.jsonl"),
        help="Golden eval set JSONL path.",
    )
    parser.add_argument(
        "--candidate-file",
        required=True,
        help="Candidate predictions JSONL path. Matched by id.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "eval_results" / "llm_judge_v1"),
        help="Directory for per-sample and summary outputs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Judge model name.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Judge API base URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature. Recommend 0 for stability.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Max tokens for judge response.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.5,
        help="Sleep time between judge API calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick preview runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files instead of appending timestamped files.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def get_api_key() -> str:
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError("Missing API key. Please set DEEPSEEK_API_KEY in .env or environment.")


def read_doc(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required doc not found: {path}")
    return path.read_text(encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
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


def index_by_id(rows: list[dict[str, Any]], source_name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"Every row in {source_name} must contain a non-empty string id")
        if row_id in indexed:
            raise ValueError(f"Duplicate id found in {source_name}: {row_id}")
        indexed[row_id] = row
    return indexed


def extract_candidate_text(row: dict[str, Any]) -> str:
    for key in ("prediction", "response", "target", "output", "answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    train_payload = row.get("train_payload")
    if isinstance(train_payload, dict):
        target = train_payload.get("target")
        if isinstance(target, str) and target.strip():
            return target.strip()
    raise ValueError(f"Could not extract candidate text from row id={row.get('id')}")


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def find_local_rule_errors(eval_row: dict[str, Any], candidate: str) -> list[str]:
    errors: list[str] = []
    payload = eval_row.get("train_payload", {})
    context = payload.get("context", {}) if isinstance(payload, dict) else {}
    meta = eval_row.get("annotation_meta", {}) if isinstance(eval_row.get("annotation_meta"), dict) else {}
    scenario = meta.get("scenario")
    decision_type = meta.get("decision_type")
    should_transfer = meta.get("should_transfer")
    shipping_status = context.get("shipping_status")
    logistics_status = context.get("logistics_status")
    order_id_provided = context.get("order_id_provided")
    is_opened = context.get("is_opened")
    has_quality_issue = context.get("has_quality_issue")

    for phrase in FORBIDDEN_PROMISES:
        if phrase in candidate:
            errors.append(f"contains_forbidden_promise:{phrase}")

    if order_id_provided == "是" and contains_any(candidate, ORDER_INFO_HINTS):
        errors.append("asks_for_order_info_again")

    if should_transfer is True and not contains_any(candidate, TRANSFER_HINTS):
        errors.append("should_transfer_but_no_human_handoff")
    if should_transfer is False and scenario != "转人工" and contains_any(candidate, TRANSFER_HINTS):
        errors.append("unexpected_human_handoff")

    if decision_type == "ask_followup" and "？" not in candidate and "吗" not in candidate and "请问" not in candidate:
        errors.append("ask_followup_without_clear_question")

    if scenario == "取消订单" and shipping_status == "未发货":
        if not contains_any(candidate, ["可以取消", "可取消"]):
            errors.append("cancel_unshipped_missing_can_cancel")
    if scenario == "取消订单" and shipping_status == "已发货":
        if not contains_any(candidate, ["无法直接取消", "不能直接取消", "不支持直接取消", "不可直接取消"]):
            errors.append("cancel_shipped_missing_cannot_cancel")
        if not contains_any(candidate, ["退货", "退货退款"]):
            errors.append("cancel_shipped_missing_return_guidance")
    if scenario == "修改地址" and (shipping_status == "已发货" or logistics_status == "已揽收"):
        if not contains_any(candidate, ["无法直接修改", "不能直接修改", "不支持修改", "不可修改"]):
            errors.append("change_address_missing_cannot_modify")
        if "物流" not in candidate:
            errors.append("change_address_missing_logistics_guidance")
    if scenario == "催发货":
        if contains_any(candidate, ["今天发", "马上发出", "一定发货", "今天给您发出"]):
            errors.append("urge_shipping_promises_shipment_time")
    if scenario == "退货条件判断" and is_opened == "是" and has_quality_issue == "否":
        if not contains_any(candidate, ["不支持售后", "不支持退货", "当前不支持", "无法办理", "无法处理"]):
            errors.append("return_opened_no_quality_missing_refusal")
    if scenario == "退货条件判断" and is_opened == "是" and has_quality_issue == "是":
        if not contains_any(candidate, ["售后", "质量问题", "退货退款"]):
            errors.append("return_opened_quality_missing_after_sales_path")
    return errors


def build_judge_messages(
    business_doc: str,
    eval_row: dict[str, Any],
    candidate_text: str,
) -> list[dict[str, str]]:
    system_prompt = (
        "你是中文电商客服模型评测裁判。"
        "你要根据业务规则、黄金样本和候选回复，对候选回复进行严格评估。"
        "只输出一个 JSON 对象，不要输出 Markdown，不要输出解释性前后缀。"
        "评分标准：所有分数均为 1-5 分，5 分最好。"
        "若候选回复出现明显业务误判、越权承诺、错误追问、错误转人工或编造事实，应判为 fatal_error=true，且 verdict=fail。"
        "不要因为候选回复措辞和黄金答案不同就扣分；重点看是否同样符合规则并解决问题。"
    )
    rubric = {
        "rule_correctness": "业务规则判断是否正确",
        "groundedness": "是否严格基于已知信息，没有编造状态、结果或时间承诺",
        "missing_info_handling": "信息不足时是否正确追问，信息充分时是否避免多余追问",
        "transfer_decision": "是否正确处理转人工边界",
        "actionability": "是否给出清晰可执行的下一步",
        "tone_control": "语气是否礼貌、专业、稳定，适配用户情绪",
        "overall_quality": "综合质量",
    }
    user_prompt = (
        "请根据以下材料评估候选回复。\n\n"
        "[业务规则摘要]\n"
        f"{business_doc}\n\n"
        "[黄金样本]\n"
        f"{json.dumps(eval_row, ensure_ascii=False, indent=2)}\n\n"
        "[候选回复]\n"
        f"{candidate_text}\n\n"
        "请输出 JSON，对以下字段打分并说明：\n"
        f"rubric={json.dumps(rubric, ensure_ascii=False)}\n"
        "输出格式必须为：\n"
        "{\n"
        '  "rule_correctness": 1-5,\n'
        '  "groundedness": 1-5,\n'
        '  "missing_info_handling": 1-5,\n'
        '  "transfer_decision": 1-5,\n'
        '  "actionability": 1-5,\n'
        '  "tone_control": 1-5,\n'
        '  "overall_quality": 1-5,\n'
        '  "fatal_error": true/false,\n'
        '  "verdict": "pass|borderline|fail",\n'
        '  "comments": "一句到三句中文简评",\n'
        '  "strengths": ["..."],\n'
        '  "issues": ["..."]\n'
        "}\n"
        "注意：若候选回复明显违反业务规则，fatal_error 必须为 true。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_chat_completions(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    parsed = json.loads(body)
    try:
        return parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {body}") from exc


def normalize_judge_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in SCORE_FIELDS:
        value = result.get(field)
        if not isinstance(value, int):
            raise ValueError(f"Judge field {field} must be int")
        if value < 1 or value > 5:
            raise ValueError(f"Judge field {field} must be between 1 and 5")
        normalized[field] = value

    fatal_error = result.get("fatal_error")
    if not isinstance(fatal_error, bool):
        raise ValueError("Judge field fatal_error must be boolean")
    verdict = result.get("verdict")
    if verdict not in {"pass", "borderline", "fail"}:
        raise ValueError("Judge field verdict must be pass/borderline/fail")
    comments = result.get("comments")
    if not isinstance(comments, str):
        raise ValueError("Judge field comments must be string")

    strengths = result.get("strengths", [])
    issues = result.get("issues", [])
    if not isinstance(strengths, list) or not all(isinstance(item, str) for item in strengths):
        raise ValueError("Judge field strengths must be array[string]")
    if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
        raise ValueError("Judge field issues must be array[string]")

    normalized["fatal_error"] = fatal_error
    normalized["verdict"] = verdict
    normalized["comments"] = comments.strip()
    normalized["strengths"] = strengths
    normalized["issues"] = issues
    return normalized


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_samples": 0,
            "score_means": {field: None for field in SCORE_FIELDS},
            "verdict_counts": {"pass": 0, "borderline": 0, "fail": 0},
            "fatal_error_count": 0,
            "hard_fail_count": 0,
            "strict_pass_count": 0,
            "strict_pass_rate": None,
            "scenario_breakdown": {},
        }

    score_means = {
        field: round(statistics.mean(row["judge"][field] for row in rows), 4)
        for field in SCORE_FIELDS
    }
    verdict_counts = {"pass": 0, "borderline": 0, "fail": 0}
    fatal_error_count = 0
    hard_fail_count = 0
    strict_pass_count = 0
    scenario_groups: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        verdict_counts[row["judge"]["verdict"]] += 1
        if row["judge"]["fatal_error"]:
            fatal_error_count += 1
        if row["local_rule_errors"]:
            hard_fail_count += 1
        if not row["local_rule_errors"] and not row["judge"]["fatal_error"] and row["judge"]["verdict"] == "pass":
            strict_pass_count += 1
        scenario = row["scenario"]
        scenario_groups.setdefault(scenario, []).append(row)

    scenario_breakdown: dict[str, Any] = {}
    for scenario, group in sorted(scenario_groups.items()):
        scenario_breakdown[scenario] = {
            "num_samples": len(group),
            "overall_quality_mean": round(statistics.mean(item["judge"]["overall_quality"] for item in group), 4),
            "rule_correctness_mean": round(statistics.mean(item["judge"]["rule_correctness"] for item in group), 4),
            "fatal_error_count": sum(1 for item in group if item["judge"]["fatal_error"]),
            "hard_fail_count": sum(1 for item in group if item["local_rule_errors"]),
        }

    return {
        "num_samples": len(rows),
        "score_means": score_means,
        "verdict_counts": verdict_counts,
        "fatal_error_count": fatal_error_count,
        "hard_fail_count": hard_fail_count,
        "strict_pass_count": strict_pass_count,
        "strict_pass_rate": round(strict_pass_count / len(rows), 4),
        "scenario_breakdown": scenario_breakdown,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    load_env_file(ENV_PATH)
    api_key = get_api_key()

    business_doc = read_doc(DOC_DIR / "业务分析文档.markdown")
    eval_rows = load_jsonl(Path(args.eval_set))
    candidate_rows = load_jsonl(Path(args.candidate_file))
    eval_by_id = index_by_id(eval_rows, "eval set")
    candidate_by_id = index_by_id(candidate_rows, "candidate file")

    selected_eval_rows = eval_rows[: args.limit] if args.limit else eval_rows
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, eval_row in enumerate(selected_eval_rows, start=1):
        row_id = eval_row["id"]
        scenario = eval_row.get("annotation_meta", {}).get("scenario", "未知")
        candidate_row = candidate_by_id.get(row_id)
        if candidate_row is None:
            errors.append({"id": row_id, "error": "missing_candidate_prediction"})
            print(f"[{index}/{len(selected_eval_rows)}] {row_id} -> missing candidate", flush=True)
            continue

        try:
            candidate_text = extract_candidate_text(candidate_row)
            local_rule_errors = find_local_rule_errors(eval_row, candidate_text)
            judge_raw = call_chat_completions(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                messages=build_judge_messages(business_doc, eval_row, candidate_text),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            judge_result = normalize_judge_result(json.loads(judge_raw))
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": row_id, "error": f"judge_error: {exc}"})
            print(f"[{index}/{len(selected_eval_rows)}] {row_id} -> judge error: {exc}", flush=True)
        else:
            result_row = {
                "id": row_id,
                "scenario": scenario,
                "golden_target": eval_row.get("train_payload", {}).get("target", ""),
                "candidate_target": candidate_text,
                "local_rule_errors": local_rule_errors,
                "judge": judge_result,
            }
            results.append(result_row)
            print(
                f"[{index}/{len(selected_eval_rows)}] {row_id} -> {judge_result['verdict']}"
                f" | local_errors={len(local_rule_errors)}"
                f" | overall={judge_result['overall_quality']}",
                flush=True,
            )
        if index < len(selected_eval_rows):
            time.sleep(args.sleep_seconds)

    summary = {
        "eval_set": str(Path(args.eval_set).resolve()),
        "candidate_file": str(Path(args.candidate_file).resolve()),
        "judge_model": args.model,
        "num_eval_rows": len(selected_eval_rows),
        "num_scored_rows": len(results),
        "num_errors": len(errors),
        "aggregate": aggregate_results(results),
        "errors": errors,
    }

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    detail_path = output_dir / "judge_results.jsonl"
    summary_path = output_dir / "judge_summary.json"
    if not args.overwrite:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        detail_path = output_dir / f"judge_results_{timestamp}.jsonl"
        summary_path = output_dir / f"judge_summary_{timestamp}.json"

    write_jsonl(detail_path, results)
    write_json(summary_path, summary)

    print("\nDone.")
    print(f"Scored rows: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Details: {detail_path}")
    print(f"Summary: {summary_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
