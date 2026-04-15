#!/usr/bin/env python3
"""Generate a higher-quality holdout eval set with DeepSeek."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_RUN_DIR = ROOT / "data" / "eval_generation" / "holdout_v3"
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "generated_eval.jsonl"
DEFAULT_REJECTED = DEFAULT_RUN_DIR / "rejected_eval.jsonl"
DEFAULT_SUMMARY = DEFAULT_RUN_DIR / "summary.json"
DEFAULT_REFERENCE_FILES = [
    ROOT / "data" / "golden_v1_train.jsonl",
    ROOT / "data" / "golden_v1_eval.jsonl",
    ROOT / "data" / "golden_v2_holdout_eval.jsonl",
]
DEFAULT_SYSTEM = (
    "你是电商客服助手，需要礼貌、专业、基于已知信息回答，不编造订单状态、物流进度、处理结果或承诺时间；"
    "信息不足时先追问，超出权限或高风险场景时转人工。"
)

SCENARIO_RULES: dict[str, dict[str, list[str]]] = {
    "催发货": {
        "subtypes": [
            "状态未知且未提供订单信息，必须先追问订单号或手机号尾号",
            "已知未发货，只能反馈催促，不能承诺发货时间或保证加急成功",
            "用户情绪焦虑甚至抱怨，回复要安抚但仍不能乱承诺",
            "多轮承接：先追问信息，用户补充后再反馈催促",
            "用户明确要求人工或投诉升级，应转人工",
        ]
    },
    "取消订单": {
        "subtypes": [
            "未发货，可取消，必须给操作路径",
            "已发货且运输中，不可取消，必须引导收到货后申请退货退款",
            "已发货且已揽收，不可取消，仍然只能引导退货退款",
            "状态未知，必须先追问是否已发货",
            "多轮承接：先判断状态，再给取消或退货路径",
        ]
    },
    "修改地址": {
        "subtypes": [
            "未发货，可修改地址，可给操作路径或追问新地址",
            "已发货，不可直接修改，必须引导顾客自行联系承运物流",
            "已揽收，不可直接修改，必须引导顾客自行联系承运物流",
            "状态未知，必须追问是否已发货，必要时补问新地址",
            "多轮承接：先追问状态或新地址，再给出正确路径",
        ]
    },
    "查询物流进度": {
        "subtypes": [
            "物流运输中，只能说明当前状态，不能承诺送达时间",
            "用户焦虑催到货，回复要安抚但不能承诺主动通知",
            "多轮承接：先确认物流状态，再解释运输中含义",
            "用户反复追问什么时候到，仍然不能给具体时间",
        ]
    },
    "退款流程说明": {
        "subtypes": [
            "信息不足，必须先追问是否签收、是否存在质量问题",
            "已签收且有质量问题，应说明走质量问题售后流程",
            "已签收且无质量问题，需要继续判断是否拆封",
            "未签收场景，回复不能编造成已签收",
            "多轮承接：先追问，再基于补充信息说明路径",
        ]
    },
    "退货条件判断": {
        "subtypes": [
            "已签收且未拆封，可申请退货退款",
            "已拆封且有质量问题，可走质量问题售后",
            "已拆封且无质量问题，不支持退货退款",
            "信息不足，必须追问是否拆封、是否有质量问题",
            "多轮承接：先追问，再给出明确判断",
        ]
    },
}

ALLOWED_DECISIONS = {"direct_answer", "ask_followup", "transfer_human", "rule_explanation"}
ALLOWED_USER_EMOTIONS = {"正常", "焦虑", "生气", "着急", "不满", "未知"}
RETRYABLE_EXCEPTIONS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    OSError,
    socket.timeout,
)
HANDLED_EXCEPTIONS = (ValueError, KeyError, json.JSONDecodeError) + RETRYABLE_EXCEPTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a high-quality holdout eval set with DeepSeek.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Output directory for this eval generation run.")
    parser.add_argument("--output-file", default=None, help="Accepted eval JSONL. Default: <run-dir>/generated_eval.jsonl")
    parser.add_argument("--rejected-file", default=None, help="Rejected JSONL. Default: <run-dir>/rejected_eval.jsonl")
    parser.add_argument("--summary-file", default=None, help="Summary JSON. Default: <run-dir>/summary.json")
    parser.add_argument(
        "--counts",
        default="催发货:8,取消订单:8,修改地址:8,查询物流进度:8,退款流程说明:8,退货条件判断:8",
        help="Per-scenario target counts, e.g. 催发货:8,取消订单:8",
    )
    parser.add_argument("--multi-turn-ratio", type=float, default=0.4, help="Target fraction of multi-turn examples.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek generation model.")
    parser.add_argument("--review-model", default=DEFAULT_MODEL, help="DeepSeek review model.")
    parser.add_argument("--temperature", type=float, default=0.45, help="Generation temperature.")
    parser.add_argument("--review-temperature", type=float, default=0.0, help="Review temperature.")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="HTTP timeout.")
    parser.add_argument("--sleep-seconds", type=float, default=0.8, help="Sleep between API calls.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--reference-files",
        default=",".join(str(path) for path in DEFAULT_REFERENCE_FILES),
        help="Comma-separated train/eval files used for de-duplication.",
    )
    parser.add_argument("--disable-llm-review", action="store_true", help="Skip second-pass LLM quality review.")
    parser.add_argument("--dry-run", action="store_true", help="Print generation plan only.")
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {idx} of {path}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_counts(raw: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid counts entry: {part}")
        scenario, count = part.split(":", 1)
        scenario = scenario.strip()
        if scenario not in SCENARIO_RULES:
            raise ValueError(f"Unsupported scenario in counts: {scenario}")
        parsed_count = int(count.strip())
        if parsed_count <= 0:
            raise ValueError(f"Count must be positive for scenario: {scenario}")
        counts[scenario] = parsed_count
    if not counts:
        raise ValueError("counts cannot be empty")
    return counts


def parse_reference_files(raw: str) -> list[Path]:
    files: list[Path] = []
    for part in raw.split(","):
        stripped = part.strip()
        if stripped:
            files.append(Path(stripped))
    return files


def pick_subtype_for_style(subtypes: list[str], turn_style: str, idx: int) -> str:
    if turn_style == "多轮":
        candidates = [item for item in subtypes if "多轮" in item]
    else:
        candidates = [item for item in subtypes if "多轮" not in item]
    if not candidates:
        candidates = subtypes
    return candidates[idx % len(candidates)]


def sample_signature(row: dict[str, Any]) -> str:
    payload = row["train_payload"]
    normalized = {
        "scenario": row["annotation_meta"].get("scenario"),
        "context": payload.get("context", {}),
        "history": payload.get("history", []),
        "target": payload.get("target", ""),
    }
    return hashlib.md5(json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def last_user_message(row: dict[str, Any]) -> str:
    history = row["train_payload"].get("history", [])
    for turn in reversed(history):
        if turn.get("role") == "user":
            return (turn.get("content") or "").strip()
    return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def build_reference_sets(files: list[Path]) -> tuple[set[str], set[str], set[str], set[str]]:
    used_ids: set[str] = set()
    used_signatures: set[str] = set()
    used_last_user: set[str] = set()
    used_targets: set[str] = set()
    for path in files:
        for row in load_jsonl(path):
            if row.get("id"):
                used_ids.add(str(row["id"]))
            try:
                used_signatures.add(sample_signature(row))
                msg = last_user_message(row)
                if msg:
                    used_last_user.add(normalize_text(msg))
                target = row.get("train_payload", {}).get("target", "")
                if target:
                    used_targets.add(normalize_text(target))
            except Exception:
                continue
    return used_ids, used_signatures, used_last_user, used_targets


def next_row_id(scenario: str, used_ids: set[str], idx: int) -> str:
    prefix = {
        "催发货": "holdout_urge",
        "取消订单": "holdout_cancel",
        "修改地址": "holdout_change",
        "查询物流进度": "holdout_logistics",
        "退款流程说明": "holdout_refund",
        "退货条件判断": "holdout_return",
    }[scenario]
    candidate = f"{prefix}_{idx:06d}"
    while candidate in used_ids:
        idx += 1
        candidate = f"{prefix}_{idx:06d}"
    used_ids.add(candidate)
    return candidate


def policy_block() -> str:
    return (
        "你在生成的是中文电商客服 holdout 评测集，不是训练集。要求更注重覆盖、难度和泛化，而不是模板重复。\n"
        "必须严格遵守业务规则：\n"
        "1. 取消订单：未发货可取消；已发货不可取消，应引导收到货后申请退货退款；状态未知先追问。\n"
        "2. 催发货：只能查询、解释、反馈催促，不能承诺发货时间、不能保证已加急；信息不足时先追问订单号或手机号尾号。\n"
        "3. 修改地址：未发货可修改地址；已发货/已揽收不可直接修改地址，需引导顾客自行联系承运物流。\n"
        "4. 查询物流进度：运输中不能承诺送达时间，不能编造主动通知。\n"
        "5. 退款流程说明：优先区分是否签收、是否有质量问题；不要编造状态。\n"
        "6. 退货条件判断：围绕是否签收、是否拆封、是否有质量问题判断；拆封后无质量问题不支持售后，拆封后有质量问题可走质量问题售后。\n"
        "7. 信息不足先追问；权限/投诉升级/用户明确要求人工时可以转人工。\n"
        "8. 输出必须是 schema_v1 单条 JSON 对象，不要 markdown，不要解释。\n"
        "9. 必须覆盖更自然的口语表达、边界表达、焦虑/投诉情绪；单轮和多轮都要有。\n"
        "10. system 字段统一使用固定值，不要改写。\n"
    )


def build_prompt(scenario: str, turn_style: str, subtype: str, idx: int) -> str:
    payload = {
        "task": "请生成 1 条新的高质量 schema_v1 电商客服评测样本。",
        "eval_goal": "这是一份 holdout eval，用于验证模型泛化能力。不要写得过于模板化，不要直接复述已有数据常见说法。",
        "requirements": [
            f"场景必须是：{scenario}",
            f"对话形式必须是：{turn_style}",
            f"重点覆盖子类型：{subtype}",
            "必须优先体现中文真实客服场景中的自然表达和边界条件。",
            "若为单轮：history 里只保留当前用户问题。",
            "若为多轮：history 至少 3 轮，且最后一轮必须是用户发言，target 只对应当前轮客服回复。",
            "上下文字段必须与场景匹配，不能乱加无关字段。",
            "target 必须严格符合平台规则，不编造状态/结果/承诺时间。",
            "若用户明确要求人工，或场景属于投诉升级，可以设置转人工。",
            "system 必须严格等于固定值。",
            "输出只包含 1 个 JSON 对象。",
        ],
        "fixed_system": DEFAULT_SYSTEM,
        "output_schema_hint": {
            "id": f"placeholder_{idx}",
            "version": "schema_v1",
            "train_payload": {
                "system": DEFAULT_SYSTEM,
                "context": {"user_request": scenario},
                "history": [{"role": "user", "content": "用户问题"}],
                "target": "当前轮客服回复",
            },
            "annotation_meta": {
                "scenario": scenario,
                "intent": "中文意图",
                "decision_type": "direct_answer / ask_followup / transfer_human / rule_explanation",
                "should_transfer": False,
                "missing_slots": [],
                "policy_basis": ["中文规则依据"],
                "risk_flags": [],
                "quality_check": {
                    "is_consistent_with_context": True,
                    "contains_forbidden_promise": False,
                    "needs_revision": False,
                },
            },
        },
    }
    return policy_block() + "\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_review_prompt(row: dict[str, Any]) -> str:
    return (
        policy_block()
        + "\n你现在是严格的评测集审稿人。请审查下面这 1 条样本。"
        + "\n只输出 1 个 JSON 对象，不要输出样本原文，不要解释，不要 markdown。"
        + "\nscores 需要按样本实际质量给出 1 到 5 的整数，不要机械照抄示例值。"
        + "\n输出格式固定为："
        + '\n{"accept": true, "reason": "一句中文结论", "main_issue": "若拒绝则写主要问题，否则为空字符串", "scores": {"rule_pass": 1, "context_consistency": 1, "naturalness": 1, "eval_value": 1}}'
        + "\n待审样本如下：\n"
        + json.dumps(row, ensure_ascii=False, indent=2)
    )


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


def normalize_accept_value(review: dict[str, Any]) -> bool:
    candidates = [review.get("accept"), review.get("accepted"), review.get("pass"), review.get("verdict")]
    for value in candidates:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "accept", "accepted", "pass", "passed", "通过", "是"}:
                return True
            if normalized in {"false", "no", "n", "reject", "rejected", "fail", "failed", "拒绝", "否", "不通过"}:
                return False
    raise ValueError(f"LLM review result missing recognizable accept field: {json.dumps(review, ensure_ascii=False)[:300]}")




def validate_turn_style(row: dict[str, Any], turn_style: str) -> None:
    history = row["train_payload"].get("history") or []
    if turn_style == "单轮":
        if len(history) != 1 or history[0].get("role") != "user":
            raise ValueError("Single-turn sample should contain exactly one user turn.")
    if turn_style == "多轮":
        if len(history) < 3:
            raise ValueError("Multi-turn sample should contain at least 3 turns.")
        if history[-1].get("role") != "user":
            raise ValueError("Multi-turn history should end with a user turn.")
        if not any(turn.get("role") == "assistant" for turn in history):
            raise ValueError("Multi-turn sample should contain assistant turns.")


def validate_history(history: list[dict[str, Any]]) -> None:
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a non-empty list")
    for turn in history:
        if turn.get("role") not in {"user", "assistant"}:
            raise ValueError("history role must be user or assistant")
        if not (turn.get("content") or "").strip():
            raise ValueError("history content cannot be empty")


def validate_generated_row(
    row: dict[str, Any],
    scenario: str,
    turn_style: str,
    used_last_user: set[str],
    used_targets: set[str],
) -> None:
    if row.get("version") != "schema_v1":
        raise ValueError("version must be schema_v1")
    train_payload = row.get("train_payload")
    meta = row.get("annotation_meta")
    if not isinstance(train_payload, dict) or not isinstance(meta, dict):
        raise ValueError("Missing train_payload or annotation_meta")
    if meta.get("scenario") != scenario:
        raise ValueError("Generated scenario mismatch")

    history = train_payload.get("history") or []
    target = (train_payload.get("target") or "").strip()
    context = train_payload.get("context") or {}
    validate_history(history)
    if not target:
        raise ValueError("Missing target")
    validate_turn_style(row, turn_style)

    system_text = (train_payload.get("system") or "").strip()
    if system_text != DEFAULT_SYSTEM:
        raise ValueError("system must match fixed instruction")

    if meta.get("decision_type") not in ALLOWED_DECISIONS:
        raise ValueError("Unsupported decision_type")
    if not isinstance(meta.get("should_transfer"), bool):
        raise ValueError("should_transfer must be bool")
    if not isinstance(meta.get("missing_slots"), list):
        raise ValueError("missing_slots must be list")
    if not isinstance(meta.get("policy_basis"), list) or not meta.get("policy_basis"):
        raise ValueError("policy_basis must be a non-empty list")
    quality_check = meta.get("quality_check")
    if not isinstance(quality_check, dict):
        raise ValueError("quality_check must be dict")
    if quality_check.get("needs_revision"):
        raise ValueError("needs_revision must be false")
    if quality_check.get("contains_forbidden_promise"):
        raise ValueError("contains_forbidden_promise must be false")

    emotion = context.get("user_emotion")
    if emotion is not None and emotion not in ALLOWED_USER_EMOTIONS:
        raise ValueError("Unsupported user_emotion")

    normalized_last_user = normalize_text(last_user_message(row))
    if normalized_last_user in used_last_user:
        raise ValueError("Exact last-user overlap with existing train/eval data")
    if normalize_text(target) in used_targets:
        raise ValueError("Target reply overlaps with existing train/eval data")

    if meta.get("should_transfer") and meta.get("decision_type") != "transfer_human":
        raise ValueError("should_transfer true must align with transfer_human")
    if meta.get("decision_type") == "transfer_human" and not meta.get("should_transfer"):
        raise ValueError("transfer_human must set should_transfer true")

    if scenario == "催发货":
        status = context.get("shipping_status")
        if status == "未知":
            if "订单号" not in target and "手机号尾号" not in target and "手机尾号" not in target:
                raise ValueError("Unknown urge case must ask for order info")
        forbidden = ["今天发", "明天发", "今晚发", "已经加急", "保证发货", "一定发出", "马上发出"]
        if any(word in target for word in forbidden):
            raise ValueError("Urge shipment reply contains forbidden promise")

    elif scenario == "取消订单":
        status = context.get("shipping_status")
        if status == "未发货" and ("取消入口" not in target and "订单详情页" not in target and "页面提示" not in target):
            raise ValueError("Unshipped cancel case should provide operation path")
        if status == "已发货" and "退货退款" not in target:
            raise ValueError("Shipped cancel case should guide return/refund")
        if status == "未知" and "发货" not in target:
            raise ValueError("Unknown cancel case should ask whether shipped")

    elif scenario == "修改地址":
        status = context.get("shipping_status")
        if status == "未发货" and ("修改地址" not in target and "订单详情页" not in target):
            raise ValueError("Unshipped change-address case should allow modification")
        if status == "已发货":
            if all(word not in target for word in ["联系物流", "承运物流", "物流方", "物流公司"]):
                raise ValueError("Shipped change-address case should guide logistics")
        if status == "未知" and "发货" not in target:
            raise ValueError("Unknown change-address case should ask shipping status")

    elif scenario == "查询物流进度":
        forbidden = ["2-3天", "明天到", "今天到", "预计", "会通知您", "稍后通知", "帮您催派送"]
        if any(word in target for word in forbidden):
            raise ValueError("Transit logistics case should not promise time or proactive notification")

    elif scenario == "退款流程说明":
        is_signed = context.get("is_signed")
        has_quality = context.get("has_quality_issue")
        if "未知" in {is_signed, has_quality} and ("签收" not in target or "质量" not in target):
            raise ValueError("Refund flow with unknown slots should ask signed/quality")

    elif scenario == "退货条件判断":
        is_opened = context.get("is_opened")
        has_quality = context.get("has_quality_issue")
        if "未知" in {is_opened, has_quality} and ("拆封" not in target or "质量" not in target):
            raise ValueError("Return judgment with unknown slots should ask opened/quality")
        if is_opened == "是" and has_quality == "否" and all(word not in target for word in ["不支持", "不能", "无法"]):
            raise ValueError("Opened no-quality case should reject return")


def make_chat_request(api_key: str, base_url: str, model: str, prompt: str, temperature: float, timeout_seconds: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "你是一个严谨的数据合成与质检助手，专门生成高质量中文电商客服评测集。"},
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def llm_review_row(
    api_key: str,
    base_url: str,
    model: str,
    row: dict[str, Any],
    temperature: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    response = make_chat_request(
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=build_review_prompt(row),
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )
    content = response["choices"][0]["message"]["content"]
    review = extract_json_object(content)
    review["accept"] = normalize_accept_value(review)
    return review


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    run_dir = Path(args.run_dir)
    output_file = Path(args.output_file) if args.output_file else run_dir / "generated_eval.jsonl"
    rejected_file = Path(args.rejected_file) if args.rejected_file else run_dir / "rejected_eval.jsonl"
    summary_file = Path(args.summary_file) if args.summary_file else run_dir / "summary.json"

    counts = parse_counts(args.counts)
    reference_files = parse_reference_files(args.reference_files)
    rng = random.Random(args.seed)

    plan: list[tuple[str, str, str]] = []
    for scenario, count in counts.items():
        subtypes = SCENARIO_RULES[scenario]["subtypes"]
        multi_count = round(count * args.multi_turn_ratio)
        single_count = count - multi_count
        styles = ["单轮"] * single_count + ["多轮"] * multi_count
        rng.shuffle(styles)
        for idx, style in enumerate(styles):
            subtype = pick_subtype_for_style(subtypes, style, idx)
            plan.append((scenario, style, subtype))
    rng.shuffle(plan)

    print(f"[info] planned_examples={len(plan)}")
    print(f"[info] scenario_counts={counts}")
    print(f"[info] llm_review={'off' if args.disable_llm_review else 'on'}")
    if args.dry_run:
        for scenario, style, subtype in plan:
            print(f"- {scenario} | {style} | {subtype}")
        return 0

    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in environment or .env")

    used_ids, used_signatures, used_last_user, used_targets = build_reference_sets(reference_files)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    next_idx = 1

    for scenario, turn_style, subtype in plan:
        prompt = build_prompt(scenario, turn_style, subtype, next_idx)
        for attempt in range(1, args.max_retries + 2):
            raw_content = ""
            try:
                response = make_chat_request(api_key, base_url, args.model, prompt, args.temperature, args.timeout_seconds)
                raw_content = response["choices"][0]["message"]["content"]
                row = extract_json_object(raw_content)
                validate_generated_row(row, scenario, turn_style, used_last_user, used_targets)

                sig = sample_signature(row)
                if sig in used_signatures:
                    raise ValueError("Duplicate semantic signature with existing train/eval data")

                if not args.disable_llm_review:
                    review = llm_review_row(
                        api_key=api_key,
                        base_url=base_url,
                        model=args.review_model,
                        row=row,
                        temperature=args.review_temperature,
                        timeout_seconds=args.timeout_seconds,
                    )
                    if not review.get("accept"):
                        raise ValueError(f"LLM review rejected sample: {review.get('main_issue') or review.get('reason')}")
                    row.setdefault("annotation_meta", {})["eval_review"] = review

                row["id"] = next_row_id(scenario, used_ids, next_idx)
                used_signatures.add(sig)
                used_last_user.add(normalize_text(last_user_message(row)))
                used_targets.add(normalize_text(row["train_payload"]["target"]))
                accepted.append(row)
                print(f"[ok] {row['id']} | {scenario} | {turn_style}")
                next_idx += 1
                break
            except HANDLED_EXCEPTIONS as exc:
                print(f"[warn] {scenario} | {turn_style} | attempt={attempt} | error={exc}", file=sys.stderr)
                if attempt > args.max_retries:
                    rejected.append(
                        {
                            "scenario": scenario,
                            "turn_style": turn_style,
                            "subtype": subtype,
                            "error": str(exc),
                            "raw_output": raw_content,
                        }
                    )
                    break
                time.sleep(args.sleep_seconds)
        time.sleep(args.sleep_seconds)

    write_jsonl(output_file, accepted)
    write_jsonl(rejected_file, rejected)
    summary = {
        "model": args.model,
        "review_model": None if args.disable_llm_review else args.review_model,
        "counts_request": counts,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "scenario_counts": dict(Counter(row["annotation_meta"]["scenario"] for row in accepted)),
        "turn_style_counts": {
            "single_turn": sum(1 for row in accepted if len(row["train_payload"]["history"]) == 1),
            "multi_turn": sum(1 for row in accepted if len(row["train_payload"]["history"]) > 1),
        },
        "output_file": str(output_file.resolve()),
        "rejected_file": str(rejected_file.resolve()),
        "reference_files": [str(path.resolve()) for path in reference_files if path.exists()],
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
