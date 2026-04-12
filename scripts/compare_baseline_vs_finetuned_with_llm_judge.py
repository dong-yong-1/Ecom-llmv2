#!/usr/bin/env python3
"""
Unified evaluation script:
1. generate baseline predictions when needed
2. generate fine-tuned predictions when needed
3. run local rule checks + LLM pairwise judging
4. write JSON summaries and a Markdown evaluation report
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "doc"
ENV_PATH = ROOT / ".env"
DEFAULT_EVAL_SET = ROOT / "data" / "golden_v1_eval.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "eval_results" / "full_eval_v1"
DEFAULT_FINETUNED_ADAPTER_PATH = ROOT / "model" / "trl_sft_run"
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
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
PAIRWISE_SCORE_FIELDS = [
    "response_a_rule_correctness",
    "response_b_rule_correctness",
    "response_a_helpfulness",
    "response_b_helpfulness",
    "response_a_safety",
    "response_b_safety",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end baseline vs fine-tuned evaluation.")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET), help="Golden eval set JSONL path.")
    parser.add_argument("--baseline-model-name-or-path", required=True, help="Baseline/base model path or HF repo.")
    parser.add_argument(
        "--finetuned-model-name-or-path",
        default=None,
        help="Optional merged fine-tuned model path. If omitted, baseline model + adapter path is used.",
    )
    parser.add_argument(
        "--finetuned-adapter-path",
        default=str(DEFAULT_FINETUNED_ADAPTER_PATH),
        help="LoRA adapter path for the fine-tuned model.",
    )
    parser.add_argument("--baseline-file", default=None, help="Optional existing baseline predictions JSONL.")
    parser.add_argument("--finetuned-file", default=None, help="Optional existing fine-tuned predictions JSONL.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for predictions, summaries, report.")
    parser.add_argument("--judge-model", default=DEFAULT_MODEL, help="Judge model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Judge API base URL.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature.")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Judge max tokens.")
    parser.add_argument("--timeout", type=int, default=180, help="Judge request timeout in seconds.")
    parser.add_argument("--sleep-seconds", type=float, default=1.5, help="Sleep time between judge API calls.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation max new tokens.")
    parser.add_argument("--generation-temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--generation-top-p", type=float, default=1.0, help="Generation top-p when sampling.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for preview runs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files.")
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_doc(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required doc not found: {path}")
    return path.read_text(encoding="utf-8")


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


def build_prompt_text(messages: list[dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def move_inputs(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    if device == "cuda":
        return {k: v.to("cuda") for k, v in inputs.items()}
    if device == "mps":
        return {k: v.to("mps") for k, v in inputs.items()}
    return inputs


def cleanup_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch, "mps", None) is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def load_generation_model(
    *,
    model_name_or_path: str,
    trust_remote_code: bool,
    adapter_path: str | None = None,
) -> tuple[Any, Any, str]:
    device = detect_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    if device == "mps":
        model = model.to("mps")
    elif device == "cpu":
        model = model.to("cpu")
    model.eval()
    return tokenizer, model, device


def generate_predictions(
    *,
    model_name_or_path: str,
    eval_rows: list[dict[str, Any]],
    output_path: Path,
    trust_remote_code: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    adapter_path: str | None = None,
) -> Path:
    tokenizer, model, device = load_generation_model(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
        adapter_path=adapter_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    do_sample = temperature > 0

    for index, row in enumerate(eval_rows, start=1):
        messages = build_prompt_messages(row)
        prompt_text = build_prompt_text(messages, tokenizer)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = move_inputs(inputs, device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        records.append({"id": row["id"], "prediction": prediction})
        print(f"[generate {index}/{len(eval_rows)}] {row['id']}", flush=True)

    write_jsonl(output_path, records)
    cleanup_model(model)
    return output_path


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


def choose_blind_order(row_id: str) -> tuple[str, str]:
    digest = hashlib.md5(row_id.encode("utf-8")).hexdigest()
    return ("baseline", "finetuned") if int(digest, 16) % 2 == 0 else ("finetuned", "baseline")


def build_judge_messages(business_doc: str, eval_row: dict[str, Any], response_a: str, response_b: str) -> list[dict[str, str]]:
    system_prompt = (
        "你是中文电商客服模型对比评测裁判。"
        "你要依据业务规则、黄金样本和两条候选回复，判断哪一条更好。"
        "你必须更重视规则正确性、边界控制、不编造、该追问时追问、该转人工时转人工。"
        "不要因为措辞更长或更像模板就偏向某一方。"
        "只输出一个 JSON 对象，不要输出 Markdown，不要输出解释前后缀。"
    )
    user_prompt = (
        "请阅读以下信息并做 pairwise 比较。\n\n"
        "[业务规则摘要]\n"
        f"{business_doc}\n\n"
        "[黄金样本]\n"
        f"{json.dumps(eval_row, ensure_ascii=False, indent=2)}\n\n"
        "[候选回复 A]\n"
        f"{response_a}\n\n"
        "[候选回复 B]\n"
        f"{response_b}\n\n"
        "请输出 JSON，格式必须为：\n"
        "{\n"
        '  "winner": "A|B|tie",\n'
        '  "confidence": 1-5,\n'
        '  "response_a_rule_correctness": 1-5,\n'
        '  "response_b_rule_correctness": 1-5,\n'
        '  "response_a_helpfulness": 1-5,\n'
        '  "response_b_helpfulness": 1-5,\n'
        '  "response_a_safety": 1-5,\n'
        '  "response_b_safety": 1-5,\n'
        '  "reason": "用中文简要说明哪条更好以及为什么",\n'
        '  "critical_issues_a": ["..."],\n'
        '  "critical_issues_b": ["..."]\n'
        "}\n"
        "若两者都差不多且都符合规则，可判 tie；若有一方明显违反规则，应优先让另一方获胜。"
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
    winner = result.get("winner")
    if winner not in {"A", "B", "tie"}:
        raise ValueError("winner must be A, B, or tie")
    confidence = result.get("confidence")
    if not isinstance(confidence, int) or confidence < 1 or confidence > 5:
        raise ValueError("confidence must be int 1-5")

    normalized: dict[str, Any] = {"winner": winner, "confidence": confidence}
    for field in PAIRWISE_SCORE_FIELDS:
        value = result.get(field)
        if not isinstance(value, int) or value < 1 or value > 5:
            raise ValueError(f"{field} must be int 1-5")
        normalized[field] = value

    reason = result.get("reason")
    if not isinstance(reason, str):
        raise ValueError("reason must be string")
    normalized["reason"] = reason.strip()

    for field in ("critical_issues_a", "critical_issues_b"):
        value = result.get(field, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be array[string]")
        normalized[field] = value
    return normalized


def resolve_winner(blind_winner: str, response_a_owner: str, response_b_owner: str) -> str:
    if blind_winner == "A":
        return response_a_owner
    if blind_winner == "B":
        return response_b_owner
    return "tie"


def compute_strict_winner(judge_winner: str, baseline_errors: list[str], finetuned_errors: list[str]) -> str:
    baseline_bad = bool(baseline_errors)
    finetuned_bad = bool(finetuned_errors)
    if baseline_bad and not finetuned_bad:
        return "finetuned"
    if finetuned_bad and not baseline_bad:
        return "baseline"
    return judge_winner


def aggregate_pairwise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_samples": 0,
            "judge_win_counts": {"baseline": 0, "finetuned": 0, "tie": 0},
            "judge_win_rate_over_all": {"baseline": None, "finetuned": None},
            "judge_win_rate_excluding_ties": {"baseline": None, "finetuned": None},
            "strict_win_counts": {"baseline": 0, "finetuned": 0, "tie": 0},
            "strict_win_rate_excluding_ties": {"baseline": None, "finetuned": None},
            "scenario_breakdown": {},
            "avg_confidence": None,
            "baseline_hard_fail_count": 0,
            "finetuned_hard_fail_count": 0,
        }

    judge_counts = {"baseline": 0, "finetuned": 0, "tie": 0}
    strict_counts = {"baseline": 0, "finetuned": 0, "tie": 0}
    scenario_groups: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        judge_counts[row["judge_winner"]] += 1
        strict_counts[row["strict_winner"]] += 1
        scenario_groups.setdefault(row["scenario"], []).append(row)

    non_tie_judge = judge_counts["baseline"] + judge_counts["finetuned"]
    non_tie_strict = strict_counts["baseline"] + strict_counts["finetuned"]
    scenario_breakdown: dict[str, Any] = {}
    for scenario, group in sorted(scenario_groups.items()):
        scenario_judge_counts = {"baseline": 0, "finetuned": 0, "tie": 0}
        for row in group:
            scenario_judge_counts[row["judge_winner"]] += 1
        non_tie = scenario_judge_counts["baseline"] + scenario_judge_counts["finetuned"]
        scenario_breakdown[scenario] = {
            "num_samples": len(group),
            "judge_win_counts": scenario_judge_counts,
            "finetuned_win_rate_excluding_ties": (
                round(scenario_judge_counts["finetuned"] / non_tie, 4) if non_tie else None
            ),
            "baseline_hard_fail_count": sum(1 for row in group if row["baseline_local_rule_errors"]),
            "finetuned_hard_fail_count": sum(1 for row in group if row["finetuned_local_rule_errors"]),
        }

    return {
        "num_samples": len(rows),
        "judge_win_counts": judge_counts,
        "judge_win_rate_over_all": {
            "baseline": round(judge_counts["baseline"] / len(rows), 4),
            "finetuned": round(judge_counts["finetuned"] / len(rows), 4),
        },
        "judge_win_rate_excluding_ties": {
            "baseline": round(judge_counts["baseline"] / non_tie_judge, 4) if non_tie_judge else None,
            "finetuned": round(judge_counts["finetuned"] / non_tie_judge, 4) if non_tie_judge else None,
        },
        "strict_win_counts": strict_counts,
        "strict_win_rate_excluding_ties": {
            "baseline": round(strict_counts["baseline"] / non_tie_strict, 4) if non_tie_strict else None,
            "finetuned": round(strict_counts["finetuned"] / non_tie_strict, 4) if non_tie_strict else None,
        },
        "avg_confidence": round(statistics.mean(row["judge"]["confidence"] for row in rows), 4),
        "baseline_hard_fail_count": sum(1 for row in rows if row["baseline_local_rule_errors"]),
        "finetuned_hard_fail_count": sum(1 for row in rows if row["finetuned_local_rule_errors"]),
        "scenario_breakdown": scenario_breakdown,
    }


def build_report(summary: dict[str, Any], top_rows: list[dict[str, Any]]) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# 评测报告",
        "",
        "## 1. 评测配置",
        "",
        f"- 评测集：`{summary['eval_set']}`",
        f"- 基座模型：`{summary['baseline_model_name_or_path']}`",
        f"- 微调模型：`{summary['finetuned_model_name_or_path']}`",
        f"- LoRA 适配器：`{summary['finetuned_adapter_path'] or '无'}`",
        f"- 基座预测文件：`{summary['baseline_file']}`",
        f"- 微调预测文件：`{summary['finetuned_file']}`",
        f"- 裁判模型：`{summary['judge_model']}`",
        "",
        "## 2. 总体结果",
        "",
        f"- 总样本数：{aggregate['num_samples']}",
        f"- 微调模型 judge 胜率（去平局）：{aggregate['judge_win_rate_excluding_ties']['finetuned']}",
        f"- 微调模型 strict 胜率（去平局）：{aggregate['strict_win_rate_excluding_ties']['finetuned']}",
        f"- 基座硬错误数：{aggregate['baseline_hard_fail_count']}",
        f"- 微调硬错误数：{aggregate['finetuned_hard_fail_count']}",
        f"- 平均裁判置信度：{aggregate['avg_confidence']}",
        "",
        "## 3. 分场景结果",
        "",
    ]
    for scenario, item in aggregate["scenario_breakdown"].items():
        lines.append(
            f"- {scenario}：样本 {item['num_samples']}，微调去平局胜率 {item['finetuned_win_rate_excluding_ties']}，"
            f"基座硬错误 {item['baseline_hard_fail_count']}，微调硬错误 {item['finetuned_hard_fail_count']}"
        )
    lines.extend(["", "## 4. 典型样本", ""])
    if not top_rows:
        lines.append("- 无")
    else:
        for row in top_rows:
            lines.extend(
                [
                    f"### {row['id']} / {row['scenario']}",
                    "",
                    f"- 裁判结果：{row['judge_winner']} / strict={row['strict_winner']}",
                    f"- 裁判理由：{row['judge']['reason']}",
                    f"- 基座错误：{', '.join(row['baseline_local_rule_errors']) if row['baseline_local_rule_errors'] else '无'}",
                    f"- 微调错误：{', '.join(row['finetuned_local_rule_errors']) if row['finetuned_local_rule_errors'] else '无'}",
                    f"- 基座回答：{row['baseline_target']}",
                    f"- 微调回答：{row['finetuned_target']}",
                    "",
                ]
            )
    if summary["errors"]:
        lines.extend(["## 5. 错误记录", ""])
        for error in summary["errors"][:20]:
            lines.append(f"- {error['id']}: {error['error']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    load_env_file(ENV_PATH)
    api_key = get_api_key()

    eval_rows = load_jsonl(Path(args.eval_set))
    if args.limit:
        eval_rows = eval_rows[: args.limit]

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    baseline_path = Path(args.baseline_file) if args.baseline_file else output_dir / "baseline_predictions.jsonl"
    finetuned_path = Path(args.finetuned_file) if args.finetuned_file else output_dir / "finetuned_predictions.jsonl"
    detail_path = output_dir / "pairwise_results.jsonl"
    summary_path = output_dir / "pairwise_summary.json"
    report_path = output_dir / "evaluation_report.md"
    if not args.overwrite:
        if args.baseline_file is None:
            baseline_path = output_dir / f"baseline_predictions_{timestamp}.jsonl"
        if args.finetuned_file is None:
            finetuned_path = output_dir / f"finetuned_predictions_{timestamp}.jsonl"
        detail_path = output_dir / f"pairwise_results_{timestamp}.jsonl"
        summary_path = output_dir / f"pairwise_summary_{timestamp}.json"
        report_path = output_dir / f"evaluation_report_{timestamp}.md"

    if args.baseline_file is None:
        print("[stage] generating baseline predictions", flush=True)
        generate_predictions(
            model_name_or_path=args.baseline_model_name_or_path,
            eval_rows=eval_rows,
            output_path=baseline_path,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            temperature=args.generation_temperature,
            top_p=args.generation_top_p,
        )
    if args.finetuned_file is None:
        print("[stage] generating finetuned predictions", flush=True)
        finetuned_model_name_or_path = args.finetuned_model_name_or_path or args.baseline_model_name_or_path
        adapter_path = args.finetuned_adapter_path
        if args.finetuned_model_name_or_path:
            adapter_path = None
        generate_predictions(
            model_name_or_path=finetuned_model_name_or_path,
            eval_rows=eval_rows,
            output_path=finetuned_path,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            temperature=args.generation_temperature,
            top_p=args.generation_top_p,
            adapter_path=adapter_path,
        )

    business_doc = read_doc(DOC_DIR / "业务分析文档.markdown")
    baseline_rows = load_jsonl(baseline_path)
    finetuned_rows = load_jsonl(finetuned_path)
    baseline_by_id = index_by_id(baseline_rows, "baseline file")
    finetuned_by_id = index_by_id(finetuned_rows, "finetuned file")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, eval_row in enumerate(eval_rows, start=1):
        row_id = eval_row["id"]
        scenario = eval_row.get("annotation_meta", {}).get("scenario", "未知")
        baseline_row = baseline_by_id.get(row_id)
        finetuned_row = finetuned_by_id.get(row_id)
        if baseline_row is None or finetuned_row is None:
            missing = []
            if baseline_row is None:
                missing.append("baseline")
            if finetuned_row is None:
                missing.append("finetuned")
            errors.append({"id": row_id, "error": f"missing_predictions:{','.join(missing)}"})
            print(f"[judge {index}/{len(eval_rows)}] {row_id} -> missing {','.join(missing)}", flush=True)
            continue

        try:
            baseline_text = extract_candidate_text(baseline_row)
            finetuned_text = extract_candidate_text(finetuned_row)
            baseline_errors = find_local_rule_errors(eval_row, baseline_text)
            finetuned_errors = find_local_rule_errors(eval_row, finetuned_text)
            response_a_owner, response_b_owner = choose_blind_order(row_id)
            response_a = baseline_text if response_a_owner == "baseline" else finetuned_text
            response_b = baseline_text if response_b_owner == "baseline" else finetuned_text
            judge_raw = call_chat_completions(
                api_key=api_key,
                base_url=args.base_url,
                model=args.judge_model,
                messages=build_judge_messages(business_doc, eval_row, response_a, response_b),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            judge = normalize_judge_result(json.loads(judge_raw))
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": row_id, "error": f"judge_error: {exc}"})
            print(f"[judge {index}/{len(eval_rows)}] {row_id} -> judge error: {exc}", flush=True)
        else:
            judge_winner = resolve_winner(judge["winner"], response_a_owner, response_b_owner)
            strict_winner = compute_strict_winner(judge_winner, baseline_errors, finetuned_errors)
            result_row = {
                "id": row_id,
                "scenario": scenario,
                "baseline_target": baseline_text,
                "finetuned_target": finetuned_text,
                "baseline_local_rule_errors": baseline_errors,
                "finetuned_local_rule_errors": finetuned_errors,
                "blind_assignment": {
                    "response_a_owner": response_a_owner,
                    "response_b_owner": response_b_owner,
                },
                "judge": judge,
                "judge_winner": judge_winner,
                "strict_winner": strict_winner,
            }
            results.append(result_row)
            print(
                f"[judge {index}/{len(eval_rows)}] {row_id} -> judge={judge_winner} | strict={strict_winner}"
                f" | baseline_err={len(baseline_errors)} | finetuned_err={len(finetuned_errors)}",
                flush=True,
            )
        if index < len(eval_rows):
            time.sleep(args.sleep_seconds)

    summary = {
        "eval_set": str(Path(args.eval_set).resolve()),
        "baseline_model_name_or_path": args.baseline_model_name_or_path,
        "finetuned_model_name_or_path": args.finetuned_model_name_or_path or args.baseline_model_name_or_path,
        "finetuned_adapter_path": (
            None
            if args.finetuned_model_name_or_path
            else (str(Path(args.finetuned_adapter_path).resolve()) if args.finetuned_adapter_path else None)
        ),
        "baseline_file": str(baseline_path.resolve()),
        "finetuned_file": str(finetuned_path.resolve()),
        "judge_model": args.judge_model,
        "num_eval_rows": len(eval_rows),
        "num_scored_rows": len(results),
        "num_errors": len(errors),
        "aggregate": aggregate_pairwise(results),
        "errors": errors,
    }

    write_jsonl(detail_path, results)
    write_json(summary_path, summary)
    report_path.write_text(build_report(summary, results[: min(5, len(results))]), encoding="utf-8")

    print("\nDone.")
    print(f"Baseline predictions: {baseline_path}")
    print(f"Finetuned predictions: {finetuned_path}")
    print(f"Details: {detail_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
