#!/usr/bin/env python3
"""Unified evaluation entry for baseline vs fine-tuned models."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
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
SCORE_FIELDS = [
    "response_a_task_completion_score",
    "response_b_task_completion_score",
    "response_a_politeness_score",
    "response_b_politeness_score",
]
BOOL_FIELDS = [
    "response_a_task_completed",
    "response_b_task_completed",
    "response_a_rule_passed",
    "response_b_rule_passed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation script.")
    parser.add_argument("--mode", choices=["formal", "explore"], default="formal", help="Evaluation mode. Use formal for fixed, reproducible comparison.")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET), help="Golden eval set JSONL path.")
    parser.add_argument("--baseline-model-name-or-path", required=True, help="Baseline/base model path or HF repo.")
    parser.add_argument("--finetuned-model-name-or-path", default=None, help="Optional merged fine-tuned model path.")
    parser.add_argument("--finetuned-adapter-path", default=str(DEFAULT_FINETUNED_ADAPTER_PATH), help="LoRA adapter path.")
    parser.add_argument("--baseline-file", default=None, help="Optional existing baseline predictions JSONL.")
    parser.add_argument("--finetuned-file", default=None, help="Optional existing finetuned predictions JSONL.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--judge-model", default=DEFAULT_MODEL, help="Judge model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Judge API base URL.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature.")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Judge max tokens.")
    parser.add_argument("--timeout", type=int, default=180, help="Judge request timeout.")
    parser.add_argument("--sleep-seconds", type=float, default=1.5, help="Sleep time between judge calls.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation max new tokens.")
    parser.add_argument("--generation-temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--generation-top-p", type=float, default=1.0, help="Generation top-p.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Sampling seed for explore mode.")
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


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_doc(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def get_scenario(row: dict[str, Any]) -> str:
    return row.get("annotation_meta", {}).get("scenario", "未知")


def select_eval_rows(eval_rows: list[dict[str, Any]], limit: int | None, mode: str, sample_seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if limit is None or limit >= len(eval_rows):
        return eval_rows, {
            "selection_mode": "full",
            "num_selected": len(eval_rows),
            "selected_ids": [row["id"] for row in eval_rows],
        }

    if mode == "formal":
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in eval_rows:
            groups.setdefault(get_scenario(row), []).append(row)
        scenario_names = sorted(groups)
        selected: list[dict[str, Any]] = []
        remaining = limit
        remaining_scenarios = len(scenario_names)

        for scenario in scenario_names:
            items = sorted(groups[scenario], key=lambda row: stable_hash(row["id"]))
            take = min(len(items), max(1, remaining // remaining_scenarios))
            selected.extend(items[:take])
            remaining -= take
            remaining_scenarios -= 1

        if len(selected) < limit:
            selected_ids = {row["id"] for row in selected}
            leftovers = sorted(
                [row for row in eval_rows if row["id"] not in selected_ids],
                key=lambda row: stable_hash(row["id"]),
            )
            selected.extend(leftovers[: limit - len(selected)])

        selected = sorted(selected[:limit], key=lambda row: stable_hash(row["id"]))
        return selected, {
            "selection_mode": "formal_stratified_fixed",
            "num_selected": len(selected),
            "selected_ids": [row["id"] for row in selected],
        }

    rng = random.Random(sample_seed)
    selected = list(eval_rows)
    rng.shuffle(selected)
    selected = selected[:limit]
    return selected, {
        "selection_mode": "explore_random_seeded",
        "sample_seed": sample_seed,
        "num_selected": len(selected),
        "selected_ids": [row["id"] for row in selected],
    }


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
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


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


def load_generation_model(model_name_or_path: str, trust_remote_code: bool, adapter_path: str | None = None) -> tuple[Any, Any, str]:
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


def generate_predictions(model_name_or_path: str, eval_rows: list[dict[str, Any]], output_path: Path, trust_remote_code: bool, max_new_tokens: int, temperature: float, top_p: float, adapter_path: str | None = None) -> Path:
    tokenizer, model, device = load_generation_model(model_name_or_path, trust_remote_code, adapter_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    do_sample = temperature > 0
    for index, row in enumerate(eval_rows, start=1):
        prompt_text = build_prompt_text(build_prompt_messages(row), tokenizer)
        inputs = move_inputs(tokenizer(prompt_text, return_tensors="pt"), device)
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
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
    raise ValueError(f"Could not extract candidate text from row id={row.get('id')}")


def choose_blind_order(row_id: str) -> tuple[str, str]:
    digest = hashlib.md5(row_id.encode("utf-8")).hexdigest()
    return ("baseline", "finetuned") if int(digest, 16) % 2 == 0 else ("finetuned", "baseline")


def build_judge_messages(business_doc: str, eval_row: dict[str, Any], response_a: str, response_b: str) -> list[dict[str, str]]:
    system_prompt = (
        "你是中文电商客服模型评测裁判。"
        "你只评估五个维度：任务完成率、平台规则符合度、礼貌度、整体胜负、badcase。"
        "不要输出 Markdown，只输出一个 JSON 对象。"
    )
    user_prompt = (
        "请阅读以下信息并评测。\n\n"
        "[平台规则]\n"
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
        '  "response_a_task_completed": true,\n'
        '  "response_b_task_completed": true,\n'
        '  "response_a_rule_passed": true,\n'
        '  "response_b_rule_passed": true,\n'
        '  "response_a_task_completion_score": 1-5,\n'
        '  "response_b_task_completion_score": 1-5,\n'
        '  "response_a_politeness_score": 1-5,\n'
        '  "response_b_politeness_score": 1-5,\n'
        '  "reason": "简要说明胜负原因",\n'
        '  "badcase_tags_a": ["..."],\n'
        '  "badcase_tags_b": ["..."]\n'
        "}\n"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def call_chat_completions(api_key: str, base_url: str, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, timeout: int) -> str:
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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
    if not isinstance(confidence, int) or not 1 <= confidence <= 5:
        raise ValueError("confidence must be int 1-5")
    normalized: dict[str, Any] = {"winner": winner, "confidence": confidence}
    for field in BOOL_FIELDS:
        value = result.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be bool")
        normalized[field] = value
    for field in SCORE_FIELDS:
        value = result.get(field)
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{field} must be int 1-5")
        normalized[field] = value
    reason = result.get("reason")
    if not isinstance(reason, str):
        raise ValueError("reason must be string")
    normalized["reason"] = reason.strip()
    for field in ("badcase_tags_a", "badcase_tags_b"):
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


def extract_owner_value(row: dict[str, Any], owner: str, field_suffix: str) -> Any:
    if row["blind_assignment"]["response_a_owner"] == owner:
        return row["judge"][f"response_a_{field_suffix}"]
    return row["judge"][f"response_b_{field_suffix}"]


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_samples": 0, "win_rate_excluding_ties": {"baseline": None, "finetuned": None}, "task_completion_rate": {"baseline": None, "finetuned": None}, "rule_pass_rate": {"baseline": None, "finetuned": None}, "avg_task_completion_score": {"baseline": None, "finetuned": None}, "avg_politeness_score": {"baseline": None, "finetuned": None}, "avg_confidence": None}
    win_counts = {"baseline": 0, "finetuned": 0, "tie": 0}
    scenario_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        win_counts[row["winner"]] += 1
        scenario_groups.setdefault(row["scenario"], []).append(row)
    non_tie = win_counts["baseline"] + win_counts["finetuned"]

    def rate(owner: str, field_suffix: str) -> float:
        return round(sum(1 for row in rows if extract_owner_value(row, owner, field_suffix)) / len(rows), 4)

    def avg(owner: str, field_suffix: str) -> float:
        return round(statistics.mean(extract_owner_value(row, owner, field_suffix) for row in rows), 4)

    scenario_breakdown: dict[str, Any] = {}
    for scenario, group in sorted(scenario_groups.items()):
        scenario_win_counts = {"baseline": 0, "finetuned": 0, "tie": 0}
        for row in group:
            scenario_win_counts[row["winner"]] += 1
        scenario_non_tie = scenario_win_counts["baseline"] + scenario_win_counts["finetuned"]
        scenario_breakdown[scenario] = {
            "num_samples": len(group),
            "win_rate_excluding_ties": round(scenario_win_counts["finetuned"] / scenario_non_tie, 4) if scenario_non_tie else None,
            "task_completion_rate": {"baseline": round(sum(1 for row in group if extract_owner_value(row, "baseline", "task_completed")) / len(group), 4), "finetuned": round(sum(1 for row in group if extract_owner_value(row, "finetuned", "task_completed")) / len(group), 4)},
            "rule_pass_rate": {"baseline": round(sum(1 for row in group if extract_owner_value(row, "baseline", "rule_passed")) / len(group), 4), "finetuned": round(sum(1 for row in group if extract_owner_value(row, "finetuned", "rule_passed")) / len(group), 4)},
        }
    return {
        "num_samples": len(rows),
        "win_counts": win_counts,
        "win_rate_excluding_ties": {"baseline": round(win_counts["baseline"] / non_tie, 4) if non_tie else None, "finetuned": round(win_counts["finetuned"] / non_tie, 4) if non_tie else None},
        "task_completion_rate": {"baseline": rate("baseline", "task_completed"), "finetuned": rate("finetuned", "task_completed")},
        "rule_pass_rate": {"baseline": rate("baseline", "rule_passed"), "finetuned": rate("finetuned", "rule_passed")},
        "avg_task_completion_score": {"baseline": avg("baseline", "task_completion_score"), "finetuned": avg("finetuned", "task_completion_score")},
        "avg_politeness_score": {"baseline": avg("baseline", "politeness_score"), "finetuned": avg("finetuned", "politeness_score")},
        "avg_confidence": round(statistics.mean(row["judge"]["confidence"] for row in rows), 4),
        "scenario_breakdown": scenario_breakdown,
    }


def build_badcases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    badcases: list[dict[str, Any]] = []
    for row in rows:
        finetuned_task_completed = extract_owner_value(row, "finetuned", "task_completed")
        finetuned_rule_passed = extract_owner_value(row, "finetuned", "rule_passed")
        finetuned_politeness_score = extract_owner_value(row, "finetuned", "politeness_score")
        finetuned_badcase_tags = row["judge"]["badcase_tags_a"] if row["blind_assignment"]["response_a_owner"] == "finetuned" else row["judge"]["badcase_tags_b"]
        if row["winner"] == "finetuned" and finetuned_task_completed and finetuned_rule_passed and finetuned_politeness_score >= 4:
            continue
        badcases.append({
            "id": row["id"],
            "scenario": row["scenario"],
            "winner": row["winner"],
            "baseline_target": row["baseline_target"],
            "finetuned_target": row["finetuned_target"],
            "finetuned_task_completed": finetuned_task_completed,
            "finetuned_rule_passed": finetuned_rule_passed,
            "finetuned_politeness_score": finetuned_politeness_score,
            "finetuned_badcase_tags": finetuned_badcase_tags,
            "judge_reason": row["judge"]["reason"],
        })
    return badcases


def main() -> int:
    args = parse_args()
    load_env_file(ENV_PATH)
    api_key = get_api_key()
    eval_rows_all = load_jsonl(Path(args.eval_set))
    eval_rows, selection_meta = select_eval_rows(eval_rows_all, args.limit, args.mode, args.sample_seed)

    if args.mode == "formal":
        args.temperature = 0.0
        args.generation_temperature = 0.0
        args.generation_top_p = 1.0

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    baseline_path = Path(args.baseline_file) if args.baseline_file else output_dir / "baseline_predictions.jsonl"
    finetuned_path = Path(args.finetuned_file) if args.finetuned_file else output_dir / "finetuned_predictions.jsonl"
    detail_path = output_dir / "pairwise_results.jsonl"
    summary_path = output_dir / "pairwise_summary.json"
    badcase_path = output_dir / "badcases.jsonl"
    selection_path = output_dir / "selected_eval_ids.json"
    if not args.overwrite and args.mode != "formal":
        if args.baseline_file is None:
            baseline_path = output_dir / f"baseline_predictions_{timestamp}.jsonl"
        if args.finetuned_file is None:
            finetuned_path = output_dir / f"finetuned_predictions_{timestamp}.jsonl"
        detail_path = output_dir / f"pairwise_results_{timestamp}.jsonl"
        summary_path = output_dir / f"pairwise_summary_{timestamp}.json"
        badcase_path = output_dir / f"badcases_{timestamp}.jsonl"
        selection_path = output_dir / f"selected_eval_ids_{timestamp}.json"

    if args.baseline_file is None:
        print("[stage] generating baseline predictions", flush=True)
        generate_predictions(args.baseline_model_name_or_path, eval_rows, baseline_path, args.trust_remote_code, args.max_new_tokens, args.generation_temperature, args.generation_top_p)
    if args.finetuned_file is None:
        print("[stage] generating finetuned predictions", flush=True)
        finetuned_model_name_or_path = args.finetuned_model_name_or_path or args.baseline_model_name_or_path
        adapter_path = None if args.finetuned_model_name_or_path else args.finetuned_adapter_path
        generate_predictions(finetuned_model_name_or_path, eval_rows, finetuned_path, args.trust_remote_code, args.max_new_tokens, args.generation_temperature, args.generation_top_p, adapter_path)

    business_doc = read_doc(DOC_DIR / "业务分析文档.markdown")
    baseline_by_id = index_by_id(load_jsonl(baseline_path), "baseline file")
    finetuned_by_id = index_by_id(load_jsonl(finetuned_path), "finetuned file")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, eval_row in enumerate(eval_rows, start=1):
        row_id = eval_row["id"]
        scenario = eval_row.get("annotation_meta", {}).get("scenario", "未知")
        baseline_row = baseline_by_id.get(row_id)
        finetuned_row = finetuned_by_id.get(row_id)
        if baseline_row is None or finetuned_row is None:
            errors.append({"id": row_id, "error": "missing_predictions"})
            continue
        try:
            baseline_text = extract_candidate_text(baseline_row)
            finetuned_text = extract_candidate_text(finetuned_row)
            response_a_owner, response_b_owner = choose_blind_order(row_id)
            response_a = baseline_text if response_a_owner == "baseline" else finetuned_text
            response_b = baseline_text if response_b_owner == "baseline" else finetuned_text
            judge = normalize_judge_result(json.loads(call_chat_completions(api_key, args.base_url, args.judge_model, build_judge_messages(business_doc, eval_row, response_a, response_b), args.temperature, args.max_tokens, args.timeout)))
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": row_id, "error": str(exc)})
        else:
            results.append({
                "id": row_id,
                "scenario": scenario,
                "baseline_target": baseline_text,
                "finetuned_target": finetuned_text,
                "blind_assignment": {"response_a_owner": response_a_owner, "response_b_owner": response_b_owner},
                "judge": judge,
                "winner": resolve_winner(judge["winner"], response_a_owner, response_b_owner),
            })
            print(f"[judge {index}/{len(eval_rows)}] {row_id}", flush=True)
        if index < len(eval_rows):
            time.sleep(args.sleep_seconds)

    badcases = build_badcases(results)
    summary = {
        "mode": args.mode,
        "eval_set": str(Path(args.eval_set).resolve()),
        "evaluation_protocol": {
            "mode": args.mode,
            "judge_temperature": args.temperature,
            "generation_temperature": args.generation_temperature,
            "generation_top_p": args.generation_top_p,
            "max_new_tokens": args.max_new_tokens,
            "selection": selection_meta,
        },
        "baseline_model_name_or_path": args.baseline_model_name_or_path,
        "finetuned_model_name_or_path": args.finetuned_model_name_or_path or args.baseline_model_name_or_path,
        "finetuned_adapter_path": None if args.finetuned_model_name_or_path else (str(Path(args.finetuned_adapter_path).resolve()) if args.finetuned_adapter_path else None),
        "baseline_file": str(baseline_path.resolve()),
        "finetuned_file": str(finetuned_path.resolve()),
        "judge_model": args.judge_model,
        "num_eval_rows": len(eval_rows),
        "num_scored_rows": len(results),
        "num_errors": len(errors),
        "badcase_count": len(badcases),
        "aggregate": build_summary(results),
        "errors": errors,
    }
    write_jsonl(detail_path, results)
    write_json(summary_path, summary)
    write_jsonl(badcase_path, badcases)
    write_json(selection_path, selection_meta)
    print(f"Summary: {summary_path}")
    print(f"Badcases: {badcase_path}")
    print(f"Selected eval ids: {selection_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
