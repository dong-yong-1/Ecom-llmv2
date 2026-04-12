#!/usr/bin/env python3
"""Run LoRA rank/target-module ablations for TRL SFT."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNNER = ROOT / "scripts" / "run_trl_sft.sh"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "trl_lora_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRL LoRA ablation experiments.")
    parser.add_argument("--model-path", required=True, help="Base model path or HF repo.")
    parser.add_argument("--ranks", default="8,16,32", help="Comma-separated LoRA ranks.")
    parser.add_argument(
        "--target-modules",
        default="attention_only,attn_mlp,all_linear",
        help="Comma-separated target module aliases.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for ablation runs.")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER), help="TRL SFT runner script path.")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands.")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra args passed through to run_trl_sft.sh, can be repeated.",
    )
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    runner = Path(args.runner)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ranks = [int(item) for item in parse_csv(args.ranks)]
    target_modules = parse_csv(args.target_modules)

    experiments: list[dict[str, Any]] = []
    for rank, target in itertools.product(ranks, target_modules):
        exp_name = f"r{rank}_{target}"
        exp_output_dir = output_root / exp_name
        cmd = [
            str(runner),
            f"--output-dir={exp_output_dir}",
        ]
        env_overrides = {
            "MODEL_PATH": args.model_path,
            "OUTPUT_DIR": str(exp_output_dir),
            "LORA_R": str(rank),
            "TARGET_MODULES": target,
            "USE_LORA": "true",
        }
        cmd.extend(args.extra_arg)
        experiments.append(
            {
                "name": exp_name,
                "rank": rank,
                "target_modules": target,
                "output_dir": str(exp_output_dir),
                "env": env_overrides,
                "cmd": cmd,
            }
        )

    records: list[dict[str, Any]] = []
    for index, exp in enumerate(experiments, start=1):
        print(f"[{index}/{len(experiments)}] {exp['name']}")
        print("  env:", json.dumps(exp["env"], ensure_ascii=False))
        print("  cmd:", " ".join(exp["cmd"]))
        if args.dry_run:
            records.append({**exp, "status": "dry_run"})
            continue

        env = dict(**exp["env"])
        full_env = dict(**env)
        full_env.update({k: v for k, v in os.environ.items() if k not in full_env})
        result = subprocess.run(exp["cmd"], env=full_env, cwd=str(ROOT), check=False)
        record = {**exp, "returncode": result.returncode}
        summary_path = Path(exp["output_dir"]) / "run_summary.json"
        if summary_path.exists():
            try:
                record["run_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                record["run_summary_read_error"] = True
        records.append(record)
        if result.returncode != 0:
            print(f"  failed with returncode={result.returncode}")

    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
