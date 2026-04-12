# 《基线模型 vs 微调模型统一评测说明 v1》

当前统一评测脚本：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/compare_baseline_vs_finetuned_with_llm_judge.py`

## 1. 现在的评测方式

评测不再拆成“先单独生成预测，再单独比较”两段流程。

现在一个脚本会直接完成：

1. 基座模型生成回答
2. 微调模型生成回答
3. 本地硬规则检查
4. LLM judge 盲比
5. 输出 JSON summary
6. 输出 Markdown 评测报告

## 2. 最小运行方式

如果你当前微调产物保存在：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/model/trl_sft_run`

那么直接运行：

```bash
python3 /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/compare_baseline_vs_finetuned_with_llm_judge.py \
  --baseline-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --finetuned-adapter-path /Users/dongyong/Project/Trea_code/ecom-llmv2/model/trl_sft_run \
  --sleep-seconds 2 \
  --overwrite
```

## 3. 这个脚本会生成什么

默认输出目录：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/eval_results/full_eval_v1`

主要产物：

- `baseline_predictions.jsonl`
- `finetuned_predictions.jsonl`
- `pairwise_results.jsonl`
- `pairwise_summary.json`
- `evaluation_report.md`

## 4. 核心指标

最终重点看：

- `judge_win_rate_excluding_ties.finetuned`
- `strict_win_rate_excluding_ties.finetuned`
- `baseline_hard_fail_count`
- `finetuned_hard_fail_count`

其中：

- `judge_win_rate_excluding_ties.finetuned` 是微调模型相对基座模型的直接胜率
- `strict_win_rate_excluding_ties.finetuned` 是加入硬规则纠偏后的更稳指标

## 5. 评测报告里会写什么

Markdown 报告会包含：

- 评测配置
- 总体结果
- 分场景结果
- 典型样本
- 错误记录

所以你后面看实验结果时，不需要只盯 JSON，也可以直接先读报告。

## 6. 预跑方式

如果你只是先看链路是否正常，可以先限制样本数：

```bash
python3 /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/compare_baseline_vs_finetuned_with_llm_judge.py \
  --baseline-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --finetuned-adapter-path /Users/dongyong/Project/Trea_code/ecom-llmv2/model/trl_sft_run \
  --limit 10 \
  --sleep-seconds 2 \
  --overwrite
```
