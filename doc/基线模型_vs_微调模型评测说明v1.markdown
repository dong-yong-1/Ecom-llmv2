# 《基线模型 vs 微调模型评测说明 v1》

当前仓库已提供脚本：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/compare_baseline_vs_finetuned_with_llm_judge.py`

## 1. 作用

这个脚本用于在同一份黄金评测集上，对：

- 基线模型输出
- 微调后模型输出

做逐条对比评测。

评测流程是：

1. 基线模型和微调模型都对同一条 eval 样本作答
2. 脚本先分别做本地硬规则检查
3. 再让 LLM judge 盲比两条回复谁更好
4. 最终统计微调模型胜率

这样更适合你的目标：不是只看微调模型“绝对分数”，而是直接看它是否稳定优于基线模型。

## 2. 输入文件

需要三份文件：

- 黄金评测集：`/Users/dongyong/Project/Trea_code/ecom-llmv2/data/golden_v1_eval.jsonl`
- 基线模型输出：例如 `/Users/dongyong/Project/Trea_code/ecom-llmv2/data/baseline_eval_predictions.jsonl`
- 微调模型输出：例如 `/Users/dongyong/Project/Trea_code/ecom-llmv2/data/finetuned_eval_predictions.jsonl`

基线和微调输出文件都需要是 `jsonl`，并且按 `id` 对齐。

最简单格式：

```json
{"id":"eval_cancel_unshipped_001","prediction":"您好，如果订单还未发货，一般是可以取消的，您可以先在订单详情页查看取消入口。"}
```

## 3. Judge 比较逻辑

脚本会把两条回复盲化成：

- `A`
- `B`

然后再让 LLM judge 比较：

- 哪条业务规则更正确
- 哪条更有帮助
- 哪条更安全
- 最终谁更好

同时会输出：

- `winner`
- `confidence`
- `reason`
- `critical_issues_a`
- `critical_issues_b`

为了减少位置偏差，脚本会根据样本 `id` 做固定盲排，不会永远让 baseline 在前面。

## 4. 本地硬规则检查

除了 LLM judge，脚本还会分别检查基线和微调回复是否出现明显红线问题，例如：

- 乱承诺
- 已发货取消却没说明不可直接取消
- 已揽收改地址却没引导联系物流
- 应转人工时未转人工
- 不该转人工时乱转人工
- 已有订单信息还重复索要订单号

## 5. 输出结果

脚本会同时输出两个胜者概念：

### 5.1 `judge_winner`

这是 LLM judge 直接判的赢家：

- `baseline`
- `finetuned`
- `tie`

### 5.2 `strict_winner`

这是“带硬规则纠偏”的赢家：

- 如果一方有明显本地硬规则错误，另一方没有
- 即使 judge 判平局，也会优先把胜利给没有硬错误的一方

这个指标更适合你这种业务规则强约束任务。

## 6. 核心指标

最终 summary 里重点看：

- `judge_win_rate_over_all`
- `judge_win_rate_excluding_ties`
- `strict_win_rate_excluding_ties`
- `scenario_breakdown`

其中最关键的是：

- `judge_win_rate_excluding_ties.finetuned`

这就是你要的“微调模型胜率”。

## 7. 运行方式

先跑小样本预览：

```bash
python3 /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/compare_baseline_vs_finetuned_with_llm_judge.py \
  --baseline-file /Users/dongyong/Project/Trea_code/ecom-llmv2/data/baseline_eval_predictions.jsonl \
  --finetuned-file /Users/dongyong/Project/Trea_code/ecom-llmv2/data/finetuned_eval_predictions.jsonl \
  --limit 5 \
  --sleep-seconds 2 \
  --overwrite
```

正式跑完整评测：

```bash
python3 /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/compare_baseline_vs_finetuned_with_llm_judge.py \
  --baseline-file /Users/dongyong/Project/Trea_code/ecom-llmv2/data/baseline_eval_predictions.jsonl \
  --finetuned-file /Users/dongyong/Project/Trea_code/ecom-llmv2/data/finetuned_eval_predictions.jsonl \
  --sleep-seconds 2 \
  --overwrite
```

## 8. 输出目录

默认输出目录：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/eval_results/pairwise_llm_judge_v1`

主要文件：

- `pairwise_results.jsonl`
- `pairwise_summary.json`

## 9. 和 verl 的衔接建议

你后面准备用 `verl` 做微调，这个评测方案是兼容的。

原因是：

- 训练数据仍然是标准的 `prompt -> response`
- 评测数据仍然是固定 `eval set`
- 基线模型和微调模型都只需要产出回复文本
- pairwise judge 脚本不依赖具体训练框架，只依赖最终预测文件

也就是说，后面如果你换成 `verl`：

- 训练阶段用 `verl`
- 推理阶段导出 `baseline_eval_predictions.jsonl`
- 推理阶段导出 `finetuned_eval_predictions.jsonl`
- 再直接跑当前 pairwise 评测脚本

评测层不需要重写。
