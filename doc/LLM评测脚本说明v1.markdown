# 《LLM 评测脚本说明 v1》

当前仓库已提供脚本：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/evaluate_with_llm_judge.py`

## 1. 作用

该脚本用于对候选模型输出进行自动评测，采用两层机制：

- 本地硬规则检查
- LLM-as-a-Judge 打分

输入包括：

- 黄金评测集：`/Users/dongyong/Project/Trea_code/ecom-llmv2/data/golden_v1_eval.jsonl`
- 候选输出文件：你自己的模型在 eval set 上生成的回复

输出包括：

- 逐条评测结果 `judge_results.jsonl`
- 汇总结果 `judge_summary.json`

## 2. 候选输出文件格式

候选输出文件需要是 `jsonl`，并且每行至少包含：

- `id`
- 候选回复文本

候选回复文本支持以下任一字段：

- `prediction`
- `response`
- `target`
- `output`
- `answer`

或者支持这种结构：

```json
{
  "id": "eval_cancel_unshipped_001",
  "train_payload": {
    "target": "您好，如果订单还未发货，一般是可以取消的，您可以先到订单详情页查看取消入口。"
  }
}
```

最简单推荐格式：

```json
{"id":"eval_cancel_unshipped_001","prediction":"您好，如果订单还未发货，一般是可以取消的，您可以先到订单详情页查看取消入口。"}
```

## 3. 评测内容

### 3.1 本地硬规则检查

脚本会先检查一些明确红线，例如：

- 是否出现乱承诺
- 已提供订单信息时是否还重复索要订单信息
- 已发货取消是否明确说不可直接取消
- 已揽收改地址是否明确引导联系物流
- 拆封无质量问题是否明确拒绝售后
- 应转人工时是否没有提人工
- 不该转人工时是否乱转人工

### 3.2 LLM Judge 维度

Judge 默认打以下分数，范围均为 `1-5`：

- `rule_correctness`
- `groundedness`
- `missing_info_handling`
- `transfer_decision`
- `actionability`
- `tone_control`
- `overall_quality`

并额外输出：

- `fatal_error`
- `verdict` (`pass` / `borderline` / `fail`)
- `comments`
- `strengths`
- `issues`

## 4. 运行方式

先确认 `.env` 中已配置 `DEEPSEEK_API_KEY`。

### 4.1 先跑少量预览

```bash
python3 /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/evaluate_with_llm_judge.py \
  --candidate-file /Users/dongyong/Project/Trea_code/ecom-llmv2/data/my_eval_predictions.jsonl \
  --limit 5 \
  --sleep-seconds 2 \
  --overwrite
```

### 4.2 正式跑完整评测集

```bash
python3 /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/evaluate_with_llm_judge.py \
  --candidate-file /Users/dongyong/Project/Trea_code/ecom-llmv2/data/my_eval_predictions.jsonl \
  --sleep-seconds 2 \
  --overwrite
```

## 5. 输出文件

默认输出目录：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/eval_results/llm_judge_v1`

默认会生成：

- 逐条结果：`judge_results.jsonl`
- 汇总结果：`judge_summary.json`

如果不加 `--overwrite`，脚本会自动加时间戳，避免覆盖旧结果。

## 6. 汇总结果怎么看

`judge_summary.json` 里重点关注：

- `score_means`
- `verdict_counts`
- `fatal_error_count`
- `hard_fail_count`
- `strict_pass_rate`
- `scenario_breakdown`

建议优先看：

1. `hard_fail_count` 是否高
2. `fatal_error_count` 是否高
3. 哪些 `scenario` 分数最低
4. `strict_pass_rate` 是否达到你预期

## 7. 推荐使用方式

最适合你的流程是：

1. 用黄金评测集生成模型候选回复
2. 用这个脚本跑一轮 LLM 评测
3. 查看最低分样本和 fatal error 样本
4. 结合 bad case 回修训练数据或 prompt

这个脚本适合做第一版自动评测基线，后面还可以继续叠加：

- 重复度分析
- 模板味分析
- 场景级别对比报表
- 多 Judge 投票
