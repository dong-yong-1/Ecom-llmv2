# 《TRL SFT 与 LoRA 消融说明 v1》

你当前阶段更适合：

- 先做 `SFT`
- 再做 `LoRA` 消融实验
- 最后用黄金评测集比较基线模型和微调模型

因此当前仓库新增了三部分脚本：

- 数据转换：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/prepare_trl_sft_dataset.py`
- 单次训练：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_sft.py`
- 训练启动封装：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_sft.sh`
- LoRA 消融：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_lora_ablation.py`

当前 uv 环境已经安装：

- `trl==1.1.0`
- `peft==0.18.1`
- `datasets==4.8.4`
- `transformers==5.5.3`
- `accelerate==1.13.0`

虚拟环境位置：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/.venv`

## 1. 为什么现在改用 TRL 做 SFT

你这一步目标是：

- 让模型先学会基础客服回复能力
- 先把业务规则、追问、转人工、承诺边界学稳
- 通过 LoRA 消融找到更合适的参数配置

这时用 `TRL + SFTTrainer` 会比直接上 RLHF / DPO 更简单，也更贴近当前目标。

## 2. 数据准备方式

当前 SFT 数据不是直接把整条 schema 原样喂给训练器，而是转换成：

- `prompt`
- `completion`

并保留对话式消息结构。

### 2.1 生成训练数据

```bash
source /Users/dongyong/Project/Trea_code/ecom-llmv2/.venv/bin/activate

python /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/prepare_trl_sft_dataset.py \
  --input /Users/dongyong/Project/Trea_code/ecom-llmv2/data/golden_v1_train.jsonl \
  --output-dir /Users/dongyong/Project/Trea_code/ecom-llmv2/data/trl_sft \
  --val-ratio 0.1
```

### 2.2 输出文件

默认输出：

- ` /Users/dongyong/Project/Trea_code/ecom-llmv2/data/trl_sft/train.jsonl`
- ` /Users/dongyong/Project/Trea_code/ecom-llmv2/data/trl_sft/val.jsonl`
- ` /Users/dongyong/Project/Trea_code/ecom-llmv2/data/trl_sft/dataset_summary.json`

## 3. 单次 SFT 训练

### 3.1 最小运行方式

```bash
source /Users/dongyong/Project/Trea_code/ecom-llmv2/.venv/bin/activate

export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct

bash /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_sft.sh
```

### 3.2 常用参数

你可以通过环境变量调整：

- `MODEL_PATH`
- `OUTPUT_DIR`
- `MAX_LENGTH`
- `LEARNING_RATE`
- `NUM_TRAIN_EPOCHS`
- `PER_DEVICE_TRAIN_BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`
- `USE_LORA`
- `LORA_R`
- `LORA_ALPHA`
- `TARGET_MODULES`

例如：

```bash
export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
export OUTPUT_DIR=/Users/dongyong/Project/Trea_code/ecom-llmv2/outputs/trl_sft_qwen25_r16_attn
export LORA_R=16
export TARGET_MODULES=attention_only
export NUM_TRAIN_EPOCHS=3

bash /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_sft.sh
```

## 4. LoRA 注入层别名

当前脚本支持这些别名：

- `attention_only`
  - `q_proj,k_proj,v_proj,o_proj`
- `mlp_only`
  - `gate_proj,up_proj,down_proj`
- `attn_mlp`
  - attention + mlp
- `all_linear`
  - 交给 PEFT 的 `all-linear`

如果你想手动指定，也可以直接传：

```bash
--target-modules q_proj,k_proj,v_proj,o_proj
```

## 5. LoRA 消融实验

你现在最适合做的第一轮消融，是同时试：

- rank：`8,16,32`
- 注入层：`attention_only,attn_mlp,all_linear`

这样总共 `9` 个实验，足够先看趋势。

### 5.1 干跑查看命令

```bash
source /Users/dongyong/Project/Trea_code/ecom-llmv2/.venv/bin/activate

python /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_lora_ablation.py \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --dry-run
```

### 5.2 正式跑消融

```bash
python /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_lora_ablation.py \
  --model-path Qwen/Qwen2.5-1.5B-Instruct
```

### 5.3 输出位置

默认输出根目录：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/outputs/trl_lora_ablation`

每个实验一个子目录，比如：

- `r8_attention_only`
- `r16_attn_mlp`
- `r32_all_linear`

同时会生成总表：

- `/Users/dongyong/Project/Trea_code/ecom-llmv2/outputs/trl_lora_ablation/ablation_summary.json`

## 6. 推荐的消融顺序

建议你不要一开始就跑特别大。

第一轮建议：

1. 先固定一个较小基座模型
2. 跑 `r=8/16/32`
3. 比较 `attention_only / attn_mlp / all_linear`
4. 用黄金评测集看哪组胜率最好

我建议你优先关注：

- `strict_win_rate_excluding_ties.finetuned`
- `fatal_error_count`
- `hard_fail_count`

不是只看 loss。

## 7. 你现在最合理的实验目标

你现在还不是在追求“最强模型”，而是在找：

- 哪个 rank 更适合你这个数据规模
- 哪组注入层最适合电商客服任务
- 哪组在规则稳定性上更好

这就是当前 SFT 阶段最重要的消融目标。

## 8. 推荐下一步

你现在最自然的下一步是：

1. 先选一个基座模型
2. 先跑单次 TRL SFT
3. 再跑第一轮 9 组 LoRA 消融
4. 最后接现有 pairwise 评测脚本看微调胜率
