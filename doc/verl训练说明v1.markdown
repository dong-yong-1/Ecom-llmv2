# 《verl 训练说明 v1》

当前仓库已提供两部分训练相关脚本：

- 数据转换脚本：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/prepare_verl_sft_dataset.py`
- 训练启动脚本：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_verl_sft.sh`

当前本地还已创建 uv 虚拟环境：

- 虚拟环境：`/Users/dongyong/Project/Trea_code/ecom-llmv2/.venv`
- 已安装关键包：`verl 0.7.1`、`pyarrow 23.0.1`、`pandas 3.0.2`、`torch 2.11.0`

## 1. 设计思路

当前项目的黄金训练数据是 `schema_v1` JSONL，格式适合标注和质检，但不适合直接喂给 `verl`。

因此训练流程拆成两步：

1. 先把 `schema_v1` 转成 `verl` 更容易消费的 SFT 数据
2. 再用 `verl` 启动训练

## 2. 数据转换脚本

脚本：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/prepare_verl_sft_dataset.py`

### 2.1 输入

默认输入：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/data/golden_v1_train.jsonl`

### 2.2 输出

默认输出目录：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/data/verl_sft`

默认会产出：

- `train.parquet`
- `val.parquet`
- `dataset_summary.json`

可选再产出：

- `train.jsonl`
- `val.jsonl`

### 2.3 转换后的字段

转换后每条样本至少包含：

- `id`
- `messages`
- `question`
- `answer`
- `scenario`
- `intent`
- `decision_type`
- `should_transfer`
- `context_json`

其中：

- `messages` = `system + history + assistant(target)` 的多轮消息结构，供 `verl` 默认的 `MultiTurnSFTDataset` 使用
- `question` = `系统指令 + 上下文 + 对话历史 + 请输出当前轮客服回复`，主要作为调试和兼容其他训练框架的辅助列
- `answer` = 原始黄金回复

### 2.4 运行方式

```bash
python3 /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/prepare_verl_sft_dataset.py \
  --input /Users/dongyong/Project/Trea_code/ecom-llmv2/data/golden_v1_train.jsonl \
  --output-dir /Users/dongyong/Project/Trea_code/ecom-llmv2/data/verl_sft \
  --val-ratio 0.1
```

如果你还想同时写出 JSONL 预览：

```bash
python3 /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/prepare_verl_sft_dataset.py \
  --jsonl-preview
```

### 2.5 依赖说明

写 parquet 需要本地环境至少安装下面之一：

- `pyarrow`
- `pandas`（并具备 parquet 写入能力）

如果都没有，脚本会直接报错提醒。

## 3. verl 训练启动脚本

脚本：

`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_verl_sft.sh`

这个脚本会先自动调用数据转换脚本，然后再启动 `verl` SFT 训练。

### 3.1 使用前需要设置

最少需要设置：

- `MODEL_PATH`

例如：

```bash
export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
```

建议先激活虚拟环境：

```bash
source /Users/dongyong/Project/Trea_code/ecom-llmv2/.venv/bin/activate
```

### 3.2 常用可调参数

你可以通过环境变量调这些参数：

- `NPROC_PER_NODE`
- `MICRO_BATCH_SIZE`
- `TRAIN_BATCH_SIZE`
- `MAX_EPOCHS`
- `LR`
- `MAX_LENGTH`
- `USE_LORA`
- `LORA_RANK`
- `OUTPUT_DIR`
- `TRAINER_DEVICE`

### 3.3 最小运行示例

```bash
export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
export NPROC_PER_NODE=1
export MICRO_BATCH_SIZE=2
export TRAIN_BATCH_SIZE=64
export MAX_EPOCHS=3
export TRAINER_DEVICE=cpu

bash /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_verl_sft.sh
```

### 3.4 增加额外 verl 参数

如果你的本地 `verl` 版本还需要额外 hydra override，可以直接接在命令后面：

```bash
bash /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_verl_sft.sh \
  trainer.project_name=ecom_llm_v1 \
  trainer.experiment_name=my_first_verl_run
```

## 4. 当前脚本默认策略

### 4.1 数据侧

- 默认从 `golden_v1_train.jsonl` 划出 `10%` 作为 `val`
- 划分是按 `scenario` 分组后稳定切分，避免验证集被单一场景占满
- `golden_v1_eval.jsonl` 不参与训练，用于后续正式评测

### 4.2 训练侧

- 默认走 `verl 0.7.1` 的 `MultiTurnSFTDataset`
- 默认使用 `messages_key=messages`
- `question/answer` 字段只作为辅助调试列保留
- 默认支持 `LoRA`

## 5. 为什么这样设计

原因很简单：

- 你的标注数据结构化程度高，适合做数据工程
- 但 `verl` 训练更适合吃比较标准的 `question -> answer` 数据
- 所以最稳的是保留原始 `schema_v1` 作为主数据，再派生一份训练格式数据

这样做的好处是：

- 标注和训练解耦
- 后续如果换训练框架，不用重做黄金数据
- 评测仍然沿用现有 eval 流程

## 6. 当前边界

这版脚本是第一版训练脚本，目标是先把训练链路跑通。

当前还没有做：

- 多轮 SFT 专项处理
- packing 优化
- 自动恢复训练
- wandb / tensorboard logger
- RL 或 RM 流程

这些后面都可以继续补。

## 7. 当前机器上的现实建议

你现在这台机器是 macOS，本地环境更适合：

- 学习 `verl` 配置结构
- 准备训练数据
- 做小规模 CPU 级别试验

如果后面要认真跑中大型模型训练，更推荐放到 Linux + CUDA 环境。

## 8. 推荐下一步

你现在最自然的下一步是：

1. 先准备好本地 `verl` 环境
2. 跑通 `prepare_verl_sft_dataset.py`
3. 用一个小模型先试跑 `run_verl_sft.sh`
4. 训练后导出 eval 输出
5. 再接现有 pairwise judge 评测脚本看微调胜率
