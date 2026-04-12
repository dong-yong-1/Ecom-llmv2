# 《AutoDL 训练与上传说明 v1》

你的代码仓库：[`dong-yong-1/Ecom-llmv2`](https://github.com/dong-yong-1/Ecom-llmv2)

当前仓库已经补齐了更适合 AutoDL 的 SFT 训练与评测脚本。按你的要求，AutoDL 上统一使用系统 Python + pip，不使用 `uv`，也不使用 `.venv`。

## 1. 本地准备并上传 GitHub

建议先把下面这些内容推到仓库：

- `scripts/prepare_trl_sft_dataset.py`
- `scripts/run_trl_sft.py`
- `scripts/run_trl_sft.sh`
- `scripts/run_trl_sft_autodl.sh`
- `scripts/run_trl_lora_ablation.py`
- `scripts/run_trl_lora_ablation_autodl.sh`
- `scripts/generate_eval_predictions.py`
- `scripts/setup_pip_env_autodl.sh`
- `requirements-train.txt`
- `requirements-train-autodl.txt`
- `.gitignore`
- `doc/TRL_SFT与LoRA消融说明v1.markdown`
- `doc/AutoDL训练与上传说明v1.markdown`

如果你当前目录还没有 git 仓库，可以用：

```bash
git init
git branch -M main
git remote add origin https://github.com/dong-yong-1/Ecom-llmv2.git
```

查看状态：

```bash
git status
```

添加文件：

```bash
git add .gitignore requirements-train.txt requirements-train-autodl.txt scripts doc data/golden_v1_train.jsonl data/golden_v1_eval.jsonl data/trl_sft
```

提交：

```bash
git commit -m "Add TRL SFT training, LoRA ablation, and AutoDL scripts"
```

首次推送：

```bash
git push -u origin main
```

如果远端仓库已经有内容，先拉一下再推：

```bash
git pull --rebase origin main
git push -u origin main
```

## 2. 在 AutoDL 上拉代码

```bash
git clone https://github.com/dong-yong-1/Ecom-llmv2.git
cd Ecom-llmv2
```

## 3. 先确认当前 Python 有没有 torch

先不要进任何虚拟环境，直接检查系统 Python：

```bash
python3 -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

如果这里能正常打印版本号，说明当前 Python 可以直接用。

如果这里报 `ModuleNotFoundError: No module named 'torch'`，说明你当前这个 Python 环境里没有 torch。这时有两种可能：

- 你切到了一个不带 torch 的 Python 环境
- 当前 AutoDL 镜像本身就没把 torch 装到这个 Python 里

## 4. 在 AutoDL 上安装训练依赖

按你的要求，这里直接往当前 Python 环境里装，不创建 `.venv`。

```bash
bash scripts/setup_pip_env_autodl.sh
```

这个脚本会：

- 使用 `python3 -m pip` 安装依赖
- 默认不安装 torch
- 最后打印当前实际使用的 Python 路径和 torch 检测结果

如果你后面换了一个没有 torch 的镜像，再手动开启 torch 安装：

```bash
export INSTALL_TORCH=true
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
bash scripts/setup_pip_env_autodl.sh
```

如果你的 CUDA 版本不是 `cu121`，再把下载源改成对应版本，比如：

```bash
export INSTALL_TORCH=true
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
bash scripts/setup_pip_env_autodl.sh
```

## 5. 单次 SFT 训练

例如：

```bash
export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
export OUTPUT_DIR=/root/autodl-tmp/ecom_trl_runs/qwen25_r16_attn_mlp
export LORA_R=16
export TARGET_MODULES=attn_mlp
export NUM_TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=8

bash scripts/run_trl_sft_autodl.sh
```

如果你想显式指定 Python：

```bash
export PYTHON_BIN=python3
bash scripts/run_trl_sft_autodl.sh
```

## 6. LoRA 消融实验

第一轮推荐：

- rank: `8,16,32`
- target modules: `attention_only,attn_mlp,all_linear`

运行：

```bash
export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
bash scripts/run_trl_lora_ablation_autodl.sh
```

如果只想先看命令：

```bash
python3 scripts/run_trl_lora_ablation.py --model-path Qwen/Qwen2.5-1.5B-Instruct --dry-run
```

## 7. 生成评测集预测

### 7.1 基线模型预测

```bash
python3 scripts/generate_eval_predictions.py \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --eval-set data/golden_v1_eval.jsonl \
  --output data/baseline_eval_predictions.jsonl
```

### 7.2 微调模型预测

如果你用 LoRA adapter：

```bash
python3 scripts/generate_eval_predictions.py \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path /root/autodl-tmp/ecom_trl_runs/qwen25_r16_attn_mlp/checkpoint-xxx \
  --eval-set data/golden_v1_eval.jsonl \
  --output data/finetuned_eval_predictions.jsonl
```

## 8. 跑基线 vs 微调评测

```bash
python3 scripts/compare_baseline_vs_finetuned_with_llm_judge.py \
  --baseline-file data/baseline_eval_predictions.jsonl \
  --finetuned-file data/finetuned_eval_predictions.jsonl \
  --sleep-seconds 2 \
  --overwrite
```

核心指标重点看：

- `judge_win_rate_excluding_ties.finetuned`
- `strict_win_rate_excluding_ties.finetuned`
- `hard_fail_count`
- `fatal_error_count`

## 9. 推荐实践

建议在 AutoDL 上这样推进：

1. 先跑一组单实验，确保训练链路通
2. 再跑第一轮 9 组 LoRA 消融
3. 每组都导出 eval predictions
4. 最后做 pairwise 胜率比较

这样你就能比较清楚地找到：

- 哪个 rank 更适合你当前数据规模
- 哪组 LoRA 注入层更适合电商客服规则任务
