# 《AutoDL 训练与上传说明 v1》

你的代码仓库：[`dong-yong-1/Ecom-llmv2`](https://github.com/dong-yong-1/Ecom-llmv2)

现在我们切回 `uv + .venv` 方案，但把本地 Mac 和服务器 GPU 环境区分开：

- 本地 Mac：`scripts/setup_uv_env_local.sh`
- AutoDL 服务器：`scripts/setup_uv_env_autodl.sh`

这样做的目的很明确：

- 本地先做最小链路验证
- 服务器再跑正式 GPU 训练
- CPU / MPS / CUDA 的 torch 安装方式分开处理，避免环境混乱

## 1. 建议先上传的文件

- `scripts/setup_uv_env_local.sh`
- `scripts/setup_uv_env_autodl.sh`
- `scripts/run_trl_sft.sh`
- `scripts/run_trl_sft_autodl.sh`
- `scripts/run_trl_sft_smoke.sh`
- `scripts/run_trl_lora_ablation.py`
- `scripts/run_trl_lora_ablation_autodl.sh`
- `scripts/run_trl_sft.py`
- `scripts/prepare_trl_sft_dataset.py`
- `requirements-train.txt`
- `doc/TRL_SFT与LoRA消融说明v1.markdown`
- `doc/AutoDL训练与上传说明v1.markdown`

如果你当前目录还没有 git 仓库，可以用：

```bash
git init
git branch -M main
git remote add origin https://github.com/dong-yong-1/Ecom-llmv2.git
```

提交并推送：

```bash
git add scripts requirements-train.txt doc
git commit -m "Add uv-based local and AutoDL training workflow"
git push -u origin main
```

## 2. AutoDL 拉代码

```bash
git clone https://github.com/dong-yong-1/Ecom-llmv2.git
cd Ecom-llmv2
```

## 3. AutoDL 上创建 uv 环境

默认按 `cu121` 安装 GPU 版 torch：

```bash
bash scripts/setup_uv_env_autodl.sh
```

如果你的 CUDA 环境更适合 `cu118`：

```bash
export TORCH_BACKEND=cu118
bash scripts/setup_uv_env_autodl.sh
```

如果你只是想临时装 CPU 版：

```bash
export TORCH_BACKEND=cpu
bash scripts/setup_uv_env_autodl.sh
```

创建完成后：

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 4. AutoDL 正式训练

例如：

```bash
source .venv/bin/activate

export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
export OUTPUT_DIR=/root/autodl-tmp/ecom_trl_runs/qwen25_r16_attn_mlp
export LORA_R=16
export TARGET_MODULES=attn_mlp
export NUM_TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=8

bash scripts/run_trl_sft_autodl.sh
```

## 5. AutoDL 跑 LoRA 消融

```bash
source .venv/bin/activate

export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
bash scripts/run_trl_lora_ablation_autodl.sh
```

## 6. 本地 Mac 先做最小链路验证

本地不要直接拿大模型硬跑，先跑 smoke test：

```bash
bash scripts/setup_uv_env_local.sh
source .venv/bin/activate
bash scripts/run_trl_sft_smoke.sh
```

这个 smoke test 默认使用一个很小的 Hugging Face 模型来验证：

- 数据准备脚本能跑
- TRL 训练入口能跑
- LoRA 链路能跑
- 输出目录和 summary 能生成

等本地这个链路通了，再把同一套代码推到服务器上跑正式训练。

## 7. 评测阶段

现在评测统一收敛成一个脚本：

- 自动生成基座模型回答
- 自动生成微调模型回答
- 自动做 LLM pairwise judge
- 自动输出 JSON summary 和 Markdown 评测报告

例如：

```bash
source .venv/bin/activate

python scripts/compare_baseline_vs_finetuned_with_llm_judge.py \
  --baseline-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --finetuned-adapter-path model/trl_sft_run \
  --sleep-seconds 2 \
  --overwrite
```

输出目录默认在：

- `eval_results/full_eval_v1`

主要文件包括：

- `baseline_predictions.jsonl`
- `finetuned_predictions.jsonl`
- `pairwise_results.jsonl`
- `pairwise_summary.json`
- `evaluation_report.md`
