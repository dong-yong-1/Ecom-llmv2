# 《TRL SFT 与 LoRA 消融说明 v1》

你当前阶段更适合：

- 先做 `SFT`
- 再做 `LoRA` 消融实验
- 最后用黄金评测集比较基线模型和微调模型

当前训练链路已经整理成两套环境：

- 本地 Mac 验证链路：`uv + .venv + local torch`
- AutoDL 正式训练：`uv + .venv + GPU torch`

## 1. 当前关键脚本

- 数据转换：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/prepare_trl_sft_dataset.py`
- 单次训练：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_sft.py`
- 训练封装：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_sft.sh`
- 本地 smoke test：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_sft_smoke.sh`
- LoRA 消融：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_lora_ablation.py`
- 本地环境安装：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/setup_uv_env_local.sh`
- AutoDL 环境安装：`/Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/setup_uv_env_autodl.sh`

## 2. 为什么先做本地 smoke test

本地 Mac 不适合直接跑正式 Qwen 训练，但非常适合先验证训练链路。

本地 smoke test 的意义是：

- 验证 `prepare_trl_sft_dataset.py` 能跑
- 验证 `run_trl_sft.py` 参数链路正确
- 验证 `TRL + LoRA` 训练入口可执行
- 验证输出目录、checkpoint、summary 能生成

这样我们就能先把“代码和环境问题”在本地解决，再把正式训练交给 AutoDL。

## 3. 本地 Mac 环境

创建环境：

```bash
bash /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/setup_uv_env_local.sh
source /Users/dongyong/Project/Trea_code/ecom-llmv2/.venv/bin/activate
```

检查 torch：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果是 Apple Silicon，还可以看 MPS：

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## 4. 本地 smoke test

最小运行：

```bash
source /Users/dongyong/Project/Trea_code/ecom-llmv2/.venv/bin/activate
bash /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_sft_smoke.sh
```

默认会：

- 使用一个很小的 Hugging Face 模型
- 把 `MAX_LENGTH` 压到较小值
- 默认只做很轻量的训练验证
- 默认 LoRA 目标层使用 `all_linear`，提高兼容性

如果你想再快一点，可以显式限制步数：

```bash
export MAX_STEPS=1
bash /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_sft_smoke.sh
```

## 5. 正式 SFT 训练

最小运行方式：

```bash
source /Users/dongyong/Project/Trea_code/ecom-llmv2/.venv/bin/activate

export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
bash /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_sft.sh
```

你可以通过环境变量调整：

- `MODEL_PATH`
- `OUTPUT_DIR`
- `MAX_LENGTH`
- `LEARNING_RATE`
- `NUM_TRAIN_EPOCHS`
- `MAX_STEPS`
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

## 6. LoRA 注入层别名

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

## 7. LoRA 消融实验

第一轮推荐：

- rank：`8,16,32`
- 注入层：`attention_only,attn_mlp,all_linear`

干跑查看命令：

```bash
source /Users/dongyong/Project/Trea_code/ecom-llmv2/.venv/bin/activate

python /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_lora_ablation.py \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --dry-run
```

正式跑：

```bash
python /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/run_trl_lora_ablation.py \
  --model-path Qwen/Qwen2.5-1.5B-Instruct
```

## 8. AutoDL 环境

创建 GPU 环境：

```bash
bash /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/setup_uv_env_autodl.sh
source /Users/dongyong/Project/Trea_code/ecom-llmv2/.venv/bin/activate
```

如果你的机器不是 `cu121`，可以覆盖：

```bash
export TORCH_BACKEND=cu118
bash /Users/dongyong/Project/Trea_code/ecom-llmv2/scripts/setup_uv_env_autodl.sh
```

## 9. 当前最合理的实验顺序

建议这样推进：

1. 本地先跑 smoke test
2. 本地确认脚本和依赖没问题
3. push 到 GitHub
4. AutoDL pull 最新代码
5. AutoDL 跑单次 SFT
6. 再跑第一轮 9 组 LoRA 消融
7. 最后直接跑统一评测脚本看微调胜率和评测报告
