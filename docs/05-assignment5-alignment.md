# 05 · Assignment 5: Alignment（对齐 / SFT / DPO / GRPO）

对应 `assignment5-alignment/cs336_alignment/`。这是后训练（post-training）作业，
把预训练 LM 对齐到指令遵循 / 推理 / 安全。

## 模块地图

| 文件 | 内容 |
|------|------|
| `sft.py` / `sft_script.py` | 监督微调（Supervised Fine-Tuning） |
| `dpo.py` | Direct Preference Optimization |
| `grpo.py` | Group Relative Policy Optimization（RLVR） |
| `drgrpo_grader.py` | 数学/代码 grader（规则奖励） |
| `data_loading.py` | 数据集加载与模板 |
| `eval_metric.py` / `eval_*.py` | 各类评测（Alpaca / GSM8K / MMLU / SST / math） |
| `infer_demo.py` / `look_at_stf.py` / `debug.py` | 推理与调试脚本 |

## 1. SFT（监督微调，`sft.py`）

用 `(prompt, response)` 配对数据，在 prompt 上 mask 掉 loss，只让模型学习
生成 response（标准 next-token CE loss）。`sft_eval` 用 vLLM 批量生成做评估。
模板化 prompt 见 `data_loading` / `prompts/`。

## 2. DPO（直接偏好优化，`dpo.py`）

不需要显式 reward model，直接在偏好数据 `(chosen, rejected)` 上优化：

```
L_DPO = -log σ( β * (log πθ(y_w|x) - log π_ref(y_w|x))
                - β * (log πθ(y_l|x) - log π_ref(y_l|x)) )
```

- `compute_per_instance_dpo_loss`：逐样本算上述损失。
- 关键：用**参考策略 `π_ref`**（冻结的 SFT 模型）约束，防止偏离太远。
- 相比 RLHF（训练 reward model + PPO），DPO 更稳定、省一个模型。

> 面试点：DPO vs PPO？→ DPO 把"奖励+RL"合成一个分类式损失，
> 无 reward model、无在线采样、训练简单；PPO 样本效率高但超参敏感、工程复杂。

## 3. GRPO（组相对策略优化，`grpo.py`）

DeepSeek-R1 用的 RLVR 方法，相对 PPO 去掉了 critic/value 网络：

- 对每个 prompt **采样一组 G 个回答** `{o_i}`。
- 用规则 grader（如 `drgrpo_grader`）给每个回答打分 `r_i`。
- **组归一化优势**：`A_i = (r_i - mean(r)) / std(r)`（同组内的相对好坏）。
- `compute_grpo_clip_loss`：带 PPO-style clip 的策略梯度损失，用优势加权。
- `grpo_microbatch_train_step`：micro-batch 更新。

> 面试点：GRPO 为什么不需要 value model？→ 用"同 prompt 多采样"的
> 组内奖励统计量当 baseline，省掉 critic，显存和稳定性都更好。

## 4. 评测（`eval_*.py`）

- **Alpaca**：指令遵循，用 `alpaca-eval` + vLLM。
- **GSM8K / math**：数学推理，`math-verify` 做符号等价校验。
- **MMLU / SST**：分类/知识选择题。
- `evaluate_vllm`：统一用 vLLM 加速生成再打分。

## 5. 测试坑

`test_sft.py` 的两个用例需 `models/Qwen2.5-Math-1.5B` 权重（自行下载）；
其余 29 个测试在根环境全部通过。
