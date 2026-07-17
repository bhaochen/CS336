# 面试：对齐（Alignment: SFT / DPO / GRPO）

## Q1. 后训练一般流程？
预训练（学世界知识）→ SFT（学指令遵循格式）→ 偏好/RL 对齐（DPO/PPO/GRPO）→ 评测。

## Q2. SFT 怎么做？（a5 `sft.py`）
- 用 `(prompt, response)` 配对，在 prompt token 上 **mask loss**，只让模型学 response 的 next-token CE。
- 把预训练 LM 变成"会按指令回答"的模型，是后续对齐的基础。

## Q3. DPO 是什么？为什么比 RLHF 简单？（a5 `dpo.py`）

### 从 RLHF 到 DPO 的直觉
RLHF 先训 reward model `r_φ`，再用 PPO 最大化 `E[r(x,y)] - β·KL(πθ||π_ref)`。
DPO 的关键观察：**这个约束优化问题的闭式解**里，reward 能表示成
`r(x,y) = β·log(πθ(y|x)/π_ref(y|x)) + const`。代回 Bradley-Terry 偏好概率，
就消掉了显式 reward model，得到只含策略比的损失：

### 损失公式
```
L_DPO(πθ; π_ref) = -E_{(x,y_w,y_l)}[ log σ( β·log(πθ(y_w|x)/π_ref(y_w|x))
                                          - β·log(πθ(y_l|x)/π_ref(y_l|x)) ) ]
```
- `y_w` = chosen，`y_l` = rejected，`σ` = sigmoid。
- `β` 控制偏离参考策略的惩罚强度。
- 代码 `compute_per_instance_dpo_loss`（`dpo.py:23`）逐样本实现上式。

### 为什么简单
| | RLHF/PPO | DPO |
|---|---|---|
| 需 reward model | ✅ | ❌ |
| 需在线采样/rollout | ✅ | ❌（离线偏好对） |
| 稳定性 | 超参敏感、易崩溃 | 稳，一个分类式损失 |
| 成本 | 高（多模型+采样） | 低 |

代价：DPO 受限于静态偏好数据，难做在线探索；复杂奖励仍常回到 reward model + PPO。

## Q4. PPO vs DPO 怎么选？
- PPO：样本效率高、能做在线探索、但超参敏感、需 reward model + critic + 在线采样，工程重。
- DPO：离线、稳定、简单；但受限于静态偏好数据，难利用新反馈。

## Q5. GRPO 是什么？为什么不需要 value model？（a5 `grpo.py`）

### 算法流程
1. 对每个 prompt `q` **采样一组 G 个回答** `{o_1,...,o_G}`（同策略 πθ）。
2. 用规则/奖励模型打分 `r_i = R(q, o_i)`（代码用 `drgrpo_grader.py`）。
3. **组相对优势**（无 value model 的关键）：
   `Â_i = (r_i - mean({r})) / std({r})`  —— `compute_group_normalized_rewards`（`grpo.py:22`）。
4. 策略梯度损失（带 PPO-style clip）：
   ```
   L = -E[ min( ρ_i·Â_i,  clip(ρ_i,1-ε,1+ε)·Â_i ) ],  ρ_i = πθ(o_i|q)/π_old(o_i|q)
   ```
   `compute_grpo_clip_loss`（`grpo.py:51`）。

### 为什么不需要 critic / value model
- PPO 用 value network 估计 baseline `V(s)` 来减方差 → 多一份梯度+优化器状态显存。
- GRPO 用**同 prompt 多采样**的组内奖励均值/标准差当 baseline：
  同组回答共享同一 prompt，奖励差异只来自回答本身 → 组内归一化天然去偏。
- 结果：省掉 critic，显存↓、训练更稳，特别适合可批量采样的推理/数学任务。

### 对比表
| | PPO | GRPO |
|---|---|---|
| value/critic | 需要 | 不需要 |
| baseline | V(s) 网络 | 组内奖励统计 |
| 采样 | 单条在线 | 一组 G 条 |
| 典型用途 | 通用 RLHF | RLVR（数学/代码，可规则打分） |

## Q6. 奖励怎么来？规则 vs 模型？
- 数学/代码可用**规则 grader**（如 `drgrpo_grader.py` 的符号等价校验）。
- 开放式有用性常用 **reward model**（常由偏好数据训出）。规则奖励可解释、无奖励 hacking 风险。

## Q7. 评测指标（a5 `eval_*.py`）
- 指令：AlpacaEval（用 vLLM 加速）。
- 数学：GSM8K / math，`math-verify` 做符号等价校验（不只看字符串）。
- 知识：MMLU / SST 选择题。
