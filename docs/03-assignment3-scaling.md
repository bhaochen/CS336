# 03 · Assignment 3: Scaling Laws（缩放定律）

对应 `assignment3-scaling/`。只有 `cs336_scaling/model.py` 和一个脚本
`chinchilla_isoflops.py`（无 `tests/` 目录）。

## 1. 模型（`model.py`）

`BasicsTransformerLM`：标准 Transformer LM，注意与 a1 的区别——
- 用 `nn.LayerNorm`（**不是** a1 的 RMSNorm）
- 用**可学习的位置编码** `nn.Embedding(context_length, d_model)`（**不是** RoPE）
- `lm_head` 与 `token_embeddings` **不共享权重**

用于训练不同规模的模型，收集 loss，拟合缩放定律。

## 2. 缩放定律（Scaling Laws）核心公式

论文结论（Kaplan et al. 2020；Chinchilla / Hoffmann et al. 2022）：

- **Loss 随参数量**：`L(N) = (N_c / N)^α` （α ≈ 0.076，Kaplan）
- **Loss 随算力**：`L(C) = (C_c / C)^β` （β ≈ 0.05）
- **Chinchilla 最优**：给定固定算力 `C`，最优参数量 `N_opt` 与训练 tokens `D_opt` 满足
  `N_opt ∝ C^0.5`，`D_opt ∝ C^0.5`——**参数量和数据量应等比例增长**，
  而非像 GPT-3 那样"参数远大于数据"。Chinchilla 用 ~1/4 参数达到 Gopher 同等 loss。

## 3. IsoFLOP / Chinchilla 实验（`chinchilla_isoflops.py`）

思路：
1. 固定一组算力预算 `C`（以 FLOPs 衡量），扫不同的 `(N, D)` 组合，使 `6ND ≈ C`。
2. 训练后在验证集测 loss，画出 **IsoFLOP 曲线**（同算力下不同规模的 loss）。
3. 每条曲线的最低点就是该算力下的最优 `N`；连接各最低点得到
   `N_opt(C)` 的经验关系，验证 `N_opt ∝ C^0.5`。

> 面试点：Chinchilla 相比 Kaplan 的结论改了什么？→
> Kaplan 认为"算力主要该砸在参数上、数据固定"；Chinchilla 证明
> **参数和 data 应同步 scaling**，此前大模型严重训练不足（undertrained）。

## 4. FLOPs 估算（面试常考）

训练一个 transformer 的近似算力：
```
C ≈ 6 * N * D      # N=参数量, D=训练 token 数
```
（前向 2ND + 反向 4ND）。推理一次前向约 `2 * N * D_prompt`。

## 5. IsoFLOP / Chinchilla 图解

```
 loss
  ^  ·           同算力 C 下扫不同 N
  │   ··        （每组 D = C/6N 取满）
  │     ··
  │      ··__最低点 = N_opt(C)
  │         ··
  │            ··
  │               ··
  └──────────────────────► model size N

  连接多条 IsoFLOP 曲线的最低点 → N_opt(C) ∝ C^0.5
```

Chinchilla 最优配比经验值：**D ≈ 20 · N**（每参数约 20 训练 token）。
