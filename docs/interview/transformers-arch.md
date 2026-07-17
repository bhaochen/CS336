# 面试：Transformer 架构

## 0. 整体架构图

```
输入 token id
    │
    ▼
┌─────────────┐
│ TokenEmbed  │  (vocab_size × d_model)
└──────┬──────┘
       │  + 位置编码(RoPE 作用于 Q/K)
       ▼
┌──────────────────────────────┐
│  Transformer Block × N       │
│  ┌────────────────────────┐  │
│  │  RMSNorm                │  │
│  │  MultiHeadSelfAttention │  │  ← Q/K 先过 RoPE
│  │    SDPA (causal mask)   │  │
│  └───────────┬────────────┘  │
│       残差相加 │              │
│  ┌────────────▼───────────┐  │
│  │  RMSNorm                │  │
│  │  SwiGLU FFN             │  │
│  └───────────┬────────────┘  │
│       残差相加 │              │
└──────────────┼───────────────┘
       │
       ▼
    RMSNorm → lm_head → logits (vocab_size)
```

> 本仓库 a1 `TransformerLM`（`module.py:284`）：TokenEmbed → N×Block → RMSNorm → lm_head。
> 注意 a3 `BasicsTransformerLM` 用 **LayerNorm + 绝对位置编码 + 不共享权重**，与 a1 不同。

## Q1. RMSNorm 和 LayerNorm 区别？（本仓库 a1 用 RMSNorm）
- LayerNorm：`(x - mean) / std * γ + β`，减均值 + 除标准差。
- RMSNorm：`x / sqrt(mean(x²) + eps) * γ`，**不减均值**，省一次计算，更稳定。
  LLaMA/GPT-NeoX 系列标配。代码：`module.py:59`。

## Q2. RoPE 是什么？为什么好？（a1 `RotaryPositionEmbedding`）

### 公式推导
把 d_model 维向量按相邻两维配对，对第 m 个位置、第 i 对 (2i, 2i+1) 施加旋转角
θ_i·m，其中频率 `θ_i = base^(-2i/d)`（代码里 `theta ** (-arange(0,d_k,2)/d_k)`，base 即 `rope_theta`）：

```
[cos(θ_i·m)  -sin(θ_i·m)]   [x_{2i}  ]
[sin(θ_i·m)   cos(θ_i·m)] · [x_{2i+1}]
```

代码 `RotaryPositionEmbedding.forward`（`module.py:152`）用等价写法：
```
x_2 = stack([-x[...,1::2], x[...,::2]]).flatten(-2)   # = rotate_half(x)
out = x*cos + x_2*sin
```
`RoPE` 类（`module.py:143`）则用复数乘法 `freqs * view_as_complex(x)` 实现（更优雅，
`torch.polar(1, freqs)` 构造单位复数 e^{i·θ_i·m}）。

### 为什么编码的是相对位置
旋转矩阵是正交阵，关键是**可加性**：旋转角相加 = 位置差。
```
(R_m q)ᵀ (R_n k) = qᵀ R_{n-m} k
```
即 Q@K 内积只依赖 **(n-m)** 这个相对位置，与绝对位置无关 → 平移不变、长度外推好。

### 对比
| | RoPE | 绝对位置 Embedding (a3) |
|---|---|---|
| 内积是否含相对位置 | ✅ 自动 | ❌ 需学 |
| 长度外推 | 好（可调整 base） | 差（超出训练长度即 OOV） |
| 额外参数 | 无 | 有 (context_length×d) |

## Q3. 为什么用 SwiGLU 替代 ReLU FFN？
- 标准 FFN：`max(0, xW1)W2`（ReLU 门控）。
- SwiGLU（`module.py:94` `PositionwiseFeedForward`）：`silu(xW1) ⊗ (xW2)`，
  其中 `silu(z)=z·σ(z)`。门控结构让网络学"哪些信息该通过"，表达力更强、收敛更好。
- 代价：参数量约为标准 FFN 的 1.5×（多一组 W2）。

### FFN 容量公式
设 d_model=d，d_ff=f。标准 FFN 参数 `2df`；SwiGLU 参数 `2·(d·(2f/3))` 在同等容量下
实际 f' 更大但仍约 2df 量级（LLaMA 取 d_ff = 8d/3 配 SwiGLU）。

## Q4. 注意力计算与因果掩码？（a1 `scaled_dot_product_attention`）
- 公式：`Attention(Q,K,V)=softmax(QKᵀ/√d_k + mask)·V`。
- 缩放因子 `1/√d_k` 的原因：Q、K 各分量方差 ~1，点积均值 0、方差 ~d_k，
  不缩放会使 softmax 进入梯度极小的饱和区；除以 √d_k 使方差回到 1。
- causal mask：上三角置 -∞，位置 i 只能看 ≤i。训练时整序列并行算下三角；推理只取末位 logit。

## Q5. 参数共享：embedding 和 lm_head 共享权重？
- 常见做法共享（GPT-2/LLaMA），省参数且对齐语义。a1 的 `TransformerLM` **不共享**；
  a3 的 `BasicsTransformerLM` 也**不共享**。面试常问利弊：共享省参、小模型更稳。

## Q6. 推理时如何高效生成（KV cache 思路）？
- 缓存已算的 K/V，每步只算新 token 的 Q 与已有 K/V 做注意力，避免重算。
- （本作业 `generate_text` 用 `inference=True` 只取末位 logit 的简化版。）

## Q7. 参数量估算（面试手算）
以 LLaMA-7B 量级为例（d=4096, layers=32, heads=32, d_ff=11008, vocab=32000）：
```
embedding : V·d           ≈ 32000·4096      ≈ 131M
per block  : 2·(d² + d·d_ff)  ≈ 2·(16.8M + 45.1M) ≈ 124M   (attn 4·d²/2 + ffn 2·d·d_ff 近似)
all blocks: 32 · 124M      ≈ 3.97B
lm_head   : V·d            ≈ 131M (若共享则不计)
```
→ ≈ 6.7B，与公开 7B 吻合。手算时常忽略 layernorm/bias 小头。

## Q8. 为什么 pre-norm（RMSNorm 在残差分支前）更稳？
- Post-norm（原 Transformer）把 norm 放残差**之后**，深层梯度易消失，需 warmup。
- Pre-norm（本仓库）：`x = x + Attn(RMSNorm(x))`，残差通道始终直通，
  梯度能无损回传，训练更稳、可省/减 warmup。a1 的 `TransformerBlock`（`module.py:263`）即 pre-norm。
