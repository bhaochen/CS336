# 01 · Assignment 1: Basics（分词 / BPE / Transformer / 优化器）

对应 `assignment1-basics/cs336_basics/`。

## 模块地图

| 文件 | 关键内容 |
|------|---------|
| `tokenizer.py` | `Tokenizer` 类：基于 vocab + merges 的 encode/decode |
| `bpe.py` | 字节级 BPE 训练：`train_bpe`、pretokenize、并行分块 |
| `module.py` | 完整 Transformer：`Linear`/`Embedding`/`RMSNorm`/`RoPE`/`Attention`/`TransformerBlock`/`TransformerLM` |
| `optimizer.py` | 手写 `SGD`、`AdamW` |
| `utils.py` | `softmax`、参数初始化、checkpoint 读写 |
| `pretokenization_example.py` | GPT-2 风格 pretokenization 示例 |

## 1. 字节级 BPE（Byte-level BPE）

核心思想：词汇表的基本单元不是字符，而是 **256 个字节值**（用 `bytes` 表示）。
这样任何文本都能无损表示，永远不会出现 unknown token。

`bpe.py` 流程：
1. **pretokenize**：按正则把文本切成词（GPT-2 用 `\r?\n| 's| ...` 这类模式），
   再把每个词拆成 UTF-8 字节元组 → `word_to_bytes_tuple`。
2. **统计字节对频率**：`find_chunk_boundaries` 把大文件按 `\n` 切分多线程处理，
   `_get_bytes_pair_freq` 统计相邻字节对。
3. **合并**：反复选出频率最高的字节对，加入 merges，直到达到 `vocab_size`。
4. **保存**：`save_vocab_and_merges` 输出 `vocab.json` + `merges.txt`。

`Tokenizer.encode` 用学到的 merges 把字节序列贪心合并成 token id；
`decode` 把 id 映射回字节再 `decode("utf-8", errors="replace")`。

> 面试点：为什么字节级 BPE 比 WordPiece/字符级好？→ 封闭 256 词表、无 OOV、
> 多语言/emoji 无损、与下游模型解耦。

## 2. Transformer 组件（`module.py`）

- **`RMSNorm`**：只除以均方根（无 mean 中心化），`x / sqrt(mean(x^2)+eps) * weight`。
  比 LayerNorm 省一次减均值，LLaMA 系列标配。
- **`RotaryPositionEmbedding` (RoPE)**：把位置信息编码进 Q/K 的旋转角度。
  `q_rot = q * cos(mθ) + rotate_half(q) * sin(mθ)`。相对位置友好、长度外推好。
- **`scaled_dot_product_attention`**：标准 `softmax(QK^T/√d)V`，支持 causal mask。
- **`MultiHeadSelfAttention`**：QKV 投影 → 分头 → RoPE → SDPA → 输出投影。
- **`TransformerBlock`**：`RMSNorm → Attention → 残差 → RMSNorm → SwiGLU FFN → 残差`。
- **`SwiGLU`**：`silu(xW1) * (xW2)`，替代 ReLU FFN，效果更好。
- **`TransformerLM`**：`token_embeddings → N×Block → RMSNorm → lm_head`。
  有 `inference=True` 分支只取最后一个 logit（生成时用）。

变体对照（用于 ablation 实验）：
`TransformerBlockNoPE` / `NoRMS` / `PostNorm` / `SiLU` / `TransformerLMNoPE` 等——
直接对应论文里"去掉某组件后损失变化"的消融研究。

## 3. 优化器（`optimizer.py`）

- **`SGD`**：手写 `p -= lr * g`，支持 momentum。
- **`AdamW`**：Adam + **解耦权重衰减**（decoupled weight decay）。
  关键区别：权重衰减 *不* 进动量/二阶矩，而是直接 `p *= (1 - lr*wd)`，
  这是 AdamW 相比 Adam+L2 泛化更好的原因。

> 面试点：AdamW 的 weight decay 为什么解耦？→ L2 把衰减混进梯度，
> 被 Adam 的二阶矩缩放后强度随参数尺度变化；解耦后衰减恒定、更稳。

## 4. 训练循环

`train_loop.py`（仓库根目录作业文件）串起：tokenizer → dataloader →
forward/backward → `AdamW` 更新 → 可选 `wandb` 记录。
