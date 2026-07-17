# 面试复习索引

按主题组织的面试题 + 要点，全部对应本仓库 `assignment1~5` 的实际实现。
答题时可以说"我在一遍手写实现里验证过"，比纯背书更有说服力。

| 主题 | 文件 |
|------|------|
| 分词与 BPE | [tokenization.md](./tokenization.md) |
| Transformer 架构 | [transformers-arch.md](./transformers-arch.md) |
| 训练系统与分布式 | [training-systems.md](./training-systems.md) |
| 缩放定律 | [scaling-laws.md](./scaling-laws.md) |
| 数据工程 | [data.md](./data.md) |
| 对齐（SFT/DPO/GRPO） | [alignment.md](./alignment.md) |

## 使用建议

- 每题先看"一句话答"，再展开"深入"。面试时先给结论，被追问再展开。
- 代码位置用 `文件:行号` 回溯（如 `module.py:59` 是 RMSNorm）。
- 重点准备：**RoPE、FlashAttention、AdamW、Chinchilla、DPO vs PPO、GRPO 无 value model**。
