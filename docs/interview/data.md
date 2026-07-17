# 面试：数据工程（Data）

## Q1. 为什么训练语料要去重？（a4 `utils.py`）
- 重复数据 → 过拟合、记忆训练集、评测污染（test 数据混进 train）。
- C4 / RedPajama / GPT-3 都证明去重显著提升下游质量。

## Q2. 精确去重 vs 模糊去重？
- **精确**：按行/文档 hash 去重（`exact_line_deduplication`）。
- **模糊（MinHash + LSH）**：`minhash_deduplication`
  - 文本规范化 → n-gram 集合 → 多个 `mmh3` 哈希取最小值得签名 → 相似签名聚为一类。
  - 能抓"改写/小改动"的近重复，精确 hash 抓不到。

## Q3. MinHash 原理一句话？
- 两个集合的 Jaccard 相似度 ≈ 它们 MinHash 签名相等的概率。
- 用 k 个独立哈希函数的"最小值"组成签名，比较签名重合度即近似相似度。

## Q4. 质量过滤怎么做？（a4）
- **规则过滤**：Gopher 启发式（平均词长、符号占比、停用词、行长度等）→ `gopher_quality_filter`。
- **分类器过滤**：训练 fasttext 二分类（wiki 高质量 vs CC 低质量）→ `quality_classifier.py`，产出 `quality.bin`。
- **毒性/NSFW**：Jigsaw fasttext 模型打分过滤。
- **语言过滤**：fasttext LID（`lid.176.bin`）保留目标语种。

## Q5. PII 怎么处理？
- 用正则识别邮箱/电话/IP，替换为占位符（`mask_email` 等），保护隐私、降法律风险。

## Q6. 数据配比（data mixing）为什么重要？
- 不同领域（web/代码/数学/书籍）比例显著影响能力；可用 ablation 找最优 mix。
