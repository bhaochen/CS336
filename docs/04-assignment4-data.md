# 04 · Assignment 4: Data（数据质量流水线）

对应 `assignment4-data/cs336_data/`。这是**工程型作业**：搭一条从原始 web 文本到
高质量训练语料的流水线。关键交付物是**你自己训练的质量分类器** `quality.bin`。

## 模块地图

| 文件 | 内容 |
|------|------|
| `utils.py` | 文本抽取、语言识别、PII 脱敏、NSFW/毒性分类、Gopher 质量过滤、去重（MinHash） |
| `quality_classifier.py` | 用 fasttext 训练二分类质量模型（wiki vs cc） |
| `download.sh` | 下载原始数据 |

## 1. 文本抽取（`extract_text_from_html_bytes`）

用 `resiliparse` 从 WARC/HTML 字节流里抽纯文本（去标签、去 boilerplate）。

## 2. 语言识别（`identify_language`）

加载 `data/lid.176.bin`（fasttext 176 语种 LID 模型），过滤非目标语言。

## 3. PII 脱敏（`mask_email` / `mask_phone_numbers` / `mask_ip_address`）

用 `regex` 匹配邮箱 / 电话 / IPv4，替换为 `[EMAIL]` 等占位符，并统计命中数。
用于 `test_pii.py`。

## 4. 毒性 / NSFW 分类

- `classify_nsfw`：加载 `jigsaw_fasttext_bigrams_nsfw_final.bin`
- `classify_toxic_speech`：加载 `jigsaw_fasttext_bigrams_hatespeech_final.bin`

## 5. Gopher 质量启发式（`gopher_quality_filter`）

基于 Gopher 论文的规则集：平均词长、符号占比、停用词占比、行长度等阈值过滤低质文本。

## 6. 去重

- **精确去重** `exact_line_deduplication`：按行 hash 去重。
- **模糊去重（MinHash + LSH）** `minhash_deduplication`：
  - `normalize_text` 规范化（去标点/小写/展开）
  - `get_minhash` 用 `mmh3` 多次哈希 n-gram 集合，取最小值做签名
  - 相似文档签名重叠高 → 聚为一类 → 只保留代表

> 面试点：为什么去重对 LLM 训练重要？→ 重复数据导致过拟合、记忆化、
> 评测污染；C4/RedPajama 等都证明去重能显著提升下游质量。

## 7. 质量分类器（`quality_classifier.py`）

这是本作业的核心：用 fasttext 训练一个**监督二分类器**，判断文本是否"高质量"。
- 正样本：Wikipedia 类高质量文本
- 负样本：Common Crawl 随机样本
- `train_model` → `fasttext.train_supervised(...)` → 输出 `quality.bin`
- 该模型被 `utils.classify_quality` 在顶层 import 加载——
  **所以 `data/quality_classifier/quality.bin` 不存在时，所有 import utils 的测试都无法收集**。

## 8. 外部资源（见 00-environment.md）

`lid.176.bin`、`nsfw`、`hatespeech` 三个 fasttext 模型我已下载；
`quality.bin` 需你自己训练。

## 9. 流水线总图

```
原始 WARC / Common Crawl
        │
        ▼
┌──────────────────┐
│ extract_text      │  resiliparse 抽纯文本（去 HTML/boilerplate）
└────────┬─────────┘
         ▼
┌──────────────────┐
│ language filter   │  lid.176.bin 留目标语种
│ PII masking       │  mask_email/phone/ip
│ NSFW/toxic filter │  jigsaw fasttext 模型
│ gopher quality    │  启发式规则阈值
└────────┬─────────┘
         ▼
┌──────────────────┐
│ deduplication     │  精确(行hash) + 模糊(MinHash+LSH)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ quality classifier│  你训练的 fasttext: wiki(+) vs cc(-) → quality.bin
└────────┬─────────┘
         ▼
   高质量训练语料
```
