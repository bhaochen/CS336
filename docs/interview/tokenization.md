# 面试：分词与 BPE

## Q1. 为什么用字节级 BPE 而不是字符级 / WordPiece？
- 字节级把词表固定在 **256 个字节**，任何 UTF-8 文本都能无损表示，**无 OOV**。
- 字符级序列太长、语义碎片化；WordPiece 仍可能出现 unknown。
- 多语言、emoji、代码都能自然处理。
- 代码：`bpe.py` 的 `word_to_bytes_tuple` 把词拆成字节元组。

## Q2. BPE 训练流程？（含公式）
1. **pretokenize**：用正则切词（GPT-2 模式），每个词转成字节序列。
2. 初始化：词 = 字节元组（如 `"low"` → `(l,o,w)` 的 UTF-8 字节）。
3. 统计**相邻字节对频率** `freq[(a,b)]`（代码 `_get_bytes_pair_freq`）。
4. 反复：选 `argmax freq` 的字节对 → 合并 → 更新词 → 记录 `merges`，
   直到 `len(vocab) == vocab_size` 或无可合并对。
5. 保存 `vocab.json` + `merges.txt`（`save_vocab_and_merges`）。

### 复杂度 & 优化
朴素实现每轮扫全语料 O(T·L)，T=总 token、L=词长。
工程优化：用**优先队列/堆**维护对频率，增量更新受影响的相邻对；
本仓库 `bpe.py` 用 `find_chunk_boundaries` 按 `\n` 把大文件切块、多线程
`pretokenize_parallelizing` 加速预分词。

## Q2b. 为什么 vocab 从字节(256)起步而不是字符？
- 字符级：中文/emoji/稀有符号基数大且仍可能 OOV，序列长。
- 字节级：基数恒为 **256**，任何 UTF-8 文本都能无损还原，**永不 OOV**，
  且 merges 在字节上学习跨语言的子词组合。代价：初始序列更长（1 字符=1~4 字节）。

## Q3. encode / decode 怎么做的？
- `encode`：用 merges 贪心地把字节序列合并成 token id。
- `decode`：id → 字节 → `bytes.decode("utf-8", errors="replace")`（容错非法 UTF-8）。
- 代码：`Tokenizer` 类（`tokenizer.py`）。

## Q4. 为什么 tokenizer 要跟模型一起保存？
merges/vocab 决定了 token→id 的映射，推理时必须和训练时完全一致，否则语义错乱。

## Q5. 大模型分词常见痛点？
- 多语种不平衡、空格/无空格语言（中文无词边界）、数字切分、长 tail token。
- 分词器膨胀（如 LLaMA 扩展到中文要加词）。
