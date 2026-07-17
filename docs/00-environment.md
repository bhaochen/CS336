# 00 · 环境与依赖

## 1. 为什么把五个 assignment 合并到一个环境

每个 assignment 原本都有自己的 `pyproject.toml` + `uv.lock`（独立虚拟环境）。
为了在根目录统一管理、避免重复下载几个 GB 的 torch/vllm，我们把依赖合并到根的
`pyproject.toml`，删除各子目录的 uv 配置，只 `uv sync` 一次。

本地源码包（`cs336_basics` / `cs336_systems` / `cs336_scaling` / `cs336_data` /
`cs336_alignment`）通过根 `pyproject.toml` 里的 editable source 指向
`assignment1-basics`，再用 `.pth` 把其余四个 assignment 目录加进 `sys.path`：

```
.venv/lib/python3.12/site-packages/cs336_assignments.pth
```

## 2. 解决过的依赖冲突

| 冲突 | 处理 |
|------|------|
| `vllm==0.7.2` 锁定 `torch==2.5.1`，与 a1/a2/a4 的 `torch~=2.6/2.7` 矛盾 | 不在根里写 `torch`，由 vllm 决定 → 最终 `torch 2.5.1+cu124`（能跑所有作业） |
| `hydra-core` 要 `antlr4==4.9.*`，`math-verify` 要 `4.13.2` | 删除 `hydra-core`（仅出现在 a4 的嵌套副本里，非必需） |
| a3 用了 `scipy` 但原 `pyproject` 没声明 | 补充 `scipy` |
| `cs336_scaling/__init__.py` 调用 `importlib.metadata.version("cs336-scaling")`，删 pyproject 后找不到元数据 | 改成 try/except，找不到时 `__version__="0.0.0"` |

## 3. 外部资源（需自行下载，非 pip 依赖）

- **a4** fasttext 模型（放到 `assignment4-data/data/`）：
  - `lid.176.bin` ← https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
  - `jigsaw_fasttext_bigrams_nsfw_final.bin` ← allenai/dolma-jigsaw-fasttext-bigrams-nsfw
  - `jigsaw_fasttext_bigrams_hatespeech_final.bin` ← allenai/dolma-jigsaw-fasttext-bigrams-hatespeech
  - `quality_classifier/quality.bin` ← **你自己训练的 assignment 交付物**，不提供下载
- **a5** 模型 `models/Qwen2.5-Math-1.5B`（SFT 测试用，需自行拉取）

## 4. 测试结果（环境验证）

| Assignment | 结果 |
|------|------|
| a1-basics | ✅ 47 passed, 1 xfailed（`test_encode_memory_usage` 作者标记预期失败，因 encode 不省内存） |
| a2-systems | ⚠️ 14 passed, 2 failed（`test_flash_backward_triton` 数值精度不达标，作业代码本身的精度问题，非环境） |
| a3-scaling | ➖ 无 `tests/` 目录，只有 `chinchilla_isoflops.py` 脚本 |
| a4-data | ⚠️ 需先训练出 `quality.bin` 才能收集测试（其余 fasttext 模型已就位） |
| a5-alignment | ✅ 29 passed, 2 errors（需 `models/Qwen2.5-Math-1.5B` 权重） |

## 5. 常用命令

```bash
uv sync                              # 安装/更新根环境依赖
cd assignment1-basics && uv run pytest -q
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
