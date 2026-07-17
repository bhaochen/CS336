# CS336 开发文档

这个目录是我在做 Stanford CS336《Language Modeling from Scratch》时的**学习 + 面试复习**笔记。所有内容基于本仓库的实际代码（`assignment1~5`），不是泛泛而谈。

## 仓库结构

```
CS336/
├── pyproject.toml          # 合并后的根依赖（所有 assignment 共用 .venv）
├── uv.lock
├── .venv/                  # 统一的 Python 3.12 虚拟环境
├── assignment1-basics/     # 分词 / BPE / Transformer / 优化器
├── assignment2-systems/     # 分布式训练 / FlashAttention / 显存
├── assignment3-scaling/     # 缩放定律（Chinchilla 等）
├── assignment4-data/        # 数据质量流水线
├── assignment5-alignment/   # SFT / DPO / GRPO / 评测
└── docs/                   # 本目录
```

## 环境说明（重要）

- 五个 assignment **共用根目录的 `.venv`**（由根的 `pyproject.toml` + `uv.lock` 管理）。
- 各 assignment 原本独立的 `pyproject.toml` / `uv.lock` **已删除**，统一在根目录 `uv sync` 安装。
- 本地源码包（`cs336_basics` 等）通过 `.venv/.../site-packages/cs336_assignments.pth` 指向各 `assignment*/` 目录，使其可被 import。
- `torch` 实际版本为 **2.5.1+cu124**（被 `vllm==0.7.2` 锁定），CUDA 可用。

运行代码的姿势：

```bash
cd assignment1-basics
uv run python train_loop.py     # 复用根 .venv，且当前目录在 sys.path 上
uv run pytest                  # 跑测试
```

> 必须 `cd` 进 assignment 目录再 `uv run`，不能从根目录跑——作业代码用了相对路径 / 当前目录 import（如 a4 的 `from utils import *`、`data/lid.176.bin`）。

## 文档导航

| 文件 | 内容 |
|------|------|
| [00-environment.md](./00-environment.md) | uv 环境、依赖冲突解决、测试结果与坑 |
| [01-assignment1-basics.md](./01-assignment1-basics.md) | 字节级 BPE、分词器、Transformer 各组件、AdamW |
| [02-assignment2-systems.md](./02-assignment2-systems.md) | DDP、梯度分桶、FlashAttention、显存/算力分析 |
| [03-assignment3-scaling.md](./03-assignment3-scaling.md) | 缩放定律、Chinchilla、IsoFLOP |
| [04-assignment4-data.md](./04-assignment4-data.md) | 数据去重、质量分类、毒性/PII 过滤 |
| [05-assignment5-alignment.md](./05-assignment5-alignment.md) | SFT、DPO、GRPO/RLVR、评测指标 |

## 面试复习

`interview/` 目录是按主题组织的面试题 + 答案，直接对应上面的代码实现：

- [interview/tokenization.md](./interview/tokenization.md)
- [interview/transformers-arch.md](./interview/transformers-arch.md)
- [interview/training-systems.md](./interview/training-systems.md)
- [interview/scaling-laws.md](./interview/scaling-laws.md)
- [interview/data.md](./interview/data.md)
- [interview/alignment.md](./interview/alignment.md)
