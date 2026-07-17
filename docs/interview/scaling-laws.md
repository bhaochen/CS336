# 面试：缩放定律（Scaling Laws）

## Q1. Kaplan 2020 的核心结论？
- Loss 随参数量：`L(N) = (N_c/N)^α`，α≈0.076。
- Loss 随算力：`L(C) = (C_c/C)^β`，β≈0.05。
- 含义：只要加算力/参数，loss 就平滑下降，无突然"聪明"的涌现阈值（涌现多是度量 artifact）。

## Q2. Chinchilla 改了什么？（重点）
- Kaplan 隐含假设"数据固定、算力主要砸参数"。
- Chinchilla 证明：给定固定算力 **C，最优是参数量 N 和训练 token 数 D 等比例增长**，
  `N_opt ∝ C^0.5`，`D_opt ∝ C^0.5`。
- 结论：GPT-3 等大模型**严重训练不足（undertrained）**——用 1/4 参数 + 更多数据
  就能达到同等 loss。

## Q3. IsoFLOP 实验怎么做的？（a3 `chinchilla_isoflops.py`）
- 固定算力预算 C，扫不同 (N, D) 使 `6ND ≈ C`。
- 训练后画 **IsoFLOP 曲线**（同算力下不同规模的 loss），最低点 = 该算力最优 N。
- 连接各最低点得经验 `N_opt(C)`，验证 ∝ C^0.5。

## Q4. Chinchilla 最优解的推导（重点，手推）
Kaplan 经验式：`L(N,D) = A/N^α + B/D^β + E`，(α≈0.076, β≈0.095)。
固定算力约束 `C = 6ND`（FLOPs），消元 D = C/(6N)：
```
L(N) = A/N^α + B·(6N/C)^β + E
dL/dN = -αA/N^{α+1} + βB(6/C)^β N^{β-1} = 0
⇒ αA/N^{α+1} = βB(6/C)^β N^{β-1}
⇒ N*^{α+β} = (αA)/(βB) · (C/6)^β
```
因 α≈β，得 `N* ∝ C^{β/(α+β)} ≈ C^0.5`，同理 `D* ∝ C^0.5`。
**结论：算力翻倍，参数和 token 各 ×√2（等比例），而非只加参数。**

### 数量级
Chinchilla 给出经验 `D ≈ 20·N`（每参数约 20 个训练 token）。
70B 模型最优训练量 ≈ 1.4T tokens（而非 Gopher 的 300B）。

## Q5. FLOPs 推导（为什么是 6ND）
- 一个 `(d,d)` 矩阵乘 `XW`：每个输出元素 = d 次乘加 = 2d FLOPs，总 `2·d_out·d_in`。
- **前向**：所有线性层参数 ≈ N，每 token 算一次 → 2ND FLOPs。
- **反向**：梯度对参数 (2ND) + 梯度对输入（同量级，常并入）≈ 再加 2ND... 合计 **≈ 6ND**
  （精确：前向 2ND，反向 wrt 参数 2ND，反向 wrt 激活 2ND）。

## Q4. 算力/数据/参数三者怎么权衡？
- 先按 Chinchilla 定 N 和 D 的比例（~20 tokens/param 的经验值）。
- 若数据有限（常见），参数要相应减小，否则过拟合/浪费。

## Q5. 缩放定律的局限？
- 只预测"交叉熵 loss"，不直接预测"下游任务能力/推理能力"。
- 数据质量、架构、训练 recipe 同样关键（data-constrained scaling）。
