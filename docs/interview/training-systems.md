# 面试：训练系统与分布式

## Q1. DP / DDP / ZeRO 区别？
- **DP**（DataParallel）：单进程内多卡复制，前向各自算、反向各自算、再 all-reduce 梯度后各自更新；主卡 gather 输出。通信少但主卡负载重、显存冗余。
- **DDP**：每卡一进程一完整副本，反向时梯度 all-reduce 求平均，各卡独立更新（参数始终一致）。本仓库 `naive_ddp.py`。
- **ZeRO**（a2 `sharding.py`）：把优化器状态 / 梯度 / 参数**分片**到各卡，显存从 O(P) 降到 O(P/N)。

## Q2. DDP 怎么隐藏通信延迟？
- **梯度分桶（bucketing）**：多个小梯度拼成大块再 all-reduce，提升带宽利用率。
- **重叠（overlap）**：反向一算完某层梯度立刻开始 all-reduce，与后续层反向并行。
- 代码：`ddp_overlap_bucketed.py` / `ddp_overlap_individual_parameters.py`。

## Q3. FlashAttention 为什么快？（a2 `flash_attention.py`）

### 标准 attention 的 IO 瓶颈
```
S = Q Kᵀ          # [N, N] 写到 HBM（慢，显存 O(N²)）
P = softmax(S)    # 再读回
O = P V
```
两次大矩阵都要落 HBM，受 **memory bandwidth** 限制，而非算力。

### FlashAttention 核心：IO 感知 + 在线 softmax
把 Q/K/V 切成小块在 **SRAM**（快）上算，关键是 softmax 可**增量更新**（online softmax）：
对每个新块 j，维护 running max `m` 和 running sum `l`：
```
m_new = max(m, rowmax(S_j))          # 更新最大值
P_j   = exp(S_j - m_new)             # 用新 max 修正
O     = O · exp(m - m_new) + P_j · V_j   # 旧结果按 exp(m-m_new) 缩放
l_new = l · exp(m - m_new) + rowsum(P_j)
# 最终 O /= l
```
这样**不物化完整 N×N 的 P**，只在 SRAM 里流式算，HBM 只读写 Q/K/V/O（O(N)）。

### 复杂度对比
| | 标准 | Flash |
|---|---|---|
| HBM 流量 | O(N²) | O(N²·d/M) → 随块小更优，有效 O(N) |
| 显存 | O(N²) | O(N) |
| 数值 | 等价 | 等价（online softmax 数学一致） |

代码：`flash_fwd_kernel`（`flash_attention.py:7`）实现前向分块；反向
`flash_bwd_kernel_q/_kv/_d`（`:118/:280/:436`）分别算 dQ / dK,dV / dP。
> 测试坑：`test_flash_backward_triton` 在 torch 2.5.1 下因反向数值精度未达 rtol=0.01 失败——作业代码本身精度问题，非环境。标准 attention 对照正常。

## Q4. 如何估算训练算力（FLOPs）？
- `C ≈ 6·N·D`（N 参数量，D token 数）：前向 2ND + 反向 4ND。
- 推理一次前向约 `2·N·D_prompt`。

## Q5. 单卡显存都花在哪？
- 模型参数（fp16 2B/param）、梯度（同）、优化器状态（Adam 的 m/v + 参数 = 12B/param 若 fp32）、
  激活值（随 batch/seq 增大）、临时 buffer。ZeRO/梯度检查点/混合精度用来压这些。

## Q6. 怎么定位性能瓶颈？（a2 `nsys_profile.py`）
- 用 nsys 抓 kernel 时间线，看是 compute-bound 还是 communication-bound / memory-bound。
