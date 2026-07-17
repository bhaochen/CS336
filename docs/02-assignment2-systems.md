# 02 · Assignment 2: Systems（分布式 / FlashAttention / 显存）

对应 `assignment2-systems/cs336_systems/`。

## 模块地图

| 文件 | 内容 |
|------|------|
| `naive_ddp.py` | 最朴素 DistributedDataParallel 实现 |
| `ddp_overlap_bucketed.py` | 梯度分桶 + 通信计算重叠 |
| `ddp_overlap_individual_parameters.py` | 逐参数级 overlap |
| `sharding.py` | `ShardingOptimizer`：ZeRO 风格分片优化器状态 |
| `flash_attention.py` | Triton 实现的 FlashAttention（fwd + bwd） |
| `distributed_communication_single_node.py` | 单节点多卡通信原语（all-reduce 等） |
| `benchmarking_script.py` / `flash_benchmarking.py` / `pytorch_attn_benchmarking.py` | 性能对比 |
| `nsys_profile.py` | nsys 性能剖析封装 |

## 1. 分布式数据并行（DDP）

- **朴素 DDP**：每个 rank 持有完整模型副本，前向/反向各自算，再把梯度
  **all-reduce** 求平均，各自用平均梯度更新。梯度同步在 `backward` 时触发。
- **梯度分桶（bucketing）**：把多个小梯度 tensor 拼进一个大 bucket 再一次性
  all-reduce，减少小包通信开销、提高带宽利用率。
- **通信计算重叠（overlap）**：在反向传播过程中，已算完的层梯度立刻开始
  all-reduce，与后面层的反向计算并行，隐藏通信延迟。
- **逐参数级 overlap**：比按层更细，进一步压缩"等待通信"的空洞。

> 面试点：DDP 为什么比 DP 省显存？→ DP 在单进程内复制多份、用 scatter/gather；
> DDP 每卡一份，只通信梯度。ZeRO 进一步把优化器状态/梯度/参数也分片（见 `sharding.py`）。

## 2. ZeRO 分片优化器（`sharding.py`）

`ShardingOptimizer` 把 optimizer 的 **state（如 Adam 的 m、v）按参数分片到各 rank**，
每卡只存 1/N 的优化器状态，显存从 `O(N_params)` 降到 `O(N_params/N)`。
更新时各自更新自己分片，再 all-gather 还原完整参数。

## 3. FlashAttention（Triton 实现，`flash_attention.py`）

标准 attention 的瓶颈：把 `N×N` 的注意力分数矩阵 **物化（materialize）** 到显存，
成 HBM 带宽瓶颈且 O(N²) 显存。

FlashAttention 的做法（IO 感知）：
1. **分块（tiling）**：把 Q/K/V 分成小块，在 SRAM 上逐块算 softmax。
2. **在线 softmax（online softmax）**：用 running max/sum 增量更新，
   无需先看完整行就能算出正确归一化。
3. **不写回中间矩阵**：只输出最终结果，把 HBM 流量降到 `O(N)`。

代码里：
- `flash_fwd_kernel`：前向，分块算 `S=QK^T`、`P=softmax(S)`、`O=PV`。
- `flash_bwd_kernel_q` / `_kv` / `_d`：反向，分别算 dQ / dK,dV / dP。
- 两个 forward 变体：tiled（分块）与 `backward_no_tile`（无分块，对比用）。

> 面试点：FlashAttention 为什么快？→ 减少 HBM 读写（IO 感知）、
> 用 SRAM 高带宽、不物化 N×N 矩阵。复杂度从显存 O(N²) 降到 O(N)。

## 4. 测试坑（已知）

`test_flash_backward_triton` 在根环境（torch 2.5.1）下因 Triton 反向 kernel 的
数值精度未达 `rtol=0.01` 而 **xfail/failed**——这是作业代码本身的精度问题，
不是环境装错。标准 attention（PyTorch 参考实现）对照测试正常。
