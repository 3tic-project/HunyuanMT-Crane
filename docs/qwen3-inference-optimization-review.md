# Qwen3 推理优化对比评审与 Crane 优化路线图

## 1. 目标

本文对以下参考实现中的 **Qwen3 / 通用 LLM 推理优化** 做系统梳理：

- `vendor/qwen_megakernel`
- `vendor/sglang`
- `vendor/vllm`
- `vendor/lmdeploy`
- `vendor/flashinfer`
- `vendor/pytorch`

并结合以下 Crane 代码路径进行对比：

- `crane-core`
- `crane-oai`
- `crane`

目标是针对 **Qwen3，尤其是 Qwen3 1.7B**，识别当前仍未利用的优化空间，并按 **收益 / 实现复杂度 / 落地优先级** 给出建议，覆盖：

- 算法层
- Attention / KV Cache 组织
- 算子融合
- CUDA Kernel
- Candle 后端
- 服务端调度与 batch serving

---

## 2. 结论摘要

### 2.1 总体判断

Crane 当前已经具备一批对 Qwen3 有实际价值的模型内优化，包括：

- 预分配 KV cache + `slice_set`
- decode 场景下的 GQA-grouped SDPA
- fused RoPE
- merged QKV / gate+up projection
- fused SiLU-mul
- batched decode 基础设施
- KV cache save / restore
- 部分 GPU sampling 路径

这说明 Crane 在“**模型内部热点小融合**”上已经做得不错。

但和 `sglang / vllm / lmdeploy / flashinfer` 相比，Crane 当前最大的差距 **不再是少几个 elementwise fused op**，而是：

- **KV cache 组织方式仍偏“张量拼装”**，不是 `paged KV / block table`
- **attention backend 仍偏 Candle matmul 路线**，不是 `FlashInfer / TRTLLM / paged decode backend`
- **还没有为 decode backend 建立 `plan / run / workspace / metadata` 风格的 ABI / FFI 层**
- **batch decode 服务路径仍有较高固定成本**
- **scheduler 仍偏 `full-prompt prefill`，缺少 `chunked prefill + page-budget admission`**
- **缺少 decode CUDA Graph / 元数据复用机制**
- **prefix cache / radix cache / block-hash cache 未建立**
- **KV cache 量化还未成为一等公民**
- **decode 热路径仍较深地依赖 Candle Tensor 抽象**

### 2.2 面向 Qwen3 1.7B 的最高优先级

如果目标是提升 **在线服务吞吐、并发能力、长上下文效率**，推荐优先级如下：

1. **Paged KV cache + block table 改造**
2. **接入 FlashInfer / TRTLLM 类 decode attention backend**
3. **Decode CUDA Graph**
4. **Prefix cache / radix cache**
5. **KV cache quantization**
6. **把已有但未 fully 接入的 fused kernel 真正用进热路径**

同时有两个经常被低估、但实际上必须前置的工程条件：

- **尽早锁定 backend-native metadata ABI**，例如 `paged_kv_indptr / indices / last_page_len / block_tables`
- **尽早把 scheduler 从“完整 prompt 一次性 prefill”推进到可做 chunked prefill 的方向**

### 2.3 不建议优先投入的方向

短期内不建议把最高优先级放在以下方向：

- 继续增加零散小型 elementwise fused op
- 直接照搬 `qwen_megakernel` 式超大 megakernel
- 对 Qwen3 做偏研究性质的架构级近似改造

原因是这些方向的 **工程代价 / 维护成本 / 通用性** 明显不如 `paged KV + backend integration + cuda graph`。

---

## 3. Crane 当前 Qwen3 能力盘点

### 3.1 已有优化

结合 `crane-core/src/models/qwen3/modeling.rs`、`crane-core/src/fused_ops/cuda_impl.rs`、`crane-oai/src/engine/backend.rs`、`crane-oai/src/engine/mod.rs`，当前 Crane 的 Qwen3 相关优化主要包括：

#### 3.1.1 KV cache 相关

- **预分配 KV cache buffer**，避免每步 decode 直接 `cat`
- 使用 `slice_set` 做 **in-place KV 写入**
- 提供 `get_kv_caches` / `set_kv_caches`
- 提供 `setup_batch_decode` / `step_batch_decode` / `extract_batch_kv`
- 支持服务层连续 batch 的 KV save / restore

#### 3.1.2 Attention 路径

- 对 `seq_len=1` decode 路径做 **GQA-grouped SDPA**
- 避免 naive KV head 扩展
- 使用 `candle_nn::rotary_emb::rope()` 作为 fused RoPE kernel
- 已支持 Q/K norm 配置路径

#### 3.1.3 MLP 路径

- `gate + up` 已合并投影
- 自定义 CUDA `fused_silu_mul`
- `down_proj` 保持单独 matmul

#### 3.1.4 采样路径

- `gpu_argmax`
- `topk_indices`
- GPU Gumbel-max
- GPU-friendly repetition penalty
- 对大词表 top-p 场景仍保留 CPU fallback

#### 3.1.5 Candle/CUDA 运行时层

- 关闭 event tracking，减少单流场景额外开销
- 有独立 `fused_ops.cu`
- 已经开始构建自定义 PTX kernel 能力

### 3.2 当前主要瓶颈

Crane 的主要瓶颈并不在“模型完全没优化”，而在以下几个结构性问题：

#### 3.2.1 batched decode 仍然是“重组型”流程

`crane-oai/src/engine/mod.rs` 当前 batched decode 逻辑基本是：

- 从每个 sequence 取出 KV
- `setup_batch_decode` 做 pad / stack
- decode 若干 round
- `extract_batch_kv` 再拆回每个 sequence

这意味着 batch serving 过程中存在明显的：

- KV tensor 重组成本
- 额外拷贝成本
- setup / extract 固定开销
- 与 batch size、轮数强耦合的服务端额外负担

#### 3.2.2 attention backend 仍以 Candle matmul 为主

虽然 decode 路径已有 grouped GQA 形状优化，但 attention 主体仍是：

- `q.matmul(k^T)`
- `softmax`
- `matmul(v)`

与 `FlashInfer / TRTLLM / paged decode backend` 相比，仍缺：

- paged KV 原生支持
- split-k / persistent / specialized decode kernel
- prefill / decode backend 分流
- backend 元数据计划与复用

#### 3.2.3 服务端没有 prefix cache / radix cache

当前没有看到：

- block-level prefix hashing
- radix tree prefix matching
- page-aligned prefix reuse
- 多请求共享前缀的低成本复用

这会直接影响 chat 服务场景下的吞吐。

#### 3.2.4 scheduler 仍是“完整 prompt prefill 优先”

`crane-oai/src/engine/scheduler.rs` 当前策略基本是：

- 等待队列一次只 prefill 一个请求
- 一个 prefill step 吃完整个 prompt
- decode step 则处理所有 running request

这个策略在 prompt 较短时还算直接，但在长 prompt 与活跃 decode 混跑时会造成明显阻塞，也不利于后续做 page-budget admission control。

#### 3.2.5 CUDA Graph 尚未进入 Qwen3 服务热路径

对于 1-token decode、固定 batch bucket、固定 metadata 的场景，decode 非常适合 CUDA Graph。但当前 Crane 路径没有建立这套机制。

而且这里的“固定 metadata”不能只理解为固定 batch size，通常还需要：

- `paged_kv_indptr / last_page_len` 的形状稳定
- `split-kv` 策略稳定
- workspace / plan buffer 生命周期稳定

#### 3.2.6 已有 kernel 仍有“写了但没 fully 落地”的情况

例如 `FusedAddRmsNorm` 在 `crane-core/src/fused_ops/cuda_impl.rs` 中已有结构和方向，但 `cuda_fwd_inplace()` 当前仍回退到 Candle 两步路径，没有真正把 fused kernel 收益吃满。

#### 3.2.7 Candle 热路径抽象仍偏厚

当前 decode 热路径仍大量建立在：

- `Tensor::new`
- `contiguous()`
- `reshape / transpose / narrow`
- 模型层直接承担 attention 细节

这对通用实现是合理的，但对 `Qwen3 1.7B` 极限 serving 来说，会让 backend-native buffer 复用、graph capture、原地写入与 workspace 生命周期管理都变得更难。

---

## 4. Vendor 优化模式总结

## 4.1 qwen_megakernel

### 4.1.1 核心特点

`vendor/qwen_megakernel/csrc/kernel.cu` 体现的是一种极端优化路线：

- 单 token decode 的 **超大 fused cooperative kernel**
- embedding、28 层 transformer、final norm 等大范围融合
- 融合 RMSNorm、QKV、RoPE、attention、MLP
- `__ldg`、warp shuffle、persistent-style scheduling
- 通过 grid sync / barrier / prefetch 提高 cache 命中与流水化程度

### 4.1.2 对 Crane 的启示

它最有价值的不是“直接照搬 megakernel”，而是以下思路：

- **将 decode 路径视为特殊问题**，与 prefill 分开处理
- **QK norm + RoPE + KV write** 可以进一步融合
- **尽量减少 kernel launch 与中间 tensor 物化**
- **为固定模型结构做更激进 specialization**

### 4.1.3 不宜直接照搬的部分

这种 megakernel 方案高度绑定：

- 固定层数
- 固定 head_dim
- 固定 hidden size
- 固定 GPU 架构
- 固定调度假设

因此对 Crane 这类通用框架来说，不适合直接作为第一阶段路线。

---

## 4.2 sglang

### 4.2.1 核心特点

`sglang` 的价值主要不在“Qwen3 模型代码本身”，而在 serving 基础设施：

- `RadixAttention`
- `RadixCache`
- `FlashInferAttnBackend`
- paged KV cache wrapper
- CUDA Graph capture
- 多种 KV pool allocator
- 对不同 attention / sliding window / sparse 场景的池化设计

### 4.2.2 对 Crane 的启示

最重要的启示有三点：

- **prefix cache 应该成为服务引擎能力，而不是模型内特性**
- **KV cache 应该抽象成 page/block allocator，不应继续依赖整张量搬运**
- **attention backend 需要独立于模型层进行规划与复用**

---

## 4.3 vllm

### 4.3.1 核心特点

`vllm` 的重点是：

- paged attention / paged KV
- FlashInfer backend
- TRTLLM backend
- KV block hash / prefix cache
- FP8 / FP4 KV dtype 管理
- 多层 metadata 管理
- `MultiLevelCascadeAttentionWrapper`

### 4.3.2 对 Crane 的启示

- **Qwen3 1.7B 的核心优化对象应从“单模型 forward”转为“KV + backend + serving metadata”**
- **KV cache dtype 需要成为可配置能力**
- **block hash / cache salt / extra key 机制是 prefix cache 可扩展性的关键**

---

## 4.4 lmdeploy

### 4.4.1 核心特点

`lmdeploy` 兼具 PyTorch rewrite 与 Turbomind CUDA 实现：

- `CudaGraphMixin`
- 专门 attention layer
- 分离 prefill / decode 执行策略
- `ProcessKV_v2` / `flattenKV_v2`
- KV cache int8 / int4 支持
- RoPE / quantized KV store 在写 cache 时处理

### 4.4.2 对 Crane 的启示

- **KV cache quantization 应尽量和 KV write / read 路径整合**
- **decode cudagraph 的输入 buffer 需要稳定形状和缓存化 metadata**
- **prefill 与 decode 应使用不同的 attention 执行模型**

---

## 4.5 flashinfer

### 4.5.1 核心特点

`flashinfer` 提供的是高性能 attention backend 基础设施：

- batch prefill / batch decode wrapper
- paged KV cache wrapper
- ragged / paged 统一接口
- split-k / persistent / cooperative kernel
- 支持不同 backend：`fa2`、`fa3`、`trtllm-gen`
- 对 workspace、计划元数据、block tables 的精细管理

### 4.5.2 对 Crane 的启示

- **attention backend 不应只暴露“算子”，还应暴露 plan / run / workspace 语义**
- **Paged decode wrapper 是服务端推理系统的真正核心接口之一**
- **graph capture 不只是固定 batch size，还要固定 `indptr / last_page_len / split-k` 等 metadata 约束**
- **persistent kernel 适合在 decode 场景作为更高阶段优化目标**

---

## 4.6 pytorch

### 4.6.1 核心特点

PyTorch 这部分不是直接 serving 实现，但提供了重要参考：

- fused qkv transform / bias rescale
- FlashAttention / mem-efficient attention / cuDNN backend dispatch
- 按输入形状和条件选择 SDP backend
- 对 dense / varlen 路径做 backend 分流

### 4.6.2 对 Crane 的启示

- **backend 选择逻辑应该被显式建模，而不是把所有场景压成一条 matmul 路线**
- **Qwen3 prefill / decode 不一定应该共用同一种 attention 实现**

---

## 5. 差距矩阵

| 维度 | Crane 当前状态 | Vendor 常见状态 | 差距判断 |
|---|---|---|---|
| QKV / gate-up 合并 | 已支持 | 普遍支持 | 差距小 |
| decode GQA 优化 | 已支持 | 普遍支持 | 差距小 |
| fused RoPE | 已支持 | 普遍支持 | 差距小 |
| fused MLP gate | 已支持 | 普遍支持 | 差距小 |
| paged KV | 未形成正式架构 | 核心能力 | **差距大** |
| prefix / radix cache | 未见正式实现 | 核心能力 | **差距大** |
| decode backend | Candle matmul 为主 | FlashInfer / TRTLLM / custom backend | **差距大** |
| backend ABI / FFI | 未形成专门 `plan/run/workspace` 接口 | backend-native 接口是常态 | **差距大** |
| decode CUDA Graph | 未见正式接入 | 常见高端优化 | **差距大** |
| chunked prefill / admission control | 仍是 full-prompt prefill | 常见 serving 优化 | **差距大** |
| graph shape 治理 | 未见 page/split-k bucket 策略 | 常见工程前提 | **差距大** |
| KV cache quantization | 局部/未系统化 | 多 backend 一等公民 | **差距大** |
| metadata / workspace 复用 | 较弱 | 很强 | **差距大** |
| Candle 热路径旁路 | 热点仍强依赖 Tensor 抽象 | 热路径常直达 backend buffer | **差距中到大** |
| persistent attention kernel | 无 | 部分实现 | 差距中到大 |
| decode 小 batch GEMV/GEMM | 仍随通用 matmul dispatch | 常见继续 specialization | 差距中到大 |
| full megakernel | 无 | 少数极端实现 | 差距存在但不应优先 |

---

## 6. 面向 Qwen3 1.7B 的优先级建议

## 6.1 P0：最高优先级

### P0-1. Paged KV cache + block table

#### 原因

这是 Crane 与 `vllm / sglang / lmdeploy` 之间最大的结构性差距。

#### 当前问题

Crane batch decode 仍依赖：

- setup 时 pad / stack
- 结束时 extract / 拆分

#### 预期收益

- 大幅降低 batch decode 固定成本
- 减少 KV 重组和额外拷贝
- 为 prefix cache、FlashInfer、CUDA Graph 提供统一基础

#### 实现难度

高

#### 推荐落点

- `crane-core`：新增 paged KV 数据结构与 layer 读写接口
- `crane-oai`：重写 batch decode 生命周期，改成 page table / seq_lens 更新逻辑

---

### P0-2. 接入 FlashInfer / TRTLLM 类 decode attention backend

#### 原因

对 Qwen3 1.7B 来说，在线 decode 性能瓶颈通常比 prefill 更敏感于：

- launch overhead
- KV bandwidth
- batch metadata 组织

#### 预期收益

- 提高多并发 decode 吞吐
- 降低长上下文下 decode 时延
- 让 paged KV 的收益真正落地

#### 实现难度

高

#### 推荐落点

- 在 `crane-core` 增加 attention backend 抽象
- 先做 `decode-only backend`
- 再逐步覆盖 prefill
- 同步建立 Rust <-> CUDA 的 `plan / run / reset_workspace / destroy` ABI
- 第一版就锁定 backend-native metadata buffer 形状与所有权

---

### P0-3. Decode CUDA Graph

#### 原因

1-token decode、固定 bucket batch、固定 page metadata 场景非常适合 graph capture。

#### 预期收益

- 降低 CPU 发射开销
- 提高稳定吞吐
- 改善小 batch / 中小模型场景下的系统效率

#### 实现难度

中

#### 前提

- 最好先完成 paged KV 或至少稳定 batch metadata 形状
- bucket 不能只按 batch size，还要考虑 `max_kv_pages_per_seq / total_kv_pages / split-k` 的稳定性

---

## 6.2 P1：高优先级

### P1-1. Chunked prefill + page-budget admission control

#### 原因

当前 scheduler 是：

- 一个请求完整 prefill
- 所有 running 请求统一 decode

这对长 prompt 混跑非常不友好，也会放大 TTFT 与 decode 抖动。

#### 预期收益

- 降低长 prefill 对活跃 decode 的阻塞
- 更容易做 paged KV 的容量治理
- 为 prefix cache 与 graph bucket 提供更稳定的运行环境

#### 实现难度

高

#### 推荐方向

- prefill 改成 chunk-based 推进
- admission 控制基于 page budget，而不是仅基于 running request 数量

---

### P1-2. Prefix cache / Radix cache

#### 原因

在 chat 服务中，大量请求共享：

- system prompt
- tool schema
- few-shot 前缀
- 模板 token

#### 预期收益

- 显著降低重复 prefill 成本
- 提高真实服务场景吞吐

#### 实现难度

高

#### 推荐方向

- 先做 block hash prefix cache
- 再考虑 radix tree 增强匹配能力

---

### P1-3. KV cache quantization

#### 原因

这不仅是显存问题，更是：

- KV 读写带宽问题
- 可承载并发数问题
- 长上下文场景下的系统容量问题

#### 预期收益

- 提高并发
- 提高长上下文可承载能力
- 为服务端成本优化提供空间

#### 实现难度

高

#### 推荐顺序

- 先支持 int8 / fp8 KV cache
- 后续再考虑更激进方案

---

### P1-4. 把 `FusedAddRmsNorm` 真正接入热路径

#### 原因

这是“已经投入但还没 fully 兑现”的优化点。

#### 当前问题

`crane-core/src/fused_ops/cuda_impl.rs` 中相关结构已存在，但 CUDA 路径实际仍回退到 Candle 两步操作。

#### 预期收益

- 减少 residual add + rmsnorm 之间的中间开销
- 对每层 decode / prefill 都有累积收益

#### 实现难度

中

---

### P1-5. Sampling 全 GPU 化

#### 当前问题

`crane-oai/src/engine/sampling.rs` 中：

- greedy + 无 repetition penalty 才走 `gpu_argmax`
- 大词表 top-p 场景仍可能 fallback 到 CPU `LogitsProcessor`
- top-k 后仍有部分 gather / 标量取值路径

#### 预期收益

- 在高 QPS、小 batch、较小模型场景下降低 host/device 往返

#### 实现难度

中

---

## 6.3 P2：次优先级，但值得长期推进

### P2-1. 融合 QK norm + RoPE + KV write

#### 来源

- `qwen_megakernel`
- `lmdeploy ProcessKV_v2`

#### 价值

- 进一步减少 decode 热路径 kernel launch
- 为 KV quantization 提供自然接入点

#### 难度

高

---

### P2-2. Prefill / decode backend 分流

#### 思路

- prefill 使用更偏 throughput 的 context attention backend
- decode 使用更偏 low-latency / paged 的 decode backend

#### 难度

高

---

### P2-3. metadata / workspace 复用机制

#### 内容

- 复用 `indptr`
- 复用 `block tables`
- 复用 `seq_lens`
- 复用 backend `plan_info`
- 复用 workspace buffer

#### 收益

对服务层很重要，但依赖 paged KV / backend 抽象先成型。

---

### P2-4. Decode 小 batch GEMV / GEMM specialization

#### 原因

当 `paged KV + decode backend + cudagraph` 这些结构性优化完成后，`QKV / O / gate_up / down` 的小 `M` matmul 往往会变成新的瓶颈。

#### 推荐方向

- 针对 `M=1` 或小 batch decode 的 `cublasLt / CUTLASS / custom matvec`
- merged `QKV / gate_up` 权重的 backend-native prepack
- 在 Qwen3 decode 路径上逐步绕开 Candle 默认 linear dispatch

#### 难度

高

---

## 7. 不建议优先投入的方向

### 7.1 直接做超大 megakernel

`qwen_megakernel` 很强，但不适合作为当前 Crane 的第一优先级：

- 过于绑定模型结构
- 维护成本高
- 通用性差
- 对不同 GPU 兼容性要求高

### 7.2 继续堆更多小型 fused op

Crane 已经有：

- fused RoPE
- fused SiLU-mul
- GPU argmax
- GPU top-k

继续在这一层加法，收益已经不如把 attention backend 与 KV 组织做对。

### 7.3 先做架构级近似 attention 改造

如：

- cross-layer attention 迁移
- 激进稀疏注意力
- 近似推理策略

这类路线偏研究性质，不适合作为当前 Qwen3 1.7B 工程优化第一优先级。

---

## 8. 推荐实施路线

## 8.1 第一阶段：先把 serving 架构做对

### 目标

围绕 `Qwen3 1.7B` 建立 production-style decode 基础设施。

### 任务

1. 设计 paged KV / block table 抽象
2. 第一版就锁定 backend-native metadata ABI
3. 建立 Rust <-> CUDA 的 `plan / run / workspace` FFI 层
4. 重写 `crane-oai` batch decode 生命周期，去掉 extract→pad→stack→extract 主循环
5. 引入 decode attention backend 抽象，并优先接入 decode-only backend
6. 建立 decode bucket + cudagraph 基础设施，分桶不只看 batch size

### 预期结果

- decode 吞吐明显提升
- batch serving 固定成本明显下降
- 后续 prefix cache / KV quant / backend 扩展有统一基础

### 2026-03-26 当前实现对照

- `paged KV / block table`：
  - 已落地第一版抽象与 allocator/metadata 测试骨架，代码位于 `crane-core/src/models/qwen3/paged_kv.rs`
  - 目前先锁 ABI 与 page/bucket 语义，真实 page-store 与 page-write kernel 仍待远端 CUDA 验证
- `backend-native metadata ABI`：
  - 已固定 `paged_kv_indptr / paged_kv_indices / paged_kv_last_page_len / block_tables`
- `plan / run / workspace` 风格 decode backend：
  - 已在 Rust 层建好 `Qwen3DecodeBackend` 与 `DecodeBackendPlan`
  - engine 已切换到 `plan_batch_decode + run_planned_batch_decode` 生命周期
- `去掉每轮 extract→pad→stack→extract`：
  - 已通过 active batch session 复用大幅减少主循环中的 setup/extract 频率
  - 当前仍是过渡实现，batch 变化或 session flush 时仍会回到旧式 batched KV 提取
- `短 prompt admission`：
  - 针对 `prompt <= 2k` 的实际服务场景，scheduler 已加入 decode-burst admission
  - waiting queue 增长、prefill 完成时，会优先保住几轮 decode，再补新 prefill；但 `running batch shrink` 不再延迟 admission，因为当前 tensor-KV 引擎在完成/取消后已经 flush active session，继续停留在 3-lane decode 没有 reuse 收益
- `decode bucket + graph 基础设施`：
  - bucket key、plan cache hit 统计和 metadata 统计已具备
  - CUDA Graph 仍未正式接入，需要远端服务器继续 capture 验证
- 当前默认仍以吞吐优先：
  - `prefill_chunk_size` 默认回到 `0`
  - 在真正 paged KV / decode backend 落地前，chunked prefill 需要显式开启，否则会频繁打断 tensor batch decode session
  - 远端验证时应重点观察 batched decode 日志里 `reuse_session=true` 是否开始连续出现，以及 decode/prefill 是否不再每轮交替

---

## 8.2 第二阶段：补齐高价值功能

### 任务

1. chunked prefill + page-budget admission control
2. prefix cache / radix cache
3. KV cache quantization
4. `FusedAddRmsNorm` 真正接入热路径
5. sampling 全 GPU 化
6. metadata / workspace 复用

---

## 8.3 第三阶段：冲击更高天花板

### 任务

1. QK norm + RoPE + KV write 融合
2. prefill / decode 后端完全分流
3. persistent decode kernel
4. decode 小 batch GEMV / GEMM specialization
5. 更强的 model-specific specialization

---

## 9. 对 Qwen3 1.7B 的最终建议

如果只保留最关键的一句话建议：

> **Crane 下一阶段最值得投入的不是再写几个局部 fused kernel，而是把 Qwen3 1.7B 的服务路径升级为 `paged KV + specialized attention backend + decode cudagraph`。**

更具体地说：

- **第一名**：`paged KV + block table`
- **第二名**：`backend-native metadata ABI + FlashInfer / TRTLLM-style decode backend`
- **第三名**：`decode CUDA Graph`
- **第四名**：`chunked prefill + page-budget scheduling`
- **第五名**：`prefix/radix cache`
- **第六名**：`KV cache quantization`

这里尤其要强调：

- **“接后端”真正要落地的是 ABI / FFI / workspace / metadata buffer，不是简单替换一个 attention op**
- **`ProcessKV_v2` 风格的 `KV page write + optional quantization` 非常值得在 paged KV 落地后尽快验证**

---

## 10. 关键证据文件

以下文件在本次评审中起关键作用：

### Crane

- `crane-core/src/models/qwen3/modeling.rs`
- `crane-core/src/models/qwen3/model.rs`
- `crane-core/src/fused_ops/cuda_impl.rs`
- `crane-core/kernels/fused_ops.cu`
- `crane-oai/src/engine/backend.rs`
- `crane-oai/src/engine/mod.rs`
- `crane-oai/src/engine/scheduler.rs`
- `crane-oai/src/engine/sampling.rs`

### Vendor

- `vendor/qwen_megakernel/csrc/kernel.cu`
- `vendor/qwen_megakernel/qwen_megakernel/model.py`
- `vendor/sglang/python/sglang/srt/models/qwen3.py`
- `vendor/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
- `vendor/sglang/python/sglang/srt/mem_cache/radix_cache.py`
- `vendor/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `vendor/vllm/vllm/model_executor/models/qwen3.py`
- `vendor/vllm/vllm/v1/attention/backends/flashinfer.py`
- `vendor/vllm/vllm/v1/core/kv_cache_utils.py`
- `vendor/lmdeploy/lmdeploy/pytorch/models/qwen3.py`
- `vendor/lmdeploy/lmdeploy/pytorch/models/utils/cudagraph.py`
- `vendor/lmdeploy/src/turbomind/models/llama/unified_attention_layer.cc`
- `vendor/lmdeploy/src/turbomind/kernels/attention/kv_cache_utils_v2.cu`
- `vendor/flashinfer/flashinfer/decode.py`
- `vendor/flashinfer/flashinfer/prefill.py`
- `vendor/flashinfer/include/flashinfer/attention/persistent.cuh`
- `vendor/pytorch/aten/src/ATen/native/transformers/cuda/attention.cu`

---

## 11. 简短行动清单

如果要立刻进入实施，建议从下面四项开始：

1. **设计 paged KV 数据结构和 layer 接口**
2. **锁定 backend-native metadata ABI，并建立 Rust/CUDA `plan-run-workspace` 接口**
3. **把 batched decode 从“KV tensor 重组”改为“metadata 更新”**
4. **把 scheduler 推向 chunked prefill + page-budget admission**

完成这些前置后，再做：

5. decode CUDA Graph
6. prefix cache
7. KV cache quantization

---

## 12. 总结

Crane 现在的问题不是“完全没优化”，而是优化重心仍偏模型内部局部融合；而 `Qwen3 1.7B` 在线推理真正的上限，更取决于：

- KV cache 是否 paged 化
- backend-native ABI / metadata buffer 是否先锁定
- attention backend 是否 specialized
- scheduler 是否支持 chunked prefill 与 page-budget 控制
- decode 是否 cudagraph 化
- prefix / cache / metadata 是否被系统化建模

因此，下一阶段最应该做的是 **服务态推理基础设施升级**，而不是继续围绕少量点状 kernel 做局部微调。
