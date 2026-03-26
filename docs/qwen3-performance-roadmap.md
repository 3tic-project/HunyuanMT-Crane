# Qwen3 极致性能工程执行 Roadmap

## 1. 文档目的

本文不是通用架构评审，也不是多模型兼容方案。

本文只回答一个问题：

> **如果暂时不考虑其他模型兼容性，只追求 `Qwen3`，尤其是 `Qwen3 1.7B` 在 Crane 中达到尽可能高的在线推理性能，工程上应该怎么分阶段实施？**

因此，本文默认允许以下策略：

- 为 `Qwen3` 单独定制数据结构和执行路径
- 为 `Qwen3 1.7B` 单独做 kernel specialization
- 暂时牺牲 `crane-core` 中通用抽象的优雅性
- 暂时不保证 `Qwen2.5 / Hunyuan / Qwen3.5` 直接复用同一路径
- 暂时只围绕 **CUDA + Qwen3** 路线优化

本文目标是给出一份 **工程可执行** 的 roadmap，包括：

- 阶段拆分
- 代码落点
- 依赖顺序
- 验收指标
- 风险控制
- 建议的取舍边界

---

## 2. 优化目标

## 2.1 北极星目标

针对 `Qwen3 1.7B`，把 Crane 的在线服务路径优化到接近 `sglang / vllm / lmdeploy` 一类 serving 系统的有效水平。

这里的“最优”不定义为单一 microbenchmark，而定义为以下几类指标的综合最优：

- **[TTFT]** 首 token 时延尽可能低
- **[Decode tok/s]** 多并发 decode 吞吐尽可能高
- **[Concurrency]** 在同等显存下支撑更高并发
- **[Context Efficiency]** 长上下文下性能退化尽可能小
- **[Serving Overhead]** 服务框架额外开销尽可能低
- **[Stability]** 不因 batch 波动、上下文长度波动而明显抖动

## 2.2 建议先锁定的目标场景

为了避免路线发散，建议第一阶段先锁定下面这个明确目标：

- **模型**：`Qwen3 1.7B Instruct`
- **精度**：`BF16`
- **硬件**：单张 NVIDIA GPU
- **任务**：chat serving
- **核心负载**：prefill + 多请求并发 decode
- **上下文档位**：`4K / 8K / 32K`
- **batch 桶位**：`1 / 2 / 4 / 8 / 16 / 32`

如果目标不锁死，工程上就会不断被“兼容性”和“抽象复用”拖慢。

---

## 3. 非目标

下面这些事情在本 roadmap 中明确 **不作为第一优先级**：

- 支持其他模型零修改复用
- 保持现有通用 backend trait 完整不变
- 先优化 CPU / Metal / 非 CUDA 路径
- 优先做更多零散 fused elementwise op
- 优先做完整 megakernel
- 优先做学术型近似 attention / 稀疏策略

这份 roadmap 的原则是：

> **只做最直接影响 Qwen3 线上性能上限的事情。**

---

## 4. 核心设计决策

如果只面向 Qwen3 极致性能，建议尽早做出以下设计决策。

## 4.1 Qwen3 走独立高性能路径

在 `crane-core/src/models/qwen3` 和 `crane-oai` 中建立 **Qwen3 high-performance path**，不要强行复用所有通用路径。

建议接受以下现实：

- `Qwen3` 的 prefill 和 decode 不是同一个问题
- `Qwen3` 的 serving 性能瓶颈更多在 `KV + attention backend + batch scheduling`
- 为 `Qwen3` 单独改造数据结构，比维护“漂亮的统一抽象”更值

## 4.2 Prefill / Decode 分家

明确把执行模型拆成两条路线：

- **Prefill Path**
  - 面向大序列、大吞吐
  - 关注 context attention 吞吐
  - 更适合 FA/FlashInfer/TRTLLM context backend

- **Decode Path**
  - 面向 `seq_len=1`
  - 关注 launch overhead、KV 带宽、batch metadata 组织
  - 更适合 paged decode backend、CUDA Graph、specialized kernel

如果 prefill / decode 不分家，后续所有优化都会被一条中庸路径稀释。

## 4.3 KV cache 从“Tensor”升级为“系统资源”

Qwen3 serving 下，KV cache 不能再被看成“每个请求一对 K/V tensor”。

它应该被建模为：

- block/page allocator
- block table / page table
- sequence to blocks 映射
- layer-local or global page metadata
- prefix cache 可复用对象
- backend 直接消费的 layout

也就是说：

> **KV cache 是服务系统的一等公民，而不是模型 forward 的副产物。**

## 4.4 接受必要的硬编码与 specialization

为了追求 Qwen3 最优性能，第一阶段可以接受以下事情：

- 针对 `head_dim=128` specialization
- 针对 `Qwen3 1.7B` 的 `num_heads / num_kv_heads / layer_count` 做常量路径
- 只支持某种 KV layout
- 只支持某几个 page size
- 只支持 BF16 主路径

这些选择在“极致性能版本”里是合理的。

## 4.5 Decode 热路径逐步脱离 Candle Tensor 抽象

如果目标是 `Qwen3 1.7B` 的极限在线性能，建议尽早接受下面这件事：

- Candle 继续负责：权重加载、冷路径、fallback、非热点算子
- Qwen3 runtime 负责：shape / bucket 决策、buffer 生命周期、上游张量准备
- backend / CUDA 层负责：`plan / run / workspace / graph-friendly input buffer`

原因很直接：

- 当前 `FusedAddRmsNorm` 已有 kernel，但仍受 Candle 原地写能力限制
- `FlashInfer / TRTLLM` 风格接口本质上不是“单个算子”，而是 `plan + run + metadata + workspace`
- 如果 decode 热路径继续建立在频繁的 `Tensor::new / contiguous / index_select` 上，后续很多优化会被抽象层开销稀释

因此这份 roadmap 里应显式预留：

- Rust <-> C/CUDA 的 ABI / FFI shim
- backend-native metadata buffer 管理
- decode-only path 上逐步减少 Candle Tensor 热路径参与度

---

## 5. 当前代码中的关键落点

后续 roadmap 会反复涉及以下文件：

### 5.1 模型层

- `crane-core/src/models/qwen3/modeling.rs`
- `crane-core/src/models/qwen3/model.rs`

### 5.2 CUDA / kernel 层

- `crane-core/src/fused_ops/cuda_impl.rs`
- `crane-core/kernels/fused_ops.cu`

### 5.3 服务层 / engine

- `crane-oai/src/engine/backend.rs`
- `crane-oai/src/engine/mod.rs`
- `crane-oai/src/engine/scheduler.rs`
- `crane-oai/src/engine/sampling.rs`

### 5.4 参考实现

- `vendor/vllm/vllm/v1/attention/backends/flashinfer.py`
- `vendor/vllm/vllm/v1/core/kv_cache_utils.py`
- `vendor/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
- `vendor/sglang/python/sglang/srt/mem_cache/radix_cache.py`
- `vendor/lmdeploy/src/turbomind/models/llama/unified_attention_layer.cc`
- `vendor/lmdeploy/src/turbomind/kernels/attention/kv_cache_utils_v2.cu`
- `vendor/flashinfer/flashinfer/decode.py`
- `vendor/flashinfer/flashinfer/prefill.py`
- `vendor/qwen_megakernel/csrc/kernel.cu`

---

## 6. 分阶段工程计划

## 6.1 Phase 0：建立基线与性能护栏

### 目标

在任何大改动之前，先把性能测量体系钉死，否则后面很容易“感觉变快了”，但无法确认。

### 交付物

- 一套固定 benchmark 工具或脚本
- 一套固定 workload
- 一套固定统计指标
- 一份当前 baseline 结果

### 必测 workload

- **[W1]** 单请求短 prompt，短生成
- **[W2]** 单请求长 prompt，短生成
- **[W3]** 8 并发，短 prompt，中等生成
- **[W4]** 32 并发，system prompt 相同
- **[W5]** 长上下文 `8K / 32K` decode
- **[W6]** 高 batch decode，固定 `decode_tokens_per_seq`
- **[W7]** 一个超长 prefill 与一批活跃 decode 请求混跑

### 必测指标

- `TTFT`
- `prefill tok/s`
- `decode tok/s`
- `end-to-end req/s`
- `p50 / p95 / p99 latency`
- `peak VRAM`
- `active KV bytes`
- `batch decode setup time`
- `decode plan / metadata build time`
- `decode graph hit rate`
- `page alloc/free time`
- `H2D metadata bytes`
- `prefix cache hit rate`
- `sampling time`

### 代码落点

- `crane-oai/src/engine/mod.rs`：增加细粒度阶段计时埋点
- benchmark 脚本：可放 `scripts/` 或 `example/benchmark/`

### 验收标准

- 每次改动后都能重复跑同一套负载
- 基线结果可复现
- 可以独立分离 `prefill / decode / sampling / setup` 时间占比

### 备注

这是后续所有 phase 的前置条件。

### 2026-03-26 当前实现状态

- 已新增 `scripts/benchmark_qwen3_serving.py`，可直接对 `crane-oai` 的 `/v1/chat/completions` 与 `/v1/stats` 跑固定 workload。
- `EngineStats` 已补充 `prefill/decode/setup/extract/mask/plan/sampling/H2D metadata/page-budget` 相关计数。
- `EngineStats` 现可暴露 `current_tracked_kv_bytes` 与 `current_estimated_kv_pages`，便于 Phase 0 持续采样。
- 针对 `prompt <= 2k` 的主场景，scheduler 已加入 decode-first admission burst：当 waiting queue 增长、prefill 晋升为 running 时，会优先保住 `4+ lane` 稳定 decode 批；但 `batch shrink` 不再延迟 admission，因为当前 tensor-KV 引擎在完成/取消后会 flush active session，此时继续跑 3-lane decode 没有 session reuse 收益。
- 在高 backlog 场景下，scheduler 现会关闭 decode-burst，直接优先把 batch 补满，避免 sustained load 下长期停留在过小 batch。
- 如果此前发生过 eviction，`effective_max_running` 也不再要求 waiting 清空才恢复，而是会在持续 headroom 下逐步放松，避免把并发长期钉死在过低值。
- 远端 CUDA 环境下建议配合 `nvidia-smi` 轮询使用；benchmark 脚本已经内置可选显存采样。

示例：

```bash
python3 scripts/benchmark_qwen3_serving.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen3-1.7B-Instruct \
  --workload w1 --workload w3 --workload w4 --workload w7 \
  --json-out /tmp/qwen3-bench.json
```

---

## 6.2 Phase 1：先去掉现有路径中的明显结构性浪费

### 目标

在不引入 paged KV 之前，先把当前 Qwen3 路径里最容易拿到的收益拿掉。

### 任务

- **[任务 1]** Qwen3 prefill / decode 路径显式分开
- **[任务 2]** 清理 batch decode 中明显可避免的 host/device 往返
- **[任务 3]** 让 `FusedAddRmsNorm` 真正进入热路径，或明确放弃并删除死方向
- **[任务 4]** 让 sampling 尽可能走 GPU 路径
- **[任务 5]** 减少 `Tensor::contiguous()` 的重复调用与中间 tensor 物化

### 重点代码

- `crane-core/src/models/qwen3/modeling.rs`
- `crane-core/src/fused_ops/cuda_impl.rs`
- `crane-oai/src/engine/sampling.rs`

### 预期收益

- 小到中等
- 主要作用是为后续改造清理现场

### 验收标准

- `sample()` 中 CPU fallback 触发比例显著下降
- 每 token kernel launch 数减少
- `batch decode setup` 的时间占比下降

### 是否必须完成

不是最终决定性 phase，但非常建议先做，否则后续难定位收益归因。

### 2026-03-26 当前实现状态

- Qwen3 prefill / decode 已在 engine 层显式分家，并支持 `prefill_chunk_size` 驱动的 chunked prefill。
- batch decode 热路径已引入 active session 复用，避免每个 scheduling round 都做 `extract -> pad/stack -> decode -> extract`。
- `crane-core/src/models/qwen3/modeling.rs` 上一轮 tensor-KV workspace / scatter 过渡实验已回滚到 `15a88b7` 对应状态：
  - 远端压测表明，这条路线在 sustained load 下仍会把瓶颈留在 `setup_batch_decode + flush/extract`
  - 因此不再继续扩大这条 tensor 搬运路径，而是把后续实现集中到 paged-KV runtime / metadata / backend ABI
- scheduler 已进一步加入短 prompt 吞吐优先的 decode-burst admission 策略，当前主要覆盖 `新请求入队 / prefill 完成` 两类事件；当前 tensor-KV 路径下，这比“slot 一空就立刻 prefill”更符合 Qwen3 1.7B serving 的真实瓶颈。对 `batch shrink`，当前实现会直接补位，因为 active batch session 已被引擎 flush，延迟 prefill 只会制造低吞吐的 3-lane decode。
- 当 waiting backlog 很高时，burst 会被自动关闭，优先扩 batch，而不是为了局部 `reuse_session=true` 牺牲整体吞吐与排队长度。
- sampling 时间已进入 engine 细粒度统计；现有 GPU fast path 继续保留 `gpu_argmax / topk / gumbel-max / repetition penalty`。
- `FusedAddRmsNorm` 仍未真正进入 Qwen3 热路径，现阶段保留为 Candle 能力限制下的待验证点，需要远端 CUDA profiling 后再继续推进。
- 由于当前仍是 tensor-KV 过渡路径，chunked prefill 会打断 active batch decode session；因此 CLI 默认值已回到 `prefill_chunk_size=0`，chunked prefill 改为显式开启项。

---

## 6.3 Phase 2：Paged KV cache 改造

### 目标

彻底替换当前 batched decode 的 `extract -> pad/stack -> decode -> extract` 流程。

### 这是全项目最关键的 phase。

### 目标状态

Qwen3 decode 不再面向“每请求一对完整 KV tensor”，而是面向：

- page/block allocator
- block table
- per-seq page list
- per-layer page storage
- backend 直接消费的 metadata

### 任务拆分

#### 任务 A：定义 paged KV 数据结构

建议新增一套只服务 Qwen3 的结构，例如：

- `PagedKvPool`
- `PagedKvLayer`
- `SeqBlockTable`
- `KvPageAllocator`

#### 任务 B：改写 Qwen3 KV write path

当前 `update_kv_cache()` 的语义是：

- 拼接或写入 contiguous tensor

新语义应改为：

- 根据 `position / page_size / block table` 写入 page 存储

#### 任务 C：改写 engine 生命周期

当前 `crane-oai/src/engine/mod.rs` 中 batch decode 依赖：

- `setup_batch_decode`
- `extract_batch_kv`

paged KV 改造后，应将其替换为：

- `prepare_decode_metadata`
- `update_seq_block_table`
- `run_decode_backend`

#### 任务 D：明确 page layout

一开始不要追求万能 layout，建议先锁一个最适合 decode backend 的 layout。

#### 任务 E：锁定 backend-native metadata ABI

第一版 paged KV 就建议把下面这些 metadata 明确定下来：

- `paged_kv_indptr`
- `paged_kv_indices`
- `paged_kv_last_page_len`
- 可选 `block_tables`
- pinned CPU mirror + GPU resident buffer

不要先发明一套只适合 Crane 自己、后面还要再转一次给 backend 的中间格式。

#### 补充建议：尽早验证 `ProcessKV_v2` 风格 KV write kernel

paged KV 一旦基本跑通，就建议尽快做一个小范围 spike：

- page write
- 可选 RoPE
- 可选 KV quantization store

即使正式上线仍放在后续 phase，这个验证也建议前置，因为它会直接影响：

- page layout 是否合适
- KV quantization 的接入方式
- decode 热路径的带宽上限

### 推荐原则

- page size 固定，如 `16` 或 `32`
- 先只做 BF16
- 先只做 decode 场景
- 先不做跨模型复用
- layout 直接对齐首个目标 backend 的 native 约束，例如 `NHD / HND`

### 重点代码

- `crane-core/src/models/qwen3/modeling.rs`
- `crane-core/src/models/qwen3/model.rs`
- `crane-oai/src/engine/backend.rs`
- `crane-oai/src/engine/mod.rs`

### 验收标准

- batch decode 路径不再需要 `extract_batch_kv`
- batch decode setup 时间显著下降
- 并发数提升时性能退化曲线明显变缓
- 长上下文下 VRAM 行为更稳定

### 风险

- 这是最具破坏性的重构
- 很可能需要先在 Qwen3 上独立走一条 backend trait 分支

### 2026-03-26 当前实现状态

- 已新增 `crane-core/src/models/qwen3/paged_kv.rs`，定义 `PagedKvConfig / PagedAttentionMetadata / DecodeBucketKey / KvPageAllocator / SeqBlockTable / PagedKvPool`。
- backend-native metadata ABI 第一版已锁定为：
  - `paged_kv_indptr`
  - `paged_kv_indices`
  - `paged_kv_last_page_len`
  - `block_tables`
- `crane-oai` 已接入 shadow paged-KV runtime：
  - 每个 sequence 维护 `SeqBlockTable`
  - prefill / decode 后都会同步 token->page 映射，统计不再只靠 `seq_len` 粗估
  - prefill admission 会显式预留 decode headroom，按 page budget 决定是否先延后 prefill
- batch decode planning 已优先走 `PagedAttentionMetadata::from_block_tables(...)`，Qwen3 backend 会直接消费这份 metadata 做 `plan_batch_decode_with_metadata(...)`。
- 当前底层 KV 存储仍是过渡态，还没有完全切到真实 paged page-store / page-write kernel。
- 当前实现的目标是先把 allocator、metadata、服务生命周期和 admission control 钉住，为后续远端 CUDA page-write / paged attention backend 留稳定接口。

---

## 6.4 Phase 3：Qwen3 专用 decode attention backend

### 目标

在 paged KV 之上，把 decode attention 从 Candle matmul 迁到 specialized backend。

### 推荐路线

按优先顺序：

1. **先接 FlashInfer decode backend**
2. 如果不足，再补自定义 decode kernel
3. 最后才考虑更激进 persistent / megakernel

### 为什么先做 decode

对在线服务来说，decode 是更容易受以下因素限制的阶段：

- launch overhead
- memory bandwidth
- page metadata 管理
- 多请求 batch scheduling

### 任务拆分

#### 任务 A：定义 Qwen3 decode backend 抽象

建议不要从通用 backend 开始，而是直接定义：

- `Qwen3DecodeBackend`

接口直接围绕：

- `plan`
- `run`
- `workspace`
- `page metadata`
- `batch bucket`

#### 任务 B：将 Qwen3 attention 层从 matmul 路径抽离

当前 Qwen3 attention 逻辑仍和模型实现强绑定。

目标是把 decode attention 变成：

- 上游只负责产生 `Q / new K / new V`
- 下游专门 backend 负责：
  - KV 写入
  - attention 计算
  - output 生成

#### 任务 C：先做 decode-only 路径贯通

不要一开始就要求 prefill 也共用 backend。

#### 任务 D：建立 backend ABI / FFI 层

如果目标是 `FlashInfer / TRTLLM` 类 backend，不能把“接入 backend”理解成调一个 Python API。

需要显式定义：

- Rust 可调用的 `plan / run / reset_workspace / destroy`
- workspace / plan_info / metadata 的所有权
- graph capture 需要的固定输入 buffer
- backend 失败时的 fallback 策略

### 验收标准

- decode 路径性能显著提升
- batch size 上升时吞吐接近线性增长更长一段区间
- attention 核心时间占比明显下降

### 风险

- Qwen3 模型层和 backend 之间的职责边界需要重划
- 一开始可能会导致代码非常不“优雅”，这是可接受的

### 2026-03-26 当前实现状态

- 已新增 `crane-core/src/models/qwen3/decode_backend.rs`，定义 `Qwen3DecodeBackend` 与 `DecodeBackendPlan`。
- `crane-oai` 的 batch decode 已切到 `plan -> run` 风格接口，decode plan cache hit、H2D metadata bytes 均可统计。
- Qwen3 backend 现已支持直接接收 block-table 驱动的 metadata plan，不必再退回“只按 `seq_lens` 重新推导 page 结构”。
- 当前默认 backend 仍是 `TensorDecodeBackend`，它的职责是先把高性能 decode backend 的 ABI、bucket 语义、engine 生命周期跑通。
- FlashInfer / 自定义 CUDA backend / CUDA Graph 仍需在远端服务器继续接线与 profiling；现阶段代码已为后续 backend 替换留出接口。

---

## 6.5 Phase 4：Decode CUDA Graph + bucketization

### 目标

进一步降低 decode CPU 发射开销和 runtime 抖动。

### 前置条件

- paged KV 已建立
- decode backend 形状与 metadata 可稳定复用

### 任务

- **[任务 1]** 设计 batch size bucket
- **[任务 2]** 设计 stable workspace / input buffers
- **[任务 3]** 为每个 bucket capture decode graph
- **[任务 4]** 让常见 decode 请求优先命中 graph 路径
- **[任务 5]** 明确 split-kv / page bucket 的 graph 兼容策略

### 推荐 bucket

- batch bucket：`1 / 2 / 4 / 8 / 16 / 32`
- page/context bucket：按 `max_kv_pages_per_seq` 或 `total_kv_pages` 再分桶

仅按 batch size 分桶通常不够，因为 split-kv / page 数变化也会改变 backend 的 plan 与 CTA 形状。
如果首版 backend 使用 split-kv，graph 路径要么固定 split size，要么显式关闭 split-kv。

### 重点代码

- `crane-oai/src/engine/mod.rs`
- Qwen3 decode backend 层
- CUDA backend 封装层

### 验收标准

- 小 batch decode tok/s 明显提升
- p95 / p99 抖动下降
- 每 token CPU 时间明显下降

### 风险

- graph capture 很怕形状不稳定
- 因此它必须建立在 paged KV + metadata 复用之上

---

## 6.6 Phase 5：Prefix cache / block-hash cache / radix cache

### 目标

解决重复前缀带来的 prefill 浪费。

### 推荐实施顺序

#### 第一层：block hash prefix cache

先做简单可控的：

- 固定 page/block hash
- shared system prompt 命中复用
- 请求附加 key 支持

#### 第二层：radix cache

在 block-hash 版本稳定后，再做更复杂的最长前缀匹配结构。

### 重点收益场景

- system prompt 相同的大量请求
- 多轮 chat
- tool schema 固定
- few-shot 模板复用

### 验收标准

- 命中前缀时 TTFT 明显下降
- 重复模板场景下 req/s 提升明显

### 风险

- cache 一致性、引用计数、淘汰策略复杂
- 建议在 paged KV 完成后再做

---

## 6.7 Phase 6：KV cache quantization

### 目标

将 KV cache 从“只支持 BF16”升级为“可作为并发和带宽优化手段”。

### 推荐顺序

1. `INT8 KV`
2. `FP8 KV`
3. 只在 decode backend 已稳定时向更激进方案推进

### 实施原则

- 不要先做训练友好方案
- 只做 serving 视角下最划算的 dtype
- 尽量把量化/反量化与 KV write / read 结合

### 重点受益

- 长上下文
- 高并发
- 显存吃紧时的可承载能力

### 验收标准

- 同等显存下可承载并发数上升
- decode 吞吐提升或至少不退化
- 质量影响在可接受范围内

---

## 6.8 Phase 7：Qwen3 专用融合 kernel

### 目标

在 serving 架构正确后，开始进一步榨 kernel 级上限。

### 推荐优先顺序

#### 优先级 1：QK norm + RoPE + KV write 融合

这是最值得做的融合点之一，因为它刚好位于 decode 热路径的关键处。

#### 优先级 2：Residual add + RMSNorm fully fused

如果 `FusedAddRmsNorm` 前面还没彻底落地，这里必须完成。

#### 优先级 3：Decode 视角下更强的 MLP / attention 局部融合

#### 优先级 4：Decode 小 batch GEMV / GEMM specialization

当 `paged KV + decode backend + cudagraph` 都稳定后，`QKV / O / gate_up / down` 的小 `M` matmul 往往会变成下一个主要瓶颈。

建议至少预留下面这些方向：

- 针对 `M=1` 或小 batch decode 的 `cublasLt / CUTLASS / custom matvec`
- merged `QKV / gate_up` 权重的 backend-native prepack
- 在 Qwen3 decode 路径上逐步绕开 Candle 默认 linear dispatch

### 重点代码

- `crane-core/kernels/fused_ops.cu`
- 新增 `qwen3_decode_kernels.cu` 一类专用文件
- `crane-core/src/fused_ops/cuda_impl.rs`

### 验收标准

- 单 token decode 的 kernel launch 数继续下降
- decode pipeline 的 GPU idle 时间继续下降

### 风险

- 如果 serving 架构没先做好，这一阶段收益会被系统开销吞掉

---

## 6.9 Phase 8：最终冲顶路线（可选）

### 目标

在前面所有结构性优化完成后，再考虑是否值得做接近 `qwen_megakernel` 的极端 specialization。

### 可选路线

- 更强的 persistent decode kernel
- 更深的 layer fusion
- 针对固定 GPU 架构的超激进 specialization
- 仅支持 Qwen3 1.7B 某个固定配置的极致版本

### 只有在以下条件满足时才建议做

- paged KV 已稳定
- decode backend 已稳定
- decode cudagraph 已稳定
- prefix cache 已稳定
- benchmark 显示瓶颈已集中在少量 GPU kernel 上

### 否则不建议提前进入这一步。

---

## 7. 模块级工作流拆分

## 7.1 Workstream A：Engine / Serving Runtime

### 负责内容

- scheduler 行为
- batch 生命周期
- metadata bucket
- prefix cache 生命周期
- decode graph 触发逻辑

### 主要代码

- `crane-oai/src/engine/mod.rs`
- `crane-oai/src/engine/backend.rs`
- `crane-oai/src/engine/scheduler.rs`

### 关键任务

- 从“张量拼装型 batch decode”迁移到“metadata 驱动型 batch decode”
- 增加 chunked prefill + page-budget admission control，避免长 prefill 持续阻塞 decode
- 增加 bucket 与 graph 机制
- prefix cache 命中路径

---

## 7.2 Workstream B：Qwen3 Model Runtime

### 负责内容

- Qwen3 attention / MLP forward 重划分
- prefill / decode 分流
- KV write path 改写

### 主要代码

- `crane-core/src/models/qwen3/modeling.rs`
- `crane-core/src/models/qwen3/model.rs`

### 关键任务

- 把 decode attention 逻辑抽离成 backend
- 把 KV cache 从 tensor 语义升级为 paged 语义

---

## 7.3 Workstream C：CUDA / Kernel

### 负责内容

- fused kernel
- decode specialized kernel
- KV read/write / quant kernel
- backend 需要的 low-level 封装

### 主要代码

- `crane-core/kernels/fused_ops.cu`
- `crane-core/src/fused_ops/cuda_impl.rs`

### 关键任务

- `FusedAddRmsNorm` 落地
- `QK norm + RoPE + KV write` 融合
- 适配 decode backend 的专用 kernel

---

## 7.4 Workstream D：Sampling / Output Path

### 负责内容

- greedy
- top-k
- top-p
- repetition penalty
- GPU sampling fallback 清理

### 主要代码

- `crane-oai/src/engine/sampling.rs`

### 关键任务

- 尽量减少 CPU fallback
- 降低 host/device 往返
- 将 sampling 开销从总时延中压低

---

## 8. 执行顺序建议

如果只有一支小团队，建议严格按下面顺序推进：

1. **Phase 0：benchmark + 基线**
2. **Phase 1：清理现有明显浪费**
3. **Phase 2：paged KV**
4. **Phase 3：decode backend**
5. **Phase 4：decode cudagraph**
6. **Phase 5：prefix cache**
7. **Phase 6：KV quantization**
8. **Phase 7：更深 kernel 融合**
9. **Phase 8：极限 specialization**

### 为什么不能先做更深 kernel

因为当前 Qwen3 最大问题不是“每层少一个 pointwise kernel”，而是：

- KV 组织方式不对
- backend 组织方式不对
- serving 路径固定成本高

先把架构做对，后面的 kernel 才有意义。

---

## 9. 里程碑定义

## Milestone M1：Qwen3 decode 不再做 KV extract / restack

### 判定标准

- `extract_batch_kv()` 不再是 decode 主路径必要步骤
- batch decode setup 时间大幅下降

## Milestone M2：Qwen3 decode 已切到 specialized backend

### 判定标准

- Qwen3 decode 不再以 Candle matmul 为主执行路径
- batch decode 吞吐显著提升

## Milestone M3：常见 batch bucket 命中 CUDA Graph

### 判定标准

- 常见 batch 桶位的 decode 已 graph capture
- p95 / p99 抖动下降

## Milestone M4：重复前缀请求可复用 prefill

### 判定标准

- system prompt / shared prefix 命中可直接复用已有 KV

## Milestone M5：Qwen3 decode kernel 进一步 specialization

### 判定标准

- QK norm + rope + kv write 已融合
- 或 decode kernel launch 进一步减少

---

## 10. 验收指标建议

每完成一个 milestone，都建议固定报告下面几组指标：

- **[单请求]** TTFT / decode tok/s
- **[8 并发]** req/s / decode tok/s / p95 latency
- **[32 并发]** req/s / decode tok/s / peak VRAM
- **[共享前缀]** TTFT 改善比例
- **[8K / 32K]** 吞吐退化曲线

建议在文档中长期维护一个表格，记录：

- baseline
- M1
- M2
- M3
- M4
- M5

否则很容易在优化过程中失去方向。

---

## 11. 风险与控制策略

## 风险 1：重构过深导致功能回退

### 控制策略

- Qwen3 高性能路径单独开关
- 新旧路径并行一段时间
- benchmark 与 correctness case 同时跟

## 风险 2：paged KV 重构复杂度过高

### 控制策略

- 先只支持 decode
- 先不做多模型通用化
- 先只支持一种 layout 与 page size

## 风险 3：backend 接入后接口边界混乱

### 控制策略

- 明确 `Qwen3Model` 只负责产生 Q / K / V / residual 等上游张量
- 把 decode attention 与 KV 组织放入独立 backend 层

## 风险 4：过早写 megakernel 导致工程失控

### 控制策略

- 把 megakernel 明确放到 Phase 8
- 只有在前面全部稳定后才允许进入

---

## 12. 不该做的事情清单

为了保证 roadmap 不跑偏，建议在项目里明确以下“停止事项”：

- **不要**在 paged KV 之前继续扩大 batch decode 的 tensor 搬运路径
- **不要**在 specialized decode backend 之前继续围绕 Candle matmul 做过多局部 patch
- **不要**在 benchmark 体系没稳定前宣称性能提升
- **不要**为了通用性把 Qwen3 高性能路径抽象得过早
- **不要**在 prefix cache 之前把重点放到更多小型 fused op
- **不要**过早进入 megakernel

---

## 13. 推荐的 4 周推进顺序

如果需要一个短期冲刺版计划，建议如下：

## Week 1

- 建立 benchmark 与细粒度 profiling
- 拆出 Qwen3 prefill / decode 路径
- 清理 sampling CPU fallback
- 明确 `FusedAddRmsNorm` 是否落地

## Week 2

- 设计并实现 paged KV 数据结构
- 在 Qwen3 decode 路径接入 paged KV write
- 让 engine 不再依赖 extract/restack 主循环

## Week 3

- 接入 decode attention backend
- 完成 Qwen3 decode 主路径切换
- 建立 workspace / metadata 复用

## Week 4

- 做 batch bucket + decode CUDA Graph
- 跑完整 benchmark 回归
- 根据瓶颈决定是否进入 prefix cache 或 kernel specialization

这 4 周做完后，系统应该已经从“优化过的通用模型实现”变成“面向 Qwen3 serving 的高性能系统”。

---

## 14. 最终建议

如果只保留一句工程建议：

> **先把 Qwen3 的 serving 基础设施做成 `paged KV + specialized decode backend + decode cudagraph`，再去做更深的 kernel 融合；不要反过来。**

如果只保留最重要的三项实施目标：

1. **Paged KV**
2. **Qwen3 decode backend**
3. **Decode CUDA Graph**

这三项决定 Qwen3 1.7B 的性能上限。

---

## 15. 文档总结

这份 roadmap 的核心立场是：

- Crane 当前并不是“完全没优化”
- 但要让 `Qwen3 1.7B` 真正达到接近一线 serving 系统的性能，必须把重点从“模型内部小融合”切到“KV / attention backend / runtime serving 架构”
- 若暂时不考虑其他模型兼容，完全应该允许 Qwen3 走一条高性能专用路径

也就是说，下一阶段最该做的不是“再多写几个 fused op”，而是：

- **把 KV cache 变成 paged 系统资源**
- **把 decode attention 变成 specialized backend**
- **把 decode 执行变成 graph-friendly runtime**

这才是 Qwen3 极致性能路线的主线。
