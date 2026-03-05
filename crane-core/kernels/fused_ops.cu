/**
 * Fused CUDA kernels for Crane transformer inference.
 *
 * Targets: sm_80+ (Ampere & newer, bf16 support)
 *
 * Kernels:
 *   1. fused_rmsnorm_residual_bf16  — RMSNorm + residual save
 *   2. fused_silu_mul_bf16          — SiLU(gate) * up  (one pass)
 *   3. fused_add_rmsnorm_bf16       — residual_add + RMSNorm (one pass)
 *   4. gpu_argmax_bf16              — GPU-side argmax over vocab (greedy decode)
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// =====================================================================
// Helpers
// =====================================================================

static constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Fast SiLU: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =====================================================================
// 1. Fused RMSNorm — one block per row
//
//    dst[row, col] = (x[row, col] / rms) * weight[col]
//    where rms = sqrt(mean(x²) + eps)
//
//    Identical to candle's rmsnorm but with explicit bf16 I/O and
//    handles up to 16384 columns per warp-tree reduction.
// =====================================================================

extern "C" __global__ void fused_rmsnorm_bf16(
    const __nv_bfloat16 *__restrict__ x,      // [rows, cols]
    __nv_bfloat16       *__restrict__ dst,     // [rows, cols]
    const __nv_bfloat16 *__restrict__ weight,  // [cols]
    const int ncols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Phase 1: compute sum of squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(x[row * ncols + col]);
        sum_sq += v * v;
    }

    // Warp reduce
    sum_sq = warp_reduce_sum_f32(sum_sq);

    // Cross-warp reduce via shared memory
    __shared__ float s_partial[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum_f32(sum_sq);
        if (lane_id == 0) s_partial[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(s_partial[0] / (float)ncols + eps);

    // Phase 2: normalize and write output
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(x[row * ncols + col]);
        float w = __bfloat162float(weight[col]);
        dst[row * ncols + col] = __float2bfloat16(v * scale * w);
    }
}

// =====================================================================
// 2. Fused SiLU(gate) * up — one pass over 2 * intermediate_size
//
//    Input:  gate_up [rows, 2*intermediate_size]  (gate||up concatenated)
//    Output: dst     [rows, intermediate_size]
//    dst[i] = silu(gate_up[i]) * gate_up[i + intermediate_size]
//
//    Saves 2 kernel launches (separate silu + mul) and 1 intermediate
//    tensor allocation.
// =====================================================================

extern "C" __global__ void fused_silu_mul_bf16(
    const __nv_bfloat16 *__restrict__ gate_up,  // [rows, 2*intermediate_size]
    __nv_bfloat16       *__restrict__ dst,       // [rows, intermediate_size]
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __nv_bfloat16 *gate_row = gate_up + row * 2 * intermediate_size;
    const __nv_bfloat16 *up_row   = gate_row + intermediate_size;
    __nv_bfloat16       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = __bfloat162float(gate_row[i]);
        float u = __bfloat162float(up_row[i]);
        dst_row[i] = __float2bfloat16(fast_silu(g) * u);
    }
}

// f16 variant
extern "C" __global__ void fused_silu_mul_f16(
    const __half *__restrict__ gate_up,
    __half       *__restrict__ dst,
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __half *gate_row = gate_up + row * 2 * intermediate_size;
    const __half *up_row   = gate_row + intermediate_size;
    __half       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = __half2float(gate_row[i]);
        float u = __half2float(up_row[i]);
        dst_row[i] = __float2half(fast_silu(g) * u);
    }
}

// f32 variant
extern "C" __global__ void fused_silu_mul_f32(
    const float *__restrict__ gate_up,
    float       *__restrict__ dst,
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float *gate_row = gate_up + row * 2 * intermediate_size;
    const float *up_row   = gate_row + intermediate_size;
    float       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = gate_row[i];
        float u = up_row[i];
        dst_row[i] = fast_silu(g) * u;
    }
}

// =====================================================================
// 3. Fused residual_add + RMSNorm — one read of hidden, write norm + residual
//
//    residual[row] += hidden[row]            (in-place update)
//    dst[row] = rmsnorm(residual[row]) * weight
//
//    Eliminates the separate add kernel + RMSNorm kernel + extra read.
// =====================================================================

extern "C" __global__ void fused_add_rmsnorm_bf16(
    __nv_bfloat16       *__restrict__ residual,  // [rows, cols] — updated in-place
    const __nv_bfloat16 *__restrict__ hidden,    // [rows, cols] — value to add
    __nv_bfloat16       *__restrict__ dst,        // [rows, cols] — normalized output
    const __nv_bfloat16 *__restrict__ weight,     // [cols]
    const int ncols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int row_offset = row * ncols;

    // Phase 1: add residual, compute sum of squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float r = __bfloat162float(residual[row_offset + col]);
        float h = __bfloat162float(hidden[row_offset + col]);
        float v = r + h;
        // Write residual back (in-place update)
        residual[row_offset + col] = __float2bfloat16(v);
        sum_sq += v * v;
    }

    // Warp + cross-warp reduce
    sum_sq = warp_reduce_sum_f32(sum_sq);
    __shared__ float s_partial[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum_f32(sum_sq);
        if (lane_id == 0) s_partial[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(s_partial[0] / (float)ncols + eps);

    // Phase 2: normalize from updated residual
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(residual[row_offset + col]);
        float w = __bfloat162float(weight[col]);
        dst[row_offset + col] = __float2bfloat16(v * scale * w);
    }
}

// =====================================================================
// 4. GPU Argmax — two-phase reduction for vocab-size vectors
//
//    Phase 1: Each block reduces a chunk of rows → per-block max + argmax
//    Phase 2: Single block reduces per-block results → final argmax
//
//    For greedy decode: replaces 303KB DtoH + CPU argmax with
//    a single scalar (4 bytes) DtoH.
// =====================================================================

extern "C" __global__ void gpu_argmax_bf16_phase1(
    const __nv_bfloat16 *__restrict__ logits,  // [vocab_size]
    float               *__restrict__ block_max_vals,
    int32_t             *__restrict__ block_max_idxs,
    const int vocab_size
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int bid = blockIdx.x;
    const int num_blocks = gridDim.x;

    // Each block handles a strided chunk
    int chunk = (vocab_size + num_blocks - 1) / num_blocks;
    int start = bid * chunk;
    int end   = min(start + chunk, vocab_size);

    float local_max = -INFINITY;
    int   local_idx = -1;

    for (int i = start + tid; i < end; i += block_size) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int   other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    // Cross-warp reduce
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    __shared__ float s_max_vals[32];
    __shared__ int   s_max_idxs[32];

    if (lane_id == 0) {
        s_max_vals[warp_id] = local_max;
        s_max_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < num_warps) {
        local_max = s_max_vals[lane_id];
        local_idx = s_max_idxs[lane_id];

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
            int   other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            block_max_vals[bid] = local_max;
            block_max_idxs[bid] = local_idx;
        }
    }
}

extern "C" __global__ void gpu_argmax_phase2(
    const float   *__restrict__ block_max_vals,
    const int32_t *__restrict__ block_max_idxs,
    int32_t       *__restrict__ output_token,
    const int num_blocks
) {
    const int tid = threadIdx.x;

    float best_val = -INFINITY;
    int   best_idx = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > best_val) {
            best_val = v;
            best_idx = block_max_idxs[i];
        }
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int   other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    // Cross-warp reduce
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    __shared__ float s_vals[32];
    __shared__ int   s_idxs[32];

    if (lane_id == 0) {
        s_vals[warp_id] = best_val;
        s_idxs[warp_id] = best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        best_val = (lane_id < num_warps) ? s_vals[lane_id] : -INFINITY;
        best_idx = (lane_id < num_warps) ? s_idxs[lane_id] : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
            int   other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            *output_token = best_idx;
        }
    }
}

// =====================================================================
// 5. Fused residual_add in-place: residual += hidden
//    Simple element-wise kernel to avoid candle's tensor add overhead.
// =====================================================================

extern "C" __global__ void fused_residual_add_bf16(
    __nv_bfloat16       *__restrict__ residual,  // [n] — updated in-place
    const __nv_bfloat16 *__restrict__ hidden,    // [n]
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = __bfloat162float(residual[idx]);
        float h = __bfloat162float(hidden[idx]);
        residual[idx] = __float2bfloat16(r + h);
    }
}

// =====================================================================
// 6. GPU TopK — two-stage block reduction (k ≤ 64)
//
//    Stage 1: Each block processes `items_per_block` elements from
//             the input, producing per-block top-k.
//    Stage 2: Single block merges all per-block results → final top-k
//             indices output.
// =====================================================================

static inline __device__ void topk_insert(
    float v,
    uint32_t i,
    float * vals,
    uint32_t * idx,
    const int k
) {
    if (v <= vals[k - 1]) return;
    int p = k - 1;
    while (p > 0 && v > vals[p - 1]) {
        vals[p] = vals[p - 1];
        idx[p] = idx[p - 1];
        --p;
    }
    vals[p] = v;
    idx[p] = i;
}

extern "C" __global__ void topk_stage1_f32(
    const float * x,
    const uint32_t n,
    const uint32_t k,
    const uint32_t items_per_block,
    float * out_vals,
    uint32_t * out_idx
) {
    const uint32_t start = blockIdx.x * items_per_block;
    const uint32_t end = min(n, start + items_per_block);
    if (start >= end) return;

    float vals[64];
    uint32_t idx[64];
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    for (uint32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = x[i];
        topk_insert(v, i, vals, idx, (int)k);
    }

    extern __shared__ uint8_t smem[];
    float * block_vals = (float *)smem;
    uint32_t * block_idx = (uint32_t *)(block_vals + (uint32_t)blockDim.x * k);

    const uint32_t base = (uint32_t)threadIdx.x * k;
    for (uint32_t j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float bvals[64];
        uint32_t bidx[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            bvals[j] = -INFINITY;
            bidx[j] = 0;
        }
        for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
            const uint32_t tb = t * k;
            for (uint32_t j = 0; j < k; ++j) {
                topk_insert(block_vals[tb + j], block_idx[tb + j], bvals, bidx, (int)k);
            }
        }
        const uint32_t out_base = blockIdx.x * k;
        for (uint32_t j = 0; j < k; ++j) {
            out_vals[out_base + j] = bvals[j];
            out_idx[out_base + j] = bidx[j];
        }
    }
}

extern "C" __global__ void topk_stage2_f32(
    const float * in_vals,
    const uint32_t * in_idx,
    const uint32_t m,
    const uint32_t k,
    uint32_t * out_idx
) {
    float vals[64];
    uint32_t idx[64];
    #pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    for (uint32_t i = threadIdx.x; i < m; i += blockDim.x) {
        float v = in_vals[i];
        uint32_t id = in_idx[i];
        topk_insert(v, id, vals, idx, (int)k);
    }

    extern __shared__ uint8_t smem2[];
    float * block_vals = (float *)smem2;
    uint32_t * block_idx = (uint32_t *)(block_vals + (uint32_t)blockDim.x * k);

    const uint32_t base = (uint32_t)threadIdx.x * k;
    for (uint32_t j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float bvals[64];
        uint32_t bidx[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            bvals[j] = -INFINITY;
            bidx[j] = 0;
        }
        for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
            const uint32_t tb = t * k;
            for (uint32_t j = 0; j < k; ++j) {
                topk_insert(block_vals[tb + j], block_idx[tb + j], bvals, bidx, (int)k);
            }
        }
        for (uint32_t j = 0; j < k; ++j) {
            out_idx[j] = bidx[j];
        }
    }
}

// =====================================================================
// 7. Fused TopK + Gumbel-Max Sampling
//
//    Single kernel: takes f32 logits, temperature, random seed →
//    returns ONE u32 token id.
//
//    Algorithm:
//      Phase 1 (per-thread): each thread scans a chunk of vocab,
//              maintaining a sorted top-k list via insertion sort.
//      Phase 2 (shared mem merge): thread 0 merges all per-thread
//              top-k lists into a single global top-k.
//      Phase 3 (Gumbel-max): thread 0 applies temperature scaling +
//              Gumbel noise to the k winners, picks argmax.
//
//    Only 4 bytes DtoH (the token id).
//
//    NOTE: k ≤ 64, uses same insertion sort as topk_stage1.
//    Uses a simple xoshiro128+ PRNG seeded per-call.
// =====================================================================

// xoshiro128+ state for Gumbel noise generation
struct Xoshiro128Plus {
    uint32_t s[4];
};

__device__ __forceinline__ uint32_t xoshiro128plus_rotl(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

__device__ __forceinline__ uint32_t xoshiro128plus_next(Xoshiro128Plus &state) {
    const uint32_t result = state.s[0] + state.s[3];
    const uint32_t t = state.s[1] << 9;
    state.s[2] ^= state.s[0];
    state.s[3] ^= state.s[1];
    state.s[1] ^= state.s[2];
    state.s[0] ^= state.s[3];
    state.s[2] ^= t;
    state.s[3] = xoshiro128plus_rotl(state.s[3], 11);
    return result;
}

// Convert u32 to uniform float in (0, 1) — open interval to avoid log(0)
__device__ __forceinline__ float xoshiro128plus_uniform(Xoshiro128Plus &state) {
    uint32_t v = xoshiro128plus_next(state);
    // Map to (2^-33, 1 - 2^-33) approximately — avoids exact 0
    return (float)(v >> 8) * (1.0f / 16777216.0f) + (1.0f / 33554432.0f);
}

extern "C" __global__ void topk_gumbel_sample_f32(
    const float * __restrict__ x,    // [vocab_size] f32 logits
    const uint32_t vocab_size,
    const uint32_t k,                // top-k (≤ 64)
    const float temperature,         // > 0
    const uint64_t seed,             // random seed
    uint32_t * __restrict__ out_token // [1] output token id
) {
    // ── Phase 1: per-thread top-k via insertion sort ──
    float vals[64];
    uint32_t idx[64];
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    for (uint32_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        topk_insert(x[i], i, vals, idx, (int)k);
    }

    // ── Phase 2: merge all threads' top-k in shared memory ──
    extern __shared__ uint8_t smem_tgs[];
    float * block_vals = (float *)smem_tgs;
    uint32_t * block_idx = (uint32_t *)(block_vals + (uint32_t)blockDim.x * k);

    const uint32_t base = (uint32_t)threadIdx.x * k;
    for (uint32_t j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        // Merge into final top-k
        float fvals[64];
        uint32_t fidx[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            fvals[j] = -INFINITY;
            fidx[j] = 0;
        }
        for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
            const uint32_t tb = t * k;
            for (uint32_t j = 0; j < k; ++j) {
                topk_insert(block_vals[tb + j], block_idx[tb + j], fvals, fidx, (int)k);
            }
        }

        // ── Phase 3: Gumbel-max sampling on the k winners ──
        // Initialize PRNG from seed
        Xoshiro128Plus rng;
        rng.s[0] = (uint32_t)(seed & 0xFFFFFFFF);
        rng.s[1] = (uint32_t)((seed >> 32) & 0xFFFFFFFF);
        rng.s[2] = rng.s[0] ^ 0x9E3779B9u;
        rng.s[3] = rng.s[1] ^ 0x6A09E667u;
        // Warm up
        for (int w = 0; w < 8; ++w) xoshiro128plus_next(rng);

        float best_score = -INFINITY;
        uint32_t best_token = 0;

        for (uint32_t j = 0; j < k; ++j) {
            if (fvals[j] <= -INFINITY) continue;
            float logit = fvals[j] / temperature;
            // Gumbel noise: -log(-log(u)), u ~ Uniform(0,1)
            float u = xoshiro128plus_uniform(rng);
            float gumbel = -logf(-logf(u));
            float score = logit + gumbel;
            if (score > best_score) {
                best_score = score;
                best_token = fidx[j];
            }
        }

        *out_token = best_token;
    }
}

// =====================================================================
// 8. Fused TopK + Top-P + Gumbel-Max Sampling
//
//    Like topk_gumbel_sample_f32 but with nucleus (top-p) filtering.
//    After finding top-k, sorts by value (already sorted from insertion),
//    computes softmax + cumsum, masks out tokens exceeding top_p threshold,
//    then applies Gumbel-max on the remaining tokens.
// =====================================================================

extern "C" __global__ void topk_topp_gumbel_sample_f32(
    const float * __restrict__ x,    // [vocab_size] f32 logits
    const uint32_t vocab_size,
    const uint32_t k,                // top-k (≤ 64)
    const float temperature,         // > 0
    const float top_p,               // (0, 1)
    const uint64_t seed,             // random seed
    uint32_t * __restrict__ out_token // [1] output token id
) {
    // ── Phase 1: per-thread top-k via insertion sort ──
    float vals[64];
    uint32_t idx[64];
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    for (uint32_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        topk_insert(x[i], i, vals, idx, (int)k);
    }

    // ── Phase 2: merge all threads' top-k in shared memory ──
    extern __shared__ uint8_t smem_tpgs[];
    float * block_vals = (float *)smem_tpgs;
    uint32_t * block_idx = (uint32_t *)(block_vals + (uint32_t)blockDim.x * k);

    const uint32_t base = (uint32_t)threadIdx.x * k;
    for (uint32_t j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        // Merge into final top-k (already sorted descending by value)
        float fvals[64];
        uint32_t fidx[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            fvals[j] = -INFINITY;
            fidx[j] = 0;
        }
        for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
            const uint32_t tb = t * k;
            for (uint32_t j = 0; j < k; ++j) {
                topk_insert(block_vals[tb + j], block_idx[tb + j], fvals, fidx, (int)k);
            }
        }

        // ── Phase 3: softmax + cumsum + top-p mask ──
        // Find max for numerical stability
        float maxv = fvals[0]; // already sorted, fvals[0] is the largest

        // Compute exp(logit/temp - max/temp) for softmax
        float exps[64];
        float sum_exp = 0.0f;
        uint32_t valid_k = 0;
        for (uint32_t j = 0; j < k; ++j) {
            if (fvals[j] <= -INFINITY) break;
            valid_k = j + 1;
            exps[j] = expf((fvals[j] - maxv) / temperature);
            sum_exp += exps[j];
        }

        // Normalize to probabilities + compute cumsum
        float cumsum = 0.0f;
        uint32_t cutoff = valid_k; // how many tokens pass the top-p filter
        for (uint32_t j = 0; j < valid_k; ++j) {
            float prob = exps[j] / sum_exp;
            cumsum += prob;
            if (cumsum > top_p && j > 0) {
                // This token pushed us over top_p — exclude it and all after
                cutoff = j;
                break;
            }
        }
        if (cutoff == 0) cutoff = 1; // always keep at least 1 token

        // ── Phase 4: Gumbel-max on the surviving tokens ──
        Xoshiro128Plus rng;
        rng.s[0] = (uint32_t)(seed & 0xFFFFFFFF);
        rng.s[1] = (uint32_t)((seed >> 32) & 0xFFFFFFFF);
        rng.s[2] = rng.s[0] ^ 0x9E3779B9u;
        rng.s[3] = rng.s[1] ^ 0x6A09E667u;
        for (int w = 0; w < 8; ++w) xoshiro128plus_next(rng);

        float best_score = -INFINITY;
        uint32_t best_token = fidx[0];

        for (uint32_t j = 0; j < cutoff; ++j) {
            float logit = fvals[j] / temperature;
            float u = xoshiro128plus_uniform(rng);
            float gumbel = -logf(-logf(u));
            float score = logit + gumbel;
            if (score > best_score) {
                best_score = score;
                best_token = fidx[j];
            }
        }

        *out_token = best_token;
    }
}

// =====================================================================
// 9. Qwen3.5 linear-attention recurrent scan (f32)
//
//    Per (batch, head) block, scans all timesteps and updates recurrent
//    state in-place while emitting outputs for each timestep.
//
//    Shapes:
//      state_in/out : [BH, K, V]
//      q            : [BH, S, K]
//      k            : [BH, S, K]
//      v            : [BH, S, V]
//      beta         : [BH, S]
//      g            : [BH, S]
//      out          : [BH, S, V]
// =====================================================================

extern "C" __global__ void qwen35_linear_scan_f32(
    const float *__restrict__ state_in,
    float *__restrict__ state_out,
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ v,
    const float *__restrict__ beta,
    const float *__restrict__ g,
    float *__restrict__ out,
    const int seq_len,
    const int k_dim,
    const int v_dim
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float shmem[];
    float *sh_k = shmem;
    float *sh_q = shmem + k_dim;
    __shared__ float sh_beta;
    __shared__ float sh_exp_g;

    const int state_base = bh * k_dim * v_dim;
    const int qk_base = bh * seq_len * k_dim;
    const int v_base = bh * seq_len * v_dim;
    const int bg_base = bh * seq_len;

    // Initialize working state from input state.
    if (tid < v_dim) {
        for (int kk = 0; kk < k_dim; ++kk) {
            const int idx = state_base + kk * v_dim + tid;
            state_out[idx] = state_in[idx];
        }
    }
    __syncthreads();

    for (int t = 0; t < seq_len; ++t) {
        if (tid < k_dim) {
            sh_k[tid] = k[qk_base + t * k_dim + tid];
            sh_q[tid] = q[qk_base + t * k_dim + tid];
        }
        if (tid == 0) {
            sh_beta = beta[bg_base + t];
            sh_exp_g = expf(g[bg_base + t]);
        }
        __syncthreads();

        if (tid < v_dim) {
            float kv_mem = 0.0f;
            for (int kk = 0; kk < k_dim; ++kk) {
                const int sidx = state_base + kk * v_dim + tid;
                kv_mem += (state_out[sidx] * sh_exp_g) * sh_k[kk];
            }

            const float delta = (v[v_base + t * v_dim + tid] - kv_mem) * sh_beta;

            float out_v = 0.0f;
            for (int kk = 0; kk < k_dim; ++kk) {
                const int sidx = state_base + kk * v_dim + tid;
                const float new_state = state_out[sidx] * sh_exp_g + sh_k[kk] * delta;
                state_out[sidx] = new_state;
                out_v += new_state * sh_q[kk];
            }

            out[v_base + t * v_dim + tid] = out_v;
        }
        __syncthreads();
    }
}

