//! Fused CUDA kernels for Crane transformer inference.
//!
//! When the `cuda` feature is enabled, this module provides:
//! - `fused_silu_mul` — Fused SiLU(gate) * up in one pass
//! - `fused_add_rmsnorm` — Fused residual_add + RMSNorm
//! - `gpu_argmax` — GPU-side argmax for greedy sampling
//! - `topk_indices` — GPU top-k on 1D f32 tensors
//! - `copy_from_slice_u32` — HtoD copy into CUDA U32 tensor
//! - `copy_from_tensor_f32` — DtoD copy between CUDA f32 tensors
//!
//! Each operation eliminates multiple kernel launches and intermediate
//! GMEM round-trips compared to the equivalent candle op chain.

#[cfg(feature = "cuda")]
mod cuda_impl;

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

// ── Non-CUDA fallbacks ──────────────────────────────────────────────

#[cfg(not(feature = "cuda"))]
mod fallback {
    use candle_core::{Result, Tensor};

    pub fn gpu_argmax(logits: &Tensor) -> Result<u32> {
        let logits = logits.flatten_all()?;
        logits.argmax(0)?.to_scalar::<u32>()
    }

    pub fn topk_indices(logits: &Tensor, k: usize) -> Result<Tensor> {
        if logits.rank() != 1 {
            candle_core::bail!("topk_indices expects a 1D tensor");
        }
        let n = logits.dims1()?;
        if k == 0 || k > n {
            candle_core::bail!("topk_indices: invalid k");
        }
        let vals = logits.to_vec1::<f32>()?;
        let mut pairs: Vec<(f32, u32)> = vals
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, i as u32))
            .collect();
        let kth = k.saturating_sub(1);
        pairs.select_nth_unstable_by(kth, |a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Greater)
        });
        pairs.truncate(k);
        pairs.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Greater)
        });
        let out: Vec<u32> = pairs.into_iter().map(|(_, i)| i).collect();
        Tensor::new(out.as_slice(), logits.device())
    }

    pub fn copy_from_slice_u32(_tensor: &Tensor, _src: &[u32]) -> Result<()> {
        candle_core::bail!("copy_from_slice_u32 requires CUDA feature")
    }

    pub fn copy_from_tensor_f32(_dst: &Tensor, _src: &Tensor) -> Result<()> {
        candle_core::bail!("copy_from_tensor_f32 requires CUDA feature")
    }
}

#[cfg(not(feature = "cuda"))]
pub use fallback::*;
