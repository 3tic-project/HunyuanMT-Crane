//! CUDA Graph capture and replay for the decode path.
//!
//! This module eliminates per-step kernel launch overhead (~3.5ms for 490
//! kernels) by capturing the entire decode forward pass as a single CUDA graph
//! and replaying it with ~5μs of CPU overhead.
//!
//! # Design
//!
//! Pre-allocated "capture var" tensors are created once at warmup time.
//! During capture the model forward pass records all GPU work referencing
//! these tensors. At replay time we write new values into the *same*
//! device-memory buffers (via `zero_set` + `slice_set`) and call
//! `cuGraphLaunch` — the graph re-reads the updated data at the baked-in
//! addresses.
//!
//! # Feature gate
//!
//! This entire module is compiled only when `feature = "cuda"`.

use candle_core::cuda_backend::cudarc::driver::sys::{
    self, CUgraphInstantiate_flags, CUmemPool_attribute, CUmemoryPool, CUstreamCaptureMode,
};
// cudarc 0.19.2 safe graph wrapper (re-exported via candle_core)
use candle_core::cuda_backend::cudarc::driver::CudaGraph;
use candle_core::cuda_backend::CudaDevice;
use candle_core::{DType, Device, Result, Tensor};
use std::collections::BTreeMap;
use std::ptr;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────
// Stream helpers
// ─────────────────────────────────────────────────────────────────────────

fn sync_stream(device: &CudaDevice) -> Result<()> {
    device
        .cuda_stream()
        .synchronize()
        .map_err(|e| candle_core::Error::Msg(format!("cuStreamSynchronize failed: {e:?}")))
}

/// Prevent CUDA from releasing pooled memory — required so that graph
/// replays see the same device addresses that were baked in at capture
/// time.
fn setup_mem_pool(device: &CudaDevice) -> Result<()> {
    let cu_device = device.cuda_stream().context().cu_device();
    unsafe {
        let mut pool: CUmemoryPool = ptr::null_mut();
        sys::cuDeviceGetDefaultMemPool(&mut pool, cu_device)
            .result()
            .map_err(|e| {
                candle_core::Error::Msg(format!("cuDeviceGetDefaultMemPool failed: {e:?}"))
            })?;

        let threshold: u64 = u64::MAX;
        sys::cuMemPoolSetAttribute(
            pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const _ as *mut _,
        )
        .result()
        .map_err(|e| {
            candle_core::Error::Msg(format!("cuMemPoolSetAttribute failed: {e:?}"))
        })?;

        sys::cuDeviceSetMemPool(cu_device, pool)
            .result()
            .map_err(|e| {
                candle_core::Error::Msg(format!("cuDeviceSetMemPool failed: {e:?}"))
            })?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Pre-allocated capture variables
// ─────────────────────────────────────────────────────────────────────────

/// Tensors whose device memory is baked into the captured graph.
/// Updated in-place (zero_set + slice_set) before each replay.
pub struct CaptureVars {
    /// `[max_bs, 1]` u32
    pub input_ids: Tensor,
    /// `[max_bs, 1, head_dim/2]` model dtype
    pub cos: Tensor,
    /// `[max_bs, 1, head_dim/2]` model dtype
    pub sin: Tensor,
    /// `[max_bs, 1, 1, max_kv_len]` model dtype
    pub attention_mask: Tensor,
    /// `[max_bs]` u32
    pub write_pos: Tensor,
    /// Per-captured-batch-size output tensor reference.
    pub outputs: BTreeMap<usize, Tensor>,
}

// ─────────────────────────────────────────────────────────────────────────
// Graph capturer
// ─────────────────────────────────────────────────────────────────────────

/// Orchestrates graph capture at multiple batch sizes and provides a
/// replay API that copies actual inputs into the pre-allocated capture
/// vars, launches the graph, and returns the (narrowed) output.
pub struct DecodeGraphCapturer {
    /// Pre-allocated capture variables. `None` before `capture()`.
    pub vars: Option<CaptureVars>,
    /// `batch_size → CudaGraph` (cudarc 0.19 safe wrapper)
    captured: BTreeMap<usize, CudaGraph>,
    /// Sorted list of captured batch sizes.
    captured_bs: Vec<usize>,
    /// The CudaDevice used for stream / pool management.
    device: Arc<CudaDevice>,
    /// Max KV length in the pre-allocated buffer (determines mask width).
    pub max_kv_len: usize,
}

// cudarc::driver::CudaGraph holds Arc<CudaStream> which is Send+Sync,
// plus raw CUgraph/CUgraphExec handles that are safe to move between
// threads (they are not thread-local). The engine runs single-threaded
// but we need Send to move DecodeGraphCapturer into the engine thread.
unsafe impl Send for DecodeGraphCapturer {}

impl DecodeGraphCapturer {
    /// Create a new capturer. Call `capture()` to actually capture graphs.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            vars: None,
            captured: BTreeMap::new(),
            captured_bs: Vec::new(),
            device,
            max_kv_len: 0,
        }
    }

    /// Compute planned batch sizes to capture (1..15 individually, then
    /// 16, 32, … up to `max_bs` in steps of 16).
    pub fn planned_batches(max_bs: usize) -> Vec<usize> {
        let small_max = max_bs.clamp(1, 15);
        let mut bs_list: Vec<usize> = (1..=small_max).collect();
        if max_bs >= 16 {
            let mut b = 16;
            while b <= max_bs {
                bs_list.push(b);
                b += 16;
            }
            if !bs_list.contains(&max_bs) {
                bs_list.push(max_bs);
            }
        }
        bs_list
    }

    /// Capture CUDA graphs at each planned batch size.
    ///
    /// The `model_forward` closure must call
    /// `HunYuanDenseV1::step_batch_decode_graph` (or the Model wrapper)
    /// with the provided tensors. It is invoked once per batch size during
    /// capture — not executed, only recorded.
    ///
    /// # Arguments
    /// - `max_bs`: largest batch size to capture
    /// - `max_kv_len`: total KV buffer width (from `setup_batch_decode`)
    /// - `half_head_dim`: `config.head_dim() / 2` (rotary embedding width)
    /// - `dtype`: model dtype (e.g. BF16)
    /// - `model_forward`: `|input_ids, cos, sin, mask, write_pos| -> logits`
    pub fn capture<F>(
        &mut self,
        max_bs: usize,
        max_kv_len: usize,
        half_head_dim: usize,
        dtype: DType,
        mut model_forward: F,
    ) -> Result<()>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    {
        self.max_kv_len = max_kv_len;
        let device = Device::Cuda((*self.device).clone());

        // ── Allocate capture vars at max_bs ──
        let input_ids = Tensor::zeros((max_bs, 1), DType::U32, &device)?;
        let cos = Tensor::zeros((max_bs, 1, half_head_dim), dtype, &device)?;
        let sin = Tensor::zeros((max_bs, 1, half_head_dim), dtype, &device)?;
        let attention_mask = Tensor::zeros((max_bs, 1, 1, max_kv_len), dtype, &device)?;
        let write_pos = Tensor::zeros(max_bs, DType::U32, &device)?;

        let mut outputs = BTreeMap::new();

        // ── Setup memory pool (prevent deallocation between captures) ──
        sync_stream(&self.device)?;
        setup_mem_pool(&self.device)?;

        let stream = self.device.cuda_stream();

        // ── Capture from largest to smallest ──
        let batches = Self::planned_batches(max_bs);
        for &bs in batches.iter().rev() {
            let ids = input_ids.narrow(0, 0, bs)?;
            let c = cos.narrow(0, 0, bs)?;
            let s = sin.narrow(0, 0, bs)?;
            let m = attention_mask.narrow(0, 0, bs)?;
            let wp = write_pos.narrow(0, 0, bs)?;

            // Begin capture — all subsequent GPU work on this stream is
            // recorded into a graph rather than executed.
            sync_stream(&self.device)?;
            stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("begin_capture failed: {e:?}"))
                })?;

            // Forward pass — recorded, not executed
            let out = model_forward(&ids, &c, &s, &m, &wp)?;

            // End capture → instantiate executable graph
            let graph = stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("end_capture failed: {e:?}"))
                })?
                .ok_or_else(|| {
                    candle_core::Error::Msg("end_capture returned None (empty graph?)".into())
                })?;

            sync_stream(&self.device)?;
            outputs.insert(bs, out);
            self.captured.insert(bs, graph);
        }

        self.captured_bs = self.captured.keys().copied().collect();
        self.captured_bs.sort_unstable();

        self.vars = Some(CaptureVars {
            input_ids,
            cos,
            sin,
            attention_mask,
            write_pos,
            outputs,
        });

        eprintln!(
            "[cuda_graph] captured {} decode graphs at batch sizes {:?}",
            self.captured.len(),
            self.captured_bs
        );

        Ok(())
    }

    /// Find the smallest captured batch size ≥ `actual_bs`.
    fn select_batch(&self, actual_bs: usize) -> Option<usize> {
        self.captured_bs.iter().copied().find(|&b| b >= actual_bs)
    }

    /// Whether a graph exists that can serve `actual_bs`.
    pub fn is_captured(&self, actual_bs: usize) -> bool {
        self.select_batch(actual_bs).is_some()
    }

    /// Replay a captured graph with new input values.
    ///
    /// Copies `input_ids`, `cos`, `sin`, `mask`, `write_pos` into the
    /// pre-allocated capture vars, launches the graph, and returns the
    /// output narrowed to `actual_bs`.
    ///
    /// # Arguments
    /// - `input_ids`: `[actual_bs, 1]` u32
    /// - `cos`: `[actual_bs, 1, D/2]`
    /// - `sin`: `[actual_bs, 1, D/2]`
    /// - `mask`: `[actual_bs, 1, 1, kv_width]` — will be padded to max_kv_len
    /// - `write_pos`: `[actual_bs]` u32
    pub fn replay(
        &self,
        input_ids: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        write_pos: &Tensor,
    ) -> Result<Tensor> {
        let actual_bs = input_ids.dim(0)?;
        let batch = self
            .select_batch(actual_bs)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "No captured graph for batch size {actual_bs} (captured: {:?})",
                    self.captured_bs
                ))
            })?;

        let vars = self.vars.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("capture() not called yet".to_string())
        })?;

        // ── Copy inputs into capture vars ──
        // zero_set() clears the full buffer (handles padding for bs < batch).
        vars.input_ids.zero_set()?;
        vars.input_ids.slice_set(&input_ids.contiguous()?, 0, 0)?;

        vars.cos.zero_set()?;
        vars.cos.slice_set(&cos.contiguous()?, 0, 0)?;

        vars.sin.zero_set()?;
        vars.sin.slice_set(&sin.contiguous()?, 0, 0)?;

        vars.attention_mask.zero_set()?;
        // mask may be narrower than max_kv_len — pad dimensions
        let mask_c = mask.contiguous()?;
        let mask_width = mask_c.dim(3)?;
        if mask_width < self.max_kv_len {
            // Write into the left portion; right portion stays as 0.0 (masked out
            // by the -1e9 values already in the real mask, right portion = 0.0
            // means "attend" which is wrong).
            // Actually, for decode, the mask has shape [N, 1, 1, used_kv_len].
            // We need to pad to max_kv_len with -inf values.
            // Instead of pad-then-copy, we write the mask into the left portion
            // of a pre-zeroed buffer, then overwrite the right portion with -1e9.
            // But zero_set already cleared it to 0.0 — we need the right portion
            // to be -1e9 (masked). Let's fill the whole thing with -1e9 first.
            let neg_inf =
                Tensor::full(-1e9f32, vars.attention_mask.shape(), vars.attention_mask.device())?
                    .to_dtype(vars.attention_mask.dtype())?;
            vars.attention_mask.slice_set(&neg_inf, 0, 0)?;
        }
        vars.attention_mask.slice_set(&mask_c, 0, 0)?;

        vars.write_pos.zero_set()?;
        vars.write_pos.slice_set(&write_pos.contiguous()?, 0, 0)?;

        // ── Launch graph ──
        let graph = self.captured.get(&batch).unwrap();
        sync_stream(&self.device)?;
        graph.launch().map_err(|e| {
            candle_core::Error::Msg(format!("cuGraphLaunch failed: {e:?}"))
        })?;
        sync_stream(&self.device)?;

        // ── Narrow output to actual batch size ──
        let out = vars.outputs.get(&batch).ok_or_else(|| {
            candle_core::Error::Msg(format!("Missing output tensor for batch {batch}"))
        })?;
        out.narrow(0, 0, actual_bs)?.contiguous()
    }
}
