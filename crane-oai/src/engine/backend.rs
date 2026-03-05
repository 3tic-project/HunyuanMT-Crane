//! Model backend abstraction for the inference engine.
//!
//! The [`ModelBackend`] trait decouples the engine from specific model implementations,
//! allowing any compatible LLM to be served through the OpenAI-compatible API.
//!
//! # Capability Levels
//!
//! | Capability        | Required | Effect when absent                            |
//! |-------------------|----------|-----------------------------------------------|
//! | `forward_step`    | Yes      | —                                             |
//! | KV cache swap     | No       | `max_concurrent` capped to 1                  |
//! | Batch decode      | No       | Sequences decoded sequentially per step        |

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

// ─────────────────────────────────────────────────────────────
//  Trait
// ─────────────────────────────────────────────────────────────

/// Core abstraction over different model backends.
///
/// All models must support single-sequence forward passes and KV cache clearing.
/// Optionally, models can support KV cache extraction/restoration (for concurrent
/// sequence serving) and batched decoding (for GPU-efficient parallel generation).
pub trait ModelBackend: Send + 'static {
    /// Run a forward pass for a single sequence.
    ///
    /// * `input_ids` — token IDs to process
    /// * `start_pos` — KV cache position (0 for a fresh sequence)
    ///
    /// Returns logits tensor, typically `[1, seq_len, vocab_size]`.
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor>;

    /// Clear all KV caches.
    fn clear_kv_cache(&mut self);

    /// Number of transformer layers (for KV cache vector sizing).
    fn num_layers(&self) -> usize;

    /// Device the model is running on.
    fn device(&self) -> &Device;

    /// Data type of model weights.
    fn dtype(&self) -> DType;

    /// Reference to the underlying tokenizer.
    fn tokenizer(&self) -> &tokenizers::Tokenizer;

    /// The model's end-of-sequence token ID(s).
    fn eos_token_id(&self) -> Vec<u32>;

    /// Warm up the model with a small forward pass.
    fn warmup(&mut self);

    // ── KV cache swap (for concurrent sequence serving) ───────

    /// Whether this backend supports extracting and restoring KV caches.
    fn supports_kv_swap(&self) -> bool {
        false
    }

    /// Extract per-layer KV caches from the model.
    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        vec![]
    }

    /// Restore per-layer KV caches into the model.
    fn set_kv_caches(&mut self, _caches: Vec<Option<(Tensor, Tensor)>>) {}

    /// Compute bytes held by the model's active KV caches without copying.
    /// Used for memory tracking without the overhead of `get_kv_caches()`.
    fn active_kv_cache_bytes(&self) -> u64 {
        0
    }

    // ── Optional KV cache quantization hooks (for swapped-out states) ──

    /// Whether this backend can quantize inactive KV caches for memory savings.
    fn supports_kv_cache_quantization(&self) -> bool {
        false
    }

    /// Quantize swapped-out KV caches in-place.
    ///
    /// Default: no-op.
    fn quantize_kv_caches_inplace(
        &self,
        _caches: &mut [Option<(Tensor, Tensor)>],
    ) -> candle_core::Result<()> {
        Ok(())
    }

    /// Dequantize swapped-out KV caches in-place before loading into the model.
    ///
    /// Default: no-op.
    fn dequantize_kv_caches_inplace(
        &self,
        _caches: &mut [Option<(Tensor, Tensor)>],
    ) -> candle_core::Result<()> {
        Ok(())
    }


    // ── Batch decode (GPU-efficient concurrent serving) ───────

    /// Whether this backend supports batched decoding.
    fn supports_batch_decode(&self) -> bool {
        false
    }

    /// Pad and load per-sequence KV caches for batched decoding.
    fn setup_batch_decode(
        &mut self,
        _seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        _extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    /// Run one batched decode step.
    fn step_batch_decode(
        &mut self,
        _input_ids: &Tensor,
        _positions: &[usize],
        _attention_mask: Option<&Tensor>,
        _batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    /// Extract per-sequence KV caches from batched state.
    fn extract_batch_kv(
        &mut self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    /// Build attention mask for batched decoding.
    fn build_batch_decode_mask(
        &self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        candle_core::bail!("Batch decode not supported by this backend")
    }
}

// ─────────────────────────────────────────────────────────────
//  Hunyuan Dense Backend
// ─────────────────────────────────────────────────────────────

pub struct HunyuanBackend {
    pub model: crane_core::models::hunyuan_dense::Model,
}

impl HunyuanBackend {
    pub fn new(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: crane_core::models::hunyuan_dense::ModelFormat,
    ) -> Result<Self> {
        let model =
            crane_core::models::hunyuan_dense::Model::new_with_format(model_path, device, dtype, format)?;
        Ok(Self { model })
    }

    /// Quantize inactive KV tensors only when runtime dtype is fp32.
    ///
    /// For fp16/bf16 runtime there is no memory win, so keep native dtype.
    fn kv_quant_dtype(&self) -> Option<DType> {
        match self.model.dtype {
            DType::F32 => Some(DType::F16),
            _ => None,
        }
    }

    fn cast_kv_caches_dtype(
        caches: &mut [Option<(Tensor, Tensor)>],
        target: DType,
    ) -> candle_core::Result<()> {
        for cache in caches.iter_mut() {
            if let Some((k, v)) = cache.take() {
                let k = if k.dtype() == target {
                    k
                } else {
                    k.to_dtype(target)?
                };
                let v = if v.dtype() == target {
                    v
                } else {
                    v.to_dtype(target)?
                };
                *cache = Some((k, v));
            }
        }
        Ok(())
    }
}

impl ModelBackend for HunyuanBackend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.model.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        vec![120020]
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    // ── KV swap ──

    fn supports_kv_swap(&self) -> bool {
        true
    }

    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.model.get_kv_caches()
    }

    fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.model.set_kv_caches(caches);
    }

    fn active_kv_cache_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    fn supports_kv_cache_quantization(&self) -> bool {
        self.kv_quant_dtype().is_some()
    }

    fn quantize_kv_caches_inplace(
        &self,
        caches: &mut [Option<(Tensor, Tensor)>],
    ) -> candle_core::Result<()> {
        if let Some(target) = self.kv_quant_dtype() {
            Self::cast_kv_caches_dtype(caches, target)?;
        }
        Ok(())
    }

    fn dequantize_kv_caches_inplace(
        &self,
        caches: &mut [Option<(Tensor, Tensor)>],
    ) -> candle_core::Result<()> {
        Self::cast_kv_caches_dtype(caches, self.model.dtype)
    }

    // ── Batch decode ──

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.model.setup_batch_decode(seq_kv_caches, extra_room)
    }

    fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        self.model
            .step_batch_decode_with_input_ids(input_ids, positions, attention_mask, batch_kv_info)
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        self.model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        crane_core::models::hunyuan_dense::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen 3.5 Backend
// ─────────────────────────────────────────────────────────────

pub struct Qwen35Backend {
    pub model: crane_core::models::qwen35::Model,
    dtype: DType,
    layer_is_full_attn: Vec<bool>,
}

impl Qwen35Backend {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model = crane_core::models::qwen35::Model::new(model_path, device, dtype)?;
        let layer_is_full_attn = model.layer_is_full_attn();
        Ok(Self {
            model,
            dtype: *dtype,
            layer_is_full_attn,
        })
    }

    fn kv_to_layer_states(
        &self,
        caches: &[Option<(Tensor, Tensor)>],
    ) -> Vec<crane_core::models::qwen35::modeling::LayerState> {
        let mut states = Vec::with_capacity(self.layer_is_full_attn.len());
        for (idx, is_full) in self.layer_is_full_attn.iter().copied().enumerate() {
            let cache = caches.get(idx).cloned().unwrap_or(None);
            if is_full {
                states.push(crane_core::models::qwen35::modeling::LayerState::FullAttn {
                    kv_cache: cache,
                });
            } else {
                let (recurrent_state, conv_history) = match cache {
                    Some((state, history)) => (Some(state), Some(history)),
                    None => (None, None),
                };
                states.push(crane_core::models::qwen35::modeling::LayerState::LinearAttn {
                    recurrent_state,
                    conv_history,
                });
            }
        }
        states
    }

    fn layer_states_to_kv(
        &self,
        states: Vec<crane_core::models::qwen35::modeling::LayerState>,
    ) -> Vec<Option<(Tensor, Tensor)>> {
        states
            .into_iter()
            .map(|state| match state {
                crane_core::models::qwen35::modeling::LayerState::FullAttn { kv_cache } => {
                    kv_cache
                }
                crane_core::models::qwen35::modeling::LayerState::LinearAttn {
                    recurrent_state,
                    conv_history,
                } => match (recurrent_state, conv_history) {
                    (Some(state), Some(history)) => Some((state, history)),
                    _ => None,
                },
            })
            .collect()
    }

    /// Quantize inactive KV/state tensors only when runtime dtype is fp32.
    ///
    /// For fp16/bf16 runtime there is no memory win, so keep native dtype.
    fn kv_quant_dtype(&self) -> Option<DType> {
        match self.dtype {
            DType::F32 => Some(DType::F16),
            _ => None,
        }
    }

    fn cast_kv_caches_dtype(
        caches: &mut [Option<(Tensor, Tensor)>],
        target: DType,
    ) -> candle_core::Result<()> {
        for cache in caches.iter_mut() {
            if let Some((k, v)) = cache.take() {
                let k = if k.dtype() == target {
                    k
                } else {
                    k.to_dtype(target)?
                };
                let v = if v.dtype() == target {
                    v
                } else {
                    v.to_dtype(target)?
                };
                *cache = Some((k, v));
            }
        }
        Ok(())
    }
}

impl ModelBackend for Qwen35Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        self.model.device()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        self.model.tokenizer()
    }

    fn eos_token_id(&self) -> Vec<u32> {
        // Qwen3.5 chat models stop at <|im_end|> (151645).
        // Also include <|endoftext|> (151643) as a fallback.
        let tok = self.model.tokenizer();
        let mut ids = Vec::new();
        if let Some(id) = tok.token_to_id("<|im_end|>") {
            ids.push(id);
        }
        if let Some(id) = tok.token_to_id("<|endoftext|>") {
            ids.push(id);
        }
        if ids.is_empty() {
            ids.push(151645);
            ids.push(151643);
        }
        ids
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    // ── KV swap ──

    fn supports_kv_swap(&self) -> bool {
        true
    }

    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layer_states_to_kv(self.model.save_full_state())
    }

    fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        let states = self.kv_to_layer_states(&caches);
        self.model.restore_full_state(states);
    }

    fn active_kv_cache_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    fn supports_kv_cache_quantization(&self) -> bool {
        self.kv_quant_dtype().is_some()
    }

    fn quantize_kv_caches_inplace(
        &self,
        caches: &mut [Option<(Tensor, Tensor)>],
    ) -> candle_core::Result<()> {
        if let Some(target) = self.kv_quant_dtype() {
            Self::cast_kv_caches_dtype(caches, target)?;
        }
        Ok(())
    }

    fn dequantize_kv_caches_inplace(
        &self,
        caches: &mut [Option<(Tensor, Tensor)>],
    ) -> candle_core::Result<()> {
        Self::cast_kv_caches_dtype(caches, self.dtype)
    }

    // ── Batch decode ──

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        let seq_states: Vec<Vec<crane_core::models::qwen35::modeling::LayerState>> =
            seq_kv_caches
                .iter()
                .map(|c| self.kv_to_layer_states(c))
                .collect();
        self.model.setup_batch_decode(&seq_states, extra_room)
    }

    fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        self.model
            .step_batch_decode_with_input_ids(input_ids, positions, attention_mask, batch_kv_info)
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        let seq_states = self
            .model
            .extract_batch_state(kv_lens, original_max_kv, rounds_done)?;
        Ok(seq_states
            .into_iter()
            .map(|s| self.layer_states_to_kv(s))
            .collect())
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        crane_core::models::qwen35::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen 2.5 Backend
// ─────────────────────────────────────────────────────────────

pub struct Qwen25Backend {
    pub model: crane_core::models::qwen25::Model,
    dtype: DType,
}

impl Qwen25Backend {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model = crane_core::models::qwen25::Model::new(model_path, device, dtype)?;
        Ok(Self {
            model,
            dtype: *dtype,
        })
    }
}

impl ModelBackend for Qwen25Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        0 // KV swap not supported; vector is unused
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        self.model
            .tokenizer
            .tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| self.model.tokenizer.tokenizer.token_to_id("<|im_end|>"))
            .map(|id| vec![id])
            .unwrap_or_else(|| vec![151643])
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen 3 Backend
// ─────────────────────────────────────────────────────────────

pub struct Qwen3Backend {
    pub model: crane_core::models::qwen3::Model,
    dtype: DType,
}

impl Qwen3Backend {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model = crane_core::models::qwen3::Model::new(model_path, device, dtype)?;
        Ok(Self {
            model,
            dtype: *dtype,
        })
    }

    /// Quantize inactive KV tensors only when runtime dtype is fp32.
    ///
    /// For fp16/bf16 runtime there is no memory win, so keep native dtype.
    fn kv_quant_dtype(&self) -> Option<DType> {
        match self.dtype {
            DType::F32 => Some(DType::F16),
            _ => None,
        }
    }

    fn cast_kv_caches_dtype(
        caches: &mut [Option<(Tensor, Tensor)>],
        target: DType,
    ) -> candle_core::Result<()> {
        for cache in caches.iter_mut() {
            if let Some((k, v)) = cache.take() {
                let k = if k.dtype() == target {
                    k
                } else {
                    k.to_dtype(target)?
                };
                let v = if v.dtype() == target {
                    v
                } else {
                    v.to_dtype(target)?
                };
                *cache = Some((k, v));
            }
        }
        Ok(())
    }
}

impl ModelBackend for Qwen3Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        // Qwen3 chat models stop at <|im_end|> (151645).
        // Also include <|endoftext|> (151643) as a fallback.
        let tok = &self.model.tokenizer.tokenizer;
        let mut ids = Vec::new();
        if let Some(id) = tok.token_to_id("<|im_end|>") { ids.push(id); }
        if let Some(id) = tok.token_to_id("<|endoftext|>") { ids.push(id); }
        if ids.is_empty() { ids.push(151645); ids.push(151643); }
        ids
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    // ── KV swap ──

    fn supports_kv_swap(&self) -> bool {
        true
    }

    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.model.get_kv_caches()
    }

    fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.model.set_kv_caches(caches);
    }

    fn active_kv_cache_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    fn supports_kv_cache_quantization(&self) -> bool {
        self.kv_quant_dtype().is_some()
    }

    fn quantize_kv_caches_inplace(
        &self,
        caches: &mut [Option<(Tensor, Tensor)>],
    ) -> candle_core::Result<()> {
        if let Some(target) = self.kv_quant_dtype() {
            Self::cast_kv_caches_dtype(caches, target)?;
        }
        Ok(())
    }

    fn dequantize_kv_caches_inplace(
        &self,
        caches: &mut [Option<(Tensor, Tensor)>],
    ) -> candle_core::Result<()> {
        Self::cast_kv_caches_dtype(caches, self.dtype)
    }

    // ── Batch decode ──

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.model.setup_batch_decode(seq_kv_caches, extra_room)
    }

    fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        self.model
            .step_batch_decode_with_input_ids(input_ids, positions, attention_mask, batch_kv_info)
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        self.model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        crane_core::models::qwen3::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}
