use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub alpha: Option<f64>,
    pub beta_fast: Option<f64>,
    pub beta_slow: Option<f64>,
    pub factor: Option<f64>,
    pub mscale: Option<f64>,
    pub mscale_all_dim: Option<f64>,
    #[serde(rename = "type")]
    pub rope_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>,
    pub rope_scaling: Option<RopeScaling>,
    pub attention_bias: Option<bool>,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    pub use_cla: Option<bool>,
    pub cla_share_factor: Option<usize>,
}

fn default_true() -> bool {
    true
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn attention_bias(&self) -> bool {
        self.attention_bias.unwrap_or(false)
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_theta.unwrap_or(10000.0)
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary_pos_emb(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q_embed = (q.broadcast_mul(cos)? + rotate_half(q)?.broadcast_mul(sin)?)?;
    let k_embed = (k.broadcast_mul(cos)? + rotate_half(k)?.broadcast_mul(sin)?)?;
    Ok((q_embed, k_embed))
}

struct RotaryEmbedding {
    inv_freq: Tensor,
    cos_cache: Option<Tensor>,
    sin_cache: Option<Tensor>,
    cached_len: usize,
}

impl RotaryEmbedding {
    fn new(config: &Config, device: &Device) -> Result<Self> {
        let dim = config.head_dim();
        let rope_theta = config.rope_theta();

        let inv_freq = if let Some(ref scaling) = config.rope_scaling {
            if let Some(alpha) = scaling.alpha {
                if alpha > 0.0 {
                    let base = rope_theta * alpha.powf(dim as f64 / (dim as f64 - 2.0));
                    let inv: Vec<f32> = (0..dim)
                        .step_by(2)
                        .map(|i| 1.0 / base.powf(i as f64 / dim as f64) as f32)
                        .collect();
                    Tensor::new(inv.as_slice(), device)?
                } else {
                    Self::default_inv_freq(dim, rope_theta, device)?
                }
            } else {
                Self::default_inv_freq(dim, rope_theta, device)?
            }
        } else {
            Self::default_inv_freq(dim, rope_theta, device)?
        };

        Ok(Self {
            inv_freq,
            cos_cache: None,
            sin_cache: None,
            cached_len: 0,
        })
    }

    fn default_inv_freq(dim: usize, base: f64, device: &Device) -> Result<Tensor> {
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f64 / dim as f64) as f32)
            .collect();
        Tensor::new(inv.as_slice(), device)
    }

    fn forward(&mut self, seq_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        if seq_len <= self.cached_len {
            if let (Some(ref cos), Some(ref sin)) = (&self.cos_cache, &self.sin_cache) {
                return Ok((cos.narrow(0, 0, seq_len)?, sin.narrow(0, 0, seq_len)?));
            }
        }

        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        // positions: [seq_len], inv_freq: [dim/2]
        let freqs = positions
            .unsqueeze(1)?
            .matmul(&self.inv_freq.unsqueeze(0)?)?; // [seq_len, dim/2]
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [seq_len, dim]
        let cos = emb.cos()?;
        let sin = emb.sin()?;

        self.cos_cache = Some(cos.clone());
        self.sin_cache = Some(sin.clone());
        self.cached_len = seq_len;

        Ok((cos, sin))
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    query_layernorm: Option<RmsNorm>,
    key_layernorm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
    #[cfg(feature = "flash-attn")]
    use_flash_attn: bool,
}

impl Attention {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let bias = config.attention_bias();

        let q_proj = if bias {
            candle_nn::linear(config.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?
        } else {
            linear_no_bias(config.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?
        };
        let k_proj = if bias {
            candle_nn::linear(config.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?
        } else {
            linear_no_bias(config.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?
        };
        let v_proj = if bias {
            candle_nn::linear(config.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?
        } else {
            linear_no_bias(config.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?
        };
        let o_proj = if bias {
            candle_nn::linear(num_heads * head_dim, config.hidden_size, vb.pp("o_proj"))?
        } else {
            linear_no_bias(num_heads * head_dim, config.hidden_size, vb.pp("o_proj"))?
        };

        let (query_layernorm, key_layernorm) = if config.use_qk_norm {
            (
                Some(candle_nn::rms_norm(head_dim, config.rms_norm_eps, vb.pp("query_layernorm"))?),
                Some(candle_nn::rms_norm(head_dim, config.rms_norm_eps, vb.pp("key_layernorm"))?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            query_layernorm,
            key_layernorm,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache: None,
            #[cfg(feature = "flash-attn")]
            use_flash_attn: false,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape: [B, S, num_heads * head_dim] -> [B, num_heads, S, head_dim]
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings.
        // cos/sin are pre-shaped by the caller:
        //   Single-sequence: [1, 1, seq_len, head_dim]
        //   Batched decode:  [N, 1, 1, head_dim]
        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

        // Apply QK norm after RoPE
        let q = if let Some(ref norm) = self.query_layernorm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref norm) = self.key_layernorm {
            norm.forward(&k)?
        } else {
            k
        };

        // KV cache
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Expand KV heads for GQA
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 {
            let (b, kv_heads, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .reshape((b, kv_heads * n_rep, s, d))?
        } else {
            k
        };
        let v = if n_rep > 1 {
            let (b, kv_heads, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .reshape((b, kv_heads * n_rep, s, d))?
        } else {
            v
        };

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [B, num_heads, S, head_dim] -> [B, S, hidden_size]
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, ()))?;

        self.o_proj.forward(&attn_output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::Activation::Silu.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(config, vb.pp("self_attn"))?;
        let mlp = Mlp::new(config, vb.pp("mlp"))?;
        let input_layernorm =
            candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states, cos, sin, attention_mask)?;
        let hidden_states = (residual + hidden_states)?;

        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual + hidden_states
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct HunYuanDenseV1 {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary_emb: RotaryEmbedding,
    config: Config,
    dtype: DType,
}

impl HunYuanDenseV1 {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let model_vb = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(config.vocab_size, config.hidden_size, model_vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = model_vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(config, layers_vb.pp(i))?);
        }

        let norm = candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, model_vb.pp("norm"))?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        let rotary_emb = RotaryEmbedding::new(config, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config: config.clone(),
            dtype,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        // Cast embedding output to the target dtype — the safetensors file may store
        // weights in BF16 which is unsupported for CPU matmul.
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let total_len = start_pos + seq_len;
        let (full_cos, full_sin) = self.rotary_emb.forward(total_len, input_ids.device())?;
        // Slice for current positions; cast to model dtype (RoPE computes in F32)
        let cos = full_cos.narrow(0, start_pos, seq_len)?.to_dtype(self.dtype)?;
        let sin = full_sin.narrow(0, start_pos, seq_len)?.to_dtype(self.dtype)?;
        // Shape for attention: [1, 1, seq_len, head_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Build causal mask
        let attention_mask = if seq_len > 1 {
            let total_len = start_pos + seq_len;
            // Build [seq_len, total_len] mask: 1.0 where allowed, 0.0 where masked
            let mut mask_data = vec![0f32; seq_len * total_len];
            for i in 0..seq_len {
                // Each query position i can attend to all cached positions + positions 0..=i
                for j in 0..total_len {
                    if j <= start_pos + i {
                        mask_data[i * total_len + j] = 1.0;
                    }
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, total_len), input_ids.device())?;
            // Convert: 0.0 (masked) -> -1e9, 1.0 (attend) -> 0.0
            let mask = mask
                .broadcast_lt(&Tensor::new(0.5f32, input_ids.device())?)?
                .to_dtype(self.dtype)?;
            let mask = (mask * (-1e9f64))?;
            Some(mask.unsqueeze(0)?.unsqueeze(0)?) // [1, 1, seq_len, total_len]
        } else {
            None
        };

        let mut hidden_states = hidden_states;
        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states.narrow(1, seq_len - 1, 1)?)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
        self.rotary_emb.cos_cache = None;
        self.rotary_emb.sin_cache = None;
        self.rotary_emb.cached_len = 0;
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Extract per-layer KV caches (cheap: Tensor is Arc-based).
    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers
            .iter()
            .map(|l| l.self_attn.kv_cache.clone())
            .collect()
    }

    /// Restore per-layer KV caches (e.g. after swapping sequences).
    pub fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        for (layer, cache) in self.layers.iter_mut().zip(caches.into_iter()) {
            layer.self_attn.kv_cache = cache;
        }
    }

    /// Batched decode forward: process N sequences in ONE forward pass.
    ///
    /// Each sequence contributes exactly 1 new token. KV caches are padded
    /// and stacked so the GPU processes all sequences in parallel, giving
    /// much higher utilization than N sequential single-token forward passes.
    ///
    /// # Arguments
    /// - `input_ids` — `[N, 1]` tensor (one token per sequence)
    /// - `positions` — start position for each sequence (= cached KV length)
    /// - `seq_kv_caches` — per-sequence, per-layer KV caches `[n_seqs][n_layers]`
    ///
    /// # Returns
    /// `(logits, updated_seq_kv_caches)` where logits is `[N, 1, vocab]`.
    pub fn forward_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        seq_kv_caches: Vec<Vec<Option<(Tensor, Tensor)>>>,
    ) -> Result<(Tensor, Vec<Vec<Option<(Tensor, Tensor)>>>)> {
        let n_seqs = positions.len();
        let device = input_ids.device();
        let head_dim = self.config.head_dim();
        let kv_heads = self.config.num_key_value_heads;

        // 1. Embed: [N, 1] → [N, 1, hidden]
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        // 2. RoPE: gather cos/sin for each sequence's position
        let max_pos = positions.iter().copied().max().unwrap_or(0) + 1;
        let (full_cos, full_sin) = self.rotary_emb.forward(max_pos, device)?;

        let pos_ids: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let pos_tensor = Tensor::new(pos_ids.as_slice(), device)?;
        // [N, head_dim] → [N, 1, 1, head_dim] for broadcasting in attention
        let cos = full_cos
            .index_select(&pos_tensor, 0)?
            .to_dtype(self.dtype)?
            .unsqueeze(1)?
            .unsqueeze(1)?;
        let sin = full_sin
            .index_select(&pos_tensor, 0)?
            .to_dtype(self.dtype)?
            .unsqueeze(1)?
            .unsqueeze(1)?;

        // 3. Build attention mask for KV padding
        let kv_lens: &[usize] = positions; // cached length = start_pos
        let max_kv_len = kv_lens.iter().copied().max().unwrap_or(0);
        let total_kv = max_kv_len + 1; // cached + new token

        let attention_mask = if max_kv_len > 0 {
            let has_padding = kv_lens.iter().any(|&l| l < max_kv_len);
            if has_padding {
                // Mask layout per sequence i: positions 0..kv_lens[i] are real,
                // kv_lens[i]..max_kv_len are padding (→ -1e9), max_kv_len is new token (→ 0).
                let mut mask_data = vec![0f32; n_seqs * total_kv];
                for i in 0..n_seqs {
                    for j in kv_lens[i]..max_kv_len {
                        mask_data[i * total_kv + j] = -1e9;
                    }
                }
                let mask = Tensor::from_vec(mask_data, (n_seqs, total_kv), device)?
                    .to_dtype(self.dtype)?;
                Some(mask.unsqueeze(1)?.unsqueeze(1)?) // [N, 1, 1, total_kv]
            } else {
                None // All same length, no padding needed
            }
        } else {
            None // No cached data (shouldn't happen in decode)
        };

        // 4. Process each layer
        let num_layers = self.layers.len();
        let mut updated_seq_caches: Vec<Vec<Option<(Tensor, Tensor)>>> =
            (0..n_seqs).map(|_| Vec::with_capacity(num_layers)).collect();

        let mut hidden_states = hidden_states;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Gather per-sequence caches for this layer
            let layer_caches: Vec<&Option<(Tensor, Tensor)>> =
                seq_kv_caches.iter().map(|seq| &seq[layer_idx]).collect();

            // Pad and stack into batched KV cache
            let batched_kv = pad_and_stack_kv_caches(
                &layer_caches, max_kv_len, kv_heads, head_dim, device, self.dtype,
            )?;

            // Set batched KV cache in the attention layer
            layer.self_attn.kv_cache = batched_kv;

            // Forward (attention automatically concatenates new K,V to cache)
            hidden_states =
                layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;

            // Extract new KV entries and append to each sequence's original cache.
            // After the concat in attention, the batched cache is:
            //   [N, kv_heads, max_kv_len+1, head_dim]
            // The NEW token's K,V is at position max_kv_len (the last one appended).
            if let Some((ref full_k, ref full_v)) = layer.self_attn.kv_cache {
                let new_k_all = full_k.narrow(2, max_kv_len, 1)?; // [N, kv_heads, 1, d]
                let new_v_all = full_v.narrow(2, max_kv_len, 1)?;

                for i in 0..n_seqs {
                    let new_k_i = new_k_all.narrow(0, i, 1)?; // [1, kv_heads, 1, d]
                    let new_v_i = new_v_all.narrow(0, i, 1)?;

                    let updated = match &layer_caches[i] {
                        Some((old_k, old_v)) => Some((
                            Tensor::cat(&[old_k.as_ref(), &new_k_i], 2)?,
                            Tensor::cat(&[old_v.as_ref(), &new_v_i], 2)?,
                        )),
                        None => Some((new_k_i, new_v_i)),
                    };
                    updated_seq_caches[i].push(updated);
                }
            } else {
                for i in 0..n_seqs {
                    updated_seq_caches[i].push(None);
                }
            }

            // Free batched KV to release memory
            layer.self_attn.kv_cache = None;
        }

        // 5. Final norm + lm_head
        let hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?; // [N, 1, vocab]

        Ok((logits, updated_seq_caches))
    }
}

/// Pad per-sequence KV caches to `max_len` and stack into a batched tensor.
///
/// Returns `Some((K, V))` with shape `[N, kv_heads, max_len, head_dim]`,
/// or `None` if `max_len == 0`.
fn pad_and_stack_kv_caches(
    caches: &[&Option<(Tensor, Tensor)>],
    max_len: usize,
    kv_heads: usize,
    head_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<Option<(Tensor, Tensor)>> {
    if max_len == 0 {
        return Ok(None);
    }

    let n = caches.len();
    let mut padded_ks = Vec::with_capacity(n);
    let mut padded_vs = Vec::with_capacity(n);

    // Pre-allocate a zero tensor for padding (shared across sequences)
    let max_pad_needed = caches
        .iter()
        .map(|c| match c {
            Some((k, _)) => max_len.saturating_sub(k.dim(2).unwrap_or(0)),
            None => max_len,
        })
        .max()
        .unwrap_or(0);
    let zero_pad = if max_pad_needed > 0 {
        Some(Tensor::zeros(
            (1, kv_heads, max_pad_needed, head_dim),
            dtype,
            device,
        )?)
    } else {
        None
    };

    for cache in caches {
        match cache {
            Some((k, v)) => {
                let cur_len = k.dim(2)?;
                let pad_len = max_len - cur_len;
                if pad_len > 0 {
                    let pad = zero_pad.as_ref().unwrap().narrow(2, 0, pad_len)?;
                    padded_ks.push(Tensor::cat(&[k.as_ref(), &pad], 2)?);
                    padded_vs.push(Tensor::cat(&[v.as_ref(), &pad], 2)?);
                } else {
                    padded_ks.push(k.clone());
                    padded_vs.push(v.clone());
                }
            }
            None => {
                let zeros = Tensor::zeros((1, kv_heads, max_len, head_dim), dtype, device)?;
                padded_ks.push(zeros.clone());
                padded_vs.push(zeros);
            }
        }
    }

    let stacked_k = Tensor::cat(&padded_ks, 0)?; // [N, kv_heads, max_len, head_dim]
    let stacked_v = Tensor::cat(&padded_vs, 0)?;
    Ok(Some((stacked_k, stacked_v)))
}
