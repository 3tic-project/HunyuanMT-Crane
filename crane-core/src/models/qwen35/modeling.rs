use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::rotary_emb::rope;
use candle_nn::{conv1d_no_bias, linear, linear_no_bias, Conv1d, Conv1dConfig, Linear, VarBuilder};
use serde::Deserialize;
use serde_json::Value;

struct Qwen35RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen35RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let dims = x.dims();
        if dims.is_empty() {
            return Ok(x.to_dtype(x_dtype)?);
        }
        let last = dims.len() - 1;

        let var = x.sqr()?.mean(last)?;
        let denom = (var + self.eps)?.sqrt()?.unsqueeze(last)?;
        let mut y = x.broadcast_div(&denom)?;

        let mut shape = vec![1usize; dims.len()];
        shape[last] = self.weight.dims1()?;
        let scale = (&self.weight.to_dtype(DType::F32)? + 1.0)?.reshape(shape)?;
        y = y.broadcast_mul(&scale)?;

        y.to_dtype(x_dtype)
    }
}

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_head_dim() -> usize {
    256
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_linear_conv_kernel_dim() -> usize {
    4
}

fn default_linear_key_head_dim() -> usize {
    128
}

fn default_linear_value_head_dim() -> usize {
    128
}

fn default_linear_num_key_heads() -> usize {
    16
}

fn default_linear_num_value_heads() -> usize {
    32
}

fn default_partial_rotary_factor() -> f64 {
    0.25
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_parameters: Option<Value>,
    #[serde(default)]
    pub rope_scaling: Option<Value>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
    #[serde(default = "default_head_dim")]
    pub full_attention_head_dim: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

impl Config {
    pub fn post_init(&mut self) -> Result<()> {
        if self.head_dim.is_none() {
            self.head_dim = Some(self.full_attention_head_dim);
        }

        if self.layer_types.is_none() {
            let interval = self.full_attention_interval.unwrap_or(4).max(1);
            self.layer_types = Some(
                (0..self.num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % interval == 0 {
                            "full_attention".to_string()
                        } else {
                            "linear_attention".to_string()
                        }
                    })
                    .collect(),
            );
        }

        let layer_types = self.layer_types.as_ref().unwrap();
        if layer_types.len() != self.num_hidden_layers {
            candle_core::bail!(
                "layer_types length {} does not match num_hidden_layers {}",
                layer_types.len(),
                self.num_hidden_layers
            );
        }

        if let Some(theta) = extract_rope_theta(self.rope_parameters.as_ref()) {
            self.rope_theta = theta;
        } else if let Some(theta) = extract_rope_theta(self.rope_scaling.as_ref()) {
            self.rope_theta = theta;
        }

        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rotary_dim(&self) -> usize {
        let mut rd = ((self.head_dim() as f64) * self.partial_rotary_factor).round() as usize;
        if rd == 0 {
            rd = self.head_dim();
        }
        rd = rd.min(self.head_dim());
        if rd % 2 == 1 {
            rd -= 1;
        }
        rd.max(2)
    }

    pub fn layer_type(&self, idx: usize) -> &str {
        self.layer_types
            .as_ref()
            .and_then(|v| v.get(idx))
            .map(|s| s.as_str())
            .unwrap_or("full_attention")
    }
}

fn extract_rope_theta(v: Option<&Value>) -> Option<f64> {
    let Some(v) = v else {
        return None;
    };
    match v {
        Value::Object(map) => map.get("rope_theta").and_then(value_to_f64),
        _ => None,
    }
}

fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg = x.affine(-1.0, 0.0)?;
    let exp = neg.exp()?;
    ((&exp + 1.0)?).recip()
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()? + 1.0)?.log()
}

fn silu(x: &Tensor) -> Result<Tensor> {
    candle_nn::Activation::Silu.forward(x)
}

fn l2norm_last_dim(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Ok(x.clone());
    }
    let last = dims.len() - 1;
    let norm = x.sqr()?.sum(last)?;
    let denom = (norm + eps)?.sqrt()?.unsqueeze(last)?;
    x.broadcast_div(&denom)
}

fn rms_norm_gated(core: &Tensor, gate: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let (n, d) = core.dims2()?;
    let work_dtype = core.dtype();
    let var = core.sqr()?.mean(1)?;
    let denom = (var + eps)?.sqrt()?.unsqueeze(1)?;
    let normed = core.broadcast_div(&denom)?;

    let w = if weight.dims().len() == 1 {
        weight.to_dtype(work_dtype)?.reshape((1, d))?
    } else {
        weight.to_dtype(work_dtype)?
    };

    let gate = gate.to_dtype(work_dtype)?;

    let normed = normed.broadcast_mul(&w)?;
    let gated = normed.broadcast_mul(&silu(&gate)?)?;
    gated.reshape((n, d))
}

struct RotaryEmbedding {
    cos_table: Tensor,
    sin_table: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &Config, device: &Device) -> Result<Self> {
        let dim = config.rotary_dim();
        let base = config.rope_theta;
        let max_pos = config.max_position_embeddings;

        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions
            .unsqueeze(1)?
            .matmul(&inv_freq.unsqueeze(0)?)?;

        let cos_table = freqs.cos()?.contiguous()?;
        let sin_table = freqs.sin()?.contiguous()?;

        Ok(Self {
            cos_table,
            sin_table,
        })
    }

    fn forward(&self, total_len: usize, start_pos: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self
            .cos_table
            .narrow(0, 0, total_len)?
            .narrow(0, start_pos, seq_len)?;
        let sin = self
            .sin_table
            .narrow(0, 0, total_len)?
            .narrow(0, start_pos, seq_len)?;
        Ok((cos, sin))
    }
}

fn apply_partial_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, rotary_dim: usize) -> Result<Tensor> {
    let full_dim = x.dim(D::Minus1)?;
    if rotary_dim >= full_dim {
        return rope(&x.contiguous()?, cos, sin);
    }

    let last_dim = x.dims().len() - 1;
    let x_rot = x.narrow(D::Minus1, 0, rotary_dim)?;
    let x_pass = x.narrow(D::Minus1, rotary_dim, full_dim - rotary_dim)?;
    let x_rot = rope(&x_rot.contiguous()?, cos, sin)?;
    Tensor::cat(&[&x_rot, &x_pass], last_dim)
}

struct FullAttention {
    qgkv_proj: Linear,
    o_proj: Linear,
    q_norm: Option<Qwen35RmsNorm>,
    k_norm: Option<Qwen35RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
    cache_seq_len: usize,
}

impl FullAttention {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let q_dim = config.num_attention_heads * head_dim;
        let kv_dim = config.num_key_value_heads * head_dim;

        let q_proj = if config.attention_bias {
            linear(config.hidden_size, q_dim * 2, vb.pp("q_proj"))?
        } else {
            linear_no_bias(config.hidden_size, q_dim * 2, vb.pp("q_proj"))?
        };
        let k_proj = if config.attention_bias {
            linear(config.hidden_size, kv_dim, vb.pp("k_proj"))?
        } else {
            linear_no_bias(config.hidden_size, kv_dim, vb.pp("k_proj"))?
        };
        let v_proj = if config.attention_bias {
            linear(config.hidden_size, kv_dim, vb.pp("v_proj"))?
        } else {
            linear_no_bias(config.hidden_size, kv_dim, vb.pp("v_proj"))?
        };
        let o_proj = if config.attention_bias {
            linear(q_dim, config.hidden_size, vb.pp("o_proj"))?
        } else {
            linear_no_bias(q_dim, config.hidden_size, vb.pp("o_proj"))?
        };

        let qgkv_w = Tensor::cat(
            &[q_proj.weight(), k_proj.weight(), v_proj.weight()],
            0,
        )?;
        let qgkv_b = match (q_proj.bias(), k_proj.bias(), v_proj.bias()) {
            (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[qb, kb, vb], 0)?),
            _ => None,
        };
        let qgkv_proj = Linear::new(qgkv_w, qgkv_b);

        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(Qwen35RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?),
                Some(Qwen35RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            qgkv_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            rotary_dim: config.rotary_dim(),
            q_dim,
            kv_dim,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    fn update_kv_cache(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let new_seq_len = k.dim(2)?;
        let cache_seq_len = self.cache_seq_len;

        match self.kv_cache.take() {
            Some((buf_k, buf_v)) => {
                let buf_len = buf_k.dim(2)?;
                let new_total = cache_seq_len + new_seq_len;

                if new_total <= buf_len {
                    buf_k.slice_set(&k, 2, cache_seq_len)?;
                    buf_v.slice_set(&v, 2, cache_seq_len)?;
                    let k_view = buf_k.narrow(2, 0, new_total)?;
                    let v_view = buf_v.narrow(2, 0, new_total)?;
                    self.kv_cache = Some((buf_k, buf_v));
                    self.cache_seq_len = new_total;
                    Ok((k_view, v_view))
                } else {
                    let cur_k = buf_k.narrow(2, 0, cache_seq_len)?;
                    let cur_v = buf_v.narrow(2, 0, cache_seq_len)?;
                    drop(buf_k);
                    drop(buf_v);
                    let full_k = Tensor::cat(&[&cur_k, &k], 2)?;
                    let full_v = Tensor::cat(&[&cur_v, &v], 2)?;
                    let total = full_k.dim(2)?;
                    let room = 256;
                    let (b, h, _, d) = full_k.dims4()?;
                    let new_buf_k = Tensor::zeros((b, h, total + room, d), k.dtype(), k.device())?;
                    let new_buf_v = Tensor::zeros((b, h, total + room, d), v.dtype(), v.device())?;
                    new_buf_k.slice_set(&full_k, 2, 0)?;
                    new_buf_v.slice_set(&full_v, 2, 0)?;
                    self.kv_cache = Some((new_buf_k, new_buf_v));
                    self.cache_seq_len = total;
                    Ok((full_k, full_v))
                }
            }
            None => {
                let (b, h, s, d) = k.dims4()?;
                let room = 256;
                let buf_k = Tensor::zeros((b, h, s + room, d), k.dtype(), k.device())?;
                let buf_v = Tensor::zeros((b, h, s + room, d), v.dtype(), v.device())?;
                buf_k.slice_set(&k, 2, 0)?;
                buf_v.slice_set(&v, 2, 0)?;
                self.kv_cache = Some((buf_k, buf_v));
                self.cache_seq_len = s;
                Ok((k, v))
            }
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        let qgkv = self.qgkv_proj.forward(hidden_states)?;
        let qg = qgkv
            .narrow(D::Minus1, 0, self.q_dim * 2)?
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = qg
            .narrow(D::Minus1, 0, self.head_dim)?
            .reshape((b_sz, seq_len, self.q_dim))?;
        let gate = qg
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b_sz, seq_len, self.q_dim))?;
        let k = qgkv.narrow(D::Minus1, self.q_dim * 2, self.kv_dim)?;
        let v = qgkv.narrow(D::Minus1, self.q_dim * 2 + self.kv_dim, self.kv_dim)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = if let Some(ref norm) = self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref norm) = self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        let q = apply_partial_rope(&q, cos, sin, self.rotary_dim)?;
        let k = apply_partial_rope(&k, cos, sin, self.rotary_dim)?;

        let (k, v) = self.update_kv_cache(k, v)?;
        let gate = sigmoid(&gate)?;

        let n_rep = self.num_heads / self.num_kv_heads;

        if n_rep > 1 && seq_len == 1 {
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let q_g = (q.reshape((b_sz, self.num_kv_heads, n_rep, self.head_dim))? * scale)?
                .contiguous()?;
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let v = v.contiguous()?;
            let attn_weights = q_g.matmul(&k_t)?;
            let attn_weights = match attention_mask {
                Some(mask) => attn_weights.broadcast_add(mask)?,
                None => attn_weights,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v)?;
            let attn_output = attn_output
                .reshape((b_sz, self.num_heads, self.head_dim))?
                .reshape((b_sz, 1, self.q_dim))?;
            let attn_output = attn_output.broadcast_mul(&gate)?;
            return self.o_proj.forward(&attn_output);
        }

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

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q = q.contiguous()?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let v = v.contiguous()?;
        let attn_weights = (q.matmul(&k_t)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, self.q_dim))?;
        let attn_output = attn_output.broadcast_mul(&gate)?;

        self.o_proj.forward(&attn_output)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
        self.cache_seq_len = 0;
    }
}

struct LinearAttention {
    in_proj_qkvz: Linear,
    in_proj_ba: Linear,
    conv1d: Conv1d,
    dt_bias: Tensor,
    a_log: Tensor,
    norm_weight: Tensor,
    out_proj: Linear,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_dim: usize,
    rms_norm_eps: f64,
    conv_history: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl LinearAttention {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;

        let in_proj_qkv = linear_no_bias(config.hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(config.hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_qkvz = Linear::new(
            Tensor::cat(&[in_proj_qkv.weight(), in_proj_z.weight()], 0)?,
            None,
        );

        let in_proj_b = linear_no_bias(config.hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(config.hidden_size, num_v_heads, vb.pp("in_proj_a"))?;
        let in_proj_ba = Linear::new(
            Tensor::cat(&[in_proj_b.weight(), in_proj_a.weight()], 0)?,
            None,
        );

        let conv_cfg = Conv1dConfig {
            padding: config.linear_conv_kernel_dim.saturating_sub(1),
            groups: conv_dim,
            ..Default::default()
        };
        let conv1d = conv1d_no_bias(
            conv_dim,
            conv_dim,
            config.linear_conv_kernel_dim,
            conv_cfg,
            vb.pp("conv1d"),
        )?;

        let dt_bias = vb.get(num_v_heads, "dt_bias")?;
        let a_log = vb.get(num_v_heads, "A_log")?;
        let norm_weight = vb.pp("norm").get(head_v_dim, "weight")?;
        let out_proj = linear_no_bias(value_dim, config.hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj_qkvz,
            in_proj_ba,
            conv1d,
            dt_bias,
            a_log,
            norm_weight,
            out_proj,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_dim: config.linear_conv_kernel_dim,
            rms_norm_eps: config.rms_norm_eps,
            conv_history: None,
            recurrent_state: None,
        })
    }

    fn causal_depthwise_conv(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        // mixed_qkv: [B, S, conv_dim]
        let (b, s, conv_dim) = mixed_qkv.dims3()?;
        let mixed_qkv = mixed_qkv.transpose(1, 2)?; // [B, conv_dim, S]

        if s == 1 {
            let history = if let Some(prev) = &self.conv_history {
                Tensor::cat(&[prev, &mixed_qkv], 2)?
            } else {
                mixed_qkv.clone()
            };

            let total_len = history.dim(2)?;
            let keep = self.conv_kernel_dim.saturating_sub(1);
            if keep > 0 {
                let keep_len = keep.min(total_len);
                self.conv_history = Some(
                    history
                        .narrow(2, total_len - keep_len, keep_len)?
                        .contiguous()?,
                );
            } else {
                self.conv_history = None;
            }

            let window = if total_len >= self.conv_kernel_dim {
                history.narrow(2, total_len - self.conv_kernel_dim, self.conv_kernel_dim)?
            } else {
                let pad_len = self.conv_kernel_dim - total_len;
                let left_pad = Tensor::zeros((b, conv_dim, pad_len), history.dtype(), history.device())?;
                Tensor::cat(&[&left_pad, &history], 2)?
            };

            let mut weight = self
                .conv1d
                .weight()
                .reshape((1, conv_dim, self.conv_kernel_dim))?;
            if weight.dtype() != window.dtype() {
                weight = weight.to_dtype(window.dtype())?;
            }

            let mut conv_cur = window.broadcast_mul(&weight)?.sum(2)?; // [B, conv_dim]
            if let Some(bias) = self.conv1d.bias() {
                let mut bias = bias.reshape((1, conv_dim))?;
                if bias.dtype() != conv_cur.dtype() {
                    bias = bias.to_dtype(conv_cur.dtype())?;
                }
                conv_cur = conv_cur.broadcast_add(&bias)?;
            }

            let conv_cur = silu(&conv_cur.unsqueeze(2)?)?; // [B, conv_dim, 1]
            return conv_cur.transpose(1, 2)?.reshape((b, s, conv_dim));
        }

        let history = if let Some(prev) = &self.conv_history {
            Tensor::cat(&[prev, &mixed_qkv], 2)?
        } else {
            mixed_qkv.clone()
        };

        let total_len = history.dim(2)?;
        let keep = self.conv_kernel_dim.saturating_sub(1);
        if keep > 0 {
            let keep_len = keep.min(total_len);
            self.conv_history = Some(
                history
                    .narrow(2, total_len - keep_len, keep_len)?
                    .contiguous()?,
            );
        } else {
            self.conv_history = None;
        }

        let conv_full = self.conv1d.forward(&history)?; // [B, conv_dim, total_len + k - 1]
        let conv_causal = conv_full.narrow(2, 0, total_len)?;
        let conv_cur = conv_causal.narrow(2, total_len.saturating_sub(s), s)?;
        let conv_cur = silu(&conv_cur)?;

        conv_cur.transpose(1, 2)?.reshape((b, s, conv_dim))
    }

    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;

        let mixed_qkvz = self.in_proj_qkvz.forward(hidden_states)?;
        let mixed_qkv = mixed_qkvz.narrow(D::Minus1, 0, self.key_dim * 2 + self.value_dim)?;
        let mixed_qkv = self.causal_depthwise_conv(&mixed_qkv)?;

        let z = mixed_qkvz
            .narrow(D::Minus1, self.key_dim * 2 + self.value_dim, self.value_dim)?
            .reshape((b, seq_len, self.num_v_heads, self.head_v_dim))?;

        let ba = self.in_proj_ba.forward(hidden_states)?;
        let b_proj = ba.narrow(D::Minus1, 0, self.num_v_heads)?;
        let a_proj = ba.narrow(D::Minus1, self.num_v_heads, self.num_v_heads)?;

        let query = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let key = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let value = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let mut query = query.reshape((b, seq_len, self.num_k_heads, self.head_k_dim))?;
        let mut key = key.reshape((b, seq_len, self.num_k_heads, self.head_k_dim))?;
        let value = value.reshape((b, seq_len, self.num_v_heads, self.head_v_dim))?;

        if self.num_v_heads % self.num_k_heads != 0 {
            candle_core::bail!(
                "linear_num_value_heads ({}) must be divisible by linear_num_key_heads ({})",
                self.num_v_heads,
                self.num_k_heads
            );
        }

        let rep = self.num_v_heads / self.num_k_heads;
        if rep > 1 {
            query = query
                .unsqueeze(3)?
                .expand((b, seq_len, self.num_k_heads, rep, self.head_k_dim))?
                .reshape((b, seq_len, self.num_v_heads, self.head_k_dim))?;
            key = key
                .unsqueeze(3)?
                .expand((b, seq_len, self.num_k_heads, rep, self.head_k_dim))?
                .reshape((b, seq_len, self.num_v_heads, self.head_k_dim))?;
        }

        let beta = sigmoid(&b_proj.to_dtype(DType::F32)?)?;

        let dt_bias = self
            .dt_bias
            .to_dtype(DType::F32)?
            .reshape((1, 1, self.num_v_heads))?;
        let a_log = self
            .a_log
            .to_dtype(DType::F32)?
            .exp()?
            .reshape((1, 1, self.num_v_heads))?;
        let g = softplus(&a_proj.to_dtype(DType::F32)?.broadcast_add(&dt_bias)?)?;
        let g = (g.broadcast_mul(&a_log)? * -1.0)?;

        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;

        let mut state = if let Some(prev) = &self.recurrent_state {
            prev.clone()
        } else {
            Tensor::zeros(
                (b, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                hidden_states.device(),
            )?
        };

        let scale = 1.0 / (self.head_k_dim as f64).sqrt();

        if seq_len == 1 {
            let q_t = query.squeeze(1)?;
            let k_t = key.squeeze(1)?;
            let v_t = value.squeeze(1)?;
            let beta_t = beta.squeeze(1)?;
            let g_t = g.squeeze(1)?;

            let q_t = (l2norm_last_dim(&q_t, 1e-6)? * scale)?;
            let k_t = l2norm_last_dim(&k_t, 1e-6)?;

            let core = if hidden_states.device().is_cuda() {
                let bh = b * self.num_v_heads;
                let mut state_batched = state.reshape((bh, self.head_k_dim, self.head_v_dim))?;

                let q_b = q_t.reshape((bh, self.head_k_dim))?;
                let k_b = k_t.reshape((bh, self.head_k_dim))?;
                let v_b = v_t.reshape((bh, self.head_v_dim))?;
                let beta_b = beta_t.reshape((bh, 1))?;
                let g_decay = g_t.exp()?.reshape((bh, 1, 1))?;

                state_batched = state_batched.broadcast_mul(&g_decay)?;

                let kv_mem = k_b.unsqueeze(1)?.matmul(&state_batched)?.squeeze(1)?;
                let delta = v_b.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_b)?;

                let k_delta = k_b.unsqueeze(2)?.matmul(&delta.unsqueeze(1)?)?;
                state_batched = state_batched.broadcast_add(&k_delta)?;

                let out_b = q_b.unsqueeze(1)?.matmul(&state_batched)?.squeeze(1)?;
                state = state_batched.reshape((b, self.num_v_heads, self.head_k_dim, self.head_v_dim))?;
                out_b
                    .reshape((b, self.num_v_heads, self.head_v_dim))?
                    .unsqueeze(1)?
            } else {
                let g_decay = g_t.exp()?.unsqueeze(2)?.unsqueeze(3)?;
                state = state.broadcast_mul(&g_decay)?;

                let kv_mem = state.broadcast_mul(&k_t.unsqueeze(3)?)?.sum(2)?;
                let delta = v_t
                    .broadcast_sub(&kv_mem)?
                    .broadcast_mul(&beta_t.unsqueeze(2)?)?;

                let k_delta = k_t.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?;
                state = state.broadcast_add(&k_delta)?;

                state.broadcast_mul(&q_t.unsqueeze(3)?)?.sum(2)?.unsqueeze(1)?
            };

            self.recurrent_state = Some(state);

            let core_2d = core.reshape(((), self.head_v_dim))?;
            let z_2d = z.to_dtype(DType::F32)?.reshape(((), self.head_v_dim))?;
            let core_2d = rms_norm_gated(&core_2d, &z_2d, &self.norm_weight, self.rms_norm_eps)?;
            let core = core_2d
                .reshape((b, seq_len, self.num_v_heads, self.head_v_dim))?
                .reshape((b, seq_len, self.value_dim))?
                .to_dtype(hidden_states.dtype())?;

            return self.out_proj.forward(&core);
        }

        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

        if hidden_states.device().is_cuda() {
            let bh = b * self.num_v_heads;
            let q_seq = query
                .reshape((bh, seq_len, self.head_k_dim))?
                .contiguous()?;
            let k_seq = key
                .reshape((bh, seq_len, self.head_k_dim))?
                .contiguous()?;
            let v_seq = value
                .reshape((bh, seq_len, self.head_v_dim))?
                .contiguous()?;
            let beta_seq = beta.reshape((bh, seq_len))?.contiguous()?;
            let g_seq = g.reshape((bh, seq_len))?.contiguous()?;
            let state_in = state
                .reshape((bh, self.head_k_dim, self.head_v_dim))?
                .contiguous()?;

            let (state_out, out_seq) = crate::fused_ops::qwen35_linear_scan_f32(
                &state_in,
                &q_seq,
                &k_seq,
                &v_seq,
                &beta_seq,
                &g_seq,
            )?;

            state = state_out
                .reshape((b, self.num_v_heads, self.head_k_dim, self.head_v_dim))?;
            let core = out_seq.reshape((b, seq_len, self.num_v_heads, self.head_v_dim))?;
            let core_2d = core.reshape(((), self.head_v_dim))?;
            let z_2d = z.to_dtype(DType::F32)?.reshape(((), self.head_v_dim))?;
            let core_2d = rms_norm_gated(&core_2d, &z_2d, &self.norm_weight, self.rms_norm_eps)?;
            let core = core_2d
                .reshape((b, seq_len, self.num_v_heads, self.head_v_dim))?
                .reshape((b, seq_len, self.value_dim))?
                .to_dtype(hidden_states.dtype())?;

            self.recurrent_state = Some(state);
            return self.out_proj.forward(&core);
        } else {
            for t in 0..seq_len {
                let q_t = query.narrow(1, t, 1)?.squeeze(1)?;
                let k_t = key.narrow(1, t, 1)?.squeeze(1)?;
                let v_t = value.narrow(1, t, 1)?.squeeze(1)?;
                let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?;
                let g_t = g.narrow(1, t, 1)?.squeeze(1)?;

                let q_t = (l2norm_last_dim(&q_t, 1e-6)? * scale)?;
                let k_t = l2norm_last_dim(&k_t, 1e-6)?;

                let g_decay = g_t.exp()?.unsqueeze(2)?.unsqueeze(3)?;
                state = state.broadcast_mul(&g_decay)?;

                let kv_mem = state.broadcast_mul(&k_t.unsqueeze(3)?)?.sum(2)?;
                let delta = v_t
                    .broadcast_sub(&kv_mem)?
                    .broadcast_mul(&beta_t.unsqueeze(2)?)?;

                let k_delta = k_t.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?;
                state = state.broadcast_add(&k_delta)?;

                let out_t = state.broadcast_mul(&q_t.unsqueeze(3)?)?.sum(2)?;
                outputs.push(out_t.unsqueeze(1)?);
            }
        }

        let out_refs: Vec<&Tensor> = outputs.iter().collect();
        let core = Tensor::cat(&out_refs, 1)?; // [B, S, Hv, Dv]

        self.recurrent_state = Some(state);

        let core_2d = core.reshape(((), self.head_v_dim))?;
        let z_2d = z.to_dtype(DType::F32)?.reshape(((), self.head_v_dim))?;
        let core_2d = rms_norm_gated(&core_2d, &z_2d, &self.norm_weight, self.rms_norm_eps)?;
        let core = core_2d
            .reshape((b, seq_len, self.num_v_heads, self.head_v_dim))?
            .reshape((b, seq_len, self.value_dim))?
            .to_dtype(hidden_states.dtype())?;

        self.out_proj.forward(&core)
    }

    fn clear_cache(&mut self) {
        self.conv_history = None;
        self.recurrent_state = None;
    }
}

struct Mlp {
    gate_up_proj: Linear,
    intermediate_size: usize,
    down_proj: Linear,
}

impl Mlp {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("up_proj"))?;
        let gate_up_proj = Linear::new(
            Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?,
            None,
        );
        let down_proj = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            intermediate_size: config.intermediate_size,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;

        #[cfg(feature = "cuda")]
        {
            if gate_up.device().is_cuda() {
                let activated = crate::fused_ops::fused_silu_mul(
                    &gate_up.contiguous()?,
                    self.intermediate_size,
                )?;
                return self.down_proj.forward(&activated);
            }
        }

        let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let gate = silu(&gate)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

enum TokenMixer {
    Full(FullAttention),
    Linear(LinearAttention),
}

struct DecoderLayer {
    token_mixer: TokenMixer,
    mlp: Mlp,
    input_layernorm: Qwen35RmsNorm,
    post_attention_layernorm: Qwen35RmsNorm,
}

impl DecoderLayer {
    fn new(config: &Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let token_mixer = match config.layer_type(layer_idx) {
            "linear_attention" => TokenMixer::Linear(LinearAttention::new(config, vb.pp("linear_attn"))?),
            _ => TokenMixer::Full(FullAttention::new(config, vb.pp("self_attn"))?),
        };

        let mlp = Mlp::new(config, vb.pp("mlp"))?;
        let input_layernorm =
            Qwen35RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Qwen35RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            token_mixer,
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
        let hidden_states = match &mut self.token_mixer {
            TokenMixer::Full(attn) => attn.forward(&hidden_states, cos, sin, attention_mask)?,
            TokenMixer::Linear(attn) => attn.forward(&hidden_states)?,
        };
        let hidden_states = (residual + hidden_states)?;

        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual + hidden_states
    }

    fn clear_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::Full(attn) => attn.clear_cache(),
            TokenMixer::Linear(attn) => attn.clear_cache(),
        }
    }
}

pub struct Qwen35Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: Qwen35RmsNorm,
    lm_head: Linear,
    rotary_emb: RotaryEmbedding,
    dtype: DType,
}

impl Qwen35Model {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_model_prefix(config, vb, "model")
    }

    pub fn new_with_model_prefix(
        config: &Config,
        vb: VarBuilder,
        model_prefix: &str,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let model_vb = vb.pp(model_prefix);

        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            model_vb.pp("embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = model_vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(config, i, layers_vb.pp(i))?);
        }

        let norm = Qwen35RmsNorm::new(config.hidden_size, config.rms_norm_eps, model_vb.pp("norm"))?;

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
            dtype,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let total_len = start_pos + seq_len;
        let (cos, sin) = self.rotary_emb.forward(total_len, start_pos, seq_len)?;
        let cos = cos.to_dtype(self.dtype)?;
        let sin = sin.to_dtype(self.dtype)?;

        let attention_mask = if seq_len > 1 {
            let mut mask_data = vec![0f32; seq_len * total_len];
            for i in 0..seq_len {
                for j in 0..total_len {
                    if j <= start_pos + i {
                        mask_data[i * total_len + j] = 1.0;
                    }
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, total_len), input_ids.device())?;
            let mask = mask
                .broadcast_lt(&Tensor::new(0.5f32, input_ids.device())?)?
                .to_dtype(self.dtype)?;
            let mask = (mask * (-1e9f64))?;
            Some(mask.unsqueeze(0)?.unsqueeze(0)?)
        } else {
            None
        };

        let mut hidden_states = hidden_states;
        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head
            .forward(&hidden_states.narrow(1, seq_len - 1, 1)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn model_dtype(&self) -> DType {
        self.dtype
    }

    /// Returns per-layer boolean: true = full_attention, false = linear_attention.
    pub fn layer_is_full_attn(&self) -> Vec<bool> {
        self.layers
            .iter()
            .map(|l| matches!(&l.token_mixer, TokenMixer::Full(_)))
            .collect()
    }

    // ── KV Cache Management ─────────────────────────────────────────────

    /// Total bytes held by the model state caches (full-attn KV + linear-attn states).
    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| match &l.token_mixer {
                TokenMixer::Full(attn) => attn
                    .kv_cache
                    .as_ref()
                    .map(|(k, v)| {
                        let k_bytes = k.elem_count() as u64 * k.dtype().size_in_bytes() as u64;
                        let v_bytes = v.elem_count() as u64 * v.dtype().size_in_bytes() as u64;
                        k_bytes + v_bytes
                    })
                    .unwrap_or(0),
                TokenMixer::Linear(attn) => {
                    let state_bytes = attn
                        .recurrent_state
                        .as_ref()
                        .map(|s| s.elem_count() as u64 * s.dtype().size_in_bytes() as u64)
                        .unwrap_or(0);
                    let conv_bytes = attn
                        .conv_history
                        .as_ref()
                        .map(|s| s.elem_count() as u64 * s.dtype().size_in_bytes() as u64)
                        .unwrap_or(0);
                    state_bytes + conv_bytes
                }
            })
            .sum()
    }

    /// Number of full-attention layers (for KV cache vector sizing).
    pub fn num_full_attn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(&l.token_mixer, TokenMixer::Full(_)))
            .count()
    }

    /// Extract per-layer KV caches from full-attention layers (valid portion only).
    /// Linear-attention layers contribute `None`.
    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers
            .iter()
            .map(|l| match &l.token_mixer {
                TokenMixer::Full(attn) => attn.kv_cache.as_ref().map(|(k, v)| {
                    let len = attn.cache_seq_len;
                    if len > 0 && len < k.dim(2).unwrap_or(0) {
                        (
                            k.narrow(2, 0, len).unwrap_or_else(|_| k.clone()),
                            v.narrow(2, 0, len).unwrap_or_else(|_| v.clone()),
                        )
                    } else {
                        (k.clone(), v.clone())
                    }
                }),
                TokenMixer::Linear(_) => None,
            })
            .collect()
    }

    /// Restore per-layer KV caches for full-attention layers.
    /// Linear-attention layers are skipped.
    pub fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        for (layer, cache) in self.layers.iter_mut().zip(caches.into_iter()) {
            if let TokenMixer::Full(ref mut attn) = layer.token_mixer {
                let seq_len = cache
                    .as_ref()
                    .map(|(k, _)| k.dim(2).unwrap_or(0))
                    .unwrap_or(0);
                attn.kv_cache = cache;
                attn.cache_seq_len = seq_len;
            }
        }
    }

    /// Save the full state of all layers (KV caches + linear-attention recurrent state).
    pub fn save_full_state(&self) -> Vec<LayerState> {
        self.layers
            .iter()
            .map(|l| match &l.token_mixer {
                TokenMixer::Full(attn) => {
                    let kv = attn.kv_cache.as_ref().map(|(k, v)| {
                        let len = attn.cache_seq_len;
                        if len > 0 && len < k.dim(2).unwrap_or(0) {
                            (
                                k.narrow(2, 0, len).unwrap_or_else(|_| k.clone()),
                                v.narrow(2, 0, len).unwrap_or_else(|_| v.clone()),
                            )
                        } else {
                            (k.clone(), v.clone())
                        }
                    });
                    LayerState::FullAttn { kv_cache: kv }
                }
                TokenMixer::Linear(attn) => LayerState::LinearAttn {
                    recurrent_state: attn.recurrent_state.clone(),
                    conv_history: attn.conv_history.clone(),
                },
            })
            .collect()
    }

    /// Restore the full state of all layers.
    pub fn restore_full_state(&mut self, states: Vec<LayerState>) {
        for (layer, state) in self.layers.iter_mut().zip(states.into_iter()) {
            match (&mut layer.token_mixer, state) {
                (TokenMixer::Full(attn), LayerState::FullAttn { kv_cache }) => {
                    let seq_len = kv_cache
                        .as_ref()
                        .map(|(k, _)| k.dim(2).unwrap_or(0))
                        .unwrap_or(0);
                    attn.kv_cache = kv_cache;
                    attn.cache_seq_len = seq_len;
                }
                (TokenMixer::Linear(attn), LayerState::LinearAttn { recurrent_state, conv_history }) => {
                    attn.recurrent_state = recurrent_state;
                    attn.conv_history = conv_history;
                }
                _ => {}
            }
        }
    }

    // ── Batched Decode ──────────────────────────────────────────────────

    /// Pad per-sequence KV caches to the same length and load into full-attention layers.
    /// Linear-attention layers get their recurrent states stacked along the batch dim.
    /// Returns `(kv_lens, max_kv_len)`.
    pub fn setup_batch_decode(
        &mut self,
        seq_states: &[Vec<LayerState>],
        extra_room: usize,
    ) -> Result<(Vec<usize>, usize)> {
        let device = self.embed_tokens.embeddings().device();

        // Determine per-sequence KV lengths from the first full-attention layer.
        let kv_lens: Vec<usize> = seq_states
            .iter()
            .map(|states| {
                states.iter().find_map(|s| match s {
                    LayerState::FullAttn { kv_cache: Some((k, _)) } => Some(k.dim(2).unwrap_or(0)),
                    _ => None,
                }).unwrap_or(0)
            })
            .collect();
        let max_kv_len = kv_lens.iter().copied().max().unwrap_or(0);

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_states: Vec<&LayerState> =
                seq_states.iter().map(|seq| &seq[layer_idx]).collect();

            match &mut layer.token_mixer {
                TokenMixer::Full(attn) => {
                    let mut cache_owned: Vec<Option<(Tensor, Tensor)>> =
                        Vec::with_capacity(layer_states.len());
                    for s in &layer_states {
                        match s {
                            LayerState::FullAttn { kv_cache } => cache_owned.push(kv_cache.clone()),
                            _ => cache_owned.push(None),
                        }
                    }
                    let caches: Vec<&Option<(Tensor, Tensor)>> = cache_owned.iter().collect();

                    let batched = pad_and_stack_kv_caches(
                        &caches,
                        max_kv_len,
                        attn.num_kv_heads,
                        attn.head_dim,
                        device,
                        self.dtype,
                    )?;

                    if let Some((k, v)) = batched {
                        let k = k.contiguous()?;
                        let v = v.contiguous()?;
                        if extra_room > 0 {
                            let (b, h, s, d) = k.dims4()?;
                            let buf_k = Tensor::zeros(
                                (b, h, s + extra_room, d),
                                k.dtype(),
                                k.device(),
                            )?;
                            let buf_v = Tensor::zeros(
                                (b, h, s + extra_room, d),
                                v.dtype(),
                                v.device(),
                            )?;
                            buf_k.slice_set(&k, 2, 0)?;
                            buf_v.slice_set(&v, 2, 0)?;
                            attn.kv_cache = Some((buf_k, buf_v));
                        } else {
                            attn.kv_cache = Some((k, v));
                        }
                        attn.cache_seq_len = max_kv_len;
                    } else {
                        attn.kv_cache = None;
                        attn.cache_seq_len = 0;
                    }
                }
                TokenMixer::Linear(lin_attn) => {
                    // Stack recurrent states along batch dimension.
                    let states: Vec<&Tensor> = layer_states
                        .iter()
                        .filter_map(|s| match s {
                            LayerState::LinearAttn { recurrent_state: Some(r), .. } => Some(r),
                            _ => None,
                        })
                        .collect();
                    if states.len() == seq_states.len() {
                        lin_attn.recurrent_state = Some(Tensor::cat(&states, 0)?);
                    } else {
                        lin_attn.recurrent_state = None;
                    }
                    // Conv history: for decode (seq_len=1) we don't need it, clear it.
                    lin_attn.conv_history = None;
                }
            }
        }

        Ok((kv_lens, max_kv_len))
    }

    /// Run one batched decode step.
    pub fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        _batch_kv_info: Option<(&[usize], usize)>,
    ) -> Result<Tensor> {
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let max_pos = positions.iter().copied().max().unwrap_or(0) + 1;
        let device = input_ids.device();
        let (cos, sin) = self.rotary_emb.forward(max_pos, 0, max_pos)?;
        let pos_ids: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let pos_tensor = Tensor::new(pos_ids.as_slice(), device)?;
        let cos = cos
            .index_select(&pos_tensor, 0)?
            .to_dtype(self.dtype)?
            .unsqueeze(1)?;
        let sin = sin
            .index_select(&pos_tensor, 0)?
            .to_dtype(self.dtype)?
            .unsqueeze(1)?;

        let mut hidden_states = hidden_states;
        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask)?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states)
    }

    /// Extract per-sequence states from batched model state.
    pub fn extract_batch_state(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> Result<Vec<Vec<LayerState>>> {
        let n_seqs = kv_lens.len();
        let num_layers = self.layers.len();
        let mut result: Vec<Vec<LayerState>> = (0..n_seqs)
            .map(|_| Vec::with_capacity(num_layers))
            .collect();

        for layer in self.layers.iter_mut() {
            match &mut layer.token_mixer {
                TokenMixer::Full(attn) => {
                    if let Some((ref full_k, ref full_v)) = attn.kv_cache {
                        for i in 0..n_seqs {
                            let row_k = full_k.narrow(0, i, 1)?;
                            let row_v = full_v.narrow(0, i, 1)?;
                            let total = kv_lens[i] + rounds_done;
                            let offset = original_max_kv - kv_lens[i];
                            let clean = Some((
                                row_k.narrow(2, offset, total)?.contiguous()?,
                                row_v.narrow(2, offset, total)?.contiguous()?,
                            ));
                            result[i].push(LayerState::FullAttn { kv_cache: clean });
                        }
                    } else {
                        for i in 0..n_seqs {
                            result[i].push(LayerState::FullAttn { kv_cache: None });
                        }
                    }
                    attn.kv_cache = None;
                    attn.cache_seq_len = 0;
                }
                TokenMixer::Linear(lin_attn) => {
                    if let Some(ref state) = lin_attn.recurrent_state {
                        for i in 0..n_seqs {
                            let row = state.narrow(0, i, 1)?.contiguous()?;
                            result[i].push(LayerState::LinearAttn {
                                recurrent_state: Some(row),
                                conv_history: None,
                            });
                        }
                    } else {
                        for i in 0..n_seqs {
                            result[i].push(LayerState::LinearAttn {
                                recurrent_state: None,
                                conv_history: None,
                            });
                        }
                    }
                    lin_attn.recurrent_state = None;
                    lin_attn.conv_history = None;
                }
            }
        }

        Ok(result)
    }

    /// Access config values needed by backend.
    pub fn num_kv_heads(&self) -> usize {
        self.layers
            .iter()
            .find_map(|l| match &l.token_mixer {
                TokenMixer::Full(attn) => Some(attn.num_kv_heads),
                _ => None,
            })
            .unwrap_or(0)
    }

    pub fn head_dim(&self) -> usize {
        self.layers
            .iter()
            .find_map(|l| match &l.token_mixer {
                TokenMixer::Full(attn) => Some(attn.head_dim),
                _ => None,
            })
            .unwrap_or(0)
    }
}

// ── Per-layer State ──────────────────────────────────────────────────────

/// Saved state for a single decoder layer.
pub enum LayerState {
    FullAttn {
        kv_cache: Option<(Tensor, Tensor)>,
    },
    LinearAttn {
        recurrent_state: Option<Tensor>,
        conv_history: Option<Tensor>,
    },
}

// ── Utilities ────────────────────────────────────────────────────────────

/// Build attention mask for batched decode with padding-aware masking.
pub fn build_batch_decode_mask(
    kv_lens: &[usize],
    original_max_kv: usize,
    total_width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Option<Tensor>> {
    if kv_lens.iter().all(|&l| l == original_max_kv) {
        return Ok(None);
    }
    let n = kv_lens.len();
    let mut mask_data = vec![0f32; n * total_width];
    for i in 0..n {
        let pad_end = (original_max_kv - kv_lens[i]).min(total_width);
        for j in 0..pad_end {
            mask_data[i * total_width + j] = -1e9;
        }
    }
    let mask = Tensor::from_vec(mask_data, (n, total_width), device)?.to_dtype(dtype)?;
    Ok(Some(mask.unsqueeze(1)?.unsqueeze(1)?))
}

/// Pad per-sequence KV caches to `max_len` and stack (right-aligned).
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
                    padded_ks.push(Tensor::cat(&[&pad, k.as_ref()], 2)?);
                    padded_vs.push(Tensor::cat(&[&pad, v.as_ref()], 2)?);
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

    let stacked_k = Tensor::cat(&padded_ks, 0)?.contiguous()?;
    let stacked_v = Tensor::cat(&padded_vs, 0)?.contiguous()?;
    Ok(Some((stacked_k, stacked_v)))
}
