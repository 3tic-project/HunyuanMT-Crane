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
    let var = core.sqr()?.mean(1)?;
    let denom = (var + eps)?.sqrt()?.unsqueeze(1)?;
    let normed = core.broadcast_div(&denom)?;

    let w = if weight.dims().len() == 1 {
        weight.reshape((1, d))?
    } else {
        weight.clone()
    };

    let normed = normed.broadcast_mul(&w)?;
    let gated = normed.broadcast_mul(&silu(gate)?)?;
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
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
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

        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(Qwen35RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?),
                Some(Qwen35RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
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
        })
    }

    fn update_kv_cache(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        if let Some((prev_k, prev_v)) = &self.kv_cache {
            let k = Tensor::cat(&[prev_k, &k], 2)?;
            let v = Tensor::cat(&[prev_v, &v], 2)?;
            self.kv_cache = Some((k.clone(), v.clone()));
            Ok((k, v))
        } else {
            self.kv_cache = Some((k.clone(), v.clone()));
            Ok((k, v))
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

        let qg = self.q_proj.forward(hidden_states)?;
        let q = qg.narrow(D::Minus1, 0, self.q_dim)?;
        let gate = qg.narrow(D::Minus1, self.q_dim, self.q_dim)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

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

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
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

        let gate = sigmoid(&gate)?;
        let attn_output = attn_output.broadcast_mul(&gate)?;

        self.o_proj.forward(&attn_output)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

struct LinearAttention {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
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
        let in_proj_b = linear_no_bias(config.hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(config.hidden_size, num_v_heads, vb.pp("in_proj_a"))?;

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
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
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
            rms_norm_eps: config.rms_norm_eps,
            conv_history: None,
            recurrent_state: None,
        })
    }

    fn causal_depthwise_conv(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        // mixed_qkv: [B, S, conv_dim]
        let (b, s, conv_dim) = mixed_qkv.dims3()?;
        let mixed_qkv = mixed_qkv.transpose(1, 2)?; // [B, conv_dim, S]

        let history = if let Some(prev) = &self.conv_history {
            Tensor::cat(&[prev, &mixed_qkv], 2)?
        } else {
            mixed_qkv.clone()
        };

        let total_len = history.dim(2)?;
        self.conv_history = Some(history.clone());

        let conv_full = self.conv1d.forward(&history)?; // [B, conv_dim, total_len + k - 1]
        let conv_causal = conv_full.narrow(2, 0, total_len)?;
        let conv_cur = conv_causal.narrow(2, total_len.saturating_sub(s), s)?;
        let conv_cur = silu(&conv_cur)?;

        conv_cur.transpose(1, 2)?.reshape((b, s, conv_dim))
    }

    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;

        let mixed_qkv = self.in_proj_qkv.forward(hidden_states)?;
        let mixed_qkv = self.causal_depthwise_conv(&mixed_qkv)?;

        let z = self
            .in_proj_z
            .forward(hidden_states)?
            .reshape((b, seq_len, self.num_v_heads, self.head_v_dim))?;
        let b_proj = self.in_proj_b.forward(hidden_states)?;
        let a_proj = self.in_proj_a.forward(hidden_states)?;

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

        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);
        let scale = 1.0 / (self.head_k_dim as f64).sqrt();

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
        let gate = silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
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
        let dtype = vb.dtype();
        let model_vb = vb.pp("model");

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
}
