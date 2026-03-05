#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::path::{Path, PathBuf};

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::generation::based::ModelForCausalLM;
use crate::generation::GenerationConfig;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

use super::modeling::{Config, LayerState, Qwen35Model};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    Auto,
    Safetensors,
    Gguf,
}

fn config_path_for(model_path: &str) -> Option<PathBuf> {
    let p = Path::new(model_path);
    if p.is_file() {
        p.parent().map(|dir| dir.join("config.json"))
    } else {
        Some(p.join("config.json"))
    }
}

fn detect_model_prefix_from_index(model_path: &str) -> Option<String> {
    let index_path = Path::new(model_path).join("model.safetensors.index.json");
    let data = std::fs::read(index_path).ok()?;
    let root: Value = serde_json::from_slice(&data).ok()?;
    let weight_map = root.get("weight_map")?.as_object()?;

    if weight_map.contains_key("model.language_model.embed_tokens.weight") {
        return Some("model.language_model".to_string());
    }
    if weight_map.contains_key("model.embed_tokens.weight") {
        return Some("model".to_string());
    }
    if weight_map.contains_key("language_model.embed_tokens.weight") {
        return Some("language_model".to_string());
    }

    None
}

pub struct Model {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    inner: Qwen35Model,
}

impl Model {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        Self::new_with_format(model_path, device, dtype, ModelFormat::Auto)
    }

    pub fn new_with_format(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: ModelFormat,
    ) -> Result<Self> {
        let format = match format {
            ModelFormat::Auto => {
                let p = Path::new(model_path);
                if p.is_file() && p.extension().map(|e| e == "gguf").unwrap_or(false) {
                    ModelFormat::Gguf
                } else {
                    ModelFormat::Safetensors
                }
            }
            other => other,
        };

        match format {
            ModelFormat::Safetensors | ModelFormat::Auto => Self::from_pretrained(model_path, device, dtype),
            ModelFormat::Gguf => anyhow::bail!(
                "Qwen3.5 GGUF loading is not implemented yet; please use safetensors checkpoints"
            ),
        }
    }

    fn load_text_config(model_path: &str) -> Result<Config> {
        let config_path = config_path_for(model_path)
            .ok_or_else(|| anyhow::anyhow!("Invalid model path: {model_path}"))?;
        let data = std::fs::read(&config_path)?;
        let root: Value = serde_json::from_slice(&data)?;

        let mut config: Config = if let Some(text_cfg) = root.get("text_config") {
            serde_json::from_value(text_cfg.clone())?
        } else {
            serde_json::from_value(root.clone())?
        };

        if let Some(tie) = root.get("tie_word_embeddings").and_then(|v| v.as_bool()) {
            config.tie_word_embeddings = tie;
        }

        config.post_init()?;
        Ok(config)
    }

    fn from_pretrained(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let tokenizer_path = Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let config = Self::load_text_config(model_path)?;

        let mut prefixes = Vec::new();
        if let Some(detected) = detect_model_prefix_from_index(model_path) {
            prefixes.push(detected);
        }
        for fallback in ["model", "model.language_model", "language_model"] {
            if !prefixes.iter().any(|p| p == fallback) {
                prefixes.push(fallback.to_string());
            }
        }

        let mut last_err: Option<anyhow::Error> = None;
        let mut inner = None;
        for prefix in prefixes.iter() {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;
            match Qwen35Model::new_with_model_prefix(&config, vb, prefix.as_str()) {
                Ok(m) => {
                    inner = Some(m);
                    break;
                }
                Err(err) => {
                    last_err = Some(anyhow::anyhow!(
                        "prefix '{}' failed: {}",
                        prefix,
                        err
                    ));
                }
            }
        }

        let inner = match inner {
            Some(m) => m,
            None => {
                let detail = last_err
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                anyhow::bail!(
                    "Failed to load Qwen3.5 weights with tried prefixes {:?}: {}",
                    prefixes,
                    detail
                );
            }
        };

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            inner,
        })
    }

    pub fn prepare_inputs(&self, inputs: &str) -> Result<Vec<u32>> {
        let input_ids = self
            .tokenizer
            .tokenizer
            .encode(inputs, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        Ok(input_ids)
    }

    pub fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> candle_core::Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        self.inner.forward(&input, start_pos)
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }

    pub fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    pub fn layer_is_full_attn(&self) -> Vec<bool> {
        self.inner.layer_is_full_attn()
    }

    pub fn warmup(&mut self) {
        if let Err(e) = self.generate(
            &[45, 546, 456],
            &GenerationConfig::with_max_tokens(5),
            None,
        ) {
            eprintln!("warmup failed (non-fatal): {e}");
        }
        self.clear_kv_cache();
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer.tokenizer
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    // ── KV cache management (for continuous-batching engine) ────────────

    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.inner.get_kv_caches()
    }

    pub fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.inner.set_kv_caches(caches);
    }

    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.inner.active_kv_cache_bytes()
    }

    pub fn save_full_state(&self) -> Vec<LayerState> {
        self.inner.save_full_state()
    }

    pub fn restore_full_state(&mut self, states: Vec<LayerState>) {
        self.inner.restore_full_state(states);
    }

    // ── Batched decode (GPU-efficient concurrent serving) ───────────────

    pub fn setup_batch_decode(
        &mut self,
        seq_states: &[Vec<LayerState>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.inner.setup_batch_decode(seq_states, extra_room)
    }

    pub fn step_batch_decode(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        let n = positions.len();
        let input = Tensor::new(tokens, &self.device)?.reshape((n, 1))?;
        self.inner
            .step_batch_decode(&input, positions, attention_mask, batch_kv_info)
    }

    pub fn step_batch_decode_with_input_ids(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        self.inner
            .step_batch_decode(input_ids, positions, attention_mask, batch_kv_info)
    }

    pub fn extract_batch_state(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<LayerState>>> {
        self.inner
            .extract_batch_state(kv_lens, original_max_kv, rounds_done)
    }
}

impl ModelForCausalLM for Model {
    fn device(&self) -> &Device {
        &self.device
    }

    fn generate(
        &mut self,
        input_ids: &[u32],
        config: &GenerationConfig,
        streamer: Option<&mut dyn crate::generation::streamer::TokenStreamer>,
    ) -> Result<Vec<u32>> {
        let mut streamer = streamer;
        self.tokenizer.clear();
        self.clear_kv_cache();

        let mut logits_processor = LogitsProcessor::new(1024, config.temperature, config.top_p);

        let mut tokens = input_ids.to_vec();
        let mut generated_tokens = 0usize;
        let eos_token: Option<u32> = config
            .eos_token_id
            .or_else(|| self.tokenizer.get_token("<|im_end|>"))
            .or_else(|| self.tokenizer.get_token("<|endoftext|>"));
        let mut streamer_finalized = false;

        for index in 0..config.max_new_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            let logits = self.inner.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if config.repetition_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    config.repetition_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;

            if eos_token == Some(next_token) {
                if let Some(ref mut s) = streamer {
                    s.finalize()?;
                }
                streamer_finalized = true;
                break;
            }

            if let Some(ref mut s) = streamer {
                s.append(next_token)?;
            }
        }

        if let Some(ref mut s) = streamer {
            if !streamer_finalized {
                s.finalize()?;
            }
        }

        if config.report_speed {
            eprintln!("qwen35 generated {} tokens", generated_tokens);
        }

        Ok(tokens)
    }
}
