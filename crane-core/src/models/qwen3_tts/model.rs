//! High-level Qwen3-TTS model wrapper.
//!
//! Handles model loading, text tokenization, speech generation, and
//! code → waveform decoding via ONNX speech tokenizer.

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hound::{SampleFormat, WavSpec, WavWriter};
use tokenizers::Tokenizer;

use super::modeling::{Qwen3TTSConfig, Qwen3TTSModel};
use crate::utils::utils;

// ── Speech Tokenizer (ONNX decoder: codes → waveform) ──────────────────

/// ONNX-based speech tokenizer decoder. Converts 16-codebook codec tokens
/// into raw audio waveform.
pub struct SpeechTokenizerDecoder {
    model: candle_onnx::onnx::ModelProto,
    pub sample_rate: u32,
}

impl SpeechTokenizerDecoder {
    /// Load from a pre-exported ONNX file.
    pub fn new(onnx_path: &str, sample_rate: Option<u32>) -> Result<Self> {
        if !std::path::Path::new(onnx_path).exists() {
            anyhow::bail!(
                "Speech tokenizer ONNX not found at {}. \
                 Export it first: python scripts/export_qwen_tts_tokenizer_onnx.py <model_dir> {}",
                onnx_path,
                onnx_path,
            );
        }
        let model = candle_onnx::read_file(onnx_path)?;
        Ok(Self {
            model,
            sample_rate: sample_rate.unwrap_or(24000),
        })
    }

    /// Decode `[batch, num_quantizers, seq_len]` codes → `[batch, 1, samples]`.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let inputs =
            std::collections::HashMap::from_iter([("codes".to_string(), codes.clone())]);
        let out = candle_onnx::simple_eval(&self.model, inputs)?;
        let out_names = &self.model.graph.as_ref().unwrap().output;
        let audio = out.get(&out_names[0].name).unwrap().clone();
        Ok(audio)
    }

    /// Convenience: decode and write a WAV file.
    pub fn decode_to_wav(&self, codes: &Tensor, filename: &str) -> Result<String> {
        let audio = self.decode(codes)?;
        Self::save_wav(&audio, filename, self.sample_rate)
    }

    pub fn save_wav(audio_values: &Tensor, filename: &str, sample_rate: u32) -> Result<String> {
        let audio = audio_values.to_dtype(DType::F32)?.flatten_all()?;
        let scaled = audio.affine(32767.0, 0.0)?.clamp(-32768.0, 32767.0)?.round()?;
        let audio_i64 = scaled.to_dtype(DType::I64)?;
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(filename, spec)?;
        for sample in audio_i64.to_vec1::<i64>()? {
            writer.write_sample(sample.clamp(i16::MIN as i64, i16::MAX as i64) as i16)?;
        }
        writer.finalize()?;
        Ok(filename.to_string())
    }
}

// ── Qwen3-TTS Model ────────────────────────────────────────────────────

pub struct Model {
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub dtype: DType,
    pub config: Qwen3TTSConfig,
    inner: Qwen3TTSModel,
    speech_decoder: Option<SpeechTokenizerDecoder>,
}

impl Model {
    /// Load from a HuggingFace-style directory.
    ///
    /// Expects:
    ///   - `config.json` (Qwen3TTSConfig)
    ///   - `tokenizer.json`
    ///   - `model.safetensors` / `model-*.safetensors` (talker weights)
    ///   - `speech_tokenizer/speech_tokenizer_decoder.onnx` (optional, for waveform decode)
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model_dir = std::path::Path::new(model_path);

        // Config
        let config_data = std::fs::read(model_dir.join("config.json"))?;
        let config: Qwen3TTSConfig = serde_json::from_slice(&config_data)?;

        // Tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("tokenizer.json not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        // Safetensors
        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;
        let inner = Qwen3TTSModel::new(&config, vb)?;

        // Speech tokenizer decoder (optional ONNX)
        let onnx_path = model_dir.join("speech_tokenizer").join("speech_tokenizer_decoder.onnx");
        let speech_decoder = if onnx_path.exists() {
            Some(SpeechTokenizerDecoder::new(onnx_path.to_str().unwrap(), Some(24000))?)
        } else {
            eprintln!(
                "Warning: speech tokenizer ONNX not found at {}. \
                 Code-to-waveform decoding will not be available. \
                 Export it with: python scripts/export_qwen_tts_tokenizer_onnx.py",
                onnx_path.display()
            );
            None
        };

        Ok(Self {
            tokenizer,
            device: device.clone(),
            dtype: *dtype,
            config,
            inner,
            speech_decoder,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }

    /// Tokenize text input for TTS.
    ///
    /// Wraps with Qwen chat-style tags:
    /// `<|im_start|>system\nYou are Qwen...<|im_end|>\n<|im_start|>user\n<tts>text</tts><|im_end|>\n<|im_start|>assistant\n`
    pub fn prepare_tts_input(
        &self,
        text: &str,
        system_prompt: Option<&str>,
    ) -> Result<Vec<u32>> {
        let system = system_prompt.unwrap_or(
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group. \
             You can perceive speech, text, and other multimodal information, and you can \
             output speech and text information.",
        );
        let chat_text = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n<tts>{text}</tts><|im_end|>\n\
             <|im_start|>assistant\n"
        );
        let encoding = self.tokenizer.encode(chat_text.as_str(), false).map_err(E::msg)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Generate speech: text → codec codes → waveform tensor.
    ///
    /// Returns `(audio_tensor, sample_rate)`.  
    /// `audio_tensor` shape: `[1, 1, samples]` (f32).
    pub fn generate_speech(
        &mut self,
        text: &str,
        language: &str,
        speaker: Option<&str>,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
        system_prompt: Option<&str>,
    ) -> Result<(Tensor, u32)> {
        let input_ids = self.prepare_tts_input(text, system_prompt)?;

        let codes = self.inner.generate_speech_codes(
            &input_ids,
            language,
            speaker,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
        )?;

        if codes.is_empty() {
            anyhow::bail!("No speech codes generated");
        }

        let speech_decoder = self
            .speech_decoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Speech tokenizer ONNX not loaded; cannot decode to audio"))?;

        // Convert codes: Vec<Vec<u32>> of shape [timesteps, num_code_groups]
        // → Tensor [1, num_code_groups, timesteps]
        let num_steps = codes.len();
        let num_groups = codes[0].len();
        let flat: Vec<i64> = codes
            .iter()
            .flat_map(|frame| frame.iter().map(|&c| c as i64))
            .collect();
        let codes_tensor = Tensor::new(flat.as_slice(), &self.device)?
            .reshape((num_steps, num_groups))?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let audio = speech_decoder.decode(&codes_tensor)?;
        Ok((audio, speech_decoder.sample_rate))
    }

    /// Generate speech and write directly to a WAV file.
    pub fn generate_speech_to_file(
        &mut self,
        text: &str,
        language: &str,
        speaker: Option<&str>,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
        output_path: &str,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        let (audio, sr) = self.generate_speech(
            text,
            language,
            speaker,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
            system_prompt,
        )?;
        SpeechTokenizerDecoder::save_wav(&audio, output_path, sr)
    }

    /// Generate only the codec codes (no waveform decode).
    /// Useful when you have an external vocoder.
    pub fn generate_codes(
        &mut self,
        text: &str,
        language: &str,
        speaker: Option<&str>,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
        system_prompt: Option<&str>,
    ) -> Result<Vec<Vec<u32>>> {
        let input_ids = self.prepare_tts_input(text, system_prompt)?;
        let codes = self.inner.generate_speech_codes(
            &input_ids,
            language,
            speaker,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
        )?;
        Ok(codes)
    }

    /// Convert pre-generated codes to raw audio bytes (PCM 16-bit LE).
    pub fn codes_to_pcm(&self, codes: &[Vec<u32>]) -> Result<Vec<u8>> {
        let speech_decoder = self
            .speech_decoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Speech tokenizer ONNX not loaded"))?;

        let num_steps = codes.len();
        let num_groups = codes[0].len();
        let flat: Vec<i64> = codes
            .iter()
            .flat_map(|frame| frame.iter().map(|&c| c as i64))
            .collect();
        let codes_tensor = Tensor::new(flat.as_slice(), &self.device)?
            .reshape((num_steps, num_groups))?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let audio = speech_decoder.decode(&codes_tensor)?;
        let audio = audio.to_dtype(DType::F32)?.flatten_all()?;

        // Scale to i16 PCM
        let scaled = audio.affine(32767.0, 0.0)?.clamp(-32768.0, 32767.0)?.round()?;
        let samples = scaled.to_dtype(DType::I64)?.to_vec1::<i64>()?;

        let mut pcm_bytes = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            let s16 = s.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
            pcm_bytes.extend_from_slice(&s16.to_le_bytes());
        }
        Ok(pcm_bytes)
    }

    /// Get the sample rate of the speech tokenizer decoder.
    pub fn sample_rate(&self) -> u32 {
        self.speech_decoder
            .as_ref()
            .map(|d| d.sample_rate)
            .unwrap_or(24000)
    }
}
