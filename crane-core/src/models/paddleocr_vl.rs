use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::paddleocr_vl::{Config, PaddleOCRVLModel};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use std::path::Path;

pub struct PaddleOcrVL {
    model: PaddleOCRVLModel,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    config: Config,
    eos_token_id: u32,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    video_token_id: Option<u32>, // 只有 video 模式才會用到
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OcrTask {
    Ocr,
    Table,
    Formula,
    Chart,
    // Video 模式目前建議單獨處理，這裡先不包含在 enum 裡
}

impl OcrTask {
    pub fn prompt(&self) -> &'static str {
        match self {
            OcrTask::Ocr => "OCR:",
            OcrTask::Table => "Table Recognition:",
            OcrTask::Formula => "Formula Recognition:",
            OcrTask::Chart => "Chart Recognition:",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OcrResult {
    pub text: String,
    pub tokens_generated: usize,
    pub duration_secs: f32,
}

impl PaddleOcrVL {
    /// 從 HuggingFace repo 加載模型（推薦方式）
    pub fn from_pretrained(
        model_id: &str,
        revision: Option<&str>,
        cpu: bool,
        bf16: bool,
    ) -> Result<Self> {
        let device = if cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        let dtype = if bf16 && device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main").to_string(),
        ));

        // config
        let config_path = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

        // tokenizer
        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        // weights
        let model_file = match repo.get("model.safetensors") {
            Ok(p) => p,
            Err(_) => repo.get("pytorch_model.bin")?,
        };

        let vb = if model_file.extension().map_or(false, |e| e == "bin") {
            VarBuilder::from_pth(&model_file, dtype, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &device)? }
        };

        let model = PaddleOCRVLModel::new(&config, vb)?;

        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2);

        Ok(Self {
            model,
            tokenizer,
            device,
            dtype,
            config,
            eos_token_id,
            image_token_id: config.image_token_id,
            vision_start_token_id: config.vision_start_token_id,
            vision_end_token_id: config.vision_end_token_id,
            video_token_id: config.video_token_id,
        })
    }

    /// 從本地目錄載入（已下載好的模型）
    pub fn from_local(path: impl AsRef<Path>, cpu: bool, bf16: bool) -> Result<Self> {
        let device = if cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };
        let dtype = if bf16 && device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let path = path.as_ref();

        let config_path = path.join("config.json");
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).map_err(E::msg)?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&path.join("model.safetensors")], dtype, &device)?
        };

        let model = PaddleOCRVLModel::new(&config, vb)?;

        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
            .unwrap_or(2);

        Ok(Self {
            model,
            tokenizer,
            device,
            dtype,
            config,
            eos_token_id,
            image_token_id: config.image_token_id,
            vision_start_token_id: config.vision_start_token_id,
            vision_end_token_id: config.vision_end_token_id,
            video_token_id: config.video_token_id,
        })
    }

    pub fn recognize(
        &mut self,
        image_path: impl AsRef<Path>,
        task: OcrTask,
        max_new_tokens: usize,
    ) -> Result<OcrResult> {
        use std::time::Instant;

        let start = Instant::now();

        let (pixel_values, grid_thw) = crate::load_image(
            // 假設你把 load_image 抽成獨立函數
            image_path.as_ref().to_str().unwrap(),
            &self.device,
            self.dtype,
        )?;

        let grid_vec: Vec<Vec<u32>> = grid_thw.to_vec2()?;
        let g = &grid_vec[0];
        let spatial_merge = self.config.vision_config.spatial_merge_size as usize;
        let num_tokens = (g[1] as usize / spatial_merge) * (g[2] as usize / spatial_merge);

        let input_ids = crate::build_input_tokens(
            // 同樣假設已抽出的 helper
            &self.tokenizer,
            task,
            num_tokens,
            self.image_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
            &self.device,
        )?;

        self.model.clear_kv_cache();

        let generated = self.model.generate(
            &input_ids,
            &pixel_values,
            &grid_thw,
            max_new_tokens,
            self.eos_token_id,
        )?;

        let output_tokens: Vec<u32> = generated
            .into_iter()
            .take_while(|&t| t != self.eos_token_id)
            .collect();

        let text = self
            .tokenizer
            .decode(&output_tokens, true)?
            .trim()
            .to_string();

        let duration = start.elapsed().as_secs_f32();

        Ok(OcrResult {
            text,
            tokens_generated: output_tokens.len(),
            duration_secs: duration,
        })
    }

    /// 簡單的 streaming 版本（一行一行印）
    pub fn recognize_and_print(
        &mut self,
        image_path: impl AsRef<Path>,
        task: OcrTask,
        max_new_tokens: usize,
    ) -> Result<()> {
        let result = self.recognize(image_path, task, max_new_tokens)?;
        println!("\n{}", "=".repeat(60));
        println!("Task: {:?}", task);
        println!("{}", result.text);
        println!(
            "{} tokens in {:.2}s ({:.1} tok/s)",
            result.tokens_generated,
            result.duration_secs,
            result.tokens_generated as f32 / result.duration_secs.max(0.01)
        );
        println!("{}\n", "=".repeat(60));
        Ok(())
    }

    // 如果之後想要支援 batch / video，可以再擴充
    // pub fn recognize_batch(...)
    // pub fn recognize_video(...)
}

impl std::fmt::Debug for PaddleOcrVL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PaddleOcrVL")
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .field(
                "vision_layers",
                &self.config.vision_config.num_hidden_layers,
            )
            .field("text_layers", &self.config.num_hidden_layers)
            .finish()
    }
}
