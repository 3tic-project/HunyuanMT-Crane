//! Model factory for automatic model type detection and backend creation.
//!
//! Supports auto-detection from `config.json`'s `model_type` / `architectures`
//! fields, or explicit model type specification via CLI.

use anyhow::Result;
use candle_core::{DType, Device};
use serde::Deserialize;
use std::path::Path;

use super::backend::{HunyuanBackend, ModelBackend, Qwen25Backend, Qwen3Backend};
use crate::chat_template::{AutoChatTemplate, ChatTemplateProcessor, HunyuanChatTemplate};

// ─────────────────────────────────────────────────────────────
//  Enums
// ─────────────────────────────────────────────────────────────

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Auto,
    HunyuanDense,
    Qwen25,
    Qwen3,
}

impl ModelType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "hunyuan" | "hunyuan_dense" | "hunyuandense" => Self::HunyuanDense,
            "qwen25" | "qwen2.5" | "qwen2" => Self::Qwen25,
            "qwen3" => Self::Qwen3,
            _ => Self::Auto,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::HunyuanDense => "hunyuan",
            Self::Qwen25 => "qwen25",
            Self::Qwen3 => "qwen3",
        }
    }
}

/// Model weight format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    Auto,
    Safetensors,
    Gguf,
}

impl ModelFormat {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "safetensors" => Self::Safetensors,
            "gguf" => Self::Gguf,
            _ => Self::Auto,
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Detection
// ─────────────────────────────────────────────────────────────

/// Minimal subset of HuggingFace `config.json` for architecture detection.
#[derive(Deserialize, Default)]
struct HfConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
}

/// Auto-detect the model type from `config.json` in the model directory.
pub fn detect_model_type(model_path: &str) -> ModelType {
    let path = Path::new(model_path);

    // Locate config.json (same dir for dir paths; parent dir for GGUF files).
    let config_path = if path.is_file() {
        path.parent().map(|p| p.join("config.json"))
    } else {
        Some(path.join("config.json"))
    };

    if let Some(config_path) = config_path {
        if let Ok(data) = std::fs::read(&config_path) {
            if let Ok(config) = serde_json::from_slice::<HfConfig>(&data) {
                // 1. Check `model_type` field
                if let Some(ref mt) = config.model_type {
                    match mt.to_lowercase().as_str() {
                        "qwen2" | "qwen2.5" => return ModelType::Qwen25,
                        "qwen3" => return ModelType::Qwen3,
                        m if m.contains("hunyuan") => return ModelType::HunyuanDense,
                        _ => {}
                    }
                }

                // 2. Check `architectures` field
                if let Some(ref archs) = config.architectures {
                    for arch in archs {
                        let a = arch.to_lowercase();
                        if a.contains("hunyuan") {
                            return ModelType::HunyuanDense;
                        }
                        if a.contains("qwen3") {
                            return ModelType::Qwen3;
                        }
                        if a.contains("qwen2") {
                            return ModelType::Qwen25;
                        }
                    }
                }
            }
        }
    }

    // 3. Heuristic: check the model path name
    let path_lower = model_path.to_lowercase();
    if path_lower.contains("hunyuan") {
        ModelType::HunyuanDense
    } else if path_lower.contains("qwen3") {
        ModelType::Qwen3
    } else if path_lower.contains("qwen2") || path_lower.contains("qwen25") {
        ModelType::Qwen25
    } else {
        tracing::warn!(
            "Could not auto-detect model type from '{model_path}', defaulting to Qwen25"
        );
        ModelType::Qwen25
    }
}

// ─────────────────────────────────────────────────────────────
//  Factory
// ─────────────────────────────────────────────────────────────

/// Resolve `ModelType::Auto` to a concrete type.
fn resolve(model_type: ModelType, model_path: &str) -> ModelType {
    if model_type == ModelType::Auto {
        detect_model_type(model_path)
    } else {
        model_type
    }
}

/// Create a model backend.
pub fn create_backend(
    model_type: ModelType,
    model_path: &str,
    device: &Device,
    dtype: &DType,
    format: ModelFormat,
) -> Result<Box<dyn ModelBackend>> {
    let model_type = resolve(model_type, model_path);
    tracing::info!("Creating backend: {:?}", model_type);

    match model_type {
        ModelType::HunyuanDense => {
            let hy_fmt = match format {
                ModelFormat::Safetensors => crane_core::models::hunyuan_dense::ModelFormat::Safetensors,
                ModelFormat::Gguf => crane_core::models::hunyuan_dense::ModelFormat::Gguf,
                ModelFormat::Auto => crane_core::models::hunyuan_dense::ModelFormat::Auto,
            };
            Ok(Box::new(HunyuanBackend::new(model_path, device, dtype, hy_fmt)?))
        }
        ModelType::Qwen25 => Ok(Box::new(Qwen25Backend::new(model_path, device, dtype)?)),
        ModelType::Qwen3 => Ok(Box::new(Qwen3Backend::new(model_path, device, dtype)?)),
        ModelType::Auto => unreachable!(),
    }
}

/// Create a chat template processor for the given model.
pub fn create_chat_template(
    model_type: ModelType,
    model_path: &str,
) -> Box<dyn ChatTemplateProcessor> {
    let model_type = resolve(model_type, model_path);

    match model_type {
        ModelType::HunyuanDense => {
            // Prefer jinja template from tokenizer_config.json if available.
            match AutoChatTemplate::new(model_path) {
                Ok(t) => Box::new(t),
                Err(_) => Box::new(HunyuanChatTemplate),
            }
        }
        _ => match AutoChatTemplate::new(model_path) {
            Ok(t) => Box::new(t),
            Err(e) => {
                tracing::warn!("Failed to load chat template: {e}; using Hunyuan fallback");
                Box::new(HunyuanChatTemplate)
            }
        },
    }
}
