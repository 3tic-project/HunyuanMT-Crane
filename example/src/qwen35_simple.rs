//! Simple Qwen3.5 Chat Example
//!
//! This example mirrors `hunyuan_simple.rs` and shows a minimal
//! Qwen3.5 chat flow using the Crane SDK.

use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::{GenerationConfig, LlmModelType};
use crane::prelude::*;

fn main() -> CraneResult<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../Qwen3.5-0.8B".to_string());

    let config = ChatConfig {
        common: CommonConfig {
            model_path: model_path.clone(),
            model_type: LlmModelType::Qwen35,
            device: DeviceConfig::Cpu,
            dtype: DataType::F32,
            max_memory: None,
        },
        generation: GenerationConfig {
            max_new_tokens: 100,
            temperature: Some(0.7),
            top_p: Some(0.9),
            ..Default::default()
        },
        max_history_turns: 4,
        enable_streaming: true,
    };

    let mut chat_client = ChatClient::new(config)?;

    let response = chat_client.send_message("Hello, introduce yourself briefly.")?;
    println!("AI Response: {}", response);
    println!();

    Ok(())
}
