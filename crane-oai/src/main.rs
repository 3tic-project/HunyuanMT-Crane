mod openai_api;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde_json::json;
use tokio::sync::Mutex;
use tracing::info;

use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::hunyuan_dense::Model;

use openai_api::*;

// ── CLI ──

#[derive(Parser, Debug)]
#[command(name = "crane-oai", about = "OpenAI-compatible API server for Hunyuan models")]
struct Args {
    /// Path to model directory
    #[arg(long, default_value = "model/Hunyuan-0.5B-Instruct")]
    model_path: String,

    /// Model name to report in API responses
    #[arg(long, default_value = "hunyuan")]
    model_name: String,

    /// Host to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Use CPU even if CUDA is available
    #[arg(long)]
    cpu: bool,
}

// ── App state ──

struct AppState {
    model: Mutex<Model>,
    model_name: String,
}

// ── Chat template ──

/// Apply the Hunyuan-family chat template to a list of messages.
///
/// Template (from chat_template.jinja):
///   - With system:  <BOS>{system}<placeholder3><User>{content}<Assistant>
///   - Without:      <BOS><User>{content}<Assistant>
///
/// Multi-turn:
///   <BOS><User>{u1}<Assistant>{a1}<EOS><User>{u2}<Assistant>
fn apply_chat_template(messages: &[ChatMessage]) -> String {
    const BOS: &str = "<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>";
    const USER: &str = "<\u{ff5c}hy_User\u{ff5c}>";
    const ASSISTANT: &str = "<\u{ff5c}hy_Assistant\u{ff5c}>";
    const EOS: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}>";
    const SEP: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}>";

    let mut result = String::new();
    result.push_str(BOS);

    // Extract system message if present
    let (system_msg, loop_messages) = if !messages.is_empty() && messages[0].role == "system" {
        (Some(&messages[0].content), &messages[1..])
    } else {
        (None, &messages[..])
    };

    if let Some(sys) = system_msg {
        result.push_str(sys);
        result.push_str(SEP);
    }

    for msg in loop_messages {
        match msg.role.as_str() {
            "user" => {
                result.push_str(USER);
                result.push_str(&msg.content);
            }
            "assistant" => {
                result.push_str(ASSISTANT);
                result.push_str(&msg.content);
                result.push_str(EOS);
            }
            _ => {}
        }
    }

    // Add generation prompt
    result.push_str(ASSISTANT);
    result
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Handlers ──

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".into(),
            created: now_epoch(),
            owned_by: "crane".into(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    if req.stream {
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "Streaming not yet supported. Set \"stream\": false.",
        ));
    }

    let formatted = apply_chat_template(&req.messages);

    let mut model = state.model.lock().await;

    let input_ids = model
        .prepare_inputs(&formatted)
        .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    let prompt_tokens = input_ids.len();

    let gen_config = GenerationConfig {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature.or(Some(0.7)),
        top_p: req.top_p.or(Some(0.8)),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.05),
        repeat_last_n: 64,
        eos_token_id: Some(120020),
        report_speed: false,
        ..Default::default()
    };

    let output_ids = model
        .generate(&input_ids, &gen_config, None)
        .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    let generated_ids = &output_ids[prompt_tokens..];
    let completion_tokens = generated_ids.len();

    let text = model
        .tokenizer
        .tokenizer
        .decode(generated_ids, true)
        .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")))?;

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        created: now_epoch(),
        model: state.model_name.clone(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: text,
            },
            finish_reason: Some(
                if generated_ids.last() == Some(&120020) {
                    "stop"
                } else {
                    "length"
                }
                .into(),
            ),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(response))
}

fn make_error(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: msg.to_string(),
                r#type: "invalid_request_error".into(),
                code: None,
            },
        }),
    )
}

// ── Main ──

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Loading model from: {}", args.model_path);

    let device = if args.cpu {
        crane_core::models::Device::Cpu
    } else {
        #[cfg(feature = "cuda")]
        {
            crane_core::models::Device::cuda_if_available(0)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            #[cfg(target_os = "macos")]
            {
                crane_core::models::Device::new_metal(0).unwrap_or(crane_core::models::Device::Cpu)
            }
            #[cfg(not(target_os = "macos"))]
            {
                crane_core::models::Device::Cpu
            }
        }
    };

    // BF16 on CUDA, F32 otherwise
    #[cfg(feature = "cuda")]
    let dtype = if args.cpu {
        crane_core::models::DType::F32
    } else {
        crane_core::models::DType::BF16
    };
    #[cfg(not(feature = "cuda"))]
    let dtype = crane_core::models::DType::F32;

    info!("Device: {:?}, dtype: {:?}", device, dtype);

    let mut model = Model::new(&args.model_path, &device, &dtype)?;
    info!("Model loaded successfully");

    model.warmup();
    info!("Model warmed up");

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        model_name: args.model_name.clone(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    info!("Starting server on {}", addr);
    info!(
        "API endpoint: http://{}/v1/chat/completions",
        addr
    );

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
