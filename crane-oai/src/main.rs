mod chat_template;
mod engine;
mod openai_api;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
    routing::{get, post},
    Router,
};
use clap::Parser;
use futures::stream::Stream;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::info;

use chat_template::ChatTemplateProcessor;
use engine::model_factory::{ModelFormat, ModelType};
use engine::{EngineHandle, EngineResponse, InferenceEngine};
use openai_api::*;

// ── CLI ──

#[derive(Parser, Debug)]
#[command(
    name = "crane-oai",
    about = "OpenAI-compatible API server with continuous batching"
)]
struct Args {
    /// Path to model directory or GGUF file
    #[arg(long)]
    model_path: String,

    /// Model architecture: auto, hunyuan, qwen25, qwen3
    #[arg(long, default_value = "auto")]
    model_type: String,

    /// Model name to report in API responses (defaults to detected type)
    #[arg(long)]
    model_name: Option<String>,

    /// Host to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Use CPU even if GPU is available
    #[arg(long)]
    cpu: bool,

    /// Max concurrent sequences in decode phase
    #[arg(long, default_value_t = 32)]
    max_concurrent: usize,

    /// Tokens to decode per sequence before switching (higher = fewer KV swaps)
    #[arg(long, default_value_t = 16)]
    decode_tokens_per_seq: usize,

    /// Model weight format: auto, safetensors, or gguf
    #[arg(long, default_value = "auto")]
    format: String,
}

// ── App state ──

struct AppState {
    engine: EngineHandle,
    model_name: String,
    /// Shared tokenizer for request pre-processing (encode only).
    tokenizer: tokenizers::Tokenizer,
    /// Chat template processor (model-specific).
    chat_template: Box<dyn ChatTemplateProcessor>,
    /// Default EOS token ID for this model.
    eos_token_id: u32,
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

async fn stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let snap = state.engine.stats.snapshot();
    Json(snap)
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

/// POST /v1/chat/completions — streaming and non-streaming.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // Apply chat template.
    let formatted = state
        .chat_template
        .apply(&req.messages)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Chat template failed: {e}")))?;

    // Tokenize.
    let input_ids = state
        .tokenizer
        .encode(formatted.as_str(), true)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
        .get_ids()
        .to_vec();

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let include_usage = req
        .stream_options
        .as_ref()
        .map_or(false, |so| so.include_usage);

    // Submit to engine.
    let response_rx = state
        .engine
        .submit(
            request_id.clone(),
            input_ids,
            req.max_tokens,
            req.temperature.or(Some(0.8)),
            req.top_p.or(Some(0.95)),
            req.top_k.or(Some(40)),
            req.repetition_penalty.unwrap_or(1.05),
            state.eos_token_id,
        )
        .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;

    if req.stream {
        let model_name = state.model_name.clone();
        let stream = make_chat_sse_stream(request_id, model_name, response_rx, include_usage);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming — collect all tokens.
        let mut full_text = String::new();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();

        let mut response_rx = response_rx;
        while let Some(resp) = response_rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    full_text.push_str(&text);
                }
                EngineResponse::Finished {
                    full_text: ft,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    finish_reason: fr,
                } => {
                    full_text = ft;
                    prompt_tokens = pt;
                    completion_tokens = ct;
                    finish_reason = fr;
                    break;
                }
                EngineResponse::Error(e) => {
                    return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, &e));
                }
            }
        }

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created: now_epoch(),
            model: state.model_name.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: full_text,
                },
                finish_reason: Some(finish_reason),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

/// POST /v1/completions — text completion (no chat template).
async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let prompt = req.prompt.as_string();
    let include_usage = req
        .stream_options
        .as_ref()
        .map_or(false, |so| so.include_usage);

    let input_ids = state
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
        .get_ids()
        .to_vec();

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

    let response_rx = state
        .engine
        .submit(
            request_id.clone(),
            input_ids,
            req.max_tokens,
            req.temperature.or(Some(0.8)),
            req.top_p.or(Some(0.95)),
            req.top_k.or(Some(40)),
            req.repetition_penalty.unwrap_or(1.05),
            state.eos_token_id,
        )
        .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;

    if req.stream {
        let model_name = state.model_name.clone();
        let stream =
            make_completion_sse_stream(request_id, model_name, response_rx, include_usage);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let mut full_text = String::new();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();

        let mut response_rx = response_rx;
        while let Some(resp) = response_rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => full_text.push_str(&text),
                EngineResponse::Finished {
                    full_text: ft,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    finish_reason: fr,
                } => {
                    full_text = ft;
                    prompt_tokens = pt;
                    completion_tokens = ct;
                    finish_reason = fr;
                    break;
                }
                EngineResponse::Error(e) => {
                    return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, &e));
                }
            }
        }

        let response = CompletionResponse {
            id: request_id,
            object: "text_completion".into(),
            created: now_epoch(),
            model: state.model_name.clone(),
            choices: vec![CompletionChoice {
                index: 0,
                text: full_text,
                finish_reason: Some(finish_reason),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

// ── SSE streams ──

/// Build an SSE stream for chat completions.
fn make_chat_sse_stream(
    request_id: String,
    model_name: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let created = now_epoch();

    async_stream::stream! {
        // First chunk: role announcement.
        let first_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".into()),
                    content: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        yield Ok(Event::default().json_data(&first_chunk).unwrap());

        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;

        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    completion_tokens += 1;
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
                EngineResponse::Finished {
                    finish_reason,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    ..
                } => {
                    prompt_tokens = pt;
                    completion_tokens = ct;

                    // Final chunk with finish_reason.
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some(finish_reason),
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());

                    // Usage chunk (if requested).
                    if include_usage {
                        let usage_chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens + completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().json_data(&usage_chunk).unwrap());
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    yield Ok(Event::default().data(format!("error: {e}")));
                    break;
                }
            }
        }
    }
}

/// Build an SSE stream for text completions.
fn make_completion_sse_stream(
    request_id: String,
    model_name: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let created = now_epoch();

    async_stream::stream! {
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;

        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    completion_tokens += 1;
                    let chunk = CompletionChunk {
                        id: request_id.clone(),
                        object: "text_completion".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text,
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
                EngineResponse::Finished {
                    finish_reason,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    ..
                } => {
                    prompt_tokens = pt;
                    completion_tokens = ct;

                    let chunk = CompletionChunk {
                        id: request_id.clone(),
                        object: "text_completion".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text: String::new(),
                            finish_reason: Some(finish_reason),
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());

                    if include_usage {
                        let usage_chunk = CompletionChunk {
                            id: request_id.clone(),
                            object: "text_completion".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens + completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().json_data(&usage_chunk).unwrap());
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    yield Ok(Event::default().data(format!("error: {e}")));
                    break;
                }
            }
        }
    }
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

    // ── Device / dtype selection ──

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
                crane_core::models::Device::new_metal(0)
                    .unwrap_or(crane_core::models::Device::Cpu)
            }
            #[cfg(not(target_os = "macos"))]
            {
                crane_core::models::Device::Cpu
            }
        }
    };

    #[cfg(feature = "cuda")]
    let dtype = if args.cpu {
        crane_core::models::DType::F32
    } else {
        crane_core::models::DType::BF16
    };
    #[cfg(not(feature = "cuda"))]
    let dtype = crane_core::models::DType::F32;

    info!("Device: {:?}, dtype: {:?}", device, dtype);

    // ── Resolve model type ──

    let model_type = ModelType::from_str(&args.model_type);
    let format = ModelFormat::from_str(&args.format);

    // ── Load model via factory ──

    let mut backend =
        engine::model_factory::create_backend(model_type, &args.model_path, &device, &dtype, format)?;

    let resolved_type = if model_type == ModelType::Auto {
        engine::model_factory::detect_model_type(&args.model_path)
    } else {
        model_type
    };

    info!(
        "Model loaded successfully (type: {:?}, format: {:?})",
        resolved_type, format,
    );

    backend.warmup();
    info!("Model warmed up");

    // Clone tokenizer and get EOS token before moving backend into engine.
    let tokenizer = backend.tokenizer().clone();
    let eos_token_id = backend.eos_token_id();

    // ── Chat template ──

    let chat_template =
        engine::model_factory::create_chat_template(model_type, &args.model_path);

    // ── Model name for API responses ──

    let model_name = args.model_name.unwrap_or_else(|| {
        let base = std::path::Path::new(&args.model_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| resolved_type.display_name().to_string());
        base
    });

    // ── Start engine on dedicated thread ──

    let (engine, handle) =
        InferenceEngine::new(backend, args.max_concurrent, args.decode_tokens_per_seq);

    std::thread::Builder::new()
        .name("inference-engine".into())
        .spawn(move || engine.run())
        .expect("Failed to spawn engine thread");
    info!(
        "Inference engine started (max_concurrent={}, decode_tokens_per_seq={})",
        args.max_concurrent, args.decode_tokens_per_seq,
    );

    // ── Build router ──

    let state = Arc::new(AppState {
        engine: handle,
        model_name: model_name.clone(),
        tokenizer,
        chat_template,
        eos_token_id,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/stats", get(stats))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    info!("Starting server on {addr}");
    info!("  POST http://{addr}/v1/chat/completions");
    info!("  POST http://{addr}/v1/completions");
    info!("  GET  http://{addr}/v1/models");
    info!("  GET  http://{addr}/v1/stats");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
