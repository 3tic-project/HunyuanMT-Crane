//! Continuous-batching inference engine.
//!
//! # Architecture
//!
//! ```text
//! API handlers ──(request channel)──► Engine thread
//!       ◄──(per-request response channel)──┘
//!
//! Engine loop (each iteration = one "step"):
//!   1. Drain new requests from channel
//!   2. Detect & cancel disconnected clients
//!   3. Scheduler picks next batch (prefill > decode)
//!   4. Prefill step: run full prompt for ONE new sequence
//!   5. Decode step: batched or sequential forward for running sequences
//!   6. If idle → blocking wait for new request
//! ```
//!
//! # Module layout
//!
//! | Module          | Responsibility                                   |
//! |-----------------|--------------------------------------------------|
//! | `types`         | Public request/response types + `EngineHandle`   |
//! | `stats`         | Lock-free counters shared with API layer          |
//! | `sampling`      | Token sampling (top-k, top-p, Gumbel-max, etc.) |
//! | `scheduler`     | FIFO scheduler with prefill priority              |
//! | `sequence`      | Per-request lifecycle state                       |
//! | `backend`       | `ModelBackend` trait + concrete implementations   |
//! | `model_factory` | Auto-detection and factory creation               |

pub mod backend;
pub mod model_factory;
pub mod sampling;
pub mod scheduler;
pub mod sequence;
pub mod stats;
pub mod types;

// Re-export commonly used items for convenience.
pub use stats::{EngineStats, StatsSnapshot};
pub use types::{EngineHandle, EngineRequest, EngineResponse};

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{Device, Tensor};
use crane_core::models::qwen3::decode_backend::DecodeBackendPlan;
use crane_core::models::qwen3::paged_kv::{PagedAttentionMetadata, PagedKvPool, SeqBlockTable};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use backend::ModelBackend;
use crane_core::utils::token_output_stream::TokenOutputStream;
use sampling::SamplingBuffers;
use scheduler::{Scheduler, SchedulerOutput};
use sequence::{Sequence, SequenceStatus};

// ─────────────────────────────────────────────────────────────
//  Memory configuration
// ─────────────────────────────────────────────────────────────

/// Configuration for GPU memory limits.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum tokens per sequence (prompt + completion). 0 = unlimited.
    pub max_seq_len: usize,
    /// GPU memory limit in bytes. 0 = unlimited.
    /// This is an **absolute** limit on total GPU memory usage.
    pub gpu_memory_limit_bytes: u64,
    /// Baseline GPU memory recorded after model load + warmup.
    /// The memory gate compares `(current_used - baseline)` against
    /// `(gpu_memory_limit_bytes - baseline)` so that the limit represents
    /// the *total* allowed usage, not just KV-cache growth.
    pub baseline_gpu_bytes: u64,
}

impl MemoryConfig {
    /// Parse memory configuration from CLI arguments.
    ///
    /// `gpu_memory_limit` accepts:
    ///   - Absolute sizes: "5G", "8G", "5120M", "5368709120" (bytes)
    ///   - Utilization fraction: "0.7" (70% of total GPU memory)
    pub fn parse(max_seq_len: usize, gpu_memory_limit: Option<&str>, device: &Device) -> Self {
        let gpu_memory_limit_bytes = match gpu_memory_limit {
            Some(s) => Self::parse_memory_limit(s, device),
            None => 0,
        };
        Self {
            max_seq_len,
            gpu_memory_limit_bytes,
            baseline_gpu_bytes: 0,
        }
    }

    fn parse_memory_limit(s: &str, device: &Device) -> u64 {
        let s = s.trim();
        if s.is_empty() || s == "0" {
            return 0;
        }

        // Try absolute sizes: "5G", "8G", "5120M", "1024K"
        let upper = s.to_uppercase();
        if upper.ends_with('G') {
            if let Ok(n) = upper[..upper.len() - 1].trim().parse::<f64>() {
                return (n * (1u64 << 30) as f64) as u64;
            }
        }
        if upper.ends_with('M') {
            if let Ok(n) = upper[..upper.len() - 1].trim().parse::<f64>() {
                return (n * (1u64 << 20) as f64) as u64;
            }
        }

        // Try as a fraction (0.0 - 1.0)
        if let Ok(frac) = s.parse::<f64>() {
            if (0.0..=1.0).contains(&frac) {
                let total = Self::query_total_gpu_memory(device);
                if total > 0 {
                    return (frac * total as f64) as u64;
                }
            }
            // If > 1.0, treat as bytes
            if frac > 1.0 {
                return frac as u64;
            }
        }

        tracing::warn!("Could not parse gpu_memory_limit '{}', ignoring", s);
        0
    }

    /// Record baseline GPU memory (call after model load + warmup).
    pub fn record_baseline(&mut self, device: &Device) {
        let (used, _total) = query_gpu_memory_usage(device);
        self.baseline_gpu_bytes = used;
    }

    /// Query total GPU memory (bytes). Returns 0 if unavailable.
    fn query_total_gpu_memory(_device: &Device) -> u64 {
        #[cfg(feature = "cuda")]
        {
            if let Device::Cuda(_) = _device {
                if let Ok((_free, total)) =
                    candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                {
                    return total as u64;
                }
            }
        }
        0
    }
}

/// Query current GPU memory usage. Returns (used_bytes, total_bytes).
/// Returns (0, 0) if not on CUDA.
fn query_gpu_memory_usage(_device: &Device) -> (u64, u64) {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = _device {
            if let Ok((free, total)) =
                candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            {
                return ((total - free) as u64, total as u64);
            }
        }
    }
    (0, 0)
}

/// Format a byte count as a human-readable string (used in engine log messages).
fn format_bytes_engine(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1}G", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0}M", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{}B", bytes)
    }
}

#[derive(Clone)]
struct ActiveBatchDecodeSession {
    batch: Vec<String>,
    batch_width: usize,
    plan: Option<DecodeBackendPlan>,
    backend_name: &'static str,
}

// ─────────────────────────────────────────────────────────────
//  InferenceEngine
// ─────────────────────────────────────────────────────────────

/// KV-to-GPU overhead factor.
///
/// `tracked_kv_bytes` only captures live per-sequence KV cache tensors, which
/// is roughly 15-20% of the *real* GPU memory consumed.  Batch-decode setup
/// creates padded copies, the CUDA caching allocator retains freed blocks,
/// and forward-pass intermediates add extra pressure.  Empirically the ratio
/// between actual GPU growth over baseline and tracked KV bytes is 5-8×.
///
/// We use 6× so that `kv_budget = (limit - baseline) / 6`.  This gives the
/// engine a realistic estimate of how much KV it can afford before the GPU
/// runs out of memory.
const KV_GPU_OVERHEAD_FACTOR: u64 = 6;

/// Continuous-batching inference engine.
///
/// Runs on a dedicated OS thread (model forward passes are synchronous).
/// Communicates with async API handlers via channels.
pub struct InferenceEngine {
    model: Box<dyn ModelBackend>,
    sequences: HashMap<String, Sequence>,
    token_streams: HashMap<String, TokenOutputStream>,
    scheduler: Scheduler,
    request_rx: mpsc::UnboundedReceiver<EngineRequest>,
    active_seq_id: Option<String>,
    num_layers: usize,
    stats: Arc<EngineStats>,
    /// How many tokens to decode for one sequence before switching.
    decode_tokens_per_seq: usize,
    /// Maximum prompt tokens to prefill in a single engine step.
    prefill_chunk_size: usize,
    /// Engine start time for uptime calculation.
    start_time: Instant,
    /// Step counter for periodic stats logging.
    step_counter: u64,
    sampling_buffers: SamplingBuffers,
    /// Memory configuration for VRAM limits.
    memory_config: MemoryConfig,
    /// Timestamp of last memory-limit warning (to throttle log spam).
    last_mem_warn: Instant,
    /// Tracked total KV cache bytes across all sequences (not relying on
    /// `cuMemGetInfo` which includes CUDA allocator pool bloat).
    tracked_kv_bytes: u64,
    /// Steps remaining before cuMemGetInfo checks are re-enabled after eviction.
    /// The CUDA caching allocator doesn't instantly reflect freed memory, so we
    /// grant a short cooldown after preemption to avoid a deadlock where
    /// cuMemGetInfo always reports over-limit.
    eviction_cooldown: u32,
    /// Reusable batched-decode state kept resident in the model while the
    /// running batch remains unchanged.
    active_batch_decode: Option<ActiveBatchDecodeSession>,
    /// Shadow paged-KV allocator used for exact page accounting and metadata
    /// planning before the real paged attention backend is wired in.
    paged_kv_pool: Option<PagedKvPool>,
}

impl InferenceEngine {
    /// Create the engine and return a handle for submitting requests.
    pub fn new(
        model: Box<dyn ModelBackend>,
        max_concurrent: usize,
        decode_tokens_per_seq: usize,
        prefill_chunk_size: usize,
        memory_config: MemoryConfig,
    ) -> (Self, EngineHandle) {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let num_layers = model.num_layers();

        // Cap max_concurrent to 1 for models without KV cache swapping.
        let effective_max = if model.supports_kv_swap() {
            max_concurrent
        } else {
            1.min(max_concurrent)
        };
        if effective_max != max_concurrent {
            info!(
                "Model does not support KV swap — limiting max_concurrent to {}",
                effective_max
            );
        }
        let decode_tokens_per_seq = decode_tokens_per_seq.max(1);
        let paged_kv_pool = Self::build_shadow_paged_kv_pool(
            model.as_ref(),
            &memory_config,
            effective_max,
            decode_tokens_per_seq,
        );

        let stats = Arc::new(EngineStats::new());
        let mut scheduler = Scheduler::new(effective_max);
        let prefill_chunk_size = if prefill_chunk_size == 0 {
            usize::MAX
        } else {
            prefill_chunk_size
        };
        scheduler.set_chunked_prefill_mode(prefill_chunk_size != usize::MAX);

        let engine = Self {
            model,
            sequences: HashMap::new(),
            token_streams: HashMap::new(),
            scheduler,
            request_rx,
            active_seq_id: None,
            num_layers,
            stats: stats.clone(),
            decode_tokens_per_seq,
            prefill_chunk_size,
            start_time: Instant::now(),
            step_counter: 0,
            sampling_buffers: SamplingBuffers::new(),
            memory_config,
            last_mem_warn: Instant::now() - std::time::Duration::from_secs(60),
            tracked_kv_bytes: 0,
            eviction_cooldown: 0,
            active_batch_decode: None,
            paged_kv_pool,
        };
        let handle = EngineHandle { request_tx, stats };
        (engine, handle)
    }

    // ─────────────────────────────────────────────────────────
    //  Main loop
    // ─────────────────────────────────────────────────────────

    /// Run the engine loop (blocking — call from a dedicated thread).
    pub fn run(mut self) {
        // Log effective memory budget.
        let baseline = self.memory_config.baseline_gpu_bytes;
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit > 0 {
            let kv_budget = self.kv_budget_bytes();
            if kv_budget == 0 || limit <= baseline {
                warn!(
                    "gpu_memory_limit ({}) <= model baseline ({}). \
                     KV-cache budget is 0 — all sequences will be immediately preempted.",
                    format_bytes_engine(limit),
                    format_bytes_engine(baseline),
                );
            } else {
                info!(
                    "Memory budget: total_limit={}, model_baseline={}, kv_budget={} (overhead={}x, also checked by cuMemGetInfo)",
                    format_bytes_engine(limit),
                    format_bytes_engine(baseline),
                    format_bytes_engine(kv_budget),
                    KV_GPU_OVERHEAD_FACTOR,
                );
            }
        }
        let prefill_chunk_display = if self.prefill_chunk_size == usize::MAX {
            "full-prompt".to_string()
        } else {
            self.prefill_chunk_size.to_string()
        };
        info!(
            "Engine started (max_concurrent={}, decode_tokens_per_seq={}, prefill_chunk_size={}, max_seq_len={})",
            self.scheduler.max_running,
            self.decode_tokens_per_seq,
            prefill_chunk_display,
            if self.memory_config.max_seq_len == 0 { "unlimited".to_string() } else { self.memory_config.max_seq_len.to_string() },
        );

        loop {
            self.drain_requests();
            self.check_cancelled();

            // Decrement eviction cooldown (cuMemGetInfo grace period).
            self.eviction_cooldown = self.eviction_cooldown.saturating_sub(1);

            self.stats
                .active_sequences
                .store(self.scheduler.running.len() as u64, Ordering::Relaxed);
            self.stats
                .waiting_sequences
                .store(self.scheduler.waiting.len() as u64, Ordering::Relaxed);

            let output = self.scheduler.schedule();

            match output {
                Some(output) => {
                    // KV cache budget gate: if a prefill is scheduled but we're
                    // over the KV budget, first try to evict (preempt) the
                    // largest running sequence to make room. If still over,
                    // defer the prefill and drain existing sequences.
                    if output.is_prefill && self.is_over_kv_budget() {
                        if let (Some(used_pages), Some(page_budget)) =
                            (self.current_estimated_kv_pages(), self.kv_page_budget())
                        {
                            if used_pages > page_budget {
                                self.stats
                                    .total_page_budget_denials
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        // Attempt eviction before deferring.
                        self.evict_if_needed();

                        if self.is_over_kv_budget() && !self.scheduler.running.is_empty() {
                            // Still over budget and have running sequences to drain.
                            for seq_id in &output.batch {
                                self.scheduler.waiting.push_front(seq_id.clone());
                            }
                            let decode_batch: Vec<String> =
                                self.scheduler.running.iter().cloned().collect();
                            let decode_output = SchedulerOutput {
                                batch: decode_batch,
                                is_prefill: false,
                            };
                            self.execute_step(decode_output);
                        } else {
                            // Budget OK after eviction (or nothing running) — proceed.
                            self.execute_step(output);
                        }
                    } else {
                        self.execute_step(output);
                    }
                    self.step_counter += 1;

                    if self.step_counter % 50 == 0 {
                        self.log_stats();
                    }
                }
                None => match self.request_rx.blocking_recv() {
                    Some(req) => self.accept_request(req),
                    None => {
                        info!("Engine channel closed, shutting down");
                        self.log_stats();
                        return;
                    }
                },
            }
        }
    }

    fn log_stats(&self) {
        let snap = self.stats.snapshot();
        let uptime = self.start_time.elapsed().as_secs();
        let (gpu_used, gpu_total) = query_gpu_memory_usage(self.model.device());
        let budget = self.kv_budget_bytes();
        let budget_info = if budget < u64::MAX {
            format!(" kv_budget: {}", format_bytes_engine(budget))
        } else {
            String::new()
        };
        let gpu_info = if gpu_total > 0 {
            format!(
                " | gpu_mem: {:.1}G/{:.1}G ({:.0}%) | kv_cache: {}{}",
                gpu_used as f64 / (1u64 << 30) as f64,
                gpu_total as f64 / (1u64 << 30) as f64,
                gpu_used as f64 / gpu_total as f64 * 100.0,
                format_bytes_engine(self.tracked_kv_bytes),
                budget_info,
            )
        } else {
            format!(
                " | kv_cache: {}{}",
                format_bytes_engine(self.tracked_kv_bytes),
                budget_info
            )
        };
        let plan_total = snap.total_decode_plan_cache_hits + snap.total_decode_plan_cache_misses;
        let plan_hit_rate = if plan_total > 0 {
            snap.total_decode_plan_cache_hits as f64 / plan_total as f64 * 100.0
        } else {
            0.0
        };
        info!(
            "Engine stats | uptime={}s | requests: total={} completed={} cancelled={} failed={} | \
             sequences: active={} waiting={} | \
             tokens: prompt={} completion={} | \
             kv_swaps={} prefill_chunks={} | \
             speed: prefill={:.1} tok/s decode={:.1} tok/s | \
             decode_plan_hit={:.0}% plan_reuse={} workspace_reset={} sampling_ms={}{}",
            uptime,
            snap.total_requests,
            snap.completed_requests,
            snap.cancelled_requests,
            snap.failed_requests,
            snap.active_sequences,
            snap.waiting_sequences,
            snap.total_prompt_tokens,
            snap.total_completion_tokens,
            snap.total_kv_swaps,
            snap.total_prefill_chunks,
            snap.avg_prefill_tokens_per_sec,
            snap.avg_decode_tokens_per_sec,
            plan_hit_rate,
            snap.total_decode_plan_reuses,
            snap.total_decode_workspace_resets,
            snap.total_sampling_time_us / 1000,
            gpu_info,
        );
    }

    // ─────────────────────────────────────────────────────────
    //  Memory management
    // ─────────────────────────────────────────────────────────

    /// Recount `tracked_kv_bytes` from all sequences.
    /// For the active sequence, bytes are in the model (uses `active_kv_cache_bytes`).
    /// For other sequences, bytes are stored in `seq.kv_caches`.
    fn recount_kv_bytes(&mut self) {
        let mut total: u64 = 0;
        for (id, seq) in &self.sequences {
            if self.active_seq_id.as_deref() == Some(id.as_str()) {
                total += self.model.active_kv_cache_bytes();
            } else {
                total += sequence::kv_cache_bytes(&seq.kv_caches);
            }
        }
        if self.active_batch_decode.is_some() {
            total += self.model.active_kv_cache_bytes();
        }
        self.tracked_kv_bytes = total;
        self.stats
            .current_tracked_kv_bytes
            .store(total, Ordering::Relaxed);
        self.stats.current_estimated_kv_pages.store(
            self.current_estimated_kv_pages().unwrap_or(0) as u64,
            Ordering::Relaxed,
        );
    }

    /// KV cache budget **in KV-cache bytes** (not raw GPU bytes).
    ///
    /// Each byte of live KV cache costs roughly `KV_GPU_OVERHEAD_FACTOR` bytes
    /// of real GPU memory (due to padded batch copies, CUDA pool bloat, and
    /// forward-pass intermediates).  The budget is therefore:
    ///
    /// ```text
    /// kv_budget = (gpu_limit - baseline) / KV_GPU_OVERHEAD_FACTOR
    /// ```
    ///
    /// Returns `u64::MAX` when no limit is configured.
    fn kv_budget_bytes(&self) -> u64 {
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit == 0 {
            return u64::MAX;
        }
        let raw = limit.saturating_sub(self.memory_config.baseline_gpu_bytes);
        raw / KV_GPU_OVERHEAD_FACTOR
    }

    /// Check whether the engine should block new prefills due to memory
    /// pressure.  Two complementary checks:
    ///
    /// 1. **KV budget** — `tracked_kv_bytes > kv_budget_bytes()`.  This is the
    ///    primary admission control, using an overhead factor to estimate real
    ///    GPU cost from the tracked KV cache bytes.
    ///
    /// 2. **cuMemGetInfo hard safety** — if actual GPU memory (as reported by
    ///    the driver) exceeds the configured limit, block prefills.  This
    ///    catches cases where the overhead factor underestimates.  The check
    ///    is skipped during `eviction_cooldown` to avoid a deadlock (the CUDA
    ///    caching allocator doesn't instantly reflect freed memory).
    fn is_over_kv_budget(&mut self) -> bool {
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit == 0 {
            return false;
        }

        let budget = self.kv_budget_bytes();
        if budget == 0 {
            return true; // limit <= baseline
        }

        // Check 1: tracked KV bytes vs overhead-adjusted budget.
        if self.tracked_kv_bytes > budget {
            let now = Instant::now();
            if now.duration_since(self.last_mem_warn).as_secs() >= 5 {
                self.last_mem_warn = now;
                warn!(
                    "KV budget exceeded: kv_used={} > kv_budget={} (limit={} baseline={} overhead={}x)",
                    format_bytes_engine(self.tracked_kv_bytes),
                    format_bytes_engine(budget),
                    format_bytes_engine(limit),
                    format_bytes_engine(self.memory_config.baseline_gpu_bytes),
                    KV_GPU_OVERHEAD_FACTOR,
                );
            }
            return true;
        }

        if let (Some(used_pages), Some(page_budget)) =
            (self.current_estimated_kv_pages(), self.kv_page_budget())
        {
            if used_pages > page_budget {
                let now = Instant::now();
                if now.duration_since(self.last_mem_warn).as_secs() >= 5 {
                    self.last_mem_warn = now;
                    warn!(used_pages, page_budget, "Estimated KV page budget exceeded");
                }
                return true;
            }
        }

        // Check 2: cuMemGetInfo hard safety (skip during cooldown).
        if self.eviction_cooldown == 0 {
            let (gpu_used, _) = query_gpu_memory_usage(self.model.device());
            if gpu_used > 0 && gpu_used > limit {
                let now = Instant::now();
                if now.duration_since(self.last_mem_warn).as_secs() >= 5 {
                    self.last_mem_warn = now;
                    warn!(
                        "GPU memory hard limit exceeded: gpu_used={} > limit={} (kv_tracked={})",
                        format_bytes_engine(gpu_used),
                        format_bytes_engine(limit),
                        format_bytes_engine(self.tracked_kv_bytes),
                    );
                }
                return true;
            }
        }

        false
    }

    /// Preempt (evict) running sequences until KV usage is within budget.
    ///
    /// Eviction policy: **longest-output-first** — the sequence that has
    /// generated the most tokens (and therefore holds the largest KV cache)
    /// is evicted first. Its KV cache is dropped and it is moved back to
    /// the waiting queue for later re-prefill.
    ///
    /// This mirrors sglang's retraction strategy.
    fn evict_if_needed(&mut self) {
        let budget = self.kv_budget_bytes();
        if budget == u64::MAX {
            return;
        }

        while self.tracked_kv_bytes > budget && !self.scheduler.running.is_empty() {
            // Find the running sequence with the most generated tokens (largest KV).
            let victim_id = self
                .scheduler
                .running
                .iter()
                .filter_map(|id| {
                    self.sequences
                        .get(id)
                        .map(|seq| (id.clone(), seq.tokens.len()))
                })
                .max_by_key(|(_, len)| *len)
                .map(|(id, _)| id);

            let victim_id = match victim_id {
                Some(id) => id,
                None => break,
            };

            // Compute bytes being freed.
            let freed = self
                .sequences
                .get(&victim_id)
                .map(|seq| sequence::kv_cache_bytes(&seq.kv_caches))
                .unwrap_or(0);

            info!(
                id = %victim_id,
                freed_bytes = %format_bytes_engine(freed),
                kv_used = %format_bytes_engine(self.tracked_kv_bytes),
                kv_budget = %format_bytes_engine(budget),
                "Preempting sequence (KV cache eviction) — will re-prefill later",
            );

            // If this sequence's KV is currently loaded in the model, clear it.
            if self.active_seq_id.as_deref() == Some(&victim_id) {
                self.model.clear_kv_cache();
                self.active_seq_id = None;
            }

            // Drop KV caches and reset sequence state to Waiting.
            if let Some(seq) = self.sequences.get_mut(&victim_id) {
                seq.kv_caches = vec![None; self.num_layers];
                seq.status = SequenceStatus::Waiting;
                seq.prefill_cursor = 0;
                // Reset tokens to just the prompt to allow re-prefill.
                seq.tokens.truncate(seq.prompt_len);
            }

            self.tracked_kv_bytes = self.tracked_kv_bytes.saturating_sub(freed);

            // Move from running back to waiting (back, not front — avoid
            // immediate re-prefill which would cause thrashing).
            self.scheduler.running.retain(|id| id != &victim_id);
            self.release_sequence_paged_kv(&victim_id);
            self.scheduler.waiting.push_back(victim_id);
        }

        // Cap effective max_running to the post-eviction running count.
        // This prevents the scheduler from admitting new sequences that
        // would immediately exceed the budget again (eviction thrashing).
        // The cap is lifted when a sequence finishes naturally.
        let post_eviction_running = self.scheduler.running.len();
        self.scheduler.effective_max_running = Some(post_eviction_running);
        info!(
            "Eviction complete: capping concurrent sequences at {} (was {})",
            post_eviction_running, self.scheduler.max_running,
        );

        // Grant a cooldown period so the cuMemGetInfo hard-safety check
        // doesn't immediately re-trigger (CUDA pool retains freed blocks).
        self.eviction_cooldown = 5;
    }

    fn batch_seq_lens(&self, batch: &[String]) -> Vec<usize> {
        batch
            .iter()
            .map(|id| {
                self.sequences
                    .get(id)
                    .map(|seq| seq.start_pos())
                    .unwrap_or(0)
            })
            .collect()
    }

    fn maybe_relax_eviction_cap(&mut self) {
        let cap = match self.scheduler.effective_max_running {
            Some(cap) => cap,
            None => return,
        };

        if cap >= self.scheduler.max_running {
            self.scheduler.effective_max_running = None;
            return;
        }

        if self.eviction_cooldown > 0 || self.scheduler.running.len() < cap {
            return;
        }

        let budget = self.kv_budget_bytes();
        let has_headroom = if budget == u64::MAX {
            true
        } else {
            // Only relax once KV usage has fallen well below the prior limit.
            self.tracked_kv_bytes.saturating_mul(4) <= budget.saturating_mul(3)
        };

        let has_page_headroom = match (self.current_estimated_kv_pages(), self.kv_page_budget()) {
            (Some(used), Some(budget)) if budget > 0 => {
                used.saturating_mul(4) <= budget.saturating_mul(3)
            }
            _ => true,
        };

        if !has_headroom || !has_page_headroom {
            return;
        }

        let new_cap = (cap + 1).min(self.scheduler.max_running);
        if new_cap >= self.scheduler.max_running {
            info!(
                previous_cap = cap,
                restored_cap = self.scheduler.max_running,
                "Eviction cap fully relaxed under sustained headroom",
            );
            self.scheduler.effective_max_running = None;
        } else {
            info!(
                previous_cap = cap,
                relaxed_cap = new_cap,
                waiting = self.scheduler.waiting.len(),
                running = self.scheduler.running.len(),
                "Relaxing eviction cap under sustained headroom",
            );
            self.scheduler.effective_max_running = Some(new_cap);
        }
    }

    fn can_reuse_active_batch_decode(&self, batch: &[String]) -> bool {
        self.active_batch_decode
            .as_ref()
            .map(|session| session.batch == batch)
            .unwrap_or(false)
    }

    fn flush_active_batch_decode_session(&mut self) {
        let session = match self.active_batch_decode.take() {
            Some(session) => session,
            None => return,
        };

        let seq_lens = self.batch_seq_lens(&session.batch);
        let t_extract = Instant::now();
        match self
            .model
            .extract_batch_kv_current(&seq_lens, session.batch_width)
        {
            Ok(extracted) => {
                for (i, seq_id) in session.batch.iter().enumerate() {
                    if let Some(seq) = self.sequences.get_mut(seq_id) {
                        if let Some(caches) = extracted.get(i) {
                            seq.kv_caches = caches.clone();
                        }
                    }
                }
                let extract_us = t_extract.elapsed().as_micros() as u64;
                self.stats
                    .total_batch_decode_extract_time_us
                    .fetch_add(extract_us, Ordering::Relaxed);
            }
            Err(e) => {
                error!(
                    backend = session.backend_name,
                    "Failed to flush active batch-decode session: {e}"
                );
            }
        }

        self.model.clear_kv_cache();
        if let Err(e) = self.model.reset_decode_backend_workspace() {
            error!(
                backend = session.backend_name,
                "Failed to reset decode backend workspace: {e}"
            );
        } else {
            self.stats
                .total_decode_workspace_resets
                .fetch_add(1, Ordering::Relaxed);
        }
        self.recount_kv_bytes();
    }

    fn kv_page_budget(&self) -> Option<usize> {
        let budget = self.kv_budget_bytes();
        if budget == u64::MAX {
            return None;
        }
        let cfg = self.model.paged_kv_config()?;
        let page_bytes = cfg.total_page_size_bytes();
        if page_bytes == 0 {
            None
        } else {
            Some((budget / page_bytes) as usize)
        }
    }

    fn current_estimated_kv_pages(&self) -> Option<usize> {
        let cfg = self.model.paged_kv_config()?;
        Some(
            self.sequences
                .values()
                .map(|seq| {
                    seq.paged_kv_table
                        .as_ref()
                        .map(|table| table.block_count())
                        .unwrap_or_else(|| cfg.pages_for_tokens(seq.start_pos()))
                })
                .sum(),
        )
    }

    fn build_shadow_paged_kv_pool(
        model: &dyn ModelBackend,
        memory_config: &MemoryConfig,
        max_running: usize,
        decode_tokens_per_seq: usize,
    ) -> Option<PagedKvPool> {
        let cfg = model.paged_kv_config()?;
        let decode_headroom_pages =
            max_running.saturating_mul(cfg.pages_for_tokens(decode_tokens_per_seq));
        let capacity_pages = if memory_config.gpu_memory_limit_bytes > 0 {
            let budget = memory_config
                .gpu_memory_limit_bytes
                .saturating_sub(memory_config.baseline_gpu_bytes)
                / KV_GPU_OVERHEAD_FACTOR;
            let page_bytes = cfg.total_page_size_bytes();
            if page_bytes == 0 {
                return None;
            }
            (budget / page_bytes) as usize
        } else if memory_config.max_seq_len > 0 {
            max_running.saturating_mul(cfg.pages_for_tokens(memory_config.max_seq_len))
        } else {
            return None;
        };
        Some(PagedKvPool::new(
            cfg,
            capacity_pages.saturating_add(decode_headroom_pages),
        ))
    }

    fn decode_page_reserve_pages(&self) -> Option<usize> {
        let cfg = self.model.paged_kv_config()?;
        Some(
            self.scheduler
                .running
                .len()
                .saturating_mul(cfg.pages_for_tokens(self.decode_tokens_per_seq)),
        )
    }

    fn can_admit_prefill_by_page_budget(&self, seq_id: &str) -> bool {
        let page_budget = match self.kv_page_budget() {
            Some(budget) => budget,
            None => return true,
        };
        let seq = match self.sequences.get(seq_id) {
            Some(seq) => seq,
            None => return false,
        };
        let cfg = match self.model.paged_kv_config() {
            Some(cfg) => cfg,
            None => return true,
        };
        let next_chunk_len = if self.prefill_chunk_size == usize::MAX {
            seq.next_input_ids().len()
        } else {
            seq.next_prefill_chunk(self.prefill_chunk_size).len()
        };
        let target_tokens = seq.start_pos().saturating_add(next_chunk_len);
        let current_pages = seq
            .paged_kv_table
            .as_ref()
            .map(|table| table.block_count())
            .unwrap_or(0);
        let additional_pages = cfg
            .pages_for_tokens(target_tokens)
            .saturating_sub(current_pages);
        let used_pages = self.current_estimated_kv_pages().unwrap_or(0);
        let decode_reserve = self.decode_page_reserve_pages().unwrap_or(0);
        used_pages
            .saturating_add(additional_pages)
            .saturating_add(decode_reserve)
            <= page_budget
    }

    fn sync_sequence_paged_kv(&mut self, seq_id: &str, target_tokens: usize) -> Result<(), String> {
        let pool = match self.paged_kv_pool.as_mut() {
            Some(pool) => pool,
            None => return Ok(()),
        };
        let seq = self
            .sequences
            .get_mut(seq_id)
            .ok_or_else(|| format!("unknown sequence {seq_id}"))?;

        if target_tokens == 0 {
            if let Some(mut table) = seq.paged_kv_table.take() {
                pool.free_seq_table(&mut table);
            }
            return Ok(());
        }

        let page_size = pool.config().page_size;
        let mut table = seq
            .paged_kv_table
            .take()
            .unwrap_or_else(|| SeqBlockTable::new(page_size));

        if target_tokens < table.token_count() {
            pool.free_seq_table(&mut table);
            table = pool
                .alloc_seq_table(target_tokens)
                .map_err(|e| e.to_string())?;
        } else {
            let delta = target_tokens.saturating_sub(table.token_count());
            if delta > 0 {
                pool.append_seq_tokens(&mut table, delta)
                    .map_err(|e| e.to_string())?;
            }
        }

        seq.paged_kv_table = Some(table);
        Ok(())
    }

    fn release_sequence_paged_kv(&mut self, seq_id: &str) {
        let pool = match self.paged_kv_pool.as_mut() {
            Some(pool) => pool,
            None => return,
        };
        if let Some(seq) = self.sequences.get_mut(seq_id) {
            if let Some(mut table) = seq.paged_kv_table.take() {
                pool.free_seq_table(&mut table);
            }
        }
    }

    fn batch_paged_attention_metadata(&self, batch: &[String]) -> Option<PagedAttentionMetadata> {
        let page_size = self.model.paged_kv_config()?.page_size;
        let mut tables = Vec::with_capacity(batch.len());
        for seq_id in batch {
            let seq = self.sequences.get(seq_id)?;
            tables.push(seq.paged_kv_table.as_ref()?.clone());
        }
        Some(PagedAttentionMetadata::from_block_tables(
            &tables, page_size,
        ))
    }

    fn try_refresh_session_decode_plan(
        &mut self,
        metadata: &PagedAttentionMetadata,
    ) -> Option<DecodeBackendPlan> {
        let mut plan = self
            .active_batch_decode
            .as_ref()
            .and_then(|session| session.plan.clone())?;
        let expected_bucket = metadata.bucket_key(self.decode_tokens_per_seq, plan.split_kv);
        if plan.backend_name != self.model.decode_backend_name()
            || plan.bucket_key != expected_bucket
        {
            return None;
        }
        match self.model.refresh_batch_decode_plan(&mut plan, metadata) {
            Ok(()) => {
                plan.plan_cache_hit = true;
                Some(plan)
            }
            Err(e) => {
                warn!(
                    backend = self.model.decode_backend_name(),
                    error = %e,
                    "Decode plan refresh failed, falling back to full plan",
                );
                None
            }
        }
    }

    fn prepare_batch_decode_plan(
        &mut self,
        batch: &[String],
        kv_lens: &[usize],
        reuse_session: bool,
    ) -> candle_core::Result<(Option<DecodeBackendPlan>, bool)> {
        let paged_metadata = self.batch_paged_attention_metadata(batch);
        if let Some(metadata) = paged_metadata.as_ref() {
            if reuse_session {
                if let Some(plan) = self.try_refresh_session_decode_plan(metadata) {
                    return Ok((Some(plan), true));
                }
            }
            return self
                .model
                .plan_batch_decode_with_metadata(metadata, self.decode_tokens_per_seq)
                .map(|plan| (plan, false));
        }
        self.model
            .plan_batch_decode(kv_lens, self.decode_tokens_per_seq)
            .map(|plan| (plan, false))
    }

    /// Effective max_tokens for a request, taking server-level max_seq_len into account.
    fn effective_max_tokens(&self, prompt_len: usize, requested_max_tokens: usize) -> usize {
        if self.memory_config.max_seq_len == 0 {
            return requested_max_tokens;
        }
        let remaining = self.memory_config.max_seq_len.saturating_sub(prompt_len);
        requested_max_tokens.min(remaining)
    }

    // ─────────────────────────────────────────────────────────
    //  Request handling
    // ─────────────────────────────────────────────────────────

    fn drain_requests(&mut self) {
        while let Ok(req) = self.request_rx.try_recv() {
            self.accept_request(req);
        }
    }

    fn accept_request(&mut self, req: EngineRequest) {
        let prompt_len = req.tokens.len();
        let tokenizer = self.model.tokenizer().clone();

        // Reject prompts that already exceed max_seq_len.
        if self.memory_config.max_seq_len > 0 && prompt_len > self.memory_config.max_seq_len {
            warn!(
                id = %req.id,
                prompt_len,
                max_seq_len = self.memory_config.max_seq_len,
                "Prompt exceeds max_seq_len, rejecting request",
            );
            let _ = req.response_tx.send(EngineResponse::Error(format!(
                "Prompt length ({}) exceeds server max_seq_len ({})",
                prompt_len, self.memory_config.max_seq_len,
            )));
            self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Cap max_tokens to respect max_seq_len.
        let effective_max_tokens = self.effective_max_tokens(prompt_len, req.max_tokens);

        info!(
            id = %req.id,
            prompt_len,
            max_tokens = effective_max_tokens,
            temp = ?req.temperature,
            top_p = ?req.top_p,
            top_k = ?req.top_k,
            rep_penalty = req.repetition_penalty,
            "New request accepted (queue: waiting={} running={})",
            self.scheduler.waiting.len() + 1,
            self.scheduler.running.len(),
        );

        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_prompt_tokens
            .fetch_add(prompt_len as u64, Ordering::Relaxed);

        let seq = Sequence {
            id: req.id.clone(),
            status: SequenceStatus::Waiting,
            tokens: req.tokens,
            prompt_len,
            prefill_cursor: 0,
            kv_caches: vec![None; self.num_layers],
            paged_kv_table: None,
            logits_processor: candle_transformers::generation::LogitsProcessor::new(
                sampling::rand_seed(),
                req.temperature,
                req.top_p,
            ),
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: effective_max_tokens,
            eos_token_id: req.eos_token_id,
            repetition_penalty: req.repetition_penalty,
            repeat_last_n: 64,
            response_tx: req.response_tx,
        };

        let stream = TokenOutputStream::new(tokenizer);
        self.sequences.insert(req.id.clone(), seq);
        self.token_streams.insert(req.id.clone(), stream);
        self.scheduler.add(req.id);
    }

    // ─────────────────────────────────────────────────────────
    //  Cancellation detection
    // ─────────────────────────────────────────────────────────

    fn check_cancelled(&mut self) {
        let cancelled: Vec<String> = self
            .sequences
            .iter()
            .filter(|(_, seq)| seq.response_tx.is_closed())
            .map(|(id, _)| id.clone())
            .collect();

        for id in cancelled {
            warn!(id = %id, "Client disconnected, cancelling sequence");
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(&id);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Step execution dispatch
    // ─────────────────────────────────────────────────────────

    fn execute_step(&mut self, output: SchedulerOutput) {
        if output.is_prefill {
            debug_assert_eq!(output.batch.len(), 1);
            let seq_id = output.batch[0].clone();
            if !self.can_admit_prefill_by_page_budget(&seq_id) {
                self.stats
                    .total_page_budget_denials
                    .fetch_add(1, Ordering::Relaxed);
                debug!(
                    id = %seq_id,
                    running = self.scheduler.running.len(),
                    waiting = self.scheduler.waiting.len(),
                    "Deferring prefill due to paged-KV page budget",
                );
                self.scheduler.requeue_waiting_front(seq_id);
                if !self.scheduler.running.is_empty() {
                    let decode_batch: Vec<String> =
                        self.scheduler.running.iter().cloned().collect();
                    self.execute_step(SchedulerOutput {
                        batch: decode_batch,
                        is_prefill: false,
                    });
                }
                return;
            }
            self.step_prefill(seq_id);
        } else if self.model.supports_batch_decode() && output.batch.len() > 1 {
            // True batched decode only when there are multiple sequences.
            // For a single sequence the sequential path is far cheaper: it
            // keeps the KV cache resident in the model and avoids the
            // extract→pad→stack→extract GPU-copy cycle that batch decode
            // performs every scheduling round.
            self.step_decode_batch(output.batch);
        } else {
            self.step_decode_sequential(output.batch);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Prefill
    // ─────────────────────────────────────────────────────────

    fn step_prefill(&mut self, seq_id: String) {
        let t0 = Instant::now();

        self.flush_active_batch_decode_session();
        self.swap_in(&seq_id);

        let (input_ids, start_pos, chunk_len, remaining_prompt_tokens) = {
            let seq = self.sequences.get(&seq_id).unwrap();
            let chunk = if self.prefill_chunk_size == usize::MAX {
                seq.next_input_ids().to_vec()
            } else {
                seq.next_prefill_chunk(self.prefill_chunk_size).to_vec()
            };
            let chunk_len = chunk.len();
            let remaining = seq.remaining_prompt_tokens().saturating_sub(chunk_len);
            (chunk, seq.start_pos(), chunk_len, remaining)
        };

        if input_ids.is_empty() {
            self.send_error(&seq_id, "Prefill chunk is empty");
            return;
        }

        let logits = match self.model.forward_step(&input_ids, start_pos) {
            Ok(l) => l,
            Err(e) => {
                self.send_error(&seq_id, &format!("Prefill forward failed: {e}"));
                return;
            }
        };

        let prefill_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_prefill_time_us
            .fetch_add(prefill_us, Ordering::Relaxed);
        self.stats
            .total_prefill_chunks
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_prefill_chunk_time_us
            .fetch_add(prefill_us, Ordering::Relaxed);

        {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            seq.advance_prefill_cursor(chunk_len);
        }
        let cached_tokens = self
            .sequences
            .get(&seq_id)
            .map(|seq| seq.start_pos())
            .unwrap_or(0);
        if let Err(e) = self.sync_sequence_paged_kv(&seq_id, cached_tokens) {
            warn!(id = %seq_id, error = %e, "Shadow paged-KV tracking failed after prefill");
        }

        if self
            .sequences
            .get(&seq_id)
            .map(|seq| !seq.is_prefill_complete())
            .unwrap_or(false)
        {
            debug!(
                id = %seq_id,
                chunk_len,
                remaining_prompt_tokens,
                prefill_ms = prefill_us / 1000,
                "Prefill chunk complete",
            );
            self.recount_kv_bytes();
            self.scheduler.requeue_waiting_front(seq_id);
            return;
        }

        let t_sampling = Instant::now();
        let next_token = {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            match sampling::sample(&seq_id, seq, &logits, &mut self.sampling_buffers) {
                Ok(t) => t,
                Err(e) => {
                    self.send_error(&seq_id, &format!("Sampling failed: {e}"));
                    return;
                }
            }
        };
        let sampling_us = t_sampling.elapsed().as_micros() as u64;
        self.stats
            .total_sampling_time_us
            .fetch_add(sampling_us, Ordering::Relaxed);

        self.swap_out(&seq_id);

        let prompt_len = self
            .sequences
            .get(&seq_id)
            .map(|seq| seq.prompt_len)
            .unwrap_or(chunk_len);
        let prefill_tok_s = if prefill_us > 0 {
            (chunk_len as f64) / (prefill_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            seq.tokens.push(next_token);
            seq.status = SequenceStatus::Running;
        }

        info!(
            id = %seq_id,
            prompt_len,
            final_chunk_len = chunk_len,
            prefill_ms = prefill_us / 1000,
            prefill_tok_s = format!("{:.1}", prefill_tok_s),
            "Prefill complete, first token generated",
        );

        self.send_token(&seq_id, next_token);

        if self.sequences.get(&seq_id).unwrap().should_stop() {
            self.finish_sequence(&seq_id);
        } else {
            self.scheduler.promote_to_running(seq_id);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Batched decode
    // ─────────────────────────────────────────────────────────

    /// Decode step for all running sequences — TRUE BATCHED forward.
    ///
    /// Uses **lazy eviction**: when a sequence completes or is cancelled
    /// mid-loop, it stays in the batch tensor (wasting trivial compute)
    /// rather than triggering an expensive extract→re-setup cycle.
    fn step_decode_batch(&mut self, batch: Vec<String>) {
        let t0 = Instant::now();

        if self.active_batch_decode.is_some() && !self.can_reuse_active_batch_decode(&batch) {
            self.flush_active_batch_decode_session();
        }

        // Filter cancelled sequences.
        let cancelled: Vec<String> = batch
            .iter()
            .filter(|id| {
                self.sequences
                    .get(id.as_str())
                    .map_or(true, |s| s.response_tx.is_closed())
            })
            .cloned()
            .collect();
        for id in &cancelled {
            warn!(id = %id, "Client disconnected before decode batch");
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(id);
        }
        let batch: Vec<String> = batch
            .into_iter()
            .filter(|id| !cancelled.contains(id))
            .collect();
        if batch.is_empty() {
            return;
        }

        let batch_size = batch.len();
        let reuse_session = self.can_reuse_active_batch_decode(&batch);
        let mut kv_lens = self.batch_seq_lens(&batch);
        let mut original_max_kv = kv_lens.iter().copied().max().unwrap_or(0);

        let setup_start = Instant::now();
        if !reuse_session {
            // Flush model's single-sequence KV cache state before switching to
            // a batched decode session.
            if let Some(ref prev_id) = self.active_seq_id.take() {
                if self.sequences.contains_key(prev_id) {
                    let caches = self.model.get_kv_caches();
                    if let Some(seq) = self.sequences.get_mut(prev_id) {
                        seq.kv_caches = caches;
                    }
                }
                self.model.clear_kv_cache();
            }
            self.recount_kv_bytes();

            let kv_caches: Vec<Vec<Option<(Tensor, Tensor)>>> = batch
                .iter()
                .map(|id| self.sequences.get(id).unwrap().kv_caches.clone())
                .collect();

            match self
                .model
                .setup_batch_decode(&kv_caches, self.decode_tokens_per_seq)
            {
                Ok((setup_lens, setup_max_kv)) => {
                    kv_lens = setup_lens;
                    original_max_kv = setup_max_kv;
                }
                Err(e) => {
                    error!("Batch decode setup failed: {e}");
                    for seq_id in &batch {
                        self.send_error(seq_id, &format!("Batch decode setup failed: {e}"));
                    }
                    return;
                }
            };
            drop(kv_caches);

            // setup_batch_decode has consumed the existing per-sequence caches.
            // Drop the stale references so the model-owned batched cache is the
            // only live copy until we explicitly flush the session.
            for seq_id in &batch {
                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.kv_caches = vec![None; self.num_layers];
                }
            }
        }
        let setup_us = setup_start.elapsed().as_micros() as u64;
        if !reuse_session {
            self.stats
                .total_batch_decode_setup_time_us
                .fetch_add(setup_us, Ordering::Relaxed);
        }

        let plan_start = Instant::now();
        let (plan, reused_plan) =
            match self.prepare_batch_decode_plan(&batch, &kv_lens, reuse_session) {
                Ok(p) => p,
                Err(e) => {
                    error!("Batch decode plan failed: {e}");
                    self.active_batch_decode = Some(ActiveBatchDecodeSession {
                        batch: batch.clone(),
                        batch_width: original_max_kv,
                        plan: None,
                        backend_name: self.model.decode_backend_name(),
                    });
                    self.flush_active_batch_decode_session();
                    for seq_id in &batch {
                        self.send_error(seq_id, &format!("Batch decode plan failed: {e}"));
                    }
                    return;
                }
            };
        let plan_us = plan_start.elapsed().as_micros() as u64;
        self.stats
            .total_decode_plan_time_us
            .fetch_add(plan_us, Ordering::Relaxed);
        if let Some(ref decode_plan) = plan {
            self.stats
                .total_h2d_metadata_bytes
                .fetch_add(decode_plan.metadata.h2d_metadata_bytes(), Ordering::Relaxed);
            if reused_plan {
                self.stats
                    .total_decode_plan_reuses
                    .fetch_add(1, Ordering::Relaxed);
            }
            let counter = if reused_plan || decode_plan.plan_cache_hit {
                &self.stats.total_decode_plan_cache_hits
            } else {
                &self.stats.total_decode_plan_cache_misses
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }

        // Pre-build attention mask.
        let max_total_width = original_max_kv + self.decode_tokens_per_seq;
        let mask_start = Instant::now();
        let full_mask =
            match self
                .model
                .build_batch_decode_mask(&kv_lens, original_max_kv, max_total_width)
            {
                Ok(m) => m,
                Err(e) => {
                    error!("Mask build failed: {e}");
                    self.active_batch_decode = Some(ActiveBatchDecodeSession {
                        batch: batch.clone(),
                        batch_width: original_max_kv,
                        plan: plan.clone(),
                        backend_name: self.model.decode_backend_name(),
                    });
                    self.flush_active_batch_decode_session();
                    return;
                }
            };
        let mask_us = mask_start.elapsed().as_micros() as u64;
        self.stats
            .total_batch_decode_mask_time_us
            .fetch_add(mask_us, Ordering::Relaxed);

        // Multi-round decode loop with lazy eviction.
        let mut total_tokens_this_step = 0u64;
        let mut rounds_done = 0usize;
        let mut alive = vec![true; batch.len()];
        let mut pending_finish: Vec<String> = Vec::new();
        let mut pending_cancel: Vec<String> = Vec::new();

        let mut positions: Vec<usize> = batch
            .iter()
            .map(|id| self.sequences.get(id).unwrap().start_pos())
            .collect();

        let mut last_tokens: Vec<u32> = batch
            .iter()
            .map(|id| *self.sequences.get(id).unwrap().tokens.last().unwrap())
            .collect();

        for round in 0..self.decode_tokens_per_seq {
            if alive.iter().all(|a| !a) {
                break;
            }

            let tokens: Vec<u32> = (0..batch.len())
                .map(|i| {
                    if alive[i] {
                        *self
                            .sequences
                            .get(&batch[i])
                            .unwrap()
                            .tokens
                            .last()
                            .unwrap()
                    } else {
                        last_tokens[i]
                    }
                })
                .collect();

            let input_ids =
                match crane_core::fused_ops::copy_from_slice_u32(&tokens, self.model.device())
                    .and_then(|t| t.reshape((batch_size, 1)))
                {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Decode input_ids upload failed: {e}");
                        self.model.clear_kv_cache();
                        return;
                    }
                };

            let mask_width = original_max_kv + round + 1;
            let mask_for_round = match &full_mask {
                Some(full) => full.narrow(3, 0, mask_width).ok(),
                None => None,
            };

            let logits = if let Some(ref decode_plan) = plan {
                match self.model.run_planned_batch_decode(
                    decode_plan,
                    &input_ids,
                    &positions,
                    mask_for_round.as_ref(),
                    Some((&kv_lens, original_max_kv)),
                ) {
                    Ok(l) => l,
                    Err(e) => {
                        error!("Planned batch decode forward failed (round {round}): {e}");
                        self.active_batch_decode = Some(ActiveBatchDecodeSession {
                            batch: batch.clone(),
                            batch_width: original_max_kv,
                            plan: plan.clone(),
                            backend_name: self.model.decode_backend_name(),
                        });
                        self.flush_active_batch_decode_session();
                        for (i, seq_id) in batch.iter().enumerate() {
                            if alive[i] {
                                self.send_error(
                                    seq_id,
                                    &format!("Planned batch decode failed: {e}"),
                                );
                            }
                        }
                        return;
                    }
                }
            } else {
                match self.model.step_batch_decode(
                    &input_ids,
                    &positions,
                    mask_for_round.as_ref(),
                    Some((&kv_lens, original_max_kv)),
                ) {
                    Ok(l) => l,
                    Err(e) => {
                        error!("Batched decode forward failed (round {round}): {e}");
                        self.active_batch_decode = Some(ActiveBatchDecodeSession {
                            batch: batch.clone(),
                            batch_width: original_max_kv,
                            plan: plan.clone(),
                            backend_name: self.model.decode_backend_name(),
                        });
                        self.flush_active_batch_decode_session();
                        for (i, seq_id) in batch.iter().enumerate() {
                            if alive[i] {
                                self.send_error(seq_id, &format!("Batched decode failed: {e}"));
                            }
                        }
                        return;
                    }
                }
            };

            rounds_done += 1;

            for (i, seq_id) in batch.iter().enumerate() {
                if !alive[i] {
                    continue;
                }

                let seq_logits = match logits.narrow(0, i, 1) {
                    Ok(l) => l,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Logits extraction failed: {e}"));
                        alive[i] = false;
                        continue;
                    }
                };

                let t_sampling = Instant::now();
                let next_token = {
                    let seq = self.sequences.get_mut(seq_id).unwrap();
                    match sampling::sample(seq_id, seq, &seq_logits, &mut self.sampling_buffers) {
                        Ok(t) => t,
                        Err(e) => {
                            self.send_error(seq_id, &format!("Sampling failed: {e}"));
                            alive[i] = false;
                            continue;
                        }
                    }
                };
                self.stats
                    .total_sampling_time_us
                    .fetch_add(t_sampling.elapsed().as_micros() as u64, Ordering::Relaxed);

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }
                let cached_tokens = self
                    .sequences
                    .get(seq_id)
                    .map(|seq| seq.start_pos())
                    .unwrap_or(0);
                if let Err(e) = self.sync_sequence_paged_kv(seq_id, cached_tokens) {
                    warn!(
                        id = %seq_id,
                        error = %e,
                        "Shadow paged-KV tracking failed during batched decode"
                    );
                }
                last_tokens[i] = next_token;

                total_tokens_this_step += 1;
                self.stats
                    .total_decode_steps
                    .fetch_add(1, Ordering::Relaxed);

                self.send_token(seq_id, next_token);

                if self.sequences.get(seq_id).map_or(true, |s| s.should_stop()) {
                    alive[i] = false;
                    pending_finish.push(seq_id.clone());
                } else if self
                    .sequences
                    .get(seq_id)
                    .map_or(true, |s| s.response_tx.is_closed())
                {
                    warn!(id = %seq_id, "Client disconnected mid-batch-decode");
                    alive[i] = false;
                    pending_cancel.push(seq_id.clone());
                }
            }

            for p in positions.iter_mut() {
                *p += 1;
            }
        }

        let current_seq_lens = self.batch_seq_lens(&batch);
        let current_batch_width = current_seq_lens.iter().copied().max().unwrap_or(0);
        let keep_session =
            rounds_done > 0 && pending_finish.is_empty() && pending_cancel.is_empty();

        if keep_session {
            self.active_batch_decode = Some(ActiveBatchDecodeSession {
                batch: batch.clone(),
                batch_width: current_batch_width,
                plan,
                backend_name: self.model.decode_backend_name(),
            });
            self.active_seq_id = None;
            self.recount_kv_bytes();
        } else if rounds_done > 0 {
            self.active_batch_decode = Some(ActiveBatchDecodeSession {
                batch: batch.clone(),
                batch_width: current_batch_width,
                plan,
                backend_name: self.model.decode_backend_name(),
            });
            self.flush_active_batch_decode_session();
        }

        for id in &pending_finish {
            self.finish_sequence(id);
        }
        for id in &pending_cancel {
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(id);
        }

        let decode_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);

        if total_tokens_this_step > 0 {
            let tok_s = if decode_us > 0 {
                (total_tokens_this_step as f64) / (decode_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            debug!(
                batch_size,
                backend = self.model.decode_backend_name(),
                reuse_session,
                reuse_plan = reused_plan,
                tokens = total_tokens_this_step,
                rounds = rounds_done,
                finished = pending_finish.len(),
                setup_ms = setup_us / 1000,
                plan_ms = plan_us / 1000,
                mask_ms = mask_us / 1000,
                decode_ms = decode_us / 1000,
                tok_s = format!("{:.1}", tok_s),
                "Batched decode step complete",
            );
        }

        self.drain_requests();
        self.check_cancelled();
    }

    // ─────────────────────────────────────────────────────────
    //  Sequential decode
    // ─────────────────────────────────────────────────────────

    /// Sequential decode for backends without batch decode support.
    fn step_decode_sequential(&mut self, batch: Vec<String>) {
        let t0 = Instant::now();
        let mut total_tokens: u64 = 0;
        self.flush_active_batch_decode_session();

        for seq_id in &batch {
            if self
                .sequences
                .get(seq_id)
                .map_or(true, |s| s.response_tx.is_closed())
            {
                self.stats
                    .cancelled_requests
                    .fetch_add(1, Ordering::Relaxed);
                self.cleanup_sequence(seq_id);
                continue;
            }

            self.swap_in(seq_id);

            for _round in 0..self.decode_tokens_per_seq {
                let (input_ids, start_pos) = {
                    let seq = match self.sequences.get(seq_id) {
                        Some(s) => s,
                        None => break,
                    };
                    (seq.next_input_ids().to_vec(), seq.start_pos())
                };

                let logits = match self.model.forward_step(&input_ids, start_pos) {
                    Ok(l) => l,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Decode forward failed: {e}"));
                        break;
                    }
                };

                let t_sampling = Instant::now();
                let next_token = {
                    let seq = self.sequences.get_mut(seq_id).unwrap();
                    match sampling::sample(seq_id, seq, &logits, &mut self.sampling_buffers) {
                        Ok(t) => t,
                        Err(e) => {
                            self.send_error(seq_id, &format!("Sampling failed: {e}"));
                            break;
                        }
                    }
                };
                self.stats
                    .total_sampling_time_us
                    .fetch_add(t_sampling.elapsed().as_micros() as u64, Ordering::Relaxed);

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }
                let cached_tokens = self
                    .sequences
                    .get(seq_id)
                    .map(|seq| seq.start_pos())
                    .unwrap_or(0);
                if let Err(e) = self.sync_sequence_paged_kv(seq_id, cached_tokens) {
                    warn!(
                        id = %seq_id,
                        error = %e,
                        "Shadow paged-KV tracking failed during sequential decode"
                    );
                }

                total_tokens += 1;
                self.stats
                    .total_decode_steps
                    .fetch_add(1, Ordering::Relaxed);

                self.send_token(seq_id, next_token);

                if self.sequences.get(seq_id).map_or(true, |s| s.should_stop()) {
                    self.finish_sequence(seq_id);
                    break;
                }

                if self
                    .sequences
                    .get(seq_id)
                    .map_or(true, |s| s.response_tx.is_closed())
                {
                    warn!(id = %seq_id, "Client disconnected mid-decode");
                    self.stats
                        .cancelled_requests
                        .fetch_add(1, Ordering::Relaxed);
                    self.cleanup_sequence(seq_id);
                    break;
                }
            }

            self.swap_out(seq_id);
        }

        let decode_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);

        if total_tokens > 0 {
            let tok_s = if decode_us > 0 {
                (total_tokens as f64) / (decode_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            debug!(
                tokens = total_tokens,
                decode_ms = decode_us / 1000,
                tok_s = format!("{:.1}", tok_s),
                "Sequential decode step complete",
            );
        }

        self.drain_requests();
        self.check_cancelled();
    }

    // ─────────────────────────────────────────────────────────
    //  KV cache management
    // ─────────────────────────────────────────────────────────

    fn swap_in(&mut self, seq_id: &str) {
        if self.active_seq_id.as_deref() == Some(seq_id) {
            return;
        }

        if !self.model.supports_kv_swap() {
            if self.active_seq_id.as_deref() != Some(seq_id) {
                self.model.clear_kv_cache();
                self.active_seq_id = Some(seq_id.to_string());
            }
            return;
        }

        // Save previous active sequence's KV cache from the model.
        if let Some(ref prev_id) = self.active_seq_id.clone() {
            let caches = self.model.get_kv_caches();
            if let Some(prev_seq) = self.sequences.get_mut(prev_id) {
                prev_seq.kv_caches = caches;
            }
        }

        // Load new sequence's KV cache into the model.
        let caches = self
            .sequences
            .get(seq_id)
            .map(|s| s.kv_caches.clone())
            .unwrap_or_else(|| vec![None; self.num_layers]);
        self.model.set_kv_caches(caches);
        self.active_seq_id = Some(seq_id.to_string());

        self.recount_kv_bytes();
        self.stats
            .total_kv_swap_count
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Mark that the model finished processing `seq_id` for this scheduling
    /// round.  Instead of extracting full KV caches (expensive GPU copies),
    /// we only update byte tracking from the model's internal state.
    /// The actual KV tensors remain in the model and are saved lazily by
    /// `swap_in` when switching to a different sequence.
    fn swap_out(&mut self, seq_id: &str) {
        if !self.model.supports_kv_swap() {
            return;
        }
        if self.active_seq_id.as_deref() != Some(seq_id) {
            return;
        }
        // Drop stale seq cache references (from the last swap_in) to free
        // GPU memory.  swap_in will extract fresh caches from the model
        // when switching to a different sequence.
        if let Some(seq) = self.sequences.get_mut(seq_id) {
            if seq.kv_caches.iter().any(|c| c.is_some()) {
                seq.kv_caches = vec![None; seq.kv_caches.len()];
            }
        }
        self.recount_kv_bytes();
    }

    // ─────────────────────────────────────────────────────────
    //  Response sending
    // ─────────────────────────────────────────────────────────

    fn send_token(&mut self, seq_id: &str, token_id: u32) {
        let text = if let Some(stream) = self.token_streams.get_mut(seq_id) {
            match stream.next_token(token_id) {
                Ok(Some(t)) => t,
                Ok(None) => return,
                Err(e) => {
                    warn!(id = %seq_id, "Token decode error: {e}");
                    return;
                }
            }
        } else {
            return;
        };

        if let Some(seq) = self.sequences.get(seq_id) {
            if seq
                .response_tx
                .send(EngineResponse::Token { text, token_id })
                .is_err()
            {
                debug!(id = %seq_id, "Response channel closed (client disconnected)");
            }
        }
    }

    fn send_error(&mut self, seq_id: &str, msg: &str) {
        error!(id = %seq_id, "Engine error: {msg}");
        if let Some(seq) = self.sequences.get(seq_id) {
            let _ = seq.response_tx.send(EngineResponse::Error(msg.to_string()));
        }
        self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.cleanup_sequence(seq_id);
    }

    fn finish_sequence(&mut self, seq_id: &str) {
        let remaining = self
            .token_streams
            .get(seq_id)
            .and_then(|s| s.decode_rest().ok().flatten())
            .unwrap_or_default();

        if !remaining.is_empty() {
            if let Some(seq) = self.sequences.get(seq_id) {
                let _ = seq.response_tx.send(EngineResponse::Token {
                    text: remaining,
                    token_id: 0,
                });
            }
        }

        if let Some(seq) = self.sequences.get(seq_id) {
            let generated_ids = &seq.tokens[seq.prompt_len..];
            let completion_tokens = seq.num_generated();
            let full_text = self
                .model
                .tokenizer()
                .decode(generated_ids, true)
                .unwrap_or_default();

            let finish_reason = seq.finish_reason().to_string();

            info!(
                id = %seq_id,
                prompt_tokens = seq.prompt_len,
                completion_tokens,
                finish_reason = %finish_reason,
                "Sequence finished",
            );

            let _ = seq.response_tx.send(EngineResponse::Finished {
                full_text,
                prompt_tokens: seq.prompt_len,
                completion_tokens,
                finish_reason,
            });

            self.stats
                .total_completion_tokens
                .fetch_add(completion_tokens as u64, Ordering::Relaxed);
            self.stats
                .completed_requests
                .fetch_add(1, Ordering::Relaxed);
        }

        self.cleanup_sequence(seq_id);
    }

    fn cleanup_sequence(&mut self, seq_id: &str) {
        if self
            .active_batch_decode
            .as_ref()
            .map(|session| session.batch.iter().any(|id| id == seq_id))
            .unwrap_or(false)
        {
            self.flush_active_batch_decode_session();
        }

        // Subtract this sequence's KV bytes from the tracked total.
        // If active, bytes are in the model (not in seq.kv_caches).
        let freed = if self.active_seq_id.as_deref() == Some(seq_id) {
            self.model.active_kv_cache_bytes()
        } else if let Some(seq) = self.sequences.get(seq_id) {
            sequence::kv_cache_bytes(&seq.kv_caches)
        } else {
            0
        };
        self.tracked_kv_bytes = self.tracked_kv_bytes.saturating_sub(freed);

        self.release_sequence_paged_kv(seq_id);
        self.sequences.remove(seq_id);
        self.token_streams.remove(seq_id);
        self.scheduler.remove(seq_id);

        if self.active_seq_id.as_deref() == Some(seq_id) {
            self.active_seq_id = None;
        }
        if self.active_batch_decode.is_none() {
            self.model.clear_kv_cache();
        }

        self.recount_kv_bytes();
        self.maybe_relax_eviction_cap();
        debug!(id = %seq_id, "Sequence cleaned up");
    }
}

impl Drop for InferenceEngine {
    fn drop(&mut self) {
        if let Err(e) = self.model.destroy_decode_backend() {
            warn!("Failed to destroy decode backend state during engine shutdown: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::ModelBackend;
    use candle_core::{DType, Device, Tensor};
    use crane_core::models::qwen3::paged_kv::{KvCacheLayout, PagedKvConfig};
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Arc;
    use tokenizers::{models::bpe::BPE, Tokenizer};
    use tokio::sync::mpsc;

    struct MockPagedBackend {
        device: Device,
        tokenizer: Tokenizer,
        cfg: PagedKvConfig,
    }

    impl MockPagedBackend {
        fn new(device: Device) -> Self {
            Self {
                device,
                tokenizer: Tokenizer::new(BPE::default()),
                cfg: PagedKvConfig {
                    page_size: 16,
                    num_layers: 1,
                    num_kv_heads: 1,
                    head_dim: 1,
                    dtype: DType::F32,
                    layout: KvCacheLayout::Nhd,
                },
            }
        }
    }

    impl ModelBackend for MockPagedBackend {
        fn forward_step(
            &mut self,
            _input_ids: &[u32],
            _start_pos: usize,
        ) -> anyhow::Result<Tensor> {
            anyhow::bail!("unused in tests")
        }

        fn clear_kv_cache(&mut self) {}

        fn num_layers(&self) -> usize {
            1
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn dtype(&self) -> DType {
            DType::F32
        }

        fn tokenizer(&self) -> &Tokenizer {
            &self.tokenizer
        }

        fn eos_token_id(&self) -> Vec<u32> {
            vec![]
        }

        fn warmup(&mut self) {}

        fn supports_kv_swap(&self) -> bool {
            true
        }

        fn paged_kv_config(&self) -> Option<PagedKvConfig> {
            Some(self.cfg.clone())
        }
    }

    #[derive(Default)]
    struct PlanCounters {
        plan_calls: AtomicUsize,
        refresh_calls: AtomicUsize,
        reset_calls: AtomicUsize,
    }

    struct MockPlanBackend {
        inner: MockPagedBackend,
        counters: Arc<PlanCounters>,
    }

    impl MockPlanBackend {
        fn new(device: Device, counters: Arc<PlanCounters>) -> Self {
            Self {
                inner: MockPagedBackend::new(device),
                counters,
            }
        }
    }

    impl ModelBackend for MockPlanBackend {
        fn forward_step(
            &mut self,
            _input_ids: &[u32],
            _start_pos: usize,
        ) -> anyhow::Result<Tensor> {
            anyhow::bail!("unused in tests")
        }

        fn clear_kv_cache(&mut self) {}

        fn num_layers(&self) -> usize {
            self.inner.num_layers()
        }

        fn device(&self) -> &Device {
            self.inner.device()
        }

        fn dtype(&self) -> DType {
            self.inner.dtype()
        }

        fn tokenizer(&self) -> &Tokenizer {
            self.inner.tokenizer()
        }

        fn eos_token_id(&self) -> Vec<u32> {
            self.inner.eos_token_id()
        }

        fn warmup(&mut self) {}

        fn supports_kv_swap(&self) -> bool {
            true
        }

        fn decode_backend_name(&self) -> &'static str {
            "mock-plan"
        }

        fn paged_kv_config(&self) -> Option<PagedKvConfig> {
            self.inner.paged_kv_config()
        }

        fn plan_batch_decode_with_metadata(
            &mut self,
            metadata: &PagedAttentionMetadata,
            decode_tokens_per_seq: usize,
        ) -> candle_core::Result<Option<DecodeBackendPlan>> {
            self.counters
                .plan_calls
                .fetch_add(1, AtomicOrdering::Relaxed);
            Ok(Some(DecodeBackendPlan {
                backend_name: "mock-plan",
                bucket_key: metadata.bucket_key(decode_tokens_per_seq, false),
                metadata: metadata.clone(),
                plan_cache_hit: false,
                graph_eligible: false,
                split_kv: false,
            }))
        }

        fn refresh_batch_decode_plan(
            &mut self,
            plan: &mut DecodeBackendPlan,
            metadata: &PagedAttentionMetadata,
        ) -> candle_core::Result<()> {
            self.counters
                .refresh_calls
                .fetch_add(1, AtomicOrdering::Relaxed);
            let bucket_key =
                metadata.bucket_key(plan.bucket_key.decode_tokens_per_seq, plan.split_kv);
            if bucket_key != plan.bucket_key {
                candle_core::bail!("bucket mismatch")
            }
            plan.metadata = metadata.clone();
            plan.plan_cache_hit = true;
            Ok(())
        }

        fn reset_decode_backend_workspace(&mut self) -> candle_core::Result<()> {
            self.counters
                .reset_calls
                .fetch_add(1, AtomicOrdering::Relaxed);
            Ok(())
        }

        fn destroy_decode_backend(&mut self) -> candle_core::Result<()> {
            Ok(())
        }
    }

    fn make_sequence(id: &str, prompt_len: usize) -> Sequence {
        let (tx, _rx) = mpsc::unbounded_channel();
        Sequence {
            id: id.to_string(),
            status: SequenceStatus::Waiting,
            tokens: vec![1; prompt_len],
            prompt_len,
            prefill_cursor: 0,
            kv_caches: vec![None],
            paged_kv_table: None,
            logits_processor: candle_transformers::generation::LogitsProcessor::new(
                42,
                Some(0.0),
                Some(0.9),
            ),
            temperature: Some(0.0),
            top_p: Some(0.9),
            top_k: Some(20),
            max_tokens: 128,
            eos_token_id: vec![],
            repetition_penalty: 1.0,
            repeat_last_n: 64,
            response_tx: tx,
        }
    }

    #[test]
    fn shadow_paged_kv_pool_uses_max_seq_len_and_decode_headroom() {
        let device = Device::Cpu;
        let backend = Box::new(MockPagedBackend::new(device));
        let memory = MemoryConfig {
            max_seq_len: 128,
            gpu_memory_limit_bytes: 0,
            baseline_gpu_bytes: 0,
        };
        let (engine, _handle) = InferenceEngine::new(backend, 4, 16, 0, memory);

        let pool = engine.paged_kv_pool.as_ref().expect("shadow pool");
        assert_eq!(pool.allocator().capacity_pages(), 36);
    }

    #[test]
    fn prefill_admission_respects_page_budget_and_decode_reserve() {
        let device = Device::Cpu;
        let backend = Box::new(MockPagedBackend::new(device));
        let page_bytes = backend.cfg.total_page_size_bytes();
        let memory = MemoryConfig {
            max_seq_len: 0,
            gpu_memory_limit_bytes: page_bytes * KV_GPU_OVERHEAD_FACTOR * 2,
            baseline_gpu_bytes: 0,
        };
        let (mut engine, _handle) = InferenceEngine::new(backend, 4, 16, 0, memory);

        let mut running = make_sequence("running", 16);
        running.status = SequenceStatus::Running;
        running.tokens.push(2);
        engine.sequences.insert(running.id.clone(), running);
        engine.scheduler.running.push_back("running".to_string());
        engine.sync_sequence_paged_kv("running", 16).unwrap();

        let waiting = make_sequence("waiting", 16);
        engine.sequences.insert(waiting.id.clone(), waiting);
        engine.scheduler.waiting.push_back("waiting".to_string());

        assert!(!engine.can_admit_prefill_by_page_budget("waiting"));
    }

    #[test]
    fn batch_metadata_uses_exact_block_tables_from_shadow_pool() {
        let device = Device::Cpu;
        let backend = Box::new(MockPagedBackend::new(device));
        let memory = MemoryConfig {
            max_seq_len: 128,
            gpu_memory_limit_bytes: 0,
            baseline_gpu_bytes: 0,
        };
        let (mut engine, _handle) = InferenceEngine::new(backend, 4, 16, 0, memory);

        let mut seq_a = make_sequence("a", 17);
        seq_a.status = SequenceStatus::Running;
        let mut seq_b = make_sequence("b", 5);
        seq_b.status = SequenceStatus::Running;
        engine.sequences.insert(seq_a.id.clone(), seq_a);
        engine.sequences.insert(seq_b.id.clone(), seq_b);

        engine.sync_sequence_paged_kv("a", 17).unwrap();
        engine.sync_sequence_paged_kv("b", 5).unwrap();

        let batch = vec!["a".to_string(), "b".to_string()];
        let metadata = engine
            .batch_paged_attention_metadata(&batch)
            .expect("metadata");

        assert_eq!(metadata.seq_lens, vec![17, 5]);
        assert_eq!(metadata.max_kv_pages_per_seq, 2);
        assert_eq!(metadata.total_kv_pages, 3);
        assert_eq!(metadata.abi.paged_kv_indptr, vec![0, 2, 3]);
        assert_eq!(metadata.abi.paged_kv_indices, vec![0, 1, 2]);
        assert_eq!(metadata.abi.paged_kv_last_page_len, vec![1, 5]);
        assert_eq!(metadata.abi.block_tables, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn prepare_batch_decode_plan_reuses_session_plan_when_bucket_matches() {
        let counters = Arc::new(PlanCounters::default());
        let backend = Box::new(MockPlanBackend::new(Device::Cpu, counters.clone()));
        let memory = MemoryConfig {
            max_seq_len: 128,
            gpu_memory_limit_bytes: 0,
            baseline_gpu_bytes: 0,
        };
        let (mut engine, _handle) = InferenceEngine::new(backend, 4, 16, 0, memory);

        let mut seq_a = make_sequence("a", 17);
        seq_a.status = SequenceStatus::Running;
        let mut seq_b = make_sequence("b", 5);
        seq_b.status = SequenceStatus::Running;
        engine.sequences.insert(seq_a.id.clone(), seq_a);
        engine.sequences.insert(seq_b.id.clone(), seq_b);

        engine.sync_sequence_paged_kv("a", 17).unwrap();
        engine.sync_sequence_paged_kv("b", 5).unwrap();

        let batch = vec!["a".to_string(), "b".to_string()];
        let first_kv_lens = engine.batch_seq_lens(&batch);
        let (first_plan, reused) = engine
            .prepare_batch_decode_plan(&batch, &first_kv_lens, false)
            .unwrap();
        assert!(!reused);
        let first_plan = first_plan.expect("plan");
        assert_eq!(counters.plan_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(counters.refresh_calls.load(AtomicOrdering::Relaxed), 0);

        engine.active_batch_decode = Some(ActiveBatchDecodeSession {
            batch: batch.clone(),
            batch_width: 17,
            plan: Some(first_plan),
            backend_name: "mock-plan",
        });

        engine.sequences.get_mut("a").unwrap().tokens.push(2);
        engine.sequences.get_mut("b").unwrap().tokens.push(2);
        engine.sync_sequence_paged_kv("a", 18).unwrap();
        engine.sync_sequence_paged_kv("b", 6).unwrap();

        let second_kv_lens = engine.batch_seq_lens(&batch);
        let (second_plan, reused) = engine
            .prepare_batch_decode_plan(&batch, &second_kv_lens, true)
            .unwrap();
        assert!(reused);
        let second_plan = second_plan.expect("plan");
        assert_eq!(counters.plan_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(counters.refresh_calls.load(AtomicOrdering::Relaxed), 1);
        assert!(second_plan.plan_cache_hit);
        assert_eq!(second_plan.metadata.seq_lens, vec![18, 6]);
    }

    #[test]
    fn prepare_batch_decode_plan_falls_back_to_full_plan_when_bucket_changes() {
        let counters = Arc::new(PlanCounters::default());
        let backend = Box::new(MockPlanBackend::new(Device::Cpu, counters.clone()));
        let memory = MemoryConfig {
            max_seq_len: 128,
            gpu_memory_limit_bytes: 0,
            baseline_gpu_bytes: 0,
        };
        let (mut engine, _handle) = InferenceEngine::new(backend, 4, 16, 0, memory);

        let mut seq_a = make_sequence("a", 17);
        seq_a.status = SequenceStatus::Running;
        engine.sequences.insert(seq_a.id.clone(), seq_a);
        engine.sync_sequence_paged_kv("a", 17).unwrap();

        let batch = vec!["a".to_string()];
        let first_kv_lens = engine.batch_seq_lens(&batch);
        let (first_plan, _) = engine
            .prepare_batch_decode_plan(&batch, &first_kv_lens, false)
            .unwrap();

        engine.active_batch_decode = Some(ActiveBatchDecodeSession {
            batch: batch.clone(),
            batch_width: 17,
            plan: first_plan,
            backend_name: "mock-plan",
        });

        for _ in 0..16 {
            engine.sequences.get_mut("a").unwrap().tokens.push(2);
        }
        engine.sync_sequence_paged_kv("a", 33).unwrap();

        let second_kv_lens = engine.batch_seq_lens(&batch);
        let (_second_plan, reused) = engine
            .prepare_batch_decode_plan(&batch, &second_kv_lens, true)
            .unwrap();
        assert!(!reused);
        assert_eq!(counters.plan_calls.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(counters.refresh_calls.load(AtomicOrdering::Relaxed), 0);
    }

    #[test]
    fn flush_active_batch_decode_session_resets_backend_workspace() {
        let counters = Arc::new(PlanCounters::default());
        let backend = Box::new(MockPlanBackend::new(Device::Cpu, counters.clone()));
        let memory = MemoryConfig {
            max_seq_len: 128,
            gpu_memory_limit_bytes: 0,
            baseline_gpu_bytes: 0,
        };
        let (mut engine, _handle) = InferenceEngine::new(backend, 4, 16, 0, memory);

        let seq = make_sequence("a", 16);
        engine.sequences.insert("a".to_string(), seq);
        engine.active_batch_decode = Some(ActiveBatchDecodeSession {
            batch: vec!["a".to_string()],
            batch_width: 16,
            plan: None,
            backend_name: "mock-plan",
        });

        engine.flush_active_batch_decode_session();

        assert_eq!(counters.reset_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(
            engine
                .stats
                .total_decode_workspace_resets
                .load(Ordering::Relaxed),
            1
        );
    }
}
