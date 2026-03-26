use candle_core::{Result, Tensor};

use super::modeling::Qwen3Model;
use super::paged_kv::{DecodeBucketKey, PagedAttentionMetadata};

pub const DEFAULT_KV_PAGE_SIZE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeBackendKind {
    Tensor,
}

#[derive(Debug, Clone)]
pub struct DecodeBackendPlan {
    pub backend_name: &'static str,
    pub bucket_key: DecodeBucketKey,
    pub metadata: PagedAttentionMetadata,
    pub plan_cache_hit: bool,
    pub graph_eligible: bool,
    pub split_kv: bool,
}

pub trait Qwen3DecodeBackend: Send {
    fn name(&self) -> &'static str;

    fn kind(&self) -> DecodeBackendKind;

    fn plan(
        &mut self,
        metadata: &PagedAttentionMetadata,
        decode_tokens_per_seq: usize,
    ) -> Result<DecodeBackendPlan>;

    fn run(
        &mut self,
        model: &mut Qwen3Model,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
        plan: &DecodeBackendPlan,
    ) -> Result<Tensor>;
}

#[derive(Default)]
pub struct TensorDecodeBackend {
    last_bucket: Option<DecodeBucketKey>,
}

impl Qwen3DecodeBackend for TensorDecodeBackend {
    fn name(&self) -> &'static str {
        "tensor"
    }

    fn kind(&self) -> DecodeBackendKind {
        DecodeBackendKind::Tensor
    }

    fn plan(
        &mut self,
        metadata: &PagedAttentionMetadata,
        decode_tokens_per_seq: usize,
    ) -> Result<DecodeBackendPlan> {
        let bucket_key = metadata.bucket_key(decode_tokens_per_seq, false);
        let plan_cache_hit = self.last_bucket == Some(bucket_key);
        self.last_bucket = Some(bucket_key);
        Ok(DecodeBackendPlan {
            backend_name: self.name(),
            bucket_key,
            metadata: metadata.clone(),
            plan_cache_hit,
            graph_eligible: false,
            split_kv: false,
        })
    }

    fn run(
        &mut self,
        model: &mut Qwen3Model,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
        _plan: &DecodeBackendPlan,
    ) -> Result<Tensor> {
        model.step_batch_decode(input_ids, positions, attention_mask, batch_kv_info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3::paged_kv::PagedAttentionMetadata;

    #[test]
    fn tensor_backend_reuses_bucket_key_as_plan_cache() {
        let meta = PagedAttentionMetadata::from_seq_lens(&[32, 48], 16);
        let mut backend = TensorDecodeBackend::default();

        let first = backend.plan(&meta, 8).unwrap();
        assert!(!first.plan_cache_hit);
        assert_eq!(first.backend_name, "tensor");
        assert!(!first.graph_eligible);

        let second = backend.plan(&meta, 8).unwrap();
        assert!(second.plan_cache_hit);
        assert_eq!(second.bucket_key, first.bucket_key);
    }
}
