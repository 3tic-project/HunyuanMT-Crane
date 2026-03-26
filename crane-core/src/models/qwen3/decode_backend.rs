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

    fn refresh_plan(
        &mut self,
        plan: &mut DecodeBackendPlan,
        metadata: &PagedAttentionMetadata,
    ) -> Result<()> {
        let refreshed = self.plan(metadata, plan.bucket_key.decode_tokens_per_seq)?;
        *plan = refreshed;
        Ok(())
    }

    fn reset_workspace(&mut self) -> Result<()> {
        Ok(())
    }

    fn destroy(&mut self) -> Result<()> {
        self.reset_workspace()
    }

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

pub struct TensorDecodeBackend {
    last_bucket: Option<DecodeBucketKey>,
    active_workspace_bucket: Option<DecodeBucketKey>,
}

impl Default for TensorDecodeBackend {
    fn default() -> Self {
        Self {
            last_bucket: None,
            active_workspace_bucket: None,
        }
    }
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
        self.active_workspace_bucket = Some(bucket_key);
        Ok(DecodeBackendPlan {
            backend_name: self.name(),
            bucket_key,
            metadata: metadata.clone(),
            plan_cache_hit,
            graph_eligible: false,
            split_kv: false,
        })
    }

    fn refresh_plan(
        &mut self,
        plan: &mut DecodeBackendPlan,
        metadata: &PagedAttentionMetadata,
    ) -> Result<()> {
        let bucket_key = metadata.bucket_key(plan.bucket_key.decode_tokens_per_seq, plan.split_kv);
        if plan.backend_name != self.name() {
            candle_core::bail!(
                "decode plan backend mismatch: expected {}, got {}",
                self.name(),
                plan.backend_name
            );
        }
        if plan.bucket_key != bucket_key {
            candle_core::bail!(
                "decode plan bucket mismatch: cached {:?}, current {:?}",
                plan.bucket_key,
                bucket_key
            );
        }
        self.last_bucket = Some(bucket_key);
        self.active_workspace_bucket = Some(bucket_key);
        plan.metadata = metadata.clone();
        plan.plan_cache_hit = true;
        Ok(())
    }

    fn reset_workspace(&mut self) -> Result<()> {
        self.active_workspace_bucket = None;
        Ok(())
    }

    fn destroy(&mut self) -> Result<()> {
        self.active_workspace_bucket = None;
        self.last_bucket = None;
        Ok(())
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

    #[test]
    fn tensor_backend_refreshes_metadata_without_replanning_bucket() {
        let meta = PagedAttentionMetadata::from_seq_lens(&[17, 5], 16);
        let refreshed_meta = PagedAttentionMetadata::from_seq_lens(&[18, 6], 16);
        let mut backend = TensorDecodeBackend::default();

        let mut plan = backend.plan(&meta, 8).unwrap();
        backend.refresh_plan(&mut plan, &refreshed_meta).unwrap();

        assert!(plan.plan_cache_hit);
        assert_eq!(plan.bucket_key, refreshed_meta.bucket_key(8, false));
        assert_eq!(plan.metadata.seq_lens, vec![18, 6]);
        assert_eq!(backend.active_workspace_bucket, Some(plan.bucket_key));
    }

    #[test]
    fn tensor_backend_reset_workspace_keeps_plan_cache_but_drops_live_workspace() {
        let meta = PagedAttentionMetadata::from_seq_lens(&[32, 48], 16);
        let mut backend = TensorDecodeBackend::default();

        let plan = backend.plan(&meta, 8).unwrap();
        assert_eq!(backend.active_workspace_bucket, Some(plan.bucket_key));

        backend.reset_workspace().unwrap();
        assert_eq!(backend.active_workspace_bucket, None);
        assert_eq!(backend.last_bucket, Some(plan.bucket_key));

        backend.destroy().unwrap();
        assert_eq!(backend.active_workspace_bucket, None);
        assert_eq!(backend.last_bucket, None);
    }
}
