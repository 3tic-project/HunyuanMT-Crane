use candle_core::DType;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheLayout {
    Nhd,
    Hnd,
}

impl Default for KvCacheLayout {
    fn default() -> Self {
        Self::Nhd
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvConfig {
    pub page_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: DType,
    pub layout: KvCacheLayout,
}

impl PagedKvConfig {
    pub fn bytes_per_token_per_layer(&self) -> u64 {
        let kv_elems = (self.num_kv_heads * self.head_dim * 2) as u64;
        kv_elems * self.dtype.size_in_bytes() as u64
    }

    pub fn bytes_per_page_per_layer(&self) -> u64 {
        self.bytes_per_token_per_layer() * self.page_size as u64
    }

    pub fn total_page_size_bytes(&self) -> u64 {
        self.bytes_per_page_per_layer() * self.num_layers as u64
    }

    pub fn pages_for_tokens(&self, seq_len: usize) -> usize {
        pages_for_tokens(seq_len, self.page_size)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvPageAllocationError {
    OutOfPages {
        requested: usize,
        available: usize,
    },
}

impl fmt::Display for KvPageAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfPages {
                requested,
                available,
            } => write!(
                f,
                "paged-kv allocator ran out of pages: requested {requested}, available {available}"
            ),
        }
    }
}

impl std::error::Error for KvPageAllocationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPageAllocator {
    capacity_pages: usize,
    free_pages: Vec<u32>,
}

impl KvPageAllocator {
    pub fn new(capacity_pages: usize) -> Self {
        let mut free_pages = (0..capacity_pages as u32).collect::<Vec<_>>();
        free_pages.reverse();
        Self {
            capacity_pages,
            free_pages,
        }
    }

    pub fn capacity_pages(&self) -> usize {
        self.capacity_pages
    }

    pub fn available_pages(&self) -> usize {
        self.free_pages.len()
    }

    pub fn allocated_pages(&self) -> usize {
        self.capacity_pages.saturating_sub(self.free_pages.len())
    }

    pub fn alloc(&mut self, num_pages: usize) -> Result<Vec<u32>, KvPageAllocationError> {
        if num_pages > self.free_pages.len() {
            return Err(KvPageAllocationError::OutOfPages {
                requested: num_pages,
                available: self.free_pages.len(),
            });
        }

        let start = self.free_pages.len() - num_pages;
        let mut pages = self.free_pages.split_off(start);
        pages.sort_unstable();
        Ok(pages)
    }

    pub fn free(&mut self, pages: &[u32]) {
        self.free_pages.extend_from_slice(pages);
        self.free_pages.sort_unstable_by(|a, b| b.cmp(a));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqBlockTable {
    page_size: usize,
    token_count: usize,
    blocks: Vec<u32>,
}

impl SeqBlockTable {
    pub fn new(page_size: usize) -> Self {
        Self {
            page_size,
            token_count: 0,
            blocks: Vec::new(),
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn block_indices(&self) -> &[u32] {
        &self.blocks
    }

    pub fn last_page_len(&self) -> usize {
        last_page_len(self.token_count, self.page_size)
    }

    pub fn append_tokens(
        &mut self,
        token_count: usize,
        allocator: &mut KvPageAllocator,
    ) -> Result<(), KvPageAllocationError> {
        if token_count == 0 {
            return Ok(());
        }

        let new_total = self.token_count + token_count;
        let required_pages = pages_for_tokens(new_total, self.page_size);
        let missing_pages = required_pages.saturating_sub(self.blocks.len());
        if missing_pages > 0 {
            let mut fresh_pages = allocator.alloc(missing_pages)?;
            self.blocks.append(&mut fresh_pages);
        }
        self.token_count = new_total;
        Ok(())
    }

    pub fn release(&mut self, allocator: &mut KvPageAllocator) {
        allocator.free(&self.blocks);
        self.blocks.clear();
        self.token_count = 0;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvPool {
    config: PagedKvConfig,
    allocator: KvPageAllocator,
}

impl PagedKvPool {
    pub fn new(config: PagedKvConfig, capacity_pages: usize) -> Self {
        Self {
            config,
            allocator: KvPageAllocator::new(capacity_pages),
        }
    }

    pub fn config(&self) -> &PagedKvConfig {
        &self.config
    }

    pub fn allocator(&self) -> &KvPageAllocator {
        &self.allocator
    }

    pub fn alloc_seq_table(
        &mut self,
        token_count: usize,
    ) -> Result<SeqBlockTable, KvPageAllocationError> {
        let mut table = SeqBlockTable::new(self.config.page_size);
        table.append_tokens(token_count, &mut self.allocator)?;
        Ok(table)
    }

    pub fn append_seq_tokens(
        &mut self,
        table: &mut SeqBlockTable,
        token_count: usize,
    ) -> Result<(), KvPageAllocationError> {
        table.append_tokens(token_count, &mut self.allocator)
    }

    pub fn free_seq_table(&mut self, table: &mut SeqBlockTable) {
        table.release(&mut self.allocator);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendMetadataAbi {
    pub paged_kv_indptr: Vec<u32>,
    pub paged_kv_indices: Vec<u32>,
    pub paged_kv_last_page_len: Vec<u32>,
    pub block_tables: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedAttentionMetadata {
    pub page_size: usize,
    pub seq_lens: Vec<usize>,
    pub max_kv_pages_per_seq: usize,
    pub total_kv_pages: usize,
    pub abi: BackendMetadataAbi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodeBucketKey {
    pub batch_size: usize,
    pub page_size: usize,
    pub max_kv_pages_per_seq: usize,
    pub total_kv_pages: usize,
    pub decode_tokens_per_seq: usize,
    pub split_kv: bool,
}

impl PagedAttentionMetadata {
    pub fn from_block_tables(seq_tables: &[SeqBlockTable], page_size: usize) -> Self {
        let mut indptr = Vec::with_capacity(seq_tables.len() + 1);
        let mut indices = Vec::new();
        let mut last_page_lens = Vec::with_capacity(seq_tables.len());
        let mut block_tables = Vec::with_capacity(seq_tables.len());
        let mut total_pages = 0usize;
        let mut max_kv_pages_per_seq = 0usize;
        let mut seq_lens = Vec::with_capacity(seq_tables.len());

        indptr.push(0);
        for table in seq_tables {
            let blocks = table.block_indices().to_vec();
            total_pages += blocks.len();
            max_kv_pages_per_seq = max_kv_pages_per_seq.max(blocks.len());
            seq_lens.push(table.token_count());
            indices.extend_from_slice(&blocks);
            block_tables.push(blocks);
            indptr.push(total_pages as u32);
            last_page_lens.push(table.last_page_len() as u32);
        }

        Self {
            page_size,
            seq_lens,
            max_kv_pages_per_seq,
            total_kv_pages: total_pages,
            abi: BackendMetadataAbi {
                paged_kv_indptr: indptr,
                paged_kv_indices: indices,
                paged_kv_last_page_len: last_page_lens,
                block_tables,
            },
        }
    }

    pub fn from_seq_lens(seq_lens: &[usize], page_size: usize) -> Self {
        let mut next_page = 0u32;
        let tables = seq_lens
            .iter()
            .map(|&seq_len| {
                let page_count = pages_for_tokens(seq_len, page_size);
                let blocks = (next_page..next_page + page_count as u32).collect::<Vec<_>>();
                next_page += page_count as u32;
                SeqBlockTable {
                    page_size,
                    token_count: seq_len,
                    blocks,
                }
            })
            .collect::<Vec<_>>();
        Self::from_block_tables(&tables, page_size)
    }

    pub fn h2d_metadata_bytes(&self) -> u64 {
        let scalars = self.abi.paged_kv_indptr.len()
            + self.abi.paged_kv_indices.len()
            + self.abi.paged_kv_last_page_len.len();
        (scalars * std::mem::size_of::<u32>()) as u64
    }

    pub fn bucket_key(
        &self,
        decode_tokens_per_seq: usize,
        split_kv: bool,
    ) -> DecodeBucketKey {
        DecodeBucketKey {
            batch_size: self.seq_lens.len(),
            page_size: self.page_size,
            max_kv_pages_per_seq: self.max_kv_pages_per_seq,
            total_kv_pages: self.total_kv_pages,
            decode_tokens_per_seq,
            split_kv,
        }
    }
}

pub fn pages_for_tokens(seq_len: usize, page_size: usize) -> usize {
    if seq_len == 0 || page_size == 0 {
        0
    } else {
        seq_len.div_ceil(page_size)
    }
}

pub fn last_page_len(seq_len: usize, page_size: usize) -> usize {
    if seq_len == 0 || page_size == 0 {
        0
    } else {
        let rem = seq_len % page_size;
        if rem == 0 { page_size } else { rem }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_for_tokens_rounds_up() {
        assert_eq!(pages_for_tokens(0, 16), 0);
        assert_eq!(pages_for_tokens(1, 16), 1);
        assert_eq!(pages_for_tokens(16, 16), 1);
        assert_eq!(pages_for_tokens(17, 16), 2);
    }

    #[test]
    fn last_page_len_uses_page_size_for_exact_multiple() {
        assert_eq!(last_page_len(0, 16), 0);
        assert_eq!(last_page_len(1, 16), 1);
        assert_eq!(last_page_len(16, 16), 16);
        assert_eq!(last_page_len(17, 16), 1);
    }

    #[test]
    fn paged_metadata_builds_indptr_indices_and_tables() {
        let meta = PagedAttentionMetadata::from_seq_lens(&[0, 1, 16, 17], 16);
        assert_eq!(meta.max_kv_pages_per_seq, 2);
        assert_eq!(meta.total_kv_pages, 4);
        assert_eq!(meta.abi.paged_kv_indptr, vec![0, 0, 1, 2, 4]);
        assert_eq!(meta.abi.paged_kv_indices, vec![0, 1, 2, 3]);
        assert_eq!(meta.abi.paged_kv_last_page_len, vec![0, 1, 16, 1]);
        assert_eq!(meta.abi.block_tables[0], Vec::<u32>::new());
        assert_eq!(meta.abi.block_tables[1], vec![0]);
        assert_eq!(meta.abi.block_tables[2], vec![1]);
        assert_eq!(meta.abi.block_tables[3], vec![2, 3]);
    }

    #[test]
    fn paged_kv_config_reports_page_sizes() {
        let cfg = PagedKvConfig {
            page_size: 16,
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            layout: KvCacheLayout::Nhd,
        };
        assert_eq!(cfg.pages_for_tokens(33), 3);
        assert_eq!(cfg.bytes_per_token_per_layer(), 4096);
        assert_eq!(cfg.bytes_per_page_per_layer(), 65536);
        assert_eq!(cfg.total_page_size_bytes(), 1835008);
    }

    #[test]
    fn bucket_key_tracks_page_shape() {
        let meta = PagedAttentionMetadata::from_seq_lens(&[31, 32], 16);
        let key = meta.bucket_key(8, false);
        assert_eq!(key.batch_size, 2);
        assert_eq!(key.page_size, 16);
        assert_eq!(key.max_kv_pages_per_seq, 2);
        assert_eq!(key.total_kv_pages, 4);
        assert_eq!(key.decode_tokens_per_seq, 8);
        assert!(!key.split_kv);
    }

    #[test]
    fn allocator_allocates_and_frees_pages() {
        let mut allocator = KvPageAllocator::new(4);
        assert_eq!(allocator.capacity_pages(), 4);
        assert_eq!(allocator.available_pages(), 4);

        let first = allocator.alloc(2).unwrap();
        assert_eq!(first, vec![0, 1]);
        assert_eq!(allocator.allocated_pages(), 2);
        assert_eq!(allocator.available_pages(), 2);

        allocator.free(&first);
        assert_eq!(allocator.allocated_pages(), 0);
        assert_eq!(allocator.available_pages(), 4);
    }

    #[test]
    fn allocator_errors_when_out_of_pages() {
        let mut allocator = KvPageAllocator::new(1);
        let err = allocator.alloc(2).unwrap_err();
        assert_eq!(
            err,
            KvPageAllocationError::OutOfPages {
                requested: 2,
                available: 1,
            }
        );
    }

    #[test]
    fn seq_block_table_grows_only_when_crossing_page_boundary() {
        let mut allocator = KvPageAllocator::new(8);
        let mut table = SeqBlockTable::new(4);

        table.append_tokens(3, &mut allocator).unwrap();
        assert_eq!(table.token_count(), 3);
        assert_eq!(table.block_indices(), &[0]);
        assert_eq!(table.last_page_len(), 3);

        table.append_tokens(1, &mut allocator).unwrap();
        assert_eq!(table.block_indices(), &[0]);
        assert_eq!(table.last_page_len(), 4);

        table.append_tokens(2, &mut allocator).unwrap();
        assert_eq!(table.block_indices(), &[0, 1]);
        assert_eq!(table.last_page_len(), 2);

        table.release(&mut allocator);
        assert_eq!(table.block_count(), 0);
        assert_eq!(table.token_count(), 0);
        assert_eq!(allocator.available_pages(), 8);
    }

    #[test]
    fn metadata_from_block_tables_preserves_sparse_page_ids() {
        let meta = PagedAttentionMetadata::from_block_tables(
            &[
                SeqBlockTable {
                    page_size: 16,
                    token_count: 32,
                    blocks: vec![4, 9],
                },
                SeqBlockTable {
                    page_size: 16,
                    token_count: 17,
                    blocks: vec![10, 15],
                },
            ],
            16,
        );

        assert_eq!(meta.abi.paged_kv_indptr, vec![0, 2, 4]);
        assert_eq!(meta.abi.paged_kv_indices, vec![4, 9, 10, 15]);
        assert_eq!(meta.abi.block_tables, vec![vec![4, 9], vec![10, 15]]);
        assert_eq!(meta.abi.paged_kv_last_page_len, vec![16, 1]);
    }

    #[test]
    fn paged_kv_pool_allocates_and_recycles_seq_tables() {
        let config = PagedKvConfig {
            page_size: 8,
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::BF16,
            layout: KvCacheLayout::Nhd,
        };
        let mut pool = PagedKvPool::new(config, 8);
        let mut table = pool.alloc_seq_table(9).unwrap();
        assert_eq!(table.block_indices(), &[0, 1]);
        assert_eq!(pool.allocator().available_pages(), 6);

        pool.append_seq_tokens(&mut table, 15).unwrap();
        assert_eq!(table.block_indices(), &[0, 1, 2]);
        assert_eq!(pool.allocator().available_pages(), 5);

        pool.free_seq_table(&mut table);
        assert_eq!(pool.allocator().available_pages(), 8);
        assert_eq!(table.token_count(), 0);
    }
}
