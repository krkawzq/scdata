use super::*;

#[derive(Debug, Clone)]
pub(crate) struct SparseBatchPlan {
    pub(crate) index_pieces: Vec<SparseReadPiece>,
    pub(crate) data_pieces: Vec<SparseReadPiece>,
    pub(crate) index_groups: Vec<SparseReadGroup>,
    pub(crate) data_groups: Vec<SparseReadGroup>,
    pub(crate) index_bytes: usize,
    pub(crate) projected_indices: Option<ProjectedIndexBuffer>,
}

#[derive(Debug, Clone)]
pub(crate) struct SparseReadPiece {
    pub(crate) chunk: ChunkRef,
    pub(crate) source: ByteRange,
    pub(crate) group_offset: usize,
    pub(crate) output_offset: usize,
    pub(crate) output_row: usize,
    pub(crate) index_offset: usize,
    pub(crate) elements: usize,
    pub(crate) bytes: usize,
    pub(crate) projection_filtered: bool,
    pub(crate) contiguous_output_start: Option<usize>,
    pub(crate) projected_index_offset: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) enum ProjectedIndexBuffer {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

#[derive(Debug, Clone)]
pub(crate) enum SparseGroupSource {
    AccessItem(AccessItem),
    Memory {
        bytes: Arc<[u8]>,
        codec: SharedCodec,
        expected_size: usize,
        decoded: bool,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct SparseReadGroup {
    pub(crate) source: SparseGroupSource,
    pub(crate) slice: SliceSpec,
    pub(crate) slice_ranges: Vec<RangeCopy>,
    pub(crate) parts: Vec<usize>,
    pub(crate) bytes: usize,
}

impl SparseReadGroup {
    pub(crate) fn finalize_slice(&mut self) {
        self.slice = SliceSpec::from_ranges(std::mem::take(&mut self.slice_ranges));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SparseGroupKey {
    File {
        key: ChunkKey,
        codec: usize,
        expected_size: Option<usize>,
    },
    Memory {
        ptr: usize,
        len: usize,
        codec: usize,
        expected_size: usize,
        decoded: bool,
    },
}

pub(crate) trait CsrIndex: Copy {
    fn checked_gene(self) -> DataBankResult<usize>;
    unsafe fn unchecked_gene(self) -> usize;
}

#[derive(Clone, Copy)]
pub(crate) struct SparseProjectionCtx<'a> {
    pub(crate) num_genes: usize,
    pub(crate) output_genes: usize,
    pub(crate) projection: &'a CompiledGeneProjection,
    pub(crate) contiguous_selected_source_range: Option<(usize, usize)>,
    pub(crate) contiguous_selected_source_output_start: Option<(usize, usize)>,
}

impl CsrIndex for u32 {
    fn checked_gene(self) -> DataBankResult<usize> {
        Ok(self as usize)
    }

    unsafe fn unchecked_gene(self) -> usize {
        self as usize
    }
}

impl CsrIndex for i32 {
    fn checked_gene(self) -> DataBankResult<usize> {
        if self < 0 {
            return Err(DataBankError::CsrIndexInvalid(format!(
                "negative i32 index {self}"
            )));
        }
        Ok(self as usize)
    }

    unsafe fn unchecked_gene(self) -> usize {
        self as usize
    }
}

impl CsrIndex for u64 {
    fn checked_gene(self) -> DataBankResult<usize> {
        usize::try_from(self).map_err(|_| {
            DataBankError::CsrIndexInvalid("u64 index does not fit in usize".to_string())
        })
    }

    unsafe fn unchecked_gene(self) -> usize {
        self as usize
    }
}

impl CsrIndex for i64 {
    fn checked_gene(self) -> DataBankResult<usize> {
        if self < 0 {
            return Err(DataBankError::CsrIndexInvalid(format!(
                "negative i64 index {self}"
            )));
        }
        usize::try_from(self as u64).map_err(|_| {
            DataBankError::CsrIndexInvalid("i64 index does not fit in usize".to_string())
        })
    }

    unsafe fn unchecked_gene(self) -> usize {
        self as usize
    }
}
