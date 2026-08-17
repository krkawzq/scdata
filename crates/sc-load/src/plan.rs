use std::fs::File;
use std::ops::Range;
use std::sync::Arc;

use dyn_blosc::BlockDecoder;
use sc_compress::ByteStore;

use crate::config::SessionConfig;
use crate::convert::ConvertOp;
use crate::dtype::StorageDType;
use crate::output::OutputSpec;
use crate::scatter::{FillOp, IndexOp};
use crate::session::Session;
#[cfg(all(target_os = "linux", target_has_atomic = "64"))]
use crate::share::{SharedConfig, SharedServer};
use crate::source::OutputSlot;
use crate::Result;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanStats {
    pub input_rows: usize,
    pub block_jobs: usize,
    pub jobs: usize,
    pub data_io_ops: u64,
    pub indices_io_ops: u64,
    pub predicted_physical_bytes: u64,
    pub gap_bytes: u64,
    pub maximum_encoded_bytes_per_side: usize,
    pub maximum_decoded_bytes_per_job: usize,
    pub arena_bytes: usize,
    pub compile_working_set_bytes: usize,
    pub retained_whole_key_bytes: usize,
    pub output_ring_bytes: usize,
    pub compile_time_io_bytes: u64,
    pub compile_time_io_ops: u64,
    #[cfg(feature = "profile")]
    /// Fused cell resolution, candidate compaction, and same-block grouping.
    pub compile_resolve_ns: u64,
    #[cfg(feature = "profile")]
    pub compile_finalize_ns: u64,
    pub predicted_io_seconds: f64,
    pub cache_capacity_bytes: usize,
    pub cache_arena_bytes: usize,
    pub cache_alignment_loss_bytes: usize,
    pub unique_cache_objects: usize,
    pub residency_loads: usize,
    pub residency_reloads: usize,
    pub cache_reference_hits: usize,
    pub cache_reference_misses: usize,
    pub cache_capacity_stalls: usize,
    pub cache_fragmentation_stalls: usize,
    pub cache_horizon_max_batches: usize,
    pub output_ring_slots: usize,
    pub initialize_io_tasks: usize,
    pub executable_tasks: usize,
    pub dependency_edges: usize,
    pub independent_block_loads: usize,
    pub fused_io_tasks: usize,
    pub predicted_io_ops_saved: usize,
    pub io_payload_bytes: u64,
    pub io_span_bytes: u64,
    pub io_read_amplification: f64,
    pub maximum_decode_ops_per_io_task: usize,
    pub maximum_decoded_bytes_per_io_task: usize,
    pub initialize_fused_io_tasks: usize,
    pub regular_fused_io_tasks: usize,
}

#[derive(Clone)]
pub struct Plan {
    pub(crate) inner: Arc<PlanData>,
}

impl Plan {
    pub fn open(&self, config: SessionConfig) -> Result<Session> {
        Session::start(Arc::clone(&self.inner), config)
    }

    /// Open a shared-ring producer for multi-rank consumers.
    ///
    /// Linux only (`memfd` + futex). The standard [`Self::open`] path is unchanged.
    #[cfg(all(target_os = "linux", target_has_atomic = "64"))]
    pub fn open_shared(&self, config: SessionConfig, shared: SharedConfig) -> Result<SharedServer> {
        SharedServer::open(self, config, shared)
    }

    pub fn stats(&self) -> &PlanStats {
        &self.inner.stats
    }

    pub fn batch_size(&self) -> usize {
        self.inner.batch_size
    }

    pub fn batch_count(&self) -> usize {
        self.inner.batch_count
    }

    /// Number of output-ring generations resident at once.
    pub fn prefetch_step(&self) -> usize {
        self.inner.ring_slots
    }

    pub fn output_spec(&self) -> &OutputSpec {
        &self.inner.output
    }

    pub fn row_stride_bytes(&self) -> usize {
        self.inner.row_stride
    }

    pub fn is_empty(&self) -> bool {
        self.inner.batch_count == 0
    }
}

pub(crate) struct PlanData {
    pub batch_size: usize,
    pub batch_count: usize,
    pub ring_slots: usize,
    /// `ring_slots - 1` for power-of-two rings, otherwise `usize::MAX`.
    pub ring_mask: usize,
    pub output: OutputSpec,
    pub fill: FillOp,
    pub row_bytes: usize,
    pub row_stride: usize,
    pub sources: Vec<ReadSource>,
    pub source_plans: Vec<SourcePlan>,
    pub stats: PlanStats,
    pub static_plan: StaticPlanData,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CacheSlice {
    pub offset: usize,
    pub len: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputSlice {
    pub ring_offset: usize,
    pub len: usize,
    pub generation: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct InitializeJob {
    pub io_tasks: Range<usize>,
    pub decoded_bytes: usize,
    pub io_bytes: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct StaticJob {
    pub batch_id: u64,
    pub io_tasks: Range<usize>,
    pub csr_tasks: Range<usize>,
    pub dense_tasks: Range<usize>,
    pub output_slot: usize,
    pub output_generation: u64,
    pub completion_node: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct IoDecodeLoadTask {
    pub source: usize,
    pub file_offset: u64,
    pub file_len: usize,
    pub decode_ops: Range<usize>,
    pub earliest_consumer_batch: u64,
    pub available_after_batch: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodeOp {
    pub encoded_offset: usize,
    pub encoded_len: usize,
    pub decoder: BlockDecoder,
    pub cache: CacheSlice,
    pub ready_node: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct DenseScatterTask {
    pub data: Option<CacheSlice>,
    pub cell: CellTask,
    pub output: OutputSlice,
    pub source_plan: usize,
    pub completion_node: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct CsrScatterTask {
    pub data: CacheSlice,
    pub indices: CacheSlice,
    pub cell: CellTask,
    pub output: OutputSlice,
    pub source_plan: usize,
    pub completion_node: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DependencyGraph {
    pub initial_dependency_count: Box<[u32]>,
    pub block_ready_ranges: Box<[Range<usize>]>,
    pub block_ready_successors: Box<[usize]>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ReleasePlan {
    pub release_ranges: Box<[Range<usize>]>,
    pub released_nodes: Box<[usize]>,
}

#[derive(Clone)]
pub(crate) struct StaticPlanData {
    pub initialize: InitializeJob,
    pub jobs: Box<[StaticJob]>,
    pub io_decode_tasks: Box<[IoDecodeLoadTask]>,
    pub decode_ops: Box<[DecodeOp]>,
    pub csr_scatter_tasks: Box<[CsrScatterTask]>,
    pub dense_scatter_tasks: Box<[DenseScatterTask]>,
    pub dependencies: DependencyGraph,
    pub prefix_releases: ReleasePlan,
    pub ring_releases: ReleasePlan,
    pub cache_capacity: usize,
    pub cache_alignment: usize,
}

#[derive(Clone)]
pub(crate) struct SourcePlan {
    pub n_cols: usize,
    pub value_dtype: StorageDType,
    pub index: Option<IndexOp>,
    /// CSR source-column target byte offsets; a width-specific sentinel drops a column.
    pub feature_map: Option<CsrMap>,
    /// Optional sparse source/target list for binary lookup on high-nnz rows.
    pub csr_sparse_map: Option<CsrSparseMap>,
    /// Dense mapped path, compacted to only the source columns that survive.
    pub dense_map: Option<DenseMap>,
    /// Dense maps with highly fragmented gaps use one streaming whole-row fill.
    pub dense_fill_whole: bool,
    /// Contiguous unmapped byte ranges for CSR and low-fragmentation Dense maps.
    pub default_ranges: Arc<[OutputRange]>,
    pub convert: ConvertOp,
}

impl SourcePlan {
    #[inline(always)]
    pub(crate) fn requires_runtime_validation(&self) -> bool {
        // CSR indices are data-dependent. Dense structure and mapping extents
        // are compiler-sealed, so only conversions using Error overflow policy
        // require a row scan before the unchecked kernel.
        self.index.is_some() || self.convert.can_fail()
    }
}

pub(crate) const UNMAPPED_TARGET: usize = usize::MAX;
pub(crate) const UNMAPPED_TARGET_U32: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputRange {
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

#[derive(Clone)]
pub(crate) enum CsrMap {
    Packed32(Arc<[u32]>),
    Wide(Arc<[usize]>),
}

#[derive(Clone)]
pub(crate) enum CsrSparseMap {
    /// Low 32 bits are source column; high 32 bits are target byte offset.
    Packed32(Arc<[u64]>),
    Wide(Arc<[CsrSparseMapEntry]>),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CsrSparseMapEntry {
    pub(crate) source_column: usize,
    pub(crate) target_byte: usize,
}

#[inline]
pub(crate) fn csr_sparse_binary_is_cheaper(mapped: usize, nnz: usize) -> bool {
    if mapped == 0 || nnz == 0 {
        return false;
    }
    let comparisons = usize::BITS as usize - (nnz - 1).leading_zeros() as usize;
    // One binary-search comparison is a random index read, while the dense
    // path performs one predictable sequential map lookup per nnz. The factor
    // keeps sparse lookup for cases with a clear cost margin on real CSR rows.
    mapped.saturating_mul(comparisons.max(1)).saturating_mul(8) < nnz
}

#[derive(Clone)]
pub(crate) enum DenseMap {
    /// Low 32 bits are source-byte offset; high 32 bits are target-byte offset.
    Packed32 {
        entries: Arc<[u64]>,
        covers_output: bool,
    },
    /// Signed 32-bit source-byte offsets gathered into one contiguous target run.
    Gather32 {
        source_offsets: Arc<[i32]>,
        target_byte: u32,
        covers_output: bool,
    },
    Wide {
        entries: Arc<[DenseMapEntry]>,
        covers_output: bool,
    },
    Runs {
        entries: Arc<[DenseMapRun]>,
        covers_output: bool,
    },
}

impl DenseMap {
    #[inline(always)]
    pub(crate) fn covers_output(&self) -> bool {
        match self {
            Self::Packed32 { covers_output, .. }
            | Self::Gather32 { covers_output, .. }
            | Self::Wide { covers_output, .. }
            | Self::Runs { covers_output, .. } => *covers_output,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseMapEntry {
    pub source_byte: usize,
    pub target_byte: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseMapRun {
    pub source_byte: usize,
    pub target_byte: usize,
    pub count: usize,
}

#[derive(Clone)]
pub(crate) enum ReadSource {
    Empty,
    Positioned {
        file: Arc<File>,
        base_offset: u64,
        view_len: u64,
    },
    RangeKey {
        store: Arc<dyn ByteStore>,
        key: Arc<str>,
        declared_len: usize,
    },
    WholeKey {
        store: Arc<dyn ByteStore>,
        key: Arc<str>,
        declared_len: usize,
        cached: Option<Arc<[u8]>>,
    },
}

#[derive(Debug, Clone, Copy)]
/// Compact row-local input ranges plus a 64-byte-aligned output offset.
#[repr(C)]
pub(crate) struct CellTask {
    row_offset: usize,
    data_start: u32,
    data_end: u32,
    indices_start: u32,
    indices_end: u32,
}

impl CellTask {
    pub(crate) fn new(
        output: OutputSlot,
        data: Range<usize>,
        indices: Option<Range<usize>>,
    ) -> Option<Self> {
        let row_offset = output.row_offset();
        if data.start > data.end || row_offset & 1 != 0 {
            return None;
        }
        let indices = indices.unwrap_or(0..0);
        if indices.start > indices.end {
            return None;
        }
        Some(Self {
            row_offset,
            data_start: u32::try_from(data.start).ok()?,
            data_end: u32::try_from(data.end).ok()?,
            indices_start: u32::try_from(indices.start).ok()?,
            indices_end: u32::try_from(indices.end).ok()?,
        })
    }

    #[inline(always)]
    pub(crate) fn row_offset(self) -> usize {
        self.row_offset
    }

    #[inline(always)]
    pub(crate) fn data_range(self) -> Range<usize> {
        self.data_start as usize..self.data_end as usize
    }

    #[inline(always)]
    pub(crate) fn indices_range(self) -> Range<usize> {
        self.indices_start as usize..self.indices_end as usize
    }
}
