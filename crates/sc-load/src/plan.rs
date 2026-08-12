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
    /// Reserved for phase-schema compatibility; fused compilation reports zero.
    pub compile_same_block_ns: u64,
    #[cfg(feature = "profile")]
    pub compile_merge_runs_ns: u64,
    #[cfg(feature = "profile")]
    pub compile_finalize_ns: u64,
    pub predicted_io_seconds: f64,
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

    pub fn prefetch_step(&self) -> usize {
        self.inner.prefetch_step
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
    pub prefetch_step: usize,
    pub ring_slots: usize,
    /// `ring_slots - 1` for power-of-two rings, otherwise `usize::MAX`.
    pub ring_mask: usize,
    pub output: OutputSpec,
    pub fill: FillOp,
    pub row_bytes: usize,
    pub row_stride: usize,
    pub jobs: Vec<Job>,
    pub groups: Vec<BlockGroup>,
    pub cells: Vec<CellTask>,
    pub completions: Vec<BatchCompletion>,
    pub blocks: Vec<BlockSpec>,
    pub sources: Vec<ReadSource>,
    pub source_plans: Vec<SourcePlan>,
    pub runtime: RuntimeEnvelope,
    pub stats: PlanStats,
}

/// Compile-time summary of the immutable plan properties needed to open a
/// session. Keeping this next to the plan makes `Plan::open` independent of
/// the number of sources and jobs.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RuntimeEnvelope {
    pub all_positioned: bool,
    pub has_fuse_source: bool,
    pub maximum_data_encoded: usize,
    pub maximum_indices_encoded: usize,
    pub maximum_combined_encoded: usize,
    pub maximum_data_decoded: usize,
    pub maximum_indices_decoded: usize,
}

#[derive(Clone)]
pub(crate) struct SourcePlan {
    pub n_cols: usize,
    pub value_dtype: StorageDType,
    pub index: Option<IndexOp>,
    /// CSR source-column target byte offsets; a width-specific sentinel drops a column.
    pub feature_map: Option<CsrMap>,
    /// Dense mapped path, compacted to only the source columns that survive.
    pub dense_map: Option<DenseMap>,
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

    pub(crate) fn can_decode_direct(&self) -> bool {
        self.index.is_none() && self.dense_map.is_none() && self.convert.is_identity()
    }
}

pub(crate) const UNMAPPED_TARGET: usize = usize::MAX;
pub(crate) const UNMAPPED_TARGET_U32: u32 = u32::MAX;

#[derive(Clone)]
pub(crate) enum CsrMap {
    Packed32(Arc<[u32]>),
    Wide(Arc<[usize]>),
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

#[derive(Debug, Clone)]
pub(crate) struct Job {
    pub source_plan: u32,
    pub completions: Range<usize>,
    pub groups: Range<usize>,
    pub data: JobSide,
    pub indices: Option<JobSide>,
    pub start_step: usize,
    pub anchor: usize,
    #[cfg(all(feature = "uring", target_os = "linux"))]
    pub batch_min: usize,
    #[cfg(all(feature = "uring", target_os = "linux"))]
    pub batch_max: usize,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct BatchCompletion {
    ring_batch: u32,
    completed: u32,
}

impl BatchCompletion {
    pub(crate) fn new(ring_batch: usize, completed: usize) -> Option<Self> {
        if completed == 0 {
            return None;
        }
        Some(Self {
            ring_batch: u32::try_from(ring_batch).ok()?,
            completed: u32::try_from(completed).ok()?,
        })
    }

    #[inline(always)]
    pub(crate) fn ring_batch(self) -> usize {
        self.ring_batch as usize
    }

    #[inline(always)]
    pub(crate) fn completed(self) -> usize {
        self.completed as usize
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BlockGroup {
    data_block_plus_one: usize,
    indices_block_plus_one: usize,
    pub cells: Range<usize>,
}

impl BlockGroup {
    pub(crate) fn new(
        data_block: Option<usize>,
        indices_block: Option<usize>,
        cells: Range<usize>,
    ) -> Option<Self> {
        Some(Self {
            data_block_plus_one: match data_block {
                Some(block) => block.checked_add(1)?,
                None => 0,
            },
            indices_block_plus_one: match indices_block {
                Some(block) => block.checked_add(1)?,
                None => 0,
            },
            cells,
        })
    }

    pub(crate) fn data_block(&self) -> Option<usize> {
        self.data_block_plus_one.checked_sub(1)
    }

    pub(crate) fn indices_block(&self) -> Option<usize> {
        self.indices_block_plus_one.checked_sub(1)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct JobSide {
    pub source: usize,
    pub read_range: Range<u64>,
    #[cfg(feature = "profile")]
    pub blocks: Range<usize>,
}

impl JobSide {
    pub fn encoded_len(&self, source: &ReadSource) -> usize {
        match source {
            ReadSource::Empty => 0,
            ReadSource::Positioned { .. } | ReadSource::RangeKey { .. } => {
                usize::try_from(self.read_range.end.saturating_sub(self.read_range.start))
                    .unwrap_or(usize::MAX)
            }
            ReadSource::WholeKey { declared_len, .. } => *declared_len,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BlockSpec {
    pub decoder: BlockDecoder,
    encoded_start: u32,
    encoded_end: u32,
    /// Zero for scratch decode; otherwise final output byte offset plus one.
    direct_output_plus_one: usize,
}

impl BlockSpec {
    pub(crate) fn new(
        decoder: BlockDecoder,
        encoded: Range<usize>,
        direct_output_plus_one: usize,
    ) -> Option<Self> {
        if encoded.end.checked_sub(encoded.start)? != decoder.encoded_len() {
            return None;
        }
        Some(Self {
            decoder,
            encoded_start: u32::try_from(encoded.start).ok()?,
            encoded_end: u32::try_from(encoded.end).ok()?,
            direct_output_plus_one,
        })
    }

    pub(crate) fn encoded_range(&self) -> Range<usize> {
        self.encoded_start as usize..self.encoded_end as usize
    }

    pub(crate) fn decoded_len(&self) -> usize {
        self.decoder.decoded_len()
    }

    pub(crate) fn direct_output(&self) -> Option<usize> {
        self.direct_output_plus_one.checked_sub(1)
    }
}

#[derive(Debug, Clone, Copy)]
/// Compact immutable task descriptor. Read-only tasks intentionally share
/// cache lines; unlike counters, they cannot create false sharing. Output rows
/// are 64-byte aligned, so the two low offset bits carry immutable task flags.
#[repr(C)]
pub(crate) struct CellTask {
    row_offset_and_flags: usize,
    data_start: u32,
    data_end: u32,
    indices_start: u32,
    indices_end: u32,
}

impl CellTask {
    const FRESH_OUTPUT: usize = 1;
    const DIRECT_DECODE: usize = 2;
    const FLAGS: usize = Self::FRESH_OUTPUT | Self::DIRECT_DECODE;

    pub(crate) fn new(
        output: OutputSlot,
        data: Range<usize>,
        indices: Option<Range<usize>>,
    ) -> Option<Self> {
        let row_offset = output.row_offset();
        if data.start > data.end || row_offset & Self::FLAGS != 0 {
            return None;
        }
        let indices = indices.unwrap_or(0..0);
        if indices.start > indices.end {
            return None;
        }
        Some(Self {
            row_offset_and_flags: row_offset
                | (usize::from(output.is_fresh()) * Self::FRESH_OUTPUT),
            data_start: u32::try_from(data.start).ok()?,
            data_end: u32::try_from(data.end).ok()?,
            indices_start: u32::try_from(indices.start).ok()?,
            indices_end: u32::try_from(indices.end).ok()?,
        })
    }

    #[inline(always)]
    pub(crate) fn row_offset(self) -> usize {
        self.row_offset_and_flags & !Self::FLAGS
    }

    #[inline(always)]
    pub(crate) fn data_range(self) -> Range<usize> {
        self.data_start as usize..self.data_end as usize
    }

    #[inline(always)]
    pub(crate) fn indices_range(self) -> Range<usize> {
        self.indices_start as usize..self.indices_end as usize
    }

    #[inline(always)]
    pub(crate) fn output_is_fresh(self) -> bool {
        self.row_offset_and_flags & Self::FRESH_OUTPUT != 0
    }

    pub(crate) fn with_direct_decode(mut self) -> Self {
        self.row_offset_and_flags |= Self::DIRECT_DECODE;
        self
    }

    pub(crate) fn is_direct_decode(self) -> bool {
        self.row_offset_and_flags & Self::DIRECT_DECODE != 0
    }
}
