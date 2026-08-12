use std::collections::{hash_map::Entry, HashMap};
use std::mem::size_of;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(feature = "profile")]
use std::time::Instant;

use dyn_blosc::{BlockDecoder, DecodeLimits, Decoder};
use sc_compress::chunk_key;

use crate::config::PlanConfig;
use crate::convert::ConvertOp;
use crate::plan::{
    BatchCompletion, BlockGroup, BlockSpec, CellTask, CsrMap, DenseMap, DenseMapEntry, DenseMapRun,
    Job, JobSide, Plan, PlanData, PlanStats, ReadSource, SourcePlan, UNMAPPED_TARGET,
    UNMAPPED_TARGET_U32,
};
use crate::scatter::{FillOp, IndexOp};
use crate::source::{DatasetKind, OutputSlot};
use crate::{Error, OutputSpec, Result, RowRef, Source, SourceId};

const MAX_DENSE_CHUNK_TABLE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct PlanSpec {
    pub sources: Vec<Source>,
    pub rows: Vec<RowRef>,
    pub output: OutputSpec,
    pub batch_size: usize,
    pub prefetch_step: usize,
    pub config: PlanConfig,
}

impl PlanSpec {
    pub fn new(
        sources: Vec<Source>,
        rows: Vec<RowRef>,
        output: OutputSpec,
        batch_size: usize,
        prefetch_step: usize,
    ) -> Self {
        Self {
            sources,
            rows,
            output,
            batch_size,
            prefetch_step,
            config: PlanConfig::default(),
        }
    }

    #[must_use]
    pub fn config(mut self, config: PlanConfig) -> Self {
        self.config = config;
        self
    }
}

pub fn compile(spec: PlanSpec) -> Result<Plan> {
    Builder::new(spec)?.compile()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Side {
    Data,
    Indices,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ChunkKey {
    did: u32,
    side: Side,
    chunk: usize,
}

struct ChunkPlan {
    source: usize,
    decoder: Arc<Decoder>,
    /// Monotonic decoded block ends used for O(log B) cell lookup.
    decoded_ends: Option<Arc<[usize]>>,
    candidate_base: usize,
}

enum ChunkCache {
    Sparse(HashMap<usize, ChunkPlan>),
    Dense {
        slots: Box<[usize]>,
        plans: Vec<ChunkPlan>,
    },
}

impl ChunkCache {
    fn sparse() -> Self {
        Self::Sparse(HashMap::new())
    }

    fn dense(chunk_count: usize) -> Result<Self> {
        let mut slots = Vec::new();
        slots.try_reserve_exact(chunk_count)?;
        slots.resize(chunk_count, usize::MAX);
        Ok(Self::Dense {
            slots: slots.into_boxed_slice(),
            plans: Vec::new(),
        })
    }

    fn get(&self, chunk: usize) -> Option<&ChunkPlan> {
        match self {
            Self::Sparse(plans) => plans.get(&chunk),
            Self::Dense { slots, plans } => {
                let slot = *slots.get(chunk)?;
                if slot == usize::MAX {
                    None
                } else {
                    plans.get(slot)
                }
            }
        }
    }

    fn contains_key(&self, chunk: usize) -> bool {
        self.get(chunk).is_some()
    }

    fn try_insert(&mut self, chunk: usize, plan: ChunkPlan) -> Result<()> {
        match self {
            Self::Sparse(plans) => {
                plans.try_reserve(1)?;
                plans.insert(chunk, plan);
            }
            Self::Dense { slots, plans } => {
                let slot = slots
                    .get_mut(chunk)
                    .ok_or_else(|| Error::Invariant("dense chunk table index is missing".into()))?;
                if *slot != usize::MAX {
                    return Err(Error::Invariant(
                        "dense chunk metadata was installed twice".into(),
                    ));
                }
                plans.try_reserve(1)?;
                *slot = plans.len();
                plans.push(plan);
            }
        }
        Ok(())
    }
}

struct LoadedChunk {
    read_source: ReadSource,
    decoder: Decoder,
    decoded_ends: Option<Vec<usize>>,
    io_bytes: u64,
    io_ops: u64,
    temporary_input_bytes: usize,
    metadata_bytes: usize,
    retained_bytes: usize,
}

struct CellBlock {
    chunk_key: ChunkKey,
    source: usize,
    block_index: usize,
    encoded_range: Range<u64>,
    decoded_len: usize,
    cell_range: Range<usize>,
}

#[derive(Clone)]
struct BlockInfo {
    decoder: BlockDecoder,
    source: usize,
    encoded_range: Range<u64>,
}

impl BlockInfo {
    fn decoded_len(&self) -> usize {
        self.decoder.decoded_len()
    }
}

struct BlockCandidate {
    did: u32,
    data: Option<BlockInfo>,
    indices: Option<BlockInfo>,
}

struct CellInfo {
    did: u32,
    block_key: usize,
    output_slot: OutputSlot,
    data: Option<CellBlock>,
    indices: Option<CellBlock>,
}

#[repr(C)]
struct CompactCell {
    output_slot: OutputSlot,
    data_start: u32,
    data_end: u32,
    indices_start: u32,
    indices_end: u32,
}

impl CompactCell {
    fn from_resolved(cell: &CellInfo) -> Self {
        let data = cell.data.as_ref().map(|block| &block.cell_range);
        let indices = cell.indices.as_ref().map(|block| &block.cell_range);
        debug_assert!(data.is_none_or(|range| range.end <= u32::MAX as usize));
        debug_assert!(indices.is_none_or(|range| range.end <= u32::MAX as usize));
        Self {
            output_slot: cell.output_slot,
            data_start: data.map_or(0, |range| range.start as u32),
            data_end: data.map_or(0, |range| range.end as u32),
            indices_start: indices.map_or(0, |range| range.start as u32),
            indices_end: indices.map_or(0, |range| range.end as u32),
        }
    }

    fn data_range(&self) -> Range<usize> {
        self.data_start as usize..self.data_end as usize
    }

    fn indices_range(&self) -> Option<Range<usize>> {
        (self.indices_start != self.indices_end)
            .then_some(self.indices_start as usize..self.indices_end as usize)
    }
}

#[derive(Clone)]
enum CellLocation {
    Empty {
        chunk: usize,
    },
    Dense {
        chunk: usize,
        byte_range: Range<usize>,
    },
    Csr {
        chunk: usize,
        data_range: Range<usize>,
        indices_range: Range<usize>,
    },
}

struct ResolvedRow {
    did: u32,
    location: CellLocation,
}

enum PreloadedRows {
    General(Vec<ResolvedRow>),
    ValidatedSingleDense { chunk_cells: u64, row_bytes: usize },
}

#[derive(Debug, Clone, Copy)]
enum SourceRowLayout {
    Dense {
        n_rows: u64,
        chunk_cells: u64,
        row_bytes: usize,
    },
    Csr {
        n_rows: u64,
        chunk_cells: Option<u64>,
        data_element_size: usize,
        index_element_size: usize,
    },
}

struct TempBlockJob {
    block_key: usize,
    cell_count: usize,
    anchor: usize,
    batch_min: usize,
    batch_max: usize,
}

#[derive(Debug, Clone, Copy)]
enum MemberNode {
    Block(usize),
    Concat { left: usize, right: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RunKey {
    did: u32,
    data_source: usize,
    indices_source: Option<usize>,
}

struct TempRun {
    member_root: usize,
    block_count: usize,
    anchor: usize,
    batch_min: usize,
    batch_max: usize,
    active: bool,
    did: u32,
    /// Cached for O(1) merge decisions (avoid re-scanning blocks).
    data_source: usize,
    indices_source: Option<usize>,
    data_range: Option<Range<u64>>,
    indices_range: Option<Range<u64>>,
    data_decoded: usize,
    indices_decoded: usize,
    cell_count: usize,
}

struct MergeProjection {
    anchor: usize,
    batch_min: usize,
    batch_max: usize,
    did: u32,
    data_source: usize,
    indices_source: Option<usize>,
    data_range: Option<Range<u64>>,
    indices_range: Option<Range<u64>>,
    data_decoded: usize,
    indices_decoded: usize,
    cell_count: usize,
    block_count: usize,
}

struct MergedRuns {
    runs: Vec<TempRun>,
    members: Vec<MemberNode>,
}

enum CandidateRemap {
    Dense(Vec<usize>),
    Sparse(HashMap<usize, usize>),
}

impl CandidateRemap {
    fn new(raw_candidates: usize, cell_count: usize) -> Result<Self> {
        if raw_candidates <= cell_count.saturating_mul(2).max(1_024) {
            Ok(Self::Dense(vec![usize::MAX; raw_candidates]))
        } else {
            let mut values = HashMap::new();
            values.try_reserve(cell_count.min(raw_candidates))?;
            Ok(Self::Sparse(values))
        }
    }

    fn intern(&mut self, raw: usize, next: usize) -> Result<(usize, bool)> {
        match self {
            Self::Dense(values) => {
                if raw >= values.len() {
                    values.try_reserve(raw + 1 - values.len())?;
                    values.resize(raw + 1, usize::MAX);
                }
                let slot = &mut values[raw];
                if *slot == usize::MAX {
                    *slot = next;
                    Ok((next, true))
                } else {
                    Ok((*slot, false))
                }
            }
            Self::Sparse(values) => match values.entry(raw) {
                Entry::Occupied(entry) => Ok((*entry.get(), false)),
                Entry::Vacant(entry) => {
                    entry.insert(next);
                    Ok((next, true))
                }
            },
        }
    }
}

#[derive(Clone, Copy)]
struct Cost {
    bytes: u64,
    ops: u64,
}

struct Builder {
    request: PlanSpec,
    source_indices: HashMap<SourceId, u32>,
    registered_sources: Vec<Source>,
    row_layouts: Vec<SourceRowLayout>,
    source_plans: Vec<SourcePlan>,
    sources: Vec<ReadSource>,
    chunks: Vec<SourceChunks>,
    compile_io_bytes: u64,
    compile_io_ops: u64,
    chunk_metadata_bytes: usize,
    retained_whole_key_bytes: usize,
    peak_compile_payload_bytes: usize,
    batch_shift: Option<u32>,
    ring_slots: usize,
    ring_mask: usize,
    next_block_candidate: usize,
    block_candidates: Vec<BlockCandidate>,
}

struct SourceChunks {
    chunk_count: usize,
    data: ChunkCache,
    indices: Option<ChunkCache>,
    empty: HashMap<usize, usize>,
}

#[derive(Debug, Clone, Copy)]
struct ChunkSideOffsets {
    data: usize,
    indices: Option<usize>,
}

type FinalizedPlan = (
    Vec<Job>,
    Vec<BlockGroup>,
    Vec<CellTask>,
    Vec<BatchCompletion>,
    Vec<BlockSpec>,
    PlanStats,
);

impl Builder {
    fn new(mut request: PlanSpec) -> Result<Self> {
        if request.batch_size == 0 {
            return Err(Error::InvalidConfig("batch_size must be positive".into()));
        }
        if request.prefetch_step <= 1 {
            return Err(Error::InvalidConfig(
                "prefetch_step must be greater than 1".into(),
            ));
        }
        if request.prefetch_step > u32::MAX as usize {
            return Err(Error::InvalidConfig(
                "prefetch_step exceeds compact ring-slot representation".into(),
            ));
        }
        request.config.validate(request.prefetch_step)?;
        request.output.validate()?;

        let mut source_indices = HashMap::new();
        let mut compiled_sources = Vec::new();
        let mut row_layouts = Vec::new();
        let mut source_plans = Vec::new();
        let mut chunks = Vec::new();
        let registered_sources = std::mem::take(&mut request.sources);
        source_indices.try_reserve(registered_sources.len())?;
        compiled_sources.try_reserve_exact(registered_sources.len())?;
        row_layouts.try_reserve_exact(registered_sources.len())?;
        source_plans.try_reserve_exact(registered_sources.len())?;
        chunks.try_reserve_exact(registered_sources.len())?;
        let mut compile_io_bytes = 0u64;
        let mut compile_io_ops = 0u64;
        let mut dense_chunk_table_bytes = 0usize;
        for mut source in registered_sources {
            let source_plan = u32::try_from(source_plans.len())
                .map_err(|_| Error::ResourceLimit("registered source count exceeds u32".into()))?;
            match source_indices.entry(source.id) {
                Entry::Occupied(_) => {
                    return Err(Error::InvalidInput(format!(
                        "source id {} is registered more than once",
                        source.id.get()
                    )))
                }
                Entry::Vacant(entry) => {
                    entry.insert(source_plan);
                }
            }
            let n_cols = usize::try_from(source.dataset.n_cols()).map_err(|_| {
                Error::InvalidDataset(format!(
                    "dataset {} column count exceeds usize",
                    source.id.get()
                ))
            })?;
            let feature_targets = match source.feature_map.take() {
                Some(mapping) => {
                    if mapping.len() != n_cols {
                        return Err(Error::InvalidInput(format!(
                            "dataset {} feature map length {} does not match source n_cols {n_cols}",
                            source.id.get(),
                            mapping.len()
                        )));
                    }
                    for (col, target) in mapping.targets().iter().copied().enumerate() {
                        if target.is_some_and(|target| target >= request.output.n_cols) {
                            return Err(Error::InvalidInput(format!(
                                "source {} feature map column {col} exceeds output n_cols {}",
                                source.id.get(),
                                request.output.n_cols
                            )));
                        }
                    }
                    let targets = mapping.into_targets();
                    if n_cols == request.output.n_cols
                        && targets
                            .iter()
                            .enumerate()
                            .all(|(column, target)| *target == Some(column))
                    {
                        None
                    } else {
                        Some(targets)
                    }
                }
                None => {
                    if n_cols != request.output.n_cols {
                        return Err(Error::InvalidInput(format!(
                            "dataset {} has {n_cols} columns but output has {}; an explicit feature map is required",
                            source.id.get(),
                            request.output.n_cols
                        )));
                    }
                    None
                }
            };
            let value_dtype = source.dataset.dtype();
            let convert = ConvertOp::resolve(value_dtype, &request.output)?;
            let row_layout = match &source.dataset.kind {
                DatasetKind::Dense(meta) => {
                    let chunk_cells = meta.partition.chunk.fixed_cells_n().ok_or_else(|| {
                        Error::InvalidDataset("dense chunk partition is not fixed_cells".into())
                    })?;
                    let row_bytes = n_cols.checked_mul(value_dtype.size()).ok_or_else(|| {
                        Error::InvalidDataset("dense row byte length overflow".into())
                    })?;
                    SourceRowLayout::Dense {
                        n_rows: meta.shape[0],
                        chunk_cells,
                        row_bytes,
                    }
                }
                DatasetKind::Csr { meta, .. } => SourceRowLayout::Csr {
                    n_rows: meta.shape[0],
                    chunk_cells: meta.partition.chunk.fixed_cells_n(),
                    data_element_size: meta.data.dtype.size(),
                    index_element_size: meta.indices.dtype.size(),
                },
            };
            let index_dtype = match &source.dataset.kind {
                DatasetKind::Dense(_) => None,
                DatasetKind::Csr { meta, .. } => Some(meta.indices.dtype),
            };
            let index = index_dtype
                .map(|dtype| {
                    IndexOp::new(dtype).ok_or_else(|| {
                        Error::InvalidDataset(format!("invalid CSR index dtype {dtype}"))
                    })
                })
                .transpose()?;
            let (feature_map, dense_map) = match (feature_targets, index_dtype) {
                (None, _) => (None, None),
                (Some(targets), None) => {
                    let source_size = value_dtype.size();
                    let target_size = request.output.dtype.size();
                    (
                        None,
                        Some(build_dense_map(
                            targets,
                            source_size,
                            target_size,
                            request.output.n_cols,
                            convert.dense_gather_min_entries(),
                        )?),
                    )
                }
                (Some(targets), Some(_)) => {
                    let target_size = request.output.dtype.size();
                    let packed = request
                        .output
                        .n_cols
                        .checked_mul(target_size)
                        .is_some_and(|bytes| bytes < u32::MAX as usize);
                    if packed {
                        let mut compact = Vec::new();
                        compact.try_reserve_exact(targets.len())?;
                        for target in targets {
                            compact.push(match target {
                                Some(target) => (target * target_size) as u32,
                                None => UNMAPPED_TARGET_U32,
                            });
                        }
                        (Some(CsrMap::Packed32(Arc::from(compact))), None)
                    } else {
                        let mut compact = Vec::new();
                        compact.try_reserve_exact(targets.len())?;
                        for target in targets {
                            compact.push(match target {
                                Some(target) => {
                                    target.checked_mul(target_size).ok_or_else(|| {
                                        Error::ResourceLimit(
                                            "CSR map target byte offset overflow".into(),
                                        )
                                    })?
                                }
                                None => UNMAPPED_TARGET,
                            });
                        }
                        (Some(CsrMap::Wide(Arc::from(compact))), None)
                    }
                }
            };
            source_plans.push(SourcePlan {
                n_cols,
                value_dtype,
                index,
                feature_map,
                dense_map,
                convert,
            });
            row_layouts.push(row_layout);
            compile_io_bytes = compile_io_bytes
                .checked_add(source.dataset.initial_io_bytes)
                .ok_or_else(|| Error::ResourceLimit("compile I/O byte count overflow".into()))?;
            compile_io_ops = compile_io_ops
                .checked_add(source.dataset.initial_io_ops)
                .ok_or_else(|| {
                    Error::ResourceLimit("compile I/O operation count overflow".into())
                })?;
            let chunk_count = match &source.dataset.kind {
                DatasetKind::Dense(meta) => meta.chunks.n_chunks(),
                DatasetKind::Csr { meta, .. } => meta.chunks.n_chunks(),
            };
            let has_indices = matches!(&source.dataset.kind, DatasetKind::Csr { .. });
            let side_count = 1usize + usize::from(has_indices);
            let table_bytes = chunk_count
                .checked_mul(size_of::<usize>())
                .and_then(|bytes| bytes.checked_mul(side_count));
            let useful_density = !request.rows.is_empty()
                && chunk_count <= request.rows.len().saturating_mul(4).max(256);
            let input_row_bytes = request.rows.len().saturating_mul(size_of::<RowRef>());
            let dense_budget = request
                .config
                .limits
                .max_compile_working_set_bytes
                .saturating_sub(input_row_bytes)
                .min(MAX_DENSE_CHUNK_TABLE_BYTES);
            let use_dense = useful_density
                && table_bytes.is_some_and(|bytes| {
                    dense_chunk_table_bytes
                        .checked_add(bytes)
                        .is_some_and(|total| total <= dense_budget)
                });
            let data = if use_dense {
                ChunkCache::dense(chunk_count)?
            } else {
                ChunkCache::sparse()
            };
            let indices = if has_indices {
                Some(if use_dense {
                    ChunkCache::dense(chunk_count)?
                } else {
                    ChunkCache::sparse()
                })
            } else {
                None
            };
            if use_dense {
                dense_chunk_table_bytes = dense_chunk_table_bytes
                    .checked_add(table_bytes.ok_or_else(|| {
                        Error::ResourceLimit("dense chunk table bytes overflow".into())
                    })?)
                    .ok_or_else(|| {
                        Error::ResourceLimit("dense chunk table bytes overflow".into())
                    })?;
            }
            chunks.push(SourceChunks {
                chunk_count,
                data,
                indices,
                empty: HashMap::new(),
            });
            compiled_sources.push(source);
        }

        let batch_shift = request
            .batch_size
            .is_power_of_two()
            .then(|| request.batch_size.trailing_zeros());
        let batch_count = request.rows.len().div_ceil(request.batch_size);
        let ring_slots = request.prefetch_step.min(batch_count);
        let ring_mask = if ring_slots.is_power_of_two() {
            ring_slots - 1
        } else {
            usize::MAX
        };
        Ok(Self {
            request,
            source_indices,
            registered_sources: compiled_sources,
            row_layouts,
            source_plans,
            sources: vec![ReadSource::Empty],
            chunks,
            compile_io_bytes,
            compile_io_ops,
            chunk_metadata_bytes: dense_chunk_table_bytes,
            retained_whole_key_bytes: 0,
            peak_compile_payload_bytes: 0,
            batch_shift,
            ring_slots,
            ring_mask,
            next_block_candidate: 0,
            block_candidates: Vec::new(),
        })
    }

    fn compile(mut self) -> Result<Plan> {
        let row_bytes = self
            .request
            .output
            .n_cols
            .checked_mul(self.request.output.dtype.size())
            .ok_or_else(|| Error::ResourceLimit("output row byte length overflow".into()))?;
        let row_stride = align_up(row_bytes, 64)?;
        let output_ring_bytes = if self.request.rows.is_empty() {
            0
        } else {
            row_stride
                .checked_mul(self.request.batch_size)
                .and_then(|bytes| bytes.checked_mul(self.ring_slots))
                .ok_or_else(|| Error::ResourceLimit("output ring byte length overflow".into()))?
        };
        if output_ring_bytes > self.request.config.limits.max_output_buffer_bytes {
            return Err(Error::ResourceLimit(format!(
                "output ring has {output_ring_bytes} bytes, limit is {}",
                self.request.config.limits.max_output_buffer_bytes
            )));
        }

        let preload_required_chunks = self.should_preload_required_chunks()?;
        let retained_resolved_row_bytes = if preload_required_chunks {
            self.resolved_row_arena_element_size()
        } else {
            0
        };
        let initial_compile_bytes = self
            .request
            .rows
            .len()
            .checked_mul(
                size_of::<RowRef>()
                    .saturating_add(retained_resolved_row_bytes)
                    .saturating_add(size_of::<CompactCell>())
                    .saturating_add(size_of::<TempBlockJob>())
                    .saturating_add(size_of::<usize>() * 2)
                    .saturating_add(size_of::<BlockCandidate>()),
            )
            .and_then(|bytes| bytes.checked_add(self.chunk_metadata_bytes))
            .ok_or_else(|| Error::ResourceLimit("initial compile working set overflow".into()))?;
        self.check_compile_payload("initial compile", initial_compile_bytes)?;

        let batch_count = self.request.rows.len().div_ceil(self.request.batch_size);
        #[cfg(feature = "profile")]
        let phase_started = Instant::now();
        let resolved_rows = if preload_required_chunks {
            let (rows, required) = self.resolve_rows_and_required_chunks()?;
            self.preload_required_chunks(required)?;
            Some(rows)
        } else {
            None
        };
        let (block_jobs, cell_blocks, cells) =
            self.resolve_and_merge_same_blocks(resolved_rows, row_stride)?;
        // Resolution is the last phase that needs input rows, source lookup,
        // or dataset metadata. Release all three before allocating merge
        // arenas; chunk plans retain only the decoder/read data needed later.
        drop(std::mem::take(&mut self.request.rows));
        drop(std::mem::take(&mut self.registered_sources));
        drop(std::mem::take(&mut self.source_indices));
        drop(std::mem::take(&mut self.row_layouts));
        #[cfg(feature = "profile")]
        let compile_resolve_ns = elapsed_ns(phase_started);
        #[cfg(feature = "profile")]
        let compile_same_block_ns = 0;
        let block_job_count = block_jobs.len();
        self.check_compile_payload(
            "cross-block merge",
            checked_payload_bytes(&[
                (cells.len(), size_of::<CompactCell>()),
                (self.request.rows.len(), size_of::<RowRef>()),
                (block_jobs.len(), size_of::<TempBlockJob>()),
                (cell_blocks.len(), size_of::<usize>()),
                (block_jobs.len(), size_of::<TempRun>()),
                (block_jobs.len(), size_of::<MemberNode>() * 2),
                (block_jobs.len(), size_of::<usize>()),
                (self.block_candidates.len(), size_of::<BlockCandidate>()),
                (1, self.chunk_metadata_bytes),
                (1, self.retained_whole_key_bytes),
                (block_jobs.len(), size_of::<RunKey>() + size_of::<usize>()),
            ])?,
        )?;
        #[cfg(feature = "profile")]
        let phase_started = Instant::now();
        let merged = self.merge_runs(&block_jobs)?;
        #[cfg(feature = "profile")]
        let compile_merge_runs_ns = elapsed_ns(phase_started);
        #[cfg(feature = "profile")]
        let phase_started = Instant::now();
        let (jobs, arena_groups, arena_cells, arena_completions, arena_blocks, mut stats) = self
            .finalize(
                &cells,
                &block_jobs,
                &cell_blocks,
                &merged,
                output_ring_bytes,
            )?;
        #[cfg(feature = "profile")]
        let compile_finalize_ns = elapsed_ns(phase_started);
        stats.input_rows = cells.len();
        stats.block_jobs = block_job_count;
        stats.jobs = jobs.len();
        stats.compile_time_io_bytes = self.compile_io_bytes;
        stats.compile_time_io_ops = self.compile_io_ops;
        #[cfg(feature = "profile")]
        {
            stats.compile_resolve_ns = compile_resolve_ns;
            stats.compile_same_block_ns = compile_same_block_ns;
            stats.compile_merge_runs_ns = compile_merge_runs_ns;
            stats.compile_finalize_ns = compile_finalize_ns;
        }
        let io_ops = stats
            .data_io_ops
            .checked_add(stats.indices_io_ops)
            .ok_or_else(|| Error::ResourceLimit("predicted I/O operation count overflow".into()))?;
        stats.predicted_io_seconds = if io_ops == 0 {
            0.0
        } else {
            (stats.predicted_physical_bytes as f64
                / self.request.config.io_bandwidth_bytes_per_second)
                .max(io_ops as f64 / self.request.config.io_operations_per_second)
        };

        let runtime = runtime_envelope(&self.sources, &jobs, &arena_groups, &arena_blocks)?;
        let plan = PlanData {
            batch_size: self.request.batch_size,
            batch_count,
            prefetch_step: self.request.prefetch_step,
            ring_slots: self.ring_slots,
            ring_mask: self.ring_mask,
            fill: FillOp::new(
                &self.request.output.fill.encode()[..self.request.output.dtype.size()],
            ),
            row_bytes,
            output: self.request.output,
            row_stride,
            jobs,
            groups: arena_groups,
            cells: arena_cells,
            completions: arena_completions,
            blocks: arena_blocks,
            sources: self.sources,
            source_plans: self.source_plans,
            runtime,
            stats,
        };
        Ok(Plan {
            inner: Arc::new(plan),
        })
    }

    fn check_compile_payload(&mut self, phase: &str, bytes: usize) -> Result<()> {
        if bytes > self.request.config.limits.max_compile_working_set_bytes {
            return Err(Error::ResourceLimit(format!(
                "{phase} payload requires up to {bytes} bytes, limit is {}",
                self.request.config.limits.max_compile_working_set_bytes
            )));
        }
        self.peak_compile_payload_bytes = self.peak_compile_payload_bytes.max(bytes);
        Ok(())
    }

    fn check_chunk_load_payload(
        &mut self,
        temporary_input_bytes: usize,
        prospective_chunk_metadata: usize,
        prospective_retained_bytes: usize,
    ) -> Result<()> {
        let bytes = checked_payload_bytes(&[
            (self.request.rows.len(), size_of::<CellInfo>()),
            (self.request.rows.len(), size_of::<RowRef>()),
            (
                self.request.rows.len(),
                self.resolved_row_arena_element_size(),
            ),
            (1, self.chunk_metadata_bytes),
            (1, temporary_input_bytes),
            (1, prospective_chunk_metadata),
            (1, prospective_retained_bytes),
        ])?;
        self.check_compile_payload("chunk metadata load", bytes)
    }

    fn single_dense_layout(&self) -> Option<(SourceId, u64, u64, usize)> {
        if self.registered_sources.len() != 1 {
            return None;
        }
        let source_id = self.registered_sources.first()?.id;
        let SourceRowLayout::Dense {
            n_rows,
            chunk_cells,
            row_bytes,
        } = *self.row_layouts.first()?
        else {
            return None;
        };
        Some((source_id, n_rows, chunk_cells, row_bytes))
    }

    fn resolved_row_arena_element_size(&self) -> usize {
        if self.single_dense_layout().is_some() {
            0
        } else {
            size_of::<ResolvedRow>()
        }
    }

    fn source_slot_for(&self, index: RowRef) -> Result<u32> {
        if self.registered_sources.len() == 1 {
            if self.registered_sources[0].id == index.source {
                return Ok(0);
            }
        } else if let Some(&slot) = self.source_indices.get(&index.source) {
            return Ok(slot);
        }
        Err(Error::InvalidInput(format!(
            "source id {} is not registered",
            index.source.get()
        )))
    }

    fn resolve_rows_and_required_chunks(&mut self) -> Result<(PreloadedRows, Vec<ChunkKey>)> {
        let (offsets, possible_chunks) = self.chunk_side_offsets()?;
        let resolved_row_size = self.resolved_row_arena_element_size();
        let marker_words = possible_chunks.div_ceil(u64::BITS as usize);
        let marker_bytes = marker_words
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| Error::ResourceLimit("required chunk markers overflow".into()))?;
        self.check_compile_payload(
            "required chunk discovery",
            checked_payload_bytes(&[
                (self.request.rows.len(), size_of::<RowRef>()),
                (self.request.rows.len(), resolved_row_size),
                (offsets.len(), size_of::<ChunkSideOffsets>()),
                (1, marker_bytes),
            ])?,
        )?;
        let mut required = Vec::new();
        required.try_reserve_exact(marker_words)?;
        required.resize(marker_words, 0u64);
        let rows = if let Some((source_id, n_rows, chunk_cells, row_bytes)) =
            self.single_dense_layout()
        {
            let chunks = self
                .chunks
                .first()
                .ok_or_else(|| Error::Invariant("dense source chunk grid is missing".into()))?;
            let data_base = offsets
                .first()
                .ok_or_else(|| Error::Invariant("dense chunk side offset is missing".into()))?
                .data;
            let mut last_required = None;
            for &index in &self.request.rows {
                if index.source != source_id {
                    return Err(Error::InvalidInput(format!(
                        "source id {} is not registered",
                        index.source.get()
                    )));
                }
                if index.row >= n_rows {
                    return Err(Error::InvalidInput(format!(
                        "row {} is outside source {} with {n_rows} rows",
                        index.row,
                        index.source.get()
                    )));
                }
                let chunk = usize::try_from(index.row / chunk_cells)
                    .map_err(|_| Error::InvalidDataset("dense chunk index exceeds usize".into()))?;
                if chunk >= chunks.chunk_count {
                    return Err(Error::InvalidDataset(
                        "row resolves outside the source chunk grid".into(),
                    ));
                }
                if row_bytes != 0 {
                    let local_row = usize::try_from(index.row % chunk_cells).map_err(|_| {
                        Error::InvalidDataset("dense row offset exceeds usize".into())
                    })?;
                    let start = local_row
                        .checked_mul(row_bytes)
                        .ok_or_else(|| Error::InvalidDataset("dense row offset overflow".into()))?;
                    start
                        .checked_add(row_bytes)
                        .ok_or_else(|| Error::InvalidDataset("dense row end overflow".into()))?;
                    if last_required != Some(chunk) {
                        set_required_chunk(&mut required, data_base + chunk)?;
                        last_required = Some(chunk);
                    }
                }
            }
            PreloadedRows::ValidatedSingleDense {
                chunk_cells,
                row_bytes,
            }
        } else {
            let mut rows = Vec::new();
            rows.try_reserve_exact(self.request.rows.len())?;
            let mut last_required = [None, None];
            for &index in &self.request.rows {
                let row = self.resolve_row(index)?;
                self.record_required_chunks(&row, &offsets, &mut required, &mut last_required)?;
                rows.push(row);
            }
            PreloadedRows::General(rows)
        };
        let required_count = required.iter().try_fold(0usize, |count, word| {
            count
                .checked_add(word.count_ones() as usize)
                .ok_or_else(|| Error::ResourceLimit("required chunk count overflow".into()))
        })?;
        self.check_compile_payload(
            "required chunk collection",
            checked_payload_bytes(&[
                (self.request.rows.len(), size_of::<RowRef>()),
                (self.request.rows.len(), resolved_row_size),
                (offsets.len(), size_of::<ChunkSideOffsets>()),
                (1, marker_bytes),
                (required_count, size_of::<ChunkKey>()),
            ])?,
        )?;
        let mut requests = Vec::new();
        requests.try_reserve_exact(required_count)?;
        for (did, chunks) in self.chunks.iter().enumerate() {
            let did = u32::try_from(did)
                .map_err(|_| Error::ResourceLimit("source slot exceeds u32".into()))?;
            let source_offsets = offsets
                .get(did as usize)
                .ok_or_else(|| Error::Invariant("chunk side offsets are missing".into()))?;
            for chunk in 0..chunks.chunk_count {
                if required_chunk_is_set(&required, source_offsets.data + chunk) {
                    requests.push(ChunkKey {
                        did,
                        side: Side::Data,
                        chunk,
                    });
                }
            }
            if let Some(base) = source_offsets.indices {
                for chunk in 0..chunks.chunk_count {
                    if required_chunk_is_set(&required, base + chunk) {
                        requests.push(ChunkKey {
                            did,
                            side: Side::Indices,
                            chunk,
                        });
                    }
                }
            }
        }
        debug_assert_eq!(requests.len(), required_count);
        Ok((rows, requests))
    }

    fn resolve_row(&self, index: RowRef) -> Result<ResolvedRow> {
        let did = self.source_slot_for(index)?;
        let source = self
            .registered_sources
            .get(did as usize)
            .ok_or_else(|| Error::Invariant("row source is not registered".into()))?;
        let layout = *self
            .row_layouts
            .get(did as usize)
            .ok_or_else(|| Error::Invariant("row layout is not registered".into()))?;
        let n_rows = match layout {
            SourceRowLayout::Dense { n_rows, .. } | SourceRowLayout::Csr { n_rows, .. } => n_rows,
        };
        if index.row >= n_rows {
            return Err(Error::InvalidInput(format!(
                "row {} is outside source {} with {n_rows} rows",
                index.row,
                index.source.get()
            )));
        }

        let location = match layout {
            SourceRowLayout::Dense {
                chunk_cells,
                row_bytes,
                ..
            } => {
                let chunk = usize::try_from(index.row / chunk_cells)
                    .map_err(|_| Error::InvalidDataset("dense chunk index exceeds usize".into()))?;
                if row_bytes == 0 {
                    CellLocation::Empty { chunk }
                } else {
                    let local_row = usize::try_from(index.row % chunk_cells).map_err(|_| {
                        Error::InvalidDataset("dense row offset exceeds usize".into())
                    })?;
                    let start = local_row
                        .checked_mul(row_bytes)
                        .ok_or_else(|| Error::InvalidDataset("dense row offset overflow".into()))?;
                    let end = start
                        .checked_add(row_bytes)
                        .ok_or_else(|| Error::InvalidDataset("dense row end overflow".into()))?;
                    CellLocation::Dense {
                        chunk,
                        byte_range: start..end,
                    }
                }
            }
            SourceRowLayout::Csr {
                chunk_cells,
                data_element_size,
                index_element_size,
                ..
            } => {
                let DatasetKind::Csr { meta, indptr } = &source.dataset.kind else {
                    return Err(Error::Invariant(
                        "CSR row layout has a dense dataset".into(),
                    ));
                };
                let (chunk, chunk_start) = if let Some(chunk_cells) = chunk_cells {
                    let chunk = usize::try_from(index.row / chunk_cells).map_err(|_| {
                        Error::InvalidDataset("CSR chunk index exceeds usize".into())
                    })?;
                    (chunk, index.row - index.row % chunk_cells)
                } else {
                    let chunk = meta
                        .chunks
                        .chunk_of(index.row)
                        .map_err(|error| Error::InvalidDataset(error.to_string()))?;
                    let (start, _) = meta.chunks.cell_range(chunk, n_rows)?;
                    (chunk, start)
                };
                let row = usize::try_from(index.row)
                    .map_err(|_| Error::InvalidDataset("CSR row exceeds usize".into()))?;
                let chunk_start = usize::try_from(chunk_start)
                    .map_err(|_| Error::InvalidDataset("CSR chunk start exceeds usize".into()))?;
                let nnz0 = *indptr
                    .get(row)
                    .ok_or_else(|| Error::InvalidDataset("CSR row exceeds indptr".into()))?;
                let nnz1 = *indptr
                    .get(row + 1)
                    .ok_or_else(|| Error::InvalidDataset("CSR row end exceeds indptr".into()))?;
                let chunk_nnz0 = *indptr.get(chunk_start).ok_or_else(|| {
                    Error::InvalidDataset("CSR chunk start exceeds indptr".into())
                })?;
                let local0 = nnz0
                    .checked_sub(chunk_nnz0)
                    .ok_or_else(|| Error::InvalidDataset("CSR indptr is not monotonic".into()))?;
                let local1 = nnz1
                    .checked_sub(chunk_nnz0)
                    .ok_or_else(|| Error::InvalidDataset("CSR indptr is not monotonic".into()))?;
                if local0 == local1 {
                    CellLocation::Empty { chunk }
                } else {
                    CellLocation::Csr {
                        chunk,
                        data_range: element_range(local0, local1, data_element_size, "CSR data")?,
                        indices_range: element_range(
                            local0,
                            local1,
                            index_element_size,
                            "CSR indices",
                        )?,
                    }
                }
            }
        };
        Ok(ResolvedRow { did, location })
    }

    fn resolve_validated_single_dense_row(
        &self,
        index: RowRef,
        chunk_cells: u64,
        row_bytes: usize,
    ) -> Result<ResolvedRow> {
        debug_assert_eq!(
            self.registered_sources.first().map(|source| source.id),
            Some(index.source)
        );
        let chunk = usize::try_from(index.row / chunk_cells)
            .map_err(|_| Error::InvalidDataset("dense chunk index exceeds usize".into()))?;
        let location = if row_bytes == 0 {
            CellLocation::Empty { chunk }
        } else {
            let local_row = usize::try_from(index.row % chunk_cells)
                .map_err(|_| Error::InvalidDataset("dense row offset exceeds usize".into()))?;
            let start = local_row
                .checked_mul(row_bytes)
                .ok_or_else(|| Error::InvalidDataset("dense row offset overflow".into()))?;
            let end = start
                .checked_add(row_bytes)
                .ok_or_else(|| Error::InvalidDataset("dense row end overflow".into()))?;
            CellLocation::Dense {
                chunk,
                byte_range: start..end,
            }
        };
        Ok(ResolvedRow { did: 0, location })
    }

    fn record_required_chunks(
        &self,
        row: &ResolvedRow,
        offsets: &[ChunkSideOffsets],
        required: &mut [u64],
        last_required: &mut [Option<ChunkKey>; 2],
    ) -> Result<()> {
        let (chunk, needs_indices) = match &row.location {
            CellLocation::Empty { .. } => return Ok(()),
            CellLocation::Dense { chunk, .. } => (*chunk, false),
            CellLocation::Csr { chunk, .. } => (*chunk, true),
        };
        let chunks = self
            .chunks
            .get(row.did as usize)
            .ok_or_else(|| Error::Invariant("resolved source chunk grid is missing".into()))?;
        if chunk >= chunks.chunk_count {
            return Err(Error::InvalidDataset(
                "row resolves outside the source chunk grid".into(),
            ));
        }
        let data_key = ChunkKey {
            did: row.did,
            side: Side::Data,
            chunk,
        };
        if last_required[0] != Some(data_key) {
            let base = offsets
                .get(row.did as usize)
                .ok_or_else(|| Error::Invariant("chunk side offsets are missing".into()))?
                .data;
            set_required_chunk(required, base + chunk)?;
            last_required[0] = Some(data_key);
        }
        if needs_indices {
            if chunks.indices.is_none() {
                return Err(Error::Invariant("CSR indices chunk map is missing".into()));
            }
            let indices_key = ChunkKey {
                did: row.did,
                side: Side::Indices,
                chunk,
            };
            if last_required[1] != Some(indices_key) {
                let base = offsets
                    .get(row.did as usize)
                    .and_then(|offset| offset.indices)
                    .ok_or_else(|| Error::Invariant("indices chunk marker is missing".into()))?;
                set_required_chunk(required, base + chunk)?;
                last_required[1] = Some(indices_key);
            }
        }
        Ok(())
    }

    fn chunk_side_offsets(&self) -> Result<(Vec<ChunkSideOffsets>, usize)> {
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(self.chunks.len())?;
        let mut next = 0usize;
        for chunks in &self.chunks {
            let data = next;
            next = next
                .checked_add(chunks.chunk_count)
                .ok_or_else(|| Error::ResourceLimit("chunk count overflow".into()))?;
            let indices = if chunks.indices.is_some() {
                let base = next;
                next = next
                    .checked_add(chunks.chunk_count)
                    .ok_or_else(|| Error::ResourceLimit("chunk count overflow".into()))?;
                Some(base)
            } else {
                None
            };
            offsets.push(ChunkSideOffsets { data, indices });
        }
        Ok((offsets, next))
    }

    fn possible_chunk_side_count(&self) -> Result<usize> {
        self.chunks.iter().try_fold(0usize, |total, chunks| {
            total
                .checked_add(chunks.chunk_count)
                .and_then(|count| {
                    chunks
                        .indices
                        .as_ref()
                        .map_or(Some(count), |_| count.checked_add(chunks.chunk_count))
                })
                .ok_or_else(|| Error::ResourceLimit("chunk count overflow".into()))
        })
    }

    fn should_preload_required_chunks(&self) -> Result<bool> {
        const PARALLEL_PRELOAD_MIN_CHUNKS: usize = 256;
        const MAX_REQUIRED_CHUNK_MARKER_BYTES: usize = 64 * 1024 * 1024;
        if self.request.config.compile_io_concurrency <= 1 || self.request.rows.is_empty() {
            return Ok(false);
        }
        let Ok(possible_chunks) = self.possible_chunk_side_count() else {
            return Ok(false);
        };
        let Some(marker_bytes) = possible_chunks
            .div_ceil(u64::BITS as usize)
            .checked_mul(size_of::<u64>())
        else {
            return Ok(false);
        };
        let useful_density = possible_chunks
            <= self
                .request
                .rows
                .len()
                .saturating_mul(4)
                .max(PARALLEL_PRELOAD_MIN_CHUNKS);
        let maximum_required_chunks =
            possible_chunks.min(self.request.rows.len().saturating_mul(2));
        let preload_payload = checked_payload_bytes(&[
            (self.request.rows.len(), size_of::<CellInfo>()),
            (self.request.rows.len(), size_of::<RowRef>()),
            (
                self.request.rows.len(),
                self.resolved_row_arena_element_size(),
            ),
            (self.chunks.len(), size_of::<ChunkSideOffsets>()),
            (1, marker_bytes),
            (maximum_required_chunks, size_of::<ChunkKey>()),
            (1, self.chunk_metadata_bytes),
            (
                1,
                self.retained_whole_key_bytes
                    .max(self.request.config.limits.max_retained_whole_key_bytes),
            ),
        ]);
        Ok(possible_chunks >= PARALLEL_PRELOAD_MIN_CHUNKS
            && marker_bytes <= MAX_REQUIRED_CHUNK_MARKER_BYTES
            && useful_density
            && preload_payload.is_ok_and(|bytes| {
                bytes <= self.request.config.limits.max_compile_working_set_bytes
            }))
    }

    fn preload_required_chunks(&mut self, requests: Vec<ChunkKey>) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }
        let retained = checked_payload_bytes(&[
            (self.request.rows.len(), size_of::<CellInfo>()),
            (self.request.rows.len(), size_of::<RowRef>()),
            (
                self.request.rows.len(),
                self.resolved_row_arena_element_size(),
            ),
            (requests.len(), size_of::<ChunkKey>()),
            (1, self.chunk_metadata_bytes),
            (
                1,
                self.retained_whole_key_bytes
                    .max(self.request.config.limits.max_retained_whole_key_bytes),
            ),
        ])?;
        let available_for_loaders = self
            .request
            .config
            .limits
            .max_compile_working_set_bytes
            .saturating_sub(retained);
        let memory_concurrency = available_for_loaders
            .checked_div(self.request.config.limits.max_encoded_bytes_per_side)
            .unwrap_or(0)
            .max(1);
        let concurrency = self
            .request
            .config
            .compile_io_concurrency
            .min(memory_concurrency)
            .min(requests.len());
        if concurrency == 1 {
            for key in requests {
                self.load_missing_chunk(key)?;
            }
            return Ok(());
        }

        let next = AtomicUsize::new(0);
        let retained_whole_keys = AtomicUsize::new(self.retained_whole_key_bytes);
        let mut outcomes = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(concurrency);
            for _ in 0..concurrency {
                handles.push(scope.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&key) = requests.get(index) else {
                            break;
                        };
                        let outcome = load_chunk_detached(
                            &self.registered_sources[key.did as usize],
                            key.side,
                            key.chunk,
                            self.request.config.limits.max_encoded_bytes_per_side,
                            &retained_whole_keys,
                            self.request.config.limits.max_retained_whole_key_bytes,
                        );
                        local.push((index, key, outcome));
                    }
                    local
                }));
            }
            let mut outcomes = Vec::with_capacity(requests.len());
            for handle in handles {
                outcomes.extend(handle.join().map_err(|_| {
                    Error::Invariant("chunk metadata loader thread panicked".into())
                })?);
            }
            Ok::<_, Error>(outcomes)
        })?;
        outcomes.sort_unstable_by_key(|(index, _, _)| *index);
        for (_, key, outcome) in outcomes {
            self.install_loaded_chunk(key, outcome?)?;
        }
        Ok(())
    }

    fn resolve_cell(
        &mut self,
        ordinal: usize,
        resolved: ResolvedRow,
        row_stride: usize,
    ) -> Result<CellInfo> {
        let index = self.request.rows[ordinal];
        let source_slot = resolved.did;
        let location = resolved.location;
        let (logical_batch, cell_in_batch) = if let Some(shift) = self.batch_shift {
            (
                ordinal >> shift,
                ordinal & self.request.batch_size.saturating_sub(1),
            )
        } else {
            (
                ordinal / self.request.batch_size,
                ordinal % self.request.batch_size,
            )
        };
        let ring_batch = if self.ring_mask == usize::MAX {
            logical_batch % self.ring_slots
        } else {
            logical_batch & self.ring_mask
        };
        let row_offset = ring_batch
            .checked_mul(self.request.batch_size)
            .and_then(|row| row.checked_add(cell_in_batch))
            .and_then(|row| row.checked_mul(row_stride))
            .ok_or_else(|| Error::ResourceLimit("output row offset overflow".into()))?;
        let output_slot = OutputSlot::new(row_offset, logical_batch < self.ring_slots)
            .ok_or_else(|| Error::Invariant("output row offset is not aligned".into()))?;
        match location {
            CellLocation::Empty { chunk } => Ok(CellInfo {
                did: source_slot,
                block_key: self.empty_block_candidate(source_slot, chunk)?,
                output_slot,
                data: None,
                indices: None,
            }),
            CellLocation::Dense { chunk, byte_range } => {
                let key = ChunkKey {
                    did: source_slot,
                    side: Side::Data,
                    chunk,
                };
                let (data, candidate_base) = self.locate_chunk_cell(key, byte_range, true)?;
                let block_key = candidate_base
                    .checked_add(data.block_index)
                    .ok_or_else(|| Error::ResourceLimit("block candidate index overflow".into()))?;
                Ok(CellInfo {
                    did: source_slot,
                    block_key,
                    output_slot,
                    data: Some(data),
                    indices: None,
                })
            }
            CellLocation::Csr {
                chunk,
                data_range,
                indices_range,
            } => {
                let data_key = ChunkKey {
                    did: source_slot,
                    side: Side::Data,
                    chunk,
                };
                let (data, candidate_base) = self.locate_chunk_cell(data_key, data_range, false)?;
                let block_key = candidate_base
                    .checked_add(data.block_index)
                    .ok_or_else(|| Error::ResourceLimit("block candidate index overflow".into()))?;
                let indices_key = ChunkKey {
                    did: source_slot,
                    side: Side::Indices,
                    chunk,
                };
                let (indices, _) = self.locate_chunk_cell(indices_key, indices_range, false)?;
                if data.block_index != indices.block_index {
                    return Err(Error::InvalidDataset(format!(
                        "CSR mirrored block mismatch in dataset {} chunk {chunk}: data {} vs indices {}",
                        index.source.get(),
                        data.block_index,
                        indices.block_index
                    )));
                }
                Ok(CellInfo {
                    did: source_slot,
                    block_key,
                    output_slot,
                    data: Some(data),
                    indices: Some(indices),
                })
            }
        }
    }

    fn locate_chunk_cell(
        &mut self,
        key: ChunkKey,
        cell: Range<usize>,
        dense: bool,
    ) -> Result<(CellBlock, usize)> {
        if let Some(result) = self.locate_cached_chunk_cell(key, cell.clone(), dense) {
            return result;
        }
        self.load_missing_chunk(key)?;
        self.locate_cached_chunk_cell(key, cell, dense)
            .ok_or_else(|| Error::Invariant("inserted chunk is missing".into()))?
    }

    fn locate_cached_chunk_cell(
        &self,
        key: ChunkKey,
        cell: Range<usize>,
        dense: bool,
    ) -> Option<Result<(CellBlock, usize)>> {
        self.chunk(key).map(|plan| {
            let block = if dense {
                locate_dense_cell_block(plan, key, cell)
            } else {
                locate_cell_block(plan, key, cell)
            }?;
            Ok((block, plan.candidate_base))
        })
    }

    fn load_missing_chunk(&mut self, cache_key: ChunkKey) -> Result<()> {
        if self.chunk(cache_key).is_some() {
            return Ok(());
        }
        let source = self
            .registered_sources
            .get(cache_key.did as usize)
            .ok_or_else(|| Error::Invariant("chunk source is not registered".into()))?;
        let retained_whole_keys = AtomicUsize::new(self.retained_whole_key_bytes);
        let loaded = load_chunk_detached(
            source,
            cache_key.side,
            cache_key.chunk,
            self.request.config.limits.max_encoded_bytes_per_side,
            &retained_whole_keys,
            self.request.config.limits.max_retained_whole_key_bytes,
        )?;
        self.install_loaded_chunk(cache_key, loaded)
    }

    fn install_loaded_chunk(&mut self, key: ChunkKey, loaded: LoadedChunk) -> Result<()> {
        let retained_bytes = self
            .retained_whole_key_bytes
            .checked_add(loaded.retained_bytes)
            .ok_or_else(|| Error::ResourceLimit("retained whole-key bytes overflow".into()))?;
        self.check_chunk_load_payload(
            loaded.temporary_input_bytes,
            loaded.metadata_bytes,
            retained_bytes,
        )?;
        self.retained_whole_key_bytes = retained_bytes;
        self.chunk_metadata_bytes = self
            .chunk_metadata_bytes
            .checked_add(loaded.metadata_bytes)
            .ok_or_else(|| Error::ResourceLimit("chunk metadata byte count overflow".into()))?;
        let source_id = self.sources.len();
        self.sources.try_reserve(1)?;
        self.sources.push(loaded.read_source);
        self.compile_io_bytes = self
            .compile_io_bytes
            .checked_add(loaded.io_bytes)
            .ok_or_else(|| Error::ResourceLimit("compile I/O byte count overflow".into()))?;
        self.compile_io_ops = self
            .compile_io_ops
            .checked_add(loaded.io_ops)
            .ok_or_else(|| Error::ResourceLimit("compile I/O operation count overflow".into()))?;
        let candidate_base = if key.side == Side::Data {
            let base = self.next_block_candidate;
            self.next_block_candidate = base
                .checked_add(loaded.decoder.header().block_count())
                .ok_or_else(|| Error::ResourceLimit("block candidate count overflow".into()))?;
            base
        } else {
            usize::MAX
        };
        let plan = ChunkPlan {
            source: source_id,
            decoder: Arc::new(loaded.decoder),
            decoded_ends: loaded.decoded_ends.map(Arc::from),
            candidate_base,
        };
        let chunks = self
            .chunks
            .get_mut(key.did as usize)
            .ok_or_else(|| Error::Invariant("chunk source grid is missing".into()))?;
        if key.chunk >= chunks.chunk_count {
            return Err(Error::Invariant(
                "chunk index is outside the registered source grid".into(),
            ));
        }
        let cache = match key.side {
            Side::Data => &mut chunks.data,
            Side::Indices => chunks
                .indices
                .as_mut()
                .ok_or_else(|| Error::Invariant("dense source has no indices cache".into()))?,
        };
        if cache.contains_key(key.chunk) {
            return Err(Error::Invariant(
                "chunk metadata was installed twice".into(),
            ));
        }
        cache.try_insert(key.chunk, plan)?;
        Ok(())
    }

    fn chunk(&self, key: ChunkKey) -> Option<&ChunkPlan> {
        let chunks = self.chunks.get(key.did as usize)?;
        if key.chunk >= chunks.chunk_count {
            return None;
        }
        match key.side {
            Side::Data => chunks.data.get(key.chunk),
            Side::Indices => chunks.indices.as_ref()?.get(key.chunk),
        }
    }

    fn empty_block_candidate(&mut self, source: u32, chunk: usize) -> Result<usize> {
        if let Some(candidate) = self
            .chunks
            .get(source as usize)
            .and_then(|chunks| chunks.empty.get(&chunk))
            .copied()
        {
            return Ok(candidate);
        }
        let candidate = self.next_block_candidate;
        self.next_block_candidate = candidate
            .checked_add(1)
            .ok_or_else(|| Error::ResourceLimit("block candidate count overflow".into()))?;
        let slot = self
            .chunks
            .get_mut(source as usize)
            .ok_or_else(|| Error::Invariant("empty row source grid is missing".into()))?;
        if chunk >= slot.chunk_count {
            return Err(Error::Invariant(
                "empty row chunk is outside its grid".into(),
            ));
        }
        slot.empty.try_reserve(1)?;
        slot.empty.insert(chunk, candidate);
        Ok(candidate)
    }

    fn resolve_and_merge_same_blocks(
        &mut self,
        resolved_rows: Option<PreloadedRows>,
        row_stride: usize,
    ) -> Result<(Vec<TempBlockJob>, Vec<usize>, Vec<CompactCell>)> {
        let cell_count = self.request.rows.len();
        let mut remap = CandidateRemap::new(self.next_block_candidate, cell_count)?;
        let candidate_hint = self.next_block_candidate.min(cell_count);
        self.block_candidates.try_reserve_exact(candidate_hint)?;
        let mut latest_job = Vec::new();
        latest_job.try_reserve_exact(candidate_hint)?;
        let mut jobs = Vec::<TempBlockJob>::new();
        let initial_job_capacity = candidate_hint.saturating_mul(8).max(1_024).min(cell_count);
        jobs.try_reserve_exact(initial_job_capacity)?;
        let mut cell_blocks = Vec::new();
        cell_blocks.try_reserve_exact(cell_count)?;
        let mut cells = Vec::new();
        cells.try_reserve_exact(cell_count)?;

        enum RowCursor {
            General(std::vec::IntoIter<ResolvedRow>),
            ValidatedSingleDense { chunk_cells: u64, row_bytes: usize },
            OnDemand,
        }

        let mut resolved_rows = match resolved_rows {
            Some(PreloadedRows::General(rows)) => RowCursor::General(rows.into_iter()),
            Some(PreloadedRows::ValidatedSingleDense {
                chunk_cells,
                row_bytes,
            }) => RowCursor::ValidatedSingleDense {
                chunk_cells,
                row_bytes,
            },
            None => RowCursor::OnDemand,
        };
        for ordinal in 0..cell_count {
            let index = self.request.rows[ordinal];
            let resolved_row = match &mut resolved_rows {
                RowCursor::General(rows) => rows.next().ok_or_else(|| {
                    Error::Invariant("resolved row arena is shorter than the request".into())
                })?,
                RowCursor::ValidatedSingleDense {
                    chunk_cells,
                    row_bytes,
                } => self.resolve_validated_single_dense_row(index, *chunk_cells, *row_bytes)?,
                RowCursor::OnDemand => self.resolve_row(index)?,
            };
            let resolved = self.resolve_cell(ordinal, resolved_row, row_stride)?;
            let raw_key = resolved.block_key;
            let (block_key, is_new) = remap.intern(raw_key, self.block_candidates.len())?;
            if is_new {
                let candidate = BlockCandidate {
                    did: resolved.did,
                    data: resolved
                        .data
                        .as_ref()
                        .map(|block| self.block_info(block))
                        .transpose()?,
                    indices: resolved
                        .indices
                        .as_ref()
                        .map(|block| self.block_info(block))
                        .transpose()?,
                };
                self.validate_block_candidate(&candidate)?;
                self.block_candidates.push(candidate);
                latest_job.push(usize::MAX);
            }
            cells.push(CompactCell::from_resolved(&resolved));

            let batch = if let Some(shift) = self.batch_shift {
                ordinal >> shift
            } else {
                ordinal / self.request.batch_size
            };
            let latest = latest_job.get_mut(block_key).ok_or_else(|| {
                Error::Invariant("compact block candidate index is out of range".into())
            })?;
            let target = (*latest != usize::MAX
                && batch.saturating_sub(jobs[*latest].batch_min)
                    < self.request.config.coalescing_distance)
                .then_some(*latest);
            if let Some(job_id) = target {
                let job = &mut jobs[job_id];
                if job.cell_count == self.request.config.limits.max_cells_per_job {
                    return Err(Error::ResourceLimit(format!(
                        "same-block job at anchor {} exceeds max_cells_per_job",
                        job.anchor
                    )));
                }
                job.cell_count += 1;
                job.batch_max = batch;
                cell_blocks.push(job_id);
            } else {
                let job_id = jobs.len();
                jobs.push(TempBlockJob {
                    block_key,
                    cell_count: 1,
                    anchor: ordinal,
                    batch_min: batch,
                    batch_max: batch,
                });
                cell_blocks.push(job_id);
                *latest = job_id;
            }
        }
        if let RowCursor::General(mut rows) = resolved_rows {
            if rows.next().is_some() {
                return Err(Error::Invariant(
                    "resolved row arena is longer than the request".into(),
                ));
            }
        }
        self.next_block_candidate = self.block_candidates.len();
        Ok((jobs, cell_blocks, cells))
    }

    fn validate_block_candidate(&self, candidate: &BlockCandidate) -> Result<()> {
        let decoded = candidate
            .data
            .as_ref()
            .map_or(0, BlockInfo::decoded_len)
            .checked_add(candidate.indices.as_ref().map_or(0, BlockInfo::decoded_len))
            .ok_or_else(|| Error::ResourceLimit("decoded block size overflow".into()))?;
        if decoded > self.request.config.limits.max_decoded_bytes_per_job {
            return Err(Error::ResourceLimit(format!(
                "source block requires {decoded} decoded bytes, limit is {}",
                self.request.config.limits.max_decoded_bytes_per_job
            )));
        }
        for block in [candidate.data.as_ref(), candidate.indices.as_ref()]
            .into_iter()
            .flatten()
        {
            let encoded_len = range_len_u64(&block.encoded_range)?;
            if encoded_len > self.request.config.limits.max_encoded_bytes_per_side {
                return Err(Error::ResourceLimit(format!(
                    "source block requires {encoded_len} encoded bytes, limit is {}",
                    self.request.config.limits.max_encoded_bytes_per_side
                )));
            }
        }
        Ok(())
    }

    fn block_info(&self, block: &CellBlock) -> Result<BlockInfo> {
        let chunk = self
            .chunk(block.chunk_key)
            .ok_or_else(|| Error::Invariant("resolved block decoder chunk is missing".into()))?;
        let decoder = chunk.decoder.block_decoder(block.block_index)?;
        if decoder.decoded_len() != block.decoded_len
            || decoder.encoded_len() != range_len_u64(&block.encoded_range)?
        {
            return Err(Error::Invariant(
                "resolved block metadata disagrees with its decoder".into(),
            ));
        }
        Ok(BlockInfo {
            decoder,
            source: block.source,
            encoded_range: block.encoded_range.clone(),
        })
    }

    fn merge_runs(&self, block_jobs: &[TempBlockJob]) -> Result<MergedRuns> {
        let mut runs = Vec::new();
        let mut members = Vec::new();
        runs.try_reserve_exact(block_jobs.len())?;
        let maximum_member_nodes = block_jobs
            .len()
            .checked_mul(2)
            .ok_or_else(|| Error::ResourceLimit("run member node count overflow".into()))?;
        members.try_reserve_exact(maximum_member_nodes)?;
        let mut total = Cost { bytes: 0, ops: 0 };
        for (id, job) in block_jobs.iter().enumerate() {
            let member_root = members.len();
            members.push(MemberNode::Block(id));
            let run = self.make_run(member_root, job)?;
            let cost = run.cost()?;
            total.bytes = total
                .bytes
                .checked_add(cost.bytes)
                .ok_or_else(|| Error::ResourceLimit("predicted I/O bytes overflow".into()))?;
            total.ops = total
                .ops
                .checked_add(cost.ops)
                .ok_or_else(|| Error::ResourceLimit("predicted I/O ops overflow".into()))?;
            runs.push(run);
        }
        let threshold = self.request.config.io_bandwidth_bytes_per_second
            / self.request.config.io_operations_per_second
            + self.request.config.delta_bytes;
        let max_coalesced_io_bytes = u64::try_from(self.request.config.max_coalesced_io_bytes)
            .map_err(|_| Error::InvalidConfig("max_coalesced_io_bytes exceeds u64".into()))?;
        let mut candidates: HashMap<RunKey, usize> = HashMap::new();
        candidates.try_reserve(runs.len())?;

        for current_id in 0..runs.len() {
            let key = runs[current_id].run_key();
            let mut accepted = None;
            let right = runs[current_id].cost()?;
            // Request-order scheduling only needs the newest live predecessor.
            // Searching the whole prefetch window makes shuffled plans
            // quadratic when `coalescing_distance` is intentionally huge. The
            // physical-order pass below handles remaining spatial neighbors.
            let candidate_id = candidates.get(&key).copied().filter(|candidate_id| {
                runs[current_id]
                    .batch_min
                    .saturating_sub(runs[*candidate_id].batch_min)
                    < self.request.config.coalescing_distance
            });
            if let Some(candidate_id) = candidate_id {
                let Some(merged) = self.project_merge(&runs[candidate_id], &runs[current_id])?
                else {
                    candidates.insert(key, current_id);
                    continue;
                };
                let left = runs[candidate_id].cost()?;
                let combined = merged.cost()?;
                let new_bytes = total
                    .bytes
                    .checked_sub(left.bytes)
                    .and_then(|value| value.checked_sub(right.bytes))
                    .and_then(|value| value.checked_add(combined.bytes))
                    .ok_or_else(|| Error::Invariant("I/O byte accounting underflow".into()))?;
                let new_ops = total
                    .ops
                    .checked_sub(left.ops)
                    .and_then(|value| value.checked_sub(right.ops))
                    .and_then(|value| value.checked_add(combined.ops))
                    .ok_or_else(|| Error::Invariant("I/O operation accounting underflow".into()))?;
                let average = if new_ops == 0 {
                    0.0
                } else {
                    new_bytes as f64 / new_ops as f64
                };
                let separate_bytes = left.bytes.checked_add(right.bytes).ok_or_else(|| {
                    Error::ResourceLimit("separate I/O byte cost overflow".into())
                })?;
                // Coalescing overlapping or adjacent ranges cannot increase
                // physical bytes. It removes an operation without consuming
                // bandwidth budget, so it remains profitable even after the
                // global plan has crossed the configured bytes/op balance.
                if combined.bytes <= max_coalesced_io_bytes
                    && (combined.bytes <= separate_bytes || average < threshold)
                {
                    accepted = Some((candidate_id, merged, new_bytes, new_ops));
                }
            }
            if let Some((candidate_id, merged, new_bytes, new_ops)) = accepted {
                let member_root = members.len();
                members.push(MemberNode::Concat {
                    left: runs[candidate_id].member_root,
                    right: runs[current_id].member_root,
                });
                runs[candidate_id] = merged.into_run(member_root);
                runs[current_id].active = false;
                total = Cost {
                    bytes: new_bytes,
                    ops: new_ops,
                };
            } else {
                candidates.insert(key, current_id);
            }
        }

        // The online pass sees blocks in request order. With a shuffled access
        // sequence, a newly arrived interval can bridge two older intervals,
        // but a single online merge cannot revisit the second neighbor. A
        // physical-order consolidation pass closes the remaining spatial
        // components in O(n log n), applying the same bytes/op profitability
        // rule as the request-order pass.
        drop(candidates);
        let mut physical_groups: HashMap<RunKey, Vec<usize>> = HashMap::new();
        physical_groups.try_reserve(runs.len())?;
        for (run_id, run) in runs.iter().enumerate() {
            if run.active {
                let group = physical_groups.entry(run.run_key()).or_default();
                group.try_reserve(1)?;
                group.push(run_id);
            }
        }
        for run_ids in physical_groups.values_mut() {
            run_ids.sort_unstable_by_key(|run_id| {
                let run = &runs[*run_id];
                (
                    run.data_range.as_ref().map_or(0, |range| range.start),
                    run.indices_range.as_ref().map_or(0, |range| range.start),
                    run.anchor,
                )
            });
            let Some((&first, remaining)) = run_ids.split_first() else {
                continue;
            };
            let mut survivor = first;
            for &next in remaining {
                let Some(merged) = self.project_merge(&runs[survivor], &runs[next])? else {
                    survivor = next;
                    continue;
                };
                let left = runs[survivor].cost()?;
                let right = runs[next].cost()?;
                let combined = merged.cost()?;
                let separate_bytes = left.bytes.checked_add(right.bytes).ok_or_else(|| {
                    Error::ResourceLimit("physical consolidation byte cost overflow".into())
                })?;
                let merged_bytes = combined.bytes;
                let new_bytes = total
                    .bytes
                    .checked_sub(left.bytes)
                    .and_then(|value| value.checked_sub(right.bytes))
                    .and_then(|value| value.checked_add(combined.bytes))
                    .ok_or_else(|| Error::Invariant("I/O byte accounting underflow".into()))?;
                let new_ops = total
                    .ops
                    .checked_sub(left.ops)
                    .and_then(|value| value.checked_sub(right.ops))
                    .and_then(|value| value.checked_add(combined.ops))
                    .ok_or_else(|| Error::Invariant("I/O operation accounting underflow".into()))?;
                let average = if new_ops == 0 {
                    0.0
                } else {
                    new_bytes as f64 / new_ops as f64
                };
                if merged_bytes > max_coalesced_io_bytes
                    || (merged_bytes > separate_bytes && average >= threshold)
                {
                    survivor = next;
                    continue;
                }
                let member_root = members.len();
                members.push(MemberNode::Concat {
                    left: runs[survivor].member_root,
                    right: runs[next].member_root,
                });
                runs[survivor] = merged.into_run(member_root);
                runs[next].active = false;
                total = Cost {
                    bytes: new_bytes,
                    ops: new_ops,
                };
            }
        }
        Ok(MergedRuns { runs, members })
    }

    fn make_run(&self, member_root: usize, job: &TempBlockJob) -> Result<TempRun> {
        let candidate = self
            .block_candidates
            .get(job.block_key)
            .ok_or_else(|| Error::Invariant("block job candidate metadata is missing".into()))?;
        let data_source = candidate.data.as_ref().map_or(0, |block| block.source);
        let indices_source = candidate.indices.as_ref().map(|block| block.source);
        let data_range = self.block_physical_range(candidate.data.as_ref())?;
        let indices_range = self.block_physical_range(candidate.indices.as_ref())?;
        let data_decoded = candidate.data.as_ref().map_or(0, BlockInfo::decoded_len);
        let indices_decoded = candidate.indices.as_ref().map_or(0, BlockInfo::decoded_len);
        Ok(TempRun {
            member_root,
            block_count: 1,
            anchor: job.anchor,
            batch_min: job.batch_min,
            batch_max: job.batch_max,
            active: true,
            did: candidate.did,
            data_source,
            indices_source,
            data_range,
            indices_range,
            data_decoded,
            indices_decoded,
            cell_count: job.cell_count,
        })
    }

    fn block_physical_range(&self, block: Option<&BlockInfo>) -> Result<Option<Range<u64>>> {
        let Some(block) = block else {
            return Ok(None);
        };
        let source = &self.sources[block.source];
        if let ReadSource::WholeKey { declared_len, .. } = source {
            let declared_len = u64::try_from(*declared_len)
                .map_err(|_| Error::ResourceLimit("whole-key length exceeds u64".into()))?;
            return Ok(Some(0..declared_len));
        }
        Ok(Some(block.encoded_range.clone()))
    }

    fn project_merge(&self, left: &TempRun, right: &TempRun) -> Result<Option<MergeProjection>> {
        let batch_min = left.batch_min.min(right.batch_min);
        let batch_max = left.batch_max.max(right.batch_max);
        let batch_span = batch_max
            .saturating_sub(batch_min)
            .checked_add(1)
            .ok_or_else(|| Error::ResourceLimit("merged batch span overflow".into()))?;
        if batch_span > self.request.config.coalescing_distance {
            return Ok(None);
        }
        let block_count = left
            .block_count
            .checked_add(right.block_count)
            .ok_or_else(|| Error::ResourceLimit("merged block count overflow".into()))?;
        if block_count > self.request.config.limits.max_blocks_per_job {
            return Ok(None);
        }
        let cell_count = left
            .cell_count
            .checked_add(right.cell_count)
            .ok_or_else(|| Error::ResourceLimit("merged cell count overflow".into()))?;
        if cell_count > self.request.config.limits.max_cells_per_job {
            return Ok(None);
        }
        let data_decoded = left
            .data_decoded
            .checked_add(right.data_decoded)
            .ok_or_else(|| Error::ResourceLimit("decoded data bytes overflow".into()))?;
        let indices_decoded = left
            .indices_decoded
            .checked_add(right.indices_decoded)
            .ok_or_else(|| Error::ResourceLimit("decoded indices bytes overflow".into()))?;
        let decoded = data_decoded
            .checked_add(indices_decoded)
            .ok_or_else(|| Error::ResourceLimit("merged decoded bytes overflow".into()))?;
        if decoded > self.request.config.limits.max_decoded_bytes_per_job
            || decoded > self.request.config.target_decoded_bytes_per_job
        {
            return Ok(None);
        }
        let data_range = union_optional_range(left.data_range.clone(), right.data_range.clone());
        let indices_range =
            union_optional_range(left.indices_range.clone(), right.indices_range.clone());
        if let Some(range) = &data_range {
            if range_len_u64(range)? > self.request.config.limits.max_encoded_bytes_per_side {
                return Ok(None);
            }
        }
        if let Some(range) = &indices_range {
            if range_len_u64(range)? > self.request.config.limits.max_encoded_bytes_per_side {
                return Ok(None);
            }
        }
        Ok(Some(MergeProjection {
            anchor: left.anchor.min(right.anchor),
            batch_min,
            batch_max,
            did: left.did,
            data_source: left.data_source,
            indices_source: left.indices_source,
            data_range,
            indices_range,
            data_decoded,
            indices_decoded,
            cell_count,
            block_count,
        }))
    }

    fn finalize(
        &self,
        cells: &[CompactCell],
        block_jobs: &[TempBlockJob],
        cell_blocks: &[usize],
        merged: &MergedRuns,
        output_ring_bytes: usize,
    ) -> Result<FinalizedPlan> {
        let runs = &merged.runs;
        let active_run_count = runs.iter().filter(|run| run.active).count();
        let block_spec_count = runs
            .iter()
            .filter(|run| run.active)
            .try_fold(0usize, |total, run| {
                let sides = usize::from(run.data_range.is_some())
                    + usize::from(run.indices_range.is_some());
                run.block_count
                    .checked_mul(sides)
                    .and_then(|count| total.checked_add(count))
            })
            .ok_or_else(|| Error::ResourceLimit("compile block arena count overflow".into()))?;
        let completion_capacity = runs.iter().filter(|run| run.active).try_fold(
            0usize,
            |total, run| -> Result<usize> {
                let batch_span = run
                    .batch_max
                    .checked_sub(run.batch_min)
                    .and_then(|span| span.checked_add(1))
                    .ok_or_else(|| Error::Invariant("final run batch range is invalid".into()))?;
                let distinct_batches = run.cell_count.min(batch_span);
                let split_descriptors =
                    run.cell_count.saturating_sub(distinct_batches) / u32::MAX as usize;
                total
                    .checked_add(distinct_batches)
                    .and_then(|count| count.checked_add(split_descriptors))
                    .ok_or_else(|| {
                        Error::ResourceLimit("batch completion arena count overflow".into())
                    })
            },
        )?;
        let arena_bytes = cells
            .len()
            .checked_mul(size_of::<CellTask>())
            .and_then(|bytes| {
                completion_capacity
                    .checked_mul(size_of::<BatchCompletion>())
                    .and_then(|completion_bytes| bytes.checked_add(completion_bytes))
            })
            .and_then(|bytes| {
                block_spec_count
                    .checked_mul(size_of::<BlockSpec>())
                    .and_then(|block_bytes| bytes.checked_add(block_bytes))
            })
            .and_then(|bytes| {
                block_jobs
                    .len()
                    .checked_mul(size_of::<BlockGroup>())
                    .and_then(|group_bytes| bytes.checked_add(group_bytes))
            })
            .and_then(|bytes| {
                active_run_count
                    .checked_mul(size_of::<Job>())
                    .and_then(|job_bytes| bytes.checked_add(job_bytes))
            })
            .ok_or_else(|| Error::ResourceLimit("compile arena byte count overflow".into()))?;
        if arena_bytes > self.request.config.limits.max_compile_arena_bytes {
            return Err(Error::ResourceLimit(format!(
                "compile arena has {arena_bytes} bytes, limit is {}",
                self.request.config.limits.max_compile_arena_bytes
            )));
        }
        let retained_bytes = cells
            .len()
            .checked_mul(size_of::<CompactCell>())
            .and_then(|bytes| {
                block_jobs
                    .len()
                    .checked_mul(size_of::<TempBlockJob>())
                    .and_then(|job_bytes| bytes.checked_add(job_bytes))
            })
            .and_then(|bytes| {
                cell_blocks
                    .len()
                    .checked_mul(size_of::<usize>())
                    .and_then(|member_bytes| bytes.checked_add(member_bytes))
            })
            .and_then(|bytes| {
                runs.len()
                    .checked_mul(size_of::<TempRun>())
                    .and_then(|run_bytes| bytes.checked_add(run_bytes))
            })
            .and_then(|bytes| {
                merged
                    .members
                    .len()
                    .checked_mul(size_of::<MemberNode>())
                    .and_then(|node_bytes| bytes.checked_add(node_bytes))
            })
            .and_then(|bytes| {
                self.block_candidates
                    .len()
                    .checked_mul(size_of::<BlockCandidate>())
                    .and_then(|candidate_bytes| bytes.checked_add(candidate_bytes))
            })
            .and_then(|bytes| bytes.checked_add(self.chunk_metadata_bytes))
            .and_then(|bytes| bytes.checked_add(self.retained_whole_key_bytes))
            .ok_or_else(|| Error::ResourceLimit("retained compile byte count overflow".into()))?;
        let maximum_run_blocks = runs
            .iter()
            .filter(|run| run.active)
            .map(|run| run.block_count)
            .max()
            .unwrap_or(0);
        let finalize_temporary_bytes = block_jobs
            .len()
            .checked_mul(size_of::<usize>())
            .and_then(|bytes| {
                block_jobs
                    .len()
                    .checked_mul(size_of::<usize>())
                    .and_then(|value| value.checked_mul(3))
                    .and_then(|value| bytes.checked_add(value))
            })
            .and_then(|bytes| {
                runs.len()
                    .checked_mul(size_of::<Range<usize>>())
                    .and_then(|value| value.checked_mul(3))
                    .and_then(|value| bytes.checked_add(value))
            })
            .and_then(|bytes| {
                runs.len()
                    .checked_mul(size_of::<usize>())
                    .and_then(|value| {
                        active_run_count
                            .checked_mul(size_of::<usize>())
                            .and_then(|active_bytes| value.checked_add(active_bytes))
                    })
                    .and_then(|value| bytes.checked_add(value))
            })
            .and_then(|bytes| {
                cells
                    .len()
                    .checked_mul(size_of::<(usize, usize)>())
                    .and_then(|value| bytes.checked_add(value))
            })
            .and_then(|bytes| {
                maximum_run_blocks
                    .checked_mul(size_of::<usize>())
                    .and_then(|value| bytes.checked_add(value))
            })
            .ok_or_else(|| Error::ResourceLimit("finalize temporary byte count overflow".into()))?;
        let working_set = arena_bytes
            .checked_add(retained_bytes)
            .and_then(|bytes| bytes.checked_add(finalize_temporary_bytes))
            .ok_or_else(|| Error::ResourceLimit("compile working set overflow".into()))?;
        if working_set > self.request.config.limits.max_compile_working_set_bytes {
            return Err(Error::ResourceLimit(format!(
                "compile working-set payload requires {working_set} bytes, limit is {}",
                self.request.config.limits.max_compile_working_set_bytes
            )));
        }

        let mut active_ids = Vec::new();
        active_ids.try_reserve_exact(active_run_count)?;
        let mut run_blocks = Vec::new();
        run_blocks.try_reserve_exact(block_jobs.len())?;
        let mut run_block_ranges = vec![0..0; runs.len()];
        let mut traversal = Vec::new();
        traversal.try_reserve_exact(maximum_run_blocks)?;
        for (run_id, run) in runs.iter().enumerate().filter(|(_, run)| run.active) {
            active_ids.push(run_id);
            let start = run_blocks.len();
            append_run_members(
                run.member_root,
                &merged.members,
                &mut traversal,
                &mut run_blocks,
            )?;
            let end = run_blocks.len();
            if end - start != run.block_count {
                return Err(Error::Invariant("run member count is inconsistent".into()));
            }
            run_block_ranges[run_id] = start..end;
        }

        let mut block_to_run = vec![usize::MAX; block_jobs.len()];
        for &run_id in &active_ids {
            for &block_id in &run_blocks[run_block_ranges[run_id].clone()] {
                let owner = block_to_run
                    .get_mut(block_id)
                    .ok_or_else(|| Error::Invariant("run member block is out of range".into()))?;
                if *owner != usize::MAX {
                    return Err(Error::Invariant(
                        "block belongs to multiple final runs".into(),
                    ));
                }
                *owner = run_id;
            }
        }

        let mut run_cell_ranges = vec![0..0; runs.len()];
        let mut next_cell = 0usize;
        for &run_id in &active_ids {
            let end = next_cell
                .checked_add(runs[run_id].cell_count)
                .ok_or_else(|| Error::ResourceLimit("ordered cell range overflow".into()))?;
            run_cell_ranges[run_id] = next_cell..end;
            next_cell = end;
        }
        if next_cell != cells.len() || cell_blocks.len() != cells.len() {
            return Err(Error::Invariant(
                "final runs do not cover every cell exactly once".into(),
            ));
        }
        let mut cell_cursors = run_cell_ranges
            .iter()
            .map(|range| range.start)
            .collect::<Vec<_>>();
        let mut ordered_cells = vec![(usize::MAX, usize::MAX); cells.len()];
        for (cell_id, &block_id) in cell_blocks.iter().enumerate() {
            let run_id = *block_to_run
                .get(block_id)
                .ok_or_else(|| Error::Invariant("cell block is out of range".into()))?;
            if run_id == usize::MAX {
                return Err(Error::Invariant("cell block has no final run".into()));
            }
            let cursor = &mut cell_cursors[run_id];
            if *cursor >= run_cell_ranges[run_id].end {
                return Err(Error::Invariant("final run cell range overflow".into()));
            }
            ordered_cells[*cursor] = (cell_id, block_id);
            *cursor += 1;
        }
        for &run_id in &active_ids {
            if cell_cursors[run_id] != run_cell_ranges[run_id].end {
                return Err(Error::Invariant(
                    "final run cell range is incomplete".into(),
                ));
            }
        }

        let mut arena_completions = Vec::new();
        arena_completions.try_reserve_exact(completion_capacity)?;
        let mut run_completion_ranges = vec![0..0; runs.len()];
        for &run_id in &active_ids {
            let start = arena_completions.len();
            let ordered = &ordered_cells[run_cell_ranges[run_id].clone()];
            let mut index = 0usize;
            while index < ordered.len() {
                let logical_batch = if let Some(shift) = self.batch_shift {
                    ordered[index].0 >> shift
                } else {
                    ordered[index].0 / self.request.batch_size
                };
                let mut end = index + 1;
                while end < ordered.len() {
                    let next_batch = if let Some(shift) = self.batch_shift {
                        ordered[end].0 >> shift
                    } else {
                        ordered[end].0 / self.request.batch_size
                    };
                    if next_batch != logical_batch {
                        break;
                    }
                    end += 1;
                }
                let ring_batch = if self.ring_mask == usize::MAX {
                    logical_batch % self.ring_slots
                } else {
                    logical_batch & self.ring_mask
                };
                append_batch_completions(&mut arena_completions, ring_batch, end - index)?;
                index = end;
            }
            run_completion_ranges[run_id] = start..arena_completions.len();
        }

        let mut block_rank = vec![usize::MAX; block_jobs.len()];
        for (rank, &block_id) in run_blocks.iter().enumerate() {
            *block_rank
                .get_mut(block_id)
                .ok_or_else(|| Error::Invariant("run block rank is out of range".into()))? = rank;
        }
        for &run_id in &active_ids {
            ordered_cells[run_cell_ranges[run_id].clone()]
                .sort_unstable_by_key(|&(cell_id, block_id)| (block_rank[block_id], cell_id));
        }

        let mut direct_outputs = vec![usize::MAX; block_jobs.len()];
        let logical_row_bytes = self
            .request
            .output
            .n_cols
            .checked_mul(self.request.output.dtype.size())
            .ok_or_else(|| Error::ResourceLimit("direct output row size overflow".into()))?;
        let mut ordered_start = 0usize;
        while ordered_start < ordered_cells.len() {
            let block_id = ordered_cells[ordered_start].1;
            let mut ordered_end = ordered_start + 1;
            while ordered_end < ordered_cells.len() && ordered_cells[ordered_end].1 == block_id {
                ordered_end += 1;
            }
            let block_job = &block_jobs[block_id];
            let candidate = &self.block_candidates[block_job.block_key];
            let Some(data) = candidate.data.as_ref() else {
                ordered_start = ordered_end;
                continue;
            };
            let source_plan = &self.source_plans[candidate.did as usize];
            if candidate.indices.is_none()
                && source_plan.can_decode_direct()
                && block_job.cell_count == ordered_end - ordered_start
            {
                let first_output = cells[ordered_cells[ordered_start].0]
                    .output_slot
                    .row_offset();
                let mut decoded_cursor = 0usize;
                let mut eligible = true;
                for &(cell_id, _) in &ordered_cells[ordered_start..ordered_end] {
                    let cell = &cells[cell_id];
                    let expected_end =
                        decoded_cursor
                            .checked_add(logical_row_bytes)
                            .ok_or_else(|| {
                                Error::ResourceLimit("direct decoded extent overflow".into())
                            })?;
                    let expected_output =
                        first_output.checked_add(decoded_cursor).ok_or_else(|| {
                            Error::ResourceLimit("direct output extent overflow".into())
                        })?;
                    if cell.data_range() != (decoded_cursor..expected_end)
                        || cell.output_slot.row_offset() != expected_output
                    {
                        eligible = false;
                        break;
                    }
                    decoded_cursor = expected_end;
                }
                if eligible && decoded_cursor == data.decoded_len() {
                    direct_outputs[block_id] = first_output;
                }
            }
            ordered_start = ordered_end;
        }

        let mut jobs = Vec::new();
        let mut arena_groups = Vec::new();
        let mut arena_cells = Vec::new();
        let mut arena_blocks = Vec::new();
        let mut stats = PlanStats {
            output_ring_bytes,
            ..PlanStats::default()
        };
        jobs.try_reserve_exact(active_run_count)?;
        arena_groups.try_reserve_exact(block_jobs.len())?;
        arena_cells.try_reserve_exact(cells.len())?;
        arena_blocks.try_reserve_exact(block_spec_count)?;
        let mut data_specs = vec![usize::MAX; block_jobs.len()];
        let mut indices_specs = vec![usize::MAX; block_jobs.len()];

        for &run_id in &active_ids {
            let run = &runs[run_id];
            let block_ids = &run_blocks[run_block_ranges[run_id].clone()];
            let data_read = run.data_range.clone().unwrap_or(0..0);
            let indices_read = run.indices_range.clone();
            let data_source = run.data_source;
            let indices_source = run.indices_source;

            #[cfg(feature = "profile")]
            let data_blocks_start = arena_blocks.len();
            let mut data_decoded = 0usize;
            for block_id in block_ids {
                let candidate = &self.block_candidates[block_jobs[*block_id].block_key];
                let Some(block) = &candidate.data else {
                    continue;
                };
                let direct_output = direct_outputs[*block_id];
                data_decoded = data_decoded
                    .checked_add(block.decoded_len())
                    .ok_or_else(|| Error::ResourceLimit("decoded data bytes overflow".into()))?;
                let encoded_range = relative_range(&block.encoded_range, &data_read)?;
                let direct_output_plus_one = if direct_output == usize::MAX {
                    0
                } else {
                    direct_output.checked_add(1).ok_or_else(|| {
                        Error::ResourceLimit("direct output offset overflow".into())
                    })?
                };
                let spec_id = arena_blocks.len();
                arena_blocks.push(
                    BlockSpec::new(block.decoder, encoded_range, direct_output_plus_one)
                        .ok_or_else(|| {
                            Error::ResourceLimit(
                                "block spec exceeds compact offset representation".into(),
                            )
                        })?,
                );
                data_specs[*block_id] = spec_id;
            }
            #[cfg(feature = "profile")]
            let data_blocks_end = arena_blocks.len();

            #[cfg(feature = "profile")]
            let indices_blocks_start = arena_blocks.len();
            let mut indices_decoded = 0usize;
            for block_id in block_ids {
                let candidate = &self.block_candidates[block_jobs[*block_id].block_key];
                let Some(block) = &candidate.indices else {
                    continue;
                };
                indices_decoded = indices_decoded
                    .checked_add(block.decoded_len())
                    .ok_or_else(|| Error::ResourceLimit("decoded indices bytes overflow".into()))?;
                let read = indices_read.as_ref().ok_or_else(|| {
                    Error::Invariant("CSR indices block has no physical read".into())
                })?;
                let spec_id = arena_blocks.len();
                arena_blocks.push(
                    BlockSpec::new(
                        block.decoder,
                        relative_range(&block.encoded_range, read)?,
                        0,
                    )
                    .ok_or_else(|| {
                        Error::ResourceLimit(
                            "indices block spec exceeds compact offset representation".into(),
                        )
                    })?,
                );
                indices_specs[*block_id] = spec_id;
            }
            #[cfg(feature = "profile")]
            let indices_blocks_end = arena_blocks.len();

            let groups_start = arena_groups.len();
            let ordered = &ordered_cells[run_cell_ranges[run_id].clone()];
            let mut ordered_cursor = 0usize;
            for &block_id in block_ids {
                let candidate = &self.block_candidates[block_jobs[block_id].block_key];
                let group_cells_start = arena_cells.len();
                while ordered_cursor < ordered.len() && ordered[ordered_cursor].1 == block_id {
                    let cell = &cells[ordered[ordered_cursor].0];
                    let direct_decode = direct_outputs[block_id] != usize::MAX;
                    let data_range = if direct_decode || candidate.data.is_none() {
                        0..0
                    } else {
                        cell.data_range()
                    };
                    let indices_range = if candidate.indices.is_some() {
                        Some(cell.indices_range().ok_or_else(|| {
                            Error::Invariant("CSR compact cell has no indices range".into())
                        })?)
                    } else {
                        None
                    };
                    let task =
                        CellTask::new(cell.output_slot, data_range, indices_range).ok_or_else(
                            || {
                                Error::ResourceLimit(
                                    "cell task exceeds compact decoded-offset/aligned-row representation"
                                        .into(),
                                )
                            },
                        )?;
                    arena_cells.push(if direct_decode {
                        task.with_direct_decode()
                    } else {
                        task
                    });
                    ordered_cursor += 1;
                }
                if group_cells_start == arena_cells.len() {
                    return Err(Error::Invariant("block group has no cell tasks".into()));
                }
                let data_block = if candidate.data.is_some() {
                    let spec = data_specs[block_id];
                    if spec == usize::MAX {
                        return Err(Error::Invariant("data block spec is missing".into()));
                    }
                    Some(spec)
                } else {
                    None
                };
                let indices_block = if candidate.indices.is_some() {
                    let spec = indices_specs[block_id];
                    if spec == usize::MAX {
                        return Err(Error::Invariant("indices block spec is missing".into()));
                    }
                    Some(spec)
                } else {
                    None
                };
                arena_groups.push(
                    BlockGroup::new(
                        data_block,
                        indices_block,
                        group_cells_start..arena_cells.len(),
                    )
                    .ok_or_else(|| {
                        Error::ResourceLimit("block group index exceeds compact encoding".into())
                    })?,
                );
            }
            if ordered_cursor != ordered.len() {
                return Err(Error::Invariant(
                    "block groups do not cover every run cell".into(),
                ));
            }
            let groups_end = arena_groups.len();
            let right_exclusive = run
                .batch_max
                .checked_add(1)
                .ok_or_else(|| Error::ResourceLimit("job batch range overflow".into()))?;
            let start_step = right_exclusive.saturating_sub(self.request.prefetch_step);
            let data = JobSide {
                source: data_source,
                read_range: data_read,
                #[cfg(feature = "profile")]
                blocks: data_blocks_start..data_blocks_end,
            };
            let indices = if let Some(read_range) = indices_read {
                Some(JobSide {
                    source: indices_source.ok_or_else(|| {
                        Error::Invariant("indices read has no registered source".into())
                    })?,
                    read_range,
                    #[cfg(feature = "profile")]
                    blocks: indices_blocks_start..indices_blocks_end,
                })
            } else {
                None
            };
            let data_encoded = data.encoded_len(&self.sources[data.source]);
            let indices_encoded = indices
                .as_ref()
                .map_or(0, |side| side.encoded_len(&self.sources[side.source]));
            stats.maximum_encoded_bytes_per_side = stats
                .maximum_encoded_bytes_per_side
                .max(data_encoded)
                .max(indices_encoded);
            let decoded_bytes = data_decoded
                .checked_add(indices_decoded)
                .ok_or_else(|| Error::ResourceLimit("decoded job byte count overflow".into()))?;
            stats.maximum_decoded_bytes_per_job =
                stats.maximum_decoded_bytes_per_job.max(decoded_bytes);
            if data_encoded > 0 {
                stats.data_io_ops = stats
                    .data_io_ops
                    .checked_add(1)
                    .ok_or_else(|| Error::ResourceLimit("data I/O count overflow".into()))?;
                let data_encoded = u64::try_from(data_encoded).map_err(|_| {
                    Error::ResourceLimit("predicted data byte count exceeds u64".into())
                })?;
                stats.predicted_physical_bytes = stats
                    .predicted_physical_bytes
                    .checked_add(data_encoded)
                    .ok_or_else(|| Error::ResourceLimit("predicted bytes overflow".into()))?;
            }
            if indices_encoded > 0 {
                stats.indices_io_ops = stats
                    .indices_io_ops
                    .checked_add(1)
                    .ok_or_else(|| Error::ResourceLimit("indices I/O count overflow".into()))?;
                let indices_encoded = u64::try_from(indices_encoded).map_err(|_| {
                    Error::ResourceLimit("predicted indices byte count exceeds u64".into())
                })?;
                stats.predicted_physical_bytes = stats
                    .predicted_physical_bytes
                    .checked_add(indices_encoded)
                    .ok_or_else(|| Error::ResourceLimit("predicted bytes overflow".into()))?;
            }
            let block_encoded: usize =
                block_ids
                    .iter()
                    .try_fold(0usize, |total, id| -> Result<usize> {
                        let candidate = &self.block_candidates[block_jobs[*id].block_key];
                        let data_bytes = candidate
                            .data
                            .as_ref()
                            .map(|block| range_len_u64(&block.encoded_range))
                            .transpose()?
                            .unwrap_or(0);
                        let indices_bytes = candidate
                            .indices
                            .as_ref()
                            .map(|block| range_len_u64(&block.encoded_range))
                            .transpose()?
                            .unwrap_or(0);
                        let bytes = data_bytes.checked_add(indices_bytes).ok_or_else(|| {
                            Error::ResourceLimit("block byte count overflow".into())
                        })?;
                        total
                            .checked_add(bytes)
                            .ok_or_else(|| Error::ResourceLimit("block byte count overflow".into()))
                    })?;
            let gap_bytes = data_encoded
                .checked_add(indices_encoded)
                .and_then(|bytes| bytes.checked_sub(block_encoded))
                .ok_or_else(|| Error::Invariant("gap byte accounting underflow".into()))?;
            let gap_bytes = u64::try_from(gap_bytes)
                .map_err(|_| Error::ResourceLimit("gap byte count exceeds u64".into()))?;
            stats.gap_bytes = stats
                .gap_bytes
                .checked_add(gap_bytes)
                .ok_or_else(|| Error::ResourceLimit("gap byte count overflow".into()))?;
            jobs.push(Job {
                source_plan: run.did,
                completions: run_completion_ranges[run_id].clone(),
                groups: groups_start..groups_end,
                data,
                indices,
                start_step,
                anchor: run.anchor,
                #[cfg(all(feature = "uring", target_os = "linux"))]
                batch_min: run.batch_min,
                #[cfg(all(feature = "uring", target_os = "linux"))]
                batch_max: run.batch_max,
            });
        }
        jobs.sort_by_key(|job| (job.start_step, job.anchor));
        for pair in jobs.windows(2) {
            if pair[0].start_step > pair[1].start_step {
                return Err(Error::Invariant(
                    "job start_step order is not monotonic".into(),
                ));
            }
        }
        stats.arena_bytes = arena_bytes;
        stats.compile_working_set_bytes = working_set.max(self.peak_compile_payload_bytes);
        stats.retained_whole_key_bytes = self.retained_whole_key_bytes;
        Ok((
            jobs,
            arena_groups,
            arena_cells,
            arena_completions,
            arena_blocks,
            stats,
        ))
    }
}

fn runtime_envelope(
    sources: &[ReadSource],
    jobs: &[Job],
    groups: &[BlockGroup],
    blocks: &[BlockSpec],
) -> Result<crate::plan::RuntimeEnvelope> {
    let all_positioned = sources
        .iter()
        .all(|source| matches!(source, ReadSource::Empty | ReadSource::Positioned { .. }));
    #[cfg(target_os = "linux")]
    let has_fuse_source = {
        const FUSE_SUPER_MAGIC: u64 = 0x6573_5546;

        sources.iter().any(|source| {
            let ReadSource::Positioned { file, .. } = source else {
                return false;
            };
            rustix::fs::fstatfs(file).is_ok_and(|stats| stats.f_type as u64 == FUSE_SUPER_MAGIC)
        })
    };
    #[cfg(not(target_os = "linux"))]
    let has_fuse_source = false;

    let mut envelope = crate::plan::RuntimeEnvelope {
        all_positioned,
        has_fuse_source,
        ..crate::plan::RuntimeEnvelope::default()
    };
    for job in jobs {
        let data_encoded = job.data.encoded_len(&sources[job.data.source]);
        let indices_encoded = job
            .indices
            .as_ref()
            .map_or(0, |side| side.encoded_len(&sources[side.source]));
        let combined_encoded = data_encoded
            .checked_add(indices_encoded)
            .ok_or_else(|| Error::ResourceLimit("job encoded byte count overflow".into()))?;
        envelope.maximum_data_encoded = envelope.maximum_data_encoded.max(data_encoded);
        envelope.maximum_indices_encoded = envelope.maximum_indices_encoded.max(indices_encoded);
        envelope.maximum_combined_encoded = envelope.maximum_combined_encoded.max(combined_encoded);
    }
    for group in groups {
        let data_decoded = match group.data_block() {
            Some(block) => {
                let block = blocks
                    .get(block)
                    .ok_or_else(|| Error::Invariant("data block group index is invalid".into()))?;
                if block.direct_output().is_some() {
                    0
                } else {
                    block.decoded_len()
                }
            }
            None => 0,
        };
        let indices_decoded = match group.indices_block() {
            Some(block) => blocks
                .get(block)
                .ok_or_else(|| Error::Invariant("indices block group index is invalid".into()))?
                .decoded_len(),
            None => 0,
        };
        envelope.maximum_data_decoded = envelope.maximum_data_decoded.max(data_decoded);
        envelope.maximum_indices_decoded = envelope.maximum_indices_decoded.max(indices_decoded);
    }
    Ok(envelope)
}

fn build_dense_map(
    targets: Vec<Option<usize>>,
    source_size: usize,
    target_size: usize,
    output_columns: usize,
    gather_min_entries: Option<usize>,
) -> Result<DenseMap> {
    let mut mapped = 0usize;
    let mut run_count = 0usize;
    let mut previous: Option<(usize, usize)> = None;
    let mut first_target_byte = None;
    let mut targets_contiguous = true;
    for (source_column, target) in targets.iter().enumerate() {
        let Some(target) = *target else { continue };
        let source_byte = source_column
            .checked_mul(source_size)
            .ok_or_else(|| Error::ResourceLimit("dense map source byte offset overflow".into()))?;
        let target_byte = target
            .checked_mul(target_size)
            .ok_or_else(|| Error::ResourceLimit("dense map target byte offset overflow".into()))?;
        mapped = mapped
            .checked_add(1)
            .ok_or_else(|| Error::ResourceLimit("dense map entry count overflow".into()))?;
        if previous.is_none_or(|(previous_source, previous_target)| {
            previous_source.checked_add(source_size) != Some(source_byte)
                || previous_target.checked_add(target_size) != Some(target_byte)
        }) {
            run_count = run_count
                .checked_add(1)
                .ok_or_else(|| Error::ResourceLimit("dense map run count overflow".into()))?;
        }
        if let Some((_, previous_target)) = previous {
            targets_contiguous &= previous_target.checked_add(target_size) == Some(target_byte);
        } else {
            first_target_byte = Some(target_byte);
        }
        previous = Some((source_byte, target_byte));
    }
    let covers_output = mapped == output_columns;
    let packed = targets
        .len()
        .checked_mul(source_size)
        .is_some_and(|bytes| bytes <= u32::MAX as usize)
        && output_columns
            .checked_mul(target_size)
            .is_some_and(|bytes| bytes <= u32::MAX as usize);
    let entry_bytes = if packed {
        mapped.saturating_mul(size_of::<u64>())
    } else {
        mapped.saturating_mul(size_of::<DenseMapEntry>())
    };
    let run_bytes = run_count.saturating_mul(size_of::<DenseMapRun>());
    if run_bytes < entry_bytes {
        let mut runs: Vec<DenseMapRun> = Vec::new();
        runs.try_reserve_exact(run_count)?;
        for (source_column, target) in targets.into_iter().enumerate() {
            let Some(target) = target else { continue };
            let source_byte = source_column.checked_mul(source_size).ok_or_else(|| {
                Error::ResourceLimit("dense map source byte offset overflow".into())
            })?;
            let target_byte = target.checked_mul(target_size).ok_or_else(|| {
                Error::ResourceLimit("dense map target byte offset overflow".into())
            })?;
            let extend = runs.last().is_some_and(|run| {
                run.count
                    .checked_mul(source_size)
                    .and_then(|bytes| run.source_byte.checked_add(bytes))
                    == Some(source_byte)
                    && run
                        .count
                        .checked_mul(target_size)
                        .and_then(|bytes| run.target_byte.checked_add(bytes))
                        == Some(target_byte)
            });
            if extend {
                let run = runs.last_mut().expect("the preceding check found a run");
                run.count = run
                    .count
                    .checked_add(1)
                    .ok_or_else(|| Error::ResourceLimit("dense map run length overflow".into()))?;
            } else {
                runs.push(DenseMapRun {
                    source_byte,
                    target_byte,
                    count: 1,
                });
            }
        }
        return Ok(DenseMap::Runs {
            entries: Arc::from(runs),
            covers_output,
        });
    }
    let gather32 = packed
        && gather_min_entries.is_some_and(|minimum| mapped >= minimum)
        && targets_contiguous
        && targets
            .len()
            .checked_mul(source_size)
            .is_some_and(|bytes| bytes <= i32::MAX as usize);
    if gather32 {
        let mut source_offsets = Vec::new();
        source_offsets.try_reserve_exact(mapped)?;
        for (source_column, target) in targets.into_iter().enumerate() {
            if target.is_none() {
                continue;
            }
            let source = source_column
                .checked_mul(source_size)
                .and_then(|offset| i32::try_from(offset).ok())
                .ok_or_else(|| {
                    Error::ResourceLimit("dense gather source offset exceeds i32".into())
                })?;
            source_offsets.push(source);
        }
        let target_byte = first_target_byte
            .and_then(|offset| u32::try_from(offset).ok())
            .ok_or_else(|| Error::ResourceLimit("dense gather target offset exceeds u32".into()))?;
        return Ok(DenseMap::Gather32 {
            source_offsets: Arc::from(source_offsets),
            target_byte,
            covers_output,
        });
    }
    if packed {
        let mut packed_entries = Vec::new();
        packed_entries.try_reserve_exact(mapped)?;
        for (source_column, target) in targets.into_iter().enumerate() {
            let Some(target) = target else { continue };
            let source = source_column
                .checked_mul(source_size)
                .and_then(|offset| u32::try_from(offset).ok())
                .ok_or_else(|| {
                    Error::ResourceLimit("dense packed source offset exceeds u32".into())
                })?;
            let target = target
                .checked_mul(target_size)
                .and_then(|offset| u32::try_from(offset).ok())
                .ok_or_else(|| {
                    Error::ResourceLimit("dense packed target offset exceeds u32".into())
                })?;
            packed_entries.push(u64::from(source) | (u64::from(target) << 32));
        }
        Ok(DenseMap::Packed32 {
            entries: Arc::from(packed_entries),
            covers_output,
        })
    } else {
        let mut entries = Vec::new();
        entries.try_reserve_exact(mapped)?;
        for (source_column, target) in targets.into_iter().enumerate() {
            let Some(target) = target else { continue };
            entries.push(DenseMapEntry {
                source_byte: source_column.checked_mul(source_size).ok_or_else(|| {
                    Error::ResourceLimit("dense map source byte offset overflow".into())
                })?,
                target_byte: target.checked_mul(target_size).ok_or_else(|| {
                    Error::ResourceLimit("dense map target byte offset overflow".into())
                })?,
            });
        }
        Ok(DenseMap::Wide {
            entries: Arc::from(entries),
            covers_output,
        })
    }
}

fn load_chunk_detached(
    source: &Source,
    side: Side,
    chunk: usize,
    maximum_whole_key_bytes: usize,
    retained_whole_key_bytes: &AtomicUsize,
    maximum_retained_whole_key_bytes: usize,
) -> Result<LoadedChunk> {
    const PREFIX_PROBE_BYTES: usize = 4096;

    let path = match (&source.dataset.kind, side) {
        (DatasetKind::Dense(meta), Side::Data) => &meta.data.path,
        (DatasetKind::Csr { meta, .. }, Side::Data) => &meta.data.path,
        (DatasetKind::Csr { meta, .. }, Side::Indices) => &meta.indices.path,
        (DatasetKind::Dense(_), Side::Indices) => {
            return Err(Error::Invariant("dense source has no indices side".into()))
        }
    };
    let chunk_number = u64::try_from(chunk)
        .map_err(|_| Error::InvalidDataset("chunk index exceeds u64".into()))?;
    let key_owned = chunk_key(path, chunk_number);
    let key = key_owned.as_str();
    let store = &source.dataset.store;
    let positioned = store.open_positioned(key)?;
    let declared_u64 = if let Some(positioned) = &positioned {
        positioned.len()
    } else {
        store.len(key)?
    };
    let declared = usize::try_from(declared_u64)
        .map_err(|_| Error::ResourceLimit(format!("value '{key}' exceeds usize")))?;
    if declared > source.dataset.limits.encoded_size() {
        return Err(Error::ResourceLimit(format!(
            "encoded chunk '{key}' has {declared} bytes"
        )));
    }
    let decode_limits = DecodeLimits::unlimited()
        .maximum_decoded_size(source.dataset.limits.decoded_size())
        .maximum_block_size(source.dataset.limits.decoded_size())
        .maximum_block_count(source.dataset.limits.block_count());

    let (read_source, decoder, io_bytes, io_ops, temporary_input_bytes, retained_bytes) =
        if let Some(positioned) = positioned {
            let probe_len = declared.min(PREFIX_PROBE_BYTES);
            let mut prefix = read_positioned_exact(
                positioned.file(),
                positioned.base_offset(),
                positioned.len(),
                0,
                probe_len,
            )?;
            let prefix_len = Decoder::index_prefix_len(&prefix)?;
            if prefix_len > declared {
                return Err(Error::InvalidDataset(format!(
                    "chunk '{key}' prefix has {prefix_len} bytes, file has {declared}"
                )));
            }
            let io_ops = if prefix_len > prefix.len() {
                let offset = u64::try_from(prefix.len())
                    .map_err(|_| Error::ResourceLimit("prefix offset exceeds u64".into()))?;
                let tail_len = prefix_len - prefix.len();
                append_positioned_exact(
                    positioned.file(),
                    positioned.base_offset(),
                    positioned.len(),
                    offset,
                    tail_len,
                    &mut prefix,
                )?;
                2
            } else {
                1
            };
            let io_bytes = u64::try_from(prefix.len())
                .map_err(|_| Error::ResourceLimit("prefix byte count exceeds u64".into()))?;
            let temporary_input_bytes = prefix.len().checked_mul(2).ok_or_else(|| {
                Error::ResourceLimit("chunk prefix temporary byte count overflow".into())
            })?;
            let decoder = Decoder::from_prefix_with_limits(&prefix[..prefix_len], decode_limits)?;
            (
                ReadSource::Positioned {
                    file: Arc::clone(positioned.file()),
                    base_offset: positioned.base_offset(),
                    view_len: positioned.len(),
                },
                decoder,
                io_bytes,
                io_ops,
                temporary_input_bytes,
                0,
            )
        } else if store.supports_efficient_range_reads(key)? {
            let probe_len = declared.min(PREFIX_PROBE_BYTES);
            let mut prefix = store.read_range(key, 0, probe_len)?;
            if prefix.len() != probe_len {
                return Err(Error::StalePlan(format!(
                    "range key '{key}' returned {} prefix bytes, expected {probe_len}",
                    prefix.len()
                )));
            }
            let prefix_len = Decoder::index_prefix_len(&prefix)?;
            if prefix_len > declared {
                return Err(Error::InvalidDataset(format!(
                    "chunk '{key}' prefix has {prefix_len} bytes, key has {declared}"
                )));
            }
            let io_ops = if prefix_len > prefix.len() {
                let tail_offset = u64::try_from(prefix.len())
                    .map_err(|_| Error::ResourceLimit("prefix offset exceeds u64".into()))?;
                let tail_len = prefix_len - prefix.len();
                let tail = store.read_range(key, tail_offset, tail_len)?;
                if tail.len() != tail_len {
                    return Err(Error::StalePlan(format!(
                        "range key '{key}' returned {} tail bytes, expected {tail_len}",
                        tail.len()
                    )));
                }
                prefix.try_reserve_exact(tail.len())?;
                prefix.extend_from_slice(&tail);
                2
            } else {
                1
            };
            let io_bytes = u64::try_from(prefix.len())
                .map_err(|_| Error::ResourceLimit("prefix byte count exceeds u64".into()))?;
            let temporary_input_bytes = prefix.len().checked_mul(2).ok_or_else(|| {
                Error::ResourceLimit("chunk prefix temporary byte count overflow".into())
            })?;
            let decoder = Decoder::from_prefix_with_limits(&prefix[..prefix_len], decode_limits)?;
            (
                ReadSource::RangeKey {
                    store: Arc::clone(store),
                    key: Arc::from(key),
                    declared_len: declared,
                },
                decoder,
                io_bytes,
                io_ops,
                temporary_input_bytes,
                0,
            )
        } else {
            if declared > maximum_whole_key_bytes {
                return Err(Error::ResourceLimit(format!(
                    "whole-key chunk '{key}' requires {declared} encoded bytes, limit is {maximum_whole_key_bytes}"
                )));
            }
            let encoded = store.read_range(key, 0, declared)?;
            if encoded.len() != declared {
                return Err(Error::StalePlan(format!(
                    "whole key '{key}' changed length during compilation"
                )));
            }
            let decoder = Decoder::from_prefix_with_limits(&encoded, decode_limits)?;
            let retained = reserve_retained_bytes(
                retained_whole_key_bytes,
                maximum_retained_whole_key_bytes,
                declared,
            );
            let cached = retained.then(|| Arc::<[u8]>::from(encoded));
            (
                ReadSource::WholeKey {
                    store: Arc::clone(store),
                    key: Arc::from(key),
                    declared_len: declared,
                    cached,
                },
                decoder,
                declared_u64,
                1,
                declared,
                usize::from(retained).saturating_mul(declared),
            )
        };
    if decoder.header().encoded_size() != declared {
        return Err(Error::InvalidDataset(format!(
            "chunk '{key}' header declares {} bytes, store reports {declared}",
            decoder.header().encoded_size()
        )));
    }
    let decoded_ends = if matches!(&source.dataset.kind, DatasetKind::Csr { .. }) {
        let mut ends = Vec::new();
        ends.try_reserve_exact(decoder.header().block_count())?;
        for block in decoder.blocks() {
            ends.push(block.decoded_range().end);
        }
        Some(ends)
    } else {
        None
    };
    let metadata_bytes = chunk_metadata_charge(
        decoder.header().block_count(),
        decoded_ends.as_ref().map_or(0, Vec::len),
        key.len(),
    )?;
    Ok(LoadedChunk {
        read_source,
        decoder,
        decoded_ends,
        io_bytes,
        io_ops,
        temporary_input_bytes,
        metadata_bytes,
        retained_bytes,
    })
}

fn reserve_retained_bytes(retained: &AtomicUsize, maximum: usize, bytes: usize) -> bool {
    bytes <= maximum
        && retained
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(bytes).filter(|next| *next <= maximum)
            })
            .is_ok()
}

fn element_range(start: u64, end: u64, size: usize, context: &str) -> Result<Range<usize>> {
    let start = usize::try_from(start)
        .ok()
        .and_then(|value| value.checked_mul(size))
        .ok_or_else(|| Error::InvalidDataset(format!("{context} start overflow")))?;
    let end = usize::try_from(end)
        .ok()
        .and_then(|value| value.checked_mul(size))
        .ok_or_else(|| Error::InvalidDataset(format!("{context} end overflow")))?;
    Ok(start..end)
}

fn set_required_chunk(markers: &mut [u64], index: usize) -> Result<()> {
    let word = markers
        .get_mut(index / u64::BITS as usize)
        .ok_or_else(|| Error::Invariant("required chunk marker is missing".into()))?;
    *word |= 1u64 << (index % u64::BITS as usize);
    Ok(())
}

fn required_chunk_is_set(markers: &[u64], index: usize) -> bool {
    markers
        .get(index / u64::BITS as usize)
        .is_some_and(|word| word & (1u64 << (index % u64::BITS as usize)) != 0)
}

#[cfg(feature = "profile")]
fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn locate_cell_block(
    plan: &ChunkPlan,
    chunk_key: ChunkKey,
    cell: Range<usize>,
) -> Result<CellBlock> {
    if cell.start >= cell.end {
        return Err(Error::InvalidDataset(
            "non-empty cell has an empty byte range".into(),
        ));
    }
    let block_index = plan
        .decoded_ends
        .as_ref()
        .ok_or_else(|| Error::Invariant("CSR chunk has no decoded block index".into()))?
        .partition_point(|&decoded_end| decoded_end <= cell.start);
    locate_cell_block_at(plan, chunk_key, cell, block_index)
}

fn locate_dense_cell_block(
    plan: &ChunkPlan,
    chunk_key: ChunkKey,
    cell: Range<usize>,
) -> Result<CellBlock> {
    if cell.start >= cell.end {
        return Err(Error::InvalidDataset(
            "non-empty cell has an empty byte range".into(),
        ));
    }
    let block_size = plan.decoder.layout().maximum_block_size();
    if block_size == 0 {
        return Err(Error::InvalidDataset(
            "non-empty dense chunk has zero block size".into(),
        ));
    }
    let block_index = cell.start / block_size;
    locate_cell_block_at(plan, chunk_key, cell, block_index)
}

fn locate_cell_block_at(
    plan: &ChunkPlan,
    chunk_key: ChunkKey,
    cell: Range<usize>,
    block_index: usize,
) -> Result<CellBlock> {
    let block = plan.decoder.block(block_index).ok_or_else(|| {
        Error::InvalidDataset(format!(
            "cell byte range [{}, {}) exceeds compressed block table",
            cell.start, cell.end
        ))
    })?;
    let decoded = block.decoded_range();
    if cell.start >= decoded.start && cell.end <= decoded.end {
        let local = (cell.start - decoded.start)..(cell.end - decoded.start);
        let encoded = block.encoded_range();
        let encoded_start = u64::try_from(encoded.start)
            .map_err(|_| Error::ResourceLimit("encoded block start exceeds u64".into()))?;
        let encoded_end = u64::try_from(encoded.end)
            .map_err(|_| Error::ResourceLimit("encoded block end exceeds u64".into()))?;
        return Ok(CellBlock {
            chunk_key,
            source: plan.source,
            block_index,
            encoded_range: encoded_start..encoded_end,
            decoded_len: block.decoded_len(),
            cell_range: local,
        });
    }
    Err(Error::InvalidDataset(format!(
        "cell byte range [{}, {}) crosses or exceeds compressed block boundaries",
        cell.start, cell.end
    )))
}

fn read_positioned_exact(
    file: &std::fs::File,
    base: u64,
    view_len: u64,
    offset: u64,
    len: usize,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    append_positioned_exact(file, base, view_len, offset, len, &mut output)?;
    Ok(output)
}

fn append_positioned_exact(
    file: &std::fs::File,
    base: u64,
    view_len: u64,
    offset: u64,
    len: usize,
    output: &mut Vec<u8>,
) -> Result<()> {
    let len_u64 = u64::try_from(len)
        .map_err(|_| Error::ResourceLimit("positioned read length exceeds u64".into()))?;
    let end = offset
        .checked_add(len_u64)
        .ok_or_else(|| Error::InvalidDataset("positioned range overflow".into()))?;
    if end > view_len {
        return Err(Error::InvalidDataset(format!(
            "positioned range [{offset}, {end}) exceeds view length {view_len}"
        )));
    }
    let absolute = base
        .checked_add(offset)
        .ok_or_else(|| Error::InvalidDataset("positioned absolute offset overflow".into()))?;
    let initial_len = output.len();
    output.try_reserve_exact(len)?;
    let mut filled = 0usize;
    while filled < len {
        let current = absolute
            .checked_add(filled as u64)
            .ok_or_else(|| Error::InvalidDataset("positioned read offset overflow".into()))?;
        let read = {
            let remaining = len - filled;
            let spare = &mut output.spare_capacity_mut()[..remaining];
            match rustix::io::pread(file, spare, current) {
                Ok((initialized, _)) => initialized.len(),
                Err(error) if error == rustix::io::Errno::INTR => continue,
                Err(error) => return Err(std::io::Error::from(error).into()),
            }
        };
        match read {
            0 => {
                return Err(Error::Io {
                    kind: std::io::ErrorKind::UnexpectedEof,
                    message: format!("short positioned read at offset {current}"),
                })
            }
            read => {
                filled += read;
                // SAFETY: rustix returned an initialized prefix of the exact
                // spare-capacity slice. Earlier iterations initialized the
                // preceding bytes, and no allocation occurs before this update.
                unsafe { output.set_len(initial_len + filled) };
            }
        }
    }
    Ok(())
}

fn relative_range(block: &Range<u64>, read: &Range<u64>) -> Result<Range<usize>> {
    if block.start < read.start || block.end > read.end {
        return Err(Error::Invariant(
            "block range is outside physical read".into(),
        ));
    }
    let start = usize::try_from(block.start - read.start)
        .map_err(|_| Error::ResourceLimit("relative block start exceeds usize".into()))?;
    let end = usize::try_from(block.end - read.start)
        .map_err(|_| Error::ResourceLimit("relative block end exceeds usize".into()))?;
    Ok(start..end)
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if value == 0 {
        return Ok(0);
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| Error::ResourceLimit("aligned size overflow".into()))
}

fn range_len_u64(range: &Range<u64>) -> Result<usize> {
    let len = range
        .end
        .checked_sub(range.start)
        .ok_or_else(|| Error::Invariant("range is reversed".into()))?;
    usize::try_from(len).map_err(|_| Error::ResourceLimit("range length exceeds usize".into()))
}

fn checked_payload_bytes(parts: &[(usize, usize)]) -> Result<usize> {
    parts.iter().try_fold(0usize, |total, &(count, width)| {
        count
            .checked_mul(width)
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| Error::ResourceLimit("compile payload byte count overflow".into()))
    })
}

fn append_batch_completions(
    arena: &mut Vec<BatchCompletion>,
    ring_batch: usize,
    mut completed: usize,
) -> Result<()> {
    while completed != 0 {
        let part = completed.min(u32::MAX as usize);
        arena.push(BatchCompletion::new(ring_batch, part).ok_or_else(|| {
            Error::ResourceLimit("batch completion exceeds compact representation".into())
        })?);
        completed -= part;
    }
    Ok(())
}

fn chunk_metadata_charge(
    decoder_blocks: usize,
    decoded_end_count: usize,
    key_len: usize,
) -> Result<usize> {
    // DynBlosc retains an encoded entry and decoded start per block. CSR adds
    // a decoded-end search index; dense chunks derive block IDs arithmetically.
    let decoder_block_bytes = size_of::<u64>()
        .checked_add(size_of::<usize>())
        .ok_or_else(|| Error::ResourceLimit("chunk metadata width overflow".into()))?;
    checked_payload_bytes(&[
        (decoder_blocks, decoder_block_bytes),
        (decoded_end_count, size_of::<usize>()),
        (
            1,
            size_of::<Decoder>()
                + size_of::<ChunkPlan>()
                + size_of::<ReadSource>()
                + size_of::<usize>() * 2,
        ),
        (1, key_len),
    ])
}

impl TempRun {
    fn run_key(&self) -> RunKey {
        RunKey {
            did: self.did,
            data_source: self.data_source,
            indices_source: self.indices_source,
        }
    }

    fn cost(&self) -> Result<Cost> {
        ranges_cost(&self.data_range, &self.indices_range)
    }
}

impl MergeProjection {
    fn cost(&self) -> Result<Cost> {
        ranges_cost(&self.data_range, &self.indices_range)
    }

    fn into_run(self, member_root: usize) -> TempRun {
        TempRun {
            member_root,
            block_count: self.block_count,
            anchor: self.anchor,
            batch_min: self.batch_min,
            batch_max: self.batch_max,
            active: true,
            did: self.did,
            data_source: self.data_source,
            indices_source: self.indices_source,
            data_range: self.data_range,
            indices_range: self.indices_range,
            data_decoded: self.data_decoded,
            indices_decoded: self.indices_decoded,
            cell_count: self.cell_count,
        }
    }
}

fn append_run_members(
    root: usize,
    nodes: &[MemberNode],
    stack: &mut Vec<usize>,
    output: &mut Vec<usize>,
) -> Result<()> {
    stack.clear();
    stack.push(root);
    while let Some(node_id) = stack.pop() {
        match *nodes
            .get(node_id)
            .ok_or_else(|| Error::Invariant("run member node is out of range".into()))?
        {
            MemberNode::Block(block_id) => output.push(block_id),
            MemberNode::Concat { left, right } => {
                stack.try_reserve(2)?;
                stack.push(right);
                stack.push(left);
            }
        }
    }
    Ok(())
}

fn ranges_cost(data: &Option<Range<u64>>, indices: &Option<Range<u64>>) -> Result<Cost> {
    let mut cost = Cost { bytes: 0, ops: 0 };
    for range in [data, indices].into_iter().flatten() {
        if !range.is_empty() {
            let bytes = range
                .end
                .checked_sub(range.start)
                .ok_or_else(|| Error::Invariant("physical range is reversed".into()))?;
            cost.bytes = cost
                .bytes
                .checked_add(bytes)
                .ok_or_else(|| Error::ResourceLimit("run byte cost overflow".into()))?;
            cost.ops += 1;
        }
    }
    Ok(cost)
}

fn union_optional_range(left: Option<Range<u64>>, right: Option<Range<u64>>) -> Option<Range<u64>> {
    match (left, right) {
        (None, None) => None,
        (Some(a), None) | (None, Some(a)) => Some(a),
        (Some(a), Some(b)) => {
            // Whole-key ranges are identical; positioned ranges expand by min/max.
            Some(a.start.min(b.start)..a.end.max(b.end))
        }
    }
}

#[cfg(test)]
mod required_chunk_marker_tests {
    use super::{required_chunk_is_set, set_required_chunk};

    #[test]
    fn markers_cover_word_boundaries() {
        let mut markers = [0u64; 3];
        for index in [0, 63, 64, 129] {
            set_required_chunk(&mut markers, index).unwrap();
        }
        for index in [0, 63, 64, 129] {
            assert!(required_chunk_is_set(&markers, index));
        }
        for index in [1, 62, 65, 128, 191] {
            assert!(!required_chunk_is_set(&markers, index));
        }
        assert!(set_required_chunk(&mut markers, 192).is_err());
        assert!(!required_chunk_is_set(&markers, 192));
    }
}
