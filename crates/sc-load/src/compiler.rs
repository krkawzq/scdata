use std::collections::{hash_map::Entry, BTreeMap, BTreeSet, HashMap};
use std::mem::size_of;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
#[cfg(feature = "profile")]
use std::time::Instant;

use dyn_blosc::{BlockDecoder, DecodeLimits, Decoder};
use sc_compress::chunk_key;

use crate::config::{IoMergePolicy, PlanConfig};
use crate::convert::ConvertOp;
use crate::plan::{
    CacheSlice, CellTask, CsrMap, CsrScatterTask, DecodeOp, DenseMap, DenseMapEntry, DenseMapRun,
    DenseScatterTask, DependencyGraph, InitializeJob, IoDecodeLoadTask, OutputRange, OutputSlice,
    Plan, PlanData, PlanStats, ReadSource, ReleasePlan, SourcePlan, StaticJob, StaticPlanData,
    UNMAPPED_TARGET, UNMAPPED_TARGET_U32,
};
use crate::scatter::{FillOp, IndexOp};
use crate::source::{DatasetKind, OutputSlot};
use crate::{Error, OutputSpec, Result, RowRef, Source, SourceId};

const MAX_DENSE_CHUNK_TABLE_BYTES: usize = 64 * 1024 * 1024;
const MAX_NESTED_BUCKET_HEADER_BYTES: usize = 4 * 1024 * 1024;
const MIN_DENSE_OVERWRITE_BYTES_PER_GAP: usize = 256;

fn use_nested_buckets(count: usize) -> bool {
    count
        .checked_mul(size_of::<Vec<usize>>())
        .is_some_and(|bytes| bytes <= MAX_NESTED_BUCKET_HEADER_BYTES)
}

fn try_filled_vec<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len)?;
    values.resize(len, value);
    Ok(values)
}

#[derive(Clone)]
pub struct PlanSpec {
    pub sources: Vec<Source>,
    pub rows: Vec<RowRef>,
    pub output: OutputSpec,
    pub batch_size: usize,
    /// Number of output-ring generations resident at once.
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

#[derive(Clone)]
struct StaticCacheObject {
    side: Side,
    block: BlockInfo,
}

struct StaticResidency {
    object: usize,
    cache: CacheSlice,
    allocation_len: usize,
    earliest_consumer_batch: usize,
    available_after_batch: Option<usize>,
    compile_refcount: usize,
}

#[derive(Debug, Clone, Copy)]
struct FreeExtent {
    offset: usize,
    len: usize,
    available_after_batch: Option<usize>,
}

struct ExtentAllocator {
    by_address: BTreeMap<usize, FreeExtent>,
    by_size: BTreeSet<(usize, usize)>,
    free_bytes: usize,
    slack: usize,
}

impl ExtentAllocator {
    fn new(capacity: usize, slack: usize) -> Self {
        let extent = FreeExtent {
            offset: 0,
            len: capacity,
            available_after_batch: None,
        };
        Self {
            by_address: BTreeMap::from([(0, extent)]),
            by_size: BTreeSet::from([(capacity, 0)]),
            free_bytes: capacity,
            slack,
        }
    }

    fn total_free(&self) -> usize {
        self.free_bytes
    }

    fn largest_free(&self) -> usize {
        self.by_size.last().map_or(0, |(len, _)| *len)
    }

    fn allocate(&mut self, len: usize) -> Option<FreeExtent> {
        let best_waste = self
            .by_size
            .range((len, 0)..)
            .next()
            .map(|(extent_len, _)| extent_len - len)?;
        let allowed = best_waste.saturating_add(self.slack.max(len / 16));
        let selected = self
            .by_size
            .range((len, 0)..)
            .take_while(|(extent_len, _)| extent_len - len <= allowed)
            .filter_map(|(extent_len, offset)| {
                self.by_address.get(offset).map(|extent| {
                    (
                        epoch_sort_key(extent.available_after_batch),
                        extent_len - len,
                        *offset,
                    )
                })
            })
            .min()?;
        let offset = selected.2;
        let extent = self.remove(offset)?;
        if extent.len > len {
            self.insert(FreeExtent {
                offset: extent.offset + len,
                len: extent.len - len,
                available_after_batch: extent.available_after_batch,
            });
        }
        // The selected extent is at least `len`; splitting only returns its
        // unused suffix, so exactly `len` bytes leave the free set.
        self.free_bytes = self
            .free_bytes
            .checked_sub(len)
            .expect("selected extent remains part of the free-byte total");
        Some(FreeExtent { len, ..extent })
    }

    fn release(&mut self, offset: usize, len: usize, available_after_batch: usize) {
        // Every residency is released exactly once after a successful
        // allocation, so the total cannot exceed the fixed cache capacity.
        self.free_bytes = self
            .free_bytes
            .checked_add(len)
            .expect("released cache extent keeps the fixed capacity representable");
        let mut released = FreeExtent {
            offset,
            len,
            available_after_batch: Some(available_after_batch),
        };
        if let Some((&left_offset, left)) = self.by_address.range(..offset).next_back() {
            if left.offset + left.len == offset {
                let left = self
                    .remove(left_offset)
                    .expect("left extent remains indexed");
                released.offset = left.offset;
                released.len += left.len;
                released.available_after_batch =
                    max_epoch(released.available_after_batch, left.available_after_batch);
            }
        }
        if let Some((&right_offset, right)) = self.by_address.range(released.offset..).next() {
            if released.offset + released.len == right.offset {
                let right = self
                    .remove(right_offset)
                    .expect("right extent remains indexed");
                released.len += right.len;
                released.available_after_batch =
                    max_epoch(released.available_after_batch, right.available_after_batch);
            }
        }
        self.insert(released);
    }

    fn insert(&mut self, extent: FreeExtent) {
        debug_assert!(extent.len > 0);
        let previous = self.by_address.insert(extent.offset, extent);
        debug_assert!(previous.is_none());
        let inserted = self.by_size.insert((extent.len, extent.offset));
        debug_assert!(inserted);
    }

    fn remove(&mut self, offset: usize) -> Option<FreeExtent> {
        let extent = self.by_address.remove(&offset)?;
        let removed = self.by_size.remove(&(extent.len, extent.offset));
        debug_assert!(removed);
        Some(extent)
    }
}

fn epoch_sort_key(epoch: Option<usize>) -> (bool, usize) {
    match epoch {
        None => (false, 0),
        Some(batch) => (true, batch),
    }
}

fn max_epoch(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (None, None) => None,
        (Some(value), None) | (None, Some(value)) => Some(value),
        (Some(left), Some(right)) => Some(left.max(right)),
    }
}

fn cost_aware_group_is_feasible(
    ordered: &[usize],
    residencies: &[StaticResidency],
    objects: &[StaticCacheObject],
    merge: &crate::config::IoMergeOptions,
) -> Result<bool> {
    let Some((&first_id, rest)) = ordered.split_first() else {
        return Ok(false);
    };
    let last_id = rest.last().copied().unwrap_or(first_id);
    let first = &objects[residencies[first_id].object].block;
    let last = &objects[residencies[last_id].object].block;
    let span = last
        .encoded_range
        .end
        .checked_sub(first.encoded_range.start)
        .and_then(|span| usize::try_from(span).ok())
        .ok_or_else(|| Error::ResourceLimit("I/O fusion span exceeds usize".into()))?;
    let (payload, decoded) =
        ordered
            .iter()
            .try_fold((0usize, 0usize), |(payload, decoded), residency| {
                let object = &objects[residencies[*residency].object];
                Ok::<_, Error>((
                    payload
                        .checked_add(range_len_u64(&object.block.encoded_range)?)
                        .ok_or_else(|| {
                            Error::ResourceLimit("I/O fusion payload overflow".into())
                        })?,
                    decoded
                        .checked_add(object.block.decoded_len())
                        .ok_or_else(|| {
                            Error::ResourceLimit("I/O fusion decoded bytes overflow".into())
                        })?,
                ))
            })?;
    let gap = span
        .checked_sub(payload)
        .ok_or_else(|| Error::Invariant("I/O fusion payload exceeds physical span".into()))?;
    let amplification = if payload == 0 {
        1.0
    } else {
        span as f64 / payload as f64
    };
    Ok((ordered.len() == 1 || span <= merge.max_coalesced_io_bytes)
        && span <= merge.max_encoded_staging_bytes_per_task
        && gap <= merge.max_io_gap_bytes
        && amplification <= merge.max_io_amplification_ratio
        && decoded <= merge.max_decoded_bytes_per_io_task
        && ordered.len() <= merge.max_decode_ops_per_io_task)
}

fn cost_aware_group_ends(
    ordered: &[usize],
    residencies: &[StaticResidency],
    objects: &[StaticCacheObject],
    merge: &crate::config::IoMergeOptions,
) -> Result<Vec<usize>> {
    let count = ordered.len();
    let state_count = count
        .checked_add(1)
        .ok_or_else(|| Error::ResourceLimit("I/O fusion state count overflow".into()))?;
    let mut payload_prefix = try_filled_vec(state_count, 0usize)?;
    let mut decoded_prefix = try_filled_vec(state_count, 0usize)?;
    for (index, &residency_id) in ordered.iter().enumerate() {
        let object = &objects[residencies[residency_id].object];
        payload_prefix[index + 1] = payload_prefix[index]
            .checked_add(range_len_u64(&object.block.encoded_range)?)
            .ok_or_else(|| Error::ResourceLimit("I/O fusion payload overflow".into()))?;
        decoded_prefix[index + 1] = decoded_prefix[index]
            .checked_add(object.block.decoded_len())
            .ok_or_else(|| Error::ResourceLimit("I/O fusion decoded bytes overflow".into()))?;
    }
    let mut costs = try_filled_vec(state_count, f64::INFINITY)?;
    let mut gaps = try_filled_vec(state_count, usize::MAX)?;
    let mut tasks = try_filled_vec(state_count, 0usize)?;
    let mut previous = try_filled_vec(state_count, usize::MAX)?;
    costs[0] = 0.0;
    gaps[0] = 0;
    for end in 1..=count {
        let earliest = end.saturating_sub(merge.max_decode_ops_per_io_task);
        for start in (earliest..end).rev() {
            let first = &objects[residencies[ordered[start]].object].block;
            let last = &objects[residencies[ordered[end - 1]].object].block;
            let span = last
                .encoded_range
                .end
                .checked_sub(first.encoded_range.start)
                .and_then(|span| usize::try_from(span).ok())
                .ok_or_else(|| Error::ResourceLimit("I/O fusion span exceeds usize".into()))?;
            let payload = payload_prefix[end] - payload_prefix[start];
            let decoded = decoded_prefix[end] - decoded_prefix[start];
            let gap = span.checked_sub(payload).ok_or_else(|| {
                Error::Invariant("I/O fusion payload exceeds physical span".into())
            })?;
            let amplification = if payload == 0 {
                1.0
            } else {
                span as f64 / payload as f64
            };
            let group_count = end - start;
            if (group_count > 1 && span > merge.max_coalesced_io_bytes)
                || span > merge.max_encoded_staging_bytes_per_task
                || gap > merge.max_io_gap_bytes
                || amplification > merge.max_io_amplification_ratio
                || decoded > merge.max_decoded_bytes_per_io_task
            {
                continue;
            }
            let uncertainty = usize::from(group_count > 1) * merge.io_merge_delta_bytes;
            let Some(charged_span) = span.checked_add(uncertainty) else {
                continue;
            };
            let group_cost = 1.0 / merge.io_operations_per_second
                + charged_span as f64 / merge.io_bandwidth_bytes_per_second;
            let candidate_cost = costs[start] + group_cost;
            let Some(candidate_gap) = gaps[start].checked_add(gap) else {
                continue;
            };
            let candidate_tasks = tasks[start]
                .checked_add(1)
                .ok_or_else(|| Error::ResourceLimit("I/O fusion task count overflow".into()))?;
            let better = if !costs[end].is_finite() {
                true
            } else {
                let tolerance = f64::EPSILON * candidate_cost.abs().max(costs[end].abs()).max(1.0);
                candidate_cost + tolerance < costs[end]
                    || ((candidate_cost - costs[end]).abs() <= tolerance
                        && (candidate_gap < gaps[end]
                            || (candidate_gap == gaps[end]
                                && (candidate_tasks > tasks[end]
                                    || (candidate_tasks == tasks[end] && start < previous[end])))))
            };
            if better {
                costs[end] = candidate_cost;
                gaps[end] = candidate_gap;
                tasks[end] = candidate_tasks;
                previous[end] = start;
            }
        }
        if previous[end] == usize::MAX {
            return Err(Error::ResourceLimit(format!(
                "no feasible I/O fusion partition ends at independent load {end}"
            )));
        }
    }
    let mut cursor = count;
    let mut group_ends = Vec::new();
    group_ends.try_reserve_exact(tasks[count])?;
    while cursor > 0 {
        group_ends.push(cursor);
        cursor = previous[cursor];
    }
    group_ends.reverse();
    Ok(group_ends)
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

enum CandidateRemap {
    Dense {
        values: Vec<usize>,
        maximum_len: usize,
    },
    Sparse(HashMap<usize, usize>),
}

impl CandidateRemap {
    fn new(raw_candidates: usize, cell_count: usize) -> Result<Self> {
        let maximum_len = cell_count.saturating_mul(2).max(1_024);
        if raw_candidates <= maximum_len {
            Ok(Self::Dense {
                values: try_filled_vec(raw_candidates, usize::MAX)?,
                maximum_len,
            })
        } else {
            let mut values = HashMap::new();
            values.try_reserve(cell_count.min(raw_candidates))?;
            Ok(Self::Sparse(values))
        }
    }

    fn intern(&mut self, raw: usize, next: usize) -> Result<(usize, bool)> {
        match self {
            Self::Dense {
                values,
                maximum_len,
            } if raw < *maximum_len => {
                if raw >= values.len() {
                    let new_len = raw.checked_add(1).ok_or_else(|| {
                        Error::ResourceLimit("block candidate remap length overflow".into())
                    })?;
                    values.try_reserve(new_len - values.len())?;
                    values.resize(new_len, usize::MAX);
                }
                // SAFETY: the growth branch above established `raw < values.len()`.
                let slot = unsafe { values.get_unchecked_mut(raw) };
                if *slot == usize::MAX {
                    *slot = next;
                    Ok((next, true))
                } else {
                    Ok((*slot, false))
                }
            }
            Self::Dense { values, .. } => {
                let capacity = next.checked_add(1).ok_or_else(|| {
                    Error::ResourceLimit("block candidate remap count overflow".into())
                })?;
                let mut sparse = HashMap::new();
                sparse.try_reserve(capacity)?;
                for (raw, &mapped) in values.iter().enumerate() {
                    if mapped != usize::MAX {
                        sparse.insert(raw, mapped);
                    }
                }
                sparse.insert(raw, next);
                *self = Self::Sparse(sparse);
                Ok((next, true))
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

impl Builder {
    fn new(mut request: PlanSpec) -> Result<Self> {
        if request.batch_size == 0 {
            return Err(Error::InvalidConfig("batch_size must be positive".into()));
        }
        if request.prefetch_step == 0 {
            return Err(Error::InvalidConfig(
                "prefetch_step must be positive".into(),
            ));
        }
        request.config.validate()?;
        request.output.validate()?;
        let maximum_job_cells = request.rows.len().min(request.batch_size);
        if maximum_job_cells > request.config.limits.max_cells_per_job {
            return Err(Error::ResourceLimit(format!(
                "one batch has up to {maximum_job_cells} cells, limit is {}",
                request.config.limits.max_cells_per_job
            )));
        }

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
            let mut default_ranges = build_default_ranges(
                feature_targets.as_deref(),
                request.output.n_cols,
                request.output.dtype.size(),
            )?;
            let dense_fill_whole = if index_dtype.is_none() {
                choose_dense_whole_fill(
                    feature_targets.as_deref(),
                    request.output.dtype.size(),
                    default_ranges.len(),
                )?
            } else {
                false
            };
            if dense_fill_whole {
                default_ranges = Arc::from([]);
            }
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
                dense_fill_whole,
                default_ranges,
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
        let (cell_blocks, cells) = self.resolve_and_merge_same_blocks(resolved_rows, row_stride)?;
        // Resolution is the last phase that needs input rows, source lookup,
        // or dataset metadata. The static cache compiler consumes only compact
        // block identities, row ranges, and frozen decoder metadata.
        drop(std::mem::take(&mut self.request.rows));
        drop(std::mem::take(&mut self.registered_sources));
        drop(std::mem::take(&mut self.source_indices));
        drop(std::mem::take(&mut self.row_layouts));
        #[cfg(feature = "profile")]
        let compile_resolve_ns = elapsed_ns(phase_started);
        self.check_compile_payload(
            "static cache graph",
            checked_payload_bytes(&[
                (cells.len(), size_of::<CompactCell>()),
                (cell_blocks.len(), size_of::<usize>()),
                (self.block_candidates.len(), size_of::<BlockCandidate>()),
                (1, self.chunk_metadata_bytes),
                (1, self.retained_whole_key_bytes),
            ])?,
        )?;
        #[cfg(feature = "profile")]
        let phase_started = Instant::now();
        let mut stats = PlanStats {
            output_ring_bytes,
            compile_working_set_bytes: self.peak_compile_payload_bytes,
            retained_whole_key_bytes: self.retained_whole_key_bytes,
            ..PlanStats::default()
        };
        let static_plan =
            self.compile_static_plan(&cells, &cell_blocks, row_stride, row_bytes, &mut stats)?;
        stats.compile_working_set_bytes = self.peak_compile_payload_bytes;
        #[cfg(feature = "profile")]
        let compile_finalize_ns = elapsed_ns(phase_started);
        stats.compile_time_io_bytes = self.compile_io_bytes;
        stats.compile_time_io_ops = self.compile_io_ops;
        #[cfg(feature = "profile")]
        {
            stats.compile_resolve_ns = compile_resolve_ns;
            stats.compile_finalize_ns = compile_finalize_ns;
        }
        let io_ops = stats
            .data_io_ops
            .checked_add(stats.indices_io_ops)
            .ok_or_else(|| Error::ResourceLimit("predicted I/O operation count overflow".into()))?;
        stats.predicted_io_seconds = if io_ops == 0 {
            0.0
        } else {
            stats.predicted_physical_bytes as f64
                / self.request.config.io_merge.io_bandwidth_bytes_per_second
                + io_ops as f64 / self.request.config.io_merge.io_operations_per_second
        };

        let plan = PlanData {
            batch_size: self.request.batch_size,
            batch_count,
            ring_slots: self.ring_slots,
            ring_mask: self.ring_mask,
            fill: FillOp::new(
                &self.request.output.fill.encode()[..self.request.output.dtype.size()],
            ),
            row_bytes,
            output: self.request.output,
            row_stride,
            sources: self.sources,
            source_plans: self.source_plans,
            stats,
            static_plan,
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
            (
                maximum_required_chunks,
                size_of::<OnceLock<Result<LoadedChunk>>>(),
            ),
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
            (requests.len(), size_of::<OnceLock<Result<LoadedChunk>>>()),
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
        let mut outcomes = Vec::new();
        outcomes.try_reserve_exact(requests.len())?;
        outcomes.resize_with(requests.len(), OnceLock::new);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            handles.try_reserve_exact(concurrency)?;
            for _ in 0..concurrency {
                handles.push(std::thread::Builder::new().spawn_scoped(scope, || {
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
                        let slot = outcomes.get(index).ok_or_else(|| {
                            Error::Invariant("chunk metadata result slot is missing".into())
                        })?;
                        if slot.set(outcome).is_err() {
                            return Err(Error::Invariant(
                                "chunk metadata result was installed twice".into(),
                            ));
                        }
                    }
                    Ok::<_, Error>(())
                })?);
            }
            for handle in handles {
                handle.join().map_err(|_| {
                    Error::Invariant("chunk metadata loader thread panicked".into())
                })??;
            }
            Ok::<_, Error>(())
        })?;
        for (index, outcome) in outcomes.into_iter().enumerate() {
            let outcome = outcome.into_inner().ok_or_else(|| {
                Error::Invariant("chunk metadata result was not initialized".into())
            })?;
            self.install_loaded_chunk(requests[index], outcome?)?;
        }
        Ok(())
    }

    fn resolve_cell(
        &mut self,
        ordinal: usize,
        index: RowRef,
        resolved: ResolvedRow,
        row_stride: usize,
    ) -> Result<CellInfo> {
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
        let output_slot = OutputSlot::new(row_offset)
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
                // CSR data and indices chunks are encoded with the same block
                // element partition. Reuse the already resolved data block and
                // let the exact range check diagnose inconsistent metadata.
                let (indices, _) = self.locate_chunk_cell_at(
                    indices_key,
                    indices_range,
                    data.block_index,
                    index.source,
                )?;
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

    fn locate_chunk_cell_at(
        &mut self,
        key: ChunkKey,
        cell: Range<usize>,
        block_index: usize,
        source: SourceId,
    ) -> Result<(CellBlock, usize)> {
        if self.chunk(key).is_none() {
            self.load_missing_chunk(key)?;
        }
        let plan = self
            .chunk(key)
            .ok_or_else(|| Error::Invariant("inserted mirrored chunk is missing".into()))?;
        let block = locate_cell_block_at(plan, key, cell, block_index).map_err(|error| {
            Error::InvalidDataset(format!(
                "CSR mirrored block mismatch in dataset {} chunk {} at block {block_index}: {error}",
                source.get(),
                key.chunk
            ))
        })?;
        Ok((block, plan.candidate_base))
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
    ) -> Result<(Vec<usize>, Vec<CompactCell>)> {
        let cell_count = self.request.rows.len();
        let mut remap = CandidateRemap::new(self.next_block_candidate, cell_count)?;
        let candidate_hint = self.next_block_candidate.min(cell_count);
        self.block_candidates.try_reserve_exact(candidate_hint)?;
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
            // SAFETY: `ordinal` is produced by the exact input-row arena bound.
            let index = unsafe { *self.request.rows.get_unchecked(ordinal) };
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
            let resolved = self.resolve_cell(ordinal, index, resolved_row, row_stride)?;
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
            }
            cells.push(CompactCell::from_resolved(&resolved));
            cell_blocks.push(block_key);
        }
        if let RowCursor::General(mut rows) = resolved_rows {
            if rows.next().is_some() {
                return Err(Error::Invariant(
                    "resolved row arena is longer than the request".into(),
                ));
            }
        }
        self.next_block_candidate = self.block_candidates.len();
        Ok((cell_blocks, cells))
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

    fn compile_static_plan(
        &mut self,
        cells: &[CompactCell],
        cell_blocks: &[usize],
        row_stride: usize,
        row_bytes: usize,
        stats: &mut PlanStats,
    ) -> Result<StaticPlanData> {
        let batch_count = cells.len().div_ceil(self.request.batch_size);
        let mut objects = Vec::<StaticCacheObject>::new();
        let object_capacity = self
            .block_candidates
            .len()
            .checked_mul(2)
            .ok_or_else(|| Error::ResourceLimit("cache object count overflow".into()))?;
        objects.try_reserve_exact(object_capacity)?;
        let mut data_objects = try_filled_vec(self.block_candidates.len(), usize::MAX)?;
        let mut indices_objects = try_filled_vec(self.block_candidates.len(), usize::MAX)?;
        for (candidate_id, candidate) in self.block_candidates.iter().enumerate() {
            if let Some(block) = &candidate.data {
                data_objects[candidate_id] = objects.len();
                objects.push(StaticCacheObject {
                    side: Side::Data,
                    block: block.clone(),
                });
            }
            if let Some(block) = &candidate.indices {
                indices_objects[candidate_id] = objects.len();
                objects.push(StaticCacheObject {
                    side: Side::Indices,
                    block: block.clone(),
                });
            }
        }

        let mut requirements = Vec::<usize>::new();
        let mut requirement_ranges = Vec::<Range<usize>>::new();
        requirement_ranges.try_reserve_exact(batch_count)?;
        let mut requirement_scratch = Vec::<usize>::new();
        let maximum_batch_references = self
            .request
            .batch_size
            .min(cell_blocks.len())
            .checked_mul(2)
            .ok_or_else(|| Error::ResourceLimit("batch reference count overflow".into()))?;
        let maximum_unique_requirements = maximum_batch_references.min(objects.len()).min(
            self.request
                .config
                .limits
                .max_blocks_per_job
                .saturating_add(1),
        );
        requirement_scratch.try_reserve_exact(maximum_unique_requirements)?;
        let mut requirement_seen = try_filled_vec(objects.len(), 0u8)?;
        let mut maximum_batch_decoded_bytes = 0usize;
        for batch in 0..batch_count {
            requirement_scratch.clear();
            let row_start = batch * self.request.batch_size;
            let row_end = (row_start + self.request.batch_size).min(cell_blocks.len());
            for &candidate_id in &cell_blocks[row_start..row_end] {
                // SAFETY: `cell_blocks` contains only IDs interned into the
                // complete block-candidate arena used to size both maps.
                let candidate_objects = unsafe {
                    [
                        *data_objects.get_unchecked(candidate_id),
                        *indices_objects.get_unchecked(candidate_id),
                    ]
                };
                for object in candidate_objects {
                    if object == usize::MAX {
                        continue;
                    }
                    // SAFETY: non-sentinel object IDs were assigned while
                    // appending to `objects`, which sized this marker arena.
                    let seen = unsafe { requirement_seen.get_unchecked_mut(object) };
                    if *seen == 0 {
                        *seen = 1;
                        requirement_scratch.push(object);
                        if requirement_scratch.len() > self.request.config.limits.max_blocks_per_job
                        {
                            return Err(Error::ResourceLimit(format!(
                                "batch cache working set has more than {} objects",
                                self.request.config.limits.max_blocks_per_job
                            )));
                        }
                    }
                }
            }
            let decoded_bytes = requirement_scratch
                .iter()
                .try_fold(0usize, |total, &object| {
                    // SAFETY: scratch entries come exclusively from the
                    // validated object IDs inserted in the loop above.
                    unsafe { *requirement_seen.get_unchecked_mut(object) = 0 };
                    total
                        // SAFETY: the same object-domain proof covers this read.
                        .checked_add(unsafe { objects.get_unchecked(object) }.block.decoded_len())
                        .ok_or_else(|| {
                            Error::ResourceLimit("batch decoded cache requirement overflow".into())
                        })
                })?;
            if decoded_bytes > self.request.config.limits.max_decoded_bytes_per_job {
                return Err(Error::ResourceLimit(format!(
                    "batch {batch} requires {decoded_bytes} decoded bytes, limit is {}",
                    self.request.config.limits.max_decoded_bytes_per_job
                )));
            }
            maximum_batch_decoded_bytes = maximum_batch_decoded_bytes.max(decoded_bytes);
            let start = requirements.len();
            requirements.try_reserve(requirement_scratch.len())?;
            requirements.extend_from_slice(&requirement_scratch);
            requirement_ranges.push(start..requirements.len());
        }

        let maximum_residencies = requirements.len();
        let maximum_executable = maximum_residencies
            .checked_add(cells.len())
            .ok_or_else(|| Error::ResourceLimit("static task count overflow".into()))?;
        let maximum_cell_dependencies = cells
            .len()
            .checked_mul(2)
            .ok_or_else(|| Error::ResourceLimit("static dependency count overflow".into()))?;
        let cost_aware_scratch_width =
            if self.request.config.io_merge.policy == IoMergePolicy::CostAware {
                size_of::<usize>() * 9 + size_of::<f64>() + size_of::<bool>()
            } else {
                0
            };
        let successor_scratch_width = if use_nested_buckets(maximum_residencies) {
            size_of::<Vec<usize>>()
        } else {
            size_of::<usize>() * 2
        };
        let release_scratch_width = if use_nested_buckets(batch_count) {
            (size_of::<Vec<usize>>() + size_of::<Range<usize>>()) * 2
        } else {
            (size_of::<Range<usize>>() + size_of::<usize>() * 2) * 2
        };
        self.check_compile_payload(
            "static cache compiler",
            checked_payload_bytes(&[
                (cells.len(), size_of::<CompactCell>()),
                (cell_blocks.len(), size_of::<usize>()),
                (self.block_candidates.len(), size_of::<BlockCandidate>()),
                (objects.len(), size_of::<StaticCacheObject>()),
                (data_objects.len(), size_of::<usize>()),
                (indices_objects.len(), size_of::<usize>()),
                (requirements.len(), size_of::<usize>()),
                (requirement_ranges.len(), size_of::<Range<usize>>()),
                (requirement_scratch.capacity(), size_of::<usize>()),
                (requirement_seen.len(), size_of::<u8>()),
                (objects.len(), size_of::<Option<usize>>()),
                (objects.len(), size_of::<u64>()),
                (maximum_residencies, size_of::<StaticResidency>()),
                (requirements.len(), size_of::<usize>()),
                (maximum_residencies, size_of::<usize>()),
                (maximum_residencies, size_of::<IoDecodeLoadTask>()),
                (maximum_residencies, size_of::<DecodeOp>()),
                (maximum_residencies, size_of::<usize>()),
                (maximum_residencies, cost_aware_scratch_width),
                (cells.len(), size_of::<CsrScatterTask>()),
                (cells.len(), size_of::<[usize; 2]>()),
                (batch_count, size_of::<StaticJob>()),
                (maximum_executable, size_of::<u32>()),
                (maximum_residencies, successor_scratch_width),
                (maximum_cell_dependencies, size_of::<usize>()),
                (maximum_executable, size_of::<usize>()),
                (batch_count, release_scratch_width),
                (maximum_residencies, size_of::<FreeExtent>() * 3),
                (1, self.chunk_metadata_bytes),
                (1, self.retained_whole_key_bytes),
            ])?,
        )?;

        let capacity = self.request.config.cache_capacity_bytes;
        let alignment = self.request.config.cache_alignment;
        let mut allocator = ExtentAllocator::new(
            capacity,
            self.request.config.cache_fragmentation_slack_bytes,
        );
        let mut resident = try_filled_vec(objects.len(), None::<usize>)?;
        let mut generations = try_filled_vec(objects.len(), 0u64)?;
        let mut residencies = Vec::<StaticResidency>::new();
        let mut bindings = try_filled_vec(requirements.len(), usize::MAX)?;
        let mut next_batch = 0usize;
        let mut cache_hits = 0usize;
        let mut cache_misses = 0usize;
        let mut capacity_stalls = 0usize;
        let mut fragmentation_stalls = 0usize;
        let mut alignment_loss = 0usize;
        let mut horizon_max = 0usize;
        let mut missing = Vec::<(usize, usize)>::new();

        for current_batch in 0..batch_count {
            loop {
                if next_batch >= batch_count {
                    break;
                }
                let requirement_range = requirement_ranges[next_batch].clone();
                for position in requirement_range.clone() {
                    if bindings[position] != usize::MAX {
                        continue;
                    }
                    let object = requirements[position];
                    if let Some(residency) = resident[object] {
                        residencies[residency].compile_refcount = residencies[residency]
                            .compile_refcount
                            .checked_add(1)
                            .ok_or_else(|| {
                                Error::ResourceLimit("compile cache refcount overflow".into())
                            })?;
                        bindings[position] = residency;
                        cache_hits = cache_hits.saturating_add(1);
                    }
                }

                missing.clear();
                missing.try_reserve(requirement_range.len())?;
                missing.extend(
                    requirement_range
                        .clone()
                        .filter(|&position| bindings[position] == usize::MAX)
                        .map(|position| (requirements[position], position)),
                );
                missing.sort_unstable_by_key(|&(object, _)| {
                    (
                        std::cmp::Reverse(objects[object].block.decoded_len()),
                        object,
                    )
                });
                let mut made_progress = false;
                for &(object, position) in &missing {
                    let decoded_len = objects[object].block.decoded_len();
                    let allocation_len = align_up(decoded_len, alignment)?;
                    if allocation_len > capacity {
                        return Err(Error::ResourceLimit(format!(
                            "batch {next_batch} contains a decoded cache object requiring {allocation_len} aligned bytes, cache capacity is {capacity}"
                        )));
                    }
                    let Some(extent) = allocator.allocate(allocation_len) else {
                        if allocator.total_free() >= allocation_len {
                            fragmentation_stalls = fragmentation_stalls.saturating_add(1);
                        } else {
                            capacity_stalls = capacity_stalls.saturating_add(1);
                        }
                        continue;
                    };
                    let generation = generations[object];
                    generations[object] = generation
                        .checked_add(1)
                        .ok_or_else(|| Error::ResourceLimit("cache generation overflow".into()))?;
                    let residency = residencies.len();
                    residencies.try_reserve(1)?;
                    residencies.push(StaticResidency {
                        object,
                        cache: CacheSlice {
                            offset: extent.offset,
                            len: decoded_len,
                            generation,
                        },
                        allocation_len,
                        earliest_consumer_batch: next_batch,
                        available_after_batch: extent.available_after_batch,
                        compile_refcount: 1,
                    });
                    resident[object] = Some(residency);
                    bindings[position] = residency;
                    cache_misses = cache_misses.saturating_add(1);
                    alignment_loss = alignment_loss
                        .checked_add(allocation_len - decoded_len)
                        .ok_or_else(|| {
                            Error::ResourceLimit("cache alignment loss overflow".into())
                        })?;
                    made_progress = true;
                }
                if bindings[requirement_range.clone()]
                    .iter()
                    .all(|&residency| residency != usize::MAX)
                {
                    next_batch += 1;
                    horizon_max = horizon_max.max(next_batch.saturating_sub(current_batch));
                    continue;
                }
                if !made_progress {
                    break;
                }
            }

            let current_range = requirement_ranges[current_batch].clone();
            if bindings[current_range.clone()].contains(&usize::MAX) {
                let required = requirements[current_range.clone()].iter().try_fold(
                    0usize,
                    |total, &object| {
                        total
                            .checked_add(align_up(objects[object].block.decoded_len(), alignment)?)
                            .ok_or_else(|| {
                                Error::ResourceLimit("batch cache working set overflow".into())
                            })
                    },
                )?;
                return Err(Error::ResourceLimit(format!(
                    "batch {current_batch} requires {required} aligned decoded-cache bytes, cache capacity is {capacity}; largest free extent is {}",
                    allocator.largest_free()
                )));
            }
            for &residency_id in &bindings[current_range] {
                let residency = &mut residencies[residency_id];
                residency.compile_refcount = residency
                    .compile_refcount
                    .checked_sub(1)
                    .ok_or_else(|| Error::Invariant("compile cache refcount underflow".into()))?;
                if residency.compile_refcount == 0 {
                    if resident[residency.object] != Some(residency_id) {
                        return Err(Error::Invariant(
                            "compile resident table generation mismatch".into(),
                        ));
                    }
                    resident[residency.object] = None;
                    allocator.release(
                        residency.cache.offset,
                        residency.allocation_len,
                        current_batch,
                    );
                }
            }
        }
        if next_batch != batch_count || residencies.iter().any(|value| value.compile_refcount != 0)
        {
            return Err(Error::Invariant(
                "static cache simulation did not drain every batch reference".into(),
            ));
        }
        if resident.iter().any(Option::is_some) {
            return Err(Error::Invariant(
                "static cache simulation retained a resident object".into(),
            ));
        }

        let mut residency_order = Vec::new();
        residency_order.try_reserve_exact(residencies.len())?;
        residency_order.extend(0..residencies.len());
        residency_order.sort_unstable_by_key(|&id| {
            let residency = &residencies[id];
            let object = &objects[residency.object];
            (
                residency.available_after_batch.is_some(),
                residency.earliest_consumer_batch,
                epoch_sort_key(residency.available_after_batch),
                object.block.source,
                object.block.encoded_range.start,
                id,
            )
        });
        let mut io_tasks = Vec::new();
        io_tasks.try_reserve_exact(residencies.len())?;
        let mut decode_ops = Vec::new();
        decode_ops.try_reserve_exact(residencies.len())?;
        let mut initialize_count = 0usize;
        let mut initialize_decoded = 0usize;
        let mut initialize_io = 0usize;
        let mut load_owner = Vec::new();
        load_owner.try_reserve_exact(residencies.len())?;
        let mut data_io_ops = 0u64;
        let mut indices_io_ops = 0u64;
        let mut predicted_bytes = 0u64;
        let mut fusion_payload_bytes = 0u64;
        let mut maximum_decode_ops = 0usize;
        let mut maximum_io_decoded = 0usize;
        let merge = &self.request.config.io_merge;
        let mut residency_cursor = 0usize;
        let mut active_bucket_end = 0usize;
        let mut cost_group_ends = Vec::<usize>::new();
        let mut cost_group_cursor = 0usize;
        while residency_cursor < residency_order.len() {
            let first_id = residency_order[residency_cursor];
            let first = &residencies[first_id];
            let first_object = &objects[first.object];
            let whole_key_len = match &self.sources[first_object.block.source] {
                ReadSource::WholeKey { declared_len, .. } => Some(*declared_len),
                _ => None,
            };
            let mut read_start = first_object.block.encoded_range.start;
            let mut read_end = first_object.block.encoded_range.end;
            let mut payload_bytes = range_len_u64(&first_object.block.encoded_range)?;
            let mut decoded_bytes = first_object.block.decoded_len();
            if decoded_bytes > merge.max_decoded_bytes_per_io_task {
                return Err(Error::ResourceLimit(format!(
                    "one cache object requires {decoded_bytes} decoded bytes in an I/O task, limit is {}",
                    merge.max_decoded_bytes_per_io_task
                )));
            }
            if whole_key_len.is_none() && payload_bytes > merge.max_encoded_staging_bytes_per_task {
                return Err(Error::ResourceLimit(format!(
                    "one cache object requires {payload_bytes} encoded staging bytes, limit is {}",
                    merge.max_encoded_staging_bytes_per_task
                )));
            }
            if residency_cursor >= active_bucket_end {
                active_bucket_end = residency_cursor + 1;
                while active_bucket_end < residency_order.len() {
                    let candidate = &residencies[residency_order[active_bucket_end]];
                    let object = &objects[candidate.object];
                    if object.side != first_object.side
                        || object.block.source != first_object.block.source
                        || candidate.available_after_batch != first.available_after_batch
                        || candidate.earliest_consumer_batch != first.earliest_consumer_batch
                    {
                        break;
                    }
                    active_bucket_end += 1;
                }
                cost_group_ends.clear();
                cost_group_cursor = 0;
                if merge.policy == IoMergePolicy::CostAware && whole_key_len.is_none() {
                    let base_ends = cost_aware_group_ends(
                        &residency_order[residency_cursor..active_bucket_end],
                        &residencies,
                        &objects,
                        merge,
                    )?;
                    let bucket_len = active_bucket_end - residency_cursor;
                    let parallelism_hint = if first.available_after_batch.is_none() {
                        merge.initialize_parallelism_hint
                    } else {
                        merge.regular_io_parallelism_hint
                    };
                    let task_floor = parallelism_hint
                        .saturating_mul(merge.min_tasks_per_worker)
                        .min(bucket_len);
                    let mut additional_per_group = try_filled_vec(base_ends.len(), 0usize)?;
                    let mut candidates = Vec::<(std::cmp::Reverse<usize>, usize)>::new();
                    candidates.try_reserve_exact(base_ends.len())?;
                    let mut group_start = 0usize;
                    for (group, &group_end) in base_ends.iter().enumerate() {
                        let additional = group_end
                            .checked_sub(group_start)
                            .and_then(|length| length.checked_sub(1))
                            .ok_or_else(|| {
                                Error::Invariant("CostAware partition is not monotonic".into())
                            })?;
                        if additional != 0 {
                            candidates.push((std::cmp::Reverse(additional), group));
                        }
                        group_start = group_end;
                    }
                    candidates.sort_unstable();
                    let needed = task_floor.saturating_sub(base_ends.len());
                    let mut additional_tasks = 0usize;
                    for (std::cmp::Reverse(additional), group) in candidates {
                        if additional_tasks >= needed {
                            break;
                        }
                        let selected = additional.min(needed - additional_tasks);
                        additional_per_group[group] = selected;
                        additional_tasks =
                            additional_tasks.checked_add(selected).ok_or_else(|| {
                                Error::ResourceLimit("CostAware task floor overflow".into())
                            })?;
                    }
                    if additional_tasks < needed {
                        return Err(Error::Invariant(
                            "CostAware partition cannot satisfy its task floor".into(),
                        ));
                    }
                    // A hard-limit fallback may expand any selected group into
                    // individual blocks, so reserve the complete safe upper
                    // bound before pushing planned endpoints.
                    cost_group_ends.try_reserve_exact(bucket_len)?;
                    let mut group_start = 0usize;
                    for (group, &group_end) in base_ends.iter().enumerate() {
                        let group_len = group_end.checked_sub(group_start).ok_or_else(|| {
                            Error::Invariant("CostAware partition is not monotonic".into())
                        })?;
                        let parts =
                            additional_per_group[group].checked_add(1).ok_or_else(|| {
                                Error::ResourceLimit("CostAware task count overflow".into())
                            })?;
                        let base_len = group_len / parts;
                        let larger_parts = group_len % parts;
                        let mut split_is_feasible = true;
                        let mut end = group_start;
                        for part in 0..parts {
                            let start = end;
                            end = end
                                .checked_add(base_len + usize::from(part < larger_parts))
                                .ok_or_else(|| {
                                    Error::ResourceLimit("CostAware group end overflow".into())
                                })?;
                            split_is_feasible &= cost_aware_group_is_feasible(
                                &residency_order[residency_cursor + start..residency_cursor + end],
                                &residencies,
                                &objects,
                                merge,
                            )?;
                        }
                        if end != group_end {
                            return Err(Error::Invariant(
                                "CostAware split did not cover its source group".into(),
                            ));
                        }
                        if split_is_feasible {
                            let mut end = group_start;
                            for part in 0..parts {
                                end = end
                                    .checked_add(base_len + usize::from(part < larger_parts))
                                    .ok_or_else(|| {
                                        Error::ResourceLimit("CostAware group end overflow".into())
                                    })?;
                                cost_group_ends.push(residency_cursor + end);
                            }
                        } else {
                            for end in group_start + 1..=group_end {
                                cost_group_ends.push(residency_cursor + end);
                            }
                        }
                        group_start = group_end;
                    }
                }
            }
            let bucket_end = active_bucket_end;
            let parallelism_hint = if first.available_after_batch.is_none() {
                merge.initialize_parallelism_hint
            } else {
                merge.regular_io_parallelism_hint
            };
            let task_floor = parallelism_hint
                .saturating_mul(merge.min_tasks_per_worker)
                .min(bucket_end - residency_cursor);
            let floor_limited_end = bucket_end.saturating_sub(task_floor.saturating_sub(1));
            let policy_group_end = match merge.policy {
                IoMergePolicy::Off => residency_cursor + 1,
                _ if whole_key_len.is_some() => bucket_end,
                IoMergePolicy::Adjacent => floor_limited_end,
                IoMergePolicy::CostAware => {
                    let end = *cost_group_ends.get(cost_group_cursor).ok_or_else(|| {
                        Error::Invariant("CostAware I/O partition was exhausted early".into())
                    })?;
                    cost_group_cursor += 1;
                    end
                }
            };
            let maximum_group_end = if (whole_key_len.is_some()
                && !matches!(merge.policy, IoMergePolicy::Off))
                || (whole_key_len.is_none() && merge.policy == IoMergePolicy::CostAware)
            {
                policy_group_end
            } else {
                policy_group_end.min(floor_limited_end)
            };
            let mut group_end = residency_cursor + 1;
            while group_end < maximum_group_end {
                let candidate = &residencies[residency_order[group_end]];
                let object = &objects[candidate.object];
                let gap = object.block.encoded_range.start.saturating_sub(read_end);
                let combined_end = read_end.max(object.block.encoded_range.end);
                let combined_len = combined_end
                    .checked_sub(read_start)
                    .and_then(|len| usize::try_from(len).ok())
                    .unwrap_or(usize::MAX);
                let block_encoded = range_len_u64(&object.block.encoded_range)?;
                let Some(next_payload) = payload_bytes.checked_add(block_encoded) else {
                    break;
                };
                let Some(next_decoded) = decoded_bytes.checked_add(object.block.decoded_len())
                else {
                    break;
                };
                let next_ops = group_end + 1 - residency_cursor;
                let amplification = if next_payload == 0 {
                    1.0
                } else {
                    combined_len as f64 / next_payload as f64
                };
                let policy_allows = match merge.policy {
                    IoMergePolicy::Off => false,
                    IoMergePolicy::Adjacent => gap == 0,
                    IoMergePolicy::CostAware => true,
                };
                if object.side != first_object.side
                    || object.block.source != first_object.block.source
                    || candidate.available_after_batch != first.available_after_batch
                    || candidate.earliest_consumer_batch != first.earliest_consumer_batch
                    || !policy_allows
                    || next_ops > merge.max_decode_ops_per_io_task
                    || next_decoded > merge.max_decoded_bytes_per_io_task
                    || (whole_key_len.is_none()
                        && (combined_len > merge.max_coalesced_io_bytes
                            || combined_len > merge.max_encoded_staging_bytes_per_task
                            || amplification > merge.max_io_amplification_ratio))
                {
                    break;
                }
                read_end = combined_end;
                payload_bytes = next_payload;
                decoded_bytes = next_decoded;
                group_end += 1;
            }
            let encoded_len = if let Some(declared_len) = whole_key_len {
                read_start = 0;
                if declared_len > merge.max_encoded_staging_bytes_per_task {
                    return Err(Error::ResourceLimit(format!(
                        "whole-key I/O task requires {declared_len} staging bytes, limit is {}",
                        merge.max_encoded_staging_bytes_per_task
                    )));
                }
                declared_len
            } else {
                usize::try_from(read_end - read_start)
                    .map_err(|_| Error::ResourceLimit("coalesced I/O range exceeds usize".into()))?
            };
            let decode_start = decode_ops.len();
            for &residency_id in &residency_order[residency_cursor..group_end] {
                let residency = &residencies[residency_id];
                let object = &objects[residency.object];
                let block_len = range_len_u64(&object.block.encoded_range)?;
                decode_ops.push(DecodeOp {
                    encoded_offset: object
                        .block
                        .encoded_range
                        .start
                        .checked_sub(read_start)
                        .and_then(|offset| usize::try_from(offset).ok())
                        .ok_or_else(|| {
                            Error::ResourceLimit("coalesced block offset exceeds usize".into())
                        })?,
                    encoded_len: block_len,
                    decoder: object.block.decoder,
                    cache: residency.cache,
                    ready_node: residency_id,
                });
                if first.available_after_batch.is_none() {
                    initialize_decoded = initialize_decoded
                        .checked_add(residency.cache.len)
                        .ok_or_else(|| {
                            Error::ResourceLimit("InitializeJob decoded bytes overflow".into())
                        })?;
                }
            }
            io_tasks.push(IoDecodeLoadTask {
                source: first_object.block.source,
                file_offset: read_start,
                file_len: encoded_len,
                decode_ops: decode_start..decode_ops.len(),
                earliest_consumer_batch: first.earliest_consumer_batch as u64,
                available_after_batch: first.available_after_batch.map(|value| value as u64),
            });
            load_owner.push(first.earliest_consumer_batch);
            if first.available_after_batch.is_none() {
                initialize_count = initialize_count.checked_add(1).ok_or_else(|| {
                    Error::ResourceLimit("InitializeJob task count overflow".into())
                })?;
                initialize_io = initialize_io.checked_add(encoded_len).ok_or_else(|| {
                    Error::ResourceLimit("InitializeJob I/O bytes overflow".into())
                })?;
            }
            match first_object.side {
                Side::Data => {
                    data_io_ops = data_io_ops.checked_add(1).ok_or_else(|| {
                        Error::ResourceLimit("data I/O operation count overflow".into())
                    })?;
                }
                Side::Indices => {
                    indices_io_ops = indices_io_ops.checked_add(1).ok_or_else(|| {
                        Error::ResourceLimit("indices I/O operation count overflow".into())
                    })?;
                }
            }
            predicted_bytes =
                predicted_bytes
                    .checked_add(u64::try_from(encoded_len).map_err(|_| {
                        Error::ResourceLimit("predicted I/O bytes exceed u64".into())
                    })?)
                    .ok_or_else(|| Error::ResourceLimit("predicted I/O bytes overflow".into()))?;
            fusion_payload_bytes =
                fusion_payload_bytes
                    .checked_add(u64::try_from(payload_bytes).map_err(|_| {
                        Error::ResourceLimit("I/O fusion payload exceeds u64".into())
                    })?)
                    .ok_or_else(|| Error::ResourceLimit("I/O fusion payload overflow".into()))?;
            maximum_decode_ops = maximum_decode_ops.max(group_end - residency_cursor);
            maximum_io_decoded = maximum_io_decoded.max(decoded_bytes);
            residency_cursor = group_end;
        }

        let csr_task_capacity = cell_blocks.iter().try_fold(0usize, |count, &candidate| {
            let is_csr = self
                .block_candidates
                .get(candidate)
                .ok_or_else(|| Error::Invariant("cell block candidate is missing".into()))?
                .indices
                .is_some();
            count
                .checked_add(usize::from(is_csr))
                .ok_or_else(|| Error::ResourceLimit("CSR task count overflow".into()))
        })?;
        let dense_task_capacity = cells
            .len()
            .checked_sub(csr_task_capacity)
            .ok_or_else(|| Error::Invariant("CSR task count exceeds cell count".into()))?;
        let mut dense_tasks = Vec::<DenseScatterTask>::new();
        dense_tasks.try_reserve_exact(dense_task_capacity)?;
        let mut csr_tasks = Vec::<CsrScatterTask>::new();
        csr_tasks.try_reserve_exact(csr_task_capacity)?;
        let mut dense_loads = Vec::<Option<usize>>::new();
        dense_loads.try_reserve_exact(dense_task_capacity)?;
        let mut csr_loads = Vec::<[usize; 2]>::new();
        csr_loads.try_reserve_exact(csr_task_capacity)?;
        let mut jobs = Vec::<StaticJob>::new();
        jobs.try_reserve_exact(batch_count)?;
        let mut io_cursor = initialize_count;
        for (batch, requirement_range) in requirement_ranges.iter().enumerate() {
            let requirement_range = requirement_range.clone();
            let batch_requirements = &requirements[requirement_range.clone()];
            let batch_bindings = &bindings[requirement_range];
            for (&object, &residency) in batch_requirements.iter().zip(batch_bindings) {
                // SAFETY: requirements contain compiler-created object IDs and
                // `resident` was sized from the complete object arena.
                unsafe { *resident.get_unchecked_mut(object) = Some(residency) };
            }
            let io_start = io_cursor;
            while io_cursor < io_tasks.len() && load_owner[io_cursor] == batch {
                io_cursor += 1;
            }
            let dense_start = dense_tasks.len();
            let csr_start = csr_tasks.len();
            let row_start = batch * self.request.batch_size;
            let row_end = (row_start + self.request.batch_size).min(cells.len());
            let job_cells = row_end - row_start;
            if job_cells > self.request.config.limits.max_cells_per_job {
                return Err(Error::ResourceLimit(format!(
                    "batch {batch} has {job_cells} cells, limit is {}",
                    self.request.config.limits.max_cells_per_job
                )));
            }
            for ordinal in row_start..row_end {
                // SAFETY: the row range is clipped to `cells.len()`, and the
                // parallel cell-block arena has exactly the same length.
                let candidate_id = unsafe { *cell_blocks.get_unchecked(ordinal) };
                // SAFETY: candidate IDs were interned against this frozen arena.
                let candidate = unsafe { self.block_candidates.get_unchecked(candidate_id) };
                // SAFETY: `ordinal < row_end <= cells.len()`.
                let compact = unsafe { cells.get_unchecked(ordinal) };
                let output = OutputSlice {
                    ring_offset: compact.output_slot.row_offset(),
                    len: row_stride,
                    generation: batch as u64,
                };
                let cell = CellTask::new(
                    compact.output_slot,
                    compact.data_range(),
                    compact.indices_range(),
                )
                .ok_or_else(|| Error::ResourceLimit("static cell task range exceeds u32".into()))?;
                let lookup = |object: usize| -> Result<(CacheSlice, usize)> {
                    // SAFETY: data/indices object IDs were created from this
                    // object arena. The batch requirement pass installed every
                    // object needed by the current cell before task lowering.
                    let residency = unsafe { *resident.get_unchecked(object) }
                        .ok_or_else(|| Error::Invariant("batch cache binding is missing".into()))?;
                    // SAFETY: batch bindings store only IDs returned when a
                    // residency was appended to this frozen arena.
                    Ok((
                        unsafe { residencies.get_unchecked(residency).cache },
                        residency,
                    ))
                };
                if candidate.indices.is_some() {
                    // SAFETY: the candidate-domain proof above covers both maps.
                    let data_object = unsafe { *data_objects.get_unchecked(candidate_id) };
                    // SAFETY: a CSR candidate always installed an indices object.
                    let indices_object = unsafe { *indices_objects.get_unchecked(candidate_id) };
                    let (data, data_load) = lookup(data_object)?;
                    let (indices, indices_load) = lookup(indices_object)?;
                    csr_tasks.push(CsrScatterTask {
                        data,
                        indices,
                        cell,
                        output,
                        source_plan: candidate.did as usize,
                        completion_node: 0,
                    });
                    csr_loads.push([data_load, indices_load]);
                } else {
                    let (data, load) = if candidate.data.is_some() {
                        // SAFETY: a present data block installed this candidate map entry.
                        let data_object = unsafe { *data_objects.get_unchecked(candidate_id) };
                        let (slice, load) = lookup(data_object)?;
                        (Some(slice), Some(load))
                    } else {
                        (None, None)
                    };
                    dense_tasks.push(DenseScatterTask {
                        data,
                        cell,
                        output,
                        source_plan: candidate.did as usize,
                        completion_node: 0,
                    });
                    dense_loads.push(load);
                }
            }
            jobs.push(StaticJob {
                batch_id: batch as u64,
                io_tasks: io_start..io_cursor,
                csr_tasks: csr_start..csr_tasks.len(),
                dense_tasks: dense_start..dense_tasks.len(),
                output_slot: if self.ring_slots == 0 {
                    0
                } else {
                    batch % self.ring_slots
                },
                output_generation: batch as u64,
                completion_node: 0,
            });
            for &object in batch_requirements {
                // SAFETY: this is the same validated object slice installed at
                // the start of the batch; clearing restores the simulation map.
                unsafe { *resident.get_unchecked_mut(object) = None };
            }
        }
        if io_cursor != io_tasks.len() {
            return Err(Error::Invariant(
                "regular load tasks are not grouped by owner batch".into(),
            ));
        }

        let load_count = io_tasks.len();
        let dense_base = load_count;
        let csr_base = dense_base + dense_tasks.len();
        let executable = csr_base + csr_tasks.len();
        let mut dependency_counts = try_filled_vec(executable, 0u32)?;
        let mut block_successor_lists = use_nested_buckets(residencies.len())
            .then(|| try_filled_vec(residencies.len(), Vec::<usize>::new()))
            .transpose()?;
        let mut block_successor_counts = if block_successor_lists.is_none() {
            try_filled_vec(residencies.len(), 0usize)?
        } else {
            Vec::new()
        };
        for (load, task) in io_tasks.iter().enumerate() {
            if let Some(batch) = task.available_after_batch {
                let batch = usize::try_from(batch)
                    .map_err(|_| Error::ResourceLimit("prefix batch exceeds usize".into()))?;
                if batch >= batch_count {
                    return Err(Error::Invariant(
                        "prefix release batch is outside the plan".into(),
                    ));
                }
                dependency_counts[load] = 1;
            }
        }
        for (task_id, load) in dense_loads.iter().enumerate() {
            let node = dense_base + task_id;
            if let Some(ready) = *load {
                if let Some(lists) = block_successor_lists.as_mut() {
                    // SAFETY: every `ready` value is a residency ID returned
                    // by the static binding table used to size these lists.
                    let list = unsafe { lists.get_unchecked_mut(ready) };
                    list.try_reserve(1)?;
                    list.push(node);
                } else {
                    // SAFETY: the flat count arena has the same residency domain.
                    let successor_count =
                        unsafe { block_successor_counts.get_unchecked_mut(ready) };
                    *successor_count = successor_count
                        .checked_add(1)
                        .ok_or_else(|| Error::ResourceLimit("successor count overflow".into()))?;
                }
                // SAFETY: `dense_base + task_id` is inside the dense segment
                // used to size `dependency_counts`.
                let dependency_count = unsafe { dependency_counts.get_unchecked_mut(node) };
                *dependency_count = dependency_count
                    .checked_add(1)
                    .ok_or_else(|| Error::ResourceLimit("dependency count exceeds u32".into()))?;
            }
            let batch = dense_tasks[task_id].output.generation as usize;
            if batch >= self.ring_slots {
                // SAFETY: the dense node proof above is independent of its
                // cache-edge count.
                let dependency_count = unsafe { dependency_counts.get_unchecked_mut(node) };
                *dependency_count = dependency_count
                    .checked_add(1)
                    .ok_or_else(|| Error::ResourceLimit("dependency count exceeds u32".into()))?;
            }
            dense_tasks[task_id].completion_node = node;
        }
        for (task_id, loads) in csr_loads.iter().enumerate() {
            let node = csr_base + task_id;
            for &ready in loads {
                if let Some(lists) = block_successor_lists.as_mut() {
                    // SAFETY: CSR load IDs come from the same residency binding
                    // table used to size these successor lists.
                    let list = unsafe { lists.get_unchecked_mut(ready) };
                    list.try_reserve(1)?;
                    list.push(node);
                } else {
                    // SAFETY: the flat count arena has the same residency domain.
                    let successor_count =
                        unsafe { block_successor_counts.get_unchecked_mut(ready) };
                    *successor_count = successor_count
                        .checked_add(1)
                        .ok_or_else(|| Error::ResourceLimit("successor count overflow".into()))?;
                }
                // SAFETY: `csr_base + task_id` is inside the CSR segment used
                // to size `dependency_counts`.
                let dependency_count = unsafe { dependency_counts.get_unchecked_mut(node) };
                *dependency_count = dependency_count
                    .checked_add(1)
                    .ok_or_else(|| Error::ResourceLimit("dependency count exceeds u32".into()))?;
            }
            let batch = csr_tasks[task_id].output.generation as usize;
            if batch >= self.ring_slots {
                // SAFETY: the CSR node proof above is independent of its
                // cache-edge count.
                let dependency_count = unsafe { dependency_counts.get_unchecked_mut(node) };
                *dependency_count = dependency_count
                    .checked_add(1)
                    .ok_or_else(|| Error::ResourceLimit("dependency count exceeds u32".into()))?;
            }
            csr_tasks[task_id].completion_node = node;
        }
        for (batch, job) in jobs.iter_mut().enumerate() {
            job.completion_node = executable + batch;
        }
        let (block_ready_ranges, block_ready_successors) =
            if let Some(lists) = block_successor_lists {
                flatten_successor_lists(lists)?
            } else {
                let successor_count =
                    block_successor_counts
                        .iter()
                        .try_fold(0usize, |total, &count| {
                            total.checked_add(count).ok_or_else(|| {
                                Error::ResourceLimit("block successor arena count overflow".into())
                            })
                        })?;
                let mut ranges = Vec::new();
                ranges.try_reserve_exact(block_successor_counts.len())?;
                let mut successors = Vec::new();
                successors.try_reserve_exact(successor_count)?;
                successors.resize(successor_count, usize::MAX);
                let mut cursors = Vec::new();
                cursors.try_reserve_exact(block_successor_counts.len())?;
                let mut successor_cursor = 0usize;
                for &count in &block_successor_counts {
                    let start = successor_cursor;
                    successor_cursor = successor_cursor.checked_add(count).ok_or_else(|| {
                        Error::ResourceLimit("block successor arena count overflow".into())
                    })?;
                    ranges.push(start..successor_cursor);
                    cursors.push(start);
                }
                for (task_id, load) in dense_loads.iter().enumerate() {
                    if let Some(ready) = *load {
                        // SAFETY: the count pass reserved exactly one slot for
                        // this repeated dense edge.
                        let cursor = unsafe { cursors.get_unchecked_mut(ready) };
                        // SAFETY: count and fill traverse identical load data.
                        unsafe { *successors.get_unchecked_mut(*cursor) = dense_base + task_id };
                        *cursor += 1;
                    }
                }
                for (task_id, loads) in csr_loads.iter().enumerate() {
                    let node = csr_base + task_id;
                    for &ready in loads {
                        // SAFETY: the count pass reserved one slot per CSR edge.
                        let cursor = unsafe { cursors.get_unchecked_mut(ready) };
                        // SAFETY: count and fill traverse identical load data.
                        unsafe { *successors.get_unchecked_mut(*cursor) = node };
                        *cursor += 1;
                    }
                }
                if cursors
                    .iter()
                    .zip(&ranges)
                    .any(|(cursor, range)| *cursor != range.end)
                {
                    return Err(Error::Invariant(
                        "block successor CSR fill is incomplete".into(),
                    ));
                }
                (ranges, successors)
            };
        let prefix_releases = build_prefix_release_plan(batch_count, &io_tasks)?;
        let ring_releases = build_ring_release_plan(
            batch_count,
            self.ring_slots,
            dense_base,
            csr_base,
            &dense_tasks,
            &csr_tasks,
        )?;

        let arena_bytes = checked_payload_bytes(&[
            (io_tasks.len(), size_of::<IoDecodeLoadTask>()),
            (decode_ops.len(), size_of::<DecodeOp>()),
            (dense_tasks.len(), size_of::<DenseScatterTask>()),
            (csr_tasks.len(), size_of::<CsrScatterTask>()),
            (jobs.len(), size_of::<StaticJob>()),
            (dependency_counts.len(), size_of::<u32>()),
            (block_ready_ranges.len(), size_of::<Range<usize>>()),
            (block_ready_successors.len(), size_of::<usize>()),
            (
                prefix_releases.release_ranges.len(),
                size_of::<Range<usize>>(),
            ),
            (prefix_releases.released_nodes.len(), size_of::<usize>()),
            (
                ring_releases.release_ranges.len(),
                size_of::<Range<usize>>(),
            ),
            (ring_releases.released_nodes.len(), size_of::<usize>()),
        ])?;
        if arena_bytes > self.request.config.limits.max_compile_arena_bytes {
            return Err(Error::ResourceLimit(format!(
                "compile arena has {arena_bytes} bytes, limit is {}",
                self.request.config.limits.max_compile_arena_bytes
            )));
        }

        stats.input_rows = cells.len();
        stats.block_jobs = objects.len();
        stats.jobs = batch_count;
        stats.data_io_ops = data_io_ops;
        stats.indices_io_ops = indices_io_ops;
        stats.predicted_physical_bytes = predicted_bytes;
        stats.gap_bytes = predicted_bytes
            .checked_sub(fusion_payload_bytes)
            .ok_or_else(|| Error::Invariant("I/O payload exceeds physical span".into()))?;
        stats.maximum_encoded_bytes_per_side = objects
            .iter()
            .map(|object| range_len_u64(&object.block.encoded_range).unwrap_or(usize::MAX))
            .max()
            .unwrap_or(0);
        stats.maximum_decoded_bytes_per_job = maximum_batch_decoded_bytes;
        stats.output_ring_bytes = row_stride
            .checked_mul(self.request.batch_size)
            .and_then(|bytes| bytes.checked_mul(self.ring_slots))
            .ok_or_else(|| Error::ResourceLimit("output ring bytes overflow".into()))?;
        stats.cache_capacity_bytes = capacity;
        stats.cache_arena_bytes = capacity;
        stats.cache_alignment_loss_bytes = alignment_loss;
        stats.unique_cache_objects = objects.len();
        stats.residency_loads = residencies.len();
        stats.residency_reloads = residencies
            .len()
            .checked_sub(objects.len())
            .ok_or_else(|| Error::Invariant("fewer residencies than cache objects".into()))?;
        stats.cache_reference_hits = cache_hits;
        stats.cache_reference_misses = cache_misses;
        stats.cache_capacity_stalls = capacity_stalls;
        stats.cache_fragmentation_stalls = fragmentation_stalls;
        stats.cache_horizon_max_batches = horizon_max;
        stats.output_ring_slots = self.ring_slots;
        stats.initialize_io_tasks = initialize_count;
        stats.executable_tasks = executable;
        stats.dependency_edges = block_ready_successors.len();
        stats.independent_block_loads = residencies.len();
        stats.fused_io_tasks = io_tasks.len();
        stats.predicted_io_ops_saved =
            residencies
                .len()
                .checked_sub(io_tasks.len())
                .ok_or_else(|| {
                    Error::Invariant("I/O fusion created more tasks than independent loads".into())
                })?;
        stats.io_payload_bytes = fusion_payload_bytes;
        stats.io_span_bytes = predicted_bytes;
        stats.io_read_amplification = if fusion_payload_bytes == 0 {
            1.0
        } else {
            predicted_bytes as f64 / fusion_payload_bytes as f64
        };
        stats.maximum_decode_ops_per_io_task = maximum_decode_ops;
        stats.maximum_decoded_bytes_per_io_task = maximum_io_decoded;
        stats.initialize_fused_io_tasks = initialize_count;
        stats.regular_fused_io_tasks =
            io_tasks
                .len()
                .checked_sub(initialize_count)
                .ok_or_else(|| {
                    Error::Invariant("InitializeJob exceeds the fused I/O task arena".into())
                })?;
        stats.arena_bytes = arena_bytes;
        let _ = row_bytes;

        Ok(StaticPlanData {
            initialize: InitializeJob {
                io_tasks: 0..initialize_count,
                decoded_bytes: initialize_decoded,
                io_bytes: initialize_io,
            },
            jobs: jobs.into_boxed_slice(),
            io_decode_tasks: io_tasks.into_boxed_slice(),
            decode_ops: decode_ops.into_boxed_slice(),
            csr_scatter_tasks: csr_tasks.into_boxed_slice(),
            dense_scatter_tasks: dense_tasks.into_boxed_slice(),
            dependencies: DependencyGraph {
                initial_dependency_count: dependency_counts.into_boxed_slice(),
                block_ready_ranges: block_ready_ranges.into_boxed_slice(),
                block_ready_successors: block_ready_successors.into_boxed_slice(),
            },
            prefix_releases,
            ring_releases,
            cache_capacity: capacity,
            cache_alignment: alignment,
        })
    }
}

pub(crate) fn build_default_ranges(
    targets: Option<&[Option<usize>]>,
    output_columns: usize,
    target_size: usize,
) -> Result<Arc<[OutputRange]>> {
    let Some(targets) = targets else {
        return Ok(Arc::from([]));
    };
    let mut mapped = Vec::new();
    mapped.try_reserve_exact(targets.len().min(output_columns))?;
    mapped.extend(targets.iter().copied().flatten());
    mapped.sort_unstable();
    if mapped.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Error::Invariant(
            "feature map contains duplicate output targets".into(),
        ));
    }

    let mut ranges = Vec::new();
    ranges.try_reserve_exact(mapped.len().saturating_add(1))?;
    let mut start = 0usize;
    for target in mapped {
        if target >= output_columns {
            return Err(Error::Invariant(
                "feature map target exceeds output columns".into(),
            ));
        }
        if start < target {
            ranges.push(output_range(start, target, target_size)?);
        }
        start = target
            .checked_add(1)
            .ok_or_else(|| Error::ResourceLimit("default output range overflow".into()))?;
    }
    if start < output_columns {
        ranges.push(output_range(start, output_columns, target_size)?);
    }
    Ok(Arc::from(ranges))
}

pub(crate) fn choose_dense_whole_fill(
    targets: Option<&[Option<usize>]>,
    target_size: usize,
    gap_count: usize,
) -> Result<bool> {
    let Some(targets) = targets else {
        return Ok(false);
    };
    if gap_count == 0 {
        return Ok(false);
    }
    let mapped = targets.iter().filter(|target| target.is_some()).count();
    let mapped_bytes = mapped
        .checked_mul(target_size)
        .ok_or_else(|| Error::ResourceLimit("dense mapped byte count overflow".into()))?;
    let minimum_savings = gap_count.saturating_mul(MIN_DENSE_OVERWRITE_BYTES_PER_GAP);
    // Range-only initialization saves `mapped_bytes` of repeated stores, but
    // each gap adds a scattered fill operation. Below this ratio one streaming
    // fill is cheaper and also avoids retaining a fragmented range table.
    Ok(mapped_bytes != 0 && mapped_bytes < minimum_savings)
}

fn output_range(start: usize, end: usize, target_size: usize) -> Result<OutputRange> {
    let offset = start
        .checked_mul(target_size)
        .ok_or_else(|| Error::ResourceLimit("default output byte offset overflow".into()))?;
    let len = end
        .checked_sub(start)
        .and_then(|elements| elements.checked_mul(target_size))
        .ok_or_else(|| Error::ResourceLimit("default output byte length overflow".into()))?;
    Ok(OutputRange { offset, len })
}

pub(crate) fn build_dense_map(
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
        mapped
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| Error::ResourceLimit("dense map entry bytes overflow".into()))?
    } else {
        mapped
            .checked_mul(size_of::<DenseMapEntry>())
            .ok_or_else(|| Error::ResourceLimit("dense map entry bytes overflow".into()))?
    };
    let run_bytes = run_count
        .checked_mul(size_of::<DenseMapRun>())
        .ok_or_else(|| Error::ResourceLimit("dense map run bytes overflow".into()))?;
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
                if retained { declared } else { 0 },
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

fn flatten_successor_lists(mut lists: Vec<Vec<usize>>) -> Result<(Vec<Range<usize>>, Vec<usize>)> {
    let count = lists.iter().try_fold(0usize, |total, list| {
        total
            .checked_add(list.len())
            .ok_or_else(|| Error::ResourceLimit("block successor arena count overflow".into()))
    })?;
    let mut ranges = Vec::new();
    ranges.try_reserve_exact(lists.len())?;
    let mut successors = Vec::new();
    successors.try_reserve_exact(count)?;
    for list in &mut lists {
        let start = successors.len();
        successors.append(list);
        ranges.push(start..successors.len());
    }
    Ok((ranges, successors))
}

struct ReleaseStorage {
    ranges: Vec<Range<usize>>,
    nodes: Vec<usize>,
    cursors: Vec<usize>,
}

fn allocate_release_storage(counts: &[usize]) -> Result<ReleaseStorage> {
    let mut ranges = Vec::new();
    ranges.try_reserve_exact(counts.len())?;
    let mut cursor = 0usize;
    for &count in counts {
        let start = cursor;
        cursor = cursor
            .checked_add(count)
            .ok_or_else(|| Error::ResourceLimit("release plan entry count overflow".into()))?;
        ranges.push(start..cursor);
    }
    let mut nodes = Vec::new();
    nodes.try_reserve_exact(cursor)?;
    nodes.resize(cursor, usize::MAX);
    let mut cursors = Vec::new();
    cursors.try_reserve_exact(ranges.len())?;
    cursors.extend(ranges.iter().map(|range| range.start));
    Ok(ReleaseStorage {
        ranges,
        nodes,
        cursors,
    })
}

fn finish_release_plan(
    ranges: Vec<Range<usize>>,
    nodes: Vec<usize>,
    cursors: &[usize],
) -> Result<ReleasePlan> {
    if cursors
        .iter()
        .zip(&ranges)
        .any(|(cursor, range)| *cursor != range.end)
    {
        return Err(Error::Invariant(
            "release plan CSR fill is incomplete".into(),
        ));
    }
    Ok(ReleasePlan {
        release_ranges: ranges.into_boxed_slice(),
        released_nodes: nodes.into_boxed_slice(),
    })
}

fn flatten_release_lists(lists: Vec<Vec<usize>>) -> Result<ReleasePlan> {
    let (ranges, nodes) = flatten_successor_lists(lists)?;
    Ok(ReleasePlan {
        release_ranges: ranges.into_boxed_slice(),
        released_nodes: nodes.into_boxed_slice(),
    })
}

fn build_prefix_release_plan(
    batch_count: usize,
    io_tasks: &[IoDecodeLoadTask],
) -> Result<ReleasePlan> {
    if use_nested_buckets(batch_count) {
        let mut buckets = try_filled_vec(batch_count, Vec::<usize>::new())?;
        for (node, task) in io_tasks.iter().enumerate() {
            let Some(batch) = task.available_after_batch else {
                continue;
            };
            let batch = usize::try_from(batch)
                .map_err(|_| Error::ResourceLimit("prefix batch exceeds usize".into()))?;
            let bucket = buckets.get_mut(batch).ok_or_else(|| {
                Error::Invariant("prefix release batch is outside the plan".into())
            })?;
            bucket.try_reserve(1)?;
            bucket.push(node);
        }
        return flatten_release_lists(buckets);
    }
    let mut counts = try_filled_vec(batch_count, 0usize)?;
    for task in io_tasks {
        let Some(batch) = task.available_after_batch else {
            continue;
        };
        let batch = usize::try_from(batch)
            .map_err(|_| Error::ResourceLimit("prefix batch exceeds usize".into()))?;
        let count = counts
            .get_mut(batch)
            .ok_or_else(|| Error::Invariant("prefix release batch is outside the plan".into()))?;
        *count = count
            .checked_add(1)
            .ok_or_else(|| Error::ResourceLimit("prefix release count overflow".into()))?;
    }
    let ReleaseStorage {
        ranges,
        mut nodes,
        mut cursors,
    } = allocate_release_storage(&counts)?;
    for (node, task) in io_tasks.iter().enumerate() {
        let Some(batch) = task.available_after_batch else {
            continue;
        };
        let batch = usize::try_from(batch)
            .map_err(|_| Error::ResourceLimit("prefix batch exceeds usize".into()))?;
        // SAFETY: the count pass validated every batch and reserved exactly
        // one destination for this repeated I/O task.
        let destination = unsafe { cursors.get_unchecked_mut(batch) };
        // SAFETY: count and fill traverse the identical immutable task arena.
        unsafe { *nodes.get_unchecked_mut(*destination) = node };
        *destination += 1;
    }
    finish_release_plan(ranges, nodes, &cursors)
}

fn build_ring_release_plan(
    batch_count: usize,
    ring_slots: usize,
    dense_base: usize,
    csr_base: usize,
    dense_tasks: &[DenseScatterTask],
    csr_tasks: &[CsrScatterTask],
) -> Result<ReleasePlan> {
    if use_nested_buckets(batch_count) {
        let mut buckets = try_filled_vec(batch_count, Vec::<usize>::new())?;
        for (task_id, task) in dense_tasks.iter().enumerate() {
            let batch = usize::try_from(task.output.generation)
                .map_err(|_| Error::ResourceLimit("dense ring batch exceeds usize".into()))?;
            if batch >= ring_slots {
                let release = batch - ring_slots;
                let bucket = buckets.get_mut(release).ok_or_else(|| {
                    Error::Invariant("ring release batch is outside the plan".into())
                })?;
                bucket.try_reserve(1)?;
                bucket.push(dense_base + task_id);
            }
        }
        for (task_id, task) in csr_tasks.iter().enumerate() {
            let batch = usize::try_from(task.output.generation)
                .map_err(|_| Error::ResourceLimit("CSR ring batch exceeds usize".into()))?;
            if batch >= ring_slots {
                let release = batch - ring_slots;
                let bucket = buckets.get_mut(release).ok_or_else(|| {
                    Error::Invariant("ring release batch is outside the plan".into())
                })?;
                bucket.try_reserve(1)?;
                bucket.push(csr_base + task_id);
            }
        }
        return flatten_release_lists(buckets);
    }
    let mut counts = try_filled_vec(batch_count, 0usize)?;
    for batch in dense_tasks
        .iter()
        .map(|task| task.output.generation)
        .chain(csr_tasks.iter().map(|task| task.output.generation))
    {
        let batch = usize::try_from(batch)
            .map_err(|_| Error::ResourceLimit("ring batch exceeds usize".into()))?;
        if batch < ring_slots {
            continue;
        }
        let release = batch - ring_slots;
        let count = counts
            .get_mut(release)
            .ok_or_else(|| Error::Invariant("ring release batch is outside the plan".into()))?;
        *count = count
            .checked_add(1)
            .ok_or_else(|| Error::ResourceLimit("ring release count overflow".into()))?;
    }
    let ReleaseStorage {
        ranges,
        mut nodes,
        mut cursors,
    } = allocate_release_storage(&counts)?;
    for (task_id, task) in dense_tasks.iter().enumerate() {
        let batch = usize::try_from(task.output.generation)
            .map_err(|_| Error::ResourceLimit("dense ring batch exceeds usize".into()))?;
        if batch >= ring_slots {
            let release = batch - ring_slots;
            // SAFETY: the count pass validated this release and reserved one
            // destination for the same dense task.
            let destination = unsafe { cursors.get_unchecked_mut(release) };
            // SAFETY: both passes traverse the same dense task arena.
            unsafe { *nodes.get_unchecked_mut(*destination) = dense_base + task_id };
            *destination += 1;
        }
    }
    for (task_id, task) in csr_tasks.iter().enumerate() {
        let batch = usize::try_from(task.output.generation)
            .map_err(|_| Error::ResourceLimit("CSR ring batch exceeds usize".into()))?;
        if batch >= ring_slots {
            let release = batch - ring_slots;
            // SAFETY: the count pass validated this release and reserved one
            // destination for the same CSR task.
            let destination = unsafe { cursors.get_unchecked_mut(release) };
            // SAFETY: both passes traverse the same CSR task arena.
            unsafe { *nodes.get_unchecked_mut(*destination) = csr_base + task_id };
            *destination += 1;
        }
    }
    finish_release_plan(ranges, nodes, &cursors)
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

#[cfg(test)]
mod required_chunk_marker_tests {
    use super::{required_chunk_is_set, set_required_chunk, CandidateRemap, ExtentAllocator};

    #[test]
    fn extent_allocator_tracks_free_bytes_across_split_and_merge() {
        let mut allocator = ExtentAllocator::new(1024, 0);
        let first = allocator.allocate(256).unwrap();
        let second = allocator.allocate(128).unwrap();
        assert_eq!(allocator.total_free(), 640);
        allocator.release(first.offset, first.len, 0);
        assert_eq!(allocator.total_free(), 896);
        allocator.release(second.offset, second.len, 1);
        assert_eq!(allocator.total_free(), 1024);
        assert_eq!(allocator.largest_free(), 1024);
    }

    #[test]
    fn candidate_remap_promotes_sparse_raw_ids_without_dense_growth() {
        let mut remap = CandidateRemap::new(0, 1).unwrap();
        assert_eq!(remap.intern(7, 0).unwrap(), (0, true));
        assert_eq!(remap.intern(1_000_000, 1).unwrap(), (1, true));
        assert_eq!(remap.intern(7, 2).unwrap(), (0, false));
        assert_eq!(remap.intern(1_000_000, 2).unwrap(), (1, false));
        assert!(matches!(remap, CandidateRemap::Sparse(_)));
    }

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
