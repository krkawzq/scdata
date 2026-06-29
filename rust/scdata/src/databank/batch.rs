#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::io;
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use crate::access::{
    AccessHandle, AccessItem, AccessRequest, ChunkKey, PrefetchCancel, RangeCopy, ScheduledAccess,
    ScheduledAccessConfig, SliceSpec,
};
use crate::codecs::{CodecError, SharedCodec};

use super::array::{Array, Chunk, ChunkRef, ChunkSource, DType, DataValue};
use super::compute::{ComputeJob, DataBankComputePool};
use super::config::ScheduledPrefetchConfig;
use super::dataset::{Dataset, Dense1DDataset, Dense2DDataset, SparseCsrDataset};
use super::error::{DataBankError, DataBankResult};
use super::interner::GeneNameView;
use super::plan::{self, ByteRange, DenseSegment, RangeSegment, SparseRowSpan};

type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
type FastHashSet<K> = HashSet<K, BuildHasherDefault<FastHasher>>;
const GENE_NOT_SELECTED: usize = usize::MAX;

#[inline]
fn row_count_for_width(len: usize, width: usize) -> usize {
    len.checked_div(width).unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingGenePolicy {
    #[default]
    Zero,
    Error,
}

#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            self.write_u64(u64::from_ne_bytes(word));
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut tail = [0u8; 8];
            tail[..rem.len()].copy_from_slice(rem);
            self.write_u64(u64::from_ne_bytes(tail));
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.mix(u64::from(i));
    }

    fn write_u16(&mut self, i: u16) {
        self.mix(u64::from(i));
    }

    fn write_u32(&mut self, i: u32) {
        self.mix(u64::from(i));
    }

    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }

    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

impl FastHasher {
    fn mix(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

fn fast_hash_map_with_capacity<K, V>(capacity: usize) -> FastHashMap<K, V> {
    HashMap::with_capacity_and_hasher(capacity, BuildHasherDefault::<FastHasher>::default())
}

fn fast_hash_set_with_capacity<K>(capacity: usize) -> FastHashSet<K> {
    HashSet::with_capacity_and_hasher(capacity, BuildHasherDefault::<FastHasher>::default())
}

#[derive(Debug, Clone)]
struct CompiledGeneProjection {
    output_by_source: Vec<usize>,
    output_names: Vec<GeneNameView>,
    selected_sources: Vec<usize>,
}

#[derive(Debug, Clone)]
enum GeneAxisPlan {
    DatasetOrder,
    Requested(CompiledGeneProjection),
}

impl GeneAxisPlan {
    fn dataset_order() -> Self {
        Self::DatasetOrder
    }

    fn requested<G>(
        dataset: &Dataset,
        requested: &[G],
        missing: MissingGenePolicy,
    ) -> DataBankResult<Self>
    where
        G: AsRef<str>,
    {
        if requested_matches_dataset_order(dataset, requested)? {
            return Ok(Self::DatasetOrder);
        }
        let projection = CompiledGeneProjection::new(dataset, requested, missing)?;
        if projection.is_identity(dataset.num_genes()) {
            Ok(Self::DatasetOrder)
        } else {
            Ok(Self::Requested(projection))
        }
    }

    fn output_genes(&self, dataset_num_genes: usize) -> usize {
        match self {
            Self::DatasetOrder => dataset_num_genes,
            Self::Requested(projection) => projection.output_genes(),
        }
    }

    fn output_names<'a>(&'a self, dataset: &'a Dataset) -> &'a [GeneNameView] {
        match self {
            Self::DatasetOrder => dataset.genes().views(),
            Self::Requested(projection) => projection.output_names(),
        }
    }

    fn fill_names(&self, dataset: &Dataset, names: &mut [GeneNameView]) -> DataBankResult<()> {
        let output_names = self.output_names(dataset);
        if names.len() != output_names.len() {
            return Err(DataBankError::NameBufferSizeMismatch {
                expected: output_names.len(),
                actual: names.len(),
            });
        }
        names.copy_from_slice(output_names);
        Ok(())
    }

    fn projection(&self) -> Option<&CompiledGeneProjection> {
        match self {
            Self::DatasetOrder => None,
            Self::Requested(projection) => Some(projection),
        }
    }

    fn requires_dense_zero_fill(&self) -> bool {
        match self {
            Self::DatasetOrder => false,
            Self::Requested(projection) => projection.has_missing_outputs(),
        }
    }
}

fn requested_matches_dataset_order<G>(dataset: &Dataset, requested: &[G]) -> DataBankResult<bool>
where
    G: AsRef<str>,
{
    let genes = dataset.genes();
    if let Some(name) = genes.duplicate_name() {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "duplicate dataset gene name: {name}"
        )));
    }
    if requested.len() != genes.len() {
        return Ok(false);
    }
    Ok(requested
        .iter()
        .zip(genes.names())
        .all(|(requested, dataset_name)| requested.as_ref() == dataset_name.as_ref()))
}

impl CompiledGeneProjection {
    fn new<G>(
        dataset: &Dataset,
        requested: &[G],
        missing: MissingGenePolicy,
    ) -> DataBankResult<Self>
    where
        G: AsRef<str>,
    {
        let genes = dataset.genes();
        if let Some(name) = genes.duplicate_name() {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "duplicate dataset gene name: {name}"
            )));
        }

        let mut seen = fast_hash_set_with_capacity(requested.len());
        let mut output_by_source = vec![GENE_NOT_SELECTED; dataset.num_genes()];
        let mut output_names = Vec::with_capacity(requested.len());
        let mut selected_sources = Vec::with_capacity(requested.len().min(dataset.num_genes()));

        for gene in requested {
            let name = gene.as_ref();
            if !seen.insert(name) {
                return Err(DataBankError::DuplicateGeneName {
                    gene: name.to_string(),
                });
            }

            if let Some(source) = genes.index_of(name) {
                let output = output_names.len();
                output_by_source[source] = output;
                output_names.push(genes.views()[source]);
                selected_sources.push(source);
            } else {
                match missing {
                    MissingGenePolicy::Zero => {
                        output_names.push(GeneNameView::empty());
                    }
                    MissingGenePolicy::Error => {
                        return Err(DataBankError::GeneNameNotFound {
                            gene: name.to_string(),
                        });
                    }
                }
            }
        }

        selected_sources.sort_unstable();
        Ok(Self {
            output_by_source,
            output_names,
            selected_sources,
        })
    }

    fn output_genes(&self) -> usize {
        self.output_names.len()
    }

    fn output_names(&self) -> &[GeneNameView] {
        &self.output_names
    }

    fn output_for_source(&self, source: usize) -> Option<usize> {
        let &output = self.output_by_source.get(source)?;
        (output != GENE_NOT_SELECTED).then_some(output)
    }

    fn has_missing_outputs(&self) -> bool {
        self.selected_sources.len() != self.output_names.len()
    }

    fn is_identity(&self, dataset_num_genes: usize) -> bool {
        self.output_names.len() == dataset_num_genes
            && self.contiguous_output_for_source_run(0, dataset_num_genes) == Some(0)
    }

    fn contiguous_output_for_source_run(&self, source_start: usize, len: usize) -> Option<usize> {
        if len == 0 {
            return Some(0);
        }
        let first_output = self.output_for_source(source_start)?;
        for offset in 1..len {
            let source = source_start.checked_add(offset)?;
            let output = first_output.checked_add(offset)?;
            if self.output_for_source(source)? != output {
                return None;
            }
        }
        Some(first_output)
    }
}

pub(super) fn validate_dtype_and_cells<T: DataValue>(
    dataset: &Dataset,
    cells: &[usize],
) -> DataBankResult<()> {
    if dataset.data_dtype() != T::DTYPE {
        return Err(DataBankError::UnsupportedDType {
            dtype: T::DTYPE,
            context: "access_cells output",
        });
    }
    for &cell in cells {
        if cell >= dataset.num_cells() {
            return Err(DataBankError::CellIndexOutOfRange {
                cell,
                num_cells: dataset.num_cells(),
            });
        }
    }
    Ok(())
}

pub fn validate_access<T: DataValue>(
    dataset: &Dataset,
    cells: &[usize],
    out: &[T],
    names: Option<&[GeneNameView]>,
    output_genes: usize,
) -> DataBankResult<()> {
    validate_dtype_and_cells::<T>(dataset, cells)?;
    let expected = cells
        .len()
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("output length overflow".to_string()))?;
    if out.len() != expected {
        return Err(DataBankError::BufferSizeMismatch {
            expected,
            actual: out.len(),
        });
    }
    if let Some(names) = names {
        let expected = output_genes;
        if names.len() != expected {
            return Err(DataBankError::NameBufferSizeMismatch {
                expected,
                actual: names.len(),
            });
        }
    }
    Ok(())
}

pub fn access_cells<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
) -> DataBankResult<()> {
    let gene_axis = GeneAxisPlan::dataset_order();
    validate_access(
        dataset,
        cells,
        out,
        names.as_deref(),
        gene_axis.output_genes(dataset.num_genes()),
    )?;
    match dataset {
        Dataset::Dense1D(dataset) => {
            access_dense_1d(access, compute, access_config, dataset, cells, out)?
        }
        Dataset::Dense2D(dataset) => {
            access_dense_2d(access, compute, access_config, dataset, cells, out)?
        }
        Dataset::SparseCsr(dataset) => {
            access_sparse(access, compute, access_config, dataset, cells, out)?
        }
    }
    if let Some(names) = names {
        gene_axis.fill_names(dataset, names)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn access_cells_by_gene_names<T, G>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    gene_names: &[G],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
    missing: MissingGenePolicy,
) -> DataBankResult<()>
where
    T: DataValue,
    G: AsRef<str>,
{
    let gene_axis = GeneAxisPlan::requested(dataset, gene_names, missing)?;
    access_cells_with_gene_axis(
        access,
        compute,
        access_config,
        dataset,
        cells,
        out,
        names,
        &gene_axis,
    )
}

pub fn access_cells_by_gene_names_owned<T, G>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    gene_names: &[G],
    missing: MissingGenePolicy,
) -> DataBankResult<Vec<T>>
where
    T: DataValue,
    G: AsRef<str>,
{
    let gene_axis = GeneAxisPlan::requested(dataset, gene_names, missing)?;
    validate_dtype_and_cells::<T>(dataset, cells)?;
    let total = cells
        .len()
        .checked_mul(gene_axis.output_genes(dataset.num_genes()))
        .ok_or_else(|| DataBankError::InvalidConfig("output length overflow".to_string()))?;
    let mut out = vec![T::zero(); total];
    access_cells_with_gene_axis(
        access,
        compute,
        access_config,
        dataset,
        cells,
        &mut out,
        None,
        &gene_axis,
    )?;
    Ok(out)
}

fn access_cells_with_gene_axis<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
    gene_axis: &GeneAxisPlan,
) -> DataBankResult<()> {
    validate_access(
        dataset,
        cells,
        out,
        names.as_deref(),
        gene_axis.output_genes(dataset.num_genes()),
    )?;
    match dataset {
        Dataset::Dense1D(dataset) => access_dense_1d_projected(
            access,
            compute,
            access_config,
            dataset,
            cells,
            gene_axis,
            out,
        )?,
        Dataset::Dense2D(dataset) => access_dense_2d_projected(
            access,
            compute,
            access_config,
            dataset,
            cells,
            gene_axis,
            out,
        )?,
        Dataset::SparseCsr(dataset) => access_sparse_projected(
            access,
            compute,
            access_config,
            dataset,
            cells,
            gene_axis,
            out,
        )?,
    }
    if let Some(names) = names {
        gene_axis.fill_names(dataset, names)?;
    }
    Ok(())
}

pub unsafe fn access_cells_unchecked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
) -> DataBankResult<()> {
    match dataset {
        Dataset::SparseCsr(dataset) => {
            // SAFETY: forwarded from DataBank::access_cells_unchecked. The caller
            // promises the CSR metadata, requested cells, output buffer, dtype,
            // and gene indices are valid for unchecked scatter.
            unsafe {
                access_sparse_unchecked(access, compute, access_config, dataset, cells, out)?;
            }
            if let Some(names) = names {
                let expected = dataset.num_genes;
                if names.len() != expected {
                    return Err(DataBankError::NameBufferSizeMismatch {
                        expected,
                        actual: names.len(),
                    });
                }
                names.copy_from_slice(dataset.genes.views());
            }
            Ok(())
        }
        _ => access_cells(access, compute, access_config, dataset, cells, out, names),
    }
}

pub fn prefetch_cells(
    access: &AccessHandle,
    dataset: &Dataset,
    cells: &[usize],
) -> DataBankResult<()> {
    for &cell in cells {
        if cell >= dataset.num_cells() {
            return Err(DataBankError::CellIndexOutOfRange {
                cell,
                num_cells: dataset.num_cells(),
            });
        }
    }

    let mut keys = FastHashSet::default();
    match dataset {
        Dataset::Dense1D(dataset) => {
            for segment in plan::plan_dense_1d(dataset, cells)? {
                collect_prefetch_key(&mut keys, &segment.chunk);
            }
        }
        Dataset::Dense2D(dataset) => {
            for segment in plan::plan_dense_2d(dataset, cells)? {
                collect_prefetch_key(&mut keys, &segment.chunk);
            }
        }
        Dataset::SparseCsr(dataset) => {
            for row in plan::plan_sparse(dataset, cells)? {
                collect_range_prefetch_keys(&mut keys, &row.indices);
                collect_range_prefetch_keys(&mut keys, &row.data);
            }
        }
    }
    prefetch_keys(access, keys)
}

fn access_dense_1d<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dense1DDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<()> {
    if try_access_dense_1d_memory_identity_direct(dataset, cells, out)? {
        return Ok(());
    }
    let segments = plan::plan_dense_1d(dataset, cells)?;
    access_dense_segments(
        access,
        compute,
        access_config,
        dataset.num_genes,
        &segments,
        out,
    )
}

fn access_dense_1d_projected<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dense1DDataset,
    cells: &[usize],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if gene_axis.requires_dense_zero_fill() {
        zero_values(out);
    }
    let segments = match gene_axis.projection() {
        Some(projection) => {
            plan::plan_dense_1d_selected_sources(dataset, cells, &projection.selected_sources)?
        }
        None => plan::plan_dense_1d(dataset, cells)?,
    };
    access_dense_segments_projected(
        access,
        compute,
        access_config,
        dataset.num_genes,
        gene_axis.output_genes(dataset.num_genes),
        &segments,
        gene_axis,
        out,
    )
}

fn access_dense_2d<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dense2DDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<()> {
    let segments = plan::plan_dense_2d(dataset, cells)?;
    access_dense_segments(
        access,
        compute,
        access_config,
        dataset.num_genes,
        &segments,
        out,
    )
}

fn access_dense_2d_projected<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dense2DDataset,
    cells: &[usize],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if gene_axis.requires_dense_zero_fill() {
        zero_values(out);
    }
    let segments = match gene_axis.projection() {
        Some(projection) => {
            plan::plan_dense_2d_selected_sources(dataset, cells, &projection.selected_sources)?
        }
        None => plan::plan_dense_2d(dataset, cells)?,
    };
    access_dense_segments_projected(
        access,
        compute,
        access_config,
        dataset.num_genes,
        gene_axis.output_genes(dataset.num_genes),
        &segments,
        gene_axis,
        out,
    )
}

fn access_dense_segments<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    num_genes: usize,
    segments: &[DenseSegment],
    out: &mut [T],
) -> DataBankResult<()> {
    let groups = group_dense_segments(segments)?;
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("dense output byte length overflow".to_string())
    })?;
    let output_rows = row_count_for_width(out.len(), num_genes);
    if compute.should_parallelize(output_rows, output_bytes) {
        let loaded_groups = load_dense_groups_for_parallel(access, access_config, &groups)?;
        if try_scatter_dense_rows_parallel(
            compute,
            num_genes,
            num_genes,
            segments,
            &groups,
            &loaded_groups,
            None,
            out,
        )? {
            return Ok(());
        }
    }
    if groups
        .iter()
        .all(|group| matches!(group.source, DenseGroupSource::AccessItem(_)))
    {
        access_dense_groups_scheduled(access, access_config, num_genes, segments, &groups, out)
    } else {
        access_dense_groups_sequential(access, num_genes, segments, &groups, out)
    }
}

fn access_dense_segments_projected<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if segments.is_empty() {
        return Ok(());
    }
    let groups = group_dense_segments(segments)?;
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("dense output byte length overflow".to_string())
    })?;
    let output_rows = row_count_for_width(out.len(), output_genes);
    if compute.should_parallelize(output_rows, output_bytes) {
        let loaded_groups = load_dense_groups_for_parallel(access, access_config, &groups)?;
        if try_scatter_dense_rows_parallel(
            compute,
            dataset_num_genes,
            output_genes,
            segments,
            &groups,
            &loaded_groups,
            gene_axis.projection().cloned(),
            out,
        )? {
            return Ok(());
        }
    }
    if groups
        .iter()
        .all(|group| matches!(group.source, DenseGroupSource::AccessItem(_)))
    {
        access_dense_groups_scheduled_projected(
            access,
            access_config,
            dataset_num_genes,
            output_genes,
            segments,
            &groups,
            gene_axis,
            out,
        )
    } else {
        access_dense_groups_sequential_projected(
            access,
            dataset_num_genes,
            output_genes,
            segments,
            &groups,
            gene_axis,
            out,
        )
    }
}

fn access_dense_groups_scheduled<T: DataValue>(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    num_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    out: &mut [T],
) -> DataBankResult<()> {
    if groups.is_empty() {
        return Ok(());
    }

    let mut scheduled = access.scheduled(
        groups.iter().map(file_dense_group_access_item),
        *access_config,
    )?;
    for group in groups {
        let bytes = scheduled.next().ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled dense access ended early",
            ))
        })??;
        scatter_dense_group(num_genes, segments, group, &bytes, out)?;
    }
    if scheduled.next().is_some() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled dense access returned extra output",
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn access_dense_groups_scheduled_projected<T: DataValue>(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    let mut scheduled = access.scheduled(
        groups.iter().map(file_dense_group_access_item),
        *access_config,
    )?;
    for group in groups {
        let bytes = scheduled.next().ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled dense access ended early",
            ))
        })??;
        scatter_dense_group_projected(
            dataset_num_genes,
            output_genes,
            segments,
            group,
            &bytes,
            gene_axis,
            out,
        )?;
    }
    if scheduled.next().is_some() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled dense access returned extra output",
        )));
    }
    Ok(())
}

fn access_dense_groups_sequential<T: DataValue>(
    access: &AccessHandle,
    num_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    out: &mut [T],
) -> DataBankResult<()> {
    for group in groups {
        if try_scatter_dense_memory_identity_group(num_genes, segments, group, out)? {
            continue;
        }
        let bytes = load_dense_group(access, group)?;
        scatter_dense_group(num_genes, segments, group, &bytes, out)?;
    }
    Ok(())
}

fn access_dense_groups_sequential_projected<T: DataValue>(
    access: &AccessHandle,
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    for group in groups {
        if try_scatter_dense_memory_identity_group_projected(
            dataset_num_genes,
            output_genes,
            segments,
            group,
            gene_axis,
            out,
        )? {
            continue;
        }
        let bytes = load_dense_group(access, group)?;
        scatter_dense_group_projected(
            dataset_num_genes,
            output_genes,
            segments,
            group,
            &bytes,
            gene_axis,
            out,
        )?;
    }
    Ok(())
}

fn try_access_dense_1d_memory_identity_direct<T: DataValue>(
    dataset: &Dense1DDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<bool> {
    let Some(chunks) = MemoryIdentity1DChunks::from_array(&dataset.data)? else {
        return Ok(false);
    };
    if mem::size_of::<T>() != T::DTYPE.item_size() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: T::DTYPE.item_size(),
            actual: mem::size_of::<T>(),
        });
    }

    let chunk_len = chunks.chunk_len;
    let value_size = mem::size_of::<T>();
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();

    for (output_row, &cell) in cells.iter().enumerate() {
        let row_start = cell.checked_mul(dataset.num_genes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("Dense1D row start overflow".to_string())
        })?;
        let row_end = row_start.checked_add(dataset.num_genes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("Dense1D row end overflow".to_string())
        })?;
        let mut pos = row_start;
        let mut output_col = 0usize;
        while pos < row_end {
            let chunk_index = pos / chunk_len;
            let in_chunk = pos % chunk_len;
            let chunk = chunks.chunk_bytes(chunk_index)?;
            let chunk_start = chunk_index.checked_mul(chunk_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense1D chunk start overflow".to_string())
            })?;
            let physical_chunk_len = chunks.physical_chunk_len_at_start(chunk_start)?;
            let elements = (row_end - pos).min(physical_chunk_len - in_chunk);
            let src_start = in_chunk.checked_mul(value_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense1D source byte start overflow".to_string())
            })?;
            let bytes = elements.checked_mul(value_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense1D source byte length overflow".to_string())
            })?;
            let src_end = src_start.checked_add(bytes).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense1D source byte end overflow".to_string())
            })?;
            let dst_start = output_row
                .checked_mul(dataset.num_genes)
                .and_then(|base| base.checked_add(output_col))
                .ok_or_else(|| {
                    DataBankError::InvalidConfig("Dense1D output offset overflow".to_string())
                })?;
            let dst_end = dst_start.checked_add(elements).ok_or_else(|| {
                DataBankError::InvalidConfig("Dense1D output end overflow".to_string())
            })?;
            if src_end > chunk.len() || dst_end > out_len {
                return Err(DataBankError::InvalidArrayMeta(
                    "Dense1D direct memory copy is out of range".to_string(),
                ));
            }
            let out_ptr = out_addr as *mut T;
            unsafe {
                ptr::copy_nonoverlapping(
                    chunk.as_ptr().add(src_start),
                    out_ptr.add(dst_start).cast::<u8>(),
                    bytes,
                );
            }
            pos += elements;
            output_col += elements;
        }
    }
    Ok(true)
}

fn access_sparse<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<()> {
    zero_values(out);
    if try_scatter_sparse_single_memory_chunk_checked(dataset, cells, out)? {
        return Ok(());
    }
    if try_scatter_sparse_memory_chunks_checked(dataset, cells, out)? {
        return Ok(());
    }
    let rows = plan::plan_sparse_rows(dataset, cells)?;
    let plan = plan_sparse_batch::<T>(dataset, &rows)?;
    if sparse_plan_is_file_backed(&plan) {
        access_sparse_file_groups_checked(access, compute, access_config, dataset, &plan, out)?;
    } else {
        let index_bytes = load_sparse_index_groups(access, access_config, &plan)?;
        load_sparse_data_groups_and_scatter_checked(
            access,
            compute,
            access_config,
            dataset,
            &plan,
            index_bytes,
            out,
        )?;
    }
    Ok(())
}

fn access_sparse_projected<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    cells: &[usize],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    zero_values(out);
    let rows = plan::plan_sparse_rows(dataset, cells)?;
    let plan = plan_sparse_batch::<T>(dataset, &rows)?;
    let index_bytes = load_sparse_index_groups(access, access_config, &plan)?;
    load_sparse_data_groups_and_scatter_projected_checked(
        access,
        compute,
        access_config,
        dataset,
        &plan,
        index_bytes,
        gene_axis,
        None,
        out,
    )?;
    Ok(())
}

unsafe fn access_sparse_unchecked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<()> {
    zero_values(out);
    if unsafe { try_scatter_sparse_single_memory_chunk_unchecked(dataset, cells, out) }? {
        return Ok(());
    }
    if unsafe { try_scatter_sparse_memory_chunks_unchecked(dataset, cells, out) }? {
        return Ok(());
    }
    // SAFETY: the caller guarantees requested cell indices and CSR indptr are
    // valid. Downstream scatter also assumes all gene indices are in range.
    let rows = unsafe { plan::plan_sparse_rows_unchecked(dataset, cells) };
    let plan = plan_sparse_batch::<T>(dataset, &rows)?;
    if sparse_plan_is_file_backed(&plan) {
        unsafe {
            access_sparse_file_groups_unchecked(
                access,
                compute,
                access_config,
                dataset,
                &plan,
                out,
            )?;
        }
    } else {
        let index_bytes = load_sparse_index_groups(access, access_config, &plan)?;
        unsafe {
            load_sparse_data_groups_and_scatter_unchecked(
                access,
                compute,
                access_config,
                dataset,
                &plan,
                index_bytes,
                out,
            )?;
        }
    }
    Ok(())
}

#[derive(Debug)]
struct SparseBatchPlan {
    index_pieces: Vec<SparseReadPiece>,
    data_pieces: Vec<SparseReadPiece>,
    index_groups: Vec<SparseReadGroup>,
    data_groups: Vec<SparseReadGroup>,
    index_bytes: usize,
}

#[derive(Debug, Clone)]
struct SparseReadPiece {
    chunk: ChunkRef,
    source: ByteRange,
    group_offset: usize,
    output_offset: usize,
    output_row: usize,
    index_offset: usize,
    elements: usize,
    bytes: usize,
}

#[derive(Debug, Clone)]
enum SparseGroupSource {
    AccessItem(AccessItem),
    Memory {
        bytes: Arc<[u8]>,
        codec: SharedCodec,
        expected_size: usize,
        decoded: bool,
    },
}

#[derive(Debug, Clone)]
struct SparseReadGroup {
    source: SparseGroupSource,
    slice: SliceSpec,
    slice_ranges: Vec<RangeCopy>,
    parts: Vec<usize>,
    bytes: usize,
}

impl SparseReadGroup {
    fn finalize_slice(&mut self) {
        self.slice = SliceSpec::from_ranges(std::mem::take(&mut self.slice_ranges));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SparseGroupKey {
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

fn plan_sparse_batch<T: DataValue>(
    dataset: &SparseCsrDataset,
    rows: &[SparseRowSpan],
) -> DataBankResult<SparseBatchPlan> {
    plan_sparse_batch_with_value_size(dataset, rows, T::DTYPE.item_size())
}

fn plan_sparse_batch_with_value_size(
    dataset: &SparseCsrDataset,
    rows: &[SparseRowSpan],
    value_size: usize,
) -> DataBankResult<SparseBatchPlan> {
    let index_size = dataset.index_dtype.item_size();
    let mut index_piece_capacity = 0usize;
    let mut data_piece_capacity = 0usize;
    for row in rows {
        if row.nnz == 0 {
            continue;
        }
        let end = row.start.checked_add(row.nnz).ok_or_else(|| {
            DataBankError::IndptrInvalid("CSR row range end overflows usize".to_string())
        })?;
        index_piece_capacity = index_piece_capacity.saturating_add(plan::range_piece_count(
            &dataset.indices,
            row.start,
            end,
        )?);
        data_piece_capacity = data_piece_capacity.saturating_add(plan::range_piece_count(
            &dataset.data,
            row.start,
            end,
        )?);
    }

    let mut index_builder = SparsePieceGroupBuilder::with_capacity(index_piece_capacity);
    let mut data_builder = SparsePieceGroupBuilder::with_capacity(data_piece_capacity);
    let mut index_bytes = 0usize;

    for row in rows {
        if row.nnz == 0 {
            continue;
        }
        let end = row.start.checked_add(row.nnz).ok_or_else(|| {
            DataBankError::IndptrInvalid("CSR row range end overflows usize".to_string())
        })?;
        let row_index_offset = index_bytes;
        push_sparse_index_pieces(
            &dataset.indices,
            row.start,
            end,
            index_size,
            &mut index_bytes,
            &mut index_builder,
        )?;
        push_sparse_data_pieces(
            &dataset.data,
            row.start,
            end,
            SparseDataPieceContext {
                value_size,
                index_size,
                output_row: row.output_row,
                row_index_offset,
            },
            &mut data_builder,
        )?;
    }

    let (index_pieces, index_groups) = index_builder.finish();
    let (data_pieces, data_groups) = data_builder.finish();
    Ok(SparseBatchPlan {
        index_pieces,
        data_pieces,
        index_groups,
        data_groups,
        index_bytes,
    })
}

fn push_sparse_index_pieces(
    array: &Array,
    start: usize,
    end: usize,
    item_size: usize,
    output_offset: &mut usize,
    builder: &mut SparsePieceGroupBuilder,
) -> DataBankResult<()> {
    push_sparse_range_pieces(array, start, end, item_size, |piece| {
        let offset = *output_offset;
        *output_offset = (*output_offset).checked_add(piece.bytes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index buffer size overflow".to_string())
        })?;
        builder.push(SparseReadPiece {
            chunk: piece.chunk,
            source: piece.source,
            group_offset: 0,
            output_offset: offset,
            output_row: 0,
            index_offset: 0,
            elements: piece.elements,
            bytes: piece.bytes,
        })?;
        Ok(())
    })
}

#[derive(Clone, Copy)]
struct SparseDataPieceContext {
    value_size: usize,
    index_size: usize,
    output_row: usize,
    row_index_offset: usize,
}

fn push_sparse_data_pieces(
    array: &Array,
    start: usize,
    end: usize,
    context: SparseDataPieceContext,
    builder: &mut SparsePieceGroupBuilder,
) -> DataBankResult<()> {
    let mut row_elements = 0usize;
    push_sparse_range_pieces(array, start, end, context.value_size, |piece| {
        let index_delta = row_elements
            .checked_mul(context.index_size)
            .ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR row index byte offset overflow".to_string())
            })?;
        let index_offset = context
            .row_index_offset
            .checked_add(index_delta)
            .ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR index byte offset overflow".to_string())
            })?;
        row_elements = row_elements.checked_add(piece.elements).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR row element cursor overflow".to_string())
        })?;
        builder.push(SparseReadPiece {
            chunk: piece.chunk,
            source: piece.source,
            group_offset: 0,
            output_offset: 0,
            output_row: context.output_row,
            index_offset,
            elements: piece.elements,
            bytes: piece.bytes,
        })?;
        Ok(())
    })
}

struct PlannedRangePiece {
    chunk: ChunkRef,
    source: ByteRange,
    elements: usize,
    bytes: usize,
}

fn push_sparse_range_pieces<F>(
    array: &Array,
    start: usize,
    end: usize,
    item_size: usize,
    mut push: F,
) -> DataBankResult<()>
where
    F: FnMut(PlannedRangePiece) -> DataBankResult<()>,
{
    if array.shape.len() != 1 {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "CSR range planning requires 1D array, got shape {:?}",
            array.shape
        )));
    }
    if start > end || end > array.shape[0] {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "invalid CSR range [{start}, {end}) for length {}",
            array.shape[0]
        )));
    }

    for range in plan::plan_1d_range(array, start, end)? {
        let bytes = range.source.len();
        let expected_bytes = range.elements.checked_mul(item_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR range byte length overflow".to_string())
        })?;
        if bytes != expected_bytes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "CSR planned range has {bytes} bytes, expected {expected_bytes}"
            )));
        }
        push(PlannedRangePiece {
            chunk: range.chunk,
            source: range.source,
            elements: range.elements,
            bytes,
        })?;
    }
    Ok(())
}

struct SparsePieceGroupBuilder {
    pieces: Vec<SparseReadPiece>,
    groups: Vec<SparseReadGroup>,
    by_key: FastHashMap<SparseGroupKey, usize>,
}

impl SparsePieceGroupBuilder {
    fn with_capacity(piece_capacity: usize) -> Self {
        Self {
            pieces: Vec::with_capacity(piece_capacity),
            groups: Vec::with_capacity(piece_capacity),
            by_key: fast_hash_map_with_capacity(piece_capacity),
        }
    }

    fn push(&mut self, mut piece: SparseReadPiece) -> DataBankResult<()> {
        let key = sparse_group_key(&piece.chunk);
        let group_index = if let Some(&group_index) = self.by_key.get(&key) {
            group_index
        } else {
            let group_index = self.groups.len();
            self.by_key.insert(key, group_index);
            self.groups.push(SparseReadGroup {
                source: sparse_group_source(&piece.chunk),
                slice: SliceSpec::Full,
                slice_ranges: Vec::new(),
                parts: Vec::new(),
                bytes: 0,
            });
            group_index
        };

        let piece_index = self.pieces.len();
        let group = &mut self.groups[group_index];
        piece.group_offset = append_sparse_group_slice(group, piece.source, piece.bytes)?;
        group.parts.push(piece_index);
        self.pieces.push(piece);
        Ok(())
    }

    fn finish(mut self) -> (Vec<SparseReadPiece>, Vec<SparseReadGroup>) {
        for group in &mut self.groups {
            group.finalize_slice();
        }
        (self.pieces, self.groups)
    }
}

fn sparse_group_key(chunk: &ChunkRef) -> SparseGroupKey {
    match chunk {
        ChunkRef::AccessItem(item) => SparseGroupKey::File {
            key: item.key,
            codec: codec_id(&item.codec),
            expected_size: item.expected_size,
        },
        ChunkRef::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => SparseGroupKey::Memory {
            ptr: bytes.as_ptr() as usize,
            len: bytes.len(),
            codec: codec_id(codec),
            expected_size: *expected_size,
            decoded: *decoded,
        },
    }
}

fn sparse_group_source(chunk: &ChunkRef) -> SparseGroupSource {
    match chunk {
        ChunkRef::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = SliceSpec::Full;
            SparseGroupSource::AccessItem(item)
        }
        ChunkRef::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => SparseGroupSource::Memory {
            bytes: Arc::clone(bytes),
            codec: Arc::clone(codec),
            expected_size: *expected_size,
            decoded: *decoded,
        },
    }
}

fn append_sparse_group_slice(
    group: &mut SparseReadGroup,
    source: ByteRange,
    bytes: usize,
) -> DataBankResult<usize> {
    if source.len() != bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "CSR source range length is {}, expected {bytes}",
            source.len()
        )));
    }
    let output_offset = group.bytes;
    group
        .slice_ranges
        .push(RangeCopy::new(output_offset, source.start, source.end));
    group.bytes = group.bytes.checked_add(bytes).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR grouped read byte length overflow".to_string())
    })?;
    Ok(output_offset)
}

fn sparse_group_access_item(group: &SparseReadGroup) -> DataBankResult<AccessItem> {
    match &group.source {
        SparseGroupSource::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = group.slice.clone();
            Ok(item)
        }
        SparseGroupSource::Memory { .. } => Err(DataBankError::InvalidArrayMeta(
            "memory chunk reached CSR scheduled path".to_string(),
        )),
    }
}

fn file_sparse_group_access_item(group: &SparseReadGroup) -> AccessItem {
    match &group.source {
        SparseGroupSource::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = group.slice.clone();
            item
        }
        SparseGroupSource::Memory { .. } => {
            unreachable!("memory chunk reached CSR file-backed scheduled path")
        }
    }
}

fn load_sparse_group(access: &AccessHandle, group: &SparseReadGroup) -> DataBankResult<Vec<u8>> {
    match &group.source {
        SparseGroupSource::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = group.slice.clone();
            submit_access_item(access, item)
        }
        SparseGroupSource::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => load_memory_group(
            bytes.as_ref(),
            codec,
            *expected_size,
            *decoded,
            &group.slice,
        ),
    }
}

type SparseGroupScheduledAccess = ScheduledAccess<std::vec::IntoIter<AccessItem>>;

fn schedule_sparse_selected_file_groups(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    plan: &SparseBatchPlan,
    selected_groups: &[usize],
    cancel: Option<&Arc<PrefetchCancel>>,
) -> DataBankResult<Option<SparseGroupScheduledAccess>> {
    if cancel.is_some_and(|cancel| cancel.is_cancelled()) {
        return Err(DataBankError::PrefetchCancelled);
    }

    let mut items = Vec::with_capacity(selected_groups.len());
    for &group_index in selected_groups {
        let group = &plan.data_groups[group_index];
        if matches!(group.source, SparseGroupSource::AccessItem(_)) {
            items.push(file_sparse_group_access_item(group));
        }
    }
    if items.is_empty() {
        return Ok(None);
    }
    let mut scheduled = access.scheduled(items, *access_config)?;
    if let Some(cancel) = cancel {
        scheduled.set_cancel_handle(Arc::clone(cancel));
    }
    Ok(Some(scheduled))
}

fn next_scheduled_sparse_group_bytes(
    scheduled: &mut SparseGroupScheduledAccess,
) -> DataBankResult<Vec<u8>> {
    scheduled
        .next()
        .ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled CSR data access ended early",
            ))
        })?
        .map_err(DataBankError::Io)
}

fn finish_scheduled_sparse_group_bytes(
    scheduled: Option<SparseGroupScheduledAccess>,
) -> DataBankResult<()> {
    let Some(mut scheduled) = scheduled else {
        return Ok(());
    };
    match scheduled.next() {
        None => Ok(()),
        Some(Ok(_)) => Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled CSR data access returned extra output",
        ))),
        Some(Err(err)) => Err(DataBankError::Io(err)),
    }
}

fn load_sparse_index_groups(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    plan: &SparseBatchPlan,
) -> DataBankResult<Vec<u8>> {
    let mut out = zeroed_byte_vec(plan.index_bytes);
    if plan.index_groups.is_empty() {
        return Ok(out);
    }
    if plan
        .index_groups
        .iter()
        .all(|group| matches!(group.source, SparseGroupSource::AccessItem(_)))
    {
        load_sparse_index_groups_scheduled(access, access_config, plan, &mut out)?;
    } else {
        load_sparse_index_groups_sequential(access, plan, &mut out)?;
    }
    Ok(out)
}

fn sparse_plan_is_file_backed(plan: &SparseBatchPlan) -> bool {
    plan.index_groups
        .iter()
        .chain(plan.data_groups.iter())
        .all(|group| matches!(group.source, SparseGroupSource::AccessItem(_)))
}

fn access_sparse_file_groups_checked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if plan.index_groups.is_empty() && plan.data_groups.is_empty() {
        return Ok(());
    }
    let mut index_bytes = zeroed_byte_vec(plan.index_bytes);
    let mut scheduled = access.scheduled(
        plan.index_groups
            .iter()
            .chain(plan.data_groups.iter())
            .map(file_sparse_group_access_item),
        *access_config,
    )?;

    for group in &plan.index_groups {
        let bytes = scheduled.next().ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled CSR combined access ended during indices",
            ))
        })??;
        copy_sparse_group_to_index_buffer(&plan.index_pieces, group, &bytes, &mut index_bytes)?;
    }

    let output_rows = row_count_for_width(out.len(), dataset.num_genes);
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if compute.should_parallelize(output_rows, output_bytes) {
        let mut data_group_bytes = Vec::with_capacity(plan.data_groups.len());
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR combined access ended during data",
                ))
            })??;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            data_group_bytes.push(Arc::from(bytes.into_boxed_slice()));
        }
        try_scatter_sparse_rows_parallel_checked(
            compute,
            dataset,
            plan,
            Arc::from(index_bytes.into_boxed_slice()),
            data_group_bytes,
            dataset.num_genes,
            None,
            out,
        )?;
    } else {
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR combined access ended during data",
                ))
            })??;
            scatter_sparse_data_group_checked(
                dataset,
                &plan.data_pieces,
                group,
                &index_bytes,
                &bytes,
                out,
            )?;
        }
    }

    if scheduled.next().is_some() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled CSR combined access returned extra output",
        )));
    }
    Ok(())
}

unsafe fn access_sparse_file_groups_unchecked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if plan.index_groups.is_empty() && plan.data_groups.is_empty() {
        return Ok(());
    }
    let mut index_bytes = zeroed_byte_vec(plan.index_bytes);
    let mut scheduled = access.scheduled(
        plan.index_groups
            .iter()
            .chain(plan.data_groups.iter())
            .map(file_sparse_group_access_item),
        *access_config,
    )?;

    for group in &plan.index_groups {
        let bytes = scheduled.next().ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled CSR combined access ended during indices",
            ))
        })??;
        copy_sparse_group_to_index_buffer(&plan.index_pieces, group, &bytes, &mut index_bytes)?;
    }

    let output_rows = row_count_for_width(out.len(), dataset.num_genes);
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if compute.should_parallelize(output_rows, output_bytes) {
        let mut data_group_bytes = Vec::with_capacity(plan.data_groups.len());
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR combined access ended during data",
                ))
            })??;
            debug_assert_eq!(bytes.len(), group.bytes);
            data_group_bytes.push(Arc::from(bytes.into_boxed_slice()));
        }
        unsafe {
            try_scatter_sparse_rows_parallel_unchecked(
                compute,
                dataset,
                plan,
                Arc::from(index_bytes.into_boxed_slice()),
                data_group_bytes,
                out,
            )?;
        }
    } else {
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR combined access ended during data",
                ))
            })??;
            unsafe {
                scatter_sparse_data_group_unchecked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    &bytes,
                    out,
                );
            }
        }
    }

    if scheduled.next().is_some() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled CSR combined access returned extra output",
        )));
    }
    Ok(())
}

fn load_sparse_index_groups_scheduled(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    plan: &SparseBatchPlan,
    out: &mut [u8],
) -> DataBankResult<()> {
    let mut scheduled = access.scheduled(
        plan.index_groups.iter().map(file_sparse_group_access_item),
        *access_config,
    )?;
    for group in &plan.index_groups {
        let bytes = scheduled.next().ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled CSR index access ended early",
            ))
        })??;
        copy_sparse_group_to_index_buffer(&plan.index_pieces, group, &bytes, out)?;
    }
    if scheduled.next().is_some() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled CSR index access returned extra output",
        )));
    }
    Ok(())
}

fn load_sparse_index_groups_sequential(
    access: &AccessHandle,
    plan: &SparseBatchPlan,
    out: &mut [u8],
) -> DataBankResult<()> {
    for group in &plan.index_groups {
        if try_copy_sparse_memory_identity_group_to_index_buffer(&plan.index_pieces, group, out)? {
            continue;
        }
        let bytes = load_sparse_group(access, group)?;
        copy_sparse_group_to_index_buffer(&plan.index_pieces, group, &bytes, out)?;
    }
    Ok(())
}

fn copy_sparse_group_to_index_buffer(
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    bytes: &[u8],
    out: &mut [u8],
) -> DataBankResult<()> {
    if bytes.len() != group.bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "CSR group decoded length is {}, expected {}",
            bytes.len(),
            group.bytes
        )));
    }

    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR group piece index is invalid".to_string())
        })?;
        let src_end = piece.group_offset.checked_add(piece.bytes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR group byte offset overflow".to_string())
        })?;
        let dst_end = piece
            .output_offset
            .checked_add(piece.bytes)
            .ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR index output offset overflow".to_string())
            })?;
        if src_end > bytes.len() || dst_end > out.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR index grouped copy is out of range".to_string(),
            ));
        }
        out[piece.output_offset..dst_end].copy_from_slice(&bytes[piece.group_offset..src_end]);
    }
    Ok(())
}

fn try_copy_sparse_memory_identity_group_to_index_buffer(
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    out: &mut [u8],
) -> DataBankResult<bool> {
    let SparseGroupSource::Memory {
        bytes,
        codec,
        expected_size,
        decoded,
    } = &group.source
    else {
        return Ok(false);
    };
    if !(*decoded || codec.is_identity()) {
        return Ok(false);
    }
    if bytes.len() != *expected_size {
        return Err(CodecError::SizeMismatch {
            codec: codec.name().to_string(),
            expected: *expected_size,
            actual: bytes.len(),
        }
        .into());
    }

    copy_sparse_group_from_decoded_source_to_index_buffer(pieces, group, bytes.as_ref(), out)?;
    Ok(true)
}

fn copy_sparse_group_from_decoded_source_to_index_buffer(
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    decoded: &[u8],
    out: &mut [u8],
) -> DataBankResult<()> {
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR group piece index is invalid".to_string())
        })?;
        if piece.source.len() != piece.bytes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "CSR source range length is {}, expected {}",
                piece.source.len(),
                piece.bytes
            )));
        }
        let dst_end = piece
            .output_offset
            .checked_add(piece.bytes)
            .ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR index output offset overflow".to_string())
            })?;
        if piece.source.end > decoded.len() || dst_end > out.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR index direct copy is out of range".to_string(),
            ));
        }
        out[piece.output_offset..dst_end]
            .copy_from_slice(&decoded[piece.source.start..piece.source.end]);
    }
    Ok(())
}

fn load_sparse_data_groups_and_scatter_checked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Vec<u8>,
    out: &mut [T],
) -> DataBankResult<()> {
    if plan.data_groups.is_empty() {
        return Ok(());
    }
    // Box once; both scatter paths borrow from this shared arc.
    let index_bytes: Arc<[u8]> = Arc::from(index_bytes.into_boxed_slice());
    let output_rows = row_count_for_width(out.len(), dataset.num_genes);
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if compute.should_parallelize(output_rows, output_bytes) {
        let data_group_bytes = load_sparse_data_group_bytes(access, access_config, plan)?;
        if try_scatter_sparse_rows_parallel_checked(
            compute,
            dataset,
            plan,
            Arc::clone(&index_bytes),
            data_group_bytes,
            dataset.num_genes,
            None,
            out,
        )? {
            return Ok(());
        }
    }
    if plan
        .data_groups
        .iter()
        .all(|group| matches!(group.source, SparseGroupSource::AccessItem(_)))
    {
        let mut scheduled = access.scheduled(
            plan.data_groups.iter().map(file_sparse_group_access_item),
            *access_config,
        )?;
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR data access ended early",
                ))
            })??;
            scatter_sparse_data_group_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                &bytes,
                out,
            )?;
        }
        if scheduled.next().is_some() {
            return Err(DataBankError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "scheduled CSR data access returned extra output",
            )));
        }
    } else {
        for group in &plan.data_groups {
            if try_scatter_sparse_memory_identity_data_group_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                out,
            )? {
                continue;
            }
            let bytes = load_sparse_group(access, group)?;
            scatter_sparse_data_group_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                &bytes,
                out,
            )?;
        }
    }
    Ok(())
}

fn try_scatter_sparse_single_memory_chunk_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<bool> {
    let Some(index_bytes) = single_memory_identity_chunk_bytes(&dataset.indices)? else {
        return Ok(false);
    };
    let Some(data_bytes) = single_memory_identity_chunk_bytes(&dataset.data)? else {
        return Ok(false);
    };

    match dataset.index_dtype {
        DType::U32 => scatter_sparse_single_memory_chunk_checked_typed::<T, u32>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_bytes,
            data_bytes,
            out,
        )?,
        DType::I32 => scatter_sparse_single_memory_chunk_checked_typed::<T, i32>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_bytes,
            data_bytes,
            out,
        )?,
        DType::U64 => scatter_sparse_single_memory_chunk_checked_typed::<T, u64>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_bytes,
            data_bytes,
            out,
        )?,
        DType::I64 => scatter_sparse_single_memory_chunk_checked_typed::<T, i64>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_bytes,
            data_bytes,
            out,
        )?,
        dtype => {
            return Err(DataBankError::UnsupportedDType {
                dtype,
                context: "CSR indices",
            });
        }
    }
    Ok(true)
}

unsafe fn try_scatter_sparse_single_memory_chunk_unchecked<T: DataValue>(
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<bool> {
    let Some(index_bytes) = single_memory_identity_chunk_bytes(&dataset.indices)? else {
        return Ok(false);
    };
    let Some(data_bytes) = single_memory_identity_chunk_bytes(&dataset.data)? else {
        return Ok(false);
    };

    match dataset.index_dtype {
        DType::U32 => unsafe {
            scatter_sparse_single_memory_chunk_unchecked_typed::<T, u32>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_bytes,
                data_bytes,
                out,
            );
        },
        DType::I32 => unsafe {
            scatter_sparse_single_memory_chunk_unchecked_typed::<T, i32>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_bytes,
                data_bytes,
                out,
            );
        },
        DType::U64 => unsafe {
            scatter_sparse_single_memory_chunk_unchecked_typed::<T, u64>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_bytes,
                data_bytes,
                out,
            );
        },
        DType::I64 => unsafe {
            scatter_sparse_single_memory_chunk_unchecked_typed::<T, i64>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_bytes,
                data_bytes,
                out,
            );
        },
        _ => unreachable!("CSR index dtype was validated during registration"),
    }
    Ok(true)
}

fn single_memory_identity_chunk_bytes(array: &Array) -> DataBankResult<Option<&[u8]>> {
    let Some(chunks) = array.memory_chunks() else {
        return Ok(None);
    };
    let [chunk] = chunks else {
        return Ok(None);
    };
    let ChunkSource::Memory { bytes, decoded } = &chunk.source else {
        return Ok(None);
    };
    if !(*decoded || array.codec.is_identity()) {
        return Ok(None);
    }
    let expected_size = chunk.decoded_bytes;
    if bytes.len() != expected_size {
        return Err(CodecError::SizeMismatch {
            codec: array.codec.name().to_string(),
            expected: expected_size,
            actual: bytes.len(),
        }
        .into());
    }
    Ok(Some(bytes.as_ref()))
}

#[derive(Clone, Copy)]
struct MemoryIdentity1DChunks<'a> {
    array: &'a Array,
    chunks: &'a [Chunk],
    chunk_len: usize,
    len: usize,
    item_size: usize,
}

impl<'a> MemoryIdentity1DChunks<'a> {
    fn from_array(array: &'a Array) -> DataBankResult<Option<Self>> {
        let Some(chunks) = array.memory_chunks() else {
            return Ok(None);
        };
        let chunks_are_decoded = chunks
            .iter()
            .all(|chunk| matches!(&chunk.source, ChunkSource::Memory { decoded: true, .. }));
        if !(array.codec.is_identity() || chunks_are_decoded) {
            return Ok(None);
        }
        let [len] = array.shape.as_slice() else {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "1D direct memory path requires 1D array, got shape {:?}",
                array.shape
            )));
        };
        let Some(chunk_len) = array.regular_1d_chunk_len() else {
            return Ok(None);
        };
        Ok(Some(Self {
            array,
            chunks,
            chunk_len,
            len: *len,
            item_size: array.dtype.item_size(),
        }))
    }

    fn chunk_bytes(self, chunk_index: usize) -> DataBankResult<&'a [u8]> {
        let Some(chunk) = self.chunks.get(chunk_index) else {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "1D memory chunk index {chunk_index} is out of range"
            )));
        };
        let ChunkSource::Memory { bytes, .. } = &chunk.source else {
            return Err(DataBankError::InvalidArrayMeta(
                "non-memory chunk reached memory direct path".to_string(),
            ));
        };
        let expected_size = self.expected_chunk_bytes(chunk_index)?;
        if bytes.len() != expected_size {
            return Err(CodecError::SizeMismatch {
                codec: self.array.codec.name().to_string(),
                expected: expected_size,
                actual: bytes.len(),
            }
            .into());
        }
        Ok(bytes.as_ref())
    }

    fn expected_chunk_bytes(self, chunk_index: usize) -> DataBankResult<usize> {
        let chunk_start = chunk_index.checked_mul(self.chunk_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("1D memory chunk start overflow".to_string())
        })?;
        if chunk_start >= self.len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "1D memory chunk {chunk_index} starts past array length {}",
                self.len
            )));
        }
        self.chunks
            .get(chunk_index)
            .map(|chunk| chunk.decoded_bytes)
            .ok_or_else(|| {
                DataBankError::InvalidArrayMeta(format!(
                    "1D memory chunk index {chunk_index} is out of range"
                ))
            })
    }

    fn physical_chunk_len_at_start(self, chunk_start: usize) -> DataBankResult<usize> {
        if chunk_start >= self.len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "1D memory chunk start {chunk_start} is out of range for length {}",
                self.len
            )));
        }
        Ok(self.chunk_len.min(self.len - chunk_start))
    }
}

fn try_scatter_sparse_memory_chunks_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<bool> {
    let Some(index_chunks) = MemoryIdentity1DChunks::from_array(&dataset.indices)? else {
        return Ok(false);
    };
    let Some(data_chunks) = MemoryIdentity1DChunks::from_array(&dataset.data)? else {
        return Ok(false);
    };

    match dataset.index_dtype {
        DType::U32 => scatter_sparse_memory_chunks_checked_typed::<T, u32>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_chunks,
            data_chunks,
            out,
        )?,
        DType::I32 => scatter_sparse_memory_chunks_checked_typed::<T, i32>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_chunks,
            data_chunks,
            out,
        )?,
        DType::U64 => scatter_sparse_memory_chunks_checked_typed::<T, u64>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_chunks,
            data_chunks,
            out,
        )?,
        DType::I64 => scatter_sparse_memory_chunks_checked_typed::<T, i64>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_chunks,
            data_chunks,
            out,
        )?,
        dtype => {
            return Err(DataBankError::UnsupportedDType {
                dtype,
                context: "CSR indices",
            });
        }
    }
    Ok(true)
}

unsafe fn try_scatter_sparse_memory_chunks_unchecked<T: DataValue>(
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<bool> {
    let Some(index_chunks) = MemoryIdentity1DChunks::from_array(&dataset.indices)? else {
        return Ok(false);
    };
    let Some(data_chunks) = MemoryIdentity1DChunks::from_array(&dataset.data)? else {
        return Ok(false);
    };

    match dataset.index_dtype {
        DType::U32 => unsafe {
            scatter_sparse_memory_chunks_unchecked_typed::<T, u32>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_chunks,
                data_chunks,
                out,
            )?;
        },
        DType::I32 => unsafe {
            scatter_sparse_memory_chunks_unchecked_typed::<T, i32>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_chunks,
                data_chunks,
                out,
            )?;
        },
        DType::U64 => unsafe {
            scatter_sparse_memory_chunks_unchecked_typed::<T, u64>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_chunks,
                data_chunks,
                out,
            )?;
        },
        DType::I64 => unsafe {
            scatter_sparse_memory_chunks_unchecked_typed::<T, i64>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_chunks,
                data_chunks,
                out,
            )?;
        },
        _ => unreachable!("CSR index dtype was validated during registration"),
    }
    Ok(true)
}

fn scatter_sparse_memory_chunks_checked_typed<T, I>(
    num_genes: usize,
    indptr: &[u64],
    cells: &[usize],
    index_chunks: MemoryIdentity1DChunks<'_>,
    data_chunks: MemoryIdentity1DChunks<'_>,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    if mem::size_of::<I>() != index_chunks.item_size || mem::size_of::<T>() != data_chunks.item_size
    {
        return Err(DataBankError::InvalidArrayMeta(
            "CSR direct memory dtype size mismatch".to_string(),
        ));
    }

    for (output_row, &cell) in cells.iter().enumerate() {
        let start = usize::try_from(indptr[cell]).map_err(|_| {
            DataBankError::IndptrInvalid("CSR row start does not fit in usize".to_string())
        })?;
        let end = usize::try_from(indptr[cell + 1]).map_err(|_| {
            DataBankError::IndptrInvalid("CSR row end does not fit in usize".to_string())
        })?;
        if end < start || end > index_chunks.len || end > data_chunks.len {
            return Err(DataBankError::IndptrInvalid(
                "CSR row range is invalid for memory chunks".to_string(),
            ));
        }
        scatter_sparse_memory_chunked_row_checked_typed::<T, I>(
            num_genes,
            output_row,
            start,
            end,
            index_chunks,
            data_chunks,
            out,
        )?;
    }
    Ok(())
}

unsafe fn scatter_sparse_memory_chunks_unchecked_typed<T, I>(
    num_genes: usize,
    indptr: &[u64],
    cells: &[usize],
    index_chunks: MemoryIdentity1DChunks<'_>,
    data_chunks: MemoryIdentity1DChunks<'_>,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    debug_assert_eq!(mem::size_of::<I>(), index_chunks.item_size);
    debug_assert_eq!(mem::size_of::<T>(), data_chunks.item_size);

    for (output_row, &cell) in cells.iter().enumerate() {
        let start = unsafe { *indptr.get_unchecked(cell) as usize };
        let end = unsafe { *indptr.get_unchecked(cell + 1) as usize };
        unsafe {
            scatter_sparse_memory_chunked_row_unchecked_typed::<T, I>(
                num_genes,
                output_row,
                start,
                end,
                index_chunks,
                data_chunks,
                out,
            )?;
        }
    }
    Ok(())
}

fn scatter_sparse_memory_chunked_row_checked_typed<T, I>(
    num_genes: usize,
    output_row: usize,
    start: usize,
    end: usize,
    index_chunks: MemoryIdentity1DChunks<'_>,
    data_chunks: MemoryIdentity1DChunks<'_>,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    let value_size = mem::size_of::<T>();
    let mut pos = start;
    while pos < end {
        let (index_bytes, index_byte_start, index_rem) =
            sparse_memory_chunk_piece(index_chunks, pos, index_size)?;
        let (data_bytes, data_byte_start, data_rem) =
            sparse_memory_chunk_piece(data_chunks, pos, value_size)?;
        let elements = (end - pos).min(index_rem).min(data_rem);
        let index_bytes_len = elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR direct index byte length overflow".to_string())
        })?;
        let data_bytes_len = elements.checked_mul(value_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR direct data byte length overflow".to_string())
        })?;
        let index_byte_end = index_byte_start
            .checked_add(index_bytes_len)
            .ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR direct index byte end overflow".to_string())
            })?;
        let data_byte_end = data_byte_start.checked_add(data_bytes_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR direct data byte end overflow".to_string())
        })?;
        scatter_sparse_values_checked_typed::<T, I>(
            num_genes,
            output_row,
            elements,
            &index_bytes[index_byte_start..index_byte_end],
            &data_bytes[data_byte_start..data_byte_end],
            out,
        )?;
        pos += elements;
    }
    Ok(())
}

unsafe fn scatter_sparse_memory_chunked_row_unchecked_typed<T, I>(
    num_genes: usize,
    output_row: usize,
    start: usize,
    end: usize,
    index_chunks: MemoryIdentity1DChunks<'_>,
    data_chunks: MemoryIdentity1DChunks<'_>,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    let value_size = mem::size_of::<T>();
    let mut pos = start;
    while pos < end {
        let (index_bytes, index_byte_start, index_rem) =
            sparse_memory_chunk_piece(index_chunks, pos, index_size)?;
        let (data_bytes, data_byte_start, data_rem) =
            sparse_memory_chunk_piece(data_chunks, pos, value_size)?;
        let elements = (end - pos).min(index_rem).min(data_rem);
        let index_byte_end = index_byte_start + elements * index_size;
        let data_byte_end = data_byte_start + elements * value_size;
        unsafe {
            scatter_sparse_values_unchecked_typed::<T, I>(
                num_genes,
                output_row,
                elements,
                index_bytes.get_unchecked(index_byte_start..index_byte_end),
                data_bytes.get_unchecked(data_byte_start..data_byte_end),
                out,
            );
        }
        pos += elements;
    }
    Ok(())
}

fn sparse_memory_chunk_piece(
    chunks: MemoryIdentity1DChunks<'_>,
    pos: usize,
    item_size: usize,
) -> DataBankResult<(&[u8], usize, usize)> {
    let chunk_index = pos / chunks.chunk_len;
    let chunk_start = chunk_index.checked_mul(chunks.chunk_len).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR direct chunk start overflow".to_string())
    })?;
    let in_chunk = pos - chunk_start;
    let physical_chunk_len = chunks.physical_chunk_len_at_start(chunk_start)?;
    let remaining = physical_chunk_len.checked_sub(in_chunk).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR direct chunk cursor is out of range".to_string())
    })?;
    let byte_start = in_chunk.checked_mul(item_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR direct byte start overflow".to_string())
    })?;
    Ok((chunks.chunk_bytes(chunk_index)?, byte_start, remaining))
}

fn scatter_sparse_single_memory_chunk_checked_typed<T, I>(
    num_genes: usize,
    indptr: &[u64],
    cells: &[usize],
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    let value_size = mem::size_of::<T>();
    for (output_row, &cell) in cells.iter().enumerate() {
        let start = usize::try_from(indptr[cell]).map_err(|_| {
            DataBankError::IndptrInvalid("CSR row start does not fit in usize".to_string())
        })?;
        let end = usize::try_from(indptr[cell + 1]).map_err(|_| {
            DataBankError::IndptrInvalid("CSR row end does not fit in usize".to_string())
        })?;
        let elements = end
            .checked_sub(start)
            .ok_or_else(|| DataBankError::IndptrInvalid("CSR row range is invalid".to_string()))?;
        let index_start = start.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index byte offset overflow".to_string())
        })?;
        let index_end = end.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index byte end overflow".to_string())
        })?;
        let data_start = start.checked_mul(value_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data byte offset overflow".to_string())
        })?;
        let data_end = end.checked_mul(value_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data byte end overflow".to_string())
        })?;
        if index_end > index_bytes.len() || data_end > data_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR single-chunk scatter is out of range".to_string(),
            ));
        }
        scatter_sparse_values_checked_typed::<T, I>(
            num_genes,
            output_row,
            elements,
            &index_bytes[index_start..index_end],
            &data_bytes[data_start..data_end],
            out,
        )?;
    }
    Ok(())
}

unsafe fn scatter_sparse_single_memory_chunk_unchecked_typed<T, I>(
    num_genes: usize,
    indptr: &[u64],
    cells: &[usize],
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    let value_size = mem::size_of::<T>();
    for (output_row, &cell) in cells.iter().enumerate() {
        let start = unsafe { *indptr.get_unchecked(cell) as usize };
        let end = unsafe { *indptr.get_unchecked(cell + 1) as usize };
        let index_start = start * index_size;
        let index_end = end * index_size;
        let data_start = start * value_size;
        let data_end = end * value_size;
        unsafe {
            scatter_sparse_values_unchecked_typed::<T, I>(
                num_genes,
                output_row,
                end - start,
                index_bytes.get_unchecked(index_start..index_end),
                data_bytes.get_unchecked(data_start..data_end),
                out,
            );
        }
    }
}

fn load_sparse_data_group_bytes(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    plan: &SparseBatchPlan,
) -> DataBankResult<Vec<Arc<[u8]>>> {
    if plan
        .data_groups
        .iter()
        .all(|group| matches!(group.source, SparseGroupSource::AccessItem(_)))
    {
        let mut scheduled = access.scheduled(
            plan.data_groups.iter().map(file_sparse_group_access_item),
            *access_config,
        )?;
        let mut out = Vec::with_capacity(plan.data_groups.len());
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR data access ended early",
                ))
            })??;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            out.push(Arc::from(bytes.into_boxed_slice()));
        }
        if scheduled.next().is_some() {
            return Err(DataBankError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "scheduled CSR data access returned extra output",
            )));
        }
        Ok(out)
    } else {
        let mut out = Vec::with_capacity(plan.data_groups.len());
        for group in &plan.data_groups {
            let bytes = load_sparse_group(access, group)?;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            out.push(Arc::from(bytes.into_boxed_slice()));
        }
        Ok(out)
    }
}

fn load_sparse_selected_data_group_bytes(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    plan: &SparseBatchPlan,
    selected_groups: &[usize],
    cancel: Option<&Arc<PrefetchCancel>>,
) -> DataBankResult<Vec<Arc<[u8]>>> {
    if selected_groups.iter().all(|&group_index| {
        matches!(
            plan.data_groups[group_index].source,
            SparseGroupSource::AccessItem(_)
        )
    }) {
        let mut scheduled = schedule_sparse_selected_file_groups(
            access,
            access_config,
            plan,
            selected_groups,
            cancel,
        )?
        .expect("file-backed selected groups should create a scheduled reader");
        let mut out = Vec::with_capacity(selected_groups.len());
        for &group_index in selected_groups {
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
            }
            let group = &plan.data_groups[group_index];
            let bytes = next_scheduled_sparse_group_bytes(&mut scheduled)?;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            out.push(Arc::from(bytes.into_boxed_slice()));
        }
        finish_scheduled_sparse_group_bytes(Some(scheduled))?;
        Ok(out)
    } else {
        let mut scheduled = schedule_sparse_selected_file_groups(
            access,
            access_config,
            plan,
            selected_groups,
            cancel,
        )?;
        let mut out = Vec::with_capacity(selected_groups.len());
        for &group_index in selected_groups {
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
            }
            let group = &plan.data_groups[group_index];
            let bytes = match &group.source {
                SparseGroupSource::AccessItem(_) => {
                    let scheduled = scheduled.as_mut().ok_or_else(|| {
                        DataBankError::InvalidArrayMeta(
                            "CSR file group missing scheduled reader".to_string(),
                        )
                    })?;
                    next_scheduled_sparse_group_bytes(scheduled)?
                }
                SparseGroupSource::Memory { .. } => load_sparse_group(access, group)?,
            };
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            out.push(Arc::from(bytes.into_boxed_slice()));
        }
        finish_scheduled_sparse_group_bytes(scheduled)?;
        Ok(out)
    }
}

fn load_sparse_data_groups_and_scatter_projected_checked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Vec<u8>,
    gene_axis: &GeneAxisPlan,
    cancel: Option<&Arc<PrefetchCancel>>,
    out: &mut [T],
) -> DataBankResult<()> {
    if plan.data_groups.is_empty() {
        return Ok(());
    }

    // Box the index buffer once; both the parallel and serial scatter paths
    // borrow from this shared `Arc<[u8]>` instead of cloning the whole buffer.
    let index_bytes: Arc<[u8]> = Arc::from(index_bytes.into_boxed_slice());

    let mut selected_groups = Vec::with_capacity(plan.data_groups.len());
    for (group_index, group) in plan.data_groups.iter().enumerate() {
        if sparse_data_group_has_selected_values(
            dataset,
            &plan.data_pieces,
            group,
            index_bytes.as_ref(),
            gene_axis,
        )? {
            selected_groups.push(group_index);
        }
    }
    if selected_groups.is_empty() {
        return Ok(());
    }

    let output_genes = gene_axis.output_genes(dataset.num_genes);
    let output_rows = row_count_for_width(out.len(), output_genes);
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if compute.should_parallelize(output_rows, output_bytes) {
        let data_group_bytes = load_sparse_selected_data_group_bytes(
            access,
            access_config,
            plan,
            &selected_groups,
            cancel,
        )?;
        if try_scatter_sparse_rows_parallel_checked_with_group_indices(
            compute,
            dataset,
            plan,
            Arc::clone(&index_bytes),
            data_group_bytes,
            output_genes,
            gene_axis.projection().cloned(),
            Some(selected_groups.clone()),
            out,
        )? {
            return Ok(());
        }
    }

    if selected_groups.iter().all(|&group_index| {
        matches!(
            plan.data_groups[group_index].source,
            SparseGroupSource::AccessItem(_)
        )
    }) {
        let mut scheduled = schedule_sparse_selected_file_groups(
            access,
            access_config,
            plan,
            &selected_groups,
            cancel,
        )?
        .expect("file-backed selected groups should create a scheduled reader");
        for &group_index in &selected_groups {
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
            }
            let group = &plan.data_groups[group_index];
            let bytes = next_scheduled_sparse_group_bytes(&mut scheduled)?;
            scatter_sparse_data_group_projected_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                &bytes,
                gene_axis,
                out,
            )?;
        }
        finish_scheduled_sparse_group_bytes(Some(scheduled))?;
    } else {
        let mut scheduled = schedule_sparse_selected_file_groups(
            access,
            access_config,
            plan,
            &selected_groups,
            cancel,
        )?;
        for &group_index in &selected_groups {
            let group = &plan.data_groups[group_index];
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
            }
            if try_scatter_sparse_memory_identity_data_group_projected_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                gene_axis,
                out,
            )? {
                continue;
            }
            let bytes = match &group.source {
                SparseGroupSource::AccessItem(_) => {
                    let scheduled = scheduled.as_mut().ok_or_else(|| {
                        DataBankError::InvalidArrayMeta(
                            "CSR file group missing scheduled reader".to_string(),
                        )
                    })?;
                    next_scheduled_sparse_group_bytes(scheduled)?
                }
                SparseGroupSource::Memory { .. } => load_sparse_group(access, group)?,
            };
            scatter_sparse_data_group_projected_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                &bytes,
                gene_axis,
                out,
            )?;
        }
        finish_scheduled_sparse_group_bytes(scheduled)?;
    }
    Ok(())
}

unsafe fn load_sparse_data_groups_and_scatter_unchecked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Vec<u8>,
    out: &mut [T],
) -> DataBankResult<()> {
    if plan.data_groups.is_empty() {
        return Ok(());
    }
    // Box once; both scatter paths borrow from this shared arc.
    let index_bytes: Arc<[u8]> = Arc::from(index_bytes.into_boxed_slice());
    let output_rows = row_count_for_width(out.len(), dataset.num_genes);
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if compute.should_parallelize(output_rows, output_bytes) {
        let data_group_bytes = load_sparse_data_group_bytes(access, access_config, plan)?;
        if unsafe {
            try_scatter_sparse_rows_parallel_unchecked(
                compute,
                dataset,
                plan,
                Arc::clone(&index_bytes),
                data_group_bytes,
                out,
            )
        }? {
            return Ok(());
        }
    }
    if plan
        .data_groups
        .iter()
        .all(|group| matches!(group.source, SparseGroupSource::AccessItem(_)))
    {
        let mut scheduled = access.scheduled(
            plan.data_groups.iter().map(file_sparse_group_access_item),
            *access_config,
        )?;
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR data access ended early",
                ))
            })??;
            unsafe {
                scatter_sparse_data_group_unchecked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    index_bytes.as_ref(),
                    &bytes,
                    out,
                );
            }
        }
        if scheduled.next().is_some() {
            return Err(DataBankError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "scheduled CSR data access returned extra output",
            )));
        }
    } else {
        for group in &plan.data_groups {
            if unsafe {
                try_scatter_sparse_memory_identity_data_group_unchecked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    index_bytes.as_ref(),
                    out,
                )
            }? {
                continue;
            }
            let bytes = load_sparse_group(access, group)?;
            unsafe {
                scatter_sparse_data_group_unchecked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    index_bytes.as_ref(),
                    &bytes,
                    out,
                );
            }
        }
    }
    Ok(())
}

fn scatter_sparse_data_group_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<()> {
    if data_bytes.len() != group.bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "CSR data group decoded length is {}, expected {}",
            data_bytes.len(),
            group.bytes
        )));
    }

    match dataset.index_dtype {
        DType::U32 => scatter_sparse_data_group_checked_typed::<T, u32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            data_bytes,
            out,
        ),
        DType::I32 => scatter_sparse_data_group_checked_typed::<T, i32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            data_bytes,
            out,
        ),
        DType::U64 => scatter_sparse_data_group_checked_typed::<T, u64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            data_bytes,
            out,
        ),
        DType::I64 => scatter_sparse_data_group_checked_typed::<T, i64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            data_bytes,
            out,
        ),
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

fn try_scatter_sparse_rows_parallel_checked<T: DataValue>(
    compute: &DataBankComputePool,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Arc<[u8]>,
    data_group_bytes: Vec<Arc<[u8]>>,
    output_genes: usize,
    projection: Option<CompiledGeneProjection>,
    out: &mut [T],
) -> DataBankResult<bool> {
    try_scatter_sparse_rows_parallel_checked_with_group_indices(
        compute,
        dataset,
        plan,
        index_bytes,
        data_group_bytes,
        output_genes,
        projection,
        None,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_scatter_sparse_rows_parallel_checked_with_group_indices<T: DataValue>(
    compute: &DataBankComputePool,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Arc<[u8]>,
    data_group_bytes: Vec<Arc<[u8]>>,
    output_genes: usize,
    projection: Option<CompiledGeneProjection>,
    loaded_group_indices: Option<Vec<usize>>,
    out: &mut [T],
) -> DataBankResult<bool> {
    if output_genes == 0 || plan.data_pieces.is_empty() {
        return Ok(true);
    }
    if out.len() % output_genes != 0 {
        return Err(DataBankError::BufferSizeMismatch {
            expected: out.len().div_ceil(output_genes) * output_genes,
            actual: out.len(),
        });
    }
    let output_rows = out.len() / output_genes;
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if !compute.should_parallelize(output_rows, output_bytes) {
        return Ok(false);
    }
    let skip_unloaded_groups = loaded_group_indices.is_some() && projection.is_some();
    let group_to_loaded = sparse_loaded_group_indices(
        plan,
        &data_group_bytes,
        loaded_group_indices.as_deref(),
        skip_unloaded_groups,
    )?;

    let row_pieces = sparse_row_piece_ranges(plan, output_rows)?;
    let piece_groups = sparse_piece_group_indices(plan)?;
    let pieces: Arc<[SparseReadPiece]> = Arc::from(plan.data_pieces.clone().into_boxed_slice());
    let piece_groups: Arc<[usize]> = Arc::from(piece_groups.into_boxed_slice());
    let group_to_loaded: Arc<[usize]> = Arc::from(group_to_loaded.into_boxed_slice());
    let row_pieces = Arc::new(row_pieces);
    let data_group_bytes: Arc<[Arc<[u8]>]> = Arc::from(data_group_bytes.into_boxed_slice());
    let projection = Arc::new(projection);
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let num_genes = dataset.num_genes;
    let index_dtype = dataset.index_dtype;
    let job_count = compute.worker_count().min(output_rows).max(1);
    let rows_per_job = output_rows.div_ceil(job_count);
    let mut jobs = Vec::with_capacity(job_count);

    for row_start in (0..output_rows).step_by(rows_per_job) {
        let row_end = (row_start + rows_per_job).min(output_rows);
        let pieces = Arc::clone(&pieces);
        let piece_groups = Arc::clone(&piece_groups);
        let group_to_loaded = Arc::clone(&group_to_loaded);
        let row_pieces = Arc::clone(&row_pieces);
        let data_group_bytes = Arc::clone(&data_group_bytes);
        let index_bytes = Arc::clone(&index_bytes);
        let projection = Arc::clone(&projection);
        let job: ComputeJob = Box::new(move || match index_dtype {
            DType::U32 => scatter_sparse_row_range_checked_typed::<T, u32>(
                num_genes,
                output_genes,
                pieces.as_ref(),
                piece_groups.as_ref(),
                group_to_loaded.as_ref(),
                row_pieces.as_ref(),
                data_group_bytes.as_ref(),
                index_bytes.as_ref(),
                projection.as_ref().as_ref(),
                skip_unloaded_groups,
                row_start,
                row_end,
                out_addr,
                out_len,
            ),
            DType::I32 => scatter_sparse_row_range_checked_typed::<T, i32>(
                num_genes,
                output_genes,
                pieces.as_ref(),
                piece_groups.as_ref(),
                group_to_loaded.as_ref(),
                row_pieces.as_ref(),
                data_group_bytes.as_ref(),
                index_bytes.as_ref(),
                projection.as_ref().as_ref(),
                skip_unloaded_groups,
                row_start,
                row_end,
                out_addr,
                out_len,
            ),
            DType::U64 => scatter_sparse_row_range_checked_typed::<T, u64>(
                num_genes,
                output_genes,
                pieces.as_ref(),
                piece_groups.as_ref(),
                group_to_loaded.as_ref(),
                row_pieces.as_ref(),
                data_group_bytes.as_ref(),
                index_bytes.as_ref(),
                projection.as_ref().as_ref(),
                skip_unloaded_groups,
                row_start,
                row_end,
                out_addr,
                out_len,
            ),
            DType::I64 => scatter_sparse_row_range_checked_typed::<T, i64>(
                num_genes,
                output_genes,
                pieces.as_ref(),
                piece_groups.as_ref(),
                group_to_loaded.as_ref(),
                row_pieces.as_ref(),
                data_group_bytes.as_ref(),
                index_bytes.as_ref(),
                projection.as_ref().as_ref(),
                skip_unloaded_groups,
                row_start,
                row_end,
                out_addr,
                out_len,
            ),
            dtype => Err(DataBankError::UnsupportedDType {
                dtype,
                context: "CSR indices",
            }),
        });
        jobs.push(job);
    }

    compute.run_jobs(jobs)?;
    Ok(true)
}

unsafe fn try_scatter_sparse_rows_parallel_unchecked<T: DataValue>(
    compute: &DataBankComputePool,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Arc<[u8]>,
    data_group_bytes: Vec<Arc<[u8]>>,
    out: &mut [T],
) -> DataBankResult<bool> {
    let output_genes = dataset.num_genes;
    if output_genes == 0 || plan.data_pieces.is_empty() {
        return Ok(true);
    }
    let output_rows = out.len() / output_genes;
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if !compute.should_parallelize(output_rows, output_bytes) {
        return Ok(false);
    }
    debug_assert_eq!(data_group_bytes.len(), plan.data_groups.len());

    let row_pieces = sparse_row_piece_ranges(plan, output_rows)?;
    let piece_groups = sparse_piece_group_indices(plan)?;
    let pieces: Arc<[SparseReadPiece]> = Arc::from(plan.data_pieces.clone().into_boxed_slice());
    let piece_groups: Arc<[usize]> = Arc::from(piece_groups.into_boxed_slice());
    let row_pieces = Arc::new(row_pieces);
    let data_group_bytes: Arc<[Arc<[u8]>]> = Arc::from(data_group_bytes.into_boxed_slice());
    let out_addr = out.as_mut_ptr() as usize;
    let index_dtype = dataset.index_dtype;
    let job_count = compute.worker_count().min(output_rows).max(1);
    let rows_per_job = output_rows.div_ceil(job_count);
    let mut jobs = Vec::with_capacity(job_count);

    for row_start in (0..output_rows).step_by(rows_per_job) {
        let row_end = (row_start + rows_per_job).min(output_rows);
        let pieces = Arc::clone(&pieces);
        let piece_groups = Arc::clone(&piece_groups);
        let row_pieces = Arc::clone(&row_pieces);
        let data_group_bytes = Arc::clone(&data_group_bytes);
        let index_bytes = Arc::clone(&index_bytes);
        let job: ComputeJob = Box::new(move || match index_dtype {
            DType::U32 => unsafe {
                scatter_sparse_row_range_unchecked_typed::<T, u32>(
                    output_genes,
                    pieces.as_ref(),
                    piece_groups.as_ref(),
                    row_pieces.as_ref(),
                    data_group_bytes.as_ref(),
                    index_bytes.as_ref(),
                    row_start,
                    row_end,
                    out_addr,
                );
                Ok(())
            },
            DType::I32 => unsafe {
                scatter_sparse_row_range_unchecked_typed::<T, i32>(
                    output_genes,
                    pieces.as_ref(),
                    piece_groups.as_ref(),
                    row_pieces.as_ref(),
                    data_group_bytes.as_ref(),
                    index_bytes.as_ref(),
                    row_start,
                    row_end,
                    out_addr,
                );
                Ok(())
            },
            DType::U64 => unsafe {
                scatter_sparse_row_range_unchecked_typed::<T, u64>(
                    output_genes,
                    pieces.as_ref(),
                    piece_groups.as_ref(),
                    row_pieces.as_ref(),
                    data_group_bytes.as_ref(),
                    index_bytes.as_ref(),
                    row_start,
                    row_end,
                    out_addr,
                );
                Ok(())
            },
            DType::I64 => unsafe {
                scatter_sparse_row_range_unchecked_typed::<T, i64>(
                    output_genes,
                    pieces.as_ref(),
                    piece_groups.as_ref(),
                    row_pieces.as_ref(),
                    data_group_bytes.as_ref(),
                    index_bytes.as_ref(),
                    row_start,
                    row_end,
                    out_addr,
                );
                Ok(())
            },
            _ => unreachable!("CSR index dtype was validated during registration"),
        });
        jobs.push(job);
    }

    compute.run_jobs(jobs)?;
    Ok(true)
}

fn sparse_piece_group_indices(plan: &SparseBatchPlan) -> DataBankResult<Vec<usize>> {
    let mut piece_groups = vec![usize::MAX; plan.data_pieces.len()];
    for (group_index, group) in plan.data_groups.iter().enumerate() {
        for &piece_index in &group.parts {
            let Some(slot) = piece_groups.get_mut(piece_index) else {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR data group piece index is invalid".to_string(),
                ));
            };
            *slot = group_index;
        }
    }
    if piece_groups.contains(&usize::MAX) {
        return Err(DataBankError::InvalidArrayMeta(
            "CSR data piece is missing its group".to_string(),
        ));
    }
    Ok(piece_groups)
}

fn sparse_loaded_group_indices(
    plan: &SparseBatchPlan,
    data_group_bytes: &[Arc<[u8]>],
    loaded_group_indices: Option<&[usize]>,
    allow_unloaded_groups: bool,
) -> DataBankResult<Vec<usize>> {
    let mut group_to_loaded = vec![usize::MAX; plan.data_groups.len()];
    match loaded_group_indices {
        Some(indices) => {
            if indices.len() != data_group_bytes.len() {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR loaded group index count mismatch".to_string(),
                ));
            }
            for (loaded_index, &group_index) in indices.iter().enumerate() {
                let Some(group) = plan.data_groups.get(group_index) else {
                    return Err(DataBankError::InvalidArrayMeta(
                        "CSR loaded group index is out of range".to_string(),
                    ));
                };
                if group_to_loaded[group_index] != usize::MAX {
                    return Err(DataBankError::InvalidArrayMeta(
                        "CSR loaded group index is duplicated".to_string(),
                    ));
                }
                let bytes = &data_group_bytes[loaded_index];
                if bytes.len() != group.bytes {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "CSR data group decoded length is {}, expected {}",
                        bytes.len(),
                        group.bytes
                    )));
                }
                group_to_loaded[group_index] = loaded_index;
            }
            if !allow_unloaded_groups && group_to_loaded.contains(&usize::MAX) {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR data group bytes are incomplete".to_string(),
                ));
            }
        }
        None => {
            if data_group_bytes.len() != plan.data_groups.len() {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR data group byte count mismatch".to_string(),
                ));
            }
            for (group_index, (group, bytes)) in plan
                .data_groups
                .iter()
                .zip(data_group_bytes.iter())
                .enumerate()
            {
                if bytes.len() != group.bytes {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "CSR data group decoded length is {}, expected {}",
                        bytes.len(),
                        group.bytes
                    )));
                }
                group_to_loaded[group_index] = group_index;
            }
        }
    }
    Ok(group_to_loaded)
}

#[derive(Debug, Clone, Copy, Default)]
struct SparseRowPieceRange {
    start: usize,
    end: usize,
}

fn sparse_row_piece_ranges(
    plan: &SparseBatchPlan,
    output_rows: usize,
) -> DataBankResult<Vec<SparseRowPieceRange>> {
    let mut row_pieces = vec![SparseRowPieceRange::default(); output_rows];
    let mut piece_index = 0usize;
    for (output_row, row_range) in row_pieces.iter_mut().enumerate() {
        let start = piece_index;
        while let Some(piece) = plan.data_pieces.get(piece_index) {
            if piece.output_row < output_row {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR data pieces are not ordered by output row".to_string(),
                ));
            }
            if piece.output_row != output_row {
                break;
            }
            piece_index += 1;
        }
        *row_range = SparseRowPieceRange {
            start,
            end: piece_index,
        };
    }
    if let Some(piece) = plan.data_pieces.get(piece_index) {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "CSR data piece output row {} is out of range for {output_rows} rows",
            piece.output_row
        )));
    }
    Ok(row_pieces)
}

fn scatter_sparse_row_range_checked_typed<T, I>(
    num_genes: usize,
    output_genes: usize,
    pieces: &[SparseReadPiece],
    piece_groups: &[usize],
    group_to_loaded: &[usize],
    row_pieces: &[SparseRowPieceRange],
    data_group_bytes: &[Arc<[u8]>],
    index_bytes: &[u8],
    projection: Option<&CompiledGeneProjection>,
    skip_unloaded_groups: bool,
    row_start: usize,
    row_end: usize,
    out_addr: usize,
    out_len: usize,
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = mem::size_of::<T>();
    let index_size = mem::size_of::<I>();
    let out_ptr = out_addr as *mut T;

    for output_row in row_start..row_end {
        let row_piece_range = row_pieces.get(output_row).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR row piece index is out of range".to_string())
        })?;
        let row_base = output_row.checked_mul(output_genes).ok_or_else(|| {
            DataBankError::InvalidConfig("sparse output row overflow".to_string())
        })?;
        let row_end_offset = row_base.checked_add(output_genes).ok_or_else(|| {
            DataBankError::InvalidConfig("sparse output row end overflow".to_string())
        })?;
        if row_end_offset > out_len {
            return Err(DataBankError::BufferSizeMismatch {
                expected: row_end_offset,
                actual: out_len,
            });
        }
        // SAFETY: the caller partitions jobs by non-overlapping output row
        // ranges. This task only creates row slices inside its assigned range.
        let row_out =
            unsafe { std::slice::from_raw_parts_mut(out_ptr.add(row_base), output_genes) };
        for piece_index in row_piece_range.start..row_piece_range.end {
            let piece = pieces.get(piece_index).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR row piece index is invalid".to_string())
            })?;
            if piece.output_row != output_row {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR row piece belongs to a different output row".to_string(),
                ));
            }
            let group_index = *piece_groups.get(piece_index).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR piece group index is invalid".to_string())
            })?;
            let loaded_index = *group_to_loaded.get(group_index).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR loaded group map index is invalid".to_string())
            })?;
            if loaded_index == usize::MAX {
                if skip_unloaded_groups {
                    continue;
                }
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR data group bytes are missing".to_string(),
                ));
            }
            let data_bytes = data_group_bytes.get(loaded_index).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR data group index is invalid".to_string())
            })?;
            let data_end = piece.group_offset.checked_add(piece.bytes).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR data group offset overflow".to_string())
            })?;
            let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
            })?;
            let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
            })?;
            if data_end > data_bytes.len() || index_end > index_bytes.len() {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR row scatter is out of range".to_string(),
                ));
            }
            scatter_sparse_values_to_row_checked_typed::<T, I>(
                num_genes,
                projection,
                piece.elements,
                &index_bytes[piece.index_offset..index_end],
                &data_bytes[piece.group_offset..data_end],
                row_out,
            )?;
        }
    }
    let _ = value_size;
    Ok(())
}

fn scatter_sparse_values_to_row_checked_typed<T, I>(
    num_genes: usize,
    projection: Option<&CompiledGeneProjection>,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    row_out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = mem::size_of::<T>();
    let expected_data_bytes = elements.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR data byte length overflow".to_string())
    })?;
    if data_bytes.len() != expected_data_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_data_bytes,
            actual: data_bytes.len(),
        });
    }
    let index_size = mem::size_of::<I>();
    let expected_index_bytes = elements.checked_mul(index_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR index byte length overflow".to_string())
    })?;
    if index_bytes.len() != expected_index_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_index_bytes,
            actual: index_bytes.len(),
        });
    }

    let index_ptr = index_bytes.as_ptr().cast::<I>();
    let data_ptr = data_bytes.as_ptr();
    for nz in 0..elements {
        let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
        if gene >= num_genes {
            return Err(DataBankError::GeneIndexOutOfRange { gene, num_genes });
        }
        let Some(output_col) = projection
            .and_then(|projection| projection.output_for_source(gene))
            .or_else(|| projection.is_none().then_some(gene))
        else {
            continue;
        };
        if output_col >= row_out.len() {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene: output_col,
                num_genes: row_out.len(),
            });
        }
        let value = unsafe { ptr::read_unaligned(data_ptr.add(nz * value_size).cast::<T>()) };
        row_out[output_col] = value;
    }
    Ok(())
}

unsafe fn scatter_sparse_row_range_unchecked_typed<T, I>(
    output_genes: usize,
    pieces: &[SparseReadPiece],
    piece_groups: &[usize],
    row_pieces: &[SparseRowPieceRange],
    data_group_bytes: &[Arc<[u8]>],
    index_bytes: &[u8],
    row_start: usize,
    row_end: usize,
    out_addr: usize,
) where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = mem::size_of::<T>();
    let index_size = mem::size_of::<I>();
    let out_ptr = out_addr as *mut T;

    for output_row in row_start..row_end {
        let row_base = output_row * output_genes;
        // SAFETY: jobs are partitioned by disjoint output row ranges.
        let row_ptr = unsafe { out_ptr.add(row_base) };
        let row_piece_range = unsafe { row_pieces.get_unchecked(output_row) };
        for piece_index in row_piece_range.start..row_piece_range.end {
            let piece = unsafe { pieces.get_unchecked(piece_index) };
            let group_index = unsafe { *piece_groups.get_unchecked(piece_index) };
            let data_bytes = unsafe { data_group_bytes.get_unchecked(group_index) };
            let data_end = piece.group_offset + piece.bytes;
            let index_len = piece.elements * index_size;
            let index_end = piece.index_offset + index_len;
            let piece_indices = unsafe { index_bytes.get_unchecked(piece.index_offset..index_end) };
            let piece_data = unsafe { data_bytes.get_unchecked(piece.group_offset..data_end) };
            unsafe {
                scatter_sparse_values_to_row_unchecked_typed::<T, I>(
                    output_genes,
                    piece.elements,
                    piece_indices,
                    piece_data,
                    row_ptr,
                    value_size,
                );
            }
        }
    }
}

unsafe fn scatter_sparse_values_to_row_unchecked_typed<T, I>(
    output_genes: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    row_ptr: *mut T,
    value_size: usize,
) where
    T: DataValue,
    I: CsrIndex,
{
    let index_ptr = index_bytes.as_ptr().cast::<I>();
    let data_ptr = data_bytes.as_ptr();
    for nz in 0..elements {
        let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.unchecked_gene();
        debug_assert!(gene < output_genes);
        let value = unsafe { ptr::read_unaligned(data_ptr.add(nz * value_size).cast::<T>()) };
        unsafe { ptr::write(row_ptr.add(gene), value) };
    }
}

fn try_scatter_sparse_memory_identity_data_group_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<bool> {
    let SparseGroupSource::Memory {
        bytes,
        codec,
        expected_size,
        decoded,
    } = &group.source
    else {
        return Ok(false);
    };
    if !(*decoded || codec.is_identity()) {
        return Ok(false);
    }
    if bytes.len() != *expected_size {
        return Err(CodecError::SizeMismatch {
            codec: codec.name().to_string(),
            expected: *expected_size,
            actual: bytes.len(),
        }
        .into());
    }

    scatter_sparse_data_group_checked_from_decoded_source(
        dataset,
        pieces,
        group,
        index_bytes,
        bytes.as_ref(),
        out,
    )?;
    Ok(true)
}

fn scatter_sparse_data_group_checked_from_decoded_source<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    out: &mut [T],
) -> DataBankResult<()> {
    match dataset.index_dtype {
        DType::U32 => scatter_sparse_data_group_checked_typed_from_decoded_source::<T, u32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            decoded,
            out,
        ),
        DType::I32 => scatter_sparse_data_group_checked_typed_from_decoded_source::<T, i32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            decoded,
            out,
        ),
        DType::U64 => scatter_sparse_data_group_checked_typed_from_decoded_source::<T, u64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            decoded,
            out,
        ),
        DType::I64 => scatter_sparse_data_group_checked_typed_from_decoded_source::<T, i64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            decoded,
            out,
        ),
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

fn sparse_data_group_has_selected_values(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    gene_axis: &GeneAxisPlan,
) -> DataBankResult<bool> {
    let Some(projection) = gene_axis.projection() else {
        return Ok(true);
    };
    if projection.selected_sources.is_empty() {
        return validate_sparse_data_group_indices(dataset, pieces, group, index_bytes)
            .map(|_| false);
    }

    match dataset.index_dtype {
        DType::U32 => sparse_data_group_has_selected_values_typed::<u32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            projection,
        ),
        DType::I32 => sparse_data_group_has_selected_values_typed::<i32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            projection,
        ),
        DType::U64 => sparse_data_group_has_selected_values_typed::<u64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            projection,
        ),
        DType::I64 => sparse_data_group_has_selected_values_typed::<i64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            projection,
        ),
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

fn validate_sparse_data_group_indices(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
) -> DataBankResult<()> {
    match dataset.index_dtype {
        DType::U32 => validate_sparse_data_group_indices_typed::<u32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
        ),
        DType::I32 => validate_sparse_data_group_indices_typed::<i32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
        ),
        DType::U64 => validate_sparse_data_group_indices_typed::<u64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
        ),
        DType::I64 => validate_sparse_data_group_indices_typed::<i64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
        ),
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

fn sparse_data_group_has_selected_values_typed<I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    projection: &CompiledGeneProjection,
) -> DataBankResult<bool>
where
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR index grouped scan is out of range".to_string(),
            ));
        }
        let index_ptr = index_bytes[piece.index_offset..index_end]
            .as_ptr()
            .cast::<I>();
        for nz in 0..piece.elements {
            let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
            if gene >= num_genes {
                return Err(DataBankError::GeneIndexOutOfRange { gene, num_genes });
            }
            if projection.output_for_source(gene).is_some() {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn validate_sparse_data_group_indices_typed<I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
) -> DataBankResult<()>
where
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR index grouped scan is out of range".to_string(),
            ));
        }
        let index_ptr = index_bytes[piece.index_offset..index_end]
            .as_ptr()
            .cast::<I>();
        for nz in 0..piece.elements {
            let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
            if gene >= num_genes {
                return Err(DataBankError::GeneIndexOutOfRange { gene, num_genes });
            }
        }
    }
    Ok(())
}

fn scatter_sparse_data_group_projected_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if data_bytes.len() != group.bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "CSR data group decoded length is {}, expected {}",
            data_bytes.len(),
            group.bytes
        )));
    }
    let Some(projection) = gene_axis.projection() else {
        return scatter_sparse_data_group_checked(
            dataset,
            pieces,
            group,
            index_bytes,
            data_bytes,
            out,
        );
    };
    let projection = SparseProjectionCtx {
        num_genes: dataset.num_genes,
        output_genes: projection.output_genes(),
        projection,
    };

    match dataset.index_dtype {
        DType::U32 => scatter_sparse_data_group_projected_checked_typed::<T, u32>(
            projection,
            pieces,
            group,
            index_bytes,
            data_bytes,
            out,
        ),
        DType::I32 => scatter_sparse_data_group_projected_checked_typed::<T, i32>(
            projection,
            pieces,
            group,
            index_bytes,
            data_bytes,
            out,
        ),
        DType::U64 => scatter_sparse_data_group_projected_checked_typed::<T, u64>(
            projection,
            pieces,
            group,
            index_bytes,
            data_bytes,
            out,
        ),
        DType::I64 => scatter_sparse_data_group_projected_checked_typed::<T, i64>(
            projection,
            pieces,
            group,
            index_bytes,
            data_bytes,
            out,
        ),
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

fn try_scatter_sparse_memory_identity_data_group_projected_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<bool> {
    let SparseGroupSource::Memory {
        bytes,
        codec,
        expected_size,
        decoded,
    } = &group.source
    else {
        return Ok(false);
    };
    if !(*decoded || codec.is_identity()) {
        return Ok(false);
    }
    if bytes.len() != *expected_size {
        return Err(CodecError::SizeMismatch {
            codec: codec.name().to_string(),
            expected: *expected_size,
            actual: bytes.len(),
        }
        .into());
    }

    scatter_sparse_data_group_projected_checked_from_decoded_source(
        dataset,
        pieces,
        group,
        index_bytes,
        bytes.as_ref(),
        gene_axis,
        out,
    )?;
    Ok(true)
}

fn scatter_sparse_data_group_projected_checked_from_decoded_source<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    let Some(projection) = gene_axis.projection() else {
        return scatter_sparse_data_group_checked_from_decoded_source(
            dataset,
            pieces,
            group,
            index_bytes,
            decoded,
            out,
        );
    };
    let projection = SparseProjectionCtx {
        num_genes: dataset.num_genes,
        output_genes: projection.output_genes(),
        projection,
    };

    match dataset.index_dtype {
        DType::U32 => {
            scatter_sparse_data_group_projected_checked_typed_from_decoded_source::<T, u32>(
                projection,
                pieces,
                group,
                index_bytes,
                decoded,
                out,
            )
        }
        DType::I32 => {
            scatter_sparse_data_group_projected_checked_typed_from_decoded_source::<T, i32>(
                projection,
                pieces,
                group,
                index_bytes,
                decoded,
                out,
            )
        }
        DType::U64 => {
            scatter_sparse_data_group_projected_checked_typed_from_decoded_source::<T, u64>(
                projection,
                pieces,
                group,
                index_bytes,
                decoded,
                out,
            )
        }
        DType::I64 => {
            scatter_sparse_data_group_projected_checked_typed_from_decoded_source::<T, i64>(
                projection,
                pieces,
                group,
                index_bytes,
                decoded,
                out,
            )
        }
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

fn scatter_sparse_data_group_checked_typed<T, I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        let data_end = piece.group_offset.checked_add(piece.bytes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group offset overflow".to_string())
        })?;
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if data_end > data_bytes.len() || index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR data grouped scatter is out of range".to_string(),
            ));
        }
        scatter_sparse_values_checked_typed::<T, I>(
            num_genes,
            piece.output_row,
            piece.elements,
            &index_bytes[piece.index_offset..index_end],
            &data_bytes[piece.group_offset..data_end],
            out,
        )?;
    }
    Ok(())
}

fn scatter_sparse_data_group_checked_typed_from_decoded_source<T, I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        if piece.source.len() != piece.bytes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "CSR data source range length is {}, expected {}",
                piece.source.len(),
                piece.bytes
            )));
        }
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if piece.source.end > decoded.len() || index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR data direct scatter is out of range".to_string(),
            ));
        }
        scatter_sparse_values_checked_typed::<T, I>(
            num_genes,
            piece.output_row,
            piece.elements,
            &index_bytes[piece.index_offset..index_end],
            &decoded[piece.source.start..piece.source.end],
            out,
        )?;
    }
    Ok(())
}

fn scatter_sparse_data_group_projected_checked_typed<T, I>(
    projection: SparseProjectionCtx<'_>,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        let data_end = piece.group_offset.checked_add(piece.bytes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group offset overflow".to_string())
        })?;
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if data_end > data_bytes.len() || index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR data grouped scatter is out of range".to_string(),
            ));
        }
        scatter_sparse_values_projected_checked_typed::<T, I>(
            projection,
            piece.output_row,
            piece.elements,
            &index_bytes[piece.index_offset..index_end],
            &data_bytes[piece.group_offset..data_end],
            out,
        )?;
    }
    Ok(())
}

fn scatter_sparse_data_group_projected_checked_typed_from_decoded_source<T, I>(
    projection: SparseProjectionCtx<'_>,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        if piece.source.len() != piece.bytes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "CSR data source range length is {}, expected {}",
                piece.source.len(),
                piece.bytes
            )));
        }
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if piece.source.end > decoded.len() || index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR data projected direct scatter is out of range".to_string(),
            ));
        }
        scatter_sparse_values_projected_checked_typed::<T, I>(
            projection,
            piece.output_row,
            piece.elements,
            &index_bytes[piece.index_offset..index_end],
            &decoded[piece.source.start..piece.source.end],
            out,
        )?;
    }
    Ok(())
}

unsafe fn scatter_sparse_data_group_unchecked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) {
    debug_assert_eq!(data_bytes.len(), group.bytes);
    match dataset.index_dtype {
        DType::U32 => unsafe {
            scatter_sparse_data_group_unchecked_typed::<T, u32>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                data_bytes,
                out,
            );
        },
        DType::I32 => unsafe {
            scatter_sparse_data_group_unchecked_typed::<T, i32>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                data_bytes,
                out,
            );
        },
        DType::U64 => unsafe {
            scatter_sparse_data_group_unchecked_typed::<T, u64>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                data_bytes,
                out,
            );
        },
        DType::I64 => unsafe {
            scatter_sparse_data_group_unchecked_typed::<T, i64>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                data_bytes,
                out,
            );
        },
        _ => unreachable!("CSR index dtype was validated during registration"),
    }
}

unsafe fn try_scatter_sparse_memory_identity_data_group_unchecked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<bool> {
    let SparseGroupSource::Memory {
        bytes,
        codec,
        expected_size,
        decoded,
    } = &group.source
    else {
        return Ok(false);
    };
    if !(*decoded || codec.is_identity()) {
        return Ok(false);
    }
    if bytes.len() != *expected_size {
        return Err(CodecError::SizeMismatch {
            codec: codec.name().to_string(),
            expected: *expected_size,
            actual: bytes.len(),
        }
        .into());
    }

    unsafe {
        scatter_sparse_data_group_unchecked_from_decoded_source(
            dataset,
            pieces,
            group,
            index_bytes,
            bytes.as_ref(),
            out,
        );
    }
    Ok(true)
}

unsafe fn scatter_sparse_data_group_unchecked_from_decoded_source<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    out: &mut [T],
) {
    match dataset.index_dtype {
        DType::U32 => unsafe {
            scatter_sparse_data_group_unchecked_typed_from_decoded_source::<T, u32>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                decoded,
                out,
            );
        },
        DType::I32 => unsafe {
            scatter_sparse_data_group_unchecked_typed_from_decoded_source::<T, i32>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                decoded,
                out,
            );
        },
        DType::U64 => unsafe {
            scatter_sparse_data_group_unchecked_typed_from_decoded_source::<T, u64>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                decoded,
                out,
            );
        },
        DType::I64 => unsafe {
            scatter_sparse_data_group_unchecked_typed_from_decoded_source::<T, i64>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                decoded,
                out,
            );
        },
        _ => unreachable!("CSR index dtype was validated during registration"),
    }
}

unsafe fn scatter_sparse_data_group_unchecked_typed<T, I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        // SAFETY: group parts are produced by plan_sparse_batch and refer to the
        // same `pieces` slice passed here.
        let piece = unsafe { pieces.get_unchecked(piece_index) };
        let data_end = piece.group_offset + piece.bytes;
        let index_len = piece.elements * index_size;
        let index_end = piece.index_offset + index_len;
        // SAFETY: unchecked CSR access requires valid planned byte ranges.
        let piece_indices = unsafe { index_bytes.get_unchecked(piece.index_offset..index_end) };
        let piece_data = unsafe { data_bytes.get_unchecked(piece.group_offset..data_end) };
        unsafe {
            scatter_sparse_values_unchecked_typed::<T, I>(
                num_genes,
                piece.output_row,
                piece.elements,
                piece_indices,
                piece_data,
                out,
            );
        }
    }
}

unsafe fn scatter_sparse_data_group_unchecked_typed_from_decoded_source<T, I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    out: &mut [T],
) where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        // SAFETY: group parts are produced by plan_sparse_batch and refer to the
        // same `pieces` slice passed here.
        let piece = unsafe { pieces.get_unchecked(piece_index) };
        let index_len = piece.elements * index_size;
        let index_end = piece.index_offset + index_len;
        // SAFETY: unchecked CSR access requires valid planned byte ranges.
        let piece_indices = unsafe { index_bytes.get_unchecked(piece.index_offset..index_end) };
        let piece_data = unsafe { decoded.get_unchecked(piece.source.start..piece.source.end) };
        unsafe {
            scatter_sparse_values_unchecked_typed::<T, I>(
                num_genes,
                piece.output_row,
                piece.elements,
                piece_indices,
                piece_data,
                out,
            );
        }
    }
}

fn scatter_sparse_values_checked_typed<T, I>(
    num_genes: usize,
    output_row: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = mem::size_of::<T>();
    let row_base = output_row
        .checked_mul(num_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output row overflow".to_string()))?;
    let row_end = row_base.checked_add(num_genes).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output row end overflow".to_string())
    })?;
    if row_end > out.len() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out.len(),
        });
    }
    let expected_data_bytes = elements.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR data byte length overflow".to_string())
    })?;
    if data_bytes.len() != expected_data_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_data_bytes,
            actual: data_bytes.len(),
        });
    }
    let index_size = mem::size_of::<I>();
    let expected_index_bytes = elements.checked_mul(index_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR index byte length overflow".to_string())
    })?;
    if index_bytes.len() != expected_index_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_index_bytes,
            actual: index_bytes.len(),
        });
    }

    let index_ptr = index_bytes.as_ptr().cast::<I>();
    let data_ptr = data_bytes.as_ptr();
    for nz in 0..elements {
        let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
        if gene >= num_genes {
            return Err(DataBankError::GeneIndexOutOfRange { gene, num_genes });
        }
        let value = unsafe { ptr::read_unaligned(data_ptr.add(nz * value_size).cast::<T>()) };
        out[row_base + gene] = value;
    }
    Ok(())
}

fn scatter_sparse_values_projected_checked_typed<T, I>(
    projection: SparseProjectionCtx<'_>,
    output_row: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = mem::size_of::<T>();
    let row_base = output_row
        .checked_mul(projection.output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output row overflow".to_string()))?;
    let row_end = row_base
        .checked_add(projection.output_genes)
        .ok_or_else(|| {
            DataBankError::InvalidConfig("sparse output row end overflow".to_string())
        })?;
    if row_end > out.len() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out.len(),
        });
    }
    let expected_data_bytes = elements.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR data byte length overflow".to_string())
    })?;
    if data_bytes.len() != expected_data_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_data_bytes,
            actual: data_bytes.len(),
        });
    }
    let index_size = mem::size_of::<I>();
    let expected_index_bytes = elements.checked_mul(index_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR index byte length overflow".to_string())
    })?;
    if index_bytes.len() != expected_index_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_index_bytes,
            actual: index_bytes.len(),
        });
    }

    let index_ptr = index_bytes.as_ptr().cast::<I>();
    let data_ptr = data_bytes.as_ptr();
    for nz in 0..elements {
        let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
        if gene >= projection.num_genes {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene,
                num_genes: projection.num_genes,
            });
        }
        let Some(output_col) = projection.projection.output_for_source(gene) else {
            continue;
        };
        let value = unsafe { ptr::read_unaligned(data_ptr.add(nz * value_size).cast::<T>()) };
        out[row_base + output_col] = value;
    }
    Ok(())
}

unsafe fn scatter_sparse_values_unchecked_typed<T, I>(
    num_genes: usize,
    output_row: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) where
    T: DataValue,
    I: CsrIndex,
{
    let row_base = output_row * num_genes;
    let value_size = mem::size_of::<T>();
    let index_ptr = index_bytes.as_ptr().cast::<I>();
    let data_ptr = data_bytes.as_ptr();
    let out_ptr = unsafe { out.as_mut_ptr().add(row_base) };

    for nz in 0..elements {
        let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
        debug_assert!(
            raw_gene.checked_gene().is_ok_and(|gene| gene < num_genes),
            "unchecked CSR gene index is negative or out of range"
        );
        let gene = unsafe { raw_gene.unchecked_gene() };
        let value = unsafe { ptr::read_unaligned(data_ptr.add(nz * value_size).cast::<T>()) };
        unsafe { ptr::write(out_ptr.add(gene), value) };
    }
}

trait CsrIndex: Copy {
    fn checked_gene(self) -> DataBankResult<usize>;
    unsafe fn unchecked_gene(self) -> usize;
}

#[derive(Clone, Copy)]
struct SparseProjectionCtx<'a> {
    num_genes: usize,
    output_genes: usize,
    projection: &'a CompiledGeneProjection,
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

#[derive(Debug, Clone)]
enum DenseGroupSource {
    AccessItem(AccessItem),
    Memory {
        bytes: Arc<[u8]>,
        codec: SharedCodec,
        expected_size: usize,
        decoded: bool,
    },
}

#[derive(Debug, Clone)]
struct DenseReadGroup {
    source: DenseGroupSource,
    slice: SliceSpec,
    slice_ranges: Vec<RangeCopy>,
    parts: Vec<DenseGroupPart>,
    bytes: usize,
}

impl DenseReadGroup {
    fn finalize_slice(&mut self) {
        self.slice = SliceSpec::from_ranges(std::mem::take(&mut self.slice_ranges));
    }
}

#[derive(Debug, Clone)]
struct DenseGroupPart {
    segment_index: usize,
    group_offset: usize,
    bytes: usize,
}

#[derive(Debug, Clone)]
enum DenseLoadedGroup {
    Packed(Arc<[u8]>),
    DecodedSource(Arc<[u8]>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DenseGroupKey {
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

fn group_dense_segments(segments: &[DenseSegment]) -> DataBankResult<Vec<DenseReadGroup>> {
    let mut groups = Vec::with_capacity(segments.len());
    let mut by_key = fast_hash_map_with_capacity(segments.len());

    for (segment_index, segment) in segments.iter().enumerate() {
        let key = dense_group_key(&segment.chunk);
        let group_index = if let Some(&group_index) = by_key.get(&key) {
            group_index
        } else {
            let group_index = groups.len();
            by_key.insert(key, group_index);
            groups.push(DenseReadGroup {
                source: dense_group_source(&segment.chunk),
                slice: SliceSpec::Full,
                slice_ranges: Vec::new(),
                parts: Vec::new(),
                bytes: 0,
            });
            group_index
        };

        let group = &mut groups[group_index];
        let bytes = segment.source.len();
        let group_offset = append_group_slice(group, segment.source, bytes)?;
        group.parts.push(DenseGroupPart {
            segment_index,
            group_offset,
            bytes,
        });
    }

    for group in &mut groups {
        group.finalize_slice();
    }
    Ok(groups)
}

fn dense_group_key(chunk: &ChunkRef) -> DenseGroupKey {
    match chunk {
        ChunkRef::AccessItem(item) => DenseGroupKey::File {
            key: item.key,
            codec: codec_id(&item.codec),
            expected_size: item.expected_size,
        },
        ChunkRef::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => DenseGroupKey::Memory {
            ptr: bytes.as_ptr() as usize,
            len: bytes.len(),
            codec: codec_id(codec),
            expected_size: *expected_size,
            decoded: *decoded,
        },
    }
}

fn dense_group_source(chunk: &ChunkRef) -> DenseGroupSource {
    match chunk {
        ChunkRef::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = SliceSpec::Full;
            DenseGroupSource::AccessItem(item)
        }
        ChunkRef::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => DenseGroupSource::Memory {
            bytes: Arc::clone(bytes),
            codec: Arc::clone(codec),
            expected_size: *expected_size,
            decoded: *decoded,
        },
    }
}

fn append_group_slice(
    group: &mut DenseReadGroup,
    source: ByteRange,
    bytes: usize,
) -> DataBankResult<usize> {
    if source.len() != bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "dense source range length is {}, expected {bytes}",
            source.len()
        )));
    }
    let output_offset = group.bytes;
    group
        .slice_ranges
        .push(RangeCopy::new(output_offset, source.start, source.end));
    group.bytes = group.bytes.checked_add(bytes).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("dense grouped read byte length overflow".to_string())
    })?;
    Ok(output_offset)
}

fn group_access_item(group: &DenseReadGroup) -> DataBankResult<AccessItem> {
    match &group.source {
        DenseGroupSource::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = group.slice.clone();
            Ok(item)
        }
        DenseGroupSource::Memory { .. } => Err(DataBankError::InvalidArrayMeta(
            "memory chunk reached file scheduled path".to_string(),
        )),
    }
}

fn file_dense_group_access_item(group: &DenseReadGroup) -> AccessItem {
    match &group.source {
        DenseGroupSource::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = group.slice.clone();
            item
        }
        DenseGroupSource::Memory { .. } => {
            unreachable!("memory chunk reached dense file-backed scheduled path")
        }
    }
}

fn load_dense_group(access: &AccessHandle, group: &DenseReadGroup) -> DataBankResult<Vec<u8>> {
    match &group.source {
        DenseGroupSource::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = group.slice.clone();
            submit_access_item(access, item)
        }
        DenseGroupSource::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => load_memory_group(
            bytes.as_ref(),
            codec,
            *expected_size,
            *decoded,
            &group.slice,
        ),
    }
}

fn load_dense_groups_for_parallel(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    groups: &[DenseReadGroup],
) -> DataBankResult<Vec<DenseLoadedGroup>> {
    if groups
        .iter()
        .all(|group| matches!(group.source, DenseGroupSource::AccessItem(_)))
    {
        let mut scheduled = access.scheduled(
            groups.iter().map(file_dense_group_access_item),
            *access_config,
        )?;
        let mut out = Vec::with_capacity(groups.len());
        for group in groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled dense access ended early",
                ))
            })??;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            out.push(DenseLoadedGroup::Packed(Arc::from(
                bytes.into_boxed_slice(),
            )));
        }
        if scheduled.next().is_some() {
            return Err(DataBankError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "scheduled dense access returned extra output",
            )));
        }
        Ok(out)
    } else {
        groups
            .iter()
            .map(|group| load_dense_group_for_parallel(access, group))
            .collect()
    }
}

fn load_dense_group_for_parallel(
    access: &AccessHandle,
    group: &DenseReadGroup,
) -> DataBankResult<DenseLoadedGroup> {
    match &group.source {
        DenseGroupSource::AccessItem(_) => {
            let bytes = load_dense_group(access, group)?;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            Ok(DenseLoadedGroup::Packed(Arc::from(
                bytes.into_boxed_slice(),
            )))
        }
        DenseGroupSource::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => {
            if *decoded || codec.is_identity() {
                if bytes.len() != *expected_size {
                    return Err(CodecError::SizeMismatch {
                        codec: codec.name().to_string(),
                        expected: *expected_size,
                        actual: bytes.len(),
                    }
                    .into());
                }
                Ok(DenseLoadedGroup::DecodedSource(Arc::clone(bytes)))
            } else {
                let decoded = codec.decode(bytes.as_ref(), Some(*expected_size))?;
                if decoded.len() != *expected_size {
                    return Err(CodecError::SizeMismatch {
                        codec: codec.name().to_string(),
                        expected: *expected_size,
                        actual: decoded.len(),
                    }
                    .into());
                }
                Ok(DenseLoadedGroup::DecodedSource(Arc::from(
                    decoded.into_boxed_slice(),
                )))
            }
        }
    }
}

fn try_scatter_dense_memory_identity_group<T: DataValue>(
    num_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    out: &mut [T],
) -> DataBankResult<bool> {
    let DenseGroupSource::Memory {
        bytes,
        codec,
        expected_size,
        decoded,
    } = &group.source
    else {
        return Ok(false);
    };
    if !(*decoded || codec.is_identity()) {
        return Ok(false);
    }
    if bytes.len() != *expected_size {
        return Err(CodecError::SizeMismatch {
            codec: codec.name().to_string(),
            expected: *expected_size,
            actual: bytes.len(),
        }
        .into());
    }

    scatter_dense_group_from_decoded_source(num_genes, segments, group, bytes.as_ref(), out)?;
    Ok(true)
}

fn try_scatter_dense_memory_identity_group_projected<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<bool> {
    let DenseGroupSource::Memory {
        bytes,
        codec,
        expected_size,
        decoded,
    } = &group.source
    else {
        return Ok(false);
    };
    if !(*decoded || codec.is_identity()) {
        return Ok(false);
    }
    if bytes.len() != *expected_size {
        return Err(CodecError::SizeMismatch {
            codec: codec.name().to_string(),
            expected: *expected_size,
            actual: bytes.len(),
        }
        .into());
    }

    scatter_dense_group_from_decoded_source_projected(
        dataset_num_genes,
        output_genes,
        segments,
        group,
        bytes.as_ref(),
        gene_axis,
        out,
    )?;
    Ok(true)
}

fn scatter_dense_group_from_decoded_source<T: DataValue>(
    num_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    decoded: &[u8],
    out: &mut [T],
) -> DataBankResult<()> {
    for part in &group.parts {
        let segment = segments.get(part.segment_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group segment index is invalid".to_string())
        })?;
        let len = dense_segment_bytes::<T>(segment)?;
        if part.bytes != len || segment.source.len() != len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group part length is {}, source length is {}, expected {len}",
                part.bytes,
                segment.source.len()
            )));
        }
        if segment.source.end > decoded.len() {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense chunk decoded length is {}, expected at least {}",
                decoded.len(),
                segment.source.end
            )));
        }
        scatter_dense_segment(
            num_genes,
            segment,
            &decoded[segment.source.start..segment.source.end],
            out,
        )?;
    }
    Ok(())
}

fn scatter_dense_group_from_decoded_source_projected<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    decoded: &[u8],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    for part in &group.parts {
        let segment = segments.get(part.segment_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group segment index is invalid".to_string())
        })?;
        let len = dense_segment_bytes::<T>(segment)?;
        if part.bytes != len || segment.source.len() != len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group part length is {}, source length is {}, expected {len}",
                part.bytes,
                segment.source.len()
            )));
        }
        if segment.source.end > decoded.len() {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense chunk decoded length is {}, expected at least {}",
                decoded.len(),
                segment.source.end
            )));
        }
        scatter_dense_segment_projected(
            dataset_num_genes,
            output_genes,
            segment,
            &decoded[segment.source.start..segment.source.end],
            gene_axis,
            out,
        )?;
    }
    Ok(())
}

fn scatter_dense_group<T: DataValue>(
    num_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<()> {
    if bytes.len() != group.bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "dense group decoded length is {}, expected {}",
            bytes.len(),
            group.bytes
        )));
    }

    for part in &group.parts {
        let segment = segments.get(part.segment_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group segment index is invalid".to_string())
        })?;
        let len = dense_segment_bytes::<T>(segment)?;
        if part.bytes != len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group part length is {}, expected {len}",
                part.bytes
            )));
        }
        let end = part.group_offset.checked_add(len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group byte offset overflow".to_string())
        })?;
        if end > bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group decoded length is {}, expected at least {end}",
                bytes.len()
            )));
        }
        scatter_dense_segment(num_genes, segment, &bytes[part.group_offset..end], out)?;
    }
    Ok(())
}

fn scatter_dense_group_projected<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    bytes: &[u8],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if bytes.len() != group.bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "dense group decoded length is {}, expected {}",
            bytes.len(),
            group.bytes
        )));
    }

    for part in &group.parts {
        let segment = segments.get(part.segment_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group segment index is invalid".to_string())
        })?;
        let len = dense_segment_bytes::<T>(segment)?;
        if part.bytes != len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group part length is {}, expected {len}",
                part.bytes
            )));
        }
        let end = part.group_offset.checked_add(len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group byte offset overflow".to_string())
        })?;
        if end > bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group decoded length is {}, expected at least {end}",
                bytes.len()
            )));
        }
        scatter_dense_segment_projected(
            dataset_num_genes,
            output_genes,
            segment,
            &bytes[part.group_offset..end],
            gene_axis,
            out,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn try_scatter_dense_rows_parallel<T: DataValue>(
    compute: &DataBankComputePool,
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    loaded_groups: &[DenseLoadedGroup],
    projection: Option<CompiledGeneProjection>,
    out: &mut [T],
) -> DataBankResult<bool> {
    if output_genes == 0 || segments.is_empty() {
        return Ok(true);
    }
    if out.len() % output_genes != 0 {
        return Err(DataBankError::BufferSizeMismatch {
            expected: out.len().div_ceil(output_genes) * output_genes,
            actual: out.len(),
        });
    }
    let output_rows = out.len() / output_genes;
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("dense output byte length overflow".to_string())
    })?;
    if !compute.should_parallelize(output_rows, output_bytes) {
        return Ok(false);
    }
    if loaded_groups.len() != groups.len() {
        return Err(DataBankError::InvalidArrayMeta(
            "dense loaded group count mismatch".to_string(),
        ));
    }
    for (group, loaded) in groups.iter().zip(loaded_groups.iter()) {
        if let DenseLoadedGroup::Packed(bytes) = loaded {
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
        }
    }

    let segments: Arc<[DenseSegment]> = Arc::from(segments.to_vec().into_boxed_slice());
    let groups: Arc<[DenseReadGroup]> = Arc::from(groups.to_vec().into_boxed_slice());
    let loaded_groups: Arc<[DenseLoadedGroup]> =
        Arc::from(loaded_groups.to_vec().into_boxed_slice());
    let projection = Arc::new(projection);
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let job_count = compute.worker_count().min(groups.len()).max(1);
    let groups_per_job = groups.len().div_ceil(job_count);
    let mut jobs = Vec::with_capacity(job_count);

    for group_start in (0..groups.len()).step_by(groups_per_job) {
        let group_end = (group_start + groups_per_job).min(groups.len());
        let segments = Arc::clone(&segments);
        let groups = Arc::clone(&groups);
        let loaded_groups = Arc::clone(&loaded_groups);
        let projection = Arc::clone(&projection);
        let job: ComputeJob = Box::new(move || {
            scatter_dense_group_range_checked::<T>(
                dataset_num_genes,
                output_genes,
                segments.as_ref(),
                groups.as_ref(),
                loaded_groups.as_ref(),
                projection.as_ref().as_ref(),
                group_start,
                group_end,
                out_addr,
                out_len,
            )
        });
        jobs.push(job);
    }

    compute.run_jobs(jobs)?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn scatter_dense_group_range_checked<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    loaded_groups: &[DenseLoadedGroup],
    projection: Option<&CompiledGeneProjection>,
    group_start: usize,
    group_end: usize,
    out_addr: usize,
    out_len: usize,
) -> DataBankResult<()> {
    for group_index in group_start..group_end {
        let group = groups.get(group_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group index is invalid".to_string())
        })?;
        let loaded = loaded_groups.get(group_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense loaded group index is invalid".to_string())
        })?;
        for part in &group.parts {
            let segment = segments.get(part.segment_index).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("dense group segment index is invalid".to_string())
            })?;
            let bytes = dense_loaded_part_bytes(segment, part, loaded)?;
            if let Some(projection) = projection {
                scatter_dense_segment_to_output_projected::<T>(
                    dataset_num_genes,
                    output_genes,
                    segment,
                    bytes,
                    projection,
                    out_addr,
                    out_len,
                )?;
            } else {
                scatter_dense_segment_to_output::<T>(
                    output_genes,
                    segment,
                    bytes,
                    out_addr,
                    out_len,
                )?;
            }
        }
    }
    Ok(())
}

fn dense_loaded_part_bytes<'a>(
    segment: &DenseSegment,
    part: &DenseGroupPart,
    loaded: &'a DenseLoadedGroup,
) -> DataBankResult<&'a [u8]> {
    match loaded {
        DenseLoadedGroup::Packed(bytes) => {
            let end = part.group_offset.checked_add(part.bytes).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("dense group byte offset overflow".to_string())
            })?;
            if end > bytes.len() {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense group decoded length is {}, expected at least {end}",
                    bytes.len()
                )));
            }
            Ok(&bytes[part.group_offset..end])
        }
        DenseLoadedGroup::DecodedSource(bytes) => {
            if segment.source.len() != part.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense group part length is {}, source length is {}",
                    part.bytes,
                    segment.source.len()
                )));
            }
            if segment.source.end > bytes.len() {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense chunk decoded length is {}, expected at least {}",
                    bytes.len(),
                    segment.source.end
                )));
            }
            Ok(&bytes[segment.source.start..segment.source.end])
        }
    }
}

fn scatter_dense_segment_to_output<T: DataValue>(
    output_genes: usize,
    segment: &DenseSegment,
    bytes: &[u8],
    out_addr: usize,
    out_len: usize,
) -> DataBankResult<()> {
    let expected_bytes = dense_segment_bytes::<T>(segment)?;
    if mem::size_of::<T>() != T::DTYPE.item_size() || bytes.len() != expected_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_bytes,
            actual: bytes.len(),
        });
    }
    let dst_start = segment
        .output_row
        .checked_mul(output_genes)
        .and_then(|base| base.checked_add(segment.output_col_start))
        .ok_or_else(|| DataBankError::InvalidConfig("output offset overflow".to_string()))?;
    let dst_end = dst_start
        .checked_add(segment.output_cols)
        .ok_or_else(|| DataBankError::InvalidConfig("output offset overflow".to_string()))?;
    if dst_end > out_len {
        return Err(DataBankError::BufferSizeMismatch {
            expected: dst_end,
            actual: out_len,
        });
    }
    let out_ptr = out_addr as *mut T;
    // SAFETY: dense batch planning partitions every output row by disjoint
    // gene ranges. Group-parallel jobs therefore write non-overlapping value
    // slots even when they target the same output row.
    unsafe {
        ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out_ptr.add(dst_start).cast::<u8>(),
            expected_bytes,
        );
    }
    Ok(())
}

fn scatter_dense_segment_to_output_projected<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segment: &DenseSegment,
    bytes: &[u8],
    projection: &CompiledGeneProjection,
    out_addr: usize,
    out_len: usize,
) -> DataBankResult<()> {
    let expected_bytes = dense_segment_bytes::<T>(segment)?;
    if bytes.len() != expected_bytes || mem::size_of::<T>() != T::DTYPE.item_size() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_bytes,
            actual: bytes.len(),
        });
    }
    let source_end = segment
        .output_col_start
        .checked_add(segment.output_cols)
        .ok_or_else(|| DataBankError::InvalidConfig("dense source column overflow".to_string()))?;
    if source_end > dataset_num_genes {
        return Err(DataBankError::InvalidArrayMeta(
            "dense segment exceeds gene count".to_string(),
        ));
    }
    let row_base = segment
        .output_row
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output row overflow".to_string()))?;
    let row_end = row_base
        .checked_add(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output row end overflow".to_string()))?;
    if row_end > out_len {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out_len,
        });
    }

    let value_size = mem::size_of::<T>();
    let data_ptr = bytes.as_ptr();
    let out_ptr = out_addr as *mut T;
    if let Some(output_col_start) =
        projection.contiguous_output_for_source_run(segment.output_col_start, segment.output_cols)
    {
        let dst_start = row_base.checked_add(output_col_start).ok_or_else(|| {
            DataBankError::InvalidConfig("dense output offset overflow".to_string())
        })?;
        let dst_end = dst_start
            .checked_add(segment.output_cols)
            .ok_or_else(|| DataBankError::InvalidConfig("dense output end overflow".to_string()))?;
        if dst_end > row_end {
            return Err(DataBankError::BufferSizeMismatch {
                expected: dst_end,
                actual: out_len,
            });
        }
        // SAFETY: the projected source run maps to the same contiguous output
        // order, and group-parallel jobs write disjoint output columns.
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                out_ptr.add(dst_start).cast::<u8>(),
                expected_bytes,
            );
        }
        return Ok(());
    }
    for local_col in 0..segment.output_cols {
        let source_col = segment.output_col_start + local_col;
        let Some(output_col) = projection.output_for_source(source_col) else {
            continue;
        };
        if output_col >= output_genes {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene: output_col,
                num_genes: output_genes,
            });
        }
        let value =
            unsafe { ptr::read_unaligned(data_ptr.add(local_col * value_size).cast::<T>()) };
        // SAFETY: requested gene projection rejects duplicates, so source
        // ranges from different groups map to disjoint output columns.
        unsafe {
            ptr::write(out_ptr.add(row_base + output_col), value);
        }
    }
    Ok(())
}

fn submit_access_item(access: &AccessHandle, item: AccessItem) -> DataBankResult<Vec<u8>> {
    let (reply, rx) = oneshot::channel();
    let request = AccessRequest {
        key: item.key,
        codec: item.codec,
        expected_size: item.expected_size,
        slice: item.slice,
        reply,
    };
    access.send(request)?;
    wait_access(rx)
}

fn scatter_dense_segment<T: DataValue>(
    num_genes: usize,
    segment: &DenseSegment,
    bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<()> {
    let expected_bytes = dense_segment_bytes::<T>(segment)?;
    if bytes.len() != expected_bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "decoded segment length is {}, expected {expected_bytes}",
            bytes.len()
        )));
    }

    let dst_start = segment
        .output_row
        .checked_mul(num_genes)
        .and_then(|base| base.checked_add(segment.output_col_start))
        .ok_or_else(|| DataBankError::InvalidConfig("output offset overflow".to_string()))?;
    let dst_end = dst_start
        .checked_add(segment.output_cols)
        .ok_or_else(|| DataBankError::InvalidConfig("output offset overflow".to_string()))?;
    copy_ne_bytes_to_values(bytes, &mut out[dst_start..dst_end])
}

fn scatter_dense_segment_projected<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segment: &DenseSegment,
    bytes: &[u8],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    let expected_bytes = dense_segment_bytes::<T>(segment)?;
    if bytes.len() != expected_bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "decoded segment length is {}, expected {expected_bytes}",
            bytes.len()
        )));
    }
    let Some(projection) = gene_axis.projection() else {
        return scatter_dense_segment(output_genes, segment, bytes, out);
    };

    let source_end = segment
        .output_col_start
        .checked_add(segment.output_cols)
        .ok_or_else(|| DataBankError::InvalidConfig("dense source column overflow".to_string()))?;
    if source_end > dataset_num_genes {
        return Err(DataBankError::InvalidArrayMeta(
            "dense segment exceeds gene count".to_string(),
        ));
    }
    let row_base = segment
        .output_row
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output row overflow".to_string()))?;
    let row_end = row_base
        .checked_add(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output row end overflow".to_string()))?;
    if row_end > out.len() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out.len(),
        });
    }

    if let Some(output_col_start) =
        projection.contiguous_output_for_source_run(segment.output_col_start, segment.output_cols)
    {
        let dst_start = row_base.checked_add(output_col_start).ok_or_else(|| {
            DataBankError::InvalidConfig("dense output offset overflow".to_string())
        })?;
        let dst_end = dst_start
            .checked_add(segment.output_cols)
            .ok_or_else(|| DataBankError::InvalidConfig("dense output end overflow".to_string()))?;
        if dst_end > row_end {
            return Err(DataBankError::BufferSizeMismatch {
                expected: dst_end,
                actual: out.len(),
            });
        }
        return copy_ne_bytes_to_values(bytes, &mut out[dst_start..dst_end]);
    }

    let value_size = mem::size_of::<T>();
    let data_ptr = bytes.as_ptr();
    for local_col in 0..segment.output_cols {
        let source_col = segment.output_col_start + local_col;
        let Some(output_col) = projection.output_for_source(source_col) else {
            continue;
        };
        let value =
            unsafe { ptr::read_unaligned(data_ptr.add(local_col * value_size).cast::<T>()) };
        out[row_base + output_col] = value;
    }
    Ok(())
}

fn dense_segment_bytes<T: DataValue>(segment: &DenseSegment) -> DataBankResult<usize> {
    segment
        .output_cols
        .checked_mul(T::DTYPE.item_size())
        .ok_or_else(|| DataBankError::InvalidArrayMeta("segment byte size overflow".to_string()))
}

fn copy_ne_bytes_to_values<T: DataValue>(bytes: &[u8], out: &mut [T]) -> DataBankResult<()> {
    let expected = out
        .len()
        .checked_mul(mem::size_of::<T>())
        .ok_or_else(|| DataBankError::InvalidConfig("typed byte length overflow".to_string()))?;
    if mem::size_of::<T>() != T::DTYPE.item_size() || bytes.len() != expected {
        return Err(DataBankError::BufferSizeMismatch {
            expected,
            actual: bytes.len(),
        });
    }
    if bytes.is_empty() {
        return Ok(());
    }

    // SAFETY: DataValue is sealed and implemented only for primitive numeric
    // types plus transparent u16 half-precision wrappers in array.rs.
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), expected) };
    dst.copy_from_slice(bytes);
    Ok(())
}

fn zero_values<T: DataValue>(out: &mut [T]) {
    if out.is_empty() {
        return;
    }
    // SAFETY: DataValue is sealed to numeric primitives plus transparent u16
    // half wrappers. Their zero value is represented by all-zero bytes.
    unsafe {
        ptr::write_bytes(out.as_mut_ptr(), 0, out.len());
    }
}

fn zeroed_byte_vec(len: usize) -> Vec<u8> {
    vec![0; len]
}

fn load_memory_group(
    bytes: &[u8],
    codec: &SharedCodec,
    expected_size: usize,
    decoded: bool,
    slice: &SliceSpec,
) -> DataBankResult<Vec<u8>> {
    if decoded || codec.is_identity() {
        if bytes.len() != expected_size {
            return Err(CodecError::SizeMismatch {
                codec: codec.name().to_string(),
                expected: expected_size,
                actual: bytes.len(),
            }
            .into());
        }
        return copy_slices(bytes, slice);
    }

    let decoded = codec.decode(bytes, Some(expected_size))?;
    copy_slices(&decoded, slice)
}

fn copy_slices(data: &[u8], slice: &SliceSpec) -> DataBankResult<Vec<u8>> {
    let plan = slice.plan(data.len())?;
    let Some(ranges) = plan.ranges() else {
        return Ok(data.to_vec());
    };

    if ranges_are_packed(ranges, plan.output_len) {
        let mut out = Vec::with_capacity(plan.output_len);
        for range in ranges {
            out.extend_from_slice(&data[range.src_start..range.src_end]);
        }
        return Ok(out);
    }

    let mut out = vec![0; plan.output_len];
    for range in ranges {
        out[range.dst_offset..range.dst_offset + range.len()]
            .copy_from_slice(&data[range.src_start..range.src_end]);
    }
    Ok(out)
}

fn ranges_are_packed(ranges: &[RangeCopy], output_len: usize) -> bool {
    let mut cursor = 0usize;
    for range in ranges {
        if range.dst_offset != cursor {
            return false;
        }
        let Some(next) = cursor.checked_add(range.len()) else {
            return false;
        };
        cursor = next;
    }
    cursor == output_len
}

fn collect_prefetch_key(keys: &mut FastHashSet<ChunkKey>, chunk: &ChunkRef) {
    if let ChunkRef::AccessItem(item) = chunk {
        keys.insert(item.key);
    }
}

fn collect_range_prefetch_keys(keys: &mut FastHashSet<ChunkKey>, segments: &[RangeSegment]) {
    for segment in segments {
        collect_prefetch_key(keys, &segment.chunk);
    }
}

fn prefetch_keys(access: &AccessHandle, keys: FastHashSet<ChunkKey>) -> DataBankResult<()> {
    let mut receivers = Vec::with_capacity(keys.len());
    for key in keys {
        receivers.push(access.prefetch(key)?);
    }
    for receiver in receivers {
        wait_prefetch(receiver)?;
    }
    Ok(())
}

fn codec_id(codec: &SharedCodec) -> usize {
    Arc::as_ptr(codec) as *const () as usize
}

fn wait_access(rx: oneshot::Receiver<io::Result<Vec<u8>>>) -> DataBankResult<Vec<u8>> {
    rx.blocking_recv()
        .map_err(|_| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "access reply dropped",
            ))
        })?
        .map_err(DataBankError::Io)
}

fn wait_prefetch(rx: oneshot::Receiver<io::Result<()>>) -> DataBankResult<()> {
    rx.blocking_recv()
        .map_err(|_| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "prefetch reply dropped",
            ))
        })?
        .map_err(DataBankError::Io)
}

// ---------------------------------------------------------------------------
// Scheduled prefetch
// ---------------------------------------------------------------------------

/// The per-batch plan produced by the scheduled prefetcher.
///
/// Each batch is planned independently (chunks are not merged across batches).
/// The plan carries both the scatter metadata needed to assemble the decoded
/// bytes into a row-major output buffer and the ordered list of access items
/// that the access scheduler consumes.
enum BatchPlan {
    Dense {
        cells: Vec<usize>,
        segments: Vec<DenseSegment>,
        groups: Vec<DenseReadGroup>,
        num_genes: usize,
    },
    Sparse {
        cells: Vec<usize>,
        plan: SparseBatchPlan,
        dataset: Arc<Dataset>,
    },
}

/// Plan one batch into a [`BatchPlan`] plus its ordered access items.
///
/// Chunks are grouped within the batch only; no merging happens across
/// batches. File-backed chunks are streamed through the access scheduler;
/// memory-backed chunks stay in the batch plan and are decoded by databank when
/// the prefetched batch is assembled.
fn plan_batch_owned(
    dataset: Arc<Dataset>,
    cells: Vec<usize>,
    gene_axis: &GeneAxisPlan,
) -> DataBankResult<(BatchPlan, Vec<AccessItem>)> {
    let num_cells = dataset.as_ref().num_cells();
    for &cell in &cells {
        if cell >= num_cells {
            return Err(DataBankError::CellIndexOutOfRange { cell, num_cells });
        }
    }
    match dataset.as_ref() {
        Dataset::Dense1D(d) => {
            let segments = match gene_axis.projection() {
                Some(projection) => {
                    plan::plan_dense_1d_selected_sources(d, &cells, &projection.selected_sources)?
                }
                None => plan::plan_dense_1d(d, &cells)?,
            };
            let groups = group_dense_segments(&segments)?;
            let items = dense_group_access_items(&groups)?;
            Ok((
                BatchPlan::Dense {
                    cells,
                    segments,
                    groups,
                    num_genes: d.num_genes,
                },
                items,
            ))
        }
        Dataset::Dense2D(d) => {
            let segments = match gene_axis.projection() {
                Some(projection) => {
                    plan::plan_dense_2d_selected_sources(d, &cells, &projection.selected_sources)?
                }
                None => plan::plan_dense_2d(d, &cells)?,
            };
            let groups = group_dense_segments(&segments)?;
            let items = dense_group_access_items(&groups)?;
            Ok((
                BatchPlan::Dense {
                    cells,
                    segments,
                    groups,
                    num_genes: d.num_genes,
                },
                items,
            ))
        }
        Dataset::SparseCsr(d) => {
            let rows = plan::plan_sparse_rows(d, &cells)?;
            let value_size = d.data.dtype.item_size();
            let plan = plan_sparse_batch_with_value_size(d, &rows, value_size)?;
            let items = if gene_axis.projection().is_some() {
                // Projected CSR must decode indices before it can decide which
                // data chunks contain requested genes. Data chunks are scheduled
                // later by `scatter_sparse_prefetch_projected_data`.
                sparse_plan_index_file_access_items(&plan)?
            } else {
                sparse_plan_file_access_items(&plan)?
            };
            Ok((
                BatchPlan::Sparse {
                    cells,
                    plan,
                    dataset,
                },
                items,
            ))
        }
    }
}

fn dense_group_access_items(groups: &[DenseReadGroup]) -> DataBankResult<Vec<AccessItem>> {
    let mut items = Vec::with_capacity(groups.len());
    for group in groups {
        if matches!(group.source, DenseGroupSource::AccessItem(_)) {
            items.push(group_access_item(group)?);
        }
    }
    Ok(items)
}

fn sparse_plan_file_access_items(plan: &SparseBatchPlan) -> DataBankResult<Vec<AccessItem>> {
    let mut items = Vec::with_capacity(plan.index_groups.len() + plan.data_groups.len());
    for group in &plan.index_groups {
        if matches!(group.source, SparseGroupSource::AccessItem(_)) {
            items.push(sparse_group_access_item(group)?);
        }
    }
    for group in &plan.data_groups {
        if matches!(group.source, SparseGroupSource::AccessItem(_)) {
            items.push(sparse_group_access_item(group)?);
        }
    }
    Ok(items)
}

fn sparse_plan_index_file_access_items(plan: &SparseBatchPlan) -> DataBankResult<Vec<AccessItem>> {
    let mut items = Vec::with_capacity(plan.index_groups.len());
    for group in &plan.index_groups {
        if matches!(group.source, SparseGroupSource::AccessItem(_)) {
            items.push(sparse_group_access_item(group)?);
        }
    }
    Ok(items)
}

/// A prefetched batch: the cell indices and the databank-allocated,
/// already-scattered row-major buffer (`cells.len() * num_genes` values).
#[derive(Debug)]
pub struct PrefetchedBatch<T>
where
    T: DataValue,
{
    pub cells: Vec<usize>,
    pub buffer: Vec<T>,
    pub num_genes: usize,
}

/// Blocking iterator over scheduled prefetch results.
///
/// Accepts a user iterator yielding one batch of cell indices at a time. Each
/// batch is planned independently and its access items are streamed into the
/// access scheduler's [`ScheduledAccess`], which provides the chunk-level
/// look-ahead (`prefetch_step`, `decode_ahead_steps`, etc.). The databank-level
/// look-ahead is [`Self::prefetch_step`]: a background producer keeps a bounded
/// completed queue of decoded batches ahead of the consumer.
///
/// The databank iterator (batches) and the access iterator (chunk groups) are
/// deliberately not aligned: one batch expands to a variable number of chunk
/// groups, so the driver tracks how many `scheduled.next()` calls each batch
/// requires via its plan.
///
/// Results are cached in the completed queue, so no external output buffer is
/// accepted.
pub struct PrefetchCells<T>
where
    T: DataValue,
{
    rx: Option<flume::Receiver<DataBankResult<PrefetchedBatch<T>>>>,
    output_names: Vec<GeneNameView>,
    _dataset: Arc<Dataset>,
    prefetch_step: usize,
    cancel: Arc<PrefetchCancelRegistry>,
    producer: Option<thread::JoinHandle<()>>,
}

impl<T> PrefetchCells<T>
where
    T: DataValue,
{
    /// Configured completed-queue depth (number of decoded batches kept ahead
    /// of the consumer).
    pub fn prefetch_step(&self) -> usize {
        self.prefetch_step
    }

    pub fn gene_names(&self) -> &[GeneNameView] {
        &self.output_names
    }
}

impl<T> Iterator for PrefetchCells<T>
where
    T: DataValue,
{
    type Item = DataBankResult<PrefetchedBatch<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let rx = self.rx.as_ref()?;
        match rx.recv() {
            Ok(batch) => Some(batch),
            Err(_) => {
                if let Some(handle) = self.producer.take() {
                    let _ = handle.join();
                }
                None
            }
        }
    }
}

impl<T> Drop for PrefetchCells<T>
where
    T: DataValue,
{
    fn drop(&mut self) {
        self.cancel.cancel_all();
        self.rx.take();
        if let Some(handle) = self.producer.take() {
            let _ = handle.join();
        }
    }
}

struct PrefetchProducer<T, I>
where
    T: DataValue,
    I: Iterator,
    I::Item: AsRef<[usize]>,
{
    access: AccessHandle,
    compute: Arc<DataBankComputePool>,
    dataset: Arc<Dataset>,
    batch_source: I,
    access_config: ScheduledAccessConfig,
    gene_axis: GeneAxisPlan,
    tx: flume::Sender<DataBankResult<PrefetchedBatch<T>>>,
    cancel: Arc<PrefetchCancelRegistry>,
    prefetch_step: usize,
    profiler: ScheduledPrefetchProfiler,
}

type BatchSeq = u64;
type ScheduledBatchAccess = ScheduledAccess<std::vec::IntoIter<AccessItem>>;

struct PlannedBatch {
    seq: BatchSeq,
    plan: BatchPlan,
    scheduled: ScheduledBatchAccess,
    cancel: Arc<PrefetchCancel>,
}

struct PlannedMessage {
    seq: BatchSeq,
    result: DataBankResult<Box<PlannedBatch>>,
}

struct DoneMessage<T>
where
    T: DataValue,
{
    seq: BatchSeq,
    result: DataBankResult<PrefetchedBatch<T>>,
}

#[derive(Clone)]
struct ScheduledPrefetchProfiler {
    inner: Option<Arc<ScheduledPrefetchProfileInner>>,
}

struct ScheduledPrefetchProfileInner {
    label: String,
    producer_started: Instant,
    batch_source_next_ns: AtomicU64,
    submit_request_ns: AtomicU64,
    submit_response_ns: AtomicU64,
    coordinator_wait_ns: AtomicU64,
    output_send_ns: AtomicU64,
    request_queue_wait_ns: AtomicU64,
    request_plan_ns: AtomicU64,
    request_schedule_ns: AtomicU64,
    request_send_ns: AtomicU64,
    request_total_ns: AtomicU64,
    response_queue_wait_ns: AtomicU64,
    response_total_ns: AtomicU64,
    assemble_total_ns: AtomicU64,
    scheduled_drain_ns: AtomicU64,
    scheduled_drain_calls: AtomicU64,
    scheduled_drain_bytes: AtomicU64,
    alloc_ns: AtomicU64,
    memory_load_ns: AtomicU64,
    memory_load_calls: AtomicU64,
    memory_load_bytes: AtomicU64,
    scatter_ns: AtomicU64,
    scatter_calls: AtomicU64,
    request_jobs: AtomicU64,
    response_jobs: AtomicU64,
    emitted_batches: AtomicU64,
    request_errors: AtomicU64,
    response_errors: AtomicU64,
}

impl ScheduledPrefetchProfileInner {
    fn new(label: String) -> Self {
        Self {
            label,
            producer_started: Instant::now(),
            batch_source_next_ns: AtomicU64::new(0),
            submit_request_ns: AtomicU64::new(0),
            submit_response_ns: AtomicU64::new(0),
            coordinator_wait_ns: AtomicU64::new(0),
            output_send_ns: AtomicU64::new(0),
            request_queue_wait_ns: AtomicU64::new(0),
            request_plan_ns: AtomicU64::new(0),
            request_schedule_ns: AtomicU64::new(0),
            request_send_ns: AtomicU64::new(0),
            request_total_ns: AtomicU64::new(0),
            response_queue_wait_ns: AtomicU64::new(0),
            response_total_ns: AtomicU64::new(0),
            assemble_total_ns: AtomicU64::new(0),
            scheduled_drain_ns: AtomicU64::new(0),
            scheduled_drain_calls: AtomicU64::new(0),
            scheduled_drain_bytes: AtomicU64::new(0),
            alloc_ns: AtomicU64::new(0),
            memory_load_ns: AtomicU64::new(0),
            memory_load_calls: AtomicU64::new(0),
            memory_load_bytes: AtomicU64::new(0),
            scatter_ns: AtomicU64::new(0),
            scatter_calls: AtomicU64::new(0),
            request_jobs: AtomicU64::new(0),
            response_jobs: AtomicU64::new(0),
            emitted_batches: AtomicU64::new(0),
            request_errors: AtomicU64::new(0),
            response_errors: AtomicU64::new(0),
        }
    }
}

impl ScheduledPrefetchProfiler {
    fn from_env() -> Self {
        if !env_flag("SCDATA_DATABANK_PREFETCH_PROFILE") {
            return Self { inner: None };
        }
        let label = std::env::var("SCDATA_DATABANK_PREFETCH_PROFILE_LABEL")
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "scheduled_prefetch".to_string());
        Self {
            inner: Some(Arc::new(ScheduledPrefetchProfileInner::new(label))),
        }
    }

    fn start(&self) -> Option<Instant> {
        self.inner.as_ref().map(|_| Instant::now())
    }

    fn record_duration(
        &self,
        started: Option<Instant>,
        counter: impl FnOnce(&ScheduledPrefetchProfileInner) -> &AtomicU64,
    ) {
        let (Some(inner), Some(started)) = (&self.inner, started) else {
            return;
        };
        counter(inner).fetch_add(duration_ns(started.elapsed()), Ordering::Relaxed);
    }

    fn record_batch_source_next(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.batch_source_next_ns);
    }

    fn record_submit_request(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.submit_request_ns);
    }

    fn record_submit_response(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.submit_response_ns);
    }

    fn record_coordinator_wait(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.coordinator_wait_ns);
    }

    fn record_output_send(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.output_send_ns);
    }

    fn record_request_queue_wait(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.request_queue_wait_ns);
    }

    fn record_request_plan(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.request_plan_ns);
    }

    fn record_request_schedule(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.request_schedule_ns);
    }

    fn record_request_send(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.request_send_ns);
    }

    fn record_request_total(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.request_total_ns);
    }

    fn record_response_queue_wait(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.response_queue_wait_ns);
    }

    fn record_response_total(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.response_total_ns);
    }

    fn record_assemble_total(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.assemble_total_ns);
    }

    fn record_scheduled_drain(&self, started: Option<Instant>, bytes: usize) {
        let (Some(inner), Some(started)) = (&self.inner, started) else {
            return;
        };
        inner
            .scheduled_drain_ns
            .fetch_add(duration_ns(started.elapsed()), Ordering::Relaxed);
        inner.scheduled_drain_calls.fetch_add(1, Ordering::Relaxed);
        inner
            .scheduled_drain_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    fn record_alloc(&self, started: Option<Instant>) {
        self.record_duration(started, |inner| &inner.alloc_ns);
    }

    fn record_memory_load(&self, started: Option<Instant>, bytes: usize) {
        let (Some(inner), Some(started)) = (&self.inner, started) else {
            return;
        };
        inner
            .memory_load_ns
            .fetch_add(duration_ns(started.elapsed()), Ordering::Relaxed);
        inner.memory_load_calls.fetch_add(1, Ordering::Relaxed);
        inner
            .memory_load_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    fn record_scatter(&self, started: Option<Instant>) {
        let (Some(inner), Some(started)) = (&self.inner, started) else {
            return;
        };
        inner
            .scatter_ns
            .fetch_add(duration_ns(started.elapsed()), Ordering::Relaxed);
        inner.scatter_calls.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_request_job(&self) {
        if let Some(inner) = &self.inner {
            inner.request_jobs.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn inc_response_job(&self) {
        if let Some(inner) = &self.inner {
            inner.response_jobs.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn inc_emitted_batch(&self) {
        if let Some(inner) = &self.inner {
            inner.emitted_batches.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn inc_request_error(&self) {
        if let Some(inner) = &self.inner {
            inner.request_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn inc_response_error(&self) {
        if let Some(inner) = &self.inner {
            inner.response_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn print_summary(&self) {
        let Some(inner) = &self.inner else {
            return;
        };
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let producer_elapsed_ms = duration_ms(inner.producer_started.elapsed());
        println!(
            "databank/scheduled_prefetch_profile label={} producer_elapsed_ms={producer_elapsed_ms:.3} request_jobs={} response_jobs={} emitted_batches={} request_queue_ms={:.3} request_plan_ms={:.3} request_schedule_ms={:.3} request_send_ms={:.3} request_total_ms={:.3} response_queue_ms={:.3} response_assemble_ms={:.3} response_total_ms={:.3} scheduled_drain_ms={:.3} scheduled_drain_calls={} scheduled_drain_mib={:.3} alloc_ms={:.3} memory_load_ms={:.3} memory_load_calls={} memory_load_mib={:.3} scatter_ms={:.3} scatter_calls={} submit_request_ms={:.3} submit_response_ms={:.3} producer_wait_ms={:.3} output_send_ms={:.3} batch_source_ms={:.3} request_errors={} response_errors={}",
            inner.label,
            load(&inner.request_jobs),
            load(&inner.response_jobs),
            load(&inner.emitted_batches),
            ns_to_ms(load(&inner.request_queue_wait_ns)),
            ns_to_ms(load(&inner.request_plan_ns)),
            ns_to_ms(load(&inner.request_schedule_ns)),
            ns_to_ms(load(&inner.request_send_ns)),
            ns_to_ms(load(&inner.request_total_ns)),
            ns_to_ms(load(&inner.response_queue_wait_ns)),
            ns_to_ms(load(&inner.assemble_total_ns)),
            ns_to_ms(load(&inner.response_total_ns)),
            ns_to_ms(load(&inner.scheduled_drain_ns)),
            load(&inner.scheduled_drain_calls),
            bytes_to_mib(load(&inner.scheduled_drain_bytes)),
            ns_to_ms(load(&inner.alloc_ns)),
            ns_to_ms(load(&inner.memory_load_ns)),
            load(&inner.memory_load_calls),
            bytes_to_mib(load(&inner.memory_load_bytes)),
            ns_to_ms(load(&inner.scatter_ns)),
            load(&inner.scatter_calls),
            ns_to_ms(load(&inner.submit_request_ns)),
            ns_to_ms(load(&inner.submit_response_ns)),
            ns_to_ms(load(&inner.coordinator_wait_ns)),
            ns_to_ms(load(&inner.output_send_ns)),
            ns_to_ms(load(&inner.batch_source_next_ns)),
            load(&inner.request_errors),
            load(&inner.response_errors),
        );
    }
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => false,
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[derive(Debug)]
struct PrefetchCancelRegistry {
    cancelled: AtomicBool,
    active: Mutex<BTreeMap<BatchSeq, Arc<PrefetchCancel>>>,
}

impl PrefetchCancelRegistry {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            cancelled: AtomicBool::new(false),
            active: Mutex::new(BTreeMap::new()),
        })
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn register(&self, seq: BatchSeq, cancel: Arc<PrefetchCancel>) {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.cancelled.load(Ordering::Acquire) {
            drop(active);
            cancel.cancel_in_flight();
        } else {
            active.insert(seq, cancel);
        }
    }

    fn unregister(&self, seq: BatchSeq) {
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&seq);
    }

    fn cancel_all(&self) {
        self.cancelled.store(true, Ordering::Release);
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let cancels = active.values().cloned().collect::<Vec<_>>();
        active.clear();
        drop(active);
        for cancel in cancels {
            cancel.cancel_in_flight();
        }
    }
}

struct ProducerState<T>
where
    T: DataValue,
{
    next_read_seq: BatchSeq,
    next_emit_seq: BatchSeq,
    source_done: bool,
    stop_reading: bool,
    outstanding: usize,
    active_requests: usize,
    active_responses: usize,
    response_limit: usize,
    planned_ready: VecDeque<PlannedBatch>,
    completed: BTreeMap<BatchSeq, DataBankResult<PrefetchedBatch<T>>>,
}

impl<T> ProducerState<T>
where
    T: DataValue,
{
    fn new(prefetch_step: usize, worker_count: usize) -> Self {
        let response_limit = prefetch_step.min(worker_count.saturating_sub(1).max(1));
        Self {
            next_read_seq: 0,
            next_emit_seq: 0,
            source_done: false,
            stop_reading: false,
            outstanding: 0,
            active_requests: 0,
            active_responses: 0,
            response_limit,
            planned_ready: VecDeque::new(),
            completed: BTreeMap::new(),
        }
    }

    fn is_finished(&self) -> bool {
        self.source_done
            && self.outstanding == 0
            && self.active_requests == 0
            && self.active_responses == 0
            && self.planned_ready.is_empty()
    }
}

enum ProducerEvent<T>
where
    T: DataValue,
{
    Planned(Result<PlannedMessage, flume::RecvError>),
    Done(Result<DoneMessage<T>, flume::RecvError>),
}

struct ActiveBatchGuard {
    seq: BatchSeq,
    registry: Arc<PrefetchCancelRegistry>,
}

impl Drop for ActiveBatchGuard {
    fn drop(&mut self) {
        self.registry.unregister(self.seq);
    }
}

impl<T, I> PrefetchProducer<T, I>
where
    T: DataValue,
    I: Iterator,
    I::Item: AsRef<[usize]>,
{
    fn run(mut self) {
        if panic::catch_unwind(AssertUnwindSafe(|| self.run_pipeline())).is_err() {
            self.cancel.cancel_all();
            let _ = self.tx.send(Err(DataBankError::PrefetchProducerPanic));
        }
        self.profiler.print_summary();
    }

    fn run_pipeline(&mut self) {
        let (planned_tx, planned_rx) = flume::unbounded();
        let (done_tx, done_rx) = flume::unbounded();
        let mut state = ProducerState::<T>::new(self.prefetch_step, self.compute.worker_count());

        loop {
            if self.cancel.is_cancelled() {
                break;
            }

            let mut progressed = false;
            progressed |= self.fill_request_window(&mut state, &planned_tx);
            progressed |= self.drain_messages(&mut state, &planned_rx, &done_rx);
            progressed |= self.submit_ready_responses(&mut state, &done_tx);
            let (keep_running, emitted) = self.emit_ready(&mut state);
            progressed |= emitted;
            if !keep_running {
                break;
            }

            if state.is_finished() {
                break;
            }

            if !progressed && !self.wait_for_event(&mut state, &planned_rx, &done_rx) {
                break;
            }
        }

        self.cancel.cancel_all();
        state.planned_ready.clear();
    }

    fn fill_request_window(
        &mut self,
        state: &mut ProducerState<T>,
        planned_tx: &flume::Sender<PlannedMessage>,
    ) -> bool {
        let mut progressed = false;
        while !state.source_done
            && !state.stop_reading
            && state.outstanding < self.prefetch_step
            && !self.cancel.is_cancelled()
        {
            let next_started = self.profiler.start();
            let next = self.batch_source.next();
            self.profiler.record_batch_source_next(next_started);
            let Some(cells) = next else {
                state.source_done = true;
                progressed = true;
                break;
            };
            let seq = state.next_read_seq;
            state.next_read_seq += 1;
            state.outstanding += 1;
            state.active_requests += 1;

            let job = make_prefetch_request_job(
                seq,
                self.access.clone(),
                Arc::clone(&self.dataset),
                cells.as_ref().to_vec(),
                self.gene_axis.clone(),
                self.access_config,
                Arc::clone(&self.cancel),
                planned_tx.clone(),
                self.profiler.clone(),
                self.profiler.start(),
            );
            let submit_started = self.profiler.start();
            let submit_result = self.compute.submit_request(job);
            self.profiler.record_submit_request(submit_started);
            if let Err(err) = submit_result {
                state.active_requests = state.active_requests.saturating_sub(1);
                state.completed.insert(seq, Err(err));
                state.stop_reading = true;
            }
            progressed = true;
        }
        progressed
    }

    fn drain_messages(
        &self,
        state: &mut ProducerState<T>,
        planned_rx: &flume::Receiver<PlannedMessage>,
        done_rx: &flume::Receiver<DoneMessage<T>>,
    ) -> bool {
        let mut progressed = false;
        while let Ok(message) = planned_rx.try_recv() {
            self.handle_planned_message(state, message);
            progressed = true;
        }
        while let Ok(message) = done_rx.try_recv() {
            self.handle_done_message(state, message);
            progressed = true;
        }
        progressed
    }

    fn wait_for_event(
        &self,
        state: &mut ProducerState<T>,
        planned_rx: &flume::Receiver<PlannedMessage>,
        done_rx: &flume::Receiver<DoneMessage<T>>,
    ) -> bool {
        if state.active_requests == 0 && state.active_responses == 0 {
            return false;
        }

        let wait_started = self.profiler.start();
        let event = match (state.active_requests > 0, state.active_responses > 0) {
            (true, true) => flume::Selector::new()
                .recv(planned_rx, ProducerEvent::Planned)
                .recv(done_rx, ProducerEvent::Done)
                .wait(),
            (true, false) => ProducerEvent::Planned(planned_rx.recv()),
            (false, true) => ProducerEvent::Done(done_rx.recv()),
            (false, false) => return false,
        };
        self.profiler.record_coordinator_wait(wait_started);

        match event {
            ProducerEvent::Planned(Ok(message)) => {
                self.handle_planned_message(state, message);
                true
            }
            ProducerEvent::Done(Ok(message)) => {
                self.handle_done_message(state, message);
                true
            }
            ProducerEvent::Planned(Err(_)) | ProducerEvent::Done(Err(_)) => false,
        }
    }

    fn handle_planned_message(&self, state: &mut ProducerState<T>, message: PlannedMessage) {
        state.active_requests = state.active_requests.saturating_sub(1);
        match message.result {
            Ok(planned) => {
                if self.cancel.is_cancelled() {
                    self.cancel.unregister(planned.seq);
                } else {
                    state.planned_ready.push_back(*planned);
                }
            }
            Err(err) => {
                if !matches!(err, DataBankError::PrefetchCancelled) || !self.cancel.is_cancelled() {
                    state.stop_reading = true;
                }
                state.completed.insert(message.seq, Err(err));
            }
        }
    }

    fn handle_done_message(&self, state: &mut ProducerState<T>, message: DoneMessage<T>) {
        state.active_responses = state.active_responses.saturating_sub(1);
        if message.result.is_err()
            && (!matches!(&message.result, Err(DataBankError::PrefetchCancelled))
                || !self.cancel.is_cancelled())
        {
            state.stop_reading = true;
        }
        state.completed.insert(message.seq, message.result);
    }

    fn submit_ready_responses(
        &self,
        state: &mut ProducerState<T>,
        done_tx: &flume::Sender<DoneMessage<T>>,
    ) -> bool {
        let mut progressed = false;
        while state.active_responses < state.response_limit && !self.cancel.is_cancelled() {
            let Some(planned) = state.planned_ready.pop_front() else {
                break;
            };
            let seq = planned.seq;
            state.active_responses += 1;
            let job = make_prefetch_response_job(
                planned,
                self.access.clone(),
                Arc::clone(&self.compute),
                self.access_config,
                self.gene_axis.clone(),
                Arc::clone(&self.cancel),
                done_tx.clone(),
                self.profiler.clone(),
                self.profiler.start(),
            );
            let submit_started = self.profiler.start();
            let submit_result = self.compute.submit_response(job);
            self.profiler.record_submit_response(submit_started);
            if let Err(err) = submit_result {
                state.active_responses = state.active_responses.saturating_sub(1);
                self.cancel.unregister(seq);
                state.completed.insert(seq, Err(err));
                state.stop_reading = true;
            }
            progressed = true;
        }
        progressed
    }

    fn emit_ready(&self, state: &mut ProducerState<T>) -> (bool, bool) {
        let mut emitted = false;
        while let Some(result) = state.completed.remove(&state.next_emit_seq) {
            emitted = true;
            state.outstanding = state.outstanding.saturating_sub(1);
            state.next_emit_seq += 1;
            match result {
                Ok(batch) => {
                    let send_started = self.profiler.start();
                    let send_result = self.tx.send(Ok(batch));
                    self.profiler.record_output_send(send_started);
                    if send_result.is_err() {
                        self.cancel.cancel_all();
                        return (false, emitted);
                    }
                    self.profiler.inc_emitted_batch();
                }
                Err(DataBankError::PrefetchCancelled) if self.cancel.is_cancelled() => {
                    return (false, emitted);
                }
                Err(err) => {
                    let send_started = self.profiler.start();
                    if self.tx.send(Err(err)).is_ok() {
                        self.profiler.inc_emitted_batch();
                    }
                    self.profiler.record_output_send(send_started);
                    self.cancel.cancel_all();
                    return (false, emitted);
                }
            }
        }
        (true, emitted)
    }
}

#[allow(clippy::too_many_arguments)]
fn make_prefetch_request_job(
    seq: BatchSeq,
    access: AccessHandle,
    dataset: Arc<Dataset>,
    cells: Vec<usize>,
    gene_axis: GeneAxisPlan,
    access_config: ScheduledAccessConfig,
    registry: Arc<PrefetchCancelRegistry>,
    planned_tx: flume::Sender<PlannedMessage>,
    profiler: ScheduledPrefetchProfiler,
    queued_at: Option<Instant>,
) -> ComputeJob {
    Box::new(move || {
        profiler.inc_request_job();
        profiler.record_request_queue_wait(queued_at);
        let total_started = profiler.start();
        let result =
            panic::catch_unwind(AssertUnwindSafe(|| -> DataBankResult<Box<PlannedBatch>> {
                if registry.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
                let plan_started = profiler.start();
                let planned = plan_batch_owned(dataset, cells, &gene_axis);
                profiler.record_request_plan(plan_started);
                let (plan, items) = planned?;
                if registry.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
                let cancel = PrefetchCancel::new(access.clone());
                let schedule_started = profiler.start();
                let scheduled_result = access.scheduled(items, access_config);
                profiler.record_request_schedule(schedule_started);
                let mut scheduled = scheduled_result?;
                scheduled.set_cancel_handle(Arc::clone(&cancel));
                registry.register(seq, Arc::clone(&cancel));
                Ok(Box::new(PlannedBatch {
                    seq,
                    plan,
                    scheduled,
                    cancel,
                }))
            }))
            .unwrap_or(Err(DataBankError::ComputeWorkerPanic));
        if result.is_err() {
            profiler.inc_request_error();
        }
        let send_started = profiler.start();
        if let Err(err) = planned_tx.send(PlannedMessage { seq, result }) {
            if let Ok(planned) = err.0.result {
                registry.unregister(planned.seq);
            }
        }
        profiler.record_request_send(send_started);
        profiler.record_request_total(total_started);
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn make_prefetch_response_job<T>(
    planned: PlannedBatch,
    access: AccessHandle,
    compute: Arc<DataBankComputePool>,
    access_config: ScheduledAccessConfig,
    gene_axis: GeneAxisPlan,
    registry: Arc<PrefetchCancelRegistry>,
    done_tx: flume::Sender<DoneMessage<T>>,
    profiler: ScheduledPrefetchProfiler,
    queued_at: Option<Instant>,
) -> ComputeJob
where
    T: DataValue,
{
    Box::new(move || {
        profiler.inc_response_job();
        profiler.record_response_queue_wait(queued_at);
        let total_started = profiler.start();
        let PlannedBatch {
            seq,
            plan,
            mut scheduled,
            cancel,
        } = planned;
        let _guard = ActiveBatchGuard {
            seq,
            registry: Arc::clone(&registry),
        };
        let result = panic::catch_unwind(AssertUnwindSafe(
            || -> DataBankResult<PrefetchedBatch<T>> {
                if registry.is_cancelled() || cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
                let batch = assemble_planned_batch(
                    &access,
                    compute.as_ref(),
                    &access_config,
                    &gene_axis,
                    &cancel,
                    &profiler,
                    plan,
                    &mut scheduled,
                )?;
                if scheduled.next().is_some() {
                    return Err(DataBankError::Io(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "scheduled prefetch returned extra output",
                    )));
                }
                Ok(batch)
            },
        ))
        .unwrap_or(Err(DataBankError::ComputeWorkerPanic));
        if result.is_err() {
            profiler.inc_response_error();
        }
        let _ = done_tx.send(DoneMessage { seq, result });
        profiler.record_response_total(total_started);
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn assemble_planned_batch<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    plan: BatchPlan,
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let assemble_started = profiler.start();
    let result = match plan {
        BatchPlan::Dense {
            cells,
            segments,
            groups,
            num_genes,
        } => assemble_dense_prefetch_batch(
            access, compute, gene_axis, cancel, profiler, cells, segments, groups, num_genes,
            scheduled,
        ),
        BatchPlan::Sparse {
            cells,
            plan,
            dataset,
        } => {
            if let Dataset::SparseCsr(dataset) = dataset.as_ref() {
                assemble_sparse_prefetch_batch(
                    access,
                    compute,
                    access_config,
                    gene_axis,
                    cancel,
                    profiler,
                    cells,
                    plan,
                    dataset,
                    scheduled,
                )
            } else {
                Err(DataBankError::InvalidArrayMeta(
                    "sparse prefetch plan carried non-CSR dataset".to_string(),
                ))
            }
        }
    };
    profiler.record_assemble_total(assemble_started);
    result
}

#[allow(clippy::too_many_arguments)]
fn assemble_dense_prefetch_batch<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    cells: Vec<usize>,
    segments: Vec<DenseSegment>,
    groups: Vec<DenseReadGroup>,
    num_genes: usize,
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let output_genes = gene_axis.output_genes(num_genes);
    let total = cells
        .len()
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output length overflow".to_string()))?;
    let alloc_started = profiler.start();
    let mut buffer = vec![T::zero(); total];
    profiler.record_alloc(alloc_started);
    let output_bytes = buffer
        .len()
        .checked_mul(mem::size_of::<T>())
        .ok_or_else(|| {
            DataBankError::InvalidConfig("dense output byte length overflow".to_string())
        })?;
    if compute.should_parallelize(cells.len(), output_bytes) {
        let mut loaded_groups = Vec::with_capacity(groups.len());
        for group in &groups {
            if cancel.is_cancelled() {
                return Err(DataBankError::PrefetchCancelled);
            }
            match &group.source {
                DenseGroupSource::AccessItem(_) => {
                    let bytes = next_scheduled_bytes(scheduled, profiler)?;
                    if bytes.len() != group.bytes {
                        return Err(DataBankError::InvalidArrayMeta(format!(
                            "dense group decoded length is {}, expected {}",
                            bytes.len(),
                            group.bytes
                        )));
                    }
                    loaded_groups.push(DenseLoadedGroup::Packed(Arc::from(
                        bytes.into_boxed_slice(),
                    )));
                }
                DenseGroupSource::Memory { .. } => {
                    let load_started = profiler.start();
                    let loaded = load_dense_group_for_parallel(access, group)?;
                    let loaded_bytes = match &loaded {
                        DenseLoadedGroup::Packed(bytes) => bytes.len(),
                        DenseLoadedGroup::DecodedSource { .. } => group.bytes,
                    };
                    profiler.record_memory_load(load_started, loaded_bytes);
                    loaded_groups.push(loaded);
                }
            }
        }
        let scatter_started = profiler.start();
        let scattered = try_scatter_dense_rows_parallel(
            compute,
            num_genes,
            output_genes,
            &segments,
            &groups,
            &loaded_groups,
            gene_axis.projection().cloned(),
            &mut buffer,
        )?;
        profiler.record_scatter(scatter_started);
        if scattered {
            return Ok(PrefetchedBatch {
                cells,
                buffer,
                num_genes: output_genes,
            });
        }
        let scatter_started = profiler.start();
        scatter_dense_group_range_checked::<T>(
            num_genes,
            output_genes,
            &segments,
            &groups,
            &loaded_groups,
            gene_axis.projection(),
            0,
            groups.len(),
            buffer.as_mut_ptr() as usize,
            buffer.len(),
        )?;
        profiler.record_scatter(scatter_started);
        return Ok(PrefetchedBatch {
            cells,
            buffer,
            num_genes: output_genes,
        });
    }
    for group in &groups {
        if cancel.is_cancelled() {
            return Err(DataBankError::PrefetchCancelled);
        }
        match &group.source {
            DenseGroupSource::AccessItem(_) => {
                let bytes = next_scheduled_bytes(scheduled, profiler)?;
                let scatter_started = profiler.start();
                if gene_axis.projection().is_some() {
                    scatter_dense_group_projected(
                        num_genes,
                        output_genes,
                        &segments,
                        group,
                        &bytes,
                        gene_axis,
                        &mut buffer,
                    )?;
                } else {
                    scatter_dense_group(num_genes, &segments, group, &bytes, &mut buffer)?;
                }
                profiler.record_scatter(scatter_started);
            }
            DenseGroupSource::Memory { .. } => {
                let scatter_started = profiler.start();
                let scattered = if gene_axis.projection().is_some() {
                    try_scatter_dense_memory_identity_group_projected(
                        num_genes,
                        output_genes,
                        &segments,
                        group,
                        gene_axis,
                        &mut buffer,
                    )?
                } else {
                    try_scatter_dense_memory_identity_group(
                        num_genes,
                        &segments,
                        group,
                        &mut buffer,
                    )?
                };
                profiler.record_scatter(scatter_started);
                if !scattered {
                    let load_started = profiler.start();
                    let bytes = load_dense_group(access, group)?;
                    profiler.record_memory_load(load_started, bytes.len());
                    let scatter_started = profiler.start();
                    if gene_axis.projection().is_some() {
                        scatter_dense_group_projected(
                            num_genes,
                            output_genes,
                            &segments,
                            group,
                            &bytes,
                            gene_axis,
                            &mut buffer,
                        )?;
                    } else {
                        scatter_dense_group(num_genes, &segments, group, &bytes, &mut buffer)?;
                    }
                    profiler.record_scatter(scatter_started);
                }
            }
        }
    }
    Ok(PrefetchedBatch {
        cells,
        buffer,
        num_genes: output_genes,
    })
}

#[allow(clippy::too_many_arguments)]
fn assemble_sparse_prefetch_batch<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    cells: Vec<usize>,
    plan: SparseBatchPlan,
    dataset: &SparseCsrDataset,
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let output_genes = gene_axis.output_genes(dataset.num_genes);
    let total = cells
        .len()
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output length overflow".to_string()))?;
    let alloc_started = profiler.start();
    let mut buffer = vec![T::zero(); total];
    profiler.record_alloc(alloc_started);
    let index_bytes = load_sparse_prefetch_indices(access, cancel, profiler, &plan, scheduled)?;

    if gene_axis.projection().is_some() {
        let scatter_started = profiler.start();
        load_sparse_data_groups_and_scatter_projected_checked(
            access,
            compute,
            access_config,
            dataset,
            &plan,
            index_bytes,
            gene_axis,
            Some(cancel),
            &mut buffer,
        )?;
        profiler.record_scatter(scatter_started);
    } else {
        scatter_sparse_prefetch_data(
            access,
            compute,
            cancel,
            profiler,
            dataset,
            &plan,
            index_bytes,
            scheduled,
            &mut buffer,
        )?;
    }

    Ok(PrefetchedBatch {
        cells,
        buffer,
        num_genes: output_genes,
    })
}

fn load_sparse_prefetch_indices<J>(
    access: &AccessHandle,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    plan: &SparseBatchPlan,
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<Vec<u8>>
where
    J: Iterator<Item = AccessItem>,
{
    let alloc_started = profiler.start();
    let mut index_bytes = zeroed_byte_vec(plan.index_bytes);
    profiler.record_alloc(alloc_started);
    for group in &plan.index_groups {
        if cancel.is_cancelled() {
            return Err(DataBankError::PrefetchCancelled);
        }
        match &group.source {
            SparseGroupSource::AccessItem(_) => {
                let bytes = next_scheduled_bytes(scheduled, profiler)?;
                let scatter_started = profiler.start();
                copy_sparse_group_to_index_buffer(
                    &plan.index_pieces,
                    group,
                    &bytes,
                    &mut index_bytes,
                )?;
                profiler.record_scatter(scatter_started);
            }
            SparseGroupSource::Memory { .. } => {
                let scatter_started = profiler.start();
                if try_copy_sparse_memory_identity_group_to_index_buffer(
                    &plan.index_pieces,
                    group,
                    &mut index_bytes,
                )? {
                    profiler.record_scatter(scatter_started);
                    continue;
                }
                profiler.record_scatter(scatter_started);
                let load_started = profiler.start();
                let bytes = load_sparse_group(access, group)?;
                profiler.record_memory_load(load_started, bytes.len());
                let scatter_started = profiler.start();
                copy_sparse_group_to_index_buffer(
                    &plan.index_pieces,
                    group,
                    &bytes,
                    &mut index_bytes,
                )?;
                profiler.record_scatter(scatter_started);
            }
        }
    }
    Ok(index_bytes)
}

#[allow(clippy::too_many_arguments)]
fn scatter_sparse_prefetch_data<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Vec<u8>,
    scheduled: &mut ScheduledAccess<J>,
    buffer: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let output_rows = row_count_for_width(buffer.len(), dataset.num_genes);
    let output_bytes = buffer
        .len()
        .checked_mul(mem::size_of::<T>())
        .ok_or_else(|| {
            DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
        })?;
    if compute.should_parallelize(output_rows, output_bytes) {
        let mut data_group_bytes = Vec::with_capacity(plan.data_groups.len());
        for group in &plan.data_groups {
            if cancel.is_cancelled() {
                return Err(DataBankError::PrefetchCancelled);
            }
            let bytes = match &group.source {
                SparseGroupSource::AccessItem(_) => next_scheduled_bytes(scheduled, profiler)?,
                SparseGroupSource::Memory { .. } => {
                    let load_started = profiler.start();
                    let bytes = load_sparse_group(access, group)?;
                    profiler.record_memory_load(load_started, bytes.len());
                    bytes
                }
            };
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            data_group_bytes.push(Arc::from(bytes.into_boxed_slice()));
        }
        let index_bytes_arc: Arc<[u8]> = Arc::from(index_bytes.into_boxed_slice());
        let fallback_index = Arc::clone(&index_bytes_arc);
        let fallback_data = data_group_bytes.clone();
        let scatter_started = profiler.start();
        let scattered = try_scatter_sparse_rows_parallel_checked(
            compute,
            dataset,
            plan,
            index_bytes_arc,
            data_group_bytes,
            dataset.num_genes,
            None,
            buffer,
        )?;
        profiler.record_scatter(scatter_started);
        if scattered {
            return Ok(());
        }
        for (group, bytes) in plan.data_groups.iter().zip(fallback_data.iter()) {
            let scatter_started = profiler.start();
            scatter_sparse_data_group_checked(
                dataset,
                &plan.data_pieces,
                group,
                fallback_index.as_ref(),
                bytes.as_ref(),
                buffer,
            )?;
            profiler.record_scatter(scatter_started);
        }
        return Ok(());
    }
    for group in &plan.data_groups {
        if cancel.is_cancelled() {
            return Err(DataBankError::PrefetchCancelled);
        }
        match &group.source {
            SparseGroupSource::AccessItem(_) => {
                let bytes = next_scheduled_bytes(scheduled, profiler)?;
                let scatter_started = profiler.start();
                scatter_sparse_data_group_checked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    &bytes,
                    buffer,
                )?;
                profiler.record_scatter(scatter_started);
            }
            SparseGroupSource::Memory { .. } => {
                let scatter_started = profiler.start();
                if try_scatter_sparse_memory_identity_data_group_checked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    buffer,
                )? {
                    profiler.record_scatter(scatter_started);
                    continue;
                }
                profiler.record_scatter(scatter_started);
                let load_started = profiler.start();
                let bytes = load_sparse_group(access, group)?;
                profiler.record_memory_load(load_started, bytes.len());
                let scatter_started = profiler.start();
                scatter_sparse_data_group_checked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    &bytes,
                    buffer,
                )?;
                profiler.record_scatter(scatter_started);
            }
        }
    }
    Ok(())
}

fn next_scheduled_bytes<J>(
    scheduled: &mut ScheduledAccess<J>,
    profiler: &ScheduledPrefetchProfiler,
) -> DataBankResult<Vec<u8>>
where
    J: Iterator<Item = AccessItem>,
{
    let drain_started = profiler.start();
    let result = match scheduled.next() {
        Some(Ok(bytes)) => Ok(bytes),
        Some(Err(err)) => Err(DataBankError::Io(err)),
        None => Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "scheduled prefetch ended before batch was complete",
        ))),
    };
    profiler.record_scheduled_drain(drain_started, result.as_ref().map_or(0, Vec::len));
    result
}

/// Build a scheduled prefetcher over `batch_source`.
///
/// Each item from `batch_source` is one batch of cell indices for `dataset`.
/// The prefetcher plans batches one at a time, streams their access items into
/// the access scheduler, and assembles decoded results in a background
/// producer. Completed batches are stored in a bounded queue of depth
/// `prefetch_step`; `next()` only pops that completed queue and blocks only
/// when the producer cannot keep up.
pub fn prefetch_cells_scheduled<T, I>(
    access: &AccessHandle,
    compute: Arc<DataBankComputePool>,
    dataset: Arc<Dataset>,
    batch_source: I,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: AsRef<[usize]> + Send,
{
    config.validate().map_err(DataBankError::InvalidConfig)?;
    if dataset.data_dtype() != T::DTYPE {
        return Err(DataBankError::UnsupportedDType {
            dtype: T::DTYPE,
            context: "scheduled prefetch output",
        });
    }
    spawn_prefetch_cells(
        access.clone(),
        compute,
        dataset,
        batch_source,
        GeneAxisPlan::dataset_order(),
        config,
    )
}

pub fn prefetch_cells_scheduled_by_gene_names<T, I, G>(
    access: &AccessHandle,
    compute: Arc<DataBankComputePool>,
    dataset: Arc<Dataset>,
    batch_source: I,
    gene_names: &[G],
    missing: MissingGenePolicy,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: AsRef<[usize]> + Send,
    G: AsRef<str>,
{
    config.validate().map_err(DataBankError::InvalidConfig)?;
    if dataset.data_dtype() != T::DTYPE {
        return Err(DataBankError::UnsupportedDType {
            dtype: T::DTYPE,
            context: "scheduled prefetch output",
        });
    }
    let gene_axis = GeneAxisPlan::requested(dataset.as_ref(), gene_names, missing)?;
    spawn_prefetch_cells(
        access.clone(),
        compute,
        dataset,
        batch_source,
        gene_axis,
        config,
    )
}

fn spawn_prefetch_cells<T, I>(
    access: AccessHandle,
    compute: Arc<DataBankComputePool>,
    dataset: Arc<Dataset>,
    batch_source: I,
    gene_axis: GeneAxisPlan,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: AsRef<[usize]> + Send,
{
    let output_names = gene_axis.output_names(dataset.as_ref()).to_vec();
    let retained_dataset = Arc::clone(&dataset);
    let prefetch_step = config.prefetch_step;
    let (tx, rx) = flume::bounded(prefetch_step);
    let cancel = PrefetchCancelRegistry::new();
    let profiler = ScheduledPrefetchProfiler::from_env();
    let producer = PrefetchProducer {
        access,
        compute,
        dataset,
        batch_source,
        access_config: config.access,
        gene_axis,
        tx,
        cancel: Arc::clone(&cancel),
        prefetch_step,
        profiler,
    };
    let handle = thread::Builder::new()
        .name("databank-prefetch-producer".to_string())
        .spawn(move || producer.run())?;
    Ok(PrefetchCells {
        rx: Some(rx),
        output_names,
        _dataset: retained_dataset,
        prefetch_step,
        cancel,
        producer: Some(handle),
    })
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc, Condvar, Mutex};

    use super::super::array::{
        Array, ArrayGrid, Chunk, ChunkSource, EdgeChunkLayout, RegisteredFile,
    };
    use super::super::dataset::{Dataset, Dense2DDataset, SparseCsrDataset};
    use crate::access::SliceSpec;
    use crate::codecs::{ChunkCodec, CodecError, CodecResult, SharedCodec, UncompressedCodec};
    use crate::databank::{
        ArrayCodecMeta, ArrayMeta, ArrayOrder, ChunkStoreMeta, DType, DataBank, DataBankConfig,
        Dense1DMeta, Dense2DMeta, DirectoryChunkLocationMeta, FileChunkLocation, MissingGenePolicy,
        PrefetchedBatch, ScheduledPrefetchConfig, SparseCsrDatasetMeta,
    };

    static FILE_SEQ: AtomicU64 = AtomicU64::new(0);

    fn parallel_config() -> DataBankConfig {
        let mut config = DataBankConfig::default();
        config.fill_config.parallel = true;
        config.fill_config.num_workers = 2;
        config.fill_config.min_parallel_rows = 1;
        config.fill_config.min_parallel_bytes = 1;
        config
    }

    fn write_chunk_file(chunks: Vec<Arc<[u8]>>) -> (PathBuf, Vec<FileChunkLocation>) {
        let mut bytes = Vec::new();
        let mut locations = Vec::new();
        for chunk in &chunks {
            let offset = bytes.len();
            bytes.extend_from_slice(chunk);
            locations.push(FileChunkLocation {
                offset: offset as u64,
                len: chunk.len(),
            });
        }
        let seq = FILE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("scdata-prefetch-{}-{seq}", std::process::id()));
        std::fs::write(&path, &bytes).expect("write temp chunk file");
        (path, locations)
    }

    fn arc_u32_bytes(values: &[u32]) -> Arc<[u8]> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes.into()
    }

    fn arc_u64_bytes(values: &[u64]) -> Arc<[u8]> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes.into()
    }

    fn arc_f32_bytes(values: &[f32]) -> Arc<[u8]> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes.into()
    }

    /// Encode ``values`` as little-endian bytes, zero-padded to ``chunk_len``
    /// elements — standard zarr edge-chunk layout (every decoded chunk is
    /// exactly ``chunk_len`` elements; trailing padding is never read because
    /// the CSR indptr bounds the logical range).
    fn padded_u32_bytes(values: &[u32], chunk_len: usize) -> Arc<[u8]> {
        let mut bytes = vec![0u8; chunk_len * std::mem::size_of::<u32>()];
        for (i, value) in values.iter().enumerate() {
            let offset = i * std::mem::size_of::<u32>();
            bytes[offset..offset + std::mem::size_of::<u32>()]
                .copy_from_slice(&value.to_ne_bytes());
        }
        bytes.into()
    }

    #[test]
    fn memory_none_group_copies_raw_slices_directly() {
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let bytes = b"abcdefghijkl";
        let slice = SliceSpec::from_triples(vec![0, 2, 5, 3, 8, 11]).expect("slice spec");

        let sliced = super::load_memory_group(bytes, &codec, bytes.len(), false, &slice)
            .expect("slice raw none codec");

        assert_eq!(sliced, b"cdeijk");
    }

    #[test]
    fn memory_none_group_preserves_sparse_slice_gaps() {
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let bytes = b"abcdefghijkl";
        let slice = SliceSpec::from_triples(vec![2, 2, 5]).expect("slice spec");

        let sliced = super::load_memory_group(bytes, &codec, bytes.len(), false, &slice)
            .expect("slice raw none codec");

        assert_eq!(sliced, b"\0\0cde");
    }

    #[test]
    fn memory_none_group_preserves_size_mismatch_error() {
        let codec: SharedCodec = Arc::new(UncompressedCodec);

        let err = super::load_memory_group(b"abc", &codec, 4, false, &SliceSpec::Full)
            .expect_err("size mismatch should be surfaced");

        assert!(matches!(
            err,
            crate::databank::DataBankError::Codec(CodecError::SizeMismatch {
                expected: 4,
                actual: 3,
                ..
            })
        ));
    }

    #[test]
    fn zero_length_file_chunk_reads_as_zero_fill_without_opening_path() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let (path, locations) = write_chunk_file(vec![arc_u32_bytes(&[11, 22])]);
        let missing_path = path.with_extension("missing");
        assert!(!missing_path.exists());

        let id = bank
            .register_dense_2d(Dense2DMeta {
                gene_names: vec!["g0".to_string(), "g1".to_string()],
                data: ArrayMeta {
                    shape: vec![2, 2],
                    chunk_shape: vec![1, 2],
                    chunk_grid_shape: vec![2, 1],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Directory {
                        locations: vec![
                            DirectoryChunkLocationMeta {
                                path,
                                offset: locations[0].offset,
                                len: locations[0].len,
                            },
                            DirectoryChunkLocationMeta {
                                path: missing_path,
                                offset: 0,
                                len: 0,
                            },
                        ],
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register dense with missing chunk");

        let mut out = vec![99u32; 4];
        bank.access_cells(id, &[0, 1], &mut out, None)
            .expect("access dense with missing chunk");

        assert_eq!(out, vec![11, 22, 0, 0]);
    }

    #[test]
    fn dense_1d_memory_fast_path_rejects_oversized_chunk() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = bank
            .register_dense_1d(Dense1DMeta {
                gene_names: vec!["g0".to_string(), "g1".to_string()],
                data: ArrayMeta {
                    shape: vec![6],
                    chunk_shape: vec![4],
                    chunk_grid_shape: vec![2],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory {
                        chunks: vec![arc_u32_bytes(&[1, 2, 3, 4, 999]), arc_u32_bytes(&[5, 6])],
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register dense 1d");

        let mut out = vec![0u32; 2];
        let err = bank
            .access_cells(id, &[0], &mut out, None)
            .expect_err("oversized chunk should fail");

        assert!(matches!(
            err,
            crate::databank::DataBankError::Codec(CodecError::SizeMismatch {
                expected: 16,
                actual: 20,
                ..
            })
        ));
    }

    #[test]
    fn empty_gene_name_is_rejected_at_registration() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let err = bank
            .register_dense_2d(Dense2DMeta {
                gene_names: vec![String::new()],
                data: ArrayMeta {
                    shape: vec![1, 1],
                    chunk_shape: vec![1, 1],
                    chunk_grid_shape: vec![1, 1],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory {
                        chunks: vec![arc_u32_bytes(&[1])],
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect_err("empty gene name should fail");

        assert!(matches!(
            err,
            crate::databank::DataBankError::InvalidArrayMeta(message)
                if message.contains("must not be empty")
        ));
    }

    #[test]
    fn unregister_failure_unloads_dataset() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_file(&mut bank, 2, 2, 1);
        let file_id = match bank.registry.get(id).expect("dataset") {
            Dataset::Dense2D(dataset) => match &dataset.data.chunks[0].source {
                ChunkSource::File { file, .. } => file.id,
                _ => panic!("expected file-backed dense dataset"),
            },
            _ => panic!("expected dense dataset"),
        };
        bank.io_pool
            .unregister_file(file_id)
            .expect("manual unregister");

        let err = bank
            .unregister(id)
            .expect_err("second unregister should fail");

        assert!(matches!(err, crate::databank::DataBankError::Io(_)));
        assert!(matches!(
            bank.dataset_genes(id),
            Err(crate::databank::DataBankError::DatasetUnloaded(unloaded)) if unloaded == id
        ));
    }

    #[test]
    fn memory_dense_2d_identity_scatter_handles_partial_column_chunks() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_memory(&mut bank, 5, 5, 2, 3);
        let cells = vec![4, 0, 3];
        let expected = expected_dense_rows(&cells, 5);

        let mut out = vec![0u32; cells.len() * 5];
        bank.access_cells(id, &cells, &mut out, None)
            .expect("memory dense access");
        assert_eq!(out, expected);

        let batches: Vec<Vec<usize>> = vec![vec![4, 0], vec![3]];
        let prefetch = bank
            .prefetch_cells_scheduled::<u32, _>(
                id,
                batches.clone(),
                ScheduledPrefetchConfig::default(),
            )
            .expect("memory dense prefetch");
        let collected: Vec<Vec<u32>> = prefetch
            .map(|batch| batch.expect("prefetch batch").buffer)
            .collect();
        let expected_batches: Vec<Vec<u32>> = batches
            .iter()
            .map(|batch| expected_dense_rows(batch, 5))
            .collect();
        assert_eq!(collected, expected_batches);
    }

    #[test]
    fn csr_batch_scatter_preserves_requested_cell_order_across_grouped_chunks() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = bank
            .register_sparse_csr(SparseCsrDatasetMeta {
                gene_names: (0..5).map(|idx| format!("g{idx}")).collect(),
                indptr: vec![0, 2, 5, 6],
                indices: u32_array_meta(vec![
                    padded_u32_bytes(&[1, 3, 0, 4], 4),
                    padded_u32_bytes(&[2, 3], 4),
                ]),
                data: u32_array_meta(vec![
                    padded_u32_bytes(&[10, 30, 100, 400], 4),
                    padded_u32_bytes(&[200, 3000], 4),
                ]),
                index_dtype: DType::U32,
                num_cells: 3,
                num_genes: 5,
            })
            .expect("register CSR");

        let mut checked = vec![0u32; 10];
        bank.access_cells(id, &[1, 0], &mut checked, None)
            .expect("checked CSR access");
        assert_eq!(checked, vec![100, 0, 200, 0, 400, 0, 10, 0, 30, 0]);

        let mut unchecked = vec![0u32; 10];
        unsafe {
            bank.access_cells_unchecked(id, &[1, 0], &mut unchecked, None)
                .expect("unchecked CSR access");
        }
        assert_eq!(unchecked, checked);
    }

    #[test]
    fn compressed_csr_repeated_chunk_batch_decodes_each_chunk_once_and_preserves_order() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let (id, decodes) = register_counted_csr_memory(&mut bank, 8, 16, 2, 8);
        let cells = vec![3, 0, 1, 3];

        let out = bank
            .access_cells_alloc::<u32>(id, &cells)
            .expect("counted CSR access");

        assert_eq!(out, expected_counted_csr_rows(&cells, 16, 2));
        assert_eq!(
            decodes.load(Ordering::SeqCst),
            2,
            "one repeated index chunk and one repeated data chunk should decode once each"
        );
    }

    #[test]
    fn compressed_csr_scattered_batch_decodes_only_touched_chunks_and_preserves_order() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let (id, decodes) = register_counted_csr_memory(&mut bank, 8, 16, 2, 2);
        let cells = vec![7, 0, 5, 3];

        let out = bank
            .access_cells_alloc::<u32>(id, &cells)
            .expect("counted CSR access");

        assert_eq!(out, expected_counted_csr_rows(&cells, 16, 2));
        assert_eq!(
            decodes.load(Ordering::SeqCst),
            cells.len() * 2,
            "scattered single-hit cells should decode one index and one data chunk per touched cell"
        );
    }

    #[test]
    fn memory_csr_direct_handles_mismatched_chunk_boundaries_and_index_width() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = bank
            .register_sparse_csr(SparseCsrDatasetMeta {
                gene_names: (0..6).map(|idx| format!("g{idx}")).collect(),
                indptr: vec![0, 3, 6],
                indices: ArrayMeta {
                    shape: vec![6],
                    chunk_shape: vec![3],
                    chunk_grid_shape: vec![2],
                    dtype: DType::U64,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory {
                        chunks: vec![arc_u64_bytes(&[0, 2, 5]), arc_u64_bytes(&[1, 3, 4])],
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
                data: ArrayMeta {
                    shape: vec![6],
                    chunk_shape: vec![2],
                    chunk_grid_shape: vec![3],
                    dtype: DType::F32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory {
                        chunks: vec![
                            arc_f32_bytes(&[1.0, 2.0]),
                            arc_f32_bytes(&[5.0, 10.0]),
                            arc_f32_bytes(&[30.0, 40.0]),
                        ],
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
                index_dtype: DType::U64,
                num_cells: 2,
                num_genes: 6,
            })
            .expect("register CSR");

        let checked = bank
            .access_cells_alloc::<f32>(id, &[1, 0])
            .expect("checked CSR access");
        assert_eq!(
            checked,
            vec![
                0.0, 10.0, 0.0, 30.0, 40.0, 0.0, // cell 1
                1.0, 0.0, 2.0, 0.0, 0.0, 5.0, // cell 0
            ]
        );

        let mut unchecked = vec![0.0f32; checked.len()];
        unsafe {
            bank.access_cells_unchecked(id, &[1, 0], &mut unchecked, None)
                .expect("unchecked CSR access");
        }
        assert_eq!(unchecked, checked);
    }

    #[test]
    fn access_cells_owned_matches_access_cells_dense_2d() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_file(&mut bank, 4, 4, 2);

        let cells = vec![3, 0, 2];
        let mut borrowed = vec![0u32; cells.len() * 4];
        bank.access_cells(id, &cells, &mut borrowed, None)
            .expect("checked dense access");
        let owned = bank
            .access_cells_alloc::<u32>(id, &cells)
            .expect("owned dense access");
        assert_eq!(owned, borrowed);
    }

    #[test]
    fn access_cells_by_gene_names_reorders_dense_and_zero_fills_missing() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_memory(&mut bank, 4, 5, 2, 2);
        let cells = vec![2, 0];
        let genes = ["g3", "missing", "g1"];
        let mut out = vec![99u32; cells.len() * genes.len()];
        let mut names = vec![crate::databank::GeneNameView::empty(); genes.len()];

        bank.access_cells_by_gene_names(
            id,
            &cells,
            &genes,
            &mut out,
            Some(&mut names),
            MissingGenePolicy::Zero,
        )
        .expect("projected dense access");

        assert_eq!(out, vec![203, 0, 201, 3, 0, 1]);
        assert!(!names[0].is_empty());
        assert!(names[1].is_empty());
        assert!(!names[2].is_empty());
    }

    #[test]
    fn projected_dense_segments_keep_only_selected_source_ranges() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_memory(&mut bank, 1, 8, 1, 8);
        let dataset = bank.registry.get(id).expect("dataset");
        let dense = match dataset {
            Dataset::Dense2D(dense) => dense,
            _ => panic!("expected dense 2d dataset"),
        };
        let genes = ["g6", "g1", "g2"];
        let gene_axis = super::GeneAxisPlan::requested(dataset, &genes, MissingGenePolicy::Zero)
            .expect("gene projection");
        let projection = match gene_axis {
            super::GeneAxisPlan::Requested(projection) => projection,
            super::GeneAxisPlan::DatasetOrder => panic!("subset should remain projected"),
        };

        let segments =
            super::plan::plan_dense_2d_selected_sources(dense, &[0], &projection.selected_sources)
                .expect("selected dense plan");

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].output_col_start, 1);
        assert_eq!(segments[0].output_cols, 2);
        assert_eq!(segments[0].source.start, 4);
        assert_eq!(segments[0].source.end, 12);
        assert_eq!(segments[1].output_col_start, 6);
        assert_eq!(segments[1].output_cols, 1);
        assert_eq!(segments[1].source.start, 24);
        assert_eq!(segments[1].source.end, 28);

        let out = bank
            .access_cells_owned_by_gene_names::<u32, _>(id, &[0], &genes, MissingGenePolicy::Zero)
            .expect("projected dense access");
        assert_eq!(out, vec![6, 1, 2]);
    }

    #[test]
    fn projected_dense_1d_planning_visits_only_selected_ranges() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = bank
            .register_dense_1d(Dense1DMeta {
                gene_names: (0..8).map(|g| format!("g{g}")).collect(),
                data: ArrayMeta {
                    shape: vec![8],
                    chunk_shape: vec![4],
                    chunk_grid_shape: vec![2],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory {
                        chunks: vec![arc_u32_bytes(&[0, 1, 2, 3]), arc_u32_bytes(&[4, 5, 6, 7])],
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register dense 1d");
        let dataset = bank.registry.get(id).expect("dataset");
        let dense = match dataset {
            Dataset::Dense1D(dense) => dense,
            _ => panic!("expected dense 1d dataset"),
        };
        let genes = ["g6", "g1", "g2"];
        let gene_axis = super::GeneAxisPlan::requested(dataset, &genes, MissingGenePolicy::Zero)
            .expect("gene projection");
        let projection = match gene_axis {
            super::GeneAxisPlan::Requested(projection) => projection,
            super::GeneAxisPlan::DatasetOrder => panic!("subset should remain projected"),
        };

        let segments =
            super::plan::plan_dense_1d_selected_sources(dense, &[0], &projection.selected_sources)
                .expect("selected dense 1d plan");

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].output_col_start, 1);
        assert_eq!(segments[0].output_cols, 2);
        assert_eq!(segments[0].source.start, 4);
        assert_eq!(segments[0].source.end, 12);
        assert_eq!(segments[1].output_col_start, 6);
        assert_eq!(segments[1].output_cols, 1);
        assert_eq!(segments[1].source.start, 8);
        assert_eq!(segments[1].source.end, 12);

        let out = bank
            .access_cells_owned_by_gene_names::<u32, _>(id, &[0], &genes, MissingGenePolicy::Zero)
            .expect("projected dense 1d access");
        assert_eq!(out, vec![6, 1, 2]);
    }

    #[test]
    fn projected_dense_2d_planning_visits_only_selected_chunk_columns() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_memory(&mut bank, 1, 8, 1, 2);
        let dataset = bank.registry.get(id).expect("dataset");
        let dense = match dataset {
            Dataset::Dense2D(dense) => dense,
            _ => panic!("expected dense 2d dataset"),
        };
        let genes = ["g1", "g6"];
        let gene_axis = super::GeneAxisPlan::requested(dataset, &genes, MissingGenePolicy::Zero)
            .expect("gene projection");
        let projection = match gene_axis {
            super::GeneAxisPlan::Requested(projection) => projection,
            super::GeneAxisPlan::DatasetOrder => panic!("subset should remain projected"),
        };

        let segments =
            super::plan::plan_dense_2d_selected_sources(dense, &[0], &projection.selected_sources)
                .expect("selected dense plan");

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].output_col_start, 1);
        assert_eq!(segments[0].output_cols, 1);
        assert_eq!(segments[0].source.start, 4);
        assert_eq!(segments[0].source.end, 8);
        assert_eq!(segments[1].output_col_start, 6);
        assert_eq!(segments[1].output_cols, 1);
        assert_eq!(segments[1].source.start, 0);
        assert_eq!(segments[1].source.end, 4);
    }

    #[test]
    fn requested_full_gene_order_collapses_to_dataset_order() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_memory(&mut bank, 2, 4, 1, 2);
        let dataset = bank.registry.get(id).expect("dataset");
        let genes = ["g0", "g1", "g2", "g3"];

        let gene_axis = super::GeneAxisPlan::requested(dataset, &genes, MissingGenePolicy::Zero)
            .expect("gene projection");

        assert!(matches!(gene_axis, super::GeneAxisPlan::DatasetOrder));
    }

    #[test]
    fn projected_gene_runs_detect_contiguous_output_ranges() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_memory(&mut bank, 1, 8, 1, 8);
        let dataset = bank.registry.get(id).expect("dataset");
        let genes = ["g1", "g2", "g6"];
        let gene_axis = super::GeneAxisPlan::requested(dataset, &genes, MissingGenePolicy::Zero)
            .expect("gene projection");
        let projection = match gene_axis {
            super::GeneAxisPlan::Requested(projection) => projection,
            super::GeneAxisPlan::DatasetOrder => panic!("subset should remain projected"),
        };

        assert_eq!(projection.contiguous_output_for_source_run(1, 2), Some(0));
        assert_eq!(projection.contiguous_output_for_source_run(6, 1), Some(2));
        assert_eq!(projection.contiguous_output_for_source_run(2, 2), None);
    }

    #[test]
    fn access_cells_by_gene_names_can_error_on_missing_gene() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_memory(&mut bank, 2, 3, 1, 2);
        let mut out = vec![0u32; 1];

        let err = bank
            .access_cells_by_gene_names(
                id,
                &[0],
                &["missing"],
                &mut out,
                None,
                MissingGenePolicy::Error,
            )
            .expect_err("missing gene should error");

        assert!(matches!(
            err,
            crate::databank::DataBankError::GeneNameNotFound { .. }
        ));
    }

    #[test]
    fn access_cells_by_gene_names_rejects_duplicate_dataset_gene_names() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = bank
            .register_dense_2d(Dense2DMeta {
                gene_names: vec!["g0".to_string(), "g0".to_string()],
                data: ArrayMeta {
                    shape: vec![1, 2],
                    chunk_shape: vec![1, 2],
                    chunk_grid_shape: vec![1, 1],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory {
                        chunks: vec![arc_u32_bytes(&[1, 2])],
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register dense with duplicate gene names");
        let mut out = vec![0u32; 1];

        let err = bank
            .access_cells_by_gene_names(id, &[0], &["g0"], &mut out, None, MissingGenePolicy::Zero)
            .expect_err("duplicate dataset gene names should fail");

        assert!(matches!(
            err,
            crate::databank::DataBankError::InvalidArrayMeta(message)
                if message.contains("duplicate dataset gene name: g0")
        ));
    }

    #[test]
    fn access_cells_by_gene_names_reorders_csr() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_csr_file(&mut bank);
        let cells = vec![1, 0, 2];
        let genes = ["g4", "g1", "missing", "g2"];

        let out = bank
            .access_cells_owned_by_gene_names::<u32, _>(id, &cells, &genes, MissingGenePolicy::Zero)
            .expect("projected CSR access");

        assert_eq!(
            out,
            vec![
                400, 0, 0, 200, // cell 1
                0, 10, 0, 0, // cell 0
                0, 0, 0, 0, // cell 2 has only g3
            ]
        );
    }

    #[test]
    fn scheduled_prefetch_dense_2d_matches_direct_access() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_file(&mut bank, 4, 4, 2);

        let batches: Vec<Vec<usize>> = vec![vec![0, 1], vec![3, 2, 0], vec![1]];
        let expected: Vec<Vec<u32>> = batches
            .iter()
            .map(|cells| {
                bank.access_cells_owned::<u32>(id, cells)
                    .expect("ground truth")
            })
            .collect();

        let prefetch = bank
            .prefetch_cells_scheduled::<u32, _>(
                id,
                batches.clone(),
                ScheduledPrefetchConfig::default(),
            )
            .expect("build prefetcher");
        assert_eq!(
            prefetch.prefetch_step(),
            ScheduledPrefetchConfig::default().prefetch_step
        );

        let mut collected = Vec::new();
        for result in prefetch {
            let batch: PrefetchedBatch<u32> = result.expect("prefetch batch");
            assert_eq!(batch.num_genes, 4);
            collected.push(batch.buffer);
        }
        assert_eq!(collected, expected);
    }

    #[test]
    fn unregister_defers_file_release_for_retained_prefetch_dataset() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_file(&mut bank, 4, 4, 2);
        let dataset = bank.registry.get_arc(id).expect("dataset arc");
        let batches: Vec<Vec<usize>> = vec![vec![0, 1], vec![2, 3]];
        let expected: Vec<Vec<u32>> = batches
            .iter()
            .map(|cells| expected_dense_rows(cells, 4))
            .collect();

        bank.unregister(id)
            .expect("unregister should retire retained dataset");
        let prefetch = super::prefetch_cells_scheduled::<u32, _>(
            &bank.access,
            Arc::clone(&bank.compute),
            Arc::clone(&dataset),
            batches.into_iter(),
            ScheduledPrefetchConfig::default(),
        )
        .expect("prefetch retained dataset");
        let collected: Vec<Vec<u32>> = prefetch
            .map(|batch| batch.expect("prefetch batch").buffer)
            .collect();
        assert_eq!(collected, expected);

        drop(dataset);
        bank.cleanup_retired().expect("cleanup retired dataset");
    }

    #[test]
    fn scheduled_prefetch_emits_in_order_when_later_response_finishes_first() {
        let mut config = DataBankConfig::default();
        config.fill_config.parallel = true;
        config.fill_config.num_workers = 3;
        let mut bank = DataBank::new(config).expect("databank");
        let (started_tx, started_rx) = mpsc::channel();
        let (decoded_tx, decoded_rx) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let codec: SharedCodec = Arc::new(BlockingFirstChunkCodec {
            blocked: std::sync::atomic::AtomicBool::new(false),
            started: started_tx,
            decoded: decoded_tx,
            release: Arc::clone(&release),
        });
        let id = register_dense_2d_memory_with_codec(&mut bank, 2, 4, codec);
        let batches = vec![vec![0], vec![1]];
        let mut prefetch = bank
            .prefetch_cells_scheduled::<u32, _>(
                id,
                batches.clone(),
                ScheduledPrefetchConfig::default(),
            )
            .expect("scheduled prefetch");

        started_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("first response blocked");
        let decoded_first = decoded_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("later response decoded");
        assert_eq!(decoded_first, dense_value(1, 0));
        {
            let (lock, cvar) = &*release;
            *lock.lock().expect("release lock") = true;
            cvar.notify_all();
        }

        let first = prefetch.next().expect("first item").expect("first batch");
        let second = prefetch.next().expect("second item").expect("second batch");
        assert_eq!(first.cells, batches[0]);
        assert_eq!(first.buffer, expected_dense_rows(&batches[0], 4));
        assert_eq!(second.cells, batches[1]);
        assert_eq!(second.buffer, expected_dense_rows(&batches[1], 4));
        assert!(prefetch.next().is_none());
    }

    #[test]
    fn scheduled_prefetch_csr_matches_direct_access() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_csr_file(&mut bank);

        let batches: Vec<Vec<usize>> = vec![vec![1, 0], vec![2], vec![0, 1, 2]];
        let expected: Vec<Vec<u32>> = batches
            .iter()
            .map(|cells| {
                bank.access_cells_owned::<u32>(id, cells)
                    .expect("ground truth")
            })
            .collect();

        let prefetch = bank
            .prefetch_cells_scheduled::<u32, _>(
                id,
                batches.clone(),
                ScheduledPrefetchConfig::default(),
            )
            .expect("build prefetcher");

        let mut collected = Vec::new();
        for result in prefetch {
            let batch: PrefetchedBatch<u32> = result.expect("prefetch batch");
            assert_eq!(batch.cells, batches[collected.len()]);
            assert_eq!(batch.num_genes, 5);
            collected.push(batch.buffer);
        }
        assert_eq!(collected, expected);
    }

    #[test]
    fn scheduled_prefetch_by_gene_names_matches_direct_csr_access() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_csr_file(&mut bank);
        let batches: Vec<Vec<usize>> = vec![vec![1, 0], vec![2]];
        let genes = ["g4", "g1", "missing", "g2"];
        let expected: Vec<Vec<u32>> = batches
            .iter()
            .map(|cells| {
                bank.access_cells_owned_by_gene_names::<u32, _>(
                    id,
                    cells,
                    &genes,
                    MissingGenePolicy::Zero,
                )
                .expect("ground truth")
            })
            .collect();

        let prefetch = bank
            .prefetch_cells_scheduled_by_gene_names::<u32, _, _>(
                id,
                batches.clone(),
                &genes,
                MissingGenePolicy::Zero,
                ScheduledPrefetchConfig::default(),
            )
            .expect("build projected prefetcher");
        assert_eq!(prefetch.gene_names().len(), genes.len());

        let mut collected = Vec::new();
        for result in prefetch {
            let batch = result.expect("prefetch batch");
            assert_eq!(batch.cells, batches[collected.len()]);
            assert_eq!(batch.num_genes, genes.len());
            collected.push(batch.buffer);
        }
        assert_eq!(collected, expected);
    }

    #[test]
    fn scheduled_prefetch_memory_csr_matches_direct_access() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = bank
            .register_sparse_csr(SparseCsrDatasetMeta {
                gene_names: (0..5).map(|idx| format!("g{idx}")).collect(),
                indptr: vec![0, 2, 5, 6],
                indices: u32_array_meta(vec![
                    padded_u32_bytes(&[1, 3, 0, 4], 4),
                    padded_u32_bytes(&[2, 3], 4),
                ]),
                data: u32_array_meta(vec![
                    padded_u32_bytes(&[10, 30, 100, 400], 4),
                    padded_u32_bytes(&[200, 3000], 4),
                ]),
                index_dtype: DType::U32,
                num_cells: 3,
                num_genes: 5,
            })
            .expect("register CSR");

        let batches: Vec<Vec<usize>> = vec![vec![2], vec![1, 0], Vec::new()];
        let expected: Vec<Vec<u32>> = batches
            .iter()
            .map(|cells| {
                bank.access_cells_alloc::<u32>(id, cells)
                    .expect("ground truth")
            })
            .collect();

        let prefetch = bank
            .prefetch_cells_scheduled::<u32, _>(
                id,
                batches.clone(),
                ScheduledPrefetchConfig::default(),
            )
            .expect("build prefetcher");

        let mut collected = Vec::new();
        for result in prefetch {
            let batch = result.expect("prefetch batch");
            assert_eq!(batch.cells, batches[collected.len()]);
            assert_eq!(batch.num_genes, 5);
            collected.push(batch.buffer);
        }
        assert_eq!(collected, expected);
    }

    #[test]
    fn databank_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DataBank>();
    }

    #[test]
    fn parallel_csr_scatter_matches_sequential_checked_and_unchecked() {
        let mut seq_bank = DataBank::new(DataBankConfig::default()).expect("seq databank");
        let seq_id = register_csr_file(&mut seq_bank);
        let mut par_config = DataBankConfig::default();
        par_config.fill_config.parallel = true;
        par_config.fill_config.num_workers = 2;
        par_config.fill_config.min_parallel_rows = 1;
        par_config.fill_config.min_parallel_bytes = 1;
        let mut par_bank = DataBank::new(par_config).expect("parallel databank");
        let par_id = register_csr_file(&mut par_bank);
        let cells = vec![1, 0, 2, 1, 0, 2];

        let expected = seq_bank
            .access_cells_alloc::<u32>(seq_id, &cells)
            .expect("sequential checked");
        let checked = par_bank
            .access_cells_alloc::<u32>(par_id, &cells)
            .expect("parallel checked");
        assert_eq!(checked, expected);

        let mut unchecked = vec![0u32; cells.len() * 5];
        unsafe {
            par_bank
                .access_cells_unchecked(par_id, &cells, &mut unchecked, None)
                .expect("parallel unchecked");
        }
        assert_eq!(unchecked, expected);
    }

    #[test]
    fn scheduled_parallel_csr_scatter_matches_direct_access() {
        let mut config = DataBankConfig::default();
        config.fill_config.parallel = true;
        config.fill_config.num_workers = 2;
        config.fill_config.min_parallel_rows = 1;
        config.fill_config.min_parallel_bytes = 1;
        let mut bank = DataBank::new(config).expect("databank");
        let id = register_csr_file(&mut bank);
        let batches: Vec<Vec<usize>> = vec![vec![1, 0, 2], vec![2, 1, 0]];
        let expected: Vec<Vec<u32>> = batches
            .iter()
            .map(|cells| {
                bank.access_cells_alloc::<u32>(id, cells)
                    .expect("direct access")
            })
            .collect();

        let prefetch = bank
            .prefetch_cells_scheduled::<u32, _>(
                id,
                batches.clone(),
                ScheduledPrefetchConfig::default(),
            )
            .expect("scheduled prefetch");
        let collected: Vec<Vec<u32>> = prefetch
            .map(|batch| batch.expect("prefetch batch").buffer)
            .collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn parallel_dense_memory_scatter_matches_expected_for_full_and_projected() {
        let mut bank = DataBank::new(parallel_config()).expect("databank");
        let id = register_dense_2d_memory(&mut bank, 8, 7, 2, 3);
        let cells = vec![7, 0, 4, 3, 1];

        let full = bank
            .access_cells_alloc::<u32>(id, &cells)
            .expect("parallel dense access");
        assert_eq!(full, expected_dense_rows(&cells, 7));

        let genes = vec!["g6", "g2", "missing", "g0"];
        let projected = bank
            .access_cells_owned_by_gene_names::<u32, _>(id, &cells, &genes, MissingGenePolicy::Zero)
            .expect("parallel projected dense access");
        let expected: Vec<u32> = cells
            .iter()
            .flat_map(|&cell| {
                [
                    dense_value(cell, 6),
                    dense_value(cell, 2),
                    0,
                    dense_value(cell, 0),
                ]
            })
            .collect();
        assert_eq!(projected, expected);
    }

    #[test]
    fn scheduled_parallel_dense_file_scatter_matches_direct_access() {
        let mut bank = DataBank::new(parallel_config()).expect("databank");
        let id = register_dense_2d_file(&mut bank, 8, 6, 2);
        let batches: Vec<Vec<usize>> = vec![vec![7, 0, 3], vec![2, 5, 1]];
        let expected: Vec<Vec<u32>> = batches
            .iter()
            .map(|cells| {
                bank.access_cells_alloc::<u32>(id, cells)
                    .expect("direct dense access")
            })
            .collect();

        let prefetch = bank
            .prefetch_cells_scheduled::<u32, _>(
                id,
                batches.clone(),
                ScheduledPrefetchConfig::default(),
            )
            .expect("scheduled dense prefetch");
        let collected: Vec<Vec<u32>> = prefetch
            .map(|batch| batch.expect("prefetch batch").buffer)
            .collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn scheduled_prefetch_single_worker_parallel_scatter_does_not_deadlock() {
        let mut config = DataBankConfig::default();
        config.fill_config.parallel = true;
        config.fill_config.num_workers = 1;
        config.fill_config.min_parallel_rows = 1;
        config.fill_config.min_parallel_bytes = 1;
        let mut bank = DataBank::new(config).expect("databank");
        let id = register_dense_2d_file(&mut bank, 6, 4, 1);
        let batches: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let expected: Vec<Vec<u32>> = batches
            .iter()
            .map(|cells| {
                bank.access_cells_alloc::<u32>(id, cells)
                    .expect("direct access")
            })
            .collect();

        let prefetch = bank
            .prefetch_cells_scheduled::<u32, _>(id, batches, ScheduledPrefetchConfig::default())
            .expect("scheduled prefetch");
        let collected: Vec<Vec<u32>> = prefetch
            .map(|batch| batch.expect("prefetch batch").buffer)
            .collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn parallel_projected_csr_scatter_reads_only_selected_groups() {
        let mut seq_bank = DataBank::new(DataBankConfig::default()).expect("seq databank");
        let seq_id = register_csr_file(&mut seq_bank);
        let mut par_bank = DataBank::new(parallel_config()).expect("parallel databank");
        let par_id = register_csr_file(&mut par_bank);
        let cells = vec![1, 0, 2, 1, 0];
        let genes = vec!["g1"];

        let expected = seq_bank
            .access_cells_owned_by_gene_names::<u32, _>(
                seq_id,
                &cells,
                &genes,
                MissingGenePolicy::Zero,
            )
            .expect("sequential projected CSR");
        let checked = par_bank
            .access_cells_owned_by_gene_names::<u32, _>(
                par_id,
                &cells,
                &genes,
                MissingGenePolicy::Zero,
            )
            .expect("parallel projected CSR");
        assert_eq!(checked, expected);
        assert_eq!(checked, vec![0, 10, 0, 0, 10]);

        let batches: Vec<Vec<usize>> = vec![vec![1, 0], vec![2, 1, 0]];
        let expected_batches: Vec<Vec<u32>> = batches
            .iter()
            .map(|batch| {
                par_bank
                    .access_cells_owned_by_gene_names::<u32, _>(
                        par_id,
                        batch,
                        &genes,
                        MissingGenePolicy::Zero,
                    )
                    .expect("direct projected CSR")
            })
            .collect();
        let prefetch = par_bank
            .prefetch_cells_scheduled_by_gene_names::<u32, _, _>(
                par_id,
                batches.clone(),
                &genes,
                MissingGenePolicy::Zero,
                ScheduledPrefetchConfig::default(),
            )
            .expect("scheduled projected CSR");
        let collected: Vec<Vec<u32>> = prefetch
            .map(|batch| batch.expect("prefetch batch").buffer)
            .collect();
        assert_eq!(collected, expected_batches);
    }

    #[test]
    fn projected_csr_mixed_file_memory_data_groups_matches_direct_and_prefetch() {
        let mut seq_bank = DataBank::new(DataBankConfig::default()).expect("seq databank");
        let seq_id = register_csr_mixed_file_memory_data(&mut seq_bank);
        let mut par_bank = DataBank::new(parallel_config()).expect("parallel databank");
        let par_id = register_csr_mixed_file_memory_data(&mut par_bank);
        let cells = vec![1, 0, 2, 1, 0];
        let genes = vec!["g1", "g3"];

        let expected = seq_bank
            .access_cells_owned_by_gene_names::<u32, _>(
                seq_id,
                &cells,
                &genes,
                MissingGenePolicy::Zero,
            )
            .expect("sequential mixed projected CSR");
        let checked = par_bank
            .access_cells_owned_by_gene_names::<u32, _>(
                par_id,
                &cells,
                &genes,
                MissingGenePolicy::Zero,
            )
            .expect("parallel mixed projected CSR");
        assert_eq!(checked, expected);
        assert_eq!(checked, vec![0, 0, 10, 30, 0, 3000, 0, 0, 10, 30]);

        let batches: Vec<Vec<usize>> = vec![vec![1, 0], vec![2, 1, 0]];
        let expected_batches: Vec<Vec<u32>> = batches
            .iter()
            .map(|batch| {
                par_bank
                    .access_cells_owned_by_gene_names::<u32, _>(
                        par_id,
                        batch,
                        &genes,
                        MissingGenePolicy::Zero,
                    )
                    .expect("direct mixed projected CSR")
            })
            .collect();
        let prefetch = par_bank
            .prefetch_cells_scheduled_by_gene_names::<u32, _, _>(
                par_id,
                batches.clone(),
                &genes,
                MissingGenePolicy::Zero,
                ScheduledPrefetchConfig::default(),
            )
            .expect("scheduled mixed projected CSR");
        let collected: Vec<Vec<u32>> = prefetch
            .map(|batch| batch.expect("prefetch batch").buffer)
            .collect();
        assert_eq!(collected, expected_batches);
    }

    #[test]
    fn shared_databank_allows_concurrent_callers() {
        let mut config = DataBankConfig::default();
        config.fill_config.parallel = true;
        config.fill_config.num_workers = 2;
        config.fill_config.min_parallel_rows = 1;
        config.fill_config.min_parallel_bytes = 1;
        let mut bank = DataBank::new(config).expect("databank");
        let id = register_csr_file(&mut bank);
        let bank = Arc::new(bank);
        let cells = vec![1, 0, 2, 1];
        let expected = bank
            .access_cells_alloc::<u32>(id, &cells)
            .expect("expected");

        let mut threads = Vec::new();
        for _ in 0..4 {
            let bank = Arc::clone(&bank);
            let cells = cells.clone();
            let expected = expected.clone();
            threads.push(std::thread::spawn(move || {
                let out = bank.access_cells_alloc::<u32>(id, &cells).expect("access");
                assert_eq!(out, expected);
            }));
        }
        for thread in threads {
            thread.join().expect("caller thread");
        }
    }

    #[test]
    fn scheduled_prefetch_surfaces_plan_error_after_draining_good_batches() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_file(&mut bank, 4, 4, 2);

        // First batch is valid, second references an out-of-range cell.
        let batches: Vec<Vec<usize>> = vec![vec![0, 1], vec![99]];
        let prefetch = bank
            .prefetch_cells_scheduled::<u32, _>(id, batches, ScheduledPrefetchConfig::default())
            .expect("build prefetcher");

        let mut iter = prefetch;
        let first = iter.next().expect("first batch").expect("first ok");
        assert_eq!(first.cells, vec![0, 1]);
        let err = iter.next().expect("error item").expect_err("plan error");
        assert!(matches!(
            err,
            crate::databank::DataBankError::CellIndexOutOfRange { cell: 99, .. }
        ));
        assert!(iter.next().is_none());
    }

    #[test]
    fn scheduled_prefetch_exposes_first_ordered_error_only() {
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = register_dense_2d_file(&mut bank, 4, 4, 2);

        let batches: Vec<Vec<usize>> = vec![vec![0, 1], vec![99], vec![2, 3]];
        let mut iter = bank
            .prefetch_cells_scheduled::<u32, _>(id, batches, ScheduledPrefetchConfig::default())
            .expect("scheduled prefetch");

        let first = iter.next().expect("first batch").expect("first ok");
        assert_eq!(first.cells, vec![0, 1]);
        let err = iter.next().expect("error item").expect_err("first error");
        assert!(matches!(
            err,
            crate::databank::DataBankError::CellIndexOutOfRange { cell: 99, .. }
        ));
        assert!(iter.next().is_none());
    }

    #[derive(Clone)]
    struct CountingCodec {
        decodes: Arc<AtomicUsize>,
    }

    impl fmt::Debug for CountingCodec {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("CountingCodec").finish_non_exhaustive()
        }
    }

    impl crate::codecs::sealed::Sealed for CountingCodec {}

    impl ChunkCodec for CountingCodec {
        fn name(&self) -> &str {
            "counting"
        }

        fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
            self.decodes.fetch_add(1, Ordering::SeqCst);
            if let Some(expected_size) = expected_size {
                assert_eq!(encoded.len(), expected_size);
            }
            Ok(encoded.to_vec())
        }

        fn decoded_size_hint(
            &self,
            _encoded: &[u8],
            expected_size: Option<usize>,
        ) -> CodecResult<Option<usize>> {
            Ok(expected_size)
        }
    }

    struct BlockingFirstChunkCodec {
        blocked: std::sync::atomic::AtomicBool,
        started: mpsc::Sender<()>,
        decoded: mpsc::Sender<u32>,
        release: Arc<(Mutex<bool>, Condvar)>,
    }

    impl fmt::Debug for BlockingFirstChunkCodec {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("BlockingFirstChunkCodec")
                .finish_non_exhaustive()
        }
    }

    impl crate::codecs::sealed::Sealed for BlockingFirstChunkCodec {}

    impl ChunkCodec for BlockingFirstChunkCodec {
        fn name(&self) -> &str {
            "blocking-first"
        }

        fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
            if let Some(expected_size) = expected_size {
                assert_eq!(encoded.len(), expected_size);
            }
            let first = encoded
                .get(..std::mem::size_of::<u32>())
                .map(|bytes| {
                    let mut word = [0u8; std::mem::size_of::<u32>()];
                    word.copy_from_slice(bytes);
                    u32::from_ne_bytes(word)
                })
                .unwrap_or(0);

            if first == 0 && !self.blocked.swap(true, Ordering::SeqCst) {
                self.started.send(()).expect("started");
                let (lock, cvar) = &*self.release;
                let mut released = lock.lock().expect("release lock");
                while !*released {
                    released = cvar.wait(released).expect("release wait");
                }
            }
            self.decoded.send(first).expect("decoded");
            Ok(encoded.to_vec())
        }

        fn decoded_size_hint(
            &self,
            _encoded: &[u8],
            expected_size: Option<usize>,
        ) -> CodecResult<Option<usize>> {
            Ok(expected_size)
        }
    }

    fn register_counted_csr_memory(
        bank: &mut DataBank,
        num_cells: usize,
        num_genes: usize,
        nnz_per_cell: usize,
        chunk_len: usize,
    ) -> (crate::databank::DatasetId, Arc<AtomicUsize>) {
        let nnz = num_cells * nnz_per_cell;
        let mut indptr = Vec::with_capacity(num_cells + 1);
        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        indptr.push(0);
        for cell in 0..num_cells {
            for k in 0..nnz_per_cell {
                indices.push(counted_csr_gene(cell, k, num_genes));
                data.push(counted_csr_value(cell, k));
            }
            indptr.push(indptr.last().copied().unwrap() + nnz_per_cell as u64);
        }

        let decodes = Arc::new(AtomicUsize::new(0));
        let codec: SharedCodec = Arc::new(CountingCodec {
            decodes: Arc::clone(&decodes),
        });
        let gene_names = (0..num_genes)
            .map(|idx| format!("g{idx}"))
            .collect::<Vec<_>>();
        let genes = bank.interner.intern_dataset(&gene_names);
        let indices_chunks = chunk_u32_values(&indices, chunk_len);
        let data_chunks = chunk_u32_values(&data, chunk_len);
        let dataset = SparseCsrDataset {
            genes,
            indptr,
            indices: counted_memory_array(
                nnz,
                chunk_len,
                DType::U32,
                Arc::clone(&codec),
                indices_chunks,
            ),
            data: counted_memory_array(nnz, chunk_len, DType::U32, codec, data_chunks),
            index_dtype: DType::U32,
            num_cells,
            num_genes,
        };
        let id = bank
            .registry
            .register(Dataset::SparseCsr(dataset))
            .expect("register counted CSR");
        (id, decodes)
    }

    fn chunk_u32_values(values: &[u32], chunk_len: usize) -> Vec<Arc<[u8]>> {
        // Edge chunks padded to ``chunk_len`` (zero fill value) — standard zarr
        // edge-chunk layout: every decoded chunk is exactly ``chunk_len``
        // elements.
        values
            .chunks(chunk_len)
            .map(|chunk| {
                let mut bytes = vec![0u8; chunk_len * std::mem::size_of::<u32>()];
                for (i, value) in chunk.iter().enumerate() {
                    let offset = i * std::mem::size_of::<u32>();
                    bytes[offset..offset + std::mem::size_of::<u32>()]
                        .copy_from_slice(&value.to_ne_bytes());
                }
                Arc::from(bytes.into_boxed_slice())
            })
            .collect()
    }

    fn counted_memory_array(
        len: usize,
        chunk_len: usize,
        dtype: DType,
        codec: SharedCodec,
        raw_chunks: Vec<Arc<[u8]>>,
    ) -> Array {
        let grid_shape = vec![len.div_ceil(chunk_len)];
        let decoded_bytes = chunk_len * dtype.item_size();
        Array {
            shape: vec![len],
            dtype,
            codec,
            grid: ArrayGrid::Regular {
                chunk_shape: vec![chunk_len],
                grid_shape,
                edge: EdgeChunkLayout::Padded,
            },
            chunks: raw_chunks
                .into_iter()
                .map(|bytes| Chunk {
                    source: ChunkSource::Memory {
                        bytes,
                        decoded: false,
                    },
                    decoded_bytes,
                })
                .collect(),
            files: Vec::new(),
        }
    }

    fn counted_decoded_memory_array(
        len: usize,
        chunk_len: usize,
        dtype: DType,
        raw_chunks: Vec<Arc<[u8]>>,
    ) -> (Array, Arc<AtomicUsize>) {
        let decodes = Arc::new(AtomicUsize::new(0));
        let codec: SharedCodec = Arc::new(CountingCodec {
            decodes: Arc::clone(&decodes),
        });
        let mut array = counted_memory_array(len, chunk_len, dtype, codec, raw_chunks);
        for chunk in &mut array.chunks {
            let ChunkSource::Memory { decoded, .. } = &mut chunk.source else {
                unreachable!("counted_memory_array only creates memory chunks");
            };
            *decoded = true;
        }
        (array, decodes)
    }

    #[test]
    fn decoded_single_memory_chunk_accepts_non_identity_codec_without_decode() {
        let bytes = arc_u32_bytes(&[1, 2, 3, 4]);
        let (array, decodes) =
            counted_decoded_memory_array(4, 4, DType::U32, vec![Arc::clone(&bytes)]);

        let raw = super::single_memory_identity_chunk_bytes(&array)
            .expect("single decoded chunk")
            .expect("direct memory chunk");

        assert_eq!(raw, bytes.as_ref());
        assert_eq!(decodes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn decoded_memory_1d_chunks_accept_non_identity_codec_without_decode() {
        let chunk0 = arc_u32_bytes(&[1, 2, 3, 4]);
        let chunk1 = arc_u32_bytes(&[5, 6, 0, 0]);
        let (array, decodes) = counted_decoded_memory_array(
            6,
            4,
            DType::U32,
            vec![Arc::clone(&chunk0), Arc::clone(&chunk1)],
        );

        let chunks = super::MemoryIdentity1DChunks::from_array(&array)
            .expect("decoded memory chunks")
            .expect("direct 1D chunks");

        assert_eq!(chunks.chunk_bytes(0).expect("chunk 0"), chunk0.as_ref());
        assert_eq!(chunks.chunk_bytes(1).expect("chunk 1"), chunk1.as_ref());
        assert_eq!(decodes.load(Ordering::SeqCst), 0);
    }

    fn expected_counted_csr_rows(
        cells: &[usize],
        num_genes: usize,
        nnz_per_cell: usize,
    ) -> Vec<u32> {
        let mut out = vec![0u32; cells.len() * num_genes];
        for (row, &cell) in cells.iter().enumerate() {
            let row_offset = row * num_genes;
            for k in 0..nnz_per_cell {
                let gene = counted_csr_gene(cell, k, num_genes) as usize;
                out[row_offset + gene] = counted_csr_value(cell, k);
            }
        }
        out
    }

    fn counted_csr_gene(cell: usize, k: usize, num_genes: usize) -> u32 {
        ((cell * 3 + k * 5) % num_genes) as u32
    }

    fn counted_csr_value(cell: usize, k: usize) -> u32 {
        (cell as u32) * 1000 + k as u32 + 1
    }

    fn register_dense_2d_file(
        bank: &mut DataBank,
        num_cells: usize,
        num_genes: usize,
        chunk_rows: usize,
    ) -> crate::databank::DatasetId {
        // value(cell, gene) = cell * 100 + gene
        let chunk_cols = num_genes;
        let grid_rows = num_cells.div_ceil(chunk_rows);
        let mut chunks = Vec::new();
        for chunk_row in 0..grid_rows {
            // Padded to full chunk_rows*chunk_cols (zero fill value) to match
            // standard zarr edge-chunk layout.
            let mut bytes = vec![0u8; chunk_rows * chunk_cols * std::mem::size_of::<u32>()];
            for local_row in 0..chunk_rows {
                let cell = chunk_row * chunk_rows + local_row;
                if cell >= num_cells {
                    break;
                }
                for gene in 0..num_genes {
                    let offset = (local_row * chunk_cols + gene) * std::mem::size_of::<u32>();
                    bytes[offset..offset + std::mem::size_of::<u32>()]
                        .copy_from_slice(&((cell * 100 + gene) as u32).to_ne_bytes());
                }
            }
            chunks.push(Arc::from(bytes.into_boxed_slice()));
        }
        let (path, locations) = write_chunk_file(chunks);
        let id = bank
            .register_dense_2d(Dense2DMeta {
                gene_names: (0..num_genes).map(|g| format!("g{g}")).collect(),
                data: ArrayMeta {
                    shape: vec![num_cells, num_genes],
                    chunk_shape: vec![chunk_rows, chunk_cols],
                    chunk_grid_shape: vec![grid_rows, 1],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::FileOffset { path, locations },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register dense 2d file");
        id
    }

    fn register_dense_2d_memory(
        bank: &mut DataBank,
        num_cells: usize,
        num_genes: usize,
        chunk_rows: usize,
        chunk_cols: usize,
    ) -> crate::databank::DatasetId {
        let grid_rows = num_cells.div_ceil(chunk_rows);
        let grid_cols = num_genes.div_ceil(chunk_cols);
        let mut chunks = Vec::with_capacity(grid_rows * grid_cols);
        for chunk_row in 0..grid_rows {
            let row_start = chunk_row * chunk_rows;
            for chunk_col in 0..grid_cols {
                let col_start = chunk_col * chunk_cols;
                // Chunks are stored padded to the full ``chunk_shape`` with a
                // zero fill value, matching standard zarr edge-chunk layout
                // (the decoded buffer is always chunk_rows*chunk_cols elements).
                let mut bytes = vec![0u8; chunk_rows * chunk_cols * std::mem::size_of::<u32>()];
                for local_row in 0..chunk_rows {
                    let cell = row_start + local_row;
                    if cell >= num_cells {
                        break;
                    }
                    for local_col in 0..chunk_cols {
                        let gene = col_start + local_col;
                        if gene >= num_genes {
                            break;
                        }
                        let offset =
                            (local_row * chunk_cols + local_col) * std::mem::size_of::<u32>();
                        bytes[offset..offset + std::mem::size_of::<u32>()]
                            .copy_from_slice(&dense_value(cell, gene).to_ne_bytes());
                    }
                }
                chunks.push(Arc::from(bytes.into_boxed_slice()));
            }
        }

        bank.register_dense_2d(Dense2DMeta {
            gene_names: (0..num_genes).map(|g| format!("g{g}")).collect(),
            data: ArrayMeta {
                shape: vec![num_cells, num_genes],
                chunk_shape: vec![chunk_rows, chunk_cols],
                chunk_grid_shape: vec![grid_rows, grid_cols],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory { chunks },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register dense 2d memory")
    }

    fn register_dense_2d_memory_with_codec(
        bank: &mut DataBank,
        num_cells: usize,
        num_genes: usize,
        codec: SharedCodec,
    ) -> crate::databank::DatasetId {
        let chunk_rows = 1;
        let chunk_cols = num_genes;
        let mut chunks = Vec::with_capacity(num_cells);
        for cell in 0..num_cells {
            let mut bytes = Vec::with_capacity(num_genes * std::mem::size_of::<u32>());
            for gene in 0..num_genes {
                bytes.extend_from_slice(&dense_value(cell, gene).to_ne_bytes());
            }
            chunks.push(Chunk {
                source: ChunkSource::Memory {
                    bytes: Arc::from(bytes.into_boxed_slice()),
                    decoded: false,
                },
                decoded_bytes: chunk_cols * std::mem::size_of::<u32>(),
            });
        }
        let gene_names = (0..num_genes)
            .map(|gene| format!("g{gene}"))
            .collect::<Vec<_>>();
        let genes = bank.interner.intern_dataset(&gene_names);
        let dataset = Dense2DDataset {
            genes,
            data: Array {
                shape: vec![num_cells, num_genes],
                dtype: DType::U32,
                codec,
                grid: ArrayGrid::Regular {
                    chunk_shape: vec![chunk_rows, chunk_cols],
                    grid_shape: vec![num_cells, 1],
                    edge: EdgeChunkLayout::Padded,
                },
                chunks,
                files: Vec::new(),
            },
            num_cells,
            num_genes,
        };
        bank.registry
            .register(Dataset::Dense2D(dataset))
            .expect("register dense 2d memory with codec")
    }

    fn expected_dense_rows(cells: &[usize], num_genes: usize) -> Vec<u32> {
        cells
            .iter()
            .flat_map(|&cell| (0..num_genes).map(move |gene| dense_value(cell, gene)))
            .collect()
    }

    fn dense_value(cell: usize, gene: usize) -> u32 {
        (cell * 100 + gene) as u32
    }

    fn register_csr_file(bank: &mut DataBank) -> crate::databank::DatasetId {
        let (indices_path, indices_locations) =
            write_chunk_file(vec![arc_u32_bytes(&[1, 3, 0, 4]), arc_u32_bytes(&[2, 3])]);
        let (data_path, data_locations) = write_chunk_file(vec![
            arc_u32_bytes(&[10, 30, 100, 400]),
            arc_u32_bytes(&[200, 3000]),
        ]);
        bank.register_sparse_csr(SparseCsrDatasetMeta {
            gene_names: (0..5).map(|idx| format!("g{idx}")).collect(),
            indptr: vec![0, 2, 5, 6],
            indices: ArrayMeta {
                shape: vec![6],
                chunk_shape: vec![4],
                chunk_grid_shape: vec![2],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::FileOffset {
                    path: indices_path,
                    locations: indices_locations,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
            data: ArrayMeta {
                shape: vec![6],
                chunk_shape: vec![4],
                chunk_grid_shape: vec![2],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::FileOffset {
                    path: data_path,
                    locations: data_locations,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
            index_dtype: DType::U32,
            num_cells: 3,
            num_genes: 5,
        })
        .expect("register CSR file")
    }

    fn register_csr_mixed_file_memory_data(bank: &mut DataBank) -> crate::databank::DatasetId {
        let (data_path, data_locations) =
            write_chunk_file(vec![arc_u32_bytes(&[10, 30, 100, 400])]);
        let file_id = bank
            .io_pool
            .register_readonly_file(&data_path)
            .expect("register data file");
        let file = RegisteredFile::new(file_id).expect("registered file");
        let gene_names = (0..5).map(|idx| format!("g{idx}")).collect::<Vec<_>>();
        let genes = bank.interner.intern_dataset(&gene_names);
        let index_chunks = vec![
            Chunk {
                source: ChunkSource::Memory {
                    bytes: arc_u32_bytes(&[1, 3, 0, 4]),
                    decoded: false,
                },
                decoded_bytes: 4 * std::mem::size_of::<u32>(),
            },
            Chunk {
                source: ChunkSource::Memory {
                    bytes: arc_u32_bytes(&[2, 3]),
                    decoded: false,
                },
                decoded_bytes: 2 * std::mem::size_of::<u32>(),
            },
        ];
        let data_chunks = vec![
            Chunk {
                source: ChunkSource::File {
                    file,
                    offset: data_locations[0].offset,
                    len: data_locations[0].len,
                },
                decoded_bytes: 4 * std::mem::size_of::<u32>(),
            },
            Chunk {
                source: ChunkSource::Memory {
                    bytes: arc_u32_bytes(&[200, 3000]),
                    decoded: false,
                },
                decoded_bytes: 2 * std::mem::size_of::<u32>(),
            },
        ];
        let grid = ArrayGrid::Regular {
            chunk_shape: vec![4],
            grid_shape: vec![2],
            edge: EdgeChunkLayout::Cropped,
        };
        let dataset = SparseCsrDataset {
            genes,
            indptr: vec![0, 2, 5, 6],
            indices: Array {
                shape: vec![6],
                dtype: DType::U32,
                codec: Arc::new(UncompressedCodec),
                grid: grid.clone(),
                chunks: index_chunks,
                files: Vec::new(),
            },
            data: Array {
                shape: vec![6],
                dtype: DType::U32,
                codec: Arc::new(UncompressedCodec),
                grid,
                chunks: data_chunks,
                files: vec![file],
            },
            index_dtype: DType::U32,
            num_cells: 3,
            num_genes: 5,
        };
        bank.registry
            .register(Dataset::SparseCsr(dataset))
            .expect("register mixed CSR")
    }

    fn u32_array_meta(chunks: Vec<Arc<[u8]>>) -> ArrayMeta {
        ArrayMeta {
            shape: vec![6],
            chunk_shape: vec![4],
            chunk_grid_shape: vec![2],
            dtype: DType::U32,
            order: ArrayOrder::C,
            codec: ArrayCodecMeta::Uncompressed,
            chunks: ChunkStoreMeta::Memory { chunks },
            variable_chunks: false,
            chunk_boundaries: None,
        }
    }
}
