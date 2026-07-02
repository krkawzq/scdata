use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use super::array::DataValue;
use super::dataset::Dataset;
use super::error::{DataBankError, DataBankResult};
use super::interner::GeneNameView;

pub(super) type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
pub(super) type FastHashSet<K> = HashSet<K, BuildHasherDefault<FastHasher>>;
pub(super) const GENE_NOT_SELECTED: usize = usize::MAX;

#[inline]
pub(super) fn row_count_for_width(len: usize, width: usize) -> usize {
    len.checked_div(width).unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingGenePolicy {
    #[default]
    Zero,
    Error,
}

#[derive(Default)]
pub(super) struct FastHasher(u64);

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

pub(super) fn fast_hash_map_with_capacity<K, V>(capacity: usize) -> FastHashMap<K, V> {
    HashMap::with_capacity_and_hasher(capacity, BuildHasherDefault::<FastHasher>::default())
}

pub(super) fn fast_hash_set_with_capacity<K>(capacity: usize) -> FastHashSet<K> {
    HashSet::with_capacity_and_hasher(capacity, BuildHasherDefault::<FastHasher>::default())
}

#[derive(Debug, Clone)]
pub(super) struct CompiledGeneProjection {
    pub(super) output_by_source: Vec<usize>,
    pub(super) output_names: Vec<GeneNameView>,
    pub(super) selected_sources: Vec<usize>,
    contiguous_selected_source_output_start: Option<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub(super) enum GeneAxisPlan {
    DatasetOrder,
    Requested(Arc<CompiledGeneProjection>),
}

impl GeneAxisPlan {
    pub(super) fn dataset_order() -> Self {
        Self::DatasetOrder
    }

    pub(super) fn requested<G>(
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
            Ok(Self::Requested(Arc::new(projection)))
        }
    }

    pub(super) fn output_genes(&self, dataset_num_genes: usize) -> usize {
        match self {
            Self::DatasetOrder => dataset_num_genes,
            Self::Requested(projection) => projection.output_genes(),
        }
    }

    pub(super) fn output_names<'a>(&'a self, dataset: &'a Dataset) -> &'a [GeneNameView] {
        match self {
            Self::DatasetOrder => dataset.genes().views(),
            Self::Requested(projection) => projection.output_names(),
        }
    }

    pub(super) fn fill_names(
        &self,
        dataset: &Dataset,
        names: &mut [GeneNameView],
    ) -> DataBankResult<()> {
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

    pub(super) fn projection(&self) -> Option<&CompiledGeneProjection> {
        match self {
            Self::DatasetOrder => None,
            Self::Requested(projection) => Some(projection.as_ref()),
        }
    }

    pub(super) fn requires_dense_zero_fill(&self) -> bool {
        match self {
            Self::DatasetOrder => false,
            Self::Requested(projection) => projection.has_missing_outputs(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MultiBatchCells {
    cells: Vec<usize>,
    parts: Vec<BatchPartRange>,
}

#[derive(Debug, Clone, Copy)]
struct BatchPartRange {
    dataset_idx: usize,
    start: usize,
    len: usize,
}

impl MultiBatchCells {
    pub fn new(parts: Vec<(usize, Vec<usize>)>) -> Self {
        let total_cells = parts.iter().map(|(_, cells)| cells.len()).sum();
        let mut out_cells = Vec::with_capacity(total_cells);
        let mut out_parts = Vec::with_capacity(parts.len());
        for (dataset_idx, cells) in parts {
            let start = out_cells.len();
            let len = cells.len();
            out_cells.extend(cells);
            out_parts.push(BatchPartRange {
                dataset_idx,
                start,
                len,
            });
        }
        Self {
            cells: out_cells,
            parts: out_parts,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn from_flat_parts(cells: Vec<usize>, parts: Vec<(usize, usize)>) -> Self {
        let mut start = 0usize;
        let mut out_parts = Vec::with_capacity(parts.len());
        for (dataset_idx, len) in parts {
            out_parts.push(BatchPartRange {
                dataset_idx,
                start,
                len,
            });
            start += len;
        }
        debug_assert_eq!(start, cells.len());
        Self {
            cells,
            parts: out_parts,
        }
    }

    pub(super) fn from_single(cells: Vec<usize>) -> Self {
        let len = cells.len();
        Self {
            cells,
            parts: vec![BatchPartRange {
                dataset_idx: 0,
                start: 0,
                len,
            }],
        }
    }

    pub(super) fn part_count(&self) -> usize {
        self.parts.len()
    }

    pub(super) fn part_slices(&self) -> impl Iterator<Item = (usize, &[usize])> + '_ {
        self.parts.iter().map(|part| {
            let end = part.start + part.len;
            (part.dataset_idx, &self.cells[part.start..end])
        })
    }

    pub(super) fn into_cells(self) -> Vec<usize> {
        self.cells
    }

    pub(super) fn total_cells(&self) -> DataBankResult<usize> {
        Ok(self.cells.len())
    }
}

#[derive(Debug, Clone)]
pub(super) struct MultiGeneAxisPlan {
    pub(super) output_names: Vec<GeneNameView>,
    pub(super) output_genes: usize,
    pub(super) per_dataset: Vec<GeneAxisPlan>,
}

impl MultiGeneAxisPlan {
    pub(super) fn from_single(dataset: &Dataset, gene_axis: GeneAxisPlan) -> Self {
        let output_names = gene_axis.output_names(dataset).to_vec();
        let output_genes = output_names.len();
        Self {
            output_names,
            output_genes,
            per_dataset: vec![gene_axis],
        }
    }

    pub(super) fn dataset_order(datasets: &[Arc<Dataset>]) -> DataBankResult<Self> {
        let first = datasets.first().ok_or_else(|| {
            DataBankError::InvalidConfig("prefetch requires at least one dataset".to_string())
        })?;
        for dataset in datasets.iter().skip(1) {
            validate_same_gene_axis(first.as_ref(), dataset.as_ref())?;
        }
        Ok(Self {
            output_names: first.genes().views().to_vec(),
            output_genes: first.num_genes(),
            per_dataset: vec![GeneAxisPlan::DatasetOrder; datasets.len()],
        })
    }

    pub(super) fn requested<G>(
        datasets: &[Arc<Dataset>],
        gene_names: &[G],
        missing: MissingGenePolicy,
    ) -> DataBankResult<Self>
    where
        G: AsRef<str>,
    {
        if datasets.is_empty() {
            return Err(DataBankError::InvalidConfig(
                "prefetch requires at least one dataset".to_string(),
            ));
        }
        let mut per_dataset = Vec::with_capacity(datasets.len());
        for dataset in datasets {
            per_dataset.push(GeneAxisPlan::requested(
                dataset.as_ref(),
                gene_names,
                missing,
            )?);
        }

        let mut output_names = vec![GeneNameView::empty(); gene_names.len()];
        for (axis, dataset) in per_dataset.iter().zip(datasets.iter()) {
            let names = match axis {
                GeneAxisPlan::DatasetOrder => dataset.genes().views(),
                GeneAxisPlan::Requested(projection) => projection.output_names(),
            };
            for (dst, src) in output_names.iter_mut().zip(names.iter().copied()) {
                if dst.is_empty() && !src.is_empty() {
                    *dst = src;
                }
            }
        }

        Ok(Self {
            output_names,
            output_genes: gene_names.len(),
            per_dataset,
        })
    }

    pub(super) fn axis_for(&self, dataset_idx: usize) -> DataBankResult<&GeneAxisPlan> {
        self.per_dataset.get(dataset_idx).ok_or_else(|| {
            DataBankError::InvalidConfig(format!(
                "multi batch references dataset index {dataset_idx}, but only {} datasets were supplied",
                self.per_dataset.len()
            ))
        })
    }
}

pub(super) fn validate_same_gene_axis(left: &Dataset, right: &Dataset) -> DataBankResult<()> {
    if left.num_genes() != right.num_genes() {
        return Err(DataBankError::InvalidConfig(format!(
            "multi-dataset prefetch without gene_names requires identical gene counts, got {} and {}",
            left.num_genes(),
            right.num_genes()
        )));
    }
    for (idx, (left_name, right_name)) in left
        .genes()
        .names()
        .iter()
        .zip(right.genes().names().iter())
        .enumerate()
    {
        if left_name.as_ref() != right_name.as_ref() {
            return Err(DataBankError::InvalidConfig(format!(
                "multi-dataset prefetch without gene_names requires identical gene order; gene {idx} differs"
            )));
        }
    }
    Ok(())
}

pub(super) fn requested_matches_dataset_order<G>(
    dataset: &Dataset,
    requested: &[G],
) -> DataBankResult<bool>
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
    pub(super) fn new<G>(
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
        let contiguous_selected_source_output_start =
            contiguous_selected_source_output_start(&selected_sources, &output_by_source);
        Ok(Self {
            output_by_source,
            output_names,
            selected_sources,
            contiguous_selected_source_output_start,
        })
    }

    pub(super) fn output_genes(&self) -> usize {
        self.output_names.len()
    }

    pub(super) fn output_names(&self) -> &[GeneNameView] {
        &self.output_names
    }

    pub(super) fn output_for_source(&self, source: usize) -> Option<usize> {
        let &output = self.output_by_source.get(source)?;
        (output != GENE_NOT_SELECTED).then_some(output)
    }

    pub(super) fn has_missing_outputs(&self) -> bool {
        self.selected_sources.len() != self.output_names.len()
    }

    pub(super) fn contiguous_selected_source_range(&self) -> Option<(usize, usize)> {
        let (&start, rest) = self.selected_sources.split_first()?;
        let end = start.checked_add(self.selected_sources.len())?;
        for (offset, &source) in std::iter::once(&start).chain(rest.iter()).enumerate() {
            if source != start.checked_add(offset)? {
                return None;
            }
        }
        Some((start, end))
    }

    pub(super) fn contiguous_selected_source_output_start(&self) -> Option<(usize, usize)> {
        self.contiguous_selected_source_output_start
    }

    pub(super) fn is_identity(&self, dataset_num_genes: usize) -> bool {
        self.output_names.len() == dataset_num_genes
            && self.contiguous_output_for_source_run(0, dataset_num_genes) == Some(0)
    }

    pub(super) fn contiguous_output_for_source_run(
        &self,
        source_start: usize,
        len: usize,
    ) -> Option<usize> {
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

fn contiguous_selected_source_output_start(
    selected_sources: &[usize],
    output_by_source: &[usize],
) -> Option<(usize, usize)> {
    let (&source_start, rest) = selected_sources.split_first()?;
    let output_start = *output_by_source.get(source_start)?;
    if output_start == GENE_NOT_SELECTED {
        return None;
    }
    for (offset, &source) in std::iter::once(&source_start)
        .chain(rest.iter())
        .enumerate()
    {
        if source != source_start.checked_add(offset)? {
            return None;
        }
        let output = output_start.checked_add(offset)?;
        if *output_by_source.get(source)? != output {
            return None;
        }
    }
    Some((source_start, output_start))
}

pub(super) fn validate_dtype_and_cells<T: DataValue>(
    dataset: &Dataset,
    cells: &[usize],
) -> DataBankResult<()> {
    let src_dtype = dataset.data_dtype();
    if !src_dtype.can_cast_to(T::DTYPE) {
        return Err(DataBankError::CannotCast {
            src: src_dtype,
            dst: T::DTYPE,
            reason: "access_cells output dtype cannot hold source values (float→int forbidden)",
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
