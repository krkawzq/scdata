use super::*;

pub(crate) fn plan_sparse_batch<T: DataValue>(
    dataset: &SparseCsrDataset,
    rows: &[SparseRowSpan],
) -> DataBankResult<SparseBatchPlan> {
    plan_sparse_batch_with_value_size(dataset, rows, T::DTYPE.item_size())
}

pub(crate) fn plan_sparse_batch_with_value_size(
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
        index_piece_capacity = index_piece_capacity
            .saturating_add(dataset.indices.range_piece_count_1d(row.start, end)?);
        data_piece_capacity =
            data_piece_capacity.saturating_add(dataset.data.range_piece_count_1d(row.start, end)?);
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
        projected_indices: None,
    })
}

pub(crate) fn push_sparse_index_pieces(
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
            projection_filtered: false,
            contiguous_output_start: None,
            projected_index_offset: None,
        })?;
        Ok(())
    })
}

#[derive(Clone, Copy)]
pub(crate) struct SparseDataPieceContext {
    value_size: usize,
    index_size: usize,
    output_row: usize,
    row_index_offset: usize,
}

pub(crate) fn push_sparse_data_pieces(
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
            projection_filtered: false,
            contiguous_output_start: None,
            projected_index_offset: None,
        })?;
        Ok(())
    })
}

pub(crate) fn plan_sparse_selected_data_batch(
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: &[u8],
    gene_axis: &GeneAxisPlan,
) -> DataBankResult<SparseBatchPlan> {
    let Some(projection) = gene_axis.projection() else {
        return Ok(SparseBatchPlan {
            index_pieces: Vec::new(),
            data_pieces: plan.data_pieces.clone(),
            index_groups: Vec::new(),
            data_groups: plan.data_groups.clone(),
            index_bytes: plan.index_bytes,
            projected_indices: None,
        });
    };

    match dataset.index_dtype {
        DType::U32 => {
            plan_sparse_selected_data_batch_typed::<u32>(dataset, plan, index_bytes, projection)
        }
        DType::I32 => {
            plan_sparse_selected_data_batch_typed::<i32>(dataset, plan, index_bytes, projection)
        }
        DType::U64 => {
            plan_sparse_selected_data_batch_typed::<u64>(dataset, plan, index_bytes, projection)
        }
        DType::I64 => {
            plan_sparse_selected_data_batch_typed::<i64>(dataset, plan, index_bytes, projection)
        }
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

fn plan_sparse_selected_data_batch_typed<I>(
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: &[u8],
    projection: &CompiledGeneProjection,
) -> DataBankResult<SparseBatchPlan>
where
    I: CsrIndex,
{
    let mut builder = SparsePieceGroupBuilder::with_capacity(plan.data_pieces.len());
    let contiguous_selected_sources = projection.contiguous_selected_source_range();
    let contiguous_selected_source_output_start =
        projection.contiguous_selected_source_output_start();
    let mut projected_indices = ProjectedIndexBuilder::new(projection.output_genes());
    for piece in &plan.data_pieces {
        push_selected_sparse_data_piece_runs::<I>(
            dataset,
            piece,
            index_bytes,
            projection,
            contiguous_selected_sources,
            contiguous_selected_source_output_start,
            &mut projected_indices,
            &mut builder,
        )?;
    }
    let (data_pieces, data_groups) = builder.finish();
    Ok(SparseBatchPlan {
        index_pieces: Vec::new(),
        data_pieces,
        index_groups: Vec::new(),
        data_groups,
        index_bytes: plan.index_bytes,
        projected_indices: projected_indices.finish(),
    })
}

#[allow(clippy::too_many_arguments)]
fn push_selected_sparse_data_piece_runs<I>(
    dataset: &SparseCsrDataset,
    piece: &SparseReadPiece,
    index_bytes: &[u8],
    projection: &CompiledGeneProjection,
    contiguous_selected_sources: Option<(usize, usize)>,
    contiguous_selected_source_output_start: Option<(usize, usize)>,
    projected_indices: &mut ProjectedIndexBuilder,
    builder: &mut SparsePieceGroupBuilder,
) -> DataBankResult<()>
where
    I: CsrIndex,
{
    if piece.elements == 0 {
        return Ok(());
    }
    let value_size = piece.bytes.checked_div(piece.elements).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR selected data piece has zero elements".to_string())
    })?;
    if value_size == 0 || value_size * piece.elements != piece.bytes {
        return Err(DataBankError::InvalidArrayMeta(
            "CSR selected data piece byte length is not divisible by elements".to_string(),
        ));
    }
    let index_size = std::mem::size_of::<I>();
    let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR selected index slice length overflow".to_string())
    })?;
    let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR selected index slice offset overflow".to_string())
    })?;
    if index_end > index_bytes.len() {
        return Err(DataBankError::InvalidArrayMeta(
            "CSR selected index scan is out of range".to_string(),
        ));
    }

    let index_ptr = index_bytes[piece.index_offset..index_end]
        .as_ptr()
        .cast::<I>();
    let mut run_start = None;
    let mut run_projected_start = None;
    let sorted_contiguous_end = if super::aot::assume_sorted_csr_indices() {
        contiguous_selected_sources.map(|(_, end)| end)
    } else {
        None
    };
    for nz in 0..piece.elements {
        let gene = unsafe { std::ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
        if gene >= dataset.num_genes {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene,
                num_genes: dataset.num_genes,
            });
        }
        if sorted_contiguous_end.is_some_and(|end| gene >= end) {
            if let Some(start) = run_start.take() {
                let projected_start = run_projected_start.take();
                push_selected_sparse_data_run(
                    piece,
                    start,
                    nz,
                    value_size,
                    index_size,
                    None,
                    projected_start,
                    builder,
                )?;
            }
            break;
        }
        let selected = if let Some((start, end)) = contiguous_selected_sources {
            gene >= start && gene < end
        } else {
            projection.output_for_source(gene).is_some()
        };
        if selected {
            if run_start.is_none() {
                run_start = Some(nz);
                run_projected_start = Some(projected_indices.len());
            }
            let output_col = if let Some((source_start, output_start)) =
                contiguous_selected_source_output_start
            {
                output_start + (gene - source_start)
            } else {
                projection.output_for_source(gene).ok_or_else(|| {
                    DataBankError::InvalidArrayMeta(
                        "selected CSR gene is missing from projection".to_string(),
                    )
                })?
            };
            projected_indices.push(output_col)?;
        } else if let Some(start) = run_start.take() {
            let projected_start = run_projected_start.take();
            push_selected_sparse_data_run(
                piece,
                start,
                nz,
                value_size,
                index_size,
                None,
                projected_start,
                builder,
            )?;
        }
    }
    if let Some(start) = run_start {
        let projected_start = run_projected_start.take();
        push_selected_sparse_data_run(
            piece,
            start,
            piece.elements,
            value_size,
            index_size,
            None,
            projected_start,
            builder,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_selected_sparse_data_run(
    piece: &SparseReadPiece,
    run_start: usize,
    run_end: usize,
    value_size: usize,
    index_size: usize,
    contiguous_output_start: Option<usize>,
    projected_index_offset: Option<usize>,
    builder: &mut SparsePieceGroupBuilder,
) -> DataBankResult<()> {
    if run_start >= run_end {
        return Ok(());
    }
    let source_start_delta = run_start.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR selected data source start overflow".to_string())
    })?;
    let source_end_delta = run_end.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR selected data source end overflow".to_string())
    })?;
    let source_start = piece
        .source
        .start
        .checked_add(source_start_delta)
        .ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR selected data source start overflow".to_string())
        })?;
    let source_end = piece
        .source
        .start
        .checked_add(source_end_delta)
        .ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR selected data source end overflow".to_string())
        })?;
    if source_end > piece.source.end {
        return Err(DataBankError::InvalidArrayMeta(
            "CSR selected data run exceeds original data piece".to_string(),
        ));
    }
    let elements = run_end - run_start;
    let bytes = source_end - source_start;
    let index_offset = piece
        .index_offset
        .checked_add(run_start.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR selected index offset overflow".to_string())
        })?)
        .ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR selected index offset overflow".to_string())
        })?;
    builder.push(SparseReadPiece {
        chunk: piece.chunk.clone(),
        source: ByteRange::new(source_start, source_end)?,
        group_offset: 0,
        output_offset: 0,
        output_row: piece.output_row,
        index_offset,
        elements,
        bytes,
        projection_filtered: true,
        contiguous_output_start,
        projected_index_offset,
    })
}

enum ProjectedIndexBuilder {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl ProjectedIndexBuilder {
    fn new(output_genes: usize) -> Self {
        if output_genes <= usize::from(u16::MAX) + 1 {
            Self::U16(Vec::new())
        } else {
            Self::U32(Vec::new())
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
        }
    }

    fn push(&mut self, output_col: usize) -> DataBankResult<()> {
        match self {
            Self::U16(values) => {
                let value =
                    u16::try_from(output_col).map_err(|_| DataBankError::GeneIndexOutOfRange {
                        gene: output_col,
                        num_genes: usize::from(u16::MAX) + 1,
                    })?;
                values.push(value);
            }
            Self::U32(values) => {
                let value = u32::try_from(output_col).map_err(|_| {
                    DataBankError::CsrIndexInvalid(
                        "projected output column does not fit in u32".to_string(),
                    )
                })?;
                values.push(value);
            }
        }
        Ok(())
    }

    fn finish(self) -> Option<ProjectedIndexBuffer> {
        match self {
            Self::U16(values) if values.is_empty() => None,
            Self::U16(values) => Some(ProjectedIndexBuffer::U16(values)),
            Self::U32(values) if values.is_empty() => None,
            Self::U32(values) => Some(ProjectedIndexBuffer::U32(values)),
        }
    }
}

pub(crate) struct PlannedRangePiece {
    chunk: ChunkRef,
    source: ByteRange,
    elements: usize,
    bytes: usize,
}

pub(crate) fn push_sparse_range_pieces<F>(
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

    plan::for_each_1d_range(array, start, end, |range| {
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
        })
    })
}

pub(crate) struct SparsePieceGroupBuilder {
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

pub(crate) fn sparse_group_key(chunk: &ChunkRef) -> SparseGroupKey {
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

pub(crate) fn sparse_group_source_key(source: &SparseGroupSource) -> SparseGroupKey {
    match source {
        SparseGroupSource::AccessItem(item) => SparseGroupKey::File {
            key: item.key,
            codec: codec_id(&item.codec),
            expected_size: item.expected_size,
        },
        SparseGroupSource::Memory {
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

pub(crate) fn sparse_group_source(chunk: &ChunkRef) -> SparseGroupSource {
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

pub(crate) fn append_sparse_group_slice(
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

pub(crate) fn sparse_group_access_item(group: &SparseReadGroup) -> DataBankResult<AccessItem> {
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

pub(crate) fn file_sparse_group_access_item(group: &SparseReadGroup) -> AccessItem {
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
