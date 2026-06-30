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
        })?;
        Ok(())
    })
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
