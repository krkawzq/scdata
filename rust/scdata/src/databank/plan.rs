use super::array::{chunk_ref, logical_chunk_index, Array, ChunkRef};
use super::dataset::{Dense1DDataset, Dense2DDataset, SparseCsrDataset};
use super::error::{DataBankError, DataBankResult};

#[derive(Debug, Clone)]
pub struct DenseSegment {
    pub output_row: usize,
    pub output_col_start: usize,
    pub output_cols: usize,
    pub chunk: ChunkRef,
    pub source: ByteRange,
}

#[derive(Debug, Clone)]
pub struct RangeSegment {
    pub chunk: ChunkRef,
    pub source: ByteRange,
    pub elements: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ByteRange {
    pub start: usize,
    pub end: usize,
}

impl ByteRange {
    pub fn new(start: usize, end: usize) -> DataBankResult<Self> {
        if start > end {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "invalid byte range [{start}, {end})"
            )));
        }
        Ok(Self { start, end })
    }

    pub fn len(self) -> usize {
        self.end - self.start
    }
}

#[derive(Debug, Clone)]
pub struct SparseSegment {
    pub indices: Vec<RangeSegment>,
    pub data: Vec<RangeSegment>,
}

#[derive(Debug, Clone, Copy)]
pub struct SparseRowSpan {
    pub output_row: usize,
    pub start: usize,
    pub nnz: usize,
}

pub fn plan_dense_2d(
    dataset: &Dense2DDataset,
    cells: &[usize],
) -> DataBankResult<Vec<DenseSegment>> {
    let segment_capacity = cells
        .len()
        .checked_mul(dataset.data.chunk_grid_shape[1])
        .ok_or_else(|| {
            DataBankError::InvalidArrayMeta("Dense2D segment count overflow".to_string())
        })?;
    let mut segments = Vec::with_capacity(segment_capacity);
    for (output_row, &cell) in cells.iter().enumerate() {
        validate_cell(cell, dataset.num_cells)?;
        let chunk_row = cell / dataset.data.chunk_shape[0];
        let row_in_chunk = cell % dataset.data.chunk_shape[0];

        for chunk_col in 0..dataset.data.chunk_grid_shape[1] {
            let chunk_index =
                logical_chunk_index(&[chunk_row, chunk_col], &dataset.data.chunk_grid_shape)?;
            let col_start = chunk_col * dataset.data.chunk_shape[1];
            let cols = dataset.data.chunk_shape[1].min(dataset.num_genes - col_start);
            let row_bytes = row_slice_bytes(&dataset.data, row_in_chunk, cols)?;
            segments.push(DenseSegment {
                output_row,
                output_col_start: col_start,
                output_cols: cols,
                chunk: chunk_ref(&dataset.data, chunk_index)?,
                source: ByteRange::new(row_bytes.0, row_bytes.1)?,
            });
        }
    }
    Ok(segments)
}

pub fn plan_dense_1d(
    dataset: &Dense1DDataset,
    cells: &[usize],
) -> DataBankResult<Vec<DenseSegment>> {
    let mut segments = Vec::with_capacity(cells.len());
    for (output_row, &cell) in cells.iter().enumerate() {
        validate_cell(cell, dataset.num_cells)?;
        let row_start = cell.checked_mul(dataset.num_genes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("1D dense row start overflow".to_string())
        })?;
        let row_end = row_start.checked_add(dataset.num_genes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("1D dense row end overflow".to_string())
        })?;

        let mut output_col_start = 0usize;
        for range in plan_1d_range(&dataset.data, row_start, row_end)? {
            segments.push(DenseSegment {
                output_row,
                output_col_start,
                output_cols: range.elements,
                chunk: range.chunk,
                source: range.source,
            });
            output_col_start = output_col_start
                .checked_add(range.elements)
                .ok_or_else(|| {
                    DataBankError::InvalidArrayMeta("1D dense output column overflow".to_string())
                })?;
        }
    }
    Ok(segments)
}

pub fn plan_sparse(
    dataset: &SparseCsrDataset,
    cells: &[usize],
) -> DataBankResult<Vec<SparseSegment>> {
    let rows = plan_sparse_rows(dataset, cells)?;
    let mut segments = Vec::with_capacity(rows.len());
    for row in rows {
        let end = row.start.checked_add(row.nnz).ok_or_else(|| {
            DataBankError::IndptrInvalid("CSR row range end overflows usize".to_string())
        })?;
        segments.push(SparseSegment {
            indices: plan_1d_range(&dataset.indices, row.start, end)?,
            data: plan_1d_range(&dataset.data, row.start, end)?,
        });
    }
    Ok(segments)
}

pub fn plan_sparse_rows(
    dataset: &SparseCsrDataset,
    cells: &[usize],
) -> DataBankResult<Vec<SparseRowSpan>> {
    let mut rows = Vec::with_capacity(cells.len());
    for (output_row, &cell) in cells.iter().enumerate() {
        rows.push(plan_sparse_row(dataset, output_row, cell)?);
    }
    Ok(rows)
}

pub unsafe fn plan_sparse_rows_unchecked(
    dataset: &SparseCsrDataset,
    cells: &[usize],
) -> Vec<SparseRowSpan> {
    let mut rows = Vec::with_capacity(cells.len());
    for (output_row, &cell) in cells.iter().enumerate() {
        // SAFETY: the caller guarantees every requested cell is in range and
        // `indptr` is a valid monotonic CSR pointer array.
        let start = unsafe { *dataset.indptr.get_unchecked(cell) as usize };
        let end = unsafe { *dataset.indptr.get_unchecked(cell + 1) as usize };
        rows.push(SparseRowSpan {
            output_row,
            start,
            nnz: end - start,
        });
    }
    rows
}

fn plan_sparse_row(
    dataset: &SparseCsrDataset,
    output_row: usize,
    cell: usize,
) -> DataBankResult<SparseRowSpan> {
    validate_cell(cell, dataset.num_cells)?;
    let start = usize::try_from(dataset.indptr[cell]).map_err(|_| {
        DataBankError::IndptrInvalid("CSR row start does not fit in usize".to_string())
    })?;
    let end = usize::try_from(dataset.indptr[cell + 1]).map_err(|_| {
        DataBankError::IndptrInvalid("CSR row end does not fit in usize".to_string())
    })?;
    let nnz = end
        .checked_sub(start)
        .ok_or_else(|| DataBankError::IndptrInvalid("CSR row range is invalid".to_string()))?;
    Ok(SparseRowSpan {
        output_row,
        start,
        nnz,
    })
}

pub fn plan_1d_range(array: &Array, start: usize, end: usize) -> DataBankResult<Vec<RangeSegment>> {
    if array.shape.len() != 1 {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "1D range planning requires 1D array, got shape {:?}",
            array.shape
        )));
    }
    if start > end || end > array.shape[0] {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "invalid 1D range [{start}, {end}) for length {}",
            array.shape[0]
        )));
    }
    if start == end {
        return Ok(Vec::new());
    }

    let item_size = array.dtype.item_size();
    let chunk_len = array.chunk_shape[0];
    let mut segments = Vec::with_capacity(range_piece_count(start, end, chunk_len));
    let mut pos = start;
    while pos < end {
        let chunk_index = pos / chunk_len;
        let in_chunk = pos % chunk_len;
        let chunk_start = chunk_index.checked_mul(chunk_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("1D chunk start overflow".to_string())
        })?;
        let physical_chunk_len = chunk_len.min(array.shape[0] - chunk_start);
        let elements = (end - pos).min(physical_chunk_len - in_chunk);
        let byte_start = in_chunk.checked_mul(item_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("range byte start overflow".to_string())
        })?;
        let byte_len = elements.checked_mul(item_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("range byte length overflow".to_string())
        })?;
        let byte_end = byte_start.checked_add(byte_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("range byte end overflow".to_string())
        })?;

        segments.push(RangeSegment {
            chunk: chunk_ref(array, chunk_index)?,
            source: ByteRange::new(byte_start, byte_end)?,
            elements,
        });
        pos += elements;
    }
    Ok(segments)
}

pub(super) fn range_piece_count(start: usize, end: usize, chunk_len: usize) -> usize {
    debug_assert!(chunk_len > 0);
    if start == end {
        return 0;
    }
    debug_assert!(start < end);
    (end - 1) / chunk_len - start / chunk_len + 1
}

fn validate_cell(cell: usize, num_cells: usize) -> DataBankResult<()> {
    if cell >= num_cells {
        return Err(DataBankError::CellIndexOutOfRange { cell, num_cells });
    }
    Ok(())
}

fn row_slice_bytes(
    array: &Array,
    row_in_chunk: usize,
    cols: usize,
) -> DataBankResult<(usize, usize)> {
    let item_size = array.dtype.item_size();
    let row_width = cols
        .checked_mul(item_size)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte width overflow".to_string()))?;
    let start = row_in_chunk
        .checked_mul(row_width)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte offset overflow".to_string()))?;
    let end = start
        .checked_add(row_width)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte end overflow".to_string()))?;
    Ok((start, end))
}
