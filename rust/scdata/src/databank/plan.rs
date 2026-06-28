use super::array::{chunk_ref, Array, ChunkRef};
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
    let chunk_shape = dataset
        .data
        .regular_chunk_shape_required("Dense2D planning")?;
    let [chunk_rows, chunk_cols] = chunk_shape else {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "Dense2D data must have 2D chunks, got {chunk_shape:?}"
        )));
    };
    let grid_shape = dataset.data.chunk_grid_shape();
    let [_, grid_cols] = grid_shape else {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "Dense2D data must have 2D chunk grid, got {grid_shape:?}"
        )));
    };

    let segment_capacity = cells.len().checked_mul(*grid_cols).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("Dense2D segment count overflow".to_string())
    })?;
    let mut segments = Vec::with_capacity(segment_capacity);
    for (output_row, &cell) in cells.iter().enumerate() {
        validate_cell(cell, dataset.num_cells)?;
        let chunk_row = cell / *chunk_rows;
        let row_in_chunk = cell % *chunk_rows;

        for chunk_col in 0..*grid_cols {
            let chunk_index = dataset
                .data
                .grid
                .logical_chunk_index(&[chunk_row, chunk_col])?;
            let col_start = chunk_col.checked_mul(*chunk_cols).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense2D column start overflow".to_string())
            })?;
            let cols = (*chunk_cols).min(dataset.num_genes - col_start);
            let row_bytes =
                row_slice_bytes(&dataset.data, chunk_row, chunk_col, row_in_chunk, cols)?;
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
    let pieces = array
        .grid
        .plan_1d_range(&array.shape, array.dtype, &array.chunks, start, end)?;
    pieces
        .into_iter()
        .map(|piece| {
            Ok(RangeSegment {
                chunk: chunk_ref(array, piece.chunk_index)?,
                source: ByteRange::new(piece.byte_start, piece.byte_end)?,
                elements: piece.elements,
            })
        })
        .collect()
}

pub(super) fn range_piece_count(array: &Array, start: usize, end: usize) -> DataBankResult<usize> {
    array.range_piece_count_1d(start, end)
}

fn validate_cell(cell: usize, num_cells: usize) -> DataBankResult<()> {
    if cell >= num_cells {
        return Err(DataBankError::CellIndexOutOfRange { cell, num_cells });
    }
    Ok(())
}

fn row_slice_bytes(
    array: &Array,
    chunk_row: usize,
    chunk_col: usize,
    row_in_chunk: usize,
    cols: usize,
) -> DataBankResult<(usize, usize)> {
    let item_size = array.dtype.item_size();
    let physical_cols = array
        .grid
        .physical_row_width_2d(&array.shape, chunk_row, chunk_col)?;
    let row_width = physical_cols
        .checked_mul(item_size)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte width overflow".to_string()))?;
    let logical_width = cols
        .checked_mul(item_size)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte length overflow".to_string()))?;
    let start = row_in_chunk
        .checked_mul(row_width)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte offset overflow".to_string()))?;
    let end = start
        .checked_add(logical_width)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte end overflow".to_string()))?;
    Ok((start, end))
}
