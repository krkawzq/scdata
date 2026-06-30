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
    plan_dense_2d_impl(dataset, cells.iter().copied().enumerate(), cells.len())
}

pub fn plan_dense_2d_with_output_rows(
    dataset: &Dense2DDataset,
    cells: &[usize],
    output_rows: &[usize],
) -> DataBankResult<Vec<DenseSegment>> {
    validate_cell_rows(cells, output_rows, "Dense2D planning")?;
    plan_dense_2d_impl(
        dataset,
        cells
            .iter()
            .copied()
            .zip(output_rows.iter().copied())
            .map(|(cell, output_row)| (output_row, cell)),
        cells.len(),
    )
}

fn plan_dense_2d_impl<I>(
    dataset: &Dense2DDataset,
    rows: I,
    row_count: usize,
) -> DataBankResult<Vec<DenseSegment>>
where
    I: IntoIterator<Item = (usize, usize)>,
{
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

    let segment_capacity = row_count.checked_mul(*grid_cols).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("Dense2D segment count overflow".to_string())
    })?;
    let mut segments = Vec::with_capacity(segment_capacity);
    for (output_row, cell) in rows {
        validate_cell(cell, dataset.num_cells)?;
        let chunk_row = cell / *chunk_rows;
        let row_in_chunk = cell % *chunk_rows;

        for chunk_col in 0..*grid_cols {
            let chunk_index = chunk_row
                .checked_mul(*grid_cols)
                .and_then(|base| base.checked_add(chunk_col))
                .ok_or_else(|| {
                    DataBankError::InvalidArrayMeta("Dense2D chunk index overflow".to_string())
                })?;
            let col_start = chunk_col.checked_mul(*chunk_cols).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense2D column start overflow".to_string())
            })?;
            let cols = (*chunk_cols).min(dataset.num_genes - col_start);
            let row_bytes =
                row_slice_bytes(&dataset.data, chunk_row, chunk_col, row_in_chunk, 0, cols)?;
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

pub fn plan_dense_2d_selected_sources(
    dataset: &Dense2DDataset,
    cells: &[usize],
    selected_sources: &[usize],
) -> DataBankResult<Vec<DenseSegment>> {
    plan_dense_2d_selected_sources_impl(
        dataset,
        cells.iter().copied().enumerate(),
        cells.len(),
        selected_sources,
    )
}

pub fn plan_dense_2d_selected_sources_with_output_rows(
    dataset: &Dense2DDataset,
    cells: &[usize],
    output_rows: &[usize],
    selected_sources: &[usize],
) -> DataBankResult<Vec<DenseSegment>> {
    validate_cell_rows(cells, output_rows, "Dense2D projected planning")?;
    plan_dense_2d_selected_sources_impl(
        dataset,
        cells
            .iter()
            .copied()
            .zip(output_rows.iter().copied())
            .map(|(cell, output_row)| (output_row, cell)),
        cells.len(),
        selected_sources,
    )
}

fn plan_dense_2d_selected_sources_impl<I>(
    dataset: &Dense2DDataset,
    rows: I,
    row_count: usize,
    selected_sources: &[usize],
) -> DataBankResult<Vec<DenseSegment>>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    if selected_sources.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_shape = dataset
        .data
        .regular_chunk_shape_required("Dense2D projected planning")?;
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

    let segment_capacity = row_count
        .checked_mul(selected_sources.len())
        .ok_or_else(|| {
            DataBankError::InvalidArrayMeta("Dense2D selected segment count overflow".to_string())
        })?;
    let mut segments = Vec::with_capacity(segment_capacity);
    for (output_row, cell) in rows {
        validate_cell(cell, dataset.num_cells)?;
        let chunk_row = cell / *chunk_rows;
        let row_in_chunk = cell % *chunk_rows;
        let mut selected_pos = 0usize;

        while let Some(&run_start) = selected_sources.get(selected_pos) {
            if run_start >= dataset.num_genes {
                return Err(DataBankError::GeneIndexOutOfRange {
                    gene: run_start,
                    num_genes: dataset.num_genes,
                });
            }
            let chunk_col = run_start / *chunk_cols;
            if chunk_col >= *grid_cols {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "Dense2D selected chunk column {chunk_col} exceeds grid columns {grid_cols}"
                )));
            }
            let col_start = chunk_col.checked_mul(*chunk_cols).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense2D column start overflow".to_string())
            })?;
            let col_end = col_start
                .checked_add((*chunk_cols).min(dataset.num_genes - col_start))
                .ok_or_else(|| {
                    DataBankError::InvalidArrayMeta("Dense2D column end overflow".to_string())
                })?;

            let mut run_len = 1usize;
            selected_pos += 1;
            while let Some(&source) = selected_sources.get(selected_pos) {
                if source >= col_end || source != run_start + run_len {
                    break;
                }
                run_len += 1;
                selected_pos += 1;
            }

            let local_col = run_start - col_start;
            let row_bytes = row_slice_bytes(
                &dataset.data,
                chunk_row,
                chunk_col,
                row_in_chunk,
                local_col,
                run_len,
            )?;
            let chunk_index = chunk_row
                .checked_mul(*grid_cols)
                .and_then(|base| base.checked_add(chunk_col))
                .ok_or_else(|| {
                    DataBankError::InvalidArrayMeta(
                        "Dense2D selected chunk index overflow".to_string(),
                    )
                })?;
            segments.push(DenseSegment {
                output_row,
                output_col_start: run_start,
                output_cols: run_len,
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
    plan_dense_1d_impl(dataset, cells.iter().copied().enumerate(), cells.len())
}

pub fn plan_dense_1d_with_output_rows(
    dataset: &Dense1DDataset,
    cells: &[usize],
    output_rows: &[usize],
) -> DataBankResult<Vec<DenseSegment>> {
    validate_cell_rows(cells, output_rows, "Dense1D planning")?;
    plan_dense_1d_impl(
        dataset,
        cells
            .iter()
            .copied()
            .zip(output_rows.iter().copied())
            .map(|(cell, output_row)| (output_row, cell)),
        cells.len(),
    )
}

fn plan_dense_1d_impl<I>(
    dataset: &Dense1DDataset,
    rows: I,
    row_count: usize,
) -> DataBankResult<Vec<DenseSegment>>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let mut segments = Vec::with_capacity(row_count);
    for (output_row, cell) in rows {
        validate_cell(cell, dataset.num_cells)?;
        let row_start = cell.checked_mul(dataset.num_genes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("1D dense row start overflow".to_string())
        })?;
        let row_end = row_start.checked_add(dataset.num_genes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("1D dense row end overflow".to_string())
        })?;

        let mut output_col_start = 0usize;
        for_each_1d_range(&dataset.data, row_start, row_end, |range| {
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
            Ok(())
        })?;
    }
    Ok(segments)
}

pub fn plan_dense_1d_selected_sources(
    dataset: &Dense1DDataset,
    cells: &[usize],
    selected_sources: &[usize],
) -> DataBankResult<Vec<DenseSegment>> {
    plan_dense_1d_selected_sources_impl(
        dataset,
        cells.iter().copied().enumerate(),
        cells.len(),
        selected_sources,
    )
}

pub fn plan_dense_1d_selected_sources_with_output_rows(
    dataset: &Dense1DDataset,
    cells: &[usize],
    output_rows: &[usize],
    selected_sources: &[usize],
) -> DataBankResult<Vec<DenseSegment>> {
    validate_cell_rows(cells, output_rows, "Dense1D projected planning")?;
    plan_dense_1d_selected_sources_impl(
        dataset,
        cells
            .iter()
            .copied()
            .zip(output_rows.iter().copied())
            .map(|(cell, output_row)| (output_row, cell)),
        cells.len(),
        selected_sources,
    )
}

fn plan_dense_1d_selected_sources_impl<I>(
    dataset: &Dense1DDataset,
    rows: I,
    row_count: usize,
    selected_sources: &[usize],
) -> DataBankResult<Vec<DenseSegment>>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    if selected_sources.is_empty() {
        return Ok(Vec::new());
    }

    let segment_capacity = row_count
        .checked_mul(selected_sources.len())
        .ok_or_else(|| {
            DataBankError::InvalidArrayMeta("Dense1D selected segment count overflow".to_string())
        })?;
    let mut segments = Vec::with_capacity(segment_capacity);
    for (output_row, cell) in rows {
        validate_cell(cell, dataset.num_cells)?;
        let row_start = cell.checked_mul(dataset.num_genes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("1D dense row start overflow".to_string())
        })?;
        let mut selected_pos = 0usize;

        while let Some(&run_start) = selected_sources.get(selected_pos) {
            if run_start >= dataset.num_genes {
                return Err(DataBankError::GeneIndexOutOfRange {
                    gene: run_start,
                    num_genes: dataset.num_genes,
                });
            }

            let mut run_len = 1usize;
            selected_pos += 1;
            while let Some(&source) = selected_sources.get(selected_pos) {
                if source >= dataset.num_genes || source != run_start + run_len {
                    break;
                }
                run_len += 1;
                selected_pos += 1;
            }

            let range_start = row_start.checked_add(run_start).ok_or_else(|| {
                DataBankError::InvalidArrayMeta(
                    "1D dense selected range start overflow".to_string(),
                )
            })?;
            let range_end = range_start.checked_add(run_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("1D dense selected range end overflow".to_string())
            })?;
            let mut output_col_start = run_start;
            for_each_1d_range(&dataset.data, range_start, range_end, |range| {
                segments.push(DenseSegment {
                    output_row,
                    output_col_start,
                    output_cols: range.elements,
                    chunk: range.chunk,
                    source: range.source,
                });
                output_col_start =
                    output_col_start
                        .checked_add(range.elements)
                        .ok_or_else(|| {
                            DataBankError::InvalidArrayMeta(
                                "1D dense selected output column overflow".to_string(),
                            )
                        })?;
                Ok(())
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

pub fn plan_sparse_rows_with_output_rows(
    dataset: &SparseCsrDataset,
    cells: &[usize],
    output_rows: &[usize],
) -> DataBankResult<Vec<SparseRowSpan>> {
    validate_cell_rows(cells, output_rows, "CSR planning")?;
    let mut rows = Vec::with_capacity(cells.len());
    for (&cell, &output_row) in cells.iter().zip(output_rows.iter()) {
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
    let mut ranges = Vec::with_capacity(array.range_piece_count_1d(start, end)?);
    for_each_1d_range(array, start, end, |range| {
        ranges.push(range);
        Ok(())
    })?;
    Ok(ranges)
}

pub fn for_each_1d_range<F>(
    array: &Array,
    start: usize,
    end: usize,
    mut push: F,
) -> DataBankResult<()>
where
    F: FnMut(RangeSegment) -> DataBankResult<()>,
{
    array.grid.for_each_1d_range(
        &array.shape,
        array.dtype,
        &array.chunks,
        start,
        end,
        |piece| {
            push(RangeSegment {
                chunk: chunk_ref(array, piece.chunk_index)?,
                source: ByteRange::new(piece.byte_start, piece.byte_end)?,
                elements: piece.elements,
            })
        },
    )
}

fn validate_cell(cell: usize, num_cells: usize) -> DataBankResult<()> {
    if cell >= num_cells {
        return Err(DataBankError::CellIndexOutOfRange { cell, num_cells });
    }
    Ok(())
}

fn validate_cell_rows(cells: &[usize], output_rows: &[usize], context: &str) -> DataBankResult<()> {
    if cells.len() != output_rows.len() {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "{context} requires one output row per cell, got {} cells and {} output rows",
            cells.len(),
            output_rows.len()
        )));
    }
    Ok(())
}

fn row_slice_bytes(
    array: &Array,
    chunk_row: usize,
    chunk_col: usize,
    row_in_chunk: usize,
    col_offset: usize,
    cols: usize,
) -> DataBankResult<(usize, usize)> {
    let item_size = array.dtype.item_size();
    let physical_cols = array
        .grid
        .physical_row_width_2d(&array.shape, chunk_row, chunk_col)?;
    let row_width = physical_cols
        .checked_mul(item_size)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte width overflow".to_string()))?;
    let logical_cols_end = col_offset
        .checked_add(cols)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row column range overflow".to_string()))?;
    if logical_cols_end > physical_cols {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "row column range [{col_offset}, {logical_cols_end}) exceeds physical width {physical_cols}"
        )));
    }
    let logical_width = cols
        .checked_mul(item_size)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte length overflow".to_string()))?;
    let start = row_in_chunk
        .checked_mul(row_width)
        .and_then(|row_start| row_start.checked_add(col_offset.checked_mul(item_size)?))
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte offset overflow".to_string()))?;
    let end = start
        .checked_add(logical_width)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("row byte end overflow".to_string()))?;
    Ok((start, end))
}
