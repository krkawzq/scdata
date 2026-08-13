//! In-memory 2-D gather. Independent of sc-compress store types.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::convert::{copy_u64_1d, dispatch_csr_data, dispatch_dense, CsrData, DenseValues};
use crate::error::{from_compress, invalid_argument};

const MIN_PARALLEL_WORK: usize = 128 * 1024;
const MAX_DIRECT_LOOKUP_BYTES: usize = 32 * 1024 * 1024;
const MISSING_DESTINATION: u32 = u32::MAX;

enum AxisSpec {
    All,
    Range { start: usize, end: usize },
    Positions(Vec<usize>),
}

enum Axis {
    Range { start: usize, end: usize },
    Positions(Vec<usize>),
}

impl AxisSpec {
    fn parse(kind: &str, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        match kind {
            "all" => Ok(Self::All),
            "range" => {
                let (start, end): (u64, u64) = payload.extract().map_err(|_| {
                    invalid_argument("range axis payload must be a (start, end) uint pair")
                })?;
                Ok(Self::Range {
                    start: usize_from_u64(start, "range start")?,
                    end: usize_from_u64(end, "range end")?,
                })
            }
            "positions" => {
                let values = payload
                    .extract::<PyReadonlyArray1<'_, u64>>()
                    .map_err(|_| {
                        invalid_argument("positions axis payload must be a 1-D uint64 array")
                    })?;
                let slice = values.as_slice().map_err(|_| {
                    invalid_argument("positions axis payload must be a C-contiguous uint64 array")
                })?;
                let mut positions = Vec::new();
                try_reserve_exact(&mut positions, slice.len())?;
                for &value in slice {
                    positions.push(usize_from_u64(value, "position")?);
                }
                Ok(Self::Positions(positions))
            }
            other => Err(invalid_argument(format!(
                "unknown axis kind {other:?}; expected 'all', 'range', or 'positions'"
            ))),
        }
    }

    fn normalize(self, axis_len: usize, name: &str) -> PyResult<Axis> {
        match self {
            Self::All => Ok(Axis::Range {
                start: 0,
                end: axis_len,
            }),
            Self::Range { start, end } => {
                if start > end || end > axis_len {
                    return Err(invalid_argument(format!(
                        "{name} range [{start}, {end}) outside 0..{axis_len}"
                    )));
                }
                Ok(Axis::Range { start, end })
            }
            Self::Positions(positions) => {
                for (index, &position) in positions.iter().enumerate() {
                    if position >= axis_len {
                        return Err(invalid_argument(format!(
                            "{name} position[{index}]={position} outside 0..{axis_len}"
                        )));
                    }
                }
                if is_contiguous(&positions) {
                    let start = positions.first().copied().unwrap_or(0);
                    let end = positions.last().map_or(start, |last| last + 1);
                    Ok(Axis::Range { start, end })
                } else {
                    Ok(Axis::Positions(positions))
                }
            }
        }
    }
}

impl Axis {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Range { start, end } => end - start,
            Self::Positions(positions) => positions.len(),
        }
    }

    #[inline]
    fn as_range(&self) -> Option<(usize, usize)> {
        match self {
            Self::Range { start, end } => Some((*start, *end)),
            Self::Positions(_) => None,
        }
    }
}

pub(crate) fn dense_select_numpy<'py>(
    py: Python<'py>,
    values: &Bound<'_, PyAny>,
    row_kind: &str,
    row_payload: &Bound<'_, PyAny>,
    col_kind: &str,
    col_payload: &Bound<'_, PyAny>,
    num_workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let row_spec = AxisSpec::parse(row_kind, row_payload)?;
    let col_spec = AxisSpec::parse(col_kind, col_payload)?;
    dispatch_dense(values, |dense, shape| {
        let n_rows = usize_from_u64(shape[0], "dense row count")?;
        let n_cols = usize_from_u64(shape[1], "dense column count")?;
        let rows = row_spec.normalize(n_rows, "row")?;
        let cols = col_spec.normalize(n_cols, "col")?;
        let out = py.allow_threads(|| match dense {
            DenseValues::U16(src) => wrap_dense(
                src,
                n_rows,
                n_cols,
                &rows,
                &cols,
                num_workers,
                OwnedDense::U16,
            ),
            DenseValues::U32(src) => wrap_dense(
                src,
                n_rows,
                n_cols,
                &rows,
                &cols,
                num_workers,
                OwnedDense::U32,
            ),
            DenseValues::U64(src) => wrap_dense(
                src,
                n_rows,
                n_cols,
                &rows,
                &cols,
                num_workers,
                OwnedDense::U64,
            ),
            DenseValues::I16(src) => wrap_dense(
                src,
                n_rows,
                n_cols,
                &rows,
                &cols,
                num_workers,
                OwnedDense::I16,
            ),
            DenseValues::I32(src) => wrap_dense(
                src,
                n_rows,
                n_cols,
                &rows,
                &cols,
                num_workers,
                OwnedDense::I32,
            ),
            DenseValues::I64(src) => wrap_dense(
                src,
                n_rows,
                n_cols,
                &rows,
                &cols,
                num_workers,
                OwnedDense::I64,
            ),
            DenseValues::F32(src) => wrap_dense(
                src,
                n_rows,
                n_cols,
                &rows,
                &cols,
                num_workers,
                OwnedDense::F32,
            ),
            DenseValues::F64(src) => wrap_dense(
                src,
                n_rows,
                n_cols,
                &rows,
                &cols,
                num_workers,
                OwnedDense::F64,
            ),
        })?;
        out.into_numpy(py)
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn csr_select_numpy<'py>(
    py: Python<'py>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    n_rows: usize,
    n_cols: usize,
    row_kind: &str,
    row_payload: &Bound<'_, PyAny>,
    col_kind: &str,
    col_payload: &Bound<'_, PyAny>,
    csr_output: &str,
    num_workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = AxisSpec::parse(row_kind, row_payload)?.normalize(n_rows, "row")?;
    let cols = AxisSpec::parse(col_kind, col_payload)?.normalize(n_cols, "col")?;
    let dense = match csr_output {
        "sparse" | "csr" => false,
        "dense" => true,
        other => {
            return Err(invalid_argument(format!(
                "csr_output must be 'sparse' or 'dense', got {other:?}"
            )));
        }
    };

    // `indptr` is small relative to the payload and controls every unchecked
    // row span. Owning it keeps those spans immutable while the GIL is released.
    let indptr = copy_u64_1d(indptr, "indptr")?;
    let gathered = dispatch_csr_indices(indices, |indices| {
        dispatch_csr_values(data, |csr| {
            py.allow_threads(move || {
                dispatch_csr_gather(
                    &indptr,
                    indices,
                    csr,
                    n_rows,
                    n_cols,
                    &rows,
                    &cols,
                    dense,
                    num_workers,
                )
            })
        })
    })?;

    match gathered {
        CsrGather::Sparse {
            indptr,
            indices,
            data,
            shape,
        } => pack_csr(py, indptr, indices, data, shape),
        CsrGather::Dense(dense) => {
            let array = dense.into_numpy(py)?;
            let kind = pyo3::types::PyString::new(py, "dense");
            PyTuple::new(py, [kind.into_any(), array]).map(|tuple| tuple.into_any())
        }
    }
}

pub(crate) fn csr_to_dense_numpy<'py>(
    py: Python<'py>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    n_rows: usize,
    n_cols: usize,
    num_workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = Axis::Range {
        start: 0,
        end: n_rows,
    };
    let cols = Axis::Range {
        start: 0,
        end: n_cols,
    };
    let indptr = copy_u64_1d(indptr, "indptr")?;
    let gathered = dispatch_csr_indices(indices, |indices| {
        dispatch_csr_values(data, |csr| {
            py.allow_threads(move || {
                dispatch_csr_gather(
                    &indptr,
                    indices,
                    csr,
                    n_rows,
                    n_cols,
                    &rows,
                    &cols,
                    true,
                    num_workers,
                )
            })
        })
    })?;
    match gathered {
        CsrGather::Dense(dense) => dense.into_numpy(py),
        CsrGather::Sparse { .. } => {
            Err(invalid_argument("internal: expected dense CSR conversion"))
        }
    }
}

fn dispatch_csr_values<R>(
    data: &Bound<'_, PyAny>,
    write: impl FnOnce(CsrData<'_>) -> PyResult<R>,
) -> PyResult<R> {
    let mut result = None;
    dispatch_csr_data(data, |values| {
        result = Some(write(values));
        Ok(())
    })?;
    result.ok_or_else(|| invalid_argument("failed to read CSR data"))?
}

#[derive(Clone, Copy)]
enum IndexRef<'a> {
    U16(&'a [u16]),
    U32(&'a [u32]),
}

fn dispatch_csr_indices<R>(
    array: &Bound<'_, PyAny>,
    write: impl FnOnce(IndexRef<'_>) -> PyResult<R>,
) -> PyResult<R> {
    if let Ok(values) = array.extract::<PyReadonlyArray1<'_, u16>>() {
        let slice = values
            .as_slice()
            .map_err(|_| invalid_argument("indices must be C-contiguous"))?;
        return write(IndexRef::U16(slice));
    }
    if let Ok(values) = array.extract::<PyReadonlyArray1<'_, u32>>() {
        let slice = values
            .as_slice()
            .map_err(|_| invalid_argument("indices must be C-contiguous"))?;
        return write(IndexRef::U32(slice));
    }
    Err(invalid_argument(
        "CSR indices must be a C-contiguous 1-D uint16 or uint32 array",
    ))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_csr_gather(
    indptr: &[u64],
    indices: IndexRef<'_>,
    csr: CsrData<'_>,
    n_rows: usize,
    n_cols: usize,
    rows: &Axis,
    cols: &Axis,
    dense: bool,
    num_workers: usize,
) -> PyResult<CsrGather> {
    macro_rules! gather_values {
        ($idx:expr, $values:expr) => {
            gather_csr(
                indptr,
                $idx,
                $values,
                n_rows,
                n_cols,
                rows,
                cols,
                dense,
                num_workers,
            )
        };
    }

    match (indices, csr) {
        (IndexRef::U16(idx), CsrData::U16(values)) => gather_values!(idx, values),
        (IndexRef::U16(idx), CsrData::U32(values)) => gather_values!(idx, values),
        (IndexRef::U16(idx), CsrData::U64(values)) => gather_values!(idx, values),
        (IndexRef::U16(idx), CsrData::I16(values)) => gather_values!(idx, values),
        (IndexRef::U16(idx), CsrData::I32(values)) => gather_values!(idx, values),
        (IndexRef::U16(idx), CsrData::I64(values)) => gather_values!(idx, values),
        (IndexRef::U16(idx), CsrData::F32(values)) => gather_values!(idx, values),
        (IndexRef::U16(idx), CsrData::F64(values)) => gather_values!(idx, values),
        (IndexRef::U32(idx), CsrData::U16(values)) => gather_values!(idx, values),
        (IndexRef::U32(idx), CsrData::U32(values)) => gather_values!(idx, values),
        (IndexRef::U32(idx), CsrData::U64(values)) => gather_values!(idx, values),
        (IndexRef::U32(idx), CsrData::I16(values)) => gather_values!(idx, values),
        (IndexRef::U32(idx), CsrData::I32(values)) => gather_values!(idx, values),
        (IndexRef::U32(idx), CsrData::I64(values)) => gather_values!(idx, values),
        (IndexRef::U32(idx), CsrData::F32(values)) => gather_values!(idx, values),
        (IndexRef::U32(idx), CsrData::F64(values)) => gather_values!(idx, values),
    }
}

enum OwnedDense {
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

struct ShapedDense {
    values: OwnedDense,
    rows: usize,
    cols: usize,
}

impl ShapedDense {
    fn into_numpy(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        fn reshape<'py, T: numpy::Element + Copy>(
            py: Python<'py>,
            values: Vec<T>,
            rows: usize,
            cols: usize,
        ) -> PyResult<Bound<'py, PyAny>> {
            let expected = checked_mul(rows, cols, "dense reshape")?;
            if values.len() != expected {
                return Err(invalid_argument(format!(
                    "value count {} does not match shape [{rows}, {cols}]",
                    values.len()
                )));
            }
            PyArray1::from_vec(py, values).call_method1("reshape", (rows, cols))
        }

        match self.values {
            OwnedDense::U16(values) => reshape(py, values, self.rows, self.cols),
            OwnedDense::U32(values) => reshape(py, values, self.rows, self.cols),
            OwnedDense::U64(values) => reshape(py, values, self.rows, self.cols),
            OwnedDense::I16(values) => reshape(py, values, self.rows, self.cols),
            OwnedDense::I32(values) => reshape(py, values, self.rows, self.cols),
            OwnedDense::I64(values) => reshape(py, values, self.rows, self.cols),
            OwnedDense::F32(values) => reshape(py, values, self.rows, self.cols),
            OwnedDense::F64(values) => reshape(py, values, self.rows, self.cols),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn wrap_dense<T>(
    src: &[T],
    n_rows: usize,
    n_cols: usize,
    rows: &Axis,
    cols: &Axis,
    num_workers: usize,
    wrap: impl FnOnce(Vec<T>) -> OwnedDense,
) -> PyResult<ShapedDense>
where
    T: Copy + Send + Sync,
{
    let (values, out_rows, out_cols) = gather_dense(src, n_rows, n_cols, rows, cols, num_workers)?;
    Ok(ShapedDense {
        values: wrap(values),
        rows: out_rows,
        cols: out_cols,
    })
}

fn gather_dense<T>(
    src: &[T],
    n_rows: usize,
    n_cols: usize,
    rows: &Axis,
    cols: &Axis,
    num_workers: usize,
) -> PyResult<(Vec<T>, usize, usize)>
where
    T: Copy + Send + Sync,
{
    let expected = checked_mul(n_rows, n_cols, "dense input")?;
    if src.len() != expected {
        return Err(invalid_argument("dense buffer length does not match shape"));
    }
    let out_rows = rows.len();
    let out_cols = cols.len();
    let out_len = checked_mul(out_rows, out_cols, "dense output")?;
    if out_len == 0 {
        return Ok((Vec::new(), out_rows, out_cols));
    }

    if let (Some((row_start, row_end)), Some((0, col_end))) = (rows.as_range(), cols.as_range()) {
        if col_end == n_cols {
            let start = row_start * n_cols;
            let end = row_end * n_cols;
            return Ok((copy_slice(&src[start..end])?, out_rows, out_cols));
        }
    }

    match rows {
        Axis::Range { start, .. } => {
            gather_dense_rows(src, n_cols, out_rows, cols, num_workers, |local_row| {
                start + local_row
            })
            .map(|values| (values, out_rows, out_cols))
        }
        Axis::Positions(positions) => {
            gather_dense_rows(src, n_cols, out_rows, cols, num_workers, |local_row| {
                debug_assert!(local_row < positions.len());
                // SAFETY: the row-block iterator only supplies local rows in
                // `0..out_rows`, which equals `positions.len()` in this arm.
                unsafe { *positions.get_unchecked(local_row) }
            })
            .map(|values| (values, out_rows, out_cols))
        }
    }
}

fn gather_dense_rows<T, R>(
    src: &[T],
    n_cols: usize,
    out_rows: usize,
    cols: &Axis,
    num_workers: usize,
    source_row: R,
) -> PyResult<Vec<T>>
where
    T: Copy + Send + Sync,
    R: Fn(usize) -> usize + Sync,
{
    let out_cols = cols.len();
    let out_len = checked_mul(out_rows, out_cols, "dense output")?;
    let mut output = Vec::new();
    try_reserve_exact(&mut output, out_len)?;
    let workers = effective_workers(num_workers, out_rows, out_len);
    {
        let spare = &mut output.spare_capacity_mut()[..out_len];
        match cols {
            Axis::Range { start, .. } => {
                for_each_row_block(out_rows, out_cols, workers, spare, |job_start, block| {
                    for (offset, dst_row) in block.chunks_exact_mut(out_cols).enumerate() {
                        let src_start = source_row(job_start + offset) * n_cols + start;
                        debug_assert!(src_start + out_cols <= src.len());
                        // SAFETY: normalized row/column axes and the validated
                        // source shape prove `src_start..src_start+out_cols` is
                        // initialized. `dst_row` is an exclusive exact-width
                        // row in the uninitialized output allocation.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                src.as_ptr().add(src_start),
                                dst_row.as_mut_ptr().cast::<T>(),
                                out_cols,
                            );
                        }
                    }
                })?
            }
            Axis::Positions(positions) => {
                for_each_row_block(out_rows, out_cols, workers, spare, |job_start, block| {
                    for (offset, dst_row) in block.chunks_exact_mut(out_cols).enumerate() {
                        let src_base = source_row(job_start + offset) * n_cols;
                        for (slot, &column) in dst_row.iter_mut().zip(positions) {
                            debug_assert!(src_base + column < src.len());
                            // SAFETY: axis normalization proves every `column`
                            // is below `n_cols`; the source row is validated and
                            // each zipped destination slot is distinct.
                            slot.write(unsafe { *src.get_unchecked(src_base + column) });
                        }
                    }
                })?
            }
        }
    }
    // SAFETY: both specialized kernels write every element of every output
    // row exactly once, and all scoped workers have joined successfully.
    unsafe { output.set_len(out_len) };
    Ok(output)
}

enum CsrGather {
    Sparse {
        indptr: Vec<u64>,
        indices: Vec<u64>,
        data: OwnedDense,
        shape: [usize; 2],
    },
    Dense(ShapedDense),
}

trait CsrIndex: Copy + Send + Sync {
    fn as_usize(self) -> usize;
}

/// Numeric CSR payloads whose all-zero byte pattern is their additive zero.
///
/// # Safety
///
/// Implementors must have no drop glue and every all-zero byte sequence of the
/// type's width must be a valid value equal to `Default::default()`.
unsafe trait ZeroableValue: Copy + Default + Send + Sync {}

// SAFETY: integer zero is all-zero bits, and `u16` is `Copy` without drop glue.
unsafe impl ZeroableValue for u16 {}
// SAFETY: integer zero is all-zero bits, and `u32` is `Copy` without drop glue.
unsafe impl ZeroableValue for u32 {}
// SAFETY: integer zero is all-zero bits, and `u64` is `Copy` without drop glue.
unsafe impl ZeroableValue for u64 {}
// SAFETY: integer zero is all-zero bits, and `i16` is `Copy` without drop glue.
unsafe impl ZeroableValue for i16 {}
// SAFETY: integer zero is all-zero bits, and `i32` is `Copy` without drop glue.
unsafe impl ZeroableValue for i32 {}
// SAFETY: integer zero is all-zero bits, and `i64` is `Copy` without drop glue.
unsafe impl ZeroableValue for i64 {}
// SAFETY: positive zero is all-zero bits, and `f32` is `Copy` without drop glue.
unsafe impl ZeroableValue for f32 {}
// SAFETY: positive zero is all-zero bits, and `f64` is `Copy` without drop glue.
unsafe impl ZeroableValue for f64 {}

impl CsrIndex for u16 {
    #[inline(always)]
    fn as_usize(self) -> usize {
        usize::from(self)
    }
}

impl CsrIndex for u32 {
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_csr<I, T>(
    indptr: &[u64],
    indices: &[I],
    data: &[T],
    n_rows: usize,
    n_cols: usize,
    rows: &Axis,
    cols: &Axis,
    dense: bool,
    num_workers: usize,
) -> PyResult<CsrGather>
where
    I: CsrIndex,
    T: ZeroableValue,
    OwnedDense: From<Vec<T>>,
{
    validate_csr_layout(indptr, indices.len(), data.len(), n_rows)?;
    let out_rows = rows.len();
    let out_cols = cols.len();

    match rows {
        Axis::Range { start, .. } => gather_csr_rows(
            indptr,
            indices,
            data,
            n_cols,
            out_rows,
            out_cols,
            cols,
            dense,
            num_workers,
            |local_row| start + local_row,
        ),
        Axis::Positions(positions) => gather_csr_rows(
            indptr,
            indices,
            data,
            n_cols,
            out_rows,
            out_cols,
            cols,
            dense,
            num_workers,
            |local_row| {
                debug_assert!(local_row < positions.len());
                // SAFETY: callers enumerate exactly `out_rows`, which equals
                // the normalized positions length in this arm.
                unsafe { *positions.get_unchecked(local_row) }
            },
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_csr_rows<I, T, R>(
    indptr: &[u64],
    indices: &[I],
    data: &[T],
    n_cols: usize,
    out_rows: usize,
    out_cols: usize,
    cols: &Axis,
    dense: bool,
    num_workers: usize,
    source_row: R,
) -> PyResult<CsrGather>
where
    I: CsrIndex,
    T: ZeroableValue,
    R: Fn(usize) -> usize + Sync,
    OwnedDense: From<Vec<T>>,
{
    let selected_nnz = selected_nnz(indptr, out_rows, &source_row);
    let columns = ColumnPlan::new(n_cols, cols, selected_nnz)?;
    if dense {
        let output = gather_csr_dense(
            indptr,
            indices,
            data,
            n_cols,
            out_rows,
            out_cols,
            &columns,
            num_workers,
            selected_nnz,
            &source_row,
        )?;
        return Ok(CsrGather::Dense(ShapedDense {
            values: OwnedDense::from(output),
            rows: out_rows,
            cols: out_cols,
        }));
    }

    let (out_indptr, row_counts) = count_sparse_rows(
        indptr,
        indices,
        n_cols,
        out_rows,
        &columns,
        num_workers,
        selected_nnz,
        &source_row,
    )?;
    let total_nnz = usize_from_u64(*out_indptr.last().unwrap_or(&0), "selected nnz")?;
    let mut out_indices = Vec::new();
    let mut out_data = Vec::new();
    try_reserve_exact(&mut out_indices, total_nnz)?;
    try_reserve_exact(&mut out_data, total_nnz)?;

    if total_nnz != 0 {
        let workers = effective_workers(
            num_workers,
            out_rows,
            selected_nnz.saturating_add(total_nnz),
        );
        let indices_spare = &mut out_indices.spare_capacity_mut()[..total_nnz];
        let data_spare = &mut out_data.spare_capacity_mut()[..total_nnz];
        fill_sparse_rows(
            indptr,
            indices,
            data,
            n_cols,
            out_rows,
            &columns,
            &out_indptr,
            &row_counts,
            workers,
            indices_spare,
            data_spare,
            &source_row,
        )?;
        // SAFETY: `fill_sparse_rows` returns success only after every row has
        // written exactly the count established by pass 1, into disjoint spans.
        unsafe {
            out_indices.set_len(total_nnz);
            out_data.set_len(total_nnz);
        }
    }

    Ok(CsrGather::Sparse {
        indptr: out_indptr,
        indices: out_indices,
        data: OwnedDense::from(out_data),
        shape: [out_rows, out_cols],
    })
}

#[allow(clippy::too_many_arguments)]
fn gather_csr_dense<I, T, R>(
    indptr: &[u64],
    indices: &[I],
    data: &[T],
    n_cols: usize,
    out_rows: usize,
    out_cols: usize,
    columns: &ColumnPlan,
    num_workers: usize,
    selected_nnz: usize,
    source_row: &R,
) -> PyResult<Vec<T>>
where
    I: CsrIndex,
    T: ZeroableValue,
    R: Fn(usize) -> usize + Sync,
{
    let out_len = checked_mul(out_rows, out_cols, "dense CSR output")?;
    if out_rows == 0 {
        return Ok(Vec::new());
    }
    if out_cols == 0 {
        validate_selected_indices(
            indptr,
            indices,
            n_cols,
            out_rows,
            num_workers,
            selected_nnz,
            source_row,
        )?;
        return Ok(Vec::new());
    }

    let mut output = Vec::new();
    try_reserve_exact(&mut output, out_len)?;
    let workers = effective_workers(num_workers, out_rows, selected_nnz.saturating_add(out_len));
    let spare = &mut output.spare_capacity_mut()[..out_len];
    match columns {
        ColumnPlan::All => scatter_dense_rows(
            indptr,
            indices,
            data,
            n_cols,
            out_rows,
            out_cols,
            workers,
            spare,
            source_row,
            |column, value, row| {
                debug_assert!(column < row.len());
                // SAFETY: the common scanner validates `column < n_cols`, and
                // an All plan has `row.len() == n_cols`.
                unsafe { *row.get_unchecked_mut(column) = value };
            },
        )?,
        ColumnPlan::Range { start, end } => scatter_dense_rows(
            indptr,
            indices,
            data,
            n_cols,
            out_rows,
            out_cols,
            workers,
            spare,
            source_row,
            |column, value, row| {
                if column >= *start && column < *end {
                    let destination = column - start;
                    debug_assert!(destination < row.len());
                    // SAFETY: the range predicate proves the remapped column
                    // lies inside the exact-width destination row.
                    unsafe { *row.get_unchecked_mut(destination) = value };
                }
            },
        )?,
        ColumnPlan::Direct(destinations) => scatter_dense_rows(
            indptr,
            indices,
            data,
            n_cols,
            out_rows,
            out_cols,
            workers,
            spare,
            source_row,
            |column, value, row| {
                let destination = {
                    // SAFETY: the common scanner accepted `column < n_cols`, and
                    // every direct map contains exactly one slot per source column.
                    unsafe { direct_destination_unchecked(destinations, column) }
                };
                if let Some(destination) = destination {
                    debug_assert!(destination < row.len());
                    // SAFETY: direct-map destinations are normalized output
                    // positions, and this row has exactly `out_cols` slots.
                    unsafe { *row.get_unchecked_mut(destination) = value };
                }
            },
        )?,
        ColumnPlan::Lookup(lookup) => scatter_dense_rows(
            indptr,
            indices,
            data,
            n_cols,
            out_rows,
            out_cols,
            workers,
            spare,
            source_row,
            |column, value, row| {
                for &(_, destination) in lookup.destinations(column) {
                    debug_assert!(destination < row.len());
                    // SAFETY: lookup destinations are original positions in
                    // `0..out_cols`, and this row has exactly `out_cols` slots.
                    unsafe { *row.get_unchecked_mut(destination) = value };
                }
            },
        )?,
    }
    // SAFETY: `scatter_dense_rows` first zero-initializes every disjoint row
    // block, then applies sparse writes, and returns only after all workers join.
    unsafe { output.set_len(out_len) };
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn scatter_dense_rows<I, T, R, W>(
    indptr: &[u64],
    indices: &[I],
    data: &[T],
    n_cols: usize,
    out_rows: usize,
    out_cols: usize,
    workers: usize,
    output: &mut [MaybeUninit<T>],
    source_row: &R,
    write: W,
) -> PyResult<()>
where
    I: CsrIndex,
    T: ZeroableValue,
    R: Fn(usize) -> usize + Sync,
    W: Fn(usize, T, &mut [T]) + Sync,
{
    let invalid = AtomicUsize::new(usize::MAX);
    for_each_row_block(out_rows, out_cols, workers, output, |job_start, block| {
        // SAFETY: `ZeroableValue` guarantees an all-zero representation is a
        // valid value. This worker owns the entire disjoint output block.
        unsafe { std::ptr::write_bytes(block.as_mut_ptr().cast::<T>(), 0, block.len()) };
        // SAFETY: the preceding write initialized every element in this exact
        // block, whose allocation and alignment came from `Vec<T>`.
        let block =
            unsafe { std::slice::from_raw_parts_mut(block.as_mut_ptr().cast::<T>(), block.len()) };
        for (offset, dst_row) in block.chunks_exact_mut(out_cols).enumerate() {
            let src_row = source_row(job_start + offset);
            // SAFETY: CSR layout validation converts every monotone indptr
            // entry to usize and proves it does not exceed both payloads.
            let (start, end) = unsafe {
                (
                    *indptr.get_unchecked(src_row) as usize,
                    *indptr.get_unchecked(src_row + 1) as usize,
                )
            };
            for position in start..end {
                // SAFETY: `position` lies in a structurally validated CSR
                // row, so both payload slices contain this element.
                let (column, value) = unsafe {
                    (
                        indices.get_unchecked(position).as_usize(),
                        *data.get_unchecked(position),
                    )
                };
                if column >= n_cols {
                    invalid.fetch_min(position, Ordering::Relaxed);
                    continue;
                }
                write(column, value, dst_row);
            }
        }
    })?;
    check_invalid_index(&invalid, indices, n_cols)
}

#[allow(clippy::too_many_arguments)]
fn count_sparse_rows<I, R>(
    indptr: &[u64],
    indices: &[I],
    n_cols: usize,
    out_rows: usize,
    columns: &ColumnPlan,
    num_workers: usize,
    selected_nnz: usize,
    source_row: &R,
) -> PyResult<(Vec<u64>, Vec<usize>)>
where
    I: CsrIndex,
    R: Fn(usize) -> usize + Sync,
{
    let mut row_counts = zeroed(out_rows)?;
    let workers = effective_workers(num_workers, out_rows, selected_nnz);
    let invalid = AtomicUsize::new(usize::MAX);

    match columns {
        ColumnPlan::All => count_all_rows(indptr, out_rows, workers, &mut row_counts, source_row)?,
        ColumnPlan::Range { start, end } => count_rows_with(
            indptr,
            indices,
            n_cols,
            out_rows,
            workers,
            &mut row_counts,
            source_row,
            &invalid,
            |column| usize::from(column >= *start && column < *end),
        )?,
        ColumnPlan::Direct(destinations) => count_rows_with(
            indptr,
            indices,
            n_cols,
            out_rows,
            workers,
            &mut row_counts,
            source_row,
            &invalid,
            |column| {
                // SAFETY: `count_rows_with` checks the source column against
                // `n_cols`, which is also the direct map length.
                usize::from(unsafe { direct_destination_unchecked(destinations, column).is_some() })
            },
        )?,
        ColumnPlan::Lookup(lookup) => count_rows_with(
            indptr,
            indices,
            n_cols,
            out_rows,
            workers,
            &mut row_counts,
            source_row,
            &invalid,
            |column| lookup.destinations(column).len(),
        )?,
    }
    check_invalid_index(&invalid, indices, n_cols)?;

    let indptr_len = out_rows
        .checked_add(1)
        .ok_or_else(|| invalid_argument("CSR output indptr length overflow"))?;
    let mut out_indptr = Vec::new();
    try_reserve_exact(&mut out_indptr, indptr_len)?;
    out_indptr.push(0u64);
    let mut total = 0usize;
    for &count in &row_counts {
        total = total
            .checked_add(count)
            .ok_or_else(|| invalid_argument("CSR selected nnz overflow"))?;
        out_indptr.push(
            u64::try_from(total).map_err(|_| invalid_argument("CSR selected nnz exceeds u64"))?,
        );
    }
    Ok((out_indptr, row_counts))
}

fn count_all_rows<R>(
    indptr: &[u64],
    out_rows: usize,
    workers: usize,
    row_counts: &mut [usize],
    source_row: &R,
) -> PyResult<()>
where
    R: Fn(usize) -> usize + Sync,
{
    for_each_row_block(out_rows, 1, workers, row_counts, |job_start, block| {
        for (offset, slot) in block.iter_mut().enumerate() {
            let src_row = source_row(job_start + offset);
            // SAFETY: the normalized row map and validated indptr layout prove
            // both entries exist, fit usize, and form a non-decreasing span.
            let (start, end) = unsafe {
                (
                    *indptr.get_unchecked(src_row) as usize,
                    *indptr.get_unchecked(src_row + 1) as usize,
                )
            };
            *slot = end - start;
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn count_rows_with<I, R, C>(
    indptr: &[u64],
    indices: &[I],
    n_cols: usize,
    out_rows: usize,
    workers: usize,
    row_counts: &mut [usize],
    source_row: &R,
    invalid: &AtomicUsize,
    destination_count: C,
) -> PyResult<()>
where
    I: CsrIndex,
    R: Fn(usize) -> usize + Sync,
    C: Fn(usize) -> usize + Sync,
{
    for_each_row_block(out_rows, 1, workers, row_counts, |job_start, block| {
        for (offset, slot) in block.iter_mut().enumerate() {
            let src_row = source_row(job_start + offset);
            // SAFETY: the normalized row map and validated indptr layout
            // prove both entries exist and delimit a payload subrange.
            let (start, end) = unsafe {
                (
                    *indptr.get_unchecked(src_row) as usize,
                    *indptr.get_unchecked(src_row + 1) as usize,
                )
            };
            let mut count = 0usize;
            for position in start..end {
                // SAFETY: every position in this row is below indices.len().
                let column = unsafe { indices.get_unchecked(position).as_usize() };
                if column >= n_cols {
                    invalid.fetch_min(position, Ordering::Relaxed);
                    continue;
                }
                count = count.saturating_add(destination_count(column));
            }
            *slot = count;
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn fill_sparse_rows<I, T, R>(
    indptr: &[u64],
    indices: &[I],
    data: &[T],
    n_cols: usize,
    out_rows: usize,
    columns: &ColumnPlan,
    out_indptr: &[u64],
    row_counts: &[usize],
    workers: usize,
    out_indices: &mut [MaybeUninit<u64>],
    out_data: &mut [MaybeUninit<T>],
    source_row: &R,
) -> PyResult<()>
where
    I: CsrIndex,
    T: Copy + Send + Sync,
    R: Fn(usize) -> usize + Sync,
{
    let invalid = AtomicUsize::new(usize::MAX);
    let mismatch = AtomicBool::new(false);
    match columns {
        ColumnPlan::All => for_each_sparse_row_block(
            out_rows,
            workers,
            out_indptr,
            out_indices,
            out_data,
            |job_start, job_end, block_base, index_block, data_block| {
                fill_sparse_block(
                    indptr,
                    indices,
                    data,
                    n_cols,
                    job_start,
                    job_end,
                    block_base,
                    index_block,
                    data_block,
                    out_indptr,
                    row_counts,
                    source_row,
                    &invalid,
                    &mismatch,
                    |column, value, cursor, dst_indices, dst_data| {
                        // SAFETY: pass 1 counted one output for every validated
                        // source entry in an All plan; `cursor` advances once.
                        unsafe {
                            write_sparse_entry(dst_indices, dst_data, cursor, column as u64, value)
                        };
                    },
                );
            },
        )?,
        ColumnPlan::Range { start, end } => for_each_sparse_row_block(
            out_rows,
            workers,
            out_indptr,
            out_indices,
            out_data,
            |job_start, job_end, block_base, index_block, data_block| {
                fill_sparse_block(
                    indptr,
                    indices,
                    data,
                    n_cols,
                    job_start,
                    job_end,
                    block_base,
                    index_block,
                    data_block,
                    out_indptr,
                    row_counts,
                    source_row,
                    &invalid,
                    &mismatch,
                    |column, value, cursor, dst_indices, dst_data| {
                        if column >= *start && column < *end {
                            // SAFETY: pass 1 used the identical range predicate,
                            // so this row owns one remaining output slot.
                            unsafe {
                                write_sparse_entry(
                                    dst_indices,
                                    dst_data,
                                    cursor,
                                    (column - start) as u64,
                                    value,
                                )
                            };
                        }
                    },
                );
            },
        )?,
        ColumnPlan::Direct(destinations) => for_each_sparse_row_block(
            out_rows,
            workers,
            out_indptr,
            out_indices,
            out_data,
            |job_start, job_end, block_base, index_block, data_block| {
                fill_sparse_block(
                    indptr,
                    indices,
                    data,
                    n_cols,
                    job_start,
                    job_end,
                    block_base,
                    index_block,
                    data_block,
                    out_indptr,
                    row_counts,
                    source_row,
                    &invalid,
                    &mismatch,
                    |column, value, cursor, dst_indices, dst_data| {
                        let destination = {
                            // SAFETY: `fill_sparse_block` checks `column < n_cols`,
                            // and the direct map has exactly `n_cols` entries.
                            unsafe { direct_destination_unchecked(destinations, column) }
                        };
                        if let Some(destination) = destination {
                            // SAFETY: pass 1 used the identical direct map and
                            // counted exactly one slot for this destination.
                            unsafe {
                                write_sparse_entry(
                                    dst_indices,
                                    dst_data,
                                    cursor,
                                    destination as u64,
                                    value,
                                )
                            };
                        }
                    },
                );
            },
        )?,
        ColumnPlan::Lookup(lookup) => for_each_sparse_row_block(
            out_rows,
            workers,
            out_indptr,
            out_indices,
            out_data,
            |job_start, job_end, block_base, index_block, data_block| {
                fill_sparse_block(
                    indptr,
                    indices,
                    data,
                    n_cols,
                    job_start,
                    job_end,
                    block_base,
                    index_block,
                    data_block,
                    out_indptr,
                    row_counts,
                    source_row,
                    &invalid,
                    &mismatch,
                    |column, value, cursor, dst_indices, dst_data| {
                        for &(_, destination) in lookup.destinations(column) {
                            // SAFETY: pass 1 counted this exact immutable lookup
                            // span, and every destination is below out_cols.
                            unsafe {
                                write_sparse_entry(
                                    dst_indices,
                                    dst_data,
                                    cursor,
                                    destination as u64,
                                    value,
                                )
                            };
                        }
                    },
                );
            },
        )?,
    }
    check_invalid_index(&invalid, indices, n_cols)?;
    if mismatch.load(Ordering::Relaxed) {
        return Err(invalid_argument(
            "internal: CSR sparse count and fill passes disagree",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fill_sparse_block<I, T, R, E>(
    indptr: &[u64],
    indices: &[I],
    data: &[T],
    n_cols: usize,
    job_start: usize,
    job_end: usize,
    block_base: usize,
    index_block: &mut [MaybeUninit<u64>],
    data_block: &mut [MaybeUninit<T>],
    out_indptr: &[u64],
    row_counts: &[usize],
    source_row: &R,
    invalid: &AtomicUsize,
    mismatch: &AtomicBool,
    emit: E,
) where
    I: CsrIndex,
    T: Copy,
    R: Fn(usize) -> usize,
    E: Fn(usize, T, &mut usize, &mut [MaybeUninit<u64>], &mut [MaybeUninit<T>]),
{
    for local_row in job_start..job_end {
        let row_start = out_indptr[local_row] as usize - block_base;
        let row_end = out_indptr[local_row + 1] as usize - block_base;
        let dst_indices = &mut index_block[row_start..row_end];
        let dst_data = &mut data_block[row_start..row_end];
        let src_row = source_row(local_row);
        // SAFETY: normalized rows and validated indptr prove both offsets exist
        // and delimit initialized positions in `indices` and `data`.
        let (start, end) = unsafe {
            (
                *indptr.get_unchecked(src_row) as usize,
                *indptr.get_unchecked(src_row + 1) as usize,
            )
        };
        let mut cursor = 0usize;
        for position in start..end {
            // SAFETY: this structurally validated row lies within both payloads.
            let (column, value) = unsafe {
                (
                    indices.get_unchecked(position).as_usize(),
                    *data.get_unchecked(position),
                )
            };
            if column >= n_cols {
                invalid.fetch_min(position, Ordering::Relaxed);
                continue;
            }
            emit(column, value, &mut cursor, dst_indices, dst_data);
        }
        if cursor != row_counts[local_row] {
            mismatch.store(true, Ordering::Relaxed);
        }
    }
}

/// Write one entry into a row span sized by the preceding count pass.
///
/// # Safety
///
/// `cursor` must be below both destination lengths. The immutable CSR inputs
/// and column plan must be the same ones used to calculate that row's count.
#[inline(always)]
unsafe fn write_sparse_entry<T: Copy>(
    indices: &mut [MaybeUninit<u64>],
    data: &mut [MaybeUninit<T>],
    cursor: &mut usize,
    column: u64,
    value: T,
) {
    debug_assert!(*cursor < indices.len());
    debug_assert_eq!(indices.len(), data.len());
    // SAFETY: the caller guarantees `cursor` is inside both exact row spans.
    unsafe {
        indices.get_unchecked_mut(*cursor).write(column);
        data.get_unchecked_mut(*cursor).write(value);
    }
    *cursor += 1;
}

#[allow(clippy::too_many_arguments)]
fn validate_selected_indices<I, R>(
    indptr: &[u64],
    indices: &[I],
    n_cols: usize,
    out_rows: usize,
    num_workers: usize,
    selected_nnz: usize,
    source_row: &R,
) -> PyResult<()>
where
    I: CsrIndex,
    R: Fn(usize) -> usize + Sync,
{
    let mut row_counts = zeroed(out_rows)?;
    let invalid = AtomicUsize::new(usize::MAX);
    count_rows_with(
        indptr,
        indices,
        n_cols,
        out_rows,
        effective_workers(num_workers, out_rows, selected_nnz),
        &mut row_counts,
        source_row,
        &invalid,
        |_| 0,
    )?;
    check_invalid_index(&invalid, indices, n_cols)
}

fn validate_csr_layout(
    indptr: &[u64],
    indices_len: usize,
    data_len: usize,
    n_rows: usize,
) -> PyResult<()> {
    let expected_indptr = n_rows
        .checked_add(1)
        .ok_or_else(|| invalid_argument("CSR indptr length overflow"))?;
    if indptr.len() != expected_indptr {
        return Err(invalid_argument(format!(
            "CSR indptr length {} does not match n_rows+1={expected_indptr}",
            indptr.len()
        )));
    }
    if indptr.first().copied() != Some(0) {
        return Err(invalid_argument("CSR indptr[0] must be 0"));
    }

    let mut previous = 0usize;
    for (row, &offset) in indptr.iter().enumerate().skip(1) {
        let offset = usize_from_u64(offset, "CSR indptr entry")?;
        if offset < previous {
            return Err(invalid_argument(format!(
                "CSR indptr must be non-decreasing at row {}",
                row - 1
            )));
        }
        previous = offset;
    }
    if indices_len != previous || data_len != previous {
        return Err(invalid_argument(format!(
            "CSR indices/data lengths ({indices_len}, {data_len}) do not match nnz={previous}"
        )));
    }
    Ok(())
}

fn selected_nnz<R>(indptr: &[u64], out_rows: usize, source_row: &R) -> usize
where
    R: Fn(usize) -> usize,
{
    let mut total = 0usize;
    for local_row in 0..out_rows {
        let row = source_row(local_row);
        // SAFETY: normalized source rows are below the validated CSR row count.
        let (start, end) = unsafe {
            (
                *indptr.get_unchecked(row) as usize,
                *indptr.get_unchecked(row + 1) as usize,
            )
        };
        total = total.saturating_add(end - start);
    }
    total
}

fn check_invalid_index<I: CsrIndex>(
    invalid: &AtomicUsize,
    indices: &[I],
    n_cols: usize,
) -> PyResult<()> {
    let position = invalid.load(Ordering::Relaxed);
    if position == usize::MAX {
        return Ok(());
    }
    debug_assert!(position < indices.len());
    // SAFETY: only scanners over validated CSR row spans publish positions.
    let column = unsafe { indices.get_unchecked(position).as_usize() };
    Err(invalid_argument(format!(
        "CSR column index[{position}]={column} outside 0..{n_cols}"
    )))
}

enum ColumnPlan {
    All,
    Range { start: usize, end: usize },
    Direct(Vec<u32>),
    Lookup(ColumnLookup),
}

struct ColumnLookup {
    pairs: Vec<(usize, usize)>,
    offsets: Option<Vec<usize>>,
}

impl ColumnPlan {
    fn new(n_cols: usize, cols: &Axis, selected_nnz: usize) -> PyResult<Self> {
        match cols {
            Axis::Range { start: 0, end } if *end == n_cols => Ok(Self::All),
            Axis::Range { start, end } => Ok(Self::Range {
                start: *start,
                end: *end,
            }),
            Axis::Positions(positions) => {
                if should_use_direct_lookup(n_cols, selected_nnz, positions.len()) {
                    let mut destinations = Vec::new();
                    try_reserve_exact(&mut destinations, n_cols)?;
                    destinations.resize(n_cols, MISSING_DESTINATION);
                    let mut unique = true;
                    for (destination, &source) in positions.iter().enumerate() {
                        debug_assert!(source < destinations.len());
                        // SAFETY: axis normalization proved every source is
                        // below `n_cols`, which is the exact map length.
                        let slot = unsafe { destinations.get_unchecked_mut(source) };
                        if *slot != MISSING_DESTINATION {
                            unique = false;
                            break;
                        }
                        debug_assert!(destination < MISSING_DESTINATION as usize);
                        *slot = destination as u32;
                    }
                    if unique {
                        return Ok(Self::Direct(destinations));
                    }
                }

                let mut pairs = Vec::new();
                try_reserve_exact(&mut pairs, positions.len())?;
                pairs.extend(
                    positions
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(destination, source)| (source, destination)),
                );
                pairs.sort_unstable();

                let dense_limit = positions.len().saturating_mul(8).max(1024);
                let offsets = if n_cols <= dense_limit {
                    n_cols.checked_add(1).and_then(|len| {
                        let mut offsets = Vec::new();
                        offsets.try_reserve_exact(len).ok()?;
                        offsets.resize(len, 0usize);
                        for &(source, _) in &pairs {
                            offsets[source + 1] += 1;
                        }
                        for index in 1..offsets.len() {
                            offsets[index] += offsets[index - 1];
                        }
                        Some(offsets)
                    })
                } else {
                    None
                };
                Ok(Self::Lookup(ColumnLookup { pairs, offsets }))
            }
        }
    }
}

fn should_use_direct_lookup(n_cols: usize, selected_nnz: usize, selected_cols: usize) -> bool {
    if selected_cols == 0 || selected_cols >= MISSING_DESTINATION as usize {
        return false;
    }
    let Some(bytes) = n_cols.checked_mul(std::mem::size_of::<u32>()) else {
        return false;
    };
    if bytes > MAX_DIRECT_LOOKUP_BYTES {
        return false;
    }
    let search_steps = usize::BITS as usize - selected_cols.leading_zeros() as usize;
    n_cols <= selected_nnz.saturating_mul(search_steps.max(1))
}

/// Read one direct source-to-destination mapping without a redundant bounds check.
///
/// # Safety
///
/// `source` must be below `destinations.len()`.
#[inline(always)]
unsafe fn direct_destination_unchecked(destinations: &[u32], source: usize) -> Option<usize> {
    debug_assert!(source < destinations.len());
    // SAFETY: the caller guarantees that `source` addresses this map.
    let destination = unsafe { *destinations.get_unchecked(source) };
    (destination != MISSING_DESTINATION).then_some(destination as usize)
}

impl ColumnLookup {
    #[inline]
    fn destinations(&self, source: usize) -> &[(usize, usize)] {
        if let Some(offsets) = &self.offsets {
            debug_assert!(source + 1 < offsets.len());
            // SAFETY: callers validate `source < n_cols`, while dense offsets
            // contain exactly `n_cols + 1` entries. Prefix sums are bounded by
            // `pairs.len()` and describe its source-sorted groups.
            unsafe {
                let start = *offsets.get_unchecked(source);
                let end = *offsets.get_unchecked(source + 1);
                return self.pairs.get_unchecked(start..end);
            }
        }
        let start = self
            .pairs
            .partition_point(|&(candidate, _)| candidate < source);
        let end =
            start + self.pairs[start..].partition_point(|&(candidate, _)| candidate == source);
        &self.pairs[start..end]
    }
}

#[allow(clippy::too_many_arguments)]
fn for_each_sparse_row_block<T, F>(
    n_rows: usize,
    workers: usize,
    indptr: &[u64],
    indices: &mut [MaybeUninit<u64>],
    data: &mut [MaybeUninit<T>],
    work: F,
) -> PyResult<()>
where
    T: Send,
    F: Fn(usize, usize, usize, &mut [MaybeUninit<u64>], &mut [MaybeUninit<T>]) + Sync,
{
    if n_rows == 0 {
        return Ok(());
    }
    debug_assert_eq!(indptr.len(), n_rows + 1);
    debug_assert_eq!(indices.len(), data.len());
    let workers = workers.max(1).min(n_rows);
    if workers == 1 {
        work(0, n_rows, 0, indices, data);
        return Ok(());
    }

    let rows_per_worker = n_rows.div_ceil(workers);
    std::thread::scope(|scope| -> PyResult<()> {
        let mut handles = Vec::new();
        try_reserve_exact(&mut handles, workers - 1)?;
        let mut indices_remaining = indices;
        let mut data_remaining = data;
        let mut row_start = 0usize;
        while row_start < n_rows {
            let row_end = (row_start + rows_per_worker).min(n_rows);
            let nnz_start = indptr[row_start] as usize;
            let nnz_end = indptr[row_end] as usize;
            let block_len = nnz_end - nnz_start;
            let index_tail = std::mem::take(&mut indices_remaining);
            let (index_block, index_tail) = index_tail.split_at_mut(block_len);
            indices_remaining = index_tail;
            let data_tail = std::mem::take(&mut data_remaining);
            let (data_block, data_tail) = data_tail.split_at_mut(block_len);
            data_remaining = data_tail;

            if row_end == n_rows {
                work(row_start, row_end, nnz_start, index_block, data_block);
            } else {
                let work = &work;
                handles.push(
                    std::thread::Builder::new()
                        .name("scdata-slice".into())
                        .spawn_scoped(scope, move || {
                            work(row_start, row_end, nnz_start, index_block, data_block)
                        })
                        .map_err(|error| {
                            invalid_argument(format!("failed to spawn slice worker: {error}"))
                        })?,
                );
            }
            row_start = row_end;
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| invalid_argument("slice worker panicked"))?;
        }
        Ok(())
    })
}

fn for_each_row_block<T, F>(
    n_rows: usize,
    row_width: usize,
    workers: usize,
    output: &mut [T],
    work: F,
) -> PyResult<()>
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let expected = checked_mul(n_rows, row_width, "parallel row output")?;
    if output.len() != expected {
        return Err(invalid_argument(
            "parallel output length does not match row shape",
        ));
    }
    if n_rows == 0 {
        return Ok(());
    }
    let workers = workers.max(1).min(n_rows);
    if workers == 1 {
        work(0, output);
        return Ok(());
    }

    let rows_per_worker = n_rows.div_ceil(workers);
    std::thread::scope(|scope| -> PyResult<()> {
        let mut handles = Vec::new();
        try_reserve_exact(&mut handles, workers - 1)?;
        let mut remaining = output;
        let mut row_start = 0usize;
        while row_start < n_rows {
            let row_end = (row_start + rows_per_worker).min(n_rows);
            let block_len = (row_end - row_start) * row_width;
            let tail = std::mem::take(&mut remaining);
            let (block, tail) = tail.split_at_mut(block_len);
            remaining = tail;
            if row_end == n_rows {
                work(row_start, block);
            } else {
                let work = &work;
                handles.push(
                    std::thread::Builder::new()
                        .name("scdata-slice".into())
                        .spawn_scoped(scope, move || work(row_start, block))
                        .map_err(|error| {
                            invalid_argument(format!("failed to spawn slice worker: {error}"))
                        })?,
                );
            }
            row_start = row_end;
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| invalid_argument("slice worker panicked"))?;
        }
        Ok(())
    })
}

fn effective_workers(requested: usize, rows: usize, work: usize) -> usize {
    if requested <= 1 || rows <= 1 || work < MIN_PARALLEL_WORK * 2 {
        return 1;
    }
    requested
        .min(rows)
        .min(work.div_ceil(MIN_PARALLEL_WORK))
        .max(1)
}

fn pack_csr<'py>(
    py: Python<'py>,
    indptr: Vec<u64>,
    indices: Vec<u64>,
    data: OwnedDense,
    shape: [usize; 2],
) -> PyResult<Bound<'py, PyAny>> {
    let indices = PyArray1::from_vec(py, indices).into_any();
    let data = match data {
        OwnedDense::U16(values) => PyArray1::from_vec(py, values).into_any(),
        OwnedDense::U32(values) => PyArray1::from_vec(py, values).into_any(),
        OwnedDense::U64(values) => PyArray1::from_vec(py, values).into_any(),
        OwnedDense::I16(values) => PyArray1::from_vec(py, values).into_any(),
        OwnedDense::I32(values) => PyArray1::from_vec(py, values).into_any(),
        OwnedDense::I64(values) => PyArray1::from_vec(py, values).into_any(),
        OwnedDense::F32(values) => PyArray1::from_vec(py, values).into_any(),
        OwnedDense::F64(values) => PyArray1::from_vec(py, values).into_any(),
    };
    let indptr = PyArray1::from_vec(py, indptr).into_any();
    let shape = PyTuple::new(py, [shape[0] as u64, shape[1] as u64])?;
    let kind = pyo3::types::PyString::new(py, "csr");
    PyTuple::new(
        py,
        [kind.into_any(), indices, data, indptr, shape.into_any()],
    )
    .map(|tuple| tuple.into_any())
}

fn is_contiguous(positions: &[usize]) -> bool {
    positions.windows(2).all(|pair| pair[1] == pair[0] + 1)
}

fn usize_from_u64(value: u64, context: &str) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| invalid_argument(format!("{context} exceeds usize")))
}

fn checked_mul(left: usize, right: usize, context: &str) -> PyResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_argument(format!("{context} size overflow")))
}

fn try_reserve_exact<T>(values: &mut Vec<T>, additional: usize) -> PyResult<()> {
    values
        .try_reserve_exact(additional)
        .map_err(sc_compress::Error::from)
        .map_err(from_compress)
}

fn copy_slice<T: Copy>(source: &[T]) -> PyResult<Vec<T>> {
    let mut output = Vec::new();
    try_reserve_exact(&mut output, source.len())?;
    output.extend_from_slice(source);
    Ok(output)
}

fn zeroed<T>(len: usize) -> PyResult<Vec<T>>
where
    T: Default + Clone,
{
    let mut output = Vec::new();
    try_reserve_exact(&mut output, len)?;
    output.resize(len, T::default());
    Ok(output)
}

macro_rules! owned_from {
    ($ty:ty, $variant:ident) => {
        impl From<Vec<$ty>> for OwnedDense {
            fn from(values: Vec<$ty>) -> Self {
                Self::$variant(values)
            }
        }
    };
}

owned_from!(u16, U16);
owned_from!(u32, U32);
owned_from!(u64, U64);
owned_from!(i16, I16);
owned_from!(i32, I32);
owned_from!(i64, I64);
owned_from!(f32, F32);
owned_from!(f64, F64);

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::{
        direct_destination_unchecked, gather_csr, scatter_dense_rows, Axis, ColumnPlan, CsrGather,
        OwnedDense,
    };

    #[test]
    fn direct_lookup_uses_compact_unique_destinations() {
        let columns = Axis::Positions(vec![4, 0, 2]);
        let plan = ColumnPlan::new(5, &columns, 5).unwrap();
        let ColumnPlan::Direct(destinations) = plan else {
            panic!("expected a direct lookup");
        };

        // SAFETY: all queried columns are within the five-entry direct map.
        unsafe {
            assert_eq!(direct_destination_unchecked(&destinations, 0), Some(1));
            assert_eq!(direct_destination_unchecked(&destinations, 1), None);
            assert_eq!(direct_destination_unchecked(&destinations, 4), Some(0));
        }
    }

    #[test]
    fn duplicate_columns_keep_every_output_destination() {
        let columns = Axis::Positions(vec![2, 2, 4]);
        let plan = ColumnPlan::new(5, &columns, 6).unwrap();
        let ColumnPlan::Lookup(lookup) = plan else {
            panic!("duplicate columns require grouped destinations");
        };

        assert_eq!(lookup.destinations(2), &[(2, 0), (2, 1)]);
        assert_eq!(lookup.destinations(4), &[(4, 2)]);
    }

    #[test]
    fn oversized_direct_map_falls_back_to_sorted_pairs() {
        let columns = Axis::Positions(vec![7]);
        let plan = ColumnPlan::new(10_000_000, &columns, 1).unwrap();
        assert!(matches!(plan, ColumnPlan::Lookup(_)));
    }

    #[test]
    fn direct_lookup_preserves_sparse_and_dense_results() {
        let indptr = [0, 3, 5];
        let indices = [0u16, 2, 4, 1, 3];
        let data = [10i32, 20, 30, 40, 50];
        let rows = Axis::Range { start: 0, end: 2 };
        let columns = Axis::Positions(vec![4, 0, 2]);

        let sparse = gather_csr(&indptr, &indices, &data, 2, 5, &rows, &columns, false, 1).unwrap();
        let CsrGather::Sparse {
            indptr,
            indices,
            data,
            shape,
        } = sparse
        else {
            panic!("expected sparse output");
        };
        assert_eq!(indptr, [0, 3, 3]);
        assert_eq!(indices, [1, 2, 0]);
        assert_eq!(shape, [2, 3]);
        let OwnedDense::I32(data) = data else {
            panic!("expected i32 data");
        };
        assert_eq!(data, [10, 20, 30]);

        let dense = gather_csr(
            &indptr_from_rows(),
            &indices_from_rows(),
            &data_from_rows(),
            2,
            5,
            &rows,
            &columns,
            true,
            1,
        )
        .unwrap();
        let CsrGather::Dense(dense) = dense else {
            panic!("expected dense output");
        };
        let OwnedDense::I32(values) = dense.values else {
            panic!("expected i32 values");
        };
        assert_eq!(values, [30, 10, 20, 0, 0, 0]);
    }

    #[test]
    fn parallel_scatter_zero_initializes_every_unwritten_slot() {
        let indptr = [0, 1, 2, 3];
        let indices = [0u16, 1, 2];
        let data = [7i32, 8, 9];
        let source_row = |row| row;
        let mut output = Vec::with_capacity(12);
        let spare: &mut [MaybeUninit<i32>] = &mut output.spare_capacity_mut()[..12];

        scatter_dense_rows(
            &indptr,
            &indices,
            &data,
            4,
            3,
            4,
            2,
            spare,
            &source_row,
            |column, value, row| row[column] = value,
        )
        .unwrap();
        // SAFETY: the scatter routine initializes all twelve slots before it
        // returns, including the nine structurally absent entries.
        unsafe { output.set_len(12) };

        assert_eq!(output, [7, 0, 0, 0, 0, 8, 0, 0, 0, 0, 9, 0]);
    }

    fn indptr_from_rows() -> [u64; 3] {
        [0, 3, 5]
    }

    fn indices_from_rows() -> [u16; 5] {
        [0, 2, 4, 1, 3]
    }

    fn data_from_rows() -> [i32; 5] {
        [10, 20, 30, 40, 50]
    }
}
