//! Axis selection parsing and store / in-memory select bindings.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use sc_compress::{
    AxisIndex, CsrArray, CsrOutput, DenseArray, Matrix, SelectedArray, Selection, DType,
};

use crate::convert::{
    csr_data_bytes_to_array, csr_index_bytes_to_array, dense_bytes_to_array, dispatch_csr_data,
    dispatch_dense, u64_slice_to_array, CsrData, DenseValues,
};
use crate::error::{invalid_argument, ResultExt};

/// Parse one axis from `(kind, payload)`.
///
/// kind:
/// - `"all"` — payload ignored
/// - `"range"` — payload is `(start, end)`
/// - `"positions"` — payload is 1-D uint64 positions
pub(crate) fn parse_axis(kind: &str, payload: &Bound<'_, PyAny>) -> PyResult<AxisIndex> {
    match kind {
        "all" => Ok(AxisIndex::All),
        "range" => {
            let (start, end): (u64, u64) = payload.extract().map_err(|_| {
                invalid_argument("range axis payload must be a (start, end) uint pair")
            })?;
            Ok(AxisIndex::range(start, end))
        }
        "positions" => {
            let values = payload
                .extract::<PyReadonlyArray1<'_, u64>>()
                .map_err(|_| invalid_argument("positions axis payload must be a 1-D uint64 array"))?;
            let slice = values.as_slice().map_err(|_| {
                invalid_argument("positions axis payload must be a C-contiguous uint64 array")
            })?;
            Ok(AxisIndex::positions(slice.iter().copied()))
        }
        other => Err(invalid_argument(format!(
            "unknown axis kind {other:?}; expected 'all', 'range', or 'positions'"
        ))),
    }
}

pub(crate) fn parse_selection(
    row_kind: &str,
    row_payload: &Bound<'_, PyAny>,
    col_kind: &str,
    col_payload: &Bound<'_, PyAny>,
) -> PyResult<Selection> {
    Ok(Selection::new(
        parse_axis(row_kind, row_payload)?,
        parse_axis(col_kind, col_payload)?,
    ))
}

pub(crate) fn parse_csr_output(name: &str) -> PyResult<CsrOutput> {
    match name {
        "sparse" | "csr" => Ok(CsrOutput::Sparse),
        "dense" => Ok(CsrOutput::Dense),
        other => Err(invalid_argument(format!(
            "csr_output must be 'sparse' or 'dense', got {other:?}"
        ))),
    }
}

pub(crate) fn dense_array_to_py<'py>(
    py: Python<'py>,
    array: DenseArray,
) -> PyResult<Bound<'py, PyAny>> {
    let (shape, dtype, values) = array.into_parts();
    dense_bytes_to_array(py, values, dtype, [shape[0] as u64, shape[1] as u64])
}

pub(crate) fn csr_array_to_py<'py>(
    py: Python<'py>,
    array: CsrArray,
) -> PyResult<Bound<'py, PyAny>> {
    let (indptr, indices, data, shape, index_dtype, value_dtype) = array.into_parts();
    let indices = csr_index_bytes_to_array(py, indices, index_dtype)?;
    let data = csr_data_bytes_to_array(py, data, value_dtype)?;
    let indptr = u64_slice_to_array(py, &indptr)?;
    let shape_t = PyTuple::new(py, [shape[0] as u64, shape[1] as u64])?;
    let kind = pyo3::types::PyString::new(py, "csr");
    PyTuple::new(
        py,
        [kind.into_any(), indices, data, indptr, shape_t.into_any()],
    )
    .map(|t| t.into_any())
}

fn pack_selected<'py>(py: Python<'py>, selected: SelectedArray) -> PyResult<Bound<'py, PyAny>> {
    match selected {
        SelectedArray::Dense(array) => {
            let arr = dense_array_to_py(py, array)?;
            let kind = pyo3::types::PyString::new(py, "dense");
            PyTuple::new(py, [kind.into_any(), arr]).map(|t| t.into_any())
        }
        SelectedArray::Csr(array) => csr_array_to_py(py, array),
    }
}

/// Select from an opened store. Returns `("dense", ndarray)` or CSR tuple.
pub(crate) fn store_select<'py>(
    py: Python<'py>,
    matrix: &Matrix,
    row_kind: &str,
    row_payload: &Bound<'_, PyAny>,
    col_kind: &str,
    col_payload: &Bound<'_, PyAny>,
    csr_output: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let selection = parse_selection(row_kind, row_payload, col_kind, col_payload)?;
    let output = parse_csr_output(csr_output)?;
    let selected = py
        .allow_threads(|| matrix.select(selection, output))
        .map_sc()?;
    pack_selected(py, selected)
}

/// In-memory dense select on a C-contiguous 2-D NumPy buffer.
pub(crate) fn dense_select_numpy<'py>(
    py: Python<'py>,
    values: &Bound<'_, PyAny>,
    row_kind: &str,
    row_payload: &Bound<'_, PyAny>,
    col_kind: &str,
    col_payload: &Bound<'_, PyAny>,
    n_workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let selection = parse_selection(row_kind, row_payload, col_kind, col_payload)?;
    let threads = n_workers.max(1);
    dispatch_dense(values, |dense, shape| {
        let (bytes, dtype) = dense_values_to_bytes(dense);
        let n_rows = usize::try_from(shape[0])
            .map_err(|_| invalid_argument("dense row count exceeds usize"))?;
        let n_cols = usize::try_from(shape[1])
            .map_err(|_| invalid_argument("dense col count exceeds usize"))?;
        let array = DenseArray::from_bytes([n_rows, n_cols], dtype, bytes).map_sc()?;
        let out = py
            .allow_threads(|| array.select(selection.clone(), threads))
            .map_sc()?;
        dense_array_to_py(py, out)
    })
}

/// In-memory CSR select.
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
    n_workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::convert::copy_u64_1d;

    let selection = parse_selection(row_kind, row_payload, col_kind, col_payload)?;
    let output = parse_csr_output(csr_output)?;
    let indptr = copy_u64_1d(indptr, "indptr")?;
    let (indices_bytes, index_dtype) = index_array_to_bytes(indices)?;
    let (data_bytes, value_dtype) = csr_data_to_bytes(data)?;
    let array = CsrArray::from_parts(
        [n_rows, n_cols],
        index_dtype,
        value_dtype,
        indptr,
        indices_bytes,
        data_bytes,
    )
    .map_sc()?;
    let selected = py
        .allow_threads(|| array.select(selection, output, n_workers.max(1)))
        .map_sc()?;
    pack_selected(py, selected)
}

/// Densify an in-memory CSR matrix.
pub(crate) fn csr_to_dense_numpy<'py>(
    py: Python<'py>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    n_rows: usize,
    n_cols: usize,
    n_workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::convert::copy_u64_1d;

    let indptr = copy_u64_1d(indptr, "indptr")?;
    let (indices_bytes, index_dtype) = index_array_to_bytes(indices)?;
    let (data_bytes, value_dtype) = csr_data_to_bytes(data)?;
    let array = CsrArray::from_parts(
        [n_rows, n_cols],
        index_dtype,
        value_dtype,
        indptr,
        indices_bytes,
        data_bytes,
    )
    .map_sc()?;
    let dense = py
        .allow_threads(|| array.to_dense(n_workers.max(1)))
        .map_sc()?;
    dense_array_to_py(py, dense)
}

fn dense_values_to_bytes(dense: DenseValues<'_>) -> (Vec<u8>, DType) {
    match dense {
        DenseValues::U16(v) => (encode_u16(v), DType::U16),
        DenseValues::U32(v) => (encode_u32(v), DType::U32),
        DenseValues::I16(v) => (encode_i16(v), DType::I16),
        DenseValues::I32(v) => (encode_i32(v), DType::I32),
        DenseValues::F32(v) => (encode_f32(v), DType::F32),
        DenseValues::F64(v) => (encode_f64(v), DType::F64),
    }
}

fn csr_data_to_bytes(data: &Bound<'_, PyAny>) -> PyResult<(Vec<u8>, DType)> {
    let mut out = None;
    dispatch_csr_data(data, |csr| {
        out = Some(match csr {
            CsrData::U16(v) => (encode_u16(v), DType::U16),
            CsrData::U32(v) => (encode_u32(v), DType::U32),
            CsrData::I16(v) => (encode_i16(v), DType::I16),
            CsrData::I32(v) => (encode_i32(v), DType::I32),
            CsrData::F32(v) => (encode_f32(v), DType::F32),
            CsrData::F64(v) => (encode_f64(v), DType::F64),
        });
        Ok(())
    })?;
    out.ok_or_else(|| invalid_argument("failed to read CSR data"))
}

fn index_array_to_bytes(indices: &Bound<'_, PyAny>) -> PyResult<(Vec<u8>, DType)> {
    if let Ok(arr) = indices.extract::<PyReadonlyArray1<'_, u16>>() {
        let slice = arr
            .as_slice()
            .map_err(|_| invalid_argument("indices must be C-contiguous"))?;
        return Ok((encode_u16(slice), DType::U16));
    }
    if let Ok(arr) = indices.extract::<PyReadonlyArray1<'_, u32>>() {
        let slice = arr
            .as_slice()
            .map_err(|_| invalid_argument("indices must be C-contiguous"))?;
        return Ok((encode_u32(slice), DType::U32));
    }
    Err(invalid_argument(
        "CSR indices must be a C-contiguous 1-D uint16 or uint32 array",
    ))
}

fn encode_u16(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_i16(values: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_i32(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_f64(values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
