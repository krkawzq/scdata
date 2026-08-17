//! Opaque store handle and function-style open / decode / select boundary.

use std::path::PathBuf;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use sc_compress::{
    AxisIndex, CsrArray, CsrOutput, DenseArray, Kind, Matrix, ReadLimits, SelectedArray, Selection,
    StoreLocation,
};

use crate::convert::{
    csr_data_bytes_to_array, csr_index_bytes_to_array, dense_bytes_to_array, dtype_name,
    u64_slice_to_array, u64_vec_to_array,
};
use crate::error::{invalid_argument, ResultExt};
use crate::validate_num_workers;

#[pyclass(name = "_Store", module = "scdata._core", frozen)]
pub struct PyStore {
    inner: Matrix,
}

#[pyfunction]
#[pyo3(signature = (
    path,
    *,
    zip_prefix,
    max_metadata_size,
    max_encoded_size,
    max_decoded_size,
    max_block_count,
    num_workers,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "the low-level open boundary keeps resource limits and worker count explicit"
)]
pub fn store_open(
    py: Python<'_>,
    path: PathBuf,
    zip_prefix: Option<String>,
    max_metadata_size: usize,
    max_encoded_size: usize,
    max_decoded_size: usize,
    max_block_count: usize,
    num_workers: usize,
) -> PyResult<PyStore> {
    validate_num_workers(num_workers)?;
    let location = match zip_prefix {
        Some(prefix) => StoreLocation::zip(path, prefix),
        None => StoreLocation::dir(path),
    };
    let limits = ReadLimits::default()
        .maximum_metadata_size(max_metadata_size)
        .maximum_encoded_size(max_encoded_size)
        .maximum_decoded_size(max_decoded_size)
        .maximum_block_count(max_block_count)
        .threads(num_workers);
    let inner = py
        .allow_threads(|| Matrix::open_with_limits(location, limits))
        .map_sc()?;
    Ok(PyStore { inner })
}

#[pyfunction]
pub fn store_meta<'py>(py: Python<'py>, store: &PyStore) -> PyResult<Bound<'py, PyDict>> {
    let inner = &store.inner;
    let values = PyDict::new(py);
    values.set_item(
        "kind",
        match inner.kind() {
            Kind::Dense => "dense",
            Kind::Csr => "csr",
        },
    )?;
    let shape = inner.shape();
    values.set_item("shape", (shape[0], shape[1]))?;
    values.set_item("value_dtype", dtype_name(inner.value_dtype()))?;
    match inner {
        Matrix::Csr(matrix) => {
            values.set_item("index_dtype", dtype_name(matrix.index_dtype()))?;
            values.set_item("nnz", matrix.nnz())?;
        }
        Matrix::Dense(_) => {
            values.set_item("index_dtype", py.None())?;
            values.set_item("nnz", py.None())?;
        }
    }
    let limits = inner.limits();
    values.set_item("max_metadata_size", limits.metadata_size())?;
    values.set_item("max_encoded_size", limits.encoded_size())?;
    values.set_item("max_decoded_size", limits.decoded_size())?;
    values.set_item("max_block_count", limits.block_count())?;
    values.set_item("num_workers", limits.thread_count())?;
    match inner {
        Matrix::Csr(matrix) => {
            values.set_item(
                "compressor",
                compressor_to_py(py, &matrix.meta().data.compressor)?,
            )?;
            values.set_item(
                "indptr_compressor",
                compressor_to_py(py, &matrix.meta().indptr.compressor)?,
            )?;
        }
        Matrix::Dense(matrix) => {
            values.set_item(
                "compressor",
                compressor_to_py(py, &matrix.meta().data.compressor)?,
            )?;
            values.set_item("indptr_compressor", py.None())?;
        }
    }
    Ok(values)
}

fn compressor_to_py<'py>(
    py: Python<'py>,
    compressor: &sc_compress::Compressor,
) -> PyResult<Bound<'py, PyAny>> {
    let text = serde_json::to_string(compressor).map_err(|error| {
        invalid_argument(format!("failed to serialize store compressor: {error}"))
    })?;
    py.import("json")?.call_method1("loads", (text,))
}

#[pyfunction]
pub fn store_indptr<'py>(py: Python<'py>, store: &PyStore) -> PyResult<Option<Bound<'py, PyAny>>> {
    match &store.inner {
        Matrix::Csr(matrix) => Ok(Some(u64_slice_to_array(py, matrix.indptr())?)),
        Matrix::Dense(_) => Ok(None),
    }
}

#[pyfunction]
pub fn store_decode_dense_rows<'py>(
    py: Python<'py>,
    store: &PyStore,
    start: u64,
    end: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let Matrix::Dense(matrix) = &store.inner else {
        return Err(invalid_argument(
            "store_decode_dense_rows requires a dense store",
        ));
    };
    validate_row_range(start, end, matrix.n_rows())?;
    let bytes = py
        .allow_threads(|| matrix.decode_rows(start..end))
        .map_sc()?;
    let rows = end - start;
    dense_bytes_to_array(py, bytes, matrix.dtype(), [rows, matrix.n_cols()])
}

#[pyfunction]
pub fn store_decode_csr_rows<'py>(
    py: Python<'py>,
    store: &PyStore,
    start: u64,
    end: u64,
) -> PyResult<Bound<'py, PyTuple>> {
    let Matrix::Csr(matrix) = &store.inner else {
        return Err(invalid_argument(
            "store_decode_csr_rows requires a CSR store",
        ));
    };
    validate_row_range(start, end, matrix.n_rows())?;
    let start_index =
        usize::try_from(start).map_err(|_| invalid_argument("row start exceeds usize"))?;
    let end_index = usize::try_from(end).map_err(|_| invalid_argument("row end exceeds usize"))?;
    let (indices_bytes, data_bytes, local_indptr) = py
        .allow_threads(|| {
            let (indices, data) = matrix.decode_rows(start..end)?;
            let indptr = copy_local_indptr(matrix.indptr(), start_index, end_index)?;
            Ok::<_, sc_compress::Error>((indices, data, indptr))
        })
        .map_sc()?;
    let indices = csr_index_bytes_to_array(py, indices_bytes, matrix.index_dtype())?;
    let data = csr_data_bytes_to_array(py, data_bytes, matrix.value_dtype())?;
    let indptr_array = u64_vec_to_array(py, local_indptr);
    PyTuple::new(py, [indices, data, indptr_array])
}

#[pyfunction(name = "store_select")]
#[pyo3(signature = (
    store,
    row_kind,
    row_payload,
    col_kind,
    col_payload,
    *,
    csr_output,
))]
pub fn store_select_fn<'py>(
    py: Python<'py>,
    store: &PyStore,
    row_kind: &str,
    row_payload: &Bound<'_, PyAny>,
    col_kind: &str,
    col_payload: &Bound<'_, PyAny>,
    csr_output: &str,
) -> PyResult<Bound<'py, PyAny>> {
    store_select(
        py,
        &store.inner,
        row_kind,
        row_payload,
        col_kind,
        col_payload,
        csr_output,
    )
}

fn store_select<'py>(
    py: Python<'py>,
    matrix: &Matrix,
    row_kind: &str,
    row_payload: &Bound<'_, PyAny>,
    col_kind: &str,
    col_payload: &Bound<'_, PyAny>,
    csr_output: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let selection = Selection::new(
        parse_store_axis(row_kind, row_payload)?,
        parse_store_axis(col_kind, col_payload)?,
    );
    let output = match csr_output {
        "sparse" | "csr" => CsrOutput::Sparse,
        "dense" => CsrOutput::Dense,
        other => {
            return Err(invalid_argument(format!(
                "csr_output must be 'sparse' or 'dense', got {other:?}"
            )))
        }
    };
    let selected = py
        .allow_threads(|| matrix.select(selection, output))
        .map_sc()?;
    pack_selected(py, selected)
}

fn parse_store_axis(kind: &str, payload: &Bound<'_, PyAny>) -> PyResult<AxisIndex> {
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
                .map_err(|_| {
                    invalid_argument("positions axis payload must be a 1-D uint64 array")
                })?;
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

fn dense_array_to_py<'py>(py: Python<'py>, array: DenseArray) -> PyResult<Bound<'py, PyAny>> {
    let (shape, dtype, values) = array.into_parts();
    dense_bytes_to_array(py, values, dtype, shape_to_u64(shape)?)
}

fn csr_array_to_py<'py>(py: Python<'py>, array: CsrArray) -> PyResult<Bound<'py, PyAny>> {
    let (indptr, indices, data, shape, index_dtype, value_dtype) = array.into_parts();
    let indices = csr_index_bytes_to_array(py, indices, index_dtype)?;
    let data = csr_data_bytes_to_array(py, data, value_dtype)?;
    let indptr = u64_vec_to_array(py, indptr);
    let shape_t = PyTuple::new(py, shape_to_u64(shape)?)?;
    let kind = pyo3::types::PyString::new(py, "csr");
    PyTuple::new(
        py,
        [kind.into_any(), indices, data, indptr, shape_t.into_any()],
    )
    .map(|t| t.into_any())
}

fn validate_row_range(start: u64, end: u64, n_rows: u64) -> PyResult<()> {
    if start > end {
        return Err(invalid_argument(
            "row end must not be smaller than row start",
        ));
    }
    if end > n_rows {
        return Err(invalid_argument(format!(
            "row end {end} exceeds matrix row count {n_rows}"
        )));
    }
    Ok(())
}

fn copy_local_indptr(
    indptr: &[u64],
    start: usize,
    end: usize,
) -> Result<Vec<u64>, sc_compress::Error> {
    let source = indptr
        .get(start..=end)
        .ok_or_else(|| sc_compress::Error::CorruptData {
            context: "CSR indptr".into(),
            message: format!(
                "row range {start}..{end} is outside length {}",
                indptr.len()
            ),
        })?;
    let base = source[0];
    let mut local = Vec::new();
    local.try_reserve_exact(source.len())?;
    for &value in source {
        local.push(
            value
                .checked_sub(base)
                .ok_or_else(|| sc_compress::Error::CorruptData {
                    context: "CSR indptr".into(),
                    message: "row pointers are not monotonic".into(),
                })?,
        );
    }
    Ok(local)
}

fn shape_to_u64(shape: [usize; 2]) -> PyResult<[u64; 2]> {
    Ok([
        u64::try_from(shape[0]).map_err(|_| invalid_argument("row count exceeds u64"))?,
        u64::try_from(shape[1]).map_err(|_| invalid_argument("column count exceeds u64"))?,
    ])
}

#[cfg(test)]
mod tests {
    use super::{copy_local_indptr, validate_row_range};

    #[test]
    fn row_range_validation_accepts_empty_tail_and_rejects_invalid_bounds() {
        assert!(validate_row_range(4, 4, 4).is_ok());
        assert!(validate_row_range(3, 2, 4).is_err());
        assert!(validate_row_range(0, 5, 4).is_err());
    }

    #[test]
    fn local_indptr_is_rebased_and_checks_monotonicity() {
        assert_eq!(
            copy_local_indptr(&[5, 7, 7, 11], 0, 3).unwrap(),
            [0, 2, 2, 6]
        );
        assert!(copy_local_indptr(&[5, 4], 0, 1).is_err());
        assert!(copy_local_indptr(&[0, 1], 0, 2).is_err());
    }
}
