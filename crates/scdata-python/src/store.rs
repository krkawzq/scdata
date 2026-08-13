//! Opaque store handle and function-style open / decode / select boundary.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use sc_compress::{Kind, Matrix, ReadLimits, StoreLocation};

use crate::convert::{
    csr_data_bytes_to_array, csr_index_bytes_to_array, dense_bytes_to_array, dtype_name,
    u64_slice_to_array,
};
use crate::error::{invalid_argument, ResultExt};
use crate::select::store_select;
use crate::validate_n_workers;

#[pyclass(name = "_Store", module = "scdata._core", frozen)]
pub struct PyStore {
    inner: Matrix,
}

#[pyfunction]
#[pyo3(signature = (
    path,
    *,
    zip_prefix = None,
    maximum_metadata_size,
    maximum_encoded_size,
    maximum_decoded_size,
    maximum_block_count,
    n_workers,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "the low-level open boundary keeps resource limits and worker count explicit"
)]
pub fn store_open(
    py: Python<'_>,
    path: PathBuf,
    zip_prefix: Option<String>,
    maximum_metadata_size: usize,
    maximum_encoded_size: usize,
    maximum_decoded_size: usize,
    maximum_block_count: usize,
    n_workers: usize,
) -> PyResult<PyStore> {
    validate_n_workers(n_workers)?;
    let location = match zip_prefix {
        Some(prefix) => StoreLocation::zip(path, prefix),
        None => StoreLocation::dir(path),
    };
    let limits = ReadLimits::default()
        .maximum_metadata_size(maximum_metadata_size)
        .maximum_encoded_size(maximum_encoded_size)
        .maximum_decoded_size(maximum_decoded_size)
        .maximum_block_count(maximum_block_count)
        .threads(n_workers);
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
    values.set_item("maximum_metadata_size", limits.metadata_size())?;
    values.set_item("maximum_encoded_size", limits.encoded_size())?;
    values.set_item("maximum_decoded_size", limits.decoded_size())?;
    values.set_item("maximum_block_count", limits.block_count())?;
    values.set_item("n_workers", limits.thread_count())?;
    Ok(values)
}

#[pyfunction]
pub fn store_indptr<'py>(
    py: Python<'py>,
    store: &PyStore,
) -> PyResult<Option<Bound<'py, PyAny>>> {
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
        return Err(invalid_argument("store_decode_dense_rows requires a dense store"));
    };
    let bytes = py
        .allow_threads(|| matrix.decode_rows(start..end))
        .map_sc()?;
    let rows = end
        .checked_sub(start)
        .ok_or_else(|| invalid_argument("row end must not be smaller than row start"))?;
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
        return Err(invalid_argument("store_decode_csr_rows requires a CSR store"));
    };
    let (indices_bytes, data_bytes) = py
        .allow_threads(|| matrix.decode_rows(start..end))
        .map_sc()?;
    let start_index =
        usize::try_from(start).map_err(|_| invalid_argument("row start exceeds usize"))?;
    let end_index = usize::try_from(end).map_err(|_| invalid_argument("row end exceeds usize"))?;
    let indptr = matrix.indptr();
    let base = *indptr
        .get(start_index)
        .ok_or_else(|| invalid_argument("row start exceeds indptr"))?;
    let local_indptr = indptr[start_index..=end_index]
        .iter()
        .map(|value| value - base)
        .collect::<Vec<_>>();
    let indices = csr_index_bytes_to_array(py, indices_bytes, matrix.index_dtype())?;
    let data = csr_data_bytes_to_array(py, data_bytes, matrix.value_dtype())?;
    let indptr_array = u64_slice_to_array(py, &local_indptr)?;
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
    csr_output = "sparse",
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
