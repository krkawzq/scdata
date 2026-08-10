//! Low-level opened store handle. Returns raw NumPy arrays only.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use sc_compress::{Kind, Matrix, ReadLimits, StoreLocation};

use crate::convert::{
    csr_data_bytes_to_array, csr_index_bytes_to_array, dense_bytes_to_array, dtype_name,
    u64_slice_to_array,
};
use crate::error::{invalid_argument, ResultExt};
use crate::select::store_select;
use crate::validate_n_workers;

/// Opened sc-compress store (directory or ZIP prefix).
#[pyclass(name = "_Store", module = "sc_compress._core", frozen)]
pub struct PyStore {
    inner: Matrix,
}

#[pymethods]
impl PyStore {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind() {
            Kind::Dense => "dense",
            Kind::Csr => "csr",
        }
    }

    #[getter]
    fn shape(&self) -> (u64, u64) {
        let shape = self.inner.shape();
        (shape[0], shape[1])
    }

    #[getter]
    fn value_dtype(&self) -> &'static str {
        dtype_name(self.inner.value_dtype())
    }

    #[getter]
    fn index_dtype(&self) -> Option<&'static str> {
        match &self.inner {
            Matrix::Csr(matrix) => Some(dtype_name(matrix.index_dtype())),
            Matrix::Dense(_) => None,
        }
    }

    #[getter]
    fn nnz(&self) -> Option<u64> {
        match &self.inner {
            Matrix::Csr(matrix) => Some(matrix.nnz()),
            Matrix::Dense(_) => None,
        }
    }

    #[getter]
    fn maximum_metadata_size(&self) -> usize {
        self.inner.limits().metadata_size()
    }

    #[getter]
    fn maximum_encoded_size(&self) -> usize {
        self.inner.limits().encoded_size()
    }

    #[getter]
    fn maximum_decoded_size(&self) -> usize {
        self.inner.limits().decoded_size()
    }

    #[getter]
    fn maximum_block_count(&self) -> usize {
        self.inner.limits().block_count()
    }

    #[getter]
    fn n_workers(&self) -> usize {
        self.inner.limits().thread_count()
    }

    /// Resident CSR indptr as `uint64`, or `None` for dense stores.
    fn indptr<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner {
            Matrix::Csr(matrix) => Ok(Some(u64_slice_to_array(py, matrix.indptr())?)),
            Matrix::Dense(_) => Ok(None),
        }
    }

    /// Decode dense rows `[start, end)` as a 2D NumPy array.
    fn decode_dense_rows<'py>(
        &self,
        py: Python<'py>,
        start: u64,
        end: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Matrix::Dense(matrix) = &self.inner else {
            return Err(invalid_argument("decode_dense_rows requires a dense store"));
        };
        let bytes = py
            .allow_threads(|| matrix.decode_rows(start..end))
            .map_sc()?;
        let rows = end
            .checked_sub(start)
            .ok_or_else(|| invalid_argument("row end must not be smaller than row start"))?;
        dense_bytes_to_array(py, bytes, matrix.dtype(), [rows, matrix.n_cols()])
    }

    /// Decode CSR rows `[start, end)` as `(indices, data, indptr)`.
    ///
    /// `indptr` is relative to the selected row window and has dtype `uint64`.
    fn decode_csr_rows<'py>(
        &self,
        py: Python<'py>,
        start: u64,
        end: u64,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let Matrix::Csr(matrix) = &self.inner else {
            return Err(invalid_argument("decode_csr_rows requires a CSR store"));
        };
        let (indices_bytes, data_bytes) = py
            .allow_threads(|| matrix.decode_rows(start..end))
            .map_sc()?;
        let start_index =
            usize::try_from(start).map_err(|_| invalid_argument("row start exceeds usize"))?;
        let end_index =
            usize::try_from(end).map_err(|_| invalid_argument("row end exceeds usize"))?;
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

    /// On-demand 2-D select.
    ///
    /// Returns:
    /// - `("dense", ndarray)` for dense stores / densified CSR
    /// - `("csr", indices, data, indptr, shape)` for sparse CSR
    #[pyo3(signature = (
        row_kind,
        row_payload,
        col_kind,
        col_payload,
        *,
        csr_output = "sparse",
    ))]
    fn select<'py>(
        &self,
        py: Python<'py>,
        row_kind: &str,
        row_payload: &Bound<'_, PyAny>,
        col_kind: &str,
        col_payload: &Bound<'_, PyAny>,
        csr_output: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        store_select(
            py,
            &self.inner,
            row_kind,
            row_payload,
            col_kind,
            col_payload,
            csr_output,
        )
    }
}

/// Open a directory store or a matrix prefix inside a ZIP archive.
#[pyfunction(name = "_open_store")]
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
pub fn open_store(
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
