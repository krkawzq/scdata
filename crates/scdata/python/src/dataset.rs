//! Low-level opened dataset handle.

use std::path::PathBuf;

use pyo3::prelude::*;
use sc_compress::Kind;
use scdata::{Dataset, ReadLimits, StoreLocation};

use crate::error::ResultExt;

#[pyclass(name = "_Dataset", module = "scdata._core", frozen)]
pub(crate) struct PyDataset {
    pub(crate) inner: Dataset,
}

#[pymethods]
impl PyDataset {
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
    fn n_rows(&self) -> u64 {
        self.inner.n_rows()
    }

    #[getter]
    fn n_cols(&self) -> u64 {
        self.inner.n_cols()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }
}

#[pyfunction(name = "_open_dataset")]
#[pyo3(signature = (path, *, zip_prefix=None, maximum_metadata_size, maximum_encoded_size, maximum_decoded_size, maximum_block_count))]
pub(crate) fn open_dataset(
    py: Python<'_>,
    path: PathBuf,
    zip_prefix: Option<String>,
    maximum_metadata_size: usize,
    maximum_encoded_size: usize,
    maximum_decoded_size: usize,
    maximum_block_count: usize,
) -> PyResult<PyDataset> {
    let limits = ReadLimits::default()
        .maximum_metadata_size(maximum_metadata_size)
        .maximum_encoded_size(maximum_encoded_size)
        .maximum_decoded_size(maximum_decoded_size)
        .maximum_block_count(maximum_block_count);
    let location = match zip_prefix {
        Some(prefix) => StoreLocation::zip(path, prefix),
        None => StoreLocation::dir(path),
    };
    let inner = py
        .allow_threads(|| Dataset::open_with_limits(location, limits))
        .map_sc()?;
    Ok(PyDataset { inner })
}
