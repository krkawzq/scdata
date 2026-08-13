//! Opaque dataset handle and function-style open / metadata boundary.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use sc_compress::Kind;
use sc_load::{Dataset, ReadLimits, StoreLocation};

use crate::error::ResultExt;

#[pyclass(name = "_Dataset", module = "scdata._core", frozen)]
pub(crate) struct PyDataset {
    pub(crate) inner: Dataset,
}

#[pyfunction]
#[pyo3(signature = (path, *, zip_prefix=None, maximum_metadata_size, maximum_encoded_size, maximum_decoded_size, maximum_block_count))]
pub(crate) fn dataset_open(
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

#[pyfunction]
pub(crate) fn dataset_meta<'py>(
    py: Python<'py>,
    dataset: &PyDataset,
) -> PyResult<Bound<'py, PyDict>> {
    let inner = &dataset.inner;
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
    values.set_item("n_rows", inner.n_rows())?;
    values.set_item("n_cols", inner.n_cols())?;
    values.set_item("dtype", inner.dtype().as_str())?;
    Ok(values)
}
