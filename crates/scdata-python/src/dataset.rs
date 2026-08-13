//! Opaque dataset handle and function-style open / metadata boundary.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use sc_compress::Kind;
use sc_load::{Dataset, ReadLimits, StoreLocation};

use crate::error::ResultExt;
use crate::validate_num_workers;

#[pyclass(name = "_Dataset", module = "scdata._core", frozen)]
pub(crate) struct PyDataset {
    pub(crate) inner: Dataset,
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
    reason = "the low-level dataset open boundary keeps resource limits and worker count explicit"
)]
pub(crate) fn dataset_open(
    py: Python<'_>,
    path: PathBuf,
    zip_prefix: Option<String>,
    max_metadata_size: usize,
    max_encoded_size: usize,
    max_decoded_size: usize,
    max_block_count: usize,
    num_workers: usize,
) -> PyResult<PyDataset> {
    validate_num_workers(num_workers)?;
    let limits = ReadLimits::default()
        .maximum_metadata_size(max_metadata_size)
        .maximum_encoded_size(max_encoded_size)
        .maximum_decoded_size(max_decoded_size)
        .maximum_block_count(max_block_count)
        .threads(num_workers);
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
