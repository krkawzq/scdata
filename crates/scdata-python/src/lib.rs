//! Private PyO3 boundary for the public `scdata` package.
//!
//! Public Python classes live in pure Python. This module exposes opaque
//! handles and function-style operations only.

mod config;
mod convert;
mod dataset;
mod error;
mod output;
mod plan;
mod session;
#[cfg(all(target_os = "linux", target_has_atomic = "64"))]
mod shared;
mod slice;
mod stats;
mod store;
mod write;

use pyo3::prelude::*;

use crate::dataset::{dataset_meta, dataset_open, PyDataset};
use crate::plan::{plan_compile, plan_meta, plan_open, plan_stats, PyPlan};
use crate::session::{
    session_cancel, session_close, session_meta, session_next, session_stats, PySession,
};
use crate::slice::{csr_select_numpy, csr_to_dense_numpy, dense_select_numpy};
use crate::store::{
    store_decode_csr_rows, store_decode_dense_rows, store_indptr, store_meta, store_open,
    store_select_fn, PyStore,
};
use crate::write::{write_csr, write_dense};

pub(crate) fn validate_num_workers(num_workers: usize) -> PyResult<()> {
    if num_workers == 0 {
        return Err(error::invalid_argument(
            "num_workers must be greater than zero",
        ));
    }
    Ok(())
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    error::register(module)?;
    module.add_class::<PyStore>()?;
    module.add_class::<PyDataset>()?;
    module.add_class::<PyPlan>()?;
    module.add_class::<PySession>()?;
    module.add_function(wrap_pyfunction!(store_open, module)?)?;
    module.add_function(wrap_pyfunction!(store_meta, module)?)?;
    module.add_function(wrap_pyfunction!(store_indptr, module)?)?;
    module.add_function(wrap_pyfunction!(store_decode_dense_rows, module)?)?;
    module.add_function(wrap_pyfunction!(store_decode_csr_rows, module)?)?;
    module.add_function(wrap_pyfunction!(store_select_fn, module)?)?;
    module.add_function(wrap_pyfunction!(write_dense, module)?)?;
    module.add_function(wrap_pyfunction!(write_csr, module)?)?;
    module.add_function(wrap_pyfunction!(matrix_dense_select, module)?)?;
    module.add_function(wrap_pyfunction!(matrix_csr_select, module)?)?;
    module.add_function(wrap_pyfunction!(matrix_csr_to_dense, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_open, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_meta, module)?)?;
    module.add_function(wrap_pyfunction!(plan_compile, module)?)?;
    module.add_function(wrap_pyfunction!(plan_meta, module)?)?;
    module.add_function(wrap_pyfunction!(plan_stats, module)?)?;
    module.add_function(wrap_pyfunction!(plan_open, module)?)?;
    module.add_function(wrap_pyfunction!(session_next, module)?)?;
    module.add_function(wrap_pyfunction!(session_cancel, module)?)?;
    module.add_function(wrap_pyfunction!(session_close, module)?)?;
    module.add_function(wrap_pyfunction!(session_meta, module)?)?;
    module.add_function(wrap_pyfunction!(session_stats, module)?)?;
    #[cfg(all(target_os = "linux", target_has_atomic = "64"))]
    {
        use crate::plan::plan_open_shared;
        module.add_function(wrap_pyfunction!(plan_open_shared, module)?)?;
        shared::register(module)?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    values,
    row_kind,
    row_payload,
    col_kind,
    col_payload,
    *,
    num_workers,
))]
fn matrix_dense_select<'py>(
    py: Python<'py>,
    values: &Bound<'_, PyAny>,
    row_kind: &str,
    row_payload: &Bound<'_, PyAny>,
    col_kind: &str,
    col_payload: &Bound<'_, PyAny>,
    num_workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    validate_num_workers(num_workers)?;
    dense_select_numpy(
        py,
        values,
        row_kind,
        row_payload,
        col_kind,
        col_payload,
        num_workers,
    )
}

#[pyfunction]
#[pyo3(signature = (
    indptr,
    indices,
    data,
    n_rows,
    n_cols,
    row_kind,
    row_payload,
    col_kind,
    col_payload,
    *,
    csr_output,
    num_workers,
))]
#[allow(clippy::too_many_arguments)]
fn matrix_csr_select<'py>(
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
    validate_num_workers(num_workers)?;
    csr_select_numpy(
        py,
        indptr,
        indices,
        data,
        n_rows,
        n_cols,
        row_kind,
        row_payload,
        col_kind,
        col_payload,
        csr_output,
        num_workers,
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_rows, n_cols, *, num_workers))]
fn matrix_csr_to_dense<'py>(
    py: Python<'py>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    n_rows: usize,
    n_cols: usize,
    num_workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    validate_num_workers(num_workers)?;
    csr_to_dense_numpy(py, indptr, indices, data, n_rows, n_cols, num_workers)
}
