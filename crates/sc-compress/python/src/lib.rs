//! Low-level Python core for the `sc-compress` matrix format.
//!
//! Top-level NumPy ergonomics and `ScDense` / `ScCsr` live in the pure-Python package.

mod convert;
mod error;
mod select;
mod store;
mod write;

use pyo3::prelude::*;
use sc_compress::{ReadLimits, FORMAT_NAME, FORMAT_VERSION};

use crate::select::{csr_select_numpy, csr_to_dense_numpy, dense_select_numpy};
use crate::store::{open_store, PyStore};
use crate::write::{write_csr, write_dense};

/// Supported matrix value dtypes (NumPy / on-disk names).
const VALUE_DTYPES: [&str; 6] = ["u16", "u32", "i16", "i32", "f32", "f64"];
/// Supported on-disk CSR index dtypes.
const INDEX_DTYPES: [&str; 2] = ["u16", "u32"];

pub(crate) fn validate_n_workers(n_workers: usize) -> PyResult<()> {
    if n_workers == 0 {
        return Err(error::invalid_argument(
            "n_workers must be greater than zero",
        ));
    }
    Ok(())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("FORMAT_NAME", FORMAT_NAME)?;
    m.add("FORMAT_VERSION", FORMAT_VERSION)?;
    m.add("VALUE_DTYPES", VALUE_DTYPES)?;
    m.add("INDEX_DTYPES", INDEX_DTYPES)?;
    let default_limits = ReadLimits::default();
    m.add(
        "DEFAULT_MAXIMUM_METADATA_SIZE",
        default_limits.metadata_size(),
    )?;
    m.add(
        "DEFAULT_MAXIMUM_ENCODED_SIZE",
        default_limits.encoded_size(),
    )?;
    m.add(
        "DEFAULT_MAXIMUM_DECODED_SIZE",
        default_limits.decoded_size(),
    )?;
    m.add("DEFAULT_MAXIMUM_BLOCK_COUNT", default_limits.block_count())?;
    m.add("DEFAULT_N_WORKERS", default_limits.thread_count())?;
    error::register(m)?;
    m.add_class::<PyStore>()?;
    m.add_function(wrap_pyfunction!(write_dense, m)?)?;
    m.add_function(wrap_pyfunction!(write_csr, m)?)?;
    m.add_function(wrap_pyfunction!(open_store, m)?)?;
    m.add_function(wrap_pyfunction!(dense_select, m)?)?;
    m.add_function(wrap_pyfunction!(csr_select, m)?)?;
    m.add_function(wrap_pyfunction!(csr_to_dense, m)?)?;
    Ok(())
}

/// In-memory dense row/column select (returns a NumPy ndarray).
#[pyfunction(name = "_dense_select")]
#[pyo3(signature = (
    values,
    row_kind,
    row_payload,
    col_kind,
    col_payload,
    *,
    n_workers,
))]
fn dense_select<'py>(
    py: Python<'py>,
    values: &Bound<'_, PyAny>,
    row_kind: &str,
    row_payload: &Bound<'_, PyAny>,
    col_kind: &str,
    col_payload: &Bound<'_, PyAny>,
    n_workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    validate_n_workers(n_workers)?;
    dense_select_numpy(
        py,
        values,
        row_kind,
        row_payload,
        col_kind,
        col_payload,
        n_workers,
    )
}

/// In-memory CSR select. Returns `("dense", ndarray)` or CSR tuple.
#[pyfunction(name = "_csr_select")]
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
    csr_output = "sparse",
    n_workers,
))]
#[allow(clippy::too_many_arguments)]
fn csr_select<'py>(
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
    validate_n_workers(n_workers)?;
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
        n_workers,
    )
}

/// Densify an in-memory CSR matrix to a NumPy ndarray.
#[pyfunction(name = "_csr_to_dense")]
#[pyo3(signature = (indptr, indices, data, n_rows, n_cols, *, n_workers))]
fn csr_to_dense<'py>(
    py: Python<'py>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    n_rows: usize,
    n_cols: usize,
    n_workers: usize,
) -> PyResult<Bound<'py, PyAny>> {
    validate_n_workers(n_workers)?;
    csr_to_dense_numpy(py, indptr, indices, data, n_rows, n_cols, n_workers)
}
