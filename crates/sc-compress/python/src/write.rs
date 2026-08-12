//! Low-level writers. Callers must pre-validate / promote arrays in Python.

use std::path::PathBuf;

use pyo3::prelude::*;
use sc_compress::{CsrWriter, DenseWriter, Partition};

use crate::convert::{copy_u64_1d, dispatch_csr_data, dispatch_dense, CsrData, DenseValues};
use crate::error::{invalid_argument, ResultExt};
use crate::validate_n_workers;

fn parse_partition(policy: &str, n: u64, what: &str) -> PyResult<Partition> {
    if n == 0 {
        return Err(invalid_argument(format!("{what} size must be non-zero")));
    }
    match policy {
        "cells" | "fixed_cells" => Ok(Partition::fixed_cells(n)),
        "budget" | "bytes_budget" => Ok(Partition::bytes_budget(n)),
        other => Err(invalid_argument(format!(
            "{what}_policy must be 'cells' or 'budget', got {other:?}"
        ))),
    }
}

fn partitions(
    chunk_policy: &str,
    chunk_n: u64,
    block_policy: &str,
    block_n: u64,
) -> PyResult<(Partition, Partition)> {
    Ok((
        parse_partition(chunk_policy, chunk_n, "chunk")?,
        parse_partition(block_policy, block_n, "block")?,
    ))
}

/// Write a C-contiguous 2D NumPy matrix of a supported value dtype.
#[pyfunction(name = "_write_dense")]
#[pyo3(signature = (
    path,
    values,
    *,
    chunk_policy,
    chunk_n,
    block_policy,
    block_n,
    n_workers,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "the low-level dense boundary keeps partitions and worker count explicit"
)]
pub fn write_dense(
    py: Python<'_>,
    path: PathBuf,
    values: &Bound<'_, PyAny>,
    chunk_policy: &str,
    chunk_n: u64,
    block_policy: &str,
    block_n: u64,
    n_workers: usize,
) -> PyResult<()> {
    validate_n_workers(n_workers)?;
    let (chunk, block) = partitions(chunk_policy, chunk_n, block_policy, block_n)?;
    dispatch_dense(values, |values, shape| {
        let writer = DenseWriter::new(&path, chunk, block).threads(n_workers);
        match values {
            DenseValues::U16(values) => {
                py.allow_threads(|| writer.write(values, shape)).map_sc()?
            }
            DenseValues::U32(values) => {
                py.allow_threads(|| writer.write(values, shape)).map_sc()?
            }
            DenseValues::I16(values) => {
                py.allow_threads(|| writer.write(values, shape)).map_sc()?
            }
            DenseValues::I32(values) => {
                py.allow_threads(|| writer.write(values, shape)).map_sc()?
            }
            DenseValues::F32(values) => {
                py.allow_threads(|| writer.write(values, shape)).map_sc()?
            }
            DenseValues::F64(values) => {
                py.allow_threads(|| writer.write(values, shape)).map_sc()?
            }
        }
        Ok(())
    })
}

/// Write CSR arrays. `indptr` / `indices` must already be contiguous `uint64`.
#[pyfunction(name = "_write_csr")]
#[pyo3(signature = (
    path,
    indptr,
    indices,
    data,
    n_rows,
    n_cols,
    *,
    chunk_policy,
    chunk_n,
    block_policy,
    block_n,
    n_workers,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "the low-level Python boundary keeps CSR buffers, shape, and partitions explicit"
)]
pub fn write_csr(
    py: Python<'_>,
    path: PathBuf,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    n_rows: u64,
    n_cols: u64,
    chunk_policy: &str,
    chunk_n: u64,
    block_policy: &str,
    block_n: u64,
    n_workers: usize,
) -> PyResult<()> {
    validate_n_workers(n_workers)?;
    let (chunk, block) = partitions(chunk_policy, chunk_n, block_policy, block_n)?;
    let indptr = copy_u64_1d(indptr, "indptr")?;
    let indices = copy_u64_1d(indices, "indices")?;
    let shape = [n_rows, n_cols];
    dispatch_csr_data(data, |data| {
        let writer = CsrWriter::new(&path, chunk, block).threads(n_workers);
        py.allow_threads(move || match data {
            CsrData::U16(data) => writer.write_promoted(indptr, indices, data, shape),
            CsrData::U32(data) => writer.write_promoted(indptr, indices, data, shape),
            CsrData::I16(data) => writer.write_promoted(indptr, indices, data, shape),
            CsrData::I32(data) => writer.write_promoted(indptr, indices, data, shape),
            CsrData::F32(data) => writer.write_promoted(indptr, indices, data, shape),
            CsrData::F64(data) => writer.write_promoted(indptr, indices, data, shape),
        })
        .map_sc()?;
        Ok(())
    })
}
