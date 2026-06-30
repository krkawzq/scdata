use std::io;
use std::mem;
use std::ptr;
use std::sync::Arc;

use crate::access::{
    AccessHandle, AccessItem, ChunkKey, PrefetchCancel, RangeCopy, ScheduledAccess,
    ScheduledAccessConfig, SliceSpec,
};
use crate::codecs::{CodecError, SharedCodec};

use super::array::{Array, Chunk, ChunkRef, ChunkSource, DType, DataValue};
use super::compute::{ComputeJob, DataBankComputePool};
use super::dataset::SparseCsrDataset;
use super::error::{DataBankError, DataBankResult};
use super::plan::{self, ByteRange, SparseRowSpan};

use super::gene_axis::*;
use super::util::*;

mod load;
mod memory;
mod planning;
mod scatter;
mod types;

pub(crate) use load::*;
pub(crate) use memory::*;
pub(crate) use planning::*;
pub(crate) use scatter::*;
pub(crate) use types::*;

pub(crate) fn access_sparse<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<()> {
    zero_values(out);
    access_sparse_zeroed(access, compute, access_config, dataset, cells, out)
}

pub(crate) fn access_sparse_zeroed<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<()> {
    if try_scatter_sparse_single_memory_chunk_checked(dataset, cells, out)? {
        return Ok(());
    }
    if try_scatter_sparse_memory_chunks_checked(dataset, cells, out)? {
        return Ok(());
    }
    let rows = plan::plan_sparse_rows(dataset, cells)?;
    let plan = plan_sparse_batch::<T>(dataset, &rows)?;
    if sparse_plan_is_file_backed(&plan) {
        access_sparse_file_groups_checked(access, compute, access_config, dataset, &plan, out)?;
    } else {
        let index_bytes = load_sparse_index_groups(access, access_config, &plan)?;
        load_sparse_data_groups_and_scatter_checked(
            access,
            compute,
            access_config,
            dataset,
            &plan,
            index_bytes,
            out,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn access_sparse_projected<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    cells: &[usize],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
    out_is_zeroed: bool,
) -> DataBankResult<()> {
    if !out_is_zeroed {
        zero_values(out);
    }
    let rows = plan::plan_sparse_rows(dataset, cells)?;
    let plan = plan_sparse_batch::<T>(dataset, &rows)?;
    let index_bytes = load_sparse_index_groups(access, access_config, &plan)?;
    load_sparse_data_groups_and_scatter_projected_checked(
        access,
        compute,
        access_config,
        dataset,
        &plan,
        index_bytes,
        gene_axis,
        None,
        None,
        out,
    )?;
    Ok(())
}

pub(crate) unsafe fn access_sparse_unchecked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<()> {
    zero_values(out);
    if unsafe { try_scatter_sparse_single_memory_chunk_unchecked(dataset, cells, out) }? {
        return Ok(());
    }
    if unsafe { try_scatter_sparse_memory_chunks_unchecked(dataset, cells, out) }? {
        return Ok(());
    }
    // SAFETY: the caller guarantees requested cell indices and CSR indptr are
    // valid. Downstream scatter also assumes all gene indices are in range.
    let rows = unsafe { plan::plan_sparse_rows_unchecked(dataset, cells) };
    let plan = plan_sparse_batch::<T>(dataset, &rows)?;
    if sparse_plan_is_file_backed(&plan) {
        unsafe {
            access_sparse_file_groups_unchecked(
                access,
                compute,
                access_config,
                dataset,
                &plan,
                out,
            )?;
        }
    } else {
        let index_bytes = load_sparse_index_groups(access, access_config, &plan)?;
        unsafe {
            load_sparse_data_groups_and_scatter_unchecked(
                access,
                compute,
                access_config,
                dataset,
                &plan,
                index_bytes,
                out,
            )?;
        }
    }
    Ok(())
}
