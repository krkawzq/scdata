#![allow(clippy::too_many_arguments)]

use crate::access::{AccessHandle, ScheduledAccessConfig};

use super::array::DataValue;
use super::compute::DataBankComputePool;
use super::dataset::Dataset;
use super::error::{DataBankError, DataBankResult};
use super::interner::GeneNameView;
use super::plan::{self};

use super::dense::*;
use super::gene_axis::*;
use super::sparse::*;
use super::util::*;

pub fn validate_access<T: DataValue>(
    dataset: &Dataset,
    cells: &[usize],
    out: &[T],
    names: Option<&[GeneNameView]>,
    output_genes: usize,
) -> DataBankResult<()> {
    validate_dtype_and_cells::<T>(dataset, cells)?;
    let expected = cells
        .len()
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("output length overflow".to_string()))?;
    if out.len() != expected {
        return Err(DataBankError::BufferSizeMismatch {
            expected,
            actual: out.len(),
        });
    }
    if let Some(names) = names {
        let expected = output_genes;
        if names.len() != expected {
            return Err(DataBankError::NameBufferSizeMismatch {
                expected,
                actual: names.len(),
            });
        }
    }
    Ok(())
}

pub fn access_cells<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
) -> DataBankResult<()> {
    let gene_axis = GeneAxisPlan::dataset_order();
    validate_access(
        dataset,
        cells,
        out,
        names.as_deref(),
        gene_axis.output_genes(dataset.num_genes()),
    )?;
    access_cells_validated(access, compute, access_config, dataset, cells, out, false)?;
    if let Some(names) = names {
        gene_axis.fill_names(dataset, names)?;
    }
    Ok(())
}

pub(super) fn access_cells_validated<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    out: &mut [T],
    // `true` means every slot in `out` is already `T::zero()` (currently only
    // databank-owned freshly allocated buffers pass this). Sparse and projected
    // dense paths rely on zero fill for missing entries.
    out_is_zeroed: bool,
) -> DataBankResult<()> {
    match dataset {
        Dataset::Dense1D(dataset) => {
            access_dense_1d(access, compute, access_config, dataset, cells, out)?
        }
        Dataset::Dense2D(dataset) => {
            access_dense_2d(access, compute, access_config, dataset, cells, out)?
        }
        Dataset::SparseCsr(dataset) => {
            if out_is_zeroed {
                access_sparse_zeroed(access, compute, access_config, dataset, cells, out)?
            } else {
                access_sparse(access, compute, access_config, dataset, cells, out)?
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn access_cells_by_gene_names<T, G>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    gene_names: &[G],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
    missing: MissingGenePolicy,
) -> DataBankResult<()>
where
    T: DataValue,
    G: AsRef<str>,
{
    let gene_axis = GeneAxisPlan::requested(dataset, gene_names, missing)?;
    access_cells_with_gene_axis(
        access,
        compute,
        access_config,
        dataset,
        cells,
        out,
        names,
        &gene_axis,
    )
}

pub fn access_cells_by_gene_names_owned<T, G>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    gene_names: &[G],
    missing: MissingGenePolicy,
) -> DataBankResult<Vec<T>>
where
    T: DataValue,
    G: AsRef<str>,
{
    let gene_axis = GeneAxisPlan::requested(dataset, gene_names, missing)?;
    validate_dtype_and_cells::<T>(dataset, cells)?;
    let output_genes = gene_axis.output_genes(dataset.num_genes());
    let total = cells
        .len()
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("output length overflow".to_string()))?;
    let mut out = vec![T::zero(); total];
    access_cells_with_gene_axis_validated(
        access,
        compute,
        access_config,
        dataset,
        cells,
        &mut out,
        None,
        &gene_axis,
        true,
    )?;
    Ok(out)
}

pub(super) fn access_cells_with_gene_axis<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
    gene_axis: &GeneAxisPlan,
) -> DataBankResult<()> {
    let output_genes = gene_axis.output_genes(dataset.num_genes());
    validate_access(dataset, cells, out, names.as_deref(), output_genes)?;
    access_cells_with_gene_axis_validated(
        access,
        compute,
        access_config,
        dataset,
        cells,
        out,
        names,
        gene_axis,
        false,
    )
}

pub(super) fn access_cells_with_gene_axis_validated<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
    gene_axis: &GeneAxisPlan,
    // Same invariant as `access_cells_validated`: when true, every output slot
    // already contains `T::zero()`, so zero-fill-only paths can skip clearing.
    out_is_zeroed: bool,
) -> DataBankResult<()> {
    if gene_axis.projection().is_none() {
        access_cells_validated(
            access,
            compute,
            access_config,
            dataset,
            cells,
            out,
            out_is_zeroed,
        )?;
        if let Some(names) = names {
            gene_axis.fill_names(dataset, names)?;
        }
        return Ok(());
    }
    match dataset {
        Dataset::Dense1D(dataset) => access_dense_1d_projected(
            access,
            compute,
            access_config,
            dataset,
            cells,
            gene_axis,
            out,
            out_is_zeroed,
        )?,
        Dataset::Dense2D(dataset) => access_dense_2d_projected(
            access,
            compute,
            access_config,
            dataset,
            cells,
            gene_axis,
            out,
            out_is_zeroed,
        )?,
        Dataset::SparseCsr(dataset) => access_sparse_projected(
            access,
            compute,
            access_config,
            dataset,
            cells,
            gene_axis,
            out,
            out_is_zeroed,
        )?,
    }
    if let Some(names) = names {
        gene_axis.fill_names(dataset, names)?;
    }
    Ok(())
}

pub unsafe fn access_cells_unchecked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dataset,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
) -> DataBankResult<()> {
    match dataset {
        Dataset::SparseCsr(dataset) => {
            // SAFETY: forwarded from DataBank::access_cells_unchecked. The caller
            // promises the CSR metadata, requested cells, output buffer, dtype,
            // and gene indices are valid for unchecked scatter.
            unsafe {
                access_sparse_unchecked(access, compute, access_config, dataset, cells, out)?;
            }
            if let Some(names) = names {
                let expected = dataset.num_genes;
                if names.len() != expected {
                    return Err(DataBankError::NameBufferSizeMismatch {
                        expected,
                        actual: names.len(),
                    });
                }
                names.copy_from_slice(dataset.genes.views());
            }
            Ok(())
        }
        _ => access_cells(access, compute, access_config, dataset, cells, out, names),
    }
}

pub fn prefetch_cells(
    access: &AccessHandle,
    dataset: &Dataset,
    cells: &[usize],
) -> DataBankResult<()> {
    for &cell in cells {
        if cell >= dataset.num_cells() {
            return Err(DataBankError::CellIndexOutOfRange {
                cell,
                num_cells: dataset.num_cells(),
            });
        }
    }

    let mut keys = FastHashSet::default();
    match dataset {
        Dataset::Dense1D(dataset) => {
            for segment in plan::plan_dense_1d(dataset, cells)? {
                collect_prefetch_key(&mut keys, &segment.chunk);
            }
        }
        Dataset::Dense2D(dataset) => {
            for segment in plan::plan_dense_2d(dataset, cells)? {
                collect_prefetch_key(&mut keys, &segment.chunk);
            }
        }
        Dataset::SparseCsr(dataset) => {
            for row in plan::plan_sparse(dataset, cells)? {
                collect_range_prefetch_keys(&mut keys, &row.indices);
                collect_range_prefetch_keys(&mut keys, &row.data);
            }
        }
    }
    prefetch_keys(access, keys)
}
