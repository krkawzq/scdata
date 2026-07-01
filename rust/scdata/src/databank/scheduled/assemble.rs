use std::io;
use std::sync::Arc;

use crate::access::{
    AccessHandle, AccessItem, PrefetchCancel, ScheduledAccess, ScheduledAccessConfig,
};

use super::super::array::{DType, DataValue};
use super::super::compute::DataBankComputePool;
use super::super::config::ProjectedSparseDataGroupStrategy;
use super::super::dataset::{Dataset, SparseCsrDataset};
use super::super::error::{DataBankError, DataBankResult};
use super::super::plan::DenseSegment;

use super::super::dense::*;
use super::super::gene_axis::*;
use super::super::sparse::*;
use super::super::util::*;

use super::profile::*;
use super::types::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_planned_batch<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axes: &MultiGeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    plan: BatchPlan,
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let assemble_started = profiler.start_assemble_total();
    let result = match plan {
        BatchPlan::Multi(plan) => assemble_multi_prefetch_batch(
            access,
            compute,
            access_config,
            projected_sparse_data_strategy,
            cancel,
            profiler,
            plan,
            scheduled,
        ),
        BatchPlan::Single {
            dataset_idx,
            cells,
            plan,
        } => {
            let gene_axis = gene_axes.axis_for(dataset_idx)?;
            assemble_single_planned_batch(
                access,
                compute,
                access_config,
                projected_sparse_data_strategy,
                gene_axis,
                cancel,
                profiler,
                cells,
                plan,
                scheduled,
            )
        }
    };
    profiler.record_assemble_total(assemble_started);
    if let Ok(batch) = &result {
        profiler.record_assembled_batch(
            batch.cells.len(),
            batch.num_genes,
            batch.buffer.len(),
            batch.buffer.len().saturating_mul(std::mem::size_of::<T>()),
        );
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_single_planned_batch<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    cells: Vec<usize>,
    plan: SingleDatasetPlan,
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    match plan {
        SingleDatasetPlan::Dense {
            active_rows: _,
            segments,
            groups,
            num_genes,
            src_dtype,
        } => assemble_dense_prefetch_batch(
            access, compute, gene_axis, cancel, profiler, cells, segments, groups, num_genes,
            src_dtype, scheduled,
        ),
        SingleDatasetPlan::Sparse {
            active_rows: _,
            plan,
            dataset,
        } => {
            if let Dataset::SparseCsr(dataset) = dataset.as_ref() {
                assemble_sparse_prefetch_batch(
                    access,
                    compute,
                    access_config,
                    projected_sparse_data_strategy,
                    gene_axis,
                    cancel,
                    profiler,
                    cells,
                    plan,
                    dataset,
                    scheduled,
                )
            } else {
                Err(DataBankError::InvalidArrayMeta(
                    "sparse prefetch plan carried non-CSR dataset".to_string(),
                ))
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_multi_prefetch_batch<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    plan: MultiDatasetPlan,
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let MultiDatasetPlan {
        output_cells,
        parts,
        total_cells,
        output_genes,
    } = plan;
    let total = total_cells
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("multi output length overflow".to_string()))?;
    let alloc_started = profiler.start_alloc();
    let mut buffer = vec![T::zero(); total];
    profiler.record_alloc(alloc_started);

    for part in parts {
        if cancel.is_cancelled() {
            return Err(DataBankError::PrefetchCancelled);
        }
        scatter_multi_part_into(
            access,
            compute,
            access_config,
            projected_sparse_data_strategy,
            cancel,
            profiler,
            part,
            total_cells,
            output_genes,
            &mut buffer,
            scheduled,
        )?;
    }

    Ok(PrefetchedBatch {
        cells: output_cells,
        buffer,
        num_genes: output_genes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_multi_part_into<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    part: MultiBatchPlanPart,
    total_cells: usize,
    output_genes: usize,
    buffer: &mut [T],
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<()>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    match part.plan {
        SingleDatasetPlan::Dense {
            active_rows,
            segments,
            groups,
            num_genes,
            src_dtype,
        } => {
            if active_rows != part.active_rows {
                return Err(DataBankError::InvalidConfig(format!(
                    "multi batch part {} active row mismatch: plan has {active_rows}, part has {}",
                    part.dataset_idx, part.active_rows
                )));
            }
            let part_output_genes = part.gene_axis.output_genes(num_genes);
            if part_output_genes != output_genes {
                return Err(DataBankError::InvalidConfig(format!(
                    "multi batch part {} produced {part_output_genes} genes, expected {output_genes}",
                    part.dataset_idx
                )));
            }
            scatter_dense_prefetch_into(
                access,
                compute,
                &part.gene_axis,
                cancel,
                profiler,
                total_cells,
                active_rows,
                &segments,
                &groups,
                num_genes,
                src_dtype,
                scheduled,
                buffer,
            )?;
        }
        SingleDatasetPlan::Sparse {
            active_rows,
            plan,
            dataset,
        } => {
            if active_rows != part.active_rows {
                return Err(DataBankError::InvalidConfig(format!(
                    "multi batch part {} active row mismatch: plan has {active_rows}, part has {}",
                    part.dataset_idx, part.active_rows
                )));
            }
            let Dataset::SparseCsr(dataset) = dataset.as_ref() else {
                return Err(DataBankError::InvalidArrayMeta(
                    "sparse prefetch plan carried non-CSR dataset".to_string(),
                ));
            };
            let part_output_genes = part.gene_axis.output_genes(dataset.num_genes);
            if part_output_genes != output_genes {
                return Err(DataBankError::InvalidConfig(format!(
                    "multi batch part {} produced {part_output_genes} genes, expected {output_genes}",
                    part.dataset_idx
                )));
            }
            scatter_sparse_prefetch_into(
                access,
                compute,
                access_config,
                projected_sparse_data_strategy,
                &part.gene_axis,
                cancel,
                profiler,
                total_cells,
                active_rows,
                &plan,
                dataset,
                scheduled,
                buffer,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_dense_prefetch_batch<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    cells: Vec<usize>,
    segments: Vec<DenseSegment>,
    groups: Vec<DenseReadGroup>,
    num_genes: usize,
    src_dtype: DType,
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let output_genes = gene_axis.output_genes(num_genes);
    let total = cells
        .len()
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output length overflow".to_string()))?;
    let alloc_started = profiler.start_alloc();
    let mut buffer = vec![T::zero(); total];
    profiler.record_alloc(alloc_started);
    scatter_dense_prefetch_into(
        access,
        compute,
        gene_axis,
        cancel,
        profiler,
        cells.len(),
        cells.len(),
        &segments,
        &groups,
        num_genes,
        src_dtype,
        scheduled,
        &mut buffer,
    )?;
    Ok(PrefetchedBatch {
        cells,
        buffer,
        num_genes: output_genes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_dense_prefetch_into<T, J>(
    access: &AccessHandle,
    _compute: &DataBankComputePool,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    rows: usize,
    active_rows: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    num_genes: usize,
    src_dtype: DType,
    scheduled: &mut ScheduledAccess<J>,
    buffer: &mut [T],
) -> DataBankResult<usize>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let output_genes = gene_axis.output_genes(num_genes);
    let total = rows
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output length overflow".to_string()))?;
    if buffer.len() != total {
        return Err(DataBankError::BufferSizeMismatch {
            expected: total,
            actual: buffer.len(),
        });
    }
    if active_rows > rows {
        return Err(DataBankError::InvalidConfig(format!(
            "dense active row count {active_rows} exceeds output rows {rows}"
        )));
    }
    // The parallel-scatter branch is intentionally absent here: this function
    // runs inside a compute worker, where `should_parallelize` is guarded by
    // `!in_databank_worker()` and would never fire. Scheduled batches rely on
    // inter-batch parallelism (multiple in-flight response jobs) instead.
    for group in groups {
        if cancel.is_cancelled() {
            return Err(DataBankError::PrefetchCancelled);
        }
        match &group.source {
            DenseGroupSource::AccessItem(_) => {
                let bytes = next_scheduled_bytes(scheduled, profiler)?;
                let scatter_started = profiler.start_scatter();
                if gene_axis.projection().is_some() {
                    scatter_dense_group_projected(
                        num_genes,
                        output_genes,
                        segments,
                        group,
                        &bytes,
                        src_dtype,
                        gene_axis,
                        buffer,
                    )?;
                } else {
                    scatter_dense_group(num_genes, segments, group, &bytes, src_dtype, buffer)?;
                }
                profiler.record_scatter(scatter_started);
            }
            DenseGroupSource::Memory { .. } => {
                let scatter_started = profiler.start_scatter();
                let scattered = if gene_axis.projection().is_some() {
                    try_scatter_dense_memory_identity_group_projected(
                        num_genes,
                        output_genes,
                        segments,
                        group,
                        src_dtype,
                        gene_axis,
                        buffer,
                    )?
                } else {
                    try_scatter_dense_memory_identity_group(
                        num_genes, segments, group, src_dtype, buffer,
                    )?
                };
                profiler.record_scatter(scatter_started);
                if !scattered {
                    let load_started = profiler.start_memory_load();
                    let bytes = load_dense_group(access, group)?;
                    profiler.record_memory_load(load_started, bytes.len());
                    let scatter_started = profiler.start_scatter();
                    if gene_axis.projection().is_some() {
                        scatter_dense_group_projected(
                            num_genes,
                            output_genes,
                            segments,
                            group,
                            &bytes,
                            src_dtype,
                            gene_axis,
                            buffer,
                        )?;
                    } else {
                        scatter_dense_group(num_genes, segments, group, &bytes, src_dtype, buffer)?;
                    }
                    profiler.record_scatter(scatter_started);
                }
            }
        }
    }
    Ok(output_genes)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_sparse_prefetch_batch<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    cells: Vec<usize>,
    plan: SparseBatchPlan,
    dataset: &SparseCsrDataset,
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let output_genes = gene_axis.output_genes(dataset.num_genes);
    let total = cells
        .len()
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output length overflow".to_string()))?;
    let alloc_started = profiler.start_alloc();
    let mut buffer = vec![T::zero(); total];
    profiler.record_alloc(alloc_started);
    scatter_sparse_prefetch_into(
        access,
        compute,
        access_config,
        projected_sparse_data_strategy,
        gene_axis,
        cancel,
        profiler,
        cells.len(),
        cells.len(),
        &plan,
        dataset,
        scheduled,
        &mut buffer,
    )?;

    Ok(PrefetchedBatch {
        cells,
        buffer,
        num_genes: output_genes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_sparse_prefetch_into<T, J>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    rows: usize,
    active_rows: usize,
    plan: &SparseBatchPlan,
    dataset: &SparseCsrDataset,
    scheduled: &mut ScheduledAccess<J>,
    buffer: &mut [T],
) -> DataBankResult<usize>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let output_genes = gene_axis.output_genes(dataset.num_genes);
    let total = rows
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output length overflow".to_string()))?;
    if buffer.len() != total {
        return Err(DataBankError::BufferSizeMismatch {
            expected: total,
            actual: buffer.len(),
        });
    }
    if active_rows > rows {
        return Err(DataBankError::InvalidConfig(format!(
            "sparse active row count {active_rows} exceeds output rows {rows}"
        )));
    }
    let index_bytes = load_sparse_prefetch_indices(access, cancel, profiler, plan, scheduled)?;

    if gene_axis.projection().is_some() {
        let scatter_started = profiler.start_scatter();
        if projected_sparse_data_strategy == ProjectedSparseDataGroupStrategy::ReadAll {
            scatter_sparse_prefetch_projected_read_all_data(
                access,
                cancel,
                profiler,
                dataset,
                plan,
                index_bytes,
                gene_axis,
                scheduled,
                buffer,
            )?;
        } else {
            load_sparse_data_groups_and_scatter_projected_checked(
                access,
                compute,
                access_config,
                projected_sparse_data_strategy,
                dataset,
                plan,
                index_bytes,
                gene_axis,
                Some(cancel),
                Some(active_rows),
                Some(profiler),
                buffer,
            )?;
        }
        profiler.record_scatter(scatter_started);
    } else {
        scatter_sparse_prefetch_data(
            access,
            compute,
            cancel,
            profiler,
            dataset,
            plan,
            index_bytes,
            scheduled,
            active_rows,
            buffer,
        )?;
    }

    Ok(output_genes)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_sparse_prefetch_projected_read_all_data<T, J>(
    access: &AccessHandle,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Vec<u8>,
    gene_axis: &GeneAxisPlan,
    scheduled: &mut ScheduledAccess<J>,
    buffer: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    let selected_bytes = plan
        .data_groups
        .iter()
        .fold(0usize, |total, group| total.saturating_add(group.bytes));
    profiler.record_sparse_projected_groups(
        ProjectedSparseDataGroupStrategy::ReadAll,
        plan.data_groups.len(),
        plan.data_groups.len(),
        selected_bytes,
    );

    for group in &plan.data_groups {
        if cancel.is_cancelled() {
            return Err(DataBankError::PrefetchCancelled);
        }
        match &group.source {
            SparseGroupSource::AccessItem(_) => {
                let load_started = profiler.start_sparse_projected_data_load();
                let bytes = next_scheduled_bytes(scheduled, profiler);
                profiler.record_sparse_projected_data_load(
                    load_started,
                    bytes.as_ref().map_or(0, Vec::len),
                );
                let bytes = bytes?;
                scatter_sparse_data_group_projected_checked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    &bytes,
                    gene_axis,
                    buffer,
                )?;
            }
            SparseGroupSource::Memory { .. } => {
                if try_scatter_sparse_memory_identity_data_group_projected_checked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    gene_axis,
                    buffer,
                )? {
                    continue;
                }
                let load_started = profiler.start_sparse_projected_data_load();
                let bytes = load_sparse_group(access, group);
                profiler.record_sparse_projected_data_load(
                    load_started,
                    bytes.as_ref().map_or(0, Vec::len),
                );
                let bytes = bytes?;
                scatter_sparse_data_group_projected_checked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    &bytes,
                    gene_axis,
                    buffer,
                )?;
            }
        }
    }
    Ok(())
}

pub(crate) fn load_sparse_prefetch_indices<J>(
    access: &AccessHandle,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    plan: &SparseBatchPlan,
    scheduled: &mut ScheduledAccess<J>,
) -> DataBankResult<Vec<u8>>
where
    J: Iterator<Item = AccessItem>,
{
    let alloc_started = profiler.start_alloc();
    let mut index_bytes = zeroed_byte_vec(plan.index_bytes);
    profiler.record_alloc(alloc_started);
    for group in &plan.index_groups {
        if cancel.is_cancelled() {
            return Err(DataBankError::PrefetchCancelled);
        }
        match &group.source {
            SparseGroupSource::AccessItem(_) => {
                let bytes = next_scheduled_bytes(scheduled, profiler)?;
                let scatter_started = profiler.start_scatter();
                copy_sparse_group_to_index_buffer(
                    &plan.index_pieces,
                    group,
                    &bytes,
                    &mut index_bytes,
                )?;
                profiler.record_scatter(scatter_started);
            }
            SparseGroupSource::Memory { .. } => {
                let scatter_started = profiler.start_scatter();
                if try_copy_sparse_memory_identity_group_to_index_buffer(
                    &plan.index_pieces,
                    group,
                    &mut index_bytes,
                )? {
                    profiler.record_scatter(scatter_started);
                    continue;
                }
                profiler.record_scatter(scatter_started);
                let load_started = profiler.start_memory_load();
                let bytes = load_sparse_group(access, group)?;
                profiler.record_memory_load(load_started, bytes.len());
                let scatter_started = profiler.start_scatter();
                copy_sparse_group_to_index_buffer(
                    &plan.index_pieces,
                    group,
                    &bytes,
                    &mut index_bytes,
                )?;
                profiler.record_scatter(scatter_started);
            }
        }
    }
    Ok(index_bytes)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_sparse_prefetch_data<T, J>(
    access: &AccessHandle,
    _compute: &DataBankComputePool,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Vec<u8>,
    scheduled: &mut ScheduledAccess<J>,
    _active_rows: usize,
    buffer: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    J: Iterator<Item = AccessItem>,
{
    // The parallel-scatter branch is intentionally absent here: this function
    // runs inside a compute worker, where `should_parallelize` is guarded by
    // `!in_databank_worker()` and would never fire. Scheduled batches rely on
    // inter-batch parallelism (multiple in-flight response jobs) instead.
    for group in &plan.data_groups {
        if cancel.is_cancelled() {
            return Err(DataBankError::PrefetchCancelled);
        }
        match &group.source {
            SparseGroupSource::AccessItem(_) => {
                let bytes = next_scheduled_bytes(scheduled, profiler)?;
                let scatter_started = profiler.start_scatter();
                scatter_sparse_data_group_checked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    &bytes,
                    buffer,
                )?;
                profiler.record_scatter(scatter_started);
            }
            SparseGroupSource::Memory { .. } => {
                let scatter_started = profiler.start_scatter();
                if try_scatter_sparse_memory_identity_data_group_checked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    buffer,
                )? {
                    profiler.record_scatter(scatter_started);
                    continue;
                }
                profiler.record_scatter(scatter_started);
                let load_started = profiler.start_memory_load();
                let bytes = load_sparse_group(access, group)?;
                profiler.record_memory_load(load_started, bytes.len());
                let scatter_started = profiler.start_scatter();
                scatter_sparse_data_group_checked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    &bytes,
                    buffer,
                )?;
                profiler.record_scatter(scatter_started);
            }
        }
    }
    Ok(())
}

pub(crate) fn next_scheduled_bytes<J>(
    scheduled: &mut ScheduledAccess<J>,
    profiler: &ScheduledPrefetchProfiler,
) -> DataBankResult<Vec<u8>>
where
    J: Iterator<Item = AccessItem>,
{
    let drain_started = profiler.start_scheduled_drain();
    let result = match scheduled.next() {
        Some(Ok(bytes)) => Ok(bytes),
        Some(Err(err)) => Err(DataBankError::Io(err)),
        None => Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "scheduled prefetch ended before batch was complete",
        ))),
    };
    profiler.record_scheduled_drain(drain_started, result.as_ref().map_or(0, Vec::len));
    result
}
