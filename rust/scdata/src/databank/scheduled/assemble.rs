use std::io;
use std::sync::Arc;

use crate::access::{AccessHandle, PrefetchCancel, ScheduledAccessConfig};

use super::super::array::{DType, DataValue};
use super::super::compute::DataBankComputePool;
use super::super::config::{NativeMode, ProjectedSparseDataGroupStrategy};
use super::super::dataset::{Dataset, SparseCsrDataset};
use super::super::error::{DataBankError, DataBankResult};
use super::super::plan::DenseSegment;

use super::super::dense::*;
use super::super::gene_axis::*;
use super::super::sparse::*;
use super::super::util::*;

use super::native_access::{
    build_scheduled_batch_access, load_native_items_ordered_async,
    load_native_items_ordered_blocking, run_native_custom_blocking, NativeScheduledContext,
    ScheduledBatchAccess,
};
use super::profile::*;
use super::types::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_planned_batch<T>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axes: &MultiGeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    plan: BatchPlan,
    scheduled: &mut ScheduledBatchAccess,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
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
            native,
            native_mode,
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
                native,
                native_mode,
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
pub(crate) fn assemble_single_planned_batch<T>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    cells: Vec<usize>,
    plan: SingleDatasetPlan,
    scheduled: &mut ScheduledBatchAccess,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
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
            preloaded_index_bytes,
            selected_data_scheduled,
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
                    native,
                    native_mode,
                    cells,
                    plan,
                    preloaded_index_bytes,
                    selected_data_scheduled,
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
pub(crate) fn assemble_multi_prefetch_batch<T>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    plan: MultiDatasetPlan,
    scheduled: &mut ScheduledBatchAccess,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
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
    let mut buffer = zeroed_values(total);
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
            native,
            native_mode,
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
pub(crate) fn scatter_multi_part_into<T>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    part: MultiBatchPlanPart,
    total_cells: usize,
    output_genes: usize,
    buffer: &mut [T],
    scheduled: &mut ScheduledBatchAccess,
) -> DataBankResult<()>
where
    T: DataValue,
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
            preloaded_index_bytes,
            selected_data_scheduled,
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
                native,
                native_mode,
                total_cells,
                active_rows,
                &plan,
                preloaded_index_bytes,
                selected_data_scheduled,
                dataset,
                scheduled,
                buffer,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_dense_prefetch_batch<T>(
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
    scheduled: &mut ScheduledBatchAccess,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
{
    let output_genes = gene_axis.output_genes(num_genes);
    let total = cells
        .len()
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output length overflow".to_string()))?;
    let alloc_started = profiler.start_alloc();
    let mut buffer = zeroed_values(total);
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
pub(crate) fn scatter_dense_prefetch_into<T>(
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
    scheduled: &mut ScheduledBatchAccess,
    buffer: &mut [T],
) -> DataBankResult<usize>
where
    T: DataValue,
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
pub(crate) fn assemble_sparse_prefetch_batch<T>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    cells: Vec<usize>,
    plan: SparseBatchPlan,
    preloaded_index_bytes: Option<Arc<[u8]>>,
    selected_data_scheduled: bool,
    dataset: &SparseCsrDataset,
    scheduled: &mut ScheduledBatchAccess,
) -> DataBankResult<PrefetchedBatch<T>>
where
    T: DataValue,
{
    let output_genes = gene_axis.output_genes(dataset.num_genes);
    let total = cells
        .len()
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output length overflow".to_string()))?;
    let alloc_started = profiler.start_alloc();
    let mut buffer = zeroed_values(total);
    profiler.record_alloc(alloc_started);
    scatter_sparse_prefetch_into(
        access,
        compute,
        access_config,
        projected_sparse_data_strategy,
        gene_axis,
        cancel,
        profiler,
        native,
        native_mode,
        cells.len(),
        cells.len(),
        &plan,
        preloaded_index_bytes,
        selected_data_scheduled,
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
pub(crate) fn scatter_sparse_prefetch_into<T>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    rows: usize,
    active_rows: usize,
    plan: &SparseBatchPlan,
    preloaded_index_bytes: Option<Arc<[u8]>>,
    selected_data_scheduled: bool,
    dataset: &SparseCsrDataset,
    scheduled: &mut ScheduledBatchAccess,
    buffer: &mut [T],
) -> DataBankResult<usize>
where
    T: DataValue,
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
    let index_bytes = match preloaded_index_bytes {
        Some(bytes) => bytes,
        None => Arc::from(
            load_sparse_prefetch_indices(access, cancel, profiler, plan, scheduled)?
                .into_boxed_slice(),
        ),
    };

    if gene_axis.projection().is_some() {
        let scatter_started = profiler.start_scatter();
        if projected_sparse_data_strategy == ProjectedSparseDataGroupStrategy::ReadAll
            || should_read_all_small_projected_sparse_plan(
                projected_sparse_data_strategy,
                true,
                plan,
            )
        {
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
            scatter_sparse_prefetch_projected_selected_data(
                access,
                compute,
                access_config,
                native,
                native_mode,
                dataset,
                plan,
                index_bytes,
                gene_axis,
                cancel,
                active_rows,
                profiler,
                selected_data_scheduled,
                scheduled,
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
pub(crate) fn scatter_sparse_prefetch_projected_read_all_data<T>(
    access: &AccessHandle,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Arc<[u8]>,
    gene_axis: &GeneAxisPlan,
    scheduled: &mut ScheduledBatchAccess,
    buffer: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
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

    if read_all_selected_scatter_enabled() {
        let scan_started = profiler.start_sparse_projected_index_scan();
        let selected_plan =
            plan_sparse_selected_data_batch(dataset, plan, index_bytes.as_ref(), gene_axis)?;
        profiler.record_sparse_projected_index_scan(scan_started);

        let mut data_group_bytes = Vec::with_capacity(plan.data_groups.len());
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
                    if bytes.len() != group.bytes {
                        return Err(DataBankError::InvalidArrayMeta(format!(
                            "CSR data group decoded length is {}, expected {}",
                            bytes.len(),
                            group.bytes
                        )));
                    }
                    data_group_bytes.push(Arc::from(bytes.into_boxed_slice()));
                }
                SparseGroupSource::Memory { .. } => {
                    let load_started = profiler.start_sparse_projected_data_load();
                    let bytes = load_sparse_group(access, group);
                    profiler.record_sparse_projected_data_load(
                        load_started,
                        bytes.as_ref().map_or(0, Vec::len),
                    );
                    let bytes = bytes?;
                    if bytes.len() != group.bytes {
                        return Err(DataBankError::InvalidArrayMeta(format!(
                            "CSR data group decoded length is {}, expected {}",
                            bytes.len(),
                            group.bytes
                        )));
                    }
                    data_group_bytes.push(Arc::from(bytes.into_boxed_slice()));
                }
            }
        }
        scatter_sparse_read_all_groups_with_selected_plan(
            dataset,
            plan,
            &selected_plan,
            &index_bytes,
            &data_group_bytes,
            gene_axis,
            buffer,
        )?;
        return Ok(());
    }

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_sparse_prefetch_projected_selected_data<T>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Arc<[u8]>,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    active_rows: usize,
    profiler: &ScheduledPrefetchProfiler,
    selected_data_scheduled: bool,
    scheduled: &mut ScheduledBatchAccess,
    buffer: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
{
    if plan.data_groups.is_empty() {
        return Ok(());
    }

    if selected_data_scheduled {
        let selected_bytes = sparse_data_group_bytes(plan);
        profiler.record_sparse_projected_groups(
            ProjectedSparseDataGroupStrategy::SelectedOnly,
            plan.data_groups.len(),
            plan.data_groups.len(),
            selected_bytes,
        );
        let mut data_group_bytes = Vec::with_capacity(plan.data_groups.len());
        for group in &plan.data_groups {
            if cancel.is_cancelled() {
                return Err(DataBankError::PrefetchCancelled);
            }
            let load_started = profiler.start_sparse_projected_data_load();
            let bytes = load_selected_sparse_group(access, Some(scheduled), group);
            profiler.record_sparse_projected_data_load(
                load_started,
                bytes.as_ref().map_or(0, Vec::len),
            );
            let bytes = bytes?;
            data_group_bytes.push(Arc::from(bytes.into_boxed_slice()));
        }
        let scatter_started = profiler.start_scatter();
        let result = scatter_sparse_data_groups_projected_checked_with_projected_indices(
            dataset,
            plan,
            index_bytes.as_ref(),
            &data_group_bytes,
            gene_axis,
            buffer,
        );
        profiler.record_scatter(scatter_started);
        result?;
        return Ok(());
    }

    let selected_plan = if selected_sparse_plan_is_preplanned(plan) {
        plan.clone()
    } else {
        let scan_started = profiler.start_sparse_projected_index_scan();
        let selected_plan =
            plan_sparse_selected_data_batch(dataset, plan, index_bytes.as_ref(), gene_axis)?;
        profiler.record_sparse_projected_index_scan(scan_started);
        selected_plan
    };

    let selected_groups = all_sparse_data_group_indices(&selected_plan);
    let selected_bytes = sparse_data_group_bytes(&selected_plan);
    profiler.record_sparse_projected_groups(
        ProjectedSparseDataGroupStrategy::SelectedOnly,
        plan.data_groups.len(),
        selected_plan.data_groups.len(),
        selected_bytes,
    );
    if selected_plan.data_groups.is_empty() {
        return Ok(());
    }

    let output_genes = gene_axis.output_genes(dataset.num_genes);
    let output_rows = row_count_for_width(buffer.len(), output_genes);
    let parallel_rows = active_rows.min(output_rows);
    let parallel_bytes = parallel_rows
        .checked_mul(output_genes)
        .and_then(|values| values.checked_mul(std::mem::size_of::<T>()))
        .ok_or_else(|| {
            DataBankError::InvalidConfig("sparse active output byte length overflow".to_string())
        })?;
    if !selected_data_scheduled && compute.should_parallelize(parallel_rows, parallel_bytes) {
        let load_started = profiler.start_sparse_projected_data_load();
        let data_group_bytes = load_selected_sparse_group_bytes(
            access,
            access_config,
            native,
            native_mode,
            &selected_plan,
            &selected_groups,
            cancel,
        );
        profiler.record_sparse_projected_data_load(
            load_started,
            data_group_bytes.as_ref().map_or(0, |_| selected_bytes),
        );
        let data_group_bytes = data_group_bytes?;
        if try_scatter_sparse_rows_parallel_checked_with_group_indices(
            compute,
            dataset,
            &selected_plan,
            Arc::clone(&index_bytes),
            data_group_bytes,
            output_genes,
            gene_axis.projection().cloned(),
            None,
            buffer,
        )? {
            return Ok(());
        }
    }

    if !selected_data_scheduled
        && selected_sparse_fused_scatter_enabled()
        && can_use_selected_sparse_ordered_native(
            native,
            native_mode,
            &selected_plan,
            &selected_groups,
            cancel,
        )
    {
        scatter_selected_sparse_groups_fused_native(
            access,
            native.expect("native availability checked"),
            native_mode,
            selected_plan,
            selected_groups,
            Arc::clone(&index_bytes),
            dataset,
            gene_axis,
            cancel,
            profiler,
            selected_bytes,
            buffer,
        )?;
        return Ok(());
    }

    if !selected_data_scheduled {
        if let Some(data_group_bytes) = try_load_selected_sparse_group_bytes_ordered_native(
            access,
            native,
            native_mode,
            &selected_plan,
            &selected_groups,
            cancel,
            profiler,
            selected_bytes,
        )? {
            if cancel.is_cancelled() {
                return Err(DataBankError::PrefetchCancelled);
            }
            scatter_sparse_data_groups_projected_checked_with_projected_indices(
                dataset,
                &selected_plan,
                index_bytes.as_ref(),
                &data_group_bytes,
                gene_axis,
                buffer,
            )?;
            return Ok(());
        }
    }

    let mut scheduled = schedule_selected_sparse_file_groups(
        access,
        access_config,
        native,
        native_mode,
        &selected_plan,
        &selected_groups,
        cancel,
    )?;
    for &group_index in &selected_groups {
        if cancel.is_cancelled() {
            return Err(DataBankError::PrefetchCancelled);
        }
        let group = &selected_plan.data_groups[group_index];
        if try_scatter_sparse_memory_identity_data_group_projected_checked(
            dataset,
            &selected_plan.data_pieces,
            group,
            index_bytes.as_ref(),
            gene_axis,
            buffer,
        )? {
            continue;
        }

        let load_started = profiler.start_sparse_projected_data_load();
        let bytes = load_selected_sparse_group(access, scheduled.as_mut(), group);
        profiler
            .record_sparse_projected_data_load(load_started, bytes.as_ref().map_or(0, Vec::len));
        let bytes = bytes?;
        scatter_sparse_data_group_projected_checked_with_projected_indices(
            dataset,
            &selected_plan.data_pieces,
            group,
            index_bytes.as_ref(),
            &bytes,
            gene_axis,
            selected_plan.projected_indices.as_ref(),
            buffer,
        )?;
    }
    finish_scheduled_batch_access(scheduled)
}

pub(crate) fn load_sparse_prefetch_indices(
    access: &AccessHandle,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    plan: &SparseBatchPlan,
    scheduled: &mut ScheduledBatchAccess,
) -> DataBankResult<Vec<u8>> {
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
pub(crate) fn scatter_sparse_prefetch_data<T>(
    access: &AccessHandle,
    _compute: &DataBankComputePool,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Arc<[u8]>,
    scheduled: &mut ScheduledBatchAccess,
    _active_rows: usize,
    buffer: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
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

pub(crate) fn next_scheduled_bytes(
    scheduled: &mut ScheduledBatchAccess,
    profiler: &ScheduledPrefetchProfiler,
) -> DataBankResult<Vec<u8>> {
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

fn sparse_data_group_bytes(plan: &SparseBatchPlan) -> usize {
    plan.data_groups
        .iter()
        .fold(0usize, |total, group| total.saturating_add(group.bytes))
}

fn all_sparse_data_group_indices(plan: &SparseBatchPlan) -> Vec<usize> {
    (0..plan.data_groups.len()).collect()
}

fn selected_sparse_plan_is_preplanned(plan: &SparseBatchPlan) -> bool {
    plan.index_groups.is_empty() && plan.index_pieces.is_empty() && plan.projected_indices.is_some()
}

fn load_selected_sparse_group_bytes(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    plan: &SparseBatchPlan,
    selected_groups: &[usize],
    cancel: &Arc<PrefetchCancel>,
) -> DataBankResult<Vec<Arc<[u8]>>> {
    let mut scheduled = schedule_selected_sparse_file_groups(
        access,
        access_config,
        native,
        native_mode,
        plan,
        selected_groups,
        cancel,
    )?;
    let mut out = Vec::with_capacity(selected_groups.len());
    for &group_index in selected_groups {
        if cancel.is_cancelled() {
            return Err(DataBankError::PrefetchCancelled);
        }
        let group = &plan.data_groups[group_index];
        let bytes = load_selected_sparse_group(access, scheduled.as_mut(), group)?;
        if bytes.len() != group.bytes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "CSR data group decoded length is {}, expected {}",
                bytes.len(),
                group.bytes
            )));
        }
        out.push(Arc::from(bytes.into_boxed_slice()));
    }
    finish_scheduled_batch_access(scheduled)?;
    Ok(out)
}

fn can_use_selected_sparse_ordered_native(
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    plan: &SparseBatchPlan,
    selected_groups: &[usize],
    cancel: &Arc<PrefetchCancel>,
) -> bool {
    if native_mode == NativeMode::Disabled || selected_groups.is_empty() || cancel.is_cancelled() {
        return false;
    }
    let Some(native) = native else {
        return false;
    };
    if native_mode == NativeMode::Auto && !native.config.enabled {
        return false;
    }
    selected_groups.iter().all(|&group_index| {
        matches!(
            plan.data_groups[group_index].source,
            SparseGroupSource::AccessItem(_)
        )
    })
}

fn selected_sparse_fused_scatter_enabled() -> bool {
    std::env::var("SCDATA_NATIVE_FUSED_SCATTER")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn read_all_selected_scatter_enabled() -> bool {
    std::env::var("SCDATA_READALL_SELECTED_SCATTER")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

#[allow(clippy::too_many_arguments)]
fn scatter_selected_sparse_groups_fused_native<T>(
    access: &AccessHandle,
    native: &NativeScheduledContext,
    native_mode: NativeMode,
    selected_plan: SparseBatchPlan,
    selected_groups: Vec<usize>,
    index_bytes: Arc<[u8]>,
    dataset: &SparseCsrDataset,
    gene_axis: &GeneAxisPlan,
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    selected_bytes: usize,
    buffer: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
{
    let mut items = Vec::with_capacity(selected_groups.len());
    for &group_index in &selected_groups {
        items.push(file_sparse_group_access_item(
            &selected_plan.data_groups[group_index],
        ));
    }

    let dataset_addr = dataset as *const SparseCsrDataset as usize;
    let gene_axis_addr = gene_axis as *const GeneAxisPlan as usize;
    let out_addr = buffer.as_mut_ptr() as usize;
    let out_len = buffer.len();
    let access_for_job = access.clone();
    let native_for_job = native.clone();
    let cancel_for_job = Arc::clone(cancel);

    let load_started = profiler.start_sparse_projected_data_load();
    let result = run_native_custom_blocking(
        native.clone(),
        Arc::clone(cancel),
        Box::new(move |runtime| {
            let loaded = runtime.block_on(load_native_items_ordered_async(
                access_for_job,
                native_for_job,
                items,
                native_mode,
                Arc::clone(&cancel_for_job),
            ))?;
            if loaded.len() != selected_groups.len() {
                return Err(DataBankError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "native fused selected CSR data returned wrong group count",
                )));
            }

            let mut data_group_bytes = Vec::with_capacity(loaded.len());
            for (&group_index, bytes) in selected_groups.iter().zip(loaded) {
                let group = &selected_plan.data_groups[group_index];
                if bytes.len() != group.bytes {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "CSR data group decoded length is {}, expected {}",
                        bytes.len(),
                        group.bytes
                    )));
                }
                data_group_bytes.push(Arc::from(bytes.into_boxed_slice()));
            }
            if cancel_for_job.is_cancelled() {
                return Err(DataBankError::PrefetchCancelled);
            }

            // SAFETY: the custom native command is submitted synchronously and
            // `run_native_custom_blocking` does not return until this closure has
            // finished. The referenced dataset, gene axis and output buffer are
            // all owned by the active assemble call for that whole interval.
            let dataset = unsafe { &*(dataset_addr as *const SparseCsrDataset) };
            let gene_axis = unsafe { &*(gene_axis_addr as *const GeneAxisPlan) };
            let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut T, out_len) };
            scatter_sparse_data_groups_projected_checked_with_projected_indices(
                dataset,
                &selected_plan,
                index_bytes.as_ref(),
                &data_group_bytes,
                gene_axis,
                out,
            )
        }),
    );
    profiler.record_sparse_projected_data_load(
        load_started,
        result.as_ref().map_or(0, |_| selected_bytes),
    );
    result
}

fn try_load_selected_sparse_group_bytes_ordered_native(
    access: &AccessHandle,
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    plan: &SparseBatchPlan,
    selected_groups: &[usize],
    cancel: &Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    selected_bytes: usize,
) -> DataBankResult<Option<Vec<Arc<[u8]>>>> {
    if !can_use_selected_sparse_ordered_native(native, native_mode, plan, selected_groups, cancel) {
        return Ok(None);
    }
    let native = native.expect("native availability checked");

    let mut items = Vec::with_capacity(selected_groups.len());
    for &group_index in selected_groups {
        items.push(file_sparse_group_access_item(
            &plan.data_groups[group_index],
        ));
    }
    let load_started = profiler.start_sparse_projected_data_load();
    let loaded = load_native_items_ordered_blocking(
        access.clone(),
        native.clone(),
        items,
        native_mode,
        Arc::clone(cancel),
    );
    profiler.record_sparse_projected_data_load(
        load_started,
        loaded.as_ref().map_or(0, |_| selected_bytes),
    );
    let loaded = loaded?;

    if loaded.len() != selected_groups.len() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "native ordered selected CSR data returned wrong group count",
        )));
    }
    let mut out = Vec::with_capacity(loaded.len());
    for (&group_index, bytes) in selected_groups.iter().zip(loaded) {
        let group = &plan.data_groups[group_index];
        if bytes.len() != group.bytes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "CSR data group decoded length is {}, expected {}",
                bytes.len(),
                group.bytes
            )));
        }
        out.push(Arc::from(bytes.into_boxed_slice()));
    }
    Ok(Some(out))
}

fn schedule_selected_sparse_file_groups(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
    plan: &SparseBatchPlan,
    selected_groups: &[usize],
    cancel: &Arc<PrefetchCancel>,
) -> DataBankResult<Option<ScheduledBatchAccess>> {
    if cancel.is_cancelled() {
        return Err(DataBankError::PrefetchCancelled);
    }

    let mut items = Vec::new();
    for &group_index in selected_groups {
        let group = &plan.data_groups[group_index];
        if matches!(group.source, SparseGroupSource::AccessItem(_)) {
            items.push(file_sparse_group_access_item(group));
        }
    }
    if items.is_empty() {
        return Ok(None);
    }
    build_scheduled_batch_access(
        access.clone(),
        native.cloned(),
        items,
        *access_config,
        native_mode,
        Arc::clone(cancel),
        true,
    )
    .map(Some)
}

fn load_selected_sparse_group(
    access: &AccessHandle,
    scheduled: Option<&mut ScheduledBatchAccess>,
    group: &SparseReadGroup,
) -> DataBankResult<Vec<u8>> {
    match &group.source {
        SparseGroupSource::AccessItem(_) => {
            let scheduled = scheduled.ok_or_else(|| {
                DataBankError::InvalidArrayMeta(
                    "CSR file group missing scheduled reader".to_string(),
                )
            })?;
            next_dynamic_scheduled_bytes(scheduled)
        }
        SparseGroupSource::Memory { .. } => load_sparse_group(access, group),
    }
}

fn next_dynamic_scheduled_bytes(scheduled: &mut ScheduledBatchAccess) -> DataBankResult<Vec<u8>> {
    match scheduled.next() {
        Some(Ok(bytes)) => Ok(bytes),
        Some(Err(err)) => Err(DataBankError::Io(err)),
        None => Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "scheduled CSR data access ended before selected data was complete",
        ))),
    }
}

fn finish_scheduled_batch_access(scheduled: Option<ScheduledBatchAccess>) -> DataBankResult<()> {
    let Some(mut scheduled) = scheduled else {
        return Ok(());
    };
    match scheduled.next() {
        None => Ok(()),
        Some(Ok(_)) => Err(DataBankError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "scheduled CSR data access returned extra output",
        ))),
        Some(Err(err)) => Err(DataBankError::Io(err)),
    }
}
