use std::sync::Arc;

use crate::access::AccessItem;

use super::super::dataset::{Dataset, Dense1DDataset, Dense2DDataset, SparseCsrDataset};
use super::super::error::{DataBankError, DataBankResult};
use super::super::plan::{self};

use super::super::dense::*;
use super::super::gene_axis::*;
use super::super::sparse::*;

use super::types::*;

/// Plan one batch into a [`BatchPlan`] plus its ordered access items.
///
/// Chunks are grouped within the batch only; no merging happens across
/// batches. File-backed chunks are streamed through the access scheduler;
/// memory-backed chunks stay in the batch plan and are decoded by databank when
/// the prefetched batch is assembled.
pub(crate) fn plan_single_dataset_owned(
    dataset: Arc<Dataset>,
    cells: Vec<usize>,
    output_rows: Option<Vec<usize>>,
    gene_axis: &GeneAxisPlan,
) -> DataBankResult<(Vec<usize>, SingleDatasetPlan, Vec<AccessItem>)> {
    if let Some(output_rows) = output_rows.as_ref() {
        if cells.len() != output_rows.len() {
            return Err(DataBankError::InvalidConfig(format!(
                "batch planner requires one output row per cell, got {} cells and {} rows",
                cells.len(),
                output_rows.len()
            )));
        }
    }
    let output_rows = match output_rows {
        Some(output_rows) if output_rows_are_sequential(&output_rows) => None,
        other => other,
    };
    let active_rows = cells.len();
    let explicit_rows = output_rows.as_deref();
    let projection = gene_axis.projection();
    let selected_sources = projection.map(|projection| projection.selected_sources.as_slice());
    let make_dense_1d_segments = |d: &Dense1DDataset| match (selected_sources, explicit_rows) {
        (Some(selected), Some(rows)) => {
            plan::plan_dense_1d_selected_sources_with_output_rows(d, &cells, rows, selected)
        }
        (Some(selected), None) => plan::plan_dense_1d_selected_sources(d, &cells, selected),
        (None, Some(rows)) => plan::plan_dense_1d_with_output_rows(d, &cells, rows),
        (None, None) => plan::plan_dense_1d(d, &cells),
    };
    let make_dense_2d_segments = |d: &Dense2DDataset| match (selected_sources, explicit_rows) {
        (Some(selected), Some(rows)) => {
            plan::plan_dense_2d_selected_sources_with_output_rows(d, &cells, rows, selected)
        }
        (Some(selected), None) => plan::plan_dense_2d_selected_sources(d, &cells, selected),
        (None, Some(rows)) => plan::plan_dense_2d_with_output_rows(d, &cells, rows),
        (None, None) => plan::plan_dense_2d(d, &cells),
    };
    let make_sparse_rows = |d: &SparseCsrDataset| match explicit_rows {
        Some(rows) => plan::plan_sparse_rows_with_output_rows(d, &cells, rows),
        None => plan::plan_sparse_rows(d, &cells),
    };

    // Cell range validation is performed inside the plan builders
    // (`plan_dense_*` / `plan_sparse_rows`), which call `validate_cell` per
    // cell. Re-checking here would duplicate that work on every batch.
    match dataset.as_ref() {
        Dataset::Dense1D(d) => {
            let segments = make_dense_1d_segments(d)?;
            let groups = group_dense_segments(&segments)?;
            let items = dense_group_access_items(&groups)?;
            Ok((
                cells,
                SingleDatasetPlan::Dense {
                    active_rows,
                    segments,
                    groups,
                    num_genes: d.num_genes,
                    src_dtype: d.data.dtype,
                },
                items,
            ))
        }
        Dataset::Dense2D(d) => {
            let segments = make_dense_2d_segments(d)?;
            let groups = group_dense_segments(&segments)?;
            let items = dense_group_access_items(&groups)?;
            Ok((
                cells,
                SingleDatasetPlan::Dense {
                    active_rows,
                    segments,
                    groups,
                    num_genes: d.num_genes,
                    src_dtype: d.data.dtype,
                },
                items,
            ))
        }
        Dataset::SparseCsr(d) => {
            let rows = make_sparse_rows(d)?;
            let value_size = d.data.dtype.item_size();
            let plan = plan_sparse_batch_with_value_size(d, &rows, value_size)?;
            let items = if gene_axis.projection().is_some() {
                sparse_plan_index_file_access_items(&plan)?
            } else {
                sparse_plan_file_access_items(&plan)?
            };
            Ok((
                cells,
                SingleDatasetPlan::Sparse {
                    active_rows,
                    plan,
                    dataset,
                },
                items,
            ))
        }
    }
}

pub(crate) fn plan_batch_multi(
    datasets: &[Arc<Dataset>],
    batch: MultiBatchCells,
    gene_axes: &MultiGeneAxisPlan,
) -> DataBankResult<(BatchPlan, Vec<AccessItem>)> {
    if datasets.is_empty() {
        return Err(DataBankError::InvalidConfig(
            "prefetch requires at least one dataset".to_string(),
        ));
    }

    let total_cells = batch.total_cells()?;
    let parts = batch.into_parts();

    let mut single_dataset_idx = None;
    let mut single_dataset_only = true;
    for part in &parts {
        if part.dataset_idx >= datasets.len() {
            return Err(DataBankError::InvalidConfig(format!(
                "multi batch references dataset index {}, but only {} datasets were supplied",
                part.dataset_idx,
                datasets.len()
            )));
        }
        if part.cells.is_empty() {
            continue;
        }
        match single_dataset_idx {
            Some(dataset_idx) if dataset_idx != part.dataset_idx => {
                single_dataset_only = false;
                break;
            }
            Some(_) => {}
            None => single_dataset_idx = Some(part.dataset_idx),
        }
    }
    if single_dataset_only {
        if let Some(dataset_idx) = single_dataset_idx {
            let mut cells = Vec::with_capacity(total_cells);
            for part in parts {
                cells.extend(part.cells);
            }
            let (cells, plan, items) = plan_single_dataset_owned(
                Arc::clone(&datasets[dataset_idx]),
                cells,
                None,
                gene_axes.axis_for(dataset_idx)?,
            )?;
            return Ok((
                BatchPlan::Single {
                    dataset_idx,
                    cells,
                    plan,
                },
                items,
            ));
        }
    }

    let layout = collect_multi_dataset_batch_rows(datasets, parts, total_cells)?;
    plan_multi_layout(datasets, layout, gene_axes)
}

pub(crate) fn plan_multi_layout(
    datasets: &[Arc<Dataset>],
    mut layout: MultiBatchLayout,
    gene_axes: &MultiGeneAxisPlan,
) -> DataBankResult<(BatchPlan, Vec<AccessItem>)> {
    let can_use_single_plan = layout.per_dataset.len() == 1
        && output_rows_are_sequential(&layout.per_dataset[0].output_rows);
    if can_use_single_plan {
        let dataset_batch = layout.per_dataset.pop().expect("single dataset batch");
        let dataset_idx = dataset_batch.dataset_idx;
        let (cells, plan, items) = plan_single_dataset_owned(
            Arc::clone(&datasets[dataset_idx]),
            dataset_batch.cells,
            None,
            gene_axes.axis_for(dataset_idx)?,
        )?;
        return Ok((
            BatchPlan::Single {
                dataset_idx,
                cells,
                plan,
            },
            items,
        ));
    }

    let mut parts = Vec::with_capacity(layout.per_dataset.len());
    let mut items = Vec::new();

    for dataset_batch in layout.per_dataset {
        let dataset = &datasets[dataset_batch.dataset_idx];
        let gene_axis = gene_axes.axis_for(dataset_batch.dataset_idx)?.clone();
        let active_rows = dataset_batch.cells.len();
        let (planned_cells, plan, mut part_items) = plan_single_dataset_owned(
            Arc::clone(dataset),
            dataset_batch.cells,
            Some(dataset_batch.output_rows),
            &gene_axis,
        )?;
        debug_assert_eq!(planned_cells.len(), active_rows);
        items.append(&mut part_items);
        parts.push(MultiBatchPlanPart {
            dataset_idx: dataset_batch.dataset_idx,
            gene_axis,
            active_rows,
            plan,
        });
    }

    let output_cells = layout.output_cells;
    let total_cells = output_cells.len();
    Ok((
        BatchPlan::Multi(MultiDatasetPlan {
            output_cells,
            parts,
            total_cells,
            output_genes: gene_axes.output_genes,
        }),
        items,
    ))
}

pub(crate) fn collect_multi_dataset_batch_rows(
    datasets: &[Arc<Dataset>],
    parts: Vec<BatchPart>,
    total_cells: usize,
) -> DataBankResult<MultiBatchLayout> {
    let mut all_cells = Vec::with_capacity(total_cells);
    let mut groups = Vec::<BatchRows>::new();
    let mut group_positions = fast_hash_map_with_capacity(parts.len().min(datasets.len()));
    let mut output_row = 0usize;

    for BatchPart { dataset_idx, cells } in parts {
        if dataset_idx >= datasets.len() {
            return Err(DataBankError::InvalidConfig(format!(
                "multi batch references dataset index {dataset_idx}, but only {} datasets were supplied",
                datasets.len()
            )));
        }
        if cells.is_empty() {
            continue;
        }
        let group_index = match group_positions.get(&dataset_idx).copied() {
            Some(group_index) => group_index,
            None => {
                let group_index = groups.len();
                group_positions.insert(dataset_idx, group_index);
                groups.push(BatchRows {
                    dataset_idx,
                    cells: Vec::new(),
                    output_rows: Vec::new(),
                });
                group_index
            }
        };
        let group = &mut groups[group_index];
        let part_len = cells.len();
        let next_output_row = output_row.checked_add(part_len).ok_or_else(|| {
            DataBankError::InvalidConfig("multi batch output row overflow".to_string())
        })?;
        all_cells.extend(cells.iter().copied());
        group.cells.reserve(part_len);
        group.output_rows.reserve(part_len);
        group.cells.extend(cells);
        group.output_rows.extend(output_row..next_output_row);
        output_row = next_output_row;
    }

    if output_row != total_cells {
        return Err(DataBankError::InvalidConfig(format!(
            "multi batch planned {output_row} output rows, expected {total_cells}"
        )));
    }

    Ok(MultiBatchLayout {
        output_cells: all_cells,
        per_dataset: groups,
    })
}

pub(crate) fn output_rows_are_sequential(output_rows: &[usize]) -> bool {
    output_rows
        .iter()
        .copied()
        .enumerate()
        .all(|(expected, row)| row == expected)
}

pub(crate) fn dense_group_access_items(
    groups: &[DenseReadGroup],
) -> DataBankResult<Vec<AccessItem>> {
    let mut items = Vec::with_capacity(groups.len());
    for group in groups {
        if matches!(group.source, DenseGroupSource::AccessItem(_)) {
            items.push(group_access_item(group)?);
        }
    }
    Ok(items)
}

pub(crate) fn sparse_plan_file_access_items(
    plan: &SparseBatchPlan,
) -> DataBankResult<Vec<AccessItem>> {
    let mut items = Vec::with_capacity(plan.index_groups.len() + plan.data_groups.len());
    for group in &plan.index_groups {
        if matches!(group.source, SparseGroupSource::AccessItem(_)) {
            items.push(sparse_group_access_item(group)?);
        }
    }
    for group in &plan.data_groups {
        if matches!(group.source, SparseGroupSource::AccessItem(_)) {
            items.push(sparse_group_access_item(group)?);
        }
    }
    Ok(items)
}

pub(crate) fn sparse_plan_index_file_access_items(
    plan: &SparseBatchPlan,
) -> DataBankResult<Vec<AccessItem>> {
    let mut items = Vec::with_capacity(plan.index_groups.len());
    for group in &plan.index_groups {
        if matches!(group.source, SparseGroupSource::AccessItem(_)) {
            items.push(sparse_group_access_item(group)?);
        }
    }
    Ok(items)
}
