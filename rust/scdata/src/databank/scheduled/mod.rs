use std::sync::Arc;
use std::thread;

use crate::access::AccessHandle;

use super::array::DataValue;
use super::compute::DataBankComputePool;
use super::config::ScheduledPrefetchConfig;
use super::dataset::Dataset;
use super::error::{DataBankError, DataBankResult};

use super::gene_axis::*;

mod assemble;
mod planner;
mod producer;
mod profile;
mod types;

#[cfg(test)]
pub(crate) use planner::plan_batch_multi;
#[cfg(test)]
pub(crate) use types::{BatchPlan, SingleDatasetPlan};
pub use types::{PrefetchCells, PrefetchedBatch};

use producer::*;
use profile::*;
use types::*;

/// Build a scheduled prefetcher over `batch_source`.
///
/// Each item from `batch_source` is one batch of cell indices for `dataset`.
/// The prefetcher plans batches one at a time, streams their access items into
/// the access scheduler, and assembles decoded results in a background
/// producer. Completed batches are stored in a bounded queue of depth
/// `prefetch_step`; `next()` only pops that completed queue and blocks only
/// when the producer cannot keep up.
pub fn prefetch_cells_scheduled<T, I>(
    access: &AccessHandle,
    compute: Arc<DataBankComputePool>,
    dataset: Arc<Dataset>,
    batch_source: I,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: AsRef<[usize]> + Send,
{
    config.validate().map_err(DataBankError::InvalidConfig)?;
    let src_dtype = dataset.data_dtype();
    if !src_dtype.can_cast_to(T::DTYPE) {
        return Err(DataBankError::CannotCast {
            src: src_dtype,
            dst: T::DTYPE,
            reason:
                "scheduled prefetch output dtype cannot hold source values (float→int forbidden)",
        });
    }
    spawn_prefetch_cells(
        access.clone(),
        compute,
        dataset,
        batch_source,
        GeneAxisPlan::dataset_order(),
        config,
    )
}

pub fn prefetch_cells_scheduled_multi<T, I>(
    access: &AccessHandle,
    compute: Arc<DataBankComputePool>,
    datasets: Arc<[Arc<Dataset>]>,
    batch_source: I,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: Into<MultiBatchCells> + Send,
{
    config.validate().map_err(DataBankError::InvalidConfig)?;
    validate_multi_cast::<T>(&datasets)?;
    let gene_axes = MultiGeneAxisPlan::dataset_order(datasets.as_ref())?;
    spawn_prefetch_cells_multi(
        access.clone(),
        compute,
        datasets,
        batch_source,
        gene_axes,
        config,
    )
}

pub fn prefetch_cells_scheduled_by_gene_names<T, I, G>(
    access: &AccessHandle,
    compute: Arc<DataBankComputePool>,
    dataset: Arc<Dataset>,
    batch_source: I,
    gene_names: &[G],
    missing: MissingGenePolicy,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: AsRef<[usize]> + Send,
    G: AsRef<str>,
{
    config.validate().map_err(DataBankError::InvalidConfig)?;
    let src_dtype = dataset.data_dtype();
    if !src_dtype.can_cast_to(T::DTYPE) {
        return Err(DataBankError::CannotCast {
            src: src_dtype,
            dst: T::DTYPE,
            reason:
                "scheduled prefetch output dtype cannot hold source values (float→int forbidden)",
        });
    }
    let gene_axis = GeneAxisPlan::requested(dataset.as_ref(), gene_names, missing)?;
    spawn_prefetch_cells(
        access.clone(),
        compute,
        dataset,
        batch_source,
        gene_axis,
        config,
    )
}

pub fn prefetch_cells_scheduled_multi_by_gene_names<T, I, G>(
    access: &AccessHandle,
    compute: Arc<DataBankComputePool>,
    datasets: Arc<[Arc<Dataset>]>,
    batch_source: I,
    gene_names: &[G],
    missing: MissingGenePolicy,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: Into<MultiBatchCells> + Send,
    G: AsRef<str>,
{
    config.validate().map_err(DataBankError::InvalidConfig)?;
    validate_multi_cast::<T>(&datasets)?;
    let gene_axes = MultiGeneAxisPlan::requested(datasets.as_ref(), gene_names, missing)?;
    spawn_prefetch_cells_multi(
        access.clone(),
        compute,
        datasets,
        batch_source,
        gene_axes,
        config,
    )
}

pub(crate) fn validate_multi_cast<T: DataValue>(datasets: &[Arc<Dataset>]) -> DataBankResult<()> {
    if datasets.is_empty() {
        return Err(DataBankError::InvalidConfig(
            "prefetch requires at least one dataset".to_string(),
        ));
    }
    for dataset in datasets {
        let src_dtype = dataset.data_dtype();
        if !src_dtype.can_cast_to(T::DTYPE) {
            return Err(DataBankError::CannotCast {
                src: src_dtype,
                dst: T::DTYPE,
                reason: "scheduled prefetch output dtype cannot hold source values (float→int forbidden)",
            });
        }
    }
    Ok(())
}

pub(crate) fn spawn_prefetch_cells<T, I>(
    access: AccessHandle,
    compute: Arc<DataBankComputePool>,
    dataset: Arc<Dataset>,
    batch_source: I,
    gene_axis: GeneAxisPlan,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: AsRef<[usize]> + Send,
{
    let gene_axes = MultiGeneAxisPlan::from_single(dataset.as_ref(), gene_axis);
    let datasets: Arc<[Arc<Dataset>]> = Arc::from(vec![dataset].into_boxed_slice());
    spawn_prefetch_cells_multi(
        access,
        compute,
        datasets,
        batch_source.map(|cells| MultiBatchCells::from_single(cells.as_ref().to_vec())),
        gene_axes,
        config,
    )
}

pub(crate) fn spawn_prefetch_cells_multi<T, I>(
    access: AccessHandle,
    compute: Arc<DataBankComputePool>,
    datasets: Arc<[Arc<Dataset>]>,
    batch_source: I,
    gene_axes: MultiGeneAxisPlan,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: Into<MultiBatchCells> + Send,
{
    let output_names = gene_axes.output_names.clone();
    let retained_datasets = Arc::clone(&datasets);
    let prefetch_step = config.prefetch_step;
    let (tx, rx) = flume::bounded(prefetch_step);
    let cancel = PrefetchCancelRegistry::new();
    let profiler = ScheduledPrefetchProfiler::from_env();
    let producer = PrefetchProducer {
        access,
        compute,
        datasets,
        batch_source,
        access_config: config.access,
        gene_axes: Arc::new(gene_axes),
        tx,
        cancel: Arc::clone(&cancel),
        prefetch_step,
        profiler,
    };
    let handle = thread::Builder::new()
        .name("databank-prefetch-producer".to_string())
        .spawn(move || producer.run())?;
    Ok(PrefetchCells {
        rx: Some(rx),
        output_names,
        _datasets: retained_datasets,
        prefetch_step,
        cancel,
        producer: Some(handle),
    })
}
