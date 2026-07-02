use std::sync::Arc;
use std::thread;

use crate::access::{AccessHandle, IoBackend};

use super::array::DataValue;
use super::compute::DataBankComputePool;
use super::config::{NativeAccessConfig, NativeMode, ScheduledPrefetchConfig};
use super::dataset::Dataset;
use super::error::{DataBankError, DataBankResult};

use super::gene_axis::*;

mod assemble;
mod native_access;
mod planner;
mod producer;
mod profile;
mod types;

#[cfg(test)]
pub(crate) use planner::plan_batch_multi;
#[cfg(test)]
pub(crate) use types::{BatchPlan, SingleDatasetPlan};
pub use types::{PrefetchCells, PrefetchedBatch};

use native_access::{AccessStrategy, NativeScheduledContext, ResolvedStrategy};
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
#[allow(clippy::too_many_arguments)]
pub fn prefetch_cells_scheduled<T, I>(
    access: &AccessHandle,
    compute: Arc<DataBankComputePool>,
    dataset: Arc<Dataset>,
    batch_source: I,
    config: ScheduledPrefetchConfig,
    native_config: NativeAccessConfig,
    native_io: Arc<dyn IoBackend>,
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
    let native = build_native_ctx(&config, native_config, native_io)?;
    spawn_prefetch_cells(
        access.clone(),
        compute,
        dataset,
        batch_source,
        GeneAxisPlan::dataset_order(),
        config,
        native,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefetch_cells_scheduled_multi<T, I>(
    access: &AccessHandle,
    compute: Arc<DataBankComputePool>,
    datasets: Arc<[Arc<Dataset>]>,
    batch_source: I,
    config: ScheduledPrefetchConfig,
    native_config: NativeAccessConfig,
    native_io: Arc<dyn IoBackend>,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: Into<MultiBatchCells> + Send,
{
    config.validate().map_err(DataBankError::InvalidConfig)?;
    validate_multi_cast::<T>(&datasets)?;
    let gene_axes = MultiGeneAxisPlan::dataset_order(datasets.as_ref())?;
    let native = build_native_ctx(&config, native_config, native_io)?;
    spawn_prefetch_cells_multi(
        access.clone(),
        compute,
        datasets,
        batch_source,
        gene_axes,
        config,
        native,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefetch_cells_scheduled_by_gene_names<T, I, G>(
    access: &AccessHandle,
    compute: Arc<DataBankComputePool>,
    dataset: Arc<Dataset>,
    batch_source: I,
    gene_names: &[G],
    missing: MissingGenePolicy,
    config: ScheduledPrefetchConfig,
    native_config: NativeAccessConfig,
    native_io: Arc<dyn IoBackend>,
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
    let native = build_native_ctx(&config, native_config, native_io)?;
    spawn_prefetch_cells(
        access.clone(),
        compute,
        dataset,
        batch_source,
        gene_axis,
        config,
        native,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefetch_cells_scheduled_multi_by_gene_names<T, I, G>(
    access: &AccessHandle,
    compute: Arc<DataBankComputePool>,
    datasets: Arc<[Arc<Dataset>]>,
    batch_source: I,
    gene_names: &[G],
    missing: MissingGenePolicy,
    config: ScheduledPrefetchConfig,
    native_config: NativeAccessConfig,
    native_io: Arc<dyn IoBackend>,
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
    let native = build_native_ctx(&config, native_config, native_io)?;
    spawn_prefetch_cells_multi(
        access.clone(),
        compute,
        datasets,
        batch_source,
        gene_axes,
        config,
        native,
    )
}

/// Build the native scheduled context when the caller requested a non-disabled
/// native mode, otherwise `None`. The context (carrying any profile
/// `native_io_override`) must be constructed before `resolve_strategy` runs in
/// `spawn_prefetch_cells_multi` — see §5.9 migration point 2.
fn build_native_ctx(
    config: &ScheduledPrefetchConfig,
    native_config: NativeAccessConfig,
    native_io: Arc<dyn IoBackend>,
) -> DataBankResult<Option<NativeScheduledContext>> {
    if config.native_mode == NativeMode::Disabled {
        return Ok(None);
    }
    Ok(Some(NativeScheduledContext::new(native_io, native_config)?))
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
    native: Option<NativeScheduledContext>,
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
        native,
    )
}

pub(crate) fn spawn_prefetch_cells_multi<T, I>(
    access: AccessHandle,
    compute: Arc<DataBankComputePool>,
    datasets: Arc<[Arc<Dataset>]>,
    batch_source: I,
    gene_axes: MultiGeneAxisPlan,
    config: ScheduledPrefetchConfig,
    native: Option<NativeScheduledContext>,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: Iterator + Send + 'static,
    I::Item: Into<MultiBatchCells> + Send,
{
    // Resolve the access strategy once, at spawn time. `native` (the
    // `NativeScheduledContext`, including any profile `native_io_override`)
    // is constructed by the caller before this point — see §5.9 migration
    // point 2: the override must be injected into the context before
    // `resolve_strategy` runs, which is why the context is built upstream.
    let precondition = datasets.iter().all(|d| d.is_blosc_codec());
    let resolved = resolve_strategy(config.native_mode, native, precondition)?;
    let resolved_label = resolved.label;
    let resolved_reason = resolved.reason;
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
        strategy: resolved.strategy,
        projected_sparse_data_strategy: config.projected_sparse_data_strategy,
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
        resolved_strategy: resolved_label,
        fallback_reason: resolved_reason,
        cancel,
        producer: Some(handle),
    })
}

/// Resolve the actual execution strategy once, at spawn time.
///
/// `mode` is the caller-requested *policy*; the returned [`ResolvedStrategy`]
/// carries the *resolved* [`AccessStrategy`] the session runs, plus a
/// user-facing `label` and optional fallback `reason` for observability.
/// `precondition` is the dataset-level blosc contract
/// (`datasets.iter().all(Dataset::is_blosc_codec)`); it is a single
/// O(datasets) probe at spawn time, not a per-item hot-path check.
///
/// Semantics (row = (mode, native_ctx, precondition) → (strategy, reason)):
///   (Disabled, _,            _)          → Generic   reason=None  (not requested)
///   (Auto,    None,          _)          → Generic   reason=Some  (no fast context)
///   (Auto,    Some(ctx) if !ctx.config.enabled, _) → Generic  reason=Some (disabled in config)
///   (Auto,    Some(ctx),     true)       → BloscLz4Native(ctx)  reason=None
///   (Auto,    Some(_),       false)      → Generic   reason=Some  (contract unmet → safe retreat)
///   (Force,   None,          _)          → Err       // no fast context
///   (Force,   Some(ctx),     true)       → BloscLz4Native(ctx)  reason=None
///   (Force,   Some(_),       false)      → Err       // contract unmet → hard fail, no fallback
///
/// `reason` is `Some` only when the fast path was requested (`auto`/`force`)
/// but the session fell back to `Generic`; it is `None` when the fast path is
/// active, or when fast mode was not requested (`disabled`). The error strings
/// for the `Force` arms are user-facing (Python surface names the path
/// `fast_mode`); the Rust symbols remain `NativeMode`/`native_config` for
/// internal / doc consistency.
///
/// Once `BloscLz4Native` is resolved, the native worker runs with zero
/// fallback: a decode failure is a real error. `Auto` + contract violation
/// retreats to `Generic` at the strategy level (one spawn-time decision), not
/// per-item. `Force` + contract violation is a hard `InvalidConfig` error
/// raised at spawn rather than a per-item failure inside the worker.
pub(crate) fn resolve_strategy(
    mode: NativeMode,
    native_ctx: Option<NativeScheduledContext>,
    precondition: bool,
) -> DataBankResult<ResolvedStrategy> {
    use AccessStrategy::*;
    match (mode, native_ctx, precondition) {
        (NativeMode::Disabled, _, _) => Ok(ResolvedStrategy {
            strategy: Generic,
            label: ResolvedStrategy::GENERIC_LABEL,
            reason: None,
        }),
        (NativeMode::Auto, None, _) => Ok(ResolvedStrategy {
            strategy: Generic,
            label: ResolvedStrategy::GENERIC_LABEL,
            reason: Some("fast path not configured (no fast access context)"),
        }),
        (NativeMode::Auto, Some(ctx), _) if !ctx.config.enabled => Ok(ResolvedStrategy {
            strategy: Generic,
            label: ResolvedStrategy::GENERIC_LABEL,
            reason: Some("fast_config.enabled is False (set FastAccessConfig.fast() to enable)"),
        }),
        (NativeMode::Auto, Some(ctx), true) => Ok(ResolvedStrategy {
            strategy: BloscLz4Native(ctx),
            label: ResolvedStrategy::FAST_LABEL,
            reason: None,
        }),
        (NativeMode::Auto, Some(_), false) => Ok(ResolvedStrategy {
            strategy: Generic,
            label: ResolvedStrategy::GENERIC_LABEL,
            reason: Some("dataset is not fully blosc-lz4; fast path requires blosc-lz4 datasets"),
        }),
        (NativeMode::Force, None, _) => Err(DataBankError::InvalidConfig(
            "fast_mode='force' requires the fast blosc-lz4 access path to be enabled \
             (set DataBankConfig.fast_config = FastAccessConfig.fast())"
                .to_string(),
        )),
        (NativeMode::Force, Some(ctx), true) => Ok(ResolvedStrategy {
            strategy: BloscLz4Native(ctx),
            label: ResolvedStrategy::FAST_LABEL,
            reason: None,
        }),
        (NativeMode::Force, Some(_), false) => Err(DataBankError::InvalidConfig(
            "fast_mode='force' requires every dataset to use the blosc-lz4 codec; \
             at least one dataset uses a different codec"
                .to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{FileRef, IoBackend, IoTask};
    use std::sync::Arc;

    /// `IoBackend` stub that never services a read — `resolve_strategy` only
    /// inspects `NativeScheduledContext.config.enabled`, never submits IO, so
    /// the backend is never exercised by these tests.
    struct StubIo;

    impl IoBackend for StubIo {
        fn submit_read(&self, _file: FileRef, _offset: u64, _len: usize, _priority: u8) -> IoTask {
            unimplemented!("resolve_strategy tests never submit reads")
        }
    }

    fn native_ctx(enabled: bool) -> NativeScheduledContext {
        NativeScheduledContext::new(
            Arc::new(StubIo),
            NativeAccessConfig {
                enabled,
                ..NativeAccessConfig::default()
            },
        )
        .expect("native context")
    }

    fn assert_generic(resolved: &ResolvedStrategy) {
        assert!(
            matches!(resolved.strategy, AccessStrategy::Generic),
            "expected Generic, got native",
        );
        assert!(!resolved.strategy.is_native());
        assert_eq!(
            resolved.label,
            ResolvedStrategy::GENERIC_LABEL,
            "generic resolution must carry the generic label",
        );
    }

    fn assert_native(resolved: &ResolvedStrategy) {
        assert!(
            matches!(resolved.strategy, AccessStrategy::BloscLz4Native(_)),
            "expected BloscLz4Native, got Generic",
        );
        assert!(resolved.strategy.is_native());
        assert_eq!(
            resolved.label,
            ResolvedStrategy::FAST_LABEL,
            "native resolution must carry the fast label",
        );
        assert!(
            resolved.reason.is_none(),
            "native resolution must have no fallback reason",
        );
    }

    #[test]
    fn disabled_always_resolves_generic() {
        // (Disabled, _, _) → Generic regardless of context / precondition.
        // `disabled` did not request the fast path, so the fallback reason is
        // `None` (generic is the requested behavior, not a retreat).
        for precondition in [true, false] {
            let resolved = resolve_strategy(NativeMode::Disabled, None, precondition).unwrap();
            assert_generic(&resolved);
            assert!(resolved.reason.is_none());
        }
        let resolved =
            resolve_strategy(NativeMode::Disabled, Some(native_ctx(true)), true).unwrap();
        assert_generic(&resolved);
        assert!(resolved.reason.is_none());
    }

    #[test]
    fn auto_without_context_resolves_generic() {
        // (Auto, None, _) → Generic with a fallback reason (fast path requested
        // but no context was built).
        for precondition in [true, false] {
            let resolved = resolve_strategy(NativeMode::Auto, None, precondition).unwrap();
            assert_generic(&resolved);
            assert!(resolved.reason.is_some());
        }
    }

    #[test]
    fn auto_with_disabled_context_resolves_generic() {
        // (Auto, Some(ctx) if !ctx.config.enabled, _) → Generic even when the
        // blosc contract holds; reason explains the config gate.
        let resolved = resolve_strategy(NativeMode::Auto, Some(native_ctx(false)), true).unwrap();
        assert_generic(&resolved);
        assert!(resolved.reason.is_some());
        let resolved = resolve_strategy(NativeMode::Auto, Some(native_ctx(false)), false).unwrap();
        assert_generic(&resolved);
        assert!(resolved.reason.is_some());
    }

    #[test]
    fn auto_enabled_with_contract_resolves_native() {
        // (Auto, Some(ctx), true) → BloscLz4Native(ctx), no fallback reason.
        assert_native(&resolve_strategy(NativeMode::Auto, Some(native_ctx(true)), true).unwrap());
    }

    #[test]
    fn auto_enabled_without_contract_retreats_to_generic() {
        // (Auto, Some(_), false) → Generic (strategy-level safe retreat) with a
        // fallback reason explaining the blosc contract was unmet.
        let resolved = resolve_strategy(NativeMode::Auto, Some(native_ctx(true)), false).unwrap();
        assert_generic(&resolved);
        assert!(resolved.reason.is_some());
    }

    #[test]
    fn force_without_context_is_hard_error() {
        // (Force, None, _) → Err.
        assert!(resolve_strategy(NativeMode::Force, None, true).is_err());
        assert!(resolve_strategy(NativeMode::Force, None, false).is_err());
    }

    #[test]
    fn force_with_contract_resolves_native() {
        // (Force, Some(ctx), true) → BloscLz4Native(ctx).
        assert_native(&resolve_strategy(NativeMode::Force, Some(native_ctx(true)), true).unwrap());
    }

    #[test]
    fn force_without_contract_is_hard_error() {
        // (Force, Some(_), false) → Err (hard fail, no fallback).
        assert!(resolve_strategy(NativeMode::Force, Some(native_ctx(true)), false).is_err());
    }
}
