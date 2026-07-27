use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::access::PrefetchCancel;

use super::super::array::{DType, DataValue};
use super::super::config::{ProjectedSparseDataGroupStrategy, SmallProjectedSparsePolicy};
use super::super::dataset::Dataset;
use super::super::error::DataBankResult;
use super::super::interner::GeneNameView;
use super::super::plan::DenseSegment;
use super::super::RetiredDatasets;

use super::super::dense::*;
use super::super::gene_axis::*;
use super::super::sparse::*;
use super::native_access::{AccessStrategy, ScheduledBatchAccess};

const PROJECTED_SPARSE_READ_ALL_SMALL_DATA_GROUPS: usize = 8;

pub(crate) fn should_read_all_small_projected_sparse_plan(
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    small_projected_sparse_policy: SmallProjectedSparsePolicy,
    has_projection: bool,
    plan: &SparseBatchPlan,
) -> bool {
    if small_projected_sparse_policy == SmallProjectedSparsePolicy::SelectedOnly {
        return false;
    }
    has_projection
        && projected_sparse_data_strategy == ProjectedSparseDataGroupStrategy::SelectedOnly
        && !plan.data_groups.is_empty()
        && plan.data_groups.len() <= PROJECTED_SPARSE_READ_ALL_SMALL_DATA_GROUPS
        && sparse_plan_output_rows_are_compact(plan)
}

fn sparse_plan_output_rows_are_compact(plan: &SparseBatchPlan) -> bool {
    let mut distinct_rows = 0usize;
    let mut min_row = usize::MAX;
    let mut max_row = 0usize;
    let mut last_row = None;

    for piece in &plan.data_pieces {
        let row = piece.output_row;
        if last_row != Some(row) {
            distinct_rows += 1;
            last_row = Some(row);
            min_row = min_row.min(row);
            max_row = max_row.max(row);
        }
    }

    if distinct_rows <= 1 {
        return false;
    }
    let span = max_row.saturating_sub(min_row).saturating_add(1);
    span <= distinct_rows.saturating_mul(2)
}

// ---------------------------------------------------------------------------
// Scheduled prefetch
// ---------------------------------------------------------------------------

/// The per-batch plan produced by the scheduled prefetcher.
///
/// Each batch is planned independently (chunks are not merged across batches).
/// The plan carries both the scatter metadata needed to assemble the decoded
/// bytes into a row-major output buffer and the ordered list of access items
/// that the access scheduler consumes.
pub(crate) enum BatchPlan {
    Single {
        dataset_idx: usize,
        cells: Vec<usize>,
        plan: SingleDatasetPlan,
    },
    Multi(MultiDatasetPlan),
}

pub(crate) enum SingleDatasetPlan {
    Dense {
        active_rows: usize,
        segments: Vec<DenseSegment>,
        groups: Vec<DenseReadGroup>,
        num_genes: usize,
        src_dtype: DType,
    },
    Sparse {
        active_rows: usize,
        plan: SparseBatchPlan,
        dataset: Arc<Dataset>,
        preloaded_index_bytes: Option<Arc<[u8]>>,
        selected_data_scheduled: bool,
    },
}

pub(crate) struct MultiDatasetPlan {
    pub(crate) output_cells: Vec<usize>,
    pub(crate) parts: Vec<MultiBatchPlanPart>,
    pub(crate) total_cells: usize,
    pub(crate) output_genes: usize,
}

pub(crate) struct MultiBatchPlanPart {
    pub(crate) dataset_idx: usize,
    pub(crate) gene_axis: GeneAxisPlan,
    pub(crate) active_rows: usize,
    pub(crate) plan: SingleDatasetPlan,
}

pub(crate) struct BatchRows {
    pub(crate) dataset_idx: usize,
    pub(crate) cells: Vec<usize>,
    pub(crate) output_rows: Vec<usize>,
}

pub(crate) struct MultiBatchLayout {
    pub(crate) output_cells: Vec<usize>,
    pub(crate) per_dataset: Vec<BatchRows>,
}

/// A prefetched batch: the cell indices and the databank-allocated,
/// already-scattered row-major buffer (`cells.len() * num_genes` values).
#[derive(Debug)]
pub struct PrefetchedBatch<T>
where
    T: DataValue,
{
    pub cells: Vec<usize>,
    pub buffer: Vec<T>,
    pub num_genes: usize,
}

/// Blocking iterator over scheduled prefetch results.
///
/// Accepts a user iterator yielding one batch of cell indices at a time. Each
/// batch is planned independently and its access items are streamed into the
/// access scheduler's [`ScheduledAccess`], which provides the chunk-level
/// look-ahead (`prefetch_step`, `decode_ahead_steps`, etc.). The databank-level
/// look-ahead is [`Self::prefetch_step`]: a background producer keeps a bounded
/// completed queue of decoded batches ahead of the consumer.
///
/// The databank iterator (batches) and the access iterator (chunk groups) are
/// deliberately not aligned: one batch expands to a variable number of chunk
/// groups, so the driver tracks how many `scheduled.next()` calls each batch
/// requires via its plan.
///
/// Results are cached in the completed queue, so no external output buffer is
/// accepted.
///
/// `next()` retains normal blocking iterator semantics and may wait for the
/// batch source to yield. `close()` is different: source advancement runs on a
/// detached forwarder, because arbitrary Rust `Iterator::next()` calls cannot
/// be interrupted safely. Thus `close()` never joins a source blocked forever;
/// when that call eventually returns, the forwarder observes cancellation and
/// exits without submitting another batch.
pub struct PrefetchCells<T>
where
    T: DataValue,
{
    pub(crate) rx: Option<flume::Receiver<DataBankResult<PrefetchedBatch<T>>>>,
    pub(crate) output_names: Vec<GeneNameView>,
    pub(crate) _datasets: Option<Arc<[Arc<Dataset>]>>,
    pub(crate) retired: Arc<RetiredDatasets>,
    pub(crate) prefetch_step: usize,
    pub(crate) resolved_strategy: &'static str,
    pub(crate) fallback_reason: Option<&'static str>,
    pub(crate) cancel: Arc<PrefetchCancelRegistry>,
    pub(crate) producer: Option<thread::JoinHandle<()>>,
}

impl<T> PrefetchCells<T>
where
    T: DataValue,
{
    /// Configured completed-queue depth (number of decoded batches kept ahead
    /// of the consumer).
    pub fn prefetch_step(&self) -> usize {
        self.prefetch_step
    }

    pub fn gene_names(&self) -> &[GeneNameView] {
        &self.output_names
    }

    /// Stable short name of the resolved access strategy for this session:
    /// `"blosc_lz4_fast"` (the native Blosc-LZ4 fast path engaged) or
    /// `"generic"` (the standard access-scheduler path).
    pub fn resolved_strategy(&self) -> &'static str {
        self.resolved_strategy
    }

    /// Why the fast path fell back to `generic`, when it was requested
    /// (`fast_mode` = `auto`/`force`) but did not engage. `None` when the fast
    /// path is active, or when fast mode was not requested.
    pub fn fallback_reason(&self) -> Option<&'static str> {
        self.fallback_reason
    }

    /// Cancel outstanding work, join the producer, and release retained
    /// datasets.  Safe to call repeatedly and also used by `Drop`.
    pub fn close(&mut self) {
        self.cancel.cancel_all();
        self.rx.take();
        if let Some(handle) = self.producer.take() {
            let _ = handle.join();
        }
        drop(self._datasets.take());
        // A file-release failure cannot be reported from `Drop`; the normal
        // DataBank cleanup path preserves its existing error reporting.
        let _ = self.retired.cleanup();
    }
}

impl<T> Iterator for PrefetchCells<T>
where
    T: DataValue,
{
    type Item = DataBankResult<PrefetchedBatch<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let received = self.rx.as_ref()?.recv();
        match received {
            Ok(batch) => {
                if batch.is_err() {
                    // Errors are terminal by producer contract.  Release the
                    // queue, worker, and retained datasets before surfacing
                    // the original error to the consumer.
                    self.close();
                }
                Some(batch)
            }
            Err(_) => {
                self.close();
                None
            }
        }
    }
}

impl<T> Drop for PrefetchCells<T>
where
    T: DataValue,
{
    fn drop(&mut self) {
        self.close();
    }
}

pub(crate) type BatchSeq = u64;
pub(crate) struct PlannedBatch {
    pub(crate) seq: BatchSeq,
    pub(crate) plan: BatchPlan,
    pub(crate) scheduled: ScheduledBatchAccess,
    pub(crate) strategy: AccessStrategy,
    pub(crate) cancel: Arc<PrefetchCancel>,
    /// Registration is created before preplanning and is moved to the response
    /// closure on success. Its Drop unregisters all non-success paths.
    pub(crate) registration: super::producer::ActiveBatchGuard,
}

pub(crate) struct PlannedMessage {
    pub(crate) seq: BatchSeq,
    pub(crate) result: DataBankResult<Box<PlannedBatch>>,
}

pub(crate) struct DoneMessage<T>
where
    T: DataValue,
{
    pub(crate) seq: BatchSeq,
    pub(crate) result: DataBankResult<PrefetchedBatch<T>>,
}

#[derive(Debug)]
pub(crate) struct PrefetchCancelRegistry {
    cancelled: AtomicBool,
    active: Mutex<BTreeMap<BatchSeq, Arc<PrefetchCancel>>>,
    /// Wakes the producer when it is waiting only for a slow or permanently
    /// blocking external batch source.
    cancel_tx: flume::Sender<()>,
    cancel_rx: flume::Receiver<()>,
}

impl PrefetchCancelRegistry {
    pub(crate) fn new() -> Arc<Self> {
        let (cancel_tx, cancel_rx) = flume::bounded(1);
        Arc::new(Self {
            cancelled: AtomicBool::new(false),
            active: Mutex::new(BTreeMap::new()),
            cancel_tx,
            cancel_rx,
        })
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub(crate) fn cancel_receiver(&self) -> flume::Receiver<()> {
        self.cancel_rx.clone()
    }

    pub(crate) fn register(&self, seq: BatchSeq, cancel: Arc<PrefetchCancel>) {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.cancelled.load(Ordering::Acquire) {
            drop(active);
            cancel.cancel_in_flight();
        } else {
            active.insert(seq, cancel);
        }
    }

    pub(crate) fn unregister(&self, seq: BatchSeq) {
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&seq);
    }

    pub(crate) fn cancel_all(&self) {
        self.cancelled.store(true, Ordering::Release);
        let _ = self.cancel_tx.try_send(());
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let cancels = active.values().cloned().collect::<Vec<_>>();
        active.clear();
        drop(active);
        for cancel in cancels {
            cancel.cancel_in_flight();
        }
    }
}
