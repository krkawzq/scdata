use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::access::{AccessItem, PrefetchCancel, ScheduledAccess};

use super::super::array::{DType, DataValue};
use super::super::dataset::Dataset;
use super::super::error::DataBankResult;
use super::super::interner::GeneNameView;
use super::super::plan::DenseSegment;

use super::super::dense::*;
use super::super::gene_axis::*;
use super::super::sparse::*;

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
pub struct PrefetchCells<T>
where
    T: DataValue,
{
    pub(crate) rx: Option<flume::Receiver<DataBankResult<PrefetchedBatch<T>>>>,
    pub(crate) output_names: Vec<GeneNameView>,
    pub(crate) _datasets: Arc<[Arc<Dataset>]>,
    pub(crate) prefetch_step: usize,
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
}

impl<T> Iterator for PrefetchCells<T>
where
    T: DataValue,
{
    type Item = DataBankResult<PrefetchedBatch<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let rx = self.rx.as_ref()?;
        match rx.recv() {
            Ok(batch) => Some(batch),
            Err(_) => {
                if let Some(handle) = self.producer.take() {
                    let _ = handle.join();
                }
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
        self.cancel.cancel_all();
        self.rx.take();
        if let Some(handle) = self.producer.take() {
            let _ = handle.join();
        }
    }
}

pub(crate) type BatchSeq = u64;
pub(crate) type ScheduledBatchAccess = ScheduledAccess<std::vec::IntoIter<AccessItem>>;

pub(crate) struct PlannedBatch {
    pub(crate) seq: BatchSeq,
    pub(crate) plan: BatchPlan,
    pub(crate) scheduled: ScheduledBatchAccess,
    pub(crate) cancel: Arc<PrefetchCancel>,
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
}

impl PrefetchCancelRegistry {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            cancelled: AtomicBool::new(false),
            active: Mutex::new(BTreeMap::new()),
        })
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
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
