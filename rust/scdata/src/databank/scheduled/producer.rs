use std::collections::{BTreeMap, VecDeque};
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use crate::access::{AccessHandle, AccessItem, PrefetchCancel, ScheduledAccessConfig};
use crate::profile::ProfileTimer;

use super::super::array::DataValue;
use super::super::compute::{ComputeJob, DataBankComputePool};
use super::super::config::{NativeMode, ProjectedSparseDataGroupStrategy};
use super::super::dataset::Dataset;
use super::super::error::{DataBankError, DataBankResult};

use super::super::gene_axis::*;
use super::super::sparse::*;

use super::assemble::*;
use super::native_access::{AccessStrategy, NativeScheduledContext};
use super::planner::*;
use super::profile::*;
use super::types::*;

pub(crate) struct PrefetchProducer<T, I>
where
    T: DataValue,
    I: Iterator,
    I::Item: Into<MultiBatchCells>,
{
    pub(crate) access: AccessHandle,
    pub(crate) compute: Arc<DataBankComputePool>,
    pub(crate) datasets: Arc<[Arc<Dataset>]>,
    pub(crate) batch_source: I,
    pub(crate) access_config: ScheduledAccessConfig,
    pub(crate) native_mode: NativeMode,
    pub(crate) native: Option<NativeScheduledContext>,
    pub(crate) projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    pub(crate) gene_axes: Arc<MultiGeneAxisPlan>,
    pub(crate) tx: flume::Sender<DataBankResult<PrefetchedBatch<T>>>,
    pub(crate) cancel: Arc<PrefetchCancelRegistry>,
    pub(crate) prefetch_step: usize,
    pub(crate) profiler: ScheduledPrefetchProfiler,
}

pub(crate) struct ProducerState<T>
where
    T: DataValue,
{
    next_read_seq: BatchSeq,
    next_emit_seq: BatchSeq,
    source_done: bool,
    stop_reading: bool,
    outstanding: usize,
    active_requests: usize,
    active_responses: usize,
    response_limit: usize,
    planned_ready: VecDeque<PlannedBatch>,
    completed: CompletedQueue<T>,
}

impl<T> ProducerState<T>
where
    T: DataValue,
{
    fn new(prefetch_step: usize, worker_count: usize) -> Self {
        let response_limit = scheduled_response_limit(prefetch_step, worker_count);
        Self {
            next_read_seq: 0,
            next_emit_seq: 0,
            source_done: false,
            stop_reading: false,
            outstanding: 0,
            active_requests: 0,
            active_responses: 0,
            response_limit,
            planned_ready: VecDeque::new(),
            completed: CompletedQueue::with_capacity(prefetch_step),
        }
    }

    fn is_finished(&self) -> bool {
        self.source_done
            && self.outstanding == 0
            && self.active_requests == 0
            && self.active_responses == 0
            && self.planned_ready.is_empty()
    }
}

fn scheduled_response_limit(prefetch_step: usize, worker_count: usize) -> usize {
    let default_limit = prefetch_step.min(worker_count.saturating_sub(1).max(1));
    std::env::var("SCDATA_SCHEDULED_RESPONSE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .map(|value| value.min(prefetch_step.max(1)).min(worker_count.max(1)))
        .unwrap_or(default_limit)
}

struct CompletedQueue<T>
where
    T: DataValue,
{
    entries: CompletedEntries<T>,
}

enum CompletedEntries<T>
where
    T: DataValue,
{
    Small(Vec<(BatchSeq, DataBankResult<PrefetchedBatch<T>>)>),
    Large(BTreeMap<BatchSeq, DataBankResult<PrefetchedBatch<T>>>),
}

impl<T> CompletedQueue<T>
where
    T: DataValue,
{
    const SMALL_WINDOW_LIMIT: usize = 32;

    fn with_capacity(capacity: usize) -> Self {
        let entries = if capacity <= Self::SMALL_WINDOW_LIMIT {
            CompletedEntries::Small(Vec::with_capacity(capacity))
        } else {
            CompletedEntries::Large(BTreeMap::new())
        };
        Self { entries }
    }

    fn insert(&mut self, seq: BatchSeq, result: DataBankResult<PrefetchedBatch<T>>) {
        match &mut self.entries {
            CompletedEntries::Small(entries) => entries.push((seq, result)),
            CompletedEntries::Large(entries) => {
                entries.insert(seq, result);
            }
        }
    }

    fn remove(&mut self, seq: BatchSeq) -> Option<DataBankResult<PrefetchedBatch<T>>> {
        match &mut self.entries {
            CompletedEntries::Small(entries) => {
                let pos = entries
                    .iter()
                    .position(|(candidate, _)| *candidate == seq)?;
                Some(entries.swap_remove(pos).1)
            }
            CompletedEntries::Large(entries) => entries.remove(&seq),
        }
    }
}

pub(crate) enum ProducerEvent<T>
where
    T: DataValue,
{
    Planned(Result<PlannedMessage, flume::RecvError>),
    Done(Result<DoneMessage<T>, flume::RecvError>),
}

pub(crate) struct ActiveBatchGuard {
    seq: BatchSeq,
    registry: Arc<PrefetchCancelRegistry>,
}

impl Drop for ActiveBatchGuard {
    fn drop(&mut self) {
        self.registry.unregister(self.seq);
    }
}

impl<T, I> PrefetchProducer<T, I>
where
    T: DataValue,
    I: Iterator,
    I::Item: Into<MultiBatchCells>,
{
    pub(crate) fn run(mut self) {
        let profile_round = if let Some((round, profiler)) = self.profiler.begin_round() {
            self.profiler = profiler;
            Some(round)
        } else {
            None
        };
        if panic::catch_unwind(AssertUnwindSafe(|| self.run_pipeline())).is_err() {
            self.cancel.cancel_all();
            let _ = self.tx.send(Err(DataBankError::PrefetchProducerPanic));
        }
        if let Some(round) = profile_round {
            self.profiler.print_summary(round.end());
        }
    }

    fn run_pipeline(&mut self) {
        let channel_capacity = self.prefetch_step.max(1);
        let (planned_tx, planned_rx) = flume::bounded(channel_capacity);
        let (done_tx, done_rx) = flume::bounded(channel_capacity);
        let mut state = ProducerState::<T>::new(self.prefetch_step, self.compute.worker_count());

        loop {
            if self.cancel.is_cancelled() {
                self.profiler.inc_cancelled();
                break;
            }

            let mut progressed = false;
            progressed |= self.fill_request_window(&mut state, &planned_tx);
            progressed |= self.drain_messages(&mut state, &planned_rx, &done_rx);
            progressed |= self.submit_ready_responses(&mut state, &done_tx);
            let (keep_running, emitted) = self.emit_ready(&mut state);
            progressed |= emitted;
            if !keep_running {
                break;
            }

            if state.is_finished() {
                break;
            }

            if !progressed && !self.wait_for_event(&mut state, &planned_rx, &done_rx) {
                break;
            }
        }

        self.cancel.cancel_all();
        state.planned_ready.clear();
    }

    fn fill_request_window(
        &mut self,
        state: &mut ProducerState<T>,
        planned_tx: &flume::Sender<PlannedMessage>,
    ) -> bool {
        let mut progressed = false;
        while !state.source_done
            && !state.stop_reading
            && state.outstanding < self.prefetch_step
            && !self.cancel.is_cancelled()
        {
            let next_started = self.profiler.start_batch_source_next();
            let next = self.batch_source.next();
            self.profiler.record_batch_source_next(next_started);
            let Some(cells) = next else {
                state.source_done = true;
                self.profiler.inc_source_exhausted();
                progressed = true;
                break;
            };
            let batch = cells.into();
            self.profiler
                .record_source_batch(batch.total_cells().unwrap_or(usize::MAX));
            let seq = state.next_read_seq;
            state.next_read_seq += 1;
            state.outstanding += 1;
            state.active_requests += 1;

            let job = make_prefetch_request_job(
                seq,
                self.access.clone(),
                Arc::clone(&self.datasets),
                batch,
                Arc::clone(&self.gene_axes),
                self.access_config,
                self.native_mode,
                self.native.clone(),
                self.projected_sparse_data_strategy,
                Arc::clone(&self.cancel),
                planned_tx.clone(),
                self.profiler.clone(),
                self.profiler.start_request_queue_wait(),
            );
            let submit_started = self.profiler.start_submit_request();
            let submit_result = self.compute.submit_request(job);
            self.profiler.record_submit_request(submit_started);
            if let Err(err) = submit_result {
                self.profiler.inc_submit_request_error();
                state.active_requests = state.active_requests.saturating_sub(1);
                state.completed.insert(seq, Err(err));
                state.stop_reading = true;
            }
            progressed = true;
        }
        progressed
    }

    fn drain_messages(
        &self,
        state: &mut ProducerState<T>,
        planned_rx: &flume::Receiver<PlannedMessage>,
        done_rx: &flume::Receiver<DoneMessage<T>>,
    ) -> bool {
        let mut progressed = false;
        while let Ok(message) = planned_rx.try_recv() {
            self.handle_planned_message(state, message);
            progressed = true;
        }
        while let Ok(message) = done_rx.try_recv() {
            self.handle_done_message(state, message);
            progressed = true;
        }
        progressed
    }

    fn wait_for_event(
        &self,
        state: &mut ProducerState<T>,
        planned_rx: &flume::Receiver<PlannedMessage>,
        done_rx: &flume::Receiver<DoneMessage<T>>,
    ) -> bool {
        if state.active_requests == 0 && state.active_responses == 0 {
            return false;
        }

        let wait_started = self.profiler.start_coordinator_wait();
        let event = match (state.active_requests > 0, state.active_responses > 0) {
            (true, true) => flume::Selector::new()
                .recv(planned_rx, ProducerEvent::Planned)
                .recv(done_rx, ProducerEvent::Done)
                .wait(),
            (true, false) => ProducerEvent::Planned(planned_rx.recv()),
            (false, true) => ProducerEvent::Done(done_rx.recv()),
            (false, false) => return false,
        };
        self.profiler.record_coordinator_wait(wait_started);

        match event {
            ProducerEvent::Planned(Ok(message)) => {
                self.handle_planned_message(state, message);
                true
            }
            ProducerEvent::Done(Ok(message)) => {
                self.handle_done_message(state, message);
                true
            }
            ProducerEvent::Planned(Err(_)) | ProducerEvent::Done(Err(_)) => false,
        }
    }

    fn handle_planned_message(&self, state: &mut ProducerState<T>, message: PlannedMessage) {
        state.active_requests = state.active_requests.saturating_sub(1);
        match message.result {
            Ok(planned) => {
                if self.cancel.is_cancelled() {
                    self.cancel.unregister(planned.seq);
                } else {
                    state.planned_ready.push_back(*planned);
                }
            }
            Err(err) => {
                if !matches!(err, DataBankError::PrefetchCancelled) || !self.cancel.is_cancelled() {
                    state.stop_reading = true;
                }
                state.completed.insert(message.seq, Err(err));
            }
        }
    }

    fn handle_done_message(&self, state: &mut ProducerState<T>, message: DoneMessage<T>) {
        state.active_responses = state.active_responses.saturating_sub(1);
        if message.result.is_err()
            && (!matches!(&message.result, Err(DataBankError::PrefetchCancelled))
                || !self.cancel.is_cancelled())
        {
            state.stop_reading = true;
        }
        state.completed.insert(message.seq, message.result);
    }

    fn submit_ready_responses(
        &self,
        state: &mut ProducerState<T>,
        done_tx: &flume::Sender<DoneMessage<T>>,
    ) -> bool {
        let mut progressed = false;
        while state.active_responses < state.response_limit && !self.cancel.is_cancelled() {
            let Some(planned) = state.planned_ready.pop_front() else {
                break;
            };
            let seq = planned.seq;
            state.active_responses += 1;
            let job = make_prefetch_response_job(
                planned,
                self.access.clone(),
                Arc::clone(&self.compute),
                self.access_config,
                self.projected_sparse_data_strategy,
                Arc::clone(&self.gene_axes),
                Arc::clone(&self.cancel),
                done_tx.clone(),
                self.profiler.clone(),
                self.profiler.start_response_queue_wait(),
            );
            let submit_started = self.profiler.start_submit_response();
            let submit_result = self.compute.submit_response(job);
            self.profiler.record_submit_response(submit_started);
            if let Err(err) = submit_result {
                self.profiler.inc_submit_response_error();
                state.active_responses = state.active_responses.saturating_sub(1);
                self.cancel.unregister(seq);
                state.completed.insert(seq, Err(err));
                state.stop_reading = true;
            }
            progressed = true;
        }
        progressed
    }

    fn emit_ready(&self, state: &mut ProducerState<T>) -> (bool, bool) {
        let mut emitted = false;
        while let Some(result) = state.completed.remove(state.next_emit_seq) {
            emitted = true;
            state.outstanding = state.outstanding.saturating_sub(1);
            state.next_emit_seq += 1;
            match result {
                Ok(batch) => {
                    let send_started = self.profiler.start_output_send();
                    let send_result = self.tx.send(Ok(batch));
                    self.profiler.record_output_send(send_started);
                    if send_result.is_err() {
                        self.profiler.inc_output_dropped();
                        self.cancel.cancel_all();
                        return (false, emitted);
                    }
                    self.profiler.inc_emitted_batch();
                }
                Err(DataBankError::PrefetchCancelled) if self.cancel.is_cancelled() => {
                    self.profiler.inc_cancelled();
                    return (false, emitted);
                }
                Err(err) => {
                    let send_started = self.profiler.start_output_send();
                    if self.tx.send(Err(err)).is_ok() {
                        self.profiler.inc_emitted_batch();
                        self.profiler.inc_emitted_error();
                    } else {
                        self.profiler.inc_output_dropped();
                    }
                    self.profiler.record_output_send(send_started);
                    self.cancel.cancel_all();
                    return (false, emitted);
                }
            }
        }
        (true, emitted)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn make_prefetch_request_job(
    seq: BatchSeq,
    access: AccessHandle,
    datasets: Arc<[Arc<Dataset>]>,
    batch: MultiBatchCells,
    gene_axes: Arc<MultiGeneAxisPlan>,
    access_config: ScheduledAccessConfig,
    native_mode: NativeMode,
    native: Option<NativeScheduledContext>,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    registry: Arc<PrefetchCancelRegistry>,
    planned_tx: flume::Sender<PlannedMessage>,
    profiler: ScheduledPrefetchProfiler,
    queued_at: ProfileTimer,
) -> ComputeJob {
    Box::new(move || {
        profiler.inc_request_job();
        profiler.record_request_queue_wait(queued_at);
        let total_started = profiler.start_request_total();
        let result =
            panic::catch_unwind(AssertUnwindSafe(|| -> DataBankResult<Box<PlannedBatch>> {
                if registry.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
                let plan_started = profiler.start_request_plan();
                let planned = plan_batch_multi(
                    datasets.as_ref(),
                    batch,
                    gene_axes.as_ref(),
                    projected_sparse_data_strategy,
                );
                profiler.record_request_plan(plan_started);
                let (mut plan, mut items) = planned?;
                if registry.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
                let cancel = PrefetchCancel::new(access.clone());
                preplan_selected_sparse_request(
                    &access,
                    native.clone(),
                    native_mode,
                    access_config,
                    projected_sparse_data_strategy,
                    gene_axes.as_ref(),
                    Arc::clone(&cancel),
                    &profiler,
                    &mut plan,
                    &mut items,
                )?;
                if registry.is_cancelled() || cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
                let schedule_started = profiler.start_request_schedule();
                let native_for_response = native.clone();
                // Transitional: construct the strategy from (native_mode,
                // native) and delegate to AccessStrategy::build. Step 3
                // replaces this with a single resolve_strategy() at spawn.
                let strategy = AccessStrategy::from_mode_and_ctx(native_mode, native)?;
                let scheduled_result = strategy.build(
                    access.clone(),
                    items,
                    access_config,
                    native_mode,
                    Arc::clone(&cancel),
                    false,
                );
                profiler.record_request_schedule(schedule_started);
                let scheduled = scheduled_result?;
                registry.register(seq, Arc::clone(&cancel));
                Ok(Box::new(PlannedBatch {
                    seq,
                    plan,
                    scheduled,
                    native: native_for_response,
                    native_mode,
                    cancel,
                }))
            }))
            .unwrap_or(Err(DataBankError::ComputeWorkerPanic));
        if result.is_err() {
            profiler.inc_request_error();
        }
        let send_started = profiler.start_request_send();
        if let Err(err) = planned_tx.send(PlannedMessage { seq, result }) {
            if let Ok(planned) = err.0.result {
                registry.unregister(planned.seq);
            }
        }
        profiler.record_request_send(send_started);
        profiler.record_request_total(total_started);
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn preplan_selected_sparse_request(
    access: &AccessHandle,
    native: Option<NativeScheduledContext>,
    native_mode: NativeMode,
    access_config: ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axes: &MultiGeneAxisPlan,
    cancel: Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    plan: &mut BatchPlan,
    items: &mut Vec<AccessItem>,
) -> DataBankResult<()> {
    if !preplan_selected_sparse_enabled()
        || projected_sparse_data_strategy != ProjectedSparseDataGroupStrategy::SelectedOnly
    {
        return Ok(());
    }

    let mut changed = false;
    match plan {
        BatchPlan::Single {
            dataset_idx, plan, ..
        } => {
            changed |= preplan_single_selected_sparse_request(
                access,
                native.clone(),
                native_mode,
                access_config,
                projected_sparse_data_strategy,
                gene_axes.axis_for(*dataset_idx)?,
                Arc::clone(&cancel),
                profiler,
                plan,
            )?;
        }
        BatchPlan::Multi(multi) => {
            for part in &mut multi.parts {
                changed |= preplan_single_selected_sparse_request(
                    access,
                    native.clone(),
                    native_mode,
                    access_config,
                    projected_sparse_data_strategy,
                    &part.gene_axis,
                    Arc::clone(&cancel),
                    profiler,
                    &mut part.plan,
                )?;
                if cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
            }
        }
    }
    if changed {
        *items = batch_plan_file_access_items(plan)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn preplan_single_selected_sparse_request(
    access: &AccessHandle,
    native: Option<NativeScheduledContext>,
    native_mode: NativeMode,
    access_config: ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axis: &GeneAxisPlan,
    cancel: Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    plan: &mut SingleDatasetPlan,
) -> DataBankResult<bool> {
    let SingleDatasetPlan::Sparse {
        plan: sparse_plan,
        dataset,
        preloaded_index_bytes,
        selected_data_scheduled,
        ..
    } = plan
    else {
        return Ok(false);
    };
    if *selected_data_scheduled || preloaded_index_bytes.is_some() || cancel.is_cancelled() {
        return Ok(false);
    }
    if gene_axis.projection().is_none()
        || should_read_all_small_projected_sparse_plan(
            projected_sparse_data_strategy,
            true,
            sparse_plan,
        )
    {
        return Ok(false);
    }
    let Dataset::SparseCsr(dataset) = dataset.as_ref() else {
        return Ok(false);
    };

    let index_items = sparse_plan_index_file_access_items(sparse_plan)?;
    let index_strategy = AccessStrategy::from_mode_and_ctx(native_mode, native.clone())?;
    let mut index_scheduled = index_strategy.build(
        access.clone(),
        index_items,
        access_config,
        native_mode,
        Arc::clone(&cancel),
        false,
    )?;
    let index_bytes =
        load_sparse_prefetch_indices(access, &cancel, profiler, sparse_plan, &mut index_scheduled)?;
    if let Some(extra) = index_scheduled.next() {
        return match extra {
            Ok(_) => Err(DataBankError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "scheduled CSR index preplan returned extra output",
            ))),
            Err(err) => Err(DataBankError::Io(err)),
        };
    }
    if cancel.is_cancelled() {
        return Err(DataBankError::PrefetchCancelled);
    }

    let selected_plan =
        plan_sparse_selected_data_batch(dataset, sparse_plan, index_bytes.as_ref(), gene_axis)?;
    let defer_selected_data = preplan_selected_sparse_defer_data_enabled()
        && can_defer_selected_sparse_data_to_response(native.as_ref(), native_mode);
    *sparse_plan = selected_plan;
    *preloaded_index_bytes = Some(Arc::from(index_bytes.into_boxed_slice()));
    *selected_data_scheduled = !defer_selected_data;
    Ok(true)
}

fn batch_plan_file_access_items(plan: &BatchPlan) -> DataBankResult<Vec<AccessItem>> {
    let mut items = Vec::new();
    match plan {
        BatchPlan::Single { plan, .. } => append_single_plan_file_access_items(plan, &mut items)?,
        BatchPlan::Multi(multi) => {
            for part in &multi.parts {
                append_single_plan_file_access_items(&part.plan, &mut items)?;
            }
        }
    }
    Ok(items)
}

fn append_single_plan_file_access_items(
    plan: &SingleDatasetPlan,
    items: &mut Vec<AccessItem>,
) -> DataBankResult<()> {
    match plan {
        SingleDatasetPlan::Dense { groups, .. } => {
            items.append(&mut dense_group_access_items(groups)?);
        }
        SingleDatasetPlan::Sparse {
            plan,
            preloaded_index_bytes,
            selected_data_scheduled,
            ..
        } => {
            if !selected_sparse_data_deferred_to_response(
                plan,
                preloaded_index_bytes,
                *selected_data_scheduled,
            ) {
                items.append(&mut sparse_plan_file_access_items(plan)?);
            }
        }
    }
    Ok(())
}

fn selected_sparse_data_deferred_to_response(
    plan: &SparseBatchPlan,
    preloaded_index_bytes: &Option<Arc<[u8]>>,
    selected_data_scheduled: bool,
) -> bool {
    preloaded_index_bytes.is_some()
        && !selected_data_scheduled
        && plan.index_groups.is_empty()
        && plan.index_pieces.is_empty()
}

fn preplan_selected_sparse_enabled() -> bool {
    std::env::var("SCDATA_PREPLAN_SELECTED_SPARSE")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off"))
        .unwrap_or(true)
}

fn preplan_selected_sparse_defer_data_enabled() -> bool {
    std::env::var("SCDATA_PREPLAN_SELECTED_SPARSE_DEFER_DATA")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn can_defer_selected_sparse_data_to_response(
    native: Option<&NativeScheduledContext>,
    native_mode: NativeMode,
) -> bool {
    match (native_mode, native) {
        (NativeMode::Disabled, _) | (NativeMode::Force, None) | (NativeMode::Auto, None) => false,
        (NativeMode::Auto, Some(native)) => native.config.enabled,
        (NativeMode::Force, Some(_)) => true,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn make_prefetch_response_job<T>(
    planned: PlannedBatch,
    access: AccessHandle,
    compute: Arc<DataBankComputePool>,
    access_config: ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    gene_axes: Arc<MultiGeneAxisPlan>,
    registry: Arc<PrefetchCancelRegistry>,
    done_tx: flume::Sender<DoneMessage<T>>,
    profiler: ScheduledPrefetchProfiler,
    queued_at: ProfileTimer,
) -> ComputeJob
where
    T: DataValue,
{
    Box::new(move || {
        profiler.inc_response_job();
        profiler.record_response_queue_wait(queued_at);
        let total_started = profiler.start_response_total();
        let PlannedBatch {
            seq,
            plan,
            mut scheduled,
            native,
            native_mode,
            cancel,
        } = planned;
        let _guard = ActiveBatchGuard {
            seq,
            registry: Arc::clone(&registry),
        };
        let result = panic::catch_unwind(AssertUnwindSafe(
            || -> DataBankResult<PrefetchedBatch<T>> {
                if registry.is_cancelled() || cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
                let batch = assemble_planned_batch(
                    &access,
                    compute.as_ref(),
                    &access_config,
                    projected_sparse_data_strategy,
                    gene_axes.as_ref(),
                    &cancel,
                    &profiler,
                    native.as_ref(),
                    native_mode,
                    plan,
                    &mut scheduled,
                )?;
                if scheduled.next().is_some() {
                    return Err(DataBankError::Io(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "scheduled prefetch returned extra output",
                    )));
                }
                Ok(batch)
            },
        ))
        .unwrap_or(Err(DataBankError::ComputeWorkerPanic));
        if result.is_err() {
            profiler.inc_response_error();
        }
        let _ = done_tx.send(DoneMessage { seq, result });
        profiler.record_response_total(total_started);
        Ok(())
    })
}
