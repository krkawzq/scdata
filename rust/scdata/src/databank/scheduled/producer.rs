use std::collections::{BTreeMap, VecDeque};
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;

use tokio::sync::Semaphore;

use crate::access::{AccessHandle, AccessItem, PrefetchCancel, ScheduledAccessConfig};
use crate::env::fastpath;
use crate::profile::ProfileTimer;

use super::super::array::DataValue;
use super::super::compute::{ComputeJob, DataBankComputePool};
use super::super::config::{ProjectedSparseDataGroupStrategy, SmallProjectedSparsePolicy};
use super::super::dataset::Dataset;
use super::super::error::{DataBankError, DataBankResult};
use super::super::RetiredCleanupGuard;

use super::super::gene_axis::*;
use super::super::sparse::*;

use super::assemble::*;
use super::native_access::AccessStrategy;
use super::planner::*;
use super::profile::*;
use super::types::*;

pub(crate) struct PrefetchProducer<T>
where
    T: DataValue,
{
    pub(crate) access: AccessHandle,
    pub(crate) compute: Arc<DataBankComputePool>,
    pub(crate) datasets: Arc<[Arc<Dataset>]>,
    /// Values are forwarded by a dedicated source thread.  The producer never
    /// owns the user iterator, so close can join it even if `Iterator::next()`
    /// is permanently blocked in user code.
    pub(crate) source_rx: flume::Receiver<DataBankResult<MultiBatchCells>>,
    pub(crate) cleanup: RetiredCleanupGuard,
    pub(crate) access_config: ScheduledAccessConfig,
    pub(crate) strategy: AccessStrategy,
    pub(crate) projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    pub(crate) small_projected_sparse_policy: SmallProjectedSparsePolicy,
    pub(crate) response_limit: Option<usize>,
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
    /// One source item selected while waiting for another event. It is consumed
    /// by the next fill pass so no batch is lost between the selector and the
    /// request-submission path.
    pending_source: Option<MultiBatchCells>,
    planned_ready: VecDeque<PlannedBatch>,
    completed: CompletedQueue<T>,
}

impl<T> ProducerState<T>
where
    T: DataValue,
{
    fn new(prefetch_step: usize, worker_count: usize, response_limit: Option<usize>) -> Self {
        let response_limit = scheduled_response_limit(prefetch_step, worker_count, response_limit);
        Self {
            next_read_seq: 0,
            next_emit_seq: 0,
            source_done: false,
            stop_reading: false,
            outstanding: 0,
            active_requests: 0,
            active_responses: 0,
            response_limit,
            pending_source: None,
            planned_ready: VecDeque::new(),
            completed: CompletedQueue::with_capacity(prefetch_step),
        }
    }

    fn is_finished(&self) -> bool {
        self.source_done
            && self.outstanding == 0
            && self.active_requests == 0
            && self.active_responses == 0
            && self.pending_source.is_none()
            && self.planned_ready.is_empty()
    }
}

fn scheduled_response_limit(
    prefetch_step: usize,
    worker_count: usize,
    response_limit: Option<usize>,
) -> usize {
    let default_limit = prefetch_step.min(worker_count.saturating_sub(1).max(1));
    response_limit
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
    Cancelled,
    Source(Result<DataBankResult<MultiBatchCells>, flume::RecvError>),
    Planned(Result<PlannedMessage, flume::RecvError>),
    Done(Result<DoneMessage<T>, flume::RecvError>),
}

const CANCELLABLE_WAIT: Duration = Duration::from_millis(20);
/// Hard process-wide ceiling for detached user `Iterator::next()` calls.
///
/// A synchronous iterator cannot be safely force-stopped. A permanently
/// blocking source therefore keeps its permit until it returns, which bounds
/// orphaned forwarder threads across every DataBank in this process.
const MAX_BATCH_SOURCE_FORWARDERS: usize = 64;
static BATCH_SOURCE_FORWARDER_LIMITER: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn batch_source_forwarder_limiter() -> Arc<Semaphore> {
    Arc::clone(
        BATCH_SOURCE_FORWARDER_LIMITER
            .get_or_init(|| Arc::new(Semaphore::new(MAX_BATCH_SOURCE_FORWARDERS))),
    )
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

/// Detach the user-provided batch iterator from the cancellable producer.
///
/// Rust cannot asynchronously interrupt arbitrary `Iterator::next()` code. The
/// forwarder therefore is deliberately not joined: after session cancellation
/// it exits as soon as a pending `next()` returns, while `PrefetchCells::close`
/// can still join the producer immediately. A source that never returns from
/// `next()` keeps only this forwarder thread and its own state alive. Such
/// threads are bounded by a fixed process-wide permit limit; once exhausted,
/// creating another scheduled prefetch fails instead of leaking another OS
/// thread.
pub(crate) fn spawn_batch_source_forwarder<I>(
    batch_source: I,
    cancel: Arc<PrefetchCancelRegistry>,
    profiler: ScheduledPrefetchProfiler,
    capacity: usize,
) -> io::Result<flume::Receiver<DataBankResult<MultiBatchCells>>>
where
    I: Iterator + Send + 'static,
    I::Item: Into<MultiBatchCells> + Send,
{
    spawn_batch_source_forwarder_with_limiter(
        batch_source,
        cancel,
        profiler,
        capacity,
        batch_source_forwarder_limiter(),
    )
}

fn spawn_batch_source_forwarder_with_limiter<I>(
    batch_source: I,
    cancel: Arc<PrefetchCancelRegistry>,
    profiler: ScheduledPrefetchProfiler,
    capacity: usize,
    limiter: Arc<Semaphore>,
) -> io::Result<flume::Receiver<DataBankResult<MultiBatchCells>>>
where
    I: Iterator + Send + 'static,
    I::Item: Into<MultiBatchCells> + Send,
{
    spawn_batch_source_forwarder_with_limiter_and_spawner(
        batch_source,
        cancel,
        profiler,
        capacity,
        limiter,
        |task| {
            thread::Builder::new()
                .name("databank-prefetch-source".to_string())
                .spawn(task)
                .map(|_| ())
        },
    )
}

fn spawn_batch_source_forwarder_with_limiter_and_spawner<I, S>(
    mut batch_source: I,
    cancel: Arc<PrefetchCancelRegistry>,
    profiler: ScheduledPrefetchProfiler,
    capacity: usize,
    limiter: Arc<Semaphore>,
    spawn: S,
) -> io::Result<flume::Receiver<DataBankResult<MultiBatchCells>>>
where
    I: Iterator + Send + 'static,
    I::Item: Into<MultiBatchCells> + Send,
    S: FnOnce(Box<dyn FnOnce() + Send + 'static>) -> io::Result<()>,
{
    let permit = limiter.try_acquire_owned().map_err(|_| {
        io::Error::new(
            io::ErrorKind::WouldBlock,
            "scheduled prefetch source-forwarder limit reached",
        )
    })?;
    let (tx, rx) = flume::bounded(capacity.max(1));
    spawn(Box::new(move || {
        // Do not release this when the session closes: an arbitrary user
        // `next()` may still be blocked. Drop releases it only when the
        // detached source thread actually returns or unwinds.
        let _permit = permit;
        while !cancel.is_cancelled() {
            let next_started = profiler.start_batch_source_next();
            let next = panic::catch_unwind(AssertUnwindSafe(|| batch_source.next()));
            profiler.record_batch_source_next(next_started);
            let cells = match next {
                Ok(Some(cells)) => cells,
                Ok(None) => break,
                Err(_) => {
                    let _ = send_source_message(
                        &tx,
                        &cancel,
                        Err(DataBankError::PrefetchProducerPanic),
                    );
                    return;
                }
            };
            let batch = match panic::catch_unwind(AssertUnwindSafe(|| cells.into())) {
                Ok(batch) => batch,
                Err(_) => {
                    let _ = send_source_message(
                        &tx,
                        &cancel,
                        Err(DataBankError::PrefetchProducerPanic),
                    );
                    return;
                }
            };
            profiler.record_source_batch(batch.total_cells().unwrap_or(usize::MAX));
            if !send_source_message(&tx, &cancel, Ok(batch)) {
                return;
            }
        }
    }))?;
    Ok(rx)
}

fn send_source_message(
    tx: &flume::Sender<DataBankResult<MultiBatchCells>>,
    cancel: &PrefetchCancelRegistry,
    mut message: DataBankResult<MultiBatchCells>,
) -> bool {
    loop {
        if cancel.is_cancelled() {
            return false;
        }
        match tx.send_timeout(message, CANCELLABLE_WAIT) {
            Ok(()) => return true,
            Err(flume::SendTimeoutError::Timeout(next_message)) => message = next_message,
            Err(flume::SendTimeoutError::Disconnected(_)) => return false,
        }
    }
}

impl<T> PrefetchProducer<T>
where
    T: DataValue,
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
        let mut state = ProducerState::<T>::new(
            self.prefetch_step,
            self.compute.worker_count(),
            self.response_limit,
        );

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
            let batch = match state.pending_source.take() {
                Some(batch) => batch,
                None => match self.source_rx.try_recv() {
                    Ok(Ok(batch)) => batch,
                    Ok(Err(err)) => {
                        self.handle_source_error(state, err);
                        progressed = true;
                        break;
                    }
                    Err(flume::TryRecvError::Empty) => break,
                    Err(flume::TryRecvError::Disconnected) => {
                        state.source_done = true;
                        self.profiler.inc_source_exhausted();
                        progressed = true;
                        break;
                    }
                },
            };
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
                self.strategy.clone(),
                self.projected_sparse_data_strategy,
                self.small_projected_sparse_policy,
                Arc::clone(&self.cancel),
                planned_tx.clone(),
                self.profiler.clone(),
                self.cleanup.clone(),
                self.profiler.start_request_queue_wait(),
            );
            let submit_started = self.profiler.start_submit_request();
            let submit_result = self
                .compute
                .submit_request_cancellable(job, || self.cancel.is_cancelled());
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

    fn handle_source_error(&self, state: &mut ProducerState<T>, err: DataBankError) {
        // Preserve the existing ordered terminal-error contract: batches read
        // before the source failure are emitted first, then this synthetic
        // source position reports the panic/error.
        state.source_done = true;
        state.stop_reading = true;
        let seq = state.next_read_seq;
        state.next_read_seq += 1;
        state.outstanding += 1;
        state.completed.insert(seq, Err(err));
    }

    fn wait_for_event(
        &self,
        state: &mut ProducerState<T>,
        planned_rx: &flume::Receiver<PlannedMessage>,
        done_rx: &flume::Receiver<DoneMessage<T>>,
    ) -> bool {
        let can_read_source = !state.source_done
            && !state.stop_reading
            && state.outstanding < self.prefetch_step
            && state.pending_source.is_none();
        if !can_read_source && state.active_requests == 0 && state.active_responses == 0 {
            return false;
        }

        let cancel_rx = self.cancel.cancel_receiver();
        let mut selector = flume::Selector::new().recv(&cancel_rx, |_| ProducerEvent::Cancelled);
        if can_read_source {
            selector = selector.recv(&self.source_rx, ProducerEvent::Source);
        }
        if state.active_requests > 0 {
            selector = selector.recv(planned_rx, ProducerEvent::Planned);
        }
        if state.active_responses > 0 {
            selector = selector.recv(done_rx, ProducerEvent::Done);
        }

        let wait_started = self.profiler.start_coordinator_wait();
        let event = selector.wait();
        self.profiler.record_coordinator_wait(wait_started);

        match event {
            ProducerEvent::Cancelled => false,
            ProducerEvent::Source(Ok(Ok(batch))) => {
                state.pending_source = Some(batch);
                true
            }
            ProducerEvent::Source(Ok(Err(err))) => {
                self.handle_source_error(state, err);
                true
            }
            ProducerEvent::Source(Err(_)) => {
                state.source_done = true;
                self.profiler.inc_source_exhausted();
                true
            }
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
                self.small_projected_sparse_policy,
                Arc::clone(&self.gene_axes),
                Arc::clone(&self.cancel),
                done_tx.clone(),
                self.profiler.clone(),
                self.cleanup.clone(),
                self.profiler.start_response_queue_wait(),
            );
            let submit_started = self.profiler.start_submit_response();
            let submit_result = self
                .compute
                .submit_response_cancellable(job, || self.cancel.is_cancelled());
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
    strategy: AccessStrategy,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    small_projected_sparse_policy: SmallProjectedSparsePolicy,
    registry: Arc<PrefetchCancelRegistry>,
    planned_tx: flume::Sender<PlannedMessage>,
    profiler: ScheduledPrefetchProfiler,
    cleanup: RetiredCleanupGuard,
    queued_at: ProfileTimer,
) -> ComputeJob {
    Box::new(move || {
        // Keep the cleanup callback with this queued closure. It is especially
        // important when cancellation drops the last dataset Arc from a job
        // after the consumer already returned from `close`.
        let _cleanup = cleanup;
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
                    small_projected_sparse_policy,
                );
                profiler.record_request_plan(plan_started);
                let (mut plan, mut items) = planned?;
                if registry.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
                let cancel = PrefetchCancel::new(access.clone());
                // Register before projected-sparse preplanning can build a
                // ScheduledAccess and block in its first `next()`. The guard
                // unregisters on every error, panic unwind, or dropped channel;
                // successful work transfers it to the response lifecycle.
                registry.register(seq, Arc::clone(&cancel));
                let registration = ActiveBatchGuard {
                    seq,
                    registry: Arc::clone(&registry),
                };
                if registry.is_cancelled() || cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
                preplan_selected_sparse_request(
                    &access,
                    &strategy,
                    access_config,
                    projected_sparse_data_strategy,
                    small_projected_sparse_policy,
                    gene_axes.as_ref(),
                    Arc::clone(&cancel),
                    &profiler,
                    &mut plan,
                    &mut items,
                )?;
                if registry.is_cancelled() || cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
                // Profile migration point 1 (§5.9): strategy.build() must stay
                // inside the start_request_schedule / record_request_schedule
                // span — do not move it into AccessStrategy::build.
                let schedule_started = profiler.start_request_schedule();
                let strategy_for_response = strategy.clone();
                let use_targeted_cache = batch_plan_uses_targeted_selected_sparse_cache(&plan);
                let scheduled_result = strategy.build(
                    access.clone(),
                    items,
                    access_config,
                    Arc::clone(&cancel),
                    false,
                    use_targeted_cache,
                );
                profiler.record_request_schedule(schedule_started);
                let scheduled = scheduled_result?;
                Ok(Box::new(PlannedBatch {
                    seq,
                    plan,
                    scheduled,
                    strategy: strategy_for_response,
                    cancel,
                    registration,
                }))
            }))
            .unwrap_or(Err(DataBankError::ComputeWorkerPanic));
        if result.is_err() {
            profiler.inc_request_error();
        }
        let send_started = profiler.start_request_send();
        // On a disconnected channel the message (and its registration guard)
        // is dropped here, which unregisters the preplan handle automatically.
        let _ = planned_tx.send(PlannedMessage { seq, result });
        profiler.record_request_send(send_started);
        profiler.record_request_total(total_started);
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn preplan_selected_sparse_request(
    access: &AccessHandle,
    strategy: &AccessStrategy,
    access_config: ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    small_projected_sparse_policy: SmallProjectedSparsePolicy,
    gene_axes: &MultiGeneAxisPlan,
    cancel: Arc<PrefetchCancel>,
    profiler: &ScheduledPrefetchProfiler,
    plan: &mut BatchPlan,
    items: &mut Vec<AccessItem>,
) -> DataBankResult<()> {
    if !fastpath::preplan_selected_sparse_enabled()
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
                strategy,
                access_config,
                projected_sparse_data_strategy,
                small_projected_sparse_policy,
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
                    strategy,
                    access_config,
                    projected_sparse_data_strategy,
                    small_projected_sparse_policy,
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
    strategy: &AccessStrategy,
    access_config: ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    small_projected_sparse_policy: SmallProjectedSparsePolicy,
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
            small_projected_sparse_policy,
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
    let mut index_scheduled = strategy.build(
        access.clone(),
        index_items,
        access_config,
        Arc::clone(&cancel),
        false,
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
    let defer_selected_data = fastpath::preplan_selected_sparse_defer_data_enabled()
        && can_defer_selected_sparse_data_to_response(strategy);
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

fn batch_plan_uses_targeted_selected_sparse_cache(plan: &BatchPlan) -> bool {
    match plan {
        BatchPlan::Single { plan, .. } => single_plan_uses_targeted_selected_sparse_cache(plan),
        BatchPlan::Multi(multi) => multi
            .parts
            .iter()
            .any(|part| single_plan_uses_targeted_selected_sparse_cache(&part.plan)),
    }
}

fn single_plan_uses_targeted_selected_sparse_cache(plan: &SingleDatasetPlan) -> bool {
    match plan {
        SingleDatasetPlan::Sparse {
            plan,
            preloaded_index_bytes,
            selected_data_scheduled,
            ..
        } => {
            preloaded_index_bytes.is_some()
                && *selected_data_scheduled
                && !plan.data_groups.is_empty()
        }
        SingleDatasetPlan::Dense { .. } => false,
    }
}

/// Whether selected sparse data can be deferred to the response phase.
///
/// Deferral is only meaningful on the native path: the response worker must
/// be able to reconstruct the scheduled native access, which only exists when
/// the strategy resolved to `BloscLz4Native`. With the strategy resolved at
/// spawn time, the old `(native_mode, native)` pairing collapses to
/// `strategy.is_native()`.
fn can_defer_selected_sparse_data_to_response(strategy: &AccessStrategy) -> bool {
    strategy.is_native()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn make_prefetch_response_job<T>(
    planned: PlannedBatch,
    access: AccessHandle,
    compute: Arc<DataBankComputePool>,
    access_config: ScheduledAccessConfig,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    small_projected_sparse_policy: SmallProjectedSparsePolicy,
    gene_axes: Arc<MultiGeneAxisPlan>,
    registry: Arc<PrefetchCancelRegistry>,
    done_tx: flume::Sender<DoneMessage<T>>,
    profiler: ScheduledPrefetchProfiler,
    cleanup: RetiredCleanupGuard,
    queued_at: ProfileTimer,
) -> ComputeJob
where
    T: DataValue,
{
    Box::new(move || {
        let _cleanup = cleanup;
        profiler.inc_response_job();
        profiler.record_response_queue_wait(queued_at);
        let total_started = profiler.start_response_total();
        let PlannedBatch {
            seq,
            plan,
            mut scheduled,
            strategy,
            cancel,
            registration: _registration,
        } = planned;
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
                    small_projected_sparse_policy,
                    gene_axes.as_ref(),
                    &cancel,
                    &profiler,
                    &strategy,
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

#[cfg(test)]
mod tests {
    use super::{
        spawn_batch_source_forwarder_with_limiter,
        spawn_batch_source_forwarder_with_limiter_and_spawner,
    };
    use crate::databank::gene_axis::MultiBatchCells;
    use crate::databank::scheduled::profile::ScheduledPrefetchProfiler;
    use crate::databank::scheduled::types::PrefetchCancelRegistry;
    use std::io;
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Semaphore;

    struct GatedSource {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }

    impl Iterator for GatedSource {
        type Item = MultiBatchCells;

        fn next(&mut self) -> Option<Self::Item> {
            let _ = self.started.send(());
            let _ = self.release.recv();
            None
        }
    }

    #[test]
    fn source_forwarder_limiter_rejects_without_spawning() {
        let limiter = Arc::new(Semaphore::new(1));
        let held = limiter
            .clone()
            .try_acquire_owned()
            .expect("hold only source-forwarder permit");
        let err = spawn_batch_source_forwarder_with_limiter(
            std::iter::empty::<MultiBatchCells>(),
            PrefetchCancelRegistry::new(),
            ScheduledPrefetchProfiler::from_env(),
            1,
            Arc::clone(&limiter),
        )
        .expect_err("a full process limiter must reject another forwarder");
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
        drop(held);
        assert!(limiter.try_acquire().is_ok());
    }

    #[test]
    fn source_forwarder_spawn_failure_releases_permit() {
        let limiter = Arc::new(Semaphore::new(1));
        let err = spawn_batch_source_forwarder_with_limiter_and_spawner(
            std::iter::empty::<MultiBatchCells>(),
            PrefetchCancelRegistry::new(),
            ScheduledPrefetchProfiler::from_env(),
            1,
            Arc::clone(&limiter),
            |_task| Err(io::Error::other("injected source-forwarder spawn failure")),
        )
        .expect_err("injected spawn failure");
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert!(
            limiter.try_acquire().is_ok(),
            "the dropped task closure must release its owned permit"
        );
    }

    #[test]
    fn gated_source_releases_limiter_permit_after_returning() {
        let limiter = Arc::new(Semaphore::new(1));
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let rx = spawn_batch_source_forwarder_with_limiter(
            GatedSource {
                started: started_tx,
                release: release_rx,
            },
            PrefetchCancelRegistry::new(),
            ScheduledPrefetchProfiler::from_env(),
            1,
            Arc::clone(&limiter),
        )
        .expect("spawn gated source forwarder");
        started_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("source entered blocking next");
        assert!(
            limiter.try_acquire().is_err(),
            "the blocked source must retain its process-wide permit"
        );
        release_tx.send(()).expect("release gated source");
        assert!(
            rx.recv_timeout(Duration::from_secs(2)).is_err(),
            "forwarder should finish and disconnect after the source returns"
        );
        assert!(
            limiter.try_acquire().is_ok(),
            "normal forwarder exit must release its permit"
        );
    }
}
