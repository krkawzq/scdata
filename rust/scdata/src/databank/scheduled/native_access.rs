use std::io;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use tokio::task::JoinSet;

use crate::access::{
    AccessHandle, AccessItem, IoBackend, PrefetchCancel, ScheduledAccess, ScheduledAccessConfig,
};

use super::super::config::NativeAccessConfig;
use super::super::error::{DataBankError, DataBankResult};
use super::super::native::{
    load_access_items_blosc_lz4_native, NativeBlockDecodedCache, NativeBlockIndexCache,
    NativeBlockPayloadCache,
};

#[derive(Clone)]
pub(crate) struct NativeScheduledContext {
    pub(crate) io: Arc<dyn IoBackend>,
    pub(crate) config: NativeAccessConfig,
    pub(crate) index_cache: Arc<NativeBlockIndexCache>,
    pub(crate) block_cache: Option<Arc<NativeBlockPayloadCache>>,
    pub(crate) decoded_cache: Option<Arc<NativeBlockDecodedCache>>,
    executor: Arc<NativeScheduledExecutor>,
}

impl NativeScheduledContext {
    pub(crate) fn new(io: Arc<dyn IoBackend>, config: NativeAccessConfig) -> DataBankResult<Self> {
        let executor = Arc::new(NativeScheduledExecutor::new(
            config.load.scheduler_workers.max(1),
        )?);
        Ok(Self {
            io,
            config,
            index_cache: Arc::new(NativeBlockIndexCache::new()),
            block_cache: native_block_payload_cache_from_env(),
            decoded_cache: native_decoded_block_cache_from_env(),
            executor,
        })
    }
}

/// The resolved access strategy for a single prefetch session.
///
/// Produced once at spawn time by `resolve_strategy` (see `scheduled/mod.rs`),
/// then carried by the producer / assemble layers for the whole session. It
/// replaces the previous `(native_mode, Option<NativeScheduledContext>)` pair
/// that was threaded through every function: the only question those callers
/// actually had — "are we on the native execution path?" — is now answered by
/// [`Self::is_native`].
///
/// `NativeMode` stays as the *policy* requested by the caller; `AccessStrategy`
/// is the *resolved* strategy actually running. Once resolved to
/// [`Self::BloscLz4Native`], the native worker runs with zero fallback: a decode
/// failure is a real error, never a silent retreat to the generic path.
#[derive(Clone)]
pub(crate) enum AccessStrategy {
    /// Generic access-scheduler chunk reads + decode pool path.
    Generic,
    /// Blosc-LZ4 native direct read + block-level scatter path. Zero fallback.
    BloscLz4Native(NativeScheduledContext),
}

impl AccessStrategy {
    /// Whether this session runs on the native execution path. Replaces the
    /// scattered `(native_mode, native)` paired queries.
    pub(crate) fn is_native(&self) -> bool {
        matches!(self, Self::BloscLz4Native(_))
    }

    /// Build this batch's scheduled access iterator. Replaces the 5-arm
    /// `build_scheduled_batch_access` match. The native path runs with zero
    /// fallback: a decode failure surfaces as an `io::Error` to the consumer.
    pub(crate) fn build(
        &self,
        access: AccessHandle,
        items: Vec<AccessItem>,
        access_config: ScheduledAccessConfig,
        cancel: Arc<PrefetchCancel>,
        allow_small_command_grouping: bool,
    ) -> DataBankResult<ScheduledBatchAccess> {
        match self {
            Self::Generic => build_generic_scheduled_access(access, items, access_config, cancel),
            Self::BloscLz4Native(ctx) => NativeScheduledAccess::spawn(
                ctx.clone(),
                items,
                access_config,
                cancel,
                allow_small_command_grouping,
            )
            .map(ScheduledBatchAccess::Native),
        }
    }
}

/// The resolved access strategy for one prefetch session, plus a user-facing
/// label and optional fallback reason for observability.
///
/// Produced once at spawn time by `resolve_strategy` (see `scheduled/mod.rs`).
/// `strategy` is the resolved [`AccessStrategy`] the session actually runs;
/// `label` is a stable short name (`"generic"` / `"blosc_lz4_fast"`) surfaced to
/// Python via `PrefetchCells.resolved_strategy`; `reason` is `Some` when the
/// fast path was *requested* (`auto`/`force`) but the session fell back to
/// generic, explaining why — it is `None` when the fast path is active, or when
/// fast mode was not requested (`disabled`).
#[derive(Clone)]
pub(crate) struct ResolvedStrategy {
    pub(crate) strategy: AccessStrategy,
    pub(crate) label: &'static str,
    pub(crate) reason: Option<&'static str>,
}

impl ResolvedStrategy {
    pub(crate) const GENERIC_LABEL: &'static str = "generic";
    pub(crate) const FAST_LABEL: &'static str = "blosc_lz4_fast";
}

fn native_block_payload_cache_from_env() -> Option<Arc<NativeBlockPayloadCache>> {
    let capacity = std::env::var("SCDATA_NATIVE_BLOCK_CACHE_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    (capacity > 0).then(|| Arc::new(NativeBlockPayloadCache::new(capacity)))
}

fn native_decoded_block_cache_from_env() -> Option<Arc<NativeBlockDecodedCache>> {
    let capacity = std::env::var("SCDATA_NATIVE_DECODED_BLOCK_CACHE_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    (capacity > 0).then(|| Arc::new(NativeBlockDecodedCache::new(capacity)))
}

pub(crate) enum ScheduledBatchAccess {
    Generic(ScheduledAccess<std::vec::IntoIter<AccessItem>>),
    Native(NativeScheduledAccess),
}

impl Iterator for ScheduledBatchAccess {
    type Item = io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Generic(scheduled) => scheduled.next(),
            Self::Native(scheduled) => scheduled.next(),
        }
    }
}

pub(crate) async fn load_native_items_ordered_async(
    native: NativeScheduledContext,
    items: Vec<AccessItem>,
    cancel: Arc<PrefetchCancel>,
) -> DataBankResult<Vec<Vec<u8>>> {
    let total_items = items.len();
    let batch = items.into_iter().enumerate().collect::<Vec<_>>();
    let results = load_native_batch(native, batch, cancel).await;
    let mut ordered = (0..total_items).map(|_| None).collect::<Vec<_>>();
    for (seq, result) in results {
        if seq >= total_items {
            return Err(DataBankError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "native inline batch result index out of bounds",
            )));
        }
        ordered[seq] = Some(result.map_err(DataBankError::Io)?);
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(seq, result)| {
            result.ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("native inline batch missing result {seq}"),
                ))
            })
        })
        .collect()
}

pub(crate) fn load_native_items_ordered_blocking(
    native: NativeScheduledContext,
    items: Vec<AccessItem>,
    cancel: Arc<PrefetchCancel>,
) -> DataBankResult<Vec<Vec<u8>>> {
    let (reply, rx) = mpsc::sync_channel(1);
    let executor = Arc::clone(&native.executor);
    executor.submit_ordered(NativeOrderedCommand {
        native,
        items,
        cancel,
        reply,
    })?;
    rx.recv().map_err(|_| {
        DataBankError::Io(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "native ordered selected-data worker ended",
        ))
    })?
}

pub(crate) type NativeCustomJob =
    Box<dyn FnOnce(&tokio::runtime::Runtime) -> DataBankResult<()> + Send + 'static>;

pub(crate) fn run_native_custom_blocking(
    native: NativeScheduledContext,
    cancel: Arc<PrefetchCancel>,
    job: NativeCustomJob,
) -> DataBankResult<()> {
    let (reply, rx) = mpsc::sync_channel(1);
    let executor = Arc::clone(&native.executor);
    executor.submit_custom(NativeCustomCommand { cancel, job, reply })?;
    rx.recv().map_err(|_| {
        DataBankError::Io(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "native custom selected-data worker ended",
        ))
    })?
}

fn build_generic_scheduled_access(
    access: AccessHandle,
    items: Vec<AccessItem>,
    access_config: ScheduledAccessConfig,
    cancel: Arc<PrefetchCancel>,
) -> DataBankResult<ScheduledBatchAccess> {
    let mut scheduled = access.scheduled(items, access_config)?;
    scheduled.set_cancel_handle(cancel);
    Ok(ScheduledBatchAccess::Generic(scheduled))
}

pub(crate) struct NativeScheduledAccess {
    rx: Option<flume::Receiver<io::Result<Vec<u8>>>>,
}

impl NativeScheduledAccess {
    fn spawn(
        native: NativeScheduledContext,
        items: Vec<AccessItem>,
        access_config: ScheduledAccessConfig,
        cancel: Arc<PrefetchCancel>,
        allow_small_command_grouping: bool,
    ) -> DataBankResult<Self> {
        let window = native_window(&native.config, access_config);
        let (tx, rx) = flume::bounded(window.max(1));
        let executor = Arc::clone(&native.executor);
        executor.submit(NativeScheduledCommand {
            native,
            items,
            cancel,
            allow_small_command_grouping,
            window,
            tx,
        })?;
        Ok(Self { rx: Some(rx) })
    }
}

impl Iterator for NativeScheduledAccess {
    type Item = io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.as_ref()?.recv().ok()
    }
}

impl Drop for NativeScheduledAccess {
    fn drop(&mut self) {
        self.rx.take();
    }
}

struct NativeScheduledCommand {
    native: NativeScheduledContext,
    items: Vec<AccessItem>,
    cancel: Arc<PrefetchCancel>,
    allow_small_command_grouping: bool,
    window: usize,
    tx: flume::Sender<io::Result<Vec<u8>>>,
}

struct NativeOrderedCommand {
    native: NativeScheduledContext,
    items: Vec<AccessItem>,
    cancel: Arc<PrefetchCancel>,
    reply: mpsc::SyncSender<DataBankResult<Vec<Vec<u8>>>>,
}

struct NativeCustomCommand {
    cancel: Arc<PrefetchCancel>,
    job: NativeCustomJob,
    reply: mpsc::SyncSender<DataBankResult<()>>,
}

enum NativeExecutorCommand {
    Scheduled(NativeScheduledCommand),
    Ordered(NativeOrderedCommand),
    Custom(NativeCustomCommand),
}

struct NativeScheduledExecutor {
    tx: Option<flume::Sender<NativeExecutorCommand>>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl NativeScheduledExecutor {
    fn new(workers: usize) -> io::Result<Self> {
        let workers = workers.max(1);
        let (tx, rx) = flume::unbounded();
        let mut threads = Vec::with_capacity(workers);
        for worker_idx in 0..workers {
            let worker_rx = rx.clone();
            let handle = thread::Builder::new()
                .name(format!("databank-native-scheduled-{worker_idx}"))
                .spawn(move || native_scheduled_worker_loop(worker_rx))?;
            threads.push(handle);
        }
        Ok(Self {
            tx: Some(tx),
            threads,
        })
    }

    fn submit(&self, command: NativeScheduledCommand) -> io::Result<()> {
        let tx = self.tx.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "native scheduled executor closed",
            )
        })?;
        tx.send(NativeExecutorCommand::Scheduled(command))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "native scheduled executor closed",
                )
            })
    }

    fn submit_ordered(&self, command: NativeOrderedCommand) -> io::Result<()> {
        let tx = self.tx.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "native scheduled executor closed",
            )
        })?;
        tx.send(NativeExecutorCommand::Ordered(command))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "native scheduled executor closed",
                )
            })
    }

    fn submit_custom(&self, command: NativeCustomCommand) -> io::Result<()> {
        let tx = self.tx.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "native scheduled executor closed",
            )
        })?;
        tx.send(NativeExecutorCommand::Custom(command))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "native scheduled executor closed",
                )
            })
    }
}

impl Drop for NativeScheduledExecutor {
    fn drop(&mut self) {
        self.tx.take();
        let current = thread::current().id();
        while let Some(handle) = self.threads.pop() {
            if handle.thread().id() == current {
                continue;
            }
            let _ = handle.join();
        }
    }
}

fn native_scheduled_worker_loop(rx: flume::Receiver<NativeExecutorCommand>) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            while let Ok(command) = rx.recv() {
                fail_native_executor_command(command, err.to_string());
            }
            return;
        }
    };

    let mut pending = None;
    loop {
        let command = if let Some(command) = pending.take() {
            command
        } else {
            match rx.recv() {
                Ok(command) => command,
                Err(_) => break,
            }
        };
        let command = match command {
            NativeExecutorCommand::Scheduled(command) => command,
            NativeExecutorCommand::Ordered(command) => {
                runtime.block_on(run_native_ordered(command));
                continue;
            }
            NativeExecutorCommand::Custom(command) => {
                run_native_custom(&runtime, command);
                continue;
            }
        };

        let mut commands = vec![command];
        if should_group_native_commands(&commands[0]) {
            let mut total_items = commands[0].items.len();
            let max_items = native_command_group_max_items(&commands[0]);
            while total_items < max_items {
                match rx.try_recv() {
                    Ok(NativeExecutorCommand::Scheduled(next)) => {
                        if should_group_native_commands(&next)
                            && total_items + next.items.len() <= max_items
                        {
                            total_items += next.items.len();
                            commands.push(next);
                        } else {
                            pending = Some(NativeExecutorCommand::Scheduled(next));
                            break;
                        }
                    }
                    Ok(next) => {
                        pending = Some(next);
                        break;
                    }
                    Err(_) => break,
                }
            }
        }

        if commands.len() == 1 {
            let command = commands.pop().expect("one command");
            runtime.block_on(run_native_scheduled(
                command.native,
                command.items,
                command.cancel,
                command.window,
                command.tx,
            ));
        } else {
            runtime.block_on(run_native_scheduled_small_commands(commands));
        }
    }
}

fn fail_native_executor_command(command: NativeExecutorCommand, message: String) {
    match command {
        NativeExecutorCommand::Scheduled(command) => {
            let _ = command.tx.send(Err(io::Error::other(message)));
            command.cancel.cancel_in_flight();
        }
        NativeExecutorCommand::Ordered(command) => {
            let _ = command
                .reply
                .send(Err(DataBankError::Io(io::Error::other(message))));
            command.cancel.cancel_in_flight();
        }
        NativeExecutorCommand::Custom(command) => {
            let _ = command
                .reply
                .send(Err(DataBankError::Io(io::Error::other(message))));
            command.cancel.cancel_in_flight();
        }
    }
}

async fn run_native_ordered(command: NativeOrderedCommand) {
    let NativeOrderedCommand {
        native,
        items,
        cancel,
        reply,
    } = command;
    let result = load_native_items_ordered_async(native, items, Arc::clone(&cancel)).await;
    if result.is_err() {
        cancel.cancel_in_flight();
    }
    let _ = reply.send(result);
}

fn run_native_custom(runtime: &tokio::runtime::Runtime, command: NativeCustomCommand) {
    let NativeCustomCommand { cancel, job, reply } = command;
    let result = job(runtime);
    if result.is_err() {
        cancel.cancel_in_flight();
    }
    let _ = reply.send(result);
}

fn should_group_native_commands(command: &NativeScheduledCommand) -> bool {
    (command.allow_small_command_grouping || native_cross_command_grouping_enabled())
        && !command.cancel.is_cancelled()
        && !command.items.is_empty()
        && command.items.len() > 1
        && command.items.len() < native_command_group_max_items(command)
}

async fn run_native_scheduled(
    native: NativeScheduledContext,
    items: Vec<AccessItem>,
    cancel: Arc<PrefetchCancel>,
    window: usize,
    tx: flume::Sender<io::Result<Vec<u8>>>,
) {
    // Items are numbered `0..total_items` by `enumerate()`, so a slot-indexed
    // Vec replaces the BTreeMap used for in-order emission: insertion is a
    // direct index write and the drain is a contiguous scan from `next_emit`,
    // with no per-completion node allocation or key comparison. The native
    // worker submits tasks in source order and IO usually completes in
    // submission order, so reordering is shallow and the O(1) slot path wins.
    let total_items = items.len();
    let mut source = items.into_iter().enumerate();
    let mut tasks = JoinSet::new();
    let mut completed: Vec<Option<io::Result<Vec<u8>>>> = (0..total_items).map(|_| None).collect();
    let mut next_emit = 0usize;
    let mut source_done = false;

    loop {
        while !source_done && tasks.len() < window && !cancel.is_cancelled() {
            let Some(first) = source.next() else {
                source_done = true;
                break;
            };
            let batch_size = native_item_batch_size(&native);
            let mut batch = Vec::with_capacity(batch_size);
            batch.push(first);
            while batch.len() < batch_size {
                let Some(next) = source.next() else {
                    source_done = true;
                    break;
                };
                batch.push(next);
            }
            let native = native.clone();
            let cancel = Arc::clone(&cancel);
            tasks.spawn(async move { load_native_batch(native, batch, cancel).await });
        }

        // Reclaim every already-finished task without awaiting. On a
        // current-thread runtime `join_next().await` returns a single
        // completion per poll, but other tasks often finish during the same
        // await window (they were polled together); `try_join_next` drains
        // them in a tight loop so the next iteration emits a burst in order
        // instead of paying one await round-trip per completion.
        while let Some(joined) = tasks.try_join_next() {
            match joined {
                Ok(results) => {
                    for (seq, result) in results {
                        if seq < total_items {
                            completed[seq] = Some(result);
                        }
                    }
                }
                Err(err) => {
                    let _ = tx.send(Err(io::Error::other(format!(
                        "native scheduled worker failed: {err}"
                    ))));
                    cancel.cancel_in_flight();
                    return;
                }
            }
        }

        while next_emit < total_items {
            let Some(result) = completed[next_emit].take() else {
                break;
            };
            let should_stop = result.is_err();
            if tx.send(result).is_err() {
                cancel.cancel_in_flight();
                return;
            }
            next_emit += 1;
            if should_stop {
                cancel.cancel_in_flight();
                return;
            }
        }

        if source_done && tasks.is_empty() {
            return;
        }
        if cancel.is_cancelled() {
            let _ = tx.send(Err(io::Error::other("scheduled item cancelled")));
            return;
        }

        let Some(joined) = tasks.join_next().await else {
            return;
        };
        match joined {
            Ok(results) => {
                for (seq, result) in results {
                    if seq < total_items {
                        completed[seq] = Some(result);
                    }
                }
            }
            Err(err) => {
                let _ = tx.send(Err(io::Error::other(format!(
                    "native scheduled worker failed: {err}"
                ))));
                cancel.cancel_in_flight();
                return;
            }
        }
    }
}

async fn load_native_batch(
    native: NativeScheduledContext,
    batch: Vec<(usize, AccessItem)>,
    cancel: Arc<PrefetchCancel>,
) -> Vec<(usize, io::Result<Vec<u8>>)> {
    if cancel.is_cancelled() {
        return batch
            .into_iter()
            .map(|(seq, _)| (seq, Err(io::Error::other("scheduled item cancelled"))))
            .collect();
    }

    let items = batch
        .iter()
        .map(|(_, item)| item.clone())
        .collect::<Vec<_>>();
    let native_result = load_access_items_blosc_lz4_native(
        Arc::clone(&native.io),
        native.config.load.coalesce.clone(),
        &native.index_cache,
        native.block_cache.clone(),
        native.decoded_cache.clone(),
        &items,
        0,
    )
    .await;

    // Zero fallback: the strategy resolved to native at spawn time, so a native
    // decode failure is a real error. `None` (the loader declined this item —
    // e.g. a non-lz4 blosc variant or an unsupported block table) and `Err`
    // (IO / decode error) both surface as `io::Error` rather than retreating to
    // the generic access path.
    match native_result {
        Ok(results) => {
            let mut out = Vec::with_capacity(batch.len());
            for ((seq, _), result) in batch.into_iter().zip(results) {
                let result = match result {
                    Some(bytes) => Ok(bytes),
                    None => Err(io::Error::other(
                        "native loader returned no result for blosc item",
                    )),
                };
                out.push((seq, result));
            }
            out
        }
        Err(err) => {
            let message = err.to_string();
            batch
                .into_iter()
                .map(|(seq, _)| (seq, Err(io::Error::other(message.clone()))))
                .collect()
        }
    }
}

struct SmallCommandState {
    tx: flume::Sender<io::Result<Vec<u8>>>,
    cancel: Arc<PrefetchCancel>,
    item_count: usize,
}

async fn run_native_scheduled_small_commands(commands: Vec<NativeScheduledCommand>) {
    if commands.is_empty() {
        return;
    }
    let native = commands[0].native.clone();
    let mut states = Vec::with_capacity(commands.len());
    let mut slots = Vec::new();
    let mut batch = Vec::new();

    for (command_idx, command) in commands.into_iter().enumerate() {
        let item_count = command.items.len();
        states.push(SmallCommandState {
            tx: command.tx,
            cancel: command.cancel,
            item_count,
        });
        if states[command_idx].cancel.is_cancelled() {
            continue;
        }
        for (item_idx, item) in command.items.into_iter().enumerate() {
            let flat_seq = slots.len();
            slots.push((command_idx, item_idx));
            batch.push((flat_seq, item));
        }
    }

    let mut completed: Vec<Vec<Option<io::Result<Vec<u8>>>>> = states
        .iter()
        .map(|state| (0..state.item_count).map(|_| None).collect())
        .collect();
    for (command_idx, state) in states.iter().enumerate() {
        if state.cancel.is_cancelled() {
            for slot in &mut completed[command_idx] {
                *slot = Some(Err(io::Error::other("scheduled item cancelled")));
            }
        }
    }

    if !batch.is_empty() {
        let results = load_native_batch_uncancelled(native, batch).await;
        for (flat_seq, result) in results {
            let Some(&(command_idx, item_idx)) = slots.get(flat_seq) else {
                continue;
            };
            if let Some(slot) = completed
                .get_mut(command_idx)
                .and_then(|items| items.get_mut(item_idx))
            {
                *slot = Some(result);
            }
        }
    }

    for (command_idx, state) in states.into_iter().enumerate() {
        for slot in completed[command_idx].iter_mut().take(state.item_count) {
            let result = slot.take().unwrap_or_else(|| {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "native grouped scheduled item missing output",
                ))
            });
            let should_stop = result.is_err();
            if state.tx.send(result).is_err() {
                state.cancel.cancel_in_flight();
                break;
            }
            if should_stop {
                state.cancel.cancel_in_flight();
                break;
            }
        }
    }
}

async fn load_native_batch_uncancelled(
    native: NativeScheduledContext,
    batch: Vec<(usize, AccessItem)>,
) -> Vec<(usize, io::Result<Vec<u8>>)> {
    let items = batch
        .iter()
        .map(|(_, item)| item.clone())
        .collect::<Vec<_>>();
    let native_result = load_access_items_blosc_lz4_native(
        Arc::clone(&native.io),
        native.config.load.coalesce.clone(),
        &native.index_cache,
        native.block_cache.clone(),
        native.decoded_cache.clone(),
        &items,
        0,
    )
    .await;

    // Zero fallback, same as `load_native_batch`: `None` and `Err` both surface
    // as `io::Error`. This variant is the small-command-grouping path and does
    // not consult the cancel flag before loading (cancellation is checked per
    // command in `run_native_scheduled_small_commands`).
    match native_result {
        Ok(results) => {
            let mut out = Vec::with_capacity(batch.len());
            for ((seq, _), result) in batch.into_iter().zip(results) {
                let result = match result {
                    Some(bytes) => Ok(bytes),
                    None => Err(io::Error::other(
                        "native loader returned no result for blosc item",
                    )),
                };
                out.push((seq, result));
            }
            out
        }
        Err(err) => {
            let message = err.to_string();
            batch
                .into_iter()
                .map(|(seq, _)| (seq, Err(io::Error::other(message.clone()))))
                .collect()
        }
    }
}

fn native_window(config: &NativeAccessConfig, access_config: ScheduledAccessConfig) -> usize {
    let access_window = access_config
        .prefetch_step
        .max(access_config.decode_ahead_steps)
        .max(access_config.ready_ahead_steps)
        .max(1);
    config
        .request_prefetch_blocks
        .max(1)
        .min(access_window.max(config.fused_workers.max(1)))
}

fn native_item_batch_size(native: &NativeScheduledContext) -> usize {
    if let Some(value) = std::env::var("SCDATA_NATIVE_ITEM_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
    {
        return value.clamp(8, 4096);
    }
    if native.block_cache.is_none() && native.decoded_cache.is_none() {
        return native
            .config
            .fused_workers
            .saturating_mul(8)
            .clamp(128, 512);
    }
    native_item_batch_size_from_config(&native.config)
}

fn native_item_batch_size_from_config(config: &NativeAccessConfig) -> usize {
    config.fused_workers.saturating_mul(4).clamp(8, 256)
}

fn native_batch_multi_item_min() -> usize {
    8
}

fn native_small_command_group_max_items(native: &NativeScheduledContext) -> usize {
    native_item_batch_size(native)
        .min(16)
        .max(native_batch_multi_item_min())
}

fn native_command_group_max_items(command: &NativeScheduledCommand) -> usize {
    if native_cross_command_grouping_enabled() {
        return std::env::var("SCDATA_NATIVE_COMMAND_GROUP_MAX_ITEMS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 1)
            .unwrap_or_else(|| native_item_batch_size(&command.native).saturating_mul(4))
            .clamp(native_batch_multi_item_min(), 4096);
    }
    native_small_command_group_max_items(&command.native)
}

fn native_cross_command_grouping_enabled() -> bool {
    std::env::var("SCDATA_NATIVE_COMMAND_GROUPING")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{
        load_native_batch, load_native_items_ordered_async, load_native_items_ordered_blocking,
        AccessStrategy, NativeScheduledContext, ScheduledBatchAccess,
    };
    use crate::access::{
        AccessConfig, AccessCpuConfig, AccessHandle, AccessItem, AccessProfile, AccessScheduler,
        ChunkKey, DecodeBackend, DecodeTask, FileRef, IoBackend, IoTask, PrefetchCancel,
        ScheduledAccessConfig, SliceSpec,
    };
    use crate::codecs::{codec_from_json_str, CodecError, DecodeSlice, SharedCodec};
    use crate::databank::config::{
        NativeAccessConfig, NativeBloscConfig, NativeLoadCoalesceConfig, NativeLoadConfig,
    };
    use std::io;
    use std::sync::Arc;
    use std::time::Duration;

    /// Re-implementation of the `manual_blosc_lz4_raw_blocks` fixture from
    /// `native::item::tests` — builds an uncompressed-block Blosc-LZ4 chunk
    /// whose decode is the concatenation of `blocks`. Self-contained so this
    /// module's tests do not depend on a private sibling test helper.
    fn manual_blosc_lz4_raw_blocks(blocks: &[&[u8]]) -> Vec<u8> {
        assert!(!blocks.is_empty());
        let blocksize = blocks[0].len();
        assert!(blocks.iter().all(|block| block.len() == blocksize));
        let decoded_size = blocks.iter().map(|block| block.len()).sum::<usize>();
        let table_bytes = blocks.len() * 4;
        let compressed_size =
            blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + table_bytes + decoded_size + table_bytes;
        let mut encoded = vec![0u8; blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + table_bytes];
        encoded[0] = blosc_src::BLOSC_VERSION_FORMAT as u8;
        encoded[1] = blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8;
        encoded[2] = (blosc_src::BLOSC_LZ4_FORMAT << 5) as u8;
        encoded[3] = 1;
        encoded[4..8].copy_from_slice(&(decoded_size as u32).to_le_bytes());
        encoded[8..12].copy_from_slice(&(blocksize as u32).to_le_bytes());
        encoded[12..16].copy_from_slice(&(compressed_size as u32).to_le_bytes());

        let mut offset = blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + table_bytes;
        for (idx, block) in blocks.iter().enumerate() {
            let table_offset = blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + idx * 4;
            encoded[table_offset..table_offset + 4].copy_from_slice(&(offset as i32).to_le_bytes());
            encoded.extend_from_slice(&(block.len() as i32).to_le_bytes());
            encoded.extend_from_slice(block);
            offset += 4 + block.len();
        }
        assert_eq!(encoded.len(), compressed_size);
        encoded
    }

    /// `IoBackend` serving a single in-memory byte buffer at `base_offset`.
    /// Same shape as the fixture in `native::item::tests` (minus the read log,
    /// which these tests do not assert on).
    struct RangeIo {
        file: FileRef,
        base_offset: u64,
        bytes: Arc<[u8]>,
    }

    impl RangeIo {
        fn new(file: FileRef, base_offset: u64, bytes: Vec<u8>) -> Self {
            Self {
                file,
                base_offset,
                bytes: Arc::from(bytes.into_boxed_slice()),
            }
        }
    }

    impl IoBackend for RangeIo {
        fn submit_read(&self, file: FileRef, offset: u64, len: usize, _priority: u8) -> IoTask {
            assert_eq!(file, self.file);
            let start = usize::try_from(offset - self.base_offset).expect("offset");
            let end = start + len;
            let data: Arc<[u8]> = Arc::from(self.bytes[start..end].to_vec().into_boxed_slice());
            Box::pin(async move { Ok(data) })
        }
    }

    /// `IoBackend` that always fails — exercises the `Err` half of zero-fallback.
    struct ErrorIo;
    impl IoBackend for ErrorIo {
        fn submit_read(&self, _file: FileRef, _offset: u64, _len: usize, _priority: u8) -> IoTask {
            Box::pin(async move { Err(io::Error::new(io::ErrorKind::UnexpectedEof, "boom")) })
        }
    }

    /// `IoBackend`/`DecodeBackend` stubs backing the `AccessHandle` that only
    /// exists to construct a `PrefetchCancel`. The native path never drives them
    /// — the handle is used solely for `cancel_in_flight` →
    /// `send_scheduled_cancel`, a no-op on an idle scheduler.
    struct CancelIo;
    impl IoBackend for CancelIo {
        fn submit_read(&self, _file: FileRef, _offset: u64, _len: usize, _priority: u8) -> IoTask {
            Box::pin(async { Err(io::Error::new(io::ErrorKind::UnexpectedEof, "cancel-only io")) })
        }
    }

    struct CancelDecode;
    impl DecodeBackend for CancelDecode {
        fn submit_decode(
            &self,
            _codec: SharedCodec,
            _encoded: Arc<[u8]>,
            _expected_size: Option<usize>,
            _slice: Option<DecodeSlice>,
        ) -> DecodeTask {
            Box::pin(async { Err(CodecError::Unsupported { codec: "cancel-only".to_string() }) })
        }
    }

    fn blosc_codec() -> SharedCodec {
        codec_from_json_str(r#"{"id":"blosc","cname":"lz4"}"#).expect("codec")
    }

    fn native_config() -> NativeAccessConfig {
        NativeAccessConfig {
            enabled: true,
            fused_workers: 1,
            request_prefetch_batches: 1,
            request_prefetch_blocks: 4,
            memory_budget_bytes: 4096,
            response_queue_bytes_soft_limit: 1024,
            response_queue_bytes_hard_limit: 2048,
            load: NativeLoadConfig {
                scheduler_workers: 1,
                io_workers: 1,
                coalesce: NativeLoadCoalesceConfig::default(),
            },
            blosc: NativeBloscConfig::default(),
        }
    }

    fn make_ctx(io: Arc<dyn IoBackend>) -> NativeScheduledContext {
        NativeScheduledContext::new(io, native_config()).expect("native context")
    }

    /// Build a minimal `AccessHandle` + `PrefetchCancel` pair. The handle backs
    /// the cancel; the native path otherwise ignores it.
    fn make_cancel() -> (AccessHandle, Arc<PrefetchCancel>) {
        let config = AccessConfig {
            queue_capacity: 4,
            scheduler_shards: 1,
            cache_capacity_bytes: 16,
            memory_budget_bytes: 32,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: AccessCpuConfig {
                num_workers: 1,
                queue_capacity: 4,
                cpus: None,
            },
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(CancelIo), Box::new(CancelDecode))
            .expect("access handle");
        let cancel = PrefetchCancel::new(handle.clone());
        (handle, cancel)
    }

    /// Build `n` distinct whole-chunk Blosc-LZ4 items laid out contiguously in
    /// one buffer. Item `i` decodes to `[b'A'+i; 4] ++ [b'0'+i; 4]`.
    fn distinct_chunks(n: usize) -> (Arc<dyn IoBackend>, Vec<AccessItem>, Vec<Vec<u8>>) {
        let file = FileRef::new(7);
        let codec = blosc_codec();
        let mut buffer = Vec::new();
        let mut items = Vec::new();
        let mut expected = Vec::new();
        for i in 0..n {
            let block_a = vec![b'A' + i as u8; 4];
            let block_b = vec![b'0' + i as u8; 4];
            let encoded = manual_blosc_lz4_raw_blocks(&[&block_a, &block_b]);
            let offset = 1000u64 + buffer.len() as u64;
            buffer.extend_from_slice(&encoded);
            items.push(AccessItem::new(
                ChunkKey::new(file, offset, encoded.len()),
                codec.clone(),
                Some(8),
            ));
            expected.push([&block_a[..], &block_b].concat());
        }
        let io: Arc<dyn IoBackend> = Arc::new(RangeIo::new(file, 1000, buffer));
        (io, items, expected)
    }

    /// Drain a `ScheduledBatchAccess` on a worker thread with a hard timeout so
    /// a stuck native executor fails the test instead of hanging the suite.
    fn drain_timeout(scheduled: ScheduledBatchAccess, timeout: Duration) -> Vec<Vec<u8>> {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<io::Result<Vec<u8>>>>();
        std::thread::spawn(move || {
            let mut out = Vec::new();
            for item in scheduled {
                out.push(item);
            }
            let _ = tx.send(out);
        });
        rx.recv_timeout(timeout)
            .expect("native scheduled iterator did not drain in time")
            .into_iter()
            .map(|r| r.expect("native item"))
            .collect()
    }

    /// Drive a native async future on a dedicated current-thread runtime with a
    /// hard timeout. `tokio::time` is not enabled in this crate, so the timeout
    /// is enforced via an `mpsc::recv_timeout` on the test thread.
    fn run_async_timeout<F, T>(future: F, timeout: Duration, label: &str) -> T
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel::<T>();
        let label = label.to_string();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("native test runtime");
            let result = runtime.block_on(future);
            let _ = tx.send(result);
        });
        rx.recv_timeout(timeout)
            .unwrap_or_else(|_| panic!("{label} did not complete in time"))
    }

    // ----- zero-fallback: `load_native_batch` mapping -------------------------

    #[tokio::test]
    async fn load_native_batch_maps_none_to_io_error() {
        // Zero-fallback (`None` half): a sliced item whose chunk is shorter than
        // the minimal Blosc header declines to `None` inside the native loader.
        // `load_native_batch` must surface that as an `io::Error` for the slot —
        // never a silent retreat to the generic access path.
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let io = Arc::new(RangeIo::new(file, 1000, encoded.clone()));
        let ctx = make_ctx(io);
        let (_handle, cancel) = make_cancel();
        let codec = blosc_codec();
        let short = AccessItem::new(ChunkKey::new(file, 2000, 8), codec.clone(), Some(4))
            .with_slice_spec(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice"));
        let normal = AccessItem::new(ChunkKey::new(file, 1000, encoded.len()), codec, Some(8))
            .with_slice_spec(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice"));
        let batch = vec![(0_usize, short), (1, normal)];

        let out = load_native_batch(ctx, batch, cancel).await;

        assert_eq!(out.len(), 2);
        let err = out[0].1.as_ref().expect_err("None must become io::Error");
        assert!(err.to_string().contains("no result"), "{err}");
        assert_eq!(out[1].1.as_ref().expect("normal ok").as_slice(), b"abcd");
    }

    #[tokio::test]
    async fn load_native_batch_maps_decode_error_to_io_error() {
        // Zero-fallback (`Err` half): an IO failure inside the native loader
        // surfaces as `io::Error` for every item in the batch, never a silent
        // retreat.
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let ctx = make_ctx(Arc::new(ErrorIo));
        let (_handle, cancel) = make_cancel();
        let codec = blosc_codec();
        let item = AccessItem::new(ChunkKey::new(file, 1000, encoded.len()), codec, Some(8))
            .with_slice_spec(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice"));
        let batch = vec![(0_usize, item.clone()), (1, item)];

        let out = load_native_batch(ctx, batch, cancel).await;

        assert_eq!(out.len(), 2);
        for (_, result) in &out {
            let err = result.as_ref().expect_err("Err must propagate to every item");
            assert!(err.to_string().contains("boom"), "{err}");
        }
    }

    #[tokio::test]
    async fn load_native_batch_returns_decoded_bytes_in_order() {
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let io = Arc::new(RangeIo::new(file, 1000, encoded.clone()));
        let ctx = make_ctx(io);
        let (_handle, cancel) = make_cancel();
        let codec = blosc_codec();
        let item = |slice: SliceSpec| {
            AccessItem::new(ChunkKey::new(file, 1000, encoded.len()), codec.clone(), Some(8))
                .with_slice_spec(slice)
        };
        let batch = vec![
            (0_usize, item(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice"))),
            (1, item(SliceSpec::from_triples(vec![0, 4, 8]).expect("slice"))),
        ];

        let out = load_native_batch(ctx, batch, cancel).await;

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].1.as_ref().expect("ok").as_slice(), b"abcd");
        assert_eq!(out[1].1.as_ref().expect("ok").as_slice(), b"efgh");
    }

    // ----- scheduled iterator + dispatch -------------------------------------

    #[test]
    fn access_strategy_build_native_dispatches_ordered_output() {
        // `AccessStrategy::BloscLz4Native(ctx).build(...)` must dispatch to the
        // native scheduled iterator (`ScheduledBatchAccess::Native`) and emit
        // every item's decoded bytes in submission order. The native arm ignores
        // `access`; it is only the cancel's backing handle.
        let (io, items, expected) = distinct_chunks(5);
        let ctx = make_ctx(io);
        let (handle, cancel) = make_cancel();
        let access_config = ScheduledAccessConfig {
            prefetch_step: 4,
            decode_ahead_steps: 4,
            ready_ahead_steps: 4,
        };
        let strategy = AccessStrategy::BloscLz4Native(ctx);
        let scheduled = strategy
            .build(handle, items, access_config, cancel, false)
            .expect("native build");

        let got = drain_timeout(scheduled, Duration::from_secs(5));
        assert_eq!(got.len(), expected.len());
        for (got, expected) in got.iter().zip(&expected) {
            assert_eq!(got.as_slice(), expected.as_slice());
        }
    }

    #[test]
    fn load_native_items_ordered_async_returns_in_order() {
        let (io, items, expected) = distinct_chunks(5);
        let ctx = make_ctx(io);
        let (_handle, cancel) = make_cancel();
        let out = run_async_timeout(
            load_native_items_ordered_async(ctx, items, cancel),
            Duration::from_secs(5),
            "ordered async",
        )
        .expect("ordered async");
        assert_eq!(out.len(), expected.len());
        for (got, expected) in out.iter().zip(&expected) {
            assert_eq!(got.as_slice(), expected.as_slice());
        }
    }

    #[test]
    fn load_native_items_ordered_blocking_returns_in_order() {
        // The blocking ordered path submits an `Ordered` command to the native
        // executor and waits on a sync channel. Driven on a worker thread with a
        // timeout so a stuck executor fails the test instead of hanging it.
        let (io, items, expected) = distinct_chunks(5);
        let ctx = make_ctx(io);
        let (_handle, cancel) = make_cancel();
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = load_native_items_ordered_blocking(ctx, items, cancel);
            let _ = tx.send(result);
        });
        let out = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("ordered blocking did not complete in time")
            .expect("ordered blocking");
        worker.join().expect("worker thread");
        assert_eq!(out.len(), expected.len());
        for (got, expected) in out.iter().zip(&expected) {
            assert_eq!(got.as_slice(), expected.as_slice());
        }
    }

    #[test]
    fn load_native_items_ordered_async_propagates_decode_error() {
        // Zero-fallback through the ordered-async wrapper: an IO error must
        // surface as `DataBankError::Io`, not a silent empty/generic retreat.
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let ctx = make_ctx(Arc::new(ErrorIo));
        let (_handle, cancel) = make_cancel();
        let codec = blosc_codec();
        let item = AccessItem::new(ChunkKey::new(file, 1000, encoded.len()), codec, Some(8))
            .with_slice_spec(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice"));
        let items = vec![item];

        let err = run_async_timeout(
            load_native_items_ordered_async(ctx, items, cancel),
            Duration::from_secs(5),
            "ordered async",
        )
        .expect_err("IO error must propagate");
        assert!(err.to_string().contains("boom"), "{err}");
    }
}
