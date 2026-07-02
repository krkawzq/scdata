use std::io;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use tokio::sync::oneshot;
use tokio::task::JoinSet;

use crate::access::{
    AccessHandle, AccessItem, AccessRequest, IoBackend, PrefetchCancel, ScheduledAccess,
    ScheduledAccessConfig,
};

use super::super::config::{NativeAccessConfig, NativeMode};
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

    /// Borrow the native context; only valid when [`Self::is_native`] is true.
    /// Prefer `if let AccessStrategy::BloscLz4Native(ctx) = &strategy { ... }`
    /// so exhaustiveness guarantees safety, rather than `expect`.
    pub(crate) fn native_ctx(&self) -> Option<&NativeScheduledContext> {
        match self {
            Self::Generic => None,
            Self::BloscLz4Native(ctx) => Some(ctx),
        }
    }
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

pub(crate) fn build_scheduled_batch_access(
    access: AccessHandle,
    native: Option<NativeScheduledContext>,
    items: Vec<AccessItem>,
    access_config: ScheduledAccessConfig,
    native_mode: NativeMode,
    cancel: Arc<PrefetchCancel>,
    allow_small_command_grouping: bool,
) -> DataBankResult<ScheduledBatchAccess> {
    match (native_mode, native) {
        (NativeMode::Disabled, _) => {
            build_generic_scheduled_access(access, items, access_config, cancel)
        }
        (NativeMode::Auto, Some(ctx)) if !ctx.config.enabled => {
            build_generic_scheduled_access(access, items, access_config, cancel)
        }
        (NativeMode::Auto | NativeMode::Force, Some(ctx)) => NativeScheduledAccess::spawn(
            access,
            ctx,
            items,
            access_config,
            native_mode,
            cancel,
            allow_small_command_grouping,
        )
        .map(ScheduledBatchAccess::Native),
        (NativeMode::Auto, None) => {
            build_generic_scheduled_access(access, items, access_config, cancel)
        }
        (NativeMode::Force, None) => Err(DataBankError::InvalidConfig(
            "native_mode='force' requested but native access context is unavailable".to_string(),
        )),
    }
}

pub(crate) async fn load_native_items_ordered_async(
    access: AccessHandle,
    native: NativeScheduledContext,
    items: Vec<AccessItem>,
    native_mode: NativeMode,
    cancel: Arc<PrefetchCancel>,
) -> DataBankResult<Vec<Vec<u8>>> {
    let total_items = items.len();
    let batch = items.into_iter().enumerate().collect::<Vec<_>>();
    let results = load_native_batch_or_fallback(access, native, batch, native_mode, cancel).await;
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
    access: AccessHandle,
    native: NativeScheduledContext,
    items: Vec<AccessItem>,
    native_mode: NativeMode,
    cancel: Arc<PrefetchCancel>,
) -> DataBankResult<Vec<Vec<u8>>> {
    let (reply, rx) = mpsc::sync_channel(1);
    let executor = Arc::clone(&native.executor);
    executor.submit_ordered(NativeOrderedCommand {
        access,
        native,
        items,
        native_mode,
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
        access: AccessHandle,
        native: NativeScheduledContext,
        items: Vec<AccessItem>,
        access_config: ScheduledAccessConfig,
        native_mode: NativeMode,
        cancel: Arc<PrefetchCancel>,
        allow_small_command_grouping: bool,
    ) -> DataBankResult<Self> {
        let window = native_window(&native.config, access_config);
        let (tx, rx) = flume::bounded(window.max(1));
        let executor = Arc::clone(&native.executor);
        executor.submit(NativeScheduledCommand {
            access,
            native,
            items,
            native_mode,
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
    access: AccessHandle,
    native: NativeScheduledContext,
    items: Vec<AccessItem>,
    native_mode: NativeMode,
    cancel: Arc<PrefetchCancel>,
    allow_small_command_grouping: bool,
    window: usize,
    tx: flume::Sender<io::Result<Vec<u8>>>,
}

struct NativeOrderedCommand {
    access: AccessHandle,
    native: NativeScheduledContext,
    items: Vec<AccessItem>,
    native_mode: NativeMode,
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
                command.access,
                command.native,
                command.items,
                command.native_mode,
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
        access,
        native,
        items,
        native_mode,
        cancel,
        reply,
    } = command;
    let result =
        load_native_items_ordered_async(access, native, items, native_mode, Arc::clone(&cancel))
            .await;
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
    access: AccessHandle,
    native: NativeScheduledContext,
    items: Vec<AccessItem>,
    native_mode: NativeMode,
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
            let batch_size = native_item_batch_size(&native.config);
            let mut batch = Vec::with_capacity(batch_size);
            batch.push(first);
            while batch.len() < batch_size {
                let Some(next) = source.next() else {
                    source_done = true;
                    break;
                };
                batch.push(next);
            }
            let access = access.clone();
            let native = native.clone();
            let cancel = Arc::clone(&cancel);
            tasks.spawn(async move {
                load_native_batch_or_fallback(access, native, batch, native_mode, cancel).await
            });
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

async fn load_native_batch_or_fallback(
    access: AccessHandle,
    native: NativeScheduledContext,
    batch: Vec<(usize, AccessItem)>,
    native_mode: NativeMode,
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

    match native_result {
        Ok(results) => {
            let mut out = Vec::with_capacity(batch.len());
            for ((seq, item), result) in batch.into_iter().zip(results) {
                let result = match result {
                    Some(bytes) => Ok(bytes),
                    None if native_mode == NativeMode::Force => Err(io::Error::other(
                        "native_mode='force' does not support this access item",
                    )),
                    None => load_generic_access_item(access.clone(), item).await,
                };
                out.push((seq, result));
            }
            out
        }
        Err(err) if native_mode == NativeMode::Force || !native.config.fallback_to_generic => {
            let message = err.to_string();
            batch
                .into_iter()
                .map(|(seq, _)| (seq, Err(io::Error::other(message.clone()))))
                .collect()
        }
        Err(_) => {
            let mut out = Vec::with_capacity(batch.len());
            for (seq, item) in batch {
                out.push((seq, load_generic_access_item(access.clone(), item).await));
            }
            out
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
    let access = commands[0].access.clone();
    let native = commands[0].native.clone();
    let native_mode = commands[0].native_mode;
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
        let results =
            load_native_batch_or_fallback_uncancelled(access, native, batch, native_mode).await;
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
        for item_idx in 0..state.item_count {
            let result = completed[command_idx][item_idx].take().unwrap_or_else(|| {
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

async fn load_native_batch_or_fallback_uncancelled(
    access: AccessHandle,
    native: NativeScheduledContext,
    batch: Vec<(usize, AccessItem)>,
    native_mode: NativeMode,
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

    match native_result {
        Ok(results) => {
            let mut out = Vec::with_capacity(batch.len());
            for ((seq, item), result) in batch.into_iter().zip(results) {
                let result = match result {
                    Some(bytes) => Ok(bytes),
                    None if native_mode == NativeMode::Force => Err(io::Error::other(
                        "native_mode='force' does not support this access item",
                    )),
                    None => load_generic_access_item(access.clone(), item).await,
                };
                out.push((seq, result));
            }
            out
        }
        Err(err) if native_mode == NativeMode::Force || !native.config.fallback_to_generic => {
            let message = err.to_string();
            batch
                .into_iter()
                .map(|(seq, _)| (seq, Err(io::Error::other(message.clone()))))
                .collect()
        }
        Err(_) => {
            let mut out = Vec::with_capacity(batch.len());
            for (seq, item) in batch {
                out.push((seq, load_generic_access_item(access.clone(), item).await));
            }
            out
        }
    }
}

async fn load_generic_access_item(access: AccessHandle, item: AccessItem) -> io::Result<Vec<u8>> {
    let (reply, rx) = oneshot::channel();
    access
        .send_async(AccessRequest {
            key: item.key,
            codec: item.codec,
            expected_size: item.expected_size,
            slice: item.slice,
            reply,
        })
        .await
        .map_err(|err| io::Error::other(err.to_string()))?;
    rx.await
        .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "native fallback task ended"))?
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

fn native_item_batch_size(config: &NativeAccessConfig) -> usize {
    config.fused_workers.saturating_mul(4).clamp(8, 256)
}

fn native_batch_multi_item_min() -> usize {
    8
}

fn native_small_command_group_max_items(config: &NativeAccessConfig) -> usize {
    native_item_batch_size(config)
        .min(16)
        .max(native_batch_multi_item_min())
}

fn native_command_group_max_items(command: &NativeScheduledCommand) -> usize {
    if native_cross_command_grouping_enabled() {
        return std::env::var("SCDATA_NATIVE_COMMAND_GROUP_MAX_ITEMS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 1)
            .unwrap_or_else(|| native_item_batch_size(&command.native.config).saturating_mul(4))
            .clamp(native_batch_multi_item_min(), 4096);
    }
    native_small_command_group_max_items(&command.native.config)
}

fn native_cross_command_grouping_enabled() -> bool {
    std::env::var("SCDATA_NATIVE_COMMAND_GROUPING")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}
