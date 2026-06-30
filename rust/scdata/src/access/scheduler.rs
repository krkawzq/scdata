//! Sharded chunk access scheduler.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use flume::TrySendError;
use tokio::runtime::Builder;
use tokio::sync::{futures::OwnedNotified, oneshot, Notify};
use tokio::task::{JoinError, JoinSet, LocalSet};

#[cfg(test)]
use super::backend::FileRef;
use super::backend::{DecodeBackend, DecodeTask, IoBackend, IoTask};
use super::cache::{
    CacheInsertErrorKind, CachePayload, CachePayloadKind, ChunkCache, PinGuard, PinnedChunk,
};
use super::cpu::{AccessCpuConfig, AccessCpuPool};
use super::error::AccessError;
use super::inflight::{InflightTable, RegisterResult};
use super::key::{shard_for_key, ChunkKey, DecodeKey};
use super::membudget::MemBudget;
use super::profile::{
    AccessCacheHitKind, AccessCacheInsertFailureKind, AccessCommandKind, AccessCommandRejectKind,
    AccessProfile, ACCESS_DECODE_SCOPE, ACCESS_INFLIGHT_SCOPE, ACCESS_IO_SCOPE,
    ACCESS_MATERIALIZE_SCOPE, ACCESS_RESERVE_SCOPE,
};
use super::scheduled::{ScheduledStage, ScheduledStore, StagedBytes};
#[cfg(test)]
use super::slice::RangeCopy;
use super::slice::{SlicePlan, SliceSpec};
use crate::codecs::{CodecError, SharedCodec};
use crate::profile::{ProfileSnapshot, ProfileTimer};

type StateHandle = Rc<RefCell<SchedulerState>>;

static NEXT_SCHEDULED_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct SchedulerDeps {
    io: Arc<dyn IoBackend>,
    decode: Arc<dyn DecodeBackend>,
    cpu: Arc<AccessCpuPool>,
    profile: AccessProfile,
    priority: u8,
    keep_decoded: bool,
}

/// A chunk access without its reply channel.
#[derive(Debug, Clone)]
pub struct AccessItem {
    pub key: ChunkKey,
    pub codec: SharedCodec,
    pub expected_size: Option<usize>,
    /// Optional scatter-copy plan over decoded bytes.
    pub slice: SliceSpec,
}

impl AccessItem {
    pub fn new(key: ChunkKey, codec: SharedCodec, expected_size: Option<usize>) -> Self {
        Self {
            key,
            codec,
            expected_size,
            slice: SliceSpec::Full,
        }
    }

    pub fn with_slice(mut self, slice: Option<Vec<usize>>) -> Self {
        self.slice = SliceSpec::from_optional_triples_deferred(slice);
        self
    }

    pub fn with_slice_spec(mut self, slice: SliceSpec) -> Self {
        self.slice = slice;
        self
    }
}

/// One caller-submitted chunk read.
pub struct AccessRequest {
    pub key: ChunkKey,
    pub codec: SharedCodec,
    pub expected_size: Option<usize>,
    /// Optional scatter-copy plan over decoded bytes.
    pub slice: SliceSpec,
    pub reply: oneshot::Sender<io::Result<Vec<u8>>>,
}

impl AccessRequest {
    pub fn new(
        key: ChunkKey,
        codec: SharedCodec,
        expected_size: Option<usize>,
        reply: oneshot::Sender<io::Result<Vec<u8>>>,
    ) -> Self {
        Self {
            key,
            codec,
            expected_size,
            slice: SliceSpec::Full,
            reply,
        }
    }

    pub fn with_slice(mut self, slice: Option<Vec<usize>>) -> Self {
        self.slice = SliceSpec::from_optional_triples_deferred(slice);
        self
    }

    pub fn with_slice_spec(mut self, slice: SliceSpec) -> Self {
        self.slice = slice;
        self
    }

    fn into_parts(self) -> (AccessItem, oneshot::Sender<io::Result<Vec<u8>>>) {
        let Self {
            key,
            codec,
            expected_size,
            slice,
            reply,
        } = self;
        (
            AccessItem {
                key,
                codec,
                expected_size,
                slice,
            },
            reply,
        )
    }
}

/// A raw IO prefetch request.
pub struct PrefetchRequest {
    pub key: ChunkKey,
    pub reply: Option<oneshot::Sender<io::Result<()>>>,
}

impl PrefetchRequest {
    pub fn new(key: ChunkKey) -> Self {
        Self { key, reply: None }
    }

    pub fn with_reply(mut self, reply: oneshot::Sender<io::Result<()>>) -> Self {
        self.reply = Some(reply);
        self
    }
}

/// Access scheduler settings.
#[derive(Debug, Clone)]
pub struct AccessConfig {
    /// Bounded request channel capacity. Default: 1024.
    pub queue_capacity: usize,
    /// Number of independent scheduler shards. Default: 1.
    ///
    /// Each shard owns one current-thread Tokio runtime and a disjoint share of
    /// the cache and memory budgets. Requests are routed by [`ChunkKey`], so
    /// duplicate work for the same chunk is still deduplicated within a shard.
    pub scheduler_shards: usize,
    /// Cache capacity for raw and decoded resident chunk bytes. Default: 256 MiB.
    pub cache_capacity_bytes: usize,
    /// Total byte budget for cache, in-flight reads, and scheduled staging.
    pub memory_budget_bytes: usize,
    /// Default IO priority. Lower values are higher priority.
    pub default_io_priority: u8,
    /// Keep decoded chunks in the cache after first decode.
    pub keep_decoded: bool,
    /// CPU worker pool used for access-side copy, scatter-copy, and ready work.
    pub cpu: AccessCpuConfig,
    /// Optional shared profiler for access-layer events. Disabled by default
    /// unless `SCDATA_PROFILE` is enabled in the environment.
    pub profile: AccessProfile,
}

impl Default for AccessConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1024,
            scheduler_shards: 1,
            cache_capacity_bytes: 256 * 1024 * 1024,
            memory_budget_bytes: 512 * 1024 * 1024,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: AccessCpuConfig::default(),
            profile: AccessProfile::from_env(),
        }
    }
}

impl AccessConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.queue_capacity == 0 {
            return Err("queue_capacity must be greater than 0".to_string());
        }
        if self.scheduler_shards == 0 {
            return Err("scheduler_shards must be greater than 0".to_string());
        }
        if self.cache_capacity_bytes == 0 {
            return Err("cache_capacity_bytes must be greater than 0".to_string());
        }
        if self.memory_budget_bytes == 0 {
            return Err("memory_budget_bytes must be greater than 0".to_string());
        }
        if self.memory_budget_bytes < self.cache_capacity_bytes {
            return Err(
                "memory_budget_bytes must be >= cache_capacity_bytes (need in-flight headroom)"
                    .to_string(),
            );
        }
        if self.scheduler_shards > self.cache_capacity_bytes {
            return Err("scheduler_shards must not exceed cache_capacity_bytes".to_string());
        }
        if self.scheduler_shards > self.memory_budget_bytes {
            return Err("scheduler_shards must not exceed memory_budget_bytes".to_string());
        }
        self.cpu.validate()?;
        Ok(())
    }

    fn for_shard(&self, shard_idx: usize) -> Self {
        let shard_count = self.scheduler_shards;
        let mut config = self.clone();
        config.scheduler_shards = 1;
        config.cache_capacity_bytes =
            split_budget(self.cache_capacity_bytes, shard_count, shard_idx);
        config.memory_budget_bytes = split_budget(self.memory_budget_bytes, shard_count, shard_idx);
        config
    }
}

#[inline]
fn split_budget(total: usize, parts: usize, idx: usize) -> usize {
    debug_assert!(parts > 0);
    debug_assert!(idx < parts);
    let base = total / parts;
    let rem = total % parts;
    base + usize::from(idx < rem)
}

/// Look-ahead distances used by [`AccessHandle::scheduled`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledAccessConfig {
    pub prefetch_step: usize,
    pub decode_ahead_steps: usize,
    pub ready_ahead_steps: usize,
}

impl Default for ScheduledAccessConfig {
    fn default() -> Self {
        Self {
            prefetch_step: 2,
            decode_ahead_steps: 1,
            ready_ahead_steps: 0,
        }
    }
}

impl ScheduledAccessConfig {
    fn max_ahead(self) -> usize {
        self.prefetch_step
            .max(self.decode_ahead_steps)
            .max(self.ready_ahead_steps)
    }
}

/// Request submission handle.
#[derive(Debug, Clone)]
pub struct AccessHandle {
    inner: Arc<AccessHandleInner>,
}

#[derive(Debug)]
struct AccessHandleInner {
    shards: Vec<flume::Sender<AccessCommandEnvelope>>,
    profile: AccessProfile,
}

impl AccessHandle {
    /// Return the profiler shared by all access scheduler shards.
    pub fn profiler(&self) -> AccessProfile {
        self.inner.profile.clone()
    }

    /// Snapshot access scheduler profile metrics.
    pub fn profile_snapshot(&self) -> ProfileSnapshot {
        self.inner.profile.snapshot()
    }

    /// Snapshot and reset access scheduler profile metrics.
    pub fn profile_snapshot_and_reset(&self) -> ProfileSnapshot {
        self.inner.profile.snapshot_and_reset()
    }

    /// Reset access scheduler profile metrics.
    pub fn reset_profile(&self) {
        self.inner.profile.reset_metrics();
    }

    /// Submit a request, blocking while the bounded queue is full.
    pub fn send(&self, request: AccessRequest) -> Result<(), AccessError> {
        self.send_command(AccessCommand::Read(request))
    }

    /// Submit a request from async code, awaiting queue capacity.
    pub async fn send_async(&self, request: AccessRequest) -> Result<(), AccessError> {
        self.send_command_async(AccessCommand::Read(request)).await
    }

    /// Submit a request without waiting for queue capacity.
    pub fn try_send(&self, request: AccessRequest) -> Result<(), AccessError> {
        self.try_send_command(AccessCommand::Read(request))
    }

    /// Prefetch raw compressed bytes into the cache.
    pub fn prefetch(
        &self,
        key: ChunkKey,
    ) -> Result<oneshot::Receiver<io::Result<()>>, AccessError> {
        let (reply, rx) = oneshot::channel();
        self.send_prefetch(PrefetchRequest::new(key).with_reply(reply))?;
        Ok(rx)
    }

    /// Submit a prefetch request with an optional caller-provided reply.
    pub fn send_prefetch(&self, request: PrefetchRequest) -> Result<(), AccessError> {
        self.send_command(AccessCommand::Prefetch(request))
    }

    /// Build a scheduled iterator over predictable access items.
    pub fn scheduled<I>(
        &self,
        items: I,
        config: ScheduledAccessConfig,
    ) -> Result<ScheduledAccess<I::IntoIter>, AccessError>
    where
        I: IntoIterator<Item = AccessItem>,
    {
        ScheduledAccess::new(self.clone(), items.into_iter(), config)
    }

    fn send_scheduled_decode(&self, id: u64, item: AccessItem) -> Result<(), AccessError> {
        self.send_command(AccessCommand::ScheduledDecode(ScheduledDecodeRequest {
            id,
            item,
        }))
    }

    fn send_scheduled_prefetch(&self, key: ChunkKey) -> Result<(), AccessError> {
        self.send_command(AccessCommand::Prefetch(PrefetchRequest::new(key)))
    }

    fn send_scheduled_ensure_ready(&self, id: u64, item: AccessItem) -> Result<(), AccessError> {
        self.send_command(AccessCommand::ScheduledEnsureReady(
            ScheduledEnsureReadyRequest { id, item },
        ))
    }

    fn send_scheduled_take(
        &self,
        id: u64,
        item: AccessItem,
    ) -> Result<oneshot::Receiver<io::Result<Vec<u8>>>, AccessError> {
        let (reply, rx) = oneshot::channel();
        self.send_command(AccessCommand::ScheduledTake(ScheduledTakeRequest {
            id,
            item,
            reply,
        }))?;
        Ok(rx)
    }

    fn send_scheduled_cancel(&self, id: u64, key: ChunkKey) -> Result<(), AccessError> {
        self.send_command(AccessCommand::ScheduledCancel(ScheduledCancelRequest {
            id,
            key,
        }))
    }

    fn send_command(&self, command: AccessCommand) -> Result<(), AccessError> {
        let shard = self.shard_for_key(command.key());
        let envelope = AccessCommandEnvelope::new(command, &self.inner.profile);
        match self.inner.shards[shard].send(envelope) {
            Ok(()) => Ok(()),
            Err(_) => {
                self.inner
                    .profile
                    .record_command_rejected(AccessCommandRejectKind::Shutdown);
                Err(AccessError::Shutdown)
            }
        }
    }

    async fn send_command_async(&self, command: AccessCommand) -> Result<(), AccessError> {
        let shard = self.shard_for_key(command.key());
        let envelope = AccessCommandEnvelope::new(command, &self.inner.profile);
        match self.inner.shards[shard].send_async(envelope).await {
            Ok(()) => Ok(()),
            Err(_) => {
                self.inner
                    .profile
                    .record_command_rejected(AccessCommandRejectKind::Shutdown);
                Err(AccessError::Shutdown)
            }
        }
    }

    fn try_send_command(&self, command: AccessCommand) -> Result<(), AccessError> {
        let shard = self.shard_for_key(command.key());
        let envelope = AccessCommandEnvelope::new(command, &self.inner.profile);
        self.inner.shards[shard]
            .try_send(envelope)
            .map_err(|err| match err {
                TrySendError::Full(_) => {
                    self.inner
                        .profile
                        .record_command_rejected(AccessCommandRejectKind::QueueFull);
                    AccessError::QueueFull {
                        capacity: self.inner.shards[shard].capacity().unwrap_or(0),
                    }
                }
                TrySendError::Disconnected(_) => {
                    self.inner
                        .profile
                        .record_command_rejected(AccessCommandRejectKind::Shutdown);
                    AccessError::Shutdown
                }
            })
    }

    #[inline]
    fn shard_for_key(&self, key: ChunkKey) -> usize {
        shard_for_key(key, self.inner.shards.len())
    }
}

/// Shared cancellation handle for a scheduled-access consumer.
///
/// When a prefetch consumer is dropped while its producer is still blocked in
/// [`ScheduledAccess::next`]'s `blocking_recv`, the drop happens on the consumer
/// thread — the producer's `ScheduledAccess` lives on the producer thread's
/// stack and its own `Drop` cannot run yet. This handle closes that gap: the
/// producer records the id/key of the entry it is currently blocked on, and the
/// consumer's drop reads it back to issue a scheduler-side cancel, which wakes
/// the blocked `scheduled_take` task and unblocks the producer's `join`.
#[derive(Debug)]
pub struct PrefetchCancel {
    flag: AtomicBool,
    in_flight: Mutex<Option<(u64, ChunkKey)>>,
    handle: AccessHandle,
}

impl PrefetchCancel {
    /// Create a handle bound to `handle` for issuing scheduler cancels.
    pub fn new(handle: AccessHandle) -> Arc<Self> {
        Arc::new(Self {
            flag: AtomicBool::new(false),
            in_flight: Mutex::new(None),
            handle,
        })
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Mark the entry currently blocked in `blocking_recv`, so a concurrent
    /// [`cancel_in_flight`] can target it.
    fn set_in_flight(&self, id: u64, key: ChunkKey) {
        *self.in_flight_lock() = Some((id, key));
    }

    /// Clear the in-flight entry once `blocking_recv` returned.
    fn clear_in_flight(&self) {
        *self.in_flight_lock() = None;
    }

    /// Request cancellation and, if the producer is currently blocked on an
    /// entry, cancel it at the scheduler so `blocking_recv` wakes promptly.
    /// Returns the cancelled key, if any.
    pub fn cancel_in_flight(&self) -> Option<ChunkKey> {
        self.flag.store(true, Ordering::Release);
        // A poisoned lock only happens if the producer panicked while holding
        // it (the critical section is a single assignment, so this is unlikely);
        // recover the inner value rather than panicking the consumer's drop too.
        let entry = self.in_flight_lock().take();
        if let Some((id, key)) = entry {
            let _ = self.handle.send_scheduled_cancel(id, key);
            Some(key)
        } else {
            None
        }
    }

    fn in_flight_lock(&self) -> std::sync::MutexGuard<'_, Option<(u64, ChunkKey)>> {
        self.in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Blocking iterator returned by [`AccessHandle::scheduled`].
///
/// `Iterator::next` waits for the scheduled result with a blocking receive. Use
/// it from synchronous consumer code; do not call it on a Tokio runtime worker
/// thread that must keep driving other futures.
pub struct ScheduledAccess<I>
where
    I: Iterator<Item = AccessItem>,
{
    handle: AccessHandle,
    source: I,
    config: ScheduledAccessConfig,
    buffer: VecDeque<ScheduledClientEntry>,
    /// Optional shared cancel handle. When set, each `next()` records the entry
    /// it blocks on so an external `cancel_in_flight` can wake it; otherwise a
    /// dropped consumer would stall in `blocking_recv` until the IO completes.
    cancel: Option<Arc<PrefetchCancel>>,
    position: usize,
    next_index: usize,
    next_prefetch_index: usize,
    next_decode_index: usize,
    next_ready_index: usize,
    source_done: bool,
    pending_error: Option<io::Error>,
}

impl<I> ScheduledAccess<I>
where
    I: Iterator<Item = AccessItem>,
{
    fn new(
        handle: AccessHandle,
        source: I,
        config: ScheduledAccessConfig,
    ) -> Result<Self, AccessError> {
        let mut scheduled = Self {
            handle,
            source,
            config,
            buffer: VecDeque::new(),
            cancel: None,
            position: 0,
            next_index: 0,
            next_prefetch_index: 0,
            next_decode_index: 0,
            next_ready_index: 0,
            source_done: false,
            pending_error: None,
        };
        scheduled.schedule_available()?;
        Ok(scheduled)
    }

    /// Attach a shared cancel handle so external code (e.g. a prefetch
    /// consumer's `Drop`) can interrupt a `next()` blocked in `blocking_recv`.
    pub fn set_cancel_handle(&mut self, cancel: Arc<PrefetchCancel>) {
        self.cancel = Some(cancel);
    }

    fn schedule_available(&mut self) -> Result<(), AccessError> {
        self.fill_buffer();

        let io_limit = self.position.saturating_add(self.config.prefetch_step);
        let decode_limit = self.position.saturating_add(self.config.decode_ahead_steps);
        let ready_limit = self.position.saturating_add(self.config.ready_ahead_steps);

        self.schedule_ready_until(ready_limit)?;
        self.schedule_decode_until(decode_limit)?;
        self.schedule_prefetch_until(io_limit)?;

        Ok(())
    }

    fn schedule_ready_until(&mut self, limit: usize) -> Result<(), AccessError> {
        self.next_ready_index = self.next_ready_index.max(self.front_index());
        while self.next_ready_index <= limit {
            let Some(offset) = self.buffer_offset(self.next_ready_index) else {
                break;
            };
            let request = {
                let entry = &self.buffer[offset];
                (!entry.ready_sent).then(|| (entry.id, entry.item.clone()))
            };
            if let Some((id, item)) = request {
                self.handle.send_scheduled_ensure_ready(id, item)?;
                let entry = &mut self.buffer[offset];
                entry.ready_sent = true;
                entry.decode_sent = true;
                entry.prefetch_sent = true;
            }
            self.next_ready_index += 1;
        }
        Ok(())
    }

    fn schedule_decode_until(&mut self, limit: usize) -> Result<(), AccessError> {
        self.next_decode_index = self.next_decode_index.max(self.front_index());
        while self.next_decode_index <= limit {
            let Some(offset) = self.buffer_offset(self.next_decode_index) else {
                break;
            };
            let request = {
                let entry = &self.buffer[offset];
                (!entry.decode_sent).then(|| (entry.id, entry.item.clone()))
            };
            if let Some((id, item)) = request {
                self.handle.send_scheduled_decode(id, item)?;
                let entry = &mut self.buffer[offset];
                entry.decode_sent = true;
                entry.prefetch_sent = true;
            }
            self.next_decode_index += 1;
        }
        Ok(())
    }

    fn schedule_prefetch_until(&mut self, limit: usize) -> Result<(), AccessError> {
        self.next_prefetch_index = self.next_prefetch_index.max(self.front_index());
        while self.next_prefetch_index <= limit {
            let Some(offset) = self.buffer_offset(self.next_prefetch_index) else {
                break;
            };
            let request = {
                let entry = &self.buffer[offset];
                (!entry.prefetch_sent).then_some(entry.item.key)
            };
            if let Some(key) = request {
                self.handle.send_scheduled_prefetch(key)?;
                self.buffer[offset].prefetch_sent = true;
            }
            self.next_prefetch_index += 1;
        }
        Ok(())
    }

    fn front_index(&self) -> usize {
        self.buffer
            .front()
            .map_or(self.next_index, |entry| entry.index)
    }

    fn buffer_offset(&self, index: usize) -> Option<usize> {
        let front = self.buffer.front()?.index;
        let offset = index.checked_sub(front)?;
        (offset < self.buffer.len()).then_some(offset)
    }

    fn fill_buffer(&mut self) {
        if self.source_done {
            return;
        }

        let target_index = self.position.saturating_add(self.config.max_ahead());
        while self.next_index <= target_index {
            let Some(item) = self.source.next() else {
                self.source_done = true;
                break;
            };
            self.buffer.push_back(ScheduledClientEntry::new(
                self.next_index,
                item,
                NEXT_SCHEDULED_ID.fetch_add(1, Ordering::Relaxed),
            ));
            self.next_index += 1;
        }
    }
}

impl<I> Iterator for ScheduledAccess<I>
where
    I: Iterator<Item = AccessItem>,
{
    type Item = io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(err) = self.pending_error.take() {
            return Some(Err(err));
        }

        if self.buffer.is_empty() {
            return None;
        }

        if !self.buffer.front().is_some_and(|entry| entry.ready_sent) {
            let entry = self.buffer.front_mut().expect("front checked above");
            match self
                .handle
                .send_scheduled_ensure_ready(entry.id, entry.item.clone())
            {
                Ok(()) => {
                    entry.ready_sent = true;
                }
                Err(err) => return Some(Err(access_error_to_io(err))),
            }
        }

        let entry = self.buffer.pop_front().expect("front checked above");
        let id = entry.id;
        let key = entry.item.key;
        let rx = match self.handle.send_scheduled_take(id, entry.item) {
            Ok(rx) => rx,
            Err(err) => return Some(Err(access_error_to_io(err))),
        };
        // Publish the in-flight entry before blocking so a concurrent
        // `PrefetchCancel::cancel_in_flight` can target it and wake the
        // scheduler-side `scheduled_take` task (which awaits on a `Notify`).
        // Then re-check the flag: if the consumer dropped between `send` and
        // `set_in_flight`, `cancel_in_flight` could not yet see this entry, so
        // we issue the cancel ourselves to avoid blocking forever.
        if let Some(cancel) = self.cancel.as_ref() {
            cancel.set_in_flight(id, key);
            if cancel.is_cancelled() {
                let _ = self.handle.send_scheduled_cancel(id, key);
            }
        }
        let result = rx.blocking_recv().unwrap_or_else(|_| {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "ready task ended",
            ))
        });
        if let Some(cancel) = self.cancel.as_ref() {
            cancel.clear_in_flight();
        }

        self.position += 1;
        if let Err(err) = self.schedule_available() {
            self.pending_error = Some(access_error_to_io(err));
        }

        Some(result)
    }
}

struct ScheduledClientEntry {
    id: u64,
    index: usize,
    item: AccessItem,
    prefetch_sent: bool,
    decode_sent: bool,
    ready_sent: bool,
}

impl ScheduledClientEntry {
    fn new(index: usize, item: AccessItem, id: u64) -> Self {
        Self {
            id,
            index,
            item,
            prefetch_sent: false,
            decode_sent: false,
            ready_sent: false,
        }
    }
}

impl<I> Drop for ScheduledAccess<I>
where
    I: Iterator<Item = AccessItem>,
{
    fn drop(&mut self) {
        while let Some(entry) = self.buffer.pop_front() {
            let _ = self.handle.send_scheduled_cancel(entry.id, entry.item.key);
        }
        // Clear any published in-flight slot. The entry blocked in
        // `blocking_recv` is owned by `next`'s stack frame and handled by the
        // external `PrefetchCancel` handle; here we only ensure no stale entry
        // remains if the iterator is dropped between `next` calls.
        if let Some(cancel) = self.cancel.as_ref() {
            cancel.clear_in_flight();
        }
    }
}

/// Namespace for starting the access scheduler.
pub struct AccessScheduler;

impl AccessScheduler {
    /// Start the scheduler on one or more dedicated current-thread Tokio runtimes.
    pub fn spawn(
        config: AccessConfig,
        io: Box<dyn IoBackend>,
        decode: Box<dyn DecodeBackend>,
    ) -> io::Result<AccessHandle> {
        config
            .validate()
            .map_err(|msg| io::Error::new(io::ErrorKind::InvalidInput, msg))?;

        let io: Arc<dyn IoBackend> = Arc::from(io);
        let decode: Arc<dyn DecodeBackend> = Arc::from(decode);
        let cpu = Arc::new(AccessCpuPool::new(config.cpu.clone())?);
        let profile = config.profile.clone();
        let shard_count = config.scheduler_shards;
        let mut shards = Vec::with_capacity(shard_count);

        for shard_idx in 0..shard_count {
            let runtime = Builder::new_current_thread().enable_all().build()?;
            let (tx, rx) = flume::bounded(config.queue_capacity);
            let shard_config = config.for_shard(shard_idx);
            let shard_io = Arc::clone(&io);
            let shard_decode = Arc::clone(&decode);
            let shard_cpu = Arc::clone(&cpu);
            let thread_name = if shard_count == 1 {
                "scdata-access".to_string()
            } else {
                format!("scdata-access-{shard_idx}")
            };

            thread::Builder::new().name(thread_name).spawn(move || {
                let local = LocalSet::new();
                local.block_on(
                    &runtime,
                    run_scheduler(rx, shard_config, shard_io, shard_decode, shard_cpu),
                );
            })?;

            shards.push(tx);
        }

        Ok(AccessHandle {
            inner: Arc::new(AccessHandleInner { shards, profile }),
        })
    }
}

struct AccessCommandEnvelope {
    command: AccessCommand,
    queued_at: ProfileTimer,
}

impl AccessCommandEnvelope {
    fn new(command: AccessCommand, profile: &AccessProfile) -> Self {
        Self {
            command,
            queued_at: profile.command_timer(),
        }
    }
}

enum AccessCommand {
    Read(AccessRequest),
    Prefetch(PrefetchRequest),
    ScheduledDecode(ScheduledDecodeRequest),
    ScheduledEnsureReady(ScheduledEnsureReadyRequest),
    ScheduledTake(ScheduledTakeRequest),
    ScheduledCancel(ScheduledCancelRequest),
}

impl AccessCommand {
    #[inline]
    fn key(&self) -> ChunkKey {
        match self {
            Self::Read(request) => request.key,
            Self::Prefetch(request) => request.key,
            Self::ScheduledDecode(request) => request.item.key,
            Self::ScheduledEnsureReady(request) => request.item.key,
            Self::ScheduledTake(request) => request.item.key,
            Self::ScheduledCancel(request) => request.key,
        }
    }

    #[inline]
    fn profile_kind(&self) -> AccessCommandKind {
        match self {
            Self::Read(_) => AccessCommandKind::Read,
            Self::Prefetch(_) => AccessCommandKind::Prefetch,
            Self::ScheduledDecode(_) => AccessCommandKind::ScheduledDecode,
            Self::ScheduledEnsureReady(_) => AccessCommandKind::ScheduledEnsureReady,
            Self::ScheduledTake(_) => AccessCommandKind::ScheduledTake,
            Self::ScheduledCancel(_) => AccessCommandKind::ScheduledCancel,
        }
    }
}

struct ScheduledDecodeRequest {
    id: u64,
    item: AccessItem,
}

struct ScheduledEnsureReadyRequest {
    id: u64,
    item: AccessItem,
}

struct ScheduledTakeRequest {
    id: u64,
    item: AccessItem,
    reply: oneshot::Sender<io::Result<Vec<u8>>>,
}

struct ScheduledCancelRequest {
    id: u64,
    key: ChunkKey,
}

#[derive(Debug)]
struct SchedulerState {
    cache: ChunkCache,
    inflight: InflightTable,
    decode_inflight: HashMap<DecodeKey, Arc<Notify>>,
    budget: MemBudget,
    scheduled: ScheduledStore,
}

impl SchedulerState {
    fn new(config: &AccessConfig) -> Self {
        let budget = MemBudget::new(config.memory_budget_bytes);
        let cache = ChunkCache::new(config.cache_capacity_bytes, budget.release_notifier());
        Self {
            cache,
            inflight: InflightTable::new(),
            decode_inflight: HashMap::new(),
            budget,
            scheduled: ScheduledStore::new(),
        }
    }

    fn evict_staged_buffer(&mut self) -> Option<usize> {
        self.scheduled.evict_one_buffer()
    }
}

async fn run_scheduler(
    rx: flume::Receiver<AccessCommandEnvelope>,
    config: AccessConfig,
    io: Arc<dyn IoBackend>,
    decode: Arc<dyn DecodeBackend>,
    cpu: Arc<AccessCpuPool>,
) {
    let state = Rc::new(RefCell::new(SchedulerState::new(&config)));
    let deps = SchedulerDeps {
        io,
        decode,
        cpu,
        profile: config.profile.clone(),
        priority: config.default_io_priority,
        keep_decoded: config.keep_decoded,
    };
    let mut tasks = JoinSet::new();
    let mut accepting = true;

    loop {
        if accepting {
            tokio::select! {
                command = rx.recv_async() => {
                    match command {
                        Ok(command) => spawn_command(
                            &mut tasks,
                            Rc::clone(&state),
                            &deps,
                            command,
                        ),
                        Err(_) => accepting = false,
                    }
                }
                result = tasks.join_next(), if !tasks.is_empty() => {
                    log_task_result(result);
                }
            }
        } else if tasks.is_empty() {
            break;
        } else {
            log_task_result(tasks.join_next().await);
        }
    }
}

fn spawn_command(
    tasks: &mut JoinSet<()>,
    state: StateHandle,
    deps: &SchedulerDeps,
    envelope: AccessCommandEnvelope,
) {
    let AccessCommandEnvelope { command, queued_at } = envelope;
    deps.profile
        .record_command(queued_at, command.profile_kind());

    match command {
        AccessCommand::Prefetch(request) => {
            match try_prefetch_inline(&state, &deps.profile, request) {
                Ok(()) => {}
                Err(request) => {
                    tasks.spawn_local(handle_command(
                        state,
                        deps.clone(),
                        AccessCommand::Prefetch(request),
                    ));
                }
            }
        }
        AccessCommand::ScheduledDecode(request) => {
            if try_scheduled_decode_inline(&state, &deps.profile, &request) {
                return;
            }

            if let Some(notify) = begin_scheduled_pending(&state, request.id) {
                tasks.spawn_local(handle_scheduled_decode(
                    state,
                    deps.clone(),
                    request,
                    notify,
                ));
            }
        }
        AccessCommand::ScheduledEnsureReady(request) => {
            match try_scheduled_ensure_ready_inline(&state, &deps.profile, &request) {
                EnsureInline::Handled => {}
                EnsureInline::NeedsAsync => {
                    tasks.spawn_local(handle_scheduled_ensure_ready(
                        state,
                        deps.clone(),
                        request,
                        None,
                    ));
                }
                EnsureInline::Missing => {
                    let owner = begin_scheduled_pending(&state, request.id);
                    tasks.spawn_local(handle_scheduled_ensure_ready(
                        state,
                        deps.clone(),
                        request,
                        owner,
                    ));
                }
            }
        }
        AccessCommand::ScheduledTake(request) => {
            match try_scheduled_take_inline(&state, &deps.profile, request) {
                Ok(()) => {}
                Err(request) => {
                    tasks.spawn_local(handle_command(
                        state,
                        deps.clone(),
                        AccessCommand::ScheduledTake(request),
                    ));
                }
            }
        }
        AccessCommand::ScheduledCancel(request) => {
            cancel_scheduled(&state, &deps.profile, request.id);
        }
        command => {
            tasks.spawn_local(handle_command(state, deps.clone(), command));
        }
    }
}

fn try_scheduled_decode_inline(
    state: &StateHandle,
    profile: &AccessProfile,
    request: &ScheduledDecodeRequest,
) -> bool {
    let mut cache_hit = None;
    let mut decode_cached = false;
    let mut identity_decode = false;
    let mut state_ref = state.borrow_mut();
    if state_ref.scheduled.contains(&request.id) {
        return true;
    }

    let decode_key = DecodeKey::new(
        request.item.key,
        &request.item.codec,
        request.item.expected_size,
    );
    let handled = if state_ref.cache.pin_decoded(&decode_key).is_some() {
        cache_hit = Some(AccessCacheHitKind::Decoded);
        decode_cached = true;
        state_ref
            .scheduled
            .insert(request.id, ScheduledStage::Complete);
        true
    } else if let Some(pinned) = state_ref.cache.pin_raw(&request.item.key) {
        if request.item.codec.is_identity() {
            cache_hit = Some(AccessCacheHitKind::Raw);
            identity_decode = true;
            let stage = match validate_identity_decoded_size(
                &request.item.codec,
                pinned.data.len(),
                request.item.expected_size,
            ) {
                Ok(()) => ScheduledStage::Complete,
                Err(err) => ScheduledStage::Failed(err.to_string()),
            };
            state_ref.scheduled.insert(request.id, stage);
            true
        } else {
            false
        }
    } else {
        false
    };
    drop(state_ref);

    if let Some(kind) = cache_hit {
        profile.record_cache_hit(kind);
    }
    if decode_cached {
        profile.record_decode_cached();
    }
    if identity_decode {
        profile.record_identity_decode();
    }
    handled
}

fn register_decode_waiter_after_raw_load(
    state: &StateHandle,
    profile: &AccessProfile,
    item: &AccessItem,
) -> DecodeRegister {
    register_decode_waiter_inner(state, profile, item, false)
}

#[cfg(test)]
fn register_decode_waiter(state: &StateHandle, item: &AccessItem) -> DecodeRegister {
    register_decode_waiter_inner(state, &AccessProfile::disabled(), item, true)
}

fn register_decode_waiter_inner(
    state: &StateHandle,
    profile: &AccessProfile,
    item: &AccessItem,
    check_cache: bool,
) -> DecodeRegister {
    enum RegisterAction {
        Ready(PinnedChunk),
        Wait(OwnedNotified),
        First(DecodeKey),
    }

    let key = DecodeKey::new(item.key, &item.codec, item.expected_size);

    let action = {
        let mut state_ref = state.borrow_mut();
        if check_cache {
            if let Some(pinned) = state_ref.cache.pin_decoded(&key) {
                RegisterAction::Ready(pinned)
            } else if let Some(notify) = state_ref.decode_inflight.get(&key) {
                RegisterAction::Wait(Arc::clone(notify).notified_owned())
            } else {
                state_ref
                    .decode_inflight
                    .insert(key.clone(), Arc::new(Notify::new()));
                RegisterAction::First(key)
            }
        } else if let Some(notify) = state_ref.decode_inflight.get(&key) {
            RegisterAction::Wait(Arc::clone(notify).notified_owned())
        } else {
            state_ref
                .decode_inflight
                .insert(key.clone(), Arc::new(Notify::new()));
            RegisterAction::First(key)
        }
    };

    match action {
        RegisterAction::Ready(pinned) => {
            profile.record_cache_hit(AccessCacheHitKind::Decoded);
            profile.record_decode_cached();
            DecodeRegister::Ready(pinned)
        }
        RegisterAction::Wait(notify) => DecodeRegister::Wait {
            notify,
            started: profile.timer(ACCESS_DECODE_SCOPE),
        },
        RegisterAction::First(key) => {
            profile.record_decode_first();
            DecodeRegister::First(key)
        }
    }
}

enum EnsureInline {
    Handled,
    NeedsAsync,
    Missing,
}

enum InlineStagedAction {
    MoveOwned,
    Fail(AccessError),
    NeedsAsync,
}

fn inline_staged_action(data: &StagedBytes, slice: &SliceSpec) -> InlineStagedAction {
    let StagedBytes::Owned(data) = data else {
        return InlineStagedAction::NeedsAsync;
    };

    match slice.plan(data.len()) {
        Ok(plan) if plan.ranges.is_none() => InlineStagedAction::MoveOwned,
        Ok(_) => InlineStagedAction::NeedsAsync,
        Err(err) => InlineStagedAction::Fail(err),
    }
}

fn try_prefetch_inline(
    state: &StateHandle,
    profile: &AccessProfile,
    request: PrefetchRequest,
) -> Result<(), PrefetchRequest> {
    enum PrefetchInline {
        Hit,
        TooLarge,
        NeedsAsync,
    }

    let action = {
        let state_ref = state.borrow();
        if state_ref.cache.contains_raw(&request.key) {
            PrefetchInline::Hit
        } else if request.key.len > state_ref.cache.capacity() {
            PrefetchInline::TooLarge
        } else {
            PrefetchInline::NeedsAsync
        }
    };

    let result = {
        match action {
            PrefetchInline::Hit => {
                profile.record_cache_hit(AccessCacheHitKind::Raw);
                Some(Ok(()))
            }
            PrefetchInline::TooLarge => {
                profile.record_cache_miss();
                profile.record_cache_too_large();
                Some(Err(AccessError::OutOfMemory))
            }
            PrefetchInline::NeedsAsync => None,
        }
    };

    if let Some(result) = result {
        send_prefetch_reply(request.reply, result);
        Ok(())
    } else {
        Err(request)
    }
}

fn try_scheduled_ensure_ready_inline(
    state: &StateHandle,
    profile: &AccessProfile,
    request: &ScheduledEnsureReadyRequest,
) -> EnsureInline {
    let mut materialize_record = None;
    let mut state_ref = state.borrow_mut();
    let action = match state_ref.scheduled.get(&request.id) {
        None => return EnsureInline::Missing,
        Some(ScheduledStage::Ready { .. } | ScheduledStage::Failed(_)) => {
            return EnsureInline::Handled;
        }
        Some(ScheduledStage::Cancelled) => {
            state_ref.scheduled.remove(&request.id);
            return EnsureInline::Handled;
        }
        Some(ScheduledStage::Pending(_) | ScheduledStage::Complete) => {
            return EnsureInline::NeedsAsync;
        }
        Some(ScheduledStage::Decoded { data, .. }) => {
            inline_staged_action(data, &request.item.slice)
        }
    };

    match action {
        InlineStagedAction::MoveOwned => {
            if let Some(ScheduledStage::Decoded {
                data: StagedBytes::Owned(data),
                bytes,
            }) = state_ref.scheduled.remove(&request.id)
            {
                materialize_record = Some((bytes, false));
                state_ref
                    .scheduled
                    .insert(request.id, ScheduledStage::Ready { data, bytes });
            }
            drop(state_ref);
            if let Some((bytes, error)) = materialize_record {
                profile.record_materialize(ProfileTimer::disabled(), bytes, error);
            }
            EnsureInline::Handled
        }
        InlineStagedAction::Fail(err) => {
            if let Some(ScheduledStage::Decoded { bytes, .. }) =
                state_ref.scheduled.remove(&request.id)
            {
                state_ref.budget.release(bytes);
            }
            state_ref
                .scheduled
                .insert(request.id, ScheduledStage::Failed(err.to_string()));
            drop(state_ref);
            profile.record_materialize(ProfileTimer::disabled(), 0, true);
            EnsureInline::Handled
        }
        InlineStagedAction::NeedsAsync => EnsureInline::NeedsAsync,
    }
}

enum TakeInlineAction {
    Ready,
    MoveOwnedDecoded,
    Fail(AccessError),
    FailedMessage(String),
    Cancelled,
    NeedsAsync,
}

fn try_scheduled_take_inline(
    state: &StateHandle,
    profile: &AccessProfile,
    request: ScheduledTakeRequest,
) -> Result<(), ScheduledTakeRequest> {
    let mut materialize_record = None;
    let result = {
        let mut state_ref = state.borrow_mut();
        let action = match state_ref.scheduled.get(&request.id) {
            Some(ScheduledStage::Ready { .. }) => TakeInlineAction::Ready,
            Some(ScheduledStage::Decoded { data, .. }) => {
                match inline_staged_action(data, &request.item.slice) {
                    InlineStagedAction::MoveOwned => TakeInlineAction::MoveOwnedDecoded,
                    InlineStagedAction::Fail(err) => TakeInlineAction::Fail(err),
                    InlineStagedAction::NeedsAsync => TakeInlineAction::NeedsAsync,
                }
            }
            Some(ScheduledStage::Failed(message)) => {
                TakeInlineAction::FailedMessage(message.clone())
            }
            Some(ScheduledStage::Cancelled) => TakeInlineAction::Cancelled,
            Some(ScheduledStage::Pending(_) | ScheduledStage::Complete) | None => {
                TakeInlineAction::NeedsAsync
            }
        };

        match action {
            TakeInlineAction::Ready => {
                if let Some(ScheduledStage::Ready { data, bytes }) =
                    state_ref.scheduled.remove(&request.id)
                {
                    state_ref.budget.release(bytes);
                    Some(Ok(data))
                } else {
                    None
                }
            }
            TakeInlineAction::MoveOwnedDecoded => {
                if let Some(ScheduledStage::Decoded {
                    data: StagedBytes::Owned(data),
                    bytes,
                }) = state_ref.scheduled.remove(&request.id)
                {
                    state_ref.budget.release(bytes);
                    materialize_record = Some((bytes, false));
                    Some(Ok(data))
                } else {
                    None
                }
            }
            TakeInlineAction::Fail(err) => {
                if let Some(ScheduledStage::Decoded { bytes, .. }) =
                    state_ref.scheduled.remove(&request.id)
                {
                    state_ref.budget.release(bytes);
                }
                materialize_record = Some((0, true));
                Some(Err(err))
            }
            TakeInlineAction::FailedMessage(message) => {
                state_ref.scheduled.remove(&request.id);
                Some(Err(AccessError::Io(io::Error::other(message))))
            }
            TakeInlineAction::Cancelled => {
                state_ref.scheduled.remove(&request.id);
                Some(Err(AccessError::Io(io::Error::other(
                    "scheduled item cancelled",
                ))))
            }
            TakeInlineAction::NeedsAsync => None,
        }
    };

    if let Some((bytes, error)) = materialize_record {
        profile.record_materialize(ProfileTimer::disabled(), bytes, error);
    }

    if let Some(result) = result {
        send_reply(request.reply, result);
        Ok(())
    } else {
        Err(request)
    }
}

fn log_task_result(result: Option<Result<(), JoinError>>) {
    if let Some(Err(err)) = result {
        eprintln!("[access] request task failed: {err}");
    }
}

async fn handle_command(state: StateHandle, deps: SchedulerDeps, command: AccessCommand) {
    match command {
        AccessCommand::Read(request) => {
            let (item, reply) = request.into_parts();
            let result = access_item(
                state,
                deps.io.as_ref(),
                deps.decode.as_ref(),
                deps.cpu.as_ref(),
                &deps.profile,
                item,
                deps.priority,
                deps.keep_decoded,
            )
            .await;
            send_reply(reply, result);
        }
        AccessCommand::Prefetch(request) => {
            let result = prefetch_key(
                state,
                deps.io.as_ref(),
                &deps.profile,
                request.key,
                deps.priority,
            )
            .await;
            send_prefetch_reply(request.reply, result);
        }
        AccessCommand::ScheduledTake(request) => {
            let result = scheduled_take(state, &deps, request.id, request.item).await;
            send_reply(request.reply, result);
        }
        AccessCommand::ScheduledDecode(_)
        | AccessCommand::ScheduledEnsureReady(_)
        | AccessCommand::ScheduledCancel(_) => {}
    }
}

#[allow(clippy::too_many_arguments)]
async fn access_item(
    state: StateHandle,
    io: &dyn IoBackend,
    decode: &dyn DecodeBackend,
    cpu: &AccessCpuPool,
    profile: &AccessProfile,
    item: AccessItem,
    priority: u8,
    keep_decoded: bool,
) -> Result<Vec<u8>, AccessError> {
    let decoded =
        decoded_item_data(state, io, decode, profile, &item, priority, keep_decoded).await?;
    materialize_decoded(profile, cpu, decoded, &item.slice).await
}

enum DecodedOutput {
    Owned(Vec<u8>),
    Shared {
        data: Arc<[u8]>,
        _guard: Option<DecodedGuard>,
    },
}

impl DecodedOutput {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(data) => data.as_slice(),
            Self::Shared { data, .. } => data.as_ref(),
        }
    }

    fn has_budget_guard(&self) -> bool {
        matches!(
            self,
            Self::Shared {
                _guard: Some(_),
                ..
            }
        )
    }
}

enum DecodedGuard {
    Pin { _pin: PinGuard },
    Reservation { _reservation: MemoryReservation },
}

enum MaterializeSource {
    Owned(Vec<u8>),
    Shared(Arc<[u8]>),
}

async fn materialize_decoded(
    profile: &AccessProfile,
    cpu: &AccessCpuPool,
    decoded: DecodedOutput,
    slice: &SliceSpec,
) -> Result<Vec<u8>, AccessError> {
    let started = profile.timer(ACCESS_MATERIALIZE_SCOPE);
    let result = async {
        let plan = slice.plan(decoded.as_slice().len())?;
        materialize_decoded_planned(cpu, decoded, plan).await
    }
    .await;
    profile.record_materialize(
        started,
        result.as_ref().map_or(0, Vec::len),
        result.is_err(),
    );
    result
}

async fn materialize_decoded_planned(
    cpu: &AccessCpuPool,
    decoded: DecodedOutput,
    plan: SlicePlan,
) -> Result<Vec<u8>, AccessError> {
    match decoded {
        DecodedOutput::Owned(data) if plan.ranges.is_none() => Ok(data),
        DecodedOutput::Owned(data) => cpu.materialize_owned(data, plan).await,
        DecodedOutput::Shared { data, _guard } => {
            let input = Arc::clone(&data);
            let result = cpu.materialize(input, plan).await;
            drop(_guard);
            result
        }
    }
}

async fn materialize_source(
    cpu: &AccessCpuPool,
    source: MaterializeSource,
    plan: SlicePlan,
) -> Result<Vec<u8>, AccessError> {
    match source {
        MaterializeSource::Owned(data) if plan.ranges.is_none() => Ok(data),
        MaterializeSource::Owned(data) => cpu.materialize_owned(data, plan).await,
        MaterializeSource::Shared(data) => cpu.materialize(data, plan).await,
    }
}

fn into_materialize_source(decoded: DecodedOutput) -> MaterializeSource {
    match decoded {
        DecodedOutput::Owned(data) => MaterializeSource::Owned(data),
        DecodedOutput::Shared { data, _guard } => {
            drop(_guard);
            MaterializeSource::Shared(data)
        }
    }
}

async fn decoded_item_data(
    state: StateHandle,
    io: &dyn IoBackend,
    decode: &dyn DecodeBackend,
    profile: &AccessProfile,
    item: &AccessItem,
    priority: u8,
    keep_decoded: bool,
) -> Result<DecodedOutput, AccessError> {
    let decode_key = DecodeKey::new(item.key, &item.codec, item.expected_size);
    loop {
        let chunk = load_chunk(
            Rc::clone(&state),
            io,
            profile,
            item.key,
            priority,
            true,
            CacheUse::RawOrDecoded(decode_key.clone()),
        )
        .await?;

        match chunk.kind {
            CachePayloadKind::Decoded => {
                profile.record_decode_cached();
                return Ok(chunk.into_decoded_output());
            }
            CachePayloadKind::Raw => {
                if item.codec.is_identity() {
                    profile.record_identity_decode();
                    validate_identity_decoded_size(
                        &item.codec,
                        chunk.data.len(),
                        item.expected_size,
                    )?;
                    return Ok(chunk.into_decoded_output());
                }

                if !keep_decoded {
                    let started = profile.timer(ACCESS_DECODE_SCOPE);
                    let decode_task: DecodeTask = decode.submit_decode(
                        Arc::clone(&item.codec),
                        Arc::clone(&chunk.data),
                        item.expected_size,
                    );
                    let decoded = decode_task.await;
                    profile.record_decode(
                        started,
                        chunk.data.len(),
                        decoded.as_ref().ok().map(Vec::len),
                        decoded.is_err(),
                    );
                    let decoded = decoded?;
                    return Ok(DecodedOutput::Owned(decoded));
                }

                match register_decode_waiter_after_raw_load(&state, profile, item) {
                    DecodeRegister::Ready(pinned) => {
                        return Ok(DecodedOutput::Shared {
                            data: pinned.data,
                            _guard: Some(DecodedGuard::Pin { _pin: pinned.guard }),
                        });
                    }
                    DecodeRegister::Wait { notify, started } => {
                        drop(chunk);
                        notify.await;
                        profile.record_decode_waiter(started);
                    }
                    DecodeRegister::First(key) => {
                        let started = profile.timer(ACCESS_DECODE_SCOPE);
                        let decode_task: DecodeTask = decode.submit_decode(
                            Arc::clone(&item.codec),
                            Arc::clone(&chunk.data),
                            item.expected_size,
                        );
                        let decoded = decode_task.await;
                        profile.record_decode(
                            started,
                            chunk.data.len(),
                            decoded.as_ref().ok().map(Vec::len),
                            decoded.is_err(),
                        );
                        drop(chunk);

                        let decoded = match decoded {
                            Ok(decoded) => decoded,
                            Err(err) => {
                                finish_decode_waiter(&state, &key);
                                return Err(AccessError::Codec(err));
                            }
                        };

                        let decoded: Arc<[u8]> = Arc::from(decoded.into_boxed_slice());
                        let result = if let Some(pinned) = try_cache_decoded(
                            Rc::clone(&state),
                            item.key,
                            key.clone(),
                            Arc::clone(&item.codec),
                            Arc::clone(&decoded),
                            profile,
                        )
                        .await
                        {
                            Ok(DecodedOutput::Shared {
                                data: pinned.data,
                                _guard: Some(DecodedGuard::Pin { _pin: pinned.guard }),
                            })
                        } else {
                            profile.record_uncached_fallback();
                            Ok(DecodedOutput::Shared {
                                data: decoded,
                                _guard: None,
                            })
                        };
                        finish_decode_waiter(&state, &key);
                        return result;
                    }
                }
            }
        }
    }
}

enum DecodeRegister {
    Ready(PinnedChunk),
    Wait {
        notify: OwnedNotified,
        started: ProfileTimer,
    },
    First(DecodeKey),
}

fn finish_decode_waiter(state: &StateHandle, key: &DecodeKey) {
    let notify = state.borrow_mut().decode_inflight.remove(key);
    if let Some(notify) = notify {
        notify.notify_waiters();
    }
}

#[derive(Debug, Clone)]
enum CacheUse {
    RawOnly,
    RawOrDecoded(DecodeKey),
}

fn pin_cached_chunk(
    cache: &ChunkCache,
    key: &ChunkKey,
    cache_use: &CacheUse,
) -> Option<PinnedChunk> {
    match cache_use {
        CacheUse::RawOnly => cache.pin_raw(key),
        CacheUse::RawOrDecoded(decode_key) => {
            cache.pin_decoded(decode_key).or_else(|| cache.pin_raw(key))
        }
    }
}

fn validate_identity_decoded_size(
    codec: &SharedCodec,
    actual: usize,
    expected_size: Option<usize>,
) -> Result<(), AccessError> {
    if let Some(expected) = expected_size {
        if expected != actual {
            return Err(AccessError::Codec(CodecError::SizeMismatch {
                codec: codec.name().to_string(),
                expected,
                actual,
            }));
        }
    }
    Ok(())
}

enum LoadAction {
    Hit(PinnedChunk),
    Wait(OwnedNotified),
    Read,
}

struct LoadedChunk {
    data: Arc<[u8]>,
    kind: CachePayloadKind,
    _pin: Option<PinGuard>,
    _reservation: Option<MemoryReservation>,
}

impl LoadedChunk {
    fn cached(pinned: PinnedChunk) -> Self {
        Self {
            data: pinned.data,
            kind: pinned.kind,
            _pin: Some(pinned.guard),
            _reservation: None,
        }
    }

    fn uncached_raw(data: Arc<[u8]>, state: Weak<RefCell<SchedulerState>>, bytes: usize) -> Self {
        Self {
            data,
            kind: CachePayloadKind::Raw,
            _pin: None,
            _reservation: Some(MemoryReservation { state, bytes }),
        }
    }

    fn is_cached(&self) -> bool {
        self._pin.is_some()
    }

    fn into_decoded_output(self) -> DecodedOutput {
        let Self {
            data,
            _pin,
            _reservation,
            ..
        } = self;
        let _guard = match (_pin, _reservation) {
            (Some(pin), None) => Some(DecodedGuard::Pin { _pin: pin }),
            (None, Some(reservation)) => Some(DecodedGuard::Reservation {
                _reservation: reservation,
            }),
            (None, None) => None,
            (Some(_), Some(_)) => unreachable!("loaded chunk cannot be both cached and uncached"),
        };
        DecodedOutput::Shared { data, _guard }
    }

    fn into_staged_shared(self) -> (StagedBytes, usize) {
        let Self {
            data,
            _pin,
            _reservation,
            ..
        } = self;
        debug_assert!(
            _pin.is_none(),
            "cached identity chunks should complete from cache"
        );
        if let Some(reservation) = _reservation {
            reservation.commit();
        }
        let bytes = data.len();
        (StagedBytes::Shared(data), bytes)
    }
}

struct MemoryReservation {
    state: Weak<RefCell<SchedulerState>>,
    bytes: usize,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if self.bytes == 0 {
            return;
        }
        if let Some(state) = self.state.upgrade() {
            state.borrow_mut().budget.release(self.bytes);
        }
    }
}

impl MemoryReservation {
    fn commit(mut self) {
        self.bytes = 0;
    }
}

async fn load_chunk(
    state: StateHandle,
    io: &dyn IoBackend,
    profile: &AccessProfile,
    key: ChunkKey,
    priority: u8,
    allow_uncached: bool,
    cache_use: CacheUse,
) -> Result<LoadedChunk, AccessError> {
    loop {
        let mut cache_hit = None;
        let mut cache_miss = false;
        let mut inflight_first = false;
        let action = {
            let mut state = state.borrow_mut();
            if let Some(pinned) = pin_cached_chunk(&state.cache, &key, &cache_use) {
                cache_hit = Some(match pinned.kind {
                    CachePayloadKind::Raw => AccessCacheHitKind::Raw,
                    CachePayloadKind::Decoded => AccessCacheHitKind::Decoded,
                });
                LoadAction::Hit(pinned)
            } else {
                cache_miss = true;
                match state.inflight.try_register(key) {
                    RegisterResult::First => {
                        inflight_first = true;
                        LoadAction::Read
                    }
                    RegisterResult::Waiter(notify) => LoadAction::Wait(notify),
                }
            }
        };

        profile.record_load(cache_hit, cache_miss, inflight_first);

        match action {
            LoadAction::Hit(pinned) => return Ok(LoadedChunk::cached(pinned)),
            LoadAction::Wait(notify) => {
                let started = profile.timer(ACCESS_INFLIGHT_SCOPE);
                notify.await;
                profile.record_inflight_wait(started);
            }
            LoadAction::Read => {
                return read_first(state, io, profile, key, priority, allow_uncached, cache_use)
                    .await;
            }
        }
    }
}

async fn read_first(
    state: StateHandle,
    io: &dyn IoBackend,
    profile: &AccessProfile,
    key: ChunkKey,
    priority: u8,
    allow_uncached: bool,
    cache_use: CacheUse,
) -> Result<LoadedChunk, AccessError> {
    if let Err(err) = reserve_bytes(profile, Rc::clone(&state), key.len).await {
        state.borrow_mut().inflight.complete(&key);
        return Err(err);
    }

    let started = profile.timer(ACCESS_IO_SCOPE);
    let io_task: IoTask = io.submit_read(key.file, key.offset, key.len, priority);
    let data = match io_task.await {
        Ok(data) => data,
        Err(err) => {
            profile.record_io_read(started, key.len, None, true);
            finish_failed_read(&state, &key, key.len);
            return Err(AccessError::Io(err));
        }
    };

    if data.len() != key.len {
        profile.record_io_read(started, key.len, Some(data.len()), true);
        finish_failed_read(&state, &key, key.len);
        return Err(AccessError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "read {} bytes for chunk {:?}, expected {} bytes",
                data.len(),
                key,
                key.len
            ),
        )));
    }

    profile.record_io_read(started, key.len, Some(data.len()), false);
    finish_successful_raw_read(&state, profile, key, data, allow_uncached, cache_use)
}

async fn reserve_bytes(
    profile: &AccessProfile,
    state: StateHandle,
    bytes: usize,
) -> Result<(), AccessError> {
    if bytes == 0 {
        return Ok(());
    }

    let started = profile.timer(ACCESS_RESERVE_SCOPE);
    let result = reserve_bytes_inner(profile, state, bytes).await;
    profile.record_reserve(started, bytes, result.is_ok());
    result
}

async fn reserve_bytes_inner(
    profile: &AccessProfile,
    state: StateHandle,
    bytes: usize,
) -> Result<(), AccessError> {
    debug_assert!(bytes > 0);

    loop {
        let wait = {
            let mut state = state.borrow_mut();
            if bytes > state.budget.capacity() {
                return Err(AccessError::OutOfMemory);
            }

            if state.budget.try_reserve(bytes) {
                return Ok(());
            }

            while state.budget.available() < bytes {
                let Some(evicted) = state.cache.evict_one() else {
                    break;
                };
                state.budget.release(evicted.bytes);
                profile.record_cache_eviction(1, evicted.bytes);
            }

            while state.budget.available() < bytes {
                let Some(evicted_bytes) = state.evict_staged_buffer() else {
                    break;
                };
                state.budget.release(evicted_bytes);
                profile.record_staged_eviction(evicted_bytes);
            }

            if state.budget.try_reserve(bytes) {
                return Ok(());
            }

            state.budget.release_notifier().notified_owned()
        };

        wait.await;
    }
}

fn try_reserve_bytes(profile: &AccessProfile, state: &StateHandle, bytes: usize) -> bool {
    if bytes == 0 {
        return true;
    }

    let started = profile.timer(ACCESS_RESERVE_SCOPE);
    let success = try_reserve_bytes_inner(profile, state, bytes);
    profile.record_reserve(started, bytes, success);
    success
}

fn try_reserve_bytes_inner(profile: &AccessProfile, state: &StateHandle, bytes: usize) -> bool {
    debug_assert!(bytes > 0);

    let mut state = state.borrow_mut();
    if bytes > state.budget.capacity() {
        return false;
    }

    if state.budget.try_reserve(bytes) {
        return true;
    }

    while state.budget.available() < bytes {
        let Some(evicted) = state.cache.evict_one() else {
            break;
        };
        state.budget.release(evicted.bytes);
        profile.record_cache_eviction(1, evicted.bytes);
    }

    while state.budget.available() < bytes {
        let Some(evicted_bytes) = state.evict_staged_buffer() else {
            break;
        };
        state.budget.release(evicted_bytes);
        profile.record_staged_eviction(evicted_bytes);
    }

    state.budget.try_reserve(bytes)
}

fn finish_failed_read(state: &StateHandle, key: &ChunkKey, reserved_bytes: usize) {
    let mut state = state.borrow_mut();
    state.budget.release(reserved_bytes);
    state.inflight.complete(key);
}

fn finish_successful_raw_read(
    state: &StateHandle,
    profile: &AccessProfile,
    key: ChunkKey,
    data: Arc<[u8]>,
    allow_uncached: bool,
    _cache_use: CacheUse,
) -> Result<LoadedChunk, AccessError> {
    let mut insert_record = None;
    let mut failure_record = None;
    let mut uncached_fallback = false;
    let pinned = {
        let mut state_ref = state.borrow_mut();
        let pinned = match state_ref
            .cache
            .insert_if_absent(key, CachePayload::Raw(Arc::clone(&data)))
        {
            Ok(outcome) if outcome.inserted => {
                insert_record = Some((
                    outcome.evicted_count,
                    outcome.evicted_bytes,
                    outcome.replaced_bytes,
                ));
                state_ref.budget.release(outcome.evicted_bytes);
                Some(
                    state_ref
                        .cache
                        .pin_raw(&key)
                        .expect("inserted cache entry must be pinnable"),
                )
            }
            Ok(outcome) => {
                state_ref
                    .budget
                    .release(outcome.evicted_bytes.saturating_add(outcome.replaced_bytes));
                let existing = state_ref
                    .cache
                    .pin_raw(&key)
                    .expect("existing cache entry must be pinnable");
                state_ref.budget.release(key.len);
                Some(existing)
            }
            Err(err) => {
                failure_record = Some((
                    cache_insert_failure_kind(err.kind),
                    err.evicted_count,
                    err.evicted_bytes,
                ));
                state_ref.budget.release(err.evicted_bytes);
                if !allow_uncached {
                    state_ref.budget.release(key.len);
                } else {
                    uncached_fallback = true;
                }
                None
            }
        };

        state_ref.inflight.complete(&key);
        pinned
    };

    if let Some((evicted_count, evicted_bytes, replaced_bytes)) = insert_record {
        profile.record_cache_insert(evicted_count, evicted_bytes, replaced_bytes);
    }
    if let Some((kind, evicted_count, evicted_bytes)) = failure_record {
        profile.record_cache_insert_failure(kind, evicted_count, evicted_bytes);
    }
    if uncached_fallback {
        profile.record_uncached_fallback();
    }

    match pinned {
        Some(pinned) => Ok(LoadedChunk::cached(pinned)),
        None if allow_uncached => Ok(LoadedChunk::uncached_raw(
            data,
            Rc::downgrade(state),
            key.len,
        )),
        None => Err(AccessError::OutOfMemory),
    }
}

fn cache_insert_failure_kind(kind: CacheInsertErrorKind) -> AccessCacheInsertFailureKind {
    match kind {
        CacheInsertErrorKind::ItemTooLarge => AccessCacheInsertFailureKind::TooLarge,
        CacheInsertErrorKind::AllPinned => AccessCacheInsertFailureKind::AllPinned,
    }
}

async fn try_cache_decoded(
    state: StateHandle,
    key: ChunkKey,
    decode_key: DecodeKey,
    codec: SharedCodec,
    data: Arc<[u8]>,
    profile: &AccessProfile,
) -> Option<PinnedChunk> {
    if data.len() > state.borrow().cache.capacity() {
        profile.record_cache_insert_failure(AccessCacheInsertFailureKind::TooLarge, 0, 0);
        return None;
    }

    if reserve_bytes(profile, Rc::clone(&state), data.len())
        .await
        .is_err()
    {
        return None;
    }

    let mut insert_record = None;
    let mut failure_record = None;
    let pinned = {
        let mut state_ref = state.borrow_mut();
        match state_ref.cache.insert_or_replace(
            key,
            CachePayload::Decoded {
                data: Arc::clone(&data),
                decode_key: decode_key.clone(),
                _codec: codec,
            },
        ) {
            Ok(outcome) => {
                insert_record = Some((
                    outcome.evicted_count,
                    outcome.evicted_bytes,
                    outcome.replaced_bytes,
                ));
                state_ref
                    .budget
                    .release(outcome.evicted_bytes.saturating_add(outcome.replaced_bytes));
                state_ref.cache.pin_decoded(&decode_key)
            }
            Err(err) => {
                failure_record = Some((
                    cache_insert_failure_kind(err.kind),
                    err.evicted_count,
                    err.evicted_bytes,
                ));
                state_ref
                    .budget
                    .release(err.evicted_bytes.saturating_add(data.len()));
                None
            }
        }
    };

    if let Some((evicted_count, evicted_bytes, replaced_bytes)) = insert_record {
        profile.record_cache_insert(evicted_count, evicted_bytes, replaced_bytes);
    }
    if let Some((kind, evicted_count, evicted_bytes)) = failure_record {
        profile.record_cache_insert_failure(kind, evicted_count, evicted_bytes);
    }

    pinned
}

async fn prefetch_key(
    state: StateHandle,
    io: &dyn IoBackend,
    profile: &AccessProfile,
    key: ChunkKey,
    priority: u8,
) -> Result<(), AccessError> {
    if state.borrow().cache.contains_raw(&key) {
        profile.record_cache_hit(AccessCacheHitKind::Raw);
        return Ok(());
    }
    if key.len > state.borrow().cache.capacity() {
        profile.record_cache_miss();
        profile.record_cache_too_large();
        return Err(AccessError::OutOfMemory);
    }

    let chunk = load_chunk(state, io, profile, key, priority, false, CacheUse::RawOnly).await?;
    if chunk.kind == CachePayloadKind::Raw {
        Ok(())
    } else {
        Err(AccessError::OutOfMemory)
    }
}

fn begin_scheduled_pending(state: &StateHandle, id: u64) -> Option<Arc<Notify>> {
    let mut state_ref = state.borrow_mut();
    if state_ref.scheduled.contains(&id) {
        return None;
    }

    let notify = Arc::new(Notify::new());
    state_ref
        .scheduled
        .insert(id, ScheduledStage::Pending(Arc::clone(&notify)));
    Some(notify)
}

fn cancel_scheduled(state: &StateHandle, profile: &AccessProfile, id: u64) {
    let mut wake = None;
    let mut cancelled = false;
    let mut state_ref = state.borrow_mut();
    match state_ref.scheduled.remove(&id) {
        Some(ScheduledStage::Pending(pending_notify)) => {
            cancelled = true;
            state_ref.scheduled.insert(id, ScheduledStage::Cancelled);
            wake = Some(pending_notify);
        }
        Some(ScheduledStage::Decoded { bytes, .. }) | Some(ScheduledStage::Ready { bytes, .. }) => {
            cancelled = true;
            state_ref.budget.release(bytes);
        }
        Some(ScheduledStage::Complete)
        | Some(ScheduledStage::Failed(_))
        | Some(ScheduledStage::Cancelled)
        | None => {}
    }
    drop(state_ref);

    if cancelled {
        profile.record_scheduled_cancelled();
    }
    if let Some(notify) = wake {
        notify.notify_waiters();
    }
}

async fn handle_scheduled_decode(
    state: StateHandle,
    deps: SchedulerDeps,
    request: ScheduledDecodeRequest,
    notify: Arc<Notify>,
) {
    let result = decode_for_schedule(
        Rc::clone(&state),
        deps.io.as_ref(),
        deps.decode.as_ref(),
        &deps.profile,
        request.item,
        deps.priority,
        deps.keep_decoded,
    )
    .await;

    finish_scheduled_decode(&state, &deps.profile, request.id, result);
    notify.notify_waiters();
}

fn finish_scheduled_decode(
    state: &StateHandle,
    profile: &AccessProfile,
    id: u64,
    result: Result<StageDecodeOutput, AccessError>,
) {
    let mut staged_bytes = None;
    let mut state_ref = state.borrow_mut();
    if matches!(
        state_ref.scheduled.get(&id),
        Some(ScheduledStage::Cancelled)
    ) {
        if let Ok(StageDecodeOutput::Staged { bytes, .. }) = result {
            state_ref.budget.release(bytes);
        }
        state_ref.scheduled.remove(&id);
        return;
    }

    state_ref.scheduled.insert(
        id,
        match result {
            Ok(StageDecodeOutput::Cached) => ScheduledStage::Complete,
            Ok(StageDecodeOutput::Staged { data, bytes }) => {
                staged_bytes = Some(bytes);
                ScheduledStage::Decoded { data, bytes }
            }
            Err(err) => ScheduledStage::Failed(err.to_string()),
        },
    );
    drop(state_ref);

    if let Some(bytes) = staged_bytes {
        profile.record_staged_bytes(bytes);
    }
}

enum StageDecodeOutput {
    Cached,
    Staged { data: StagedBytes, bytes: usize },
}

async fn decode_for_schedule(
    state: StateHandle,
    io: &dyn IoBackend,
    decode: &dyn DecodeBackend,
    profile: &AccessProfile,
    item: AccessItem,
    priority: u8,
    keep_decoded: bool,
) -> Result<StageDecodeOutput, AccessError> {
    let decode_key = DecodeKey::new(item.key, &item.codec, item.expected_size);

    loop {
        let chunk = load_chunk(
            Rc::clone(&state),
            io,
            profile,
            item.key,
            priority,
            true,
            CacheUse::RawOrDecoded(decode_key.clone()),
        )
        .await?;
        if chunk.kind == CachePayloadKind::Decoded {
            profile.record_decode_cached();
            return Ok(StageDecodeOutput::Cached);
        }
        if item.codec.is_identity() {
            profile.record_identity_decode();
            validate_identity_decoded_size(&item.codec, chunk.data.len(), item.expected_size)?;
            if chunk.is_cached() {
                return Ok(StageDecodeOutput::Cached);
            }
            let (data, bytes) = chunk.into_staged_shared();
            return Ok(StageDecodeOutput::Staged { data, bytes });
        }

        if !keep_decoded {
            let started = profile.timer(ACCESS_DECODE_SCOPE);
            let decode_task: DecodeTask = decode.submit_decode(
                Arc::clone(&item.codec),
                Arc::clone(&chunk.data),
                item.expected_size,
            );
            let decoded = decode_task.await;
            profile.record_decode(
                started,
                chunk.data.len(),
                decoded.as_ref().ok().map(Vec::len),
                decoded.is_err(),
            );
            let decoded = decoded?;
            drop(chunk);

            let bytes = decoded.len();
            reserve_bytes(profile, Rc::clone(&state), bytes).await?;
            return Ok(StageDecodeOutput::Staged {
                data: StagedBytes::Owned(decoded),
                bytes,
            });
        }

        match register_decode_waiter_after_raw_load(&state, profile, &item) {
            DecodeRegister::Ready(_) => return Ok(StageDecodeOutput::Cached),
            DecodeRegister::Wait { notify, started } => {
                drop(chunk);
                notify.await;
                profile.record_decode_waiter(started);
                continue;
            }
            DecodeRegister::First(key) => {
                let started = profile.timer(ACCESS_DECODE_SCOPE);
                let decode_task: DecodeTask = decode.submit_decode(
                    Arc::clone(&item.codec),
                    Arc::clone(&chunk.data),
                    item.expected_size,
                );
                let decoded = decode_task.await;
                profile.record_decode(
                    started,
                    chunk.data.len(),
                    decoded.as_ref().ok().map(Vec::len),
                    decoded.is_err(),
                );
                drop(chunk);

                let decoded = match decoded {
                    Ok(decoded) => decoded,
                    Err(err) => {
                        finish_decode_waiter(&state, &key);
                        return Err(AccessError::Codec(err));
                    }
                };

                let decoded: Arc<[u8]> = Arc::from(decoded.into_boxed_slice());
                let output = if try_cache_decoded(
                    Rc::clone(&state),
                    item.key,
                    key.clone(),
                    Arc::clone(&item.codec),
                    Arc::clone(&decoded),
                    profile,
                )
                .await
                .is_some()
                {
                    Ok(StageDecodeOutput::Cached)
                } else {
                    profile.record_uncached_fallback();
                    reserve_bytes(profile, Rc::clone(&state), decoded.len())
                        .await
                        .map(|()| StageDecodeOutput::Staged {
                            bytes: decoded.len(),
                            data: StagedBytes::Shared(decoded),
                        })
                };
                finish_decode_waiter(&state, &key);
                return output;
            }
        }
    }
}

enum ReadyAction {
    Wait(OwnedNotified),
    ConvertDecoded {
        data: StagedBytes,
        bytes: usize,
        notify: Arc<Notify>,
    },
    TakeDecoded {
        data: StagedBytes,
        bytes: usize,
    },
    PrepareDirect {
        notify: Arc<Notify>,
    },
    Direct,
    Done,
    Failed(String),
}

async fn handle_scheduled_ensure_ready(
    state: StateHandle,
    deps: SchedulerDeps,
    request: ScheduledEnsureReadyRequest,
    owner: Option<Arc<Notify>>,
) {
    if let Some(notify) = owner {
        let result = match decode_for_schedule(
            Rc::clone(&state),
            deps.io.as_ref(),
            deps.decode.as_ref(),
            &deps.profile,
            request.item.clone(),
            deps.priority,
            deps.keep_decoded,
        )
        .await
        {
            Ok(output) => {
                async_ready_from_decode_output(
                    Rc::clone(&state),
                    &deps,
                    request.item.clone(),
                    output,
                )
                .await
            }
            Err(err) => Err(err),
        };

        finish_scheduled_ready(&state, &deps.profile, request.id, result, 0);
        notify.notify_waiters();
        return;
    }

    loop {
        let action = {
            let mut state_ref = state.borrow_mut();
            match state_ref.scheduled.remove(&request.id) {
                Some(ScheduledStage::Pending(notify)) => {
                    let wait = Arc::clone(&notify).notified_owned();
                    state_ref
                        .scheduled
                        .insert(request.id, ScheduledStage::Pending(Arc::clone(&notify)));
                    ReadyAction::Wait(wait)
                }
                Some(ScheduledStage::Decoded { data, bytes }) => {
                    let notify = Arc::new(Notify::new());
                    state_ref
                        .scheduled
                        .insert(request.id, ScheduledStage::Pending(Arc::clone(&notify)));
                    ReadyAction::ConvertDecoded {
                        data,
                        bytes,
                        notify,
                    }
                }
                Some(ScheduledStage::Ready { data, bytes }) => {
                    state_ref
                        .scheduled
                        .insert(request.id, ScheduledStage::Ready { data, bytes });
                    ReadyAction::Done
                }
                Some(ScheduledStage::Complete) => {
                    let notify = Arc::new(Notify::new());
                    state_ref
                        .scheduled
                        .insert(request.id, ScheduledStage::Pending(Arc::clone(&notify)));
                    ReadyAction::PrepareDirect { notify }
                }
                Some(ScheduledStage::Cancelled) | None => ReadyAction::Done,
                Some(ScheduledStage::Failed(message)) => {
                    state_ref
                        .scheduled
                        .insert(request.id, ScheduledStage::Failed(message.clone()));
                    ReadyAction::Failed(message)
                }
            }
        };

        match action {
            ReadyAction::Wait(wait) => wait.await,
            ReadyAction::ConvertDecoded {
                data,
                bytes,
                notify,
            } => {
                let result = make_ready_from_staged_data(
                    Rc::clone(&state),
                    deps.cpu.as_ref(),
                    &deps.profile,
                    data,
                    &request.item.slice,
                    bytes,
                )
                .await;
                finish_scheduled_ready(&state, &deps.profile, request.id, result, 0);
                notify.notify_waiters();
                return;
            }
            ReadyAction::PrepareDirect { notify } => {
                let result = make_ready_direct(Rc::clone(&state), &deps, request.item).await;
                finish_scheduled_ready(&state, &deps.profile, request.id, result, 0);
                notify.notify_waiters();
                return;
            }
            ReadyAction::Done | ReadyAction::Failed(_) => return,
            ReadyAction::Direct => unreachable!(),
            ReadyAction::TakeDecoded { .. } => unreachable!(),
        }
    }
}

async fn async_ready_from_decode_output(
    state: StateHandle,
    deps: &SchedulerDeps,
    item: AccessItem,
    output: StageDecodeOutput,
) -> Result<ReadyBuffer, AccessError> {
    match output {
        StageDecodeOutput::Cached => make_ready_direct(state, deps, item).await,
        StageDecodeOutput::Staged { data, bytes } => {
            make_ready_from_staged_data(
                state,
                deps.cpu.as_ref(),
                &deps.profile,
                data,
                &item.slice,
                bytes,
            )
            .await
        }
    }
}

fn finish_scheduled_ready(
    state: &StateHandle,
    profile: &AccessProfile,
    id: u64,
    result: Result<ReadyBuffer, AccessError>,
    release_staged_bytes: usize,
) {
    let mut staged_bytes = None;
    let mut state_ref = state.borrow_mut();
    if release_staged_bytes > 0 {
        state_ref.budget.release(release_staged_bytes);
    }

    if matches!(
        state_ref.scheduled.get(&id),
        Some(ScheduledStage::Cancelled)
    ) {
        if let Ok(buffer) = result {
            state_ref.budget.release(buffer.bytes);
        }
        state_ref.scheduled.remove(&id);
        return;
    }

    state_ref.scheduled.insert(
        id,
        match result {
            Ok(buffer) => {
                staged_bytes = Some(buffer.bytes);
                ScheduledStage::Ready {
                    data: buffer.data,
                    bytes: buffer.bytes,
                }
            }
            Err(err) => ScheduledStage::Failed(err.to_string()),
        },
    );
    drop(state_ref);

    if let Some(bytes) = staged_bytes {
        profile.record_staged_bytes(bytes);
    }
}

struct ReadyBuffer {
    data: Vec<u8>,
    bytes: usize,
}

async fn make_ready_direct(
    state: StateHandle,
    deps: &SchedulerDeps,
    item: AccessItem,
) -> Result<ReadyBuffer, AccessError> {
    let decoded = decoded_item_data(
        Rc::clone(&state),
        deps.io.as_ref(),
        deps.decode.as_ref(),
        &deps.profile,
        &item,
        deps.priority,
        deps.keep_decoded,
    )
    .await?;
    let materialize_started = deps.profile.timer(ACCESS_MATERIALIZE_SCOPE);
    let plan = match item.slice.plan(decoded.as_slice().len()) {
        Ok(plan) => plan,
        Err(err) => {
            deps.profile
                .record_materialize(materialize_started, 0, true);
            return Err(err);
        }
    };
    let bytes = plan.output_len;

    if decoded.has_budget_guard() && try_reserve_bytes(&deps.profile, &state, bytes) {
        let data = match materialize_decoded_planned(deps.cpu.as_ref(), decoded, plan).await {
            Ok(data) => {
                deps.profile
                    .record_materialize(materialize_started, data.len(), false);
                data
            }
            Err(err) => {
                state.borrow_mut().budget.release(bytes);
                deps.profile
                    .record_materialize(materialize_started, 0, true);
                return Err(err);
            }
        };
        return Ok(ReadyBuffer { data, bytes });
    }

    let source_bytes = match &decoded {
        DecodedOutput::Owned(data) if plan.ranges.is_some() => data.len(),
        DecodedOutput::Owned(_) => 0,
        DecodedOutput::Shared { data, .. } => data.len(),
    };
    let peak_bytes = bytes
        .checked_add(source_bytes)
        .ok_or(AccessError::OutOfMemory)?;
    if peak_bytes > state.borrow().budget.capacity() {
        return Err(AccessError::OutOfMemory);
    }
    let source = into_materialize_source(decoded);

    reserve_bytes(&deps.profile, Rc::clone(&state), source_bytes).await?;
    if let Err(err) = reserve_bytes(&deps.profile, Rc::clone(&state), bytes).await {
        state.borrow_mut().budget.release(source_bytes);
        return Err(err);
    }

    let data = match materialize_source(deps.cpu.as_ref(), source, plan).await {
        Ok(data) => {
            deps.profile
                .record_materialize(materialize_started, data.len(), false);
            data
        }
        Err(err) => {
            state
                .borrow_mut()
                .budget
                .release(source_bytes.saturating_add(bytes));
            deps.profile
                .record_materialize(materialize_started, 0, true);
            return Err(err);
        }
    };
    if source_bytes > 0 {
        state.borrow_mut().budget.release(source_bytes);
    }
    Ok(ReadyBuffer { data, bytes })
}

async fn make_ready_from_staged_data(
    state: StateHandle,
    cpu: &AccessCpuPool,
    profile: &AccessProfile,
    data: StagedBytes,
    slice: &SliceSpec,
    staged_bytes: usize,
) -> Result<ReadyBuffer, AccessError> {
    let materialize_started = profile.timer(ACCESS_MATERIALIZE_SCOPE);
    let plan = match slice.plan(data.as_slice().len()) {
        Ok(plan) => plan,
        Err(err) => {
            state.borrow_mut().budget.release(staged_bytes);
            profile.record_materialize(materialize_started, 0, true);
            return Err(err);
        }
    };
    let bytes = plan.output_len;

    if plan.ranges.is_none() {
        match data {
            StagedBytes::Owned(data) => {
                profile.record_materialize(materialize_started, data.len(), false);
                return Ok(ReadyBuffer { data, bytes });
            }
            StagedBytes::Shared(data) => {
                let peak_fits = match staged_bytes.checked_add(bytes) {
                    Some(peak) => peak <= state.borrow().budget.capacity(),
                    None => false,
                };
                if !peak_fits {
                    state.borrow_mut().budget.release(staged_bytes);
                    profile.record_materialize(materialize_started, 0, true);
                    return Err(AccessError::OutOfMemory);
                }

                if let Err(err) = reserve_bytes(profile, Rc::clone(&state), bytes).await {
                    state.borrow_mut().budget.release(staged_bytes);
                    profile.record_materialize(materialize_started, 0, true);
                    return Err(err);
                }

                let data = match cpu.materialize(data, plan).await {
                    Ok(data) => {
                        profile.record_materialize(materialize_started, data.len(), false);
                        data
                    }
                    Err(err) => {
                        state
                            .borrow_mut()
                            .budget
                            .release(staged_bytes.saturating_add(bytes));
                        profile.record_materialize(materialize_started, 0, true);
                        return Err(err);
                    }
                };
                state.borrow_mut().budget.release(staged_bytes);
                return Ok(ReadyBuffer { data, bytes });
            }
        }
    }

    let peak_fits = match staged_bytes.checked_add(bytes) {
        Some(peak) => peak <= state.borrow().budget.capacity(),
        None => false,
    };
    if !peak_fits {
        state.borrow_mut().budget.release(staged_bytes);
        profile.record_materialize(materialize_started, 0, true);
        return Err(AccessError::OutOfMemory);
    }

    if let Err(err) = reserve_bytes(profile, Rc::clone(&state), bytes).await {
        state.borrow_mut().budget.release(staged_bytes);
        profile.record_materialize(materialize_started, 0, true);
        return Err(err);
    }

    let result = match data {
        StagedBytes::Owned(data) => cpu.materialize_owned(data, plan).await,
        StagedBytes::Shared(data) => cpu.materialize(data, plan).await,
    };
    let data = match result {
        Ok(data) => {
            profile.record_materialize(materialize_started, data.len(), false);
            data
        }
        Err(err) => {
            state
                .borrow_mut()
                .budget
                .release(staged_bytes.saturating_add(bytes));
            profile.record_materialize(materialize_started, 0, true);
            return Err(err);
        }
    };
    state.borrow_mut().budget.release(staged_bytes);
    Ok(ReadyBuffer { data, bytes })
}

async fn scheduled_take(
    state: StateHandle,
    deps: &SchedulerDeps,
    id: u64,
    item: AccessItem,
) -> Result<Vec<u8>, AccessError> {
    loop {
        let action = {
            let mut state_ref = state.borrow_mut();
            match state_ref.scheduled.remove(&id) {
                Some(ScheduledStage::Pending(notify)) => {
                    let wait = Arc::clone(&notify).notified_owned();
                    state_ref
                        .scheduled
                        .insert(id, ScheduledStage::Pending(Arc::clone(&notify)));
                    ReadyAction::Wait(wait)
                }
                Some(ScheduledStage::Ready { data, bytes }) => {
                    state_ref.budget.release(bytes);
                    return Ok(data);
                }
                Some(ScheduledStage::Decoded { data, bytes }) => {
                    ReadyAction::TakeDecoded { data, bytes }
                }
                Some(ScheduledStage::Complete) | None => ReadyAction::Direct,
                Some(ScheduledStage::Failed(message)) => ReadyAction::Failed(message),
                Some(ScheduledStage::Cancelled) => {
                    ReadyAction::Failed("scheduled item cancelled".to_string())
                }
            }
        };

        match action {
            ReadyAction::Wait(wait) => wait.await,
            ReadyAction::Direct => {
                return access_item(
                    state,
                    deps.io.as_ref(),
                    deps.decode.as_ref(),
                    deps.cpu.as_ref(),
                    &deps.profile,
                    item,
                    deps.priority,
                    deps.keep_decoded,
                )
                .await;
            }
            ReadyAction::TakeDecoded { data, bytes } => {
                let buffer = make_ready_from_staged_data(
                    Rc::clone(&state),
                    deps.cpu.as_ref(),
                    &deps.profile,
                    data,
                    &item.slice,
                    bytes,
                )
                .await?;
                state.borrow_mut().budget.release(buffer.bytes);
                return Ok(buffer.data);
            }
            ReadyAction::Failed(message) => {
                return Err(AccessError::Io(io::Error::other(message)));
            }
            ReadyAction::ConvertDecoded { .. }
            | ReadyAction::PrepareDirect { .. }
            | ReadyAction::Done => {
                unreachable!()
            }
        }
    }
}

#[cfg(test)]
fn copy_slices(data: &[u8], slice: Option<&[usize]>) -> Result<Vec<u8>, AccessError> {
    let spec = slice_spec_from_borrowed_triples(slice)?;
    let plan = spec.plan(data.len())?;
    Ok(copy_slices_prevalidated(
        data,
        plan.ranges(),
        plan.output_len,
    ))
}

#[cfg(test)]
fn copy_slices_prevalidated(
    data: &[u8],
    slice: Option<&[RangeCopy]>,
    output_len: usize,
) -> Vec<u8> {
    let Some(slice) = slice else {
        return data.to_vec();
    };

    let mut out = vec![0; output_len];
    for range in slice {
        out[range.dst_offset..range.dst_offset + range.len()]
            .copy_from_slice(&data[range.src_start..range.src_end]);
    }
    out
}

#[cfg(test)]
fn slice_spec_from_borrowed_triples(slice: Option<&[usize]>) -> Result<SliceSpec, AccessError> {
    SliceSpec::from_optional_triples(slice.map(Vec::from))
}

fn send_reply(reply: oneshot::Sender<io::Result<Vec<u8>>>, result: Result<Vec<u8>, AccessError>) {
    let _ = reply.send(result.map_err(access_error_to_io));
}

fn send_prefetch_reply(
    reply: Option<oneshot::Sender<io::Result<()>>>,
    result: Result<(), AccessError>,
) {
    if let Some(reply) = reply {
        let _ = reply.send(result.map_err(access_error_to_io));
    }
}

fn access_error_to_io(err: AccessError) -> io::Error {
    io::Error::other(err.to_string())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "profile")]
    use super::super::profile::{
        access_profile_registry, test_metrics, ACCESS_CACHE_SCOPE, ACCESS_IO_SCOPE,
    };
    use super::*;
    use crate::codecs::{ChunkCodec, CodecError, CodecResult, SharedCodec, UncompressedCodec};
    #[cfg(feature = "profile")]
    use crate::profile::{ProfileConfig, ProfileRegistry, ProfileRuntime};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    #[derive(Debug)]
    struct StaticIo {
        payload: Arc<[u8]>,
        reads: Arc<AtomicUsize>,
        gate: Mutex<Option<oneshot::Receiver<()>>>,
    }

    impl StaticIo {
        fn new(payload: &'static [u8]) -> Self {
            Self {
                payload: Arc::from(payload),
                reads: Arc::new(AtomicUsize::new(0)),
                gate: Mutex::new(None),
            }
        }

        fn counted(payload: &'static [u8]) -> (Self, Arc<AtomicUsize>) {
            let io = Self::new(payload);
            let reads = Arc::clone(&io.reads);
            (io, reads)
        }

        fn gated(payload: &'static [u8]) -> (Self, oneshot::Sender<()>, Arc<AtomicUsize>) {
            let (tx, rx) = oneshot::channel();
            let reads = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    payload: Arc::from(payload),
                    reads: Arc::clone(&reads),
                    gate: Mutex::new(Some(rx)),
                },
                tx,
                reads,
            )
        }
    }

    impl IoBackend for StaticIo {
        fn submit_read(&self, _file: FileRef, _offset: u64, _len: usize, _priority: u8) -> IoTask {
            self.reads.fetch_add(1, Ordering::SeqCst);
            let payload = Arc::clone(&self.payload);
            let gate = self.gate.lock().expect("gate lock").take();
            Box::pin(async move {
                if let Some(gate) = gate {
                    let _ = gate.await;
                }
                Ok(payload)
            })
        }
    }

    #[derive(Debug)]
    struct IdentityDecode {
        decodes: Arc<AtomicUsize>,
    }

    impl IdentityDecode {
        fn new() -> Self {
            Self {
                decodes: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn counted() -> (Self, Arc<AtomicUsize>) {
            let decode = Self::new();
            let decodes = Arc::clone(&decode.decodes);
            (decode, decodes)
        }
    }

    impl DecodeBackend for IdentityDecode {
        fn submit_decode(
            &self,
            _codec: SharedCodec,
            encoded: Arc<[u8]>,
            expected_size: Option<usize>,
        ) -> DecodeTask {
            self.decodes.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if let Some(expected) = expected_size {
                    if encoded.len() != expected {
                        return Err(CodecError::SizeMismatch {
                            codec: "identity".to_string(),
                            expected,
                            actual: encoded.len(),
                        });
                    }
                }
                Ok(encoded.to_vec())
            })
        }
    }

    #[derive(Debug)]
    struct CodecBackedDecode {
        decodes: Arc<AtomicUsize>,
    }

    impl CodecBackedDecode {
        fn counted() -> (Self, Arc<AtomicUsize>) {
            let decodes = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    decodes: Arc::clone(&decodes),
                },
                decodes,
            )
        }
    }

    impl DecodeBackend for CodecBackedDecode {
        fn submit_decode(
            &self,
            codec: SharedCodec,
            encoded: Arc<[u8]>,
            expected_size: Option<usize>,
        ) -> DecodeTask {
            self.decodes.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { codec.decode(&encoded, expected_size) })
        }
    }

    #[derive(Debug)]
    struct PrefixCodec {
        prefix: u8,
    }

    impl crate::codecs::sealed::Sealed for PrefixCodec {}

    impl ChunkCodec for PrefixCodec {
        fn name(&self) -> &str {
            "prefix"
        }

        fn cache_key(&self) -> crate::codecs::CodecCacheKey {
            crate::codecs::CodecCacheKey::Pipeline(vec![
                crate::codecs::CodecCacheKey::Static("test-prefix"),
                crate::codecs::CodecCacheKey::Unsupported(format!("prefix-byte:{}", self.prefix)),
            ])
        }

        fn decode(&self, encoded: &[u8], _expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
            let mut decoded = Vec::with_capacity(encoded.len() + 1);
            decoded.push(self.prefix);
            decoded.extend_from_slice(encoded);
            Ok(decoded)
        }
    }

    #[derive(Debug)]
    struct ExpandCodec {
        byte: u8,
        output_len: usize,
    }

    impl crate::codecs::sealed::Sealed for ExpandCodec {}

    impl ChunkCodec for ExpandCodec {
        fn name(&self) -> &str {
            "expand"
        }

        fn decode(&self, _encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
            if let Some(expected) = expected_size {
                if expected != self.output_len {
                    return Err(CodecError::SizeMismatch {
                        codec: self.name().to_string(),
                        expected,
                        actual: self.output_len,
                    });
                }
            }
            Ok(vec![self.byte; self.output_len])
        }
    }

    fn test_config() -> AccessConfig {
        AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 64,
            memory_budget_bytes: 128,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        }
    }

    fn test_cpu_config() -> AccessCpuConfig {
        AccessCpuConfig {
            num_workers: 2,
            queue_capacity: 32,
            cpus: None,
        }
    }

    fn request(key: ChunkKey) -> (AccessRequest, oneshot::Receiver<io::Result<Vec<u8>>>) {
        request_with_codec(key, Arc::new(UncompressedCodec), Some(key.len))
    }

    fn request_with_codec(
        key: ChunkKey,
        codec: SharedCodec,
        expected_size: Option<usize>,
    ) -> (AccessRequest, oneshot::Receiver<io::Result<Vec<u8>>>) {
        let (tx, rx) = oneshot::channel();
        (AccessRequest::new(key, codec, expected_size, tx), rx)
    }

    fn recv_reply(rx: oneshot::Receiver<io::Result<Vec<u8>>>) -> io::Result<Vec<u8>> {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(async { rx.await.expect("reply") })
    }

    fn recv_reply_timeout(
        rx: oneshot::Receiver<io::Result<Vec<u8>>>,
        timeout: Duration,
    ) -> io::Result<Vec<u8>> {
        let (tx, done) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(recv_reply(rx));
        });
        done.recv_timeout(timeout).unwrap_or_else(|_| {
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for reply",
            ))
        })
    }

    fn recv_prefetch(rx: oneshot::Receiver<io::Result<()>>) -> io::Result<()> {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(async { rx.await.expect("prefetch reply") })
    }

    fn recv_scheduled_next_timeout<I>(
        mut scheduled: ScheduledAccess<I>,
        timeout: Duration,
    ) -> Option<io::Result<Vec<u8>>>
    where
        I: Iterator<Item = AccessItem> + Send + 'static,
    {
        let (tx, done) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(scheduled.next());
        });
        done.recv_timeout(timeout).unwrap_or_else(|_| {
            Some(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for scheduled item",
            )))
        })
    }

    fn wait_for_reads(reads: &AtomicUsize, expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(1);
        while reads.load(Ordering::SeqCst) < expected {
            assert!(Instant::now() < deadline, "timed out waiting for reads");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[cfg(feature = "profile")]
    fn metric(snapshot: &ProfileSnapshot, id: crate::profile::ProfileMetricId) -> u64 {
        snapshot.metric_value(id).unwrap_or(0)
    }

    #[test]
    fn config_defaults_are_valid() {
        AccessConfig::default()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn config_rejects_zero_queue_capacity() {
        let config = AccessConfig {
            queue_capacity: 0,
            scheduler_shards: 1,
            ..AccessConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_rejects_zero_scheduler_shards() {
        let config = AccessConfig {
            scheduler_shards: 0,
            ..AccessConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_rejects_more_scheduler_shards_than_cache_bytes() {
        let config = AccessConfig {
            scheduler_shards: 5,
            cache_capacity_bytes: 4,
            memory_budget_bytes: 8,
            ..AccessConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_rejects_memory_budget_smaller_than_cache() {
        let config = AccessConfig {
            cache_capacity_bytes: 1024,
            memory_budget_bytes: 512,
            ..AccessConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn shard_router_spreads_aligned_chunk_offsets() {
        let mut seen = [false; 4];
        for idx in 0..32 {
            let key = ChunkKey::new(FileRef(1), idx * 64 * 1024, 64 * 1024);
            seen[shard_for_key(key, seen.len())] = true;
        }

        assert!(seen.iter().all(|seen| *seen));
    }

    #[test]
    fn chunk_key_equality() {
        let a = ChunkKey::new(FileRef(1), 1024, 4096);
        let b = ChunkKey::new(FileRef(1), 1024, 4096);
        let c = ChunkKey::new(FileRef(1), 2048, 4096);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn slice_spec_scatter_copies_left_closed_right_open_ranges() {
        assert_eq!(copy_slices(b"abcdef", Some(&[])).expect("empty slice"), b"");
        assert_eq!(
            copy_slices(b"abcdef", Some(&[0, 0, 3, 3, 2, 5])).expect("slice"),
            b"abccde"
        );
        assert_eq!(
            copy_slices(b"abcdef", Some(&[2, 0, 3])).expect("slice with gap"),
            b"\0\0abc"
        );
        assert_eq!(
            copy_slices(b"abcdef", Some(&[0, 0, 4, 2, 4, 6])).expect("overlap"),
            b"abef"
        );
    }

    #[test]
    fn slice_spec_rejects_invalid_triples() {
        assert!(copy_slices(b"abc", Some(&[0])).is_err());
        assert!(copy_slices(b"abc", Some(&[0, 0])).is_err());
        assert!(copy_slices(b"abc", Some(&[0, 2, 1])).is_err());
        assert!(copy_slices(b"abc", Some(&[0, 0, 4])).is_err());
        assert!(copy_slices(b"abc", Some(&[usize::MAX, 0, 1])).is_err());
    }

    #[test]
    fn scheduler_reads_and_decodes_chunk() {
        let io = StaticIo::new(b"abcdef");
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(IdentityDecode::new()))
                .expect("spawn scheduler");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let (request, rx) = request(key);

        handle.send(request).expect("send request");

        assert_eq!(recv_reply(rx).expect("decode"), b"abcdef");
    }

    #[test]
    fn access_request_can_return_sliced_output() {
        let io = StaticIo::new(b"abcdef");
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(IdentityDecode::new()))
                .expect("spawn scheduler");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let (request, rx) = request(key);

        handle
            .send(request.with_slice(Some(vec![0, 1, 4, 3, 0, 2])))
            .expect("send request");

        assert_eq!(recv_reply(rx).expect("decode"), b"bcdab");
    }

    #[test]
    fn access_request_with_invalid_slice_returns_error() {
        let io = StaticIo::new(b"abcdef");
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(IdentityDecode::new()))
                .expect("spawn scheduler");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let (request, rx) = request(key);

        handle
            .send(request.with_slice(Some(vec![0, 0])))
            .expect("send request");

        let err = recv_reply(rx).expect_err("invalid slice should fail");
        assert_eq!(
            err.to_string(),
            "invalid slice spec: slice spec must contain off/start/end triples"
        );
    }

    #[test]
    fn concurrent_same_key_requests_share_one_read() {
        let (io, gate, reads) = StaticIo::gated(b"abcdef");
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(IdentityDecode::new()))
                .expect("spawn scheduler");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let (first, first_rx) = request(key);
        let (second, second_rx) = request(key);

        handle.send(first).expect("send first");
        wait_for_reads(&reads, 1);
        handle.send(second).expect("send second");
        thread::sleep(Duration::from_millis(10));
        gate.send(()).expect("release read");

        assert_eq!(recv_reply(first_rx).expect("first decode"), b"abcdef");
        assert_eq!(recv_reply(second_rx).expect("second decode"), b"abcdef");
        assert_eq!(reads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn keep_decoded_reuses_decoded_cache() {
        let (io, reads) = StaticIo::counted(b"x");
        let (decode, decodes) = CodecBackedDecode::counted();
        let mut config = test_config();
        config.keep_decoded = true;
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'a' });

        let (first, first_rx) = request_with_codec(key, Arc::clone(&codec), None);
        handle.send(first).expect("send first");
        assert_eq!(recv_reply(first_rx).expect("first"), b"ax");

        let (second, second_rx) = request_with_codec(key, Arc::clone(&codec), None);
        handle.send(second).expect("send second");
        assert_eq!(recv_reply(second_rx).expect("second"), b"ax");

        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn keep_decoded_reuses_equivalent_codec_instances() {
        let (io, reads) = StaticIo::counted(b"x");
        let (decode, decodes) = CodecBackedDecode::counted();
        let mut config = test_config();
        config.keep_decoded = true;
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 1);

        let (first, first_rx) =
            request_with_codec(key, Arc::new(PrefixCodec { prefix: b'a' }), None);
        handle.send(first).expect("send first");
        assert_eq!(recv_reply(first_rx).expect("first"), b"ax");

        let (second, second_rx) =
            request_with_codec(key, Arc::new(PrefixCodec { prefix: b'a' }), None);
        handle.send(second).expect("send second");
        assert_eq!(recv_reply(second_rx).expect("second"), b"ax");

        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn decode_waiter_registered_before_finish_observes_wake() {
        let config = test_config();
        let state = Rc::new(RefCell::new(SchedulerState::new(&config)));
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'a' });
        let item = AccessItem::new(key, Arc::clone(&codec), None);

        assert!(matches!(
            register_decode_waiter(&state, &item),
            DecodeRegister::First(_)
        ));
        let waiter = match register_decode_waiter(&state, &item) {
            DecodeRegister::Wait { notify, .. } => notify,
            DecodeRegister::Ready(_) | DecodeRegister::First(_) => {
                panic!("second decode registration should wait")
            }
        };
        let decode_key = DecodeKey::new(key, &codec, None);
        finish_decode_waiter(&state, &decode_key);

        Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(waiter);
    }

    #[test]
    fn keep_decoded_does_not_reuse_cache_for_different_codec() {
        let (io, reads) = StaticIo::counted(b"x");
        let (decode, decodes) = CodecBackedDecode::counted();
        let mut config = test_config();
        config.keep_decoded = true;
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let first_codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'a' });
        let second_codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'b' });

        let (first, first_rx) = request_with_codec(key, first_codec, None);
        handle.send(first).expect("send first");
        assert_eq!(recv_reply(first_rx).expect("first"), b"ax");

        let (second, second_rx) = request_with_codec(key, second_codec, None);
        handle.send(second).expect("send second");
        assert_eq!(recv_reply(second_rx).expect("second"), b"bx");

        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn keep_decoded_disabled_decodes_every_access() {
        let (io, reads) = StaticIo::counted(b"x");
        let (decode, decodes) = CodecBackedDecode::counted();
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'a' });

        let (first, first_rx) = request_with_codec(key, Arc::clone(&codec), None);
        handle.send(first).expect("send first");
        assert_eq!(recv_reply(first_rx).expect("first"), b"ax");

        let (second, second_rx) = request_with_codec(key, codec, None);
        handle.send(second).expect("send second");
        assert_eq!(recv_reply(second_rx).expect("second"), b"ax");

        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn prefetch_loads_raw_cache_without_decoding() {
        let (io, reads) = StaticIo::counted(b"abcdef");
        let (decode, decodes) = IdentityDecode::counted();
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 6);

        recv_prefetch(handle.prefetch(key).expect("prefetch submit")).expect("prefetch");
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 0);

        let (request, rx) = request(key);
        handle.send(request).expect("send request");
        assert_eq!(recv_reply(rx).expect("decode"), b"abcdef");
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn identity_codec_bypasses_decode_pool_on_raw_hit() {
        let (io, reads) = StaticIo::counted(b"abcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let (request, rx) = request_with_codec(key, codec, Some(6));

        handle.send(request).expect("send request");

        assert_eq!(recv_reply(rx).expect("identity fast path"), b"abcdef");
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 0);
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_access_read_cache_and_materialize_events() {
        let (io, reads) = StaticIo::counted(b"abcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let profile = AccessProfile::enabled("access-test");
        let mut config = test_config();
        config.profile = profile.clone();
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let round = profile.start();

        let (first, first_rx) = request_with_codec(key, Arc::clone(&codec), Some(6));
        handle.send(first).expect("send first");
        assert_eq!(recv_reply(first_rx).expect("first"), b"abcdef");

        let first_snapshot = handle.profile_snapshot();
        assert_eq!(first_snapshot.label, "access-test");
        assert_eq!(metric(&first_snapshot, test_metrics::COMMANDS), 1);
        assert_eq!(metric(&first_snapshot, test_metrics::READ_COMMANDS), 1);
        assert_eq!(metric(&first_snapshot, test_metrics::CACHE_MISSES), 1);
        assert_eq!(metric(&first_snapshot, test_metrics::CACHE_HITS), 0);
        assert_eq!(metric(&first_snapshot, test_metrics::INFLIGHT_FIRST), 1);
        assert_eq!(metric(&first_snapshot, test_metrics::IO_READS), 1);
        assert_eq!(metric(&first_snapshot, test_metrics::IO_REQUESTED_BYTES), 6);
        assert_eq!(metric(&first_snapshot, test_metrics::IO_READ_BYTES), 6);
        assert_eq!(metric(&first_snapshot, test_metrics::IDENTITY_DECODE), 1);
        assert_eq!(metric(&first_snapshot, test_metrics::MATERIALIZE_CALLS), 1);
        assert_eq!(metric(&first_snapshot, test_metrics::MATERIALIZE_BYTES), 6);
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 0);

        let (second, second_rx) = request_with_codec(key, Arc::clone(&codec), Some(6));
        handle.send(second).expect("send second");
        assert_eq!(recv_reply(second_rx).expect("second"), b"abcdef");

        let total = profile.snapshot();
        assert_eq!(metric(&total, test_metrics::COMMANDS), 2);
        assert_eq!(metric(&total, test_metrics::READ_COMMANDS), 2);
        assert_eq!(metric(&total, test_metrics::CACHE_HITS), 1);
        assert_eq!(metric(&total, test_metrics::RAW_CACHE_HITS), 1);
        assert_eq!(metric(&total, test_metrics::CACHE_MISSES), 1);
        assert_eq!(metric(&total, test_metrics::IO_READS), 1);
        assert_eq!(metric(&total, test_metrics::IDENTITY_DECODE), 2);
        assert_eq!(metric(&total, test_metrics::MATERIALIZE_CALLS), 2);
        assert_eq!(metric(&total, test_metrics::MATERIALIZE_BYTES), 12);

        let reset = handle.profile_snapshot_and_reset();
        assert_eq!(metric(&reset, test_metrics::COMMANDS), 2);
        assert_eq!(
            metric(&handle.profile_snapshot(), test_metrics::COMMANDS),
            0
        );

        let (third, third_rx) = request_with_codec(key, codec, Some(6));
        handle.send(third).expect("send third");
        assert_eq!(recv_reply(third_rx).expect("third"), b"abcdef");
        let after_reset = round.end();
        assert_eq!(metric(&after_reset, test_metrics::COMMANDS), 1);
        assert_eq!(metric(&after_reset, test_metrics::READ_COMMANDS), 1);
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_respects_access_scope_switches() {
        let (io, _reads) = StaticIo::counted(b"abcdef");
        let (decode, _decodes) = CodecBackedDecode::counted();
        let profile = AccessProfile::from_runtime(ProfileRuntime::new_lazy(
            ProfileConfig::enabled("access-scope").disable_scope("access.io"),
            access_profile_registry,
        ));
        let mut config = test_config();
        config.profile = profile.clone();
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let round = profile.start();

        let (request, rx) = request_with_codec(key, codec, Some(6));
        handle.send(request).expect("send request");
        assert_eq!(recv_reply(rx).expect("read"), b"abcdef");

        let snapshot = handle.profile_snapshot();
        assert_eq!(metric(&snapshot, test_metrics::COMMANDS), 1);
        assert_eq!(metric(&snapshot, test_metrics::CACHE_MISSES), 1);
        assert_eq!(metric(&snapshot, test_metrics::IO_READS), 0);
        assert_eq!(metric(&snapshot, test_metrics::IO_REQUESTED_BYTES), 0);
        assert_eq!(metric(&snapshot, test_metrics::IO_READ_BYTES), 0);
        assert_eq!(metric(&snapshot, test_metrics::IDENTITY_DECODE), 1);
        assert_eq!(metric(&snapshot, test_metrics::MATERIALIZE_CALLS), 1);
        assert!(
            !snapshot
                .scopes
                .iter()
                .find(|scope| scope.id == ACCESS_IO_SCOPE)
                .expect("access.io scope")
                .enabled
        );

        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_queue_full_rejections() {
        let profile = AccessProfile::enabled("queue-full");
        let (tx, _rx) = flume::bounded(0);
        let handle = AccessHandle {
            inner: Arc::new(AccessHandleInner {
                shards: vec![tx],
                profile: profile.clone(),
            }),
        };
        let round = profile.start();
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let (request, _reply) = request(key);

        assert!(matches!(
            handle.try_send(request),
            Err(AccessError::QueueFull { capacity: 0 })
        ));

        let snapshot = round.end();
        assert_eq!(metric(&snapshot, test_metrics::COMMANDS), 0);
        assert_eq!(metric(&snapshot, test_metrics::COMMAND_REJECTIONS), 1);
        assert_eq!(metric(&snapshot, test_metrics::COMMAND_QUEUE_FULL), 1);
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_raw_and_decoded_cache_inserts() {
        let (io, reads) = StaticIo::counted(b"bcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let profile = AccessProfile::enabled("cache-replace");
        let mut config = test_config();
        config.keep_decoded = true;
        config.profile = profile.clone();
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 5);
        let codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'a' });
        let round = profile.start();

        let (request, rx) = request_with_codec(key, codec, Some(6));
        handle.send(request).expect("send request");
        assert_eq!(recv_reply(rx).expect("read"), b"abcdef");

        let snapshot = round.end();
        assert_eq!(metric(&snapshot, test_metrics::CACHE_INSERTS), 2);
        assert_eq!(metric(&snapshot, test_metrics::CACHE_REPLACEMENTS), 0);
        assert_eq!(metric(&snapshot, test_metrics::CACHE_REPLACED_BYTES), 0);
        assert_eq!(metric(&snapshot, test_metrics::CACHE_EVICTIONS), 0);
        assert_eq!(metric(&snapshot, test_metrics::CACHE_UNCACHED_FALLBACKS), 0);
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_cache_eviction_under_capacity_pressure() {
        let (io, reads) = StaticIo::counted(b"abcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let profile = AccessProfile::enabled("cache-evict");
        let mut config = test_config();
        config.cache_capacity_bytes = 6;
        config.memory_budget_bytes = 12;
        config.profile = profile.clone();
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let first_key = ChunkKey::new(FileRef(1), 0, 6);
        let second_key = ChunkKey::new(FileRef(1), 6, 6);
        let round = profile.start();

        let (first, first_rx) = request_with_codec(first_key, Arc::clone(&codec), Some(6));
        handle.send(first).expect("send first");
        assert_eq!(recv_reply(first_rx).expect("first"), b"abcdef");
        let (second, second_rx) = request_with_codec(second_key, codec, Some(6));
        handle.send(second).expect("send second");
        assert_eq!(recv_reply(second_rx).expect("second"), b"abcdef");

        let snapshot = round.end();
        assert_eq!(metric(&snapshot, test_metrics::CACHE_INSERTS), 2);
        assert_eq!(metric(&snapshot, test_metrics::CACHE_EVICTIONS), 1);
        assert_eq!(metric(&snapshot, test_metrics::CACHE_EVICTED_BYTES), 6);
        assert_eq!(reads.load(Ordering::SeqCst), 2);
        assert_eq!(decodes.load(Ordering::SeqCst), 0);
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_uncached_fallback_for_oversized_chunk() {
        let (io, reads) = StaticIo::counted(b"abcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let profile = AccessProfile::enabled("uncached");
        let mut config = test_config();
        config.cache_capacity_bytes = 1;
        config.memory_budget_bytes = 12;
        config.profile = profile.clone();
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let round = profile.start();

        let (request, rx) = request_with_codec(key, codec, Some(6));
        handle.send(request).expect("send request");
        assert_eq!(recv_reply(rx).expect("read"), b"abcdef");

        let snapshot = round.end();
        assert_eq!(metric(&snapshot, test_metrics::CACHE_INSERTS), 0);
        assert_eq!(metric(&snapshot, test_metrics::CACHE_INSERT_FAILURES), 1);
        assert_eq!(metric(&snapshot, test_metrics::CACHE_TOO_LARGE), 1);
        assert_eq!(metric(&snapshot, test_metrics::CACHE_UNCACHED_FALLBACKS), 1);
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 0);
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_registers_access_scopes_for_active_runtime() {
        let runtime = ProfileRuntime::new_lazy(
            ProfileConfig::enabled("access-active"),
            ProfileRegistry::new,
        );
        let round = runtime.start();
        let profile = AccessProfile::from_runtime(runtime.clone());

        profile.record_cache_miss();

        let snapshot = runtime.snapshot();
        assert_eq!(metric(&snapshot, test_metrics::CACHE_MISSES), 1);
        assert!(snapshot
            .scopes
            .iter()
            .any(|scope| scope.id == ACCESS_CACHE_SCOPE && scope.enabled));

        round.end();
    }

    #[test]
    fn identity_codec_size_mismatch_fails_without_decode_pool() {
        let (io, _reads) = StaticIo::counted(b"abcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let (request, rx) = request_with_codec(key, codec, Some(5));

        handle.send(request).expect("send request");

        let err = recv_reply(rx).expect_err("identity size mismatch");
        assert!(err.to_string().contains("expected 5 bytes, got 6"));
        assert_eq!(decodes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn oversized_prefetch_fails_without_reading() {
        let (io, reads) = StaticIo::counted(b"abcdefgh");
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 4,
            memory_budget_bytes: 8,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(IdentityDecode::new()))
            .expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 8);

        let err = recv_prefetch(handle.prefetch(key).expect("prefetch submit"))
            .expect_err("prefetch cannot cache oversized raw chunk");
        assert!(err.to_string().contains("memory budget"));
        assert_eq!(reads.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn standalone_prefetch_reuses_raw_after_decoded_is_cached() {
        let (io, reads) = StaticIo::counted(b"x");
        let (decode, decodes) = CodecBackedDecode::counted();
        let mut config = test_config();
        config.keep_decoded = true;
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let first_codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'a' });
        let second_codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'b' });

        let (request, rx) = request_with_codec(key, first_codec, None);
        handle.send(request).expect("send request");
        assert_eq!(recv_reply(rx).expect("first"), b"ax");
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 1);

        recv_prefetch(handle.prefetch(key).expect("prefetch submit")).expect("prefetch");
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 1);

        let (second, second_rx) = request_with_codec(key, second_codec, None);
        handle.send(second).expect("send second");
        assert_eq!(recv_reply(second_rx).expect("second"), b"bx");
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn oversized_decoded_cache_attempt_keeps_existing_raw_entries() {
        let (io, reads) = StaticIo::counted(b"data");
        let (decode, decodes) = CodecBackedDecode::counted();
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 8,
            memory_budget_bytes: 20,
            default_io_priority: 0,
            keep_decoded: true,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let expanding_key = ChunkKey::new(FileRef(1), 0, 4);
        let hot_raw_key = ChunkKey::new(FileRef(1), 4, 4);

        recv_prefetch(handle.prefetch(hot_raw_key).expect("prefetch submit")).expect("prefetch");
        assert_eq!(reads.load(Ordering::SeqCst), 1);

        let expanding_codec: SharedCodec = Arc::new(ExpandCodec {
            byte: b'z',
            output_len: 16,
        });
        let (request, rx) = request_with_codec(expanding_key, expanding_codec, Some(16));
        handle.send(request).expect("send expanding request");
        assert_eq!(recv_reply(rx).expect("expanded"), vec![b'z'; 16]);
        assert_eq!(reads.load(Ordering::SeqCst), 2);
        assert_eq!(decodes.load(Ordering::SeqCst), 1);

        let (hot_request, hot_rx) =
            request_with_codec(hot_raw_key, Arc::new(UncompressedCodec), Some(4));
        handle.send(hot_request).expect("send hot raw request");
        assert_eq!(recv_reply(hot_rx).expect("hot raw"), b"data");
        assert_eq!(reads.load(Ordering::SeqCst), 2);
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn scheduled_decode_staging_oom_releases_decode_waiters() {
        let io = StaticIo::new(b"x");
        let (decode, _decodes) = CodecBackedDecode::counted();
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 4,
            memory_budget_bytes: 4,
            default_io_priority: 0,
            keep_decoded: true,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(ExpandCodec {
            byte: b'z',
            output_len: 8,
        });
        let item = AccessItem::new(key, Arc::clone(&codec), Some(8));

        let mut scheduled = handle
            .scheduled(
                vec![item.clone()],
                ScheduledAccessConfig {
                    prefetch_step: 0,
                    decode_ahead_steps: 0,
                    ready_ahead_steps: 0,
                },
            )
            .expect("scheduled");

        let scheduled_err = scheduled
            .next()
            .expect("scheduled item")
            .expect_err("decoded staging should exceed budget");
        assert!(scheduled_err.to_string().contains("memory budget"));

        let (request, rx) = request_with_codec(key, codec, Some(8));
        handle.send(request).expect("send request");
        assert_eq!(
            recv_reply_timeout(rx, Duration::from_secs(1)).expect("request after failed schedule"),
            vec![b'z'; 8]
        );
    }

    #[test]
    fn scheduled_ready_reuses_decoded_staging_budget() {
        let io = StaticIo::new(b"x");
        let (decode, decodes) = CodecBackedDecode::counted();
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 1,
            memory_budget_bytes: 8,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(ExpandCodec {
            byte: b'z',
            output_len: 8,
        });
        let item = AccessItem::new(key, codec, Some(8));

        let scheduled = handle
            .scheduled(
                vec![item],
                ScheduledAccessConfig {
                    prefetch_step: 0,
                    decode_ahead_steps: 0,
                    ready_ahead_steps: 0,
                },
            )
            .expect("scheduled");

        assert_eq!(
            recv_scheduled_next_timeout(scheduled, Duration::from_secs(1))
                .expect("scheduled item")
                .expect("ready result"),
            vec![b'z'; 8]
        );
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn scheduled_ready_reports_oom_when_decoded_cache_fills_budget() {
        let (io, reads) = StaticIo::counted(b"bcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 6,
            memory_budget_bytes: 6,
            default_io_priority: 0,
            keep_decoded: true,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 5);
        let codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'a' });
        let item = AccessItem::new(key, codec, None);

        let scheduled = handle
            .scheduled(
                vec![item],
                ScheduledAccessConfig {
                    prefetch_step: 0,
                    decode_ahead_steps: 0,
                    ready_ahead_steps: 0,
                },
            )
            .expect("scheduled");

        let err = recv_scheduled_next_timeout(scheduled, Duration::from_secs(1))
            .expect("scheduled item")
            .expect_err("ready copy needs cache input plus output budget");
        assert!(err.to_string().contains("memory budget"));
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn scheduled_unsliced_staged_decode_moves_into_ready_without_extra_budget() {
        let io = StaticIo::new(b"x");
        let (decode, decodes) = CodecBackedDecode::counted();
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 1,
            memory_budget_bytes: 8,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(ExpandCodec {
            byte: b'z',
            output_len: 8,
        });
        let item = AccessItem::new(key, codec, Some(8));

        let scheduled = handle
            .scheduled(
                vec![item],
                ScheduledAccessConfig {
                    prefetch_step: 0,
                    decode_ahead_steps: 0,
                    ready_ahead_steps: 0,
                },
            )
            .expect("scheduled");

        assert_eq!(
            recv_scheduled_next_timeout(scheduled, Duration::from_secs(1))
                .expect("scheduled item")
                .expect("ready result"),
            vec![b'z'; 8]
        );
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn scheduled_identity_slice_staged_decode_moves_without_extra_budget() {
        let io = StaticIo::new(b"x");
        let (decode, decodes) = CodecBackedDecode::counted();
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 1,
            memory_budget_bytes: 8,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(ExpandCodec {
            byte: b'z',
            output_len: 8,
        });
        let item = AccessItem::new(key, codec, Some(8)).with_slice(Some(vec![0, 0, 4, 4, 4, 8]));

        let scheduled = handle
            .scheduled(
                vec![item],
                ScheduledAccessConfig {
                    prefetch_step: 0,
                    decode_ahead_steps: 0,
                    ready_ahead_steps: 0,
                },
            )
            .expect("scheduled");

        assert_eq!(
            recv_scheduled_next_timeout(scheduled, Duration::from_secs(1))
                .expect("scheduled item")
                .expect("ready result"),
            vec![b'z'; 8]
        );
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn scheduled_ready_keeps_decoded_cache_when_output_budget_fits() {
        let (io, reads) = StaticIo::counted(b"bcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 6,
            memory_budget_bytes: 9,
            default_io_priority: 0,
            keep_decoded: true,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 5);
        let codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'a' });
        let scheduled_item =
            AccessItem::new(key, Arc::clone(&codec), None).with_slice(Some(vec![0, 0, 3]));

        let scheduled = handle
            .scheduled(
                vec![scheduled_item],
                ScheduledAccessConfig {
                    prefetch_step: 0,
                    decode_ahead_steps: 0,
                    ready_ahead_steps: 0,
                },
            )
            .expect("scheduled");

        assert_eq!(
            recv_scheduled_next_timeout(scheduled, Duration::from_secs(1))
                .expect("scheduled item")
                .expect("ready result"),
            b"abc"
        );

        let (request, rx) = request_with_codec(key, codec, None);
        handle.send(request).expect("send cached request");
        assert_eq!(recv_reply(rx).expect("cached full chunk"), b"abcdef");
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn scheduled_sliced_staged_decode_requires_output_budget() {
        let io = StaticIo::new(b"x");
        let (decode, decodes) = CodecBackedDecode::counted();
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 1,
            memory_budget_bytes: 8,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(ExpandCodec {
            byte: b'z',
            output_len: 8,
        });
        let item = AccessItem::new(key, codec, Some(8)).with_slice(Some(vec![0, 0, 4]));

        let scheduled = handle
            .scheduled(
                vec![item],
                ScheduledAccessConfig {
                    prefetch_step: 0,
                    decode_ahead_steps: 0,
                    ready_ahead_steps: 0,
                },
            )
            .expect("scheduled");

        let err = recv_scheduled_next_timeout(scheduled, Duration::from_secs(1))
            .expect("scheduled item")
            .expect_err("scatter-copy needs an output buffer in addition to staging");
        assert!(err.to_string().contains("memory budget"));
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn direct_ready_sliced_owned_decode_requires_input_and_output_budget() {
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 1,
            memory_budget_bytes: 8,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let state = Rc::new(RefCell::new(SchedulerState::new(&config)));
        let cpu = Arc::new(AccessCpuPool::new(test_cpu_config()).expect("cpu pool"));
        let deps = SchedulerDeps {
            io: Arc::new(StaticIo::new(b"x")),
            decode: Arc::new(CodecBackedDecode::counted().0),
            cpu,
            profile: AccessProfile::disabled(),
            priority: 0,
            keep_decoded: false,
        };
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(ExpandCodec {
            byte: b'z',
            output_len: 8,
        });
        let item = AccessItem::new(key, codec, Some(8)).with_slice(Some(vec![0, 0, 4]));

        let result = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(make_ready_direct(state, &deps, item));

        assert!(matches!(result, Err(AccessError::OutOfMemory)));
    }

    #[test]
    fn direct_ready_uncached_shared_decode_requires_input_and_output_budget() {
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 1,
            memory_budget_bytes: 8,
            default_io_priority: 0,
            keep_decoded: true,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let state = Rc::new(RefCell::new(SchedulerState::new(&config)));
        let cpu = Arc::new(AccessCpuPool::new(test_cpu_config()).expect("cpu pool"));
        let deps = SchedulerDeps {
            io: Arc::new(StaticIo::new(b"x")),
            decode: Arc::new(CodecBackedDecode::counted().0),
            cpu,
            profile: AccessProfile::disabled(),
            priority: 0,
            keep_decoded: true,
        };
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(ExpandCodec {
            byte: b'z',
            output_len: 8,
        });
        let item = AccessItem::new(key, codec, Some(8)).with_slice(Some(vec![0, 0, 4]));

        let result = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(make_ready_direct(state, &deps, item));

        assert!(matches!(result, Err(AccessError::OutOfMemory)));
    }

    #[cfg(feature = "profile")]
    #[test]
    fn scheduled_future_ready_buffers_are_evictable_under_memory_pressure() {
        let (io, gate, reads) = StaticIo::gated(b"x");
        let (decode, decodes) = CodecBackedDecode::counted();
        let profile = AccessProfile::enabled("scheduled-evict");
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 2,
            memory_budget_bytes: 5,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: test_cpu_config(),
            profile: profile.clone(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let codec: SharedCodec = Arc::new(ExpandCodec {
            byte: b'z',
            output_len: 4,
        });
        let items = vec![
            AccessItem::new(ChunkKey::new(FileRef(1), 0, 1), Arc::clone(&codec), Some(4)),
            AccessItem::new(ChunkKey::new(FileRef(1), 1, 1), codec, Some(4)),
        ];
        let round = profile.start();

        let scheduled = handle
            .scheduled(
                items,
                ScheduledAccessConfig {
                    prefetch_step: 0,
                    decode_ahead_steps: 0,
                    ready_ahead_steps: 1,
                },
            )
            .expect("scheduled");

        wait_for_reads(&reads, 2);
        wait_for_reads(&decodes, 1);
        thread::sleep(Duration::from_millis(50));
        gate.send(()).expect("release first read");

        assert_eq!(
            recv_scheduled_next_timeout(scheduled, Duration::from_secs(1))
                .expect("scheduled item")
                .expect("ready result"),
            vec![b'z'; 4]
        );
        let snapshot = round.end();
        assert_eq!(metric(&snapshot, test_metrics::SCHEDULED_EVICTIONS), 1);
        assert_eq!(metric(&snapshot, test_metrics::SCHEDULED_EVICTED_BYTES), 4);
    }

    #[test]
    fn scheduled_cancel_wakes_blocked_take() {
        let (io, gate, reads) = StaticIo::gated(b"x");
        let (decode, _decodes) = CodecBackedDecode::counted();
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(decode)).expect("spawn");
        let cancel = PrefetchCancel::new(handle.clone());
        let key = ChunkKey::new(FileRef(1), 0, 1);
        let codec: SharedCodec = Arc::new(ExpandCodec {
            byte: b'z',
            output_len: 4,
        });
        let item = AccessItem::new(key, codec, Some(4));
        let mut scheduled = handle
            .scheduled(
                vec![item],
                ScheduledAccessConfig {
                    prefetch_step: 0,
                    decode_ahead_steps: 0,
                    ready_ahead_steps: 0,
                },
            )
            .expect("scheduled");
        scheduled.set_cancel_handle(Arc::clone(&cancel));

        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(scheduled.next());
        });
        wait_for_reads(&reads, 1);

        cancel.cancel_in_flight();
        let result = rx.recv_timeout(Duration::from_secs(1)).unwrap_or_else(|_| {
            Some(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for cancelled scheduled item",
            )))
        });
        assert!(result
            .expect("scheduled item")
            .expect_err("cancelled item should return an error")
            .to_string()
            .contains("cancelled"));

        let _ = gate.send(());
    }

    #[test]
    fn scheduled_iterator_returns_ready_results() {
        let (io, reads) = StaticIo::counted(b"bcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 5);
        let codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'a' });
        let items = vec![
            AccessItem::new(key, Arc::clone(&codec), None).with_slice(Some(vec![0, 0, 3])),
            AccessItem::new(key, Arc::clone(&codec), None).with_slice(Some(vec![0, 3, 6])),
        ];

        let mut scheduled = handle
            .scheduled(
                items,
                ScheduledAccessConfig {
                    prefetch_step: 1,
                    decode_ahead_steps: 1,
                    ready_ahead_steps: 1,
                },
            )
            .expect("scheduled");

        assert_eq!(scheduled.next().expect("first").expect("first ok"), b"abc");
        assert_eq!(
            scheduled.next().expect("second").expect("second ok"),
            b"def"
        );
        assert!(scheduled.next().is_none());
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn scheduled_identity_iterator_bypasses_decode_pool() {
        let (io, reads) = StaticIo::counted(b"abcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let handle =
            AccessScheduler::spawn(test_config(), Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let items = vec![
            AccessItem::new(key, Arc::clone(&codec), Some(6)).with_slice(Some(vec![0, 0, 3])),
            AccessItem::new(key, Arc::clone(&codec), Some(6)).with_slice(Some(vec![0, 3, 6])),
        ];

        let mut scheduled = handle
            .scheduled(
                items,
                ScheduledAccessConfig {
                    prefetch_step: 1,
                    decode_ahead_steps: 1,
                    ready_ahead_steps: 1,
                },
            )
            .expect("scheduled");

        assert_eq!(scheduled.next().expect("first").expect("first ok"), b"abc");
        assert_eq!(
            scheduled.next().expect("second").expect("second ok"),
            b"def"
        );
        assert!(scheduled.next().is_none());
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn scheduled_uncached_identity_decode_bypasses_decode_pool() {
        let (io, reads) = StaticIo::counted(b"abcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let config = AccessConfig {
            queue_capacity: 32,
            scheduler_shards: 1,
            cache_capacity_bytes: 1,
            memory_budget_bytes: 12,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: test_cpu_config(),
            profile: AccessProfile::disabled(),
        };
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 6);
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let item = AccessItem::new(key, codec, Some(6));

        let mut scheduled = handle
            .scheduled(
                vec![item],
                ScheduledAccessConfig {
                    prefetch_step: 0,
                    decode_ahead_steps: 0,
                    ready_ahead_steps: 0,
                },
            )
            .expect("scheduled");

        assert_eq!(
            scheduled.next().expect("first").expect("first ok"),
            b"abcdef"
        );
        assert!(scheduled.next().is_none());
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn scheduled_iterator_reuses_decoded_cache_when_enabled() {
        let (io, reads) = StaticIo::counted(b"bcdef");
        let (decode, decodes) = CodecBackedDecode::counted();
        let mut config = test_config();
        config.keep_decoded = true;
        let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode)).expect("spawn");
        let key = ChunkKey::new(FileRef(1), 0, 5);
        let codec: SharedCodec = Arc::new(PrefixCodec { prefix: b'a' });
        let items = vec![
            AccessItem::new(key, Arc::clone(&codec), None).with_slice(Some(vec![0, 0, 3])),
            AccessItem::new(key, Arc::clone(&codec), None).with_slice(Some(vec![0, 3, 6])),
        ];

        let mut scheduled = handle
            .scheduled(
                items,
                ScheduledAccessConfig {
                    prefetch_step: 1,
                    decode_ahead_steps: 1,
                    ready_ahead_steps: 1,
                },
            )
            .expect("scheduled");

        assert_eq!(scheduled.next().expect("first").expect("first ok"), b"abc");
        assert_eq!(
            scheduled.next().expect("second").expect("second ok"),
            b"def"
        );
        assert!(scheduled.next().is_none());
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
    }
}
