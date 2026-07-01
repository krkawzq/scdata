mod profile;
mod threaded;

#[cfg(feature = "uring")]
mod uring;

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fs::{File, Metadata, OpenOptions};
use std::future::Future;
use std::io;
use std::mem::MaybeUninit;
use std::os::unix::fs::FileExt;
#[cfg(feature = "uring")]
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock, Weak};
use std::task::{Context, Poll};
use std::thread;

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use crate::profile::{ProfileRuntime, ProfileSnapshot, ProfileTimer};

use self::profile::{
    record_iopool_cancelled_before_dispatch, record_iopool_dedup_hit, record_iopool_dispatched,
    record_iopool_immediate, record_iopool_operation, record_iopool_queue_full,
    record_iopool_submit, IoCommandKind, IoPoolProfile,
};

const TARGETED_READY_NOTIFIES: usize = 8;

/// Opaque handle returned by [`IoPool::register_file`].
pub type FileId = usize;

/// Which IO backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// io_uring driver. The queue semantics are identical to the threaded
    /// backend.
    Uring,
    /// Thread-pool pread/pwrite fallback.
    Threaded,
}

/// Operation kind used by [`RequestKey`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    Read,
}

/// Deduplication key for operations that may be shared by multiple callers.
///
/// Only reads are deduplicated. Writes carry caller-owned bytes, and sync-like
/// operations are sequence points whose meaning depends on their submit order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestKey {
    pub file: FileId,
    pub offset: u64,
    pub len: usize,
    pub kind: OpKind,
}

/// Settings shared by every backend.
#[derive(Debug, Clone)]
pub struct BaseIoConfig {
    /// Maximum number of IO operations dispatched to the backend at once.
    /// Duplicate reads do not consume another slot. Default: 1024.
    pub max_in_flight: usize,
    /// Maximum number of unique IO operations admitted into the internal queue.
    /// Duplicate reads do not consume another slot. Default: 4096.
    pub queue_capacity: usize,
    /// Number of priority levels. `0` is the highest priority. Default: 3.
    pub priority_levels: usize,
    /// Number of independent internal queues. Default: 1.
    ///
    /// Values greater than 1 reduce lock contention across independent files.
    /// Requests for the same file are routed to the same shard so per-file
    /// ordering barriers and read deduplication remain intact.
    pub queue_shards: usize,
    /// Enable a read-optimized queue path for workloads where reads are known
    /// not to overlap with writes/truncates. Default: false.
    ///
    /// Duplicate reads are still deduplicated. Reads submitted while a barrier
    /// is pending on the same file fall back to the fully ordered path.
    pub assume_non_overlapping_reads: bool,
}

impl Default for BaseIoConfig {
    fn default() -> Self {
        Self {
            max_in_flight: 1024,
            queue_capacity: 4096,
            priority_levels: 3,
            queue_shards: 1,
            assume_non_overlapping_reads: false,
        }
    }
}

impl BaseIoConfig {
    pub(super) fn in_flight_limit(&self) -> usize {
        self.max_in_flight.max(1)
    }

    pub(super) fn priority_levels(&self) -> usize {
        self.priority_levels.max(1)
    }

    pub(super) fn queue_capacity(&self) -> usize {
        self.queue_capacity.max(1)
    }

    pub(super) fn queue_shards(&self) -> usize {
        self.queue_shards.max(1)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.max_in_flight == 0 {
            return Err("max_in_flight must be greater than 0".to_string());
        }
        if self.queue_capacity == 0 {
            return Err("queue_capacity must be greater than 0".to_string());
        }
        if self.priority_levels == 0 {
            return Err("priority_levels must be greater than 0".to_string());
        }
        if self.queue_shards == 0 {
            return Err("queue_shards must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// Configuration for the io_uring backend.
#[derive(Debug, Clone)]
pub struct UringConfig {
    pub base: BaseIoConfig,
    /// Submission/completion queue depth. Default: 256.
    pub entries: u32,
    /// Number of io_uring driver threads/rings. Default: 1.
    pub drivers: usize,
    /// io-wq bounded worker limit per NUMA node. `0` keeps the kernel default.
    pub iowq_bounded_workers: u32,
    /// io-wq unbounded worker limit per NUMA node. `0` keeps the kernel default.
    pub iowq_unbounded_workers: u32,
    /// Number of sparse fixed-file slots to register. `0` disables fixed
    /// files. Default: 4096.
    pub registered_files: u32,
}

impl Default for UringConfig {
    fn default() -> Self {
        Self {
            base: BaseIoConfig::default(),
            entries: 256,
            drivers: 1,
            iowq_bounded_workers: 0,
            iowq_unbounded_workers: 0,
            registered_files: 4096,
        }
    }
}

impl UringConfig {
    pub(super) fn validate(&self) -> Result<(), String> {
        self.base.validate()?;
        if self.entries < 2 {
            return Err("entries must be greater than 1".to_string());
        }
        if self.drivers == 0 {
            return Err("drivers must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// Configuration for the thread-pool pread/pwrite backend.
#[derive(Debug, Clone)]
pub struct ThreadedConfig {
    pub base: BaseIoConfig,
    /// Number of worker threads. Default: 8.
    pub num_workers: usize,
    /// Optional CPU affinity allow-list. When `Some`, workers are pinned to
    /// these CPUs round-robin, so `num_workers` may be greater than
    /// `cpus.len()`. When `None`, workers are left unpinned so the scheduler can
    /// place blocking IO threads away from CPU-bound decode work.
    pub cpus: Option<Vec<usize>>,
}

impl Default for ThreadedConfig {
    fn default() -> Self {
        Self {
            base: BaseIoConfig::default(),
            num_workers: 8,
            cpus: None,
        }
    }
}

impl ThreadedConfig {
    pub(super) fn worker_count(&self) -> usize {
        self.num_workers.max(1)
    }

    pub(super) fn effective_worker_count(&self) -> usize {
        self.worker_count().min(self.base.in_flight_limit()).max(1)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.base.validate()?;

        if self.num_workers == 0 {
            return Err("num_workers must be greater than 0".to_string());
        }

        if let Some(cpus) = &self.cpus {
            if cpus.is_empty() {
                return Err("cpus list must not be empty".to_string());
            }

            let unique = cpus.iter().copied().collect::<BTreeSet<_>>();
            if unique.len() != cpus.len() {
                return Err("cpus list contains duplicate entries".to_string());
            }
        }

        Ok(())
    }
}

/// Configuration for the IO backend.
#[derive(Debug, Clone)]
pub enum IoConfig {
    Uring(UringConfig),
    Threaded(ThreadedConfig),
}

impl Default for IoConfig {
    fn default() -> Self {
        Self::Threaded(ThreadedConfig::default())
    }
}

impl IoConfig {
    pub fn kind(&self) -> BackendKind {
        match self {
            Self::Uring(_) => BackendKind::Uring,
            Self::Threaded(_) => BackendKind::Threaded,
        }
    }

    pub fn base(&self) -> &BaseIoConfig {
        match self {
            Self::Uring(config) => &config.base,
            Self::Threaded(config) => &config.base,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Uring(config) => config.validate(),
            Self::Threaded(config) => config.validate(),
        }
    }

    pub(super) fn in_flight_limit(&self) -> usize {
        self.base().in_flight_limit()
    }

    pub(super) fn priority_levels(&self) -> usize {
        self.base().priority_levels()
    }

    pub(super) fn queue_capacity(&self) -> usize {
        self.base().queue_capacity()
    }

    pub(super) fn queue_shards(&self) -> usize {
        let requested = self.base().queue_shards();
        let max_active = self.base().in_flight_limit();
        let consumers = match self {
            Self::Uring(config) => config.drivers.max(1),
            Self::Threaded(config) => config.effective_worker_count(),
        };
        requested.min(consumers).min(max_active).max(1)
    }
}

/// A caller-submitted IO request.
#[derive(Debug)]
pub enum IoCommand {
    Read {
        file: FileId,
        offset: u64,
        len: usize,
        priority: usize,
    },
    Write {
        file: FileId,
        offset: u64,
        buf: Vec<u8>,
        priority: usize,
    },
    Fsync {
        file: FileId,
        priority: usize,
    },
    SyncData {
        file: FileId,
        priority: usize,
    },
    Truncate {
        file: FileId,
        len: u64,
        priority: usize,
    },
    Metadata {
        file: FileId,
        priority: usize,
    },
}

impl IoCommand {
    pub fn read(file: FileId, offset: u64, len: usize, priority: usize) -> Self {
        Self::Read {
            file,
            offset,
            len,
            priority,
        }
    }

    pub fn write(file: FileId, offset: u64, buf: Vec<u8>, priority: usize) -> Self {
        Self::Write {
            file,
            offset,
            buf,
            priority,
        }
    }

    pub fn fsync(file: FileId, priority: usize) -> Self {
        Self::Fsync { file, priority }
    }

    pub fn sync_all(file: FileId, priority: usize) -> Self {
        Self::Fsync { file, priority }
    }

    pub fn sync_data(file: FileId, priority: usize) -> Self {
        Self::SyncData { file, priority }
    }

    pub fn truncate(file: FileId, len: u64, priority: usize) -> Self {
        Self::Truncate {
            file,
            len,
            priority,
        }
    }

    pub fn metadata(file: FileId, priority: usize) -> Self {
        Self::Metadata { file, priority }
    }

    pub fn priority(&self) -> usize {
        match self {
            Self::Read { priority, .. }
            | Self::Write { priority, .. }
            | Self::Fsync { priority, .. }
            | Self::SyncData { priority, .. }
            | Self::Truncate { priority, .. }
            | Self::Metadata { priority, .. } => *priority,
        }
    }

    #[inline]
    fn profile_kind(&self) -> IoCommandKind {
        match self {
            Self::Read { .. } => IoCommandKind::Read,
            Self::Write { .. } => IoCommandKind::Write,
            Self::Fsync { .. } => IoCommandKind::Fsync,
            Self::SyncData { .. } => IoCommandKind::SyncData,
            Self::Truncate { .. } => IoCommandKind::Truncate,
            Self::Metadata { .. } => IoCommandKind::Metadata,
        }
    }

    pub fn dedup_key(&self) -> Option<RequestKey> {
        match self {
            Self::Read {
                file, offset, len, ..
            } => Some(RequestKey {
                file: *file,
                offset: *offset,
                len: *len,
                kind: OpKind::Read,
            }),
            Self::Write { .. }
            | Self::Fsync { .. }
            | Self::SyncData { .. }
            | Self::Truncate { .. }
            | Self::Metadata { .. } => None,
        }
    }

    pub fn file(&self) -> FileId {
        match self {
            Self::Read { file, .. }
            | Self::Write { file, .. }
            | Self::Fsync { file, .. }
            | Self::SyncData { file, .. }
            | Self::Truncate { file, .. }
            | Self::Metadata { file, .. } => *file,
        }
    }

    fn immediate_output(&self) -> Option<IoOutput> {
        match self {
            Self::Read { len: 0, .. } => Some(IoOutput::Read(Arc::from(Vec::new()))),
            Self::Write { buf, .. } if buf.is_empty() => Some(IoOutput::Write { bytes: 0 }),
            Self::Read { .. }
            | Self::Write { .. }
            | Self::Fsync { .. }
            | Self::SyncData { .. }
            | Self::Truncate { .. }
            | Self::Metadata { .. } => None,
        }
    }

    fn into_operation(self, handle: Option<Arc<File>>) -> IoOperation {
        match self {
            Self::Read {
                file, offset, len, ..
            } => IoOperation::Read {
                file,
                handle,
                offset,
                len,
            },
            Self::Write {
                file, offset, buf, ..
            } => IoOperation::Write {
                file,
                handle,
                offset,
                buf,
            },
            Self::Fsync { file, .. } => IoOperation::Fsync { file, handle },
            Self::SyncData { file, .. } => IoOperation::SyncData { file, handle },
            Self::Truncate { file, len, .. } => IoOperation::Truncate { file, handle, len },
            Self::Metadata { file, .. } => IoOperation::Metadata { file, handle },
        }
    }
}

#[derive(Debug)]
pub(super) enum IoOperation {
    Read {
        file: FileId,
        handle: Option<Arc<File>>,
        offset: u64,
        len: usize,
    },
    Write {
        file: FileId,
        handle: Option<Arc<File>>,
        offset: u64,
        buf: Vec<u8>,
    },
    Fsync {
        file: FileId,
        handle: Option<Arc<File>>,
    },
    SyncData {
        file: FileId,
        handle: Option<Arc<File>>,
    },
    Truncate {
        file: FileId,
        handle: Option<Arc<File>>,
        len: u64,
    },
    Metadata {
        file: FileId,
        handle: Option<Arc<File>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoMetadata {
    pub len: u64,
    pub is_file: bool,
    pub is_dir: bool,
    pub readonly: bool,
}

impl From<Metadata> for IoMetadata {
    fn from(metadata: Metadata) -> Self {
        Self {
            len: metadata.len(),
            is_file: metadata.is_file(),
            is_dir: metadata.is_dir(),
            readonly: metadata.permissions().readonly(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoOutput {
    Read(Arc<[u8]>),
    Write { bytes: usize },
    SyncAll,
    SyncData,
    Truncate,
    Metadata(IoMetadata),
}

impl IoOutput {
    pub fn read_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Read(bytes) => Some(bytes),
            _ => None,
        }
    }

    pub fn into_read_bytes(self) -> io::Result<Arc<[u8]>> {
        match self {
            Self::Read(bytes) => Ok(bytes),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected read output, got {other:?}"),
            )),
        }
    }

    pub fn bytes_written(&self) -> Option<usize> {
        match self {
            Self::Write { bytes } => Some(*bytes),
            _ => None,
        }
    }

    pub fn metadata(&self) -> Option<&IoMetadata> {
        match self {
            Self::Metadata(metadata) => Some(metadata),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct SharedIoError {
    kind: io::ErrorKind,
    raw_os_error: Option<i32>,
    message: Arc<str>,
}

impl From<io::Error> for SharedIoError {
    fn from(err: io::Error) -> Self {
        Self {
            kind: err.kind(),
            raw_os_error: err.raw_os_error(),
            message: Arc::from(err.to_string()),
        }
    }
}

impl SharedIoError {
    fn into_io_error(self) -> io::Error {
        if let Some(raw_os_error) = self.raw_os_error {
            return io::Error::from_raw_os_error(raw_os_error);
        }
        io::Error::new(self.kind, self.message.to_string())
    }

    fn to_io_error(&self) -> io::Error {
        if let Some(raw_os_error) = self.raw_os_error {
            return io::Error::from_raw_os_error(raw_os_error);
        }
        io::Error::new(self.kind, self.message.to_string())
    }
}

type SharedIoResult = Result<IoOutput, SharedIoError>;
type IoResultSender = tokio::sync::oneshot::Sender<SharedIoResult>;

fn shared_to_io_result(result: SharedIoResult) -> io::Result<IoOutput> {
    result.map_err(SharedIoError::into_io_error)
}

#[derive(Debug)]
enum Waiters {
    One(IoResultSender),
    Many(Vec<IoResultSender>),
}

impl Waiters {
    fn new(waiter: IoResultSender) -> Self {
        Self::One(waiter)
    }

    fn push(&mut self, waiter: IoResultSender) {
        match self {
            Self::One(_) => {
                let first = match std::mem::replace(self, Self::Many(Vec::with_capacity(2))) {
                    Self::One(first) => first,
                    Self::Many(_) => unreachable!("waiters variant changed during replace"),
                };
                let Self::Many(waiters) = self else {
                    unreachable!("waiters replaced with Many");
                };
                waiters.push(first);
                waiters.push(waiter);
            }
            Self::Many(waiters) => waiters.push(waiter),
        }
    }

    fn send(self, result: SharedIoResult) {
        match self {
            Self::One(waiter) => {
                let _ = waiter.send(result);
            }
            Self::Many(mut waiters) => {
                let Some(last) = waiters.pop() else {
                    return;
                };
                for waiter in waiters {
                    let _ = waiter.send(result.clone());
                }
                let _ = last.send(result);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct WorkId(u64);

#[derive(Debug)]
pub(super) struct IoWork {
    pub(super) id: WorkId,
    pub(super) op: IoOperation,
    pub(super) queued_at: ProfileTimer,
    pub(super) operation_started: ProfileTimer,
}

#[cfg(feature = "uring")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FixedFileUpdate {
    pub(super) file: FileId,
    pub(super) fd: RawFd,
}

#[derive(Debug)]
pub(super) enum FileSlot {
    Open(Arc<File>),
    Closing { handle: Arc<File>, reopenable: bool },
    Closed,
}

pub(super) type FileTable = RwLock<Vec<FileSlot>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClosingMark {
    NewlyMarked,
    AlreadyClosing,
}

impl FileSlot {
    fn open_file(&self) -> Option<Arc<File>> {
        match self {
            Self::Open(file) => Some(Arc::clone(file)),
            Self::Closing { .. } | Self::Closed => None,
        }
    }
}

#[cfg(feature = "uring")]
#[derive(Debug)]
pub(super) enum QueueWake {
    Work(IoWork),
    Notified,
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InflightState {
    Queued,
    Dispatched,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    Read { file: FileId },
    Write { file: FileId },
    Sync { file: FileId },
    Truncate { file: FileId },
    Metadata { file: FileId },
}

impl Access {
    fn for_command(cmd: &IoCommand) -> Self {
        match cmd {
            IoCommand::Read { file, .. } => Self::Read { file: *file },
            IoCommand::Write { file, .. } => Self::Write { file: *file },
            IoCommand::Fsync { file, .. } | IoCommand::SyncData { file, .. } => {
                Self::Sync { file: *file }
            }
            IoCommand::Truncate { file, .. } => Self::Truncate { file: *file },
            IoCommand::Metadata { file, .. } => Self::Metadata { file: *file },
        }
    }

    fn file(self) -> FileId {
        match self {
            Self::Read { file, .. }
            | Self::Write { file, .. }
            | Self::Sync { file }
            | Self::Truncate { file }
            | Self::Metadata { file } => file,
        }
    }
}

#[derive(Debug)]
struct Inflight {
    key: Option<RequestKey>,
    op: Option<IoOperation>,
    access: Access,
    waiters: Waiters,
    _queue_permit: QueuePermit,
    refcount: usize,
    state: InflightState,
    priority: usize,
    seq: u64,
    fast_read: bool,
    queued_at: ProfileTimer,
}

struct QueuePermit {
    permit: Option<OwnedSemaphorePermit>,
    signal: Arc<QueueSignal>,
}

impl std::fmt::Debug for QueuePermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("QueuePermit")
    }
}

impl QueuePermit {
    fn new(permit: OwnedSemaphorePermit, signal: Arc<QueueSignal>) -> Self {
        Self {
            permit: Some(permit),
            signal,
        }
    }
}

impl Drop for QueuePermit {
    fn drop(&mut self) {
        if self.signal.waiters.load(Ordering::Acquire) == 0 {
            self.permit.take();
            return;
        }

        let guard = self
            .signal
            .mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.permit.take();
        drop(guard);
        self.signal.available.notify_one();
    }
}

#[derive(Debug, Default)]
struct QueueSignal {
    waiters: AtomicUsize,
    mutex: Mutex<()>,
    available: Condvar,
}

#[derive(Debug, Default)]
struct FileActive {
    all: BTreeSet<(u64, WorkId)>,
    writes: BTreeSet<(u64, WorkId)>,
    barriers: BTreeSet<(u64, WorkId)>,
    truncates: BTreeSet<(u64, WorkId)>,
    metadata: BTreeSet<(u64, WorkId)>,
}

#[derive(Debug)]
struct Inner {
    priority_levels: usize,
    queued: usize,
    fast_read_path: bool,
    fast_ready: Vec<VecDeque<WorkId>>,
    fast_active_reads: Vec<usize>,
    ready: BTreeSet<(usize, u64, WorkId)>,
    ready_refresh: Vec<WorkId>,
    #[cfg(feature = "uring")]
    fixed_file_updates: Vec<FixedFileUpdate>,
    #[cfg(feature = "uring")]
    fixed_file_update_base: usize,
    #[cfg(feature = "uring")]
    fixed_file_update_cursors: HashMap<RawFd, usize>,
    table: HashMap<WorkId, Inflight>,
    dedup: HashMap<RequestKey, WorkId>,
    active_by_file: Vec<Option<FileActive>>,
    next_id: u64,
    next_seq: u64,
    shutdown: bool,
    failure: Option<SharedIoError>,
}

#[derive(Debug)]
struct ActiveLimiter {
    max: usize,
    active: AtomicUsize,
    wake_epoch: AtomicUsize,
    signal: QueueSignal,
}

impl ActiveLimiter {
    fn new(max: usize) -> Self {
        Self {
            max: max.max(1),
            active: AtomicUsize::new(0),
            wake_epoch: AtomicUsize::new(0),
            signal: QueueSignal::default(),
        }
    }

    fn try_acquire(&self) -> bool {
        let mut current = self.active.load(Ordering::Relaxed);
        loop {
            if current >= self.max {
                return false;
            }
            match self.active.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(next) => current = next,
            }
        }
    }

    fn release(&self) {
        let previous = self.active.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        self.notify_one_waiter();
    }

    #[cfg(any(feature = "uring", test))]
    fn release_many(&self, count: usize) {
        if count == 0 {
            return;
        }
        let previous = self.active.fetch_sub(count, Ordering::AcqRel);
        debug_assert!(previous >= count);
        self.notify_all_waiters();
    }

    fn wait_for_capacity_until<F>(&self, mut should_stop: F) -> bool
    where
        F: FnMut() -> bool,
    {
        if self.active.load(Ordering::Acquire) < self.max {
            return false;
        }
        if should_stop() {
            return true;
        }

        let observed_epoch = self.wake_epoch.load(Ordering::Acquire);
        self.signal.waiters.fetch_add(1, Ordering::AcqRel);
        let mut guard = self
            .signal
            .mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut stopped = false;
        while self.active.load(Ordering::Acquire) >= self.max
            && self.wake_epoch.load(Ordering::Acquire) == observed_epoch
        {
            if should_stop() {
                stopped = true;
                break;
            }
            guard = self
                .signal
                .available
                .wait(guard)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        drop(guard);
        self.signal.waiters.fetch_sub(1, Ordering::AcqRel);
        stopped
    }

    fn notify_one_waiter(&self) {
        if self.signal.waiters.load(Ordering::Acquire) == 0 {
            return;
        }
        let guard = self
            .signal
            .mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.wake_epoch.fetch_add(1, Ordering::AcqRel);
        drop(guard);
        self.signal.available.notify_one();
    }

    fn notify_all_waiters(&self) {
        if self.signal.waiters.load(Ordering::Acquire) == 0 {
            return;
        }
        let guard = self
            .signal
            .mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.wake_epoch.fetch_add(1, Ordering::AcqRel);
        drop(guard);
        self.signal.available.notify_all();
    }

    #[cfg(test)]
    fn waiter_count(&self) -> usize {
        self.signal.waiters.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub(super) struct QueueCore {
    inner: Mutex<Inner>,
    active_limiter: Arc<ActiveLimiter>,
    queue_limiter: Arc<Semaphore>,
    queue_signal: Arc<QueueSignal>,
    queue_capacity: usize,
    shutdown_flag: AtomicBool,
    profiler: IoPoolProfile,
    available: Condvar,
    inactive_waiters: AtomicUsize,
    #[cfg(feature = "uring")]
    wake_fds: Mutex<Vec<RawFd>>,
    #[cfg(feature = "uring")]
    wake_pending: AtomicBool,
}

#[derive(Debug)]
enum QueueSubmitError {
    Full(IoCommand),
    Io(io::Error),
}

impl From<io::Error> for QueueSubmitError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

enum PopAttempt {
    Work(IoWork),
    NoReady,
    ActiveFull,
}

impl QueueCore {
    #[cfg(test)]
    pub(super) fn new(priority_levels: usize, max_active: usize) -> Self {
        Self::new_with_capacity(priority_levels, max_active, max_active)
    }

    #[cfg(test)]
    pub(super) fn new_with_capacity(
        priority_levels: usize,
        queue_capacity: usize,
        max_active: usize,
    ) -> Self {
        Self::new_with_fast_read_path(priority_levels, queue_capacity, max_active, false)
    }

    #[cfg(test)]
    pub(super) fn new_with_fast_read_path(
        priority_levels: usize,
        queue_capacity: usize,
        max_active: usize,
        fast_read_path: bool,
    ) -> Self {
        Self::new_with_limiter(
            priority_levels,
            queue_capacity,
            max_active,
            fast_read_path,
            Arc::new(ActiveLimiter::new(max_active)),
            Arc::new(Semaphore::new(queue_capacity.max(1))),
            Arc::new(QueueSignal::default()),
        )
    }

    #[cfg(test)]
    fn new_with_limiter(
        priority_levels: usize,
        queue_capacity: usize,
        max_active: usize,
        fast_read_path: bool,
        active_limiter: Arc<ActiveLimiter>,
        queue_limiter: Arc<Semaphore>,
        queue_signal: Arc<QueueSignal>,
    ) -> Self {
        Self::new_with_limiter_and_profile(
            priority_levels,
            queue_capacity,
            max_active,
            fast_read_path,
            active_limiter,
            queue_limiter,
            queue_signal,
            IoPoolProfile::disabled(),
        )
    }

    fn new_with_limiter_and_profile(
        priority_levels: usize,
        queue_capacity: usize,
        _max_active: usize,
        fast_read_path: bool,
        active_limiter: Arc<ActiveLimiter>,
        queue_limiter: Arc<Semaphore>,
        queue_signal: Arc<QueueSignal>,
        profiler: IoPoolProfile,
    ) -> Self {
        let priority_levels = priority_levels.max(1);
        let queue_capacity = queue_capacity.max(1);
        Self {
            inner: Mutex::new(Inner {
                priority_levels,
                queued: 0,
                fast_read_path,
                fast_ready: (0..priority_levels).map(|_| VecDeque::new()).collect(),
                fast_active_reads: Vec::new(),
                ready: BTreeSet::new(),
                ready_refresh: Vec::new(),
                #[cfg(feature = "uring")]
                fixed_file_updates: Vec::new(),
                #[cfg(feature = "uring")]
                fixed_file_update_base: 0,
                #[cfg(feature = "uring")]
                fixed_file_update_cursors: HashMap::new(),
                table: HashMap::new(),
                dedup: HashMap::new(),
                active_by_file: Vec::new(),
                next_id: 0,
                next_seq: 0,
                shutdown: false,
                failure: None,
            }),
            active_limiter,
            queue_limiter,
            queue_signal,
            queue_capacity,
            shutdown_flag: AtomicBool::new(false),
            profiler,
            available: Condvar::new(),
            inactive_waiters: AtomicUsize::new(0),
            #[cfg(feature = "uring")]
            wake_fds: Mutex::new(Vec::new()),
            #[cfg(feature = "uring")]
            wake_pending: AtomicBool::new(false),
        }
    }

    #[cfg(test)]
    fn submit(self: &Arc<Self>, cmd: IoCommand) -> io::Result<IoFuture> {
        match self.try_submit_with_handle(cmd, None, true) {
            Ok(future) => Ok(future),
            Err(QueueSubmitError::Full(cmd)) => {
                let permit = self.blocking_acquire_queue_permit()?;
                self.submit_with_handle_permit(cmd, None, permit, false)
            }
            Err(QueueSubmitError::Io(err)) => Err(err),
        }
    }

    async fn acquire_queue_permit(self: &Arc<Self>) -> io::Result<QueuePermit> {
        Arc::clone(&self.queue_limiter)
            .acquire_owned()
            .await
            .map(|permit| QueuePermit::new(permit, Arc::clone(&self.queue_signal)))
            .map_err(|_| self.queue_closed_error())
    }

    fn blocking_acquire_queue_permit(self: &Arc<Self>) -> io::Result<QueuePermit> {
        match Arc::clone(&self.queue_limiter).try_acquire_owned() {
            Ok(permit) => {
                return Ok(QueuePermit::new(permit, Arc::clone(&self.queue_signal)));
            }
            Err(TryAcquireError::NoPermits) => {}
            Err(TryAcquireError::Closed) => {
                return Err(self.queue_closed_error());
            }
        }

        self.queue_signal.waiters.fetch_add(1, Ordering::AcqRel);
        let mut guard = self
            .queue_signal
            .mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = loop {
            match Arc::clone(&self.queue_limiter).try_acquire_owned() {
                Ok(permit) => {
                    break Ok(QueuePermit::new(permit, Arc::clone(&self.queue_signal)));
                }
                Err(TryAcquireError::NoPermits) => {
                    guard = self
                        .queue_signal
                        .available
                        .wait(guard)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
                Err(TryAcquireError::Closed) => {
                    break Err(());
                }
            }
        };
        drop(guard);
        self.queue_signal.waiters.fetch_sub(1, Ordering::AcqRel);
        match result {
            Ok(permit) => Ok(permit),
            Err(()) => Err(self.queue_closed_error()),
        }
    }

    fn queue_full_error(&self) -> io::Error {
        io::Error::new(
            io::ErrorKind::WouldBlock,
            format!("IO queue reached queue_capacity={}", self.queue_capacity),
        )
    }

    fn queue_closed_error(&self) -> io::Error {
        let inner = self.inner.lock().unwrap();
        if let Some(failure) = &inner.failure {
            return failure.to_io_error();
        }
        if inner.shutdown {
            return io::Error::new(io::ErrorKind::BrokenPipe, "IO queue is shut down");
        }
        io::Error::new(
            io::ErrorKind::BrokenPipe,
            "IO queue capacity limiter is closed",
        )
    }

    fn try_submit_with_handle(
        self: &Arc<Self>,
        cmd: IoCommand,
        handle: Option<Arc<File>>,
        record_submit: bool,
    ) -> Result<IoFuture, QueueSubmitError> {
        if record_submit {
            record_iopool_submit(&self.profiler, cmd.profile_kind());
        }
        let priority = cmd.priority();
        let key = cmd.dedup_key();
        let access = Access::for_command(&cmd);
        let immediate = cmd.immediate_output();
        let (tx, rx) = tokio::sync::oneshot::channel();

        let mut inner = self.inner.lock().unwrap();
        if let Some(failure) = &inner.failure {
            return Err(QueueSubmitError::Io(failure.to_io_error()));
        }
        if inner.shutdown {
            return Err(QueueSubmitError::Io(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "IO queue is shut down",
            )));
        }
        if priority >= inner.priority_levels {
            return Err(QueueSubmitError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "priority {priority} is out of range for {} priority levels",
                    inner.priority_levels
                ),
            )));
        }

        if let Some(output) = immediate {
            if Self::can_complete_immediately_locked(&inner, access) {
                drop(inner);
                record_iopool_immediate(&self.profiler);
                let _ = tx.send(Ok(output));
                return Ok(IoFuture::detached(rx));
            }
        }

        if let Some(key) = key {
            if let Some(id) = inner.dedup.get(&key).copied() {
                if inner
                    .table
                    .get(&id)
                    .is_some_and(|inflight| Self::can_dedup_locked(&inner, id, inflight, access))
                {
                    if inner
                        .table
                        .get(&id)
                        .is_some_and(|inflight| inflight.refcount == usize::MAX)
                    {
                        return Err(QueueSubmitError::Io(refcount_overflow()));
                    }
                    Self::promote_priority_locked(&mut inner, id, priority);
                    let inflight = inner
                        .table
                        .get_mut(&id)
                        .expect("deduplicated entry checked above");
                    inflight.refcount += 1;
                    inflight.waiters.push(tx);
                    drop(inner);
                    record_iopool_dedup_hit(&self.profiler);
                    return Ok(IoFuture::new(rx, Arc::downgrade(self), id));
                }
                if !inner.table.contains_key(&id) {
                    inner.dedup.remove(&key);
                }
            }
        }

        let queue_permit = match Arc::clone(&self.queue_limiter).try_acquire_owned() {
            Ok(permit) => QueuePermit::new(permit, Arc::clone(&self.queue_signal)),
            Err(TryAcquireError::NoPermits) => {
                drop(inner);
                record_iopool_queue_full(&self.profiler);
                return Err(QueueSubmitError::Full(cmd));
            }
            Err(TryAcquireError::Closed) => {
                drop(inner);
                return Err(QueueSubmitError::Io(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "IO queue capacity limiter is closed",
                )));
            }
        };

        Self::insert_locked(
            self,
            inner,
            cmd,
            handle,
            queue_permit,
            priority,
            key,
            access,
            tx,
            rx,
        )
    }

    fn submit_with_handle_permit(
        self: &Arc<Self>,
        cmd: IoCommand,
        handle: Option<Arc<File>>,
        queue_permit: QueuePermit,
        record_submit: bool,
    ) -> io::Result<IoFuture> {
        if record_submit {
            record_iopool_submit(&self.profiler, cmd.profile_kind());
        }
        let priority = cmd.priority();
        let key = cmd.dedup_key();
        let access = Access::for_command(&cmd);
        let immediate = cmd.immediate_output();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let mut inner = self.inner.lock().unwrap();
        if let Some(failure) = &inner.failure {
            return Err(failure.to_io_error());
        }
        if inner.shutdown {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "IO queue is shut down",
            ));
        }
        if priority >= inner.priority_levels {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "priority {priority} is out of range for {} priority levels",
                    inner.priority_levels
                ),
            ));
        }
        if let Some(output) = immediate {
            if Self::can_complete_immediately_locked(&inner, access) {
                drop(inner);
                record_iopool_immediate(&self.profiler);
                drop(queue_permit);
                let _ = tx.send(Ok(output));
                return Ok(IoFuture::detached(rx));
            }
        }
        if let Some(key) = key {
            if let Some(id) = inner.dedup.get(&key).copied() {
                if inner
                    .table
                    .get(&id)
                    .is_some_and(|inflight| Self::can_dedup_locked(&inner, id, inflight, access))
                {
                    if inner
                        .table
                        .get(&id)
                        .is_some_and(|inflight| inflight.refcount == usize::MAX)
                    {
                        return Err(refcount_overflow());
                    }
                    Self::promote_priority_locked(&mut inner, id, priority);
                    let inflight = inner
                        .table
                        .get_mut(&id)
                        .expect("deduplicated entry checked above");
                    inflight.refcount += 1;
                    inflight.waiters.push(tx);
                    drop(inner);
                    drop(queue_permit);
                    record_iopool_dedup_hit(&self.profiler);
                    return Ok(IoFuture::new(rx, Arc::downgrade(self), id));
                }
                if !inner.table.contains_key(&id) {
                    inner.dedup.remove(&key);
                }
            }
        }
        Self::insert_locked(
            self,
            inner,
            cmd,
            handle,
            queue_permit,
            priority,
            key,
            access,
            tx,
            rx,
        )
        .map_err(|err| match err {
            QueueSubmitError::Io(err) => err,
            QueueSubmitError::Full(_) => unreachable!("queue permit was already acquired"),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_locked(
        this: &Arc<Self>,
        mut inner: std::sync::MutexGuard<'_, Inner>,
        cmd: IoCommand,
        handle: Option<Arc<File>>,
        queue_permit: QueuePermit,
        priority: usize,
        key: Option<RequestKey>,
        access: Access,
        tx: IoResultSender,
        rx: tokio::sync::oneshot::Receiver<SharedIoResult>,
    ) -> Result<IoFuture, QueueSubmitError> {
        let id = WorkId(inner.next_id);
        inner.next_id = inner.next_id.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::OutOfMemory, "IO work id counter overflowed")
        })?;
        let seq = inner.next_seq;
        inner.next_seq = inner.next_seq.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::OutOfMemory, "IO sequence counter overflowed")
        })?;

        let fast_read = Self::can_use_fast_read_locked(&inner, access);
        let queued_at = this.profiler.queue_timer();
        let operation = cmd.into_operation(handle);

        inner.table.insert(
            id,
            Inflight {
                key,
                op: Some(operation),
                access,
                waiters: Waiters::new(tx),
                _queue_permit: queue_permit,
                refcount: 1,
                state: InflightState::Queued,
                priority,
                seq,
                fast_read,
                queued_at,
            },
        );
        if fast_read {
            Self::insert_fast_read_locked(&mut inner, access.file(), priority, id);
        } else {
            Self::insert_active_index_locked(&mut inner, access.file(), seq, id);
        }
        if let Some(key) = key {
            inner.dedup.insert(key, id);
        }
        inner.queued += 1;
        let became_ready = fast_read || Self::insert_ready_if_dispatchable_locked(&mut inner, id);
        drop(inner);

        if became_ready {
            this.notify_one();
        }
        Ok(IoFuture::new(rx, Arc::downgrade(this), id))
    }

    pub(super) fn pop(&self) -> Option<IoWork> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            match self.pop_ready_with_active_locked(&mut inner) {
                PopAttempt::Work(work) => {
                    drop(inner);
                    return Some(self.profile_dispatched(work));
                }
                PopAttempt::NoReady => {}
                PopAttempt::ActiveFull => {
                    if inner.shutdown {
                        return None;
                    }
                    drop(inner);
                    if self
                        .active_limiter
                        .wait_for_capacity_until(|| self.is_shutdown())
                    {
                        return None;
                    }
                    inner = self.inner.lock().unwrap();
                    continue;
                }
            }

            if inner.shutdown {
                return None;
            }

            inner = self.available.wait(inner).unwrap();
        }
    }

    #[cfg(test)]
    fn try_pop_for_test(&self) -> Option<IoWork> {
        let mut inner = self.inner.lock().unwrap();
        match self.pop_ready_with_active_locked(&mut inner) {
            PopAttempt::Work(work) => {
                drop(inner);
                Some(self.profile_dispatched(work))
            }
            PopAttempt::NoReady | PopAttempt::ActiveFull => None,
        }
    }

    #[cfg(feature = "uring")]
    pub(super) fn pop_or_notification(&self, fixed_file_update_cursor: usize) -> QueueWake {
        let mut inner = self.inner.lock().unwrap();
        loop {
            match self.pop_ready_with_active_locked(&mut inner) {
                PopAttempt::Work(work) => {
                    drop(inner);
                    return QueueWake::Work(self.profile_dispatched(work));
                }
                PopAttempt::NoReady => {}
                PopAttempt::ActiveFull => {
                    if inner.shutdown {
                        return QueueWake::Shutdown;
                    }
                    drop(inner);
                    if self
                        .active_limiter
                        .wait_for_capacity_until(|| self.is_shutdown())
                    {
                        return QueueWake::Shutdown;
                    }
                    inner = self.inner.lock().unwrap();
                    continue;
                }
            }

            if inner.shutdown {
                return QueueWake::Shutdown;
            }
            if Self::fixed_file_update_tail_locked(&inner) > fixed_file_update_cursor {
                return QueueWake::Notified;
            }

            inner = self.available.wait(inner).unwrap();
            if !inner.shutdown
                && inner.ready.is_empty()
                && Self::fixed_file_update_tail_locked(&inner) > fixed_file_update_cursor
            {
                return QueueWake::Notified;
            }
        }
    }

    #[cfg(feature = "uring")]
    pub(super) fn try_pop_batch(&self, out: &mut Vec<IoWork>, limit: usize) {
        if limit == 0 {
            return;
        }
        let start = out.len();
        let mut inner = self.inner.lock().unwrap();
        for _ in 0..limit {
            match self.pop_ready_with_active_locked(&mut inner) {
                PopAttempt::Work(work) => out.push(work),
                PopAttempt::NoReady | PopAttempt::ActiveFull => break,
            }
        }
        drop(inner);
        for work in &mut out[start..] {
            self.profile_dispatched_in_place(work);
        }
    }

    fn pop_ready_with_active_locked(&self, inner: &mut Inner) -> PopAttempt {
        if !Self::has_ready_candidate_locked(inner) {
            return PopAttempt::NoReady;
        }
        if !self.active_limiter.try_acquire() {
            return PopAttempt::ActiveFull;
        }
        match Self::pop_ready_locked(inner) {
            Some(work) => PopAttempt::Work(work),
            None => {
                self.active_limiter.release();
                PopAttempt::NoReady
            }
        }
    }

    fn has_ready_candidate_locked(inner: &Inner) -> bool {
        !inner.ready.is_empty() || inner.fast_ready.iter().any(|ready| !ready.is_empty())
    }

    fn is_shutdown(&self) -> bool {
        self.shutdown_flag.load(Ordering::Acquire)
    }

    fn profile_dispatched(&self, mut work: IoWork) -> IoWork {
        self.profile_dispatched_in_place(&mut work);
        work
    }

    fn profile_dispatched_in_place(&self, work: &mut IoWork) {
        record_iopool_dispatched(&self.profiler, work.queued_at);
        work.operation_started = self.profiler.operation_timer();
    }

    pub(super) fn complete(
        &self,
        id: WorkId,
        operation_started: ProfileTimer,
        result: io::Result<IoOutput>,
    ) {
        let (waiters, ready_count, file_inactive) = {
            let mut inner = self.inner.lock().unwrap();
            let Some(inflight) = Self::remove_locked(&mut inner, id) else {
                return;
            };
            let access = inflight.access;
            let file = access.file();
            let waiters = inflight.waiters;
            let ready_count = Self::refresh_ready_after_removed_locked(&mut inner, access);
            let file_inactive = !Self::file_active_locked(&inner, file);
            (waiters, ready_count, file_inactive)
        };

        if self.profiler.is_recording() {
            let (read_bytes, write_bytes) = io_result_profile_bytes(&result);
            record_iopool_operation(
                &self.profiler,
                operation_started,
                read_bytes,
                write_bytes,
                result.is_err(),
            );
        }
        self.active_limiter.release();
        self.notify_ready_count(ready_count);
        self.notify_file_inactive(file_inactive);
        let result = result.map_err(SharedIoError::from);
        waiters.send(result);
    }

    fn release(&self, id: WorkId) {
        let mut inner = self.inner.lock().unwrap();
        let Some(inflight) = inner.table.get_mut(&id) else {
            return;
        };

        inflight.refcount = inflight.refcount.saturating_sub(1);
        let should_remove = inflight.refcount == 0 && inflight.state == InflightState::Queued;
        let access = inflight.access;
        let file = access.file();
        if should_remove {
            inner.queued = inner.queued.saturating_sub(1);
            Self::remove_locked(&mut inner, id);
            let ready_count = Self::refresh_ready_after_removed_locked(&mut inner, access);
            let file_inactive = !Self::file_active_locked(&inner, file);
            drop(inner);
            record_iopool_cancelled_before_dispatch(&self.profiler);
            self.notify_ready_count(ready_count);
            self.notify_file_inactive(file_inactive);
        }
    }

    pub(super) fn shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::Release);
        let mut inner = self.inner.lock().unwrap();
        inner.shutdown = true;
        drop(inner);
        self.close_queue_capacity_limiter();
        self.active_limiter.notify_all_waiters();
        self.notify_all();
    }

    #[cfg(any(feature = "uring", test))]
    pub(super) fn fail(&self, err: io::Error) {
        self.shutdown_flag.store(true, Ordering::Release);
        let failure = SharedIoError::from(err);
        let (waiters, released_active) = {
            let mut inner = self.inner.lock().unwrap();
            inner.shutdown = true;
            inner.failure = Some(failure.clone());
            let released_active = inner
                .table
                .values()
                .filter(|inflight| inflight.state == InflightState::Dispatched)
                .count();
            (Self::drain_locked(&mut inner), released_active)
        };

        self.active_limiter.release_many(released_active);
        self.active_limiter.notify_all_waiters();
        for waiters in waiters {
            waiters.send(Err(failure.clone()));
        }
        self.close_queue_capacity_limiter();
        self.notify_all();
    }

    fn close_queue_capacity_limiter(&self) {
        let guard = self
            .queue_signal
            .mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.queue_limiter.close();
        drop(guard);
        self.queue_signal.available.notify_all();
    }

    fn notify_one(&self) {
        self.available.notify_one();
        self.wake_backend();
    }

    fn notify_all(&self) {
        self.available.notify_all();
        self.wake_backend();
    }

    fn notify_ready_count(&self, ready_count: usize) {
        match ready_count {
            0 => {}
            1 => {
                self.available.notify_one();
                self.wake_backend();
            }
            count if count <= TARGETED_READY_NOTIFIES => {
                for _ in 0..count {
                    self.available.notify_one();
                }
                self.wake_backend();
            }
            _ => self.notify_all(),
        }
    }

    fn notify_file_inactive(&self, file_inactive: bool) {
        if file_inactive && self.inactive_waiters.load(Ordering::Acquire) > 0 {
            self.available.notify_all();
        }
    }

    #[cfg(feature = "uring")]
    pub(super) fn register_wake_fd(&self, fd: RawFd) {
        let mut inner = self.inner.lock().unwrap();
        let base = inner.fixed_file_update_base;
        inner.fixed_file_update_cursors.insert(fd, base);
        drop(inner);
        self.wake_fds.lock().unwrap().push(fd);
    }

    #[cfg(feature = "uring")]
    pub(super) fn unregister_wake_fd(&self, fd: RawFd) {
        let mut inner = self.inner.lock().unwrap();
        inner.fixed_file_update_cursors.remove(&fd);
        Self::compact_fixed_file_updates_locked(&mut inner);
        drop(inner);
        let mut wake_fds = self.wake_fds.lock().unwrap();
        if let Some(index) = wake_fds.iter().position(|wake_fd| *wake_fd == fd) {
            wake_fds.swap_remove(index);
        }
        drop(wake_fds);
        self.wake_pending.store(false, Ordering::Release);
    }

    #[cfg(feature = "uring")]
    pub(super) fn acknowledge_wake(&self) {
        self.wake_pending.store(false, Ordering::Release);
    }

    #[cfg(feature = "uring")]
    pub(super) fn push_fixed_file_update(&self, update: FixedFileUpdate) {
        let mut inner = self.inner.lock().unwrap();
        inner.fixed_file_updates.push(update);
        drop(inner);
        self.notify_all();
    }

    #[cfg(feature = "uring")]
    pub(super) fn fixed_file_updates_since(
        &self,
        reader_fd: RawFd,
        cursor: usize,
        out: &mut Vec<FixedFileUpdate>,
    ) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let base = inner.fixed_file_update_base;
        let tail = Self::fixed_file_update_tail_locked(&inner);
        let cursor = cursor.max(base).min(tail);
        let offset = cursor - base;
        out.extend_from_slice(&inner.fixed_file_updates[offset..]);
        inner.fixed_file_update_cursors.insert(reader_fd, tail);
        Self::compact_fixed_file_updates_locked(&mut inner);
        tail
    }

    #[cfg(feature = "uring")]
    fn fixed_file_update_tail_locked(inner: &Inner) -> usize {
        inner.fixed_file_update_base + inner.fixed_file_updates.len()
    }

    #[cfg(feature = "uring")]
    fn compact_fixed_file_updates_locked(inner: &mut Inner) {
        let Some(min_cursor) = inner.fixed_file_update_cursors.values().copied().min() else {
            return;
        };
        let drain_len = min_cursor
            .saturating_sub(inner.fixed_file_update_base)
            .min(inner.fixed_file_updates.len());
        if drain_len == 0 {
            return;
        }
        inner.fixed_file_updates.drain(..drain_len);
        inner.fixed_file_update_base += drain_len;
    }

    #[cfg(feature = "uring")]
    fn wake_backend(&self) {
        if self.wake_pending.swap(true, Ordering::AcqRel) {
            return;
        }
        for fd in self.wake_fds.lock().unwrap().iter().copied() {
            wake_eventfd(fd);
        }
    }

    #[cfg(not(feature = "uring"))]
    fn wake_backend(&self) {}

    fn remove_locked(inner: &mut Inner, id: WorkId) -> Option<Inflight> {
        let inflight = inner.table.remove(&id)?;
        if inflight.fast_read {
            Self::remove_fast_read_locked(inner, inflight.access.file());
        } else {
            inner.ready.remove(&Self::ready_key_locked(&inflight, id));
            Self::remove_active_index_locked(inner, inflight.access, inflight.seq, id);
        }
        if let Some(key) = inflight.key {
            if inner
                .dedup
                .get(&key)
                .is_some_and(|dedup_id| *dedup_id == id)
            {
                inner.dedup.remove(&key);
            }
        }
        Some(inflight)
    }

    fn ready_key_locked(inflight: &Inflight, id: WorkId) -> (usize, u64, WorkId) {
        (inflight.priority, inflight.seq, id)
    }

    fn can_use_fast_read_locked(inner: &Inner, access: Access) -> bool {
        if !inner.fast_read_path || !matches!(access, Access::Read { .. }) {
            return false;
        }
        inner
            .active_by_file
            .get(access.file())
            .and_then(Option::as_ref)
            .map_or(true, |active| active.barriers.is_empty())
    }

    fn can_complete_immediately_locked(inner: &Inner, access: Access) -> bool {
        let Some(active) = inner
            .active_by_file
            .get(access.file())
            .and_then(Option::as_ref)
        else {
            return true;
        };

        match access {
            Access::Read { .. } => active.barriers.is_empty(),
            Access::Write { .. } => active.barriers.is_empty() && active.metadata.is_empty(),
            Access::Sync { .. } | Access::Truncate { .. } | Access::Metadata { .. } => false,
        }
    }

    fn insert_fast_read_locked(inner: &mut Inner, file: FileId, priority: usize, id: WorkId) {
        if inner.fast_active_reads.len() <= file {
            inner.fast_active_reads.resize(file + 1, 0);
        }
        inner.fast_active_reads[file] += 1;
        inner.fast_ready[priority].push_back(id);
    }

    fn remove_fast_read_locked(inner: &mut Inner, file: FileId) {
        if let Some(count) = inner.fast_active_reads.get_mut(file) {
            *count = count.saturating_sub(1);
        }
    }

    fn fast_active_read_count(inner: &Inner, file: FileId) -> usize {
        inner.fast_active_reads.get(file).copied().unwrap_or(0)
    }

    fn next_fast_ready_key_locked(inner: &mut Inner) -> Option<(usize, u64, WorkId)> {
        for priority in 0..inner.fast_ready.len() {
            while let Some(id) = inner.fast_ready[priority].front().copied() {
                let Some(inflight) = inner.table.get(&id) else {
                    inner.fast_ready[priority].pop_front();
                    continue;
                };
                if inflight.fast_read
                    && inflight.priority == priority
                    && inflight.refcount > 0
                    && inflight.state == InflightState::Queued
                {
                    return Some((priority, inflight.seq, id));
                }
                inner.fast_ready[priority].pop_front();
            }
        }
        None
    }

    fn insert_ready_if_dispatchable_locked(inner: &mut Inner, id: WorkId) -> bool {
        let Some(inflight) = inner.table.get(&id) else {
            return false;
        };
        if inflight.fast_read {
            return false;
        }
        if inflight.refcount == 0 || inflight.state != InflightState::Queued {
            return false;
        }
        if Self::is_dispatchable_locked(inner, id, inflight) {
            return inner.ready.insert(Self::ready_key_locked(inflight, id));
        }
        false
    }

    fn refresh_ready_after_removed_locked(inner: &mut Inner, removed: Access) -> usize {
        let file = removed.file();
        if inner
            .active_by_file
            .get(file)
            .and_then(Option::as_ref)
            .is_none()
        {
            return 0;
        };

        let mut ids = std::mem::take(&mut inner.ready_refresh);
        ids.clear();
        {
            let active = inner
                .active_by_file
                .get(file)
                .and_then(Option::as_ref)
                .expect("active file checked above");
            match removed {
                Access::Read { .. } => ids.extend(active.barriers.iter().map(|(_, id)| *id)),
                Access::Write { .. } => ids.extend(
                    active
                        .metadata
                        .iter()
                        .chain(active.barriers.iter())
                        .map(|(_, id)| *id),
                ),
                Access::Metadata { .. } => ids.extend(
                    active
                        .writes
                        .iter()
                        .chain(active.barriers.iter())
                        .map(|(_, id)| *id),
                ),
                Access::Sync { .. } | Access::Truncate { .. } => {
                    ids.extend(active.all.iter().map(|(_, id)| *id))
                }
            }
        }

        let mut ready_count = 0;
        for id in ids.iter().copied() {
            if Self::insert_ready_if_dispatchable_locked(inner, id) {
                ready_count += 1;
            }
        }
        ids.clear();
        inner.ready_refresh = ids;
        ready_count
    }

    #[cfg(any(feature = "uring", test))]
    fn drain_locked(inner: &mut Inner) -> Vec<Waiters> {
        inner.queued = 0;
        inner.ready.clear();
        inner.ready_refresh.clear();
        #[cfg(feature = "uring")]
        inner.fixed_file_updates.clear();
        #[cfg(feature = "uring")]
        {
            inner.fixed_file_update_base = 0;
            for cursor in inner.fixed_file_update_cursors.values_mut() {
                *cursor = 0;
            }
        }
        for ready in &mut inner.fast_ready {
            ready.clear();
        }
        inner.fast_active_reads.clear();
        inner.dedup.clear();
        inner.active_by_file.clear();

        inner
            .table
            .drain()
            .map(|(_, inflight)| inflight.waiters)
            .collect()
    }

    fn pop_ready_locked(inner: &mut Inner) -> Option<IoWork> {
        loop {
            let fast_key = Self::next_fast_ready_key_locked(inner);
            let slow_key = inner.ready.first().copied();
            let use_fast = match (fast_key, slow_key) {
                (Some(fast), Some(slow)) => fast <= slow,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => return None,
            };

            if use_fast {
                let (priority, seq, id) = fast_key.expect("fast key checked above");
                let Some(front) = inner.fast_ready[priority].pop_front() else {
                    continue;
                };
                debug_assert_eq!(front, id);
                let Some(inflight) = inner.table.get(&id) else {
                    continue;
                };
                if inflight.priority != priority
                    || inflight.seq != seq
                    || inflight.refcount == 0
                    || inflight.state != InflightState::Queued
                    || !inflight.fast_read
                {
                    continue;
                }

                let (queued_at, op) = {
                    let inflight = inner.table.get_mut(&id).expect("fast entry exists");
                    inflight.state = InflightState::Dispatched;
                    (inflight.queued_at, inflight.op.take())
                };
                inner.queued = inner.queued.saturating_sub(1);
                let op = op.expect("queued fast entry has an operation");
                return Some(IoWork {
                    id,
                    op,
                    queued_at,
                    operation_started: ProfileTimer::disabled(),
                });
            }

            let key = inner.ready.pop_first().expect("slow key checked above");
            let (priority, seq, id) = key;

            let Some(inflight) = inner.table.get(&id) else {
                continue;
            };
            if inflight.priority != priority
                || inflight.seq != seq
                || inflight.refcount == 0
                || inflight.state != InflightState::Queued
                || !Self::is_dispatchable_locked(inner, id, inflight)
            {
                continue;
            }

            let (queued_at, op) = {
                let inflight = inner.table.get_mut(&id).expect("dispatchable entry exists");
                inflight.state = InflightState::Dispatched;
                (inflight.queued_at, inflight.op.take())
            };
            inner.queued = inner.queued.saturating_sub(1);
            let op = op.expect("queued dispatchable entry has an operation");
            return Some(IoWork {
                id,
                op,
                queued_at,
                operation_started: ProfileTimer::disabled(),
            });
        }
    }

    fn is_dispatchable_locked(inner: &Inner, id: WorkId, candidate: &Inflight) -> bool {
        let Some(active) = inner
            .active_by_file
            .get(candidate.access.file())
            .and_then(Option::as_ref)
        else {
            return true;
        };

        match candidate.access {
            Access::Read { .. } => !Self::has_earlier_active(&active.barriers, candidate.seq, id),
            Access::Write { .. } => {
                !Self::has_earlier_active(&active.barriers, candidate.seq, id)
                    && !Self::has_earlier_active(&active.metadata, candidate.seq, id)
            }
            Access::Sync { .. } | Access::Truncate { .. } => {
                Self::fast_active_read_count(inner, candidate.access.file()) == 0
                    && !Self::has_earlier_active(&active.all, candidate.seq, id)
            }
            Access::Metadata { .. } => {
                !Self::has_earlier_active(&active.barriers, candidate.seq, id)
                    && !Self::has_earlier_active(&active.writes, candidate.seq, id)
            }
        }
    }

    fn can_dedup_locked(
        inner: &Inner,
        id: WorkId,
        inflight: &Inflight,
        requested_access: Access,
    ) -> bool {
        if inflight.refcount == 0 || inflight.access != requested_access {
            return false;
        }

        let Some(active) = inner
            .active_by_file
            .get(inflight.access.file())
            .and_then(Option::as_ref)
        else {
            return true;
        };

        !Self::has_later_or_equal_active(&active.truncates, inflight.seq, id)
    }

    fn has_earlier_active(set: &BTreeSet<(u64, WorkId)>, seq: u64, id: WorkId) -> bool {
        set.first().is_some_and(|key| *key < (seq, id))
    }

    fn has_later_or_equal_active(set: &BTreeSet<(u64, WorkId)>, seq: u64, id: WorkId) -> bool {
        set.range((seq, id)..).next().is_some()
    }

    fn promote_priority_locked(inner: &mut Inner, id: WorkId, priority: usize) {
        let Some(inflight) = inner.table.get(&id) else {
            return;
        };
        if priority >= inflight.priority {
            return;
        }
        if inflight.state != InflightState::Queued {
            let inflight = inner.table.get_mut(&id).expect("entry checked above");
            inflight.priority = priority;
            return;
        }
        if inflight.fast_read {
            let inflight = inner.table.get_mut(&id).expect("entry checked above");
            inflight.priority = priority;
            if let Some(ready) = inner.fast_ready.get_mut(priority) {
                ready.push_back(id);
            }
            return;
        }

        let old_ready_key = Self::ready_key_locked(inflight, id);
        inner.ready.remove(&old_ready_key);

        let inflight = inner.table.get_mut(&id).expect("entry checked above");
        inflight.priority = priority;
        let _ = Self::insert_ready_if_dispatchable_locked(inner, id);
    }

    fn insert_active_index_locked(inner: &mut Inner, file: FileId, seq: u64, id: WorkId) {
        let Some(inflight) = inner.table.get(&id) else {
            return;
        };
        let access = inflight.access;
        if inner.active_by_file.len() <= file {
            inner.active_by_file.resize_with(file + 1, || None);
        }
        let active = inner.active_by_file[file].get_or_insert_with(FileActive::default);
        let key = (seq, id);
        active.all.insert(key);
        match access {
            Access::Read { .. } => {}
            Access::Write { .. } => {
                active.writes.insert(key);
            }
            Access::Sync { .. } => {
                active.barriers.insert(key);
            }
            Access::Truncate { .. } => {
                active.barriers.insert(key);
                active.truncates.insert(key);
            }
            Access::Metadata { .. } => {
                active.metadata.insert(key);
            }
        }
    }

    fn remove_active_index_locked(inner: &mut Inner, access: Access, seq: u64, id: WorkId) {
        let file = access.file();
        let Some(Some(active)) = inner.active_by_file.get_mut(file) else {
            return;
        };
        let key = (seq, id);
        active.all.remove(&key);
        match access {
            Access::Read { .. } => {}
            Access::Write { .. } => {
                active.writes.remove(&key);
            }
            Access::Sync { .. } => {
                active.barriers.remove(&key);
            }
            Access::Truncate { .. } => {
                active.barriers.remove(&key);
                active.truncates.remove(&key);
            }
            Access::Metadata { .. } => {
                active.metadata.remove(&key);
            }
        }
        if active.all.is_empty() {
            inner.active_by_file[file] = None;
        }
    }

    fn has_active_file(&self, file: FileId) -> bool {
        let inner = self.inner.lock().unwrap();
        Self::file_active_locked(&inner, file)
    }

    fn wait_until_file_inactive(&self, file: FileId) {
        let mut inner = self.inner.lock().unwrap();
        if !Self::file_active_locked(&inner, file) {
            return;
        }
        self.inactive_waiters.fetch_add(1, Ordering::AcqRel);
        while Self::file_active_locked(&inner, file) {
            inner = self.available.wait(inner).unwrap();
        }
        self.inactive_waiters.fetch_sub(1, Ordering::AcqRel);
    }

    fn file_active_locked(inner: &Inner, file: FileId) -> bool {
        inner
            .active_by_file
            .get(file)
            .and_then(Option::as_ref)
            .is_some()
            || Self::fast_active_read_count(inner, file) > 0
    }

    #[cfg(test)]
    fn debug_counts(&self) -> (usize, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.table.len(), inner.queued)
    }

    #[cfg(test)]
    fn debug_active_index_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner
            .active_by_file
            .iter()
            .filter_map(Option::as_ref)
            .map(|active| active.all.len())
            .sum()
    }

    #[cfg(test)]
    fn debug_active_waiter_count(&self) -> usize {
        self.active_limiter.waiter_count()
    }

    #[cfg(test)]
    fn debug_fast_active_read_count(&self, file: FileId) -> usize {
        let inner = self.inner.lock().unwrap();
        Self::fast_active_read_count(&inner, file)
    }

    #[cfg(test)]
    fn debug_fast_ready_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.fast_ready.iter().map(VecDeque::len).sum()
    }

    #[cfg(all(test, feature = "uring"))]
    fn debug_fixed_file_update_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.fixed_file_updates.len()
    }
}

/// Future returned by [`IoPool::submit`].
///
/// Dropping the future before completion decrements the queue reference count.
/// If it was the last waiter and the request has not yet been dispatched, the
/// queued work is removed before workers see it.
#[derive(Debug)]
pub struct IoFuture {
    rx: Option<tokio::sync::oneshot::Receiver<SharedIoResult>>,
    queue: Weak<QueueCore>,
    id: WorkId,
    released: bool,
}

impl IoFuture {
    fn new(
        rx: tokio::sync::oneshot::Receiver<SharedIoResult>,
        queue: Weak<QueueCore>,
        id: WorkId,
    ) -> Self {
        Self {
            rx: Some(rx),
            queue,
            id,
            released: false,
        }
    }

    fn detached(rx: tokio::sync::oneshot::Receiver<SharedIoResult>) -> Self {
        Self {
            rx: Some(rx),
            queue: Weak::new(),
            id: WorkId(0),
            released: true,
        }
    }

    pub fn blocking_recv(mut self) -> io::Result<IoOutput> {
        self.released = true;
        let Some(rx) = self.rx.take() else {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "IO future was already consumed",
            ));
        };
        match rx.blocking_recv() {
            Ok(result) => shared_to_io_result(result),
            Err(_) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "IO result sender was dropped",
            )),
        }
    }

    pub fn blocking_recv_read(self) -> io::Result<Arc<[u8]>> {
        self.blocking_recv()?.into_read_bytes()
    }

    pub fn try_recv(&mut self) -> io::Result<Option<IoOutput>> {
        let Some(mut rx) = self.rx.take() else {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "IO future was already consumed",
            ));
        };

        match rx.try_recv() {
            Ok(result) => {
                self.released = true;
                shared_to_io_result(result).map(Some)
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                self.rx = Some(rx);
                Ok(None)
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                self.released = true;
                Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "IO result sender was dropped",
                ))
            }
        }
    }

    pub fn try_recv_read(&mut self) -> io::Result<Option<Arc<[u8]>>> {
        self.try_recv()?.map(IoOutput::into_read_bytes).transpose()
    }
}

impl Future for IoFuture {
    type Output = io::Result<IoOutput>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let poll_result = {
            let Some(rx) = self.rx.as_mut() else {
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "IO future was already consumed",
                )));
            };
            Pin::new(rx).poll(cx)
        };

        match poll_result {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(result)) => {
                self.released = true;
                self.rx = None;
                Poll::Ready(shared_to_io_result(result))
            }
            Poll::Ready(Err(_)) => {
                self.released = true;
                self.rx = None;
                Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "IO result sender was dropped",
                )))
            }
        }
    }
}

impl Drop for IoFuture {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(queue) = self.queue.upgrade() {
            queue.release(self.id);
        }
    }
}

/// The IO execution pool.
pub struct IoPool {
    queues: Vec<Arc<QueueCore>>,
    file_table: Arc<FileTable>,
    free_file_ids: Mutex<Vec<FileId>>,
    threads: Vec<thread::JoinHandle<()>>,
    profiler: IoPoolProfile,
    #[cfg(feature = "uring")]
    uses_uring: bool,
}

impl IoPool {
    pub fn new(config: IoConfig) -> io::Result<Self> {
        Self::new_with_iopool_profile(config, IoPoolProfile::from_env())
    }

    pub fn new_with_profile(config: IoConfig, profiler: ProfileRuntime) -> io::Result<Self> {
        Self::new_with_iopool_profile(config, IoPoolProfile::new(profiler))
    }

    fn new_with_iopool_profile(config: IoConfig, profiler: IoPoolProfile) -> io::Result<Self> {
        config
            .validate()
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;

        let queue_count = config.queue_shards();
        let in_flight_limit = config.in_flight_limit();
        let queue_capacity = config.queue_capacity();
        let active_limiter = Arc::new(ActiveLimiter::new(in_flight_limit));
        let queue_limiter = Arc::new(Semaphore::new(queue_capacity));
        let queue_signal = Arc::new(QueueSignal::default());
        let fast_read_path = config.base().assume_non_overlapping_reads;
        let queues = (0..queue_count)
            .map(|_| {
                Arc::new(QueueCore::new_with_limiter_and_profile(
                    config.priority_levels(),
                    queue_capacity,
                    in_flight_limit,
                    fast_read_path,
                    Arc::clone(&active_limiter),
                    Arc::clone(&queue_limiter),
                    Arc::clone(&queue_signal),
                    profiler.clone(),
                ))
            })
            .collect::<Vec<_>>();
        let file_table = Arc::new(RwLock::new(Vec::new()));
        #[cfg(feature = "uring")]
        let uses_uring = matches!(&config, IoConfig::Uring(_));

        let threads = match config {
            IoConfig::Threaded(config) => {
                threaded::start(config, &queues, Arc::clone(&file_table))?
            }
            IoConfig::Uring(config) => Self::start_uring(config, &queues, &file_table)?,
        };

        Ok(Self {
            queues,
            file_table,
            free_file_ids: Mutex::new(Vec::new()),
            threads,
            profiler,
            #[cfg(feature = "uring")]
            uses_uring,
        })
    }

    pub fn profile(&self) -> &ProfileRuntime {
        self.profiler.runtime()
    }

    pub fn profile_snapshot(&self) -> ProfileSnapshot {
        self.profiler.snapshot()
    }

    pub fn profile_snapshot_and_reset(&self) -> ProfileSnapshot {
        self.profiler.snapshot_and_reset()
    }

    pub fn reset_profile(&self) {
        self.profiler.reset_metrics();
    }

    /// Register a file for positioned IO and return its opaque handle.
    ///
    /// This preserves the historical behavior: try read-write first, then fall
    /// back to read-only. Use [`Self::register_readwrite_file`] when writes must
    /// be supported.
    pub fn register_file(&self, path: &Path) -> io::Result<FileId> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .or_else(|_| File::open(path))?;
        self.register_existing_file(file)
    }

    pub fn register_readwrite_file(&self, path: &Path) -> io::Result<FileId> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        self.register_existing_file(file)
    }

    pub fn register_readonly_file(&self, path: &Path) -> io::Result<FileId> {
        self.register_existing_file(File::open(path)?)
    }

    pub fn register_file_with_options(
        &self,
        path: &Path,
        options: &OpenOptions,
    ) -> io::Result<FileId> {
        self.register_existing_file(options.open(path)?)
    }

    pub fn register_existing_file(&self, file: File) -> io::Result<FileId> {
        #[cfg(feature = "uring")]
        let fd = file.as_raw_fd();
        let file = Arc::new(file);
        let mut table = self.file_table.write().unwrap();
        let mut free_file_ids = self.free_file_ids.lock().unwrap();
        let id = loop {
            let Some(candidate) = free_file_ids.pop() else {
                let id = table.len();
                table.push(FileSlot::Open(Arc::clone(&file)));
                break id;
            };
            if table
                .get(candidate)
                .is_some_and(|slot| matches!(slot, FileSlot::Closed))
            {
                table[candidate] = FileSlot::Open(Arc::clone(&file));
                break candidate;
            }
        };
        drop(free_file_ids);
        drop(table);
        #[cfg(feature = "uring")]
        self.update_fixed_file(id, fd);
        Ok(id)
    }

    pub fn unregister_file(&self, file: FileId) -> io::Result<()> {
        self.mark_file_closing(file, false)?;
        for queue in &self.queues {
            queue.wait_until_file_inactive(file);
        }
        self.finish_unregister_file(file)
    }

    /// Try to unregister a file without waiting for active IO.
    ///
    /// If this returns `WouldBlock`, the file is restored to the open state when
    /// this call was the one that marked it closing.
    pub fn try_unregister_file(&self, file: FileId) -> io::Result<()> {
        let mark = self.mark_file_closing(file, true)?;
        if mark == ClosingMark::AlreadyClosing {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("file_id {file} is already closing"),
            ));
        }
        if self.queues.iter().any(|queue| queue.has_active_file(file)) {
            let _ = self.reopen_file_if_reopenable(file)?;
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("file_id {file} still has active IO"),
            ));
        }
        if self.finish_unregister_file_if_reopenable(file)? {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("file_id {file} is already closing"),
            ))
        }
    }

    fn mark_file_closing(&self, file: FileId, reopenable: bool) -> io::Result<ClosingMark> {
        let mut table = self.file_table.write().unwrap();
        let Some(slot) = table.get_mut(file) else {
            return Err(invalid_file_id(file));
        };
        match slot {
            FileSlot::Open(handle) => {
                let handle = Arc::clone(handle);
                *slot = FileSlot::Closing { handle, reopenable };
                Ok(ClosingMark::NewlyMarked)
            }
            FileSlot::Closing {
                reopenable: current,
                ..
            } => {
                if !reopenable {
                    *current = false;
                }
                Ok(ClosingMark::AlreadyClosing)
            }
            FileSlot::Closed => Err(invalid_file_id(file)),
        }
    }

    fn reopen_file_if_reopenable(&self, file: FileId) -> io::Result<bool> {
        let mut table = self.file_table.write().unwrap();
        let Some(slot) = table.get_mut(file) else {
            return Err(invalid_file_id(file));
        };
        match slot {
            FileSlot::Closing {
                handle,
                reopenable: true,
            } => {
                let handle = Arc::clone(handle);
                *slot = FileSlot::Open(handle);
                Ok(true)
            }
            FileSlot::Closing {
                reopenable: false, ..
            } => Ok(false),
            FileSlot::Open(_) => Ok(true),
            FileSlot::Closed => Err(invalid_file_id(file)),
        }
    }

    fn finish_unregister_file(&self, file: FileId) -> io::Result<()> {
        let mut table = self.file_table.write().unwrap();
        let Some(slot) = table.get_mut(file) else {
            return Err(invalid_file_id(file));
        };
        match slot {
            FileSlot::Open(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("file_id {file} is not closing"),
                ));
            }
            FileSlot::Closing { .. } => {
                *slot = FileSlot::Closed;
            }
            FileSlot::Closed => return Err(invalid_file_id(file)),
        }
        drop(table);
        #[cfg(feature = "uring")]
        self.update_fixed_file(file, -1);
        self.free_file_ids.lock().unwrap().push(file);
        Ok(())
    }

    fn finish_unregister_file_if_reopenable(&self, file: FileId) -> io::Result<bool> {
        let mut table = self.file_table.write().unwrap();
        let Some(slot) = table.get_mut(file) else {
            return Err(invalid_file_id(file));
        };
        match slot {
            FileSlot::Open(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("file_id {file} is not closing"),
                ));
            }
            FileSlot::Closing {
                reopenable: true, ..
            } => {
                *slot = FileSlot::Closed;
            }
            FileSlot::Closing {
                reopenable: false, ..
            } => return Ok(false),
            FileSlot::Closed => return Err(invalid_file_id(file)),
        }
        drop(table);
        #[cfg(feature = "uring")]
        self.update_fixed_file(file, -1);
        self.free_file_ids.lock().unwrap().push(file);
        Ok(true)
    }

    pub fn try_submit(&self, cmd: IoCommand) -> io::Result<IoFuture> {
        let file = cmd.file();
        let queue = Arc::clone(self.select_queue(&cmd));
        let future = {
            let table = self.file_table.read().unwrap();
            let handle = registered_file_locked(&table, file)?;
            queue.try_submit_with_handle(cmd, Some(handle), true)
        };
        match future {
            Ok(future) => Ok(future),
            Err(QueueSubmitError::Full(_)) => Err(queue.queue_full_error()),
            Err(QueueSubmitError::Io(err)) => Err(err),
        }
    }

    pub fn submit(&self, cmd: IoCommand) -> io::Result<IoFuture> {
        let file = cmd.file();
        let queue = Arc::clone(self.select_queue(&cmd));
        let future = {
            let table = self.file_table.read().unwrap();
            let handle = registered_file_locked(&table, file)?;
            queue.try_submit_with_handle(cmd, Some(handle), true)
        };
        match future {
            Ok(future) => Ok(future),
            Err(QueueSubmitError::Full(cmd)) => {
                let permit = queue.blocking_acquire_queue_permit()?;
                {
                    let table = self.file_table.read().unwrap();
                    let handle = registered_file_locked(&table, file)?;
                    queue.submit_with_handle_permit(cmd, Some(handle), permit, false)
                }
            }
            Err(QueueSubmitError::Io(err)) => Err(err),
        }
    }

    pub async fn submit_async(&self, cmd: IoCommand) -> io::Result<IoFuture> {
        let file = cmd.file();
        let queue = Arc::clone(self.select_queue(&cmd));
        let future = {
            let table = self.file_table.read().unwrap();
            let handle = registered_file_locked(&table, file)?;
            queue.try_submit_with_handle(cmd, Some(handle), true)
        };
        match future {
            Ok(future) => Ok(future),
            Err(QueueSubmitError::Full(cmd)) => {
                let permit = queue.acquire_queue_permit().await?;
                {
                    let table = self.file_table.read().unwrap();
                    let handle = registered_file_locked(&table, file)?;
                    queue.submit_with_handle_permit(cmd, Some(handle), permit, false)
                }
            }
            Err(QueueSubmitError::Io(err)) => Err(err),
        }
    }

    fn select_queue(&self, cmd: &IoCommand) -> &Arc<QueueCore> {
        self.queue_for_file(cmd.file())
    }

    #[cfg(feature = "uring")]
    fn update_fixed_file(&self, file: FileId, fd: RawFd) {
        if self.uses_uring {
            self.queue_for_file(file)
                .push_fixed_file_update(FixedFileUpdate { file, fd });
        }
    }

    fn queue_for_file(&self, file: FileId) -> &Arc<QueueCore> {
        &self.queues[file % self.queues.len()]
    }

    #[cfg(feature = "uring")]
    fn start_uring(
        config: UringConfig,
        queues: &[Arc<QueueCore>],
        file_table: &Arc<FileTable>,
    ) -> io::Result<Vec<thread::JoinHandle<()>>> {
        uring::start(config, queues, Arc::clone(file_table))
    }

    #[cfg(not(feature = "uring"))]
    fn start_uring(
        _config: UringConfig,
        _queues: &[Arc<QueueCore>],
        _file_table: &Arc<FileTable>,
    ) -> io::Result<Vec<thread::JoinHandle<()>>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "io_uring backend not compiled in (enable the 'uring' feature)",
        ))
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        for queue in &self.queues {
            queue.shutdown();
        }
        while let Some(handle) = self.threads.pop() {
            if handle.join().is_err() {
                eprintln!("[iopool] backend thread panicked during shutdown");
            }
        }
    }
}

pub(super) fn execute_work(file_table: &Arc<FileTable>, op: IoOperation) -> io::Result<IoOutput> {
    match op {
        IoOperation::Read {
            file,
            handle,
            offset,
            len,
        } => {
            let file = operation_file(file_table, file, handle)?;
            let mut buf = uninit_read_buffer(len);
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), len) };
            file.read_exact_at(bytes, offset)?;
            let buf = unsafe { assume_init_read_buffer(buf) };
            Ok(IoOutput::Read(Arc::from(buf)))
        }
        IoOperation::Write {
            file,
            handle,
            offset,
            buf,
        } => {
            let file = operation_file(file_table, file, handle)?;
            let bytes = buf.len();
            write_all_at(&file, &buf, offset)?;
            Ok(IoOutput::Write { bytes })
        }
        IoOperation::Fsync { file, handle } => {
            let file = operation_file(file_table, file, handle)?;
            file.sync_all()?;
            Ok(IoOutput::SyncAll)
        }
        IoOperation::SyncData { file, handle } => {
            let file = operation_file(file_table, file, handle)?;
            file.sync_data()?;
            Ok(IoOutput::SyncData)
        }
        IoOperation::Truncate { file, handle, len } => {
            let file = operation_file(file_table, file, handle)?;
            file.set_len(len)?;
            Ok(IoOutput::Truncate)
        }
        IoOperation::Metadata { file, handle } => {
            let file = operation_file(file_table, file, handle)?;
            Ok(IoOutput::Metadata(file.metadata()?.into()))
        }
    }
}

#[inline]
fn io_result_profile_bytes(result: &io::Result<IoOutput>) -> (usize, usize) {
    match result {
        Ok(IoOutput::Read(bytes)) => (bytes.len(), 0),
        Ok(IoOutput::Write { bytes }) => (0, *bytes),
        Ok(IoOutput::SyncAll | IoOutput::SyncData | IoOutput::Truncate | IoOutput::Metadata(_))
        | Err(_) => (0, 0),
    }
}

fn operation_file(
    file_table: &Arc<FileTable>,
    file_id: FileId,
    handle: Option<Arc<File>>,
) -> io::Result<Arc<File>> {
    match handle {
        Some(file) => Ok(file),
        None => registered_file(file_table, file_id),
    }
}

pub(super) fn registered_file(
    file_table: &Arc<FileTable>,
    file_id: FileId,
) -> io::Result<Arc<File>> {
    let table = file_table.read().unwrap();
    registered_file_locked(&table, file_id)
}

fn registered_file_locked(table: &[FileSlot], file_id: FileId) -> io::Result<Arc<File>> {
    table
        .get(file_id)
        .and_then(FileSlot::open_file)
        .ok_or_else(|| invalid_file_id(file_id))
}

fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match file.write_at(buf, offset)? {
            0 => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                ));
            }
            n => {
                buf = &buf[n..];
                offset = offset.checked_add(n as u64).ok_or_else(offset_overflow)?;
            }
        }
    }
    Ok(())
}

pub(super) fn uninit_read_buffer(len: usize) -> Box<[MaybeUninit<u8>]> {
    let mut buf = Vec::with_capacity(len);
    // `MaybeUninit<u8>` does not require initialization before becoming live.
    unsafe {
        buf.set_len(len);
    }
    buf.into_boxed_slice()
}

pub(super) unsafe fn assume_init_read_buffer(buf: Box<[MaybeUninit<u8>]>) -> Box<[u8]> {
    let raw = Box::into_raw(buf) as *mut [u8];
    unsafe { Box::from_raw(raw) }
}

fn invalid_file_id(file_id: FileId) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("invalid file_id {file_id}"),
    )
}

fn offset_overflow() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "IO offset overflowed")
}

fn refcount_overflow() -> io::Error {
    io::Error::new(
        io::ErrorKind::OutOfMemory,
        "IO waiter reference count overflowed",
    )
}

#[cfg(feature = "uring")]
fn wake_eventfd(fd: RawFd) {
    let value: u64 = 1;
    loop {
        let written = unsafe {
            libc::write(
                fd,
                (&value as *const u64).cast::<libc::c_void>(),
                std::mem::size_of::<u64>(),
            )
        };
        if written == std::mem::size_of::<u64>() as isize {
            return;
        }

        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EINTR) => continue,
            Some(libc::EAGAIN) => return,
            _ => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "profile")]
    use crate::iopool::profile::iopool_profile_registry;
    use crate::iopool::profile::{test_metrics, IoPoolProfile};
    #[cfg(feature = "profile")]
    use crate::profile::{ProfileMetricId, ProfileRegistry, ProfileRuntime};
    use std::io::{self, ErrorKind, Write};
    #[cfg(feature = "profile")]
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    #[cfg(feature = "profile")]
    static PROFILE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    #[cfg(feature = "profile")]
    const PROFILE_ENV_KEYS: &[&str] = &[
        "SCDATA_PROFILE",
        "SCDATA_PROFILE_LABEL",
        "SCDATA_PROFILE_COMPONENTS",
        "SCDATA_PROFILE_COMPONENT_ENABLE",
        "SCDATA_PROFILE_COMPONENT_DISABLE",
        "SCDATA_PROFILE_SCOPES",
        "SCDATA_PROFILE_SCOPE_ENABLE",
        "SCDATA_PROFILE_SCOPE_DISABLE",
    ];

    fn temp_file(name: &str, content: &[u8]) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("scdata_iopool_{}_{}", name, std::process::id()));
        let mut file = File::create(&path).expect("create temp file");
        file.write_all(content).expect("write temp file");
        file.flush().expect("flush temp file");
        path
    }

    fn make_pool() -> IoPool {
        IoPool::new(IoConfig::Threaded(ThreadedConfig {
            base: BaseIoConfig {
                max_in_flight: 16,
                queue_capacity: 64,
                priority_levels: 3,
                queue_shards: 1,
                assume_non_overlapping_reads: false,
            },
            num_workers: 2,
            cpus: None,
        }))
        .expect("create pool")
    }

    fn make_unstarted_pool(queue_count: usize, max_active: usize) -> IoPool {
        make_unstarted_pool_with_capacity(queue_count, max_active, max_active)
    }

    fn make_unstarted_pool_with_capacity(
        queue_count: usize,
        queue_capacity: usize,
        max_active: usize,
    ) -> IoPool {
        let active_limiter = Arc::new(ActiveLimiter::new(max_active));
        let queue_limiter = Arc::new(Semaphore::new(queue_capacity.max(1)));
        let queue_signal = Arc::new(QueueSignal::default());
        let queues = (0..queue_count)
            .map(|_| {
                Arc::new(QueueCore::new_with_limiter(
                    3,
                    queue_capacity,
                    max_active,
                    false,
                    Arc::clone(&active_limiter),
                    Arc::clone(&queue_limiter),
                    Arc::clone(&queue_signal),
                ))
            })
            .collect::<Vec<_>>();
        IoPool {
            queues,
            file_table: Arc::new(RwLock::new(Vec::new())),
            free_file_ids: Mutex::new(Vec::new()),
            threads: Vec::new(),
            profiler: IoPoolProfile::disabled(),
            #[cfg(feature = "uring")]
            uses_uring: false,
        }
    }

    #[test]
    fn blocking_submit_waiting_for_queue_capacity_wakes_on_shutdown() {
        let path = temp_file("blocking_submit_shutdown", b"abcdef");
        let pool = Arc::new(make_unstarted_pool_with_capacity(1, 1, 1));
        let file = pool.register_file(&path).expect("register");
        let held = pool
            .submit(IoCommand::read(file, 0, 1, 0))
            .expect("held submit");

        let (started_tx, started_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let worker_pool = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            started_tx.send(()).expect("signal started");
            let result = worker_pool.submit(IoCommand::read(file, 1, 1, 0));
            result_tx
                .send(result.map(|_| ()))
                .expect("send submit result");
        });

        started_rx.recv().expect("submit thread started");
        pool.queues[0].shutdown();
        let result = match result_rx.recv_timeout(Duration::from_millis(500)) {
            Ok(result) => result,
            Err(err) => {
                drop(held);
                let _ = handle.join();
                panic!("blocking submit did not wake after queue shutdown: {err}");
            }
        };
        assert_eq!(
            result
                .expect_err("shutdown should reject blocked submit")
                .kind(),
            ErrorKind::BrokenPipe
        );
        drop(held);
        handle.join().expect("submit thread joined");
    }

    #[cfg(feature = "profile")]
    fn enabled_profile(label: &'static str) -> ProfileRuntime {
        ProfileRuntime::enabled_lazy(label, iopool_profile_registry)
    }

    #[cfg(feature = "profile")]
    fn enabled_queue_profile(label: &'static str) -> (ProfileRuntime, IoPoolProfile) {
        let runtime = enabled_profile(label);
        let profile = IoPoolProfile::new(runtime.clone());
        (runtime, profile)
    }

    #[cfg(feature = "profile")]
    fn metric(snapshot: &ProfileSnapshot, id: ProfileMetricId) -> u64 {
        snapshot.metric_value(id).unwrap_or(0)
    }

    #[cfg(feature = "profile")]
    fn with_profile_env_enabled<T>(f: impl FnOnce() -> T) -> T {
        let _guard = PROFILE_ENV_LOCK.lock().unwrap();
        let saved = PROFILE_ENV_KEYS
            .iter()
            .map(|key| (*key, std::env::var_os(key)))
            .collect::<Vec<_>>();

        std::env::set_var("SCDATA_PROFILE", "1");
        for key in PROFILE_ENV_KEYS
            .iter()
            .copied()
            .filter(|key| *key != "SCDATA_PROFILE")
        {
            std::env::remove_var(key);
        }

        let result = catch_unwind(AssertUnwindSafe(f));
        for (key, value) in saved {
            if let Some(value) = value {
                std::env::set_var(key, value);
            } else {
                std::env::remove_var(key);
            }
        }

        match result {
            Ok(value) => value,
            Err(payload) => resume_unwind(payload),
        }
    }

    #[cfg(feature = "uring")]
    fn make_unstarted_uring_pool(queue_count: usize, max_active: usize) -> IoPool {
        let active_limiter = Arc::new(ActiveLimiter::new(max_active));
        let queue_limiter = Arc::new(Semaphore::new(max_active.max(1)));
        let queue_signal = Arc::new(QueueSignal::default());
        let queues = (0..queue_count)
            .map(|_| {
                Arc::new(QueueCore::new_with_limiter(
                    3,
                    max_active,
                    max_active,
                    false,
                    Arc::clone(&active_limiter),
                    Arc::clone(&queue_limiter),
                    Arc::clone(&queue_signal),
                ))
            })
            .collect::<Vec<_>>();
        IoPool {
            queues,
            file_table: Arc::new(RwLock::new(Vec::new())),
            free_file_ids: Mutex::new(Vec::new()),
            threads: Vec::new(),
            profiler: IoPoolProfile::disabled(),
            uses_uring: true,
        }
    }

    #[cfg(feature = "uring")]
    fn test_eventfd() -> RawFd {
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        assert!(fd >= 0, "create eventfd: {}", io::Error::last_os_error());
        fd
    }

    #[cfg(feature = "uring")]
    fn close_test_fd(fd: RawFd) {
        unsafe {
            libc::close(fd);
        }
    }

    #[test]
    fn global_in_flight_limit_spans_shards() {
        let path_a = temp_file("global_limit_a", b"abcdefgh");
        let path_b = temp_file("global_limit_b", b"ABCDEFGH");
        let pool = make_unstarted_pool_with_capacity(2, 4, 2);
        let file_a = pool.register_readonly_file(&path_a).expect("register a");
        let file_b = pool.register_readonly_file(&path_b).expect("register b");

        let first = pool
            .submit(IoCommand::read(file_a, 0, 1, 0))
            .expect("first read");
        let second = pool
            .submit(IoCommand::read(file_b, 0, 1, 0))
            .expect("second read");
        let third = pool
            .submit(IoCommand::read(file_a, 1, 1, 0))
            .expect("third read should queue behind active limit");

        let queue_a = pool.queue_for_file(file_a);
        let queue_b = pool.queue_for_file(file_b);
        let first_work = queue_a.try_pop_for_test().expect("dispatch first read");
        let second_work = queue_b.try_pop_for_test().expect("dispatch second read");
        assert!(
            queue_a.try_pop_for_test().is_none(),
            "global active limit should stall third dispatch"
        );

        queue_a.complete(
            first_work.id,
            first_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"a".to_vec()))),
        );
        let third_work = queue_a
            .try_pop_for_test()
            .expect("released active permit should dispatch third read");
        queue_b.complete(
            second_work.id,
            second_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"A".to_vec()))),
        );
        queue_a.complete(
            third_work.id,
            third_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"b".to_vec()))),
        );
        assert_eq!(&*first.blocking_recv_read().expect("first"), b"a");
        assert_eq!(&*second.blocking_recv_read().expect("second"), b"A");
        assert_eq!(&*third.blocking_recv_read().expect("third"), b"b");
    }

    #[test]
    fn active_release_wakes_other_shard_waiting_on_global_limit() {
        let path_a = temp_file("active_wake_a", b"abcdefgh");
        let path_b = temp_file("active_wake_b", b"ABCDEFGH");
        let pool = Arc::new(make_unstarted_pool_with_capacity(2, 4, 1));
        let file_a = pool.register_readonly_file(&path_a).expect("register a");
        let file_b = pool.register_readonly_file(&path_b).expect("register b");
        let queue_a = Arc::clone(pool.queue_for_file(file_a));
        let queue_b = Arc::clone(pool.queue_for_file(file_b));
        assert!(
            !Arc::ptr_eq(&queue_a, &queue_b),
            "test requires distinct queue shards"
        );

        let first = pool
            .submit(IoCommand::read(file_a, 0, 1, 0))
            .expect("first read");
        let first_work = queue_a
            .try_pop_for_test()
            .expect("first shard should occupy active slot");
        let second = pool
            .submit(IoCommand::read(file_b, 0, 1, 0))
            .expect("second read");

        let (work_tx, work_rx) = mpsc::channel();
        let waiting_queue = Arc::clone(&queue_b);
        let handle = thread::spawn(move || {
            work_tx.send(waiting_queue.pop()).expect("send popped work");
        });

        for _ in 0..500 {
            if queue_b.debug_active_waiter_count() > 0 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(
            queue_b.debug_active_waiter_count(),
            1,
            "second shard should be waiting for global active capacity"
        );

        queue_a.complete(
            first_work.id,
            first_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"a".to_vec()))),
        );
        let second_work = match work_rx.recv_timeout(Duration::from_millis(500)) {
            Ok(Some(work)) => work,
            Ok(None) => panic!("second shard stopped before dispatching work"),
            Err(err) => {
                queue_b.shutdown();
                let _ = handle.join();
                panic!("active release did not wake second shard: {err}");
            }
        };
        queue_b.complete(
            second_work.id,
            second_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"A".to_vec()))),
        );

        assert_eq!(&*first.blocking_recv_read().expect("first"), b"a");
        assert_eq!(&*second.blocking_recv_read().expect("second"), b"A");
        handle.join().expect("worker joined");
    }

    #[test]
    fn queue_shutdown_wakes_shard_waiting_on_active_limit() {
        let path_a = temp_file("active_shutdown_a", b"abcdefgh");
        let path_b = temp_file("active_shutdown_b", b"ABCDEFGH");
        let pool = Arc::new(make_unstarted_pool_with_capacity(2, 4, 1));
        let file_a = pool.register_readonly_file(&path_a).expect("register a");
        let file_b = pool.register_readonly_file(&path_b).expect("register b");
        let queue_a = Arc::clone(pool.queue_for_file(file_a));
        let queue_b = Arc::clone(pool.queue_for_file(file_b));
        assert!(
            !Arc::ptr_eq(&queue_a, &queue_b),
            "test requires distinct queue shards"
        );

        let first = pool
            .submit(IoCommand::read(file_a, 0, 1, 0))
            .expect("first read");
        let first_work = queue_a
            .try_pop_for_test()
            .expect("first shard should occupy active slot");
        let second = pool
            .submit(IoCommand::read(file_b, 0, 1, 0))
            .expect("second read");

        let (done_tx, done_rx) = mpsc::channel();
        let waiting_queue = Arc::clone(&queue_b);
        let handle = thread::spawn(move || {
            done_tx
                .send(waiting_queue.pop().is_none())
                .expect("send shutdown result");
        });

        for _ in 0..500 {
            if queue_b.debug_active_waiter_count() > 0 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(
            queue_b.debug_active_waiter_count(),
            1,
            "second shard should be waiting for global active capacity"
        );

        queue_b.shutdown();
        assert!(
            done_rx
                .recv_timeout(Duration::from_millis(500))
                .expect("active waiter should wake on shutdown"),
            "queue pop should stop after shutdown"
        );
        drop(second);
        queue_a.complete(
            first_work.id,
            first_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"a".to_vec()))),
        );
        assert_eq!(&*first.blocking_recv_read().expect("first"), b"a");
        handle.join().expect("worker joined");
    }

    #[test]
    fn active_wait_observes_queue_shutdown_before_waiter_registration() {
        let path_a = temp_file("active_preregister_shutdown_a", b"abcdefgh");
        let path_b = temp_file("active_preregister_shutdown_b", b"ABCDEFGH");
        let pool = make_unstarted_pool_with_capacity(2, 4, 1);
        let file_a = pool.register_readonly_file(&path_a).expect("register a");
        let file_b = pool.register_readonly_file(&path_b).expect("register b");
        let queue_a = pool.queue_for_file(file_a);
        let queue_b = pool.queue_for_file(file_b);
        assert!(
            !Arc::ptr_eq(queue_a, queue_b),
            "test requires distinct queue shards"
        );

        let first = pool
            .submit(IoCommand::read(file_a, 0, 1, 0))
            .expect("first read");
        let first_work = queue_a
            .try_pop_for_test()
            .expect("first shard should occupy active slot");
        let second = pool
            .submit(IoCommand::read(file_b, 0, 1, 0))
            .expect("second read");

        queue_b.shutdown();
        assert!(
            queue_b
                .active_limiter
                .wait_for_capacity_until(|| queue_b.is_shutdown()),
            "active wait should return immediately when queue is already shut down"
        );
        drop(second);
        queue_a.complete(
            first_work.id,
            first_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"a".to_vec()))),
        );
        assert_eq!(&*first.blocking_recv_read().expect("first"), b"a");
    }

    #[test]
    fn same_file_can_use_full_global_in_flight_limit() {
        let path = temp_file("same_file_full_limit", b"abcdefghijklmnopqrstuvwxyz");
        let pool = make_unstarted_pool_with_capacity(2, 8, 4);
        let file = pool.register_readonly_file(&path).expect("register");

        let reads = (0..4)
            .map(|offset| {
                pool.submit(IoCommand::read(file, offset, 1, 0))
                    .expect("same-file read within global limit")
            })
            .collect::<Vec<_>>();
        let fifth = pool
            .submit(IoCommand::read(file, 4, 1, 0))
            .expect("fifth read should queue behind active limit");

        let queue = pool.queue_for_file(file);
        let mut works = (0..4)
            .map(|_| {
                queue
                    .try_pop_for_test()
                    .expect("dispatch within active limit")
            })
            .collect::<Vec<_>>();
        assert!(
            queue.try_pop_for_test().is_none(),
            "fifth same-file read should wait for active capacity"
        );
        let first_work = works.pop().expect("held work");
        queue.complete(
            first_work.id,
            first_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"a".to_vec()))),
        );
        let fifth_work = queue
            .try_pop_for_test()
            .expect("released active permit should dispatch fifth read");
        for work in works {
            queue.complete(
                work.id,
                work.operation_started,
                Ok(IoOutput::Read(Arc::from(b"x".to_vec()))),
            );
        }
        queue.complete(
            fifth_work.id,
            fifth_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"e".to_vec()))),
        );
        drop(reads);
        assert_eq!(&*fifth.blocking_recv_read().expect("fifth"), b"e");
    }

    #[test]
    fn zero_length_read_and_write_bypass_in_flight_limit() {
        let path = temp_file("zero_len_bypass", b"abcdef");
        let pool = make_unstarted_pool_with_capacity(1, 2, 1);
        let file = pool.register_readonly_file(&path).expect("register");
        let held = pool
            .submit(IoCommand::read(file, 0, 1, 0))
            .expect("occupy in-flight permit");
        let held_work = pool
            .queue_for_file(file)
            .try_pop_for_test()
            .expect("dispatch held read");

        let empty = pool
            .submit(IoCommand::read(file, 3, 0, 0))
            .expect("zero-length read should complete immediately")
            .blocking_recv_read()
            .expect("zero read");
        assert!(empty.is_empty());

        let written = pool
            .submit(IoCommand::write(file, 3, Vec::new(), 0))
            .expect("zero-length write should complete immediately")
            .blocking_recv()
            .expect("zero write");
        assert_eq!(written.bytes_written(), Some(0));

        let next = pool
            .submit(IoCommand::read(file, 1, 1, 0))
            .expect("normal read should queue while active permit is held");
        assert!(
            pool.queue_for_file(file).try_pop_for_test().is_none(),
            "detached zero-length futures must not release active permits"
        );

        pool.queue_for_file(file).complete(
            held_work.id,
            held_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"a".to_vec()))),
        );
        let next_work = pool
            .queue_for_file(file)
            .try_pop_for_test()
            .expect("next read should dispatch after active release");
        pool.queue_for_file(file).complete(
            next_work.id,
            next_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"b".to_vec()))),
        );
        assert_eq!(&*held.blocking_recv_read().expect("held"), b"a");
        assert_eq!(&*next.blocking_recv_read().expect("next"), b"b");
    }

    #[test]
    fn zero_length_operations_respect_prior_barrier() {
        let queue = Arc::new(QueueCore::new(2, 8));
        let sync = queue.submit(IoCommand::sync_all(0, 0)).expect("sync");
        let sync_work = queue.pop().expect("dispatch sync");

        let mut read = queue
            .submit(IoCommand::read(0, 42, 0, 0))
            .expect("zero read behind barrier");
        let mut write = queue
            .submit(IoCommand::write(0, 42, Vec::new(), 0))
            .expect("zero write behind barrier");

        assert!(read.try_recv().expect("read pending").is_none());
        assert!(write.try_recv().expect("write pending").is_none());

        queue.complete(
            sync_work.id,
            sync_work.operation_started,
            Ok(IoOutput::SyncAll),
        );
        assert!(matches!(
            sync.blocking_recv().expect("sync"),
            IoOutput::SyncAll
        ));

        let read_work = queue.pop().expect("zero read dispatches after barrier");
        match read_work.op {
            IoOperation::Read { len, .. } => assert_eq!(len, 0),
            _ => panic!("expected zero read"),
        }
        queue.complete(
            read_work.id,
            read_work.operation_started,
            Ok(IoOutput::Read(Arc::from(Vec::new()))),
        );
        assert!(read.blocking_recv_read().expect("zero read").is_empty());

        let write_work = queue.pop().expect("zero write dispatches after barrier");
        match write_work.op {
            IoOperation::Write { buf, .. } => assert!(buf.is_empty()),
            _ => panic!("expected zero write"),
        }
        queue.complete(
            write_work.id,
            write_work.operation_started,
            Ok(IoOutput::Write { bytes: 0 }),
        );
        assert_eq!(
            write.blocking_recv().expect("zero write").bytes_written(),
            Some(0)
        );
    }

    #[test]
    fn queue_shards_respect_consumers_and_max_in_flight() {
        let config = IoConfig::Threaded(ThreadedConfig {
            base: BaseIoConfig {
                max_in_flight: 1,
                queue_capacity: 4,
                priority_levels: 2,
                queue_shards: 4,
                assume_non_overlapping_reads: false,
            },
            num_workers: 4,
            cpus: None,
        });

        assert_eq!(config.queue_shards(), 1);
    }

    #[test]
    fn same_file_commands_route_to_same_shard() {
        let pool = make_unstarted_pool(2, 4);
        let read = Arc::as_ptr(pool.select_queue(&IoCommand::read(3, 0, 1, 0)));
        let sync = Arc::as_ptr(pool.select_queue(&IoCommand::sync_all(3, 0)));
        let metadata = Arc::as_ptr(pool.select_queue(&IoCommand::metadata(3, 0)));
        let other_file = Arc::as_ptr(pool.select_queue(&IoCommand::read(4, 0, 1, 0)));

        assert_eq!(read, sync);
        assert_eq!(read, metadata);
        assert_ne!(read, other_file);
    }

    #[test]
    fn excessive_shards_do_not_leave_work_without_worker() {
        let path = temp_file("excessive_shards", b"abcdefghijklmnopqrstuvwxyz");
        let pool = IoPool::new(IoConfig::Threaded(ThreadedConfig {
            base: BaseIoConfig {
                max_in_flight: 1,
                queue_capacity: 4,
                priority_levels: 2,
                queue_shards: 4,
                assume_non_overlapping_reads: false,
            },
            num_workers: 4,
            cpus: None,
        }))
        .expect("create constrained pool");

        assert_eq!(pool.queues.len(), 1);
        let file = pool.register_readonly_file(&path).expect("register");
        let bytes = pool
            .submit(IoCommand::read(file, 3, 4, 0))
            .expect("submit")
            .blocking_recv_read()
            .expect("read");
        assert_eq!(&*bytes, b"defg");
    }

    #[test]
    fn try_unregister_file_restores_open_file_on_would_block() {
        let path = temp_file("try_unregister_rollback", b"hello");
        let pool = make_unstarted_pool(1, 4);
        let file = pool.register_readonly_file(&path).expect("register");
        let future = pool
            .submit(IoCommand::read(file, 0, 5, 0))
            .expect("submit queued read");

        let err = pool
            .try_unregister_file(file)
            .expect_err("active file should not unregister");
        assert_eq!(err.kind(), ErrorKind::WouldBlock);

        drop(future);
        let second = pool
            .submit(IoCommand::read(file, 0, 5, 0))
            .expect("file should be open after failed try_unregister");
        drop(second);
        pool.unregister_file(file)
            .expect("unregister after releases");
    }

    #[test]
    fn blocking_unregister_claim_prevents_try_unregister_reopen() {
        let path = temp_file("try_unregister_claimed", b"hello");
        let pool = make_unstarted_pool(1, 4);
        let file = pool.register_readonly_file(&path).expect("register");
        let future = pool
            .submit(IoCommand::read(file, 0, 5, 0))
            .expect("submit queued read");

        assert_eq!(
            pool.mark_file_closing(file, true)
                .expect("try mark closing"),
            ClosingMark::NewlyMarked
        );
        assert_eq!(
            pool.mark_file_closing(file, false)
                .expect("blocking unregister claims closing file"),
            ClosingMark::AlreadyClosing
        );
        assert!(!pool
            .reopen_file_if_reopenable(file)
            .expect("try unregister should not reopen claimed closing file"));

        drop(future);
        pool.finish_unregister_file(file)
            .expect("blocking unregister can finish");
        let err = pool
            .submit(IoCommand::read(file, 0, 1, 0))
            .expect_err("file should remain closed");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn try_unregister_does_not_finish_file_already_closing() {
        let path = temp_file("try_unregister_already_closing", b"hello");
        let pool = make_unstarted_pool(1, 4);
        let file = pool.register_readonly_file(&path).expect("register");

        pool.mark_file_closing(file, false)
            .expect("blocking unregister marks closing");
        let err = pool
            .try_unregister_file(file)
            .expect_err("try unregister should not claim an already closing file");
        assert_eq!(err.kind(), ErrorKind::WouldBlock);

        pool.finish_unregister_file(file)
            .expect("original unregister can finish");
        let err = pool
            .submit(IoCommand::read(file, 0, 1, 0))
            .expect_err("file should be closed");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn try_unregister_finish_loses_to_blocking_claim() {
        let path = temp_file("try_unregister_finish_claimed", b"hello");
        let pool = make_unstarted_pool(1, 4);
        let file = pool.register_readonly_file(&path).expect("register");

        assert_eq!(
            pool.mark_file_closing(file, true)
                .expect("try unregister marks closing"),
            ClosingMark::NewlyMarked
        );
        assert_eq!(
            pool.mark_file_closing(file, false)
                .expect("blocking unregister claims closing"),
            ClosingMark::AlreadyClosing
        );
        assert!(!pool
            .finish_unregister_file_if_reopenable(file)
            .expect("try unregister should not finish claimed file"));

        pool.finish_unregister_file(file)
            .expect("blocking unregister can finish");
        let err = pool
            .submit(IoCommand::read(file, 0, 1, 0))
            .expect_err("file should be closed");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn unregister_reuses_closed_file_id() {
        let path = temp_file("reuse_file_id", b"hello");
        let pool = make_pool();

        let first = pool.register_readonly_file(&path).expect("register first");
        pool.unregister_file(first).expect("unregister first");
        let second = pool.register_readonly_file(&path).expect("register second");

        assert_eq!(first, second);
    }

    #[cfg(feature = "uring")]
    #[test]
    fn fixed_file_update_wakes_empty_queue() {
        let queue = Arc::new(QueueCore::new(1, 1));
        let fd = test_eventfd();
        queue.register_wake_fd(fd);
        queue.push_fixed_file_update(FixedFileUpdate { file: 7, fd: -1 });

        assert!(matches!(queue.pop_or_notification(0), QueueWake::Notified));

        let mut updates = Vec::new();
        assert_eq!(queue.fixed_file_updates_since(fd, 0, &mut updates), 1);
        assert_eq!(updates, vec![FixedFileUpdate { file: 7, fd: -1 }]);
        queue.unregister_wake_fd(fd);
        close_test_fd(fd);
    }

    #[cfg(feature = "uring")]
    #[test]
    fn fixed_file_updates_compact_after_all_readers_advance() {
        let queue = Arc::new(QueueCore::new(1, 1));
        let first_fd = test_eventfd();
        let second_fd = test_eventfd();
        queue.register_wake_fd(first_fd);
        queue.register_wake_fd(second_fd);

        queue.push_fixed_file_update(FixedFileUpdate { file: 3, fd: 10 });

        let mut first_updates = Vec::new();
        let first_cursor = queue.fixed_file_updates_since(first_fd, 0, &mut first_updates);
        assert_eq!(first_cursor, 1);
        assert_eq!(first_updates, vec![FixedFileUpdate { file: 3, fd: 10 }]);
        assert_eq!(queue.debug_fixed_file_update_count(), 1);

        let mut second_updates = Vec::new();
        let second_cursor = queue.fixed_file_updates_since(second_fd, 0, &mut second_updates);
        assert_eq!(second_cursor, 1);
        assert_eq!(second_updates, vec![FixedFileUpdate { file: 3, fd: 10 }]);
        assert_eq!(queue.debug_fixed_file_update_count(), 0);

        queue.push_fixed_file_update(FixedFileUpdate { file: 4, fd: -1 });

        first_updates.clear();
        assert_eq!(
            queue.fixed_file_updates_since(first_fd, first_cursor, &mut first_updates),
            2
        );
        assert_eq!(first_updates, vec![FixedFileUpdate { file: 4, fd: -1 }]);
        assert_eq!(queue.debug_fixed_file_update_count(), 1);

        second_updates.clear();
        assert_eq!(
            queue.fixed_file_updates_since(second_fd, second_cursor, &mut second_updates),
            2
        );
        assert_eq!(second_updates, vec![FixedFileUpdate { file: 4, fd: -1 }]);
        assert_eq!(queue.debug_fixed_file_update_count(), 0);

        queue.unregister_wake_fd(first_fd);
        queue.unregister_wake_fd(second_fd);
        close_test_fd(first_fd);
        close_test_fd(second_fd);
    }

    #[cfg(feature = "uring")]
    #[test]
    fn fixed_file_updates_only_target_file_shard() {
        let path_a = temp_file("fixed_update_shard_a", b"a");
        let path_b = temp_file("fixed_update_shard_b", b"b");
        let pool = make_unstarted_uring_pool(2, 4);

        let file_a = pool.register_readonly_file(&path_a).expect("register a");
        assert_eq!(file_a, 0);
        assert_eq!(pool.queues[0].debug_fixed_file_update_count(), 1);
        assert_eq!(pool.queues[1].debug_fixed_file_update_count(), 0);

        let file_b = pool.register_readonly_file(&path_b).expect("register b");
        assert_eq!(file_b, 1);
        assert_eq!(pool.queues[0].debug_fixed_file_update_count(), 1);
        assert_eq!(pool.queues[1].debug_fixed_file_update_count(), 1);

        pool.unregister_file(file_a).expect("unregister a");
        assert_eq!(pool.queues[0].debug_fixed_file_update_count(), 2);
        assert_eq!(pool.queues[1].debug_fixed_file_update_count(), 1);
    }

    #[test]
    fn read_exact() {
        let path = temp_file("read_exact", b"abcdefghijklmnopqrstuvwxyz");
        let pool = make_pool();
        let file = pool.register_file(&path).expect("register");

        let result = pool
            .submit(IoCommand::read(file, 2, 5, 0))
            .expect("submit")
            .blocking_recv_read()
            .expect("read");

        assert_eq!(&*result, b"cdefg");
    }

    #[test]
    fn write_then_read() {
        let path = temp_file("write_then_read", b"hello world");
        let pool = make_pool();
        let file = pool.register_file(&path).expect("register");

        pool.submit(IoCommand::write(file, 6, b"rust!".to_vec(), 1))
            .expect("submit write")
            .blocking_recv()
            .expect("write");
        pool.submit(IoCommand::fsync(file, 1))
            .expect("submit fsync")
            .blocking_recv()
            .expect("fsync");

        let result = pool
            .submit(IoCommand::read(file, 0, 11, 1))
            .expect("submit read")
            .blocking_recv_read()
            .expect("read");
        assert_eq!(&*result, b"hello rust!");
    }

    #[test]
    fn sharded_threaded_pool_reads_from_multiple_files() {
        let path_a = temp_file("sharded_threaded_reads_a", b"abcdefghijklmnopqrstuvwxyz");
        let path_b = temp_file("sharded_threaded_reads_b", b"ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        let pool = IoPool::new(IoConfig::Threaded(ThreadedConfig {
            base: BaseIoConfig {
                max_in_flight: 8,
                queue_capacity: 16,
                priority_levels: 2,
                queue_shards: 2,
                assume_non_overlapping_reads: false,
            },
            num_workers: 4,
            cpus: None,
        }))
        .expect("create sharded pool");
        let file_a = pool.register_file(&path_a).expect("register file a");
        let file_b = pool.register_file(&path_b).expect("register file b");

        assert_ne!(
            Arc::as_ptr(pool.select_queue(&IoCommand::read(file_a, 0, 2, 0))),
            Arc::as_ptr(pool.select_queue(&IoCommand::read(file_b, 0, 2, 0)))
        );

        let reads = (0..4)
            .map(|index| {
                pool.submit(IoCommand::read(file_a, index * 2, 2, 0))
                    .expect("submit read")
            })
            .chain((0..4).map(|index| {
                pool.submit(IoCommand::read(file_b, index * 2, 2, 0))
                    .expect("submit read")
            }))
            .collect::<Vec<_>>();
        let mut out = Vec::new();
        for future in reads {
            out.extend_from_slice(&future.blocking_recv_read().expect("read"));
        }
        assert_eq!(&out, b"abcdefghABCDEFGH");
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_threaded_io_operations() {
        let path = temp_file("profile_threaded_io", b"hello world");
        let profiler = enabled_profile("iopool-threaded-test");
        let pool = IoPool::new_with_profile(
            IoConfig::Threaded(ThreadedConfig {
                base: BaseIoConfig {
                    max_in_flight: 8,
                    queue_capacity: 16,
                    priority_levels: 2,
                    queue_shards: 1,
                    assume_non_overlapping_reads: false,
                },
                num_workers: 1,
                cpus: None,
            }),
            profiler.clone(),
        )
        .expect("create profiled pool");
        let round = profiler.start();
        assert!(pool.profile().is_global_enabled());
        let file = pool.register_file(&path).expect("register");

        let empty = pool
            .submit(IoCommand::read(file, 0, 0, 0))
            .expect("submit empty read")
            .blocking_recv_read()
            .expect("empty read");
        assert!(empty.is_empty());
        let read = pool
            .submit(IoCommand::read(file, 0, 5, 0))
            .expect("submit read")
            .blocking_recv_read()
            .expect("read");
        assert_eq!(&*read, b"hello");
        let written = pool
            .submit(IoCommand::write(file, 6, b"rust!".to_vec(), 0))
            .expect("submit write")
            .blocking_recv()
            .expect("write");
        assert_eq!(written.bytes_written(), Some(5));

        let snapshot = profiler.snapshot();
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 3);
        assert_eq!(metric(&snapshot, test_metrics::READ_SUBMITS), 2);
        assert_eq!(metric(&snapshot, test_metrics::WRITE_SUBMITS), 1);
        assert_eq!(metric(&snapshot, test_metrics::IMMEDIATE_COMPLETIONS), 1);
        assert_eq!(metric(&snapshot, test_metrics::DISPATCHED), 2);
        assert_eq!(metric(&snapshot, test_metrics::OPERATION_CALLS), 2);
        assert_eq!(metric(&snapshot, test_metrics::READ_BYTES), 5);
        assert_eq!(metric(&snapshot, test_metrics::WRITE_BYTES), 5);
        assert_eq!(metric(&snapshot, test_metrics::OPERATION_ERRORS), 0);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_queue_dedup_dispatch_and_cancel() {
        let (runtime, profiler) = enabled_queue_profile("iopool-queue-test");
        let queue = Arc::new(QueueCore::new_with_limiter_and_profile(
            3,
            8,
            8,
            false,
            Arc::new(ActiveLimiter::new(8)),
            Arc::new(Semaphore::new(8)),
            Arc::new(QueueSignal::default()),
            profiler.clone(),
        ));
        let round = runtime.start();

        let first = queue
            .submit(IoCommand::read(0, 10, 4, 2))
            .expect("first submit");
        let second = queue
            .submit(IoCommand::read(0, 10, 4, 0))
            .expect("dedup submit");
        let cancelled = queue
            .submit(IoCommand::read(0, 20, 4, 1))
            .expect("cancelled submit");
        drop(cancelled);

        let work = queue.pop().expect("dispatch deduplicated read");
        queue.complete(
            work.id,
            work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"rust".to_vec()))),
        );
        assert_eq!(&*first.blocking_recv_read().expect("first result"), b"rust");
        assert_eq!(
            &*second.blocking_recv_read().expect("second result"),
            b"rust"
        );

        let snapshot = runtime.snapshot();
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 3);
        assert_eq!(metric(&snapshot, test_metrics::READ_SUBMITS), 3);
        assert_eq!(metric(&snapshot, test_metrics::DEDUP_HITS), 1);
        assert_eq!(
            metric(&snapshot, test_metrics::CANCELLED_BEFORE_DISPATCH),
            1
        );
        assert_eq!(metric(&snapshot, test_metrics::DISPATCHED), 1);
        assert_eq!(metric(&snapshot, test_metrics::OPERATION_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::READ_BYTES), 4);
        assert_eq!(metric(&snapshot, test_metrics::OPERATION_ERRORS), 0);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_queue_capacity_rejections() {
        let (runtime, profiler) = enabled_queue_profile("iopool-capacity-test");
        let queue = Arc::new(QueueCore::new_with_limiter_and_profile(
            2,
            1,
            8,
            false,
            Arc::new(ActiveLimiter::new(8)),
            Arc::new(Semaphore::new(1)),
            Arc::new(QueueSignal::default()),
            profiler.clone(),
        ));
        let round = runtime.start();
        let held = queue
            .submit(IoCommand::read(0, 0, 1, 0))
            .expect("held submit");

        let err = queue
            .try_submit_with_handle(IoCommand::read(0, 1, 1, 0), None, true)
            .expect_err("queue capacity should reject second unique read");
        assert!(matches!(err, QueueSubmitError::Full(_)));

        let snapshot = runtime.snapshot();
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 2);
        assert_eq!(metric(&snapshot, test_metrics::QUEUE_FULL), 1);
        drop(held);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_global_active_limit_dispatch_backpressure() {
        let (runtime, profiler) = enabled_queue_profile("iopool-active-limit-test");
        let limiter = Arc::new(ActiveLimiter::new(1));
        let queue_signal = Arc::new(QueueSignal::default());
        let first_queue = Arc::new(QueueCore::new_with_limiter_and_profile(
            2,
            8,
            1,
            false,
            Arc::clone(&limiter),
            Arc::new(Semaphore::new(8)),
            Arc::clone(&queue_signal),
            profiler.clone(),
        ));
        let second_queue = Arc::new(QueueCore::new_with_limiter_and_profile(
            2,
            8,
            1,
            false,
            limiter,
            Arc::new(Semaphore::new(8)),
            queue_signal,
            profiler.clone(),
        ));
        let round = runtime.start();
        let held = first_queue
            .submit(IoCommand::read(0, 0, 1, 0))
            .expect("held submit");
        let waiting = second_queue
            .submit(IoCommand::read(1, 0, 1, 0))
            .expect("second submit should queue");

        let first_work = first_queue.try_pop_for_test().expect("dispatch first read");
        assert!(
            second_queue.try_pop_for_test().is_none(),
            "global active limit should stall second dispatch"
        );
        first_queue.complete(
            first_work.id,
            first_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"a".to_vec()))),
        );
        let second_work = second_queue
            .try_pop_for_test()
            .expect("second read should dispatch after active release");
        second_queue.complete(
            second_work.id,
            second_work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"b".to_vec()))),
        );

        let snapshot = runtime.snapshot();
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 2);
        assert_eq!(metric(&snapshot, test_metrics::QUEUE_FULL), 0);
        assert_eq!(metric(&snapshot, test_metrics::DISPATCHED), 2);
        assert_eq!(&*held.blocking_recv_read().expect("held"), b"a");
        assert_eq!(&*waiting.blocking_recv_read().expect("waiting"), b"b");
        round.end();
    }

    #[test]
    fn disabled_profile_does_not_record_metrics() {
        let profiler = IoPoolProfile::disabled();
        let queue = Arc::new(QueueCore::new_with_limiter_and_profile(
            2,
            4,
            4,
            false,
            Arc::new(ActiveLimiter::new(4)),
            Arc::new(Semaphore::new(4)),
            Arc::new(QueueSignal::default()),
            profiler.clone(),
        ));

        let empty = queue
            .submit(IoCommand::read(0, 0, 0, 0))
            .expect("submit immediate read");
        assert!(empty.blocking_recv_read().expect("read").is_empty());

        let snapshot = profiler.snapshot();
        assert!(!snapshot.global_enabled);
        assert!(snapshot.metrics.is_empty());
        assert_eq!(snapshot.metric_value(test_metrics::SUBMIT_CALLS), None);
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_all_submit_kinds() {
        let (runtime, profiler) = enabled_queue_profile("iopool-submit-kinds-test");
        let queue = Arc::new(QueueCore::new_with_limiter_and_profile(
            2,
            8,
            8,
            false,
            Arc::new(ActiveLimiter::new(8)),
            Arc::new(Semaphore::new(8)),
            Arc::new(QueueSignal::default()),
            profiler,
        ));
        let round = runtime.start();

        let futures = vec![
            queue
                .submit(IoCommand::read(0, 0, 1, 0))
                .expect("read submit"),
            queue
                .submit(IoCommand::write(0, 0, vec![1], 0))
                .expect("write submit"),
            queue.submit(IoCommand::fsync(0, 0)).expect("fsync submit"),
            queue
                .submit(IoCommand::sync_data(0, 0))
                .expect("sync data submit"),
            queue
                .submit(IoCommand::truncate(0, 1, 0))
                .expect("truncate submit"),
            queue
                .submit(IoCommand::metadata(0, 0))
                .expect("metadata submit"),
        ];

        let snapshot = runtime.snapshot();
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 6);
        assert_eq!(metric(&snapshot, test_metrics::READ_SUBMITS), 1);
        assert_eq!(metric(&snapshot, test_metrics::WRITE_SUBMITS), 1);
        assert_eq!(metric(&snapshot, test_metrics::FSYNC_SUBMITS), 1);
        assert_eq!(metric(&snapshot, test_metrics::SYNC_DATA_SUBMITS), 1);
        assert_eq!(metric(&snapshot, test_metrics::TRUNCATE_SUBMITS), 1);
        assert_eq!(metric(&snapshot, test_metrics::METADATA_SUBMITS), 1);

        drop(futures);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_records_operation_errors() {
        let (runtime, profiler) = enabled_queue_profile("iopool-error-test");
        let queue = Arc::new(QueueCore::new_with_limiter_and_profile(
            2,
            4,
            4,
            false,
            Arc::new(ActiveLimiter::new(4)),
            Arc::new(Semaphore::new(4)),
            Arc::new(QueueSignal::default()),
            profiler,
        ));
        let round = runtime.start();

        let future = queue
            .submit(IoCommand::read(0, 0, 4, 0))
            .expect("submit read");
        let work = queue.pop().expect("dispatch read");
        queue.complete(
            work.id,
            work.operation_started,
            Err(io::Error::other("profiled failure")),
        );
        assert_eq!(
            future.blocking_recv().expect_err("read should fail").kind(),
            ErrorKind::Other
        );

        let snapshot = runtime.snapshot();
        assert_eq!(metric(&snapshot, test_metrics::DISPATCHED), 1);
        assert_eq!(metric(&snapshot, test_metrics::OPERATION_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::READ_BYTES), 0);
        assert_eq!(metric(&snapshot, test_metrics::OPERATION_ERRORS), 1);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn env_auto_profile_snapshot_and_reset_keeps_recording() {
        with_profile_env_enabled(|| {
            let path = temp_file("profile_env_auto_reset", b"hello");
            let pool = IoPool::new(IoConfig::Threaded(ThreadedConfig {
                base: BaseIoConfig {
                    max_in_flight: 4,
                    queue_capacity: 16,
                    priority_levels: 2,
                    queue_shards: 1,
                    assume_non_overlapping_reads: false,
                },
                num_workers: 1,
                cpus: None,
            }))
            .expect("create env-profiled pool");
            assert!(pool.profile().is_recording());

            let file = pool.register_file(&path).expect("register");
            let first = pool
                .submit(IoCommand::read(file, 0, 0, 0))
                .expect("first submit")
                .blocking_recv_read()
                .expect("first read");
            assert!(first.is_empty());

            let first_snapshot = pool.profile_snapshot_and_reset();
            assert_eq!(metric(&first_snapshot, test_metrics::SUBMIT_CALLS), 1);
            assert_eq!(
                metric(&first_snapshot, test_metrics::IMMEDIATE_COMPLETIONS),
                1
            );
            assert!(pool.profile().is_recording());
            assert_eq!(
                metric(&pool.profile_snapshot(), test_metrics::SUBMIT_CALLS),
                0
            );

            let second = pool
                .submit(IoCommand::read(file, 1, 0, 0))
                .expect("second submit")
                .blocking_recv_read()
                .expect("second read");
            assert!(second.is_empty());
            pool.reset_profile();
            assert!(pool.profile().is_recording());
            assert_eq!(
                metric(&pool.profile_snapshot(), test_metrics::SUBMIT_CALLS),
                0
            );
        });
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_registers_iopool_scopes_for_active_runtime() {
        let runtime =
            ProfileRuntime::enabled_lazy("iopool-late-registry-test", ProfileRegistry::new);
        let round = runtime.start();
        let queue = Arc::new(QueueCore::new_with_limiter_and_profile(
            2,
            4,
            4,
            false,
            Arc::new(ActiveLimiter::new(4)),
            Arc::new(Semaphore::new(4)),
            Arc::new(QueueSignal::default()),
            IoPoolProfile::new(runtime.clone()),
        ));

        let future = queue
            .submit(IoCommand::read(0, 0, 4, 0))
            .expect("submit read");
        let work = queue.pop().expect("dispatch read");
        queue.complete(
            work.id,
            work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"rust".to_vec()))),
        );
        assert_eq!(&*future.blocking_recv_read().expect("read"), b"rust");

        let snapshot = runtime.snapshot();
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::DISPATCHED), 1);
        assert_eq!(metric(&snapshot, test_metrics::OPERATION_CALLS), 1);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn profile_cache_refreshes_between_rounds() {
        let (runtime, profiler) = enabled_queue_profile("iopool-cache-refresh-test");
        let queue = Arc::new(QueueCore::new_with_limiter_and_profile(
            2,
            4,
            4,
            false,
            Arc::new(ActiveLimiter::new(4)),
            Arc::new(Semaphore::new(4)),
            Arc::new(QueueSignal::default()),
            profiler,
        ));

        let round = runtime.start();
        let first = queue
            .submit(IoCommand::read(0, 0, 0, 0))
            .expect("first immediate read");
        assert!(first.blocking_recv_read().expect("first").is_empty());
        let first_snapshot = round.end();
        assert_eq!(metric(&first_snapshot, test_metrics::SUBMIT_CALLS), 1);
        assert_eq!(
            metric(&first_snapshot, test_metrics::IMMEDIATE_COMPLETIONS),
            1
        );

        let round = runtime.start();
        let second = queue
            .submit(IoCommand::read(0, 1, 0, 0))
            .expect("second immediate read");
        assert!(second.blocking_recv_read().expect("second").is_empty());
        let second_snapshot = round.end();
        assert_eq!(metric(&second_snapshot, test_metrics::SUBMIT_CALLS), 1);
        assert_eq!(
            metric(&second_snapshot, test_metrics::IMMEDIATE_COMPLETIONS),
            1
        );
    }

    #[test]
    fn fast_read_path_tracks_active_reads_without_active_index() {
        let queue = Arc::new(QueueCore::new_with_fast_read_path(2, 8, 8, true));
        let future = queue
            .submit(IoCommand::read(0, 0, 4, 1))
            .expect("submit fast read");

        assert_eq!(queue.debug_counts(), (1, 1));
        assert_eq!(queue.debug_active_index_count(), 0);
        assert_eq!(queue.debug_fast_active_read_count(0), 1);

        let work = queue.pop().expect("pop fast read");
        assert!(matches!(work.op, IoOperation::Read { .. }));
        assert_eq!(queue.debug_counts(), (1, 0));
        assert_eq!(queue.debug_fast_active_read_count(0), 1);

        queue.complete(
            work.id,
            work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"rust".to_vec()))),
        );
        assert_eq!(&*future.blocking_recv_read().expect("read"), b"rust");
        assert_eq!(queue.debug_counts(), (0, 0));
        assert_eq!(queue.debug_fast_active_read_count(0), 0);
    }

    #[test]
    fn sync_barrier_preserves_write_then_read_order() {
        let path = temp_file("sync_barrier_write_then_read", b"hello world");
        let pool = make_pool();
        let file = pool.register_file(&path).expect("register");

        let write = pool
            .submit(IoCommand::write(file, 6, b"rust!".to_vec(), 2))
            .expect("submit write");
        let sync = pool
            .submit(IoCommand::sync_all(file, 1))
            .expect("submit sync");
        let read = pool
            .submit(IoCommand::read(file, 0, 11, 0))
            .expect("submit read");

        assert_eq!(
            &*read.blocking_recv_read().expect("read after write"),
            b"hello rust!"
        );
        assert_eq!(
            write
                .blocking_recv()
                .expect("write")
                .bytes_written()
                .expect("write output"),
            5
        );
        assert!(matches!(
            sync.blocking_recv().expect("sync"),
            IoOutput::SyncAll
        ));
    }

    #[test]
    fn duplicate_reads_share_single_queue_entry() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let first = queue
            .submit(IoCommand::read(0, 10, 4, 2))
            .expect("first submit");
        let second = queue
            .submit(IoCommand::read(0, 10, 4, 0))
            .expect("second submit");

        assert_eq!(queue.debug_counts(), (1, 1));
        drop(first);
        assert_eq!(queue.debug_counts(), (1, 1));
        drop(second);
        assert_eq!(queue.debug_counts(), (0, 0));
        queue.shutdown();
        assert!(queue.pop().is_none());
    }

    #[test]
    fn duplicate_read_refcount_overflow_is_rejected() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let first = queue
            .submit(IoCommand::read(0, 10, 4, 2))
            .expect("first submit");

        {
            let mut inner = queue.inner.lock().unwrap();
            inner
                .table
                .get_mut(&WorkId(0))
                .expect("first work")
                .refcount = usize::MAX;
        }

        let err = queue
            .submit(IoCommand::read(0, 10, 4, 0))
            .expect_err("overflowing duplicate should be rejected");
        assert_eq!(err.kind(), ErrorKind::OutOfMemory);

        {
            let mut inner = queue.inner.lock().unwrap();
            inner
                .table
                .get_mut(&WorkId(0))
                .expect("first work")
                .refcount = 1;
        }
        drop(first);
        assert_eq!(queue.debug_counts(), (0, 0));
    }

    #[test]
    fn writes_are_not_deduplicated() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let first = queue
            .submit(IoCommand::write(0, 0, b"a".to_vec(), 1))
            .expect("first submit");
        let second = queue
            .submit(IoCommand::write(0, 0, b"b".to_vec(), 1))
            .expect("second submit");

        assert_eq!(queue.debug_counts(), (2, 2));
        drop(first);
        drop(second);
        assert_eq!(queue.debug_counts(), (0, 0));
    }

    #[test]
    fn sync_operations_are_not_deduplicated() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let first = queue.submit(IoCommand::fsync(0, 1)).expect("first fsync");
        let second = queue
            .submit(IoCommand::sync_all(0, 1))
            .expect("second fsync");

        assert_eq!(queue.debug_counts(), (2, 2));
        drop(first);
        drop(second);
    }

    #[test]
    fn duplicate_read_after_intervening_truncate_is_not_shared() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let _first_read = queue
            .submit(IoCommand::read(0, 0, 4, 2))
            .expect("first read");
        let _truncate = queue
            .submit(IoCommand::truncate(0, 0, 2))
            .expect("truncate");
        let _second_read = queue
            .submit(IoCommand::read(0, 0, 4, 0))
            .expect("second read");

        assert_eq!(queue.debug_counts(), (3, 3));
    }

    #[test]
    fn higher_priority_duplicate_promotes_queued_read() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let _other = queue
            .submit(IoCommand::read(0, 20, 1, 1))
            .expect("other read");
        let _low = queue
            .submit(IoCommand::read(0, 10, 1, 2))
            .expect("low read");
        let _duplicate = queue
            .submit(IoCommand::read(0, 10, 1, 0))
            .expect("duplicate read");

        let work = queue.pop().expect("work");
        match work.op {
            IoOperation::Read { offset, .. } => assert_eq!(offset, 10),
            _ => panic!("expected read"),
        }
    }

    #[test]
    fn dispatched_fast_read_duplicate_does_not_leave_stale_ready_entry() {
        let queue = Arc::new(QueueCore::new_with_fast_read_path(3, 8, 8, true));
        let first = queue
            .submit(IoCommand::read(0, 10, 4, 2))
            .expect("first read");
        let work = queue.pop().expect("dispatch first read");
        assert_eq!(queue.debug_fast_ready_count(), 0);

        let duplicate = queue
            .submit(IoCommand::read(0, 10, 4, 0))
            .expect("dedup dispatched read");
        assert_eq!(queue.debug_fast_ready_count(), 0);

        queue.complete(
            work.id,
            work.operation_started,
            Ok(IoOutput::Read(Arc::from(b"rust".to_vec()))),
        );
        assert_eq!(&*first.blocking_recv_read().expect("first result"), b"rust");
        assert_eq!(
            &*duplicate.blocking_recv_read().expect("duplicate result"),
            b"rust"
        );
    }

    #[test]
    fn range_read_can_bypass_earlier_write_under_no_overlap_assumption() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let _write = queue
            .submit(IoCommand::write(0, 0, b"a".to_vec(), 2))
            .expect("write");
        let _read = queue.submit(IoCommand::read(0, 0, 1, 0)).expect("read");

        let work = queue.pop().expect("work");
        match work.op {
            IoOperation::Read { offset, .. } => assert_eq!(offset, 0),
            _ => panic!("expected read"),
        }
    }

    #[test]
    fn disjoint_read_can_bypass_lower_priority_write() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let _write = queue
            .submit(IoCommand::write(0, 0, b"a".to_vec(), 2))
            .expect("write");
        let _read = queue.submit(IoCommand::read(0, 10, 1, 0)).expect("read");

        let work = queue.pop().expect("work");
        match work.op {
            IoOperation::Read { offset, .. } => assert_eq!(offset, 10),
            _ => panic!("expected read"),
        }
    }

    #[test]
    fn sync_orders_after_earlier_write() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let _write = queue
            .submit(IoCommand::write(0, 0, b"a".to_vec(), 2))
            .expect("write");
        let _sync = queue.submit(IoCommand::fsync(0, 0)).expect("sync");

        let work = queue.pop().expect("work");
        match work.op {
            IoOperation::Write { .. } => {}
            _ => panic!("expected write before sync"),
        }
    }

    #[test]
    fn barrier_blocked_high_priority_work_does_not_hide_other_files() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let _write = queue
            .submit(IoCommand::write(0, 0, b"a".to_vec(), 2))
            .expect("write");
        let _blocked = queue.submit(IoCommand::fsync(0, 0)).expect("sync");
        let _other_file = queue.submit(IoCommand::read(1, 0, 1, 1)).expect("read");

        let work = queue.pop().expect("work");
        match work.op {
            IoOperation::Read { file, .. } => assert_eq!(file, 1),
            _ => panic!("expected read from other file"),
        }
    }

    #[test]
    fn active_file_index_tracks_cancel_and_complete() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let first = queue.submit(IoCommand::read(0, 0, 1, 0)).expect("first");
        let second = queue.submit(IoCommand::read(1, 0, 1, 0)).expect("second");

        assert_eq!(queue.debug_active_index_count(), 2);
        drop(first);
        assert_eq!(queue.debug_counts(), (1, 1));
        assert_eq!(queue.debug_active_index_count(), 1);

        let work = queue.pop().expect("work");
        queue.complete(
            work.id,
            work.operation_started,
            Ok(IoOutput::Read(Arc::from(vec![0u8]))),
        );
        assert_eq!(queue.debug_counts(), (0, 0));
        assert_eq!(queue.debug_active_index_count(), 0);
        drop(second);
    }

    #[test]
    fn dropped_dispatched_barrier_still_blocks_same_file_work() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let sync = queue.submit(IoCommand::fsync(0, 2)).expect("sync");
        let sync_work = queue.pop().expect("dispatch sync");
        drop(sync);

        let _blocked_read = queue.submit(IoCommand::read(0, 0, 1, 0)).expect("read");
        let _other_file = queue.submit(IoCommand::read(1, 0, 1, 1)).expect("other");

        let work = queue.pop().expect("other file should still dispatch");
        match work.op {
            IoOperation::Read { file, .. } => assert_eq!(file, 1),
            _ => panic!("expected other-file read"),
        }

        queue.complete(
            sync_work.id,
            sync_work.operation_started,
            Ok(IoOutput::SyncAll),
        );
        let work = queue.pop().expect("same-file read should unblock");
        match work.op {
            IoOperation::Read { file, offset, .. } => {
                assert_eq!(file, 0);
                assert_eq!(offset, 0);
            }
            _ => panic!("expected read after barrier completion"),
        }
    }

    #[test]
    fn priority_zero_is_highest() {
        let queue = Arc::new(QueueCore::new(3, 8));
        let _low = queue
            .submit(IoCommand::read(0, 20, 1, 2))
            .expect("low submit");
        let _high = queue
            .submit(IoCommand::read(0, 10, 1, 0))
            .expect("high submit");

        let work = queue.pop().expect("work");
        match work.op {
            IoOperation::Read { offset, .. } => assert_eq!(offset, 10),
            _ => panic!("expected read"),
        }
    }

    #[test]
    fn invalid_priority_is_rejected() {
        let queue = Arc::new(QueueCore::new(2, 8));
        let err = queue
            .submit(IoCommand::read(0, 0, 1, 2))
            .expect_err("priority should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn queue_fail_wakes_waiters_and_rejects_submits() {
        let queue = Arc::new(QueueCore::new(2, 8));
        let future = queue.submit(IoCommand::read(0, 0, 1, 0)).expect("submit");

        queue.fail(io::Error::new(io::ErrorKind::BrokenPipe, "backend failed"));

        let err = future
            .blocking_recv_read()
            .expect_err("future should receive backend failure");
        assert_eq!(err.kind(), ErrorKind::BrokenPipe);
        let err = queue
            .submit(IoCommand::read(0, 0, 1, 0))
            .expect_err("failed queue should reject new work");
        assert_eq!(err.kind(), ErrorKind::BrokenPipe);
        assert!(queue.pop().is_none());
    }

    #[test]
    fn sync_data_truncate_metadata_and_unregister() {
        let path = temp_file("metadata_ops", b"hello world");
        let pool = make_pool();
        let file = pool.register_file(&path).expect("register");

        assert!(matches!(
            pool.submit(IoCommand::sync_data(file, 1))
                .expect("submit sync_data")
                .blocking_recv()
                .expect("sync_data"),
            IoOutput::SyncData
        ));
        assert!(matches!(
            pool.submit(IoCommand::truncate(file, 5, 2))
                .expect("submit truncate")
                .blocking_recv()
                .expect("truncate"),
            IoOutput::Truncate
        ));

        let metadata = pool
            .submit(IoCommand::metadata(file, 0))
            .expect("submit metadata")
            .blocking_recv()
            .expect("metadata");
        assert_eq!(metadata.metadata().expect("metadata output").len, 5);

        pool.unregister_file(file).expect("unregister");
        let err = pool
            .submit(IoCommand::read(file, 0, 1, 0))
            .expect_err("unregistered file should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn readonly_registration_surfaces_write_error() {
        let path = temp_file("readonly_registration", b"hello");
        let pool = make_pool();
        let file = pool
            .register_readonly_file(&path)
            .expect("register readonly");

        let err = pool
            .submit(IoCommand::write(file, 0, b"x".to_vec(), 0))
            .expect("submit write")
            .blocking_recv()
            .expect_err("write on readonly file should fail");
        assert!(
            err.raw_os_error() == Some(libc::EBADF) || err.kind() == ErrorKind::PermissionDenied
        );
    }

    #[cfg(feature = "uring")]
    #[test]
    fn uring_backend_read_write_when_available() {
        let path = temp_file("uring_read_write", b"hello world");
        let pool = match IoPool::new(IoConfig::Uring(UringConfig {
            base: BaseIoConfig {
                max_in_flight: 16,
                queue_capacity: 64,
                priority_levels: 3,
                queue_shards: 1,
                assume_non_overlapping_reads: false,
            },
            entries: 8,
            drivers: 1,
            iowq_bounded_workers: 0,
            iowq_unbounded_workers: 0,
            registered_files: 16,
        })) {
            Ok(pool) => pool,
            Err(err)
                if matches!(
                    err.raw_os_error(),
                    Some(libc::ENOSYS | libc::EPERM | libc::EACCES)
                ) =>
            {
                eprintln!("skipping io_uring runtime test: {err}");
                return;
            }
            Err(err) => panic!("create uring pool: {err}"),
        };
        let file = pool.register_file(&path).expect("register");

        let write = pool
            .submit(IoCommand::write(file, 6, b"rust!".to_vec(), 2))
            .expect("submit write");
        assert_eq!(
            write
                .blocking_recv()
                .expect("write")
                .bytes_written()
                .expect("write output"),
            5
        );
        let read = pool
            .submit(IoCommand::read(file, 0, 11, 0))
            .expect("submit read");
        assert_eq!(&*read.blocking_recv_read().expect("read"), b"hello rust!");

        pool.submit(IoCommand::sync_data(file, 1))
            .expect("submit sync_data")
            .blocking_recv()
            .expect("sync_data");
        pool.submit(IoCommand::truncate(file, 5, 1))
            .expect("submit truncate")
            .blocking_recv()
            .expect("truncate");
        let metadata = pool
            .submit(IoCommand::metadata(file, 0))
            .expect("submit metadata")
            .blocking_recv()
            .expect("metadata");
        assert_eq!(metadata.metadata().expect("metadata output").len, 5);
    }

    #[test]
    fn threaded_workers_and_cpu_allow_list_are_decoupled() {
        let config = ThreadedConfig {
            base: BaseIoConfig::default(),
            num_workers: 8,
            cpus: Some(vec![0, 1]),
        };
        config.validate().expect("workers may exceed cpu count");

        let empty = ThreadedConfig {
            cpus: Some(Vec::new()),
            ..ThreadedConfig::default()
        };
        assert!(empty.validate().is_err());

        let duplicate = ThreadedConfig {
            cpus: Some(vec![0, 0]),
            ..ThreadedConfig::default()
        };
        assert!(duplicate.validate().is_err());
    }
}
