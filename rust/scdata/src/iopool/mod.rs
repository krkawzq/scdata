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
#[cfg(feature = "uring")]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock, Weak};
use std::task::{Context, Poll};
use std::thread;

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
    #[deprecated(note = "writes are not deduplicated")]
    Write,
    #[deprecated(note = "fsync operations are not deduplicated")]
    Fsync,
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
    /// Maximum number of unique active IO operations. Duplicate reads do not
    /// consume another slot. Default: 1024.
    pub max_in_flight: usize,
    /// Number of priority levels. `0` is the highest priority. Default: 3.
    pub priority_levels: usize,
    /// Number of independent internal queues. Default: 1.
    ///
    /// Values greater than 1 reduce lock contention for high-throughput,
    /// non-overlapping IO workloads. Cross-request ordering and deduplication
    /// are only guaranteed within a shard.
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

    pub(super) fn queue_shards(&self) -> usize {
        self.queue_shards.max(1)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.max_in_flight == 0 {
            return Err("max_in_flight must be greater than 0".to_string());
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

    pub(super) fn queue_shards(&self) -> usize {
        let requested = self.base().queue_shards();
        let consumers = match self {
            Self::Uring(config) => config.drivers.max(1),
            Self::Threaded(config) => config.worker_count(),
        };
        requested.min(consumers).max(1)
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
    Closing { _handle: Arc<File> },
    Closed,
}

pub(super) type FileTable = RwLock<Vec<FileSlot>>;

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
    fn for_operation(op: &IoOperation) -> Self {
        match op {
            IoOperation::Read { file, .. } => Self::Read { file: *file },
            IoOperation::Write { file, .. } => Self::Write { file: *file },
            IoOperation::Fsync { file, .. } | IoOperation::SyncData { file, .. } => {
                Self::Sync { file: *file }
            }
            IoOperation::Truncate { file, .. } => Self::Truncate { file: *file },
            IoOperation::Metadata { file, .. } => Self::Metadata { file: *file },
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
    refcount: usize,
    state: InflightState,
    priority: usize,
    seq: u64,
    fast_read: bool,
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
    table: HashMap<WorkId, Inflight>,
    dedup: HashMap<RequestKey, WorkId>,
    active_by_file: Vec<Option<FileActive>>,
    next_id: u64,
    next_seq: u64,
    max_active: usize,
    shutdown: bool,
    failure: Option<SharedIoError>,
}

#[derive(Debug)]
pub(super) struct QueueCore {
    inner: Mutex<Inner>,
    available: Condvar,
    #[cfg(feature = "uring")]
    wake_fds: Mutex<Vec<RawFd>>,
    #[cfg(feature = "uring")]
    wake_pending: AtomicBool,
}

impl QueueCore {
    #[cfg(test)]
    pub(super) fn new(priority_levels: usize, max_active: usize) -> Self {
        Self::new_with_fast_read_path(priority_levels, max_active, false)
    }

    pub(super) fn new_with_fast_read_path(
        priority_levels: usize,
        max_active: usize,
        fast_read_path: bool,
    ) -> Self {
        let priority_levels = priority_levels.max(1);
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
                table: HashMap::with_capacity(max_active),
                dedup: HashMap::with_capacity(max_active),
                active_by_file: Vec::new(),
                next_id: 0,
                next_seq: 0,
                max_active: max_active.max(1),
                shutdown: false,
                failure: None,
            }),
            available: Condvar::new(),
            #[cfg(feature = "uring")]
            wake_fds: Mutex::new(Vec::new()),
            #[cfg(feature = "uring")]
            wake_pending: AtomicBool::new(false),
        }
    }

    #[cfg(test)]
    fn submit(self: &Arc<Self>, cmd: IoCommand) -> io::Result<IoFuture> {
        self.submit_with_handle(cmd, None)
    }

    fn submit_with_handle(
        self: &Arc<Self>,
        cmd: IoCommand,
        handle: Option<Arc<File>>,
    ) -> io::Result<IoFuture> {
        let priority = cmd.priority();
        let key = cmd.dedup_key();
        let operation = cmd.into_operation(handle);
        let access = Access::for_operation(&operation);
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

        if let Some(key) = key {
            if let Some(id) = inner.dedup.get(&key).copied() {
                if inner
                    .table
                    .get(&id)
                    .is_some_and(|inflight| Self::can_dedup_locked(&inner, id, inflight, access))
                {
                    Self::promote_priority_locked(&mut inner, id, priority);
                    let inflight = inner
                        .table
                        .get_mut(&id)
                        .expect("deduplicated entry checked above");
                    inflight.refcount += 1;
                    inflight.waiters.push(tx);
                    return Ok(IoFuture::new(rx, Arc::downgrade(self), id));
                }
                if !inner.table.contains_key(&id) {
                    inner.dedup.remove(&key);
                }
            }
        }

        if inner.table.len() >= inner.max_active {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("IO queue reached max_in_flight={}", inner.max_active),
            ));
        }

        let id = WorkId(inner.next_id);
        inner.next_id = inner.next_id.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::OutOfMemory, "IO work id counter overflowed")
        })?;
        let seq = inner.next_seq;
        inner.next_seq = inner.next_seq.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::OutOfMemory, "IO sequence counter overflowed")
        })?;

        let fast_read = Self::can_use_fast_read_locked(&inner, access);

        inner.table.insert(
            id,
            Inflight {
                key,
                op: Some(operation),
                access,
                waiters: Waiters::new(tx),
                refcount: 1,
                state: InflightState::Queued,
                priority,
                seq,
                fast_read,
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
            self.notify_one();
        }
        Ok(IoFuture::new(rx, Arc::downgrade(self), id))
    }

    pub(super) fn pop(&self) -> Option<IoWork> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if let Some(work) = Self::pop_ready_locked(&mut inner) {
                return Some(work);
            }

            if inner.shutdown {
                return None;
            }

            inner = self.available.wait(inner).unwrap();
        }
    }

    #[cfg(feature = "uring")]
    pub(super) fn pop_or_notification(&self, fixed_file_update_cursor: usize) -> QueueWake {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if let Some(work) = Self::pop_ready_locked(&mut inner) {
                return QueueWake::Work(work);
            }

            if inner.shutdown {
                return QueueWake::Shutdown;
            }
            if inner.fixed_file_updates.len() > fixed_file_update_cursor {
                return QueueWake::Notified;
            }

            inner = self.available.wait(inner).unwrap();
            if !inner.shutdown
                && inner.ready.is_empty()
                && inner.fixed_file_updates.len() > fixed_file_update_cursor
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
        let mut inner = self.inner.lock().unwrap();
        for _ in 0..limit {
            let Some(work) = Self::pop_ready_locked(&mut inner) else {
                break;
            };
            out.push(work);
        }
    }

    pub(super) fn complete(&self, id: WorkId, result: io::Result<IoOutput>) {
        let (waiters, ready_count) = {
            let mut inner = self.inner.lock().unwrap();
            let Some(inflight) = Self::remove_locked(&mut inner, id) else {
                return;
            };
            let access = inflight.access;
            let waiters = inflight.waiters;
            let ready_count = Self::refresh_ready_after_removed_locked(&mut inner, access);
            (waiters, ready_count)
        };

        self.notify_ready_count(ready_count);
        self.available.notify_all();
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
        if should_remove {
            inner.queued = inner.queued.saturating_sub(1);
            Self::remove_locked(&mut inner, id);
            let ready_count = Self::refresh_ready_after_removed_locked(&mut inner, access);
            drop(inner);
            self.notify_ready_count(ready_count);
            self.available.notify_all();
        }
    }

    pub(super) fn shutdown(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.shutdown = true;
        drop(inner);
        self.notify_all();
    }

    #[cfg(any(feature = "uring", test))]
    pub(super) fn fail(&self, err: io::Error) {
        let failure = SharedIoError::from(err);
        let waiters = {
            let mut inner = self.inner.lock().unwrap();
            inner.shutdown = true;
            inner.failure = Some(failure.clone());
            Self::drain_locked(&mut inner)
        };

        for waiters in waiters {
            waiters.send(Err(failure.clone()));
        }
        self.notify_all();
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

    #[cfg(feature = "uring")]
    #[allow(dead_code)]
    pub(super) fn register_wake_fd(&self, fd: RawFd) {
        self.wake_fds.lock().unwrap().push(fd);
    }

    #[cfg(feature = "uring")]
    #[allow(dead_code)]
    pub(super) fn unregister_wake_fd(&self, fd: RawFd) {
        let mut wake_fds = self.wake_fds.lock().unwrap();
        if let Some(index) = wake_fds.iter().position(|wake_fd| *wake_fd == fd) {
            wake_fds.swap_remove(index);
        }
        drop(wake_fds);
        self.wake_pending.store(false, Ordering::Release);
    }

    #[cfg(feature = "uring")]
    #[allow(dead_code)]
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
        cursor: usize,
        out: &mut Vec<FixedFileUpdate>,
    ) -> usize {
        let inner = self.inner.lock().unwrap();
        let cursor = cursor.min(inner.fixed_file_updates.len());
        out.extend_from_slice(&inner.fixed_file_updates[cursor..]);
        inner.fixed_file_updates.len()
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

                let op = {
                    let inflight = inner.table.get_mut(&id).expect("fast entry exists");
                    inflight.state = InflightState::Dispatched;
                    inflight.op.take()
                };
                inner.queued = inner.queued.saturating_sub(1);
                return op.map(|op| IoWork { id, op });
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

            let op = {
                let inflight = inner.table.get_mut(&id).expect("dispatchable entry exists");
                inflight.state = InflightState::Dispatched;
                inflight.op.take()
            };
            inner.queued = inner.queued.saturating_sub(1);
            return op.map(|op| IoWork { id, op });
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
        while Self::file_active_locked(&inner, file) {
            inner = self.available.wait(inner).unwrap();
        }
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
    fn debug_fast_active_read_count(&self, file: FileId) -> usize {
        let inner = self.inner.lock().unwrap();
        Self::fast_active_read_count(&inner, file)
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
        let Some(rx) = self.rx.as_mut() else {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "IO future was already consumed",
            )));
        };

        match Pin::new(rx).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(result)) => {
                self.released = true;
                Poll::Ready(shared_to_io_result(result))
            }
            Poll::Ready(Err(_)) => {
                self.released = true;
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
    submit_cursor: AtomicUsize,
    file_table: Arc<FileTable>,
    threads: Vec<thread::JoinHandle<()>>,
    #[cfg(feature = "uring")]
    uses_uring: bool,
}

impl IoPool {
    pub fn new(config: IoConfig) -> io::Result<Self> {
        config
            .validate()
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;

        let queue_count = config.queue_shards();
        let per_queue_limit = split_limit(config.in_flight_limit(), queue_count);
        let fast_read_path = config.base().assume_non_overlapping_reads;
        let queues = (0..queue_count)
            .map(|_| {
                Arc::new(QueueCore::new_with_fast_read_path(
                    config.priority_levels(),
                    per_queue_limit,
                    fast_read_path,
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
            submit_cursor: AtomicUsize::new(0),
            file_table,
            threads,
            #[cfg(feature = "uring")]
            uses_uring,
        })
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
        let id = table.len();
        table.push(FileSlot::Open(file));
        drop(table);
        #[cfg(feature = "uring")]
        self.update_fixed_file(id, fd);
        Ok(id)
    }

    pub fn unregister_file(&self, file: FileId) -> io::Result<()> {
        self.mark_file_closing(file)?;
        for queue in &self.queues {
            queue.wait_until_file_inactive(file);
        }
        self.finish_unregister_file(file)
    }

    pub fn try_unregister_file(&self, file: FileId) -> io::Result<()> {
        self.mark_file_closing(file)?;
        if self.queues.iter().any(|queue| queue.has_active_file(file)) {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("file_id {file} still has active IO"),
            ));
        }
        self.finish_unregister_file(file)
    }

    fn mark_file_closing(&self, file: FileId) -> io::Result<()> {
        let mut table = self.file_table.write().unwrap();
        let Some(slot) = table.get_mut(file) else {
            return Err(invalid_file_id(file));
        };
        match slot {
            FileSlot::Open(handle) => {
                *slot = FileSlot::Closing {
                    _handle: Arc::clone(handle),
                };
                Ok(())
            }
            FileSlot::Closing { .. } => Ok(()),
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
        Ok(())
    }

    pub fn submit(&self, cmd: IoCommand) -> io::Result<IoFuture> {
        let file = cmd.file();
        let queue = self.select_queue(&cmd);
        let table = self.file_table.read().unwrap();
        let handle = registered_file_locked(&table, file)?;
        let future = queue.submit_with_handle(cmd, Some(handle));
        drop(table);
        future
    }

    fn select_queue(&self, cmd: &IoCommand) -> &Arc<QueueCore> {
        if self.queues.len() == 1 {
            return &self.queues[0];
        }

        let shard = match cmd {
            IoCommand::Read {
                file, offset, len, ..
            } if *len != 0 => {
                let block = (*offset / *len as u64) as usize;
                block.wrapping_add(file.wrapping_mul(0x9e37_79b1)) % self.queues.len()
            }
            _ => self.submit_cursor.fetch_add(1, Ordering::Relaxed) % self.queues.len(),
        };
        &self.queues[shard]
    }

    #[cfg(feature = "uring")]
    fn update_fixed_file(&self, file: FileId, fd: RawFd) {
        if self.uses_uring {
            for queue in &self.queues {
                queue.push_fixed_file_update(FixedFileUpdate { file, fd });
            }
        }
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

fn split_limit(total: usize, shards: usize) -> usize {
    let shards = shards.max(1);
    (total / shards + usize::from(total % shards != 0)).max(1)
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
                offset += n as u64;
            }
        }
    }
    Ok(())
}

pub(super) fn uninit_read_buffer(len: usize) -> Box<[MaybeUninit<u8>]> {
    let mut buf = Vec::with_capacity(len);
    buf.resize_with(len, MaybeUninit::uninit);
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
    use std::io::{ErrorKind, Write};

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
                priority_levels: 3,
                queue_shards: 1,
                assume_non_overlapping_reads: false,
            },
            num_workers: 2,
            cpus: None,
        }))
        .expect("create pool")
    }

    #[cfg(feature = "uring")]
    #[test]
    fn fixed_file_update_wakes_empty_queue() {
        let queue = Arc::new(QueueCore::new(1, 1));
        queue.push_fixed_file_update(FixedFileUpdate { file: 7, fd: -1 });

        assert!(matches!(queue.pop_or_notification(0), QueueWake::Notified));

        let mut updates = Vec::new();
        assert_eq!(queue.fixed_file_updates_since(0, &mut updates), 1);
        assert_eq!(updates, vec![FixedFileUpdate { file: 7, fd: -1 }]);
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
    fn sharded_threaded_pool_reads_from_multiple_queues() {
        let path = temp_file("sharded_threaded_reads", b"abcdefghijklmnopqrstuvwxyz");
        let pool = IoPool::new(IoConfig::Threaded(ThreadedConfig {
            base: BaseIoConfig {
                max_in_flight: 8,
                priority_levels: 2,
                queue_shards: 2,
                assume_non_overlapping_reads: false,
            },
            num_workers: 4,
            cpus: None,
        }))
        .expect("create sharded pool");
        let file = pool.register_file(&path).expect("register");

        let reads = (0..8)
            .map(|index| {
                pool.submit(IoCommand::read(file, index * 2, 2, 0))
                    .expect("submit read")
            })
            .collect::<Vec<_>>();
        let mut out = Vec::new();
        for future in reads {
            out.extend_from_slice(&future.blocking_recv_read().expect("read"));
        }
        assert_eq!(&out, b"abcdefghijklmnop");
    }

    #[test]
    fn fast_read_path_tracks_active_reads_without_active_index() {
        let queue = Arc::new(QueueCore::new_with_fast_read_path(2, 8, true));
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

        queue.complete(work.id, Ok(IoOutput::Read(Arc::from(b"rust".to_vec()))));
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
        queue.complete(work.id, Ok(IoOutput::Read(Arc::from(vec![0u8]))));
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

        queue.complete(sync_work.id, Ok(IoOutput::SyncAll));
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
