mod blocking;

#[cfg(all(feature = "uring", target_os = "linux"))]
mod uring;

#[cfg(not(target_os = "linux"))]
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ptr::NonNull;
#[cfg(feature = "profile")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
#[cfg(feature = "profile")]
use std::time::Instant;

use crate::dtype::{OutputDType, OutputValue};
#[cfg(all(feature = "uring", target_os = "linux"))]
use crate::plan::Job;
#[cfg(feature = "profile")]
use crate::plan::JobSide;
use crate::plan::PlanData;
use crate::scatter::{scatter_row_prevalidated, validate_row};
use crate::{Error, IoMode, Result, SessionConfig};
use dyn_blosc::DecodeWorkspace;
use parking_lot::{Condvar, Mutex};

const RUNNING: u8 = 0;
const FAILED: u8 = 1;
const CANCELLED: u8 = 2;
const FINISHED: u8 = 3;
const WINDOW_BROADCAST_THRESHOLD: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Running,
    Failed,
    Cancelled,
    Finished,
}

impl SessionState {
    fn from_raw(raw: u8) -> Self {
        match raw {
            RUNNING => Self::Running,
            FAILED => Self::Failed,
            CANCELLED => Self::Cancelled,
            FINISHED => Self::Finished,
            _ => Self::Failed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeStats {
    pub requested_io_mode: IoMode,
    pub actual_io_mode: IoMode,
    pub worker_count: usize,
    pub max_inflight_jobs_per_worker: usize,
    pub max_inflight_encoded_bytes_per_worker: usize,
    pub max_decoded_bytes_per_worker: usize,
    #[cfg(feature = "profile")]
    pub physical_read_ops: u64,
    #[cfg(feature = "profile")]
    pub physical_read_bytes: u64,
    #[cfg(feature = "profile")]
    pub short_read_retries: u64,
    #[cfg(feature = "profile")]
    pub whole_key_materializations: u64,
    #[cfg(feature = "profile")]
    pub uring_prepared_read_sqes: u64,
    #[cfg(feature = "profile")]
    pub uring_submitted_read_sqes: u64,
    #[cfg(feature = "profile")]
    pub uring_submit_calls: u64,
    #[cfg(feature = "profile")]
    pub uring_cqes: u64,
    #[cfg(feature = "profile")]
    pub uring_cancel_requests: u64,
    #[cfg(feature = "profile")]
    pub uring_cancel_cqes: u64,
    #[cfg(feature = "profile")]
    pub io_wait_nanoseconds: u64,
    #[cfg(feature = "profile")]
    pub decode_nanoseconds: u64,
    #[cfg(feature = "profile")]
    pub data_decode_nanoseconds: u64,
    #[cfg(feature = "profile")]
    pub indices_decode_nanoseconds: u64,
    #[cfg(feature = "profile")]
    pub validation_nanoseconds: u64,
    #[cfg(feature = "profile")]
    pub scatter_nanoseconds: u64,
    #[cfg(feature = "profile")]
    pub scatter_kernel_nanoseconds: u64,
    #[cfg(feature = "profile")]
    pub completion_nanoseconds: u64,
    #[cfg(feature = "profile")]
    pub window_wait_nanoseconds: u64,
    #[cfg(feature = "profile")]
    pub consumer_wait_nanoseconds: u64,
    #[cfg(feature = "profile")]
    pub completed_jobs: u64,
    #[cfg(feature = "profile")]
    pub completed_cells: u64,
    #[cfg(feature = "profile")]
    pub decoded_blocks: u64,
    #[cfg(feature = "profile")]
    pub decoded_bytes: u64,
    #[cfg(feature = "profile")]
    pub claim_cas_retries: u64,
    #[cfg(feature = "profile")]
    pub window_block_events: u64,
    #[cfg(feature = "profile")]
    pub local_full_events: u64,
    #[cfg(feature = "profile")]
    pub peak_inflight_jobs: usize,
    #[cfg(feature = "profile")]
    pub peak_inflight_read_ops: usize,
    #[cfg(feature = "profile")]
    pub peak_inflight_encoded_bytes: usize,
    #[cfg(feature = "profile")]
    pub workers: Vec<WorkerRuntimeStats>,
    pub state: SessionState,
}

/// Per-worker cumulative profile. Timings are wall-clock residency of worker
/// phases, so sums across workers intentionally exceed session wall time.
#[cfg(feature = "profile")]
#[derive(Debug, Clone)]
pub struct WorkerRuntimeStats {
    pub worker_id: usize,
    pub completed_jobs: u64,
    pub completed_cells: u64,
    pub decoded_blocks: u64,
    pub decoded_bytes: u64,
    pub data_decode_nanoseconds: u64,
    pub indices_decode_nanoseconds: u64,
    pub validation_nanoseconds: u64,
    pub scatter_kernel_nanoseconds: u64,
    pub completion_nanoseconds: u64,
    pub window_wait_nanoseconds: u64,
    pub io_wait_nanoseconds: u64,
    pub physical_read_ops: u64,
    pub physical_read_bytes: u64,
    pub short_read_retries: u64,
    pub whole_key_materializations: u64,
    pub claim_cas_retries: u64,
    pub window_block_events: u64,
    pub local_full_events: u64,
    pub uring_prepared_read_sqes: u64,
    pub uring_submitted_read_sqes: u64,
    pub uring_submit_calls: u64,
    pub uring_cqes: u64,
    pub uring_cancel_requests: u64,
    pub uring_cancel_cqes: u64,
}

#[repr(align(64))]
struct CachePadded<T>(T);

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[repr(C, align(64))]
struct ConsumerWait {
    waiting: AtomicBool,
    lock: Mutex<()>,
    condvar: Condvar,
}

impl ConsumerWait {
    fn new() -> Self {
        Self {
            waiting: AtomicBool::new(false),
            lock: Mutex::new(()),
            condvar: Condvar::new(),
        }
    }
}

#[repr(C, align(64))]
struct WindowWait {
    waiters: AtomicUsize,
    lock: Mutex<()>,
    condvar: Condvar,
}

impl WindowWait {
    fn new() -> Self {
        Self {
            waiters: AtomicUsize::new(0),
            lock: Mutex::new(()),
            condvar: Condvar::new(),
        }
    }
}

#[repr(C, align(64))]
pub(crate) struct BatchSlot {
    remaining: AtomicUsize,
    generation: AtomicUsize,
}

pub(crate) struct AlignedBuffer {
    pointer: NonNull<u8>,
    len: usize,
    /// When false, the bytes are borrowed from an external mapping (for example
    /// a shared memfd ring) and this type must not unmap or free them.
    owned: bool,
}

impl AlignedBuffer {
    fn zeroed(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                pointer: NonNull::<CachePadded<()>>::dangling().cast(),
                len,
                owned: true,
            });
        }
        #[cfg(target_os = "linux")]
        {
            use rustix::mm::{Advice, MapFlags, ProtFlags};

            // SAFETY: a null hint requests a fresh anonymous mapping. The
            // mapping is private, writable, page-aligned, and owned solely by
            // this `AlignedBuffer` until its matching `munmap` in `Drop`.
            let pointer = unsafe {
                rustix::mm::mmap_anonymous(
                    std::ptr::null_mut(),
                    len,
                    ProtFlags::READ | ProtFlags::WRITE,
                    MapFlags::PRIVATE | MapFlags::NORESERVE,
                )
            }
            .map_err(|error| Error::Allocation(error.to_string()))?;
            let pointer = NonNull::new(pointer.cast::<u8>())
                .ok_or_else(|| Error::Allocation("mmap returned a null pointer".into()))?;
            // SAFETY: `pointer..pointer+len` is the live anonymous mapping just
            // created above. Transparent huge pages preserve byte semantics;
            // this is a best-effort placement hint, so kernel policy may ignore
            // it without changing correctness.
            let _ =
                unsafe { rustix::mm::madvise(pointer.as_ptr().cast(), len, Advice::LinuxHugepage) };
            Ok(Self {
                pointer,
                len,
                owned: true,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let layout = Layout::from_size_align(len, 64)
                .map_err(|error| Error::Allocation(error.to_string()))?;
            // SAFETY: `layout` has non-zero size and valid 64-byte alignment.
            // Zeroing once makes untouched row padding and fresh zero-filled
            // output rows initialized without repeated per-row stores.
            let pointer = unsafe { alloc_zeroed(layout) };
            let pointer = NonNull::new(pointer)
                .ok_or_else(|| Error::Allocation(format!("failed to allocate {len} bytes")))?;
            Ok(Self {
                pointer,
                len,
                owned: true,
            })
        }
    }

    /// Borrow an externally owned output ring. The caller retains mapping
    /// ownership and must outlive every session that uses this buffer.
    pub(crate) fn from_shared(pointer: NonNull<u8>, len: usize) -> Self {
        if len == 0 {
            return Self {
                pointer: NonNull::<CachePadded<()>>::dangling().cast(),
                len,
                owned: false,
            };
        }
        Self {
            pointer,
            len,
            owned: false,
        }
    }

    unsafe fn slice(&self, offset: usize, len: usize) -> &[u8] {
        debug_assert!(offset <= self.len && len <= self.len - offset);
        // SAFETY: the caller proves the requested range is initialized, within
        // this allocation, and not concurrently written for the borrow.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr().add(offset), len) }
    }

    unsafe fn row_pointer(&self, offset: usize, len: usize) -> *mut u8 {
        debug_assert!(offset <= self.len && len <= self.len - offset);
        // SAFETY: the checked offset lies within this allocation. Dereferencing
        // and aliasing requirements remain with the caller.
        unsafe { self.pointer.as_ptr().add(offset) }
    }
}

// SAFETY: access is synchronized by static row ownership, per-slot generation,
// release/acquire completion counters, and the single-consumer lease protocol.
unsafe impl Send for AlignedBuffer {}
// SAFETY: see the `Send` implementation; no unsynchronized shared references
// are constructed while worker writes are possible.
unsafe impl Sync for AlignedBuffer {}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.len == 0 || !self.owned {
            return;
        }
        #[cfg(target_os = "linux")]
        {
            // SAFETY: Linux construction always obtains this exact pointer and
            // length from `mmap_anonymous`; all worker references are gone
            // before `SessionInner` and this buffer are dropped.
            let _ = unsafe { rustix::mm::munmap(self.pointer.as_ptr().cast(), self.len) };
        }
        #[cfg(not(target_os = "linux"))]
        {
            let layout =
                Layout::from_size_align(self.len, 64).expect("allocation layout stays valid");
            // SAFETY: this pointer was allocated with the identical layout and
            // has not been deallocated or moved.
            unsafe { dealloc(self.pointer.as_ptr(), layout) };
        }
    }
}

pub struct Session {
    inner: Arc<SessionInner>,
    workers: Vec<JoinHandle<()>>,
}

#[derive(Clone)]
pub struct CancellationHandle {
    inner: Arc<SessionInner>,
}

pub struct Batch<'a> {
    session: &'a mut Session,
    logical_batch: usize,
    rows: usize,
    released: bool,
}

impl Session {
    pub(crate) fn start(plan: Arc<PlanData>, config: SessionConfig) -> Result<Self> {
        let output = AlignedBuffer::zeroed(plan.stats.output_ring_bytes)?;
        Self::start_with_output(plan, config, output)
    }

    pub(crate) fn start_with_output(
        plan: Arc<PlanData>,
        config: SessionConfig,
        output: AlignedBuffer,
    ) -> Result<Self> {
        config.validate()?;
        let requested = config.io_mode;
        let selected = choose_io_mode(&plan, requested)?;
        #[cfg(all(feature = "uring", target_os = "linux"))]
        let mut actual = selected;
        #[cfg(not(all(feature = "uring", target_os = "linux")))]
        let actual = selected;
        #[cfg(all(feature = "uring", target_os = "linux"))]
        let mut prepared_rings = Vec::new();
        #[cfg(all(feature = "uring", target_os = "linux"))]
        if let IoMode::Uring { queue_depth } = actual {
            prepared_rings.try_reserve_exact(config.worker_count)?;
            for _ in 0..config.worker_count {
                match io_uring::IoUring::new(queue_depth) {
                    Ok(ring) => prepared_rings.push(ring),
                    Err(_error) if matches!(requested, IoMode::Auto { .. }) => {
                        prepared_rings.clear();
                        actual = IoMode::Blocking;
                        break;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }
        validate_worker_capacity(&plan, &config, actual)?;
        if output.len != plan.stats.output_ring_bytes {
            return Err(Error::Invariant(format!(
                "output ring length {} does not match plan requirement {}",
                output.len, plan.stats.output_ring_bytes
            )));
        }
        let slots = plan.ring_slots;
        let mut batch_slots = Vec::new();
        batch_slots.try_reserve_exact(slots)?;
        for slot in 0..slots {
            if slot < plan.batch_count {
                batch_slots.push(BatchSlot {
                    remaining: AtomicUsize::new(batch_len(&plan, slot)),
                    generation: AtomicUsize::new(slot),
                });
            } else {
                batch_slots.push(BatchSlot {
                    remaining: AtomicUsize::new(0),
                    generation: AtomicUsize::new(usize::MAX),
                });
            }
        }
        let initial_state = if plan.batch_count == 0 {
            FINISHED
        } else {
            RUNNING
        };
        let inner = Arc::new(SessionInner {
            plan,
            output,
            batch_slots,
            consume_idx: CachePadded(AtomicUsize::new(0)),
            next_job: CachePadded(AtomicUsize::new(0)),
            state: CachePadded(AtomicU8::new(initial_state)),
            first_error: Mutex::new(None),
            consumer_wait: ConsumerWait::new(),
            window_wait: WindowWait::new(),
            stats: AtomicRuntimeStats::new(requested, actual, &config),
        });
        let mut workers = Vec::new();
        if initial_state == RUNNING {
            workers.try_reserve_exact(config.worker_count)?;
            match actual {
                IoMode::Blocking => {
                    for worker_id in 0..config.worker_count {
                        let worker = Arc::clone(&inner);
                        let spawned = std::thread::Builder::new()
                            .name(format!("sc-load-blocking-{worker_id}"))
                            .spawn(move || {
                                worker_entry(worker, |inner| blocking::run_worker(inner, worker_id))
                            });
                        match spawned {
                            Ok(worker) => workers.push(worker),
                            Err(error) => {
                                stop_started_workers(&inner, &mut workers);
                                return Err(error.into());
                            }
                        }
                    }
                }
                IoMode::Uring { .. } => {
                    #[cfg(all(feature = "uring", target_os = "linux"))]
                    {
                        debug_assert_eq!(prepared_rings.len(), config.worker_count);
                        for (worker_id, ring) in prepared_rings.into_iter().enumerate() {
                            let worker = Arc::clone(&inner);
                            let worker_config = config.clone();
                            let spawned = std::thread::Builder::new()
                                .name(format!("sc-load-uring-{worker_id}"))
                                .spawn(move || {
                                    worker_entry(worker, |inner| {
                                        uring::run_worker(inner, ring, worker_config, worker_id)
                                    })
                                });
                            match spawned {
                                Ok(worker) => workers.push(worker),
                                Err(error) => {
                                    stop_started_workers(&inner, &mut workers);
                                    return Err(error.into());
                                }
                            }
                        }
                    }
                    #[cfg(not(all(feature = "uring", target_os = "linux")))]
                    unreachable!("choose_io_mode rejects unavailable io_uring");
                }
                IoMode::Auto { .. } => unreachable!("actual mode is never Auto"),
            }
        }
        Ok(Self { inner, workers })
    }

    pub fn state(&self) -> SessionState {
        SessionState::from_raw(self.inner.state.load(Ordering::Acquire))
    }

    pub fn stats(&self) -> RuntimeStats {
        self.inner.stats.snapshot(self.state())
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }

    pub fn cancellation_handle(&self) -> CancellationHandle {
        CancellationHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Wait up to `timeout` for logical batch `logical` to complete in the ring.
    ///
    /// Used by the shared-ring producer. Does not advance `consume_idx` and does
    /// not require exclusive `Batch` ownership of prior leases. A bounded wait
    /// lets the shared control plane observe cross-process cancellation even
    /// while session workers are still active.
    pub(crate) fn wait_ready_for(
        &self,
        logical: usize,
        timeout: std::time::Duration,
    ) -> Result<bool> {
        let plan = &self.inner.plan;
        if logical >= plan.batch_count {
            return Err(Error::InvalidInput(format!(
                "logical batch {logical} is outside plan batch_count {}",
                plan.batch_count
            )));
        }
        let slot = ring_slot(plan, logical);
        match self.state() {
            SessionState::Failed => return Err(self.inner.execution_error()),
            SessionState::Cancelled => return Err(Error::Cancelled),
            SessionState::Finished => {
                return Err(Error::Invariant(
                    "session finished before shared wait_ready".into(),
                ))
            }
            SessionState::Running => {}
        }
        let generation = self.inner.batch_slots[slot]
            .generation
            .load(Ordering::Acquire);
        if generation != logical {
            return Err(Error::Invariant(format!(
                "ring slot {slot} has generation {generation}, expected {logical}"
            )));
        }
        if self.inner.batch_slots[slot]
            .remaining
            .load(Ordering::Acquire)
            == 0
        {
            return Ok(true);
        }
        let mut guard = self.inner.consumer_wait.lock.lock();
        let _ = self
            .inner
            .consumer_wait
            .waiting
            .swap(true, Ordering::AcqRel);
        let result = (|| {
            match self.state() {
                SessionState::Failed => return Err(self.inner.execution_error()),
                SessionState::Cancelled => return Err(Error::Cancelled),
                SessionState::Finished => {
                    return Err(Error::Invariant(
                        "session finished before shared wait_ready".into(),
                    ))
                }
                SessionState::Running => {}
            }
            let generation = self.inner.batch_slots[slot]
                .generation
                .load(Ordering::Acquire);
            if generation != logical {
                return Err(Error::Invariant(format!(
                    "ring slot {slot} has generation {generation}, expected {logical}"
                )));
            }
            if self.inner.batch_slots[slot]
                .remaining
                .load(Ordering::Acquire)
                == 0
            {
                return Ok(true);
            }
            self.inner
                .consumer_wait
                .condvar
                .wait_for(&mut guard, timeout);
            match self.state() {
                SessionState::Failed => return Err(self.inner.execution_error()),
                SessionState::Cancelled => return Err(Error::Cancelled),
                SessionState::Finished => {
                    return Err(Error::Invariant(
                        "session finished before shared wait_ready".into(),
                    ))
                }
                SessionState::Running => {}
            }
            let generation = self.inner.batch_slots[slot]
                .generation
                .load(Ordering::Acquire);
            if generation != logical {
                return Err(Error::Invariant(format!(
                    "ring slot {slot} has generation {generation}, expected {logical}"
                )));
            }
            Ok(self.inner.batch_slots[slot]
                .remaining
                .load(Ordering::Acquire)
                == 0)
        })();
        self.inner
            .consumer_wait
            .waiting
            .store(false, Ordering::Release);
        drop(guard);
        result
    }

    pub(crate) fn consume_idx(&self) -> usize {
        self.inner.consume_idx.load(Ordering::Acquire)
    }

    pub(crate) fn terminal_error(&self) -> Error {
        match self.state() {
            SessionState::Failed => self.inner.execution_error(),
            SessionState::Cancelled => Error::Cancelled,
            state => Error::Invariant(format!(
                "requested a terminal session error while state is {state:?}"
            )),
        }
    }

    pub(crate) fn batch_count(&self) -> usize {
        self.inner.plan.batch_count
    }

    /// Commit a shared-path release for `logical`.
    ///
    /// Caller must ensure releases are applied in order (`logical == consume_idx`).
    pub(crate) fn commit_release(&self, logical: usize) -> Result<()> {
        let expected = self.inner.consume_idx.load(Ordering::Acquire);
        if logical != expected {
            return Err(Error::Invariant(format!(
                "shared commit_release expected logical {expected}, got {logical}"
            )));
        }
        self.release_batch_inner(logical);
        Ok(())
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch<'_>>> {
        let logical = self.inner.consume_idx.load(Ordering::Acquire);
        if logical >= self.inner.plan.batch_count {
            return match self.state() {
                SessionState::Failed => Err(self.inner.execution_error()),
                SessionState::Cancelled => Err(Error::Cancelled),
                _ => Ok(None),
            };
        }
        let slot = ring_slot(&self.inner.plan, logical);
        match self.state() {
            SessionState::Failed => return Err(self.inner.execution_error()),
            SessionState::Cancelled => return Err(Error::Cancelled),
            SessionState::Finished => return Ok(None),
            SessionState::Running => {}
        }
        let generation = self.inner.batch_slots[slot]
            .generation
            .load(Ordering::Acquire);
        if generation != logical {
            return Err(Error::Invariant(format!(
                "ring slot {slot} has generation {generation}, expected {logical}"
            )));
        }
        if self.inner.batch_slots[slot]
            .remaining
            .load(Ordering::Acquire)
            == 0
        {
            return Ok(Some(Batch {
                rows: batch_len(&self.inner.plan, logical),
                session: self,
                logical_batch: logical,
                released: false,
            }));
        }
        #[cfg(feature = "profile")]
        let wait_start = Instant::now();
        let mut guard = self.inner.consumer_wait.lock.lock();
        let ready = loop {
            let _ = self
                .inner
                .consumer_wait
                .waiting
                .swap(true, Ordering::AcqRel);
            match self.state() {
                SessionState::Failed => break Err(self.inner.execution_error()),
                SessionState::Cancelled => break Err(Error::Cancelled),
                SessionState::Finished => break Ok(false),
                SessionState::Running => {}
            }
            let generation = self.inner.batch_slots[slot]
                .generation
                .load(Ordering::Acquire);
            if generation != logical {
                break Err(Error::Invariant(format!(
                    "ring slot {slot} has generation {generation}, expected {logical}"
                )));
            }
            if self.inner.batch_slots[slot]
                .remaining
                .load(Ordering::Acquire)
                == 0
            {
                break Ok(true);
            }
            self.inner.consumer_wait.condvar.wait(&mut guard);
        };
        self.inner
            .consumer_wait
            .waiting
            .store(false, Ordering::Release);
        drop(guard);
        #[cfg(feature = "profile")]
        self.inner
            .stats
            .consumer_wait_ns
            .fetch_add(elapsed_ns(wait_start), Ordering::Relaxed);
        if !ready? {
            return Ok(None);
        }
        Ok(Some(Batch {
            rows: batch_len(&self.inner.plan, logical),
            session: self,
            logical_batch: logical,
            released: false,
        }))
    }

    fn release_batch(&mut self, logical: usize) {
        self.release_batch_inner(logical);
    }

    fn release_batch_inner(&self, logical: usize) {
        let plan = &self.inner.plan;
        let slot = ring_slot(plan, logical);
        let next_generation = logical + plan.ring_slots;
        if next_generation < plan.batch_count {
            self.inner.batch_slots[slot]
                .remaining
                .store(batch_len(plan, next_generation), Ordering::Relaxed);
            self.inner.batch_slots[slot]
                .generation
                .store(next_generation, Ordering::Release);
        }
        let next = logical + 1;
        self.inner.consume_idx.store(next, Ordering::Release);
        if next == plan.batch_count {
            let _ = self.inner.state.compare_exchange(
                RUNNING,
                FINISHED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            self.inner.wake_all_window_workers();
        } else {
            self.inner.wake_window_workers_for_progress();
        }
    }
}

impl CancellationHandle {
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    pub fn state(&self) -> SessionState {
        SessionState::from_raw(self.inner.state.load(Ordering::Acquire))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.inner.cancel();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Batch<'_> {
    pub fn logical_batch(&self) -> usize {
        self.logical_batch
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn n_cols(&self) -> usize {
        self.session.inner.plan.output.n_cols
    }

    pub fn dtype(&self) -> OutputDType {
        self.session.inner.plan.output.dtype
    }

    pub fn row_stride_bytes(&self) -> usize {
        self.session.inner.plan.row_stride
    }

    /// Padded row-major bytes. Use [`Self::row`] to omit cache-line padding.
    pub fn bytes(&self) -> &[u8] {
        let plan = &self.session.inner.plan;
        let slot = ring_slot(plan, self.logical_batch);
        let offset = slot * plan.batch_size * plan.row_stride;
        let len = self.rows * plan.row_stride;
        // SAFETY: a lease is returned only after the release/acquire completion
        // handshake, and the mutable borrow of `Session` prevents a second
        // consumer lease. The slot is not reusable until this lease drops.
        unsafe { self.session.inner.output.slice(offset, len) }
    }

    pub fn row(&self, row: usize) -> Option<&[u8]> {
        if row >= self.rows {
            return None;
        }
        let row_bytes = self.n_cols().checked_mul(self.dtype().size())?;
        let start = row.checked_mul(self.row_stride_bytes())?;
        self.bytes().get(start..start + row_bytes)
    }

    /// Typed view of a single logical row (exactly `n_cols` elements, no padding).
    pub fn row_as<T: OutputValue>(&self, row: usize) -> Result<&[T]> {
        if T::DTYPE != self.dtype() {
            return Err(Error::InvalidInput(format!(
                "requested {} view but batch dtype is {}",
                T::DTYPE,
                self.dtype()
            )));
        }
        let bytes = self
            .row(row)
            .ok_or_else(|| Error::InvalidInput(format!("row {row} out of range")))?;
        #[cfg(target_endian = "big")]
        return Err(Error::Unsupported(
            "typed batch views require a little-endian target; use row bytes".into(),
        ));
        #[cfg(target_endian = "little")]
        {
            if std::mem::size_of::<T>() != self.dtype().size()
                || std::mem::align_of::<T>() > 64
                || bytes.as_ptr().align_offset(std::mem::align_of::<T>()) != 0
            {
                return Err(Error::Invariant(
                    "sealed output type layout does not match batch dtype".into(),
                ));
            }
            let len = bytes.len() / std::mem::size_of::<T>();
            // SAFETY: OutputValue is sealed to primitive numeric types, the
            // layout and alignment were checked, and this target is little-endian.
            Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len) })
        }
    }

    /// Contiguous logical values. Padded multi-row batches use [`Self::row_as`]
    /// or [`Self::as_padded_slice`] instead.
    pub fn as_slice<T: OutputValue>(&self) -> Result<&[T]> {
        if self.rows == 1 {
            return self.row_as(0);
        }
        let row_bytes = self
            .n_cols()
            .checked_mul(self.dtype().size())
            .ok_or_else(|| Error::Invariant("logical row byte length overflow".into()))?;
        if self.row_stride_bytes() != row_bytes {
            return Err(Error::Unsupported(
                "batch rows are cache-line padded; use row_as or as_padded_slice".into(),
            ));
        }
        self.as_padded_slice()
    }

    /// Full ring-slot storage, including zero-initialized row padding.
    pub fn as_padded_slice<T: OutputValue>(&self) -> Result<&[T]> {
        if T::DTYPE != self.dtype() {
            return Err(Error::InvalidInput(format!(
                "requested {} view but batch dtype is {}",
                T::DTYPE,
                self.dtype()
            )));
        }
        #[cfg(target_endian = "big")]
        return Err(Error::Unsupported(
            "typed batch views require a little-endian target; use bytes".into(),
        ));
        #[cfg(target_endian = "little")]
        {
            let bytes = self.bytes();
            if std::mem::size_of::<T>() != self.dtype().size()
                || !bytes.len().is_multiple_of(std::mem::size_of::<T>())
                || bytes.as_ptr().align_offset(std::mem::align_of::<T>()) != 0
            {
                return Err(Error::Invariant(
                    "batch byte length not aligned to element type".into(),
                ));
            }
            let len = bytes.len() / std::mem::size_of::<T>();
            // SAFETY: OutputValue is sealed, the layout was checked, and every
            // logical byte and padding byte is initialized before publication.
            Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len) })
        }
    }

    pub fn release(mut self) {
        self.release_inner();
    }

    fn release_inner(&mut self) {
        if !self.released {
            self.session.release_batch(self.logical_batch);
            self.released = true;
        }
    }
}

impl Drop for Batch<'_> {
    fn drop(&mut self) {
        self.release_inner();
    }
}

pub(crate) struct SessionInner {
    pub plan: Arc<PlanData>,
    output: AlignedBuffer,
    batch_slots: Vec<BatchSlot>,
    consume_idx: CachePadded<AtomicUsize>,
    next_job: CachePadded<AtomicUsize>,
    state: CachePadded<AtomicU8>,
    first_error: Mutex<Option<Arc<Error>>>,
    consumer_wait: ConsumerWait,
    window_wait: WindowWait,
    stats: AtomicRuntimeStats,
}

impl SessionInner {
    fn is_running(&self) -> bool {
        self.state.load(Ordering::Acquire) == RUNNING
    }

    fn fail(&self, error: Error) {
        let mut first_error = self.first_error.lock();
        if self.state.load(Ordering::Acquire) != RUNNING {
            return;
        }
        *first_error = Some(Arc::new(error));
        if self
            .state
            .compare_exchange(RUNNING, FAILED, Ordering::Release, Ordering::Acquire)
            .is_ok()
        {
            drop(first_error);
            self.wake_all_waiters();
        }
    }

    fn cancel(&self) {
        if self
            .state
            .compare_exchange(RUNNING, CANCELLED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.wake_all_waiters();
        }
    }

    fn execution_error(&self) -> Error {
        self.first_error
            .lock()
            .clone()
            .map(Error::Session)
            .unwrap_or_else(|| {
                Error::Session(Arc::new(Error::Invariant("missing first error".into())))
            })
    }

    #[cfg(all(feature = "uring", target_os = "linux"))]
    fn claim_job(&self, _worker_id: usize, capacity: impl Fn(&Job) -> bool) -> Claim {
        loop {
            if !self.is_running() {
                return Claim::Stopped;
            }
            let next = self.next_job.load(Ordering::Relaxed);
            let Some(job) = self.plan.jobs.get(next) else {
                return Claim::Exhausted;
            };
            debug_assert!(job.batch_min <= job.batch_max);
            debug_assert_eq!(
                job.start_step,
                (job.batch_max + 1).saturating_sub(self.plan.prefetch_step)
            );
            if self.consume_idx.load(Ordering::Acquire) < job.start_step {
                #[cfg(feature = "profile")]
                self.worker_stats(_worker_id)
                    .window_blocks
                    .fetch_add(1, Ordering::Relaxed);
                return Claim::WindowBlocked;
            }
            if !capacity(job) {
                #[cfg(feature = "profile")]
                self.worker_stats(_worker_id)
                    .local_full
                    .fetch_add(1, Ordering::Relaxed);
                return Claim::LocalFull;
            }
            if self
                .next_job
                .compare_exchange_weak(next, next + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                self.wake_window_workers_if_claimable();
                return Claim::Claimed(next);
            }
            #[cfg(feature = "profile")]
            self.worker_stats(_worker_id)
                .claim_retries
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn claim_blocking_jobs(&self, _worker_id: usize, maximum: usize) -> RangeClaim {
        debug_assert!(maximum > 0);
        loop {
            if !self.is_running() {
                return RangeClaim::Stopped;
            }
            let next = self.next_job.load(Ordering::Relaxed);
            let Some(job) = self.plan.jobs.get(next) else {
                return RangeClaim::Exhausted;
            };
            let consumed = self.consume_idx.load(Ordering::Acquire);
            if consumed < job.start_step {
                #[cfg(feature = "profile")]
                self.worker_stats(_worker_id)
                    .window_blocks
                    .fetch_add(1, Ordering::Relaxed);
                return RangeClaim::WindowBlocked;
            }
            let limit = next.saturating_add(maximum).min(self.plan.jobs.len());
            let mut end = next + 1;
            while end < limit && self.plan.jobs[end].start_step <= consumed {
                end += 1;
            }
            if self
                .next_job
                .compare_exchange_weak(next, end, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                self.wake_window_workers_if_claimable();
                return RangeClaim::Claimed(next..end);
            }
            #[cfg(feature = "profile")]
            self.worker_stats(_worker_id)
                .claim_retries
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn wait_for_window(&self, _worker_id: usize) {
        #[cfg(feature = "profile")]
        let started = Instant::now();
        let mut guard = self.window_wait.lock.lock();
        self.window_wait.waiters.fetch_add(1, Ordering::AcqRel);
        while self.is_running()
            && self
                .plan
                .jobs
                .get(self.next_job.load(Ordering::Relaxed))
                .is_some_and(|job| self.consume_idx.load(Ordering::Acquire) < job.start_step)
        {
            self.window_wait.condvar.wait(&mut guard);
        }
        let previous = self.window_wait.waiters.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        drop(guard);
        #[cfg(feature = "profile")]
        self.worker_stats(_worker_id)
            .window_wait_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
    }

    fn wake_consumer(&self) {
        if self.consumer_wait.waiting.swap(false, Ordering::AcqRel) {
            let _guard = self.consumer_wait.lock.lock();
            self.consumer_wait.condvar.notify_one();
        }
    }

    fn wake_window_workers_if_claimable(&self) {
        let next = self.next_job.load(Ordering::Relaxed);
        let consumed = self.consume_idx.load(Ordering::Acquire);
        if self
            .plan
            .jobs
            .get(next)
            .is_some_and(|job| job.start_step <= consumed)
        {
            self.wake_window_workers_for_progress();
        }
    }

    fn wake_window_workers_for_progress(&self) {
        // The zero-valued RMW completes the registration handshake even when no
        // worker is currently asleep. A worker registering immediately after it
        // therefore observes the release that made its window eligible.
        let waiters = self.window_wait.waiters.fetch_add(0, Ordering::AcqRel);
        if waiters > 0 {
            let _guard = self.window_wait.lock.lock();
            if waiters <= WINDOW_BROADCAST_THRESHOLD {
                self.window_wait.condvar.notify_all();
            } else {
                self.window_wait.condvar.notify_one();
            }
        }
    }

    fn wake_all_window_workers(&self) {
        if self.window_wait.waiters.fetch_add(0, Ordering::AcqRel) > 0 {
            let _guard = self.window_wait.lock.lock();
            self.window_wait.condvar.notify_all();
        }
    }

    fn wake_all_waiters(&self) {
        self.wake_consumer();
        self.wake_all_window_workers();
    }

    fn decode_and_commit(
        &self,
        job_idx: usize,
        data_encoded: &[u8],
        indices_encoded: Option<&[u8]>,
        scratch: &mut WorkerScratch,
        _worker_id: usize,
    ) -> Result<()> {
        let job = self
            .plan
            .jobs
            .get(job_idx)
            .ok_or_else(|| Error::Invariant("claimed job is missing".into()))?;
        let source_plan = self
            .plan
            .source_plans
            .get(job.source_plan as usize)
            .ok_or_else(|| Error::Invariant("job source plan is missing".into()))?;
        let groups = self
            .plan
            .groups
            .get(job.groups.clone())
            .ok_or_else(|| Error::Invariant("job block-group arena range is invalid".into()))?;
        for group in groups {
            if let Some(block) = group.data_block() {
                #[cfg(feature = "profile")]
                let decode_started = Instant::now();
                decode_block(
                    &self.plan,
                    block,
                    data_encoded,
                    &mut scratch.data_decoded,
                    &mut scratch.workspace,
                    &self.output,
                )?;
                #[cfg(feature = "profile")]
                self.worker_stats(_worker_id)
                    .data_decode_ns
                    .fetch_add(elapsed_ns(decode_started), Ordering::Relaxed);
            } else {
                scratch.data_decoded.set_empty();
            }
            if let Some(block) = group.indices_block() {
                #[cfg(feature = "profile")]
                let indices_started = Instant::now();
                decode_block(
                    &self.plan,
                    block,
                    indices_encoded.ok_or_else(|| {
                        Error::Invariant("CSR job is missing encoded indices".into())
                    })?,
                    &mut scratch.indices_decoded,
                    &mut scratch.workspace,
                    &self.output,
                )?;
                #[cfg(feature = "profile")]
                self.worker_stats(_worker_id)
                    .indices_decode_ns
                    .fetch_add(elapsed_ns(indices_started), Ordering::Relaxed);
            } else {
                scratch.indices_decoded.set_empty();
            }

            let data_decoded = scratch.data_decoded.as_slice();
            let indices_decoded = scratch.indices_decoded.as_slice();
            let group_tasks = self.plan.cells.get(group.cells.clone()).ok_or_else(|| {
                Error::Invariant("block group cell arena range is invalid".into())
            })?;
            #[cfg(feature = "profile")]
            let validation_started = Instant::now();
            if source_plan.requires_runtime_validation() {
                for task in group_tasks {
                    validate_row(source_plan, task, data_decoded, indices_decoded)?;
                }
            }
            #[cfg(feature = "profile")]
            self.worker_stats(_worker_id)
                .validation_ns
                .fetch_add(elapsed_ns(validation_started), Ordering::Relaxed);
            if !self.is_running() {
                return Ok(());
            }

            #[cfg(feature = "profile")]
            let scatter_started = Instant::now();
            let fill = self.plan.fill;
            let row_bytes = self.plan.row_bytes;
            for task in group_tasks {
                if task.is_direct_decode() {
                    continue;
                }
                let row_offset = task.row_offset();
                // SAFETY: each task owns one unique output ordinal, and the
                // job window keeps this ring generation writable and unleased.
                let row = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.output.row_pointer(row_offset, self.plan.row_stride),
                        self.plan.row_stride,
                    )
                };
                // SAFETY: this block group's validation covers the exact
                // scratch buffers and tasks used below. Compiler-sealed dense
                // infallible paths require no data-dependent validation.
                unsafe {
                    scatter_row_prevalidated(
                        source_plan,
                        task,
                        data_decoded,
                        indices_decoded,
                        row,
                        row_bytes,
                        fill,
                    )?;
                }
            }
            #[cfg(feature = "profile")]
            self.worker_stats(_worker_id)
                .scatter_kernel_ns
                .fetch_add(elapsed_ns(scatter_started), Ordering::Relaxed);
        }

        let completions = self
            .plan
            .completions
            .get(job.completions.clone())
            .ok_or_else(|| Error::Invariant("job completion arena range is invalid".into()))?;
        #[cfg(feature = "profile")]
        let completion_started = Instant::now();
        let mut batch_became_ready = false;
        #[cfg(feature = "profile")]
        let mut completed_cells = 0usize;
        for completion in completions {
            let ring_batch = completion.ring_batch();
            let completed = completion.completed();
            let counter = &self
                .batch_slots
                .get(ring_batch)
                .ok_or_else(|| Error::Invariant("job completion ring slot is invalid".into()))?
                .remaining;
            let previous = counter.fetch_sub(completed, Ordering::Release);
            if previous < completed {
                return Err(Error::Invariant(format!(
                    "batch completion counter underflow in ring slot {ring_batch}"
                )));
            }
            batch_became_ready |= previous == completed;
            #[cfg(feature = "profile")]
            {
                completed_cells = completed_cells.saturating_add(completed);
            }
        }
        #[cfg(feature = "profile")]
        {
            let stats = self.worker_stats(_worker_id);
            stats
                .completion_ns
                .fetch_add(elapsed_ns(completion_started), Ordering::Relaxed);
            stats.jobs.fetch_add(1, Ordering::Relaxed);
            stats.cells.fetch_add(
                u64::try_from(completed_cells).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            let block_count =
                job.data.blocks.len() + job.indices.as_ref().map_or(0, |side| side.blocks.len());
            let decoded_side_bytes = |side: &JobSide| {
                self.plan.blocks[side.blocks.clone()]
                    .iter()
                    .map(|block| block.decoded_len())
                    .fold(0usize, usize::saturating_add)
            };
            let decoded_bytes = decoded_side_bytes(&job.data)
                .saturating_add(job.indices.as_ref().map_or(0, decoded_side_bytes));
            stats
                .blocks
                .fetch_add(block_count as u64, Ordering::Relaxed);
            stats
                .decoded_bytes
                .fetch_add(decoded_bytes as u64, Ordering::Relaxed);
        }
        if batch_became_ready {
            self.wake_consumer();
        }
        Ok(())
    }

    #[cfg(feature = "profile")]
    #[inline(always)]
    pub(super) fn worker_stats(&self, worker_id: usize) -> &WorkerStats {
        debug_assert!(worker_id < self.stats.worker_stats.len());
        // SAFETY: worker IDs are assigned exactly from
        // `0..config.worker_count` when threads are spawned.
        unsafe { self.stats.worker_stats.get_unchecked(worker_id) }
    }
}

#[cfg(all(feature = "uring", target_os = "linux"))]
enum Claim {
    Claimed(usize),
    Exhausted,
    WindowBlocked,
    LocalFull,
    Stopped,
}

enum RangeClaim {
    Claimed(std::ops::Range<usize>),
    Exhausted,
    WindowBlocked,
    Stopped,
}

pub(crate) struct WorkerScratch {
    data_decoded: DecodedBuffer,
    indices_decoded: DecodedBuffer,
    workspace: DecodeWorkspace,
}

impl WorkerScratch {
    pub(crate) fn new() -> Self {
        Self {
            data_decoded: DecodedBuffer::new(),
            indices_decoded: DecodedBuffer::new(),
            workspace: DecodeWorkspace::new(),
        }
    }
}

/// Reusable decoder destination whose capacity is grown without writing bytes
/// that the decoder will immediately overwrite.
struct DecodedBuffer {
    bytes: Vec<MaybeUninit<u8>>,
    initialized: bool,
}

impl DecodedBuffer {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            initialized: true,
        }
    }

    fn prepare(&mut self, len: usize) -> Result<()> {
        self.bytes.clear();
        self.bytes.try_reserve_exact(len)?;
        // SAFETY: the vector has at least `len` capacity after the successful
        // reserve. Every bit pattern is valid for `MaybeUninit<u8>`, and no
        // byte is exposed as initialized until `finish` is called.
        unsafe { self.bytes.set_len(len) };
        self.initialized = len == 0;
        Ok(())
    }

    fn set_empty(&mut self) {
        self.bytes.clear();
        self.initialized = true;
    }

    fn finish(&mut self) {
        self.initialized = true;
    }

    fn as_slice(&self) -> &[u8] {
        assert!(
            self.initialized,
            "decoded bytes must be fully initialized before access"
        );
        // SAFETY: `finish` is reached only after every byte in the contiguous
        // compiler-built decoded extent was written successfully. u8 has no
        // invalid bit patterns and the returned borrow cannot outlive `self`.
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), self.bytes.len()) }
    }

    unsafe fn range_mut(&mut self, range: std::ops::Range<usize>) -> &mut [u8] {
        debug_assert!(range.start <= range.end && range.end <= self.bytes.len());
        // SAFETY: the caller proves the range is contained in the allocation
        // and does not retain overlapping mutable ranges. The decoder owns and
        // initializes the entire returned range before the borrow ends.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.bytes.as_mut_ptr().add(range.start).cast(),
                range.end - range.start,
            )
        }
    }
}

fn worker_entry<F>(inner: Arc<SessionInner>, run: F)
where
    F: FnOnce(Arc<SessionInner>) -> Result<()>,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(Arc::clone(&inner)))) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => inner.fail(error),
        Err(_) => inner.fail(Error::WorkerPanic),
    }
}

fn stop_started_workers(inner: &SessionInner, workers: &mut Vec<JoinHandle<()>>) {
    inner.cancel();
    for worker in workers.drain(..) {
        let _ = worker.join();
    }
}

fn choose_io_mode(plan: &PlanData, requested: IoMode) -> Result<IoMode> {
    match requested {
        IoMode::Blocking => Ok(IoMode::Blocking),
        IoMode::Uring {
            queue_depth: _queue_depth,
        } => {
            if !plan.runtime.all_positioned {
                return Err(Error::Unsupported(
                    "explicit io_uring cannot execute key-backed sources".into(),
                ));
            }
            #[cfg(all(feature = "uring", target_os = "linux"))]
            return Ok(IoMode::Uring {
                queue_depth: _queue_depth,
            });
            #[cfg(not(all(feature = "uring", target_os = "linux")))]
            return Err(Error::Unsupported(
                "io_uring requires Linux and the `uring` feature".into(),
            ));
        }
        IoMode::Auto {
            queue_depth: _queue_depth,
        } => {
            if !plan.runtime.all_positioned || plan.runtime.has_fuse_source {
                return Ok(IoMode::Blocking);
            }
            #[cfg(all(feature = "uring", target_os = "linux"))]
            return Ok(IoMode::Uring {
                queue_depth: _queue_depth,
            });
            #[cfg(not(all(feature = "uring", target_os = "linux")))]
            return Ok(IoMode::Blocking);
        }
    }
}

fn batch_len(plan: &PlanData, logical: usize) -> usize {
    let start = logical * plan.batch_size;
    plan.stats
        .input_rows
        .saturating_sub(start)
        .min(plan.batch_size)
}

#[inline(always)]
fn ring_slot(plan: &PlanData, logical: usize) -> usize {
    if plan.ring_mask != usize::MAX {
        logical & plan.ring_mask
    } else {
        logical % plan.ring_slots
    }
}

fn validate_worker_capacity(
    plan: &PlanData,
    config: &SessionConfig,
    actual_mode: IoMode,
) -> Result<()> {
    let envelope = plan.runtime;

    if matches!(actual_mode, IoMode::Uring { .. })
        && envelope.maximum_combined_encoded > config.max_inflight_encoded_bytes_per_worker
    {
        return Err(Error::ResourceLimit(format!(
            "a job requires up to {} in-flight encoded bytes, per-worker limit is {}",
            envelope.maximum_combined_encoded, config.max_inflight_encoded_bytes_per_worker
        )));
    }

    if matches!(actual_mode, IoMode::Blocking) {
        let encoded_scratch = envelope
            .maximum_data_encoded
            .checked_add(envelope.maximum_indices_encoded)
            .ok_or_else(|| Error::ResourceLimit("worker encoded scratch size overflow".into()))?;
        if encoded_scratch > config.max_inflight_encoded_bytes_per_worker {
            return Err(Error::ResourceLimit(format!(
                "blocking worker can retain {encoded_scratch} encoded scratch bytes, per-worker limit is {}",
                config.max_inflight_encoded_bytes_per_worker
            )));
        }
    }
    let decoded_scratch = envelope
        .maximum_data_decoded
        .checked_add(envelope.maximum_indices_decoded)
        .ok_or_else(|| Error::ResourceLimit("worker decoded scratch size overflow".into()))?;
    if decoded_scratch > config.max_decoded_bytes_per_worker {
        return Err(Error::ResourceLimit(format!(
            "worker can retain {decoded_scratch} decoded scratch bytes, per-worker limit is {}",
            config.max_decoded_bytes_per_worker
        )));
    }
    Ok(())
}

#[inline(always)]
fn decode_block(
    plan: &PlanData,
    block_id: usize,
    encoded: &[u8],
    decoded: &mut DecodedBuffer,
    workspace: &mut DecodeWorkspace,
    output_ring: &AlignedBuffer,
) -> Result<()> {
    let block = plan
        .blocks
        .get(block_id)
        .ok_or_else(|| Error::Invariant("block group decoder index is invalid".into()))?;
    let encoded_range = block.encoded_range();
    let input = encoded
        .get(encoded_range)
        .ok_or_else(|| Error::Invariant("block encoded range exceeds physical read".into()))?;
    let decoded_len = block.decoded_len();
    let direct_output = block.direct_output();
    let output = if let Some(offset) = direct_output {
        decoded.set_empty();
        if offset > output_ring.len || decoded_len > output_ring.len - offset {
            return Err(Error::Invariant(
                "direct decoder output exceeds batch ring".into(),
            ));
        }
        // SAFETY: compiler direct-decode eligibility proves the full block maps
        // to one contiguous worker-owned output extent. Ring generations keep
        // consumers and other jobs from aliasing it until completion.
        unsafe {
            std::slice::from_raw_parts_mut(
                output_ring.row_pointer(offset, decoded_len),
                decoded_len,
            )
        }
    } else {
        decoded.prepare(decoded_len)?;
        // SAFETY: `prepare` allocated exactly `decoded_len` uninitialized bytes;
        // this decoder call owns and initializes the complete range.
        unsafe { decoded.range_mut(0..decoded_len) }
    };
    let written = block.decoder.decode_into(input, output, workspace)?;
    if written != output.len() {
        return Err(Error::Decode(format!(
            "block decoder wrote {written} bytes, expected {}",
            output.len()
        )));
    }
    if direct_output.is_none() {
        decoded.finish();
    }
    Ok(())
}

struct AtomicRuntimeStats {
    requested: IoMode,
    actual: IoMode,
    worker_count: usize,
    max_inflight_jobs_per_worker: usize,
    max_inflight_encoded_bytes_per_worker: usize,
    max_decoded_bytes_per_worker: usize,
    #[cfg(feature = "profile")]
    worker_stats: Box<[WorkerStats]>,
    #[cfg(feature = "profile")]
    consumer_wait_ns: AtomicU64,
    #[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
    inflight_jobs: AtomicUsize,
    #[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
    inflight_ops: AtomicUsize,
    #[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
    inflight_bytes: AtomicUsize,
    #[cfg(feature = "profile")]
    peak_jobs: AtomicUsize,
    #[cfg(feature = "profile")]
    peak_ops: AtomicUsize,
    #[cfg(feature = "profile")]
    peak_bytes: AtomicUsize,
}

/// One writer-owned cache-line-exclusive shard; no cumulative telemetry is
/// shared by different workers.
#[cfg(feature = "profile")]
#[repr(C, align(256))]
pub(crate) struct WorkerStats {
    jobs: AtomicU64,
    cells: AtomicU64,
    blocks: AtomicU64,
    decoded_bytes: AtomicU64,
    data_decode_ns: AtomicU64,
    indices_decode_ns: AtomicU64,
    validation_ns: AtomicU64,
    scatter_kernel_ns: AtomicU64,
    completion_ns: AtomicU64,
    window_wait_ns: AtomicU64,
    claim_retries: AtomicU64,
    window_blocks: AtomicU64,
    local_full: AtomicU64,
    pub(super) io_wait_ns: AtomicU64,
    pub(super) read_ops: AtomicU64,
    pub(super) read_bytes: AtomicU64,
    pub(super) short_reads: AtomicU64,
    pub(super) whole_keys: AtomicU64,
    pub(super) uring_prepared: AtomicU64,
    pub(super) uring_submitted: AtomicU64,
    pub(super) uring_submit_calls: AtomicU64,
    pub(super) uring_cqes: AtomicU64,
    pub(super) uring_cancel_requests: AtomicU64,
    pub(super) uring_cancel_cqes: AtomicU64,
}

#[cfg(feature = "profile")]
impl WorkerStats {
    fn snapshot(&self, worker_id: usize) -> WorkerRuntimeStats {
        let load = |value: &AtomicU64| value.load(Ordering::Relaxed);
        WorkerRuntimeStats {
            worker_id,
            completed_jobs: load(&self.jobs),
            completed_cells: load(&self.cells),
            decoded_blocks: load(&self.blocks),
            decoded_bytes: load(&self.decoded_bytes),
            data_decode_nanoseconds: load(&self.data_decode_ns),
            indices_decode_nanoseconds: load(&self.indices_decode_ns),
            validation_nanoseconds: load(&self.validation_ns),
            scatter_kernel_nanoseconds: load(&self.scatter_kernel_ns),
            completion_nanoseconds: load(&self.completion_ns),
            window_wait_nanoseconds: load(&self.window_wait_ns),
            io_wait_nanoseconds: load(&self.io_wait_ns),
            physical_read_ops: load(&self.read_ops),
            physical_read_bytes: load(&self.read_bytes),
            short_read_retries: load(&self.short_reads),
            whole_key_materializations: load(&self.whole_keys),
            claim_cas_retries: load(&self.claim_retries),
            window_block_events: load(&self.window_blocks),
            local_full_events: load(&self.local_full),
            uring_prepared_read_sqes: load(&self.uring_prepared),
            uring_submitted_read_sqes: load(&self.uring_submitted),
            uring_submit_calls: load(&self.uring_submit_calls),
            uring_cqes: load(&self.uring_cqes),
            uring_cancel_requests: load(&self.uring_cancel_requests),
            uring_cancel_cqes: load(&self.uring_cancel_cqes),
        }
    }
}

impl AtomicRuntimeStats {
    fn new(requested: IoMode, actual: IoMode, config: &SessionConfig) -> Self {
        #[cfg(feature = "profile")]
        let worker_stats = (0..config.worker_count)
            .map(|_| WorkerStats {
                jobs: AtomicU64::new(0),
                cells: AtomicU64::new(0),
                blocks: AtomicU64::new(0),
                decoded_bytes: AtomicU64::new(0),
                data_decode_ns: AtomicU64::new(0),
                indices_decode_ns: AtomicU64::new(0),
                validation_ns: AtomicU64::new(0),
                scatter_kernel_ns: AtomicU64::new(0),
                completion_ns: AtomicU64::new(0),
                window_wait_ns: AtomicU64::new(0),
                claim_retries: AtomicU64::new(0),
                window_blocks: AtomicU64::new(0),
                local_full: AtomicU64::new(0),
                io_wait_ns: AtomicU64::new(0),
                read_ops: AtomicU64::new(0),
                read_bytes: AtomicU64::new(0),
                short_reads: AtomicU64::new(0),
                whole_keys: AtomicU64::new(0),
                uring_prepared: AtomicU64::new(0),
                uring_submitted: AtomicU64::new(0),
                uring_submit_calls: AtomicU64::new(0),
                uring_cqes: AtomicU64::new(0),
                uring_cancel_requests: AtomicU64::new(0),
                uring_cancel_cqes: AtomicU64::new(0),
            })
            .collect();
        Self {
            requested,
            actual,
            worker_count: config.worker_count,
            max_inflight_jobs_per_worker: config.max_inflight_jobs_per_worker,
            max_inflight_encoded_bytes_per_worker: config.max_inflight_encoded_bytes_per_worker,
            max_decoded_bytes_per_worker: config.max_decoded_bytes_per_worker,
            #[cfg(feature = "profile")]
            worker_stats,
            #[cfg(feature = "profile")]
            consumer_wait_ns: AtomicU64::new(0),
            #[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
            inflight_jobs: AtomicUsize::new(0),
            #[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
            inflight_ops: AtomicUsize::new(0),
            #[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
            inflight_bytes: AtomicUsize::new(0),
            #[cfg(feature = "profile")]
            peak_jobs: AtomicUsize::new(0),
            #[cfg(feature = "profile")]
            peak_ops: AtomicUsize::new(0),
            #[cfg(feature = "profile")]
            peak_bytes: AtomicUsize::new(0),
        }
    }

    fn snapshot(&self, state: SessionState) -> RuntimeStats {
        RuntimeStats {
            requested_io_mode: self.requested,
            actual_io_mode: self.actual,
            worker_count: self.worker_count,
            max_inflight_jobs_per_worker: self.max_inflight_jobs_per_worker,
            max_inflight_encoded_bytes_per_worker: self.max_inflight_encoded_bytes_per_worker,
            max_decoded_bytes_per_worker: self.max_decoded_bytes_per_worker,
            #[cfg(feature = "profile")]
            physical_read_ops: sum_worker_stat(&self.worker_stats, |stats| &stats.read_ops),
            #[cfg(feature = "profile")]
            physical_read_bytes: sum_worker_stat(&self.worker_stats, |stats| &stats.read_bytes),
            #[cfg(feature = "profile")]
            short_read_retries: sum_worker_stat(&self.worker_stats, |stats| &stats.short_reads),
            #[cfg(feature = "profile")]
            whole_key_materializations: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.whole_keys
            }),
            #[cfg(feature = "profile")]
            uring_prepared_read_sqes: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.uring_prepared
            }),
            #[cfg(feature = "profile")]
            uring_submitted_read_sqes: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.uring_submitted
            }),
            #[cfg(feature = "profile")]
            uring_submit_calls: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.uring_submit_calls
            }),
            #[cfg(feature = "profile")]
            uring_cqes: sum_worker_stat(&self.worker_stats, |stats| &stats.uring_cqes),
            #[cfg(feature = "profile")]
            uring_cancel_requests: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.uring_cancel_requests
            }),
            #[cfg(feature = "profile")]
            uring_cancel_cqes: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.uring_cancel_cqes
            }),
            #[cfg(feature = "profile")]
            io_wait_nanoseconds: sum_worker_stat(&self.worker_stats, |stats| &stats.io_wait_ns),
            #[cfg(feature = "profile")]
            decode_nanoseconds: sum_worker_stat(&self.worker_stats, |stats| &stats.data_decode_ns)
                .saturating_add(sum_worker_stat(&self.worker_stats, |stats| {
                    &stats.indices_decode_ns
                })),
            #[cfg(feature = "profile")]
            data_decode_nanoseconds: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.data_decode_ns
            }),
            #[cfg(feature = "profile")]
            indices_decode_nanoseconds: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.indices_decode_ns
            }),
            #[cfg(feature = "profile")]
            validation_nanoseconds: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.validation_ns
            }),
            #[cfg(feature = "profile")]
            scatter_nanoseconds: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.scatter_kernel_ns
            })
            .saturating_add(sum_worker_stat(&self.worker_stats, |stats| {
                &stats.completion_ns
            })),
            #[cfg(feature = "profile")]
            scatter_kernel_nanoseconds: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.scatter_kernel_ns
            }),
            #[cfg(feature = "profile")]
            completion_nanoseconds: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.completion_ns
            }),
            #[cfg(feature = "profile")]
            window_wait_nanoseconds: sum_worker_stat(&self.worker_stats, |stats| {
                &stats.window_wait_ns
            }),
            #[cfg(feature = "profile")]
            consumer_wait_nanoseconds: self.consumer_wait_ns.load(Ordering::Relaxed),
            #[cfg(feature = "profile")]
            completed_jobs: sum_worker_stat(&self.worker_stats, |stats| &stats.jobs),
            #[cfg(feature = "profile")]
            completed_cells: sum_worker_stat(&self.worker_stats, |stats| &stats.cells),
            #[cfg(feature = "profile")]
            decoded_blocks: sum_worker_stat(&self.worker_stats, |stats| &stats.blocks),
            #[cfg(feature = "profile")]
            decoded_bytes: sum_worker_stat(&self.worker_stats, |stats| &stats.decoded_bytes),
            #[cfg(feature = "profile")]
            claim_cas_retries: sum_worker_stat(&self.worker_stats, |stats| &stats.claim_retries),
            #[cfg(feature = "profile")]
            window_block_events: sum_worker_stat(&self.worker_stats, |stats| &stats.window_blocks),
            #[cfg(feature = "profile")]
            local_full_events: sum_worker_stat(&self.worker_stats, |stats| &stats.local_full),
            #[cfg(feature = "profile")]
            peak_inflight_jobs: self.peak_jobs.load(Ordering::Relaxed),
            #[cfg(feature = "profile")]
            peak_inflight_read_ops: self.peak_ops.load(Ordering::Relaxed),
            #[cfg(feature = "profile")]
            peak_inflight_encoded_bytes: self.peak_bytes.load(Ordering::Relaxed),
            #[cfg(feature = "profile")]
            workers: self
                .worker_stats
                .iter()
                .enumerate()
                .map(|(worker_id, stats)| stats.snapshot(worker_id))
                .collect(),
            state,
        }
    }
}

#[cfg(feature = "profile")]
fn sum_worker_stat(workers: &[WorkerStats], select: impl Fn(&WorkerStats) -> &AtomicU64) -> u64 {
    workers.iter().fold(0u64, |total, worker| {
        total.saturating_add(select(worker).load(Ordering::Relaxed))
    })
}

#[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
fn update_peak(peak: &AtomicUsize, value: usize) {
    let mut current = peak.load(Ordering::Relaxed);
    while value > current {
        match peak.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
fn add_inflight(current: &AtomicUsize, peak: &AtomicUsize, amount: usize) -> Result<()> {
    let previous = current
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(amount)
        })
        .map_err(|_| Error::Invariant("runtime in-flight statistic overflow".into()))?;
    update_peak(peak, previous + amount);
    Ok(())
}

#[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
fn remove_inflight(current: &AtomicUsize, amount: usize) -> Result<()> {
    current
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_sub(amount)
        })
        .map(|_| ())
        .map_err(|_| Error::Invariant("runtime in-flight statistic underflow".into()))
}

#[cfg(feature = "profile")]
fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
