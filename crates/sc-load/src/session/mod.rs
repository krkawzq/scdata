#[cfg(not(target_os = "linux"))]
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::{BTreeMap, VecDeque};
#[cfg(all(feature = "uring", target_os = "linux"))]
use std::os::fd::AsRawFd;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use dyn_blosc::DecodeWorkspace;
use parking_lot::{Condvar, Mutex};

use crate::dtype::{OutputDType, OutputValue};
use crate::plan::{
    CellTask, CsrScatterTask, DenseScatterTask, PlanData, ReadSource, ReleasePlan, SourcePlan,
};
use crate::scatter::{scatter_row_prevalidated, validate_row};
use crate::{Error, IoMode, Result, SessionConfig};

enum WorkerIo {
    Blocking,
    #[cfg(all(feature = "uring", target_os = "linux"))]
    Uring(Box<io_uring::IoUring>),
}

const RUNNING: u8 = 0;
const FAILED: u8 = 1;
const CANCELLED: u8 = 2;
const FINISHED: u8 = 3;

const NODE_WAITING: u8 = 0;
const NODE_READY: u8 = 1;
const NODE_RUNNING: u8 = 2;
const NODE_DONE: u8 = 3;

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

pub(crate) struct AlignedBuffer {
    pointer: NonNull<u8>,
    len: usize,
    owned: bool,
}

impl AlignedBuffer {
    fn anonymous(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                pointer: NonNull::dangling(),
                len,
                owned: true,
            });
        }
        #[cfg(target_os = "linux")]
        {
            use rustix::mm::{Advice, MapFlags, ProtFlags};
            // SAFETY: this requests one private anonymous mapping owned by the
            // returned buffer until its matching `munmap` in `Drop`.
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
                .ok_or_else(|| Error::Allocation("mmap returned null".into()))?;
            // SAFETY: this is the live mapping above; the advice is optional.
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
            // SAFETY: layout is non-zero and valid. Zero initialization keeps
            // output padding initialized on platforms without anonymous mmap.
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

    pub(crate) fn from_shared(pointer: NonNull<u8>, len: usize) -> Self {
        Self {
            pointer: if len == 0 {
                NonNull::dangling()
            } else {
                pointer
            },
            len,
            owned: false,
        }
    }

    fn pointer_at(&self, offset: usize, len: usize) -> Result<NonNull<u8>> {
        if offset > self.len || len > self.len - offset {
            return Err(Error::Invariant(format!(
                "buffer range {offset}..{} exceeds {} bytes",
                offset.saturating_add(len),
                self.len
            )));
        }
        // SAFETY: the validated offset lies in this allocation (or is the
        // one-past pointer for an empty range).
        Ok(unsafe { NonNull::new_unchecked(self.pointer.as_ptr().add(offset)) })
    }

    unsafe fn slice(&self, offset: usize, len: usize) -> &[u8] {
        debug_assert!(offset <= self.len && len <= self.len - offset);
        // SAFETY: caller proves the range is initialized and no writer aliases it.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr().add(offset), len) }
    }
}

// SAFETY: all access is governed by immutable static ranges, dependency
// publication, and output-generation leases.
unsafe impl Send for AlignedBuffer {}
// SAFETY: the same graph prevents unsynchronized aliased mutable access.
unsafe impl Sync for AlignedBuffer {}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.len == 0 || !self.owned {
            return;
        }
        #[cfg(target_os = "linux")]
        {
            // SAFETY: construction obtained this exact mapping and workers are joined.
            let _ = unsafe { rustix::mm::munmap(self.pointer.as_ptr().cast(), self.len) };
        }
        #[cfg(not(target_os = "linux"))]
        {
            let layout = Layout::from_size_align(self.len, 64)
                .expect("buffer allocation layout remains valid");
            // SAFETY: this matches the allocation in `anonymous`.
            unsafe { dealloc(self.pointer.as_ptr(), layout) };
        }
    }
}

struct RuntimeDecodeOp {
    encoded_offset: usize,
    encoded_len: usize,
    decoder: dyn_blosc::BlockDecoder,
    target: NonNull<u8>,
    decoded_len: usize,
    successors: NonNull<usize>,
    successor_count: usize,
}

struct RuntimeIoTask {
    source: NonNull<ReadSource>,
    file_offset: u64,
    file_len: usize,
    decode_ops: NonNull<RuntimeDecodeOp>,
    decode_op_count: usize,
    priority: u64,
}

struct RuntimeDenseTask {
    data: Option<(NonNull<u8>, usize)>,
    source: NonNull<SourcePlan>,
    output: NonNull<u8>,
    output_len: usize,
    cell: CellTask,
    batch: usize,
}

struct RuntimeCsrTask {
    data: NonNull<u8>,
    data_len: usize,
    indices: NonNull<u8>,
    indices_len: usize,
    source: NonNull<SourcePlan>,
    output: NonNull<u8>,
    output_len: usize,
    cell: CellTask,
    batch: usize,
}

// SAFETY: descriptors point only into frozen plan arenas and session-owned
// cache/output allocations that outlive every worker.
unsafe impl Send for RuntimeDecodeOp {}
// SAFETY: disjoint generation ownership makes descriptor targets thread-safe.
unsafe impl Sync for RuntimeDecodeOp {}
// SAFETY: see `RuntimeDecodeOp`; source objects are immutable.
unsafe impl Send for RuntimeIoTask {}
// SAFETY: RuntimeIoTask contains only immutable source/descriptor pointers;
// each encoded staging buffer remains worker-local.
unsafe impl Sync for RuntimeIoTask {}
// SAFETY: the static graph assigns one writer to each output row generation
// and keeps the referenced cache generation immutable during scatter.
unsafe impl Send for RuntimeDenseTask {}
// SAFETY: see the `Send` proof; concurrent readers never mutate descriptors.
unsafe impl Sync for RuntimeDenseTask {}
// SAFETY: CSR pointers obey the same immutable-cache and unique-output-row
// ownership rules established by the compiler and dependency graph.
unsafe impl Send for RuntimeCsrTask {}
// SAFETY: see the `Send` proof; descriptors and source plans are immutable.
unsafe impl Sync for RuntimeCsrTask {}

struct ExecutionPlan {
    io: Box<[RuntimeIoTask]>,
    _decode_ops: Box<[RuntimeDecodeOp]>,
    dense: Box<[RuntimeDenseTask]>,
    csr: Box<[RuntimeCsrTask]>,
    dense_base: usize,
    csr_base: usize,
}

impl ExecutionPlan {
    fn lower(plan: &PlanData, cache: &AlignedBuffer, output: &AlignedBuffer) -> Result<Self> {
        let logical = &plan.static_plan;
        if logical.cache_alignment < 64 || !logical.cache_alignment.is_power_of_two() {
            return Err(Error::Invariant("static cache alignment is invalid".into()));
        }
        let mut init_decoded = 0usize;
        let mut init_io = 0usize;
        for task in logical
            .io_decode_tasks
            .get(logical.initialize.io_tasks.clone())
            .ok_or_else(|| Error::Invariant("InitializeJob I/O range is invalid".into()))?
        {
            init_io = init_io.saturating_add(task.file_len);
            for operation in logical
                .decode_ops
                .get(task.decode_ops.clone())
                .ok_or_else(|| Error::Invariant("InitializeJob decode range is invalid".into()))?
            {
                init_decoded = init_decoded.saturating_add(operation.cache.len);
            }
        }
        if init_decoded != logical.initialize.decoded_bytes
            || init_io != logical.initialize.io_bytes
        {
            return Err(Error::Invariant(
                "InitializeJob byte accounting is inconsistent".into(),
            ));
        }
        let graph = &logical.dependencies;
        let successor_base = NonNull::new(graph.block_ready_successors.as_ptr().cast_mut())
            .unwrap_or_else(NonNull::dangling);
        for (batch, job) in logical.jobs.iter().enumerate() {
            if job.batch_id != batch as u64
                || job.output_generation != batch as u64
                || job.output_slot != batch % plan.ring_slots
                || job.io_tasks.end > logical.io_decode_tasks.len()
            {
                return Err(Error::Invariant("static Job layout is inconsistent".into()));
            }
        }
        let mut decode_ops = Vec::with_capacity(logical.decode_ops.len());
        for operation in logical.decode_ops.iter().copied() {
            let range = graph
                .block_ready_ranges
                .get(operation.ready_node)
                .ok_or_else(|| Error::Invariant("decode ready node is out of range".into()))?;
            let successors = graph
                .block_ready_successors
                .get(range.clone())
                .ok_or_else(|| Error::Invariant("decode successor range is invalid".into()))?;
            if successors
                .iter()
                .any(|&node| node >= graph.initial_dependency_count.len())
            {
                return Err(Error::Invariant(
                    "decode successor node is out of range".into(),
                ));
            }
            // SAFETY: `range` was validated against the immutable successor
            // arena, including its legal one-past position when it is empty.
            let successor_pointer =
                unsafe { NonNull::new_unchecked(successor_base.as_ptr().add(range.start)) };
            decode_ops.push(RuntimeDecodeOp {
                encoded_offset: operation.encoded_offset,
                encoded_len: operation.encoded_len,
                decoder: operation.decoder,
                target: cache.pointer_at(operation.cache.offset, operation.cache.len)?,
                decoded_len: operation.cache.len,
                successors: successor_pointer,
                successor_count: successors.len(),
            });
        }
        let decode_ops = decode_ops.into_boxed_slice();
        let decode_base =
            NonNull::new(decode_ops.as_ptr().cast_mut()).unwrap_or_else(NonNull::dangling);
        let mut io = Vec::with_capacity(logical.io_decode_tasks.len());
        for task in &logical.io_decode_tasks {
            let source = plan
                .sources
                .get(task.source)
                .ok_or_else(|| Error::Invariant("static I/O source is missing".into()))?;
            let operation_count = task.decode_ops.end - task.decode_ops.start;
            // SAFETY: the logical range was compiler-built inside decode_ops.
            let operations =
                unsafe { NonNull::new_unchecked(decode_base.as_ptr().add(task.decode_ops.start)) };
            io.push(RuntimeIoTask {
                source: NonNull::from(source),
                file_offset: task.file_offset,
                file_len: task.file_len,
                decode_ops: operations,
                decode_op_count: operation_count,
                priority: task.earliest_consumer_batch,
            });
        }
        let mut dense = Vec::with_capacity(logical.dense_scatter_tasks.len());
        for task in &logical.dense_scatter_tasks {
            dense.push(lower_dense(plan, cache, output, task)?);
        }
        let mut csr = Vec::with_capacity(logical.csr_scatter_tasks.len());
        for task in &logical.csr_scatter_tasks {
            csr.push(lower_csr(plan, cache, output, task)?);
        }
        let dense_base = io.len();
        let csr_base = dense_base + dense.len();
        Ok(Self {
            io: io.into_boxed_slice(),
            _decode_ops: decode_ops,
            dense: dense.into_boxed_slice(),
            csr: csr.into_boxed_slice(),
            dense_base,
            csr_base,
        })
    }

    fn priority(&self, node: usize) -> u64 {
        if node < self.dense_base {
            self.io[node].priority
        } else if node < self.csr_base {
            self.dense[node - self.dense_base].batch as u64
        } else {
            self.csr[node - self.csr_base].batch as u64
        }
    }
}

fn lower_dense(
    plan: &PlanData,
    cache: &AlignedBuffer,
    output: &AlignedBuffer,
    task: &DenseScatterTask,
) -> Result<RuntimeDenseTask> {
    let source = plan
        .source_plans
        .get(task.source_plan)
        .ok_or_else(|| Error::Invariant("dense source plan is missing".into()))?;
    Ok(RuntimeDenseTask {
        data: task
            .data
            .map(|slice| -> Result<_> {
                Ok((cache.pointer_at(slice.offset, slice.len)?, slice.len))
            })
            .transpose()?,
        source: NonNull::from(source),
        output: output.pointer_at(task.output.ring_offset, task.output.len)?,
        output_len: task.output.len,
        cell: task.cell,
        batch: usize::try_from(task.output.generation)
            .map_err(|_| Error::ResourceLimit("dense batch exceeds usize".into()))?,
    })
}

fn lower_csr(
    plan: &PlanData,
    cache: &AlignedBuffer,
    output: &AlignedBuffer,
    task: &CsrScatterTask,
) -> Result<RuntimeCsrTask> {
    let source = plan
        .source_plans
        .get(task.source_plan)
        .ok_or_else(|| Error::Invariant("CSR source plan is missing".into()))?;
    Ok(RuntimeCsrTask {
        data: cache.pointer_at(task.data.offset, task.data.len)?,
        data_len: task.data.len,
        indices: cache.pointer_at(task.indices.offset, task.indices.len)?,
        indices_len: task.indices.len,
        source: NonNull::from(source),
        output: output.pointer_at(task.output.ring_offset, task.output.len)?,
        output_len: task.output.len,
        cell: task.cell,
        batch: usize::try_from(task.output.generation)
            .map_err(|_| Error::ResourceLimit("CSR batch exceeds usize".into()))?,
    })
}

struct RuntimeNodeState {
    remaining: AtomicU32,
    state: AtomicU8,
}

struct ReadyState {
    buckets: BTreeMap<u64, VecDeque<usize>>,
    stopped: bool,
}

struct ReadyQueue {
    state: Mutex<ReadyState>,
    changed: Condvar,
}

impl ReadyQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(ReadyState {
                buckets: BTreeMap::new(),
                stopped: false,
            }),
            changed: Condvar::new(),
        }
    }

    fn push(&self, priority: u64, node: usize) {
        let mut state = self.state.lock();
        if state.stopped {
            return;
        }
        state.buckets.entry(priority).or_default().push_back(node);
        self.changed.notify_one();
    }

    fn pop(&self) -> Option<usize> {
        let mut state = self.state.lock();
        loop {
            if let Some((&priority, _)) = state.buckets.first_key_value() {
                let bucket = state
                    .buckets
                    .get_mut(&priority)
                    .expect("selected priority remains present");
                let node = bucket.pop_front();
                if bucket.is_empty() {
                    state.buckets.remove(&priority);
                }
                return node;
            }
            if state.stopped {
                return None;
            }
            self.changed.wait(&mut state);
        }
    }

    fn stop(&self) {
        let mut state = self.state.lock();
        state.stopped = true;
        state.buckets.clear();
        self.changed.notify_all();
    }
}

struct BatchSlot {
    generation: AtomicUsize,
    ready: AtomicBool,
}

struct RuntimeCounters {
    reads: AtomicUsize,
    read_bytes: AtomicUsize,
    decoded_blocks: AtomicUsize,
    decoded_bytes: AtomicUsize,
    completed_jobs: AtomicUsize,
    completed_cells: AtomicUsize,
    #[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
    uring_reads: AtomicUsize,
}

impl RuntimeCounters {
    fn new() -> Self {
        Self {
            reads: AtomicUsize::new(0),
            read_bytes: AtomicUsize::new(0),
            decoded_blocks: AtomicUsize::new(0),
            decoded_bytes: AtomicUsize::new(0),
            completed_jobs: AtomicUsize::new(0),
            completed_cells: AtomicUsize::new(0),
            #[cfg(all(feature = "profile", feature = "uring", target_os = "linux"))]
            uring_reads: AtomicUsize::new(0),
        }
    }
}

#[cfg(feature = "profile")]
fn uring_read_count(counters: &RuntimeCounters) -> u64 {
    #[cfg(all(feature = "uring", target_os = "linux"))]
    {
        counters.uring_reads.load(Ordering::Relaxed) as u64
    }
    #[cfg(not(all(feature = "uring", target_os = "linux")))]
    {
        let _ = counters;
        0
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

struct SessionInner {
    plan: Arc<PlanData>,
    _cache: AlignedBuffer,
    output: AlignedBuffer,
    execution: ExecutionPlan,
    nodes: Box<[RuntimeNodeState]>,
    ready: ReadyQueue,
    job_remaining: Box<[AtomicUsize]>,
    job_done: Box<[AtomicBool]>,
    prefix_cursor: Mutex<usize>,
    batch_slots: Box<[BatchSlot]>,
    consume_idx: AtomicUsize,
    state: AtomicU8,
    first_error: Mutex<Option<Error>>,
    consumer_lock: Mutex<()>,
    consumer_changed: Condvar,
    requested_io_mode: IoMode,
    actual_io_mode: IoMode,
    config: SessionConfig,
    counters: RuntimeCounters,
}

impl Session {
    pub(crate) fn start(plan: Arc<PlanData>, config: SessionConfig) -> Result<Self> {
        let output = AlignedBuffer::anonymous(plan.stats.output_ring_bytes)?;
        Self::start_with_output(plan, config, output)
    }

    pub(crate) fn start_with_output(
        plan: Arc<PlanData>,
        config: SessionConfig,
        output: AlignedBuffer,
    ) -> Result<Self> {
        config.validate()?;
        let requested_io_mode = config.io_mode;
        let all_positioned = plan
            .sources
            .iter()
            .all(|source| matches!(source, ReadSource::Empty | ReadSource::Positioned { .. }));
        #[cfg(all(feature = "uring", target_os = "linux"))]
        let mut prepared_rings = Vec::new();
        let actual_io_mode = match requested_io_mode {
            IoMode::Blocking => IoMode::Blocking,
            IoMode::Uring {
                queue_depth: _queue_depth,
            } => {
                if !all_positioned {
                    return Err(Error::Unsupported(
                        "io_uring requires positioned filesystem sources".into(),
                    ));
                }
                #[cfg(all(feature = "uring", target_os = "linux"))]
                {
                    prepared_rings.try_reserve_exact(config.worker_count)?;
                    for _ in 0..config.worker_count {
                        prepared_rings.push(io_uring::IoUring::new(_queue_depth)?);
                    }
                    IoMode::Uring {
                        queue_depth: _queue_depth,
                    }
                }
                #[cfg(not(all(feature = "uring", target_os = "linux")))]
                return Err(Error::Unsupported(
                    "io_uring requires Linux and the uring feature".into(),
                ));
            }
            IoMode::Auto { queue_depth } => {
                #[cfg(all(feature = "uring", target_os = "linux"))]
                {
                    if all_positioned {
                        let mut available = true;
                        prepared_rings.try_reserve_exact(config.worker_count)?;
                        for _ in 0..config.worker_count {
                            match io_uring::IoUring::new(queue_depth) {
                                Ok(ring) => prepared_rings.push(ring),
                                Err(_) => {
                                    available = false;
                                    prepared_rings.clear();
                                    break;
                                }
                            }
                        }
                        if available {
                            IoMode::Uring { queue_depth }
                        } else {
                            IoMode::Blocking
                        }
                    } else {
                        IoMode::Blocking
                    }
                }
                #[cfg(not(all(feature = "uring", target_os = "linux")))]
                {
                    let _ = (queue_depth, all_positioned);
                    IoMode::Blocking
                }
            }
        };
        if output.len != plan.stats.output_ring_bytes {
            return Err(Error::Invariant(format!(
                "output ring has {} bytes, plan requires {}",
                output.len, plan.stats.output_ring_bytes
            )));
        }
        let regular_io_tasks = plan
            .static_plan
            .io_decode_tasks
            .get(plan.static_plan.initialize.io_tasks.end..)
            .ok_or_else(|| Error::Invariant("InitializeJob I/O range is invalid".into()))?;
        let maximum_regular_encoded = regular_io_tasks
            .iter()
            .map(|task| task.file_len)
            .max()
            .unwrap_or(0);
        if maximum_regular_encoded > config.max_inflight_encoded_bytes_per_worker {
            return Err(Error::ResourceLimit(format!(
                "regular I/O task requires {maximum_regular_encoded} encoded bytes, per-worker limit is {}",
                config.max_inflight_encoded_bytes_per_worker
            )));
        }
        let cache = AlignedBuffer::anonymous(plan.static_plan.cache_capacity)?;
        let execution = ExecutionPlan::lower(&plan, &cache, &output)?;
        if plan.static_plan.dependencies.initial_dependency_count.len()
            != execution.io.len() + execution.dense.len() + execution.csr.len()
        {
            return Err(Error::Invariant(
                "runtime dependency count does not match executable tasks".into(),
            ));
        }
        let dependencies = &plan.static_plan.dependencies.initial_dependency_count;
        let nodes = dependencies
            .iter()
            .map(|remaining| RuntimeNodeState {
                remaining: AtomicU32::new(*remaining),
                state: AtomicU8::new(NODE_WAITING),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let job_remaining = plan
            .static_plan
            .jobs
            .iter()
            .map(|job| {
                AtomicUsize::new(
                    (job.dense_tasks.end - job.dense_tasks.start)
                        + (job.csr_tasks.end - job.csr_tasks.start),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let job_done = (0..plan.batch_count)
            .map(|_| AtomicBool::new(false))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let batch_slots = (0..plan.ring_slots)
            .map(|slot| BatchSlot {
                generation: AtomicUsize::new(slot),
                ready: AtomicBool::new(false),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let initial_state = if plan.batch_count == 0 {
            FINISHED
        } else {
            RUNNING
        };
        let inner = Arc::new(SessionInner {
            plan,
            _cache: cache,
            output,
            execution,
            nodes,
            ready: ReadyQueue::new(),
            job_remaining,
            job_done,
            prefix_cursor: Mutex::new(0),
            batch_slots,
            consume_idx: AtomicUsize::new(0),
            state: AtomicU8::new(initial_state),
            first_error: Mutex::new(None),
            consumer_lock: Mutex::new(()),
            consumer_changed: Condvar::new(),
            requested_io_mode,
            actual_io_mode,
            config: config.clone(),
            counters: RuntimeCounters::new(),
        });
        if initial_state == FINISHED {
            inner.ready.stop();
            return Ok(Self {
                inner,
                workers: Vec::new(),
            });
        }

        run_initialize(&inner)?;
        for node in 0..inner.nodes.len() {
            inner.enqueue_if_ready(node)?;
        }
        let mut workers = Vec::with_capacity(config.worker_count);
        for worker_id in 0..config.worker_count {
            let worker_inner = Arc::clone(&inner);
            #[cfg(all(feature = "uring", target_os = "linux"))]
            let worker_io = if matches!(actual_io_mode, IoMode::Uring { .. }) {
                WorkerIo::Uring(Box::new(prepared_rings.remove(0)))
            } else {
                WorkerIo::Blocking
            };
            #[cfg(not(all(feature = "uring", target_os = "linux")))]
            let worker_io = WorkerIo::Blocking;
            match std::thread::Builder::new()
                .name(format!("sc-load-dag-{worker_id}"))
                .spawn(move || worker_entry(worker_inner, worker_io))
            {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    inner.cancel();
                    for worker in workers.drain(..) {
                        let _ = worker.join();
                    }
                    return Err(error.into());
                }
            }
        }
        Ok(Self { inner, workers })
    }

    pub fn state(&self) -> SessionState {
        self.inner.state()
    }

    pub fn stats(&self) -> RuntimeStats {
        self.inner.stats()
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }

    pub fn cancellation_handle(&self) -> CancellationHandle {
        CancellationHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    pub(crate) fn wait_ready_for(
        &self,
        logical: usize,
        timeout: std::time::Duration,
    ) -> Result<bool> {
        self.inner.wait_ready(logical, Some(timeout))
    }

    pub(crate) fn consume_idx(&self) -> usize {
        self.inner.consume_idx.load(Ordering::Acquire)
    }

    pub(crate) fn terminal_error(&self) -> Error {
        self.inner.terminal_error()
    }

    pub(crate) fn batch_count(&self) -> usize {
        self.inner.plan.batch_count
    }

    pub(crate) fn commit_release(&self, logical: usize) -> Result<()> {
        let expected = self.consume_idx();
        if logical != expected {
            return Err(Error::Invariant(format!(
                "release expected batch {expected}, got {logical}"
            )));
        }
        self.inner.release_batch(logical)
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch<'_>>> {
        let logical = self.consume_idx();
        if logical >= self.inner.plan.batch_count {
            return match self.state() {
                SessionState::Failed => Err(self.terminal_error()),
                SessionState::Cancelled => Err(Error::Cancelled),
                _ => Ok(None),
            };
        }
        if !self.inner.wait_ready(logical, None)? {
            return Ok(None);
        }
        Ok(Some(Batch {
            rows: batch_len(&self.inner.plan, logical),
            session: self,
            logical_batch: logical,
            released: false,
        }))
    }

    fn release_batch(&self, logical: usize) {
        if let Err(error) = self.inner.release_batch(logical) {
            self.inner.fail(error);
        }
    }
}

impl CancellationHandle {
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    pub fn state(&self) -> SessionState {
        self.inner.state()
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

    pub fn bytes(&self) -> &[u8] {
        let plan = &self.session.inner.plan;
        let slot = ring_slot(plan, self.logical_batch);
        let offset = slot * plan.batch_size * plan.row_stride;
        let len = self.rows * plan.row_stride;
        // SAFETY: Batch is returned after release/acquire publication and the
        // mutable Session borrow holds this generation lease.
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
            "typed batch views require little endian".into(),
        ));
        #[cfg(target_endian = "little")]
        {
            if std::mem::size_of::<T>() != self.dtype().size()
                || bytes.as_ptr().align_offset(std::mem::align_of::<T>()) != 0
            {
                return Err(Error::Invariant("batch element layout mismatch".into()));
            }
            // SAFETY: OutputValue is sealed and layout/alignment were checked.
            Ok(unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr().cast::<T>(),
                    bytes.len() / std::mem::size_of::<T>(),
                )
            })
        }
    }

    pub fn as_slice<T: OutputValue>(&self) -> Result<&[T]> {
        if self.rows == 1 {
            return self.row_as(0);
        }
        let row_bytes = self
            .n_cols()
            .checked_mul(self.dtype().size())
            .ok_or_else(|| Error::Invariant("logical row bytes overflow".into()))?;
        if self.row_stride_bytes() != row_bytes {
            return Err(Error::Unsupported(
                "batch rows are padded; use row_as or as_padded_slice".into(),
            ));
        }
        self.as_padded_slice()
    }

    pub fn as_padded_slice<T: OutputValue>(&self) -> Result<&[T]> {
        if T::DTYPE != self.dtype() {
            return Err(Error::InvalidInput(format!(
                "requested {} view but batch dtype is {}",
                T::DTYPE,
                self.dtype()
            )));
        }
        let bytes = self.bytes();
        if !bytes.len().is_multiple_of(std::mem::size_of::<T>())
            || bytes.as_ptr().align_offset(std::mem::align_of::<T>()) != 0
        {
            return Err(Error::Invariant("padded batch layout mismatch".into()));
        }
        // SAFETY: the complete published stride, including padding, is initialized.
        Ok(unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr().cast::<T>(),
                bytes.len() / std::mem::size_of::<T>(),
            )
        })
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

impl SessionInner {
    fn state(&self) -> SessionState {
        SessionState::from_raw(self.state.load(Ordering::Acquire))
    }

    fn is_running(&self) -> bool {
        self.state.load(Ordering::Acquire) == RUNNING
    }

    fn fail(&self, error: Error) {
        let mut first_error = self.first_error.lock();
        if self
            .state
            .compare_exchange(RUNNING, FAILED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            *first_error = Some(error);
        }
        drop(first_error);
        self.ready.stop();
        self.wake_consumer();
    }

    fn cancel(&self) {
        let _ =
            self.state
                .compare_exchange(RUNNING, CANCELLED, Ordering::AcqRel, Ordering::Acquire);
        self.ready.stop();
        self.wake_consumer();
    }

    fn terminal_error(&self) -> Error {
        match self.state() {
            SessionState::Failed => {
                Error::Session(Arc::new(self.first_error.lock().clone().unwrap_or_else(
                    || Error::Invariant("session failed without an error".into()),
                )))
            }
            SessionState::Cancelled => Error::Cancelled,
            state => Error::Invariant(format!("session state {state:?} is not terminal")),
        }
    }

    fn enqueue_if_ready(&self, node: usize) -> Result<()> {
        let state = self
            .nodes
            .get(node)
            .ok_or_else(|| Error::Invariant("runtime node is out of range".into()))?;
        if state.remaining.load(Ordering::Acquire) != 0 {
            return Ok(());
        }
        if state
            .state
            .compare_exchange(
                NODE_WAITING,
                NODE_READY,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.ready.push(self.execution.priority(node), node);
        }
        Ok(())
    }

    fn release_dependency(&self, node: usize) -> Result<()> {
        let state = self
            .nodes
            .get(node)
            .ok_or_else(|| Error::Invariant("released node is out of range".into()))?;
        let previous = state.remaining.fetch_sub(1, Ordering::AcqRel);
        if previous == 0 {
            return Err(Error::Invariant("dependency counter underflow".into()));
        }
        if previous == 1 {
            self.enqueue_if_ready(node)?;
        }
        Ok(())
    }

    fn complete_block_ready(&self, operation: &RuntimeDecodeOp) -> Result<()> {
        // SAFETY: lowering validated this pointer/count pair against the
        // immutable plan successor arena, which outlives all workers.
        let successors = unsafe {
            std::slice::from_raw_parts(operation.successors.as_ptr(), operation.successor_count)
        };
        for &successor in successors {
            self.release_dependency(successor)?;
        }
        Ok(())
    }

    fn complete_job_task(&self, batch: usize) -> Result<()> {
        let counter = self
            .job_remaining
            .get(batch)
            .ok_or_else(|| Error::Invariant("job completion batch is invalid".into()))?;
        let previous = counter.fetch_sub(1, Ordering::AcqRel);
        if previous == 0 {
            return Err(Error::Invariant("job completion underflow".into()));
        }
        self.counters
            .completed_cells
            .fetch_add(1, Ordering::Relaxed);
        if previous != 1 {
            return Ok(());
        }
        self.counters.completed_jobs.fetch_add(1, Ordering::Relaxed);
        self.job_done[batch].store(true, Ordering::Release);
        let slot = ring_slot(&self.plan, batch);
        if self.batch_slots[slot].generation.load(Ordering::Acquire) != batch {
            return Err(Error::Invariant(
                "output generation changed before publish".into(),
            ));
        }
        {
            let _guard = self.consumer_lock.lock();
            self.batch_slots[slot].ready.store(true, Ordering::Release);
            self.consumer_changed.notify_all();
        }
        self.advance_prefix()
    }

    fn advance_prefix(&self) -> Result<()> {
        let mut cursor = self.prefix_cursor.lock();
        while *cursor < self.job_done.len() && self.job_done[*cursor].load(Ordering::Acquire) {
            self.release_nodes(&self.plan.static_plan.prefix_releases, *cursor)?;
            *cursor += 1;
        }
        Ok(())
    }

    fn release_nodes(&self, releases: &ReleasePlan, batch: usize) -> Result<()> {
        let Some(range) = releases.release_ranges.get(batch) else {
            return Ok(());
        };
        for &node in releases
            .released_nodes
            .get(range.clone())
            .ok_or_else(|| Error::Invariant("release range is invalid".into()))?
        {
            self.release_dependency(node)?;
        }
        Ok(())
    }

    fn execute(
        &self,
        node: usize,
        io: &mut WorkerIo,
        encoded: &mut Vec<u8>,
        workspace: &mut DecodeWorkspace,
    ) -> Result<()> {
        if node < self.execution.dense_base {
            self.execute_io_with(node, io, encoded, workspace)?;
            Ok(())
        } else if node < self.execution.csr_base {
            let task = &self.execution.dense[node - self.execution.dense_base];
            self.execute_dense(task)?;
            self.complete_job_task(task.batch)
        } else {
            let task = self
                .execution
                .csr
                .get(node - self.execution.csr_base)
                .ok_or_else(|| Error::Invariant("CSR node is out of range".into()))?;
            self.execute_csr(task)?;
            self.complete_job_task(task.batch)
        }
    }

    fn execute_io(
        &self,
        node: usize,
        encoded: &mut Vec<u8>,
        workspace: &mut DecodeWorkspace,
    ) -> Result<()> {
        self.execute_io_with(node, &mut WorkerIo::Blocking, encoded, workspace)
    }

    fn execute_io_with(
        &self,
        node: usize,
        io: &mut WorkerIo,
        encoded: &mut Vec<u8>,
        workspace: &mut DecodeWorkspace,
    ) -> Result<()> {
        let task = self
            .execution
            .io
            .get(node)
            .ok_or_else(|| Error::Invariant("I/O node is out of range".into()))?;
        // SAFETY: lowering points at an immutable source owned by `plan`.
        let source = unsafe { task.source.as_ref() };
        match io {
            WorkerIo::Blocking => {
                read_exact_source(source, task.file_offset, task.file_len, encoded)?
            }
            #[cfg(all(feature = "uring", target_os = "linux"))]
            WorkerIo::Uring(ring) => {
                let operations =
                    read_exact_uring(source, task.file_offset, task.file_len, encoded, ring)?;
                #[cfg(feature = "profile")]
                self.counters
                    .uring_reads
                    .fetch_add(operations, Ordering::Relaxed);
                #[cfg(not(feature = "profile"))]
                let _ = operations;
            }
        }
        self.counters.reads.fetch_add(1, Ordering::Relaxed);
        self.counters
            .read_bytes
            .fetch_add(task.file_len, Ordering::Relaxed);
        // SAFETY: lowering created a contiguous range inside the immutable
        // runtime decode-op arena.
        let operations =
            unsafe { std::slice::from_raw_parts(task.decode_ops.as_ptr(), task.decode_op_count) };
        for operation in operations {
            if !self.is_running() {
                return Err(Error::Cancelled);
            }
            let end = operation
                .encoded_offset
                .checked_add(operation.encoded_len)
                .ok_or_else(|| Error::Invariant("encoded range overflow".into()))?;
            let input = encoded
                .get(operation.encoded_offset..end)
                .ok_or_else(|| Error::StalePlan("encoded block is shorter than planned".into()))?;
            // SAFETY: the static cache compiler gives this load exclusive
            // ownership of the target generation until PrefixDone releases it.
            let output = unsafe {
                std::slice::from_raw_parts_mut(operation.target.as_ptr(), operation.decoded_len)
            };
            let written = operation.decoder.decode_into(input, output, workspace)?;
            if written != operation.decoded_len {
                return Err(Error::Decode(format!(
                    "decoder wrote {written} bytes, expected {}",
                    operation.decoded_len
                )));
            }
            self.counters.decoded_blocks.fetch_add(1, Ordering::Relaxed);
            self.counters
                .decoded_bytes
                .fetch_add(written, Ordering::Relaxed);
            self.complete_block_ready(operation)?;
        }
        Ok(())
    }

    fn execute_dense(&self, task: &RuntimeDenseTask) -> Result<()> {
        // SAFETY: lowering points into immutable plan/source memory.
        let source = unsafe { task.source.as_ref() };
        // SAFETY: cache generation dependency makes the decoded bytes immutable
        // for this scatter, and the output generation gives this row one writer.
        unsafe {
            let row = std::slice::from_raw_parts_mut(task.output.as_ptr(), task.output_len);
            if let Some((data, len)) = task.data {
                let data = std::slice::from_raw_parts(data.as_ptr(), len);
                if source.requires_runtime_validation() {
                    validate_row(source, &task.cell, data, &[])?;
                }
                scatter_row_prevalidated(
                    source,
                    &task.cell,
                    data,
                    &[],
                    row,
                    self.plan.row_bytes,
                    self.plan.fill,
                )
            } else {
                self.plan.fill.apply(row.as_mut_ptr(), self.plan.row_bytes);
                Ok(())
            }
        }
    }

    fn execute_csr(&self, task: &RuntimeCsrTask) -> Result<()> {
        // SAFETY: all pointers were range-checked during lowering; load and
        // ring dependencies provide immutable inputs and one output writer.
        unsafe {
            let source = task.source.as_ref();
            let data = std::slice::from_raw_parts(task.data.as_ptr(), task.data_len);
            let indices = std::slice::from_raw_parts(task.indices.as_ptr(), task.indices_len);
            validate_row(source, &task.cell, data, indices)?;
            let row = std::slice::from_raw_parts_mut(task.output.as_ptr(), task.output_len);
            scatter_row_prevalidated(
                source,
                &task.cell,
                data,
                indices,
                row,
                self.plan.row_bytes,
                self.plan.fill,
            )
        }
    }

    fn wait_ready(&self, logical: usize, timeout: Option<std::time::Duration>) -> Result<bool> {
        if logical >= self.plan.batch_count {
            return Err(Error::InvalidInput(format!(
                "batch {logical} is outside {} batches",
                self.plan.batch_count
            )));
        }
        let slot = ring_slot(&self.plan, logical);
        let mut guard = self.consumer_lock.lock();
        loop {
            match self.state() {
                SessionState::Failed => return Err(self.terminal_error()),
                SessionState::Cancelled => return Err(Error::Cancelled),
                SessionState::Finished => return Ok(false),
                SessionState::Running => {}
            }
            if self.batch_slots[slot].generation.load(Ordering::Acquire) != logical {
                return Err(Error::Invariant(
                    "consumer observed wrong ring generation".into(),
                ));
            }
            if self.batch_slots[slot].ready.load(Ordering::Acquire) {
                return Ok(true);
            }
            if let Some(duration) = timeout {
                self.consumer_changed.wait_for(&mut guard, duration);
                return Ok(self.batch_slots[slot].ready.load(Ordering::Acquire));
            }
            self.consumer_changed.wait(&mut guard);
        }
    }

    fn release_batch(&self, logical: usize) -> Result<()> {
        let expected = self.consume_idx.load(Ordering::Acquire);
        if logical != expected {
            return Err(Error::Invariant(format!(
                "consumer release expected {expected}, got {logical}"
            )));
        }
        let slot = ring_slot(&self.plan, logical);
        if self.batch_slots[slot].generation.load(Ordering::Acquire) != logical
            || !self.batch_slots[slot].ready.swap(false, Ordering::AcqRel)
        {
            return Err(Error::Invariant(
                "released output generation is not ready".into(),
            ));
        }
        let next_generation = logical + self.plan.ring_slots;
        if next_generation < self.plan.batch_count {
            self.batch_slots[slot]
                .generation
                .store(next_generation, Ordering::Release);
        }
        self.release_nodes(&self.plan.static_plan.ring_releases, logical)?;
        let next = logical + 1;
        self.consume_idx.store(next, Ordering::Release);
        if next == self.plan.batch_count {
            let _ =
                self.state
                    .compare_exchange(RUNNING, FINISHED, Ordering::AcqRel, Ordering::Acquire);
            self.ready.stop();
            self.wake_consumer();
        }
        Ok(())
    }

    fn wake_consumer(&self) {
        let _guard = self.consumer_lock.lock();
        self.consumer_changed.notify_all();
    }

    fn stats(&self) -> RuntimeStats {
        #[cfg(feature = "profile")]
        let load = |value: &AtomicUsize| value.load(Ordering::Relaxed) as u64;
        RuntimeStats {
            requested_io_mode: self.requested_io_mode,
            actual_io_mode: self.actual_io_mode,
            worker_count: self.config.worker_count,
            max_inflight_jobs_per_worker: self.config.max_inflight_jobs_per_worker,
            max_inflight_encoded_bytes_per_worker: self
                .config
                .max_inflight_encoded_bytes_per_worker,
            max_decoded_bytes_per_worker: self.config.max_decoded_bytes_per_worker,
            #[cfg(feature = "profile")]
            physical_read_ops: load(&self.counters.reads),
            #[cfg(feature = "profile")]
            physical_read_bytes: load(&self.counters.read_bytes),
            #[cfg(feature = "profile")]
            short_read_retries: 0,
            #[cfg(feature = "profile")]
            whole_key_materializations: 0,
            #[cfg(feature = "profile")]
            uring_prepared_read_sqes: uring_read_count(&self.counters),
            #[cfg(feature = "profile")]
            uring_submitted_read_sqes: uring_read_count(&self.counters),
            #[cfg(feature = "profile")]
            uring_submit_calls: uring_read_count(&self.counters),
            #[cfg(feature = "profile")]
            uring_cqes: uring_read_count(&self.counters),
            #[cfg(feature = "profile")]
            uring_cancel_requests: 0,
            #[cfg(feature = "profile")]
            uring_cancel_cqes: 0,
            #[cfg(feature = "profile")]
            io_wait_nanoseconds: 0,
            #[cfg(feature = "profile")]
            decode_nanoseconds: 0,
            #[cfg(feature = "profile")]
            data_decode_nanoseconds: 0,
            #[cfg(feature = "profile")]
            indices_decode_nanoseconds: 0,
            #[cfg(feature = "profile")]
            validation_nanoseconds: 0,
            #[cfg(feature = "profile")]
            scatter_nanoseconds: 0,
            #[cfg(feature = "profile")]
            scatter_kernel_nanoseconds: 0,
            #[cfg(feature = "profile")]
            completion_nanoseconds: 0,
            #[cfg(feature = "profile")]
            window_wait_nanoseconds: 0,
            #[cfg(feature = "profile")]
            consumer_wait_nanoseconds: 0,
            #[cfg(feature = "profile")]
            completed_jobs: load(&self.counters.completed_jobs),
            #[cfg(feature = "profile")]
            completed_cells: load(&self.counters.completed_cells),
            #[cfg(feature = "profile")]
            decoded_blocks: load(&self.counters.decoded_blocks),
            #[cfg(feature = "profile")]
            decoded_bytes: load(&self.counters.decoded_bytes),
            #[cfg(feature = "profile")]
            claim_cas_retries: 0,
            #[cfg(feature = "profile")]
            window_block_events: 0,
            #[cfg(feature = "profile")]
            local_full_events: 0,
            #[cfg(feature = "profile")]
            peak_inflight_jobs: 0,
            #[cfg(feature = "profile")]
            peak_inflight_read_ops: 0,
            #[cfg(feature = "profile")]
            peak_inflight_encoded_bytes: 0,
            #[cfg(feature = "profile")]
            workers: (0..self.config.worker_count)
                .map(|worker_id| WorkerRuntimeStats {
                    worker_id,
                    completed_jobs: 0,
                    completed_cells: 0,
                    decoded_blocks: 0,
                    decoded_bytes: 0,
                    data_decode_nanoseconds: 0,
                    indices_decode_nanoseconds: 0,
                    validation_nanoseconds: 0,
                    scatter_kernel_nanoseconds: 0,
                    completion_nanoseconds: 0,
                    window_wait_nanoseconds: 0,
                    io_wait_nanoseconds: 0,
                    physical_read_ops: 0,
                    physical_read_bytes: 0,
                    short_read_retries: 0,
                    whole_key_materializations: 0,
                    claim_cas_retries: 0,
                    window_block_events: 0,
                    local_full_events: 0,
                    uring_prepared_read_sqes: 0,
                    uring_submitted_read_sqes: 0,
                    uring_submit_calls: 0,
                    uring_cqes: 0,
                    uring_cancel_requests: 0,
                    uring_cancel_cqes: 0,
                })
                .collect(),
            state: self.state(),
        }
    }
}

fn run_initialize(inner: &Arc<SessionInner>) -> Result<()> {
    let range = inner.plan.static_plan.initialize.io_tasks.clone();
    if range.is_empty() {
        return Ok(());
    }
    let next = AtomicUsize::new(range.start);
    let failed = AtomicBool::new(false);
    let error = Mutex::new(None::<Error>);
    let maximum_task_bytes = inner.plan.static_plan.io_decode_tasks[range.clone()]
        .iter()
        .map(|task| task.file_len)
        .max()
        .unwrap_or(0);
    if maximum_task_bytes > inner.config.initialize_inflight_encoded_bytes {
        return Err(Error::ResourceLimit(format!(
            "InitializeJob has a {maximum_task_bytes}-byte I/O task, encoded in-flight limit is {}",
            inner.config.initialize_inflight_encoded_bytes
        )));
    }
    let encoded_worker_limit = inner
        .config
        .initialize_inflight_encoded_bytes
        .checked_div(maximum_task_bytes)
        .unwrap_or(inner.config.initialize_workers);
    let worker_count = inner
        .config
        .initialize_workers
        .min(range.end - range.start)
        .min(inner.config.initialize_inflight_io_ops)
        .min(encoded_worker_limit.max(1));
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| {
                let mut encoded = Vec::new();
                let mut workspace = DecodeWorkspace::new();
                loop {
                    if failed.load(Ordering::Acquire) {
                        return;
                    }
                    let node = next.fetch_add(1, Ordering::Relaxed);
                    if node >= range.end {
                        return;
                    }
                    let state = &inner.nodes[node];
                    if state
                        .state
                        .compare_exchange(
                            NODE_WAITING,
                            NODE_RUNNING,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_err()
                    {
                        if !failed.swap(true, Ordering::AcqRel) {
                            *error.lock() = Some(Error::Invariant(
                                "initialize node was scheduled more than once".into(),
                            ));
                        }
                        return;
                    }
                    let result = inner.execute_io(node, &mut encoded, &mut workspace);
                    match result {
                        Ok(()) => state.state.store(NODE_DONE, Ordering::Release),
                        Err(value) => {
                            if !failed.swap(true, Ordering::AcqRel) {
                                *error.lock() = Some(value);
                            }
                            return;
                        }
                    }
                }
            });
        }
    });
    if let Some(error) = error.into_inner() {
        inner.fail(error.clone());
        return Err(error);
    }
    Ok(())
}

fn worker_entry(inner: Arc<SessionInner>, mut io: WorkerIo) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut encoded = Vec::new();
        let mut workspace = DecodeWorkspace::new();
        while let Some(node) = inner.ready.pop() {
            if !inner.is_running() {
                return Ok(());
            }
            let state = &inner.nodes[node];
            if state
                .state
                .compare_exchange(
                    NODE_READY,
                    NODE_RUNNING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                return Err(Error::Invariant("ready node state is invalid".into()));
            }
            inner.execute(node, &mut io, &mut encoded, &mut workspace)?;
            state.state.store(NODE_DONE, Ordering::Release);
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => inner.fail(error),
        Err(_) => inner.fail(Error::WorkerPanic),
    }
}

#[cfg(all(feature = "uring", target_os = "linux"))]
fn read_exact_uring(
    source: &ReadSource,
    offset: u64,
    len: usize,
    output: &mut Vec<u8>,
    ring: &mut io_uring::IoUring,
) -> Result<usize> {
    use io_uring::{opcode, types};

    let ReadSource::Positioned {
        file,
        base_offset,
        view_len,
    } = source
    else {
        return Err(Error::Invariant(
            "io_uring task does not reference a positioned source".into(),
        ));
    };
    let end = offset
        .checked_add(len as u64)
        .ok_or_else(|| Error::StalePlan("io_uring range overflow".into()))?;
    if end > *view_len {
        return Err(Error::StalePlan(
            "io_uring range exceeds positioned source".into(),
        ));
    }
    let absolute = base_offset
        .checked_add(offset)
        .ok_or_else(|| Error::StalePlan("io_uring absolute offset overflow".into()))?;
    output.clear();
    output.try_reserve_exact(len)?;
    let mut filled = 0usize;
    let mut operations = 0usize;
    while filled < len {
        let request_len = (len - filled).min(u32::MAX as usize);
        let request_offset = absolute
            .checked_add(filled as u64)
            .ok_or_else(|| Error::StalePlan("io_uring read offset overflow".into()))?;
        // SAFETY: the vector keeps this allocation pinned until the matching
        // CQE below and exposes at least `request_len` spare bytes.
        let pointer = unsafe {
            output
                .spare_capacity_mut()
                .as_mut_ptr()
                .add(filled)
                .cast::<u8>()
        };
        let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), pointer, request_len as u32)
            .offset(request_offset)
            .build();
        // SAFETY: the SQE's buffer remains live and unmoved until submit_and_wait
        // returns a completion, and this worker is its sole owner.
        unsafe {
            ring.submission()
                .push(&entry)
                .map_err(|_| Error::Invariant("io_uring submission queue is full".into()))?;
        }
        loop {
            match ring.submit_and_wait(1) {
                Ok(_) => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        let completion = ring
            .completion()
            .next()
            .ok_or_else(|| Error::Invariant("io_uring produced no completion".into()))?;
        let result = completion.result();
        operations = operations.saturating_add(1);
        if result < 0 {
            return Err(std::io::Error::from_raw_os_error(-result).into());
        }
        let read = result as usize;
        if read == 0 {
            return Err(Error::Io {
                kind: std::io::ErrorKind::UnexpectedEof,
                message: format!("io_uring read ended at {filled} of {len} bytes"),
            });
        }
        if read > request_len {
            return Err(Error::Invariant(
                "io_uring completion exceeds submitted length".into(),
            ));
        }
        filled += read;
    }
    // SAFETY: every byte in the logical length was initialized by successful
    // read CQEs and the vector was not reallocated while requests were live.
    unsafe { output.set_len(len) };
    Ok(operations)
}

fn read_exact_source(
    source: &ReadSource,
    offset: u64,
    len: usize,
    output: &mut Vec<u8>,
) -> Result<()> {
    match source {
        ReadSource::Empty => Err(Error::Invariant("cache load uses an empty source".into())),
        ReadSource::Positioned {
            file,
            base_offset,
            view_len,
        } => {
            let end = offset
                .checked_add(len as u64)
                .ok_or_else(|| Error::StalePlan("positioned range overflow".into()))?;
            if end > *view_len {
                return Err(Error::StalePlan("positioned range exceeds source".into()));
            }
            output.clear();
            output.try_reserve_exact(len)?;
            let absolute = base_offset
                .checked_add(offset)
                .ok_or_else(|| Error::StalePlan("absolute offset overflow".into()))?;
            let mut filled = 0usize;
            while filled < len {
                let read_offset = absolute
                    .checked_add(filled as u64)
                    .ok_or_else(|| Error::StalePlan("read offset overflow".into()))?;
                let read = {
                    let spare = &mut output.spare_capacity_mut()[..len - filled];
                    match rustix::io::pread(file, spare, read_offset) {
                        Ok((initialized, _)) => initialized.len(),
                        Err(error) if error == rustix::io::Errno::INTR => continue,
                        Err(error) => return Err(std::io::Error::from(error).into()),
                    }
                };
                if read == 0 {
                    return Err(Error::Io {
                        kind: std::io::ErrorKind::UnexpectedEof,
                        message: format!("positioned read ended at {filled} of {len} bytes"),
                    });
                }
                filled += read;
                // SAFETY: rustix initialized exactly the reported spare prefix.
                unsafe { output.set_len(filled) };
            }
            Ok(())
        }
        ReadSource::RangeKey {
            store,
            key,
            declared_len,
        } => {
            let end = usize::try_from(offset)
                .ok()
                .and_then(|start| start.checked_add(len))
                .ok_or_else(|| Error::StalePlan("range-key extent overflow".into()))?;
            if end > *declared_len {
                return Err(Error::StalePlan(format!(
                    "range key '{key}' exceeds declared length"
                )));
            }
            store.read_range_into(key, offset, len, output)?;
            if output.len() != len {
                return Err(Error::StalePlan("range-key short read".into()));
            }
            Ok(())
        }
        ReadSource::WholeKey {
            store,
            key,
            declared_len,
            cached,
        } => {
            let start = usize::try_from(offset)
                .map_err(|_| Error::StalePlan("whole-key offset exceeds usize".into()))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| Error::StalePlan("whole-key extent overflow".into()))?;
            if end > *declared_len {
                return Err(Error::StalePlan("whole-key extent exceeds source".into()));
            }
            if let Some(cached) = cached {
                output.clear();
                output.try_reserve_exact(len)?;
                output.extend_from_slice(cached.get(start..end).ok_or_else(|| {
                    Error::StalePlan("cached whole-key extent is invalid".into())
                })?);
                return Ok(());
            }
            store.read_range_into(key, 0, *declared_len, output)?;
            if output.len() != *declared_len {
                return Err(Error::StalePlan("whole-key short read".into()));
            }
            output.copy_within(start..end, 0);
            output.truncate(len);
            Ok(())
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

fn ring_slot(plan: &PlanData, logical: usize) -> usize {
    if plan.ring_mask != usize::MAX {
        logical & plan.ring_mask
    } else {
        logical % plan.ring_slots
    }
}
