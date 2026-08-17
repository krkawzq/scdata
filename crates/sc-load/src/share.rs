//! Shared-memory output ring and futex IPC for multi-rank consumers.
//!
//! The standard [`crate::Plan::open`] path is unchanged. This module allocates a
//! sealed memfd-backed `[control | ring]` mapping, runs session workers against
//! the ring, and publishes ready ring generations to rank-local consumers.

use std::mem::ManuallyDrop;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use rustix::fs::{
    fcntl_add_seals, fcntl_get_seals, ftruncate, memfd_create, MemfdFlags, SealFlags,
};
use rustix::io::dup;
use rustix::mm::{madvise, mmap, mprotect, munmap, Advice, MapFlags, MprotectFlags, ProtFlags};
use rustix::thread::futex;

use crate::dtype::OutputDType;
use crate::plan::Plan;
use crate::session::{AlignedBuffer, CancellationHandle, RuntimeStats, Session, SessionState};
use crate::{Error, Result, SessionConfig};

const MAGIC: u64 = u64::from_le_bytes(*b"SCSHARE6");
const VERSION: u32 = 6;
const CONTROL_ALIGNMENT: usize = 64;
pub const DEFAULT_MAX_SHARED_CONTROL_BYTES: usize = 64 * 1024 * 1024;
const ABSOLUTE_MAX_CONTROL_BYTES: usize = 1024 * 1024 * 1024;
const UNPUBLISHED: u64 = u64::MAX;
const FUTEX_RECHECK_TIMEOUT: futex::Timespec = futex::Timespec {
    tv_sec: 1,
    tv_nsec: 0,
};

const STATE_RUNNING: u32 = 0;
const STATE_FAILED: u32 = 1;
const STATE_CANCELLED: u32 = 2;
const STATE_FINISHED: u32 = 3;

/// Configuration for a shared-ring producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedConfig {
    pub world_size: usize,
    pub max_control_bytes: usize,
}

impl SharedConfig {
    pub fn new(world_size: usize) -> Result<Self> {
        let config = Self {
            world_size,
            max_control_bytes: DEFAULT_MAX_SHARED_CONTROL_BYTES,
        };
        config.validate()?;
        Ok(config)
    }

    /// Set the hard limit for shared rank and ring-generation metadata.
    pub fn with_max_control_bytes(mut self, max_control_bytes: usize) -> Result<Self> {
        self.max_control_bytes = max_control_bytes;
        self.validate()?;
        Ok(self)
    }

    fn validate(self) -> Result<()> {
        if self.world_size == 0 {
            return Err(Error::InvalidConfig(
                "shared world_size must be positive".into(),
            ));
        }
        u32::try_from(self.world_size)
            .map_err(|_| Error::InvalidConfig("shared world_size exceeds u32".into()))?;
        if self.max_control_bytes == 0 {
            return Err(Error::InvalidConfig(
                "shared max_control_bytes must be positive".into(),
            ));
        }
        if self.max_control_bytes > ABSOLUTE_MAX_CONTROL_BYTES {
            return Err(Error::InvalidConfig(format!(
                "shared max_control_bytes {} exceeds the absolute limit {}",
                self.max_control_bytes, ABSOLUTE_MAX_CONTROL_BYTES
            )));
        }
        Ok(())
    }
}

#[repr(C)]
struct SharedHeader {
    magic: u64,
    version: u32,
    world_size: u32,
    dtype: u32,
    producer_pid: u32,
    producer_start_time: u64,
    n_rows: u64,
    n_cols: u64,
    batch_size: u64,
    batch_count: u64,
    ring_slots: u64,
    row_stride: u64,
    ring_offset: u64,
    ring_bytes: u64,
    /// Low 32 bits are state; high 32 bits are a stable producer error code.
    terminal: AtomicU64,
}

#[repr(C, align(64))]
struct RankControl {
    /// Non-zero while one client process owns this rank.
    owner: AtomicU64,
    /// Linux process start time for the current owner token.
    owner_start_time: AtomicU64,
    /// First rank-assigned logical batch not yet released.
    resume_logical: AtomicU64,
    ready_futex: AtomicU32,
    release_futex: AtomicU32,
    ready_waiting: AtomicU32,
    release_waiting: AtomicU32,
}

#[repr(C, align(64))]
struct RingControl {
    ready_generation: AtomicU64,
    released_generation: AtomicU64,
}

const HEADER_SIZE: usize = std::mem::size_of::<SharedHeader>();
const RANK_CONTROL_SIZE: usize = std::mem::size_of::<RankControl>();
const RING_CONTROL_SIZE: usize = std::mem::size_of::<RingControl>();

#[derive(Debug, Clone, Copy)]
struct SharedLayout {
    world_size: usize,
    n_rows: usize,
    n_cols: usize,
    batch_size: usize,
    batch_count: usize,
    ring_slots: usize,
    world_mask: usize,
    ring_mask: usize,
    row_stride: usize,
    rank_offset: usize,
    ring_control_offset: usize,
    control_bytes: usize,
    ring_bytes: usize,
    total_bytes: usize,
    dtype: OutputDType,
}

impl SharedLayout {
    fn for_plan(plan: &Plan, config: SharedConfig) -> Result<Self> {
        config.validate()?;
        let page_size = page_size()?;
        let world_size = config.world_size;
        let n_rows = plan.stats().input_rows;
        let n_cols = plan.output_spec().n_cols();
        let batch_size = plan.batch_size();
        let batch_count = plan.batch_count();
        let ring_slots = plan.inner.ring_slots;
        let row_stride = plan.row_stride_bytes();
        let (rank_offset, ring_control_offset, control_bytes) =
            control_layout(world_size, ring_slots, page_size)?;
        if control_bytes > config.max_control_bytes {
            return Err(Error::ResourceLimit(format!(
                "shared control region has {control_bytes} bytes, limit is {}",
                config.max_control_bytes
            )));
        }
        let ring_bytes = expected_ring_bytes(ring_slots, batch_size, row_stride)?;
        if ring_bytes != plan.stats().output_ring_bytes {
            return Err(Error::Invariant(format!(
                "shared ring calculation produced {ring_bytes} bytes, plan requires {}",
                plan.stats().output_ring_bytes
            )));
        }
        validate_batch_layout(n_rows, batch_size, batch_count, ring_slots)?;
        let total_bytes = mapping_bytes(control_bytes, ring_bytes, page_size)?;
        Ok(Self {
            world_size,
            n_rows,
            n_cols,
            batch_size,
            batch_count,
            ring_slots,
            world_mask: power_of_two_mask(world_size),
            ring_mask: power_of_two_mask(ring_slots),
            row_stride,
            rank_offset,
            ring_control_offset,
            control_bytes,
            ring_bytes,
            total_bytes,
            dtype: plan.output_spec().dtype(),
        })
    }

    fn from_header(header: &SharedHeader, file_bytes: usize) -> Result<Self> {
        if header.magic != MAGIC || header.version != VERSION {
            return Err(Error::InvalidDataset(
                "shared ring magic/version mismatch".into(),
            ));
        }
        let page_size = page_size()?;
        let world_size = usize::try_from(header.world_size)
            .map_err(|_| Error::InvalidDataset("shared world_size does not fit usize".into()))?;
        if world_size == 0 {
            return Err(Error::InvalidDataset(
                "shared world_size must be positive".into(),
            ));
        }
        let n_rows = u64_to_usize(header.n_rows, "n_rows")?;
        let n_cols = u64_to_usize(header.n_cols, "n_cols")?;
        let batch_size = u64_to_usize(header.batch_size, "batch_size")?;
        let batch_count = u64_to_usize(header.batch_count, "batch_count")?;
        let ring_slots = u64_to_usize(header.ring_slots, "ring_slots")?;
        let row_stride = u64_to_usize(header.row_stride, "row_stride")?;
        let ring_bytes = u64_to_usize(header.ring_bytes, "ring_bytes")?;
        let dtype = dtype_from_code(header.dtype)?;
        validate_process_id(header.producer_pid, "producer")?;
        if header.producer_start_time == 0 {
            return Err(Error::InvalidDataset(
                "shared producer process start time must be positive".into(),
            ));
        }
        match unpack_terminal(header.terminal.load(Ordering::Acquire)).0 {
            STATE_RUNNING | STATE_FAILED | STATE_CANCELLED | STATE_FINISHED => {}
            state => {
                return Err(Error::InvalidDataset(format!(
                    "shared terminal state {state} is invalid"
                )))
            }
        }
        validate_batch_layout(n_rows, batch_size, batch_count, ring_slots)?;

        let row_bytes = n_cols
            .checked_mul(dtype.size())
            .ok_or_else(|| Error::InvalidDataset("shared logical row size overflow".into()))?;
        let expected_stride = align_up(row_bytes, CONTROL_ALIGNMENT)?;
        if row_stride != expected_stride {
            return Err(Error::InvalidDataset(format!(
                "shared row_stride {row_stride} does not match expected {expected_stride}"
            )));
        }
        let expected_ring = expected_ring_bytes(ring_slots, batch_size, row_stride)?;
        if ring_bytes != expected_ring {
            return Err(Error::InvalidDataset(format!(
                "shared ring_bytes {ring_bytes} does not match expected {expected_ring}"
            )));
        }

        let (rank_offset, ring_control_offset, control_bytes) =
            control_layout(world_size, ring_slots, page_size)?;
        if control_bytes > ABSOLUTE_MAX_CONTROL_BYTES {
            return Err(Error::InvalidDataset(format!(
                "shared control region has {control_bytes} bytes, absolute limit is {ABSOLUTE_MAX_CONTROL_BYTES}"
            )));
        }
        let encoded_control = u64_to_usize(header.ring_offset, "ring_offset")?;
        if encoded_control != control_bytes {
            return Err(Error::InvalidDataset(format!(
                "shared ring_offset {encoded_control} does not match expected {control_bytes}"
            )));
        }
        let total_bytes = mapping_bytes(control_bytes, ring_bytes, page_size)?;
        if total_bytes != file_bytes {
            return Err(Error::InvalidDataset(format!(
                "shared mapping has {file_bytes} bytes, expected {total_bytes}"
            )));
        }
        Ok(Self {
            world_size,
            n_rows,
            n_cols,
            batch_size,
            batch_count,
            ring_slots,
            world_mask: power_of_two_mask(world_size),
            ring_mask: power_of_two_mask(ring_slots),
            row_stride,
            rank_offset,
            ring_control_offset,
            control_bytes,
            ring_bytes,
            total_bytes,
            dtype,
        })
    }

    fn batch_rows(self, logical: usize) -> Result<usize> {
        if logical >= self.batch_count {
            return Err(Error::InvalidInput(format!(
                "logical batch {logical} is outside batch_count {}",
                self.batch_count
            )));
        }
        let start = logical
            .checked_mul(self.batch_size)
            .ok_or_else(|| Error::Invariant("shared batch row offset overflow".into()))?;
        Ok((self.n_rows - start).min(self.batch_size))
    }

    fn next_for_rank(self, logical: usize) -> usize {
        logical
            .checked_add(self.world_size)
            .unwrap_or(self.batch_count)
            .min(self.batch_count)
    }

    #[inline(always)]
    fn rank_for(self, logical: usize) -> usize {
        if self.world_mask != usize::MAX {
            logical & self.world_mask
        } else {
            logical % self.world_size
        }
    }

    #[inline(always)]
    fn ring_slot(self, logical: usize) -> usize {
        if self.ring_mask != usize::MAX {
            logical & self.ring_mask
        } else {
            logical % self.ring_slots
        }
    }

    fn rank_batch_count(self, rank: usize) -> usize {
        if rank >= self.batch_count {
            0
        } else {
            (self.batch_count - 1 - rank) / self.world_size + 1
        }
    }
}

fn power_of_two_mask(value: usize) -> usize {
    if value.is_power_of_two() {
        value - 1
    } else {
        usize::MAX
    }
}

fn page_size() -> Result<usize> {
    let page_size = rustix::param::page_size();
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err(Error::Invariant(format!(
            "kernel page size {page_size} is invalid"
        )));
    }
    Ok(page_size)
}

fn align_up(value: usize, align: usize) -> Result<usize> {
    debug_assert!(align.is_power_of_two());
    value
        .checked_add(align - 1)
        .map(|rounded| rounded & !(align - 1))
        .ok_or_else(|| Error::Allocation("shared mapping size overflow".into()))
}

fn control_layout(
    world_size: usize,
    ring_slots: usize,
    page_size: usize,
) -> Result<(usize, usize, usize)> {
    let rank_offset = align_up(HEADER_SIZE, CONTROL_ALIGNMENT)?;
    let rank_bytes = world_size
        .checked_mul(RANK_CONTROL_SIZE)
        .ok_or_else(|| Error::Allocation("shared rank control region overflow".into()))?;
    let ring_control_offset = align_up(
        rank_offset
            .checked_add(rank_bytes)
            .ok_or_else(|| Error::Allocation("shared control region overflow".into()))?,
        CONTROL_ALIGNMENT,
    )?;
    let ring_control_bytes = ring_slots
        .checked_mul(RING_CONTROL_SIZE)
        .ok_or_else(|| Error::Allocation("shared ring control region overflow".into()))?;
    let raw = ring_control_offset
        .checked_add(ring_control_bytes)
        .ok_or_else(|| Error::Allocation("shared control region overflow".into()))?;
    Ok((rank_offset, ring_control_offset, align_up(raw, page_size)?))
}

fn expected_ring_bytes(ring_slots: usize, batch_size: usize, row_stride: usize) -> Result<usize> {
    ring_slots
        .checked_mul(batch_size)
        .and_then(|rows| rows.checked_mul(row_stride))
        .ok_or_else(|| Error::Allocation("shared ring byte length overflow".into()))
}

fn mapping_bytes(control_bytes: usize, ring_bytes: usize, page_size: usize) -> Result<usize> {
    let raw = control_bytes
        .checked_add(ring_bytes)
        .ok_or_else(|| Error::Allocation("shared mapping total size overflow".into()))?;
    let total = align_up(raw.max(control_bytes), page_size)?;
    if total > isize::MAX as usize {
        return Err(Error::Allocation(format!(
            "shared mapping has {total} bytes, exceeding the addressable slice limit {}",
            isize::MAX
        )));
    }
    Ok(total)
}

fn validate_batch_layout(
    n_rows: usize,
    batch_size: usize,
    batch_count: usize,
    ring_slots: usize,
) -> Result<()> {
    if batch_size == 0 {
        return Err(Error::InvalidDataset(
            "shared batch_size must be positive".into(),
        ));
    }
    let expected_batches = n_rows.div_ceil(batch_size);
    if batch_count != expected_batches {
        return Err(Error::InvalidDataset(format!(
            "shared batch_count {batch_count} does not match {n_rows} rows at batch_size {batch_size}"
        )));
    }
    if batch_count == u64::MAX as usize {
        return Err(Error::InvalidDataset(
            "shared batch_count exhausts the generation sentinel".into(),
        ));
    }
    if (batch_count == 0 && ring_slots != 0)
        || (batch_count > 0 && (ring_slots == 0 || ring_slots > batch_count))
    {
        return Err(Error::InvalidDataset(format!(
            "shared ring_slots {ring_slots} is invalid for batch_count {batch_count}"
        )));
    }
    Ok(())
}

fn u64_to_usize(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidDataset(format!("shared {name} does not fit usize")))
}

fn dtype_code(dtype: OutputDType) -> u32 {
    match dtype {
        OutputDType::I16 => 0,
        OutputDType::I32 => 1,
        OutputDType::U16 => 2,
        OutputDType::U32 => 3,
        OutputDType::F32 => 4,
        OutputDType::F64 => 5,
        OutputDType::I64 => 6,
        OutputDType::U64 => 7,
    }
}

fn dtype_from_code(code: u32) -> Result<OutputDType> {
    match code {
        0 => Ok(OutputDType::I16),
        1 => Ok(OutputDType::I32),
        2 => Ok(OutputDType::U16),
        3 => Ok(OutputDType::U32),
        4 => Ok(OutputDType::F32),
        5 => Ok(OutputDType::F64),
        6 => Ok(OutputDType::I64),
        7 => Ok(OutputDType::U64),
        _ => Err(Error::InvalidDataset(format!(
            "shared ring dtype code {code} is unsupported"
        ))),
    }
}

fn futex_wait(word: &AtomicU32, expected: u32) -> Result<()> {
    match futex::wait(
        word,
        futex::Flags::empty(),
        expected,
        Some(&FUTEX_RECHECK_TIMEOUT),
    ) {
        Ok(())
        | Err(rustix::io::Errno::AGAIN)
        | Err(rustix::io::Errno::INTR)
        | Err(rustix::io::Errno::TIMEDOUT) => Ok(()),
        Err(error) => Err(Error::Io {
            kind: std::io::Error::from(error).kind(),
            message: error.to_string(),
        }),
    }
}

fn futex_wake(word: &AtomicU32) {
    // The protocol admits one rank owner and one producer, so each direction
    // has at most one legitimate waiter.
    let _ = futex::wake(word, futex::Flags::empty(), 1);
}

fn signal_futex_waiter(word: &AtomicU32, waiting: &AtomicU32) {
    if waiting.swap(0, Ordering::AcqRel) != 0 {
        word.fetch_add(1, Ordering::Release);
        futex_wake(word);
    }
}

fn pack_terminal(state: u32, error_code: u32) -> u64 {
    u64::from(state) | (u64::from(error_code) << 32)
}

fn unpack_terminal(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

fn shared_error_code(error: &Error) -> u32 {
    match error {
        Error::InvalidConfig(_) => 1,
        Error::InvalidInput(_) => 2,
        Error::InvalidDataset(_) => 3,
        Error::ResourceLimit(_) => 4,
        Error::StalePlan(_) => 5,
        Error::Unsupported(_) => 6,
        Error::Io { .. } => 7,
        Error::Decode(_) => 8,
        Error::Promote(_) => 9,
        Error::Conversion(_) => 10,
        Error::Cancelled => 11,
        Error::Session(inner) => shared_error_code(inner),
        Error::WorkerPanic => 12,
        Error::Allocation(_) => 13,
        Error::Invariant(_) => 14,
    }
}

fn producer_error(code: u32) -> Error {
    Error::Session(Arc::new(Error::Invariant(format!(
        "shared producer failed (error code {code}); inspect the producer result for details"
    ))))
}

fn validate_process_id(process_id: u32, role: &str) -> Result<rustix::process::Pid> {
    let raw = i32::try_from(process_id)
        .map_err(|_| Error::InvalidDataset(format!("shared {role} pid exceeds pid_t")))?;
    rustix::process::Pid::from_raw(raw)
        .ok_or_else(|| Error::InvalidDataset(format!("shared {role} pid must be positive")))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProcessStat {
    state: u8,
    start_time: u64,
}

fn read_process_stat(process_id: u32) -> std::io::Result<ProcessStat> {
    let stat = std::fs::read_to_string(format!("/proc/{process_id}/stat"))?;
    let (_, fields) = stat.rsplit_once(')').ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("/proc/{process_id}/stat has no command terminator"),
        )
    })?;
    let fields = fields.split_whitespace().collect::<Vec<_>>();
    let state = fields
        .first()
        .and_then(|field| field.as_bytes().first())
        .copied()
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("/proc/{process_id}/stat has no process state"),
            )
        })?;
    let start_time = fields
        .get(19)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("/proc/{process_id}/stat has no start time"),
            )
        })?
        .parse::<u64>()
        .map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("/proc/{process_id}/stat has an invalid start time: {error}"),
            )
        })?;
    Ok(ProcessStat { state, start_time })
}

fn current_process_start_time(role: &str) -> Result<u64> {
    let process_id = std::process::id();
    let start_time = read_process_stat(process_id).map_err(|error| Error::Io {
        kind: error.kind(),
        message: format!("failed to read {role} process identity: {error}"),
    })?;
    if start_time.start_time == 0 {
        return Err(Error::Invariant(format!(
            "shared {role} process start time must be positive"
        )));
    }
    Ok(start_time.start_time)
}

fn process_is_alive(process_id: u32, expected_start_time: Option<u64>) -> bool {
    process_is_alive_matching(process_id, |stat| {
        expected_start_time.is_none_or(|expected| stat.start_time == expected)
    })
}

fn process_is_alive_matching(
    process_id: u32,
    identity_matches: impl FnOnce(ProcessStat) -> bool,
) -> bool {
    let Ok(pid) = validate_process_id(process_id, "owner") else {
        return false;
    };
    match rustix::process::test_kill_process(pid) {
        Ok(()) | Err(rustix::io::Errno::PERM) => {}
        Err(rustix::io::Errno::SRCH) => return false,
        Err(_) => return true,
    };
    match read_process_stat(process_id) {
        Ok(stat) => !matches!(stat.state, b'Z' | b'X') && identity_matches(stat),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => true,
    }
}

fn process_start_fingerprint(start_time: u64) -> u32 {
    (start_time as u32) ^ ((start_time >> 32) as u32).rotate_left(13)
}

fn client_token(process_id: u32, process_start_time: u64) -> u64 {
    (u64::from(process_id) << 32) | u64::from(process_start_fingerprint(process_start_time))
}

fn owner_process_is_alive(token: u64, published_start_time: u64) -> bool {
    if token == 0 {
        return false;
    }
    let process_id = (token >> 32) as u32;
    let fingerprint = token as u32;
    if published_start_time != 0 && process_start_fingerprint(published_start_time) != fingerprint {
        return false;
    }
    process_is_alive_matching(process_id, |stat| {
        if published_start_time == 0 {
            process_start_fingerprint(stat.start_time) == fingerprint
        } else {
            stat.start_time == published_start_time
        }
    })
}

struct SharedMapping {
    producer_fd: Option<OwnedFd>,
    base: NonNull<u8>,
    layout: SharedLayout,
}

// SAFETY: layout metadata is copied and validated before sharing. Control-plane
// mutation uses process-shared atomics, and ring bytes are immutable for the
// lifetime of every published generation lease.
unsafe impl Send for SharedMapping {}
// SAFETY: see `Send`; no mutable Rust reference to the mapped controls or ring
// is exposed after initialization.
unsafe impl Sync for SharedMapping {}

impl SharedMapping {
    fn create(plan: &Plan, config: SharedConfig) -> Result<Self> {
        let layout = SharedLayout::for_plan(plan, config)?;
        let fd = memfd_create(
            "sc-load-shared-ring",
            MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
        )
        .map_err(|error| Error::Allocation(error.to_string()))?;
        ftruncate(
            &fd,
            u64::try_from(layout.total_bytes)
                .map_err(|_| Error::Allocation("shared mapping exceeds u64 length".into()))?,
        )
        .map_err(|error| Error::Allocation(error.to_string()))?;
        fcntl_add_seals(&fd, SealFlags::SHRINK | SealFlags::GROW | SealFlags::SEAL)
            .map_err(|error| Error::Allocation(format!("failed to seal shared memfd: {error}")))?;

        let base = map_shared(fd.as_fd(), layout.total_bytes)?;
        let mapping = Self {
            producer_fd: Some(fd),
            base,
            layout,
        };
        mapping.initialize()?;
        if mapping.layout.ring_bytes > 0 {
            // SAFETY: the ring starts on a page boundary and the kernel rounds
            // the advised tail within this mapping. This is only a placement hint.
            let _ = unsafe {
                madvise(
                    mapping.ring_pointer().as_ptr().cast(),
                    mapping.layout.ring_bytes,
                    Advice::LinuxHugepage,
                )
            };
        }
        Ok(mapping)
    }

    fn attach(fd: BorrowedFd<'_>) -> Result<Self> {
        let required_seals = SealFlags::SHRINK | SealFlags::GROW | SealFlags::SEAL;
        let seals = fcntl_get_seals(fd).map_err(|error| {
            Error::InvalidDataset(format!("shared descriptor is not a sealed memfd: {error}"))
        })?;
        if !seals.contains(required_seals) {
            return Err(Error::InvalidDataset(
                "shared memfd is missing size/seal immutability seals".into(),
            ));
        }
        let stat = rustix::fs::fstat(fd).map_err(|error| Error::Io {
            kind: std::io::Error::from(error).kind(),
            message: error.to_string(),
        })?;
        let total_bytes = usize::try_from(stat.st_size)
            .map_err(|_| Error::InvalidDataset("shared ring size does not fit usize".into()))?;
        if total_bytes < HEADER_SIZE || total_bytes > isize::MAX as usize {
            return Err(Error::InvalidDataset(format!(
                "shared ring mapping size {total_bytes} is invalid"
            )));
        }
        let base = map_shared(fd, total_bytes)?;
        // SAFETY: the mapping contains at least HEADER_SIZE aligned bytes. The
        // producer initializes this header before any descriptor is exported.
        let header = unsafe { &*base.as_ptr().cast::<SharedHeader>() };
        let layout = match SharedLayout::from_header(header, total_bytes) {
            Ok(layout) => layout,
            Err(error) => {
                // SAFETY: `base` is the exact mapping returned above.
                let _ = unsafe { munmap(base.as_ptr().cast(), total_bytes) };
                return Err(error);
            }
        };
        let mapping = Self {
            producer_fd: None,
            base,
            layout,
        };
        if layout.total_bytes > layout.control_bytes {
            // Clients mutate only control atomics. Protect the ring view itself
            // against accidental or unsafe writes in the consumer process.
            // SAFETY: control_bytes is page-aligned and the protected tail is
            // wholly contained in this mapping.
            unsafe {
                mprotect(
                    mapping.ring_pointer().as_ptr().cast(),
                    layout.total_bytes - layout.control_bytes,
                    MprotectFlags::READ,
                )
            }
            .map_err(|error| {
                Error::Allocation(format!("failed to protect shared ring: {error}"))
            })?;
        }
        Ok(mapping)
    }

    fn initialize(&self) -> Result<()> {
        let producer_start_time = current_process_start_time("producer")?;
        let header = SharedHeader {
            magic: MAGIC,
            version: VERSION,
            world_size: u32::try_from(self.layout.world_size)
                .map_err(|_| Error::InvalidConfig("shared world_size exceeds u32".into()))?,
            dtype: dtype_code(self.layout.dtype),
            producer_pid: std::process::id(),
            producer_start_time,
            n_rows: u64::try_from(self.layout.n_rows)
                .map_err(|_| Error::InvalidConfig("shared n_rows exceeds u64".into()))?,
            n_cols: u64::try_from(self.layout.n_cols)
                .map_err(|_| Error::InvalidConfig("shared n_cols exceeds u64".into()))?,
            batch_size: u64::try_from(self.layout.batch_size)
                .map_err(|_| Error::InvalidConfig("shared batch_size exceeds u64".into()))?,
            batch_count: u64::try_from(self.layout.batch_count)
                .map_err(|_| Error::InvalidConfig("shared batch_count exceeds u64".into()))?,
            ring_slots: u64::try_from(self.layout.ring_slots)
                .map_err(|_| Error::InvalidConfig("shared ring_slots exceeds u64".into()))?,
            row_stride: u64::try_from(self.layout.row_stride)
                .map_err(|_| Error::InvalidConfig("shared row_stride exceeds u64".into()))?,
            ring_offset: u64::try_from(self.layout.control_bytes)
                .map_err(|_| Error::Allocation("shared ring_offset exceeds u64".into()))?,
            ring_bytes: u64::try_from(self.layout.ring_bytes)
                .map_err(|_| Error::Allocation("shared ring_bytes exceeds u64".into()))?,
            terminal: AtomicU64::new(pack_terminal(STATE_RUNNING, 0)),
        };
        // SAFETY: producer initialization has exclusive access to this fresh
        // mapping and writes every live control object before exporting the fd.
        unsafe {
            self.base.as_ptr().cast::<SharedHeader>().write(header);
        }
        for rank in 0..self.layout.world_size {
            let resume = rank.min(self.layout.batch_count);
            let control =
                RankControl {
                    owner: AtomicU64::new(0),
                    owner_start_time: AtomicU64::new(0),
                    resume_logical: AtomicU64::new(u64::try_from(resume).map_err(|_| {
                        Error::InvalidConfig("shared rank resume exceeds u64".into())
                    })?),
                    ready_futex: AtomicU32::new(0),
                    release_futex: AtomicU32::new(0),
                    ready_waiting: AtomicU32::new(0),
                    release_waiting: AtomicU32::new(0),
                };
            // SAFETY: offsets are checked against the initialized control layout;
            // every rank object is distinct and 64-byte aligned.
            unsafe {
                self.rank_pointer(rank)?.write(control);
            }
        }
        for slot in 0..self.layout.ring_slots {
            let control = RingControl {
                ready_generation: AtomicU64::new(UNPUBLISHED),
                released_generation: AtomicU64::new(UNPUBLISHED),
            };
            // SAFETY: every slot object is distinct, in bounds, and aligned.
            unsafe {
                self.ring_control_pointer(slot)?.write(control);
            }
        }
        Ok(())
    }

    fn header(&self) -> &SharedHeader {
        // SAFETY: create/attach establishes a live, aligned SharedHeader for the
        // entire mapping lifetime.
        unsafe { &*self.base.as_ptr().cast::<SharedHeader>() }
    }

    fn rank_pointer(&self, rank: usize) -> Result<*mut RankControl> {
        if rank >= self.layout.world_size {
            return Err(Error::InvalidInput(format!(
                "rank {rank} is outside world_size {}",
                self.layout.world_size
            )));
        }
        let offset =
            self.layout
                .rank_offset
                .checked_add(rank.checked_mul(RANK_CONTROL_SIZE).ok_or_else(|| {
                    Error::Invariant("shared rank control offset overflow".into())
                })?)
                .ok_or_else(|| Error::Invariant("shared rank control offset overflow".into()))?;
        if offset + RANK_CONTROL_SIZE > self.layout.ring_control_offset {
            return Err(Error::Invariant(
                "shared rank control exceeds its validated region".into(),
            ));
        }
        // SAFETY: offset is within the mapped control region.
        let pointer = unsafe { self.base.as_ptr().add(offset).cast::<RankControl>() };
        if pointer.align_offset(CONTROL_ALIGNMENT) != 0 {
            return Err(Error::Invariant(format!(
                "shared rank control {rank} is misaligned"
            )));
        }
        Ok(pointer)
    }

    fn rank_control(&self, rank: usize) -> Result<&RankControl> {
        // SAFETY: the producer initialized this object; all later mutation is atomic.
        Ok(unsafe { &*self.rank_pointer(rank)? })
    }

    fn ring_control_pointer(&self, slot: usize) -> Result<*mut RingControl> {
        if slot >= self.layout.ring_slots {
            return Err(Error::InvalidInput(format!(
                "shared ring slot {slot} is outside ring_slots {}",
                self.layout.ring_slots
            )));
        }
        let offset =
            self.layout
                .ring_control_offset
                .checked_add(slot.checked_mul(RING_CONTROL_SIZE).ok_or_else(|| {
                    Error::Invariant("shared ring control offset overflow".into())
                })?)
                .ok_or_else(|| Error::Invariant("shared ring control offset overflow".into()))?;
        if offset + RING_CONTROL_SIZE > self.layout.control_bytes {
            return Err(Error::Invariant(
                "shared ring control exceeds its validated region".into(),
            ));
        }
        // SAFETY: offset is within the mapped control region.
        let pointer = unsafe { self.base.as_ptr().add(offset).cast::<RingControl>() };
        if pointer.align_offset(CONTROL_ALIGNMENT) != 0 {
            return Err(Error::Invariant(format!(
                "shared ring control {slot} is misaligned"
            )));
        }
        Ok(pointer)
    }

    fn ring_control(&self, slot: usize) -> Result<&RingControl> {
        // SAFETY: the producer initialized this object; all later mutation is atomic.
        Ok(unsafe { &*self.ring_control_pointer(slot)? })
    }

    fn control_for_logical(&self, logical: usize) -> Result<&RingControl> {
        if logical >= self.layout.batch_count || self.layout.ring_slots == 0 {
            return Err(Error::InvalidInput(format!(
                "logical batch {logical} is outside batch_count {}",
                self.layout.batch_count
            )));
        }
        self.ring_control(self.layout.ring_slot(logical))
    }

    fn ring_pointer(&self) -> NonNull<u8> {
        // SAFETY: the validated control offset is within (or exactly at the end
        // of) the mapping and remains suitably aligned for every output dtype.
        unsafe { NonNull::new_unchecked(self.base.as_ptr().add(self.layout.control_bytes)) }
    }

    fn ring_bytes_at(&self, offset: usize, len: usize) -> Result<&[u8]> {
        if offset > self.layout.ring_bytes || len > self.layout.ring_bytes - offset {
            return Err(Error::Invariant(
                "shared batch view exceeds ring bounds".into(),
            ));
        }
        // SAFETY: offset/len are checked against the mapped ring extent. The
        // generation remains leased for the lifetime of the returned borrow.
        Ok(unsafe { std::slice::from_raw_parts(self.ring_pointer().as_ptr().add(offset), len) })
    }

    fn terminal(&self) -> (u32, u32) {
        unpack_terminal(self.header().terminal.load(Ordering::Acquire))
    }

    fn set_terminal(&self, state: u32, error_code: u32) -> bool {
        let changed = self
            .header()
            .terminal
            .compare_exchange(
                pack_terminal(STATE_RUNNING, 0),
                pack_terminal(state, error_code),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok();
        if changed {
            self.wake_all();
        }
        changed
    }

    fn wake_all(&self) {
        for rank in 0..self.layout.world_size.min(self.layout.batch_count) {
            if let Ok(control) = self.rank_control(rank) {
                signal_futex_waiter(&control.ready_futex, &control.ready_waiting);
                signal_futex_waiter(&control.release_futex, &control.release_waiting);
            }
        }
    }

    fn advance_rank_resume(&self, rank: usize) -> Result<()> {
        let rank_control = self.rank_control(rank)?;
        loop {
            let current_u64 = rank_control.resume_logical.load(Ordering::Acquire);
            let current = u64_to_usize(current_u64, "rank resume")?;
            if current > self.layout.batch_count {
                return Err(Error::Invariant(format!(
                    "rank {rank} resume logical {current} exceeds batch_count {}",
                    self.layout.batch_count
                )));
            }
            if current == self.layout.batch_count {
                return Ok(());
            }
            if self.layout.rank_for(current) != rank {
                return Err(Error::Invariant(format!(
                    "rank {rank} resume logical {current} has the wrong assignment"
                )));
            }
            let logical_u64 = u64::try_from(current)
                .map_err(|_| Error::Invariant("shared logical generation exceeds u64".into()))?;
            let released = self
                .control_for_logical(current)?
                .released_generation
                .load(Ordering::Acquire);
            if released != logical_u64 {
                return Ok(());
            }
            let next = self.layout.next_for_rank(current);
            let next_u64 = u64::try_from(next)
                .map_err(|_| Error::Invariant("shared rank resume exceeds u64".into()))?;
            let _ = rank_control.resume_logical.compare_exchange(
                current_u64,
                next_u64,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }

    fn producer_is_alive(&self) -> bool {
        let header = self.header();
        process_is_alive(header.producer_pid, Some(header.producer_start_time))
    }

    fn owner_is_alive(&self, rank: usize) -> Result<bool> {
        let control = self.rank_control(rank)?;
        let token = control.owner.load(Ordering::Acquire);
        if token == 0 {
            return Ok(true);
        }
        let start_time = control.owner_start_time.load(Ordering::Acquire);
        Ok(owner_process_is_alive(token, start_time))
    }
}

impl Drop for SharedMapping {
    fn drop(&mut self) {
        // SAFETY: create/attach obtained this exact mapping and length.
        let _ = unsafe { munmap(self.base.as_ptr().cast(), self.layout.total_bytes) };
    }
}

fn map_shared(fd: BorrowedFd<'_>, len: usize) -> Result<NonNull<u8>> {
    // SAFETY: null hint, non-zero page-rounded length, and a size-sealed memfd.
    let pointer = unsafe {
        mmap(
            std::ptr::null_mut(),
            len,
            ProtFlags::READ | ProtFlags::WRITE,
            MapFlags::SHARED | MapFlags::NORESERVE,
            fd,
            0,
        )
    }
    .map_err(|error| Error::Allocation(error.to_string()))?;
    match NonNull::new(pointer.cast::<u8>()) {
        Some(pointer) => Ok(pointer),
        None => {
            // SAFETY: even an address-zero mapping must be released on this
            // exceptional path before its owning descriptor is returned.
            let _ = unsafe { munmap(pointer, len) };
            Err(Error::Allocation("shared mmap returned null".into()))
        }
    }
}

/// Producer that fills a shared output ring and publishes batches to ranks.
///
/// The worker-owning session is dropped explicitly so a post-fork child can
/// discard its inherited handle without joining threads that exist only in the
/// producer process.
pub struct SharedServer {
    session: ManuallyDrop<Session>,
    mapping: Arc<SharedMapping>,
    next_publish: usize,
    process_id: u32,
}

impl SharedServer {
    pub(crate) fn open(
        plan: &Plan,
        session_config: SessionConfig,
        shared: SharedConfig,
    ) -> Result<Self> {
        let mapping = Arc::new(SharedMapping::create(plan, shared)?);
        let output = AlignedBuffer::from_shared(mapping.ring_pointer(), mapping.layout.ring_bytes);
        let session = Session::start_with_output(Arc::clone(&plan.inner), session_config, output)?;
        if session.batch_count() == 0 {
            mapping.set_terminal(STATE_FINISHED, 0);
        }
        Ok(Self {
            session: ManuallyDrop::new(session),
            mapping,
            next_publish: 0,
            process_id: std::process::id(),
        })
    }

    pub fn world_size(&self) -> usize {
        self.mapping.layout.world_size
    }

    pub fn n_rows(&self) -> usize {
        self.mapping.layout.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.mapping.layout.n_cols
    }

    pub fn dtype(&self) -> OutputDType {
        self.mapping.layout.dtype
    }

    pub fn batch_size(&self) -> usize {
        self.mapping.layout.batch_size
    }

    pub fn batch_count(&self) -> usize {
        self.mapping.layout.batch_count
    }

    pub fn row_stride_bytes(&self) -> usize {
        self.mapping.layout.row_stride
    }

    pub fn attach_fd(&self) -> Result<OwnedFd> {
        self.ensure_process()?;
        let descriptor =
            self.mapping.producer_fd.as_ref().ok_or_else(|| {
                Error::Invariant("shared producer mapping has no descriptor".into())
            })?;
        dup(descriptor).map_err(|error| Error::Io {
            kind: std::io::Error::from(error).kind(),
            message: error.to_string(),
        })
    }

    pub fn cancellation_handle(&self) -> SharedCancellationHandle {
        SharedCancellationHandle {
            session: self.session.cancellation_handle(),
            mapping: Arc::clone(&self.mapping),
            process_id: self.process_id,
        }
    }

    pub fn cancel(&self) {
        if std::process::id() != self.process_id {
            return;
        }
        self.session.cancel();
        self.mapping.set_terminal(STATE_CANCELLED, 0);
    }

    /// Drive the producer until every published generation is released.
    pub fn run(self) -> Result<()> {
        self.run_with_stats().0
    }

    /// Drive the producer and retain its final runtime counters for frontends.
    pub fn run_with_stats(mut self) -> (Result<()>, RuntimeStats) {
        if let Err(error) = self.ensure_process() {
            let stats = self.session.stats();
            return (Err(error), stats);
        }
        let mut result = self.run_inner();
        if result.is_ok() {
            if !self.mapping.set_terminal(STATE_FINISHED, 0) {
                let (state, code) = self.mapping.terminal();
                result = match state {
                    STATE_FINISHED => Ok(()),
                    STATE_CANCELLED => Err(Error::Cancelled),
                    STATE_FAILED => Err(producer_error(code)),
                    _ => Err(Error::Invariant(format!(
                        "shared producer ended with unknown state {state}"
                    ))),
                };
            }
        } else if matches!(result, Err(Error::Cancelled)) {
            self.mapping.set_terminal(STATE_CANCELLED, 0);
        } else if let Err(error) = &result {
            self.mapping
                .set_terminal(STATE_FAILED, shared_error_code(error));
        }
        let stats = self.session.stats();
        (result, stats)
    }

    fn ensure_process(&self) -> Result<()> {
        let current = std::process::id();
        if current != self.process_id {
            return Err(Error::InvalidInput(format!(
                "shared producer was opened in process {}, but is being used in process {current}",
                self.process_id
            )));
        }
        Ok(())
    }

    fn run_inner(&mut self) -> Result<()> {
        let batch_count = self.session.batch_count();
        if batch_count == 0 {
            return Ok(());
        }
        let ring_slots = self.mapping.layout.ring_slots;
        loop {
            self.check_external_terminal()?;
            self.drain_releases()?;
            let consumed = self.session.consume_idx();
            if consumed == batch_count {
                return Ok(());
            }
            match self.session.state() {
                SessionState::Failed => return Err(self.session.terminal_error()),
                SessionState::Cancelled => return Err(Error::Cancelled),
                SessionState::Finished => {
                    return Err(Error::Invariant(format!(
                        "shared session finished at batch {consumed} of {batch_count}"
                    )))
                }
                SessionState::Running => {}
            }

            let resident = self.next_publish.checked_sub(consumed).ok_or_else(|| {
                Error::Invariant("shared published cursor precedes consumed cursor".into())
            })?;
            if self.next_publish < batch_count && resident < ring_slots {
                let logical = self.next_publish;
                if !self
                    .session
                    .wait_ready_for(logical, std::time::Duration::from_secs(1))?
                {
                    continue;
                }
                self.check_external_terminal()?;
                self.publish(logical)?;
                self.next_publish += 1;
                continue;
            }
            self.wait_release(consumed)?;
        }
    }

    fn publish(&self, logical: usize) -> Result<()> {
        let logical_u64 = u64::try_from(logical)
            .map_err(|_| Error::Invariant("shared logical generation exceeds u64".into()))?;
        let control = self.mapping.control_for_logical(logical)?;
        if logical >= self.mapping.layout.ring_slots {
            let previous = logical - self.mapping.layout.ring_slots;
            let released = control.released_generation.load(Ordering::Acquire);
            if released != previous as u64 {
                return Err(Error::Invariant(format!(
                    "shared ring slot for logical {logical} still holds unreleased generation {released}"
                )));
            }
        } else if control.ready_generation.load(Ordering::Acquire) != UNPUBLISHED {
            return Err(Error::Invariant(format!(
                "shared ring slot for initial logical {logical} was already published"
            )));
        }
        control
            .ready_generation
            .store(logical_u64, Ordering::Release);
        let rank = self.mapping.layout.rank_for(logical);
        let rank_control = self.mapping.rank_control(rank)?;
        signal_futex_waiter(&rank_control.ready_futex, &rank_control.ready_waiting);
        Ok(())
    }

    fn drain_releases(&mut self) -> Result<()> {
        loop {
            let logical = self.session.consume_idx();
            if logical >= self.next_publish {
                return Ok(());
            }
            let logical_u64 = u64::try_from(logical)
                .map_err(|_| Error::Invariant("shared logical generation exceeds u64".into()))?;
            let released = self
                .mapping
                .control_for_logical(logical)?
                .released_generation
                .load(Ordering::Acquire);
            if released != logical_u64 {
                return Ok(());
            }
            let rank = self.mapping.layout.rank_for(logical);
            // Advance the durable rank cursor before ring reuse. This closes the
            // release/reuse race even when rank batches are dropped out of order.
            self.mapping.advance_rank_resume(rank)?;
            self.session.commit_release(logical)?;
        }
    }

    fn wait_release(&self, logical: usize) -> Result<()> {
        let logical_u64 = u64::try_from(logical)
            .map_err(|_| Error::Invariant("shared logical generation exceeds u64".into()))?;
        let rank = self.mapping.layout.rank_for(logical);
        let rank_control = self.mapping.rank_control(rank)?;
        let ring_control = self.mapping.control_for_logical(logical)?;
        loop {
            if ring_control.released_generation.load(Ordering::Acquire) == logical_u64 {
                return Ok(());
            }
            self.check_external_terminal()?;
            match self.session.state() {
                SessionState::Failed => return Err(self.session.terminal_error()),
                SessionState::Cancelled => return Err(Error::Cancelled),
                SessionState::Finished => {
                    return Err(Error::Invariant(format!(
                        "shared session finished before logical {logical} was released"
                    )))
                }
                SessionState::Running => {}
            }
            let _ = rank_control.release_waiting.swap(1, Ordering::AcqRel);
            let observed = rank_control.release_futex.load(Ordering::Acquire);
            if ring_control.released_generation.load(Ordering::Acquire) == logical_u64 {
                rank_control.release_waiting.store(0, Ordering::Release);
                return Ok(());
            }
            let armed_state =
                self.check_external_terminal()
                    .and_then(|()| match self.session.state() {
                        SessionState::Failed => Err(self.session.terminal_error()),
                        SessionState::Cancelled => Err(Error::Cancelled),
                        SessionState::Finished => Err(Error::Invariant(format!(
                            "shared session finished before logical {logical} was released"
                        ))),
                        SessionState::Running => Ok(()),
                    });
            if let Err(error) = armed_state {
                rank_control.release_waiting.store(0, Ordering::Release);
                return Err(error);
            }
            let wait_result = futex_wait(&rank_control.release_futex, observed);
            rank_control.release_waiting.store(0, Ordering::Release);
            wait_result?;
            if ring_control.released_generation.load(Ordering::Acquire) == logical_u64 {
                return Ok(());
            }
            if !self.mapping.owner_is_alive(rank)? {
                self.mapping.set_terminal(STATE_CANCELLED, 0);
                self.session.cancel();
                return Err(Error::Cancelled);
            }
        }
    }

    fn check_external_terminal(&self) -> Result<()> {
        match self.mapping.terminal() {
            (STATE_RUNNING, _) => Ok(()),
            (STATE_CANCELLED, _) => {
                self.session.cancel();
                Err(Error::Cancelled)
            }
            (STATE_FAILED, code) => Err(producer_error(code)),
            (STATE_FINISHED, _) => Err(Error::Invariant(
                "shared producer was marked finished before execution completed".into(),
            )),
            (state, _) => Err(Error::Invariant(format!(
                "shared producer has unknown state {state}"
            ))),
        }
    }
}

impl Drop for SharedServer {
    fn drop(&mut self) {
        if std::process::id() != self.process_id {
            return;
        }
        if self.mapping.terminal().0 == STATE_RUNNING {
            self.session.cancel();
            self.mapping.set_terminal(STATE_CANCELLED, 0);
        }
        // SAFETY: the owner process reaches this destructor exactly once. A
        // post-fork child intentionally leaves the copied JoinHandles untouched.
        unsafe { ManuallyDrop::drop(&mut self.session) };
    }
}

/// Cancellation handle that wakes both session workers and shared ACK waits.
#[derive(Clone)]
pub struct SharedCancellationHandle {
    session: CancellationHandle,
    mapping: Arc<SharedMapping>,
    process_id: u32,
}

/// Process-local handle that can cancel a client wait without taking ownership
/// of the rank consumer itself.
#[derive(Clone)]
pub struct SharedClientCancellationHandle {
    mapping: Arc<SharedMapping>,
    complete: Arc<AtomicBool>,
    process_id: u32,
}

impl SharedClientCancellationHandle {
    pub fn cancel(&self) {
        if std::process::id() == self.process_id {
            self.mapping.set_terminal(STATE_CANCELLED, 0);
        }
    }

    pub fn cancel_if_incomplete(&self) {
        if !self.complete.load(Ordering::Acquire) {
            self.cancel();
        }
    }

    pub fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

impl SharedCancellationHandle {
    pub fn cancel(&self) {
        if std::process::id() != self.process_id {
            return;
        }
        self.session.cancel();
        self.mapping.set_terminal(STATE_CANCELLED, 0);
    }

    pub fn state(&self) -> SessionState {
        match self.mapping.terminal().0 {
            STATE_RUNNING if std::process::id() == self.process_id => self.session.state(),
            STATE_RUNNING => SessionState::Running,
            STATE_FAILED => SessionState::Failed,
            STATE_CANCELLED => SessionState::Cancelled,
            STATE_FINISHED => SessionState::Finished,
            _ => SessionState::Failed,
        }
    }
}

struct ClientLease {
    mapping: Arc<SharedMapping>,
    rank: usize,
    token: u64,
    process_id: u32,
    process_start_time: u64,
}

impl ClientLease {
    fn ensure_process(&self) -> Result<()> {
        let current = std::process::id();
        if current != self.process_id {
            return Err(Error::InvalidInput(format!(
                "shared client was attached in process {}, but is being used in process {current}; attach after forking",
                self.process_id
            )));
        }
        Ok(())
    }
}

impl Drop for ClientLease {
    fn drop(&mut self) {
        if std::process::id() != self.process_id {
            return;
        }
        if self.mapping.advance_rank_resume(self.rank).is_err() {
            self.mapping.set_terminal(STATE_CANCELLED, 0);
        }
        if let Ok(control) = self.mapping.rank_control(self.rank) {
            if control.owner.load(Ordering::Acquire) == self.token
                && control
                    .owner_start_time
                    .compare_exchange(
                        self.process_start_time,
                        0,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
            {
                let _ = control.owner.compare_exchange(
                    self.token,
                    0,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            }
        }
    }
}

/// Rank-local consumer attached to a shared ring memfd.
pub struct SharedClient {
    lease: Arc<ClientLease>,
    next_logical: usize,
    complete: Arc<AtomicBool>,
}

impl SharedClient {
    /// Attach to a producer memfd as the sole live consumer for `rank`.
    ///
    /// Attach in the final consumer process. A client or batch inherited across
    /// `fork` is rejected so a child cannot release a parent's live lease.
    /// Dropping the client before it has requested every assigned batch cancels
    /// the shared session, preventing an abandoned rank from stalling the ring.
    pub fn attach(fd: BorrowedFd<'_>, rank: usize) -> Result<Self> {
        let mapping = Arc::new(SharedMapping::attach(fd)?);
        match mapping.terminal() {
            (STATE_RUNNING, _) if !mapping.producer_is_alive() => {
                return Err(Error::Session(Arc::new(Error::Invariant(format!(
                    "shared producer process {} is no longer alive",
                    mapping.header().producer_pid
                )))))
            }
            (STATE_RUNNING | STATE_FINISHED, _) => {}
            (STATE_CANCELLED, _) => return Err(Error::Cancelled),
            (STATE_FAILED, code) => return Err(producer_error(code)),
            (state, _) => {
                return Err(Error::InvalidDataset(format!(
                    "shared terminal state {state} is invalid"
                )))
            }
        }
        let rank_control = mapping.rank_control(rank)?;
        let process_id = std::process::id();
        let process_start_time = current_process_start_time("owner")?;
        let token = client_token(process_id, process_start_time);
        loop {
            match rank_control
                .owner
                .compare_exchange(0, token, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(owner) => {
                    let owner_start_time = rank_control.owner_start_time.load(Ordering::Acquire);
                    if rank_control.owner.load(Ordering::Acquire) != owner {
                        continue;
                    }
                    if !owner_process_is_alive(owner, owner_start_time) {
                        if rank_control.owner.load(Ordering::Acquire) != owner {
                            continue;
                        }
                        if let Err(error) = mapping.advance_rank_resume(rank) {
                            mapping.set_terminal(STATE_CANCELLED, 0);
                            return Err(error);
                        }
                        let resume = u64_to_usize(
                            rank_control.resume_logical.load(Ordering::Acquire),
                            "rank resume",
                        )?;
                        if resume == mapping.layout.batch_count {
                            if rank_control
                                .owner_start_time
                                .compare_exchange(
                                    owner_start_time,
                                    0,
                                    Ordering::AcqRel,
                                    Ordering::Acquire,
                                )
                                .is_err()
                            {
                                continue;
                            }
                            if rank_control
                                .owner
                                .compare_exchange(owner, 0, Ordering::AcqRel, Ordering::Acquire)
                                .is_ok()
                            {
                                continue;
                            }
                            mapping.set_terminal(STATE_CANCELLED, 0);
                            return Err(Error::Cancelled);
                        }
                        mapping.set_terminal(STATE_CANCELLED, 0);
                        return Err(Error::Cancelled);
                    }
                    return Err(Error::InvalidInput(format!(
                        "rank {rank} already has an attached shared client (owner token {owner:#x})"
                    )));
                }
            }
        }
        rank_control
            .owner_start_time
            .store(process_start_time, Ordering::Release);
        let lease = Arc::new(ClientLease {
            mapping: Arc::clone(&mapping),
            rank,
            token,
            process_id,
            process_start_time,
        });
        lease.mapping.advance_rank_resume(rank)?;
        let next_logical = u64_to_usize(
            rank_control.resume_logical.load(Ordering::Acquire),
            "rank resume",
        )?;
        if next_logical < lease.mapping.layout.batch_count
            && lease.mapping.layout.rank_for(next_logical) != rank
        {
            return Err(Error::Invariant(format!(
                "rank {rank} resume logical {next_logical} has the wrong assignment"
            )));
        }
        let complete = Arc::new(AtomicBool::new(
            next_logical >= lease.mapping.layout.batch_count,
        ));
        Ok(Self {
            lease,
            next_logical,
            complete,
        })
    }

    pub fn rank(&self) -> usize {
        self.lease.rank
    }

    pub fn world_size(&self) -> usize {
        self.lease.mapping.layout.world_size
    }

    pub fn n_rows(&self) -> usize {
        self.lease.mapping.layout.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.lease.mapping.layout.n_cols
    }

    pub fn dtype(&self) -> OutputDType {
        self.lease.mapping.layout.dtype
    }

    pub fn batch_size(&self) -> usize {
        self.lease.mapping.layout.batch_size
    }

    pub fn batch_count(&self) -> usize {
        self.lease.mapping.layout.rank_batch_count(self.rank())
    }

    pub fn next_logical_batch(&self) -> Option<usize> {
        (self.next_logical < self.lease.mapping.layout.batch_count).then_some(self.next_logical)
    }

    pub fn cancellation_handle(&self) -> SharedClientCancellationHandle {
        SharedClientCancellationHandle {
            mapping: Arc::clone(&self.lease.mapping),
            complete: Arc::clone(&self.complete),
            process_id: self.lease.process_id,
        }
    }

    pub fn next_batch(&mut self) -> Result<Option<SharedBatch>> {
        self.lease.ensure_process()?;
        let layout = self.lease.mapping.layout;
        let logical = self.next_logical;
        if logical >= layout.batch_count {
            return Ok(None);
        }
        if layout.rank_for(logical) != self.rank() {
            return Err(Error::Invariant(format!(
                "rank {} attempted to consume logical batch {logical}",
                self.rank()
            )));
        }
        let logical_u64 = u64::try_from(logical)
            .map_err(|_| Error::Invariant("shared logical generation exceeds u64".into()))?;
        let slot = layout.ring_slot(logical);
        let ring_control = self.lease.mapping.ring_control(slot)?;
        let rank_control = self.lease.mapping.rank_control(self.rank())?;
        loop {
            if self.batch_is_ready(logical, logical_u64, ring_control)? {
                break;
            }
            let _ = rank_control.ready_waiting.swap(1, Ordering::AcqRel);
            let observed = rank_control.ready_futex.load(Ordering::Acquire);
            match self.batch_is_ready(logical, logical_u64, ring_control) {
                Ok(true) => {
                    rank_control.ready_waiting.store(0, Ordering::Release);
                    break;
                }
                Ok(false) => {}
                Err(error) => {
                    rank_control.ready_waiting.store(0, Ordering::Release);
                    return Err(error);
                }
            }
            let wait_result = futex_wait(&rank_control.ready_futex, observed);
            rank_control.ready_waiting.store(0, Ordering::Release);
            wait_result?;
            if !self.batch_is_ready(logical, logical_u64, ring_control)?
                && !self.lease.mapping.producer_is_alive()
            {
                return Err(Error::Session(Arc::new(Error::Invariant(format!(
                    "shared producer process {} exited while rank {} waited for logical batch {logical}",
                    self.lease.mapping.header().producer_pid,
                    self.rank()
                )))));
            }
        }
        let rows = layout.batch_rows(logical)?;
        self.next_logical = layout.next_for_rank(logical);
        if self.next_logical >= layout.batch_count {
            self.complete.store(true, Ordering::Release);
        }
        Ok(Some(SharedBatch {
            lease: Arc::clone(&self.lease),
            logical,
            slot,
            rows,
            released: false,
        }))
    }

    fn batch_is_ready(
        &self,
        logical: usize,
        logical_u64: u64,
        ring_control: &RingControl,
    ) -> Result<bool> {
        match self.lease.mapping.terminal() {
            (STATE_RUNNING, _) => {}
            (STATE_FINISHED, _) => {
                return Err(Error::Invariant(format!(
                    "shared producer finished before logical batch {logical} was published"
                )))
            }
            (STATE_CANCELLED, _) => return Err(Error::Cancelled),
            (STATE_FAILED, code) => return Err(producer_error(code)),
            (state, _) => {
                return Err(Error::Invariant(format!(
                    "shared producer has unknown state {state}"
                )))
            }
        }
        let ready = ring_control.ready_generation.load(Ordering::Acquire);
        if ready == logical_u64 {
            return Ok(true);
        }
        if ready != UNPUBLISHED && ready > logical_u64 {
            return Err(Error::Invariant(format!(
                "shared logical batch {logical} was overwritten by generation {ready}"
            )));
        }
        if ready != UNPUBLISHED
            && ready < logical_u64
            && self.lease.mapping.layout.rank_for(ready as usize) == self.rank()
            && ring_control.released_generation.load(Ordering::Acquire) != ready
        {
            return Err(Error::InvalidInput(format!(
                "rank {} still holds logical batch {ready}; release or drop it before requesting logical batch {logical}",
                self.rank()
            )));
        }
        Ok(false)
    }

    pub fn cancel(&self) {
        self.cancellation_handle().cancel();
    }
}

impl Drop for SharedClient {
    fn drop(&mut self) {
        self.cancellation_handle().cancel_if_incomplete();
    }
}

/// An owned, read-only generation lease into the shared output ring.
///
/// Multiple batches may be held concurrently. Ring reuse remains blocked until
/// every preceding generation is released, so retaining batches deliberately
/// applies bounded backpressure to the producer.
pub struct SharedBatch {
    lease: Arc<ClientLease>,
    logical: usize,
    slot: usize,
    rows: usize,
    released: bool,
}

impl SharedBatch {
    pub fn logical_batch(&self) -> usize {
        self.logical
    }

    pub fn slot(&self) -> usize {
        self.slot
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn n_cols(&self) -> usize {
        self.lease.mapping.layout.n_cols
    }

    pub fn dtype(&self) -> OutputDType {
        self.lease.mapping.layout.dtype
    }

    pub fn row_stride_bytes(&self) -> usize {
        self.lease.mapping.layout.row_stride
    }

    pub fn bytes(&self) -> Result<&[u8]> {
        self.lease.ensure_process()?;
        let layout = self.lease.mapping.layout;
        let offset = self
            .slot
            .checked_mul(layout.batch_size)
            .and_then(|rows| rows.checked_mul(layout.row_stride))
            .ok_or_else(|| Error::Invariant("shared batch offset overflow".into()))?;
        let len = self
            .rows
            .checked_mul(layout.row_stride)
            .ok_or_else(|| Error::Invariant("shared batch length overflow".into()))?;
        self.lease.mapping.ring_bytes_at(offset, len)
    }

    pub fn release(mut self) -> Result<()> {
        self.release_inner()
    }

    fn release_inner(&mut self) -> Result<()> {
        if self.released {
            return Ok(());
        }
        self.lease.ensure_process()?;
        let layout = self.lease.mapping.layout;
        let control = self.lease.mapping.ring_control(self.slot)?;
        let logical_u64 = u64::try_from(self.logical)
            .map_err(|_| Error::Invariant("shared logical generation exceeds u64".into()))?;
        let expected = if self.logical >= layout.ring_slots {
            u64::try_from(self.logical - layout.ring_slots)
                .map_err(|_| Error::Invariant("shared previous generation exceeds u64".into()))?
        } else {
            UNPUBLISHED
        };
        let observed = control.released_generation.load(Ordering::Acquire);
        if observed != expected {
            return Err(Error::Invariant(format!(
                "shared logical batch {} release expected generation {expected}, observed {observed}",
                self.logical
            )));
        }
        // One rank owner creates exactly one non-cloneable lease per logical
        // generation, so no competing writer can pass the validation above.
        control
            .released_generation
            .store(logical_u64, Ordering::Release);
        self.released = true;
        let rank_control = self.lease.mapping.rank_control(self.lease.rank)?;
        signal_futex_waiter(&rank_control.release_futex, &rank_control.release_waiting);
        Ok(())
    }
}

impl Drop for SharedBatch {
    fn drop(&mut self) {
        if self.release_inner().is_err() {
            self.lease.mapping.set_terminal(STATE_CANCELLED, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsFd;
    use std::process::Command;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use rustix::fs::{fcntl_add_seals, ftruncate, memfd_create, MemfdFlags, SealFlags};

    use super::{
        client_token, control_layout, current_process_start_time, expected_ring_bytes, map_shared,
        mapping_bytes, owner_process_is_alive, page_size, process_is_alive, read_process_stat,
        ClientLease, SharedBatch, SharedLayout, SharedMapping, STATE_CANCELLED, UNPUBLISHED,
    };
    use crate::OutputDType;

    fn one_batch_mapping() -> SharedMapping {
        let page_size = page_size().expect("read page size");
        let (rank_offset, ring_control_offset, control_bytes) =
            control_layout(1, 1, page_size).expect("lay out controls");
        let ring_bytes = expected_ring_bytes(1, 2, 64).expect("lay out ring");
        let total_bytes = mapping_bytes(control_bytes, ring_bytes, page_size).expect("map size");
        let layout = SharedLayout {
            world_size: 1,
            n_rows: 2,
            n_cols: 1,
            batch_size: 2,
            batch_count: 1,
            ring_slots: 1,
            world_mask: 0,
            ring_mask: 0,
            row_stride: 64,
            rank_offset,
            ring_control_offset,
            control_bytes,
            ring_bytes,
            total_bytes,
            dtype: OutputDType::U32,
        };
        let fd = memfd_create(
            "sc-load-shared-ring-test",
            MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
        )
        .expect("create test memfd");
        ftruncate(&fd, total_bytes as u64).expect("size test memfd");
        fcntl_add_seals(&fd, SealFlags::SHRINK | SealFlags::GROW | SealFlags::SEAL)
            .expect("seal test memfd");
        let base = map_shared(fd.as_fd(), total_bytes).expect("map test memfd");
        let mapping = SharedMapping {
            producer_fd: Some(fd),
            base,
            layout,
        };
        mapping.initialize().expect("initialize test mapping");
        mapping
    }

    #[test]
    fn process_identity_rejects_a_reused_pid() {
        let process_id = std::process::id();
        let stat = read_process_stat(process_id).expect("read current process identity");
        assert!(process_is_alive(process_id, Some(stat.start_time)));
        assert!(!process_is_alive(
            process_id,
            Some(stat.start_time.wrapping_add(1))
        ));
    }

    #[test]
    fn attached_mapping_does_not_retain_a_descriptor() {
        let producer = one_batch_mapping();
        let descriptor = producer
            .producer_fd
            .as_ref()
            .expect("producer descriptor")
            .as_fd();
        let attached = SharedMapping::attach(descriptor).expect("attach test mapping");
        assert!(attached.producer_fd.is_none());
    }

    #[test]
    fn owner_token_binds_identity_before_start_time_publication() {
        let process_id = std::process::id();
        let stat = read_process_stat(process_id).expect("read current process identity");
        let token = client_token(process_id, stat.start_time);
        assert!(owner_process_is_alive(token, 0));
        assert!(owner_process_is_alive(token, stat.start_time));
        assert!(!owner_process_is_alive(token ^ 1, 0));
        assert!(!owner_process_is_alive(
            token,
            stat.start_time.wrapping_add(1)
        ));
    }

    #[test]
    fn rank_resume_rejects_values_past_the_terminal_cursor() {
        let mapping = one_batch_mapping();
        mapping
            .rank_control(0)
            .expect("rank control")
            .resume_logical
            .store(2, Ordering::Release);
        assert!(mapping.advance_rank_resume(0).is_err());
    }

    #[test]
    fn failed_batch_drop_cancels_instead_of_stranding_the_producer() {
        let mapping = Arc::new(one_batch_mapping());
        let process_id = std::process::id();
        let process_start_time =
            current_process_start_time("test owner").expect("process identity");
        let token = client_token(process_id, process_start_time);
        let rank_control = mapping.rank_control(0).expect("rank control");
        rank_control.owner.store(token, Ordering::Release);
        rank_control
            .owner_start_time
            .store(process_start_time, Ordering::Release);
        mapping
            .ring_control(0)
            .expect("ring control")
            .released_generation
            .store(UNPUBLISHED - 1, Ordering::Release);
        let lease = Arc::new(ClientLease {
            mapping: Arc::clone(&mapping),
            rank: 0,
            token,
            process_id,
            process_start_time,
        });
        drop(SharedBatch {
            lease,
            logical: 0,
            slot: 0,
            rows: 2,
            released: false,
        });
        assert_eq!(mapping.terminal().0, STATE_CANCELLED);
    }

    #[test]
    fn zombie_process_is_not_considered_a_live_shared_owner() {
        let mut child = Command::new("sh")
            .args(["-c", "sleep 0.05"])
            .spawn()
            .expect("spawn short-lived child");
        let process_id = child.id();
        let start_time = read_process_stat(process_id)
            .expect("read child process identity")
            .start_time;
        let deadline = Instant::now() + Duration::from_secs(2);
        let detected = loop {
            if !process_is_alive(process_id, Some(start_time)) {
                break true;
            }
            if Instant::now() >= deadline {
                break false;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        child.wait().expect("reap short-lived child");
        assert!(detected, "zombie process {process_id} remained live");
    }
}
