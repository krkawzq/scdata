use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
#[cfg(feature = "profile")]
use std::sync::atomic::Ordering;
use std::sync::Arc;
#[cfg(feature = "profile")]
use std::time::Instant;

use io_uring::{opcode, types, IoUring};

use crate::plan::{Job, JobSide, ReadSource};
use crate::{Error, Result, SessionConfig};

#[cfg(feature = "profile")]
use super::{add_inflight, remove_inflight};
use super::{Claim, SessionInner, WorkerScratch};

const DATA_SIDE: u8 = 0;
const INDICES_SIDE: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpState {
    Idle,
    Prepared,
    Submitted,
    Done,
}

struct ReadSide {
    buffer: IoBuffer,
    absolute_offset: u64,
    initialized: usize,
    submitted_len: usize,
    state: OpState,
    cancel_requested: bool,
}

impl ReadSide {
    fn done(buffer: IoBuffer) -> Self {
        Self {
            buffer,
            absolute_offset: 0,
            initialized: 0,
            submitted_len: 0,
            state: OpState::Done,
            cancel_requested: false,
        }
    }

    fn is_done(&self) -> bool {
        self.state == OpState::Done
    }

    fn bytes(&self) -> &[u8] {
        assert_eq!(
            self.initialized,
            self.buffer.len(),
            "io_uring buffer must be fully initialized before decode"
        );
        // SAFETY: CQE accounting advances `initialized` only by successful
        // kernel writes, and the equality above proves the complete logical
        // buffer was written before it is exposed as bytes.
        unsafe { self.buffer.assume_init() }
    }
}

#[derive(Default)]
struct IoBuffer {
    bytes: Vec<MaybeUninit<u8>>,
}

impl IoBuffer {
    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn prepare(&mut self, len: usize) -> Result<()> {
        self.bytes.clear();
        self.bytes.try_reserve_exact(len)?;
        // SAFETY: reserve established at least `len` capacity and every bit
        // pattern is valid for MaybeUninit. The io_uring read owns all writes;
        // bytes are not exposed until `ReadSide::bytes` proves full completion.
        unsafe { self.bytes.set_len(len) };
        Ok(())
    }

    fn clear(&mut self) {
        self.bytes.clear();
    }

    unsafe fn write_pointer_at(&mut self, offset: usize) -> *mut u8 {
        debug_assert!(offset <= self.bytes.len());
        // SAFETY: the caller keeps the allocation pinned until its CQE and
        // proves `offset` is within the logical buffer.
        unsafe { self.bytes.as_mut_ptr().add(offset).cast() }
    }

    unsafe fn assume_init(&self) -> &[u8] {
        // SAFETY: the caller proves every logical element was initialized.
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), self.bytes.len()) }
    }
}

struct InflightJob {
    job_idx: usize,
    encoded_bytes: usize,
    data: ReadSide,
    indices: Option<ReadSide>,
    queued_ready: bool,
}

impl InflightJob {
    fn is_ready(&self) -> bool {
        self.data.is_done() && self.indices.as_ref().is_none_or(ReadSide::is_done)
    }
}

struct Slot {
    generation: u32,
    job: Option<InflightJob>,
}

#[derive(Debug, Clone, Copy)]
struct Token {
    slot: usize,
    generation: u32,
    side: u8,
    kind: TokenKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Read,
    Cancel,
}

impl Token {
    #[inline(always)]
    fn encode(self) -> u64 {
        debug_assert!(self.slot <= 0x3fff_ffff);
        debug_assert!(self.side <= INDICES_SIDE);
        let kind = u64::from(matches!(self.kind, TokenKind::Cancel));
        (u64::from(self.generation) << 32)
            | ((self.slot as u64) << 2)
            | (kind << 1)
            | u64::from(self.side)
    }

    #[inline(always)]
    fn decode(value: u64) -> Self {
        let side = (value & 1) as u8;
        let kind = if (value >> 1) & 1 == 0 {
            TokenKind::Read
        } else {
            TokenKind::Cancel
        };
        let slot = ((value >> 2) & 0x3fff_ffff) as usize;
        let generation = (value >> 32) as u32;
        Self {
            slot,
            generation,
            side,
            kind,
        }
    }
}

/// Field order is intentional: the ring is dropped before slots and their
/// buffers if an unrecoverable protocol error forces the fallback teardown.
struct UringWorker {
    ring: IoUring,
    slots: Vec<Slot>,
    free_slots: Vec<usize>,
    ready_slots: VecDeque<usize>,
    prepared: VecDeque<Token>,
    completions: Vec<(u64, i32)>,
    cancel_candidates: Vec<Token>,
    buffer_pool: Vec<IoBuffer>,
    pooled_encoded_bytes: usize,
    submitted_ops: usize,
    outstanding_cancel_ops: usize,
    inflight_jobs: usize,
    inflight_encoded_bytes: usize,
    queue_depth: usize,
    config: SessionConfig,
    scratch: WorkerScratch,
    worker_id: usize,
}

pub(super) fn run_worker(
    inner: Arc<SessionInner>,
    ring: IoUring,
    config: SessionConfig,
    worker_id: usize,
) -> Result<()> {
    let queue_depth = match config.io_mode {
        crate::IoMode::Uring { queue_depth } | crate::IoMode::Auto { queue_depth } => {
            queue_depth as usize
        }
        crate::IoMode::Blocking => {
            return Err(Error::Invariant("uring worker got Blocking mode".into()))
        }
    };
    let slot_count = config.max_inflight_jobs_per_worker.min(queue_depth).max(1);
    if slot_count > 0x4000_0000 {
        return Err(Error::ResourceLimit(
            "io_uring slot count exceeds token capacity".into(),
        ));
    }
    let mut slots = Vec::new();
    slots.try_reserve_exact(slot_count)?;
    slots.extend((0..slot_count).map(|_| Slot {
        generation: 0,
        job: None,
    }));
    let free_slots = (0..slot_count).rev().collect();
    let mut worker = UringWorker {
        ring,
        slots,
        free_slots,
        ready_slots: VecDeque::with_capacity(slot_count),
        prepared: VecDeque::with_capacity(queue_depth),
        completions: Vec::with_capacity(queue_depth),
        cancel_candidates: Vec::with_capacity(queue_depth),
        buffer_pool: Vec::with_capacity(slot_count.checked_mul(2).unwrap_or(slot_count)),
        pooled_encoded_bytes: 0,
        submitted_ops: 0,
        outstanding_cancel_ops: 0,
        inflight_jobs: 0,
        inflight_encoded_bytes: 0,
        queue_depth,
        config,
        scratch: WorkerScratch::new(),
        worker_id,
    };
    worker.run(&inner)
}

impl UringWorker {
    fn run(&mut self, inner: &Arc<SessionInner>) -> Result<()> {
        loop {
            self.drain_cqes(inner)?;
            // Short-read continuations produced while draining CQEs should not
            // wait behind CPU decode work.
            self.submit(inner)?;
            let mut claim_result = Claim::LocalFull;
            if inner.is_running() {
                // Decode one completed job at a time. Each released slot is
                // refilled and submitted before decoding the next ready job so
                // bursty completions do not leave the ring idle during a long
                // CPU phase.
                while self.process_one_ready(inner)? {
                    let _ = self.refill(inner)?;
                    self.submit(inner)?;
                }
                claim_result = self.refill(inner)?;
            } else {
                self.ready_slots.clear();
                self.prepare_cancels(inner)?;
            }
            self.submit(inner)?;

            if !inner.is_running() {
                if self.prepared.is_empty()
                    && self.submitted_ops == 0
                    && self.outstanding_cancel_ops == 0
                {
                    return Ok(());
                }
                self.wait_one(inner)?;
                continue;
            }
            if !self.ready_slots.is_empty() {
                continue;
            }
            if !self.prepared.is_empty() {
                self.submit(inner)?;
                if self.submitted_ops > 0 {
                    self.wait_one(inner)?;
                } else if !self.prepared.is_empty() {
                    return Err(Error::Io {
                        kind: std::io::ErrorKind::Other,
                        message: "io_uring submit made no progress".into(),
                    });
                }
                continue;
            }
            if self.submitted_ops > 0 {
                self.wait_one(inner)?;
                continue;
            }
            match claim_result {
                Claim::Exhausted => return Ok(()),
                Claim::WindowBlocked => inner.wait_for_window(self.worker_id),
                Claim::LocalFull => {
                    if self.inflight_jobs == 0 {
                        return Err(Error::ResourceLimit(
                            "next job exceeds per-worker io_uring capacity".into(),
                        ));
                    }
                }
                Claim::Stopped => return Ok(()),
                Claim::Claimed(_) => unreachable!(),
            }
        }
    }

    fn refill(&mut self, inner: &SessionInner) -> Result<Claim> {
        let mut result = Claim::LocalFull;
        while inner.is_running() {
            result = inner.claim_job(self.worker_id, |job| self.can_admit(inner, job));
            let Claim::Claimed(job_idx) = result else {
                break;
            };
            self.admit(inner, job_idx)?;
        }
        Ok(result)
    }

    fn can_admit(&self, inner: &SessionInner, job: &Job) -> bool {
        if self.free_slots.is_empty() {
            return false;
        }
        let Some((ops, bytes)) = job_resources(inner, job) else {
            return false;
        };
        self.prepared
            .len()
            .checked_add(self.submitted_ops)
            .and_then(|total| total.checked_add(self.outstanding_cancel_ops))
            .and_then(|total| total.checked_add(ops))
            .is_some_and(|total| total <= self.queue_depth)
            && self
                .inflight_encoded_bytes
                .checked_add(bytes)
                .is_some_and(|total| total <= self.config.max_inflight_encoded_bytes_per_worker)
    }

    fn admit(&mut self, inner: &SessionInner, job_idx: usize) -> Result<()> {
        let job = &inner.plan.jobs[job_idx];
        let slot_id = self
            .free_slots
            .pop()
            .ok_or_else(|| Error::Invariant("claimed job without a free uring slot".into()))?;
        let generation = self.slots[slot_id]
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::ResourceLimit("io_uring slot generation exhausted".into()))?;
        self.slots[slot_id].generation = generation;
        let data_len = side_len(inner, &job.data)?;
        let data_buffer = self.take_buffer(data_len)?;
        let data = make_side(inner, &job.data, data_buffer)?;
        let indices = job
            .indices
            .as_ref()
            .map(|side| {
                let len = side_len(inner, side)?;
                let buffer = self.take_buffer(len)?;
                make_side(inner, side, buffer)
            })
            .transpose()?;
        let encoded_bytes = data
            .buffer
            .len()
            .checked_add(indices.as_ref().map_or(0, |side| side.buffer.len()))
            .ok_or_else(|| Error::ResourceLimit("in-flight encoded bytes overflow".into()))?;
        self.slots[slot_id].job = Some(InflightJob {
            job_idx,
            encoded_bytes,
            data,
            indices,
            queued_ready: false,
        });
        self.inflight_jobs += 1;
        self.inflight_encoded_bytes = self
            .inflight_encoded_bytes
            .checked_add(encoded_bytes)
            .ok_or_else(|| Error::ResourceLimit("in-flight encoded bytes overflow".into()))?;
        #[cfg(feature = "profile")]
        {
            add_inflight(&inner.stats.inflight_jobs, &inner.stats.peak_jobs, 1)?;
            add_inflight(
                &inner.stats.inflight_bytes,
                &inner.stats.peak_bytes,
                encoded_bytes,
            )?;
        }
        self.prepare_if_needed(
            inner,
            Token {
                slot: slot_id,
                generation,
                side: DATA_SIDE,
                kind: TokenKind::Read,
            },
        )?;
        if self.slots[slot_id]
            .job
            .as_ref()
            .and_then(|job| job.indices.as_ref())
            .is_some()
        {
            self.prepare_if_needed(
                inner,
                Token {
                    slot: slot_id,
                    generation,
                    side: INDICES_SIDE,
                    kind: TokenKind::Read,
                },
            )?;
        }
        self.queue_ready_once(slot_id)?;
        Ok(())
    }

    fn prepare_if_needed(&mut self, inner: &SessionInner, token: Token) -> Result<()> {
        if token.kind != TokenKind::Read {
            return Err(Error::Invariant(
                "read preparation got a cancel token".into(),
            ));
        }
        let slot = self
            .slots
            .get_mut(token.slot)
            .ok_or_else(|| Error::Invariant("io_uring token slot is invalid".into()))?;
        if slot.generation != token.generation {
            return Err(Error::Invariant(
                "io_uring token generation mismatch".into(),
            ));
        }
        let job = slot
            .job
            .as_mut()
            .ok_or_else(|| Error::Invariant("io_uring token references a free slot".into()))?;
        let job_idx = job.job_idx;
        let side = side_mut(job, token.side)?;
        if side.state == OpState::Done
            || side.state == OpState::Prepared
            || side.state == OpState::Submitted
        {
            return Ok(());
        }
        let remaining = side
            .buffer
            .len()
            .checked_sub(side.initialized)
            .ok_or_else(|| Error::Invariant("read side initialized beyond its buffer".into()))?;
        if remaining == 0 {
            side.state = OpState::Done;
            return Ok(());
        }
        let read_len = remaining.min(u32::MAX as usize);
        let offset = side
            .absolute_offset
            .checked_add(side.initialized as u64)
            .ok_or_else(|| Error::StalePlan("io_uring read offset overflow".into()))?;
        // SAFETY: `remaining > 0` proves `initialized < len`; the fixed slot
        // owns this allocation until the CQE and never reallocates it in flight.
        let pointer = unsafe { side.buffer.write_pointer_at(side.initialized) };
        let fd = side_fd(inner, job_idx, token.side)?;
        let entry = opcode::Read::new(types::Fd(fd), pointer, read_len as u32)
            .offset(offset)
            .build()
            .user_data(token.encode());
        // SAFETY: the pointed-to allocation is owned by this fixed slot and is
        // neither resized nor released until the matching CQE is consumed. On
        // fallback teardown `UringWorker.ring` drops before `slots`.
        unsafe {
            self.ring
                .submission()
                .push(&entry)
                .map_err(|_| Error::Invariant("io_uring SQ unexpectedly full".into()))?;
        }
        side.submitted_len = read_len;
        side.state = OpState::Prepared;
        self.prepared.push_back(token);
        #[cfg(feature = "profile")]
        {
            inner
                .worker_stats(self.worker_id)
                .uring_prepared
                .fetch_add(1, Ordering::Relaxed);
            add_inflight(&inner.stats.inflight_ops, &inner.stats.peak_ops, 1)?;
        }
        Ok(())
    }

    fn prepare_cancels(&mut self, _inner: &SessionInner) -> Result<()> {
        self.cancel_candidates.clear();
        for (slot_id, slot) in self.slots.iter().enumerate() {
            let Some(job) = &slot.job else { continue };
            if job.data.state == OpState::Submitted && !job.data.cancel_requested {
                self.cancel_candidates.push(Token {
                    slot: slot_id,
                    generation: slot.generation,
                    side: DATA_SIDE,
                    kind: TokenKind::Read,
                });
            }
            if job
                .indices
                .as_ref()
                .is_some_and(|side| side.state == OpState::Submitted && !side.cancel_requested)
            {
                self.cancel_candidates.push(Token {
                    slot: slot_id,
                    generation: slot.generation,
                    side: INDICES_SIDE,
                    kind: TokenKind::Read,
                });
            }
        }
        for index in 0..self.cancel_candidates.len() {
            let read_token = self.cancel_candidates[index];
            if self.prepared.len() + self.submitted_ops + self.outstanding_cancel_ops
                >= self.queue_depth
            {
                break;
            }
            let cancel_token = Token {
                kind: TokenKind::Cancel,
                ..read_token
            };
            let entry = opcode::AsyncCancel::new(read_token.encode())
                .build()
                .user_data(cancel_token.encode());
            // SAFETY: cancellation SQEs contain no userspace buffer pointer.
            unsafe {
                self.ring
                    .submission()
                    .push(&entry)
                    .map_err(|_| Error::Invariant("io_uring SQ unexpectedly full".into()))?;
            }
            self.token_side_mut(read_token)?.cancel_requested = true;
            self.prepared.push_back(cancel_token);
            #[cfg(feature = "profile")]
            _inner
                .worker_stats(self.worker_id)
                .uring_cancel_requests
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn submit(&mut self, _inner: &SessionInner) -> Result<()> {
        if self.prepared.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "profile")]
        _inner
            .worker_stats(self.worker_id)
            .uring_submit_calls
            .fetch_add(1, Ordering::Relaxed);
        let submitted = loop {
            match self.ring.submit() {
                Ok(count) => break count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        };
        if submitted == 0 {
            return Ok(());
        }
        if submitted > self.prepared.len() {
            return Err(Error::Invariant(
                "io_uring reported more submissions than prepared SQEs".into(),
            ));
        }
        #[cfg(feature = "profile")]
        let mut submitted_reads = 0u64;
        for _ in 0..submitted {
            let token = self
                .prepared
                .pop_front()
                .ok_or_else(|| Error::Invariant("prepared SQE queue underflow".into()))?;
            match token.kind {
                TokenKind::Read => {
                    let side = self.token_side_mut(token)?;
                    if side.state != OpState::Prepared {
                        return Err(Error::Invariant("submitted side was not Prepared".into()));
                    }
                    side.state = OpState::Submitted;
                    self.submitted_ops += 1;
                    #[cfg(feature = "profile")]
                    {
                        submitted_reads += 1;
                    }
                }
                TokenKind::Cancel => {
                    self.outstanding_cancel_ops += 1;
                }
            }
        }
        #[cfg(feature = "profile")]
        _inner
            .worker_stats(self.worker_id)
            .uring_submitted
            .fetch_add(submitted_reads, Ordering::Relaxed);
        Ok(())
    }

    fn wait_one(&mut self, inner: &SessionInner) -> Result<()> {
        #[cfg(feature = "profile")]
        let started = Instant::now();
        loop {
            match self.ring.submit_and_wait(1) {
                Ok(_) => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        #[cfg(feature = "profile")]
        inner.worker_stats(self.worker_id).io_wait_ns.fetch_add(
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.drain_cqes(inner)
    }

    fn drain_cqes(&mut self, inner: &SessionInner) -> Result<()> {
        #[cfg(feature = "profile")]
        let worker_stats = inner.worker_stats(self.worker_id);
        self.completions.clear();
        self.completions.extend(
            self.ring
                .completion()
                .map(|cqe| (cqe.user_data(), cqe.result())),
        );
        for index in 0..self.completions.len() {
            let (user_data, result) = self.completions[index];
            #[cfg(feature = "profile")]
            worker_stats.uring_cqes.fetch_add(1, Ordering::Relaxed);
            let token = Token::decode(user_data);
            if token.kind == TokenKind::Cancel {
                self.validate_cancel_token(token)?;
                self.outstanding_cancel_ops = self
                    .outstanding_cancel_ops
                    .checked_sub(1)
                    .ok_or_else(|| Error::Invariant("cancel op counter underflow".into()))?;
                #[cfg(feature = "profile")]
                worker_stats
                    .uring_cancel_cqes
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            self.submitted_ops = self
                .submitted_ops
                .checked_sub(1)
                .ok_or_else(|| Error::Invariant("submitted op counter underflow".into()))?;
            #[cfg(feature = "profile")]
            remove_inflight(&inner.stats.inflight_ops, 1)?;
            let side = self.token_side_mut(token)?;
            if side.state != OpState::Submitted {
                return Err(Error::Invariant("CQE side was not Submitted".into()));
            }
            side.state = OpState::Idle;
            if result < 0 {
                if !inner.is_running() {
                    side.state = OpState::Done;
                    continue;
                }
                return Err(std::io::Error::from_raw_os_error(-result).into());
            }
            let read = result as usize;
            if !inner.is_running() {
                side.state = OpState::Done;
                continue;
            }
            if read > side.submitted_len {
                return Err(Error::Invariant(format!(
                    "CQE reported {read} bytes for {}-byte request",
                    side.submitted_len
                )));
            }
            if read == 0 && side.initialized < side.buffer.len() {
                return Err(Error::Io {
                    kind: std::io::ErrorKind::UnexpectedEof,
                    message: "io_uring read returned zero before filling the range".into(),
                });
            }
            if read < side.submitted_len {
                #[cfg(feature = "profile")]
                worker_stats.short_reads.fetch_add(1, Ordering::Relaxed);
            }
            side.initialized = side
                .initialized
                .checked_add(read)
                .ok_or_else(|| Error::Invariant("read side initialized byte overflow".into()))?;
            #[cfg(feature = "profile")]
            {
                worker_stats.read_ops.fetch_add(1, Ordering::Relaxed);
                worker_stats
                    .read_bytes
                    .fetch_add(read as u64, Ordering::Relaxed);
            }
            if side.initialized == side.buffer.len() || !inner.is_running() {
                side.state = OpState::Done;
            } else {
                self.prepare_if_needed(inner, token)?;
            }
            self.queue_ready_once(token.slot)?;
        }
        Ok(())
    }

    fn queue_ready_once(&mut self, slot_id: usize) -> Result<()> {
        let job = self
            .slots
            .get_mut(slot_id)
            .and_then(|slot| slot.job.as_mut())
            .ok_or_else(|| Error::Invariant("ready check references free slot".into()))?;
        if job.is_ready() && !job.queued_ready {
            job.queued_ready = true;
            self.ready_slots.push_back(slot_id);
        }
        Ok(())
    }

    fn process_one_ready(&mut self, inner: &SessionInner) -> Result<bool> {
        let Some(slot_id) = self.ready_slots.pop_front() else {
            return Ok(false);
        };
        let mut job = self.slots[slot_id]
            .job
            .take()
            .ok_or_else(|| Error::Invariant("ready queue contains free slot".into()))?;
        if !job.is_ready() {
            return Err(Error::Invariant("ready queue contains reading job".into()));
        }
        let indices = job.indices.as_ref().map(ReadSide::bytes);
        inner.decode_and_commit(
            job.job_idx,
            job.data.bytes(),
            indices,
            &mut self.scratch,
            self.worker_id,
        )?;
        self.inflight_jobs = self
            .inflight_jobs
            .checked_sub(1)
            .ok_or_else(|| Error::Invariant("in-flight job counter underflow".into()))?;
        self.inflight_encoded_bytes = self
            .inflight_encoded_bytes
            .checked_sub(job.encoded_bytes)
            .ok_or_else(|| Error::Invariant("in-flight byte counter underflow".into()))?;
        #[cfg(feature = "profile")]
        {
            remove_inflight(&inner.stats.inflight_jobs, 1)?;
            remove_inflight(&inner.stats.inflight_bytes, job.encoded_bytes)?;
        }
        self.recycle_buffer(std::mem::take(&mut job.data.buffer));
        if let Some(indices) = &mut job.indices {
            self.recycle_buffer(std::mem::take(&mut indices.buffer));
        }
        self.free_slots.push(slot_id);
        Ok(true)
    }

    fn token_side_mut(&mut self, token: Token) -> Result<&mut ReadSide> {
        if token.kind != TokenKind::Read {
            return Err(Error::Invariant(
                "cancel token has no read side state".into(),
            ));
        }
        let slot = self
            .slots
            .get_mut(token.slot)
            .ok_or_else(|| Error::Invariant("CQE slot is out of range".into()))?;
        if slot.generation != token.generation {
            return Err(Error::Invariant(format!(
                "CQE generation {} does not match slot generation {}",
                token.generation, slot.generation
            )));
        }
        let job = slot
            .job
            .as_mut()
            .ok_or_else(|| Error::Invariant("CQE references a free slot".into()))?;
        side_mut(job, token.side)
    }

    fn validate_cancel_token(&mut self, token: Token) -> Result<()> {
        if token.kind != TokenKind::Cancel {
            return Err(Error::Invariant(
                "cancel validation got a read token".into(),
            ));
        }
        let slot = self
            .slots
            .get_mut(token.slot)
            .ok_or_else(|| Error::Invariant("cancel CQE slot is out of range".into()))?;
        if slot.generation != token.generation {
            return Err(Error::Invariant(format!(
                "cancel CQE generation {} does not match slot generation {}",
                token.generation, slot.generation
            )));
        }
        let job = slot
            .job
            .as_mut()
            .ok_or_else(|| Error::Invariant("cancel CQE references a free slot".into()))?;
        if !side_mut(job, token.side)?.cancel_requested {
            return Err(Error::Invariant(
                "cancel CQE has no matching cancellation request".into(),
            ));
        }
        Ok(())
    }

    fn take_buffer(&mut self, len: usize) -> Result<IoBuffer> {
        let best = self
            .buffer_pool
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.capacity() >= len)
            .min_by_key(|(_, buffer)| buffer.capacity())
            .map(|(index, _)| index);
        let mut buffer = if let Some(index) = best {
            let buffer = self.buffer_pool.swap_remove(index);
            self.pooled_encoded_bytes = self
                .pooled_encoded_bytes
                .checked_sub(buffer.capacity())
                .ok_or_else(|| {
                    Error::Invariant("pooled buffer byte accounting underflow".into())
                })?;
            buffer
        } else {
            IoBuffer::default()
        };
        buffer.prepare(len)?;
        Ok(buffer)
    }

    fn recycle_buffer(&mut self, mut buffer: IoBuffer) {
        buffer.clear();
        let retained = self
            .inflight_encoded_bytes
            .checked_add(self.pooled_encoded_bytes)
            .and_then(|bytes| bytes.checked_add(buffer.capacity()));
        if retained.is_some_and(|bytes| bytes <= self.config.max_inflight_encoded_bytes_per_worker)
        {
            self.pooled_encoded_bytes += buffer.capacity();
            self.buffer_pool.push(buffer);
        }
    }
}

fn side_len(inner: &SessionInner, side: &JobSide) -> Result<usize> {
    let source = inner
        .plan
        .sources
        .get(side.source)
        .ok_or_else(|| Error::Invariant("uring job source is missing".into()))?;
    match source {
        ReadSource::Empty => Ok(0),
        ReadSource::Positioned { view_len, .. } => {
            if side.read_range.end > *view_len || side.read_range.start > side.read_range.end {
                return Err(Error::StalePlan(
                    "uring range exceeds positioned view".into(),
                ));
            }
            usize::try_from(side.read_range.end - side.read_range.start)
                .map_err(|_| Error::ResourceLimit("uring read length exceeds usize".into()))
        }
        ReadSource::RangeKey { .. } | ReadSource::WholeKey { .. } => Err(Error::Unsupported(
            "key-backed source reached io_uring worker".into(),
        )),
    }
}

fn make_side(inner: &SessionInner, side: &JobSide, buffer: IoBuffer) -> Result<ReadSide> {
    let source = inner
        .plan
        .sources
        .get(side.source)
        .ok_or_else(|| Error::Invariant("uring job source is missing".into()))?;
    match source {
        ReadSource::Empty => Ok(ReadSide::done(buffer)),
        ReadSource::Positioned {
            base_offset,
            view_len,
            ..
        } => {
            if side.read_range.end > *view_len || side.read_range.start > side.read_range.end {
                return Err(Error::StalePlan(
                    "uring range exceeds positioned view".into(),
                ));
            }
            let absolute_offset = base_offset
                .checked_add(side.read_range.start)
                .ok_or_else(|| Error::StalePlan("uring absolute offset overflow".into()))?;
            if buffer.is_empty() {
                Ok(ReadSide::done(buffer))
            } else {
                Ok(ReadSide {
                    buffer,
                    absolute_offset,
                    initialized: 0,
                    submitted_len: 0,
                    state: OpState::Idle,
                    cancel_requested: false,
                })
            }
        }
        ReadSource::RangeKey { .. } | ReadSource::WholeKey { .. } => Err(Error::Unsupported(
            "key-backed source reached io_uring worker".into(),
        )),
    }
}

fn side_fd(inner: &SessionInner, job_idx: usize, side: u8) -> Result<i32> {
    let job = inner
        .plan
        .jobs
        .get(job_idx)
        .ok_or_else(|| Error::Invariant("uring slot job is missing".into()))?;
    let side = match side {
        DATA_SIDE => &job.data,
        INDICES_SIDE => job
            .indices
            .as_ref()
            .ok_or_else(|| Error::Invariant("indices token references dense job".into()))?,
        _ => return Err(Error::Invariant("invalid uring side".into())),
    };
    match &inner.plan.sources[side.source] {
        ReadSource::Positioned { file, .. } => Ok(file.as_raw_fd()),
        _ => Err(Error::Invariant("uring side is not positioned".into())),
    }
}

fn side_mut(job: &mut InflightJob, side: u8) -> Result<&mut ReadSide> {
    match side {
        DATA_SIDE => Ok(&mut job.data),
        INDICES_SIDE => job
            .indices
            .as_mut()
            .ok_or_else(|| Error::Invariant("indices token references dense job".into())),
        _ => Err(Error::Invariant("invalid uring side".into())),
    }
}

fn job_resources(inner: &SessionInner, job: &Job) -> Option<(usize, usize)> {
    let data_len = job.data.encoded_len(&inner.plan.sources[job.data.source]);
    let mut ops = usize::from(data_len > 0);
    let mut bytes = data_len;
    if let Some(indices) = &job.indices {
        let len = indices.encoded_len(&inner.plan.sources[indices.source]);
        ops = ops.checked_add(usize::from(len > 0))?;
        bytes = bytes.checked_add(len)?;
    }
    Some((ops, bytes))
}
