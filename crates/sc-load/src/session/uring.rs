use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::sync::Arc;

use dyn_blosc::DecodeWorkspace;
use io_uring::{opcode, types, IoUring};

use crate::plan::ReadSource;
use crate::{Error, IoMode, Result, SessionConfig};

use super::SessionInner;

const READY_BATCH: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Idle,
    Prepared,
    Submitted,
}

#[derive(Debug, Clone, Copy)]
struct Token {
    slot: usize,
    generation: u32,
    kind: TokenKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Read,
    Cancel,
}

impl Token {
    fn encode(self) -> Result<u64> {
        if self.slot > 0x7fff_ffff {
            return Err(Error::ResourceLimit(
                "io_uring slot count exceeds token capacity".into(),
            ));
        }
        Ok((u64::from(self.generation) << 32)
            | ((self.slot as u64) << 1)
            | u64::from(matches!(self.kind, TokenKind::Cancel)))
    }

    fn decode(value: u64) -> Self {
        Self {
            slot: ((value >> 1) & 0x7fff_ffff) as usize,
            generation: (value >> 32) as u32,
            kind: if value & 1 == 0 {
                TokenKind::Read
            } else {
                TokenKind::Cancel
            },
        }
    }
}

struct Slot {
    generation: u32,
    node: usize,
    buffer: Vec<MaybeUninit<u8>>,
    absolute_offset: u64,
    initialized: usize,
    submitted_len: usize,
    operations: usize,
    state: SlotState,
    cancel_requested: bool,
}

impl Slot {
    fn new() -> Self {
        Self {
            generation: 0,
            node: usize::MAX,
            buffer: Vec::new(),
            absolute_offset: 0,
            initialized: 0,
            submitted_len: 0,
            operations: 0,
            state: SlotState::Idle,
            cancel_requested: false,
        }
    }

    fn prepare(&mut self, node: usize, absolute_offset: u64, len: usize) -> Result<()> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::ResourceLimit("io_uring slot generation exhausted".into()))?;
        self.node = node;
        self.buffer.clear();
        self.buffer.try_reserve_exact(len)?;
        // SAFETY: reserve established `len` elements of capacity and every bit
        // pattern is valid for MaybeUninit. Bytes are exposed only after CQE
        // accounting proves full initialization.
        unsafe { self.buffer.set_len(len) };
        self.absolute_offset = absolute_offset;
        self.initialized = 0;
        self.submitted_len = 0;
        self.operations = 0;
        self.state = SlotState::Idle;
        self.cancel_requested = false;
        Ok(())
    }

    fn token(&self, slot: usize) -> Token {
        Token {
            slot,
            generation: self.generation,
            kind: TokenKind::Read,
        }
    }

    unsafe fn write_pointer(&mut self) -> *mut u8 {
        debug_assert!(self.initialized <= self.buffer.len());
        // SAFETY: caller keeps the slot allocation fixed through the matching
        // CQE and `initialized` is within the reserved logical buffer.
        unsafe { self.buffer.as_mut_ptr().add(self.initialized).cast() }
    }
}

/// Field order is intentional: closing the ring cancels/joins kernel requests
/// before slot buffers are released if an unrecoverable protocol error occurs.
struct UringWorker {
    ring: IoUring,
    slots: Vec<Slot>,
    free_slots: Vec<usize>,
    prepared: VecDeque<Token>,
    completions: Vec<(u64, i32)>,
    ready: Vec<usize>,
    deferred: Vec<usize>,
    buffer_pool: Vec<Vec<MaybeUninit<u8>>>,
    pooled_capacity: usize,
    active_capacity: usize,
    submitted_ops: usize,
    outstanding_cancels: usize,
    inflight_encoded_bytes: usize,
    queue_depth: usize,
    config: SessionConfig,
    workspace: DecodeWorkspace,
    #[cfg(test)]
    inject_multi_cqe_error: bool,
}

pub(super) fn run_worker(
    inner: Arc<SessionInner>,
    ring: IoUring,
    config: SessionConfig,
    _worker_id: usize,
) -> Result<()> {
    let queue_depth = match config.io_mode {
        IoMode::Uring { queue_depth } | IoMode::Auto { queue_depth } => queue_depth as usize,
        IoMode::Blocking => {
            return Err(Error::Invariant("io_uring worker got Blocking mode".into()));
        }
    };
    let slot_count = config.max_inflight_jobs_per_worker.min(queue_depth).max(1);
    let mut slots = Vec::new();
    slots.try_reserve_exact(slot_count)?;
    slots.resize_with(slot_count, Slot::new);
    let mut free_slots = Vec::new();
    free_slots.try_reserve_exact(slot_count)?;
    free_slots.extend((0..slot_count).rev());
    let mut prepared = VecDeque::new();
    prepared.try_reserve_exact(slot_count)?;
    let mut completions = Vec::new();
    completions.try_reserve_exact(queue_depth)?;
    let mut ready = Vec::new();
    ready.try_reserve_exact(READY_BATCH)?;
    let mut deferred = Vec::new();
    deferred.try_reserve_exact(READY_BATCH)?;
    let mut buffer_pool = Vec::new();
    buffer_pool.try_reserve_exact(slot_count)?;
    let mut worker = UringWorker {
        ring,
        slots,
        free_slots,
        prepared,
        completions,
        ready,
        deferred,
        buffer_pool,
        pooled_capacity: 0,
        active_capacity: 0,
        submitted_ops: 0,
        outstanding_cancels: 0,
        inflight_encoded_bytes: 0,
        queue_depth,
        config,
        workspace: DecodeWorkspace::new(),
        #[cfg(test)]
        inject_multi_cqe_error: std::env::var_os("SC_LOAD_TEST_URING_MULTI_CQE_ERROR").is_some(),
    };
    match worker.run(&inner) {
        Ok(()) => Ok(()),
        Err(error) => {
            inner.fail(error.clone());
            let _ = worker.cancel_and_drain(&inner);
            Err(error)
        }
    }
}

impl UringWorker {
    fn run(&mut self, inner: &Arc<SessionInner>) -> Result<()> {
        loop {
            self.drain_cqes(inner)?;
            self.submit_all(inner)?;

            if !inner.is_running() {
                return self.cancel_and_drain(inner);
            }

            let found_ready = self.refill(inner)?;
            self.submit_all(inner)?;
            if self.submitted_ops != 0 {
                self.wait_one(inner)?;
                continue;
            }
            if !found_ready {
                return Ok(());
            }
        }
    }

    fn refill(&mut self, inner: &SessionInner) -> Result<bool> {
        let block = self.submitted_ops == 0 && self.prepared.is_empty();
        let found = if block {
            inner.ready.pop_many(&mut self.ready, READY_BATCH)
        } else {
            inner.ready.try_pop_many(&mut self.ready, READY_BATCH)
        };
        if !found {
            return Ok(false);
        }

        self.deferred.clear();
        for index in 0..self.ready.len() {
            let node = self.ready[index];
            if !inner.is_running() {
                break;
            }
            if !inner.is_io_node(node) {
                inner.claim_ready_node(node)?;
                inner.execute_cpu_node(node)?;
                inner.finish_node(node);
                continue;
            }
            let task_bytes = inner.io_task(node)?.file_len;
            let can_admit = !self.free_slots.is_empty()
                && self
                    .inflight_encoded_bytes
                    .checked_add(task_bytes)
                    .is_some_and(|bytes| {
                        bytes <= self.config.max_inflight_encoded_bytes_per_worker
                    });
            if can_admit {
                inner.claim_ready_node(node)?;
                self.admit(inner, node)?;
            } else {
                self.deferred.push(node);
            }
        }
        inner.requeue_ready_nodes(&self.deferred);
        Ok(true)
    }

    fn admit(&mut self, inner: &SessionInner, node: usize) -> Result<()> {
        let task = inner.io_task(node)?;
        // SAFETY: lowering points into a frozen source arena owned by `plan`.
        let source = unsafe { task.source.as_ref() };
        let ReadSource::Positioned {
            base_offset,
            view_len,
            ..
        } = source
        else {
            return Err(Error::Invariant(
                "io_uring task does not reference a positioned source".into(),
            ));
        };
        let end = task
            .file_offset
            .checked_add(task.file_len as u64)
            .ok_or_else(|| Error::StalePlan("io_uring range overflow".into()))?;
        if end > *view_len {
            return Err(Error::StalePlan(
                "io_uring range exceeds positioned source".into(),
            ));
        }
        let absolute = base_offset
            .checked_add(task.file_offset)
            .ok_or_else(|| Error::StalePlan("io_uring absolute offset overflow".into()))?;
        let slot_id = self
            .free_slots
            .pop()
            .ok_or_else(|| Error::Invariant("admitted I/O task without a free slot".into()))?;
        let buffer = self.take_buffer(task.file_len)?;
        self.slots[slot_id].buffer = buffer;
        self.slots[slot_id].prepare(node, absolute, task.file_len)?;
        self.active_capacity = self
            .active_capacity
            .checked_add(self.slots[slot_id].buffer.capacity())
            .ok_or_else(|| Error::ResourceLimit("active I/O capacity overflow".into()))?;
        self.inflight_encoded_bytes = self
            .inflight_encoded_bytes
            .checked_add(task.file_len)
            .ok_or_else(|| Error::ResourceLimit("in-flight encoded bytes overflow".into()))?;
        inner.record_uring_admit(task.file_len);
        if let Err(error) = self.prepare_read(inner, slot_id) {
            let token = self.slots[slot_id].token(slot_id);
            self.release_slot(inner, token)?;
            return Err(error);
        }
        Ok(())
    }

    fn prepare_read(&mut self, inner: &SessionInner, slot_id: usize) -> Result<()> {
        if self.prepared.len() + self.submitted_ops + self.outstanding_cancels >= self.queue_depth {
            return Err(Error::Invariant("io_uring queue admission overflow".into()));
        }
        let slot = self
            .slots
            .get_mut(slot_id)
            .ok_or_else(|| Error::Invariant("io_uring slot is out of range".into()))?;
        if slot.state != SlotState::Idle {
            return Err(Error::Invariant("io_uring slot is not idle".into()));
        }
        let remaining = slot
            .buffer
            .len()
            .checked_sub(slot.initialized)
            .ok_or_else(|| Error::Invariant("io_uring initialization overflow".into()))?;
        if remaining == 0 {
            return Err(Error::Invariant("io_uring prepared an empty read".into()));
        }
        let read_len = remaining.min(u32::MAX as usize);
        let offset = slot
            .absolute_offset
            .checked_add(slot.initialized as u64)
            .ok_or_else(|| Error::StalePlan("io_uring continuation offset overflow".into()))?;
        let task = inner.io_task(slot.node)?;
        // SAFETY: the runtime task source pointer targets the immutable plan.
        let source = unsafe { task.source.as_ref() };
        let ReadSource::Positioned { file, .. } = source else {
            return Err(Error::Invariant(
                "io_uring task lost its positioned source".into(),
            ));
        };
        // SAFETY: this slot owns the fixed allocation through the CQE.
        let pointer = unsafe { slot.write_pointer() };
        let token = slot.token(slot_id);
        let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), pointer, read_len as u32)
            .offset(offset)
            .build()
            .user_data(token.encode()?);
        // SAFETY: slot reuse and buffer growth are forbidden until the token's
        // CQE has been consumed; the ring field drops before slot buffers.
        unsafe {
            self.ring
                .submission()
                .push(&entry)
                .map_err(|_| Error::Invariant("io_uring submission queue is full".into()))?;
        }
        slot.submitted_len = read_len;
        slot.state = SlotState::Prepared;
        self.prepared.push_back(token);
        inner.record_uring_prepared(1);
        Ok(())
    }

    fn submit_all(&mut self, _inner: &SessionInner) -> Result<()> {
        while !self.prepared.is_empty() {
            let submitted = loop {
                match self.ring.submit() {
                    Ok(count) => break count,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error.into()),
                }
            };
            _inner.record_uring_submit_call();
            if submitted == 0 || submitted > self.prepared.len() {
                return Err(Error::Io {
                    kind: std::io::ErrorKind::Other,
                    message: "io_uring submit made invalid progress".into(),
                });
            }
            for _ in 0..submitted {
                let token = self
                    .prepared
                    .pop_front()
                    .ok_or_else(|| Error::Invariant("prepared SQE queue underflow".into()))?;
                match token.kind {
                    TokenKind::Read => {
                        let slot = self.slot_mut(token)?;
                        if slot.state != SlotState::Prepared {
                            return Err(Error::Invariant(
                                "submitted io_uring slot was not prepared".into(),
                            ));
                        }
                        slot.state = SlotState::Submitted;
                        self.submitted_ops =
                            self.submitted_ops.checked_add(1).ok_or_else(|| {
                                Error::ResourceLimit("submitted I/O count overflow".into())
                            })?;
                        _inner.record_uring_submitted(1);
                    }
                    TokenKind::Cancel => {
                        self.outstanding_cancels =
                            self.outstanding_cancels.checked_add(1).ok_or_else(|| {
                                Error::ResourceLimit("cancel I/O count overflow".into())
                            })?;
                        _inner.record_uring_cancel_request();
                    }
                }
            }
        }
        Ok(())
    }

    fn wait_one(&mut self, inner: &SessionInner) -> Result<()> {
        #[cfg(feature = "profile")]
        let _timer = inner.profile_io_wait();
        loop {
            match self.ring.submit_and_wait(1) {
                Ok(_) => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        self.drain_cqes(inner)
    }

    fn drain_cqes(&mut self, inner: &SessionInner) -> Result<()> {
        self.completions.clear();
        self.completions.extend(
            self.ring
                .completion()
                .map(|completion| (completion.user_data(), completion.result())),
        );
        #[cfg(test)]
        let inject_error = if self.inject_multi_cqe_error && self.completions.len() > 1 {
            self.inject_multi_cqe_error = false;
            true
        } else {
            false
        };
        let mut first_error = None;
        for index in 0..self.completions.len() {
            let (user_data, result) = self.completions[index];
            #[cfg(test)]
            let result = if inject_error && index == 0 {
                -rustix::io::Errno::IO.raw_os_error()
            } else {
                result
            };
            let token = Token::decode(user_data);
            inner.record_uring_cqe();
            if token.kind == TokenKind::Cancel {
                if let Err(error) = self.validate_token(token) {
                    remember_error(inner, &mut first_error, error);
                }
                match self.outstanding_cancels.checked_sub(1) {
                    Some(remaining) => self.outstanding_cancels = remaining,
                    None => remember_error(
                        inner,
                        &mut first_error,
                        Error::Invariant("cancel I/O count underflow".into()),
                    ),
                }
                inner.record_uring_cancel_cqe();
                continue;
            }
            match self.submitted_ops.checked_sub(1) {
                Some(remaining) => self.submitted_ops = remaining,
                None => {
                    remember_error(
                        inner,
                        &mut first_error,
                        Error::Invariant("submitted I/O count underflow".into()),
                    );
                    continue;
                }
            }
            if !inner.is_running() {
                if let Err(error) = self.release_slot(inner, token) {
                    remember_error(inner, &mut first_error, error);
                }
                continue;
            }
            if result < 0 {
                let error = std::io::Error::from_raw_os_error(-result);
                remember_error(inner, &mut first_error, error.into());
                if let Err(error) = self.release_slot(inner, token) {
                    remember_error(inner, &mut first_error, error);
                }
                continue;
            }
            let read = result as usize;
            let slot = match self.slot_mut(token) {
                Ok(slot) => slot,
                Err(error) => {
                    remember_error(inner, &mut first_error, error);
                    continue;
                }
            };
            if slot.state != SlotState::Submitted {
                remember_error(
                    inner,
                    &mut first_error,
                    Error::Invariant("io_uring completion references a non-submitted slot".into()),
                );
                continue;
            }
            if read == 0 {
                let error = Error::Io {
                    kind: std::io::ErrorKind::UnexpectedEof,
                    message: format!(
                        "io_uring read ended at {} of {} bytes",
                        slot.initialized,
                        slot.buffer.len()
                    ),
                };
                remember_error(inner, &mut first_error, error);
                if let Err(error) = self.release_slot(inner, token) {
                    remember_error(inner, &mut first_error, error);
                }
                continue;
            }
            if read > slot.submitted_len {
                remember_error(
                    inner,
                    &mut first_error,
                    Error::Invariant("io_uring completion exceeds submitted length".into()),
                );
                if let Err(error) = self.release_slot(inner, token) {
                    remember_error(inner, &mut first_error, error);
                }
                continue;
            }
            if read < slot.submitted_len {
                inner.record_short_read();
            }
            slot.initialized += read;
            slot.operations += 1;
            slot.state = SlotState::Idle;
            if slot.initialized < slot.buffer.len() {
                if let Err(error) = self.prepare_read(inner, token.slot) {
                    remember_error(inner, &mut first_error, error);
                    if let Err(error) = self.release_slot(inner, token) {
                        remember_error(inner, &mut first_error, error);
                    }
                }
                continue;
            }

            let node = slot.node;
            let operations = slot.operations;
            let bytes = slot.buffer.len();
            let encoded_pointer = slot.buffer.as_ptr().cast::<u8>();
            inner.record_reads(operations, bytes);
            // SAFETY: the slot remains occupied and its buffer cannot move or
            // be reused until decode completes and `release_slot` runs below.
            let encoded = unsafe { std::slice::from_raw_parts(encoded_pointer, bytes) };
            if let Err(error) = inner.decode_io(node, encoded, &mut self.workspace, true) {
                remember_error(inner, &mut first_error, error);
                if let Err(error) = self.release_slot(inner, token) {
                    remember_error(inner, &mut first_error, error);
                }
                continue;
            }
            inner.finish_node(node);
            if let Err(error) = self.release_slot(inner, token) {
                remember_error(inner, &mut first_error, error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn cancel_and_drain(&mut self, inner: &SessionInner) -> Result<()> {
        self.submit_all(inner)?;
        self.prepare_cancels(inner)?;
        self.submit_all(inner)?;
        while self.submitted_ops != 0 || self.outstanding_cancels != 0 {
            loop {
                match self.ring.submit_and_wait(1) {
                    Ok(_) => break,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error.into()),
                }
            }
            self.drain_cqes(inner)?;
        }
        Ok(())
    }

    fn prepare_cancels(&mut self, inner: &SessionInner) -> Result<()> {
        let mut cancel_tokens = Vec::new();
        for (slot_id, slot) in self.slots.iter_mut().enumerate() {
            if slot.state == SlotState::Submitted && !slot.cancel_requested {
                slot.cancel_requested = true;
                cancel_tokens.push(Token {
                    slot: slot_id,
                    generation: slot.generation,
                    kind: TokenKind::Cancel,
                });
            }
        }
        for cancel_token in cancel_tokens {
            let read_token = Token {
                kind: TokenKind::Read,
                ..cancel_token
            };
            let entry = opcode::AsyncCancel::new(read_token.encode()?)
                .build()
                .user_data(cancel_token.encode()?);
            // SAFETY: cancellation SQEs contain no userspace buffer pointer;
            // slot generations remain fixed until all CQEs are drained.
            unsafe {
                self.ring
                    .submission()
                    .push(&entry)
                    .map_err(|_| Error::Invariant("io_uring cancel SQ is full".into()))?;
            }
            self.prepared.push_back(cancel_token);
            inner.record_uring_prepared(1);
        }
        Ok(())
    }

    fn slot_mut(&mut self, token: Token) -> Result<&mut Slot> {
        let slot = self
            .slots
            .get_mut(token.slot)
            .ok_or_else(|| Error::Invariant("io_uring token slot is invalid".into()))?;
        if slot.generation != token.generation {
            return Err(Error::Invariant(
                "io_uring token generation mismatch".into(),
            ));
        }
        Ok(slot)
    }

    fn validate_token(&self, token: Token) -> Result<()> {
        let slot = self
            .slots
            .get(token.slot)
            .ok_or_else(|| Error::Invariant("io_uring token slot is invalid".into()))?;
        if slot.generation != token.generation {
            return Err(Error::Invariant(
                "io_uring token generation mismatch".into(),
            ));
        }
        Ok(())
    }

    fn release_slot(&mut self, inner: &SessionInner, token: Token) -> Result<()> {
        let (bytes, buffer) = {
            let slot = self.slot_mut(token)?;
            let bytes = slot.buffer.len();
            slot.node = usize::MAX;
            slot.initialized = 0;
            slot.submitted_len = 0;
            slot.operations = 0;
            slot.state = SlotState::Idle;
            slot.cancel_requested = false;
            (bytes, std::mem::take(&mut slot.buffer))
        };
        self.inflight_encoded_bytes = self
            .inflight_encoded_bytes
            .checked_sub(bytes)
            .ok_or_else(|| Error::Invariant("in-flight encoded bytes underflow".into()))?;
        inner.record_uring_release(bytes);
        self.active_capacity = self
            .active_capacity
            .checked_sub(buffer.capacity())
            .ok_or_else(|| Error::Invariant("active I/O capacity underflow".into()))?;
        self.return_buffer(buffer);
        self.free_slots.push(token.slot);
        Ok(())
    }

    fn take_buffer(&mut self, len: usize) -> Result<Vec<MaybeUninit<u8>>> {
        if let Some((index, _)) = self
            .buffer_pool
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.capacity() >= len)
            .min_by_key(|(_, buffer)| buffer.capacity())
        {
            let buffer = self.buffer_pool.swap_remove(index);
            self.pooled_capacity = self.pooled_capacity.saturating_sub(buffer.capacity());
            return Ok(buffer);
        }
        while self
            .active_capacity
            .saturating_add(self.pooled_capacity)
            .saturating_add(len)
            > self.config.max_inflight_encoded_bytes_per_worker
        {
            let Some(buffer) = self.buffer_pool.pop() else {
                break;
            };
            self.pooled_capacity = self.pooled_capacity.saturating_sub(buffer.capacity());
        }
        if self.active_capacity.saturating_add(len)
            > self.config.max_inflight_encoded_bytes_per_worker
        {
            return Err(Error::ResourceLimit(
                "io_uring retained buffer capacity exceeds the per-worker encoded limit".into(),
            ));
        }
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(len)?;
        if self.active_capacity.saturating_add(buffer.capacity())
            > self.config.max_inflight_encoded_bytes_per_worker
        {
            return Err(Error::ResourceLimit(
                "allocator capacity exceeds the per-worker encoded limit".into(),
            ));
        }
        Ok(buffer)
    }

    fn return_buffer(&mut self, mut buffer: Vec<MaybeUninit<u8>>) {
        buffer.clear();
        if self
            .active_capacity
            .saturating_add(self.pooled_capacity)
            .saturating_add(buffer.capacity())
            <= self.config.max_inflight_encoded_bytes_per_worker
        {
            self.pooled_capacity = self.pooled_capacity.saturating_add(buffer.capacity());
            self.buffer_pool.push(buffer);
        }
    }
}

fn remember_error(inner: &SessionInner, first_error: &mut Option<Error>, error: Error) {
    if first_error.is_none() {
        inner.fail(error.clone());
        *first_error = Some(error);
    }
}
