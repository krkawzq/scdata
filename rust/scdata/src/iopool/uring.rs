//! io_uring backend for positioned file IO.

use std::fs::File;
use std::io;
use std::mem::MaybeUninit;
use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::Arc;
use std::thread;

use io_uring::{opcode, squeue, types, IoUring};

use super::{
    assume_init_read_buffer, operation_file, uninit_read_buffer, FileTable, FixedFileUpdate,
    IoMetadata, IoOperation, IoOutput, IoWork, QueueCore, QueueWake, UringConfig, WorkId,
};

const WAKE_TOKEN: u64 = 1 << 63;
static EMPTY_PATH: [libc::c_char; 1] = [0];

pub(super) fn start(
    config: UringConfig,
    queues: &[Arc<QueueCore>],
    file_table: Arc<FileTable>,
) -> io::Result<Vec<thread::JoinHandle<()>>> {
    let max_pending = config.entries as usize - 1;
    let registered_files = config.registered_files;
    let iowq_workers = [config.iowq_bounded_workers, config.iowq_unbounded_workers];
    let mut handles: Vec<thread::JoinHandle<()>> = Vec::with_capacity(config.drivers);
    let mut started_wake_fds = Vec::with_capacity(config.drivers);
    let queues = queues.to_vec();

    for driver_index in 0..config.drivers {
        let ring = IoUring::new(config.entries)?;
        configure_iowq_workers(&ring, iowq_workers)?;
        let wake_fd = EventFd::new()?;
        let wake_raw_fd = wake_fd.as_raw_fd();
        let queue = Arc::clone(&queues[driver_index % queues.len()]);
        queue.register_wake_fd(wake_raw_fd);
        started_wake_fds.push((Arc::clone(&queue), wake_raw_fd));

        let thread_file_table = Arc::clone(&file_table);
        let handle = match thread::Builder::new()
            .name(format!("iopool-uring-driver-{driver_index}"))
            .spawn(move || {
                driver_loop(
                    ring,
                    max_pending,
                    registered_files,
                    queue,
                    thread_file_table,
                    wake_fd,
                )
            }) {
            Ok(handle) => handle,
            Err(err) => {
                for queue in &queues {
                    queue.shutdown();
                }
                for (queue, fd) in started_wake_fds {
                    queue.unregister_wake_fd(fd);
                }
                for handle in handles {
                    let _ = handle.join();
                }
                return Err(err);
            }
        };
        handles.push(handle);
    }

    Ok(handles)
}

fn configure_iowq_workers(ring: &IoUring, mut workers: [u32; 2]) -> io::Result<()> {
    if workers == [0, 0] {
        return Ok(());
    }
    ring.submitter().register_iowq_max_workers(&mut workers)
}

fn driver_loop(
    mut ring: IoUring,
    max_pending: usize,
    registered_files: u32,
    queue: Arc<QueueCore>,
    file_table: Arc<FileTable>,
    wake_fd: EventFd,
) {
    let mut pending = PendingSlots::new(max_pending);
    let mut fixed_files = FixedFileRegistry::new(&ring, registered_files);
    let mut completions = Vec::with_capacity(max_pending + 1);
    let mut fixed_file_updates = Vec::new();
    let mut fixed_file_update_cursor = 0;
    let mut work_batch = Vec::with_capacity(max_pending);
    let mut wake_poll_armed = false;

    loop {
        fixed_file_update_cursor = apply_fixed_file_updates(
            &ring,
            &mut fixed_files,
            &queue,
            fixed_file_update_cursor,
            &mut fixed_file_updates,
        );

        if pending.is_empty() {
            match queue.pop_or_notification(fixed_file_update_cursor) {
                QueueWake::Work(work) => submit_or_complete(
                    work,
                    &mut ring,
                    &mut pending,
                    &mut fixed_files,
                    &queue,
                    &file_table,
                ),
                QueueWake::Notified => continue,
                QueueWake::Shutdown => break,
            }
        }

        queue.try_pop_batch(&mut work_batch, pending.free_len());
        for work in work_batch.drain(..) {
            submit_or_complete(
                work,
                &mut ring,
                &mut pending,
                &mut fixed_files,
                &queue,
                &file_table,
            );
        }

        if pending.is_empty() {
            continue;
        }

        if !wake_poll_armed {
            if let Err(err) = submit_wake_poll(&mut ring, wake_fd.as_raw_fd()) {
                fail_driver(&queue, &mut pending, err);
                break;
            }
            wake_poll_armed = true;
        }

        if let Err(err) = ring.submit_and_wait(1) {
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            fail_driver(&queue, &mut pending, err);
            break;
        }

        match reap_completions(
            &mut ring,
            &mut pending,
            &mut fixed_files,
            &mut completions,
            &queue,
            wake_fd.as_raw_fd(),
        ) {
            Ok(outcome) => {
                if outcome.wake_seen {
                    wake_poll_armed = false;
                }
            }
            Err(err) => {
                fail_driver(&queue, &mut pending, err);
                break;
            }
        }
    }

    queue.unregister_wake_fd(wake_fd.as_raw_fd());
}

fn submit_or_complete(
    work: IoWork,
    ring: &mut IoUring,
    pending: &mut PendingSlots,
    fixed_files: &mut FixedFileRegistry,
    queue: &QueueCore,
    file_table: &Arc<FileTable>,
) {
    let id = work.id;
    match Pending::prepare(work, file_table) {
        Ok(Prepared::Ring(pending_work)) => {
            if let Err(err) = submit_pending(ring, pending, fixed_files, pending_work) {
                queue.complete(id, Err(err));
            }
        }
        Ok(Prepared::Complete(result)) => queue.complete(id, result),
        Err(err) => queue.complete(id, Err(err)),
    }
}

fn submit_pending(
    ring: &mut IoUring,
    pending: &mut PendingSlots,
    fixed_files: &mut FixedFileRegistry,
    mut pending_work: Pending,
) -> io::Result<()> {
    let slot = pending.reserve()?;
    let result = (|| {
        let target = pending_work.file_target(ring, fixed_files);
        let entry = pending_work.entry(slot_token(slot)?, target)?;
        push_entry(ring, &entry)
    })();

    match result {
        Ok(()) => {
            pending.place(slot, pending_work);
            Ok(())
        }
        Err(err) => {
            pending.release_reserved(slot);
            Err(err)
        }
    }
}

fn push_entry(ring: &mut IoUring, entry: &squeue::Entry) -> io::Result<()> {
    loop {
        let pushed = unsafe { ring.submission().push(entry).is_ok() };
        if pushed {
            return Ok(());
        }
        ring.submit()?;
    }
}

fn submit_wake_poll(ring: &mut IoUring, wake_fd: RawFd) -> io::Result<()> {
    let entry = opcode::PollAdd::new(types::Fd(wake_fd), libc::POLLIN as u32)
        .build()
        .user_data(WAKE_TOKEN);
    push_entry(ring, &entry)
}

fn reap_completions(
    ring: &mut IoUring,
    pending: &mut PendingSlots,
    fixed_files: &mut FixedFileRegistry,
    completions: &mut Vec<(u64, i32)>,
    queue: &QueueCore,
    wake_fd: RawFd,
) -> io::Result<ReapOutcome> {
    completions.clear();
    completions.extend(ring.completion().map(|cqe| (cqe.user_data(), cqe.result())));
    let mut outcome = ReapOutcome::default();

    for (token, result) in completions.iter().copied() {
        if token == WAKE_TOKEN {
            if result < 0 {
                return Err(io::Error::from_raw_os_error(-result));
            }
            outcome.wake_seen = true;
            continue;
        }

        let Some(slot) = token_slot(token) else {
            continue;
        };
        let Some(pending_work) = pending.remove(slot) else {
            continue;
        };
        match pending_work.on_completion(result) {
            PendingResult::Complete(id, result) => queue.complete(id, result),
            PendingResult::Continue(pending_work) => {
                let id = pending_work.id();
                if let Err(err) = submit_pending(ring, pending, fixed_files, pending_work) {
                    queue.complete(id, Err(err));
                }
            }
        }
    }
    if outcome.wake_seen {
        drain_eventfd(wake_fd);
        queue.acknowledge_wake();
    }
    Ok(outcome)
}

fn apply_fixed_file_updates(
    ring: &IoUring,
    fixed_files: &mut FixedFileRegistry,
    queue: &QueueCore,
    cursor: usize,
    updates: &mut Vec<FixedFileUpdate>,
) -> usize {
    let next_cursor = queue.fixed_file_updates_since(cursor, updates);
    for update in updates.drain(..) {
        fixed_files.apply_update(ring, update);
    }
    next_cursor
}

fn fail_driver(queue: &QueueCore, pending: &mut PendingSlots, err: io::Error) {
    let kind = err.kind();
    let message = err.to_string();
    for pending_work in pending.drain() {
        queue.complete(
            pending_work.id(),
            Err(io::Error::new(kind, message.clone())),
        );
    }
    queue.fail(io::Error::new(kind, message));
}

#[derive(Default)]
struct ReapOutcome {
    wake_seen: bool,
}

struct EventFd(RawFd);

impl EventFd {
    fn new() -> io::Result<Self> {
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(fd))
    }
}

impl AsRawFd for EventFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

impl Drop for EventFd {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.0);
        }
    }
}

fn drain_eventfd(fd: RawFd) {
    let mut value: u64 = 0;
    loop {
        let read = unsafe {
            libc::read(
                fd,
                (&mut value as *mut u64).cast::<libc::c_void>(),
                std::mem::size_of::<u64>(),
            )
        };
        if read == std::mem::size_of::<u64>() as isize {
            continue;
        }

        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EINTR) => continue,
            Some(libc::EAGAIN) => return,
            _ => return,
        }
    }
}

#[derive(Clone, Copy)]
enum FileTarget {
    Fd(RawFd),
    Fixed(u32),
}

struct FixedFileRegistry {
    enabled: bool,
    slots: Vec<Option<RawFd>>,
}

impl FixedFileRegistry {
    fn new(ring: &IoUring, capacity: u32) -> Self {
        if capacity == 0 {
            return Self {
                enabled: false,
                slots: Vec::new(),
            };
        }

        match ring.submitter().register_files_sparse(capacity) {
            Ok(()) => Self {
                enabled: true,
                slots: vec![None; capacity as usize],
            },
            Err(_) => Self {
                enabled: false,
                slots: Vec::new(),
            },
        }
    }

    fn target_for(&mut self, ring: &IoUring, file_id: usize, fd: RawFd) -> FileTarget {
        if !self.enabled {
            return FileTarget::Fd(fd);
        }
        let Ok(slot) = u32::try_from(file_id) else {
            return FileTarget::Fd(fd);
        };
        let Some(current) = self.slots.get_mut(slot as usize) else {
            return FileTarget::Fd(fd);
        };
        if *current != Some(fd) {
            match ring.submitter().register_files_update(slot, &[fd]) {
                Ok(1) => *current = Some(fd),
                _ => {
                    self.disable();
                    return FileTarget::Fd(fd);
                }
            }
        }
        FileTarget::Fixed(slot)
    }

    fn apply_update(&mut self, ring: &IoUring, update: FixedFileUpdate) {
        if !self.enabled {
            return;
        }
        let Ok(slot) = u32::try_from(update.file) else {
            return;
        };
        let Some(current) = self.slots.get_mut(slot as usize) else {
            return;
        };

        if update.fd < 0 {
            if current.is_some() {
                if ring.submitter().register_files_update(slot, &[-1]).ok() == Some(1) {
                    *current = None;
                } else {
                    self.disable();
                }
            }
            return;
        }

        if *current == Some(update.fd) {
            return;
        }
        if ring
            .submitter()
            .register_files_update(slot, &[update.fd])
            .ok()
            == Some(1)
        {
            *current = Some(update.fd);
        } else {
            self.disable();
        }
    }

    fn disable(&mut self) {
        self.enabled = false;
        self.slots.clear();
    }
}

struct PendingSlots {
    slots: Vec<Option<Pending>>,
    free: Vec<usize>,
    len: usize,
}

impl PendingSlots {
    fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || None);
        Self {
            slots,
            free: (0..capacity).rev().collect(),
            len: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn free_len(&self) -> usize {
        self.free.len()
    }

    fn reserve(&mut self) -> io::Result<usize> {
        self.free.pop().ok_or_else(|| {
            io::Error::new(io::ErrorKind::WouldBlock, "io_uring pending slots are full")
        })
    }

    fn place(&mut self, slot: usize, pending: Pending) {
        debug_assert!(slot < self.slots.len());
        debug_assert!(self.slots[slot].is_none());
        self.slots[slot] = Some(pending);
        self.len += 1;
    }

    fn release_reserved(&mut self, slot: usize) {
        debug_assert!(slot < self.slots.len());
        debug_assert!(self.slots[slot].is_none());
        self.free.push(slot);
    }

    fn remove(&mut self, slot: usize) -> Option<Pending> {
        let pending = self.slots.get_mut(slot)?.take()?;
        self.free.push(slot);
        self.len -= 1;
        Some(pending)
    }

    fn drain(&mut self) -> Vec<Pending> {
        let mut drained = Vec::with_capacity(self.len);
        for slot in &mut self.slots {
            if let Some(pending) = slot.take() {
                drained.push(pending);
            }
        }
        self.free.clear();
        self.free.extend((0..self.slots.len()).rev());
        self.len = 0;
        drained
    }
}

fn slot_token(slot: usize) -> io::Result<u64> {
    let token = slot as u64;
    if token >= WAKE_TOKEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "io_uring slot index exceeds token space",
        ));
    }
    Ok(token)
}

fn token_slot(token: u64) -> Option<usize> {
    if token & WAKE_TOKEN != 0 {
        return None;
    }
    usize::try_from(token).ok()
}

enum Prepared {
    Ring(Pending),
    Complete(io::Result<IoOutput>),
}

enum Pending {
    Read {
        id: WorkId,
        file_id: usize,
        file: Arc<File>,
        buf: Box<[MaybeUninit<u8>]>,
        offset: u64,
        done: usize,
    },
    Write {
        id: WorkId,
        file_id: usize,
        file: Arc<File>,
        buf: Vec<u8>,
        offset: u64,
        done: usize,
    },
    Fsync {
        id: WorkId,
        file_id: usize,
        file: Arc<File>,
        data_only: bool,
    },
    Truncate {
        id: WorkId,
        file_id: usize,
        file: Arc<File>,
        len: u64,
    },
    Metadata {
        id: WorkId,
        file: Arc<File>,
        statx: Box<libc::statx>,
    },
}

impl Pending {
    fn prepare(work: IoWork, file_table: &Arc<FileTable>) -> io::Result<Prepared> {
        match work.op {
            IoOperation::Read {
                file,
                handle,
                offset,
                len,
            } => {
                if len == 0 {
                    return Ok(Prepared::Complete(Ok(IoOutput::Read(
                        Arc::from(Vec::new()),
                    ))));
                }
                Ok(Prepared::Ring(Self::Read {
                    id: work.id,
                    file_id: file,
                    file: operation_file(file_table, file, handle)?,
                    buf: uninit_read_buffer(len),
                    offset,
                    done: 0,
                }))
            }
            IoOperation::Write {
                file,
                handle,
                offset,
                buf,
            } => {
                if buf.is_empty() {
                    return Ok(Prepared::Complete(Ok(IoOutput::Write { bytes: 0 })));
                }
                Ok(Prepared::Ring(Self::Write {
                    id: work.id,
                    file_id: file,
                    file: operation_file(file_table, file, handle)?,
                    buf,
                    offset,
                    done: 0,
                }))
            }
            IoOperation::Fsync { file, handle } => Ok(Prepared::Ring(Self::Fsync {
                id: work.id,
                file_id: file,
                file: operation_file(file_table, file, handle)?,
                data_only: false,
            })),
            IoOperation::SyncData { file, handle } => Ok(Prepared::Ring(Self::Fsync {
                id: work.id,
                file_id: file,
                file: operation_file(file_table, file, handle)?,
                data_only: true,
            })),
            IoOperation::Truncate { file, handle, len } => Ok(Prepared::Ring(Self::Truncate {
                id: work.id,
                file_id: file,
                file: operation_file(file_table, file, handle)?,
                len,
            })),
            IoOperation::Metadata { file, handle } => Ok(Prepared::Ring(Self::Metadata {
                id: work.id,
                file: operation_file(file_table, file, handle)?,
                statx: Box::new(unsafe { std::mem::zeroed() }),
            })),
        }
    }

    fn id(&self) -> WorkId {
        match self {
            Self::Read { id, .. }
            | Self::Write { id, .. }
            | Self::Fsync { id, .. }
            | Self::Truncate { id, .. }
            | Self::Metadata { id, .. } => *id,
        }
    }

    fn file_target(&self, ring: &IoUring, fixed_files: &mut FixedFileRegistry) -> FileTarget {
        match self {
            Self::Metadata { file, .. } => FileTarget::Fd(file.as_raw_fd()),
            Self::Read { file_id, file, .. }
            | Self::Write { file_id, file, .. }
            | Self::Fsync { file_id, file, .. }
            | Self::Truncate { file_id, file, .. } => {
                fixed_files.target_for(ring, *file_id, file.as_raw_fd())
            }
        }
    }

    fn entry(&mut self, token: u64, target: FileTarget) -> io::Result<squeue::Entry> {
        match self {
            Self::Read {
                buf, offset, done, ..
            } => {
                let len = chunk_len(buf.len() - *done);
                let ptr = unsafe { buf.as_mut_ptr().add(*done).cast::<u8>() };
                let offset = offset
                    .checked_add(*done as u64)
                    .ok_or_else(offset_overflow)?;
                let entry = match target {
                    FileTarget::Fd(fd) => opcode::Read::new(types::Fd(fd), ptr, len)
                        .offset(offset)
                        .build(),
                    FileTarget::Fixed(slot) => opcode::Read::new(types::Fixed(slot), ptr, len)
                        .offset(offset)
                        .build(),
                };
                Ok(entry.user_data(token).flags(squeue::Flags::ASYNC))
            }
            Self::Write {
                buf, offset, done, ..
            } => {
                let len = chunk_len(buf.len() - *done);
                let ptr = unsafe { buf.as_ptr().add(*done) };
                let offset = offset
                    .checked_add(*done as u64)
                    .ok_or_else(offset_overflow)?;
                let entry = match target {
                    FileTarget::Fd(fd) => opcode::Write::new(types::Fd(fd), ptr, len)
                        .offset(offset)
                        .build(),
                    FileTarget::Fixed(slot) => opcode::Write::new(types::Fixed(slot), ptr, len)
                        .offset(offset)
                        .build(),
                };
                Ok(entry.user_data(token).flags(squeue::Flags::ASYNC))
            }
            Self::Fsync { data_only, .. } => {
                let flags = if *data_only {
                    types::FsyncFlags::DATASYNC
                } else {
                    types::FsyncFlags::empty()
                };
                let entry = match target {
                    FileTarget::Fd(fd) => opcode::Fsync::new(types::Fd(fd)).flags(flags).build(),
                    FileTarget::Fixed(slot) => {
                        opcode::Fsync::new(types::Fixed(slot)).flags(flags).build()
                    }
                };
                Ok(entry.user_data(token))
            }
            Self::Truncate { len, .. } => {
                let entry = match target {
                    FileTarget::Fd(fd) => opcode::Ftruncate::new(types::Fd(fd), *len).build(),
                    FileTarget::Fixed(slot) => {
                        opcode::Ftruncate::new(types::Fixed(slot), *len).build()
                    }
                };
                Ok(entry.user_data(token))
            }
            Self::Metadata { file, statx, .. } => {
                let statx = statx.as_mut() as *mut libc::statx;
                Ok(opcode::Statx::new(
                    types::Fd(file.as_raw_fd()),
                    EMPTY_PATH.as_ptr(),
                    statx.cast::<types::statx>(),
                )
                .flags(libc::AT_EMPTY_PATH)
                .mask(libc::STATX_BASIC_STATS)
                .build()
                .user_data(token))
            }
        }
    }

    fn on_completion(mut self, result: i32) -> PendingResult {
        if result < 0 {
            return self.on_error(io::Error::from_raw_os_error(-result));
        }

        match &mut self {
            Self::Read { id, buf, done, .. } => {
                let n = result as usize;
                if n > buf.len() - *done {
                    return PendingResult::Complete(*id, Err(invalid_completion()));
                }
                if n == 0 {
                    return PendingResult::Complete(
                        *id,
                        Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "failed to fill whole buffer",
                        )),
                    );
                }
                *done += n;
                if *done == buf.len() {
                    let buf = std::mem::replace(buf, uninit_read_buffer(0));
                    let buf = unsafe { assume_init_read_buffer(buf) };
                    PendingResult::Complete(*id, Ok(IoOutput::Read(Arc::from(buf))))
                } else {
                    PendingResult::Continue(self)
                }
            }
            Self::Write { id, buf, done, .. } => {
                let n = result as usize;
                if n > buf.len() - *done {
                    return PendingResult::Complete(*id, Err(invalid_completion()));
                }
                if n == 0 {
                    return PendingResult::Complete(
                        *id,
                        Err(io::Error::new(
                            io::ErrorKind::WriteZero,
                            "failed to write whole buffer",
                        )),
                    );
                }
                *done += n;
                if *done == buf.len() {
                    PendingResult::Complete(*id, Ok(IoOutput::Write { bytes: *done }))
                } else {
                    PendingResult::Continue(self)
                }
            }
            Self::Fsync { id, data_only, .. } => {
                let output = if *data_only {
                    IoOutput::SyncData
                } else {
                    IoOutput::SyncAll
                };
                PendingResult::Complete(*id, Ok(output))
            }
            Self::Truncate { id, .. } => PendingResult::Complete(*id, Ok(IoOutput::Truncate)),
            Self::Metadata { id, statx, .. } => {
                PendingResult::Complete(*id, Ok(IoOutput::Metadata(statx_to_metadata(statx))))
            }
        }
    }

    fn on_error(self, err: io::Error) -> PendingResult {
        let pending = match self {
            Self::Metadata { id, file, .. } => {
                return PendingResult::Complete(
                    id,
                    file.metadata()
                        .map(IoMetadata::from)
                        .map(IoOutput::Metadata),
                );
            }
            pending => pending,
        };
        let id = pending.id();

        if is_uring_opcode_unsupported(&err) {
            match pending {
                Self::Truncate { id, file, len, .. } => {
                    return PendingResult::Complete(
                        id,
                        file.set_len(len).map(|_| IoOutput::Truncate),
                    );
                }
                Self::Read { .. } | Self::Write { .. } | Self::Fsync { .. } => {}
                Self::Metadata { .. } => unreachable!("metadata returns above"),
            }
        }

        PendingResult::Complete(id, Err(err))
    }
}

enum PendingResult {
    Complete(WorkId, io::Result<IoOutput>),
    Continue(Pending),
}

fn chunk_len(remaining: usize) -> u32 {
    remaining.min(u32::MAX as usize) as u32
}

fn offset_overflow() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "IO offset overflowed")
}

fn invalid_completion() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "io_uring completion exceeded remaining buffer length",
    )
}

fn is_uring_opcode_unsupported(err: &io::Error) -> bool {
    matches!(
        err.raw_os_error(),
        Some(libc::EINVAL | libc::EOPNOTSUPP | libc::ENOSYS)
    )
}

fn statx_to_metadata(statx: &libc::statx) -> IoMetadata {
    let mode = statx.stx_mode as libc::mode_t;
    let kind = mode & libc::S_IFMT;
    IoMetadata {
        len: statx.stx_size,
        is_file: kind == libc::S_IFREG,
        is_dir: kind == libc::S_IFDIR,
        readonly: mode & 0o222 == 0,
    }
}
