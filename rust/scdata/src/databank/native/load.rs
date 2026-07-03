use std::collections::{HashMap, VecDeque};
use std::io;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use tokio::task::JoinSet;

use crate::access::{FileRef, IoBackend, IoTask};

use super::super::config::NativeLoadCoalesceConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeLoadRequest {
    pub(crate) id: u64,
    pub(crate) file: FileRef,
    pub(crate) offset: u64,
    pub(crate) len: usize,
    pub(crate) priority: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CoalescedRead {
    pub(crate) file: FileRef,
    pub(crate) offset: u64,
    pub(crate) len: usize,
    pub(crate) priority: u8,
    pub(crate) children: Vec<CoalescedChild>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoalescedChild {
    pub(crate) request_id: u64,
    pub(crate) relative_offset: usize,
    pub(crate) len: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct NativeLoadCompletion {
    pub(crate) request_id: u64,
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) range: Range<usize>,
}

#[derive(Debug, Clone)]
struct NativeSharedLoadCompletion {
    bytes: Arc<[u8]>,
    range: Range<usize>,
}

impl NativeSharedLoadCompletion {
    fn for_request(&self, request: NativeLoadRequest) -> NativeLoadCompletion {
        NativeLoadCompletion {
            request_id: request.id,
            bytes: Arc::clone(&self.bytes),
            range: self.range.clone(),
        }
    }
}

impl From<&NativeLoadCompletion> for NativeSharedLoadCompletion {
    fn from(completion: &NativeLoadCompletion) -> Self {
        Self {
            bytes: Arc::clone(&completion.bytes),
            range: completion.range.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct NativeSharedLoadError {
    kind: io::ErrorKind,
    message: Arc<str>,
}

impl NativeSharedLoadError {
    fn from_error(err: &io::Error) -> Self {
        Self {
            kind: err.kind(),
            message: Arc::from(err.to_string()),
        }
    }

    fn to_error(&self) -> io::Error {
        io::Error::new(self.kind, self.message.to_string())
    }
}

type NativeSharedLoadResult = Result<NativeSharedLoadCompletion, NativeSharedLoadError>;

#[derive(Debug)]
struct NativeInFlightPayloadEntry {
    notify: Notify,
    result: Mutex<Option<NativeSharedLoadResult>>,
}

impl NativeInFlightPayloadEntry {
    fn new() -> Self {
        Self {
            notify: Notify::new(),
            result: Mutex::new(None),
        }
    }

    async fn wait(&self) -> io::Result<NativeSharedLoadCompletion> {
        loop {
            let notified = self.notify.notified();
            if let Some(result) = self
                .result
                .lock()
                .expect("native in-flight payload lock poisoned")
                .clone()
            {
                return result.map_err(|err| err.to_error());
            }
            notified.await;
        }
    }

    fn complete(&self, result: NativeSharedLoadResult) {
        *self
            .result
            .lock()
            .expect("native in-flight payload lock poisoned") = Some(result);
        self.notify.notify_waiters();
    }
}

#[derive(Debug)]
struct NativeOwnedInFlightPayload {
    request: NativeLoadRequest,
    entry: Arc<NativeInFlightPayloadEntry>,
}

#[derive(Debug)]
struct NativeWaitingInFlightPayload {
    request: NativeLoadRequest,
    entry: Arc<NativeInFlightPayloadEntry>,
}

#[derive(Debug)]
enum NativeInFlightRegistration {
    Owner(Arc<NativeInFlightPayloadEntry>),
    Waiter(Arc<NativeInFlightPayloadEntry>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct NativeBlockCacheKey {
    pub(crate) file: FileRef,
    pub(crate) offset: u64,
    pub(crate) len: usize,
}

impl NativeBlockCacheKey {
    pub(crate) fn from_request(request: NativeLoadRequest) -> Self {
        Self {
            file: request.file,
            offset: request.offset,
            len: request.len,
        }
    }
}

#[derive(Debug)]
struct NativeBlockPayloadEntry {
    bytes: Arc<[u8]>,
    bytes_len: usize,
}

#[derive(Debug, Default)]
struct NativeBlockPayloadCacheState {
    entries: HashMap<NativeBlockCacheKey, NativeBlockPayloadEntry>,
    order: VecDeque<NativeBlockCacheKey>,
    bytes: usize,
}

#[derive(Debug)]
struct NativeBlockPayloadCacheShard {
    capacity_bytes: usize,
    state: Mutex<NativeBlockPayloadCacheState>,
}

impl NativeBlockPayloadCacheShard {
    fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            state: Mutex::new(NativeBlockPayloadCacheState::default()),
        }
    }

    fn get(&self, request: NativeLoadRequest) -> Option<NativeLoadCompletion> {
        let key = NativeBlockCacheKey::from_request(request);
        let state = self
            .state
            .lock()
            .expect("native block payload cache lock poisoned");
        let entry = state.entries.get(&key)?;
        Some(NativeLoadCompletion {
            request_id: request.id,
            bytes: Arc::clone(&entry.bytes),
            range: 0..entry.bytes_len,
        })
    }

    fn insert(&self, request: NativeLoadRequest, bytes: Arc<[u8]>) {
        if self.capacity_bytes == 0 || bytes.is_empty() || bytes.len() > self.capacity_bytes {
            return;
        }
        let key = NativeBlockCacheKey::from_request(request);
        let mut state = self
            .state
            .lock()
            .expect("native block payload cache lock poisoned");
        if let Some(old_len) = state.entries.get(&key).map(|entry| entry.bytes_len) {
            state.bytes = state.bytes.saturating_sub(old_len);
            state.bytes = state.bytes.saturating_add(bytes.len());
            if let Some(old) = state.entries.get_mut(&key) {
                old.bytes_len = bytes.len();
                old.bytes = bytes;
            }
            return;
        }
        state.bytes = state.bytes.saturating_add(bytes.len());
        state.order.push_back(key);
        state.entries.insert(
            key,
            NativeBlockPayloadEntry {
                bytes_len: bytes.len(),
                bytes,
            },
        );
        while state.bytes > self.capacity_bytes {
            let Some(victim) = state.order.pop_front() else {
                break;
            };
            if let Some(entry) = state.entries.remove(&victim) {
                state.bytes = state.bytes.saturating_sub(entry.bytes_len);
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct NativeBlockPayloadCache {
    shards: Vec<NativeBlockPayloadCacheShard>,
}

impl NativeBlockPayloadCache {
    pub(crate) fn new(capacity_bytes: usize, shards: usize) -> Self {
        let shard_count = shards.max(1);
        let shard_capacity = capacity_bytes.div_ceil(shard_count).max(1);
        let shards = (0..shard_count)
            .map(|_| NativeBlockPayloadCacheShard::new(shard_capacity))
            .collect();
        Self { shards }
    }

    fn get(&self, request: NativeLoadRequest) -> Option<NativeLoadCompletion> {
        let key = NativeBlockCacheKey::from_request(request);
        self.shard_for_key(key).get(request)
    }

    fn insert(&self, request: NativeLoadRequest, bytes: Arc<[u8]>) {
        let key = NativeBlockCacheKey::from_request(request);
        self.shard_for_key(key).insert(request, bytes);
    }

    fn shard_for_key(&self, key: NativeBlockCacheKey) -> &NativeBlockPayloadCacheShard {
        let hash = key.file.0.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ key.offset.rotate_left(17)
            ^ (key.len as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        &self.shards[(hash as usize) % self.shards.len()]
    }
}

/// Shares exact payload reads while they are currently in flight.
///
/// This is deliberately not a payload cache: the global map entry is removed as
/// soon as the owner read completes. Existing waiters keep only an `Arc` to the
/// completed entry long enough to receive the same completion.
#[derive(Debug, Default)]
struct NativeInFlightPayloadShard {
    entries: Mutex<HashMap<NativeBlockCacheKey, Arc<NativeInFlightPayloadEntry>>>,
}

#[derive(Debug)]
pub(crate) struct NativeInFlightPayloadReads {
    shards: Vec<NativeInFlightPayloadShard>,
}

impl NativeInFlightPayloadReads {
    const DEFAULT_SHARDS: usize = 64;

    pub(crate) fn new() -> Self {
        Self::with_shards(Self::DEFAULT_SHARDS)
    }

    pub(crate) fn with_shards(shards: usize) -> Self {
        let shard_count = shards.max(1);
        let shards = (0..shard_count)
            .map(|_| NativeInFlightPayloadShard::default())
            .collect();
        Self { shards }
    }

    fn register(&self, request: NativeLoadRequest) -> NativeInFlightRegistration {
        let key = NativeBlockCacheKey::from_request(request);
        let shard = self.shard_for_key(key);
        let mut entries = shard
            .entries
            .lock()
            .expect("native in-flight payload table lock poisoned");
        if let Some(entry) = entries.get(&key) {
            return NativeInFlightRegistration::Waiter(Arc::clone(entry));
        }
        let entry = Arc::new(NativeInFlightPayloadEntry::new());
        entries.insert(key, Arc::clone(&entry));
        NativeInFlightRegistration::Owner(entry)
    }

    fn complete(
        &self,
        request: NativeLoadRequest,
        entry: &Arc<NativeInFlightPayloadEntry>,
        result: NativeSharedLoadResult,
    ) {
        entry.complete(result);
        let key = NativeBlockCacheKey::from_request(request);
        let shard = self.shard_for_key(key);
        let mut entries = shard
            .entries
            .lock()
            .expect("native in-flight payload table lock poisoned");
        if entries
            .get(&key)
            .map(|current| Arc::ptr_eq(current, entry))
            .unwrap_or(false)
        {
            entries.remove(&key);
        }
    }

    fn shard_for_key(&self, key: NativeBlockCacheKey) -> &NativeInFlightPayloadShard {
        let hash = key.file.0.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ key.offset.rotate_left(17)
            ^ (key.len as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        &self.shards[(hash as usize) % self.shards.len()]
    }
}

#[derive(Clone)]
pub(crate) struct NativeLoadModule {
    io: Arc<dyn IoBackend>,
    coalesce: NativeLoadCoalesceConfig,
    block_cache: Option<Arc<NativeBlockPayloadCache>>,
    in_flight: Option<Arc<NativeInFlightPayloadReads>>,
}

impl NativeLoadModule {
    pub(crate) fn new(io: Arc<dyn IoBackend>, coalesce: NativeLoadCoalesceConfig) -> Self {
        Self {
            io,
            coalesce,
            block_cache: None,
            in_flight: None,
        }
    }

    pub(crate) fn with_block_cache(
        io: Arc<dyn IoBackend>,
        coalesce: NativeLoadCoalesceConfig,
        block_cache: Option<Arc<NativeBlockPayloadCache>>,
    ) -> Self {
        Self {
            io,
            coalesce,
            block_cache,
            in_flight: None,
        }
    }

    pub(crate) fn with_caches(
        io: Arc<dyn IoBackend>,
        coalesce: NativeLoadCoalesceConfig,
        block_cache: Option<Arc<NativeBlockPayloadCache>>,
        in_flight: Option<Arc<NativeInFlightPayloadReads>>,
    ) -> Self {
        Self {
            io,
            coalesce,
            block_cache,
            in_flight,
        }
    }

    /// Read a single range directly, bypassing coalescing and the JoinSet.
    ///
    /// Used on the index-cache miss path (`build_block_index`), where each
    /// read is small (Blosc header / table prefix) and independent — the
    /// coalesce+JoinSet machinery of [`load`](Self::load) would only add task
    /// spawn and index-bookkeeping overhead for a single request.
    pub(crate) async fn load_single(
        &self,
        file: FileRef,
        offset: u64,
        len: usize,
        priority: u8,
    ) -> io::Result<Arc<[u8]>> {
        let bytes = self.io.submit_read(file, offset, len, priority).await?;
        if bytes.len() != len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "native load_single returned {} bytes, expected {}",
                    bytes.len(),
                    len
                ),
            ));
        }
        Ok(bytes)
    }

    pub(crate) async fn load(
        &self,
        requests: &[NativeLoadRequest],
    ) -> io::Result<Vec<NativeLoadCompletion>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let (cached, misses) = self.split_cached_requests(requests);
        if misses.is_empty() {
            return merge_cached_and_loaded_completions(requests, cached, Vec::new());
        }
        let registered = self.register_inflight_requests(&misses);
        let reads = coalesce_load_requests_presorted(&registered.misses, &self.coalesce);
        self.load_registered_misses(requests, cached, registered, reads)
            .await
    }

    pub(crate) async fn load_unsorted(
        &self,
        requests: &[NativeLoadRequest],
    ) -> io::Result<Vec<NativeLoadCompletion>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let (cached, misses) = self.split_cached_requests(requests);
        if misses.is_empty() {
            return merge_cached_and_loaded_completions(requests, cached, Vec::new());
        }
        let registered = self.register_inflight_requests(&misses);
        let reads = coalesce_load_requests(&registered.misses, &self.coalesce);
        self.load_registered_misses(requests, cached, registered, reads)
            .await
    }

    async fn load_registered_misses(
        &self,
        original_order: &[NativeLoadRequest],
        cached: Vec<NativeLoadCompletion>,
        registered: RegisteredNativeLoads,
        reads: Vec<CoalescedRead>,
    ) -> io::Result<Vec<NativeLoadCompletion>> {
        let loaded = if registered.misses.is_empty() {
            Vec::new()
        } else {
            match self
                .load_misses_coalesced(&registered.misses, &registered.misses, Vec::new(), reads)
                .await
            {
                Ok(loaded) => {
                    self.complete_inflight_success(&registered.owners, &loaded);
                    loaded
                }
                Err(err) => {
                    self.complete_inflight_error(&registered.owners, &err);
                    return Err(err);
                }
            }
        };
        let had_waiters = !registered.waiters.is_empty();
        let waited = self.await_inflight_waiters(registered.waiters).await?;
        let loaded = loaded.into_iter().chain(waited).collect();
        if had_waiters {
            return merge_completions_by_request_id(original_order, cached, loaded);
        }
        merge_cached_and_loaded_completions(original_order, cached, loaded)
    }

    fn register_inflight_requests(&self, requests: &[NativeLoadRequest]) -> RegisteredNativeLoads {
        let Some(in_flight) = &self.in_flight else {
            return RegisteredNativeLoads {
                misses: requests.to_vec(),
                owners: Vec::new(),
                waiters: Vec::new(),
            };
        };
        let mut misses = Vec::new();
        let mut owners = Vec::new();
        let mut waiters = Vec::new();
        for &request in requests {
            match in_flight.register(request) {
                NativeInFlightRegistration::Owner(entry) => {
                    misses.push(request);
                    owners.push(NativeOwnedInFlightPayload { request, entry });
                }
                NativeInFlightRegistration::Waiter(entry) => {
                    waiters.push(NativeWaitingInFlightPayload { request, entry });
                }
            }
        }
        RegisteredNativeLoads {
            misses,
            owners,
            waiters,
        }
    }

    fn complete_inflight_success(
        &self,
        owners: &[NativeOwnedInFlightPayload],
        loaded: &[NativeLoadCompletion],
    ) {
        let Some(in_flight) = &self.in_flight else {
            return;
        };
        debug_assert_eq!(
            owners.len(),
            loaded.len(),
            "native in-flight owner/completion count mismatch"
        );
        for (owner, completion) in owners.iter().zip(loaded) {
            in_flight.complete(
                owner.request,
                &owner.entry,
                Ok(NativeSharedLoadCompletion::from(completion)),
            );
        }
    }

    fn complete_inflight_error(&self, owners: &[NativeOwnedInFlightPayload], err: &io::Error) {
        let Some(in_flight) = &self.in_flight else {
            return;
        };
        let shared = NativeSharedLoadError::from_error(err);
        for owner in owners {
            in_flight.complete(owner.request, &owner.entry, Err(shared.clone()));
        }
    }

    async fn await_inflight_waiters(
        &self,
        waiters: Vec<NativeWaitingInFlightPayload>,
    ) -> io::Result<Vec<NativeLoadCompletion>> {
        if waiters.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(waiters.len());
        for waiter in waiters {
            let completion = waiter.entry.wait().await?;
            out.push(completion.for_request(waiter.request));
        }
        Ok(out)
    }

    async fn load_misses_coalesced(
        &self,
        original_order: &[NativeLoadRequest],
        miss_order: &[NativeLoadRequest],
        cached: Vec<NativeLoadCompletion>,
        reads: Vec<CoalescedRead>,
    ) -> io::Result<Vec<NativeLoadCompletion>> {
        if reads.is_empty() {
            return merge_cached_and_loaded_completions(original_order, cached, Vec::new());
        }
        let loaded = if self.io.prefers_inline_reads() {
            self.load_coalesced_inline(miss_order, reads).await?
        } else {
            self.load_coalesced_parallel(miss_order, reads).await?
        };
        merge_cached_and_loaded_completions(original_order, cached, loaded)
    }

    fn split_cached_requests(
        &self,
        output_order: &[NativeLoadRequest],
    ) -> (Vec<NativeLoadCompletion>, Vec<NativeLoadRequest>) {
        let Some(cache) = &self.block_cache else {
            return (Vec::new(), output_order.to_vec());
        };
        let mut cached = Vec::new();
        let mut misses = Vec::new();
        for request in output_order {
            if let Some(completion) = cache.get(*request) {
                cached.push(completion);
            } else {
                misses.push(*request);
            }
        }
        (cached, misses)
    }

    async fn load_coalesced_parallel(
        &self,
        output_order: &[NativeLoadRequest],
        reads: Vec<CoalescedRead>,
    ) -> io::Result<Vec<NativeLoadCompletion>> {
        // Submit every coalesced read up front and await them concurrently.
        // `IoBackend::submit_read` only constructs the future; the underlying
        // IO is dispatched when the future is first polled, so a serial
        // `await` loop would force reads to run one at a time. Driving them
        // together on a JoinSet lets the IoPool execute them in parallel.
        let mut pending = JoinSet::new();
        for (slot, read) in reads.into_iter().enumerate() {
            let task = self
                .io
                .submit_read(read.file, read.offset, read.len, read.priority);
            pending.spawn(async move {
                let bytes = task.await?;
                io::Result::Ok((slot, read, bytes))
            });
        }

        // Completions indexed by request id. Batch-native loading may sort
        // requests by file/offset before coalescing, so the first request is
        // not necessarily the lowest id. The planner still assigns a compact
        // contiguous id range; indexing by min id keeps lookup O(1) without a
        // HashMap in the hot path.
        let (min_id, id_span) = request_id_bounds(output_order)?;
        let mut by_id: Vec<Option<NativeLoadCompletion>> = (0..id_span).map(|_| None).collect();
        let request_by_id = if self.block_cache.is_some() {
            Some(build_request_by_id(output_order, min_id, id_span)?)
        } else {
            None
        };
        while let Some(joined) = pending.join_next().await {
            let (slot, read, bytes) = joined
                .map_err(|err| io::Error::other(format!("native load task panicked: {err}")))??;
            if bytes.len() != read.len {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "native load read returned {} bytes, expected {}",
                        bytes.len(),
                        read.len
                    ),
                ));
            }
            let _ = slot; // slot retained for future ordering diagnostics
            for child in read.children {
                let start = child.relative_offset;
                let end = start.checked_add(child.len).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "native load child overflow")
                })?;
                if end > bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "native load child range exceeds coalesced read",
                    ));
                }
                let idx = request_id_index(child.request_id, min_id, id_span)?;
                if by_id[idx].is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("native load duplicate child id {}", child.request_id),
                    ));
                }
                by_id[idx] = Some(NativeLoadCompletion {
                    request_id: child.request_id,
                    bytes: Arc::clone(&bytes),
                    range: start..end,
                });
                if let Some(request_by_id) = request_by_id.as_ref() {
                    let request = request_by_id[idx].ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "native load child request missing",
                        )
                    })?;
                    self.insert_child_cache(request, &bytes, start, end);
                }
            }
        }

        output_order
            .iter()
            .map(|request| {
                let idx = request_id_index(request.id, min_id, id_span)?;
                by_id[idx].take().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("native load missing completion for request {}", request.id),
                    )
                })
            })
            .collect()
    }

    async fn load_coalesced_inline(
        &self,
        output_order: &[NativeLoadRequest],
        reads: Vec<CoalescedRead>,
    ) -> io::Result<Vec<NativeLoadCompletion>> {
        let (min_id, id_span) = request_id_bounds(output_order)?;
        let mut by_id: Vec<Option<NativeLoadCompletion>> = (0..id_span).map(|_| None).collect();
        let request_by_id = if self.block_cache.is_some() {
            Some(build_request_by_id(output_order, min_id, id_span)?)
        } else {
            None
        };

        for read in reads {
            let bytes = self
                .io
                .submit_read(read.file, read.offset, read.len, read.priority)
                .await?;
            if bytes.len() != read.len {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "native inline load read returned {} bytes, expected {}",
                        bytes.len(),
                        read.len
                    ),
                ));
            }
            for child in read.children {
                let start = child.relative_offset;
                let end = start.checked_add(child.len).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "native load child overflow")
                })?;
                if end > bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "native load child range exceeds coalesced read",
                    ));
                }
                let idx = request_id_index(child.request_id, min_id, id_span)?;
                if by_id[idx].is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("native inline load duplicate child id {}", child.request_id),
                    ));
                }
                by_id[idx] = Some(NativeLoadCompletion {
                    request_id: child.request_id,
                    bytes: Arc::clone(&bytes),
                    range: start..end,
                });
                if let Some(request_by_id) = request_by_id.as_ref() {
                    let request = request_by_id[idx].ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "native inline load child request missing",
                        )
                    })?;
                    self.insert_child_cache(request, &bytes, start, end);
                }
            }
        }

        output_order
            .iter()
            .map(|request| {
                let idx = request_id_index(request.id, min_id, id_span)?;
                by_id[idx].take().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("native load missing completion for request {}", request.id),
                    )
                })
            })
            .collect()
    }

    fn insert_child_cache(
        &self,
        request: NativeLoadRequest,
        bytes: &Arc<[u8]>,
        start: usize,
        end: usize,
    ) {
        let Some(cache) = &self.block_cache else {
            return;
        };
        let cached = if start == 0 && end == bytes.len() {
            Arc::clone(bytes)
        } else {
            Arc::from(bytes[start..end].to_vec().into_boxed_slice())
        };
        cache.insert(request, cached);
    }
}

struct RegisteredNativeLoads {
    misses: Vec<NativeLoadRequest>,
    owners: Vec<NativeOwnedInFlightPayload>,
    waiters: Vec<NativeWaitingInFlightPayload>,
}

fn build_request_by_id(
    output_order: &[NativeLoadRequest],
    min_id: u64,
    id_span: usize,
) -> io::Result<Vec<Option<NativeLoadRequest>>> {
    let mut by_id = vec![None; id_span];
    for request in output_order {
        let idx = request_id_index(request.id, min_id, id_span)?;
        if by_id[idx].is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("native load duplicate request id {}", request.id),
            ));
        }
        by_id[idx] = Some(*request);
    }
    Ok(by_id)
}

fn merge_cached_and_loaded_completions(
    output_order: &[NativeLoadRequest],
    cached: Vec<NativeLoadCompletion>,
    loaded: Vec<NativeLoadCompletion>,
) -> io::Result<Vec<NativeLoadCompletion>> {
    if cached.is_empty() {
        debug_assert_completions_match_order(output_order, &loaded);
        return Ok(loaded);
    }
    if loaded.is_empty() && cached.len() == output_order.len() {
        debug_assert_completions_match_order(output_order, &cached);
        return Ok(cached);
    }
    merge_completions_by_request_id(output_order, cached, loaded)
}

fn merge_completions_by_request_id(
    output_order: &[NativeLoadRequest],
    cached: Vec<NativeLoadCompletion>,
    loaded: Vec<NativeLoadCompletion>,
) -> io::Result<Vec<NativeLoadCompletion>> {
    let (min_id, id_span) = request_id_bounds(output_order)?;
    let mut by_id: Vec<Option<NativeLoadCompletion>> = (0..id_span).map(|_| None).collect();
    let mut completion_count = 0usize;
    for completion in cached.into_iter().chain(loaded) {
        let idx = request_id_index(completion.request_id, min_id, id_span)?;
        if by_id[idx].is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "native load duplicate completion id {}",
                    completion.request_id
                ),
            ));
        }
        by_id[idx] = Some(completion);
        completion_count += 1;
    }
    if completion_count != output_order.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "native load completion count mismatch: got {}, expected {}",
                completion_count,
                output_order.len()
            ),
        ));
    }
    output_order
        .iter()
        .map(|request| {
            let idx = request_id_index(request.id, min_id, id_span)?;
            by_id[idx].take().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("native load missing completion for request {}", request.id),
                )
            })
        })
        .collect()
}

fn request_id_bounds(output_order: &[NativeLoadRequest]) -> io::Result<(u64, usize)> {
    let min_id = output_order
        .iter()
        .map(|request| request.id)
        .min()
        .unwrap_or(0);
    let max_id = output_order
        .iter()
        .map(|request| request.id)
        .max()
        .unwrap_or(min_id);
    let id_span = max_id
        .checked_sub(min_id)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "native load id overflow"))?;
    let id_span = usize::try_from(id_span)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "native load id span too large"))?;
    debug_assert!(
        id_span >= output_order.len(),
        "native load id span too small"
    );
    Ok((min_id, id_span))
}

fn request_id_index(request_id: u64, min_id: u64, id_span: usize) -> io::Result<usize> {
    let idx = usize::try_from(request_id.checked_sub(min_id).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "native load request id underflow",
        )
    })?)
    .map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "native load request id overflow",
        )
    })?;
    if idx >= id_span {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("native load request id {request_id} out of range"),
        ));
    }
    Ok(idx)
}

fn debug_assert_completions_match_order(
    output_order: &[NativeLoadRequest],
    completions: &[NativeLoadCompletion],
) {
    debug_assert_eq!(
        output_order.len(),
        completions.len(),
        "native load completion count mismatch"
    );
    debug_assert!(
        output_order
            .iter()
            .zip(completions)
            .all(|(request, completion)| request.id == completion.request_id),
        "native load completions must be returned in request order"
    );
}

pub(crate) fn coalesce_load_requests(
    requests: &[NativeLoadRequest],
    config: &NativeLoadCoalesceConfig,
) -> Vec<CoalescedRead> {
    if requests.is_empty() {
        return Vec::new();
    }
    let mut sorted = requests.to_vec();
    sorted.sort_by_key(|request| (request.file.0, request.offset));

    let mut out = Vec::new();
    let mut current = CoalescedBuilder::new(sorted[0]);
    for request in sorted.into_iter().skip(1) {
        if current.can_absorb(request, config) {
            current.absorb(request);
        } else {
            out.push(current.finish());
            current = CoalescedBuilder::new(request);
        }
    }
    out.push(current.finish());
    out
}

/// Coalesce pre-sorted requests, skipping the `to_vec` + sort that
/// [`coalesce_load_requests`] performs.
///
/// `requests` must be sorted ascending by `(file, offset)` — the planner
/// emits reads in block order, which already satisfies this. The native hot
/// path calls this directly; the sorting variant is kept for tests and any
/// future caller that cannot guarantee ordering.
pub(crate) fn coalesce_load_requests_presorted(
    requests: &[NativeLoadRequest],
    config: &NativeLoadCoalesceConfig,
) -> Vec<CoalescedRead> {
    if requests.is_empty() {
        return Vec::new();
    }
    debug_assert!(
        requests
            .windows(2)
            .all(|w| (w[0].file.0, w[0].offset) <= (w[1].file.0, w[1].offset)),
        "coalesce_load_requests_presorted: input must be sorted by (file, offset)",
    );
    let mut out = Vec::new();
    let mut current = CoalescedBuilder::new(requests[0]);
    for request in requests.iter().skip(1).copied() {
        if current.can_absorb(request, config) {
            current.absorb(request);
        } else {
            out.push(current.finish());
            current = CoalescedBuilder::new(request);
        }
    }
    out.push(current.finish());
    out
}

#[derive(Debug)]
struct CoalescedBuilder {
    file: FileRef,
    offset: u64,
    end: u64,
    useful_len: usize,
    priority: u8,
    children: Vec<PendingChild>,
}

#[derive(Debug)]
struct PendingChild {
    request_id: u64,
    offset: u64,
    len: usize,
}

impl CoalescedBuilder {
    fn new(request: NativeLoadRequest) -> Self {
        let end = request_end(request);
        Self {
            file: request.file,
            offset: request.offset,
            end,
            useful_len: request.len,
            priority: request.priority,
            children: vec![PendingChild {
                request_id: request.id,
                offset: request.offset,
                len: request.len,
            }],
        }
    }

    fn can_absorb(&self, request: NativeLoadRequest, config: &NativeLoadCoalesceConfig) -> bool {
        if request.file != self.file {
            return false;
        }
        let request_end = request_end(request);
        let merged_start = self.offset.min(request.offset);
        let merged_end = self.end.max(request_end);
        let Some(merged_len) = usize::try_from(merged_end - merged_start).ok() else {
            return false;
        };
        if merged_len > config.max_merged_len {
            return false;
        }

        let gap = request.offset.saturating_sub(self.end);
        let Ok(gap) = usize::try_from(gap) else {
            return false;
        };
        if gap > config.max_gap_bytes {
            return false;
        }

        let useful_len = self.useful_len.saturating_add(request.len);
        if useful_len > merged_len {
            return true;
        }
        let waste = merged_len - useful_len;
        let waste_ratio = waste as f32 / merged_len as f32;
        waste_ratio <= config.max_waste_ratio
    }

    fn absorb(&mut self, request: NativeLoadRequest) {
        self.end = self.end.max(request_end(request));
        self.offset = self.offset.min(request.offset);
        self.useful_len = self.useful_len.saturating_add(request.len);
        self.priority = self.priority.min(request.priority);
        self.children.push(PendingChild {
            request_id: request.id,
            offset: request.offset,
            len: request.len,
        });
    }

    fn finish(self) -> CoalescedRead {
        let len = usize::try_from(self.end - self.offset).expect("coalesced read length overflow");
        let children = self
            .children
            .into_iter()
            .map(|child| CoalescedChild {
                request_id: child.request_id,
                relative_offset: usize::try_from(child.offset - self.offset)
                    .expect("coalesced child offset overflow"),
                len: child.len,
            })
            .collect();
        CoalescedRead {
            file: self.file,
            offset: self.offset,
            len,
            priority: self.priority,
            children,
        }
    }
}

fn request_end(request: NativeLoadRequest) -> u64 {
    request
        .offset
        .checked_add(request.len as u64)
        .expect("native load request end overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(id: u64, file: u64, offset: u64, len: usize) -> NativeLoadRequest {
        NativeLoadRequest {
            id,
            file: FileRef::new(file),
            offset,
            len,
            priority: 1,
        }
    }

    fn config() -> NativeLoadCoalesceConfig {
        NativeLoadCoalesceConfig {
            max_window_us: 0,
            max_merged_len: 1024,
            max_gap_bytes: 16,
            max_waste_ratio: 0.25,
            min_children: 2,
        }
    }

    fn completion(id: u64, offset: u64, len: usize) -> NativeLoadCompletion {
        NativeLoadCompletion {
            request_id: id,
            bytes: Arc::from(bytes(offset, len).into_boxed_slice()),
            range: 0..len,
        }
    }

    #[test]
    fn merge_completions_restores_request_order() {
        let requests = [request(10, 7, 100, 20), request(11, 7, 124, 20)];
        let completions = merge_completions_by_request_id(
            &requests,
            vec![completion(11, 124, 20)],
            vec![completion(10, 100, 20)],
        )
        .expect("merge completions");

        assert_eq!(completions[0].request_id, 10);
        assert_eq!(completions[1].request_id, 11);
        assert_eq!(
            &completions[0].bytes[completions[0].range.clone()],
            &bytes(100, 20)
        );
        assert_eq!(
            &completions[1].bytes[completions[1].range.clone()],
            &bytes(124, 20)
        );
    }

    #[test]
    fn merge_completions_rejects_duplicate_ids() {
        let requests = [request(10, 7, 100, 20), request(11, 7, 124, 20)];
        let err = merge_completions_by_request_id(
            &requests,
            vec![completion(10, 100, 20)],
            vec![completion(10, 124, 20)],
        )
        .expect_err("duplicate completion id should fail");

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("duplicate completion id 10"));
    }

    #[test]
    fn coalesces_nearby_same_file_ranges() {
        let reads =
            coalesce_load_requests(&[request(1, 7, 100, 20), request(2, 7, 124, 20)], &config());
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0].offset, 100);
        assert_eq!(reads[0].len, 44);
        assert_eq!(
            reads[0].children,
            vec![
                CoalescedChild {
                    request_id: 1,
                    relative_offset: 0,
                    len: 20,
                },
                CoalescedChild {
                    request_id: 2,
                    relative_offset: 24,
                    len: 20,
                },
            ]
        );
    }

    #[test]
    fn keeps_different_files_separate() {
        let reads =
            coalesce_load_requests(&[request(1, 1, 0, 16), request(2, 2, 8, 16)], &config());
        assert_eq!(reads.len(), 2);
        assert_ne!(reads[0].file, reads[1].file);
    }

    #[test]
    fn rejects_high_waste_merge() {
        let reads =
            coalesce_load_requests(&[request(1, 1, 0, 16), request(2, 1, 100, 16)], &config());
        assert_eq!(reads.len(), 2);
    }

    #[tokio::test]
    async fn load_splits_coalesced_read_back_to_requests() {
        let loader = NativeLoadModule::new(Arc::new(MockIoBackend), config());
        let completions = loader
            .load(&[request(1, 7, 100, 20), request(2, 7, 124, 20)])
            .await
            .expect("native load");

        assert_eq!(completions.len(), 2);
        assert_eq!(completions[0].request_id, 1);
        assert_eq!(
            &completions[0].bytes[completions[0].range.clone()],
            &bytes(100, 20)
        );
        assert_eq!(completions[1].request_id, 2);
        assert_eq!(
            &completions[1].bytes[completions[1].range.clone()],
            &bytes(124, 20)
        );
        assert!(Arc::ptr_eq(&completions[0].bytes, &completions[1].bytes));
    }

    #[tokio::test]
    async fn block_payload_cache_reuses_exact_request() {
        let io = Arc::new(CountingIoBackend::default());
        let cache = Arc::new(NativeBlockPayloadCache::new(4096, 8));
        let loader = NativeLoadModule::with_block_cache(io.clone(), config(), Some(cache));
        let first = loader
            .load(&[request(1, 7, 100, 20)])
            .await
            .expect("first native load");
        let second = loader
            .load(&[request(2, 7, 100, 20)])
            .await
            .expect("second native load");

        assert_eq!(io.reads(), 1);
        assert_eq!(&first[0].bytes[first[0].range.clone()], &bytes(100, 20));
        assert_eq!(&second[0].bytes[second[0].range.clone()], &bytes(100, 20));
    }

    #[tokio::test]
    async fn in_flight_payload_reads_share_concurrent_exact_request() {
        let io = Arc::new(YieldingCountingIoBackend::default());
        let in_flight = Arc::new(NativeInFlightPayloadReads::new());
        let loader = NativeLoadModule::with_caches(io.clone(), config(), None, Some(in_flight));
        let first_request = [request(1, 7, 100, 20)];
        let second_request = [request(2, 7, 100, 20)];

        let (first, second) =
            tokio::join!(loader.load(&first_request), loader.load(&second_request),);
        let first = first.expect("first native load");
        let second = second.expect("second native load");

        assert_eq!(io.reads(), 1);
        assert_eq!(first[0].request_id, 1);
        assert_eq!(second[0].request_id, 2);
        assert_eq!(&first[0].bytes[first[0].range.clone()], &bytes(100, 20));
        assert_eq!(&second[0].bytes[second[0].range.clone()], &bytes(100, 20));
    }

    #[tokio::test]
    async fn in_flight_waiters_preserve_request_order_with_uncached_misses() {
        let io = Arc::new(BlockingCountingIoBackend::new(2));
        let in_flight = Arc::new(NativeInFlightPayloadReads::new());
        let loader = NativeLoadModule::with_caches(io.clone(), config(), None, Some(in_flight));

        let first_loader = loader.clone();
        let first = tokio::spawn(async move { first_loader.load(&[request(1, 7, 100, 20)]).await });
        wait_for_reads(&io, 1).await;

        let second_loader = loader.clone();
        let second = tokio::spawn(async move {
            second_loader
                .load(&[request(2, 7, 100, 20), request(3, 7, 124, 20)])
                .await
        });
        wait_for_reads(&io, 2).await;
        io.release().await;

        let first = first.await.expect("first task").expect("first native load");
        let second = second
            .await
            .expect("second task")
            .expect("second native load");

        assert_eq!(io.reads(), 2);
        assert_eq!(first[0].request_id, 1);
        assert_eq!(second[0].request_id, 2);
        assert_eq!(second[1].request_id, 3);
        assert_eq!(&second[0].bytes[second[0].range.clone()], &bytes(100, 20));
        assert_eq!(&second[1].bytes[second[1].range.clone()], &bytes(124, 20));
    }

    struct MockIoBackend;

    impl IoBackend for MockIoBackend {
        fn submit_read(&self, _file: FileRef, offset: u64, len: usize, _priority: u8) -> IoTask {
            Box::pin(async move { Ok(Arc::from(bytes(offset, len).into_boxed_slice())) })
        }
    }

    #[derive(Default)]
    struct CountingIoBackend {
        reads: std::sync::atomic::AtomicUsize,
    }

    impl CountingIoBackend {
        fn reads(&self) -> usize {
            self.reads.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl IoBackend for CountingIoBackend {
        fn submit_read(&self, _file: FileRef, offset: u64, len: usize, _priority: u8) -> IoTask {
            self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async move { Ok(Arc::from(bytes(offset, len).into_boxed_slice())) })
        }
    }

    #[derive(Default)]
    struct YieldingCountingIoBackend {
        reads: std::sync::atomic::AtomicUsize,
    }

    impl YieldingCountingIoBackend {
        fn reads(&self) -> usize {
            self.reads.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl IoBackend for YieldingCountingIoBackend {
        fn submit_read(&self, _file: FileRef, offset: u64, len: usize, _priority: u8) -> IoTask {
            self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async move {
                tokio::task::yield_now().await;
                Ok(Arc::from(bytes(offset, len).into_boxed_slice()))
            })
        }
    }

    struct BlockingCountingIoBackend {
        reads: std::sync::atomic::AtomicUsize,
        barrier: Arc<tokio::sync::Barrier>,
    }

    impl BlockingCountingIoBackend {
        fn new(reads: usize) -> Self {
            Self {
                reads: std::sync::atomic::AtomicUsize::new(0),
                barrier: Arc::new(tokio::sync::Barrier::new(reads + 1)),
            }
        }

        fn reads(&self) -> usize {
            self.reads.load(std::sync::atomic::Ordering::SeqCst)
        }

        async fn release(&self) {
            self.barrier.wait().await;
        }
    }

    impl IoBackend for BlockingCountingIoBackend {
        fn submit_read(&self, _file: FileRef, offset: u64, len: usize, _priority: u8) -> IoTask {
            self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let barrier = Arc::clone(&self.barrier);
            Box::pin(async move {
                barrier.wait().await;
                Ok(Arc::from(bytes(offset, len).into_boxed_slice()))
            })
        }
    }

    async fn wait_for_reads(io: &BlockingCountingIoBackend, reads: usize) {
        while io.reads() < reads {
            tokio::task::yield_now().await;
        }
    }

    fn bytes(offset: u64, len: usize) -> Vec<u8> {
        (0..len)
            .map(|idx| offset.wrapping_add(idx as u64) as u8)
            .collect()
    }
}
