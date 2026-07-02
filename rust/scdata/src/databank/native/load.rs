use std::collections::{HashMap, VecDeque};
use std::io;
use std::ops::Range;
use std::sync::{Arc, Mutex};

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
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        const SHARDS: usize = 8;
        let shard_capacity = capacity_bytes.div_ceil(SHARDS).max(1);
        let shards = (0..SHARDS)
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

#[derive(Clone)]
pub(crate) struct NativeLoadModule {
    io: Arc<dyn IoBackend>,
    coalesce: NativeLoadCoalesceConfig,
    block_cache: Option<Arc<NativeBlockPayloadCache>>,
}

impl NativeLoadModule {
    pub(crate) fn new(io: Arc<dyn IoBackend>, coalesce: NativeLoadCoalesceConfig) -> Self {
        Self {
            io,
            coalesce,
            block_cache: None,
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

        let reads = coalesce_load_requests_presorted(requests, &self.coalesce);
        self.load_coalesced(requests, reads).await
    }

    pub(crate) async fn load_unsorted(
        &self,
        requests: &[NativeLoadRequest],
    ) -> io::Result<Vec<NativeLoadCompletion>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let reads = coalesce_load_requests(requests, &self.coalesce);
        self.load_coalesced(requests, reads).await
    }

    async fn load_coalesced(
        &self,
        output_order: &[NativeLoadRequest],
        reads: Vec<CoalescedRead>,
    ) -> io::Result<Vec<NativeLoadCompletion>> {
        let original_order = output_order.to_vec();
        let (cached, output_order, reads) = self.split_cached_completions(output_order, reads)?;
        if reads.is_empty() {
            return merge_cached_and_loaded_completions(&original_order, cached, Vec::new());
        }
        let loaded = if self.io.prefers_inline_reads() {
            self.load_coalesced_inline(&output_order, reads).await?
        } else {
            self.load_coalesced_parallel(&output_order, reads).await?
        };
        merge_cached_and_loaded_completions(&original_order, cached, loaded)
    }

    fn split_cached_completions(
        &self,
        output_order: &[NativeLoadRequest],
        reads: Vec<CoalescedRead>,
    ) -> io::Result<(
        Vec<NativeLoadCompletion>,
        Vec<NativeLoadRequest>,
        Vec<CoalescedRead>,
    )> {
        let Some(cache) = &self.block_cache else {
            return Ok((Vec::new(), output_order.to_vec(), reads));
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
        if cached.is_empty() {
            return Ok((cached, misses, reads));
        }
        if misses.is_empty() {
            return Ok((cached, Vec::new(), Vec::new()));
        }
        let mut miss_by_id = HashMap::with_capacity(misses.len());
        for request in &misses {
            miss_by_id.insert(request.id, *request);
        }
        let mut filtered_reads = Vec::new();
        for mut read in reads {
            read.children
                .retain(|child| miss_by_id.contains_key(&child.request_id));
            if read.children.is_empty() {
                continue;
            }
            let first_child = read
                .children
                .iter()
                .map(|child| child.relative_offset)
                .min();
            let last_child = read.children.iter().try_fold(0usize, |end, child| {
                child
                    .relative_offset
                    .checked_add(child.len)
                    .map(|child_end| end.max(child_end))
            });
            let (Some(first_child), Some(last_child)) = (first_child, last_child) else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "native cached read child overflow",
                ));
            };
            read.offset = read.offset.checked_add(first_child as u64).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "native cached read offset overflow",
                )
            })?;
            read.len = last_child.checked_sub(first_child).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "native cached read length underflow",
                )
            })?;
            for child in &mut read.children {
                child.relative_offset =
                    child
                        .relative_offset
                        .checked_sub(first_child)
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "native cached child offset underflow",
                            )
                        })?;
            }
            filtered_reads.push(read);
        }
        Ok((cached, misses, filtered_reads))
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
        let id_span = usize::try_from(id_span).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "native load id span too large")
        })?;
        debug_assert!(
            id_span >= output_order.len(),
            "native load id span too small"
        );
        let mut by_id: Vec<Option<NativeLoadCompletion>> = (0..id_span).map(|_| None).collect();
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
                let idx = usize::try_from(child.request_id - min_id).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "native load child id overflow")
                })?;
                if idx >= by_id.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "native load child id out of range",
                    ));
                }
                by_id[idx] = Some(NativeLoadCompletion {
                    request_id: child.request_id,
                    bytes: Arc::clone(&bytes),
                    range: start..end,
                });
                self.insert_child_cache(output_order, child.request_id, &bytes, start, end);
            }
        }

        output_order
            .iter()
            .map(|request| {
                let idx = usize::try_from(request.id - min_id).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "native load output id overflow")
                })?;
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
        let id_span = usize::try_from(id_span).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "native load id span too large")
        })?;
        debug_assert!(
            id_span >= output_order.len(),
            "native load id span too small"
        );
        let mut by_id: Vec<Option<NativeLoadCompletion>> = (0..id_span).map(|_| None).collect();

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
                let idx = usize::try_from(child.request_id - min_id).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "native load child id overflow")
                })?;
                if idx >= by_id.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "native load child id out of range",
                    ));
                }
                by_id[idx] = Some(NativeLoadCompletion {
                    request_id: child.request_id,
                    bytes: Arc::clone(&bytes),
                    range: start..end,
                });
                self.insert_child_cache(output_order, child.request_id, &bytes, start, end);
            }
        }

        output_order
            .iter()
            .map(|request| {
                let idx = usize::try_from(request.id - min_id).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "native load output id overflow")
                })?;
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
        output_order: &[NativeLoadRequest],
        request_id: u64,
        bytes: &Arc<[u8]>,
        start: usize,
        end: usize,
    ) {
        let Some(cache) = &self.block_cache else {
            return;
        };
        let Some(request) = output_order
            .iter()
            .copied()
            .find(|request| request.id == request_id)
        else {
            return;
        };
        cache.insert(
            request,
            Arc::from(bytes[start..end].to_vec().into_boxed_slice()),
        );
    }
}

fn merge_cached_and_loaded_completions(
    output_order: &[NativeLoadRequest],
    cached: Vec<NativeLoadCompletion>,
    loaded: Vec<NativeLoadCompletion>,
) -> io::Result<Vec<NativeLoadCompletion>> {
    if cached.is_empty() {
        return Ok(loaded);
    }
    let mut by_id = cached
        .into_iter()
        .chain(loaded)
        .map(|completion| (completion.request_id, completion))
        .collect::<HashMap<_, _>>();
    output_order
        .iter()
        .map(|request| {
            by_id.remove(&request.id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("native load missing completion for request {}", request.id),
                )
            })
        })
        .collect()
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
        let cache = Arc::new(NativeBlockPayloadCache::new(4096));
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

    fn bytes(offset: u64, len: usize) -> Vec<u8> {
        (0..len)
            .map(|idx| offset.wrapping_add(idx as u64) as u8)
            .collect()
    }
}
