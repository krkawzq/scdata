//! Pin-aware LRU cache for raw and decoded chunk bytes.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use tokio::sync::Notify;

use super::scheduler::{ChunkKey, DecodeKey};
use crate::codecs::SharedCodec;

/// The representation currently stored for one chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CachePayloadKind {
    Raw,
    Decoded,
}

#[derive(Debug, Clone)]
pub(crate) enum CachePayload {
    Raw(Arc<[u8]>),
    Decoded {
        data: Arc<[u8]>,
        decode_key: DecodeKey,
        _codec: SharedCodec,
    },
}

impl CachePayload {
    pub(crate) fn kind(&self) -> CachePayloadKind {
        match self {
            Self::Raw(_) => CachePayloadKind::Raw,
            Self::Decoded { .. } => CachePayloadKind::Decoded,
        }
    }

    pub(crate) fn data(&self) -> Arc<[u8]> {
        match self {
            Self::Raw(data) | Self::Decoded { data, .. } => Arc::clone(data),
        }
    }

    pub(crate) fn decode_key(&self) -> Option<DecodeKey> {
        match self {
            Self::Raw(_) => None,
            Self::Decoded { decode_key, .. } => Some(*decode_key),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Raw(data) | Self::Decoded { data, .. } => data.len(),
        }
    }
}

#[derive(Debug)]
struct CacheEntry {
    payload: CachePayload,
    refcount: usize,
    stamp: u64,
}

#[derive(Debug)]
struct CacheInner {
    capacity: usize,
    used: usize,
    entries: HashMap<ChunkKey, CacheEntry>,
    lru: VecDeque<(ChunkKey, u64)>,
    clock: u64,
}

impl CacheInner {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            used: 0,
            entries: HashMap::new(),
            lru: VecDeque::new(),
            clock: 0,
        }
    }

    fn touch(&mut self, key: ChunkKey) {
        let stamp = {
            let Some(entry) = self.entries.get_mut(&key) else {
                return;
            };
            self.clock = self.clock.wrapping_add(1);
            entry.stamp = self.clock;
            entry.stamp
        };
        self.lru.push_back((key, stamp));
        self.compact_lru_if_needed();
    }

    fn compact_lru_if_needed(&mut self) {
        let threshold = self.entries.len().saturating_mul(4).max(64);
        if self.lru.len() <= threshold {
            return;
        }

        let mut compacted = VecDeque::with_capacity(self.entries.len());
        while let Some((key, stamp)) = self.lru.pop_front() {
            if self
                .entries
                .get(&key)
                .is_some_and(|entry| entry.stamp == stamp)
            {
                compacted.push_back((key, stamp));
            }
        }
        self.lru = compacted;
    }

    fn evict_one(&mut self) -> Option<EvictedEntry> {
        self.evict_one_except(None)
    }

    fn evict_one_except(&mut self, skip_key: Option<ChunkKey>) -> Option<EvictedEntry> {
        let visits = self.lru.len();
        for _ in 0..visits {
            let Some((key, stamp)) = self.lru.pop_front() else {
                break;
            };

            let Some(entry) = self.entries.get(&key) else {
                continue;
            };
            if entry.stamp != stamp {
                continue;
            }
            if Some(key) == skip_key || entry.refcount > 0 {
                self.lru.push_back((key, stamp));
                continue;
            }

            let entry = self
                .entries
                .remove(&key)
                .expect("entry checked before eviction");
            let bytes = entry.payload.len();
            self.used = self.used.saturating_sub(bytes);
            return Some(EvictedEntry { key, bytes });
        }
        None
    }

    fn evict_until_fits(
        &mut self,
        additional_bytes: usize,
        skip_key: Option<ChunkKey>,
    ) -> Result<usize, CacheInsertError> {
        let mut evicted_bytes = 0;
        while self.used.saturating_add(additional_bytes) > self.capacity {
            let Some(evicted) = self.evict_one_except(skip_key) else {
                return Err(CacheInsertError::all_pinned(evicted_bytes));
            };
            evicted_bytes += evicted.bytes;
        }
        Ok(evicted_bytes)
    }
}

/// A pinned cache hit. Keeping the guard alive prevents eviction.
pub(crate) struct PinnedChunk {
    pub(crate) data: Arc<[u8]>,
    pub(crate) kind: CachePayloadKind,
    pub(crate) decode_key: Option<DecodeKey>,
    pub(crate) guard: PinGuard,
}

impl fmt::Debug for PinnedChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedChunk")
            .field("kind", &self.kind)
            .field("len", &self.data.len())
            .field("guard", &self.guard)
            .finish()
    }
}

/// RAII guard that unpins a cache entry on drop.
pub(crate) struct PinGuard {
    key: ChunkKey,
    inner: Weak<RefCell<CacheInner>>,
    notify: Arc<Notify>,
}

impl PinGuard {
    fn new(key: ChunkKey, inner: Weak<RefCell<CacheInner>>, notify: Arc<Notify>) -> Self {
        Self { key, inner, notify }
    }

    #[allow(dead_code)]
    pub(crate) fn key(&self) -> ChunkKey {
        self.key
    }
}

impl fmt::Debug for PinGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinGuard")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl Drop for PinGuard {
    fn drop(&mut self) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };

        let became_unpinned = {
            let mut inner = inner.borrow_mut();
            let Some(entry) = inner.entries.get_mut(&self.key) else {
                return;
            };
            entry.refcount = entry.refcount.saturating_sub(1);
            entry.refcount == 0
        };

        if became_unpinned {
            self.notify.notify_waiters();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EvictedEntry {
    pub(crate) key: ChunkKey,
    pub(crate) bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InsertOutcome {
    pub(crate) inserted: bool,
    pub(crate) evicted_bytes: usize,
    pub(crate) replaced_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheInsertErrorKind {
    ItemTooLarge,
    AllPinned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CacheInsertError {
    pub(crate) kind: CacheInsertErrorKind,
    pub(crate) evicted_bytes: usize,
}

impl CacheInsertError {
    fn item_too_large() -> Self {
        Self {
            kind: CacheInsertErrorKind::ItemTooLarge,
            evicted_bytes: 0,
        }
    }

    fn all_pinned(evicted_bytes: usize) -> Self {
        Self {
            kind: CacheInsertErrorKind::AllPinned,
            evicted_bytes,
        }
    }
}

/// LRU cache storing either compressed bytes or decoded bytes per chunk.
#[derive(Debug, Clone)]
pub(crate) struct ChunkCache {
    inner: Rc<RefCell<CacheInner>>,
    unpin_notify: Arc<Notify>,
}

impl ChunkCache {
    pub(crate) fn new(capacity: usize, unpin_notify: Arc<Notify>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(CacheInner::new(capacity))),
            unpin_notify,
        }
    }

    /// Return whether any representation for the key is already cached.
    pub(crate) fn contains(&self, key: &ChunkKey) -> bool {
        self.inner.borrow().entries.contains_key(key)
    }

    /// Pin and clone the cached bytes in one synchronous operation.
    pub(crate) fn pin_and_get(&self, key: &ChunkKey) -> Option<PinnedChunk> {
        let mut inner = self.inner.borrow_mut();
        let entry = inner.entries.get_mut(key)?;
        entry.refcount += 1;
        let kind = entry.payload.kind();
        let data = entry.payload.data();
        let decode_key = entry.payload.decode_key();
        inner.touch(*key);
        Some(PinnedChunk {
            data,
            kind,
            decode_key,
            guard: PinGuard::new(
                *key,
                Rc::downgrade(&self.inner),
                Arc::clone(&self.unpin_notify),
            ),
        })
    }

    /// Insert only when no cached representation exists for this key.
    pub(crate) fn insert_if_absent(
        &self,
        key: ChunkKey,
        payload: CachePayload,
    ) -> Result<InsertOutcome, CacheInsertError> {
        let size = payload.len();
        let mut inner = self.inner.borrow_mut();

        if size > inner.capacity {
            return Err(CacheInsertError::item_too_large());
        }

        if inner.entries.contains_key(&key) {
            inner.touch(key);
            return Ok(InsertOutcome {
                inserted: false,
                evicted_bytes: 0,
                replaced_bytes: 0,
            });
        }

        let evicted_bytes = inner.evict_until_fits(size, None)?;
        inner.used += size;
        inner.entries.insert(
            key,
            CacheEntry {
                payload,
                refcount: 0,
                stamp: 0,
            },
        );
        inner.touch(key);
        Ok(InsertOutcome {
            inserted: true,
            evicted_bytes,
            replaced_bytes: 0,
        })
    }

    /// Insert the payload, replacing an unpinned existing representation.
    pub(crate) fn insert_or_replace(
        &self,
        key: ChunkKey,
        payload: CachePayload,
    ) -> Result<InsertOutcome, CacheInsertError> {
        let size = payload.len();
        let mut inner = self.inner.borrow_mut();

        if size > inner.capacity {
            return Err(CacheInsertError::item_too_large());
        }

        let replaced_bytes = if let Some(entry) = inner.entries.get(&key) {
            if entry.refcount > 0 {
                return Err(CacheInsertError::all_pinned(0));
            }
            entry.payload.len()
        } else {
            0
        };

        if replaced_bytes > 0 {
            inner.used = inner.used.saturating_sub(replaced_bytes);
        }

        let evicted = match inner.evict_until_fits(size, Some(key)) {
            Ok(evicted) => evicted,
            Err(err) => {
                if replaced_bytes > 0 {
                    inner.used += replaced_bytes;
                }
                return Err(err);
            }
        };

        if replaced_bytes > 0 {
            let entry = inner
                .entries
                .get_mut(&key)
                .expect("replacement target checked above");
            entry.payload = payload;
            entry.refcount = 0;
        } else {
            inner.entries.insert(
                key,
                CacheEntry {
                    payload,
                    refcount: 0,
                    stamp: 0,
                },
            );
        }
        inner.used += size;
        inner.touch(key);

        Ok(InsertOutcome {
            inserted: true,
            evicted_bytes: evicted,
            replaced_bytes,
        })
    }

    /// Evict one least-recently used unpinned entry.
    pub(crate) fn evict_one(&self) -> Option<EvictedEntry> {
        self.inner.borrow_mut().evict_one()
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.inner.borrow().entries.len()
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[allow(dead_code)]
    pub(crate) fn used_bytes(&self) -> usize {
        self.inner.borrow().used
    }

    #[allow(dead_code)]
    pub(crate) fn capacity(&self) -> usize {
        self.inner.borrow().capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::FileRef;
    use crate::codecs::UncompressedCodec;

    fn key(offset: u64) -> ChunkKey {
        ChunkKey::new(FileRef(1), offset, 4)
    }

    fn cache(capacity: usize) -> ChunkCache {
        ChunkCache::new(capacity, Arc::new(Notify::new()))
    }

    fn raw(bytes: &'static [u8]) -> CachePayload {
        CachePayload::Raw(Arc::from(bytes))
    }

    fn decoded(bytes: &'static [u8], offset: u64) -> CachePayload {
        CachePayload::Decoded {
            data: Arc::from(bytes),
            decode_key: DecodeKey {
                chunk: key(offset),
                codec: 1,
                expected_size: Some(bytes.len()),
            },
            _codec: Arc::new(UncompressedCodec),
        }
    }

    #[test]
    fn cache_hits_promote_entries() {
        let cache = cache(8);
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 8);

        cache.insert_if_absent(key(0), raw(b"aaaa")).unwrap();
        cache.insert_if_absent(key(4), raw(b"bbbb")).unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.used_bytes(), 8);

        assert_eq!(&*cache.pin_and_get(&key(0)).expect("hit").data, b"aaaa");
        cache.insert_if_absent(key(8), raw(b"cccc")).unwrap();

        assert!(cache.pin_and_get(&key(0)).is_some());
        assert!(cache.pin_and_get(&key(4)).is_none());
        assert!(cache.pin_and_get(&key(8)).is_some());
    }

    #[test]
    fn pinned_entries_are_not_evicted() {
        let cache = cache(8);
        cache.insert_if_absent(key(0), raw(b"aaaa")).unwrap();
        cache.insert_if_absent(key(4), raw(b"bbbb")).unwrap();
        let pinned = cache.pin_and_get(&key(0)).expect("pin");

        cache.insert_if_absent(key(8), raw(b"cccc")).unwrap();

        assert!(cache.pin_and_get(&key(0)).is_some());
        assert!(cache.pin_and_get(&key(4)).is_none());
        assert!(cache.pin_and_get(&key(8)).is_some());
        assert_eq!(pinned.guard.key(), key(0));
    }

    #[test]
    fn all_pinned_cache_rejects_insert_until_unpinned() {
        let cache = cache(4);
        cache.insert_if_absent(key(0), raw(b"aaaa")).unwrap();
        let pinned = cache.pin_and_get(&key(0)).expect("pin");

        assert_eq!(
            cache
                .insert_if_absent(key(4), raw(b"bbbb"))
                .expect_err("all entries are pinned")
                .kind,
            CacheInsertErrorKind::AllPinned
        );

        drop(pinned);
        assert!(cache.insert_if_absent(key(4), raw(b"bbbb")).is_ok());
        assert!(cache.pin_and_get(&key(0)).is_none());
        assert!(cache.pin_and_get(&key(4)).is_some());
    }

    #[test]
    fn partial_eviction_failure_reports_evicted_bytes() {
        let cache = cache(8);
        cache.insert_if_absent(key(0), raw(b"aaaa")).unwrap();
        cache.insert_if_absent(key(4), raw(b"bbbb")).unwrap();
        let _pinned = cache.pin_and_get(&key(4)).expect("pin");

        let err = cache
            .insert_if_absent(key(8), raw(b"cccccccc"))
            .expect_err("remaining entry is pinned");

        assert_eq!(err.kind, CacheInsertErrorKind::AllPinned);
        assert_eq!(err.evicted_bytes, 4);
        assert_eq!(cache.used_bytes(), 4);
        assert!(cache.pin_and_get(&key(0)).is_none());
        assert!(cache.pin_and_get(&key(4)).is_some());
    }

    #[test]
    fn decoded_payload_replaces_unpinned_raw_payload() {
        let cache = cache(16);
        cache.insert_if_absent(key(0), raw(b"raw")).unwrap();

        let outcome = cache
            .insert_or_replace(key(0), decoded(b"decoded", 0))
            .unwrap();
        assert_eq!(outcome.replaced_bytes, 3);

        let pinned = cache.pin_and_get(&key(0)).expect("decoded hit");
        assert_eq!(pinned.kind, CachePayloadKind::Decoded);
        assert_eq!(pinned.decode_key.expect("decode key").chunk, key(0));
        assert_eq!(&*pinned.data, b"decoded");
    }
}
