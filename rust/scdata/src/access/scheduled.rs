//! Server-side scheduled access staging state.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use tokio::sync::Notify;

#[derive(Debug, Default)]
pub(crate) struct ScheduledStore {
    entries: HashMap<u64, ScheduledStage>,
    evictable_ids: HashSet<u64>,
    evictable: VecDeque<u64>,
}

impl ScheduledStore {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn contains(&self, id: &u64) -> bool {
        self.entries.contains_key(id)
    }

    pub(crate) fn get(&self, id: &u64) -> Option<&ScheduledStage> {
        self.entries.get(id)
    }

    pub(crate) fn insert(&mut self, id: u64, stage: ScheduledStage) -> Option<ScheduledStage> {
        let is_evictable = stage.is_evictable();
        let old = self.entries.insert(id, stage);
        let old_evictable = old.as_ref().is_some_and(ScheduledStage::is_evictable);
        if is_evictable {
            if self.evictable_ids.insert(id) {
                self.evictable.push_back(id);
            }
            self.compact_evictable_if_needed();
        } else if old_evictable {
            self.evictable_ids.remove(&id);
        }
        old
    }

    pub(crate) fn remove(&mut self, id: &u64) -> Option<ScheduledStage> {
        let stage = self.entries.remove(id);
        if stage.as_ref().is_some_and(ScheduledStage::is_evictable) {
            self.evictable_ids.remove(id);
        }
        if self.entries.is_empty() {
            self.evictable_ids.clear();
            self.evictable.clear();
        }
        stage
    }

    pub(crate) fn evict_one_buffer(&mut self) -> Option<usize> {
        while let Some(id) = self.evictable.pop_front() {
            if !self.evictable_ids.remove(&id) {
                continue;
            }
            let Some(stage) = self.entries.remove(&id) else {
                continue;
            };
            match stage {
                ScheduledStage::Decoded { bytes, .. } | ScheduledStage::Ready { bytes, .. } => {
                    return Some(bytes);
                }
                stage => {
                    self.entries.insert(id, stage);
                }
            }
        }
        None
    }

    fn compact_evictable_if_needed(&mut self) {
        let threshold = self.entries.len().saturating_mul(4).max(64);
        if self.evictable.len() <= threshold {
            return;
        }

        let mut compacted = VecDeque::with_capacity(self.entries.len().min(self.evictable.len()));
        let mut compacted_ids = HashSet::with_capacity(self.evictable_ids.len());
        while let Some(id) = self.evictable.pop_front() {
            if compacted_ids.contains(&id) {
                continue;
            }
            if self
                .entries
                .get(&id)
                .is_some_and(ScheduledStage::is_evictable)
            {
                compacted_ids.insert(id);
                compacted.push_back(id);
            }
        }
        self.evictable_ids = compacted_ids;
        self.evictable = compacted;
    }

    #[cfg(test)]
    fn evictable_len(&self) -> usize {
        self.evictable.len()
    }
}

#[derive(Debug)]
pub(crate) enum ScheduledStage {
    Pending(Arc<Notify>),
    Complete,
    Decoded { data: StagedBytes, bytes: usize },
    Ready { data: Vec<u8>, bytes: usize },
    Failed(String),
    Cancelled,
}

impl ScheduledStage {
    fn is_evictable(&self) -> bool {
        matches!(self, Self::Decoded { .. } | Self::Ready { .. })
    }
}

#[derive(Debug)]
pub(crate) enum StagedBytes {
    Owned(Vec<u8>),
    Shared(Arc<[u8]>),
}

impl StagedBytes {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(data) => data.as_slice(),
            Self::Shared(data) => data.as_ref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evictable_queue_does_not_grow_with_removed_entries() {
        let mut store = ScheduledStore::new();

        for id in 0..256 {
            store.insert(
                id,
                ScheduledStage::Ready {
                    data: Vec::new(),
                    bytes: 0,
                },
            );
            store.remove(&id);
        }

        assert!(store.evictable_len() <= 64);
        assert!(store.entries.is_empty());
    }

    #[test]
    fn evictable_queue_deduplicates_repeated_stage_updates() {
        let mut store = ScheduledStore::new();

        for bytes in 1..=128 {
            store.insert(
                7,
                ScheduledStage::Ready {
                    data: Vec::new(),
                    bytes,
                },
            );
        }

        assert_eq!(store.evictable_len(), 1);
        assert_eq!(store.evict_one_buffer(), Some(128));
        assert_eq!(store.evict_one_buffer(), None);
    }
}
