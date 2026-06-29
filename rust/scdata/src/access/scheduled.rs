//! Server-side scheduled access staging state.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tokio::sync::Notify;

#[derive(Debug, Default)]
pub(crate) struct ScheduledStore {
    entries: HashMap<u64, ScheduledStage>,
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
        if is_evictable {
            self.evictable.push_back(id);
            self.compact_evictable_if_needed();
        }
        old
    }

    pub(crate) fn remove(&mut self, id: &u64) -> Option<ScheduledStage> {
        let stage = self.entries.remove(id);
        if self.entries.is_empty() {
            self.evictable.clear();
        }
        stage
    }

    pub(crate) fn evict_one_buffer(&mut self) -> Option<usize> {
        while let Some(id) = self.evictable.pop_front() {
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
        while let Some(id) = self.evictable.pop_front() {
            if self
                .entries
                .get(&id)
                .is_some_and(ScheduledStage::is_evictable)
            {
                compacted.push_back(id);
            }
        }
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
}
