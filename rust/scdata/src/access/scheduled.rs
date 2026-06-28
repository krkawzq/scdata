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
        if stage.is_evictable() {
            self.evictable.push_back(id);
        }
        self.entries.insert(id, stage)
    }

    pub(crate) fn remove(&mut self, id: &u64) -> Option<ScheduledStage> {
        self.entries.remove(id)
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
