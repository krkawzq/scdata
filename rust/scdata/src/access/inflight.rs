//! In-flight deduplication for compressed chunk reads.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Notify;

use super::scheduler::ChunkKey;

#[derive(Debug)]
struct InflightEntry {
    notify: Arc<Notify>,
}

/// Deduplicates concurrent misses for the same chunk key.
#[derive(Debug, Default)]
pub(crate) struct InflightTable {
    entries: HashMap<ChunkKey, InflightEntry>,
}

/// Result of attempting to register a miss.
pub(crate) enum RegisterResult {
    /// The caller owns the read and must submit IO.
    First,
    /// Another task owns the read; wait for this notification and retry.
    Waiter(Arc<Notify>),
}

impl InflightTable {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn try_register(&mut self, key: ChunkKey) -> RegisterResult {
        if let Some(entry) = self.entries.get(&key) {
            return RegisterResult::Waiter(Arc::clone(&entry.notify));
        }

        self.entries.insert(
            key,
            InflightEntry {
                notify: Arc::new(Notify::new()),
            },
        );
        RegisterResult::First
    }

    /// Remove the entry and wake every waiter. Waiters always re-check cache.
    pub(crate) fn complete(&mut self, key: &ChunkKey) {
        if let Some(entry) = self.entries.remove(key) {
            entry.notify.notify_waiters();
        }
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::FileRef;

    #[test]
    fn first_registration_owns_the_read() {
        let mut table = InflightTable::new();
        let key = ChunkKey::new(FileRef(1), 8, 4);

        assert!(matches!(table.try_register(key), RegisterResult::First));
        assert!(matches!(table.try_register(key), RegisterResult::Waiter(_)));
        assert_eq!(table.len(), 1);

        table.complete(&key);
        assert!(table.is_empty());
        assert!(matches!(table.try_register(key), RegisterResult::First));
    }
}
