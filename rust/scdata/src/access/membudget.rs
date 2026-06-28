//! Memory accounting shared by cached and in-flight compressed chunks.

use std::sync::Arc;

use tokio::sync::Notify;

/// Byte budget for compressed chunk buffers.
#[derive(Debug)]
pub(crate) struct MemBudget {
    capacity: usize,
    used: usize,
    release_notify: Arc<Notify>,
}

impl MemBudget {
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity: capacity_bytes,
            used: 0,
            release_notify: Arc::new(Notify::new()),
        }
    }

    /// Try to reserve bytes for a buffer that is about to become live.
    pub(crate) fn try_reserve(&mut self, bytes: usize) -> bool {
        let Some(next_used) = self.used.checked_add(bytes) else {
            return false;
        };
        if next_used > self.capacity {
            return false;
        }
        self.used = next_used;
        true
    }

    /// Release bytes that are no longer live.
    pub(crate) fn release(&mut self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.used = self.used.saturating_sub(bytes);
        self.release_notify.notify_waiters();
    }

    pub(crate) fn release_notifier(&self) -> Arc<Notify> {
        Arc::clone(&self.release_notify)
    }

    pub(crate) fn available(&self) -> usize {
        self.capacity.saturating_sub(self.used)
    }

    #[allow(dead_code)]
    pub(crate) fn used(&self) -> usize {
        self.used
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserves_and_releases_bytes() {
        let mut budget = MemBudget::new(10);

        assert!(budget.try_reserve(4));
        assert_eq!(budget.used(), 4);
        assert_eq!(budget.available(), 6);
        assert!(!budget.try_reserve(7));

        budget.release(3);
        assert_eq!(budget.used(), 1);
        assert!(budget.try_reserve(9));
        assert_eq!(budget.available(), 0);
    }
}
