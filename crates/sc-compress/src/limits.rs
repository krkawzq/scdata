use crate::error::{Error, Result};
use crate::parallel::{default_threads, validate_threads};

/// Resource limits applied while opening and decoding a store.
///
/// Compound readers apply the decoded limit to all explicit simultaneously
/// resident buffers they control, including outputs, decoder indexes, scratch
/// space, and encoded ranges, rather than to each allocation independently.
///
/// `threads` controls sc-compress chunk-level decode parallelism (dyn-blosc
/// block parallelism stays single-threaded). Concurrent workers share the
/// decoded-size budget and are admitted serially when there is not enough
/// temporary-memory headroom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadLimits {
    maximum_metadata_size: usize,
    maximum_encoded_size: usize,
    maximum_decoded_size: usize,
    maximum_block_count: usize,
    threads: usize,
}

impl Default for ReadLimits {
    fn default() -> Self {
        Self {
            maximum_metadata_size: 1024 * 1024,
            maximum_encoded_size: 1024 * 1024 * 1024,
            maximum_decoded_size: 1024 * 1024 * 1024,
            maximum_block_count: 1_000_000,
            threads: default_threads(),
        }
    }
}

impl ReadLimits {
    pub fn unlimited() -> Self {
        Self {
            maximum_metadata_size: usize::MAX,
            maximum_encoded_size: usize::MAX,
            maximum_decoded_size: usize::MAX,
            maximum_block_count: usize::MAX,
            threads: default_threads(),
        }
    }

    #[must_use]
    pub const fn maximum_metadata_size(mut self, bytes: usize) -> Self {
        self.maximum_metadata_size = bytes;
        self
    }

    #[must_use]
    pub const fn maximum_encoded_size(mut self, bytes: usize) -> Self {
        self.maximum_encoded_size = bytes;
        self
    }

    #[must_use]
    pub const fn maximum_decoded_size(mut self, bytes: usize) -> Self {
        self.maximum_decoded_size = bytes;
        self
    }

    #[must_use]
    pub const fn maximum_block_count(mut self, count: usize) -> Self {
        self.maximum_block_count = count;
        self
    }

    /// Number of workers used for chunk-level decode parallelism.
    #[must_use]
    pub fn threads(mut self, count: usize) -> Self {
        self.threads = count;
        self
    }

    pub const fn metadata_size(self) -> usize {
        self.maximum_metadata_size
    }

    pub const fn encoded_size(self) -> usize {
        self.maximum_encoded_size
    }

    pub const fn decoded_size(self) -> usize {
        self.maximum_decoded_size
    }

    pub const fn block_count(self) -> usize {
        self.maximum_block_count
    }

    pub const fn thread_count(self) -> usize {
        self.threads
    }

    pub(crate) fn validate(self) -> Result<Self> {
        validate_threads(self.threads)?;
        Ok(self)
    }

    pub(crate) fn check_encoded(self, bytes: usize, context: &str) -> Result<()> {
        check_limit(bytes, self.maximum_encoded_size, context, "encoded")
    }

    pub(crate) fn check_decoded(self, bytes: usize, context: &str) -> Result<()> {
        check_limit(bytes, self.maximum_decoded_size, context, "decoded")
    }

    pub(crate) fn check_decoded_sum(
        self,
        sizes: impl IntoIterator<Item = usize>,
        context: &str,
    ) -> Result<usize> {
        let mut total = 0usize;
        for size in sizes {
            total = total
                .checked_add(size)
                .ok_or_else(|| Error::corrupt(context, "decoded size sum overflow"))?;
        }
        self.check_decoded(total, context)?;
        Ok(total)
    }
}

fn check_limit(actual: usize, maximum: usize, context: &str, kind: &str) -> Result<()> {
    if actual > maximum {
        return Err(Error::corrupt(
            context,
            format!("{kind} size {actual} exceeds configured limit {maximum}"),
        ));
    }
    Ok(())
}
