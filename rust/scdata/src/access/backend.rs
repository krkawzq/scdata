//! Backend callback traits used by the access scheduler.
//!
//! The scheduler depends only on these traits. Concrete IO and decode pools
//! can stay in their own modules and expose lightweight adapters here.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;

use crate::codecs::{CodecResult, SharedCodec};

/// Opaque file handle understood by an [`IoBackend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileRef(pub u64);

impl FileRef {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Future returned by an IO backend after a positioned read is submitted.
pub type IoTask = Pin<Box<dyn Future<Output = io::Result<Arc<[u8]>>> + Send>>;

/// Future returned by a decode backend after a chunk decode is submitted.
pub type DecodeTask = Pin<Box<dyn Future<Output = CodecResult<Vec<u8>>> + Send>>;

/// Positioned-read backend used by the access scheduler.
pub trait IoBackend: Send + Sync + 'static {
    /// Submit a positioned read and return a future for the compressed bytes.
    fn submit_read(&self, file: FileRef, offset: u64, len: usize, priority: u8) -> IoTask;
}

/// Decode backend used by the access scheduler.
pub trait DecodeBackend: Send + Sync + 'static {
    /// Submit one decode for one caller-owned output buffer.
    fn submit_decode(
        &self,
        codec: SharedCodec,
        encoded: Arc<[u8]>,
        expected_size: Option<usize>,
    ) -> DecodeTask;
}
