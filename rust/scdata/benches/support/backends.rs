//! In-memory mock IO / decode backends for the access scheduler benches.
//!
//! `SliceIo` serves positioned reads from a shared byte slice (no real IO), and
//! `CodecDecode` dispatches decodes through the codec trait directly. Together
//! they let the `access` benches isolate scheduler overhead from disk and pool
//! plumbing.

use std::io;
use std::sync::Arc;

use _scdata::access::{DecodeBackend, DecodeTask, FileRef, IoBackend, IoTask};
use _scdata::codecs::SharedCodec;

#[derive(Debug)]
pub struct SliceIo {
    data: Arc<[u8]>,
}

impl SliceIo {
    pub fn new(data: Arc<[u8]>) -> Self {
        Self { data }
    }
}

impl IoBackend for SliceIo {
    fn submit_read(&self, _file: FileRef, offset: u64, len: usize, _priority: u8) -> IoTask {
        let data = Arc::clone(&self.data);
        Box::pin(async move {
            let start = usize::try_from(offset).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "offset does not fit usize")
            })?;
            let end = start.checked_add(len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "read range overflow")
            })?;
            let bytes = data.get(start..end).ok_or_else(|| {
                io::Error::new(io::ErrorKind::UnexpectedEof, "read range out of bounds")
            })?;
            Ok(Arc::from(bytes))
        })
    }
}

#[derive(Debug)]
pub struct CodecDecode;

impl DecodeBackend for CodecDecode {
    fn submit_decode(
        &self,
        codec: SharedCodec,
        encoded: Arc<[u8]>,
        expected_size: Option<usize>,
    ) -> DecodeTask {
        Box::pin(async move { codec.decode(&encoded, expected_size) })
    }
}
