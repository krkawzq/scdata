use std::io;
use std::sync::Arc;

use crate::access::{DecodeBackend, DecodeTask, FileRef, IoBackend, IoTask};
use crate::codecs::{DecodePool, DecodeRequest, SharedCodec};
use crate::iopool::{IoCommand, IoPool};

#[derive(Clone)]
pub struct IoPoolBackend {
    pool: Arc<IoPool>,
}

impl IoPoolBackend {
    pub fn new(pool: Arc<IoPool>) -> Self {
        Self { pool }
    }
}

impl IoBackend for IoPoolBackend {
    fn submit_read(&self, file: FileRef, offset: u64, len: usize, priority: u8) -> IoTask {
        let pool = Arc::clone(&self.pool);
        Box::pin(async move {
            let file_id = usize::try_from(file.0).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("file ref {} does not fit FileId", file.0),
                )
            })?;
            let output = pool
                .submit(IoCommand::read(file_id, offset, len, priority as usize))?
                .await?;
            output.into_read_bytes()
        })
    }
}

#[derive(Clone)]
pub struct DecodePoolBackend {
    pool: Arc<DecodePool>,
}

impl DecodePoolBackend {
    pub fn new(pool: Arc<DecodePool>) -> Self {
        Self { pool }
    }
}

impl DecodeBackend for DecodePoolBackend {
    fn submit_decode(
        &self,
        codec: SharedCodec,
        encoded: Arc<[u8]>,
        expected_size: Option<usize>,
    ) -> DecodeTask {
        let pool = Arc::clone(&self.pool);
        Box::pin(async move {
            let mut request = DecodeRequest::new(codec, encoded);
            if let Some(expected_size) = expected_size {
                request = request.with_expected_size(expected_size);
            }
            pool.submit_async(request).await?.await
        })
    }
}
