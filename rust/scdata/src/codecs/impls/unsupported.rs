use super::super::spec::{sealed, ChunkCodec};
use super::super::{CodecError, CodecResult};

#[derive(Debug)]
pub struct UnsupportedCodec {
    name: String,
}

impl sealed::Sealed for UnsupportedCodec {}

impl UnsupportedCodec {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl ChunkCodec for UnsupportedCodec {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn decode(&self, _encoded: &[u8], _expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        Err(CodecError::Unsupported {
            codec: self.name.clone(),
        })
    }
}
