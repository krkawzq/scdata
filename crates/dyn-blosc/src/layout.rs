use std::ops::Range;

use crate::error::{reserve_exact, Error, Result};
use crate::format::{BloscVersion, Codec, Header, Index, Shuffle, HEADER_LEN};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLayout {
    Fixed { block_size: usize },
    Variable { maximum_block_size: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metadata {
    pub version: BloscVersion,
    pub decoded_size: usize,
    pub encoded_size: usize,
    pub maximum_block_size: usize,
    pub block_count: usize,
    pub element_size: usize,
    pub codec: Codec,
    pub shuffle: Shuffle,
    pub split_blocks: bool,
    pub is_raw: bool,
    pub block_layout: BlockLayout,
}

/// One validated block's location in a versioned Blosc chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockRange {
    index: usize,
    /// Byte offset of the encoded block relative to the chunk start.
    encoded_offset: usize,
    encoded_len: usize,
    /// Byte offset of this block in the decoded payload.
    decoded_offset: usize,
    decoded_len: usize,
}

impl BlockRange {
    pub fn index(self) -> usize {
        self.index
    }

    pub fn encoded_range(self) -> Range<usize> {
        self.encoded_offset..self.encoded_offset + self.encoded_len
    }

    pub fn decoded_len(self) -> usize {
        self.decoded_len
    }

    pub fn decoded_range(self) -> Range<usize> {
        self.decoded_offset..self.decoded_offset + self.decoded_len
    }
}

/// Sizes of one independently encoded block payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDescriptor {
    decoded_len: usize,
    encoded_len: usize,
}

impl BlockDescriptor {
    pub fn new(decoded_len: usize, encoded_len: usize) -> Result<Self> {
        if decoded_len == 0 {
            return Err(Error::InvalidArgument(
                "decoded block length must be non-zero".into(),
            ));
        }
        if encoded_len == 0 {
            return Err(Error::InvalidArgument(
                "encoded block length must be non-zero".into(),
            ));
        }
        if decoded_len > i32::MAX as usize {
            return Err(Error::InvalidArgument(format!(
                "decoded block length {decoded_len} exceeds the wire-format limit {}",
                i32::MAX
            )));
        }
        if encoded_len > i32::MAX as usize {
            return Err(Error::InvalidArgument(format!(
                "encoded block length {encoded_len} exceeds the wire-format limit {}",
                i32::MAX
            )));
        }
        Ok(Self {
            decoded_len,
            encoded_len,
        })
    }

    pub fn decoded_len(self) -> usize {
        self.decoded_len
    }

    pub fn encoded_len(self) -> usize {
        self.encoded_len
    }
}

/// Validated versioned Blosc header and block-index layout without payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkLayout {
    pub(crate) header: Header,
    pub(crate) index: Option<Index>,
}

impl ChunkLayout {
    pub(crate) fn from_parts(header: Header, index: Option<Index>) -> Self {
        Self { header, index }
    }

    pub fn header(&self) -> Header {
        self.header
    }

    /// Largest decoded block length in this layout.
    ///
    /// Raw chunks contain one payload block spanning the decoded size.
    /// Compressed Blosc1 chunks use the fixed header block size; compressed
    /// DynBlosc chunks derive the maximum from the block index.
    pub fn maximum_block_size(&self) -> usize {
        if self.header.decoded_size() == 0 {
            return 0;
        }
        if self.header.is_raw() {
            return self.header.decoded_size();
        }
        match self.header {
            Header::Blosc1(header) => header.block_size(),
            Header::DynBlosc(_) => self
                .index
                .as_ref()
                .map(Index::maximum_decoded_length)
                .unwrap_or(0),
        }
    }

    pub fn metadata(&self) -> Metadata {
        let maximum_block_size = self.maximum_block_size();
        Metadata {
            version: self.header.version(),
            decoded_size: self.header.decoded_size(),
            encoded_size: self.header.encoded_size(),
            maximum_block_size,
            block_count: self.header.block_count(),
            element_size: self.header.element_size(),
            codec: self
                .header
                .codec()
                .expect("ChunkLayout construction validates the codec"),
            shuffle: self.header.shuffle(),
            split_blocks: self.header.split_blocks(),
            is_raw: self.header.is_raw(),
            block_layout: match self.header {
                Header::Blosc1(_) => BlockLayout::Fixed {
                    block_size: maximum_block_size,
                },
                Header::DynBlosc(_) => BlockLayout::Variable { maximum_block_size },
            },
        }
    }

    pub fn prefix_len(&self) -> usize {
        if self.header.decoded_size() == 0 || self.header.is_raw() {
            HEADER_LEN
        } else {
            self.header
                .index_prefix_len()
                .expect("ChunkLayout construction validates the index size")
        }
    }

    pub fn block(&self, block_index: usize) -> Option<BlockRange> {
        if self.header.decoded_size() == 0 {
            return None;
        }
        if self.header.is_raw() {
            return (block_index == 0).then(|| {
                let decoded_len = self.header.decoded_size();
                BlockRange {
                    index: 0,
                    encoded_offset: HEADER_LEN,
                    encoded_len: decoded_len,
                    decoded_offset: 0,
                    decoded_len,
                }
            });
        }

        let index = self
            .index
            .as_ref()
            .expect("compressed non-empty layouts have an index");
        let entry = index.entry(block_index)?;
        let encoded_range = index
            .encoded_range(self.header, block_index)
            .expect("ChunkLayout construction validates encoded ranges");
        Some(BlockRange {
            index: block_index,
            encoded_offset: encoded_range.start,
            encoded_len: encoded_range.end - encoded_range.start,
            decoded_offset: index
                .decoded_start(block_index)
                .expect("entry and decoded-start tables have equal lengths"),
            decoded_len: entry.decoded_length as usize,
        })
    }

    pub fn blocks(&self) -> Blocks<'_> {
        Blocks {
            layout: self,
            next: 0,
        }
    }

    pub fn write_prefix(&self, output: &mut [u8]) -> Result<usize> {
        let prefix_len = self.prefix_len();
        if output.len() < prefix_len {
            return Err(Error::BufferTooSmall {
                need: prefix_len,
                have: output.len(),
            });
        }
        self.header.write(output)?;
        if let Some(index) = &self.index {
            index.write(&mut output[HEADER_LEN..prefix_len])?;
        }
        Ok(prefix_len)
    }

    pub fn prefix(&self) -> Result<Vec<u8>> {
        let prefix_len = self.prefix_len();
        let mut output = Vec::new();
        reserve_exact(&mut output, prefix_len)?;
        output.resize(prefix_len, 0);
        self.write_prefix(&mut output)?;
        Ok(output)
    }

    /// Assemble a complete chunk from externally owned block payloads.
    pub fn assemble<'a, I>(&self, payloads: I) -> Result<Vec<u8>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut output = self.prefix()?;
        reserve_exact(&mut output, self.header.encoded_size() - self.prefix_len())?;
        let mut payloads = payloads.into_iter();
        for block in self.blocks() {
            let payload = payloads.next().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "missing encoded payload for block {}",
                    block.index()
                ))
            })?;
            if payload.len() != block.encoded_len {
                return Err(Error::InvalidArgument(format!(
                    "encoded payload for block {} has {} bytes, expected {}",
                    block.index(),
                    payload.len(),
                    block.encoded_len
                )));
            }
            output.extend_from_slice(payload);
        }
        if payloads.next().is_some() {
            return Err(Error::InvalidArgument(
                "more encoded payloads were supplied than the layout contains".into(),
            ));
        }
        if output.len() != self.header.encoded_size() {
            return Err(Error::InvalidFormat(
                "assembled size does not match layout".into(),
            ));
        }
        Ok(output)
    }

    pub(crate) fn index(&self) -> &Index {
        self.index
            .as_ref()
            .expect("compressed non-empty layouts have a validated index")
    }

    pub(crate) fn matches_chunk(&self, encoded: &[u8]) -> Result<()> {
        if encoded.len() != self.header.encoded_size() {
            return Err(Error::SchemaMismatch(format!(
                "encoded length {} differs from schema length {}",
                encoded.len(),
                self.header.encoded_size()
            )));
        }
        let supplied_header =
            Header::parse(encoded).map_err(|error| Error::SchemaMismatch(error.to_string()))?;
        if supplied_header != self.header {
            return Err(Error::SchemaMismatch("header differs".into()));
        }
        if let Some(index) = &self.index {
            index.ensure_matches_prefix(encoded)?;
        }
        Ok(())
    }
}

pub struct Blocks<'a> {
    layout: &'a ChunkLayout,
    next: usize,
}

impl Iterator for Blocks<'_> {
    type Item = BlockRange;

    fn next(&mut self) -> Option<Self::Item> {
        let block = self.layout.block(self.next)?;
        self.next += 1;
        Some(block)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.layout.header.block_count() - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Blocks<'_> {}
