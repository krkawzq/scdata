use std::ops::Range;

use crate::error::{Error, Result};
use crate::format::{blosc1, dyn_blosc, Header};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockEntry {
    pub encoded_offset: u32,
    pub decoded_length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Index {
    Blosc1(blosc1::Index),
    DynBlosc(dyn_blosc::Index),
}

impl Index {
    pub(crate) fn parse(header: Header, encoded: &[u8]) -> Result<Self> {
        match header {
            Header::Blosc1(header) => blosc1::Index::parse(header, encoded).map(Self::Blosc1),
            Header::DynBlosc(header) => {
                dyn_blosc::Index::parse(header, encoded).map(Self::DynBlosc)
            }
        }
    }

    pub(crate) fn new_dyn(entries: Vec<BlockEntry>) -> Result<Self> {
        dyn_blosc::Index::new(entries).map(Self::DynBlosc)
    }

    pub(crate) fn new_blosc1(
        encoded_offsets: Vec<u32>,
        block_size: usize,
        decoded_size: usize,
        encoded_size: usize,
    ) -> Result<Self> {
        blosc1::Index::from_offsets(encoded_offsets, block_size, decoded_size, encoded_size)
            .map(Self::Blosc1)
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Blosc1(index) => index.len(),
            Self::DynBlosc(index) => index.len(),
        }
    }

    pub(crate) fn maximum_decoded_length(&self) -> usize {
        match self {
            Self::Blosc1(index) => index.maximum_decoded_length(),
            Self::DynBlosc(index) => index.maximum_decoded_length(),
        }
    }

    pub(crate) fn has_uniform_decoded_layout(&self) -> bool {
        match self {
            Self::Blosc1(_) => true,
            Self::DynBlosc(index) => index.has_uniform_decoded_layout(),
        }
    }

    /// Return a semantic block entry without a bounds check.
    ///
    /// # Safety
    ///
    /// `block` must be less than [`Self::len`].
    pub(crate) unsafe fn entry_unchecked(&self, block: usize) -> BlockEntry {
        match self {
            Self::Blosc1(index) => BlockEntry {
                // SAFETY: guaranteed by the caller.
                encoded_offset: unsafe { index.encoded_offset_unchecked(block) },
                // SAFETY: guaranteed by the caller.
                decoded_length: unsafe { index.decoded_length_unchecked(block) } as u32,
            },
            Self::DynBlosc(index) => {
                // SAFETY: guaranteed by the caller.
                unsafe { index.entry_unchecked(block) }
            }
        }
    }

    /// Return a decoded block start without a bounds check.
    ///
    /// # Safety
    ///
    /// `block` must be less than [`Self::len`].
    pub(crate) unsafe fn decoded_start_unchecked(&self, block: usize) -> usize {
        match self {
            // SAFETY: guaranteed by the caller.
            Self::Blosc1(index) => unsafe { index.decoded_start_unchecked(block) },
            // SAFETY: guaranteed by the caller.
            Self::DynBlosc(index) => unsafe { index.decoded_start_unchecked(block) },
        }
    }

    /// Return an encoded block range without a bounds check.
    ///
    /// # Safety
    ///
    /// `block` must be less than [`Self::len`], and `header` must be the header
    /// from which this index was parsed or constructed.
    pub(crate) unsafe fn encoded_range_unchecked(
        &self,
        header: Header,
        block: usize,
    ) -> Range<usize> {
        match (self, header) {
            (Self::Blosc1(index), Header::Blosc1(header)) => {
                // SAFETY: guaranteed by the caller.
                unsafe { index.encoded_range_unchecked(header.encoded_size(), block) }
            }
            (Self::DynBlosc(index), Header::DynBlosc(header)) => {
                // SAFETY: guaranteed by the caller.
                unsafe { index.encoded_range_unchecked(header.encoded_size(), block) }
            }
            _ => unreachable!("header and index versions always match"),
        }
    }

    pub(crate) fn encoded_range(&self, header: Header, block: usize) -> Result<Range<usize>> {
        if block >= self.len() {
            return Err(Error::InvalidArgument(format!(
                "block {block} is out of range"
            )));
        }
        // SAFETY: checked immediately above; layout construction keeps the
        // header and index versions paired.
        Ok(unsafe { self.encoded_range_unchecked(header, block) })
    }

    pub(crate) fn decoded_start(&self, block: usize) -> Result<usize> {
        if block >= self.len() {
            return Err(Error::InvalidArgument(format!(
                "block {block} is out of range"
            )));
        }
        // SAFETY: checked immediately above.
        Ok(unsafe { self.decoded_start_unchecked(block) })
    }

    pub(crate) fn entry(&self, block: usize) -> Option<BlockEntry> {
        if block >= self.len() {
            return None;
        }
        // SAFETY: checked immediately above.
        Some(unsafe { self.entry_unchecked(block) })
    }

    pub(crate) fn blocks_intersecting(&self, range: &Range<usize>) -> Range<usize> {
        match self {
            Self::Blosc1(index) => index.blocks_intersecting(range),
            Self::DynBlosc(index) => index.blocks_intersecting(range),
        }
    }

    pub(crate) fn write(&self, out: &mut [u8]) -> Result<()> {
        match self {
            Self::Blosc1(index) => index.write(out),
            Self::DynBlosc(index) => index.write(out),
        }
    }

    pub(crate) fn ensure_matches_prefix(&self, encoded: &[u8]) -> Result<()> {
        match self {
            Self::Blosc1(index) => index.ensure_matches_prefix(encoded),
            Self::DynBlosc(index) => index.ensure_matches_prefix(encoded),
        }
    }
}
