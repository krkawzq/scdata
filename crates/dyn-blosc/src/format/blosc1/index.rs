use std::ops::Range;

use crate::error::{vector_with_capacity, Error, Result};

use super::{Header, HEADER_LEN};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Index {
    encoded_offsets: Vec<u32>,
    encoded_ends: Vec<u32>,
    block_size: usize,
    decoded_size: usize,
}

impl Index {
    pub(crate) fn parse(header: Header, encoded: &[u8]) -> Result<Self> {
        let need = header.index_prefix_len()?;
        if encoded.len() < need {
            return Err(Error::InvalidFormat(format!(
                "need {need} bytes for Blosc1 header and index, have {}",
                encoded.len()
            )));
        }
        if header.is_raw() {
            return Err(Error::InvalidFormat(
                "raw Blosc1 chunks do not contain a block index".into(),
            ));
        }
        let count = header.block_count();
        let mut encoded_offsets = vector_with_capacity(count)?;
        for block in 0..count {
            let base = HEADER_LEN + block * 4;
            // SAFETY: the prefix-length check proves this four-byte entry exists.
            let offset = unsafe {
                i32::from_le(std::ptr::read_unaligned(
                    encoded.as_ptr().add(base).cast::<i32>(),
                ))
            };
            if offset < 0 {
                return Err(Error::InvalidFormat(format!(
                    "Blosc1 block {block} has negative encoded offset {offset}"
                )));
            }
            let offset = offset as usize;
            if offset < need || offset >= header.encoded_size() {
                return Err(Error::InvalidFormat(format!(
                    "Blosc1 block {block} encoded offset {offset} is outside the payload"
                )));
            }
            encoded_offsets.push(offset as u32);
        }
        let encoded_ends =
            physical_ends(&encoded_offsets, need, header.encoded_size()).map_err(|error| {
                Error::InvalidFormat(format!("invalid Blosc1 block offsets: {error}"))
            })?;
        Ok(Self {
            encoded_offsets,
            encoded_ends,
            block_size: header.block_size(),
            decoded_size: header.decoded_size(),
        })
    }

    pub(crate) fn from_offsets(
        encoded_offsets: Vec<u32>,
        block_size: usize,
        decoded_size: usize,
        encoded_size: usize,
    ) -> Result<Self> {
        let expected = if decoded_size == 0 {
            0
        } else {
            decoded_size.div_ceil(block_size)
        };
        if encoded_offsets.len() != expected {
            return Err(Error::InvalidOptions(format!(
                "Blosc1 index has {} offsets, expected {expected}",
                encoded_offsets.len()
            )));
        }
        let prefix_len = HEADER_LEN
            .checked_add(
                encoded_offsets
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| Error::InvalidOptions("Blosc1 index size overflow".into()))?,
            )
            .ok_or_else(|| Error::InvalidOptions("Blosc1 index prefix overflow".into()))?;
        let encoded_ends = physical_ends(&encoded_offsets, prefix_len, encoded_size)
            .map_err(|error| Error::InvalidOptions(error.to_string()))?;
        Ok(Self {
            encoded_offsets,
            encoded_ends,
            block_size,
            decoded_size,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.encoded_offsets.len()
    }

    pub(crate) fn maximum_decoded_length(&self) -> usize {
        if self.encoded_offsets.is_empty() {
            0
        } else {
            self.block_size.min(self.decoded_size)
        }
    }

    pub(crate) unsafe fn encoded_offset_unchecked(&self, block: usize) -> u32 {
        // SAFETY: guaranteed by the caller.
        unsafe { *self.encoded_offsets.get_unchecked(block) }
    }

    pub(crate) unsafe fn decoded_start_unchecked(&self, block: usize) -> usize {
        debug_assert!(block < self.len());
        block * self.block_size
    }

    pub(crate) unsafe fn decoded_length_unchecked(&self, block: usize) -> usize {
        // SAFETY: guaranteed by the caller.
        let start = unsafe { self.decoded_start_unchecked(block) };
        self.block_size.min(self.decoded_size - start)
    }

    pub(crate) unsafe fn encoded_range_unchecked(
        &self,
        encoded_size: usize,
        block: usize,
    ) -> Range<usize> {
        // SAFETY: guaranteed by the caller.
        let start = unsafe { self.encoded_offset_unchecked(block) as usize };
        // SAFETY: the offset and end tables always have equal lengths.
        let end = unsafe { *self.encoded_ends.get_unchecked(block) as usize };
        debug_assert!(end <= encoded_size);
        start..end
    }

    pub(crate) fn blocks_intersecting(&self, range: &Range<usize>) -> Range<usize> {
        if range.is_empty() || self.encoded_offsets.is_empty() {
            return 0..0;
        }
        let first = range.start / self.block_size;
        let end = range.end.div_ceil(self.block_size).min(self.len());
        first.min(end)..end
    }

    pub(crate) fn write(&self, out: &mut [u8]) -> Result<()> {
        let need = self
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::InvalidOptions("Blosc1 index size overflow".into()))?;
        if out.len() < need {
            return Err(Error::BufferTooSmall {
                need,
                have: out.len(),
            });
        }
        for (block, &offset) in self.encoded_offsets.iter().enumerate() {
            // SAFETY: the length check reserves four bytes for every entry;
            // unaligned writes are valid for byte-backed storage.
            unsafe {
                std::ptr::write_unaligned(
                    out.as_mut_ptr().add(block * 4).cast::<u32>(),
                    offset.to_le(),
                );
            }
        }
        Ok(())
    }

    pub(crate) fn ensure_matches_prefix(&self, encoded: &[u8]) -> Result<()> {
        let need = HEADER_LEN
            .checked_add(self.len().checked_mul(4).ok_or_else(|| {
                Error::SchemaMismatch("Blosc1 index prefix length overflows usize".into())
            })?)
            .ok_or_else(|| {
                Error::SchemaMismatch("Blosc1 index prefix length overflows usize".into())
            })?;
        if encoded.len() < need {
            return Err(Error::SchemaMismatch(format!(
                "need {need} bytes for Blosc1 header and index, have {}",
                encoded.len()
            )));
        }
        for (block, &expected) in self.encoded_offsets.iter().enumerate() {
            let base = HEADER_LEN + block * 4;
            // SAFETY: the prefix-length check proves this entry exists.
            let actual = unsafe {
                u32::from_le(std::ptr::read_unaligned(
                    encoded.as_ptr().add(base).cast::<u32>(),
                ))
            };
            if actual != expected {
                return Err(Error::SchemaMismatch(format!(
                    "Blosc1 block index entry {block} differs"
                )));
            }
        }
        Ok(())
    }
}

fn physical_ends(offsets: &[u32], prefix_len: usize, encoded_size: usize) -> Result<Vec<u32>> {
    let mut physical_order = vector_with_capacity(offsets.len())?;
    physical_order.extend(offsets.iter().copied().enumerate());
    physical_order.sort_unstable_by_key(|&(_, offset)| offset);
    if physical_order
        .first()
        .is_some_and(|&(_, offset)| offset as usize != prefix_len)
    {
        return Err(Error::InvalidFormat(format!(
            "first physical block starts at {}, expected {prefix_len}",
            physical_order[0].1
        )));
    }
    for pair in physical_order.windows(2) {
        if pair[0].1 == pair[1].1 {
            return Err(Error::InvalidFormat(format!(
                "blocks {} and {} share encoded offset {}",
                pair[0].0, pair[1].0, pair[0].1
            )));
        }
    }
    let mut ends = vector_with_capacity(offsets.len())?;
    ends.resize(offsets.len(), 0);
    for (position, &(logical_block, _)) in physical_order.iter().enumerate() {
        let end = physical_order
            .get(position + 1)
            .map_or(encoded_size, |&(_, offset)| offset as usize);
        if end > u32::MAX as usize {
            return Err(Error::InvalidFormat(
                "Blosc1 encoded block end exceeds u32".into(),
            ));
        }
        ends[logical_block] = end as u32;
    }
    Ok(ends)
}
