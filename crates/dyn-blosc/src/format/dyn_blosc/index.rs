use std::ops::Range;

use crate::error::{vector_with_capacity, Error, Result};
use crate::format::index::BlockEntry;

use super::{Header, HEADER_LEN};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Index {
    entries: Vec<BlockEntry>,
    decoded_starts: Vec<usize>,
    maximum_decoded_length: usize,
    uniform_decoded_layout: bool,
}

impl Index {
    pub(crate) fn new(entries: Vec<BlockEntry>) -> Result<Self> {
        let mut decoded_starts = vector_with_capacity(entries.len())?;
        let mut offset = 0usize;
        let mut maximum_decoded_length = 0usize;
        for entry in &entries {
            decoded_starts.push(offset);
            maximum_decoded_length = maximum_decoded_length.max(entry.decoded_length as usize);
            offset = offset
                .checked_add(entry.decoded_length as usize)
                .ok_or_else(|| {
                    Error::InvalidFormat("decoded block offsets overflow usize".into())
                })?;
        }
        let uniform_decoded_layout = uniform_decoded_layout(&entries);
        Ok(Self {
            entries,
            decoded_starts,
            maximum_decoded_length,
            uniform_decoded_layout,
        })
    }

    pub(crate) fn parse(header: Header, encoded: &[u8]) -> Result<Self> {
        let need = header.index_prefix_len()?;
        if encoded.len() < need {
            return Err(Error::InvalidFormat(format!(
                "need {need} bytes for DynBlosc header and index, have {}",
                encoded.len()
            )));
        }
        if header.is_raw() {
            return Err(Error::InvalidFormat(
                "raw DynBlosc chunks do not contain a block index".into(),
            ));
        }

        let entry_count = header.block_count();
        let mut entries = vector_with_capacity(entry_count)?;
        let mut decoded_starts = vector_with_capacity(entry_count)?;
        let mut decoded_sum = 0u64;
        let mut maximum_decoded_length = 0usize;
        for block in 0..entry_count {
            let base = HEADER_LEN + block * 8;
            // SAFETY: the prefix-length check proves this eight-byte entry exists.
            let wire_entry = unsafe {
                u64::from_le(std::ptr::read_unaligned(
                    encoded.as_ptr().add(base).cast::<u64>(),
                ))
            };
            let encoded_offset = wire_entry as u32 as i32;
            let decoded_length = (wire_entry >> 32) as u32 as i32;
            if encoded_offset < 0 || decoded_length <= 0 {
                return Err(Error::InvalidFormat(format!(
                    "DynBlosc block {block} has invalid encoded offset {encoded_offset} or decoded length {decoded_length}"
                )));
            }
            let encoded_offset = encoded_offset as u32;
            let decoded_length = decoded_length as u32;
            if decoded_length as usize > header.decoded_size() {
                return Err(Error::InvalidFormat(format!(
                    "DynBlosc block {block} decoded length {decoded_length} exceeds decoded size {}",
                    header.decoded_size()
                )));
            }
            if (encoded_offset as usize) < need
                || (encoded_offset as usize) >= header.encoded_size()
            {
                return Err(Error::InvalidFormat(format!(
                    "DynBlosc block {block} encoded offset {encoded_offset} is outside the payload"
                )));
            }
            if block == 0 && encoded_offset as usize != need {
                return Err(Error::InvalidFormat(format!(
                    "first DynBlosc block starts at {encoded_offset}, expected {need}"
                )));
            }
            if entries
                .last()
                .is_some_and(|previous: &BlockEntry| encoded_offset <= previous.encoded_offset)
            {
                return Err(Error::InvalidFormat(format!(
                    "DynBlosc block offsets are not strictly increasing at {block}"
                )));
            }
            if header.element_size() > 1
                && !decoded_sum.is_multiple_of(header.element_size() as u64)
            {
                return Err(Error::InvalidFormat(format!(
                    "DynBlosc block {block} decoded start {decoded_sum} is not a multiple of element size {}",
                    header.element_size()
                )));
            }
            decoded_starts.push(decoded_sum as usize);
            decoded_sum += decoded_length as u64;
            maximum_decoded_length = maximum_decoded_length.max(decoded_length as usize);
            entries.push(BlockEntry {
                encoded_offset,
                decoded_length,
            });
        }
        if decoded_sum != header.decoded_size() as u64 {
            return Err(Error::InvalidFormat(format!(
                "decoded block lengths sum to {decoded_sum}, expected {}",
                header.decoded_size()
            )));
        }
        let uniform_decoded_layout = uniform_decoded_layout(&entries);
        Ok(Self {
            entries,
            decoded_starts,
            maximum_decoded_length,
            uniform_decoded_layout,
        })
    }

    pub(crate) fn maximum_decoded_length(&self) -> usize {
        self.maximum_decoded_length
    }

    pub(crate) fn has_uniform_decoded_layout(&self) -> bool {
        self.uniform_decoded_layout
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) unsafe fn entry_unchecked(&self, block: usize) -> BlockEntry {
        // SAFETY: guaranteed by the caller.
        unsafe { *self.entries.get_unchecked(block) }
    }

    pub(crate) unsafe fn decoded_start_unchecked(&self, block: usize) -> usize {
        // SAFETY: entry and decoded-start tables always have equal lengths.
        unsafe { *self.decoded_starts.get_unchecked(block) }
    }

    pub(crate) unsafe fn encoded_range_unchecked(
        &self,
        encoded_size: usize,
        block: usize,
    ) -> Range<usize> {
        // SAFETY: guaranteed by the caller.
        let start = unsafe { self.entries.get_unchecked(block).encoded_offset as usize };
        let end = if block + 1 < self.entries.len() {
            // SAFETY: the branch proves the next entry exists.
            unsafe { self.entries.get_unchecked(block + 1).encoded_offset as usize }
        } else {
            encoded_size
        };
        start..end
    }

    pub(crate) fn blocks_intersecting(&self, range: &Range<usize>) -> Range<usize> {
        if range.is_empty() || self.entries.is_empty() {
            return 0..0;
        }
        let first = self
            .decoded_starts
            .partition_point(|&start| start <= range.start)
            .saturating_sub(1);
        let end = self
            .decoded_starts
            .partition_point(|&start| start < range.end);
        first..end
    }

    pub(crate) fn write(&self, out: &mut [u8]) -> Result<()> {
        let need = self.entries.len() * 8;
        if out.len() < need {
            return Err(Error::BufferTooSmall {
                need,
                have: out.len(),
            });
        }
        for (block, entry) in self.entries.iter().enumerate() {
            let wire = u64::from(entry.encoded_offset) | (u64::from(entry.decoded_length) << 32);
            // SAFETY: the length check reserves eight bytes for every entry.
            unsafe {
                std::ptr::write_unaligned(
                    out.as_mut_ptr().add(block * 8).cast::<u64>(),
                    wire.to_le(),
                );
            }
        }
        Ok(())
    }

    pub(crate) fn ensure_matches_prefix(&self, encoded: &[u8]) -> Result<()> {
        let need = HEADER_LEN
            .checked_add(self.entries.len().checked_mul(8).ok_or_else(|| {
                Error::SchemaMismatch("DynBlosc index prefix length overflows usize".into())
            })?)
            .ok_or_else(|| {
                Error::SchemaMismatch("DynBlosc index prefix length overflows usize".into())
            })?;
        if encoded.len() < need {
            return Err(Error::SchemaMismatch(format!(
                "need {need} bytes for DynBlosc header and index, have {}",
                encoded.len()
            )));
        }
        for (block, expected) in self.entries.iter().enumerate() {
            let base = HEADER_LEN + block * 8;
            // SAFETY: the prefix-length check proves this entry exists.
            let actual = unsafe {
                u64::from_le(std::ptr::read_unaligned(
                    encoded.as_ptr().add(base).cast::<u64>(),
                ))
            };
            let expected =
                u64::from(expected.encoded_offset) | (u64::from(expected.decoded_length) << 32);
            if actual != expected {
                return Err(Error::SchemaMismatch(format!(
                    "DynBlosc block index entry {block} differs"
                )));
            }
        }
        Ok(())
    }
}

fn uniform_decoded_layout(entries: &[BlockEntry]) -> bool {
    let Some((first, rest)) = entries.split_first() else {
        return true;
    };
    let Some((last, middle)) = rest.split_last() else {
        return true;
    };
    middle
        .iter()
        .all(|entry| entry.decoded_length == first.decoded_length)
        && last.decoded_length <= first.decoded_length
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(lengths: &[u32]) -> Vec<BlockEntry> {
        lengths
            .iter()
            .enumerate()
            .map(|(index, &decoded_length)| BlockEntry {
                encoded_offset: (HEADER_LEN + lengths.len() * 8 + index) as u32,
                decoded_length,
            })
            .collect()
    }

    #[test]
    fn uniform_layout_allows_a_short_final_block_only() {
        assert!(Index::new(entries(&[64, 64, 64, 16]))
            .unwrap()
            .has_uniform_decoded_layout());
        assert!(!Index::new(entries(&[64, 16, 64, 16]))
            .unwrap()
            .has_uniform_decoded_layout());
        assert!(!Index::new(entries(&[64, 64, 96]))
            .unwrap()
            .has_uniform_decoded_layout());
    }
}
