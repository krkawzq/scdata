use std::ops::Range;

use crate::error::{vector_with_capacity, Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteMapping {
    source: Range<usize>,
    destination_start: usize,
}

impl ByteMapping {
    pub fn new(source: Range<usize>, destination_start: usize) -> Result<Self> {
        let length = source
            .end
            .checked_sub(source.start)
            .ok_or_else(|| Error::InvalidArgument("source range is reversed".into()))?;
        destination_start
            .checked_add(length)
            .ok_or_else(|| Error::InvalidArgument("destination range overflow".into()))?;
        Ok(Self {
            source,
            destination_start,
        })
    }

    pub fn source(&self) -> &Range<usize> {
        &self.source
    }

    pub fn destination_start(&self) -> usize {
        self.destination_start
    }

    pub fn len(&self) -> usize {
        self.source.end - self.source.start
    }

    pub fn is_empty(&self) -> bool {
        self.source.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteSelection {
    mappings: Vec<ByteMapping>,
    output_len: usize,
    fully_covers_output: bool,
}

impl ByteSelection {
    pub fn new(mappings: Vec<ByteMapping>, output_len: usize) -> Result<Self> {
        let mut destinations = vector_with_capacity(mappings.len())?;
        for mapping in &mappings {
            let end = mapping
                .destination_start
                .checked_add(mapping.len())
                .ok_or_else(|| Error::InvalidArgument("destination range overflow".into()))?;
            if end > output_len {
                return Err(Error::InvalidArgument(format!(
                    "destination range {}..{end} exceeds output length {output_len}",
                    mapping.destination_start
                )));
            }
            if !mapping.is_empty() {
                destinations.push(mapping.destination_start..end);
            }
        }
        destinations.sort_unstable_by_key(|range| range.start);
        if destinations
            .windows(2)
            .any(|ranges| ranges[0].end > ranges[1].start)
        {
            return Err(Error::InvalidArgument("destination ranges overlap".into()));
        }
        let fully_covers_output = if output_len == 0 {
            true
        } else {
            destinations.first().is_some_and(|range| range.start == 0)
                && destinations
                    .windows(2)
                    .all(|ranges| ranges[0].end == ranges[1].start)
                && destinations
                    .last()
                    .is_some_and(|range| range.end == output_len)
        };
        Ok(Self {
            mappings,
            output_len,
            fully_covers_output,
        })
    }

    pub fn contiguous(source: Range<usize>) -> Result<Self> {
        let output_len = source
            .end
            .checked_sub(source.start)
            .ok_or_else(|| Error::InvalidArgument("source range is reversed".into()))?;
        Self::new(vec![ByteMapping::new(source, 0)?], output_len)
    }

    pub fn output_len(&self) -> usize {
        self.output_len
    }

    pub fn mappings(&self) -> &[ByteMapping] {
        &self.mappings
    }

    pub(crate) fn fully_covers_output(&self) -> bool {
        self.fully_covers_output
    }
}
