//! Validated decoded-byte slice specifications.

use std::sync::Arc;

use super::error::AccessError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeCopy {
    pub dst_offset: usize,
    pub src_start: usize,
    pub src_end: usize,
}

impl RangeCopy {
    pub fn new(dst_offset: usize, src_start: usize, src_end: usize) -> Self {
        Self {
            dst_offset,
            src_start,
            src_end,
        }
    }

    pub(crate) fn len(self) -> usize {
        self.src_end - self.src_start
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SliceSpec {
    #[default]
    Full,
    Scatter(Arc<[RangeCopy]>),
    #[doc(hidden)]
    Invalid(String),
}

impl SliceSpec {
    pub fn full() -> Self {
        Self::Full
    }

    pub fn from_triples(triples: Vec<usize>) -> Result<Self, AccessError> {
        if triples.len() % 3 != 0 {
            return Err(AccessError::InvalidSlice(
                "slice spec must contain off/start/end triples".to_string(),
            ));
        }

        let mut ranges = Vec::with_capacity(triples.len() / 3);
        for triple in triples.chunks_exact(3) {
            ranges.push(RangeCopy::new(triple[0], triple[1], triple[2]));
        }
        Ok(Self::Scatter(Arc::from(ranges.into_boxed_slice())))
    }

    pub(crate) fn from_ranges(ranges: Vec<RangeCopy>) -> Self {
        Self::Scatter(Arc::from(ranges.into_boxed_slice()))
    }

    pub fn from_optional_triples(slice: Option<Vec<usize>>) -> Result<Self, AccessError> {
        match slice {
            Some(slice) => Self::from_triples(slice),
            None => Ok(Self::Full),
        }
    }

    pub(crate) fn from_optional_triples_deferred(slice: Option<Vec<usize>>) -> Self {
        Self::from_optional_triples(slice).unwrap_or_else(|err| Self::Invalid(err.to_string()))
    }

    pub(crate) fn plan(&self, data_len: usize) -> Result<SlicePlan, AccessError> {
        let ranges = match self {
            Self::Full => {
                return Ok(SlicePlan {
                    ranges: None,
                    output_len: data_len,
                    shape: SliceShape::Full,
                });
            }
            Self::Scatter(ranges) => ranges,
            Self::Invalid(message) => return Err(AccessError::InvalidSlice(message.clone())),
        };

        let mut output_len = 0usize;
        let mut cursor = 0usize;
        let mut identity = true;
        let mut sequential = true;
        for range in ranges.iter() {
            if range.src_start > range.src_end {
                return Err(AccessError::InvalidSlice(format!(
                    "slice start {} is greater than end {}",
                    range.src_start, range.src_end
                )));
            }
            if range.src_end > data_len {
                return Err(AccessError::InvalidSlice(format!(
                    "slice end {} exceeds decoded length {data_len}",
                    range.src_end
                )));
            }

            let span = range.src_end - range.src_start;
            let out_end = range.dst_offset.checked_add(span).ok_or_else(|| {
                AccessError::InvalidSlice("slice output length overflow".to_string())
            })?;

            identity &= range.dst_offset == cursor && range.src_start == cursor;
            if sequential && range.dst_offset == output_len {
                output_len = out_end;
            } else {
                sequential = false;
                output_len = output_len.max(out_end);
            }
            cursor = range.src_end;
            if !sequential {
                output_len = output_len.max(out_end);
            }
        }
        identity &= cursor == data_len && output_len == data_len;

        let shape = if identity {
            SliceShape::Full
        } else if sequential {
            SliceShape::Sequential
        } else {
            SliceShape::Sparse
        };

        Ok(SlicePlan {
            ranges: if identity {
                None
            } else {
                Some(Arc::clone(ranges))
            },
            output_len,
            shape,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SliceShape {
    Full,
    Sequential,
    Sparse,
}

#[derive(Debug, Clone)]
pub(crate) struct SlicePlan {
    pub(crate) ranges: Option<Arc<[RangeCopy]>>,
    pub(crate) output_len: usize,
    pub(crate) shape: SliceShape,
}

impl SlicePlan {
    pub(crate) fn ranges(&self) -> Option<&[RangeCopy]> {
        self.ranges.as_deref()
    }

    pub(crate) fn range_count(&self) -> usize {
        self.ranges.as_ref().map_or(0, |ranges| ranges.len())
    }
}
