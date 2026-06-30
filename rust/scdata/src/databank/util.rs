use std::io;
use std::ptr;
use std::sync::Arc;

use tokio::sync::oneshot;

use crate::access::{AccessHandle, AccessItem, AccessRequest, ChunkKey, RangeCopy, SliceSpec};
use crate::codecs::{CodecError, SharedCodec};

use super::array::{ChunkRef, DType, DataValue};
use super::error::{DataBankError, DataBankResult};
use super::plan::{DenseSegment, RangeSegment};

use super::gene_axis::*;

pub(super) fn submit_access_item(
    access: &AccessHandle,
    item: AccessItem,
) -> DataBankResult<Vec<u8>> {
    let (reply, rx) = oneshot::channel();
    let request = AccessRequest {
        key: item.key,
        codec: item.codec,
        expected_size: item.expected_size,
        slice: item.slice,
        reply,
    };
    access.send(request)?;
    wait_access(rx)
}

pub(super) fn scatter_dense_segment<T: DataValue>(
    num_genes: usize,
    segment: &DenseSegment,
    bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()> {
    let expected_bytes = dense_segment_bytes(segment, src_dtype)?;
    if bytes.len() != expected_bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "decoded segment length is {}, expected {expected_bytes}",
            bytes.len()
        )));
    }

    let dst_start = segment
        .output_row
        .checked_mul(num_genes)
        .and_then(|base| base.checked_add(segment.output_col_start))
        .ok_or_else(|| DataBankError::InvalidConfig("output offset overflow".to_string()))?;
    let dst_end = dst_start
        .checked_add(segment.output_cols)
        .ok_or_else(|| DataBankError::InvalidConfig("output offset overflow".to_string()))?;
    copy_ne_bytes_to_values(bytes, src_dtype, &mut out[dst_start..dst_end])
}

pub(super) fn scatter_dense_segment_projected<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segment: &DenseSegment,
    bytes: &[u8],
    src_dtype: DType,
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    let expected_bytes = dense_segment_bytes(segment, src_dtype)?;
    if bytes.len() != expected_bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "decoded segment length is {}, expected {expected_bytes}",
            bytes.len()
        )));
    }
    let Some(projection) = gene_axis.projection() else {
        return scatter_dense_segment(output_genes, segment, bytes, src_dtype, out);
    };

    let source_end = segment
        .output_col_start
        .checked_add(segment.output_cols)
        .ok_or_else(|| DataBankError::InvalidConfig("dense source column overflow".to_string()))?;
    if source_end > dataset_num_genes {
        return Err(DataBankError::InvalidArrayMeta(
            "dense segment exceeds gene count".to_string(),
        ));
    }
    let row_base = segment
        .output_row
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output row overflow".to_string()))?;
    let row_end = row_base
        .checked_add(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output row end overflow".to_string()))?;
    if row_end > out.len() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out.len(),
        });
    }

    if let Some(output_col_start) =
        projection.contiguous_output_for_source_run(segment.output_col_start, segment.output_cols)
    {
        let dst_start = row_base.checked_add(output_col_start).ok_or_else(|| {
            DataBankError::InvalidConfig("dense output offset overflow".to_string())
        })?;
        let dst_end = dst_start
            .checked_add(segment.output_cols)
            .ok_or_else(|| DataBankError::InvalidConfig("dense output end overflow".to_string()))?;
        if dst_end > row_end {
            return Err(DataBankError::BufferSizeMismatch {
                expected: dst_end,
                actual: out.len(),
            });
        }
        return copy_ne_bytes_to_values(bytes, src_dtype, &mut out[dst_start..dst_end]);
    }

    // Per-element scatter with cast: read one source element and route it to
    // its projected output column.  `cast_slice_from` on a 1-element slice
    // reuses the same engine (identity fast path or conversion).
    let src_size = src_dtype.item_size();
    for local_col in 0..segment.output_cols {
        let source_col = segment.output_col_start + local_col;
        let Some(output_col) = projection.output_for_source(source_col) else {
            continue;
        };
        let src_elem = &bytes[local_col * src_size..(local_col + 1) * src_size];
        let dst_slot = &mut out[row_base + output_col..row_base + output_col + 1];
        T::cast_slice_from(src_elem, src_dtype, dst_slot)?;
    }
    Ok(())
}

pub(super) fn dense_segment_bytes(
    segment: &DenseSegment,
    src_dtype: DType,
) -> DataBankResult<usize> {
    segment
        .output_cols
        .checked_mul(src_dtype.item_size())
        .ok_or_else(|| DataBankError::InvalidArrayMeta("segment byte size overflow".to_string()))
}

pub(super) fn copy_ne_bytes_to_values<T: DataValue>(
    bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()> {
    // `cast_slice_from` already contains the identical-dtype fast path (raw
    // byte copy) and the per-element cast path, so this is the single entry
    // point for both the zero-overhead and the converting scatter.
    T::cast_slice_from(bytes, src_dtype, out)
}

pub(super) fn zero_values<T: DataValue>(out: &mut [T]) {
    if out.is_empty() {
        return;
    }
    // SAFETY: DataValue is sealed to numeric primitives plus transparent u16
    // half wrappers. Their zero value is represented by all-zero bytes.
    unsafe {
        ptr::write_bytes(out.as_mut_ptr(), 0, out.len());
    }
}

pub(super) fn zeroed_byte_vec(len: usize) -> Vec<u8> {
    vec![0; len]
}

pub(super) fn load_memory_group(
    bytes: &[u8],
    codec: &SharedCodec,
    expected_size: usize,
    decoded: bool,
    slice: &SliceSpec,
) -> DataBankResult<Vec<u8>> {
    if decoded || codec.is_identity() {
        if bytes.len() != expected_size {
            return Err(CodecError::SizeMismatch {
                codec: codec.name().to_string(),
                expected: expected_size,
                actual: bytes.len(),
            }
            .into());
        }
        return copy_slices(bytes, slice);
    }

    let decoded = codec.decode(bytes, Some(expected_size))?;
    copy_slices(&decoded, slice)
}

pub(super) fn copy_slices(data: &[u8], slice: &SliceSpec) -> DataBankResult<Vec<u8>> {
    let plan = slice.plan(data.len())?;
    let Some(ranges) = plan.ranges() else {
        return Ok(data.to_vec());
    };

    if ranges_are_packed(ranges, plan.output_len) {
        let mut out = Vec::with_capacity(plan.output_len);
        for range in ranges {
            out.extend_from_slice(&data[range.src_start..range.src_end]);
        }
        return Ok(out);
    }

    let mut out = vec![0; plan.output_len];
    for range in ranges {
        out[range.dst_offset..range.dst_offset + range.len()]
            .copy_from_slice(&data[range.src_start..range.src_end]);
    }
    Ok(out)
}

pub(super) fn ranges_are_packed(ranges: &[RangeCopy], output_len: usize) -> bool {
    let mut cursor = 0usize;
    for range in ranges {
        if range.dst_offset != cursor {
            return false;
        }
        let Some(next) = cursor.checked_add(range.len()) else {
            return false;
        };
        cursor = next;
    }
    cursor == output_len
}

pub(super) fn collect_prefetch_key(keys: &mut FastHashSet<ChunkKey>, chunk: &ChunkRef) {
    if let ChunkRef::AccessItem(item) = chunk {
        keys.insert(item.key);
    }
}

pub(super) fn collect_range_prefetch_keys(
    keys: &mut FastHashSet<ChunkKey>,
    segments: &[RangeSegment],
) {
    for segment in segments {
        collect_prefetch_key(keys, &segment.chunk);
    }
}

pub(super) fn prefetch_keys(
    access: &AccessHandle,
    keys: FastHashSet<ChunkKey>,
) -> DataBankResult<()> {
    let mut receivers = Vec::with_capacity(keys.len());
    for key in keys {
        receivers.push(access.prefetch(key)?);
    }
    for receiver in receivers {
        wait_prefetch(receiver)?;
    }
    Ok(())
}

pub(super) fn codec_id(codec: &SharedCodec) -> usize {
    Arc::as_ptr(codec) as *const () as usize
}

pub(super) fn wait_access(rx: oneshot::Receiver<io::Result<Vec<u8>>>) -> DataBankResult<Vec<u8>> {
    rx.blocking_recv()
        .map_err(|_| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "access reply dropped",
            ))
        })?
        .map_err(DataBankError::Io)
}

pub(super) fn wait_prefetch(rx: oneshot::Receiver<io::Result<()>>) -> DataBankResult<()> {
    rx.blocking_recv()
        .map_err(|_| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "prefetch reply dropped",
            ))
        })?
        .map_err(DataBankError::Io)
}
