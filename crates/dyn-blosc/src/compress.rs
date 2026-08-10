use crate::codec::{compress_piece, decompress_piece, DecodeContext};
use crate::encoder::{EncodeWorkspace, Encoder};
use crate::error::{
    join_workers, reserve_exact, resize_zeroed, vector_with_capacity, Error, Result,
};
use crate::filter::{apply_filter, reverse_filter};
use crate::format::{
    blosc1, dyn_blosc, encode_flags, BlockEntry, BloscVersion, Codec, Index, Shuffle, HEADER_LEN,
};
use crate::partition::balanced_ranges;

const MAX_SPLITS: usize = 16;
const MIN_SPLIT_ELEMENTS: usize = 128;

pub(crate) fn encode(source: &[u8], encoder: &Encoder) -> Result<Vec<u8>> {
    encoder.validate(source.len())?;
    let block_lengths = encoder.plan_blocks(source.len())?;
    match encoder.version {
        BloscVersion::Blosc1 => encode_blosc1(source, encoder, &block_lengths),
        BloscVersion::DynBlosc => encode_dyn_blosc(source, encoder, &block_lengths),
    }
}

fn encode_dyn_blosc(source: &[u8], encoder: &Encoder, block_lengths: &[u32]) -> Result<Vec<u8>> {
    if source.is_empty() {
        return encode_empty(encoder, BloscVersion::DynBlosc);
    }

    if encoder.level == 0 {
        return encode_raw(source, encoder, BloscVersion::DynBlosc);
    }

    let block_count = block_lengths.len();
    let index_bytes = block_count
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidOptions("block index size overflow".into()))?;
    let prefix_len = HEADER_LEN
        .checked_add(index_bytes)
        .ok_or_else(|| Error::InvalidOptions("block index prefix overflow".into()))?;
    let max_block_size = block_lengths.iter().copied().max().unwrap_or(0);

    let (mut output, entries) = if encoder.threads == 1 || block_count == 1 {
        encode_blocks_sequential(
            source,
            encoder,
            block_lengths,
            max_block_size as usize,
            prefix_len,
        )?
    } else {
        encode_blocks_parallel(
            source,
            encoder,
            block_lengths,
            max_block_size as usize,
            prefix_len,
        )?
    };

    let encoded_size = output.len();
    if encoded_size > i32::MAX as usize {
        return Err(Error::InvalidOptions(format!(
            "encoded size {encoded_size} exceeds the wire-format limit {}",
            i32::MAX
        )));
    }
    let header = dyn_blosc::Header::new(
        encoder.codec.format_version(),
        encode_flags(encoder.codec, encoder.shuffle, encoder.split_blocks, false),
        encoder.element_size as u8,
        source.len() as u32,
        encoded_size as u32,
        block_count as u32,
    )?;
    header.write(&mut output)?;
    Index::new_dyn(entries)?.write(&mut output[HEADER_LEN..prefix_len])?;
    Ok(output)
}

fn encode_blosc1(source: &[u8], encoder: &Encoder, block_lengths: &[u32]) -> Result<Vec<u8>> {
    if source.is_empty() {
        return encode_empty(encoder, BloscVersion::Blosc1);
    }
    if source.len() > (i32::MAX as usize - HEADER_LEN) {
        return Err(Error::InvalidOptions(format!(
            "Blosc1 source length {} exceeds the wire-format limit {}",
            source.len(),
            i32::MAX as usize - HEADER_LEN
        )));
    }
    if encoder.level == 0 || source.len() < MIN_SPLIT_ELEMENTS {
        return encode_raw(source, encoder, BloscVersion::Blosc1);
    }

    let block_count = block_lengths.len();
    let prefix_len = HEADER_LEN
        .checked_add(
            block_count
                .checked_mul(4)
                .ok_or_else(|| Error::InvalidOptions("Blosc1 index size overflow".into()))?,
        )
        .ok_or_else(|| Error::InvalidOptions("Blosc1 index prefix overflow".into()))?;
    let block_size = block_lengths[0] as usize;
    let (mut output, entries) = if encoder.threads == 1 || block_count == 1 {
        encode_blocks_sequential(source, encoder, block_lengths, block_size, prefix_len)?
    } else {
        encode_blocks_parallel(source, encoder, block_lengths, block_size, prefix_len)?
    };

    if output.len() >= source.len() + HEADER_LEN {
        return encode_raw(source, encoder, BloscVersion::Blosc1);
    }
    let encoded_size = output.len();
    let header = blosc1::Header::new(
        encoder.codec.format_version(),
        encode_flags(encoder.codec, encoder.shuffle, encoder.split_blocks, false),
        encoder.element_size as u8,
        source.len() as u32,
        block_size as u32,
        encoded_size as u32,
    )?;
    header.write(&mut output)?;
    let mut offsets = vector_with_capacity(entries.len())?;
    offsets.extend(entries.iter().map(|entry| entry.encoded_offset));
    Index::new_blosc1(offsets, block_size, source.len(), encoded_size)?
        .write(&mut output[HEADER_LEN..prefix_len])?;
    Ok(output)
}

fn encode_empty(encoder: &Encoder, version: BloscVersion) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    reserve_exact(&mut output, HEADER_LEN)?;
    output.resize(HEADER_LEN, 0);
    let flags = encode_flags(encoder.codec, Shuffle::None, false, false);
    match version {
        BloscVersion::Blosc1 => blosc1::Header::new(
            encoder.codec.format_version(),
            flags,
            encoder.element_size as u8,
            0,
            0,
            HEADER_LEN as u32,
        )?
        .write(&mut output)?,
        BloscVersion::DynBlosc => dyn_blosc::Header::new(
            encoder.codec.format_version(),
            flags,
            encoder.element_size as u8,
            0,
            HEADER_LEN as u32,
            0,
        )?
        .write(&mut output)?,
    }
    Ok(output)
}

fn encode_raw(source: &[u8], encoder: &Encoder, version: BloscVersion) -> Result<Vec<u8>> {
    let encoded_size = HEADER_LEN
        .checked_add(source.len())
        .ok_or_else(|| Error::InvalidOptions("encoded size overflow".into()))?;
    if encoded_size > u32::MAX as usize {
        return Err(Error::InvalidOptions(format!(
            "encoded size {encoded_size} exceeds the wire-format limit"
        )));
    }
    let mut output = Vec::new();
    reserve_exact(&mut output, encoded_size)?;
    output.resize(encoded_size, 0);
    let flags = encode_flags(encoder.codec, Shuffle::None, false, true);
    match version {
        BloscVersion::Blosc1 => blosc1::Header::new(
            encoder.codec.format_version(),
            flags,
            encoder.element_size as u8,
            source.len() as u32,
            blosc1_raw_block_size(source.len()) as u32,
            encoded_size as u32,
        )?
        .write(&mut output)?,
        BloscVersion::DynBlosc => dyn_blosc::Header::new(
            encoder.codec.format_version(),
            flags,
            encoder.element_size as u8,
            source.len() as u32,
            encoded_size as u32,
            1,
        )?
        .write(&mut output)?,
    }
    output[HEADER_LEN..].copy_from_slice(source);
    Ok(output)
}

fn blosc1_raw_block_size(decoded_size: usize) -> usize {
    decoded_size.min(blosc1::MAX_BLOCK_SIZE)
}

fn encode_blocks_sequential(
    source: &[u8],
    encoder: &Encoder,
    block_lengths: &[u32],
    max_block_size: usize,
    prefix_len: usize,
) -> Result<(Vec<u8>, Vec<BlockEntry>)> {
    let maximum_payload_size = block_lengths.iter().try_fold(0usize, |sum, &length| {
        sum.checked_add(maximum_encoded_block_len(length as usize, encoder)?)
            .ok_or_else(|| Error::InvalidOptions("encoded size overflow".into()))
    })?;
    let maximum_encoded_size = prefix_len
        .checked_add(maximum_payload_size)
        .ok_or_else(|| Error::InvalidOptions("encoded size overflow".into()))?;
    let mut output = Vec::new();
    reserve_exact(&mut output, maximum_encoded_size)?;
    output.resize(prefix_len, 0);
    let mut entries = vector_with_capacity(block_lengths.len())?;
    let mut workspace = EncodeWorkspace::new();
    workspace.prepare(max_block_size, encoder.shuffle)?;
    let mut source_offset = 0usize;

    for &length in block_lengths {
        let length = length as usize;
        let block = &source[source_offset..source_offset + length];
        source_offset += length;
        let encoded_offset = wire_offset(output.len())?;
        entries.push(BlockEntry {
            encoded_offset,
            decoded_length: length as u32,
        });
        let full_blosc1_block = encoder.version != BloscVersion::Blosc1 || length == max_block_size;
        encode_block(
            block,
            encoder,
            full_blosc1_block,
            &mut output,
            &mut workspace,
        )?;
    }
    Ok((output, entries))
}

fn encode_blocks_parallel(
    source: &[u8],
    encoder: &Encoder,
    block_lengths: &[u32],
    max_block_size: usize,
    prefix_len: usize,
) -> Result<(Vec<u8>, Vec<BlockEntry>)> {
    let partitions = balanced_ranges(
        block_lengths.len(),
        source.len(),
        encoder.threads,
        |block| block_lengths[block] as usize,
    )?;
    let mut payloads = vector_with_capacity(partitions.len())?;
    payloads.resize_with(partitions.len(), Vec::new);
    let mut encoded_lengths = vector_with_capacity(block_lengths.len())?;
    encoded_lengths.resize(block_lengths.len(), 0u32);
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = vector_with_capacity(partitions.len())?;
        let mut source_tail = source;
        let mut encoded_lengths_tail = encoded_lengths.as_mut_slice();
        for (partition, payload) in partitions.iter().zip(&mut payloads) {
            let lengths = &block_lengths[partition.clone()];
            let decoded_length = lengths.iter().map(|&length| length as usize).sum();
            let (worker_source, remaining_source) = source_tail.split_at(decoded_length);
            source_tail = remaining_source;
            let (worker_encoded_lengths, remaining_encoded_lengths) =
                encoded_lengths_tail.split_at_mut(partition.len());
            encoded_lengths_tail = remaining_encoded_lengths;
            handles.push(scope.spawn(move || -> Result<()> {
                let maximum_payload_size = lengths.iter().try_fold(0usize, |sum, &length| {
                    sum.checked_add(maximum_encoded_block_len(length as usize, encoder)?)
                        .ok_or_else(|| Error::InvalidOptions("encoded size overflow".into()))
                })?;
                reserve_exact(payload, maximum_payload_size)?;
                let mut workspace = EncodeWorkspace::new();
                let local_maximum_block_size = lengths.iter().copied().max().unwrap_or(0) as usize;
                workspace.prepare(local_maximum_block_size, encoder.shuffle)?;
                let mut decoded_offset = 0usize;
                for (&length, encoded_length) in lengths.iter().zip(worker_encoded_lengths) {
                    let length = length as usize;
                    let encoded_start = payload.len();
                    let full_blosc1_block =
                        encoder.version != BloscVersion::Blosc1 || length == max_block_size;
                    encode_block(
                        &worker_source[decoded_offset..decoded_offset + length],
                        encoder,
                        full_blosc1_block,
                        payload,
                        &mut workspace,
                    )?;
                    *encoded_length =
                        u32::try_from(payload.len() - encoded_start).map_err(|_| {
                            Error::InvalidOptions("encoded block length exceeds u32".into())
                        })?;
                    decoded_offset += length;
                }
                Ok(())
            }));
        }
        debug_assert!(source_tail.is_empty());
        debug_assert!(encoded_lengths_tail.is_empty());
        join_workers(handles)
    })?;

    let payload_size = payloads.iter().try_fold(0usize, |sum, payload| {
        sum.checked_add(payload.len())
            .ok_or_else(|| Error::InvalidOptions("encoded size overflow".into()))
    })?;
    let encoded_size = prefix_len
        .checked_add(payload_size)
        .ok_or_else(|| Error::InvalidOptions("encoded size overflow".into()))?;
    let mut output = Vec::new();
    reserve_exact(&mut output, encoded_size)?;
    output.resize(prefix_len, 0);
    let mut entries = vector_with_capacity(block_lengths.len())?;
    for (partition, payload) in partitions.iter().zip(payloads) {
        let mut encoded_offset = output.len();
        for block in partition.clone() {
            entries.push(BlockEntry {
                encoded_offset: wire_offset(encoded_offset)?,
                decoded_length: block_lengths[block],
            });
            encoded_offset = encoded_offset
                .checked_add(encoded_lengths[block] as usize)
                .ok_or_else(|| Error::InvalidOptions("encoded size overflow".into()))?;
        }
        debug_assert_eq!(encoded_offset, output.len() + payload.len());
        output.extend_from_slice(&payload);
    }
    Ok((output, entries))
}

fn wire_offset(offset: usize) -> Result<u32> {
    if offset > i32::MAX as usize {
        return Err(Error::InvalidOptions(format!(
            "block offset {offset} exceeds the wire-format limit {}",
            i32::MAX
        )));
    }
    Ok(offset as u32)
}

fn should_split(split_blocks: bool, element_size: usize, decoded_length: usize) -> bool {
    split_blocks
        && element_size > 1
        && element_size <= MAX_SPLITS
        && decoded_length.is_multiple_of(element_size)
        && decoded_length / element_size >= MIN_SPLIT_ELEMENTS
}

pub(crate) fn maximum_encoded_block_len(decoded_len: usize, encoder: &Encoder) -> Result<usize> {
    let piece_count = if should_split(encoder.split_blocks, encoder.element_size, decoded_len) {
        encoder.element_size
    } else {
        1
    };
    decoded_len
        .checked_add(
            piece_count
                .checked_mul(4)
                .ok_or_else(|| Error::InvalidOptions("block payload size overflow".into()))?,
        )
        .ok_or_else(|| Error::InvalidOptions("block payload size overflow".into()))
}

pub(crate) fn encode_block_payload(
    block: &[u8],
    encoder: &Encoder,
    output: &mut Vec<u8>,
    workspace: &mut EncodeWorkspace,
) -> Result<()> {
    if block.is_empty() {
        return Err(Error::InvalidOptions(
            "encoded block source must be non-empty".into(),
        ));
    }
    if block.len() > i32::MAX as usize {
        return Err(Error::InvalidOptions(format!(
            "block length {} exceeds the wire-format limit {}",
            block.len(),
            i32::MAX
        )));
    }
    encode_block(block, encoder, true, output, workspace)
}

fn encode_block(
    block: &[u8],
    encoder: &Encoder,
    allow_split: bool,
    output: &mut Vec<u8>,
    workspace: &mut EncodeWorkspace,
) -> Result<()> {
    reserve_exact(output, maximum_encoded_block_len(block.len(), encoder)?)?;
    workspace.prepare(block.len(), encoder.shuffle)?;
    let filtered = if encoder.shuffle == Shuffle::None {
        block
    } else {
        apply_filter(
            encoder.shuffle,
            encoder.element_size,
            block,
            &mut workspace.filtered[..block.len()],
            &mut workspace.bit_temp,
        )?;
        &workspace.filtered[..block.len()]
    };

    let split =
        allow_split && should_split(encoder.split_blocks, encoder.element_size, block.len());
    let split_count = if split { encoder.element_size } else { 1 };
    let piece_size = block.len() / split_count;
    for piece in filtered.chunks_exact(piece_size) {
        let max_encoded_size = piece_size
            .checked_add(piece_size / 8)
            .and_then(|size| size.checked_add(64))
            .ok_or_else(|| Error::InvalidOptions("codec scratch size overflow".into()))?;
        resize_zeroed(&mut workspace.codec, max_encoded_size)?;
        let compressed_size = compress_piece(
            encoder.codec,
            encoder.level,
            piece,
            &mut workspace.codec[..max_encoded_size],
            split,
            &mut workspace.codec_context,
        )?;
        if compressed_size > max_encoded_size {
            return Err(Error::Codec(format!(
                "{} compressor returned {compressed_size} bytes for a {max_encoded_size}-byte output buffer",
                encoder.codec.name()
            )));
        }
        let stored = if compressed_size == 0 || compressed_size >= piece_size {
            piece
        } else {
            &workspace.codec[..compressed_size]
        };
        // SAFETY: the reservation above covers the decoded length plus one
        // four-byte prefix per piece, and `stored.len() <= piece_size`.
        unsafe { append_piece_unchecked(output, stored) };
    }
    Ok(())
}

/// Append a length-prefixed encoded piece without repeating Vec capacity checks.
///
/// # Safety
///
/// `output` must have at least `4 + piece.len()` bytes of spare capacity, and
/// `piece.len()` must fit in a positive `i32`. `piece` must not overlap the
/// spare capacity of `output`.
#[inline]
unsafe fn append_piece_unchecked(output: &mut Vec<u8>, piece: &[u8]) {
    debug_assert!(!piece.is_empty());
    debug_assert!(piece.len() <= i32::MAX as usize);
    debug_assert!(output.capacity() - output.len() >= 4 + piece.len());

    let start = output.len();
    let length = (piece.len() as i32).to_le_bytes();
    // SAFETY: guaranteed by the function contract. Both copies target
    // initialized-or-spare Vec storage, do not overlap their sources, and the
    // final length stays within capacity.
    unsafe {
        let destination = output.as_mut_ptr().add(start);
        std::ptr::copy_nonoverlapping(length.as_ptr(), destination, length.len());
        std::ptr::copy_nonoverlapping(piece.as_ptr(), destination.add(length.len()), piece.len());
        output.set_len(start + length.len() + piece.len());
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockParameters {
    pub codec: Codec,
    pub shuffle: Shuffle,
    pub element_size: usize,
    pub split_count: usize,
}

pub(crate) fn decode_block_into(
    encoded: &[u8],
    decoded_length: usize,
    parameters: BlockParameters,
    destination: &mut [u8],
    filtered: &mut [u8],
    bit_temp: &mut [u8],
    codec_context: &mut DecodeContext,
) -> Result<()> {
    if destination.len() < decoded_length {
        return Err(Error::BufferTooSmall {
            need: decoded_length,
            have: destination.len(),
        });
    }
    if parameters.split_count == 0 || !decoded_length.is_multiple_of(parameters.split_count) {
        return Err(Error::InvalidFormat(format!(
            "invalid split count {} for decoded block length {decoded_length}",
            parameters.split_count
        )));
    }
    let piece_size = decoded_length / parameters.split_count;

    let decoded = if parameters.shuffle == Shuffle::None {
        &mut destination[..decoded_length]
    } else {
        if filtered.len() < decoded_length {
            return Err(Error::BufferTooSmall {
                need: decoded_length,
                have: filtered.len(),
            });
        }
        &mut filtered[..decoded_length]
    };

    if parameters.split_count == 1 {
        decode_single_piece(encoded, decoded, parameters.codec, codec_context)?;
    } else {
        let mut encoded_offset = 0usize;
        for piece_output in decoded.chunks_exact_mut(piece_size) {
            if encoded.len().saturating_sub(encoded_offset) < 4 {
                return Err(Error::InvalidFormat("truncated compressed block".into()));
            }
            // SAFETY: the remaining-length check above proves that four bytes are
            // available starting at `encoded_offset`.
            let compressed_size = unsafe {
                i32::from_le_bytes([
                    *encoded.get_unchecked(encoded_offset),
                    *encoded.get_unchecked(encoded_offset + 1),
                    *encoded.get_unchecked(encoded_offset + 2),
                    *encoded.get_unchecked(encoded_offset + 3),
                ])
            };
            if compressed_size <= 0 {
                return Err(Error::InvalidFormat(format!(
                    "compressed piece has invalid length {compressed_size}"
                )));
            }
            let compressed_size = compressed_size as usize;
            encoded_offset += 4;
            let piece_end = encoded_offset
                .checked_add(compressed_size)
                .ok_or_else(|| Error::InvalidFormat("compressed piece length overflow".into()))?;
            if piece_end > encoded.len() {
                return Err(Error::InvalidFormat(
                    "compressed piece overruns block".into(),
                ));
            }
            // SAFETY: `piece_end` was checked against the slice length, and
            // `encoded_offset <= piece_end` follows from checked addition.
            let piece_input = unsafe { encoded.get_unchecked(encoded_offset..piece_end) };
            if compressed_size == piece_size {
                piece_output.copy_from_slice(piece_input);
            } else {
                let actual =
                    decompress_piece(parameters.codec, piece_input, piece_output, codec_context)?;
                if actual != piece_size {
                    return Err(Error::Codec(format!(
                        "decoded {actual} bytes, expected {piece_size}"
                    )));
                }
            }
            encoded_offset = piece_end;
        }
        if encoded_offset != encoded.len() {
            return Err(Error::InvalidFormat(format!(
                "compressed block has {} trailing bytes",
                encoded.len() - encoded_offset
            )));
        }
    }

    if parameters.shuffle != Shuffle::None {
        reverse_filter(
            parameters.shuffle,
            parameters.element_size,
            &filtered[..decoded_length],
            &mut destination[..decoded_length],
            bit_temp,
        )?;
    }
    Ok(())
}

fn decode_single_piece(
    encoded: &[u8],
    decoded: &mut [u8],
    codec: Codec,
    codec_context: &mut DecodeContext,
) -> Result<()> {
    const PREFIX_LEN: usize = 4;
    if encoded.len() < PREFIX_LEN {
        return Err(Error::InvalidFormat("truncated compressed block".into()));
    }
    // SAFETY: the length check above proves that the full prefix is present.
    let compressed_size = unsafe {
        i32::from_le_bytes([
            *encoded.get_unchecked(0),
            *encoded.get_unchecked(1),
            *encoded.get_unchecked(2),
            *encoded.get_unchecked(3),
        ])
    };
    if compressed_size <= 0 {
        return Err(Error::InvalidFormat(format!(
            "compressed piece has invalid length {compressed_size}"
        )));
    }
    let compressed_size = compressed_size as usize;
    let piece_end = PREFIX_LEN
        .checked_add(compressed_size)
        .ok_or_else(|| Error::InvalidFormat("compressed piece length overflow".into()))?;
    if piece_end > encoded.len() {
        return Err(Error::InvalidFormat(
            "compressed piece overruns block".into(),
        ));
    }
    if piece_end != encoded.len() {
        return Err(Error::InvalidFormat(format!(
            "compressed block has {} trailing bytes",
            encoded.len() - piece_end
        )));
    }
    let piece_input = &encoded[PREFIX_LEN..];
    if compressed_size == decoded.len() {
        decoded.copy_from_slice(piece_input);
    } else {
        let actual = decompress_piece(codec, piece_input, decoded, codec_context)?;
        if actual != decoded.len() {
            return Err(Error::Codec(format!(
                "decoded {actual} bytes, expected {}",
                decoded.len()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_blosc1_block_size_respects_the_format_limit() {
        assert_eq!(
            blosc1_raw_block_size(blosc1::MAX_BLOCK_SIZE - 1),
            blosc1::MAX_BLOCK_SIZE - 1
        );
        assert_eq!(
            blosc1_raw_block_size(blosc1::MAX_BLOCK_SIZE),
            blosc1::MAX_BLOCK_SIZE
        );
        assert_eq!(
            blosc1_raw_block_size(blosc1::MAX_BLOCK_SIZE + 1),
            blosc1::MAX_BLOCK_SIZE
        );
        assert_eq!(
            blosc1_raw_block_size(blosc1::MAX_BUFFER_SIZE),
            blosc1::MAX_BLOCK_SIZE
        );
    }
}
