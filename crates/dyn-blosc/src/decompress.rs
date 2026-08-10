use std::ops::Range;

use crate::codec::DecodeContext;
use crate::compress::{decode_block_into, BlockParameters};
use crate::error::{join_workers, resize_zeroed, vector_with_capacity, Error, LimitKind, Result};
use crate::format::{BloscVersion, Header, Index, Shuffle, HEADER_LEN};
use crate::layout::{BlockRange, Blocks, ChunkLayout, Metadata};
use crate::partition::balanced_ranges;
use crate::ranges::{ByteMapping, ByteSelection};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    pub maximum_decoded_size: usize,
    pub maximum_block_size: usize,
    pub maximum_block_count: usize,
}

impl DecodeLimits {
    pub const fn unlimited() -> Self {
        Self {
            maximum_decoded_size: usize::MAX,
            maximum_block_size: usize::MAX,
            maximum_block_count: usize::MAX,
        }
    }

    #[must_use]
    pub const fn maximum_decoded_size(mut self, bytes: usize) -> Self {
        self.maximum_decoded_size = bytes;
        self
    }

    #[must_use]
    pub const fn maximum_block_size(mut self, bytes: usize) -> Self {
        self.maximum_block_size = bytes;
        self
    }

    #[must_use]
    pub const fn maximum_block_count(mut self, count: usize) -> Self {
        self.maximum_block_count = count;
        self
    }

    fn check_header(self, header: Header) -> Result<()> {
        check_limit(
            LimitKind::DecodedSize,
            header.decoded_size(),
            self.maximum_decoded_size,
        )?;
        check_limit(
            LimitKind::BlockCount,
            header.block_count(),
            self.maximum_block_count,
        )?;
        // Blosc1 exposes a fixed block size in the header; DynBlosc derives it
        // from the index and is checked after the layout is built.
        if let Header::Blosc1(header) = header {
            check_limit(
                LimitKind::BlockSize,
                header.block_size(),
                self.maximum_block_size,
            )?;
        }
        Ok(())
    }

    fn check_layout(self, layout: &ChunkLayout) -> Result<()> {
        self.check_header(layout.header())?;
        check_limit(
            LimitKind::BlockSize,
            layout.maximum_block_size(),
            self.maximum_block_size,
        )
    }
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self::unlimited()
    }
}

fn check_limit(kind: LimitKind, actual: usize, limit: usize) -> Result<()> {
    if actual > limit {
        Err(Error::LimitExceeded {
            kind,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

/// Reusable scratch memory for block and selection decoding.
#[derive(Debug, Default)]
pub struct DecodeWorkspace {
    filtered: Vec<u8>,
    bit_temp: Vec<u8>,
    block: Vec<u8>,
    codec_context: DecodeContext,
}

impl DecodeWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    fn prepare(&mut self, block_len: usize, shuffle: Shuffle, decoded_block: bool) -> Result<()> {
        if shuffle != Shuffle::None {
            resize_zeroed(&mut self.filtered, block_len)?;
        }
        if shuffle == Shuffle::Bits {
            resize_zeroed(&mut self.bit_temp, block_len)?;
        }
        if decoded_block {
            resize_zeroed(&mut self.block, block_len)?;
        }
        Ok(())
    }
}

/// Self-contained decoder for one validated block payload.
///
/// Build this once with [`Decoder::block_decoder`] when a caller retains chunk
/// metadata but loads and decodes individual blocks repeatedly. Unlike
/// [`Decoder`], this compact value does not retain the chunk index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDecoder {
    encoded_len: u32,
    decoded_len: u32,
    element_size: u8,
    split_count: u8,
    codec: crate::format::Codec,
    shuffle: Shuffle,
    raw: bool,
}

impl BlockDecoder {
    fn new(
        encoded_len: usize,
        decoded_len: usize,
        parameters: BlockParameters,
        raw: bool,
    ) -> Result<Self> {
        Ok(Self {
            encoded_len: u32::try_from(encoded_len)
                .map_err(|_| Error::InvalidFormat("encoded block length exceeds u32".into()))?,
            decoded_len: u32::try_from(decoded_len)
                .map_err(|_| Error::InvalidFormat("decoded block length exceeds u32".into()))?,
            element_size: u8::try_from(parameters.element_size)
                .map_err(|_| Error::InvalidFormat("element size exceeds u8".into()))?,
            split_count: u8::try_from(parameters.split_count)
                .map_err(|_| Error::InvalidFormat("split count exceeds u8".into()))?,
            codec: parameters.codec,
            shuffle: parameters.shuffle,
            raw,
        })
    }

    #[inline]
    pub fn encoded_len(self) -> usize {
        self.encoded_len as usize
    }

    #[inline]
    pub fn decoded_len(self) -> usize {
        self.decoded_len as usize
    }

    /// Decode one externally loaded payload into caller-owned memory.
    ///
    /// The encoded slice must contain exactly this block's payload. On error,
    /// bytes in `output` may already have been modified.
    #[inline]
    pub fn decode_into(
        self,
        encoded_block: &[u8],
        output: &mut [u8],
        workspace: &mut DecodeWorkspace,
    ) -> Result<usize> {
        let encoded_len = self.encoded_len();
        let decoded_len = self.decoded_len();
        if encoded_block.len() != encoded_len {
            return Err(Error::InvalidArgument(format!(
                "encoded block has {} bytes, expected {encoded_len}",
                encoded_block.len()
            )));
        }
        if output.len() < decoded_len {
            return Err(Error::BufferTooSmall {
                need: decoded_len,
                have: output.len(),
            });
        }
        if self.raw {
            output[..decoded_len].copy_from_slice(encoded_block);
            return Ok(decoded_len);
        }

        workspace.prepare(decoded_len, self.shuffle, false)?;
        decode_block_into(
            encoded_block,
            decoded_len,
            BlockParameters {
                codec: self.codec,
                shuffle: self.shuffle,
                element_size: self.element_size as usize,
                split_count: self.split_count as usize,
            },
            &mut output[..decoded_len],
            &mut workspace.filtered,
            &mut workspace.bit_temp,
            &mut workspace.codec_context,
        )?;
        Ok(decoded_len)
    }
}

/// Validated Blosc1 or DynBlosc chunk schema and decode tools.
///
/// Holds only the header and block index. Compressed payload bytes are never
/// stored; every decode method takes them as arguments from the caller.
#[derive(Debug, Clone)]
pub struct Decoder {
    layout: ChunkLayout,
    threads: usize,
    limits: DecodeLimits,
}

impl Decoder {
    /// Build a schema from header + block index only.
    ///
    /// The slice need not contain the compressed payload; it must cover at least
    /// [`Self::index_prefix_len`] bytes for the chunk.
    pub fn from_prefix(prefix: &[u8]) -> Result<Self> {
        Self::from_prefix_with_limits(prefix, DecodeLimits::unlimited())
    }

    /// Build a schema while enforcing limits before allocating the block index.
    pub fn from_prefix_with_limits(prefix: &[u8], limits: DecodeLimits) -> Result<Self> {
        let header = Header::parse(prefix)?;
        limits.check_header(header)?;
        let index = if header.decoded_size() == 0 || header.is_raw() {
            None
        } else {
            Some(Index::parse(header, prefix)?)
        };
        let layout = ChunkLayout::from_parts(header, index);
        limits.check_layout(&layout)?;
        Ok(Self {
            layout,
            threads: 1,
            limits,
        })
    }

    /// Build a schema from a complete encoded chunk.
    ///
    /// Validates `encoded.len() == header.encoded_size` but does not retain the
    /// bytes.
    pub fn from_encoded(encoded: &[u8]) -> Result<Self> {
        Self::from_encoded_with_limits(encoded, DecodeLimits::unlimited())
    }

    pub fn from_encoded_with_limits(encoded: &[u8], limits: DecodeLimits) -> Result<Self> {
        let decoder = Self::from_prefix_with_limits(encoded, limits)?;
        if encoded.len() != decoder.layout.header.encoded_size() {
            return Err(Error::InvalidFormat(format!(
                "encoded size {} does not match input length {}",
                decoder.layout.header.encoded_size(),
                encoded.len()
            )));
        }
        Ok(decoder)
    }

    pub fn from_layout(layout: ChunkLayout) -> Self {
        Self::from_layout_with_limits(layout, DecodeLimits::unlimited())
            .expect("an unlimited limit accepts every validated layout")
    }

    pub fn from_layout_with_limits(layout: ChunkLayout, limits: DecodeLimits) -> Result<Self> {
        limits.check_layout(&layout)?;
        Ok(Self {
            layout,
            threads: 1,
            limits,
        })
    }

    /// Number of leading bytes (header + index) needed to build a [`Decoder`].
    ///
    /// `bytes` must contain at least [`crate::HEADER_LEN`] bytes so the shared
    /// header can be parsed.
    pub fn index_prefix_len(bytes: &[u8]) -> Result<usize> {
        let header = Header::parse(bytes)?;
        if header.decoded_size() == 0 || header.is_raw() {
            Ok(HEADER_LEN)
        } else {
            header.index_prefix_len()
        }
    }

    #[must_use]
    pub fn threads(mut self, count: usize) -> Self {
        self.threads = count;
        self
    }

    pub fn with_limits(mut self, limits: DecodeLimits) -> Result<Self> {
        limits.check_layout(&self.layout)?;
        self.limits = limits;
        Ok(self)
    }

    pub fn metadata(&self) -> Metadata {
        self.layout.metadata()
    }

    pub fn header(&self) -> Header {
        self.layout.header()
    }

    pub fn layout(&self) -> &ChunkLayout {
        &self.layout
    }

    pub fn block(&self, block_index: usize) -> Option<BlockRange> {
        self.layout.block(block_index)
    }

    pub fn blocks(&self) -> Blocks<'_> {
        self.layout.blocks()
    }

    /// Prepare a compact decoder for one independently loaded block payload.
    pub fn block_decoder(&self, block_index: usize) -> Result<BlockDecoder> {
        self.check_configuration()?;
        let range = self.block_or_error(block_index)?;
        BlockDecoder::new(
            range.encoded_range().len(),
            range.decoded_len(),
            self.block_parameters(block_index, range.decoded_len())?,
            self.layout.header.is_raw(),
        )
    }

    pub fn decode(&self, encoded: &[u8]) -> Result<Vec<u8>> {
        self.check_chunk(encoded)?;
        let decoded_size = self.layout.header.decoded_size();
        let mut output = Vec::new();
        resize_zeroed(&mut output, decoded_size)?;
        self.decode_into_validated(encoded, &mut output)?;
        Ok(output)
    }

    /// Decode a complete chunk into caller-owned memory.
    ///
    /// On error, bytes in `output` may already have been modified.
    pub fn decode_into(&self, encoded: &[u8], output: &mut [u8]) -> Result<usize> {
        self.check_chunk(encoded)?;
        self.decode_into_validated(encoded, output)
    }

    fn decode_into_validated(&self, encoded: &[u8], output: &mut [u8]) -> Result<usize> {
        let decoded_size = self.layout.header.decoded_size();
        if output.len() < decoded_size {
            return Err(Error::BufferTooSmall {
                need: decoded_size,
                have: output.len(),
            });
        }
        if decoded_size == 0 {
            return Ok(0);
        }
        if self.layout.header.is_raw() {
            let header_len = HEADER_LEN;
            let payload = encoded
                .get(header_len..header_len + decoded_size)
                .ok_or_else(|| Error::InvalidFormat("raw payload is truncated".into()))?;
            output[..decoded_size].copy_from_slice(payload);
            return Ok(decoded_size);
        }

        let index = self.index();
        if self.threads == 1 || index.len() == 1 {
            self.decode_blocks(
                encoded,
                0..index.len(),
                &mut output[..decoded_size],
                0,
                self.layout.maximum_block_size(),
            )?;
        } else {
            self.decode_parallel(encoded, &mut output[..decoded_size])?;
        }
        Ok(decoded_size)
    }

    pub fn decode_block(&self, block_index: usize, encoded_block: &[u8]) -> Result<Vec<u8>> {
        self.check_configuration()?;
        let range = self.block_or_error(block_index)?;
        let mut output = Vec::new();
        resize_zeroed(&mut output, range.decoded_len())?;
        let mut workspace = DecodeWorkspace::new();
        self.decode_block_into(block_index, encoded_block, &mut output, &mut workspace)?;
        Ok(output)
    }

    /// Decode one externally loaded block with reusable caller-owned scratch.
    ///
    /// On error, bytes in `output` may already have been modified.
    pub fn decode_block_into(
        &self,
        block_index: usize,
        encoded_block: &[u8],
        output: &mut [u8],
        workspace: &mut DecodeWorkspace,
    ) -> Result<usize> {
        self.block_decoder(block_index)?
            .decode_into(encoded_block, output, workspace)
    }

    pub fn decode_bytes(&self, encoded: &[u8], range: Range<usize>) -> Result<Vec<u8>> {
        self.decode_selection(encoded, &ByteSelection::contiguous(range)?)
    }

    pub fn decode_items(&self, encoded: &[u8], range: Range<usize>) -> Result<Vec<u8>> {
        let element_size = self.layout.header.element_size();
        let start = range
            .start
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidArgument("item range start overflow".into()))?;
        let end = range
            .end
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidArgument("item range end overflow".into()))?;
        self.decode_bytes(encoded, start..end)
    }

    pub fn decode_selection(&self, encoded: &[u8], selection: &ByteSelection) -> Result<Vec<u8>> {
        self.check_chunk(encoded)?;
        self.validate_selection(selection)?;
        let mut output = Vec::new();
        resize_zeroed(&mut output, selection.output_len())?;
        let mut workspace = DecodeWorkspace::new();
        self.decode_selection_into_validated(
            encoded,
            selection,
            &mut output,
            &mut workspace,
            false,
        )?;
        Ok(output)
    }

    /// Decode selected byte ranges with reusable caller-owned scratch.
    ///
    /// On error, bytes in `output` may already have been modified.
    pub fn decode_selection_into(
        &self,
        encoded: &[u8],
        selection: &ByteSelection,
        output: &mut [u8],
        workspace: &mut DecodeWorkspace,
    ) -> Result<usize> {
        self.check_chunk(encoded)?;
        self.validate_selection(selection)?;
        self.decode_selection_into_validated(encoded, selection, output, workspace, true)
    }

    fn decode_selection_into_validated(
        &self,
        encoded: &[u8],
        selection: &ByteSelection,
        output: &mut [u8],
        workspace: &mut DecodeWorkspace,
        clear_unmapped: bool,
    ) -> Result<usize> {
        if output.len() < selection.output_len() {
            return Err(Error::BufferTooSmall {
                need: selection.output_len(),
                have: output.len(),
            });
        }
        let decoded_size = self.layout.header.decoded_size();
        let output = &mut output[..selection.output_len()];
        if clear_unmapped && !selection.fully_covers_output() {
            output.fill(0);
        }
        if self.layout.header.is_raw() {
            let header_len = HEADER_LEN;
            let payload = encoded
                .get(header_len..header_len + decoded_size)
                .ok_or_else(|| Error::InvalidFormat("raw payload is truncated".into()))?;
            copy_mappings(payload, 0, selection.mappings(), output);
            return Ok(output.len());
        }
        if decoded_size == 0 {
            return Ok(output.len());
        }

        let index = self.index();
        let mut spans = vector_with_capacity(selection.mappings().len())?;
        for (mapping_index, mapping) in selection.mappings().iter().enumerate() {
            let blocks = index.blocks_intersecting(mapping.source());
            if !blocks.is_empty() {
                spans.push(MappingSpan {
                    first_block: blocks.start,
                    end_block: blocks.end,
                    mapping_index,
                });
            }
        }
        if spans.is_empty() {
            return Ok(output.len());
        }
        spans.sort_unstable_by_key(|span| span.first_block);
        let mut active = vector_with_capacity(spans.len())?;
        let mut next_span = 0usize;
        let mut block_index = spans[0].first_block;
        let mut maximum_direct_block = 0usize;
        let mut maximum_buffered_block = 0usize;
        while next_span < spans.len() || !active.is_empty() {
            active.retain(|span: &MappingSpan| span.end_block > block_index);
            if active.is_empty()
                && next_span < spans.len()
                && block_index < spans[next_span].first_block
            {
                block_index = spans[next_span].first_block;
            }
            while next_span < spans.len() && spans[next_span].first_block <= block_index {
                let span = spans[next_span];
                if span.end_block > block_index {
                    active.push(span);
                }
                next_span += 1;
            }
            if active.is_empty() {
                continue;
            }
            // SAFETY: `blocks_intersecting` only yields indexes in the validated
            // index entry count.
            let (entry, block_start, encoded_range) = unsafe {
                (
                    index.entry_unchecked(block_index),
                    index.decoded_start_unchecked(block_index),
                    index.encoded_range_unchecked(self.layout.header, block_index),
                )
            };
            // SAFETY: index validation bounds every encoded range by the
            // complete chunk length checked by `check_chunk`.
            let encoded_block = unsafe { encoded.get_unchecked(encoded_range) };
            let decoded_length = entry.decoded_length as usize;
            let parameters = self.block_parameters(block_index, decoded_length)?;
            let direct_destination = if active.len() == 1 {
                let mapping = &selection.mappings()[active[0].mapping_index];
                let block_end = block_start + decoded_length;
                (mapping.source().start <= block_start && mapping.source().end >= block_end).then(
                    || {
                        let start =
                            mapping.destination_start() + (block_start - mapping.source().start);
                        start..start + decoded_length
                    },
                )
            } else {
                None
            };
            if direct_destination.is_some() {
                if decoded_length > maximum_direct_block {
                    workspace.prepare(decoded_length, self.layout.header.shuffle(), false)?;
                    maximum_direct_block = decoded_length;
                }
            } else if decoded_length > maximum_buffered_block {
                workspace.prepare(decoded_length, self.layout.header.shuffle(), true)?;
                maximum_direct_block = maximum_direct_block.max(decoded_length);
                maximum_buffered_block = decoded_length;
            }
            if let Some(destination) = direct_destination {
                decode_block_into(
                    encoded_block,
                    decoded_length,
                    parameters,
                    &mut output[destination],
                    &mut workspace.filtered,
                    &mut workspace.bit_temp,
                    &mut workspace.codec_context,
                )?;
            } else {
                decode_block_into(
                    encoded_block,
                    decoded_length,
                    parameters,
                    &mut workspace.block[..decoded_length],
                    &mut workspace.filtered,
                    &mut workspace.bit_temp,
                    &mut workspace.codec_context,
                )?;
                copy_mappings(
                    &workspace.block[..decoded_length],
                    block_start,
                    active
                        .iter()
                        .map(|span| &selection.mappings()[span.mapping_index]),
                    output,
                );
            }
            block_index += 1;
        }
        Ok(output.len())
    }

    fn check_configuration(&self) -> Result<()> {
        if self.threads == 0 {
            return Err(Error::InvalidArgument(
                "decoder thread count must be non-zero".into(),
            ));
        }
        self.limits.check_layout(&self.layout)
    }

    fn check_chunk(&self, encoded: &[u8]) -> Result<()> {
        self.check_configuration()?;
        self.layout.matches_chunk(encoded)
    }

    fn validate_selection(&self, selection: &ByteSelection) -> Result<()> {
        let decoded_size = self.layout.header.decoded_size();
        for mapping in selection.mappings() {
            if mapping.source().end > decoded_size {
                return Err(Error::InvalidArgument(format!(
                    "source range {:?} exceeds decoded size {decoded_size}",
                    mapping.source()
                )));
            }
        }
        Ok(())
    }

    fn index(&self) -> &Index {
        self.layout.index()
    }

    fn block_or_error(&self, block_index: usize) -> Result<BlockRange> {
        self.block(block_index)
            .ok_or_else(|| Error::InvalidArgument(format!("block {block_index} is out of range")))
    }

    fn block_parameters(
        &self,
        block_index: usize,
        decoded_length: usize,
    ) -> Result<BlockParameters> {
        let element_size = self.layout.header.element_size();
        let split_count = if !self.layout.header.split_blocks() {
            1
        } else {
            match self.layout.header.version() {
                BloscVersion::Blosc1 => {
                    let full_block = decoded_length == self.layout.maximum_block_size();
                    if full_block && element_size <= 16 && decoded_length / element_size >= 128 {
                        element_size
                    } else {
                        1
                    }
                }
                BloscVersion::DynBlosc => {
                    if element_size > 1
                        && element_size <= 16
                        && decoded_length.is_multiple_of(element_size)
                        && decoded_length / element_size >= 128
                    {
                        element_size
                    } else {
                        1
                    }
                }
            }
        };
        debug_assert!(block_index < self.layout.header.block_count());
        Ok(BlockParameters {
            codec: self.layout.header.codec()?,
            shuffle: self.layout.header.shuffle(),
            element_size,
            split_count,
        })
    }

    fn decode_parallel(&self, encoded: &[u8], output: &mut [u8]) -> Result<()> {
        if self.index().has_uniform_decoded_layout() {
            self.decode_parallel_by_block_count(encoded, output)
        } else {
            self.decode_parallel_by_decoded_size(encoded, output)
        }
    }

    fn decode_parallel_by_block_count(&self, encoded: &[u8], output: &mut [u8]) -> Result<()> {
        let index = self.index();
        let block_count = index.len();
        let blocks_per_thread = block_count.div_ceil(self.threads);
        let maximum_block_size = self.layout.maximum_block_size();
        std::thread::scope(|scope| -> Result<()> {
            let worker_count = block_count.div_ceil(blocks_per_thread);
            let mut handles = vector_with_capacity(worker_count)?;
            let mut output_tail = output;
            for first_block in (0..block_count).step_by(blocks_per_thread) {
                let last_block = (first_block + blocks_per_thread).min(block_count);
                // SAFETY: the loop only produces values below `block_count`.
                let decoded_start = unsafe { index.decoded_start_unchecked(first_block) };
                let decoded_end = if last_block == block_count {
                    self.layout.header.decoded_size()
                } else {
                    // SAFETY: this branch proves `last_block < block_count`.
                    unsafe { index.decoded_start_unchecked(last_block) }
                };
                let length = decoded_end - decoded_start;
                let (thread_output, tail) = output_tail.split_at_mut(length);
                output_tail = tail;
                handles.push(scope.spawn(move || {
                    self.decode_blocks(
                        encoded,
                        first_block..last_block,
                        thread_output,
                        decoded_start,
                        maximum_block_size,
                    )
                }));
            }
            join_workers(handles)
        })
    }

    fn decode_parallel_by_decoded_size(&self, encoded: &[u8], output: &mut [u8]) -> Result<()> {
        let index = self.index();
        let block_count = index.len();
        let partitions = balanced_ranges(
            block_count,
            self.layout.header.decoded_size(),
            self.threads,
            |block| {
                // SAFETY: the partitioner only requests weights in
                // `0..block_count`.
                unsafe { index.entry_unchecked(block).decoded_length as usize }
            },
        )?;
        std::thread::scope(|scope| -> Result<()> {
            let mut handles = vector_with_capacity(partitions.len())?;
            let mut output_tail = output;
            for blocks in partitions {
                // SAFETY: balanced partitions are non-empty and bounded by
                // `block_count`.
                let decoded_start = unsafe { index.decoded_start_unchecked(blocks.start) };
                let decoded_end = if blocks.end == block_count {
                    self.layout.header.decoded_size()
                } else {
                    // SAFETY: this branch proves `blocks.end < block_count`.
                    unsafe { index.decoded_start_unchecked(blocks.end) }
                };
                let length = decoded_end - decoded_start;
                let maximum_block_size = blocks
                    .clone()
                    .map(|block| {
                        // SAFETY: every partition is bounded by `block_count`.
                        unsafe { index.entry_unchecked(block).decoded_length as usize }
                    })
                    .max()
                    .unwrap_or(0);
                let (thread_output, tail) = output_tail.split_at_mut(length);
                output_tail = tail;
                handles.push(scope.spawn(move || {
                    self.decode_blocks(
                        encoded,
                        blocks,
                        thread_output,
                        decoded_start,
                        maximum_block_size,
                    )
                }));
            }
            debug_assert!(output_tail.is_empty());
            join_workers(handles)
        })
    }

    fn decode_blocks(
        &self,
        encoded: &[u8],
        blocks: Range<usize>,
        output: &mut [u8],
        output_decoded_start: usize,
        maximum_block_size: usize,
    ) -> Result<()> {
        let index = self.index();
        let mut workspace = DecodeWorkspace::new();
        workspace.prepare(maximum_block_size, self.layout.header.shuffle(), false)?;
        for block_index in blocks {
            // SAFETY: every caller constructs `blocks` within the validated
            // index entry count.
            let (entry, absolute_decoded_start, encoded_range) = unsafe {
                (
                    index.entry_unchecked(block_index),
                    index.decoded_start_unchecked(block_index),
                    index.encoded_range_unchecked(self.layout.header, block_index),
                )
            };
            let decoded_start = absolute_decoded_start - output_decoded_start;
            let decoded_end = decoded_start + entry.decoded_length as usize;
            // SAFETY: index validation bounds the encoded range by the complete
            // chunk, and the caller provides exactly the decoded span covered
            // by `blocks`.
            let (encoded_block, decoded_block) = unsafe {
                (
                    encoded.get_unchecked(encoded_range),
                    output.get_unchecked_mut(decoded_start..decoded_end),
                )
            };
            let parameters = self.block_parameters(block_index, entry.decoded_length as usize)?;
            decode_block_into(
                encoded_block,
                entry.decoded_length as usize,
                parameters,
                decoded_block,
                &mut workspace.filtered,
                &mut workspace.bit_temp,
                &mut workspace.codec_context,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct MappingSpan {
    first_block: usize,
    end_block: usize,
    mapping_index: usize,
}

fn copy_mappings<'a>(
    block: &[u8],
    block_start: usize,
    mappings: impl IntoIterator<Item = &'a ByteMapping>,
    output: &mut [u8],
) {
    let block_end = block_start + block.len();
    for mapping in mappings {
        let start = mapping.source().start.max(block_start);
        let end = mapping.source().end.min(block_end);
        if start >= end {
            continue;
        }
        let source_start = start - block_start;
        let length = end - start;
        let destination_start = mapping.destination_start() + (start - mapping.source().start);
        output[destination_start..destination_start + length]
            .copy_from_slice(&block[source_start..source_start + length]);
    }
}
