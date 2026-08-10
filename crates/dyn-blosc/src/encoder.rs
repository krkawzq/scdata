use crate::codec::EncodeContext;
use crate::compress;
use crate::error::{resize_zeroed, vector_with_capacity, Error, Result};
use crate::format::{
    blosc1, dyn_blosc, encode_flags, BlockEntry, BloscVersion, Codec, Header, Index, Shuffle,
    HEADER_LEN,
};
use crate::layout::{BlockDescriptor, ChunkLayout};

#[derive(Debug, Clone)]
pub(crate) enum BlockPartition {
    Automatic,
    Fixed(usize),
    Variable(Vec<usize>),
}

/// Encoding schema and tools for Blosc1 or DynBlosc chunks.
///
/// Holds only compression settings. Source bytes are never stored; every encode
/// method takes them as arguments from the caller.
///
/// The builder methods are intentionally infallible so configurations remain
/// easy to compose. [`Encoder::encode`] / [`Encoder::encode_block`] validate
/// the complete configuration before reading the source.
#[derive(Debug, Clone)]
pub struct Encoder {
    pub(crate) version: BloscVersion,
    pub(crate) level: u8,
    pub(crate) shuffle: Shuffle,
    pub(crate) element_size: usize,
    pub(crate) codec: Codec,
    pub(crate) split_blocks: bool,
    pub(crate) partition: BlockPartition,
    pub(crate) threads: usize,
}

/// Reusable scratch memory for block encoding.
#[derive(Debug, Default)]
pub struct EncodeWorkspace {
    pub(crate) filtered: Vec<u8>,
    pub(crate) bit_temp: Vec<u8>,
    pub(crate) codec: Vec<u8>,
    pub(crate) codec_context: EncodeContext,
}

impl EncodeWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn prepare(&mut self, block_len: usize, shuffle: Shuffle) -> Result<()> {
        if shuffle != Shuffle::None {
            resize_zeroed(&mut self.filtered, block_len)?;
        }
        if shuffle == Shuffle::Bits {
            resize_zeroed(&mut self.bit_temp, block_len)?;
        }
        Ok(())
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self {
            version: BloscVersion::DynBlosc,
            level: 5,
            shuffle: Shuffle::Bytes,
            element_size: 4,
            codec: Codec::Lz4,
            split_blocks: false,
            partition: BlockPartition::Automatic,
            threads: 1,
        }
    }
}

impl Encoder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    /// Select the emitted wire format.
    ///
    /// Blosc1 uses a fixed decoded block size; DynBlosc additionally supports
    /// caller-provided variable block lengths.
    pub fn version(mut self, version: BloscVersion) -> Self {
        self.version = version;
        self
    }

    #[must_use]
    pub fn codec(mut self, codec: Codec) -> Self {
        self.codec = codec;
        self
    }

    #[must_use]
    pub fn compression_level(mut self, level: u8) -> Self {
        self.level = level;
        self
    }

    #[must_use]
    pub fn shuffle(mut self, shuffle: Shuffle) -> Self {
        self.shuffle = shuffle;
        self
    }

    #[must_use]
    pub fn element_size(mut self, bytes: usize) -> Self {
        self.element_size = bytes;
        self
    }

    /// Enables or disables splitting each filtered block into element streams.
    #[must_use]
    pub fn split_blocks(mut self, enabled: bool) -> Self {
        self.split_blocks = enabled;
        self
    }

    #[must_use]
    pub fn automatic_block_size(mut self) -> Self {
        self.partition = BlockPartition::Automatic;
        self
    }

    #[must_use]
    pub fn block_size(mut self, bytes: usize) -> Self {
        self.partition = BlockPartition::Fixed(bytes);
        self
    }

    #[must_use]
    pub fn block_lengths<I>(mut self, lengths: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.partition = BlockPartition::Variable(lengths.into_iter().collect());
        self
    }

    #[must_use]
    pub fn threads(mut self, count: usize) -> Self {
        self.threads = count;
        self
    }

    /// Encode a complete versioned Blosc chunk (header + index + payloads).
    pub fn encode(&self, source: &[u8]) -> Result<Vec<u8>> {
        compress::encode(source, self)
    }

    /// Encode one block's wire payload (piece length prefixes + compressed bytes).
    ///
    /// Does not write a chunk header or block index; callers that need a full
    /// chunk should use [`Self::encode`] or assemble those themselves.
    pub fn encode_block(&self, source: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        let mut workspace = EncodeWorkspace::new();
        self.encode_block_into(source, &mut output, &mut workspace)?;
        Ok(output)
    }

    /// Append one block's wire payload using caller-owned reusable scratch memory.
    ///
    /// If encoding fails, `output` is restored to its original length.
    pub fn encode_block_into(
        &self,
        source: &[u8],
        output: &mut Vec<u8>,
        workspace: &mut EncodeWorkspace,
    ) -> Result<usize> {
        self.validate(source.len())?;
        if self.level == 0 {
            return Err(Error::InvalidOptions(
                "encode_block requires compression_level > 0 (raw/memcpy is chunk-level only)"
                    .into(),
            ));
        }
        let start = output.len();
        match compress::encode_block_payload(source, self, output, workspace) {
            Ok(()) => Ok(output.len() - start),
            Err(error) => {
                output.truncate(start);
                Err(error)
            }
        }
    }

    /// Maximum wire-payload length produced by [`Self::encode_block`].
    pub fn maximum_encoded_block_len(&self, decoded_len: usize) -> Result<usize> {
        self.validate(decoded_len)?;
        if decoded_len == 0 {
            return Err(Error::InvalidOptions(
                "encoded block source must be non-empty".into(),
            ));
        }
        if decoded_len > i32::MAX as usize {
            return Err(Error::InvalidOptions(format!(
                "block length {decoded_len} exceeds the wire-format limit {}",
                i32::MAX
            )));
        }
        if self.level == 0 {
            return Err(Error::InvalidOptions(
                "block payloads require compression_level > 0".into(),
            ));
        }
        compress::maximum_encoded_block_len(decoded_len, self)
    }

    /// Build a payload-free chunk layout for independently encoded blocks.
    ///
    /// Each descriptor must correspond to a payload produced with this
    /// encoder's codec, shuffle, element-size, and split settings.
    pub fn chunk_layout(&self, blocks: &[BlockDescriptor]) -> Result<ChunkLayout> {
        self.validate(0)?;
        if self.level == 0 && !blocks.is_empty() {
            return Err(Error::InvalidOptions(
                "block layouts require compression_level > 0".into(),
            ));
        }
        if blocks.len() > u32::MAX as usize {
            return Err(Error::InvalidOptions(format!(
                "block count {} exceeds the wire-format limit {}",
                blocks.len(),
                u32::MAX
            )));
        }
        if blocks.is_empty() {
            let header = match self.version {
                BloscVersion::Blosc1 => Header::Blosc1(blosc1::Header::new(
                    self.codec.format_version(),
                    encode_flags(self.codec, Shuffle::None, false, false),
                    self.element_size as u8,
                    0,
                    0,
                    HEADER_LEN as u32,
                )?),
                BloscVersion::DynBlosc => Header::DynBlosc(dyn_blosc::Header::new(
                    self.codec.format_version(),
                    encode_flags(self.codec, Shuffle::None, false, false),
                    self.element_size as u8,
                    0,
                    HEADER_LEN as u32,
                    0,
                )?),
            };
            return Ok(ChunkLayout::from_parts(header, None));
        }
        let index_width = match self.version {
            BloscVersion::Blosc1 => 4,
            BloscVersion::DynBlosc => 8,
        };
        let index_bytes = blocks
            .len()
            .checked_mul(index_width)
            .ok_or_else(|| Error::InvalidOptions("block index size overflow".into()))?;
        let prefix_len = HEADER_LEN
            .checked_add(index_bytes)
            .ok_or_else(|| Error::InvalidOptions("block index prefix overflow".into()))?;
        let block_size = blocks[0].decoded_len();
        if self.version == BloscVersion::Blosc1 && block_size > blosc1::MAX_BLOCK_SIZE {
            return Err(Error::InvalidOptions(format!(
                "Blosc1 block size {block_size} exceeds {}",
                blosc1::MAX_BLOCK_SIZE
            )));
        }

        let mut blosc1_offsets = match self.version {
            BloscVersion::Blosc1 => vector_with_capacity(blocks.len())?,
            BloscVersion::DynBlosc => Vec::new(),
        };
        let mut dyn_entries = match self.version {
            BloscVersion::Blosc1 => Vec::new(),
            BloscVersion::DynBlosc => vector_with_capacity(blocks.len())?,
        };
        let mut decoded_size = 0usize;
        let mut maximum_block_size = 0usize;
        let mut encoded_size = prefix_len;
        for (index, block) in blocks.iter().enumerate() {
            let is_last = index + 1 == blocks.len();
            if !is_last && !block.decoded_len().is_multiple_of(self.element_size) {
                return Err(Error::InvalidOptions(format!(
                    "block {index} decoded length {} is not a multiple of element size {}",
                    block.decoded_len(),
                    self.element_size
                )));
            }
            let maximum_encoded_len =
                compress::maximum_encoded_block_len(block.decoded_len(), self)?;
            if block.encoded_len() > maximum_encoded_len {
                return Err(Error::InvalidOptions(format!(
                    "block {index} encoded length {} exceeds the maximum {maximum_encoded_len} \
                     for decoded length {}",
                    block.encoded_len(),
                    block.decoded_len()
                )));
            }
            if self.version == BloscVersion::Blosc1
                && ((!is_last && block.decoded_len() != block_size)
                    || (is_last && block.decoded_len() > block_size))
            {
                return Err(Error::InvalidOptions(format!(
                    "Blosc1 block {index} has decoded length {}, expected {block_size} \
                     except for a shorter final block",
                    block.decoded_len()
                )));
            }
            if encoded_size > i32::MAX as usize {
                return Err(Error::InvalidOptions(format!(
                    "block offset {encoded_size} exceeds the wire-format limit {}",
                    i32::MAX
                )));
            }
            match self.version {
                BloscVersion::Blosc1 => blosc1_offsets.push(encoded_size as u32),
                BloscVersion::DynBlosc => dyn_entries.push(BlockEntry {
                    encoded_offset: encoded_size as u32,
                    decoded_length: block.decoded_len() as u32,
                }),
            }
            decoded_size = decoded_size
                .checked_add(block.decoded_len())
                .ok_or_else(|| Error::InvalidOptions("decoded size overflow".into()))?;
            maximum_block_size = maximum_block_size.max(block.decoded_len());
            encoded_size = encoded_size
                .checked_add(block.encoded_len())
                .ok_or_else(|| Error::InvalidOptions("encoded size overflow".into()))?;
        }
        if decoded_size > u32::MAX as usize {
            return Err(Error::InvalidOptions(format!(
                "decoded size {decoded_size} exceeds the wire-format limit {}",
                u32::MAX
            )));
        }
        if self.version == BloscVersion::Blosc1 && decoded_size > blosc1::MAX_BUFFER_SIZE {
            return Err(Error::InvalidOptions(format!(
                "Blosc1 decoded size {decoded_size} exceeds {}",
                blosc1::MAX_BUFFER_SIZE
            )));
        }
        if encoded_size > i32::MAX as usize {
            return Err(Error::InvalidOptions(format!(
                "encoded size {encoded_size} exceeds the wire-format limit {}",
                i32::MAX
            )));
        }
        let (header, index) = match self.version {
            BloscVersion::Blosc1 => (
                Header::Blosc1(blosc1::Header::new(
                    self.codec.format_version(),
                    encode_flags(self.codec, self.shuffle, self.split_blocks, false),
                    self.element_size as u8,
                    decoded_size as u32,
                    maximum_block_size as u32,
                    encoded_size as u32,
                )?),
                Index::new_blosc1(
                    blosc1_offsets,
                    maximum_block_size,
                    decoded_size,
                    encoded_size,
                )?,
            ),
            BloscVersion::DynBlosc => {
                let index = Index::new_dyn(dyn_entries)?;
                (
                    Header::DynBlosc(dyn_blosc::Header::new(
                        self.codec.format_version(),
                        encode_flags(self.codec, self.shuffle, self.split_blocks, false),
                        self.element_size as u8,
                        decoded_size as u32,
                        encoded_size as u32,
                        blocks.len() as u32,
                    )?),
                    index,
                )
            }
        };
        Ok(ChunkLayout::from_parts(header, Some(index)))
    }

    pub(crate) fn validate(&self, source_len: usize) -> Result<()> {
        if self.level > 9 {
            return Err(Error::InvalidOptions(
                "compression level must be in 0..=9".into(),
            ));
        }
        if !(1..=u8::MAX as usize).contains(&self.element_size) {
            return Err(Error::InvalidOptions(
                "element size must be in 1..=255".into(),
            ));
        }
        if self.threads == 0 {
            return Err(Error::InvalidOptions(
                "thread count must be non-zero".into(),
            ));
        }
        if source_len > u32::MAX as usize {
            return Err(Error::InvalidOptions(format!(
                "source length {source_len} exceeds the wire-format limit {}",
                u32::MAX
            )));
        }
        if self.version == BloscVersion::Blosc1 && source_len > blosc1::MAX_BUFFER_SIZE {
            return Err(Error::InvalidOptions(format!(
                "Blosc1 source length {source_len} exceeds {}",
                blosc1::MAX_BUFFER_SIZE
            )));
        }
        Ok(())
    }

    pub(crate) fn plan_blocks(&self, source_len: usize) -> Result<Vec<u32>> {
        if self.version == BloscVersion::Blosc1
            && matches!(self.partition, BlockPartition::Variable(_))
        {
            return Err(Error::InvalidOptions(
                "Blosc1 requires a fixed block size; variable block lengths are DynBlosc-only"
                    .into(),
            ));
        }
        let lengths = if let BlockPartition::Variable(lengths) = &self.partition {
            if lengths.is_empty() || lengths.contains(&0) {
                return Err(Error::InvalidOptions(
                    "variable block lengths must be non-empty and non-zero".into(),
                ));
            }
            let sum = lengths.iter().try_fold(0usize, |sum, &length| {
                sum.checked_add(length)
                    .ok_or_else(|| Error::InvalidOptions("block length sum overflow".into()))
            })?;
            if sum != source_len {
                return Err(Error::InvalidOptions(format!(
                    "block lengths sum to {sum}, but source length is {source_len}"
                )));
            }
            let mut planned = vector_with_capacity(lengths.len())?;
            for &length in lengths {
                if length > i32::MAX as usize {
                    return Err(Error::InvalidOptions(format!(
                        "block length {length} exceeds the wire-format limit {}",
                        i32::MAX
                    )));
                }
                planned.push(length as u32);
            }
            planned
        } else {
            if source_len == 0 {
                if matches!(self.partition, BlockPartition::Fixed(0)) {
                    return Err(Error::InvalidOptions("block size must be non-zero".into()));
                }
                return Ok(Vec::new());
            }
            let block_size = match self.partition {
                BlockPartition::Automatic => {
                    automatic_block_size(source_len, self.element_size, self.level)
                }
                BlockPartition::Fixed(0) => {
                    return Err(Error::InvalidOptions("block size must be non-zero".into()));
                }
                BlockPartition::Fixed(bytes) => bytes.min(source_len),
                BlockPartition::Variable(_) => unreachable!(),
            };
            if self.version == BloscVersion::Blosc1 && block_size > blosc1::MAX_BLOCK_SIZE {
                return Err(Error::InvalidOptions(format!(
                    "Blosc1 block size {block_size} exceeds {}",
                    blosc1::MAX_BLOCK_SIZE
                )));
            }
            let mut lengths = vector_with_capacity(source_len.div_ceil(block_size))?;
            let mut remaining = source_len;
            while remaining != 0 {
                let length = remaining.min(block_size);
                if length > i32::MAX as usize {
                    return Err(Error::InvalidOptions(format!(
                        "block length {length} exceeds the wire-format limit {}",
                        i32::MAX
                    )));
                }
                lengths.push(length as u32);
                remaining -= length;
            }
            lengths
        };

        ensure_element_aligned_block_boundaries(&lengths, self.element_size)?;
        Ok(lengths)
    }
}

/// Decoded block starts must fall on element boundaries: every block except the
/// last must have a length that is a multiple of `element_size`.
fn ensure_element_aligned_block_boundaries(lengths: &[u32], element_size: usize) -> Result<()> {
    if element_size <= 1 || lengths.len() <= 1 {
        return Ok(());
    }
    for (index, &length) in lengths[..lengths.len() - 1].iter().enumerate() {
        if !(length as usize).is_multiple_of(element_size) {
            return Err(Error::InvalidOptions(format!(
                "block {index} length {length} is not a multiple of element size {element_size}"
            )));
        }
    }
    Ok(())
}

fn automatic_block_size(source_len: usize, element_size: usize, level: u8) -> usize {
    const L1_SIZE: usize = 32 * 1024;
    const MIN_BLOCK_SIZE: usize = 128;

    // Keep a single block when the payload is shorter than one element so the
    // only decoded start remains element-aligned.
    if source_len < element_size {
        return source_len;
    }
    let mut block_size = source_len.min(L1_SIZE);
    block_size = match level {
        0 => block_size / 4,
        1 => block_size / 2,
        2 => block_size,
        3 => block_size.saturating_mul(2),
        4 | 5 => block_size.saturating_mul(4),
        6..=9 => block_size.saturating_mul(8),
        _ => block_size,
    };
    block_size = block_size.clamp(MIN_BLOCK_SIZE.min(source_len), source_len);
    if element_size > 1 && block_size >= element_size {
        block_size = (block_size / element_size * element_size).max(element_size);
    }
    block_size
}
