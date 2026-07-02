use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::codecs::{
    blosc_lz4_block_split_count, decode_blosc_lz4_block, decode_blosc_lz4_block_partial_prefixes,
    unshuffle_bytes, BloscLz4Block, CodecError, CodecResult,
};

use super::load::NativeBlockCacheKey;
use super::metadata::{NativeBloscBlockIndex, NativeBloscBlockRange};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeBlockConsumer {
    pub(crate) decoded_start: usize,
    pub(crate) decoded_end: usize,
    pub(crate) output_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeBlockOutputConsumer {
    pub(crate) output_index: usize,
    pub(crate) consumer: NativeBlockConsumer,
}

#[derive(Debug, Default)]
pub(crate) struct NativeBlockScratch {
    shuffled: Vec<u8>,
    decoded: Vec<u8>,
    split_prefixes: Vec<usize>,
}

#[derive(Debug)]
struct NativeBlockDecodedEntry {
    decoded: Arc<[u8]>,
    bytes_len: usize,
}

#[derive(Debug, Default)]
struct NativeBlockDecodedCacheState {
    entries: HashMap<NativeBlockCacheKey, NativeBlockDecodedEntry>,
    order: VecDeque<NativeBlockCacheKey>,
    bytes: usize,
}

#[derive(Debug)]
struct NativeBlockDecodedCacheShard {
    capacity_bytes: usize,
    state: Mutex<NativeBlockDecodedCacheState>,
}

impl NativeBlockDecodedCacheShard {
    fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            state: Mutex::new(NativeBlockDecodedCacheState::default()),
        }
    }

    fn get(&self, key: NativeBlockCacheKey) -> Option<Arc<[u8]>> {
        self.state
            .lock()
            .expect("native decoded block cache lock poisoned")
            .entries
            .get(&key)
            .map(|entry| Arc::clone(&entry.decoded))
    }

    fn insert(&self, key: NativeBlockCacheKey, decoded: Arc<[u8]>) {
        if self.capacity_bytes == 0 || decoded.is_empty() || decoded.len() > self.capacity_bytes {
            return;
        }
        let mut state = self
            .state
            .lock()
            .expect("native decoded block cache lock poisoned");
        if let Some(old_len) = state.entries.get(&key).map(|entry| entry.bytes_len) {
            state.bytes = state.bytes.saturating_sub(old_len);
            state.bytes = state.bytes.saturating_add(decoded.len());
            if let Some(old) = state.entries.get_mut(&key) {
                old.bytes_len = decoded.len();
                old.decoded = decoded;
            }
            return;
        }
        state.bytes = state.bytes.saturating_add(decoded.len());
        state.order.push_back(key);
        state.entries.insert(
            key,
            NativeBlockDecodedEntry {
                bytes_len: decoded.len(),
                decoded,
            },
        );
        while state.bytes > self.capacity_bytes {
            let Some(victim) = state.order.pop_front() else {
                break;
            };
            if let Some(entry) = state.entries.remove(&victim) {
                state.bytes = state.bytes.saturating_sub(entry.bytes_len);
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct NativeBlockDecodedCache {
    shards: Vec<NativeBlockDecodedCacheShard>,
}

impl NativeBlockDecodedCache {
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        const SHARDS: usize = 8;
        let shard_capacity = capacity_bytes.div_ceil(SHARDS).max(1);
        let shards = (0..SHARDS)
            .map(|_| NativeBlockDecodedCacheShard::new(shard_capacity))
            .collect();
        Self { shards }
    }

    fn get(&self, key: NativeBlockCacheKey) -> Option<Arc<[u8]>> {
        self.shard_for_key(key).get(key)
    }

    fn insert(&self, key: NativeBlockCacheKey, decoded: Arc<[u8]>) {
        self.shard_for_key(key).insert(key, decoded);
    }

    fn shard_for_key(&self, key: NativeBlockCacheKey) -> &NativeBlockDecodedCacheShard {
        let hash = key.file.0.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ key.offset.rotate_left(17)
            ^ (key.len as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        &self.shards[(hash as usize) % self.shards.len()]
    }
}

impl NativeBlockScratch {
    /// Resize the shuffled scratch to `len` without zero-filling the new tail.
    ///
    /// The buffer's `[old_len..len]` bytes are left uninitialized; callers
    /// (`decode_blosc_lz4_block` on the shuffled path) write exactly `len`
    /// bytes before reading them, so the uninitialized tail is never observed.
    fn resize_shuffled_uninit(&mut self, len: usize) {
        resize_uninit(&mut self.shuffled, len);
    }

    /// Resize the decoded scratch to `len` without zero-filling. See
    /// [`resize_shuffled_uninit`](Self::resize_shuffled_uninit).
    fn resize_decoded_uninit(&mut self, len: usize) {
        resize_uninit(&mut self.decoded, len);
    }
}

/// Grow `buf` to `len` without writing the new bytes.
///
/// `Vec::resize(len, 0)` spends a memset zeroing bytes that the Blosc decode
/// and unshuffle paths overwrite completely. This skips that write: it
/// `reserve`s the capacity, then `set_len`s. Callers must overwrite the full
/// `[0..len)` range before reading it.
fn resize_uninit(buf: &mut Vec<u8>, len: usize) {
    if len > buf.capacity() {
        buf.reserve(len - buf.len());
    }
    // SAFETY: `reserve` (or the existing capacity) guarantees `capacity >= len`.
    // The bytes in `[old_len..len]` are uninitialized, but every caller writes
    // exactly `len` bytes before reading, so no uninitialized memory is ever
    // observed. `set_len` on a shrink is also safe: the dropped tail was
    // already initialized on a prior call.
    unsafe { buf.set_len(len) };
}

pub(crate) fn scatter_loaded_blosc_block(
    index: &NativeBloscBlockIndex,
    block_idx: usize,
    loaded_block: &[u8],
    consumers: &[NativeBlockConsumer],
    output: &mut [u8],
    scratch: &mut NativeBlockScratch,
) -> CodecResult<()> {
    scatter_loaded_blosc_block_cached(
        index,
        block_idx,
        None,
        loaded_block,
        consumers,
        output,
        scratch,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_loaded_blosc_block_cached(
    index: &NativeBloscBlockIndex,
    block_idx: usize,
    key: Option<NativeBlockCacheKey>,
    loaded_block: &[u8],
    consumers: &[NativeBlockConsumer],
    output: &mut [u8],
    scratch: &mut NativeBlockScratch,
    decoded_cache: Option<&NativeBlockDecodedCache>,
) -> CodecResult<()> {
    // `block_idx` is the array index produced by the planner (it iterates
    // `index.blocks` in order), so the `block_idx == array position` invariant
    // holds by construction — direct indexing skips the `Option` and the
    // error-formatting allocation of `.get().ok_or_else(...)` on every block.
    debug_assert!(
        block_idx < index.blocks.len(),
        "native scatter block_idx {block_idx} out of range",
    );
    let block = &index.blocks[block_idx];
    if loaded_block.len() != block.compressed_len {
        return Err(decode_error(format!(
            "native loaded block has {} bytes, expected {}",
            loaded_block.len(),
            block.compressed_len
        )));
    }

    if index.memcpyed {
        return scatter_decoded_block(block, loaded_block, consumers, output);
    }

    if let (Some(cache), Some(key)) = (decoded_cache, key) {
        if let Some(decoded) = cache.get(key) {
            if decoded.len() == block.decoded_len {
                return scatter_decoded_block(block, &decoded, consumers, output);
            }
        }
        decode_full_block_to_scratch(index, block, loaded_block, scratch)?;
        scatter_decoded_block(block, &scratch.decoded, consumers, output)?;
        cache.insert(key, Arc::from(scratch.decoded.clone().into_boxed_slice()));
        return Ok(());
    }

    // The loaded slice is exactly one Blosc block's compressed payload, so the
    // source range starts at 0 and spans the whole loaded buffer.
    let block_desc = BloscLz4Block {
        size: block.decoded_len,
        leftover: block.leftover,
        src_offset: 0,
        src_limit: loaded_block.len(),
    };

    if index.byte_shuffled {
        // Decode LZ4 into the shuffled plane layout, then either unshuffle the
        // whole block (high consumer coverage) or gather only the requested
        // byte ranges directly out of the shuffled planes (low coverage).
        scratch.resize_shuffled_uninit(block.decoded_len);
        if should_full_unshuffle(block, consumers) {
            decode_blosc_lz4_block(
                "blosc",
                loaded_block,
                index.header,
                block_desc,
                &mut scratch.shuffled,
            )?;
            scratch.resize_decoded_uninit(block.decoded_len);
            unshuffle_bytes(index.type_size, &scratch.shuffled, &mut scratch.decoded);
            scatter_decoded_block(block, &scratch.decoded, consumers, output)
        } else {
            fill_required_shuffled_split_prefixes(
                index.header,
                index.type_size,
                block,
                consumers,
                &mut scratch.split_prefixes,
            )?;
            if should_partial_decode_shuffled(block, &scratch.split_prefixes) {
                decode_blosc_lz4_block_partial_prefixes(
                    "blosc",
                    loaded_block,
                    index.header,
                    block_desc,
                    &scratch.split_prefixes,
                    &mut scratch.shuffled,
                )?;
            } else {
                decode_blosc_lz4_block(
                    "blosc",
                    loaded_block,
                    index.header,
                    block_desc,
                    &mut scratch.shuffled,
                )?;
            }
            scatter_shuffled_block_ranges(
                index.type_size,
                block,
                &scratch.shuffled,
                consumers,
                output,
            )
        }
    } else {
        scratch.resize_decoded_uninit(block.decoded_len);
        decode_blosc_lz4_block(
            "blosc",
            loaded_block,
            index.header,
            block_desc,
            &mut scratch.decoded,
        )?;
        scatter_decoded_block(block, &scratch.decoded, consumers, output)
    }
}

pub(crate) fn scatter_loaded_blosc_block_multi_output(
    index: &NativeBloscBlockIndex,
    block_idx: usize,
    loaded_block: &[u8],
    targets: &[NativeBlockOutputConsumer],
    outputs: &mut [Option<Vec<u8>>],
    scratch: &mut NativeBlockScratch,
) -> CodecResult<()> {
    scatter_loaded_blosc_block_multi_output_cached(
        index,
        block_idx,
        None,
        loaded_block,
        targets,
        outputs,
        scratch,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_loaded_blosc_block_multi_output_cached(
    index: &NativeBloscBlockIndex,
    block_idx: usize,
    key: Option<NativeBlockCacheKey>,
    loaded_block: &[u8],
    targets: &[NativeBlockOutputConsumer],
    outputs: &mut [Option<Vec<u8>>],
    scratch: &mut NativeBlockScratch,
    decoded_cache: Option<&NativeBlockDecodedCache>,
) -> CodecResult<()> {
    debug_assert!(
        block_idx < index.blocks.len(),
        "native scatter block_idx {block_idx} out of range",
    );
    let block = &index.blocks[block_idx];
    if loaded_block.len() != block.compressed_len {
        return Err(decode_error(format!(
            "native loaded block has {} bytes, expected {}",
            loaded_block.len(),
            block.compressed_len
        )));
    }

    if index.memcpyed {
        return scatter_decoded_block_to_outputs(block, loaded_block, targets, outputs);
    }

    if let (Some(cache), Some(key)) = (decoded_cache, key) {
        if let Some(decoded) = cache.get(key) {
            if decoded.len() == block.decoded_len {
                return scatter_decoded_block_to_outputs(block, &decoded, targets, outputs);
            }
        }
        decode_full_block_to_scratch(index, block, loaded_block, scratch)?;
        scatter_decoded_block_to_outputs(block, &scratch.decoded, targets, outputs)?;
        cache.insert(key, Arc::from(scratch.decoded.clone().into_boxed_slice()));
        return Ok(());
    }

    let block_desc = BloscLz4Block {
        size: block.decoded_len,
        leftover: block.leftover,
        src_offset: 0,
        src_limit: loaded_block.len(),
    };

    if index.byte_shuffled {
        scratch.resize_shuffled_uninit(block.decoded_len);
        if should_full_unshuffle_targets(block, targets) {
            decode_blosc_lz4_block(
                "blosc",
                loaded_block,
                index.header,
                block_desc,
                &mut scratch.shuffled,
            )?;
            scratch.resize_decoded_uninit(block.decoded_len);
            unshuffle_bytes(index.type_size, &scratch.shuffled, &mut scratch.decoded);
            scatter_decoded_block_to_outputs(block, &scratch.decoded, targets, outputs)
        } else {
            fill_required_shuffled_split_prefixes_targets(
                index.header,
                index.type_size,
                block,
                targets,
                &mut scratch.split_prefixes,
            )?;
            if should_partial_decode_shuffled(block, &scratch.split_prefixes) {
                decode_blosc_lz4_block_partial_prefixes(
                    "blosc",
                    loaded_block,
                    index.header,
                    block_desc,
                    &scratch.split_prefixes,
                    &mut scratch.shuffled,
                )?;
            } else {
                decode_blosc_lz4_block(
                    "blosc",
                    loaded_block,
                    index.header,
                    block_desc,
                    &mut scratch.shuffled,
                )?;
            }
            scatter_shuffled_block_ranges_to_outputs(
                index.type_size,
                block,
                &scratch.shuffled,
                targets,
                outputs,
            )
        }
    } else {
        scratch.resize_decoded_uninit(block.decoded_len);
        decode_blosc_lz4_block(
            "blosc",
            loaded_block,
            index.header,
            block_desc,
            &mut scratch.decoded,
        )?;
        scatter_decoded_block_to_outputs(block, &scratch.decoded, targets, outputs)
    }
}

fn should_full_unshuffle(block: &NativeBloscBlockRange, consumers: &[NativeBlockConsumer]) -> bool {
    requested_block_bytes(block, consumers) * 2 >= block.decoded_len
}

fn decode_full_block_to_scratch(
    index: &NativeBloscBlockIndex,
    block: &NativeBloscBlockRange,
    loaded_block: &[u8],
    scratch: &mut NativeBlockScratch,
) -> CodecResult<()> {
    let block_desc = BloscLz4Block {
        size: block.decoded_len,
        leftover: block.leftover,
        src_offset: 0,
        src_limit: loaded_block.len(),
    };
    if index.byte_shuffled {
        scratch.resize_shuffled_uninit(block.decoded_len);
        decode_blosc_lz4_block(
            "blosc",
            loaded_block,
            index.header,
            block_desc,
            &mut scratch.shuffled,
        )?;
        scratch.resize_decoded_uninit(block.decoded_len);
        unshuffle_bytes(index.type_size, &scratch.shuffled, &mut scratch.decoded);
    } else {
        scratch.resize_decoded_uninit(block.decoded_len);
        decode_blosc_lz4_block(
            "blosc",
            loaded_block,
            index.header,
            block_desc,
            &mut scratch.decoded,
        )?;
    }
    Ok(())
}

fn should_full_unshuffle_targets(
    block: &NativeBloscBlockRange,
    targets: &[NativeBlockOutputConsumer],
) -> bool {
    requested_block_bytes_targets(block, targets) * 2 >= block.decoded_len
}

fn should_partial_decode_shuffled(block: &NativeBloscBlockRange, split_prefixes: &[usize]) -> bool {
    split_prefixes.iter().sum::<usize>() * 4 < block.decoded_len * 3
}

fn fill_required_shuffled_split_prefixes(
    header: crate::codecs::BloscHeader,
    typesize: usize,
    block: &NativeBloscBlockRange,
    consumers: &[NativeBlockConsumer],
    prefixes: &mut Vec<usize>,
) -> CodecResult<()> {
    reset_shuffled_split_prefixes(header, block, prefixes)?;
    let block_start = block.decoded_offset;
    let block_end = block_start + block.decoded_len;
    let split_size = split_size_for_prefixes(block, prefixes.len())?;
    for consumer in consumers {
        let start = consumer.decoded_start.max(block_start);
        let end = consumer.decoded_end.min(block_end);
        if start >= end {
            continue;
        }
        record_required_shuffled_range(
            typesize,
            block.decoded_len,
            split_size,
            start - block_start,
            end - block_start,
            prefixes,
        )?;
    }
    Ok(())
}

fn fill_required_shuffled_split_prefixes_targets(
    header: crate::codecs::BloscHeader,
    typesize: usize,
    block: &NativeBloscBlockRange,
    targets: &[NativeBlockOutputConsumer],
    prefixes: &mut Vec<usize>,
) -> CodecResult<()> {
    reset_shuffled_split_prefixes(header, block, prefixes)?;
    let block_start = block.decoded_offset;
    let block_end = block_start + block.decoded_len;
    let split_size = split_size_for_prefixes(block, prefixes.len())?;
    for target in targets {
        let consumer = target.consumer;
        let start = consumer.decoded_start.max(block_start);
        let end = consumer.decoded_end.min(block_end);
        if start >= end {
            continue;
        }
        record_required_shuffled_range(
            typesize,
            block.decoded_len,
            split_size,
            start - block_start,
            end - block_start,
            prefixes,
        )?;
    }
    Ok(())
}

fn reset_shuffled_split_prefixes(
    header: crate::codecs::BloscHeader,
    block: &NativeBloscBlockRange,
    prefixes: &mut Vec<usize>,
) -> CodecResult<()> {
    let block_desc = BloscLz4Block {
        size: block.decoded_len,
        leftover: block.leftover,
        src_offset: 0,
        src_limit: block.compressed_len,
    };
    let nsplits = blosc_lz4_block_split_count(header, block_desc);
    let _ = split_size_for_prefixes(block, nsplits)?;
    prefixes.clear();
    prefixes.resize(nsplits, 0);
    Ok(())
}

fn split_size_for_prefixes(block: &NativeBloscBlockRange, nsplits: usize) -> CodecResult<usize> {
    if nsplits == 0 || block.decoded_len % nsplits != 0 {
        return Err(decode_error("invalid Blosc split block size"));
    }
    Ok(block.decoded_len / nsplits)
}

fn record_required_shuffled_range(
    typesize: usize,
    decoded_len: usize,
    split_size: usize,
    rel_start: usize,
    rel_end: usize,
    prefixes: &mut [usize],
) -> CodecResult<()> {
    if rel_start >= rel_end {
        return Ok(());
    }
    if typesize <= 1 {
        return record_required_shuffled_offset(rel_end - 1, split_size, prefixes);
    }
    let elements = decoded_len / typesize;
    let main_len = elements * typesize;
    if rel_end - rel_start >= 64
        && rel_end <= main_len
        && rel_start % typesize == 0
        && rel_end % typesize == 0
    {
        let elem_end = rel_end / typesize;
        if elem_end == 0 {
            return Ok(());
        }
        let last_element = elem_end - 1;
        for byte in 0..typesize {
            record_required_shuffled_offset(byte * elements + last_element, split_size, prefixes)?;
        }
        return Ok(());
    }

    for rel in rel_start..rel_end {
        let shuffled = shuffled_offset_for_decoded_byte(typesize, decoded_len, rel);
        record_required_shuffled_offset(shuffled, split_size, prefixes)?;
    }
    Ok(())
}

fn record_required_shuffled_offset(
    shuffled: usize,
    split_size: usize,
    prefixes: &mut [usize],
) -> CodecResult<()> {
    let split_idx = shuffled / split_size;
    let split_offset = shuffled % split_size;
    let Some(prefix) = prefixes.get_mut(split_idx) else {
        return Err(decode_error("native shuffled prefix exceeds split count"));
    };
    *prefix = (*prefix).max(split_offset + 1);
    Ok(())
}

fn shuffled_offset_for_decoded_byte(typesize: usize, decoded_len: usize, rel: usize) -> usize {
    if typesize <= 1 {
        return rel;
    }
    let elements = decoded_len / typesize;
    let main_len = elements * typesize;
    if rel < main_len {
        let element = rel / typesize;
        let byte = rel % typesize;
        byte * elements + element
    } else {
        rel
    }
}

fn requested_block_bytes(
    block: &NativeBloscBlockRange,
    consumers: &[NativeBlockConsumer],
) -> usize {
    let block_start = block.decoded_offset;
    let block_end = block_start + block.decoded_len;
    consumers
        .iter()
        .map(|consumer| {
            let start = consumer.decoded_start.max(block_start);
            let end = consumer.decoded_end.min(block_end);
            end.saturating_sub(start)
        })
        .sum()
}

fn requested_block_bytes_targets(
    block: &NativeBloscBlockRange,
    targets: &[NativeBlockOutputConsumer],
) -> usize {
    let block_start = block.decoded_offset;
    let block_end = block_start + block.decoded_len;
    targets
        .iter()
        .map(|target| {
            let consumer = target.consumer;
            let start = consumer.decoded_start.max(block_start);
            let end = consumer.decoded_end.min(block_end);
            end.saturating_sub(start)
        })
        .sum()
}

fn scatter_decoded_block(
    block: &NativeBloscBlockRange,
    decoded: &[u8],
    consumers: &[NativeBlockConsumer],
    output: &mut [u8],
) -> CodecResult<()> {
    let block_start = block.decoded_offset;
    let block_end = block_start + block.decoded_len;
    for consumer in consumers {
        // The three bounds below hold by construction: the planner emits a
        // consumer only with `decoded_start < decoded_end` (it skips empty
        // ranges) over a block it intersects, and the slice plan guarantees the
        // output window fits. `decoded` is sized to `block.decoded_len` by every
        // caller (memcpyed payload or `resize_decoded_uninit`). Mirrors the
        // `block_idx` invariant in `scatter_loaded_blosc_block` — direct indexing
        // skips the per-consumer `Option`/`format!` error path on the hot loop.
        debug_assert!(
            consumer.decoded_start <= consumer.decoded_end,
            "native scatter consumer decoded range inverted",
        );
        let start = consumer.decoded_start.max(block_start);
        let end = consumer.decoded_end.min(block_end);
        if start >= end {
            continue;
        }
        let dst_start = consumer.output_offset + (start - consumer.decoded_start);
        let dst_end = dst_start + (end - start);
        debug_assert!(
            dst_end <= output.len(),
            "native scatter consumer output range exceeds buffer",
        );
        let src_start = start - block_start;
        let src_end = end - block_start;
        debug_assert!(
            src_end <= decoded.len(),
            "native scatter consumer decoded range exceeds buffer",
        );
        output[dst_start..dst_end].copy_from_slice(&decoded[src_start..src_end]);
    }
    Ok(())
}

fn scatter_decoded_block_to_outputs(
    block: &NativeBloscBlockRange,
    decoded: &[u8],
    targets: &[NativeBlockOutputConsumer],
    outputs: &mut [Option<Vec<u8>>],
) -> CodecResult<()> {
    for target in targets {
        let Some(output) = outputs
            .get_mut(target.output_index)
            .and_then(Option::as_mut)
        else {
            return Err(decode_error("native multi-output target is missing output"));
        };
        scatter_decoded_block(
            block,
            decoded,
            std::slice::from_ref(&target.consumer),
            output,
        )?;
    }
    Ok(())
}

fn scatter_shuffled_block_ranges(
    typesize: usize,
    block: &NativeBloscBlockRange,
    shuffled: &[u8],
    consumers: &[NativeBlockConsumer],
    output: &mut [u8],
) -> CodecResult<()> {
    if typesize <= 1 {
        return scatter_decoded_block(block, shuffled, consumers, output);
    }
    let elements = shuffled.len() / typesize;
    let main_len = elements * typesize;
    let block_start = block.decoded_offset;
    let block_end = block_start + block.decoded_len;
    for consumer in consumers {
        // Same construction guarantees as `scatter_decoded_block`: the planner
        // emits only non-empty intersecting consumers whose output window fits.
        debug_assert!(
            consumer.decoded_start <= consumer.decoded_end,
            "native scatter consumer decoded range inverted",
        );
        let start = consumer.decoded_start.max(block_start);
        let end = consumer.decoded_end.min(block_end);
        if start >= end {
            continue;
        }
        let dst = consumer.output_offset + (start - consumer.decoded_start);
        let dst_end = dst + (end - start);
        debug_assert!(
            dst_end <= output.len(),
            "native scatter consumer output range exceeds buffer",
        );
        copy_shuffled_range(
            typesize,
            elements,
            main_len,
            block_start,
            shuffled,
            start,
            end,
            output,
            dst,
        )?;
    }
    Ok(())
}

fn scatter_shuffled_block_ranges_to_outputs(
    typesize: usize,
    block: &NativeBloscBlockRange,
    shuffled: &[u8],
    targets: &[NativeBlockOutputConsumer],
    outputs: &mut [Option<Vec<u8>>],
) -> CodecResult<()> {
    for target in targets {
        let Some(output) = outputs
            .get_mut(target.output_index)
            .and_then(Option::as_mut)
        else {
            return Err(decode_error("native multi-output target is missing output"));
        };
        scatter_shuffled_block_ranges(
            typesize,
            block,
            shuffled,
            std::slice::from_ref(&target.consumer),
            output,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn copy_shuffled_range(
    typesize: usize,
    elements: usize,
    main_len: usize,
    block_start: usize,
    shuffled: &[u8],
    start: usize,
    end: usize,
    output: &mut [u8],
    dst: usize,
) -> CodecResult<()> {
    let rel_start = start - block_start;
    let rel_end = end - block_start;
    if rel_end <= main_len && rel_start % typesize == 0 && rel_end % typesize == 0 {
        match typesize {
            2 => {
                copy_shuffled_range_2(elements, shuffled, rel_start / 2, rel_end / 2, output, dst);
                return Ok(());
            }
            4 => {
                copy_shuffled_range_4(elements, shuffled, rel_start / 4, rel_end / 4, output, dst);
                return Ok(());
            }
            _ => {
                copy_shuffled_range_generic(
                    typesize,
                    elements,
                    shuffled,
                    rel_start / typesize,
                    rel_end / typesize,
                    output,
                    dst,
                );
                return Ok(());
            }
        }
    }

    let mut dst = dst;
    for rel in rel_start..rel_end {
        let src = if rel < main_len {
            let element = rel / typesize;
            let byte = rel % typesize;
            byte * elements + element
        } else {
            rel
        };
        if src >= shuffled.len() {
            return Err(decode_error("native shuffled block range exceeds buffer"));
        }
        output[dst] = shuffled[src];
        dst += 1;
    }
    Ok(())
}

fn copy_shuffled_range_2(
    elements: usize,
    shuffled: &[u8],
    elem_start: usize,
    elem_end: usize,
    output: &mut [u8],
    dst: usize,
) {
    // The aligned fast path (caller guarantees `elem_end <= elements` and
    // `dst + 2*count <= output.len()`) interleaves the two byte planes with
    // unaligned u16 stores. This matches the per-element work in `unshuffle_2`
    // but is restricted to the requested element range, so a low-coverage
    // random request no longer pays for a full-block unshuffle. The contiguous
    // u16 store stream lets the compiler auto-vectorize, unlike the previous
    // per-byte `output[dst]` / `output[dst+1]` stores.
    #[cfg(target_endian = "little")]
    unsafe {
        let s0 = shuffled.as_ptr();
        let s1 = s0.add(elements);
        let out = output.as_mut_ptr().add(dst).cast::<u16>();
        for idx in 0..(elem_end - elem_start) {
            let elem = elem_start + idx;
            let value = (*s0.add(elem) as u16) | ((*s1.add(elem) as u16) << 8);
            std::ptr::write_unaligned(out.add(idx), value);
        }
        return;
    }
    #[cfg(not(target_endian = "little"))]
    {
        let plane1 = elements;
        let mut dst = dst;
        for elem in elem_start..elem_end {
            output[dst] = shuffled[elem];
            output[dst + 1] = shuffled[plane1 + elem];
            dst += 2;
        }
    }
}

fn copy_shuffled_range_4(
    elements: usize,
    shuffled: &[u8],
    elem_start: usize,
    elem_end: usize,
    output: &mut [u8],
    dst: usize,
) {
    #[cfg(target_endian = "little")]
    unsafe {
        let s0 = shuffled.as_ptr();
        let s1 = s0.add(elements);
        let s2 = s1.add(elements);
        let s3 = s2.add(elements);
        let out = output.as_mut_ptr().add(dst).cast::<u32>();
        for idx in 0..(elem_end - elem_start) {
            let elem = elem_start + idx;
            let value = (*s0.add(elem) as u32)
                | ((*s1.add(elem) as u32) << 8)
                | ((*s2.add(elem) as u32) << 16)
                | ((*s3.add(elem) as u32) << 24);
            std::ptr::write_unaligned(out.add(idx), value);
        }
        return;
    }
    #[cfg(not(target_endian = "little"))]
    {
        let plane1 = elements;
        let plane2 = elements * 2;
        let plane3 = elements * 3;
        let mut dst = dst;
        for elem in elem_start..elem_end {
            output[dst] = shuffled[elem];
            output[dst + 1] = shuffled[plane1 + elem];
            output[dst + 2] = shuffled[plane2 + elem];
            output[dst + 3] = shuffled[plane3 + elem];
            dst += 4;
        }
    }
}

fn copy_shuffled_range_generic(
    typesize: usize,
    elements: usize,
    shuffled: &[u8],
    elem_start: usize,
    elem_end: usize,
    output: &mut [u8],
    dst: usize,
) {
    // typesize == 8 is the only common case not handled by the 2/4 specialists
    // (typesize 1 is routed around `scatter_shuffled_block_ranges` entirely).
    // Give it the same unaligned u64 word-store treatment.
    if typesize == 8 {
        #[cfg(target_endian = "little")]
        unsafe {
            let s0 = shuffled.as_ptr();
            let s1 = s0.add(elements);
            let s2 = s1.add(elements);
            let s3 = s2.add(elements);
            let s4 = s3.add(elements);
            let s5 = s4.add(elements);
            let s6 = s5.add(elements);
            let s7 = s6.add(elements);
            let out = output.as_mut_ptr().add(dst).cast::<u64>();
            for idx in 0..(elem_end - elem_start) {
                let elem = elem_start + idx;
                let value = (*s0.add(elem) as u64)
                    | ((*s1.add(elem) as u64) << 8)
                    | ((*s2.add(elem) as u64) << 16)
                    | ((*s3.add(elem) as u64) << 24)
                    | ((*s4.add(elem) as u64) << 32)
                    | ((*s5.add(elem) as u64) << 40)
                    | ((*s6.add(elem) as u64) << 48)
                    | ((*s7.add(elem) as u64) << 56);
                std::ptr::write_unaligned(out.add(idx), value);
            }
            return;
        }
    }
    let mut dst = dst;
    for elem in elem_start..elem_end {
        for byte in 0..typesize {
            output[dst + byte] = shuffled[byte * elements + elem];
        }
        dst += typesize;
    }
}

fn decode_error(message: impl Into<String>) -> CodecError {
    CodecError::Decode {
        codec: "blosc".to_string(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::FileRef;
    use crate::databank::native::metadata::build_blosc_lz4_block_index;
    use std::ffi::CString;
    use std::os::raw::c_void;

    #[test]
    fn scatters_one_loaded_block_to_multiple_consumers() {
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let index = build_blosc_lz4_block_index(&encoded)
            .expect("valid index")
            .expect("Blosc LZ4 index");
        let block = index.blocks[0];
        let loaded = &encoded
            [block.payload_relative_offset..block.payload_relative_offset + block.compressed_len];
        let mut output = vec![0u8; 8];
        let mut scratch = NativeBlockScratch::default();

        scatter_loaded_blosc_block(
            &index,
            0,
            loaded,
            &[
                NativeBlockConsumer {
                    decoded_start: 1,
                    decoded_end: 3,
                    output_offset: 0,
                },
                NativeBlockConsumer {
                    decoded_start: 2,
                    decoded_end: 4,
                    output_offset: 4,
                },
            ],
            &mut output,
            &mut scratch,
        )
        .expect("scatter");

        assert_eq!(&output, b"bc\0\0cd\0\0");
    }

    #[test]
    fn scatters_memcpyed_loaded_range_without_decode() {
        let encoded = manual_blosc_lz4_memcpyed(b"abcdefgh");
        let index = build_blosc_lz4_block_index(&encoded)
            .expect("valid index")
            .expect("Blosc LZ4 index");
        let block = index.blocks[0];
        let loaded = &encoded
            [block.payload_relative_offset..block.payload_relative_offset + block.compressed_len];
        let mut output = vec![0u8; 4];
        let mut scratch = NativeBlockScratch::default();

        scatter_loaded_blosc_block(
            &index,
            0,
            loaded,
            &[NativeBlockConsumer {
                decoded_start: 2,
                decoded_end: 6,
                output_offset: 0,
            }],
            &mut output,
            &mut scratch,
        )
        .expect("scatter");

        assert_eq!(&output, b"cdef");
    }

    #[test]
    fn scatters_byte_shuffled_loaded_block() {
        let raw = (0..4096)
            .flat_map(|value: u32| value.to_le_bytes())
            .collect::<Vec<_>>();
        let encoded = blosc_lz4_encode(&raw, blosc_src::BLOSC_SHUFFLE as i32, 4, 1024);
        let index = build_blosc_lz4_block_index(&encoded)
            .expect("valid index")
            .expect("Blosc LZ4 index");
        assert!(index.byte_shuffled);
        let block = index.blocks[0];
        let loaded = &encoded
            [block.payload_relative_offset..block.payload_relative_offset + block.compressed_len];
        let mut output = vec![0u8; 64];
        let mut scratch = NativeBlockScratch::default();

        scatter_loaded_blosc_block(
            &index,
            block.block_idx,
            loaded,
            &[NativeBlockConsumer {
                decoded_start: 16,
                decoded_end: 80,
                output_offset: 0,
            }],
            &mut output,
            &mut scratch,
        )
        .expect("scatter");

        assert_eq!(&output, &raw[16..80]);
    }

    #[test]
    fn decoded_cache_scatters_repeated_block() {
        let raw = (0..4096)
            .flat_map(|value: u32| value.to_le_bytes())
            .collect::<Vec<_>>();
        let encoded = blosc_lz4_encode(&raw, blosc_src::BLOSC_SHUFFLE as i32, 4, 1024);
        let index = build_blosc_lz4_block_index(&encoded)
            .expect("valid index")
            .expect("Blosc LZ4 index");
        let block = index.blocks[0];
        let loaded = &encoded
            [block.payload_relative_offset..block.payload_relative_offset + block.compressed_len];
        let key = NativeBlockCacheKey {
            file: FileRef::new(17),
            offset: 1024,
            len: loaded.len(),
        };
        let cache = NativeBlockDecodedCache::new(1024 * 1024);
        let mut scratch = NativeBlockScratch::default();
        let mut first = vec![0u8; 64];
        let mut second = vec![0u8; 64];

        scatter_loaded_blosc_block_cached(
            &index,
            block.block_idx,
            Some(key),
            loaded,
            &[NativeBlockConsumer {
                decoded_start: 16,
                decoded_end: 80,
                output_offset: 0,
            }],
            &mut first,
            &mut scratch,
            Some(&cache),
        )
        .expect("first scatter");
        scatter_loaded_blosc_block_cached(
            &index,
            block.block_idx,
            Some(key),
            loaded,
            &[NativeBlockConsumer {
                decoded_start: 16,
                decoded_end: 80,
                output_offset: 0,
            }],
            &mut second,
            &mut scratch,
            Some(&cache),
        )
        .expect("second scatter");

        assert_eq!(&first, &raw[16..80]);
        assert_eq!(first, second);
    }

    fn manual_blosc_lz4_raw_blocks(blocks: &[&[u8]]) -> Vec<u8> {
        assert!(!blocks.is_empty());
        let blocksize = blocks[0].len();
        assert!(blocks.iter().all(|block| block.len() == blocksize));
        let decoded_size = blocks.iter().map(|block| block.len()).sum::<usize>();
        let table_bytes = blocks.len() * 4;
        let compressed_size =
            blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + table_bytes + decoded_size + table_bytes;
        let mut encoded = vec![0u8; blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + table_bytes];
        encoded[0] = blosc_src::BLOSC_VERSION_FORMAT as u8;
        encoded[1] = blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8;
        encoded[2] = (blosc_src::BLOSC_LZ4_FORMAT << 5) as u8;
        encoded[3] = 1;
        encoded[4..8].copy_from_slice(&(decoded_size as u32).to_le_bytes());
        encoded[8..12].copy_from_slice(&(blocksize as u32).to_le_bytes());
        encoded[12..16].copy_from_slice(&(compressed_size as u32).to_le_bytes());

        let mut offset = blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + table_bytes;
        for (idx, block) in blocks.iter().enumerate() {
            let table_offset = blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + idx * 4;
            encoded[table_offset..table_offset + 4].copy_from_slice(&(offset as i32).to_le_bytes());
            encoded.extend_from_slice(&(block.len() as i32).to_le_bytes());
            encoded.extend_from_slice(block);
            offset += 4 + block.len();
        }
        assert_eq!(encoded.len(), compressed_size);
        encoded
    }

    fn manual_blosc_lz4_memcpyed(raw: &[u8]) -> Vec<u8> {
        let compressed_size = blosc_src::BLOSC_MAX_OVERHEAD as usize + raw.len();
        let mut encoded = vec![0u8; blosc_src::BLOSC_MAX_OVERHEAD as usize];
        encoded[0] = blosc_src::BLOSC_VERSION_FORMAT as u8;
        encoded[1] = blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8;
        encoded[2] = (blosc_src::BLOSC_LZ4_FORMAT << 5) as u8 | blosc_src::BLOSC_MEMCPYED as u8;
        encoded[3] = 1;
        encoded[4..8].copy_from_slice(&(raw.len() as u32).to_le_bytes());
        encoded[8..12].copy_from_slice(&(raw.len() as u32).to_le_bytes());
        encoded[12..16].copy_from_slice(&(compressed_size as u32).to_le_bytes());
        encoded.extend_from_slice(raw);
        encoded
    }

    fn blosc_lz4_encode(raw: &[u8], shuffle: i32, typesize: usize, blocksize: usize) -> Vec<u8> {
        let compressor = CString::new("lz4").expect("static compressor name");
        let mut encoded = vec![0u8; raw.len() + blosc_src::BLOSC_MAX_OVERHEAD as usize];
        let written = unsafe {
            blosc_src::blosc_compress_ctx(
                5,
                shuffle,
                typesize,
                raw.len(),
                raw.as_ptr().cast::<c_void>(),
                encoded.as_mut_ptr().cast::<c_void>(),
                encoded.len(),
                compressor.as_ptr(),
                blocksize,
                1,
            )
        };
        assert!(written > 0, "Blosc compression failed with {written}");
        encoded.truncate(written as usize);
        encoded
    }
}
