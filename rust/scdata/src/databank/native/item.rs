use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::sync::Arc;

use crate::access::{AccessError, AccessItem, IoBackend, SliceSpec};
use crate::codecs::{
    blosc_lz4_header_table_len_from_prefix, try_blosc_lz4_plan_from_prefix, CodecError,
};
use crate::databank::config::NativeLoadCoalesceConfig;

use super::executor::{
    scatter_loaded_blosc_block_cached, scatter_loaded_blosc_block_multi_output_cached,
    NativeBlockDecodedCache, NativeBlockOutputConsumer, NativeBlockScratch,
};
use super::load::{NativeBlockPayloadCache, NativeLoadCompletion, NativeLoadModule};
use super::metadata::{index_from_plan, NativeBlockIndexCache, NativeBloscBlockIndex};
use super::planner::{plan_blosc_slice_reads, NativeSliceBlockPlan};

const BLOCK_REQUEST_ID_BASE: u64 = 2;

thread_local! {
    /// Reusable decode scratch buffer for the native scatter path.
    ///
    /// The scheduled native worker runs on a single-threaded `current_thread`
    /// runtime, so one scratch per worker thread is enough to amortize the
    /// `shuffled`/`decoded` Vec allocations across items — a 192 KiB block
    /// would otherwise allocate and free on every item. The borrow is held
    /// only inside the synchronous scatter loop (no `.await` while held), so
    /// the `RefCell` cannot be borrowed across a suspension point.
    static NATIVE_SCRATCH: RefCell<NativeBlockScratch> = RefCell::new(NativeBlockScratch::default());
}

pub(crate) async fn load_access_item_blosc_lz4_native(
    io: Arc<dyn IoBackend>,
    coalesce: NativeLoadCoalesceConfig,
    index_cache: &NativeBlockIndexCache,
    block_cache: Option<Arc<NativeBlockPayloadCache>>,
    decoded_cache: Option<Arc<NativeBlockDecodedCache>>,
    item: &AccessItem,
    priority: u8,
) -> Result<Option<Vec<u8>>, AccessError> {
    // The dispatch layer guarantees every item reaching the native path is
    // blosc (`Dataset::is_blosc_codec` precondition at spawn time), so the
    // `codec.name() != "blosc"` guard is gone. `None` still covers the
    // remaining decline paths (short chunk, non-lz4 header, unsupported block
    // table); `load_native_batch` turns those into `io::Error` fail-fast.
    let loader = NativeLoadModule::with_block_cache(io, coalesce, block_cache);
    if item.slice == SliceSpec::Full {
        return load_full_blosc_item(&loader, item, priority)
            .await
            .map(Some);
    }
    if item.key.len < blosc_src::BLOSC_MIN_HEADER_LENGTH as usize {
        return Ok(None);
    }

    let Some(index) = index_cache
        .get_or_insert_with(item.key, || build_block_index(&loader, item.key, priority))
        .await?
    else {
        return Ok(None);
    };
    validate_expected_size(index.decoded_size, item.expected_size)?;

    let Some(slice_plan) = plan_blosc_slice_reads(
        &index,
        item.key,
        &item.slice,
        priority,
        BLOCK_REQUEST_ID_BASE,
    )?
    else {
        return load_full_blosc_item(&loader, item, priority)
            .await
            .map(Some);
    };
    if slice_plan.output_len == 0 || slice_plan.reads.is_empty() {
        return Ok(Some(vec![0u8; slice_plan.output_len]));
    }

    let requests = slice_plan
        .reads
        .iter()
        .map(|read| read.request)
        .collect::<Vec<_>>();
    let completions = loader.load(&requests).await?;

    // `NativeLoadModule::load` returns completions in request order (it drains
    // its internal index by iterating `requests`), so `completions[i]` lines
    // up with `slice_plan.reads[i]` — no per-id remap needed.
    if completions.len() != slice_plan.reads.len() {
        return Err(AccessError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "native load completion count mismatch",
        )));
    }

    let mut output = output_buffer_for_slice_plan(&slice_plan);
    NATIVE_SCRATCH.with(|cell| -> Result<(), AccessError> {
        let mut scratch = cell.borrow_mut();
        for (read, completion) in slice_plan.reads.iter().zip(completions) {
            let loaded = completion_slice(&completion)?;
            scatter_loaded_blosc_block_cached(
                &index,
                read.block_idx,
                Some(super::load::NativeBlockCacheKey::from_request(read.request)),
                loaded,
                &read.consumers,
                &mut output,
                &mut scratch,
                decoded_cache.as_deref(),
            )?;
        }
        Ok(())
    })?;
    Ok(Some(output))
}

pub(crate) async fn load_access_items_blosc_lz4_native(
    io: Arc<dyn IoBackend>,
    coalesce: NativeLoadCoalesceConfig,
    index_cache: &NativeBlockIndexCache,
    block_cache: Option<Arc<NativeBlockPayloadCache>>,
    decoded_cache: Option<Arc<NativeBlockDecodedCache>>,
    items: &[AccessItem],
    priority: u8,
) -> Result<Vec<Option<Vec<u8>>>, AccessError> {
    if items.len() <= 1 {
        let mut results = Vec::with_capacity(items.len());
        for item in items {
            results.push(
                load_access_item_blosc_lz4_native(
                    Arc::clone(&io),
                    coalesce.clone(),
                    index_cache,
                    block_cache.clone(),
                    decoded_cache.clone(),
                    item,
                    priority,
                )
                .await?,
            );
        }
        return Ok(results);
    }

    let batch_coalesce = cross_item_coalesce_config(coalesce.clone());
    let loader = NativeLoadModule::with_block_cache(Arc::clone(&io), coalesce, block_cache.clone());
    let mut results = (0..items.len()).map(|_| None).collect::<Vec<_>>();
    let mut planned_outputs = (0..items.len()).map(|_| None).collect::<Vec<_>>();
    let mut block_jobs = Vec::new();
    let mut block_job_by_range = HashMap::new();
    let mut full_jobs = Vec::new();
    let mut full_job_by_range = HashMap::new();
    let mut read_jobs = Vec::new();
    let mut requests = Vec::new();

    for (item_idx, item) in items.iter().enumerate() {
        // The blosc guard is gone: every item on the native path is blosc by
        // the spawn-time precondition. Decline paths below (short chunk,
        // non-lz4 header, unsupported block table) leave the slot as `None`,
        // which `load_native_batch` converts to an `io::Error`.
        if item.slice == SliceSpec::Full {
            append_full_chunk_job(
                item_idx,
                item,
                priority,
                &mut requests,
                &mut read_jobs,
                &mut full_jobs,
                &mut full_job_by_range,
            )?;
            continue;
        }
        if item.key.len < blosc_src::BLOSC_MIN_HEADER_LENGTH as usize {
            continue;
        }

        let Some(index) = index_cache
            .get_or_insert_with(item.key, || build_block_index(&loader, item.key, priority))
            .await?
        else {
            continue;
        };
        validate_expected_size(index.decoded_size, item.expected_size)?;

        let Some(slice_plan) = plan_blosc_slice_reads(&index, item.key, &item.slice, priority, 0)?
        else {
            append_full_chunk_job(
                item_idx,
                item,
                priority,
                &mut requests,
                &mut read_jobs,
                &mut full_jobs,
                &mut full_job_by_range,
            )?;
            continue;
        };

        if slice_plan.output_len == 0 || slice_plan.reads.is_empty() {
            results[item_idx] = Some(vec![0u8; slice_plan.output_len]);
            continue;
        }

        planned_outputs[item_idx] = Some(output_buffer_for_slice_plan(&slice_plan));
        for read in slice_plan.reads {
            let key = (read.request.file, read.request.offset, read.request.len);
            let job_index = if let Some(&job_index) = block_job_by_range.get(&key) {
                job_index
            } else {
                let request_id = next_native_request_id(requests.len())?;
                let mut request = read.request;
                request.id = request_id;
                requests.push(request);
                read_jobs.push(PlannedNativeRead::Block(block_jobs.len()));
                let job_index = block_jobs.len();
                block_job_by_range.insert(key, job_index);
                block_jobs.push(PlannedNativeBlock {
                    index: Arc::clone(&index),
                    block_idx: read.block_idx,
                    request,
                    targets: Vec::new(),
                });
                job_index
            };
            let job = &mut block_jobs[job_index];
            debug_assert_eq!(job.block_idx, read.block_idx);
            for consumer in read.consumers {
                job.targets.push(NativeBlockOutputConsumer {
                    output_index: item_idx,
                    consumer,
                });
            }
        }
    }

    if requests.is_empty() {
        return Ok(results);
    }

    let batch_loader = NativeLoadModule::with_block_cache(io, batch_coalesce, block_cache);
    let completions = batch_loader.load_unsorted(&requests).await?;
    if completions.len() != requests.len() {
        return Err(AccessError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "native batch load completion count mismatch",
        )));
    }

    NATIVE_SCRATCH.with(|cell| -> Result<(), AccessError> {
        let mut scratch = cell.borrow_mut();
        for (read_job, completion) in read_jobs.iter().zip(&completions) {
            match *read_job {
                PlannedNativeRead::Block(block_job_idx) => {
                    let block_job = block_jobs.get(block_job_idx).ok_or_else(|| {
                        AccessError::Io(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "native batch block job index out of bounds",
                        ))
                    })?;
                    let loaded = completion_slice(completion)?;
                    scatter_loaded_blosc_block_multi_output_cached(
                        &block_job.index,
                        block_job.block_idx,
                        Some(super::load::NativeBlockCacheKey::from_request(
                            block_job.request,
                        )),
                        loaded,
                        &block_job.targets,
                        &mut planned_outputs,
                        &mut scratch,
                        decoded_cache.as_deref(),
                    )?;
                }
                PlannedNativeRead::Full(full_job_idx) => {
                    let full_job = full_jobs.get(full_job_idx).ok_or_else(|| {
                        AccessError::Io(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "native batch full job index out of bounds",
                        ))
                    })?;
                    let loaded = completion_slice(completion)?;
                    for &item_idx in &full_job.output_indices {
                        let item = items.get(item_idx).ok_or_else(|| {
                            AccessError::Io(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "native batch full output index out of bounds",
                            ))
                        })?;
                        results[item_idx] = Some(
                            item.codec
                                .decode(loaded, item.expected_size)
                                .map_err(AccessError::from)?,
                        );
                    }
                }
            }
        }
        Ok(())
    })?;

    for (item_idx, output) in planned_outputs.into_iter().enumerate() {
        if output.is_some() {
            results[item_idx] = output;
        }
    }
    Ok(results)
}

async fn load_full_blosc_item(
    loader: &NativeLoadModule,
    item: &AccessItem,
    priority: u8,
) -> Result<Vec<u8>, AccessError> {
    let encoded = loader
        .load_single(item.key.file, item.key.offset, item.key.len, priority)
        .await?;
    item.codec
        .decode(&encoded, item.expected_size)
        .map_err(Into::into)
}

#[derive(Debug, Clone, Copy)]
enum PlannedNativeRead {
    Block(usize),
    Full(usize),
}

struct PlannedNativeBlock {
    index: Arc<NativeBloscBlockIndex>,
    block_idx: usize,
    request: super::load::NativeLoadRequest,
    targets: Vec<NativeBlockOutputConsumer>,
}

struct PlannedFullChunk {
    output_indices: Vec<usize>,
}

fn append_full_chunk_job(
    item_idx: usize,
    item: &AccessItem,
    priority: u8,
    requests: &mut Vec<super::load::NativeLoadRequest>,
    read_jobs: &mut Vec<PlannedNativeRead>,
    full_jobs: &mut Vec<PlannedFullChunk>,
    full_job_by_range: &mut HashMap<(crate::access::FileRef, u64, usize), usize>,
) -> Result<(), AccessError> {
    let key = (item.key.file, item.key.offset, item.key.len);
    let job_index = if let Some(&job_index) = full_job_by_range.get(&key) {
        job_index
    } else {
        let request_id = next_native_request_id(requests.len())?;
        requests.push(super::load::NativeLoadRequest {
            id: request_id,
            file: item.key.file,
            offset: item.key.offset,
            len: item.key.len,
            priority,
        });
        read_jobs.push(PlannedNativeRead::Full(full_jobs.len()));
        let job_index = full_jobs.len();
        full_job_by_range.insert(key, job_index);
        full_jobs.push(PlannedFullChunk {
            output_indices: Vec::new(),
        });
        job_index
    };
    full_jobs[job_index].output_indices.push(item_idx);
    Ok(())
}

fn next_native_request_id(request_count: usize) -> Result<u64, AccessError> {
    BLOCK_REQUEST_ID_BASE
        .checked_add(request_count as u64)
        .ok_or_else(|| AccessError::InvalidSlice("native request id overflow".to_string()))
}

fn output_buffer_for_slice_plan(slice_plan: &NativeSliceBlockPlan) -> Vec<u8> {
    if slice_plan.output_fully_covered {
        uninit_u8_vec(slice_plan.output_len)
    } else {
        vec![0u8; slice_plan.output_len]
    }
}

fn uninit_u8_vec(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    // SAFETY: callers use this only for `NativeSliceBlockPlan::output_fully_covered`,
    // which means the slice ranges cover every output byte exactly before the Vec
    // is returned to consumers. Dropping a partially written Vec<u8> on an error is
    // also safe because `u8` has no drop glue.
    unsafe { out.set_len(len) };
    out
}

fn cross_item_coalesce_config(config: NativeLoadCoalesceConfig) -> NativeLoadCoalesceConfig {
    config
}

/// Read the Blosc header + block offset table and build the validated block
/// index. Used as the cache-miss path of `NativeBlockIndexCache`.
///
/// Returns `Ok(None)` when the chunk is a Blosc variant the native path does
/// not support; the caller should fall back to the generic decode path.
///
/// To avoid a serial two-step (read 16 B header, then read the full table),
/// we issue one optimistic read of `HEADER_TABLE_READ_HINT` bytes. The Blosc
/// header alone determines `table_len = 16 + nblocks * 4`, so when the hint
/// already covers the table — the common case, ~124 blocks at 512 B — a single
/// IO builds the index. Larger tables fall back to a second exact-sized read.
async fn build_block_index(
    loader: &NativeLoadModule,
    key: crate::access::ChunkKey,
    priority: u8,
) -> Result<Option<Arc<NativeBloscBlockIndex>>, AccessError> {
    /// Optimistic header+table prefetch size. Covers `nblocks` up to
    /// `(HINT - 16) / 4`; 512 B ⇒ 124 blocks, enough for the vast majority of
    /// Blosc chunks produced by scRNA encoders. Exceeding this is fine — it
    /// just costs one extra exact-sized read.
    const HEADER_TABLE_READ_HINT: usize = 512;

    let hint_len = HEADER_TABLE_READ_HINT.min(key.len);
    let prefix = loader
        .load_single(key.file, key.offset, hint_len, priority)
        .await?;

    // The first `BLOSC_MIN_HEADER_LENGTH` bytes are the header; from it we
    // derive the exact table length without a second IO round-trip.
    let header_bytes = prefix
        .get(..blosc_src::BLOSC_MIN_HEADER_LENGTH as usize)
        .ok_or_else(|| {
            AccessError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "native Blosc prefix shorter than minimal header",
            ))
        })?;
    let Some(table_len) = blosc_lz4_header_table_len_from_prefix("blosc", header_bytes)? else {
        return Ok(None);
    };
    if table_len > key.len {
        return Err(AccessError::Codec(CodecError::Decode {
            codec: "blosc".to_string(),
            message: "native Blosc block table exceeds chunk length".to_string(),
        }));
    }

    // Fast path: the optimistic prefix already covers the whole table, so we
    // parse directly from it — no second IO, no copy. Only large block tables
    // (nblocks > ~124 at the 512 B hint) need a second exact-sized read.
    let plan = if prefix.len() >= table_len {
        match try_blosc_lz4_plan_from_prefix("blosc", &prefix[..table_len])? {
            Some(plan) => plan,
            None => return Ok(None),
        }
    } else {
        let header_table = loader
            .load_single(key.file, key.offset, table_len, priority)
            .await?;
        match try_blosc_lz4_plan_from_prefix("blosc", &header_table[..table_len])? {
            Some(plan) => plan,
            None => return Ok(None),
        }
    };
    Ok(Some(Arc::new(index_from_plan(plan))))
}

fn validate_expected_size(
    decoded_size: usize,
    expected_size: Option<usize>,
) -> Result<(), AccessError> {
    if let Some(expected) = expected_size {
        if expected != decoded_size {
            return Err(AccessError::Codec(CodecError::SizeMismatch {
                codec: "blosc".to_string(),
                expected,
                actual: decoded_size,
            }));
        }
    }
    Ok(())
}

fn completion_slice(completion: &NativeLoadCompletion) -> Result<&[u8], AccessError> {
    // `NativeLoadModule::load` validates each child range against the coalesced
    // buffer before constructing the completion, so the range is in-bounds by
    // construction — direct indexing skips the `Option` and the error-construction
    // allocation of `.get().ok_or_else(...)` on every block.
    debug_assert!(
        completion.range.start <= completion.range.end
            && completion.range.end <= completion.bytes.len(),
        "native load completion range exceeds buffer",
    );
    Ok(&completion.bytes[completion.range.clone()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{ChunkKey, FileRef, IoTask};
    use crate::codecs::codec_from_json_str;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct RangeIo {
        file: FileRef,
        base_offset: u64,
        bytes: Arc<[u8]>,
        reads: Mutex<Vec<(u64, usize)>>,
    }

    impl RangeIo {
        fn new(file: FileRef, base_offset: u64, bytes: Vec<u8>) -> Self {
            Self {
                file,
                base_offset,
                bytes: Arc::from(bytes.into_boxed_slice()),
                reads: Mutex::new(Vec::new()),
            }
        }

        fn reads(&self) -> Vec<(u64, usize)> {
            self.reads.lock().expect("reads lock").clone()
        }
    }

    impl IoBackend for RangeIo {
        fn submit_read(&self, file: FileRef, offset: u64, len: usize, _priority: u8) -> IoTask {
            assert_eq!(file, self.file);
            self.reads.lock().expect("reads lock").push((offset, len));
            let start = usize::try_from(offset - self.base_offset).expect("offset");
            let end = start + len;
            let data: Arc<[u8]> = Arc::from(self.bytes[start..end].to_vec().into_boxed_slice());
            Box::pin(async move { Ok(data) })
        }
    }

    #[tokio::test]
    async fn loads_partial_blosc_lz4_blocks_and_scatters_output() {
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let base_offset = 1000;
        let io = Arc::new(RangeIo::new(file, base_offset, encoded.clone()));
        let item = AccessItem::new(
            ChunkKey::new(file, base_offset, encoded.len()),
            codec_from_json_str(r#"{"id":"blosc","cname":"lz4"}"#).expect("codec"),
            Some(8),
        )
        .with_slice_spec(SliceSpec::from_triples(vec![0, 1, 4, 3, 5, 8]).expect("slice"));

        let out = load_access_item_blosc_lz4_native(
            io.clone(),
            coalesce_config(),
            &index_cache(),
            None,
            None,
            &item,
            0,
        )
        .await
        .expect("native load")
        .expect("native result");

        assert_eq!(&out, b"bcdfgh");
        // One optimistic header+table read (covers the whole 40 B chunk here),
        // then one coalesced read for both blocks (gap 0 ⇒ merged to 16 B).
        assert_eq!(io.reads(), vec![(1000, 40), (1024, 16)]);
    }

    #[tokio::test]
    async fn returns_none_for_full_slice() {
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let base_offset = 1000;
        let io = Arc::new(RangeIo::new(file, base_offset, encoded.clone()));
        let item = AccessItem::new(
            ChunkKey::new(file, base_offset, encoded.len()),
            codec_from_json_str(r#"{"id":"blosc","cname":"lz4"}"#).expect("codec"),
            Some(8),
        );

        let out = load_access_item_blosc_lz4_native(
            io,
            coalesce_config(),
            &index_cache(),
            None,
            None,
            &item,
            0,
        )
        .await
        .expect("native load")
        .expect("native full result");

        assert_eq!(&out, b"abcdefgh");
    }

    #[tokio::test]
    async fn reuses_cached_block_index_across_items() {
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let base_offset = 1000;
        let io = Arc::new(RangeIo::new(file, base_offset, encoded.clone()));
        let cache = NativeBlockIndexCache::new();

        let item = |slice: SliceSpec| {
            AccessItem::new(
                ChunkKey::new(file, base_offset, encoded.len()),
                codec_from_json_str(r#"{"id":"blosc","cname":"lz4"}"#).expect("codec"),
                Some(8),
            )
            .with_slice_spec(slice)
        };

        let first = load_access_item_blosc_lz4_native(
            io.clone(),
            coalesce_config(),
            &cache,
            None,
            None,
            &item(SliceSpec::from_triples(vec![0, 0, 2]).expect("slice")),
            0,
        )
        .await
        .expect("native load")
        .expect("native result");
        let reads_after_first = io.reads();

        let second = load_access_item_blosc_lz4_native(
            io.clone(),
            coalesce_config(),
            &cache,
            None,
            None,
            &item(SliceSpec::from_triples(vec![0, 4, 6]).expect("slice")),
            0,
        )
        .await
        .expect("native load")
        .expect("native result");
        let reads_after_second = io.reads();

        // Second call hits the cache: no header/table reads, only the block.
        assert_eq!(&first, b"ab");
        assert_eq!(&second, b"ef");
        // First call: one optimistic 40 B read (header+table, covers the whole
        // chunk) + the block 0 payload. Second call: block 1 payload only.
        assert_eq!(reads_after_first, vec![(1000, 40), (1024, 8)]);
        assert_eq!(
            reads_after_second[reads_after_first.len()..],
            vec![(1032, 8)]
        );
    }

    fn index_cache() -> NativeBlockIndexCache {
        NativeBlockIndexCache::new()
    }

    fn coalesce_config() -> NativeLoadCoalesceConfig {
        NativeLoadCoalesceConfig {
            max_window_us: 0,
            max_merged_len: 1024,
            max_gap_bytes: 16,
            max_waste_ratio: 0.5,
            min_children: 2,
        }
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

    /// `IoBackend` that always fails — exercises the zero-fallback error origin
    /// (`load_access_items_blosc_lz4_native` returns `Err`, never a silent `None`
    /// or a retreat to the generic path).
    struct ErrorIo;

    impl IoBackend for ErrorIo {
        fn submit_read(&self, _file: FileRef, _offset: u64, _len: usize, _priority: u8) -> IoTask {
            Box::pin(async move { Err(io::Error::new(io::ErrorKind::UnexpectedEof, "boom")) })
        }
    }

    fn blosc_codec() -> crate::codecs::SharedCodec {
        codec_from_json_str(r#"{"id":"blosc","cname":"lz4"}"#).expect("codec")
    }

    #[tokio::test]
    async fn batch_loads_multiple_sliced_items_in_order() {
        // >1 items exercises the batch arm (cross-item coalesce + multi-output
        // scatter), not the single-item fallback. Two slices of one chunk come
        // back in submission order.
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let base_offset = 1000;
        let io = Arc::new(RangeIo::new(file, base_offset, encoded.clone()));
        let codec = blosc_codec();
        let item = |slice: SliceSpec| {
            AccessItem::new(
                ChunkKey::new(file, base_offset, encoded.len()),
                codec.clone(),
                Some(8),
            )
            .with_slice_spec(slice)
        };
        let items = vec![
            item(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice")), // "abcd"
            item(SliceSpec::from_triples(vec![0, 4, 8]).expect("slice")), // "efgh"
        ];

        let out = load_access_items_blosc_lz4_native(
            io,
            coalesce_config(),
            &index_cache(),
            None,
            None,
            &items,
            0,
        )
        .await
        .expect("native batch");

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].as_deref(), Some(b"abcd".as_slice()));
        assert_eq!(out[1].as_deref(), Some(b"efgh".as_slice()));
    }

    #[tokio::test]
    async fn batch_dedups_shared_block_reads_across_items() {
        // Both slices live in block 0 (decoded bytes 0..4 = "abcd"). The batch
        // planner keys block reads by `(file, offset, len)` in
        // `block_job_by_range`, so the second item reuses the first's block job
        // instead of re-reading the block payload.
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let base_offset = 1000;
        let io = Arc::new(RangeIo::new(file, base_offset, encoded.clone()));
        let codec = blosc_codec();
        let item = |slice: SliceSpec| {
            AccessItem::new(
                ChunkKey::new(file, base_offset, encoded.len()),
                codec.clone(),
                Some(8),
            )
            .with_slice_spec(slice)
        };
        let items = vec![
            item(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice")), // "abcd"
            item(SliceSpec::from_triples(vec![0, 0, 2]).expect("slice")), // "ab"
        ];

        let out = load_access_items_blosc_lz4_native(
            io.clone(),
            coalesce_config(),
            &index_cache(),
            None,
            None,
            &items,
            0,
        )
        .await
        .expect("native batch");

        assert_eq!(out[0].as_deref(), Some(b"abcd".as_slice()));
        assert_eq!(out[1].as_deref(), Some(b"ab".as_slice()));
        // One optimistic header+table read (covers the whole 40 B chunk) + one
        // block-0 payload read, shared across both items.
        assert_eq!(io.reads(), vec![(1000, 40), (1024, 8)]);
    }

    #[tokio::test]
    async fn batch_mixed_full_and_sliced_items() {
        // A `Full` item (whole-chunk decode) alongside a sliced item on the same
        // chunk: the full-chunk job and the block-slice job take different code
        // paths through the batch planner but must both land correct output.
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let base_offset = 1000;
        let io = Arc::new(RangeIo::new(file, base_offset, encoded.clone()));
        let codec = blosc_codec();
        let key = ChunkKey::new(file, base_offset, encoded.len());
        let items = vec![
            AccessItem::new(key, codec.clone(), Some(8)), // Full → "abcdefgh"
            AccessItem::new(key, codec.clone(), Some(8))
                .with_slice_spec(SliceSpec::from_triples(vec![0, 4, 8]).expect("slice")), // "efgh"
        ];

        let out = load_access_items_blosc_lz4_native(
            io,
            coalesce_config(),
            &index_cache(),
            None,
            None,
            &items,
            0,
        )
        .await
        .expect("native batch");

        assert_eq!(out[0].as_deref(), Some(b"abcdefgh".as_slice()));
        assert_eq!(out[1].as_deref(), Some(b"efgh".as_slice()));
    }

    #[tokio::test]
    async fn batch_leaves_none_for_short_chunk_with_slice() {
        // A sliced item whose chunk is shorter than the minimal Blosc header
        // (16 B) declines on the native path: the batch loop `continue`s, leaving
        // the slot as `None`. This `None` is the zero-fallback source —
        // `load_native_batch` turns it into an `io::Error` rather than retreating
        // to the generic path (see `native_access::tests`).
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let io = Arc::new(RangeIo::new(file, 1000, encoded.clone()));
        let codec = blosc_codec();
        let short = AccessItem::new(ChunkKey::new(file, 2000, 8), codec.clone(), Some(4))
            .with_slice_spec(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice"));
        let normal =
            AccessItem::new(ChunkKey::new(file, 1000, encoded.len()), codec.clone(), Some(8))
                .with_slice_spec(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice"));
        let items = vec![short, normal];

        let out = load_access_items_blosc_lz4_native(
            io,
            coalesce_config(),
            &index_cache(),
            None,
            None,
            &items,
            0,
        )
        .await
        .expect("decline is Ok(None), not Err");

        assert_eq!(out.len(), 2);
        assert!(out[0].is_none(), "short chunk with slice must decline to None");
        assert_eq!(out[1].as_deref(), Some(b"abcd".as_slice()));
    }

    #[tokio::test]
    async fn batch_propagates_io_error() {
        // An IO failure while building the block index surfaces as `Err` — never
        // a silent `None` or a generic-path retreat. This is the `Err` half of
        // the zero-fallback contract.
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let io = Arc::new(ErrorIo);
        let codec = blosc_codec();
        let item = AccessItem::new(ChunkKey::new(file, 1000, encoded.len()), codec, Some(8))
            .with_slice_spec(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice"));
        let items = vec![item.clone(), item];

        let err = load_access_items_blosc_lz4_native(
            io,
            coalesce_config(),
            &index_cache(),
            None,
            None,
            &items,
            0,
        )
        .await
        .expect_err("IO error must propagate, not decline to None");

        assert!(err.to_string().contains("boom"), "{err}");
    }

    #[tokio::test]
    async fn batch_rejects_expected_size_mismatch() {
        // `expected_size` is validated against the block-index decoded size
        // before any payload read. A mismatch is a hard `CodecError::SizeMismatch`
        // — not a decline-to-`None`.
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let file = FileRef::new(7);
        let io = Arc::new(RangeIo::new(file, 1000, encoded.clone()));
        let codec = blosc_codec();
        let item = AccessItem::new(ChunkKey::new(file, 1000, encoded.len()), codec, Some(99))
            .with_slice_spec(SliceSpec::from_triples(vec![0, 0, 4]).expect("slice"));
        let items = vec![item.clone(), item];

        let err = load_access_items_blosc_lz4_native(
            io,
            coalesce_config(),
            &index_cache(),
            None,
            None,
            &items,
            0,
        )
        .await
        .expect_err("size mismatch must error");

        let msg = err.to_string();
        assert!(msg.contains("size mismatch"), "{msg}");
        assert!(msg.contains("expected 99"), "{msg}");
    }
}
