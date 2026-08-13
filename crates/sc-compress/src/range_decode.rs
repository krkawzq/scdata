use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::sync::{Condvar, Mutex};

use dyn_blosc::{
    BloscVersion, DecodeLimits as BloscDecodeLimits, DecodeWorkspace, Decoder,
    Header as BloscHeader, Shuffle as BloscShuffle, HEADER_LEN,
};

use crate::codec::Compressor;
use crate::error::{Error, Result};
use crate::limits::ReadLimits;
use crate::parallel;
use crate::storage::{ByteStore, PositionedValue};

/// Per-decode admission control for encoded buffers owned by concurrent block
/// workers.
///
/// Final outputs, retained schemas, and reusable per-worker decode scratch are
/// charged as `resident_decoded` before workers start. Each active worker then
/// reserves the encoded payload for its current block.
pub(crate) struct DecodeBudget {
    resident_decoded: usize,
    maximum_decoded: usize,
    available: Mutex<usize>,
    wake: Condvar,
}

impl DecodeBudget {
    pub(crate) fn new(resident_decoded: usize, limits: ReadLimits) -> Result<Self> {
        limits.check_decoded(resident_decoded, "parallel decode resident output")?;
        let capacity = limits
            .decoded_size()
            .checked_sub(resident_decoded)
            .ok_or_else(|| Error::corrupt("parallel decode", "resident output exceeds limit"))?;
        Ok(Self {
            resident_decoded,
            maximum_decoded: limits.decoded_size(),
            available: Mutex::new(capacity),
            wake: Condvar::new(),
        })
    }

    pub(crate) fn acquire(&self, working_set_bound: usize) -> Result<DecodePermit<'_>> {
        let capacity = self.maximum_decoded - self.resident_decoded;
        let charge = working_set_bound.min(capacity);
        if charge == 0 {
            return Ok(DecodePermit {
                budget: self,
                charge,
                context_resident: self.resident_decoded,
            });
        }

        let mut available = self
            .available
            .lock()
            .map_err(|_| Error::corrupt("parallel decode", "budget mutex poisoned"))?;
        while *available < charge {
            available = self
                .wake
                .wait(available)
                .map_err(|_| Error::corrupt("parallel decode", "budget mutex poisoned"))?;
        }
        *available -= charge;
        drop(available);
        Ok(DecodePermit {
            budget: self,
            charge,
            // Assume every byte not reserved by this worker may be in use by
            // peers. The exact per-chunk checks can therefore spend at most
            // this worker's charge without exceeding the aggregate limit.
            context_resident: self.maximum_decoded - charge,
        })
    }
}

pub(crate) struct DecodePermit<'a> {
    budget: &'a DecodeBudget,
    charge: usize,
    context_resident: usize,
}

impl DecodePermit<'_> {
    pub(crate) const fn context_resident(&self) -> usize {
        self.context_resident
    }
}

impl Drop for DecodePermit<'_> {
    fn drop(&mut self) {
        if self.charge == 0 {
            return;
        }
        let mut available = self
            .budget
            .available
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *available = available.saturating_add(self.charge);
        drop(available);
        self.budget.wake.notify_all();
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RangeDecodeContext<'a> {
    store: &'a dyn ByteStore,
    resident_decoded: usize,
    limits: ReadLimits,
}

impl<'a> RangeDecodeContext<'a> {
    pub(crate) const fn new(
        store: &'a dyn ByteStore,
        resident_decoded: usize,
        limits: ReadLimits,
    ) -> Self {
        Self {
            store,
            resident_decoded,
            limits,
        }
    }
}

/// One decoded byte range and its final destination range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScatterMapping {
    pub(crate) source: Range<usize>,
    pub(crate) destination: Range<usize>,
}

/// All byte mappings read from one compressed chunk.
#[derive(Debug)]
pub(crate) struct BloscScatterRequest {
    pub(crate) key: String,
    pub(crate) expected: usize,
    pub(crate) mappings: Vec<ScatterMapping>,
}

struct PreparedChunk {
    key: String,
    decoder: Decoder,
    encoded_len: usize,
    positioned: Option<PositionedValue>,
    full_encoded: Option<Vec<u8>>,
    schema_resident: usize,
}

#[derive(Debug, Clone, Copy)]
struct ScatterPiece {
    source_in_block: usize,
    destination: usize,
    len: usize,
}

#[derive(Debug)]
struct BlockTask {
    chunk: usize,
    block: Option<usize>,
    pieces: Range<usize>,
    encoded_working_set: usize,
    decoded_working_set: usize,
}

struct DecodeWorker {
    decoded: Vec<u8>,
    workspace: DecodeWorkspace,
}

struct ScatterDestination {
    base: *mut u8,
    len: usize,
}

// SAFETY: the pointer remains valid for the complete scoped execution.
// Request validation proves that concurrently scheduled pieces write
// pairwise-disjoint in-bounds ranges, and the caller cannot access the
// destination again until every scoped worker has joined.
unsafe impl Send for ScatterDestination {}
// SAFETY: shared access only reads this descriptor. Dereferences are writes to
// the disjoint ranges established by `validate_scatter_requests`.
unsafe impl Sync for ScatterDestination {}

impl DecodeWorker {
    fn new() -> Self {
        Self {
            decoded: Vec::new(),
            workspace: DecodeWorkspace::new(),
        }
    }
}

/// Decode every selected block once and scatter its intersecting byte ranges.
///
/// Efficient range stores are planned together so one bounded worker pool can
/// schedule blocks across chunk boundaries. Stores that cannot serve physical
/// ranges are processed one chunk at a time because the complete encoded value
/// must remain resident while its selected blocks are decoded.
pub(crate) fn decode_blosc_scatter_into(
    compressor: &Compressor,
    context: RangeDecodeContext<'_>,
    requests: &[BloscScatterRequest],
    destination: &mut [u8],
) -> Result<()> {
    compressor.validate()?;
    validate_scatter_requests(requests, destination.len())?;
    if requests.is_empty() {
        return Ok(());
    }

    let mut all_efficient = true;
    for request in requests {
        all_efficient &= context.store.supports_efficient_range_reads(&request.key)?;
    }
    if all_efficient {
        let batch_size = context.limits.thread_count().max(1);
        for batch in requests.chunks(batch_size) {
            decode_efficient_batch(compressor, context, batch, destination)?;
        }
        return Ok(());
    }

    for request in requests {
        let prepared =
            prepare_scatter_chunks(compressor, context, std::slice::from_ref(request), true)?;
        run_scatter_tasks(
            context,
            std::slice::from_ref(request),
            &prepared,
            destination,
        )?;
    }
    Ok(())
}

fn decode_efficient_batch(
    compressor: &Compressor,
    context: RangeDecodeContext<'_>,
    requests: &[BloscScatterRequest],
    destination: &mut [u8],
) -> Result<()> {
    match prepare_scatter_chunks(compressor, context, requests, false) {
        Ok(prepared) => run_scatter_tasks(context, requests, &prepared, destination),
        Err(error) if requests.len() > 1 && is_decoded_limit_error(&error) => {
            let middle = requests.len() / 2;
            decode_efficient_batch(compressor, context, &requests[..middle], destination)?;
            decode_efficient_batch(compressor, context, &requests[middle..], destination)
        }
        Err(error) => Err(error),
    }
}

fn is_decoded_limit_error(error: &Error) -> bool {
    matches!(
        error,
        Error::CorruptData { message, .. }
            if message.starts_with("decoded size ")
                && message.contains(" exceeds configured limit ")
    )
}

fn prepare_scatter_chunks(
    compressor: &Compressor,
    context: RangeDecodeContext<'_>,
    requests: &[BloscScatterRequest],
    read_full: bool,
) -> Result<Vec<PreparedChunk>> {
    let mut prepared = Vec::new();
    prepared.try_reserve_exact(requests.len())?;
    let mut accumulated_schema = 0usize;
    for request in requests {
        let chunk =
            prepare_scatter_chunk(compressor, context, request, read_full, accumulated_schema)?;
        accumulated_schema = accumulated_schema
            .checked_add(chunk.schema_resident)
            .ok_or_else(|| Error::corrupt(&request.key, "schema resident size overflow"))?;
        prepared.push(chunk);
    }
    Ok(prepared)
}

fn prepare_scatter_chunk(
    compressor: &Compressor,
    context: RangeDecodeContext<'_>,
    request: &BloscScatterRequest,
    read_full: bool,
    accumulated_schema: usize,
) -> Result<PreparedChunk> {
    let mut positioned = if read_full {
        None
    } else {
        context.store.open_positioned(&request.key)?
    };
    let encoded_len = usize::try_from(if let Some(value) = &positioned {
        value.len()
    } else {
        context.store.len(&request.key)?
    })
    .map_err(|_| Error::corrupt(&request.key, "encoded size exceeds usize"))?;
    context.limits.check_encoded(encoded_len, &request.key)?;
    if encoded_len < HEADER_LEN {
        return Err(Error::corrupt(
            &request.key,
            format!("encoded chunk is shorter than the {HEADER_LEN}-byte header"),
        ));
    }

    let mut full_encoded = if read_full {
        context.limits.check_decoded_sum(
            [context.resident_decoded, accumulated_schema, encoded_len],
            "Blosc encoded scatter working set",
        )?;
        Some(
            context
                .store
                .read_limited(&request.key, context.limits.encoded_size())?,
        )
    } else {
        None
    };
    let header_storage = if full_encoded.is_none() {
        context.limits.check_decoded_sum(
            [context.resident_decoded, accumulated_schema, HEADER_LEN],
            "Blosc scatter header working set",
        )?;
        Some(read_exact_range(
            context.store,
            positioned.as_ref(),
            &request.key,
            0,
            HEADER_LEN,
        )?)
    } else {
        None
    };
    let header = full_encoded
        .as_deref()
        .and_then(|encoded| encoded.get(..HEADER_LEN))
        .or(header_storage.as_deref())
        .ok_or_else(|| Error::corrupt(&request.key, "missing encoded header"))?;
    let parsed_header = BloscHeader::parse(header)?;
    validate_scatter_header(
        compressor,
        &request.key,
        request.expected,
        encoded_len,
        parsed_header,
    )?;
    if parsed_header.block_count() > context.limits.block_count() {
        return Err(Error::corrupt(
            &request.key,
            format!(
                "block count {} exceeds configured limit {}",
                parsed_header.block_count(),
                context.limits.block_count()
            ),
        ));
    }
    let prefix_len = Decoder::index_prefix_len(header)?;
    if prefix_len > encoded_len {
        return Err(Error::corrupt(
            &request.key,
            format!("index prefix length {prefix_len} exceeds encoded size {encoded_len}"),
        ));
    }
    context.limits.check_encoded(prefix_len, &request.key)?;
    let decoder_index_resident = prefix_len
        .saturating_sub(HEADER_LEN)
        .checked_mul(2)
        .ok_or_else(|| Error::corrupt(&request.key, "decoder index working set overflow"))?;
    let prefix_storage = if full_encoded.is_none() && prefix_len > HEADER_LEN {
        context.limits.check_decoded_sum(
            [
                context.resident_decoded,
                accumulated_schema,
                HEADER_LEN,
                prefix_len,
                decoder_index_resident,
            ],
            "Blosc scatter schema working set",
        )?;
        Some(read_exact_range(
            context.store,
            positioned.as_ref(),
            &request.key,
            0,
            prefix_len,
        )?)
    } else {
        None
    };
    let prefix = full_encoded
        .as_deref()
        .and_then(|encoded| encoded.get(..prefix_len))
        .or(prefix_storage.as_deref())
        .unwrap_or(header);
    let decoder_limits = BloscDecodeLimits::unlimited()
        .maximum_decoded_size(request.expected)
        .maximum_block_size(request.expected)
        .maximum_block_count(context.limits.block_count());
    let decoder = Decoder::from_prefix_with_limits(prefix, decoder_limits)?;
    if full_encoded.is_none()
        && request.mappings.len() == 1
        && request.mappings[0].source == (0..request.expected)
    {
        drop(prefix_storage);
        drop(header_storage);
        let maximum_scratch = if decoder.header().is_raw() {
            Some(0)
        } else {
            decoder
                .layout()
                .maximum_block_size()
                .checked_mul(shuffle_scratch_buffers(decoder.header().shuffle()))
        };
        let optional_full_resident = context
            .resident_decoded
            .checked_add(accumulated_schema)
            .and_then(|resident| resident.checked_add(decoder_index_resident))
            .and_then(|resident| resident.checked_add(encoded_len))
            .and_then(|resident| maximum_scratch.and_then(|scratch| resident.checked_add(scratch)));
        if optional_full_resident.is_some_and(|resident| resident <= context.limits.decoded_size())
        {
            full_encoded = Some(read_exact_range(
                context.store,
                positioned.as_ref(),
                &request.key,
                0,
                encoded_len,
            )?);
            positioned = None;
        }
    }
    let schema_resident = decoder_index_resident
        .checked_add(full_encoded.as_ref().map_or(0, Vec::len))
        .ok_or_else(|| Error::corrupt(&request.key, "schema resident size overflow"))?;
    context.limits.check_decoded_sum(
        [
            context.resident_decoded,
            accumulated_schema,
            schema_resident,
        ],
        "Blosc scatter retained schema",
    )?;
    Ok(PreparedChunk {
        key: request.key.clone(),
        decoder,
        encoded_len,
        positioned,
        full_encoded,
        schema_resident,
    })
}

fn validate_scatter_header(
    compressor: &Compressor,
    key: &str,
    expected: usize,
    encoded_len: usize,
    header: BloscHeader,
) -> Result<()> {
    let expected_version = if compressor.is_blosc1() {
        BloscVersion::Blosc1
    } else if compressor.is_dyn_blosc() {
        BloscVersion::DynBlosc
    } else {
        return Err(Error::invalid_argument(format!(
            "range decode requires a Blosc compressor, got `{}`",
            compressor.id()
        )));
    };
    if header.version() != expected_version {
        return Err(Error::corrupt(
            key,
            format!(
                "metadata declares {}, encoded chunk is {:?}",
                compressor.id(),
                header.version()
            ),
        ));
    }
    if header.decoded_size() != expected {
        return Err(Error::corrupt(
            key,
            format!(
                "decoded size {} does not match expected {expected}",
                header.decoded_size()
            ),
        ));
    }
    if header.encoded_size() != encoded_len {
        return Err(Error::corrupt(
            key,
            format!(
                "encoded size {} does not match store value length {encoded_len}",
                header.encoded_size()
            ),
        ));
    }
    Ok(())
}

fn validate_scatter_requests(
    requests: &[BloscScatterRequest],
    destination_len: usize,
) -> Result<()> {
    let mapping_count = requests.iter().try_fold(0usize, |count, request| {
        count
            .checked_add(request.mappings.len())
            .ok_or_else(|| Error::invalid_argument("scatter mapping count overflow"))
    })?;
    let mut destinations = Vec::new();
    destinations.try_reserve_exact(mapping_count)?;
    for request in requests {
        for mapping in &request.mappings {
            if mapping.source.start > mapping.source.end || mapping.source.end > request.expected {
                return Err(Error::invalid_argument(format!(
                    "scatter source [{}, {}) exceeds 0..{} for '{}'",
                    mapping.source.start, mapping.source.end, request.expected, request.key
                )));
            }
            if mapping.destination.start > mapping.destination.end
                || mapping.destination.end > destination_len
                || mapping.source.len() != mapping.destination.len()
            {
                return Err(Error::invalid_argument(format!(
                    "invalid scatter destination [{}, {}) for {} output bytes",
                    mapping.destination.start, mapping.destination.end, destination_len
                )));
            }
            if !mapping.destination.is_empty() {
                destinations.push(mapping.destination.clone());
            }
        }
    }
    destinations.sort_unstable_by_key(|range| (range.start, range.end));
    if destinations
        .windows(2)
        .any(|pair| pair[0].end > pair[1].start)
    {
        return Err(Error::invalid_argument(
            "scatter destination ranges must not overlap",
        ));
    }
    Ok(())
}

fn run_scatter_tasks(
    context: RangeDecodeContext<'_>,
    requests: &[BloscScatterRequest],
    chunks: &[PreparedChunk],
    destination: &mut [u8],
) -> Result<()> {
    let (pieces, tasks) = build_scatter_tasks(requests, chunks)?;
    if tasks.is_empty() {
        return Ok(());
    }
    let schema_resident = chunks.iter().try_fold(0usize, |total, chunk| {
        total
            .checked_add(chunk.schema_resident)
            .ok_or_else(|| Error::corrupt(&chunk.key, "schema resident size overflow"))
    })?;
    let planner_resident = pieces
        .capacity()
        .checked_mul(std::mem::size_of::<ScatterPiece>())
        .and_then(|resident| {
            resident.checked_add(
                tasks
                    .capacity()
                    .checked_mul(std::mem::size_of::<BlockTask>())?,
            )
        })
        .ok_or_else(|| Error::corrupt("parallel block scatter", "planner size overflow"))?;
    let base_resident = context.limits.check_decoded_sum(
        [context.resident_decoded, schema_resident, planner_resident],
        "parallel block scatter resident output",
    )?;
    let maximum_encoded = tasks
        .iter()
        .map(|task| task.encoded_working_set)
        .max()
        .unwrap_or(0);
    let maximum_workers = context.limits.thread_count().min(tasks.len());
    let mut selected = None;
    for worker_count in (1..=maximum_workers).rev() {
        let retained_scratch = top_k_sum(
            tasks.iter().map(|task| task.decoded_working_set),
            worker_count,
        )?;
        let required = base_resident
            .checked_add(retained_scratch)
            .and_then(|resident| resident.checked_add(maximum_encoded));
        if required.is_some_and(|required| required <= context.limits.decoded_size()) {
            selected = Some((worker_count, retained_scratch));
            break;
        }
    }
    let Some((worker_count, retained_scratch)) = selected else {
        let minimum_scratch = top_k_sum(tasks.iter().map(|task| task.decoded_working_set), 1)?;
        context.limits.check_decoded_sum(
            [base_resident, minimum_scratch, maximum_encoded],
            "parallel block scatter minimum working set",
        )?;
        unreachable!("the minimum working-set check must reject this selection");
    };
    let resident = base_resident
        .checked_add(retained_scratch)
        .ok_or_else(|| Error::corrupt("parallel block scatter", "resident size overflow"))?;
    let budget = DecodeBudget::new(resident, context.limits)?;
    let destination = ScatterDestination {
        base: destination.as_mut_ptr(),
        len: destination.len(),
    };

    parallel::try_for_each_stream_init(
        worker_count,
        tasks.len(),
        |emit| {
            for task_index in 0..tasks.len() {
                emit(task_index)?;
            }
            Ok(())
        },
        DecodeWorker::new,
        |task_index, worker| {
            let task = &tasks[task_index];
            let chunk = &chunks[task.chunk];
            let permit = budget.acquire(task.encoded_working_set)?;
            context.limits.check_decoded_sum(
                [permit.context_resident(), task.encoded_working_set],
                "parallel block scatter working set",
            )?;
            if let Some(block_index) = task.block {
                decode_compressed_task(
                    context.store,
                    chunk,
                    block_index,
                    &pieces[task.pieces.clone()],
                    worker,
                    &destination,
                )
            } else {
                decode_raw_task(
                    context.store,
                    chunk,
                    &pieces[task.pieces.clone()],
                    &destination,
                )
            }
        },
    )
}

fn build_scatter_tasks(
    requests: &[BloscScatterRequest],
    chunks: &[PreparedChunk],
) -> Result<(Vec<ScatterPiece>, Vec<BlockTask>)> {
    let mut pieces = Vec::new();
    let mut tasks = Vec::new();
    const BLOCK_BUCKET_LIMIT: usize = 8192;
    for (chunk_index, (request, chunk)) in requests.iter().zip(chunks).enumerate() {
        if chunk.decoder.header().is_raw() {
            continue;
        }
        let block_count = chunk.decoder.header().block_count();
        if block_count > 0 && block_count <= BLOCK_BUCKET_LIMIT {
            let mut buckets = Vec::new();
            buckets.try_reserve_exact(block_count)?;
            buckets.resize_with(block_count, Vec::new);
            push_compressed_pieces(request, chunk, |block_index, piece| {
                buckets[block_index].push(piece);
                Ok(())
            })?;
            for (block_index, block_pieces) in buckets.into_iter().enumerate() {
                if block_pieces.is_empty() {
                    continue;
                }
                push_compressed_block_task(
                    &mut pieces,
                    &mut tasks,
                    chunks,
                    chunk_index,
                    block_index,
                    block_pieces,
                )?;
            }
            continue;
        }

        let mut records = Vec::new();
        push_compressed_pieces(request, chunk, |block_index, piece| {
            records.push((block_index, piece));
            Ok(())
        })?;
        records.sort_unstable_by_key(|(block, _)| *block);
        let mut cursor = 0usize;
        while cursor < records.len() {
            let block_index = records[cursor].0;
            let start = cursor;
            cursor += 1;
            while cursor < records.len() && records[cursor].0 == block_index {
                cursor += 1;
            }
            let block_pieces = records[start..cursor]
                .iter()
                .map(|(_, piece)| *piece)
                .collect();
            push_compressed_block_task(
                &mut pieces,
                &mut tasks,
                chunks,
                chunk_index,
                block_index,
                block_pieces,
            )?;
        }
    }

    for (chunk_index, (request, chunk)) in requests.iter().zip(chunks).enumerate() {
        if !chunk.decoder.header().is_raw() {
            continue;
        }
        let mut mappings = Vec::new();
        mappings.try_reserve_exact(request.mappings.len())?;
        mappings.extend(
            request
                .mappings
                .iter()
                .filter(|mapping| !mapping.source.is_empty()),
        );
        mappings.sort_unstable_by_key(|mapping| {
            (
                mapping.source.start,
                mapping.source.end,
                mapping.destination.start,
            )
        });
        pieces.try_reserve(mappings.len())?;
        let mut cursor = 0usize;
        while cursor < mappings.len() {
            let window_start = mappings[cursor].source.start;
            let mut window_end = mappings[cursor].source.end;
            let start = pieces.len();
            while cursor < mappings.len() {
                let mapping = mappings[cursor];
                let coalesced_end = window_end.saturating_add(RAW_SCATTER_COALESCE_GAP);
                let span = mapping.source.end.saturating_sub(window_start);
                if pieces.len() > start
                    && (mapping.source.start > coalesced_end || span > RAW_SCATTER_MAX_WINDOW)
                {
                    break;
                }
                window_end = window_end.max(mapping.source.end);
                pieces.push(ScatterPiece {
                    source_in_block: mapping.source.start,
                    destination: mapping.destination.start,
                    len: mapping.source.len(),
                });
                cursor += 1;
            }
            tasks.try_reserve(1)?;
            tasks.push(BlockTask {
                chunk: chunk_index,
                block: None,
                pieces: start..pieces.len(),
                encoded_working_set: if chunk.full_encoded.is_some() {
                    0
                } else {
                    window_end - window_start
                },
                decoded_working_set: 0,
            });
        }
    }
    Ok((pieces, tasks))
}

fn decode_compressed_task(
    store: &dyn ByteStore,
    chunk: &PreparedChunk,
    block_index: usize,
    pieces: &[ScatterPiece],
    worker: &mut DecodeWorker,
    destination: &ScatterDestination,
) -> Result<()> {
    let block = chunk
        .decoder
        .block(block_index)
        .ok_or_else(|| Error::corrupt(&chunk.key, "block index is out of range"))?;
    let encoded_range = block.encoded_range();
    let encoded_storage;
    let encoded = if let Some(full) = &chunk.full_encoded {
        full.get(encoded_range.clone())
            .ok_or_else(|| Error::corrupt(&chunk.key, "encoded block range exceeds chunk"))?
    } else {
        encoded_storage = read_exact_range(
            store,
            chunk.positioned.as_ref(),
            &chunk.key,
            encoded_range.start,
            encoded_range.len(),
        )?;
        &encoded_storage
    };
    let DecodeWorker { decoded, workspace } = worker;
    if decoded.len() < block.decoded_len() {
        decoded.try_reserve_exact(block.decoded_len() - decoded.len())?;
        decoded.resize(block.decoded_len(), 0);
    }
    let decoded = &mut decoded[..block.decoded_len()];
    chunk
        .decoder
        .decode_block_into(block_index, encoded, decoded, workspace)?;
    scatter_pieces(decoded, 0, pieces, destination)
}

fn decode_raw_task(
    store: &dyn ByteStore,
    chunk: &PreparedChunk,
    pieces: &[ScatterPiece],
    destination: &ScatterDestination,
) -> Result<()> {
    let first = pieces
        .first()
        .ok_or_else(|| Error::corrupt(&chunk.key, "raw scatter task has no piece"))?;
    let source_start = first.source_in_block;
    let source_end = pieces.iter().try_fold(source_start, |end, piece| {
        piece
            .source_in_block
            .checked_add(piece.len)
            .map(|piece_end| end.max(piece_end))
            .ok_or_else(|| Error::corrupt(&chunk.key, "raw scatter end overflow"))
    })?;
    let source_len = source_end - source_start;
    let offset = HEADER_LEN
        .checked_add(source_start)
        .ok_or_else(|| Error::corrupt(&chunk.key, "raw scatter offset overflow"))?;
    let end = offset
        .checked_add(source_len)
        .ok_or_else(|| Error::corrupt(&chunk.key, "raw scatter range overflow"))?;
    let storage;
    let source = if let Some(full) = &chunk.full_encoded {
        full.get(offset..end)
            .ok_or_else(|| Error::corrupt(&chunk.key, "raw scatter exceeds chunk"))?
    } else {
        storage = read_exact_range(
            store,
            chunk.positioned.as_ref(),
            &chunk.key,
            offset,
            source_len,
        )?;
        &storage
    };
    scatter_pieces(source, source_start, pieces, destination)
}

fn scatter_pieces(
    source: &[u8],
    source_base: usize,
    pieces: &[ScatterPiece],
    destination: &ScatterDestination,
) -> Result<()> {
    for piece in pieces {
        let source_start = piece
            .source_in_block
            .checked_sub(source_base)
            .ok_or_else(|| Error::invalid_argument("scatter source precedes buffer"))?;
        let source_end = source_start
            .checked_add(piece.len)
            .ok_or_else(|| Error::invalid_argument("scatter source overflow"))?;
        let destination_end = piece
            .destination
            .checked_add(piece.len)
            .ok_or_else(|| Error::invalid_argument("scatter destination overflow"))?;
        if source_end > source.len() || destination_end > destination.len {
            return Err(Error::invalid_argument("scatter piece is out of bounds"));
        }
        // SAFETY: `validate_scatter_requests` proves every destination range is
        // in bounds and pairwise disjoint. `build_scatter_tasks` partitions each
        // mapping at decoded-block boundaries without changing those ranges.
        // The source is a separate initialized decode/read buffer, and the
        // scoped worker pool finishes before `destination` can be accessed again.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.as_ptr().add(source_start),
                destination.base.add(piece.destination),
                piece.len,
            );
        }
    }
    Ok(())
}

const RAW_SCATTER_COALESCE_GAP: usize = 4 * 1024;
const RAW_SCATTER_MAX_WINDOW: usize = 1024 * 1024;

fn top_k_sum(values: impl IntoIterator<Item = usize>, k: usize) -> Result<usize> {
    if k == 0 {
        return Ok(0);
    }
    let mut largest = BinaryHeap::new();
    largest.try_reserve(k)?;
    for value in values {
        if largest.len() < k {
            largest.push(Reverse(value));
        } else if largest.peek().is_some_and(|smallest| value > smallest.0) {
            largest.pop();
            largest.push(Reverse(value));
        }
    }
    largest.into_iter().try_fold(0usize, |total, value| {
        total
            .checked_add(value.0)
            .ok_or_else(|| Error::invalid_argument("parallel decode scratch size overflow"))
    })
}

const fn shuffle_scratch_buffers(shuffle: BloscShuffle) -> usize {
    match shuffle {
        BloscShuffle::None => 1,
        BloscShuffle::Bytes => 2,
        BloscShuffle::Bits => 3,
    }
}

fn push_compressed_pieces(
    request: &BloscScatterRequest,
    chunk: &PreparedChunk,
    mut emit: impl FnMut(usize, ScatterPiece) -> Result<()>,
) -> Result<()> {
    let block_count = chunk.decoder.header().block_count();
    for mapping in &request.mappings {
        if mapping.source.is_empty() {
            continue;
        }
        let blocks =
            intersecting_blocks(&chunk.decoder, block_count, &mapping.source, &request.key)?;
        for block_index in blocks {
            let block = chunk
                .decoder
                .block(block_index)
                .ok_or_else(|| Error::corrupt(&request.key, "block index is out of range"))?;
            let decoded = block.decoded_range();
            let overlap_start = mapping.source.start.max(decoded.start);
            let overlap_end = mapping.source.end.min(decoded.end);
            if overlap_start < overlap_end {
                emit(
                    block_index,
                    ScatterPiece {
                        source_in_block: overlap_start - decoded.start,
                        destination: mapping.destination.start
                            + (overlap_start - mapping.source.start),
                        len: overlap_end - overlap_start,
                    },
                )?;
            }
        }
    }
    Ok(())
}

fn push_compressed_block_task(
    pieces: &mut Vec<ScatterPiece>,
    tasks: &mut Vec<BlockTask>,
    chunks: &[PreparedChunk],
    chunk_index: usize,
    block_index: usize,
    block_pieces: Vec<ScatterPiece>,
) -> Result<()> {
    if block_pieces.is_empty() {
        return Ok(());
    }
    let chunk = &chunks[chunk_index];
    let block = chunk
        .decoder
        .block(block_index)
        .ok_or_else(|| Error::corrupt(&chunk.key, "block index is out of range"))?;
    let encoded_range = block.encoded_range();
    if encoded_range.end > chunk.encoded_len {
        return Err(Error::corrupt(
            &chunk.key,
            "encoded block range exceeds chunk",
        ));
    }
    let decoded_working_set = block
        .decoded_len()
        .checked_mul(shuffle_scratch_buffers(chunk.decoder.header().shuffle()))
        .ok_or_else(|| Error::corrupt(&chunk.key, "decode scratch size overflow"))?;
    let start = pieces.len();
    pieces.try_reserve(block_pieces.len())?;
    pieces.extend(block_pieces);
    tasks.try_reserve(1)?;
    tasks.push(BlockTask {
        chunk: chunk_index,
        block: Some(block_index),
        pieces: start..pieces.len(),
        encoded_working_set: if chunk.full_encoded.is_some() {
            0
        } else {
            encoded_range.len()
        },
        decoded_working_set,
    });
    Ok(())
}

fn intersecting_blocks(
    decoder: &Decoder,
    block_count: usize,
    selection: &Range<usize>,
    key: &str,
) -> Result<Range<usize>> {
    let mut low = 0usize;
    let mut high = block_count;
    while low < high {
        let middle = low + (high - low) / 2;
        let block = decoder
            .block(middle)
            .ok_or_else(|| Error::corrupt(key, "block index is out of range"))?;
        if block.decoded_range().end <= selection.start {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let first = low;
    high = block_count;
    while low < high {
        let middle = low + (high - low) / 2;
        let block = decoder
            .block(middle)
            .ok_or_else(|| Error::corrupt(key, "block index is out of range"))?;
        if block.decoded_range().start < selection.end {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok(first..low)
}

fn read_exact_range(
    store: &dyn ByteStore,
    positioned: Option<&PositionedValue>,
    key: &str,
    offset: usize,
    len: usize,
) -> Result<Vec<u8>> {
    let offset = u64::try_from(offset)
        .map_err(|_| Error::corrupt(key, "encoded range offset exceeds u64"))?;
    let bytes = if let Some(value) = positioned {
        let len_u64 = u64::try_from(len)
            .map_err(|_| Error::corrupt(key, "encoded range length exceeds u64"))?;
        let end = offset
            .checked_add(len_u64)
            .ok_or_else(|| Error::corrupt(key, "encoded range end overflow"))?;
        if end > value.len() {
            return Err(Error::corrupt(
                key,
                format!(
                    "encoded range [{offset}, {end}) exceeds positioned value length {}",
                    value.len()
                ),
            ));
        }
        let absolute = value
            .base_offset()
            .checked_add(offset)
            .ok_or_else(|| Error::corrupt(key, "positioned range offset overflow"))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len)?;
        bytes.resize(len, 0);
        value.file().read_exact_at(&mut bytes, absolute)?;
        bytes
    } else {
        store.read_range(key, offset, len)?
    };
    if bytes.len() != len {
        return Err(Error::corrupt(
            key,
            format!(
                "encoded range at offset {offset} returned {} bytes, expected {len}",
                bytes.len()
            ),
        ));
    }
    Ok(bytes)
}
