use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use crate::codec::Compressor;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::io_util::{read_meta, write_meta};
use crate::limits::ReadLimits;
use crate::meta::{ArrayMeta, ChunkGridMeta, DenseMeta, MetaBody, MetaFile, PartitionMeta};
use crate::numeric::{encode_matrix_values_into, MatrixValue};
use crate::parallel::{self, default_threads, validate_threads};
use crate::partition::{dense_blosc1_block_size, visit_dense_chunks, Partition};
use crate::range_decode::{
    decode_blosc_scatter_into, BloscScatterRequest, RangeDecodeContext, ScatterMapping,
};
use crate::select::NormalizedAxis;
use crate::storage::{chunk_key, ByteStore, DirectoryTransaction, StoreLocation};

const DATA_DIR: &str = "data";

/// Configures chunking and compression for a dense matrix.
#[derive(Debug, Clone)]
pub struct DenseWriter {
    dir: std::path::PathBuf,
    chunk: Partition,
    block: Partition,
    compressor: Compressor,
    threads: usize,
}

impl DenseWriter {
    /// Create a writer. Callers must pass chunk/block partitions explicitly.
    pub fn new(dir: impl AsRef<Path>, chunk: Partition, block: Partition) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            chunk,
            block,
            // `block_size` is overwritten from `block` × n_genes × dtype at write time.
            compressor: Compressor::blosc1_lz4(1),
            threads: default_threads(),
        }
    }

    /// Sets Blosc1 codec options. Any configured `block_size` is ignored and
    /// recomputed from [`Self::block`] when writing.
    pub fn compressor(mut self, compressor: Compressor) -> Self {
        self.compressor = compressor;
        self
    }

    /// Maximum workers used by the bounded chunk encode/write pipeline.
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Write a row-major `cells × genes` matrix.
    pub fn write<V: MatrixValue>(self, values: &[V], shape: [u64; 2]) -> Result<()> {
        let n_cells = shape[0];
        let n_genes = shape[1];
        let dtype = V::DTYPE;
        let element_size = dtype.size();
        let expected = n_cells
            .checked_mul(n_genes)
            .ok_or_else(|| Error::invalid_argument("dense size overflow"))?;
        if values.len() as u64 != expected {
            return Err(Error::invalid_argument(format!(
                "values length {} does not match shape [{n_cells}, {n_genes}]",
                values.len()
            )));
        }
        let threads = validate_threads(self.threads)?;
        self.chunk.validate()?;
        self.block.validate()?;
        if !self.chunk.is_fixed_cells() {
            return Err(Error::invalid_argument(
                "dense chunks require fixed_cells partition",
            ));
        }
        let Some(block_cells) = self.block.fixed_cells_n() else {
            return Err(Error::invalid_argument(
                "dense blocks require fixed_cells partition",
            ));
        };
        if !self.compressor.is_blosc1() {
            return Err(Error::invalid_argument(
                "dense data chunks require blosc1 compressor",
            ));
        }
        let block_size = dense_blosc1_block_size(block_cells, n_genes, element_size)?;
        let compressor = match self.compressor {
            Compressor::Blosc1 { options, .. } => Compressor::blosc1(options, block_size),
            _ => {
                return Err(Error::invalid_argument(
                    "dense data chunks require blosc1 compressor",
                ))
            }
        };
        compressor.validate()?;

        let mut transaction = DirectoryTransaction::new(&self.dir)?;

        let n_genes_usize = usize::try_from(n_genes)
            .map_err(|_| Error::invalid_argument("dense column count exceeds usize"))?;
        let chunk_cells = self
            .chunk
            .fixed_cells_n()
            .ok_or_else(|| Error::invalid_argument("dense chunks require fixed_cells partition"))?;
        let chunk_count = usize::try_from(n_cells.div_ceil(chunk_cells))
            .map_err(|_| Error::invalid_argument("dense chunk count exceeds usize"))?;

        let mut offsets = Vec::new();
        offsets.try_reserve_exact(chunk_count)?;
        let store = transaction.store();
        parallel::try_for_each_stream_init(
            threads,
            chunk_count,
            |emit| {
                visit_dense_chunks(n_cells, n_genes, &self.chunk, |id, span| {
                    offsets.push(span.cell_start);
                    emit((id, span))
                })
            },
            Vec::new,
            |(id, span), chunk_bytes| {
                let value_start = usize::try_from(span.cell_start)
                    .ok()
                    .and_then(|start| start.checked_mul(n_genes_usize))
                    .ok_or_else(|| Error::invalid_argument("dense value offset overflow"))?;
                let value_end = usize::try_from(span.cell_end)
                    .ok()
                    .and_then(|end| end.checked_mul(n_genes_usize))
                    .ok_or_else(|| Error::invalid_argument("dense value offset overflow"))?;
                let chunk_values = values
                    .get(value_start..value_end)
                    .ok_or_else(|| Error::invalid_argument("dense chunk exceeds input values"))?;
                encode_matrix_values_into(chunk_values, chunk_bytes)?;
                let encoded = compressor.encode_buffer(chunk_bytes, element_size)?;
                let id = u64::try_from(id)
                    .map_err(|_| Error::invalid_argument("dense chunk id exceeds u64"))?;
                store.write_value(&chunk_key(DATA_DIR, id), &encoded)
            },
        )?;

        let meta = MetaFile::dense(DenseMeta {
            shape: [n_cells, n_genes],
            partition: PartitionMeta {
                chunk: self.chunk,
                block: self.block,
            },
            data: ArrayMeta::new(DATA_DIR, dtype, compressor),
            chunks: ChunkGridMeta::from_cell_starts(offsets),
        });
        let store = transaction.store_mut();
        write_meta(store, &meta)?;
        transaction.commit()
    }
}

/// Opened dense matrix backed by any [`ByteStore`].
#[derive(Clone)]
pub struct DenseMatrix {
    store: Arc<dyn ByteStore>,
    meta: DenseMeta,
    limits: ReadLimits,
}

/// Open a dense matrix from a directory or a zip prefix.
pub fn open_dense(location: impl Into<StoreLocation>) -> Result<DenseMatrix> {
    open_dense_with_limits(location, ReadLimits::default())
}

/// Open a dense matrix with explicit resource limits.
pub fn open_dense_with_limits(
    location: impl Into<StoreLocation>,
    limits: ReadLimits,
) -> Result<DenseMatrix> {
    let store = location.into().open()?;
    DenseMatrix::from_store_with_limits(store, limits)
}

impl DenseMatrix {
    pub fn from_store(store: Arc<dyn ByteStore>) -> Result<Self> {
        Self::from_store_with_limits(store, ReadLimits::default())
    }

    pub fn from_store_with_limits(store: Arc<dyn ByteStore>, limits: ReadLimits) -> Result<Self> {
        let limits = limits.validate()?;
        let file = read_meta(store.as_ref(), limits)?;
        let MetaBody::Dense(meta) = file.into_body() else {
            return Err(Error::invalid_meta("store kind is not dense"));
        };
        Ok(Self::from_parts(store, meta, limits))
    }

    /// Construct from already-validated metadata (single meta read path).
    pub(crate) fn from_parts(
        store: Arc<dyn ByteStore>,
        meta: DenseMeta,
        limits: ReadLimits,
    ) -> Self {
        Self {
            store,
            meta,
            limits,
        }
    }

    pub fn meta(&self) -> &DenseMeta {
        &self.meta
    }

    pub fn store(&self) -> &Arc<dyn ByteStore> {
        &self.store
    }

    pub fn limits(&self) -> ReadLimits {
        self.limits
    }

    /// Consume into store + meta + limits without cloning.
    pub fn into_parts(self) -> (Arc<dyn ByteStore>, DenseMeta, ReadLimits) {
        (self.store, self.meta, self.limits)
    }

    pub fn shape(&self) -> [u64; 2] {
        self.meta.shape
    }

    pub fn n_rows(&self) -> u64 {
        self.meta.shape[0]
    }

    pub fn n_cols(&self) -> u64 {
        self.meta.shape[1]
    }

    pub fn dtype(&self) -> DType {
        self.meta.data.dtype
    }

    pub fn decode_all(&self) -> Result<Vec<u8>> {
        self.decode_rows(0..self.n_rows())
    }

    /// Decode the half-open row range, returning contiguous row-major bytes.
    pub fn decode_rows(&self, rows: Range<u64>) -> Result<Vec<u8>> {
        self.decode_selected_rows(&NormalizedAxis::Contiguous {
            start: rows.start,
            end: rows.end,
        })
    }

    pub(crate) fn decode_selected_rows(&self, rows: &NormalizedAxis) -> Result<Vec<u8>> {
        self.decode_selection(
            rows,
            &NormalizedAxis::Contiguous {
                start: 0,
                end: self.n_cols(),
            },
        )
    }

    pub(crate) fn decode_selection(
        &self,
        rows: &NormalizedAxis,
        cols: &NormalizedAxis,
    ) -> Result<Vec<u8>> {
        rows.validate(self.n_rows())?;
        cols.validate(self.n_cols())?;
        let row_bytes = checked_byte_len(self.n_cols(), self.meta.data.dtype.size(), "dense row")?;
        let output_row_bytes = checked_byte_len(
            cols.len(),
            self.meta.data.dtype.size(),
            "dense selected row",
        )?;
        let output_len = checked_byte_len(rows.len(), output_row_bytes, "dense output")?;
        self.limits.check_decoded(output_len, "dense output")?;
        let mut output = zeroed_vec(output_len)?;
        if rows.is_empty() || cols.is_empty() {
            return Ok(output);
        }

        let requests = plan_dense_scatter_requests(
            self,
            rows,
            cols,
            row_bytes,
            output_row_bytes,
            self.meta.data.dtype.size(),
        )?;
        decode_blosc_scatter_into(
            &self.meta.data.compressor,
            RangeDecodeContext::new(self.store.as_ref(), output_len, self.limits),
            &requests,
            &mut output,
        )?;
        Ok(output)
    }
}

fn plan_dense_scatter_requests(
    matrix: &DenseMatrix,
    rows: &NormalizedAxis,
    cols: &NormalizedAxis,
    row_bytes: usize,
    output_row_bytes: usize,
    element_size: usize,
) -> Result<Vec<BloscScatterRequest>> {
    let column_runs = dense_column_runs(cols, element_size)?;
    match rows {
        NormalizedAxis::Contiguous { start, end } => {
            let mut requests = Vec::new();
            for chunk_id in matrix.meta.chunks.overlapping_chunks(*start, *end) {
                let (chunk_start, chunk_end) =
                    matrix.meta.chunks.cell_range(chunk_id, matrix.n_rows())?;
                let overlap_start = (*start).max(chunk_start);
                let overlap_end = (*end).min(chunk_end);
                if overlap_start >= overlap_end {
                    continue;
                }
                let expected = checked_byte_len(chunk_end - chunk_start, row_bytes, "dense chunk")?;
                let mut mappings = Vec::new();
                if is_full_dense_row(&column_runs, row_bytes) {
                    mappings.try_reserve_exact(1)?;
                    let source_start = checked_byte_len(
                        overlap_start - chunk_start,
                        row_bytes,
                        "dense source rows",
                    )?;
                    let destination_start = checked_byte_len(
                        overlap_start - *start,
                        output_row_bytes,
                        "dense destination rows",
                    )?;
                    let len = checked_byte_len(
                        overlap_end - overlap_start,
                        row_bytes,
                        "dense selected rows",
                    )?;
                    mappings.push(ScatterMapping {
                        source: source_start..source_start + len,
                        destination: destination_start..destination_start + len,
                    });
                } else {
                    for source_row in overlap_start..overlap_end {
                        append_dense_row_mappings(
                            &mut mappings,
                            source_row - chunk_start,
                            source_row - *start,
                            row_bytes,
                            output_row_bytes,
                            &column_runs,
                        )?;
                    }
                }
                requests.try_reserve(1)?;
                requests.push(BloscScatterRequest {
                    key: chunk_key(&matrix.meta.data.path, chunk_id as u64),
                    expected,
                    mappings,
                });
            }
            Ok(requests)
        }
        NormalizedAxis::Gather { positions } => {
            let mut items = Vec::new();
            items.try_reserve_exact(positions.len())?;
            for (destination, &source) in positions.iter().enumerate() {
                items.push((matrix.meta.chunks.chunk_of(source)?, source, destination));
            }
            items.sort_unstable();
            let mut requests = Vec::new();
            let mut cursor = 0usize;
            while cursor < items.len() {
                let chunk_id = items[cursor].0;
                let (chunk_start, chunk_end) =
                    matrix.meta.chunks.cell_range(chunk_id, matrix.n_rows())?;
                let mut mappings = Vec::new();
                while cursor < items.len() && items[cursor].0 == chunk_id {
                    let (_, source, destination) = items[cursor];
                    append_dense_row_mappings(
                        &mut mappings,
                        source - chunk_start,
                        u64::try_from(destination).map_err(|_| {
                            Error::invalid_argument("dense destination row exceeds u64")
                        })?,
                        row_bytes,
                        output_row_bytes,
                        &column_runs,
                    )?;
                    cursor += 1;
                }
                requests.try_reserve(1)?;
                requests.push(BloscScatterRequest {
                    key: chunk_key(&matrix.meta.data.path, chunk_id as u64),
                    expected: checked_byte_len(chunk_end - chunk_start, row_bytes, "dense chunk")?,
                    mappings,
                });
            }
            Ok(requests)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DenseColumnRun {
    source: usize,
    destination: usize,
    len: usize,
}

fn is_full_dense_row(runs: &[DenseColumnRun], row_bytes: usize) -> bool {
    matches!(
        runs,
        [DenseColumnRun {
            source: 0,
            destination: 0,
            len,
        }] if *len == row_bytes
    )
}

fn dense_column_runs(cols: &NormalizedAxis, element_size: usize) -> Result<Vec<DenseColumnRun>> {
    match cols {
        NormalizedAxis::Contiguous { start, end } => Ok(vec![DenseColumnRun {
            source: checked_byte_len(*start, element_size, "dense source columns")?,
            destination: 0,
            len: checked_byte_len(end - start, element_size, "dense selected columns")?,
        }]),
        NormalizedAxis::Gather { positions } => {
            let mut runs: Vec<DenseColumnRun> = Vec::new();
            runs.try_reserve_exact(positions.len())?;
            for (destination, &source) in positions.iter().enumerate() {
                let source = checked_byte_len(source, element_size, "dense source column")?;
                let destination = destination
                    .checked_mul(element_size)
                    .ok_or_else(|| Error::invalid_argument("dense destination column overflow"))?;
                if let Some(previous) = runs.last_mut() {
                    if previous.source.checked_add(previous.len) == Some(source)
                        && previous.destination.checked_add(previous.len) == Some(destination)
                    {
                        previous.len = previous
                            .len
                            .checked_add(element_size)
                            .ok_or_else(|| Error::invalid_argument("dense column run overflow"))?;
                        continue;
                    }
                }
                runs.push(DenseColumnRun {
                    source,
                    destination,
                    len: element_size,
                });
            }
            Ok(runs)
        }
    }
}

fn append_dense_row_mappings(
    mappings: &mut Vec<ScatterMapping>,
    source_row: u64,
    destination_row: u64,
    row_bytes: usize,
    output_row_bytes: usize,
    column_runs: &[DenseColumnRun],
) -> Result<()> {
    mappings.try_reserve(column_runs.len())?;
    let source_base = checked_byte_len(source_row, row_bytes, "dense source row")?;
    let destination_base =
        checked_byte_len(destination_row, output_row_bytes, "dense destination row")?;
    for run in column_runs {
        let source_start = source_base
            .checked_add(run.source)
            .ok_or_else(|| Error::invalid_argument("dense source mapping overflow"))?;
        let source_end = source_start
            .checked_add(run.len)
            .ok_or_else(|| Error::invalid_argument("dense source mapping end overflow"))?;
        let destination_start = destination_base
            .checked_add(run.destination)
            .ok_or_else(|| Error::invalid_argument("dense destination mapping overflow"))?;
        let destination_end = destination_start
            .checked_add(run.len)
            .ok_or_else(|| Error::invalid_argument("dense destination mapping end overflow"))?;
        push_scatter_mapping(
            mappings,
            source_start..source_end,
            destination_start..destination_end,
        );
    }
    Ok(())
}

fn push_scatter_mapping(
    mappings: &mut Vec<ScatterMapping>,
    source: Range<usize>,
    destination: Range<usize>,
) {
    if let Some(previous) = mappings.last_mut() {
        if previous.source.end == source.start && previous.destination.end == destination.start {
            previous.source.end = source.end;
            previous.destination.end = destination.end;
            return;
        }
    }
    mappings.push(ScatterMapping {
        source,
        destination,
    });
}

fn checked_byte_len(count: u64, element_bytes: usize, context: &str) -> Result<usize> {
    let element_size = u64::try_from(element_bytes)
        .map_err(|_| Error::invalid_argument(format!("{context} element size exceeds u64")))?;
    usize::try_from(
        count
            .checked_mul(element_size)
            .ok_or_else(|| Error::invalid_argument(format!("{context} byte length overflow")))?,
    )
    .map_err(|_| Error::invalid_argument(format!("{context} byte length exceeds usize")))
}

fn zeroed_vec(len: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output.try_reserve_exact(len)?;
    output.resize(len, 0);
    Ok(output)
}
