use std::borrow::Cow;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use crate::array::{CsrArray, DenseArray, SelectedArray};
use crate::codec::Compressor;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::io_util::{read_meta, u64_slice_from_le_bytes, u64_slice_to_le_bytes, write_meta};
use crate::kernel::{output_index_dtype, read_index_unchecked, write_index, GatherColumns};
use crate::limits::ReadLimits;
use crate::meta::{ArrayMeta, ChunkGridMeta, CsrMeta, MetaBody, MetaFile, PartitionMeta};
use crate::numeric::{
    csr_index_dtype_from_or_mask, encode_csr_indices_into, encode_matrix_values_into,
    promote_csr_indices, promote_indptr, IntegerIndex, MatrixValue,
};
use crate::parallel::{self, default_threads, validate_threads};
use crate::partition::{plan_csr_blocks, validate_indptr, visit_csr_chunks, BlockTable, Partition};
use crate::range_decode::{
    decode_blosc_scatter_into, BloscScatterRequest, RangeDecodeContext, ScatterMapping,
};
use crate::select::{CsrOutput, NormalizedAxis};
use crate::storage::{chunk_key, ByteStore, ByteStoreMut, DirectoryTransaction, StoreLocation};

const INDPTR_FILE: &str = "indptr";
const INDICES_DIR: &str = "indices";
const DATA_DIR: &str = "data";

enum CsrWriteTask<'a, V> {
    Indices {
        id: u64,
        values: &'a [u64],
        blocks: Arc<BlockTable>,
    },
    Data {
        id: u64,
        values: &'a [V],
        blocks: Arc<BlockTable>,
    },
}

#[derive(Default)]
struct CsrWriteWorkspace {
    chunk: Vec<u8>,
}

/// Configures chunking and compression for a CSR matrix.
#[derive(Debug, Clone)]
pub struct CsrWriter {
    dir: std::path::PathBuf,
    chunk: Partition,
    block: Partition,
    compressor: Compressor,
    indptr_compressor: Compressor,
    threads: usize,
}

impl CsrWriter {
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            chunk: Partition::fixed_cells(1024),
            block: Partition::fixed_cells(16),
            compressor: Compressor::dyn_blosc_lz4(),
            indptr_compressor: Compressor::zstd(3),
            threads: default_threads(),
        }
    }

    pub fn chunk(mut self, partition: Partition) -> Self {
        self.chunk = partition;
        self
    }

    pub fn block(mut self, partition: Partition) -> Self {
        self.block = partition;
        self
    }

    pub fn compressor(mut self, compressor: Compressor) -> Self {
        self.compressor = compressor;
        self
    }

    pub fn indptr_compressor(mut self, compressor: Compressor) -> Self {
        self.indptr_compressor = compressor;
        self
    }

    /// Maximum workers used by the bounded chunk encode/write pipeline.
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Write CSR arrays from typed `indptr` / `indices` / `data` slices.
    pub fn write<P, I, V>(
        self,
        indptr: &[P],
        indices: &[I],
        data: &[V],
        shape: [u64; 2],
    ) -> Result<()>
    where
        P: IntegerIndex,
        I: IntegerIndex,
        V: MatrixValue,
    {
        let indptr = promote_indptr(indptr)?;
        let (index_or_mask, indices) = promote_csr_indices(indices, shape[1])?;
        self.write_promoted_with_mask(indptr, indices, data, shape, index_or_mask)
    }

    /// Write CSR arrays whose offsets and indices are already represented as
    /// owned `u64` values.
    ///
    /// This adapter-oriented entry point avoids an additional full-size index
    /// copy after a foreign-language binding has validated and promoted its
    /// input. All ordinary CSR invariants are still checked, rows are
    /// canonicalized, and indices are stored using the narrowest supported
    /// on-disk dtype.
    pub fn write_promoted<V: MatrixValue>(
        self,
        indptr: Vec<u64>,
        indices: Vec<u64>,
        data: &[V],
        shape: [u64; 2],
    ) -> Result<()> {
        let mut index_or_mask = 0u64;
        for (position, &index) in indices.iter().enumerate() {
            if index >= shape[1] {
                return Err(Error::invalid_argument(format!(
                    "csr index at position {position} is {index}, outside 0..{}",
                    shape[1]
                )));
            }
            index_or_mask |= index;
        }
        self.write_promoted_with_mask(indptr, indices, data, shape, index_or_mask)
    }

    fn write_promoted_with_mask<V: MatrixValue>(
        self,
        indptr: Vec<u64>,
        mut indices: Vec<u64>,
        data: &[V],
        shape: [u64; 2],
        index_or_mask: u64,
    ) -> Result<()> {
        let threads = validate_threads(self.threads)?;
        self.chunk.validate()?;
        self.block.validate()?;
        self.compressor.validate()?;
        self.indptr_compressor.validate()?;
        if !self.compressor.is_dyn_blosc() {
            return Err(Error::invalid_argument(
                "csr indices/data chunks require dyn-blosc compressor",
            ));
        }

        validate_indptr(&indptr)?;
        let n_cells = indptr.len() - 1;
        let nnz = indptr
            .last()
            .copied()
            .ok_or_else(|| Error::invalid_argument("indptr must not be empty"))?;
        if shape[0] != n_cells as u64 {
            return Err(Error::invalid_argument(format!(
                "shape[0] {} does not match indptr n_cells {n_cells}",
                shape[0]
            )));
        }
        let n_genes = shape[1];
        let value_dtype = V::DTYPE;
        let data_size = value_dtype.size();

        if indices.len() as u64 != nnz {
            return Err(Error::invalid_argument(format!(
                "indices length {} != nnz {nnz}",
                indices.len()
            )));
        }
        if data.len() as u64 != nnz {
            return Err(Error::invalid_argument(format!(
                "data length {} != nnz {nnz}",
                data.len()
            )));
        }

        let index_dtype = if nnz == 0 {
            DType::U16
        } else {
            csr_index_dtype_from_or_mask(index_or_mask)?
        };
        let index_size = index_dtype.size();
        let data = canonicalize_csr_rows(&indptr, &mut indices, data)?;

        let mut transaction = DirectoryTransaction::new(&self.dir)?;

        let cost_element = index_size.max(data_size);
        let mut offsets = Vec::new();
        let chunk_upper_bound = if let Some(chunk_cells) = self.chunk.fixed_cells_n() {
            let chunk_cells = usize::try_from(chunk_cells)
                .map_err(|_| Error::invalid_argument("csr chunk size exceeds usize"))?;
            let chunk_count = n_cells.div_ceil(chunk_cells);
            offsets.try_reserve_exact(chunk_count)?;
            chunk_count
        } else {
            n_cells
        };
        let store = transaction.store();
        parallel::try_for_each_stream_init(
            threads,
            chunk_upper_bound.saturating_mul(2),
            |emit| {
                visit_csr_chunks(&indptr, cost_element, &self.chunk, |id, span| {
                    if offsets.len() == offsets.capacity() {
                        offsets.try_reserve(1)?;
                    }
                    offsets.push(span.cell_start);
                    let cell_start = usize::try_from(span.cell_start)
                        .map_err(|_| Error::invalid_argument("chunk cell start exceeds usize"))?;
                    let cell_end = usize::try_from(span.cell_end)
                        .map_err(|_| Error::invalid_argument("chunk cell end exceeds usize"))?;
                    let blocks = Arc::new(plan_csr_blocks(
                        &indptr[cell_start..=cell_end],
                        cost_element,
                        &self.block,
                    )?);
                    let nnz_start = usize::try_from(span.nnz_start)
                        .map_err(|_| Error::invalid_argument("chunk nnz start exceeds usize"))?;
                    let nnz_end = usize::try_from(span.nnz_end)
                        .map_err(|_| Error::invalid_argument("chunk nnz end exceeds usize"))?;
                    let indices_values = indices
                        .get(nnz_start..nnz_end)
                        .ok_or_else(|| Error::invalid_argument("chunk exceeds csr indices"))?;
                    let data_values = data
                        .get(nnz_start..nnz_end)
                        .ok_or_else(|| Error::invalid_argument("chunk exceeds csr data"))?;
                    let id = u64::try_from(id)
                        .map_err(|_| Error::invalid_argument("csr chunk id exceeds u64"))?;
                    emit(CsrWriteTask::Indices {
                        id,
                        values: indices_values,
                        blocks: Arc::clone(&blocks),
                    })?;
                    emit(CsrWriteTask::Data {
                        id,
                        values: data_values,
                        blocks,
                    })
                })
            },
            CsrWriteWorkspace::default,
            |task, workspace| match task {
                CsrWriteTask::Indices { id, values, blocks } => {
                    encode_csr_indices_into(values, index_dtype, &mut workspace.chunk)?;
                    let encoded = self.compressor.encode_partitioned(
                        &workspace.chunk,
                        index_size,
                        blocks.as_ref(),
                    )?;
                    store.write_value(&chunk_key(INDICES_DIR, id), &encoded)
                }
                CsrWriteTask::Data { id, values, blocks } => {
                    encode_matrix_values_into(values, &mut workspace.chunk)?;
                    let encoded = self.compressor.encode_partitioned(
                        &workspace.chunk,
                        data_size,
                        blocks.as_ref(),
                    )?;
                    store.write_value(&chunk_key(DATA_DIR, id), &encoded)
                }
            },
        )?;

        let indptr_bytes = u64_slice_to_le_bytes(&indptr)?;
        let encoded_indptr = self
            .indptr_compressor
            .encode_buffer(&indptr_bytes, DType::U64.size())?;
        let store = transaction.store_mut();
        store.write(INDPTR_FILE, &encoded_indptr)?;

        let meta = MetaFile::csr(CsrMeta {
            shape: [n_cells as u64, n_genes],
            nnz,
            partition: PartitionMeta {
                chunk: self.chunk,
                block: self.block,
            },
            indptr: ArrayMeta::new(INDPTR_FILE, DType::U64, self.indptr_compressor),
            indices: ArrayMeta::new(INDICES_DIR, index_dtype, self.compressor.clone()),
            data: ArrayMeta::new(DATA_DIR, value_dtype, self.compressor),
            chunks: ChunkGridMeta::from_cell_starts(offsets),
        });
        write_meta(store, &meta)?;
        transaction.commit()
    }
}

/// Opened CSR matrix backed by any [`ByteStore`].
#[derive(Clone)]
pub struct CsrMatrix {
    store: Arc<dyn ByteStore>,
    meta: CsrMeta,
    /// Shared indptr so consumers (e.g. scdata) can pin without copying.
    indptr: Arc<[u64]>,
    limits: ReadLimits,
}

/// Open a CSR matrix from a directory or a zip prefix.
pub fn open_csr(location: impl Into<StoreLocation>) -> Result<CsrMatrix> {
    open_csr_with_limits(location, ReadLimits::default())
}

/// Open a CSR matrix with explicit resource limits.
pub fn open_csr_with_limits(
    location: impl Into<StoreLocation>,
    limits: ReadLimits,
) -> Result<CsrMatrix> {
    let store = location.into().open()?;
    CsrMatrix::from_store_with_limits(store, limits)
}

impl CsrMatrix {
    pub fn from_store(store: Arc<dyn ByteStore>) -> Result<Self> {
        Self::from_store_with_limits(store, ReadLimits::default())
    }

    pub fn from_store_with_limits(store: Arc<dyn ByteStore>, limits: ReadLimits) -> Result<Self> {
        let limits = limits.validate()?;
        let file = read_meta(store.as_ref(), limits)?;
        let MetaBody::Csr(meta) = file.into_body() else {
            return Err(Error::invalid_meta("store kind is not csr"));
        };
        let (indptr, _encoded_len) = load_indptr(store.as_ref(), &meta, limits)?;
        Ok(Self::from_parts(store, meta, indptr, limits))
    }

    /// Construct from already-validated metadata and indptr (single meta read path).
    pub(crate) fn from_parts(
        store: Arc<dyn ByteStore>,
        meta: CsrMeta,
        indptr: Arc<[u64]>,
        limits: ReadLimits,
    ) -> Self {
        Self {
            store,
            meta,
            indptr,
            limits,
        }
    }

    pub fn meta(&self) -> &CsrMeta {
        &self.meta
    }

    pub fn indptr(&self) -> &[u64] {
        &self.indptr
    }

    /// Cheap shared handle to the resident indptr (no element copy).
    pub fn indptr_shared(&self) -> Arc<[u64]> {
        Arc::clone(&self.indptr)
    }

    #[must_use]
    pub fn store(&self) -> &Arc<dyn ByteStore> {
        &self.store
    }

    pub fn limits(&self) -> ReadLimits {
        self.limits
    }

    /// Consume into store + meta + shared indptr + limits without copying indptr.
    pub fn into_parts(self) -> (Arc<dyn ByteStore>, CsrMeta, Arc<[u64]>, ReadLimits) {
        (self.store, self.meta, self.indptr, self.limits)
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

    pub fn nnz(&self) -> u64 {
        self.meta.nnz
    }

    pub fn index_dtype(&self) -> DType {
        self.meta.indices.dtype
    }

    pub fn value_dtype(&self) -> DType {
        self.meta.data.dtype
    }

    pub fn decode_all(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        self.decode_rows(0..self.n_rows())
    }

    /// Decode indices and values for a half-open row range.
    pub fn decode_rows(&self, rows: Range<u64>) -> Result<(Vec<u8>, Vec<u8>)> {
        let (_, indices, data) = self.decode_selected_rows(&NormalizedAxis::Contiguous {
            start: rows.start,
            end: rows.end,
        })?;
        Ok((indices, data))
    }

    pub(crate) fn decode_selected_rows(
        &self,
        rows: &NormalizedAxis,
    ) -> Result<(Vec<u64>, Vec<u8>, Vec<u8>)> {
        let DecodedCsrIndices {
            indptr,
            indices,
            nnz_requests,
        } = self.decode_selected_indices(rows)?;
        let n_nnz = indptr.last().copied().unwrap_or(0);
        let data_size = self.meta.data.dtype.size();
        let data_len = checked_meta_byte_len(n_nnz, data_size, "data output")?;
        let source_indptr_len = self.source_indptr_resident()?;
        let output_indptr_len = resident_bytes::<u64>(indptr.len(), "selected indptr")?;
        let resident_decoded = self.limits.check_decoded_sum(
            [
                source_indptr_len,
                output_indptr_len,
                indices.len(),
                data_len,
            ],
            "csr selected resident output",
        )?;
        let mut data = zeroed_vec(data_len)?;
        if n_nnz == 0 {
            return Ok((indptr, indices, data));
        }

        // Index decoding and bounds validation finish before any data request
        // is issued, so corrupt indices cannot trigger value I/O.
        let data_requests = scale_csr_requests(&self.meta.data.path, &nnz_requests, data_size)?;
        decode_blosc_scatter_into(
            &self.meta.data.compressor,
            RangeDecodeContext::new(self.store.as_ref(), resident_decoded, self.limits),
            &data_requests,
            &mut data,
        )?;

        Ok((indptr, indices, data))
    }

    fn decode_selected_indices(&self, rows: &NormalizedAxis) -> Result<DecodedCsrIndices> {
        rows.validate(self.n_rows())?;
        let indptr = selected_indptr(self, rows)?;
        let n_nnz = indptr.last().copied().unwrap_or(0);
        let index_size = self.meta.indices.dtype.size();
        let indices_len = checked_meta_byte_len(n_nnz, index_size, "indices output")?;
        let source_indptr_len = self.source_indptr_resident()?;
        let output_indptr_len = resident_bytes::<u64>(indptr.len(), "selected indptr")?;
        let resident_decoded = self.limits.check_decoded_sum(
            [source_indptr_len, output_indptr_len, indices_len],
            "csr selected indices resident output",
        )?;
        let mut indices = zeroed_vec(indices_len)?;
        let nnz_requests = plan_csr_nnz_requests(self, rows, &indptr)?;
        if n_nnz != 0 {
            let requests = scale_csr_requests(&self.meta.indices.path, &nnz_requests, index_size)?;
            decode_blosc_scatter_into(
                &self.meta.indices.compressor,
                RangeDecodeContext::new(self.store.as_ref(), resident_decoded, self.limits),
                &requests,
                &mut indices,
            )?;
            if let Some((position, value)) =
                first_out_of_bounds_index(&indices, self.index_dtype(), self.n_cols())
            {
                return Err(Error::corrupt(
                    "csr indices",
                    format!(
                        "index at decoded position {position} is {value}, outside 0..{}",
                        self.n_cols()
                    ),
                ));
            }
        }
        Ok(DecodedCsrIndices {
            indptr,
            indices,
            nnz_requests,
        })
    }

    pub(crate) fn decode_selection(
        &self,
        rows: &NormalizedAxis,
        cols: &NormalizedAxis,
        output: CsrOutput,
    ) -> Result<SelectedArray> {
        rows.validate(self.n_rows())?;
        cols.validate(self.n_cols())?;
        let n_rows = usize::try_from(rows.len())
            .map_err(|_| Error::invalid_argument("selected row count exceeds usize"))?;
        let n_cols = usize::try_from(cols.len())
            .map_err(|_| Error::invalid_argument("selected column count exceeds usize"))?;
        if n_rows == 0 || n_cols == 0 {
            return match output {
                CsrOutput::Sparse => Ok(SelectedArray::Csr(CsrArray::empty(
                    [n_rows, n_cols],
                    output_index_dtype(n_cols)?,
                    self.value_dtype(),
                )?)),
                CsrOutput::Dense => Ok(SelectedArray::Dense(DenseArray::zeros(
                    [n_rows, n_cols],
                    self.value_dtype(),
                )?)),
            };
        }

        let DecodedCsrIndices {
            indptr,
            indices,
            nnz_requests: _,
        } = self.decode_selected_indices(rows)?;
        let SelectedColumnPlan { layout, mappings } = plan_selected_columns(
            &indptr,
            &indices,
            cols,
            SelectedColumnContext {
                index_dtype: self.index_dtype(),
                source_n_cols: self.n_cols(),
                output,
                additional_resident: self.source_indptr_resident()?,
                limits: self.limits,
            },
        )?;
        let nnz_requests = plan_selected_data_requests(self, rows, &indptr, mappings)?;
        let data_requests = scale_csr_requests(
            &self.meta.data.path,
            &nnz_requests,
            self.value_dtype().size(),
        )?;
        drop(nnz_requests);
        let request_resident = scatter_requests_resident(&data_requests)?;
        drop(indices);
        drop(indptr);

        let source_indptr_len = self.source_indptr_resident()?;
        let value_size = self.value_dtype().size();
        match layout {
            SelectedColumnLayout::Sparse {
                indptr,
                index_dtype,
                indices,
            } => {
                let n_nnz = indptr.last().copied().unwrap_or(0);
                let data_len = checked_meta_byte_len(n_nnz, value_size, "selected data")?;
                let resident = self.limits.check_decoded_sum(
                    [
                        source_indptr_len,
                        resident_bytes::<u64>(indptr.len(), "selected output indptr")?,
                        indices.len(),
                        data_len,
                        request_resident,
                    ],
                    "CSR direct sparse selection resident output",
                )?;
                let mut data = zeroed_vec(data_len)?;
                decode_blosc_scatter_into(
                    &self.meta.data.compressor,
                    RangeDecodeContext::new(self.store.as_ref(), resident, self.limits),
                    &data_requests,
                    &mut data,
                )?;
                Ok(SelectedArray::Csr(CsrArray::from_parts_validated(
                    [n_rows, n_cols],
                    index_dtype,
                    self.value_dtype(),
                    indptr,
                    indices,
                    data,
                )))
            }
            SelectedColumnLayout::Dense => {
                let output_len = n_rows
                    .checked_mul(n_cols)
                    .and_then(|elements| elements.checked_mul(value_size))
                    .ok_or_else(|| Error::invalid_argument("dense selection size overflow"))?;
                let resident = self.limits.check_decoded_sum(
                    [source_indptr_len, output_len, request_resident],
                    "CSR direct dense selection resident output",
                )?;
                let mut values = zeroed_vec(output_len)?;
                decode_blosc_scatter_into(
                    &self.meta.data.compressor,
                    RangeDecodeContext::new(self.store.as_ref(), resident, self.limits),
                    &data_requests,
                    &mut values,
                )?;
                Ok(SelectedArray::Dense(DenseArray::from_bytes(
                    [n_rows, n_cols],
                    self.value_dtype(),
                    values,
                )?))
            }
        }
    }

    fn source_indptr_resident(&self) -> Result<usize> {
        resident_bytes::<u64>(self.indptr.len(), "csr indptr")
    }
}

struct DecodedCsrIndices {
    indptr: Vec<u64>,
    indices: Vec<u8>,
    nnz_requests: Vec<NnzScatterRequest>,
}

fn selected_indptr(matrix: &CsrMatrix, rows: &NormalizedAxis) -> Result<Vec<u64>> {
    let output_len = usize::try_from(rows.len())
        .map_err(|_| Error::invalid_argument("selected row count exceeds usize"))?
        .checked_add(1)
        .ok_or_else(|| Error::invalid_argument("selected indptr length overflow"))?;
    let mut output = Vec::new();
    output.try_reserve_exact(output_len)?;
    output.push(0);
    match rows {
        NormalizedAxis::Contiguous { start, end } => {
            let start = usize::try_from(*start)
                .map_err(|_| Error::invalid_argument("row start exceeds usize"))?;
            let end = usize::try_from(*end)
                .map_err(|_| Error::invalid_argument("row end exceeds usize"))?;
            let base = matrix.indptr[start];
            for &offset in &matrix.indptr[start + 1..=end] {
                output.push(offset - base);
            }
        }
        NormalizedAxis::Gather { positions } => {
            for &position in positions {
                let row = usize::try_from(position)
                    .map_err(|_| Error::invalid_argument("row position exceeds usize"))?;
                let row_nnz = matrix.indptr[row + 1] - matrix.indptr[row];
                let next = output
                    .last()
                    .copied()
                    .unwrap_or(0u64)
                    .checked_add(row_nnz)
                    .ok_or_else(|| Error::invalid_argument("selected CSR nnz overflow"))?;
                output.push(next);
            }
        }
    }
    Ok(output)
}

struct NnzScatterRequest {
    chunk: usize,
    expected_nnz: u64,
    mappings: Vec<(Range<u64>, Range<u64>)>,
}

#[derive(Debug, Clone, Copy)]
struct SelectedValueMapping {
    source_position: usize,
    destination_position: usize,
}

enum SelectedColumnLayout {
    Sparse {
        indptr: Vec<u64>,
        index_dtype: DType,
        indices: Vec<u8>,
    },
    Dense,
}

struct SelectedColumnPlan {
    layout: SelectedColumnLayout,
    mappings: Vec<SelectedValueMapping>,
}

#[derive(Clone, Copy)]
struct SelectedColumnContext {
    index_dtype: DType,
    source_n_cols: u64,
    output: CsrOutput,
    additional_resident: usize,
    limits: ReadLimits,
}

fn plan_selected_columns(
    indptr: &[u64],
    indices: &[u8],
    cols: &NormalizedAxis,
    context: SelectedColumnContext,
) -> Result<SelectedColumnPlan> {
    let SelectedColumnContext {
        index_dtype,
        source_n_cols,
        output,
        additional_resident,
        limits,
    } = context;
    let n_rows = indptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| Error::invalid_argument("selected CSR indptr is empty"))?;
    let index_size = index_dtype.size();
    let nnz = usize::try_from(indptr.last().copied().unwrap_or(0))
        .map_err(|_| Error::invalid_argument("selected CSR nnz exceeds usize"))?;
    let expected_indices = nnz
        .checked_mul(index_size)
        .ok_or_else(|| Error::invalid_argument("selected CSR indices size overflow"))?;
    if indptr.first() != Some(&0)
        || indptr.windows(2).any(|window| window[1] < window[0])
        || indices.len() != expected_indices
    {
        return Err(Error::invalid_argument("invalid selected CSR index layout"));
    }
    cols.validate(source_n_cols)?;
    let source_n_cols = usize::try_from(source_n_cols)
        .map_err(|_| Error::invalid_argument("CSR column count exceeds usize"))?;
    let row_counts_bytes = resident_bytes::<usize>(n_rows, "selected row counts")?;
    let gather_upper_bound = match cols {
        NormalizedAxis::Gather { positions } => positions
            .len()
            .checked_mul(std::mem::size_of::<(u64, u32)>())
            .ok_or_else(|| Error::invalid_argument("column gather map size overflow"))?,
        NormalizedAxis::Contiguous { .. } => 0,
    };
    limits.check_decoded_sum(
        [
            additional_resident,
            resident_bytes::<u64>(indptr.len(), "selected source indptr")?,
            indices.len(),
            row_counts_bytes,
            gather_upper_bound,
        ],
        "CSR direct column planning preflight",
    )?;
    let gather = match cols {
        NormalizedAxis::Gather { positions } => Some(GatherColumns::new(source_n_cols, positions)?),
        NormalizedAxis::Contiguous { .. } => None,
    };
    let col_range = match cols {
        NormalizedAxis::Contiguous { start, end } => Some((*start, *end)),
        NormalizedAxis::Gather { .. } => None,
    };

    let mut row_counts = Vec::new();
    row_counts.try_reserve_exact(n_rows)?;
    let mut total = 0usize;
    for row in 0..n_rows {
        let start = usize::try_from(indptr[row])
            .map_err(|_| Error::invalid_argument("selected row start exceeds usize"))?;
        let end = usize::try_from(indptr[row + 1])
            .map_err(|_| Error::invalid_argument("selected row end exceeds usize"))?;
        let count = if let Some((first_col, past_last_col)) = col_range {
            let first = lower_bound_decoded_index(indices, start, end, index_size, first_col);
            let past_last =
                lower_bound_decoded_index(indices, first, end, index_size, past_last_col);
            past_last - first
        } else {
            let gather = gather
                .as_ref()
                .expect("gather lookup exists for non-contiguous columns");
            let mut count = 0usize;
            for position in start..end {
                // SAFETY: the exact packed index length and indptr bounds were
                // validated above, so every row position contains one index.
                let source = unsafe { read_index_unchecked(indices, position, index_size) };
                count = count
                    .checked_add(gather.destinations(source).len())
                    .ok_or_else(|| Error::invalid_argument("selected CSR nnz overflow"))?;
            }
            count
        };
        total = total
            .checked_add(count)
            .ok_or_else(|| Error::invalid_argument("selected CSR nnz overflow"))?;
        row_counts.push(count);
    }

    let n_out_cols = usize::try_from(cols.len())
        .map_err(|_| Error::invalid_argument("selected column count exceeds usize"))?;
    let sparse_index_dtype = if output == CsrOutput::Sparse {
        output_index_dtype(n_out_cols)?
    } else {
        DType::U16
    };
    let sparse_index_size = sparse_index_dtype.size();
    let mapping_bytes = total
        .checked_mul(
            std::mem::size_of::<SelectedValueMapping>()
                + std::mem::size_of::<(Range<u64>, Range<u64>)>()
                + std::mem::size_of::<ScatterMapping>(),
        )
        .ok_or_else(|| Error::invalid_argument("selected mapping size overflow"))?;
    let gather_bytes = gather
        .as_ref()
        .map_or(Ok(0), GatherColumns::resident_bytes)?;
    let scratch_entries = row_counts.iter().copied().max().unwrap_or(0);
    let scratch_bytes = scratch_entries
        .checked_mul(std::mem::size_of::<(u32, usize)>())
        .ok_or_else(|| Error::invalid_argument("selected row scratch size overflow"))?;
    let layout_bytes = match output {
        CsrOutput::Sparse => resident_bytes::<u64>(indptr.len(), "selected output indptr")?
            .checked_add(
                total
                    .checked_mul(sparse_index_size)
                    .ok_or_else(|| Error::invalid_argument("selected indices size overflow"))?,
            )
            .ok_or_else(|| Error::invalid_argument("selected sparse layout size overflow"))?,
        CsrOutput::Dense => 0,
    };
    limits.check_decoded_sum(
        [
            additional_resident,
            resident_bytes::<u64>(indptr.len(), "selected source indptr")?,
            indices.len(),
            row_counts_bytes,
            mapping_bytes,
            gather_bytes,
            scratch_bytes,
            layout_bytes,
        ],
        "CSR direct column planning working set",
    )?;

    let mut mappings = Vec::new();
    mappings.try_reserve_exact(total)?;
    let mut sparse_indptr = Vec::new();
    let mut sparse_indices = Vec::new();
    if output == CsrOutput::Sparse {
        sparse_indptr.try_reserve_exact(n_rows + 1)?;
        sparse_indptr.push(0);
        for &count in &row_counts {
            let count = u64::try_from(count)
                .map_err(|_| Error::invalid_argument("selected row nnz exceeds u64"))?;
            sparse_indptr.push(
                sparse_indptr
                    .last()
                    .copied()
                    .unwrap_or(0u64)
                    .checked_add(count)
                    .ok_or_else(|| Error::invalid_argument("selected CSR nnz overflow"))?,
            );
        }
        sparse_indices = zeroed_vec(
            total
                .checked_mul(sparse_index_size)
                .ok_or_else(|| Error::invalid_argument("selected indices size overflow"))?,
        )?;
    }

    let mut output_cursor = 0usize;
    let mut gather_entries = Vec::new();
    for row in 0..n_rows {
        let start = usize::try_from(indptr[row])
            .map_err(|_| Error::invalid_argument("selected row start exceeds usize"))?;
        let end = usize::try_from(indptr[row + 1])
            .map_err(|_| Error::invalid_argument("selected row end exceeds usize"))?;
        if let Some((first_col, past_last_col)) = col_range {
            let first = lower_bound_decoded_index(indices, start, end, index_size, first_col);
            let past_last =
                lower_bound_decoded_index(indices, first, end, index_size, past_last_col);
            for position in first..past_last {
                // SAFETY: both lower bounds lie inside this validated row.
                let source = unsafe { read_index_unchecked(indices, position, index_size) };
                let destination_col = usize::try_from(source - first_col)
                    .map_err(|_| Error::invalid_argument("selected column exceeds usize"))?;
                let destination_position = match output {
                    CsrOutput::Sparse => {
                        write_index(
                            &mut sparse_indices,
                            output_cursor,
                            sparse_index_size,
                            destination_col as u64,
                        )?;
                        output_cursor
                    }
                    CsrOutput::Dense => row
                        .checked_mul(n_out_cols)
                        .and_then(|base| base.checked_add(destination_col))
                        .ok_or_else(|| Error::invalid_argument("dense destination overflow"))?,
                };
                mappings.push(SelectedValueMapping {
                    source_position: position,
                    destination_position,
                });
                if output == CsrOutput::Sparse {
                    output_cursor += 1;
                }
            }
        } else {
            let gather = gather
                .as_ref()
                .expect("gather lookup exists for non-contiguous columns");
            gather_entries.clear();
            gather_entries.try_reserve(row_counts[row])?;
            for position in start..end {
                // SAFETY: the exact packed index length and row bounds were
                // validated before the scan.
                let source = unsafe { read_index_unchecked(indices, position, index_size) };
                for &(_, destination) in gather.destinations(source) {
                    gather_entries.push((destination, position));
                }
            }
            if output == CsrOutput::Sparse {
                gather_entries.sort_unstable_by_key(|&(destination, _)| destination);
            }
            for &(destination_col, source_position) in &gather_entries {
                let destination_col = destination_col as usize;
                let destination_position = match output {
                    CsrOutput::Sparse => {
                        write_index(
                            &mut sparse_indices,
                            output_cursor,
                            sparse_index_size,
                            destination_col as u64,
                        )?;
                        output_cursor
                    }
                    CsrOutput::Dense => row
                        .checked_mul(n_out_cols)
                        .and_then(|base| base.checked_add(destination_col))
                        .ok_or_else(|| Error::invalid_argument("dense destination overflow"))?,
                };
                mappings.push(SelectedValueMapping {
                    source_position,
                    destination_position,
                });
                if output == CsrOutput::Sparse {
                    output_cursor += 1;
                }
            }
        }
    }
    debug_assert_eq!(mappings.len(), total);
    debug_assert!(output != CsrOutput::Sparse || output_cursor == total);

    let layout = match output {
        CsrOutput::Sparse => SelectedColumnLayout::Sparse {
            indptr: sparse_indptr,
            index_dtype: sparse_index_dtype,
            indices: sparse_indices,
        },
        CsrOutput::Dense => SelectedColumnLayout::Dense,
    };
    Ok(SelectedColumnPlan { layout, mappings })
}

#[inline]
fn lower_bound_decoded_index(
    indices: &[u8],
    mut start: usize,
    mut end: usize,
    index_size: usize,
    target: u64,
) -> usize {
    debug_assert!(end
        .checked_mul(index_size)
        .is_some_and(|bytes| bytes <= indices.len()));
    while start < end {
        let middle = start + (end - start) / 2;
        // SAFETY: `middle < end` and the caller supplies a validated packed row.
        if unsafe { read_index_unchecked(indices, middle, index_size) } < target {
            start = middle + 1;
        } else {
            end = middle;
        }
    }
    start
}

fn plan_selected_data_requests(
    matrix: &CsrMatrix,
    rows: &NormalizedAxis,
    selected_indptr: &[u64],
    mappings: impl IntoIterator<Item = SelectedValueMapping>,
) -> Result<Vec<NnzScatterRequest>> {
    let n_selected_rows = selected_indptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| Error::invalid_argument("selected CSR indptr is empty"))?;
    let mut requests = Vec::new();
    requests.try_reserve_exact(matrix.meta.chunks.n_chunks())?;
    requests.resize_with(matrix.meta.chunks.n_chunks(), || None);
    let mut selected_row = 0usize;

    for mapping in mappings {
        while selected_row < n_selected_rows
            && mapping.source_position
                >= usize::try_from(selected_indptr[selected_row + 1])
                    .map_err(|_| Error::invalid_argument("selected CSR row end exceeds usize"))?
        {
            selected_row += 1;
        }
        if selected_row >= n_selected_rows {
            return Err(Error::invalid_argument(
                "selected value position is outside selected indptr",
            ));
        }
        let selected_row_start = usize::try_from(selected_indptr[selected_row])
            .map_err(|_| Error::invalid_argument("selected CSR row start exceeds usize"))?;
        if mapping.source_position < selected_row_start {
            return Err(Error::invalid_argument(
                "selected value positions are not grouped by row",
            ));
        }
        let source_row = normalized_axis_position(rows, selected_row)?;
        let source_row_usize = usize::try_from(source_row)
            .map_err(|_| Error::invalid_argument("source row exceeds usize"))?;
        let chunk = matrix.meta.chunks.chunk_of(source_row)?;
        let (chunk_row_start, chunk_row_end) =
            matrix.meta.chunks.cell_range(chunk, matrix.n_rows())?;
        let chunk_row_start = usize::try_from(chunk_row_start)
            .map_err(|_| Error::invalid_meta("chunk row start exceeds usize"))?;
        let chunk_row_end = usize::try_from(chunk_row_end)
            .map_err(|_| Error::invalid_meta("chunk row end exceeds usize"))?;
        let chunk_nnz_start = matrix.indptr[chunk_row_start];
        let chunk_nnz_end = matrix.indptr[chunk_row_end];
        let row_offset = mapping.source_position - selected_row_start;
        let row_offset = u64::try_from(row_offset)
            .map_err(|_| Error::invalid_argument("source row offset exceeds u64"))?;
        let source_global = matrix.indptr[source_row_usize]
            .checked_add(row_offset)
            .ok_or_else(|| Error::invalid_argument("source nnz position overflow"))?;
        if source_global >= matrix.indptr[source_row_usize + 1] {
            return Err(Error::invalid_argument(
                "selected value position exceeds its source row",
            ));
        }
        let source = source_global - chunk_nnz_start;
        let source_end = source
            .checked_add(1)
            .ok_or_else(|| Error::invalid_argument("source nnz range overflow"))?;
        let destination = u64::try_from(mapping.destination_position)
            .map_err(|_| Error::invalid_argument("destination nnz position exceeds u64"))?;
        let destination_end = destination
            .checked_add(1)
            .ok_or_else(|| Error::invalid_argument("destination nnz range overflow"))?;
        let request = requests[chunk].get_or_insert_with(|| NnzScatterRequest {
            chunk,
            expected_nnz: chunk_nnz_end - chunk_nnz_start,
            mappings: Vec::new(),
        });
        push_nnz_mapping(
            &mut request.mappings,
            source..source_end,
            destination..destination_end,
        )?;
    }

    Ok(requests.into_iter().flatten().collect())
}

fn normalized_axis_position(axis: &NormalizedAxis, position: usize) -> Result<u64> {
    match axis {
        NormalizedAxis::Contiguous { start, end } => {
            let position = u64::try_from(position)
                .map_err(|_| Error::invalid_argument("row position exceeds u64"))?;
            let source = start
                .checked_add(position)
                .ok_or_else(|| Error::invalid_argument("row position overflow"))?;
            if source >= *end {
                return Err(Error::invalid_argument("row position is out of bounds"));
            }
            Ok(source)
        }
        NormalizedAxis::Gather { positions } => positions
            .get(position)
            .copied()
            .ok_or_else(|| Error::invalid_argument("row position is out of bounds")),
    }
}

fn plan_csr_nnz_requests(
    matrix: &CsrMatrix,
    rows: &NormalizedAxis,
    output_indptr: &[u64],
) -> Result<Vec<NnzScatterRequest>> {
    match rows {
        NormalizedAxis::Contiguous { start, end } => {
            let start_row = usize::try_from(*start)
                .map_err(|_| Error::invalid_argument("row start exceeds usize"))?;
            let output_base = matrix.indptr[start_row];
            let mut requests = Vec::new();
            for chunk in matrix.meta.chunks.overlapping_chunks(*start, *end) {
                let (chunk_row_start, chunk_row_end) =
                    matrix.meta.chunks.cell_range(chunk, matrix.n_rows())?;
                let chunk_start = usize::try_from(chunk_row_start)
                    .map_err(|_| Error::invalid_meta("chunk row start exceeds usize"))?;
                let chunk_end = usize::try_from(chunk_row_end)
                    .map_err(|_| Error::invalid_meta("chunk row end exceeds usize"))?;
                let chunk_nnz_start = matrix.indptr[chunk_start];
                let chunk_nnz_end = matrix.indptr[chunk_end];
                let overlap_row_start = usize::try_from((*start).max(chunk_row_start))
                    .map_err(|_| Error::invalid_meta("overlap row start exceeds usize"))?;
                let overlap_row_end = usize::try_from((*end).min(chunk_row_end))
                    .map_err(|_| Error::invalid_meta("overlap row end exceeds usize"))?;
                let source_start = matrix.indptr[overlap_row_start] - chunk_nnz_start;
                let source_end = matrix.indptr[overlap_row_end] - chunk_nnz_start;
                if source_start == source_end {
                    continue;
                }
                let destination_start = matrix.indptr[overlap_row_start] - output_base;
                let destination_end = matrix.indptr[overlap_row_end] - output_base;
                let mut mappings = Vec::new();
                mappings.try_reserve_exact(1)?;
                mappings.push((source_start..source_end, destination_start..destination_end));
                requests.try_reserve(1)?;
                requests.push(NnzScatterRequest {
                    chunk,
                    expected_nnz: chunk_nnz_end - chunk_nnz_start,
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
                let chunk = items[cursor].0;
                let (chunk_row_start, chunk_row_end) =
                    matrix.meta.chunks.cell_range(chunk, matrix.n_rows())?;
                let chunk_start = usize::try_from(chunk_row_start)
                    .map_err(|_| Error::invalid_meta("chunk row start exceeds usize"))?;
                let chunk_end = usize::try_from(chunk_row_end)
                    .map_err(|_| Error::invalid_meta("chunk row end exceeds usize"))?;
                let chunk_nnz_start = matrix.indptr[chunk_start];
                let chunk_nnz_end = matrix.indptr[chunk_end];
                let mut mappings = Vec::new();
                while cursor < items.len() && items[cursor].0 == chunk {
                    let (_, source, destination) = items[cursor];
                    let source_row = usize::try_from(source)
                        .map_err(|_| Error::invalid_argument("row position exceeds usize"))?;
                    let source_start = matrix.indptr[source_row] - chunk_nnz_start;
                    let source_end = matrix.indptr[source_row + 1] - chunk_nnz_start;
                    let destination_start = output_indptr[destination];
                    let destination_end = output_indptr[destination + 1];
                    push_nnz_mapping(
                        &mut mappings,
                        source_start..source_end,
                        destination_start..destination_end,
                    )?;
                    cursor += 1;
                }
                if !mappings.is_empty() {
                    requests.try_reserve(1)?;
                    requests.push(NnzScatterRequest {
                        chunk,
                        expected_nnz: chunk_nnz_end - chunk_nnz_start,
                        mappings,
                    });
                }
            }
            Ok(requests)
        }
    }
}

fn push_nnz_mapping(
    mappings: &mut Vec<(Range<u64>, Range<u64>)>,
    source: Range<u64>,
    destination: Range<u64>,
) -> Result<()> {
    if source.is_empty() {
        return Ok(());
    }
    if let Some((previous_source, previous_destination)) = mappings.last_mut() {
        if previous_source.end == source.start && previous_destination.end == destination.start {
            previous_source.end = source.end;
            previous_destination.end = destination.end;
            return Ok(());
        }
    }
    mappings.try_reserve(1)?;
    mappings.push((source, destination));
    Ok(())
}

fn scale_csr_requests(
    path: &str,
    requests: &[NnzScatterRequest],
    element_size: usize,
) -> Result<Vec<BloscScatterRequest>> {
    let mut output = Vec::new();
    output.try_reserve_exact(requests.len())?;
    for request in requests {
        let mut mappings = Vec::new();
        mappings.try_reserve_exact(request.mappings.len())?;
        for (source, destination) in &request.mappings {
            mappings.push(ScatterMapping {
                source: checked_meta_byte_len(source.start, element_size, "CSR source")?
                    ..checked_meta_byte_len(source.end, element_size, "CSR source")?,
                destination: checked_meta_byte_len(
                    destination.start,
                    element_size,
                    "CSR destination",
                )?
                    ..checked_meta_byte_len(destination.end, element_size, "CSR destination")?,
            });
        }
        output.push(BloscScatterRequest {
            key: chunk_key(path, request.chunk as u64),
            expected: checked_meta_byte_len(request.expected_nnz, element_size, "CSR chunk")?,
            mappings,
        });
    }
    Ok(output)
}

fn checked_byte_len(count: u64, element_size: usize, context: &str) -> Result<usize> {
    let element_size = u64::try_from(element_size)
        .map_err(|_| Error::invalid_argument(format!("{context} element size exceeds u64")))?;
    let bytes = count
        .checked_mul(element_size)
        .ok_or_else(|| Error::invalid_argument(format!("{context} byte length overflow")))?;
    usize::try_from(bytes)
        .map_err(|_| Error::invalid_argument(format!("{context} byte length exceeds usize")))
}

fn resident_bytes<T>(count: usize, context: &str) -> Result<usize> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::invalid_argument(format!("{context} resident size overflow")))
}

fn scatter_requests_resident(requests: &[BloscScatterRequest]) -> Result<usize> {
    requests.iter().try_fold(
        resident_bytes::<BloscScatterRequest>(requests.len(), "scatter requests")?,
        |resident, request| {
            resident
                .checked_add(request.key.len())
                .and_then(|resident| {
                    resident.checked_add(
                        request
                            .mappings
                            .len()
                            .checked_mul(std::mem::size_of::<ScatterMapping>())?,
                    )
                })
                .ok_or_else(|| Error::invalid_argument("scatter request resident size overflow"))
        },
    )
}

/// Decode and validate CSR indptr. Returns `(indptr, encoded_byte_len)`.
pub(crate) fn load_indptr(
    store: &dyn ByteStore,
    meta: &CsrMeta,
    limits: ReadLimits,
) -> Result<(Arc<[u64]>, u64)> {
    let indptr_len = meta.shape[0]
        .checked_add(1)
        .ok_or_else(|| Error::invalid_meta("csr indptr length overflow"))?;
    let expected_bytes = checked_meta_byte_len(indptr_len, DType::U64.size(), "indptr")?;
    limits.check_decoded(expected_bytes, "csr indptr")?;
    let encoded = store.read_limited(&meta.indptr.path, limits.encoded_size())?;
    let encoded_len = encoded.len() as u64;
    limits.check_decoded_sum(
        [encoded.len(), expected_bytes],
        "csr indptr decode working set",
    )?;
    let decoded =
        meta.indptr
            .compressor
            .decode_exact_with_limits(&encoded, expected_bytes, limits)?;
    drop(encoded);
    limits.check_decoded_sum(
        [expected_bytes, expected_bytes],
        "csr indptr conversion working set",
    )?;
    let indptr = u64_slice_from_le_bytes(&decoded)?;
    drop(decoded);
    validate_indptr(&indptr).map_err(|error| Error::invalid_meta(error.to_string()))?;
    if indptr.last().copied() != Some(meta.nnz) {
        return Err(Error::invalid_meta(format!(
            "csr indptr ends at {}, metadata declares nnz {}",
            indptr.last().copied().unwrap_or(0),
            meta.nnz
        )));
    }
    Ok((Arc::from(indptr), encoded_len))
}

fn checked_meta_byte_len(count: u64, element_size: usize, context: &str) -> Result<usize> {
    checked_byte_len(count, element_size, context)
        .map_err(|error| Error::invalid_meta(error.to_string()))
}

fn zeroed_vec(len: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output.try_reserve_exact(len)?;
    output.resize(len, 0);
    Ok(output)
}

fn first_out_of_bounds_index(indices: &[u8], dtype: DType, n_cols: u64) -> Option<(usize, u64)> {
    match dtype {
        DType::U16 => indices
            .chunks_exact(2)
            .map(|bytes| u64::from(u16::from_le_bytes([bytes[0], bytes[1]])))
            .enumerate()
            .find(|(_, value)| *value >= n_cols),
        DType::U32 => indices
            .chunks_exact(4)
            .map(|bytes| u64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])))
            .enumerate()
            .find(|(_, value)| *value >= n_cols),
        DType::U64 | DType::I16 | DType::I32 | DType::F32 | DType::F64 => None,
    }
}

/// Sort each CSR row by column index and reject duplicates.
fn canonicalize_csr_rows<'a, V: MatrixValue>(
    indptr: &[u64],
    indices: &mut [u64],
    data: &'a [V],
) -> Result<Cow<'a, [V]>> {
    let mut requires_sort = false;
    for window in indptr.windows(2) {
        let start = usize::try_from(window[0])
            .map_err(|_| Error::invalid_argument("csr row start exceeds usize"))?;
        let end = usize::try_from(window[1])
            .map_err(|_| Error::invalid_argument("csr row end exceeds usize"))?;
        let row_indices = indices
            .get(start..end)
            .ok_or_else(|| Error::invalid_argument("csr row exceeds indices"))?;
        requires_sort |= !row_indices.windows(2).all(|pair| pair[0] < pair[1]);
    }
    if !requires_sort {
        return Ok(Cow::Borrowed(data));
    }

    let mut owned_data = Vec::new();
    owned_data.try_reserve_exact(data.len())?;
    owned_data.extend_from_slice(data);
    let mut order = Vec::new();
    let mut scratch_indices = Vec::new();
    let mut scratch_data = Vec::new();

    for (row, window) in indptr.windows(2).enumerate() {
        let start = usize::try_from(window[0])
            .map_err(|_| Error::invalid_argument("csr row start exceeds usize"))?;
        let end = usize::try_from(window[1])
            .map_err(|_| Error::invalid_argument("csr row end exceeds usize"))?;
        let n = end
            .checked_sub(start)
            .ok_or_else(|| Error::invalid_argument("csr row range is not monotonic"))?;
        if n <= 1 {
            continue;
        }

        let row_indices = indices
            .get(start..end)
            .ok_or_else(|| Error::invalid_argument("csr row exceeds indices"))?;
        if row_indices.windows(2).all(|pair| pair[0] < pair[1]) {
            continue;
        }

        order.clear();
        if order.capacity() < n {
            order.try_reserve_exact(n - order.capacity())?;
        }
        order.extend(
            row_indices
                .iter()
                .copied()
                .enumerate()
                .map(|(local, index)| (index, local)),
        );
        order.sort_unstable_by_key(|&(idx, _)| idx);

        for pair in order.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(Error::invalid_argument(format!(
                    "csr row {row} has duplicate column index {}",
                    pair[0].0
                )));
            }
        }

        if scratch_indices.capacity() < n {
            scratch_indices.try_reserve_exact(n - scratch_indices.capacity())?;
        }
        scratch_indices.resize(n, 0);
        if scratch_data.capacity() < n {
            scratch_data.try_reserve_exact(n - scratch_data.capacity())?;
        }
        scratch_data.clear();

        let row_data = owned_data
            .get(start..end)
            .ok_or_else(|| Error::invalid_argument("csr row exceeds data"))?;
        for (new_pos, &(idx, old_pos)) in order.iter().enumerate() {
            let value = *row_data
                .get(old_pos)
                .ok_or_else(|| Error::invalid_argument("csr source exceeds row data"))?;
            scratch_indices[new_pos] = idx;
            scratch_data.push(value);
        }
        indices
            .get_mut(start..end)
            .ok_or_else(|| Error::invalid_argument("csr row exceeds indices"))?
            .copy_from_slice(&scratch_indices);
        owned_data
            .get_mut(start..end)
            .ok_or_else(|| Error::invalid_argument("csr row exceeds data"))?
            .copy_from_slice(&scratch_data);
    }
    Ok(Cow::Owned(owned_data))
}
