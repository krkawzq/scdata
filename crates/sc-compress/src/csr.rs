use std::borrow::Cow;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use crate::array::{CsrArray, DenseArray, SelectedArray};
use crate::codec::Compressor;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::io_util::{read_meta, u64_slice_from_le_bytes, u64_slice_to_le_bytes, write_meta};
use crate::kernel::{
    output_index_dtype, read_index_unchecked, write_index_unchecked, GatherColumns,
};
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
use crate::select::{visit_run_chunks, AxisRun, CsrOutput, NormalizedAxis};
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
    /// Create a writer. Callers must pass chunk/block partitions explicitly.
    pub fn new(dir: impl AsRef<Path>, chunk: Partition, block: Partition) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            chunk,
            block,
            compressor: Compressor::dyn_blosc_lz4(),
            indptr_compressor: Compressor::zstd(3),
            threads: default_threads(),
        }
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
    /// Shared indptr so consumers (e.g. sc-load) can pin without copying.
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
        let decoded = self.decode_selected_indices_unvalidated(rows)?;
        validate_decoded_csr_indices(
            &decoded.indptr,
            &decoded.indices,
            self.index_dtype(),
            self.n_cols(),
        )?;
        Ok(decoded)
    }

    fn decode_selected_indices_unvalidated(
        &self,
        rows: &NormalizedAxis,
    ) -> Result<DecodedCsrIndices> {
        rows.validate(self.n_rows())?;
        let indptr = selected_indptr(self, rows)?;
        let n_nnz = indptr.last().copied().unwrap_or(0);
        let index_size = self.meta.indices.dtype.size();
        let indices_len = checked_meta_byte_len(n_nnz, index_size, "indices output")?;
        let source_indptr_len = self.source_indptr_resident()?;
        let output_indptr_len = resident_bytes::<u64>(indptr.len(), "selected indptr")?;
        let base_resident = self.limits.check_decoded_sum(
            [source_indptr_len, output_indptr_len, indices_len],
            "csr selected indices resident output",
        )?;
        let mut indices = zeroed_vec(indices_len)?;
        let nnz_requests = if n_nnz == 0 {
            Vec::new()
        } else {
            let (request_count, mapping_count) = csr_nnz_plan_upper_bound(self, rows)?;
            let plan_resident =
                csr_nnz_plan_upper_resident(request_count, mapping_count, &self.meta.indices.path)?;
            self.limits.check_decoded_sum(
                [base_resident, plan_resident],
                "csr selected resident output",
            )?;
            plan_csr_nnz_requests(self, rows, &indptr)?
        };
        if n_nnz != 0 {
            let requests = scale_csr_requests(&self.meta.indices.path, &nnz_requests, index_size)?;
            let resident_decoded = self.limits.check_decoded_sum(
                [
                    base_resident,
                    nnz_requests_resident(&nnz_requests)?,
                    scatter_requests_resident(&requests)?,
                ],
                "csr selected resident output",
            )?;
            decode_blosc_scatter_into(
                &self.meta.indices.compressor,
                RangeDecodeContext::new(self.store.as_ref(), resident_decoded, self.limits),
                &requests,
                &mut indices,
            )?;
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
        } = self.decode_selected_indices_unvalidated(rows)?;
        let SelectedColumnPlan {
            layout,
            data_requests,
        } = plan_selected_columns(
            self,
            rows,
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
    rows.visit_runs(|run| append_selected_indptr_run(matrix, run, &mut output))?;
    Ok(output)
}

fn append_selected_indptr_run(
    matrix: &CsrMatrix,
    run: AxisRun,
    output: &mut Vec<u64>,
) -> Result<()> {
    if run.source_step == 1 {
        let start = usize::try_from(run.source)
            .map_err(|_| Error::invalid_argument("row start exceeds usize"))?;
        let count = usize::try_from(run.count)
            .map_err(|_| Error::invalid_argument("selected row count exceeds usize"))?;
        let end = start
            .checked_add(count)
            .ok_or_else(|| Error::invalid_argument("selected row end overflow"))?;
        let base_out = output.last().copied().unwrap_or(0);
        let base_in = matrix.indptr[start];
        for &offset in &matrix.indptr[start + 1..=end] {
            output.push(
                base_out
                    .checked_add(offset - base_in)
                    .ok_or_else(|| Error::invalid_argument("selected CSR nnz overflow"))?,
            );
        }
        return Ok(());
    }
    for index in 0..run.count {
        let row = usize::try_from(run.nth(index)?)
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
    Ok(())
}

struct NnzScatterRequest {
    chunk: usize,
    expected_nnz: u64,
    mappings: Vec<(Range<u64>, Range<u64>)>,
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
    data_requests: Vec<BloscScatterRequest>,
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
    matrix: &CsrMatrix,
    rows: &NormalizedAxis,
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
        NormalizedAxis::Strided { len, .. } => usize::try_from(*len)
            .ok()
            .and_then(|len| len.checked_mul(std::mem::size_of::<(u64, u32)>()))
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
    let gather = column_gather(source_n_cols, cols)?;
    let col_range = cols.as_range().map(|range| (range.start, range.end));

    let mut row_counts = Vec::new();
    row_counts.try_reserve_exact(n_rows)?;
    row_counts.resize(n_rows, 0usize);
    let threads = limits.thread_count().max(1);
    let job = 64usize;
    let job_count = n_rows.div_ceil(job);
    let mut remaining = row_counts.as_mut_slice();
    let mut row_start = 0usize;
    parallel::try_for_each_stream(
        threads,
        job_count,
        |emit| {
            while row_start < n_rows {
                let row_end = (row_start + job).min(n_rows);
                let tail = std::mem::take(&mut remaining);
                let (block, tail) = tail.split_at_mut(row_end - row_start);
                remaining = tail;
                emit((row_start, block))?;
                row_start = row_end;
            }
            Ok(())
        },
        |(start, block)| {
            for (offset, slot) in block.iter_mut().enumerate() {
                let row = start + offset;
                let row_start = usize::try_from(indptr[row])
                    .map_err(|_| Error::invalid_argument("selected row start exceeds usize"))?;
                let row_end = usize::try_from(indptr[row + 1])
                    .map_err(|_| Error::invalid_argument("selected row end exceeds usize"))?;
                *slot = count_selected_columns_checked(
                    indices,
                    row_start,
                    row_end,
                    index_size,
                    source_n_cols as u64,
                    col_range,
                    gather.as_ref(),
                )?;
            }
            Ok(())
        },
    )?;
    let total = row_counts.iter().try_fold(0usize, |total, &count| {
        total
            .checked_add(count)
            .ok_or_else(|| Error::invalid_argument("selected CSR nnz overflow"))
    })?;

    let n_out_cols = usize::try_from(cols.len())
        .map_err(|_| Error::invalid_argument("selected column count exceeds usize"))?;
    let sparse_index_dtype = if output == CsrOutput::Sparse {
        output_index_dtype(n_out_cols)?
    } else {
        DType::U16
    };
    let sparse_index_size = sparse_index_dtype.size();
    let mapping_upper_bound = if output == CsrOutput::Sparse && col_range.is_some() {
        row_counts.iter().filter(|&&count| count != 0).count()
    } else {
        total
    };
    let mapping_bytes = mapping_upper_bound
        .checked_mul(2)
        .and_then(|capacity| capacity.checked_mul(std::mem::size_of::<ScatterMapping>()))
        .ok_or_else(|| Error::invalid_argument("selected mapping size overflow"))?;
    let active_request_upper_bound = row_counts
        .iter()
        .filter(|&&count| count != 0)
        .count()
        .min(matrix.meta.chunks.n_chunks());
    let request_entry_bytes = std::mem::size_of::<(usize, BloscScatterRequest)>()
        .checked_mul(2)
        .and_then(|bytes| {
            bytes.checked_add(
                matrix
                    .meta
                    .data
                    .path
                    .len()
                    .saturating_add(1)
                    .saturating_add(20),
            )
        })
        .ok_or_else(|| Error::invalid_argument("selected request size overflow"))?;
    let request_bytes = active_request_upper_bound
        .checked_mul(request_entry_bytes)
        .ok_or_else(|| Error::invalid_argument("selected request size overflow"))?;
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
            request_bytes,
            gather_bytes,
            scratch_bytes,
            layout_bytes,
        ],
        "CSR direct column planning working set",
    )?;

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

    let value_size = matrix.value_dtype().size();
    let mut requests = BTreeMap::<usize, BloscScatterRequest>::new();
    let mut output_cursor = 0usize;
    let mut gather_entries = Vec::new();
    for row in 0..n_rows {
        let start = usize::try_from(indptr[row])
            .map_err(|_| Error::invalid_argument("selected row start exceeds usize"))?;
        let end = usize::try_from(indptr[row + 1])
            .map_err(|_| Error::invalid_argument("selected row end exceeds usize"))?;
        if row_counts[row] == 0 {
            continue;
        }
        let source_row = normalized_axis_position(rows, row)?;
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
        let request = match requests.entry(chunk) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(BloscScatterRequest {
                key: chunk_key(&matrix.meta.data.path, chunk as u64),
                expected: checked_meta_byte_len(
                    chunk_nnz_end - chunk_nnz_start,
                    value_size,
                    "CSR chunk",
                )?,
                mappings: Vec::new(),
            }),
        };
        let row_data = SelectedDataRow {
            selected_start: start,
            selected_end: end,
            source_start: matrix.indptr[source_row_usize],
            source_end: matrix.indptr[source_row_usize + 1],
            chunk_start: chunk_nnz_start,
            value_size,
        };
        if let Some((first_col, past_last_col)) = col_range {
            let first = lower_bound_decoded_index(indices, start, end, index_size, first_col);
            let past_last =
                lower_bound_decoded_index(indices, first, end, index_size, past_last_col);
            match output {
                CsrOutput::Sparse => {
                    let destination_start = output_cursor;
                    for position in first..past_last {
                        // SAFETY: both lower bounds lie inside this validated row.
                        let source = unsafe { read_index_unchecked(indices, position, index_size) };
                        let destination_col =
                            usize::try_from(source - first_col).map_err(|_| {
                                Error::invalid_argument("selected column exceeds usize")
                            })?;
                        // SAFETY: `sparse_indices` was sized from the exact
                        // count pass, `output_cursor < total`, and the remapped
                        // column fits the selected output index dtype.
                        unsafe {
                            write_index_unchecked(
                                sparse_indices.as_mut_ptr(),
                                output_cursor,
                                sparse_index_size,
                                destination_col as u64,
                            );
                        }
                        output_cursor += 1;
                    }
                    row_data.push_mapping(
                        request,
                        first..past_last,
                        destination_start..output_cursor,
                    )?;
                }
                CsrOutput::Dense => {
                    let row_base = row
                        .checked_mul(n_out_cols)
                        .ok_or_else(|| Error::invalid_argument("dense destination overflow"))?;
                    for position in first..past_last {
                        // SAFETY: both lower bounds lie inside this validated row.
                        let source = unsafe { read_index_unchecked(indices, position, index_size) };
                        let destination_col =
                            usize::try_from(source - first_col).map_err(|_| {
                                Error::invalid_argument("selected column exceeds usize")
                            })?;
                        let destination = row_base
                            .checked_add(destination_col)
                            .ok_or_else(|| Error::invalid_argument("dense destination overflow"))?;
                        row_data.push_mapping(
                            request,
                            position..position + 1,
                            destination..destination + 1,
                        )?;
                    }
                }
            }
        } else {
            let gather = gather
                .as_ref()
                .expect("gather lookup exists for non-contiguous columns");
            gather_entries.clear();
            gather_entries.try_reserve(row_counts[row])?;
            gather.collect_hits(indices, start, end, index_size, &mut gather_entries)?;
            if output == CsrOutput::Sparse && !gather.destinations_are_ordered() {
                gather_entries.sort_unstable_by_key(|&(destination, _)| destination);
            }
            for &(destination_col, source_position) in &gather_entries {
                let destination_col = destination_col as usize;
                let destination_position = match output {
                    CsrOutput::Sparse => {
                        // SAFETY: `sparse_indices` was sized from the exact
                        // count pass, `output_cursor < total`, and destinations
                        // originate from validated selected-column positions.
                        unsafe {
                            write_index_unchecked(
                                sparse_indices.as_mut_ptr(),
                                output_cursor,
                                sparse_index_size,
                                destination_col as u64,
                            );
                        }
                        output_cursor
                    }
                    CsrOutput::Dense => row
                        .checked_mul(n_out_cols)
                        .and_then(|base| base.checked_add(destination_col))
                        .ok_or_else(|| Error::invalid_argument("dense destination overflow"))?,
                };
                row_data.push_mapping(
                    request,
                    source_position..source_position + 1,
                    destination_position..destination_position + 1,
                )?;
                if output == CsrOutput::Sparse {
                    output_cursor += 1;
                }
            }
        }
    }
    debug_assert!(output != CsrOutput::Sparse || output_cursor == total);

    let mut data_requests = Vec::new();
    data_requests.try_reserve_exact(requests.len())?;
    data_requests.extend(requests.into_values());

    let layout = match output {
        CsrOutput::Sparse => SelectedColumnLayout::Sparse {
            indptr: sparse_indptr,
            index_dtype: sparse_index_dtype,
            indices: sparse_indices,
        },
        CsrOutput::Dense => SelectedColumnLayout::Dense,
    };
    Ok(SelectedColumnPlan {
        layout,
        data_requests,
    })
}

fn count_selected_columns_checked(
    indices: &[u8],
    start: usize,
    end: usize,
    index_size: usize,
    source_n_cols: u64,
    col_range: Option<(u64, u64)>,
    gather: Option<&GatherColumns>,
) -> Result<usize> {
    if let Some((first_col, past_last_col)) = col_range {
        let mut count = 0usize;
        let mut previous = None;
        for position in start..end {
            // SAFETY: the caller supplies one complete validated packed row.
            let source = unsafe { read_index_unchecked(indices, position, index_size) };
            validate_decoded_index(position, source, previous, source_n_cols)?;
            previous = Some(source);
            count += usize::from(source >= first_col && source < past_last_col);
        }
        return Ok(count);
    }

    let gather = gather.expect("gather lookup exists for non-contiguous columns");
    if gather.prefer_binary_search(end.saturating_sub(start)) {
        validate_decoded_index_row(indices, start, end, index_size, source_n_cols)?;
        return gather.count_hits(indices, start, end, index_size);
    }
    let mut count = 0usize;
    let mut previous = None;
    for position in start..end {
        // SAFETY: the caller supplies one complete validated packed row.
        let source = unsafe { read_index_unchecked(indices, position, index_size) };
        validate_decoded_index(position, source, previous, source_n_cols)?;
        previous = Some(source);
        count = count
            .checked_add(gather.destinations(source).len())
            .ok_or_else(|| Error::invalid_argument("CSR selected nnz overflow"))?;
    }
    Ok(count)
}

fn validate_decoded_index_row(
    indices: &[u8],
    start: usize,
    end: usize,
    index_size: usize,
    source_n_cols: u64,
) -> Result<()> {
    let mut previous = None;
    for position in start..end {
        // SAFETY: the caller supplies one complete validated packed row.
        let source = unsafe { read_index_unchecked(indices, position, index_size) };
        validate_decoded_index(position, source, previous, source_n_cols)?;
        previous = Some(source);
    }
    Ok(())
}

fn validate_decoded_index(
    position: usize,
    source: u64,
    previous: Option<u64>,
    source_n_cols: u64,
) -> Result<()> {
    if source >= source_n_cols {
        return Err(Error::corrupt(
            "csr indices",
            format!("index at decoded position {position} is {source}, outside 0..{source_n_cols}"),
        ));
    }
    if previous.is_some_and(|previous| previous >= source) {
        return Err(Error::corrupt(
            "csr indices",
            format!(
                "indices at decoded position {position} are not strictly increasing within the row"
            ),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct SelectedDataRow {
    selected_start: usize,
    selected_end: usize,
    source_start: u64,
    source_end: u64,
    chunk_start: u64,
    value_size: usize,
}

impl SelectedDataRow {
    fn push_mapping(
        self,
        request: &mut BloscScatterRequest,
        source: Range<usize>,
        destination: Range<usize>,
    ) -> Result<()> {
        if source.is_empty() {
            return Ok(());
        }
        if source.start < self.selected_start
            || source.end > self.selected_end
            || source.len() != destination.len()
        {
            return Err(Error::invalid_argument(
                "selected data mapping is outside its CSR row",
            ));
        }
        let local_start = u64::try_from(source.start - self.selected_start)
            .map_err(|_| Error::invalid_argument("selected row offset exceeds u64"))?;
        let local_end = u64::try_from(source.end - self.selected_start)
            .map_err(|_| Error::invalid_argument("selected row offset exceeds u64"))?;
        let source_start = self
            .source_start
            .checked_add(local_start)
            .ok_or_else(|| Error::invalid_argument("source nnz position overflow"))?;
        let source_end = self
            .source_start
            .checked_add(local_end)
            .ok_or_else(|| Error::invalid_argument("source nnz position overflow"))?;
        if source_end > self.source_end || source_start < self.chunk_start {
            return Err(Error::invalid_argument(
                "selected data mapping exceeds its source CSR row",
            ));
        }
        let destination_start = u64::try_from(destination.start)
            .map_err(|_| Error::invalid_argument("destination position exceeds u64"))?;
        let destination_end = u64::try_from(destination.end)
            .map_err(|_| Error::invalid_argument("destination position exceeds u64"))?;
        push_scatter_mapping(
            &mut request.mappings,
            ScatterMapping {
                source: checked_meta_byte_len(
                    source_start - self.chunk_start,
                    self.value_size,
                    "CSR source",
                )?
                    ..checked_meta_byte_len(
                        source_end - self.chunk_start,
                        self.value_size,
                        "CSR source",
                    )?,
                destination: checked_meta_byte_len(
                    destination_start,
                    self.value_size,
                    "CSR destination",
                )?
                    ..checked_meta_byte_len(destination_end, self.value_size, "CSR destination")?,
            },
        )
    }
}

fn push_scatter_mapping(mappings: &mut Vec<ScatterMapping>, mapping: ScatterMapping) -> Result<()> {
    if let Some(previous) = mappings.last_mut() {
        if previous.source.end == mapping.source.start
            && previous.destination.end == mapping.destination.start
        {
            previous.source.end = mapping.source.end;
            previous.destination.end = mapping.destination.end;
            return Ok(());
        }
    }
    mappings.try_reserve(1)?;
    mappings.push(mapping);
    Ok(())
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

fn column_gather(n_cols: usize, cols: &NormalizedAxis) -> Result<Option<GatherColumns>> {
    match cols {
        NormalizedAxis::Contiguous { .. } => Ok(None),
        NormalizedAxis::Gather { positions } => Ok(Some(GatherColumns::new(n_cols, positions)?)),
        NormalizedAxis::Strided { .. } => {
            let positions = cols.to_positions();
            Ok(Some(GatherColumns::new(n_cols, &positions)?))
        }
    }
}

fn normalized_axis_position(axis: &NormalizedAxis, position: usize) -> Result<u64> {
    axis.nth(
        u64::try_from(position).map_err(|_| Error::invalid_argument("row position exceeds u64"))?,
    )
}

fn csr_nnz_plan_upper_bound(matrix: &CsrMatrix, rows: &NormalizedAxis) -> Result<(usize, usize)> {
    let mut request_count = 0usize;
    let mut mapping_count = 0usize;
    rows.visit_runs(|run| {
        visit_run_chunks(
            run,
            |row| matrix.meta.chunks.chunk_of(row),
            |chunk| matrix.meta.chunks.cell_range(chunk, matrix.n_rows()),
            |_, subrun| {
                request_count = request_count
                    .checked_add(1)
                    .ok_or_else(|| Error::invalid_argument("CSR request count overflow"))?;
                let mappings = if subrun.source_step == 1 {
                    1
                } else {
                    usize::try_from(subrun.count)
                        .map_err(|_| Error::invalid_argument("CSR mapping count exceeds usize"))?
                };
                mapping_count = mapping_count
                    .checked_add(mappings)
                    .ok_or_else(|| Error::invalid_argument("CSR mapping count overflow"))?;
                Ok(())
            },
        )
    })?;
    Ok((
        request_count.min(matrix.meta.chunks.n_chunks()),
        mapping_count,
    ))
}

fn csr_nnz_plan_upper_resident(
    request_count: usize,
    mapping_count: usize,
    path: &str,
) -> Result<usize> {
    let request_bytes = std::mem::size_of::<NnzScatterRequest>()
        .checked_add(std::mem::size_of::<BloscScatterRequest>())
        .and_then(|bytes| bytes.checked_add(4 * std::mem::size_of::<usize>()))
        .and_then(|bytes| bytes.checked_add(path.len().saturating_add(21)))
        .and_then(|bytes| bytes.checked_mul(request_count))
        .ok_or_else(|| Error::invalid_argument("CSR request plan size overflow"))?;
    let mapping_bytes = std::mem::size_of::<(Range<u64>, Range<u64>)>()
        .checked_add(std::mem::size_of::<ScatterMapping>())
        .and_then(|bytes| bytes.checked_mul(mapping_count))
        .ok_or_else(|| Error::invalid_argument("CSR mapping plan size overflow"))?;
    request_bytes
        .checked_add(mapping_bytes)
        .ok_or_else(|| Error::invalid_argument("CSR request plan resident size overflow"))
}

fn plan_csr_nnz_requests(
    matrix: &CsrMatrix,
    rows: &NormalizedAxis,
    output_indptr: &[u64],
) -> Result<Vec<NnzScatterRequest>> {
    let mut requests = BTreeMap::<usize, NnzScatterRequest>::new();
    rows.visit_runs(|run| {
        visit_run_chunks(
            run,
            |row| matrix.meta.chunks.chunk_of(row),
            |chunk| matrix.meta.chunks.cell_range(chunk, matrix.n_rows()),
            |chunk, subrun| {
                append_csr_run_nnz_request(matrix, output_indptr, &mut requests, chunk, subrun)
            },
        )
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(requests.len())?;
    output.extend(requests.into_values());
    Ok(output)
}

fn append_csr_run_nnz_request(
    matrix: &CsrMatrix,
    output_indptr: &[u64],
    requests: &mut BTreeMap<usize, NnzScatterRequest>,
    chunk: usize,
    run: AxisRun,
) -> Result<()> {
    let (chunk_row_start, chunk_row_end) = matrix.meta.chunks.cell_range(chunk, matrix.n_rows())?;
    let chunk_start = usize::try_from(chunk_row_start)
        .map_err(|_| Error::invalid_meta("chunk row start exceeds usize"))?;
    let chunk_end = usize::try_from(chunk_row_end)
        .map_err(|_| Error::invalid_meta("chunk row end exceeds usize"))?;
    let chunk_nnz_start = matrix.indptr[chunk_start];
    let chunk_nnz_end = matrix.indptr[chunk_end];
    if run.source_step == 1 {
        let start = usize::try_from(run.source)
            .map_err(|_| Error::invalid_argument("row start exceeds usize"))?;
        let count = usize::try_from(run.count)
            .map_err(|_| Error::invalid_argument("selected row count exceeds usize"))?;
        let end = start
            .checked_add(count)
            .ok_or_else(|| Error::invalid_argument("selected row end overflow"))?;
        let destination = usize::try_from(run.destination)
            .map_err(|_| Error::invalid_argument("destination row exceeds usize"))?;
        return push_chunk_nnz_mapping(
            requests,
            chunk,
            chunk_nnz_end - chunk_nnz_start,
            (matrix.indptr[start] - chunk_nnz_start)..(matrix.indptr[end] - chunk_nnz_start),
            output_indptr[destination]..output_indptr[destination + count],
        );
    }
    for index in 0..run.count {
        let source_row = usize::try_from(run.nth(index)?)
            .map_err(|_| Error::invalid_argument("row position exceeds usize"))?;
        let destination = usize::try_from(
            run.destination
                .checked_add(index)
                .ok_or_else(|| Error::invalid_argument("destination row overflow"))?,
        )
        .map_err(|_| Error::invalid_argument("destination row exceeds usize"))?;
        push_chunk_nnz_mapping(
            requests,
            chunk,
            chunk_nnz_end - chunk_nnz_start,
            (matrix.indptr[source_row] - chunk_nnz_start)
                ..(matrix.indptr[source_row + 1] - chunk_nnz_start),
            output_indptr[destination]..output_indptr[destination + 1],
        )?;
    }
    Ok(())
}

fn push_chunk_nnz_mapping(
    requests: &mut BTreeMap<usize, NnzScatterRequest>,
    chunk: usize,
    expected_nnz: u64,
    source: Range<u64>,
    destination: Range<u64>,
) -> Result<()> {
    if source.is_empty() {
        return Ok(());
    }
    let request = match requests.entry(chunk) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(NnzScatterRequest {
            chunk,
            expected_nnz,
            mappings: Vec::new(),
        }),
    };
    push_nnz_mapping(&mut request.mappings, source, destination)
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
                .checked_add(request.key.capacity())
                .and_then(|resident| {
                    resident.checked_add(
                        request
                            .mappings
                            .capacity()
                            .checked_mul(std::mem::size_of::<ScatterMapping>())?,
                    )
                })
                .ok_or_else(|| Error::invalid_argument("scatter request resident size overflow"))
        },
    )
}

fn nnz_requests_resident(requests: &[NnzScatterRequest]) -> Result<usize> {
    requests.iter().try_fold(
        resident_bytes::<NnzScatterRequest>(requests.len(), "CSR nnz requests")?,
        |resident, request| {
            resident
                .checked_add(
                    request
                        .mappings
                        .capacity()
                        .checked_mul(std::mem::size_of::<(Range<u64>, Range<u64>)>())
                        .ok_or_else(|| {
                            Error::invalid_argument("CSR nnz request mapping size overflow")
                        })?,
                )
                .ok_or_else(|| Error::invalid_argument("CSR nnz request resident size overflow"))
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

fn validate_decoded_csr_indices(
    indptr: &[u64],
    indices: &[u8],
    dtype: DType,
    n_cols: u64,
) -> Result<()> {
    if !dtype.is_csr_index() {
        return Err(Error::corrupt(
            "csr indices",
            format!("invalid CSR index dtype {dtype}"),
        ));
    }
    let index_size = dtype.size();
    for bounds in indptr.windows(2) {
        let start = usize::try_from(bounds[0])
            .map_err(|_| Error::corrupt("csr indices", "row start exceeds usize"))?;
        let end = usize::try_from(bounds[1])
            .map_err(|_| Error::corrupt("csr indices", "row end exceeds usize"))?;
        validate_decoded_index_row(indices, start, end, index_size, n_cols)?;
    }
    Ok(())
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
