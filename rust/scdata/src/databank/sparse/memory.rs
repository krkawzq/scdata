use super::*;

pub(crate) fn try_scatter_sparse_single_memory_chunk_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<bool> {
    let Some(index_bytes) = single_memory_identity_chunk_bytes(&dataset.indices)? else {
        return Ok(false);
    };
    let Some(data_bytes) = single_memory_identity_chunk_bytes(&dataset.data)? else {
        return Ok(false);
    };
    let src_dtype = dataset.data.dtype;

    match dataset.index_dtype {
        DType::U32 => scatter_sparse_single_memory_chunk_checked_typed::<T, u32>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_bytes,
            data_bytes,
            src_dtype,
            out,
        )?,
        DType::I32 => scatter_sparse_single_memory_chunk_checked_typed::<T, i32>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_bytes,
            data_bytes,
            src_dtype,
            out,
        )?,
        DType::U64 => scatter_sparse_single_memory_chunk_checked_typed::<T, u64>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_bytes,
            data_bytes,
            src_dtype,
            out,
        )?,
        DType::I64 => scatter_sparse_single_memory_chunk_checked_typed::<T, i64>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_bytes,
            data_bytes,
            src_dtype,
            out,
        )?,
        dtype => {
            return Err(DataBankError::UnsupportedDType {
                dtype,
                context: "CSR indices",
            });
        }
    }
    Ok(true)
}

pub(crate) unsafe fn try_scatter_sparse_single_memory_chunk_unchecked<T: DataValue>(
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<bool> {
    let Some(index_bytes) = single_memory_identity_chunk_bytes(&dataset.indices)? else {
        return Ok(false);
    };
    let Some(data_bytes) = single_memory_identity_chunk_bytes(&dataset.data)? else {
        return Ok(false);
    };
    let src_dtype = dataset.data.dtype;

    match dataset.index_dtype {
        DType::U32 => unsafe {
            scatter_sparse_single_memory_chunk_unchecked_typed::<T, u32>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_bytes,
                data_bytes,
                src_dtype,
                out,
            );
        },
        DType::I32 => unsafe {
            scatter_sparse_single_memory_chunk_unchecked_typed::<T, i32>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_bytes,
                data_bytes,
                src_dtype,
                out,
            );
        },
        DType::U64 => unsafe {
            scatter_sparse_single_memory_chunk_unchecked_typed::<T, u64>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_bytes,
                data_bytes,
                src_dtype,
                out,
            );
        },
        DType::I64 => unsafe {
            scatter_sparse_single_memory_chunk_unchecked_typed::<T, i64>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_bytes,
                data_bytes,
                src_dtype,
                out,
            );
        },
        _ => unreachable!("CSR index dtype was validated during registration"),
    }
    Ok(true)
}

pub(crate) fn single_memory_identity_chunk_bytes(array: &Array) -> DataBankResult<Option<&[u8]>> {
    let Some(chunks) = array.memory_chunks() else {
        return Ok(None);
    };
    let [chunk] = chunks else {
        return Ok(None);
    };
    let ChunkSource::Memory { bytes, decoded } = &chunk.source else {
        return Ok(None);
    };
    if !(*decoded || array.codec.is_identity()) {
        return Ok(None);
    }
    let expected_size = chunk.decoded_bytes;
    if bytes.len() != expected_size {
        return Err(CodecError::SizeMismatch {
            codec: array.codec.name().to_string(),
            expected: expected_size,
            actual: bytes.len(),
        }
        .into());
    }
    Ok(Some(bytes.as_ref()))
}

#[derive(Clone, Copy)]
pub(crate) struct MemoryIdentity1DChunks<'a> {
    pub(crate) array: &'a Array,
    pub(crate) chunks: &'a [Chunk],
    pub(crate) chunk_len: usize,
    pub(crate) len: usize,
    pub(crate) item_size: usize,
}

impl<'a> MemoryIdentity1DChunks<'a> {
    pub(crate) fn from_array(array: &'a Array) -> DataBankResult<Option<Self>> {
        let Some(chunks) = array.memory_chunks() else {
            return Ok(None);
        };
        let chunks_are_decoded = chunks
            .iter()
            .all(|chunk| matches!(&chunk.source, ChunkSource::Memory { decoded: true, .. }));
        if !(array.codec.is_identity() || chunks_are_decoded) {
            return Ok(None);
        }
        let [len] = array.shape.as_slice() else {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "1D direct memory path requires 1D array, got shape {:?}",
                array.shape
            )));
        };
        let Some(chunk_len) = array.regular_1d_chunk_len() else {
            return Ok(None);
        };
        Ok(Some(Self {
            array,
            chunks,
            chunk_len,
            len: *len,
            item_size: array.dtype.item_size(),
        }))
    }

    pub(crate) fn chunk_bytes(self, chunk_index: usize) -> DataBankResult<&'a [u8]> {
        let Some(chunk) = self.chunks.get(chunk_index) else {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "1D memory chunk index {chunk_index} is out of range"
            )));
        };
        let ChunkSource::Memory { bytes, .. } = &chunk.source else {
            return Err(DataBankError::InvalidArrayMeta(
                "non-memory chunk reached memory direct path".to_string(),
            ));
        };
        let expected_size = self.expected_chunk_bytes(chunk_index)?;
        if bytes.len() != expected_size {
            return Err(CodecError::SizeMismatch {
                codec: self.array.codec.name().to_string(),
                expected: expected_size,
                actual: bytes.len(),
            }
            .into());
        }
        Ok(bytes.as_ref())
    }

    pub(crate) fn expected_chunk_bytes(self, chunk_index: usize) -> DataBankResult<usize> {
        let chunk_start = chunk_index.checked_mul(self.chunk_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("1D memory chunk start overflow".to_string())
        })?;
        if chunk_start >= self.len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "1D memory chunk {chunk_index} starts past array length {}",
                self.len
            )));
        }
        self.chunks
            .get(chunk_index)
            .map(|chunk| chunk.decoded_bytes)
            .ok_or_else(|| {
                DataBankError::InvalidArrayMeta(format!(
                    "1D memory chunk index {chunk_index} is out of range"
                ))
            })
    }

    pub(crate) fn physical_chunk_len_at_start(self, chunk_start: usize) -> DataBankResult<usize> {
        if chunk_start >= self.len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "1D memory chunk start {chunk_start} is out of range for length {}",
                self.len
            )));
        }
        Ok(self.chunk_len.min(self.len - chunk_start))
    }
}

pub(crate) fn try_scatter_sparse_memory_chunks_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<bool> {
    let Some(index_chunks) = MemoryIdentity1DChunks::from_array(&dataset.indices)? else {
        return Ok(false);
    };
    let Some(data_chunks) = MemoryIdentity1DChunks::from_array(&dataset.data)? else {
        return Ok(false);
    };
    let src_dtype = dataset.data.dtype;

    match dataset.index_dtype {
        DType::U32 => scatter_sparse_memory_chunks_checked_typed::<T, u32>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_chunks,
            data_chunks,
            src_dtype,
            out,
        )?,
        DType::I32 => scatter_sparse_memory_chunks_checked_typed::<T, i32>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_chunks,
            data_chunks,
            src_dtype,
            out,
        )?,
        DType::U64 => scatter_sparse_memory_chunks_checked_typed::<T, u64>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_chunks,
            data_chunks,
            src_dtype,
            out,
        )?,
        DType::I64 => scatter_sparse_memory_chunks_checked_typed::<T, i64>(
            dataset.num_genes,
            &dataset.indptr,
            cells,
            index_chunks,
            data_chunks,
            src_dtype,
            out,
        )?,
        dtype => {
            return Err(DataBankError::UnsupportedDType {
                dtype,
                context: "CSR indices",
            });
        }
    }
    Ok(true)
}

pub(crate) unsafe fn try_scatter_sparse_memory_chunks_unchecked<T: DataValue>(
    dataset: &SparseCsrDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<bool> {
    let Some(index_chunks) = MemoryIdentity1DChunks::from_array(&dataset.indices)? else {
        return Ok(false);
    };
    let Some(data_chunks) = MemoryIdentity1DChunks::from_array(&dataset.data)? else {
        return Ok(false);
    };
    let src_dtype = dataset.data.dtype;

    match dataset.index_dtype {
        DType::U32 => unsafe {
            scatter_sparse_memory_chunks_unchecked_typed::<T, u32>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_chunks,
                data_chunks,
                src_dtype,
                out,
            )?;
        },
        DType::I32 => unsafe {
            scatter_sparse_memory_chunks_unchecked_typed::<T, i32>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_chunks,
                data_chunks,
                src_dtype,
                out,
            )?;
        },
        DType::U64 => unsafe {
            scatter_sparse_memory_chunks_unchecked_typed::<T, u64>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_chunks,
                data_chunks,
                src_dtype,
                out,
            )?;
        },
        DType::I64 => unsafe {
            scatter_sparse_memory_chunks_unchecked_typed::<T, i64>(
                dataset.num_genes,
                &dataset.indptr,
                cells,
                index_chunks,
                data_chunks,
                src_dtype,
                out,
            )?;
        },
        _ => unreachable!("CSR index dtype was validated during registration"),
    }
    Ok(true)
}

pub(crate) fn scatter_sparse_memory_chunks_checked_typed<T, I>(
    num_genes: usize,
    indptr: &[u64],
    cells: &[usize],
    index_chunks: MemoryIdentity1DChunks<'_>,
    data_chunks: MemoryIdentity1DChunks<'_>,
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    if mem::size_of::<I>() != index_chunks.item_size
        || src_dtype.item_size() != data_chunks.item_size
    {
        return Err(DataBankError::InvalidArrayMeta(
            "CSR direct memory dtype size mismatch".to_string(),
        ));
    }

    for (output_row, &cell) in cells.iter().enumerate() {
        let start = usize::try_from(indptr[cell]).map_err(|_| {
            DataBankError::IndptrInvalid("CSR row start does not fit in usize".to_string())
        })?;
        let end = usize::try_from(indptr[cell + 1]).map_err(|_| {
            DataBankError::IndptrInvalid("CSR row end does not fit in usize".to_string())
        })?;
        if end < start || end > index_chunks.len || end > data_chunks.len {
            return Err(DataBankError::IndptrInvalid(
                "CSR row range is invalid for memory chunks".to_string(),
            ));
        }
        scatter_sparse_memory_chunked_row_checked_typed::<T, I>(
            num_genes,
            output_row,
            start,
            end,
            index_chunks,
            data_chunks,
            src_dtype,
            out,
        )?;
    }
    Ok(())
}

unsafe fn scatter_sparse_memory_chunks_unchecked_typed<T, I>(
    num_genes: usize,
    indptr: &[u64],
    cells: &[usize],
    index_chunks: MemoryIdentity1DChunks<'_>,
    data_chunks: MemoryIdentity1DChunks<'_>,
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    debug_assert_eq!(mem::size_of::<I>(), index_chunks.item_size);
    debug_assert_eq!(src_dtype.item_size(), data_chunks.item_size);

    for (output_row, &cell) in cells.iter().enumerate() {
        let start = unsafe { *indptr.get_unchecked(cell) as usize };
        let end = unsafe { *indptr.get_unchecked(cell + 1) as usize };
        unsafe {
            scatter_sparse_memory_chunked_row_unchecked_typed::<T, I>(
                num_genes,
                output_row,
                start,
                end,
                index_chunks,
                data_chunks,
                src_dtype,
                out,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_sparse_memory_chunked_row_checked_typed<T, I>(
    num_genes: usize,
    output_row: usize,
    start: usize,
    end: usize,
    index_chunks: MemoryIdentity1DChunks<'_>,
    data_chunks: MemoryIdentity1DChunks<'_>,
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    let value_size = src_dtype.item_size();
    let mut pos = start;
    while pos < end {
        let (index_bytes, index_byte_start, index_rem) =
            sparse_memory_chunk_piece(index_chunks, pos, index_size)?;
        let (data_bytes, data_byte_start, data_rem) =
            sparse_memory_chunk_piece(data_chunks, pos, value_size)?;
        let elements = (end - pos).min(index_rem).min(data_rem);
        let index_bytes_len = elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR direct index byte length overflow".to_string())
        })?;
        let data_bytes_len = elements.checked_mul(value_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR direct data byte length overflow".to_string())
        })?;
        let index_byte_end = index_byte_start
            .checked_add(index_bytes_len)
            .ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR direct index byte end overflow".to_string())
            })?;
        let data_byte_end = data_byte_start.checked_add(data_bytes_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR direct data byte end overflow".to_string())
        })?;
        scatter_sparse_values_checked_typed::<T, I>(
            num_genes,
            output_row,
            elements,
            &index_bytes[index_byte_start..index_byte_end],
            &data_bytes[data_byte_start..data_byte_end],
            src_dtype,
            out,
        )?;
        pos += elements;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn scatter_sparse_memory_chunked_row_unchecked_typed<T, I>(
    num_genes: usize,
    output_row: usize,
    start: usize,
    end: usize,
    index_chunks: MemoryIdentity1DChunks<'_>,
    data_chunks: MemoryIdentity1DChunks<'_>,
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    let value_size = src_dtype.item_size();
    let mut pos = start;
    while pos < end {
        let (index_bytes, index_byte_start, index_rem) =
            sparse_memory_chunk_piece(index_chunks, pos, index_size)?;
        let (data_bytes, data_byte_start, data_rem) =
            sparse_memory_chunk_piece(data_chunks, pos, value_size)?;
        let elements = (end - pos).min(index_rem).min(data_rem);
        let index_byte_end = index_byte_start + elements * index_size;
        let data_byte_end = data_byte_start + elements * value_size;
        unsafe {
            scatter_sparse_values_unchecked_typed::<T, I>(
                num_genes,
                output_row,
                elements,
                index_bytes.get_unchecked(index_byte_start..index_byte_end),
                data_bytes.get_unchecked(data_byte_start..data_byte_end),
                src_dtype,
                out,
            );
        }
        pos += elements;
    }
    Ok(())
}

pub(crate) fn sparse_memory_chunk_piece(
    chunks: MemoryIdentity1DChunks<'_>,
    pos: usize,
    item_size: usize,
) -> DataBankResult<(&[u8], usize, usize)> {
    let chunk_index = pos / chunks.chunk_len;
    let chunk_start = chunk_index.checked_mul(chunks.chunk_len).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR direct chunk start overflow".to_string())
    })?;
    let in_chunk = pos - chunk_start;
    let physical_chunk_len = chunks.physical_chunk_len_at_start(chunk_start)?;
    let remaining = physical_chunk_len.checked_sub(in_chunk).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR direct chunk cursor is out of range".to_string())
    })?;
    let byte_start = in_chunk.checked_mul(item_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR direct byte start overflow".to_string())
    })?;
    Ok((chunks.chunk_bytes(chunk_index)?, byte_start, remaining))
}

pub(crate) fn scatter_sparse_single_memory_chunk_checked_typed<T, I>(
    num_genes: usize,
    indptr: &[u64],
    cells: &[usize],
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    let value_size = src_dtype.item_size();
    for (output_row, &cell) in cells.iter().enumerate() {
        let start = usize::try_from(indptr[cell]).map_err(|_| {
            DataBankError::IndptrInvalid("CSR row start does not fit in usize".to_string())
        })?;
        let end = usize::try_from(indptr[cell + 1]).map_err(|_| {
            DataBankError::IndptrInvalid("CSR row end does not fit in usize".to_string())
        })?;
        let elements = end
            .checked_sub(start)
            .ok_or_else(|| DataBankError::IndptrInvalid("CSR row range is invalid".to_string()))?;
        let index_start = start.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index byte offset overflow".to_string())
        })?;
        let index_end = end.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index byte end overflow".to_string())
        })?;
        let data_start = start.checked_mul(value_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data byte offset overflow".to_string())
        })?;
        let data_end = end.checked_mul(value_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data byte end overflow".to_string())
        })?;
        if index_end > index_bytes.len() || data_end > data_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR single-chunk scatter is out of range".to_string(),
            ));
        }
        scatter_sparse_values_checked_typed::<T, I>(
            num_genes,
            output_row,
            elements,
            &index_bytes[index_start..index_end],
            &data_bytes[data_start..data_end],
            src_dtype,
            out,
        )?;
    }
    Ok(())
}

unsafe fn scatter_sparse_single_memory_chunk_unchecked_typed<T, I>(
    num_genes: usize,
    indptr: &[u64],
    cells: &[usize],
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    let value_size = src_dtype.item_size();
    for (output_row, &cell) in cells.iter().enumerate() {
        let start = unsafe { *indptr.get_unchecked(cell) as usize };
        let end = unsafe { *indptr.get_unchecked(cell + 1) as usize };
        let index_start = start * index_size;
        let index_end = end * index_size;
        let data_start = start * value_size;
        let data_end = end * value_size;
        unsafe {
            scatter_sparse_values_unchecked_typed::<T, I>(
                num_genes,
                output_row,
                end - start,
                index_bytes.get_unchecked(index_start..index_end),
                data_bytes.get_unchecked(data_start..data_end),
                src_dtype,
                out,
            );
        }
    }
}
