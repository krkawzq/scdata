use std::{mem, ptr, slice, sync::OnceLock};

use super::super::array::{Bf16Bits, DType, DataValue, F16Bits};
use super::super::error::{DataBankError, DataBankResult};
use super::super::gene_axis::GENE_NOT_SELECTED;
use super::{CsrIndex, SparseProjectionCtx};

pub(crate) fn scatter_sparse_values_projected_aot_checked<T, I>(
    projection: SparseProjectionCtx<'_>,
    output_row: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = src_dtype.item_size();
    let row_base = output_row
        .checked_mul(projection.output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output row overflow".to_string()))?;
    let row_end = row_base
        .checked_add(projection.output_genes)
        .ok_or_else(|| {
            DataBankError::InvalidConfig("sparse output row end overflow".to_string())
        })?;
    if row_end > out.len() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out.len(),
        });
    }
    let expected_data_bytes = elements.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR data byte length overflow".to_string())
    })?;
    if data_bytes.len() != expected_data_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_data_bytes,
            actual: data_bytes.len(),
        });
    }
    let index_size = mem::size_of::<I>();
    let expected_index_bytes = elements.checked_mul(index_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR index byte length overflow".to_string())
    })?;
    if index_bytes.len() != expected_index_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_index_bytes,
            actual: index_bytes.len(),
        });
    }
    if projection.projection.output_by_source.len() < projection.num_genes {
        return Err(DataBankError::InvalidArrayMeta(
            "gene projection is shorter than dataset gene count".to_string(),
        ));
    }

    macro_rules! dispatch_dst {
        ($dst:ty) => {{
            let out = unsafe { output_as_mut::<T, $dst>(out) };
            dispatch_source::<I, $dst>(
                projection,
                row_base,
                elements,
                index_bytes,
                data_bytes,
                src_dtype,
                out,
            )
        }};
    }

    match T::DTYPE {
        DType::U8 => dispatch_dst!(u8),
        DType::I8 => dispatch_dst!(i8),
        DType::U16 => dispatch_dst!(u16),
        DType::I16 => dispatch_dst!(i16),
        DType::U32 => dispatch_dst!(u32),
        DType::I32 => dispatch_dst!(i32),
        DType::U64 => dispatch_dst!(u64),
        DType::I64 => dispatch_dst!(i64),
        DType::F16 => dispatch_dst!(F16Bits),
        DType::BF16 => dispatch_dst!(Bf16Bits),
        DType::F32 => dispatch_dst!(f32),
        DType::F64 => dispatch_dst!(f64),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn scatter_sparse_values_projected_selected_aot_unchecked<T, I>(
    projection: SparseProjectionCtx<'_>,
    output_row: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    contiguous_output_start: Option<usize>,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = src_dtype.item_size();
    let row_base = output_row
        .checked_mul(projection.output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output row overflow".to_string()))?;
    let row_end = row_base
        .checked_add(projection.output_genes)
        .ok_or_else(|| {
            DataBankError::InvalidConfig("sparse output row end overflow".to_string())
        })?;
    if row_end > out.len() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out.len(),
        });
    }
    let expected_data_bytes = elements.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR data byte length overflow".to_string())
    })?;
    if data_bytes.len() != expected_data_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_data_bytes,
            actual: data_bytes.len(),
        });
    }
    let index_size = mem::size_of::<I>();
    let expected_index_bytes = elements.checked_mul(index_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR index byte length overflow".to_string())
    })?;
    if index_bytes.len() != expected_index_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_index_bytes,
            actual: index_bytes.len(),
        });
    }
    if projection.projection.output_by_source.len() < projection.num_genes {
        return Err(DataBankError::InvalidArrayMeta(
            "gene projection is shorter than dataset gene count".to_string(),
        ));
    }

    if let Some(output_start) = contiguous_output_start {
        let output_end = output_start.checked_add(elements).ok_or_else(|| {
            DataBankError::InvalidConfig("sparse projected output run overflow".to_string())
        })?;
        if output_end > projection.output_genes {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene: output_end,
                num_genes: projection.output_genes,
            });
        }
        let dst_start = row_base.checked_add(output_start).ok_or_else(|| {
            DataBankError::InvalidConfig("sparse projected output start overflow".to_string())
        })?;
        let dst_end = row_base.checked_add(output_end).ok_or_else(|| {
            DataBankError::InvalidConfig("sparse projected output end overflow".to_string())
        })?;
        T::cast_slice_from(data_bytes, src_dtype, &mut out[dst_start..dst_end])?;
        return Ok(());
    }

    macro_rules! dispatch_dst {
        ($dst:ty) => {{
            let out = unsafe { output_as_mut::<T, $dst>(out) };
            unsafe {
                dispatch_selected_source::<I, $dst>(
                    projection,
                    row_base,
                    elements,
                    index_bytes,
                    data_bytes,
                    src_dtype,
                    out,
                )
            }
        }};
    }

    match T::DTYPE {
        DType::U8 => dispatch_dst!(u8),
        DType::I8 => dispatch_dst!(i8),
        DType::U16 => dispatch_dst!(u16),
        DType::I16 => dispatch_dst!(i16),
        DType::U32 => dispatch_dst!(u32),
        DType::I32 => dispatch_dst!(i32),
        DType::U64 => dispatch_dst!(u64),
        DType::I64 => dispatch_dst!(i64),
        DType::F16 => dispatch_dst!(F16Bits),
        DType::BF16 => dispatch_dst!(Bf16Bits),
        DType::F32 => dispatch_dst!(f32),
        DType::F64 => dispatch_dst!(f64),
    }
}

pub(crate) unsafe fn scatter_sparse_values_projected_cols_aot_unchecked<T, O>(
    output_row: usize,
    output_genes: usize,
    elements: usize,
    projected_cols: &[O],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    O: ProjectedCol,
{
    let value_size = src_dtype.item_size();
    let row_base = output_row
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output row overflow".to_string()))?;
    let row_end = row_base.checked_add(output_genes).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output row end overflow".to_string())
    })?;
    if row_end > out.len() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out.len(),
        });
    }
    let expected_data_bytes = elements.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR data byte length overflow".to_string())
    })?;
    if data_bytes.len() != expected_data_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_data_bytes,
            actual: data_bytes.len(),
        });
    }
    if projected_cols.len() != elements {
        return Err(DataBankError::BufferSizeMismatch {
            expected: elements,
            actual: projected_cols.len(),
        });
    }

    macro_rules! dispatch_dst {
        ($dst:ty) => {{
            let out = unsafe { output_as_mut::<T, $dst>(out) };
            unsafe {
                dispatch_projected_cols_source::<O, $dst>(
                    row_base,
                    output_genes,
                    elements,
                    projected_cols,
                    data_bytes,
                    src_dtype,
                    out,
                )
            }
        }};
    }

    match T::DTYPE {
        DType::U8 => dispatch_dst!(u8),
        DType::I8 => dispatch_dst!(i8),
        DType::U16 => dispatch_dst!(u16),
        DType::I16 => dispatch_dst!(i16),
        DType::U32 => dispatch_dst!(u32),
        DType::I32 => dispatch_dst!(i32),
        DType::U64 => dispatch_dst!(u64),
        DType::I64 => dispatch_dst!(i64),
        DType::F16 => dispatch_dst!(F16Bits),
        DType::BF16 => dispatch_dst!(Bf16Bits),
        DType::F32 => dispatch_dst!(f32),
        DType::F64 => dispatch_dst!(f64),
    }
}

unsafe fn output_as_mut<T, D>(out: &mut [T]) -> &mut [D]
where
    T: DataValue,
    D: AotDst,
{
    debug_assert_eq!(T::DTYPE, D::DTYPE);
    debug_assert_eq!(mem::size_of::<T>(), mem::size_of::<D>());
    unsafe { slice::from_raw_parts_mut(out.as_mut_ptr().cast::<D>(), out.len()) }
}

fn dispatch_source<I, D>(
    projection: SparseProjectionCtx<'_>,
    row_base: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [D],
) -> DataBankResult<()>
where
    I: CsrIndex,
    D: AotDst,
{
    macro_rules! source_arm {
        ($src_dtype:expr, $src:ty) => {{
            if D::DTYPE == $src_dtype {
                unsafe {
                    scatter_projected_copy::<I, D>(
                        projection,
                        row_base,
                        elements,
                        index_bytes,
                        data_bytes,
                        out,
                    )
                }
            } else {
                scatter_projected_cast::<I, $src, D>(
                    projection,
                    row_base,
                    elements,
                    index_bytes,
                    data_bytes,
                    out,
                )
            }
        }};
    }

    match src_dtype {
        DType::U8 => source_arm!(DType::U8, SrcU8),
        DType::I8 => source_arm!(DType::I8, SrcI8),
        DType::U16 => source_arm!(DType::U16, SrcU16),
        DType::I16 => source_arm!(DType::I16, SrcI16),
        DType::U32 => source_arm!(DType::U32, SrcU32),
        DType::I32 => source_arm!(DType::I32, SrcI32),
        DType::U64 => source_arm!(DType::U64, SrcU64),
        DType::I64 => source_arm!(DType::I64, SrcI64),
        DType::F16 => source_arm!(DType::F16, SrcF16),
        DType::BF16 => source_arm!(DType::BF16, SrcBf16),
        DType::F32 => source_arm!(DType::F32, SrcF32),
        DType::F64 => source_arm!(DType::F64, SrcF64),
    }
}

unsafe fn dispatch_selected_source<I, D>(
    projection: SparseProjectionCtx<'_>,
    row_base: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [D],
) -> DataBankResult<()>
where
    I: CsrIndex,
    D: AotDst,
{
    macro_rules! source_arm {
        ($src_dtype:expr, $src:ty) => {{
            if D::DTYPE == $src_dtype {
                unsafe {
                    scatter_projected_selected_copy::<I, D>(
                        projection,
                        row_base,
                        elements,
                        index_bytes,
                        data_bytes,
                        out,
                    )
                }
            } else {
                unsafe {
                    scatter_projected_selected_cast::<I, $src, D>(
                        projection,
                        row_base,
                        elements,
                        index_bytes,
                        data_bytes,
                        out,
                    )
                }
            }
        }};
    }

    match src_dtype {
        DType::U8 => source_arm!(DType::U8, SrcU8),
        DType::I8 => source_arm!(DType::I8, SrcI8),
        DType::U16 => source_arm!(DType::U16, SrcU16),
        DType::I16 => source_arm!(DType::I16, SrcI16),
        DType::U32 => source_arm!(DType::U32, SrcU32),
        DType::I32 => source_arm!(DType::I32, SrcI32),
        DType::U64 => source_arm!(DType::U64, SrcU64),
        DType::I64 => source_arm!(DType::I64, SrcI64),
        DType::F16 => source_arm!(DType::F16, SrcF16),
        DType::BF16 => source_arm!(DType::BF16, SrcBf16),
        DType::F32 => source_arm!(DType::F32, SrcF32),
        DType::F64 => source_arm!(DType::F64, SrcF64),
    }
}

unsafe fn dispatch_projected_cols_source<O, D>(
    row_base: usize,
    output_genes: usize,
    elements: usize,
    projected_cols: &[O],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [D],
) -> DataBankResult<()>
where
    O: ProjectedCol,
    D: AotDst,
{
    macro_rules! source_arm {
        ($src_dtype:expr, $src:ty) => {{
            if D::DTYPE == $src_dtype {
                unsafe {
                    scatter_projected_cols_copy::<O, D>(
                        row_base,
                        output_genes,
                        elements,
                        projected_cols,
                        data_bytes,
                        out,
                    )
                }
            } else {
                unsafe {
                    scatter_projected_cols_cast::<O, $src, D>(
                        row_base,
                        output_genes,
                        elements,
                        projected_cols,
                        data_bytes,
                        out,
                    )
                }
            }
        }};
    }

    match src_dtype {
        DType::U8 => source_arm!(DType::U8, SrcU8),
        DType::I8 => source_arm!(DType::I8, SrcI8),
        DType::U16 => source_arm!(DType::U16, SrcU16),
        DType::I16 => source_arm!(DType::I16, SrcI16),
        DType::U32 => source_arm!(DType::U32, SrcU32),
        DType::I32 => source_arm!(DType::I32, SrcI32),
        DType::U64 => source_arm!(DType::U64, SrcU64),
        DType::I64 => source_arm!(DType::I64, SrcI64),
        DType::F16 => source_arm!(DType::F16, SrcF16),
        DType::BF16 => source_arm!(DType::BF16, SrcBf16),
        DType::F32 => source_arm!(DType::F32, SrcF32),
        DType::F64 => source_arm!(DType::F64, SrcF64),
    }
}

unsafe fn scatter_projected_copy<I, D>(
    projection: SparseProjectionCtx<'_>,
    row_base: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [D],
) -> DataBankResult<()>
where
    I: CsrIndex,
    D: AotDst,
{
    let index_ptr = index_bytes.as_ptr().cast::<I>();
    let data_ptr = data_bytes.as_ptr().cast::<D>();
    let out_ptr = out.as_mut_ptr();
    let output_by_source = projection.projection.output_by_source.as_ptr();

    if assume_sorted_csr_indices() {
        if let (Some((source_start, source_end)), Some((output_source_start, output_start))) = (
            projection.contiguous_selected_source_range,
            projection.contiguous_selected_source_output_start,
        ) {
            if source_start == output_source_start {
                for nz in 0..elements {
                    let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
                    let gene = raw_gene.checked_gene()?;
                    if gene >= projection.num_genes {
                        return Err(DataBankError::GeneIndexOutOfRange {
                            gene,
                            num_genes: projection.num_genes,
                        });
                    }
                    if gene >= source_end {
                        break;
                    }
                    if gene < source_start {
                        continue;
                    }
                    let output_col = output_start + (gene - source_start);
                    debug_assert!(output_col < projection.output_genes);
                    let value = unsafe { ptr::read_unaligned(data_ptr.add(nz)) };
                    unsafe {
                        ptr::write(out_ptr.add(row_base + output_col), value);
                    }
                }
                return Ok(());
            }
        }
    }

    for nz in 0..elements {
        let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
        let gene = raw_gene.checked_gene()?;
        if gene >= projection.num_genes {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene,
                num_genes: projection.num_genes,
            });
        }
        let output_col = unsafe { *output_by_source.add(gene) };
        if output_col == GENE_NOT_SELECTED {
            continue;
        }
        if output_col >= projection.output_genes {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene: output_col,
                num_genes: projection.output_genes,
            });
        }
        let value = unsafe { ptr::read_unaligned(data_ptr.add(nz)) };
        unsafe {
            ptr::write(out_ptr.add(row_base + output_col), value);
        }
    }
    Ok(())
}

unsafe fn scatter_projected_cols_copy<O, D>(
    row_base: usize,
    output_genes: usize,
    elements: usize,
    projected_cols: &[O],
    data_bytes: &[u8],
    out: &mut [D],
) -> DataBankResult<()>
where
    O: ProjectedCol,
    D: AotDst,
{
    let col_ptr = projected_cols.as_ptr();
    let data_ptr = data_bytes.as_ptr().cast::<D>();
    let out_ptr = out.as_mut_ptr();
    for nz in 0..elements {
        let output_col = unsafe { (*col_ptr.add(nz)).to_usize() };
        debug_assert!(
            output_col < output_genes,
            "projected CSR output column out of range"
        );
        let value = unsafe { ptr::read_unaligned(data_ptr.add(nz)) };
        unsafe {
            ptr::write(out_ptr.add(row_base + output_col), value);
        }
    }
    Ok(())
}

unsafe fn scatter_projected_selected_copy<I, D>(
    projection: SparseProjectionCtx<'_>,
    row_base: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [D],
) -> DataBankResult<()>
where
    I: CsrIndex,
    D: AotDst,
{
    let index_ptr = index_bytes.as_ptr().cast::<I>();
    let data_ptr = data_bytes.as_ptr().cast::<D>();
    let out_ptr = out.as_mut_ptr();

    if projection.contiguous_selected_source_output_start == Some((0, 0)) {
        for nz in 0..elements {
            let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
            debug_assert!(
                raw_gene
                    .checked_gene()
                    .is_ok_and(|gene| gene < projection.output_genes),
                "selected CSR gene index is negative or out of output range"
            );
            let output_col = unsafe { raw_gene.unchecked_gene() };
            let value = unsafe { ptr::read_unaligned(data_ptr.add(nz)) };
            unsafe {
                ptr::write(out_ptr.add(row_base + output_col), value);
            }
        }
    } else if let Some((source_start, output_start)) =
        projection.contiguous_selected_source_output_start
    {
        for nz in 0..elements {
            let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
            debug_assert!(
                raw_gene
                    .checked_gene()
                    .is_ok_and(|gene| gene < projection.num_genes),
                "selected CSR gene index is negative or out of range"
            );
            let gene = unsafe { raw_gene.unchecked_gene() };
            debug_assert!(
                gene >= source_start,
                "selected CSR gene is before contiguous range"
            );
            let output_col = output_start + (gene - source_start);
            debug_assert!(
                output_col < projection.output_genes,
                "selected CSR piece contains an out-of-range contiguous gene"
            );
            let value = unsafe { ptr::read_unaligned(data_ptr.add(nz)) };
            unsafe {
                ptr::write(out_ptr.add(row_base + output_col), value);
            }
        }
    } else {
        let output_by_source = projection.projection.output_by_source.as_ptr();
        for nz in 0..elements {
            let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
            debug_assert!(
                raw_gene
                    .checked_gene()
                    .is_ok_and(|gene| gene < projection.num_genes),
                "selected CSR gene index is negative or out of range"
            );
            let gene = unsafe { raw_gene.unchecked_gene() };
            let output_col = unsafe { *output_by_source.add(gene) };
            debug_assert!(
                output_col != GENE_NOT_SELECTED && output_col < projection.output_genes,
                "selected CSR piece contains an unselected gene"
            );
            let value = unsafe { ptr::read_unaligned(data_ptr.add(nz)) };
            unsafe {
                ptr::write(out_ptr.add(row_base + output_col), value);
            }
        }
    }
    Ok(())
}

fn scatter_projected_cast<I, S, D>(
    projection: SparseProjectionCtx<'_>,
    row_base: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [D],
) -> DataBankResult<()>
where
    I: CsrIndex,
    S: AotSrc,
    D: AotDst,
{
    if S::IS_FLOAT && D::IS_INT {
        if elements == 0 {
            return Ok(());
        }
        return Err(DataBankError::CannotCast {
            src: S::DTYPE,
            dst: D::DTYPE,
            reason: "float-to-int cast is not permitted",
        });
    }

    let index_ptr = index_bytes.as_ptr().cast::<I>();
    let data_ptr = data_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let output_by_source = projection.projection.output_by_source.as_ptr();

    if assume_sorted_csr_indices() {
        if let (Some((source_start, source_end)), Some((output_source_start, output_start))) = (
            projection.contiguous_selected_source_range,
            projection.contiguous_selected_source_output_start,
        ) {
            if source_start == output_source_start {
                for nz in 0..elements {
                    let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
                    let gene = raw_gene.checked_gene()?;
                    if gene >= projection.num_genes {
                        return Err(DataBankError::GeneIndexOutOfRange {
                            gene,
                            num_genes: projection.num_genes,
                        });
                    }
                    if gene >= source_end {
                        break;
                    }
                    if gene < source_start {
                        continue;
                    }
                    let output_col = output_start + (gene - source_start);
                    debug_assert!(output_col < projection.output_genes);
                    let raw = unsafe { S::read(data_ptr, nz) };
                    let value = S::cast_to_dst::<D>(raw);
                    unsafe {
                        ptr::write(out_ptr.add(row_base + output_col), value);
                    }
                }
                return Ok(());
            }
        }
    }

    for nz in 0..elements {
        let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
        let gene = raw_gene.checked_gene()?;
        if gene >= projection.num_genes {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene,
                num_genes: projection.num_genes,
            });
        }
        let output_col = unsafe { *output_by_source.add(gene) };
        if output_col == GENE_NOT_SELECTED {
            continue;
        }
        if output_col >= projection.output_genes {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene: output_col,
                num_genes: projection.output_genes,
            });
        }
        let raw = unsafe { S::read(data_ptr, nz) };
        let value = S::cast_to_dst::<D>(raw);
        unsafe {
            ptr::write(out_ptr.add(row_base + output_col), value);
        }
    }
    Ok(())
}

pub(crate) fn assume_sorted_csr_indices() -> bool {
    static ASSUME_SORTED: OnceLock<bool> = OnceLock::new();
    *ASSUME_SORTED.get_or_init(|| {
        std::env::var("SCDATA_ASSUME_SORTED_CSR_INDICES")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(false)
    })
}

unsafe fn scatter_projected_cols_cast<O, S, D>(
    row_base: usize,
    output_genes: usize,
    elements: usize,
    projected_cols: &[O],
    data_bytes: &[u8],
    out: &mut [D],
) -> DataBankResult<()>
where
    O: ProjectedCol,
    S: AotSrc,
    D: AotDst,
{
    if S::IS_FLOAT && D::IS_INT {
        if elements == 0 {
            return Ok(());
        }
        return Err(DataBankError::CannotCast {
            src: S::DTYPE,
            dst: D::DTYPE,
            reason: "float-to-int cast is not permitted",
        });
    }

    let col_ptr = projected_cols.as_ptr();
    let data_ptr = data_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr();
    for nz in 0..elements {
        let output_col = unsafe { (*col_ptr.add(nz)).to_usize() };
        debug_assert!(
            output_col < output_genes,
            "projected CSR output column out of range"
        );
        let raw = unsafe { S::read(data_ptr, nz) };
        let value = S::cast_to_dst::<D>(raw);
        unsafe {
            ptr::write(out_ptr.add(row_base + output_col), value);
        }
    }
    Ok(())
}

unsafe fn scatter_projected_selected_cast<I, S, D>(
    projection: SparseProjectionCtx<'_>,
    row_base: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [D],
) -> DataBankResult<()>
where
    I: CsrIndex,
    S: AotSrc,
    D: AotDst,
{
    if S::IS_FLOAT && D::IS_INT {
        if elements == 0 {
            return Ok(());
        }
        return Err(DataBankError::CannotCast {
            src: S::DTYPE,
            dst: D::DTYPE,
            reason: "float-to-int cast is not permitted",
        });
    }

    let index_ptr = index_bytes.as_ptr().cast::<I>();
    let data_ptr = data_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr();

    if projection.contiguous_selected_source_output_start == Some((0, 0)) {
        for nz in 0..elements {
            let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
            debug_assert!(
                raw_gene
                    .checked_gene()
                    .is_ok_and(|gene| gene < projection.output_genes),
                "selected CSR gene index is negative or out of output range"
            );
            let output_col = unsafe { raw_gene.unchecked_gene() };
            let raw = unsafe { S::read(data_ptr, nz) };
            let value = S::cast_to_dst::<D>(raw);
            unsafe {
                ptr::write(out_ptr.add(row_base + output_col), value);
            }
        }
    } else if let Some((source_start, output_start)) =
        projection.contiguous_selected_source_output_start
    {
        for nz in 0..elements {
            let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
            debug_assert!(
                raw_gene
                    .checked_gene()
                    .is_ok_and(|gene| gene < projection.num_genes),
                "selected CSR gene index is negative or out of range"
            );
            let gene = unsafe { raw_gene.unchecked_gene() };
            debug_assert!(
                gene >= source_start,
                "selected CSR gene is before contiguous range"
            );
            let output_col = output_start + (gene - source_start);
            debug_assert!(
                output_col < projection.output_genes,
                "selected CSR piece contains an out-of-range contiguous gene"
            );
            let raw = unsafe { S::read(data_ptr, nz) };
            let value = S::cast_to_dst::<D>(raw);
            unsafe {
                ptr::write(out_ptr.add(row_base + output_col), value);
            }
        }
    } else {
        let output_by_source = projection.projection.output_by_source.as_ptr();
        for nz in 0..elements {
            let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
            debug_assert!(
                raw_gene
                    .checked_gene()
                    .is_ok_and(|gene| gene < projection.num_genes),
                "selected CSR gene index is negative or out of range"
            );
            let gene = unsafe { raw_gene.unchecked_gene() };
            let output_col = unsafe { *output_by_source.add(gene) };
            debug_assert!(
                output_col != GENE_NOT_SELECTED && output_col < projection.output_genes,
                "selected CSR piece contains an unselected gene"
            );
            let raw = unsafe { S::read(data_ptr, nz) };
            let value = S::cast_to_dst::<D>(raw);
            unsafe {
                ptr::write(out_ptr.add(row_base + output_col), value);
            }
        }
    }
    Ok(())
}

pub(crate) trait ProjectedCol: Copy {
    fn to_usize(self) -> usize;
}

impl ProjectedCol for u16 {
    #[inline]
    fn to_usize(self) -> usize {
        usize::from(self)
    }
}

impl ProjectedCol for u32 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

trait AotSrc {
    type Raw: Copy;
    const DTYPE: DType;
    const IS_FLOAT: bool;

    unsafe fn read(data_ptr: *const u8, index: usize) -> Self::Raw;
    fn cast_to_dst<D: AotDst>(raw: Self::Raw) -> D;
}

macro_rules! int_src {
    ($name:ident, $raw:ty, $dtype:expr) => {
        struct $name;

        impl AotSrc for $name {
            type Raw = $raw;
            const DTYPE: DType = $dtype;
            const IS_FLOAT: bool = false;

            #[inline]
            unsafe fn read(data_ptr: *const u8, index: usize) -> Self::Raw {
                unsafe { ptr::read_unaligned(data_ptr.cast::<$raw>().add(index)) }
            }

            #[inline]
            fn cast_to_dst<D: AotDst>(raw: Self::Raw) -> D {
                D::from_i128(i128::from(raw))
            }
        }
    };
}

macro_rules! uint_src {
    ($name:ident, $raw:ty, $dtype:expr) => {
        struct $name;

        impl AotSrc for $name {
            type Raw = $raw;
            const DTYPE: DType = $dtype;
            const IS_FLOAT: bool = false;

            #[inline]
            unsafe fn read(data_ptr: *const u8, index: usize) -> Self::Raw {
                unsafe { ptr::read_unaligned(data_ptr.cast::<$raw>().add(index)) }
            }

            #[inline]
            fn cast_to_dst<D: AotDst>(raw: Self::Raw) -> D {
                D::from_i128(i128::from(raw))
            }
        }
    };
}

macro_rules! float_src {
    ($name:ident, $raw:ty, $dtype:expr, $to_f64:expr) => {
        struct $name;

        impl AotSrc for $name {
            type Raw = $raw;
            const DTYPE: DType = $dtype;
            const IS_FLOAT: bool = true;

            #[inline]
            unsafe fn read(data_ptr: *const u8, index: usize) -> Self::Raw {
                unsafe { ptr::read_unaligned(data_ptr.cast::<$raw>().add(index)) }
            }

            #[inline]
            fn cast_to_dst<D: AotDst>(raw: Self::Raw) -> D {
                D::from_f64(($to_f64)(raw))
            }
        }
    };
}

uint_src!(SrcU8, u8, DType::U8);
int_src!(SrcI8, i8, DType::I8);
uint_src!(SrcU16, u16, DType::U16);
int_src!(SrcI16, i16, DType::I16);
uint_src!(SrcU32, u32, DType::U32);
int_src!(SrcI32, i32, DType::I32);
uint_src!(SrcU64, u64, DType::U64);
int_src!(SrcI64, i64, DType::I64);
float_src!(SrcF16, u16, DType::F16, |bits| half::f16::from_bits(bits)
    .to_f64());
float_src!(SrcBf16, u16, DType::BF16, |bits| half::bf16::from_bits(
    bits
)
.to_f64());
float_src!(SrcF32, f32, DType::F32, f64::from);
float_src!(SrcF64, f64, DType::F64, |value| value);

trait AotDst: Copy {
    const DTYPE: DType;
    const IS_INT: bool;

    fn from_i128(value: i128) -> Self;
    fn from_f64(value: f64) -> Self;
}

macro_rules! int_dst {
    ($ty:ty, $dtype:expr) => {
        impl AotDst for $ty {
            const DTYPE: DType = $dtype;
            const IS_INT: bool = true;

            #[inline]
            fn from_i128(value: i128) -> Self {
                value as $ty
            }

            #[inline]
            fn from_f64(value: f64) -> Self {
                value as $ty
            }
        }
    };
}

macro_rules! float_dst {
    ($ty:ty, $dtype:expr) => {
        impl AotDst for $ty {
            const DTYPE: DType = $dtype;
            const IS_INT: bool = false;

            #[inline]
            fn from_i128(value: i128) -> Self {
                value as $ty
            }

            #[inline]
            fn from_f64(value: f64) -> Self {
                value as $ty
            }
        }
    };
}

int_dst!(u8, DType::U8);
int_dst!(i8, DType::I8);
int_dst!(u16, DType::U16);
int_dst!(i16, DType::I16);
int_dst!(u32, DType::U32);
int_dst!(i32, DType::I32);
int_dst!(u64, DType::U64);
int_dst!(i64, DType::I64);
float_dst!(f32, DType::F32);
float_dst!(f64, DType::F64);

impl AotDst for F16Bits {
    const DTYPE: DType = DType::F16;
    const IS_INT: bool = false;

    #[inline]
    fn from_i128(value: i128) -> Self {
        Self(half::f16::from_f64(value as f64).to_bits())
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        Self(half::f16::from_f64(value).to_bits())
    }
}

impl AotDst for Bf16Bits {
    const DTYPE: DType = DType::BF16;
    const IS_INT: bool = false;

    #[inline]
    fn from_i128(value: i128) -> Self {
        Self(half::bf16::from_f64(value as f64).to_bits())
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        Self(half::bf16::from_f64(value).to_bits())
    }
}
