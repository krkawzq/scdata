//! Dense / CSR initialization and scatter using pre-bound output kernels.

use sc_compress::DType as StorageDType;

use crate::plan::{
    csr_sparse_binary_is_cheaper, CsrMap, CsrSparseMap, CsrSparseMapEntry, DenseMap, SourcePlan,
    UNMAPPED_TARGET, UNMAPPED_TARGET_U32,
};
use crate::{Error, Result};

type FillFn = unsafe fn(output: *mut u8, len: usize, word: u64);
type IndexValidatorFn = unsafe fn(indices: *const u8, count: usize, n_cols: usize) -> bool;
type IndexReaderFn = unsafe fn(index: *const u8) -> usize;

#[derive(Debug, Clone, Copy)]
pub(crate) struct IndexOp {
    pub(crate) size: u8,
    pub(crate) shift: u8,
    validate: IndexValidatorFn,
    read: IndexReaderFn,
}

#[cfg(feature = "profile")]
pub(crate) mod profile;

impl IndexOp {
    pub(crate) fn new(dtype: StorageDType) -> Option<Self> {
        let (size, scalar, avx2, avx512): (
            u8,
            IndexValidatorFn,
            IndexValidatorFn,
            Option<IndexValidatorFn>,
        ) = match dtype {
            StorageDType::U16 => (
                2,
                validate_u16_scalar,
                validate_u16_avx2_dispatch,
                Some(validate_u16_avx512_dispatch),
            ),
            StorageDType::U32 => (
                4,
                validate_u32_scalar,
                validate_u32_avx2_dispatch,
                Some(validate_u32_avx512_dispatch),
            ),
            _ => return None,
        };
        let has_avx512 = if dtype == StorageDType::U16 {
            has_avx512bw()
        } else {
            has_avx512f()
        };
        let validate = if has_avx512 {
            avx512.unwrap_or_else(|| if has_avx2() { avx2 } else { scalar })
        } else if has_avx2() {
            avx2
        } else {
            scalar
        };
        let read = if size == 2 { read_u16 } else { read_u32 };
        Some(Self {
            size,
            shift: size.trailing_zeros() as u8,
            validate,
            read,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_scalar(dtype: StorageDType) -> Option<Self> {
        let mut op = Self::new(dtype)?;
        op.validate = match dtype {
            StorageDType::U16 => validate_u16_scalar,
            StorageDType::U32 => validate_u32_scalar,
            _ => return None,
        };
        Some(op)
    }

    #[cfg(all(test, target_arch = "x86_64", target_endian = "little"))]
    pub(crate) fn new_avx2_for_test(dtype: StorageDType) -> Option<Self> {
        if !has_avx2() {
            return None;
        }
        let mut op = Self::new_scalar(dtype)?;
        op.validate = match dtype {
            StorageDType::U16 => validate_u16_avx2_dispatch,
            StorageDType::U32 => validate_u32_avx2_dispatch,
            _ => return None,
        };
        Some(op)
    }

    #[cfg(all(test, target_arch = "x86_64", target_endian = "little"))]
    pub(crate) fn new_avx512_for_test(dtype: StorageDType) -> Option<Self> {
        let mut op = Self::new_scalar(dtype)?;
        op.validate = match dtype {
            StorageDType::U16 if has_avx512bw() => validate_u16_avx512_dispatch,
            StorageDType::U32 if has_avx512f() => validate_u32_avx512_dispatch,
            _ => return None,
        };
        Some(op)
    }

    #[inline(always)]
    pub(crate) unsafe fn validate(self, indices: *const u8, count: usize, n_cols: usize) -> bool {
        // SAFETY: caller proves the index buffer contains `count` complete
        // elements of the dtype bound into this operator.
        unsafe { (self.validate)(indices, count, n_cols) }
    }

    #[inline(always)]
    unsafe fn read(self, index: *const u8) -> usize {
        // SAFETY: caller proves one complete bound-dtype index is available.
        unsafe { (self.read)(index) }
    }
}

/// Pre-bound fill kernel. The compiler resolves the byte-pattern class once,
/// so scatter workers do not branch on output dtype or fill representation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FillOp {
    word: u64,
    fill: FillFn,
}

impl FillOp {
    pub(crate) fn new(fill: &[u8]) -> Self {
        debug_assert!(matches!(fill.len(), 2 | 4 | 8));
        let mut repeated = [0u8; 8];
        let mask = fill.len() - 1;
        for (index, byte) in repeated.iter_mut().enumerate() {
            // Element widths are powers of two, so wrapping the source pattern
            // is a mask instead of a remainder in this compile-time setup path.
            *byte = fill[index & mask];
        }
        let word = u64::from_le_bytes(repeated);
        let first = repeated[0];
        let fill = if word == 0 {
            fill_zero as FillFn
        } else if repeated.iter().all(|byte| *byte == first) {
            fill_uniform as FillFn
        } else {
            fill_repeated_scalar as FillFn
        };
        Self { word, fill }
    }

    #[inline(always)]
    pub(crate) unsafe fn apply(self, row: *mut u8, row_bytes: usize) {
        // SAFETY: caller proves `row..row+row_bytes` is writable.
        unsafe { (self.fill)(row, row_bytes, self.word) };
    }

    #[inline(always)]
    fn is_zero(self) -> bool {
        self.word == 0
    }

    #[inline(always)]
    unsafe fn apply_default_ranges(self, source: &SourcePlan, row: *mut u8) {
        for range in source.default_ranges.iter() {
            // SAFETY: the compiler built disjoint ranges inside the logical
            // output row, and the caller owns that complete row. Short gaps use
            // direct stores so fragmented feature maps do not make one indirect
            // memset-style call per missing output column.
            unsafe {
                let output = row.add(range.offset);
                if range.len < 64 {
                    fill_repeated_scalar(output, range.len, self.word);
                } else {
                    self.apply(output, range.len);
                }
            }
        }
    }
}

const CSR_MAP_IDENTITY: u8 = 0;
const CSR_MAP_PACKED: u8 = 1;
const CSR_MAP_WIDE: u8 = 2;

#[derive(Clone, Copy)]
struct CsrMapPointers {
    packed: *const u32,
    wide: *const usize,
}

#[inline(always)]
unsafe fn csr_map_target<const MAP_KIND: u8>(
    column: usize,
    destination_shift: u8,
    maps: CsrMapPointers,
) -> usize {
    if MAP_KIND == CSR_MAP_IDENTITY {
        column << destination_shift
    } else if MAP_KIND == CSR_MAP_PACKED {
        // SAFETY: structural validation proved the column lies in this map.
        let target = unsafe { *maps.packed.add(column) };
        if target == UNMAPPED_TARGET_U32 {
            UNMAPPED_TARGET
        } else {
            target as usize
        }
    } else {
        debug_assert_eq!(MAP_KIND, CSR_MAP_WIDE);
        // SAFETY: structural validation proved the column lies in this map.
        unsafe { *maps.wide.add(column) }
    }
}

unsafe fn validate_mapped_csr_values<const MAP_KIND: u8>(
    index: IndexOp,
    convert: &crate::convert::ConvertOp,
    values: *const u8,
    indices: *const u8,
    count: usize,
    maps: CsrMapPointers,
) -> Result<()> {
    let value_size = usize::from(convert.src_size);
    for element in 0..count {
        // SAFETY: row structure validation proved both arrays contain count
        // complete elements and every column lies within the selected map.
        let column = unsafe { index.read(indices.add(element << index.shift)) };
        // SAFETY: the same proof covers the selected map representation.
        let target = unsafe { csr_map_target::<MAP_KIND>(column, convert.dst_shift, maps) };
        if target != UNMAPPED_TARGET {
            // SAFETY: the equal-count check covers this complete source value;
            // the temporary slice is confined to this validation call.
            let value = unsafe {
                std::slice::from_raw_parts(values.add(element << convert.src_shift), value_size)
            };
            convert.validate_one(value)?;
        }
    }
    Ok(())
}

#[inline(always)]
fn unpack_sparse_packed(entry: u64) -> (usize, usize) {
    (entry as u32 as usize, (entry >> 32) as u32 as usize)
}

unsafe fn find_csr_column<const INDEX_BYTES: usize>(
    indices: *const u8,
    count: usize,
    target: usize,
) -> Option<usize> {
    let mut start = 0usize;
    let mut end = count;
    while start < end {
        let middle = start + (end - start) / 2;
        // SAFETY: `middle < count` addresses one complete validated index.
        let value = unsafe { read_sparse_index::<INDEX_BYTES>(indices, middle) };
        if value < target {
            start = middle + 1;
        } else {
            end = middle;
        }
    }
    if start < count {
        // SAFETY: `start < count` addresses one complete validated index.
        let value = unsafe { read_sparse_index::<INDEX_BYTES>(indices, start) };
        if value == target {
            return Some(start);
        }
    }
    None
}

#[inline(always)]
unsafe fn read_sparse_index<const INDEX_BYTES: usize>(indices: *const u8, element: usize) -> usize {
    if INDEX_BYTES == 2 {
        // SAFETY: caller proves one complete possibly unaligned u16 index.
        usize::from(u16::from_le(unsafe {
            indices.add(element * 2).cast::<u16>().read_unaligned()
        }))
    } else {
        debug_assert_eq!(INDEX_BYTES, 4);
        // SAFETY: caller proves one complete possibly unaligned u32 index.
        u32::from_le(unsafe { indices.add(element * 4).cast::<u32>().read_unaligned() }) as usize
    }
}

#[inline(always)]
unsafe fn find_bound_csr_column(
    index: IndexOp,
    indices: *const u8,
    count: usize,
    target: usize,
) -> Option<usize> {
    if index.size == 2 {
        // SAFETY: caller validated `count` complete u16 indices.
        unsafe { find_csr_column::<2>(indices, count, target) }
    } else {
        debug_assert_eq!(index.size, 4);
        // SAFETY: caller validated `count` complete u32 indices.
        unsafe { find_csr_column::<4>(indices, count, target) }
    }
}

unsafe fn validate_sparse_packed_csr_values(
    index: IndexOp,
    convert: &crate::convert::ConvertOp,
    values: *const u8,
    indices: *const u8,
    count: usize,
    entries: &[u64],
) -> Result<()> {
    let value_size = usize::from(convert.src_size);
    for &entry in entries {
        let (source_column, _) = unpack_sparse_packed(entry);
        // SAFETY: canonical indices and `count` were validated by the caller.
        let position = unsafe { find_bound_csr_column(index, indices, count, source_column) };
        if let Some(position) = position {
            // SAFETY: the found position addresses one complete source value.
            let value = unsafe {
                std::slice::from_raw_parts(values.add(position << convert.src_shift), value_size)
            };
            convert.validate_one(value)?;
        }
    }
    Ok(())
}

unsafe fn validate_sparse_wide_csr_values(
    index: IndexOp,
    convert: &crate::convert::ConvertOp,
    values: *const u8,
    indices: *const u8,
    count: usize,
    entries: &[CsrSparseMapEntry],
) -> Result<()> {
    let value_size = usize::from(convert.src_size);
    for entry in entries {
        // SAFETY: canonical indices and `count` were validated by the caller.
        let position = unsafe { find_bound_csr_column(index, indices, count, entry.source_column) };
        if let Some(position) = position {
            // SAFETY: the found position addresses one complete source value.
            let value = unsafe {
                std::slice::from_raw_parts(values.add(position << convert.src_shift), value_size)
            };
            convert.validate_one(value)?;
        }
    }
    Ok(())
}

unsafe fn scatter_sparse_packed_csr(
    index: IndexOp,
    convert: &crate::convert::ConvertOp,
    values: *const u8,
    indices: *const u8,
    output: *mut u8,
    count: usize,
    entries: &[u64],
) -> Result<()> {
    for &entry in entries {
        let (source_column, target_byte) = unpack_sparse_packed(entry);
        // SAFETY: canonical indices and `count` were validated by the caller.
        let position = unsafe { find_bound_csr_column(index, indices, count, source_column) };
        if let Some(position) = position {
            // SAFETY: validation covers the found value and compiler-built
            // target; mapped targets are unique and output is row-exclusive.
            unsafe {
                convert.convert_one_prevalidated(
                    values.add(position << convert.src_shift),
                    output.add(target_byte),
                )?;
            }
        }
    }
    Ok(())
}

unsafe fn scatter_sparse_wide_csr(
    index: IndexOp,
    convert: &crate::convert::ConvertOp,
    values: *const u8,
    indices: *const u8,
    output: *mut u8,
    count: usize,
    entries: &[CsrSparseMapEntry],
) -> Result<()> {
    for entry in entries {
        // SAFETY: canonical indices and `count` were validated by the caller.
        let position = unsafe { find_bound_csr_column(index, indices, count, entry.source_column) };
        if let Some(position) = position {
            // SAFETY: validation covers the found value and compiler-built
            // target; mapped targets are unique and output is row-exclusive.
            unsafe {
                convert.convert_one_prevalidated(
                    values.add(position << convert.src_shift),
                    output.add(entry.target_byte),
                )?;
            }
        }
    }
    Ok(())
}

unsafe fn scatter_csr_scalar<const MAP_KIND: u8>(
    index: IndexOp,
    convert: &crate::convert::ConvertOp,
    values: *const u8,
    indices: *const u8,
    output: *mut u8,
    count: usize,
    maps: CsrMapPointers,
) -> Result<()> {
    for element in 0..count {
        // SAFETY: prevalidation proved both arrays contain count complete
        // elements and every column lies within the selected map.
        let (column, value) = unsafe {
            (
                index.read(indices.add(element << index.shift)),
                values.add(element << convert.src_shift),
            )
        };
        // SAFETY: the same proof covers the selected map representation.
        let target = unsafe { csr_map_target::<MAP_KIND>(column, convert.dst_shift, maps) };
        if target != UNMAPPED_TARGET {
            // SAFETY: feature-map validation bounds the destination, and
            // successful conversion validation covers this source value.
            unsafe { convert.convert_one_prevalidated(value, output.add(target))? };
        }
    }
    Ok(())
}

/// Validate structure, canonical CSR indices, and every conversion that can fail.
pub(crate) fn validate_row(
    source: &SourcePlan,
    task: &crate::plan::CellTask,
    data: &[u8],
    indices: &[u8],
) -> Result<()> {
    let values = data
        .get(task.data_range())
        .ok_or_else(|| Error::Decode("cell data range exceeds decoded buffer".into()))?;
    let value_size = source.value_dtype.size();

    if source.index.is_none() {
        let expected = source
            .n_cols
            .checked_shl(u32::from(source.convert.src_shift))
            .ok_or_else(|| Error::Decode("dense row size overflow".into()))?;
        if values.len() != expected {
            return Err(Error::Decode(format!(
                "dense row {} has {} bytes, expected {expected}",
                task.row_offset(),
                values.len()
            )));
        }
        if source.convert.can_fail() {
            if let Some(entries) = source.dense_map.as_ref() {
                match entries {
                    DenseMap::Packed32 { entries, .. } => {
                        for &entry in entries.iter() {
                            let source_byte = entry as u32 as usize;
                            // SAFETY: compiler-packed offsets point at complete
                            // source elements inside the validated dense row.
                            let value = unsafe {
                                values.get_unchecked(source_byte..source_byte + value_size)
                            };
                            source.convert.validate_one(value)?;
                        }
                    }
                    DenseMap::Gather32 { source_offsets, .. } => {
                        for &source_byte in source_offsets.iter() {
                            // SAFETY: compiler-built nonnegative offsets point
                            // at complete source elements in this dense row.
                            let value = unsafe {
                                values.get_unchecked(
                                    source_byte as usize..source_byte as usize + value_size,
                                )
                            };
                            source.convert.validate_one(value)?;
                        }
                    }
                    DenseMap::Wide { entries, .. } => {
                        for entry in entries.iter() {
                            // SAFETY: compiler-built dense entries point at
                            // complete source elements inside the validated row.
                            let value = unsafe {
                                values.get_unchecked(
                                    entry.source_byte..entry.source_byte + value_size,
                                )
                            };
                            source.convert.validate_one(value)?;
                        }
                    }
                    DenseMap::Runs { entries, .. } => {
                        for run in entries.iter() {
                            let bytes = run.count.checked_mul(value_size).ok_or_else(|| {
                                Error::Invariant("dense validation run size overflow".into())
                            })?;
                            // SAFETY: compiler-built run offsets and lengths
                            // lie inside the already size-validated dense row.
                            let values = unsafe {
                                values.get_unchecked(run.source_byte..run.source_byte + bytes)
                            };
                            source.convert.validate_slice(values)?;
                        }
                    }
                }
            } else {
                source.convert.validate_slice(values)?;
            }
        }
        return Ok(());
    }

    let index_range = task.indices_range();
    let index_bytes = indices
        .get(index_range)
        .ok_or_else(|| Error::Decode("cell indices range exceeds decoded buffer".into()))?;
    let index = source
        .index
        .ok_or_else(|| Error::Invariant("CSR validation has no index operator".into()))?;
    let index_size = usize::from(index.size);
    let value_shift = source.convert.src_shift;
    if values.len() & (value_size - 1) != 0
        || index_bytes.len() & (index_size - 1) != 0
        || values.len() >> value_shift != index_bytes.len() >> index.shift
    {
        return Err(Error::Decode(format!(
            "CSR row {} has mismatched indices/data lengths",
            task.row_offset()
        )));
    }
    let count = index_bytes.len() >> index.shift;
    let map = source.feature_map.as_ref();
    // SAFETY: the byte-alignment and equal-count checks above prove the index
    // buffer contains exactly `count` complete elements of the bound dtype.
    if !unsafe { index.validate(index_bytes.as_ptr(), count, source.n_cols) } {
        let mut previous = None;
        for element in 0..count {
            // SAFETY: `element < count` and the alignment proof cover this read.
            let col = unsafe { index.read(index_bytes.as_ptr().add(element << index.shift)) };
            if col >= source.n_cols {
                return Err(Error::Decode(format!(
                    "CSR row {} contains index {col}, outside 0..{}",
                    task.row_offset(),
                    source.n_cols
                )));
            }
            if previous.is_some_and(|previous| col <= previous) {
                return Err(Error::Decode(format!(
                    "CSR row {} indices are not strictly increasing at column {col}",
                    task.row_offset()
                )));
            }
            previous = Some(col);
        }
        return Err(Error::Invariant(
            "CSR index validator rejected a row without a diagnostic".into(),
        ));
    }
    if source.convert.can_fail() {
        let sparse_validated = match source.csr_sparse_map.as_ref() {
            Some(CsrSparseMap::Packed32(entries))
                if csr_sparse_binary_is_cheaper(entries.len(), count) =>
            {
                // SAFETY: structural validation covers both arrays and the
                // compiler-built sparse entries are ordered and in range.
                unsafe {
                    validate_sparse_packed_csr_values(
                        index,
                        &source.convert,
                        values.as_ptr(),
                        index_bytes.as_ptr(),
                        count,
                        entries,
                    )?;
                }
                true
            }
            Some(CsrSparseMap::Wide(entries))
                if csr_sparse_binary_is_cheaper(entries.len(), count) =>
            {
                // SAFETY: the same proof covers wide sparse entries.
                unsafe {
                    validate_sparse_wide_csr_values(
                        index,
                        &source.convert,
                        values.as_ptr(),
                        index_bytes.as_ptr(),
                        count,
                        entries,
                    )?;
                }
                true
            }
            _ => false,
        };
        if !sparse_validated {
            match map {
                Some(CsrMap::Packed32(entries)) => {
                    // SAFETY: the structural checks above cover both complete
                    // arrays, every source column, and the packed map extent.
                    unsafe {
                        validate_mapped_csr_values::<CSR_MAP_PACKED>(
                            index,
                            &source.convert,
                            values.as_ptr(),
                            index_bytes.as_ptr(),
                            count,
                            CsrMapPointers {
                                packed: entries.as_ptr(),
                                wide: std::ptr::null(),
                            },
                        )?;
                    }
                }
                Some(CsrMap::Wide(entries)) => {
                    // SAFETY: the same proof covers the wide map representation.
                    unsafe {
                        validate_mapped_csr_values::<CSR_MAP_WIDE>(
                            index,
                            &source.convert,
                            values.as_ptr(),
                            index_bytes.as_ptr(),
                            count,
                            CsrMapPointers {
                                packed: std::ptr::null(),
                                wide: entries.as_ptr(),
                            },
                        )?;
                    }
                }
                None => source.convert.validate_slice(values)?,
            }
        }
    }
    Ok(())
}

/// Initialize and scatter a row whose structure and fallible conversions were checked.
///
/// # Safety
///
/// `validate_row(source, task, data, indices)` must have succeeded for these
/// exact immutable buffers, unless this is an infallible dense source whose
/// ranges and mapping were sealed by the compiler and whose decoder produced
/// the exact planned lengths. `row` must be the unique output row assigned to
/// `task`, with space for the plan's logical row bytes and aligned padding.
pub(crate) unsafe fn scatter_row_prevalidated(
    source: &SourcePlan,
    task: &crate::plan::CellTask,
    data: &[u8],
    indices: &[u8],
    row: &mut [u8],
    row_bytes: usize,
    fill: FillOp,
) -> Result<()> {
    // SAFETY: this forwards the caller's full validation and ownership proof;
    // a general destination may contain bytes from an older generation.
    unsafe {
        scatter_row_prevalidated_inner(
            source,
            task,
            data,
            indices,
            row,
            row_bytes,
            RowInitialization {
                fill,
                is_zeroed: false,
            },
        )
    }
}

/// Scatter into a row whose complete logical prefix is already zero.
///
/// # Safety
///
/// The safety contract of [`scatter_row_prevalidated`] applies, and every byte
/// in `row[..row_bytes]` must contain zero before this call.
pub(crate) unsafe fn scatter_row_prevalidated_zeroed(
    source: &SourcePlan,
    task: &crate::plan::CellTask,
    data: &[u8],
    indices: &[u8],
    row: &mut [u8],
    row_bytes: usize,
    fill: FillOp,
) -> Result<()> {
    // SAFETY: the caller supplies the base scatter proof plus the stronger
    // zeroed-destination invariant required by the final argument.
    unsafe {
        scatter_row_prevalidated_inner(
            source,
            task,
            data,
            indices,
            row,
            row_bytes,
            RowInitialization {
                fill,
                is_zeroed: true,
            },
        )
    }
}

#[derive(Clone, Copy)]
struct RowInitialization {
    fill: FillOp,
    is_zeroed: bool,
}

unsafe fn scatter_row_prevalidated_inner(
    source: &SourcePlan,
    task: &crate::plan::CellTask,
    data: &[u8],
    indices: &[u8],
    row: &mut [u8],
    row_bytes: usize,
    initialization: RowInitialization,
) -> Result<()> {
    let RowInitialization { fill, is_zeroed } = initialization;
    let dst_size = source.convert.dst_size as usize;
    debug_assert!(row.len() >= row_bytes);
    debug_assert_eq!(row_bytes & (dst_size - 1), 0);
    // SAFETY: the caller's successful validation proved this exact range is
    // inside the immutable decoded data buffer.
    let values = unsafe { data.get_unchecked(task.data_range()) };
    let convert = &source.convert;
    let map = source.feature_map.as_ref();
    let dense_map = source.dense_map.as_ref();

    if source.index.is_none() && dense_map.is_none() {
        // Dense identity mapping covers every logical output column, so filling
        // it first would be pure memory traffic. Padding starts zero and no
        // kernel writes it, so it requires no per-generation operation.
        // SAFETY: dense validation proved exactly `source.n_cols` complete
        // values, and the row-width check above proves the destination extent.
        return unsafe {
            convert.convert_slice_unchecked(values.as_ptr(), row.as_mut_ptr(), source.n_cols)
        };
    }

    if let Some(index) = source.index {
        // sc-compress canonicalizes every CSR row before writing. validate_row
        // repeats that boundary check for corrupt or externally modified
        // stores, so the commit phase can rely on unique, increasing indices.
        // FeatureMap also rejects duplicate destinations; together these
        // invariants give every mapped output element at most one nnz writer.
        // CSR absence is a structural zero; OutputSpec::fill applies only to
        // output columns that have no source feature mapping.
        if !is_zeroed {
            // SAFETY: the function contract proves the logical row prefix writable.
            unsafe { row.as_mut_ptr().write_bytes(0, row_bytes) };
        }
        if !fill.is_zero() {
            // SAFETY: compiler-built default ranges are disjoint and bounded.
            unsafe { fill.apply_default_ranges(source, row.as_mut_ptr()) };
        }
        let index_range = task.indices_range();
        let index_size = usize::from(index.size);
        // SAFETY: validation proved the range, element alignment, equal nnz,
        // strict index ordering, column bounds, and conversion policy.
        let index_bytes = unsafe { indices.get_unchecked(index_range) };
        let count = values.len() >> convert.src_shift;
        match source.csr_sparse_map.as_ref() {
            Some(CsrSparseMap::Packed32(entries))
                if csr_sparse_binary_is_cheaper(entries.len(), count) =>
            {
                // SAFETY: validation covered the canonical row, every sparse
                // entry, mapped conversion, and unique output target.
                unsafe {
                    scatter_sparse_packed_csr(
                        index,
                        convert,
                        values.as_ptr(),
                        index_bytes.as_ptr(),
                        row.as_mut_ptr(),
                        count,
                        entries,
                    )?;
                }
                return Ok(());
            }
            Some(CsrSparseMap::Wide(entries))
                if csr_sparse_binary_is_cheaper(entries.len(), count) =>
            {
                // SAFETY: the same proof covers wide sparse targets.
                unsafe {
                    scatter_sparse_wide_csr(
                        index,
                        convert,
                        values.as_ptr(),
                        index_bytes.as_ptr(),
                        row.as_mut_ptr(),
                        count,
                        entries,
                    )?;
                }
                return Ok(());
            }
            _ => {}
        }
        // SAFETY: validation proved both CSR arrays contain `count` complete
        // elements, every index/map target is in range, and this row is unique.
        if unsafe {
            convert.convert_csr_prevalidated(
                values.as_ptr(),
                index_bytes.as_ptr(),
                row.as_mut_ptr(),
                count,
                index_size,
                map,
            )
        } {
            return Ok(());
        }
        // The bound fast kernel is unavailable only for per-value fallback
        // conversion. Select its map representation once for the whole row.
        match map {
            None => {
                // SAFETY: prevalidation established the exact source, index,
                // identity-target, and output extents used here.
                unsafe {
                    scatter_csr_scalar::<CSR_MAP_IDENTITY>(
                        index,
                        convert,
                        values.as_ptr(),
                        index_bytes.as_ptr(),
                        row.as_mut_ptr(),
                        count,
                        CsrMapPointers {
                            packed: std::ptr::null(),
                            wide: std::ptr::null(),
                        },
                    )?;
                }
            }
            Some(CsrMap::Packed32(entries)) => {
                // SAFETY: the same proof covers the packed map extent.
                unsafe {
                    scatter_csr_scalar::<CSR_MAP_PACKED>(
                        index,
                        convert,
                        values.as_ptr(),
                        index_bytes.as_ptr(),
                        row.as_mut_ptr(),
                        count,
                        CsrMapPointers {
                            packed: entries.as_ptr(),
                            wide: std::ptr::null(),
                        },
                    )?;
                }
            }
            Some(CsrMap::Wide(entries)) => {
                // SAFETY: the same proof covers the wide map extent.
                unsafe {
                    scatter_csr_scalar::<CSR_MAP_WIDE>(
                        index,
                        convert,
                        values.as_ptr(),
                        index_bytes.as_ptr(),
                        row.as_mut_ptr(),
                        count,
                        CsrMapPointers {
                            packed: std::ptr::null(),
                            wide: entries.as_ptr(),
                        },
                    )?;
                }
            }
        }
    } else {
        let entries = dense_map
            .ok_or_else(|| Error::Invariant("dense mapped path has no compact mapping".into()))?;
        debug_assert!(
            !entries.covers_output()
                || !source.dense_fill_whole && source.default_ranges.is_empty()
        );
        if !(is_zeroed && fill.is_zero()) {
            if source.dense_fill_whole {
                // Highly fragmented gaps favor one streaming fill even though
                // mapped positions are overwritten by the conversion kernel.
                // SAFETY: the function contract proves the logical row writable.
                unsafe { fill.apply(row.as_mut_ptr(), row_bytes) };
            } else {
                // Mapped values and default ranges partition the logical row,
                // so low-fragmentation outputs write every byte exactly once.
                // SAFETY: compiler-built default ranges are disjoint and bounded.
                unsafe { fill.apply_default_ranges(source, row.as_mut_ptr()) };
            }
        }
        // SAFETY: compiler-built offsets point at complete source/destination
        // elements and the caller prevalidated every mapped conversion.
        unsafe {
            convert.convert_map_prevalidated(values.as_ptr(), row.as_mut_ptr(), entries)?;
        }
    }
    Ok(())
}

/// Initialize an empty source row without reading decoded buffers.
///
/// # Safety
///
/// `row` must uniquely own at least `row_bytes` writable bytes for `source`.
pub(crate) unsafe fn initialize_empty_row(
    source: &SourcePlan,
    row: *mut u8,
    row_bytes: usize,
    fill: FillOp,
) {
    // SAFETY: this forwards the caller's ownership proof; the destination may
    // contain bytes from an older output generation.
    unsafe { initialize_empty_row_inner(source, row, row_bytes, fill, false) };
}

/// Initialize an empty source row whose logical output prefix is already zero.
///
/// # Safety
///
/// The safety contract of [`initialize_empty_row`] applies, and
/// `row[..row_bytes]` must already contain zero.
pub(crate) unsafe fn initialize_empty_row_zeroed(
    source: &SourcePlan,
    row: *mut u8,
    row_bytes: usize,
    fill: FillOp,
) {
    // SAFETY: the caller supplies the base ownership proof plus a zeroed row.
    unsafe { initialize_empty_row_inner(source, row, row_bytes, fill, true) };
}

unsafe fn initialize_empty_row_inner(
    source: &SourcePlan,
    row: *mut u8,
    row_bytes: usize,
    fill: FillOp,
    row_is_zeroed: bool,
) {
    if source.index.is_some() {
        // An empty CSR row contains structural zeros in every mapped column.
        if !row_is_zeroed {
            // SAFETY: the caller proves the complete logical prefix writable.
            unsafe { row.write_bytes(0, row_bytes) };
        }
        if !fill.is_zero() {
            // SAFETY: compiler-built default ranges are disjoint and bounded.
            unsafe { fill.apply_default_ranges(source, row) };
        }
    } else {
        // A dense row reaches this path only when its stored width is zero;
        // every logical output column therefore belongs to a default range.
        if !(row_is_zeroed && fill.is_zero()) {
            // SAFETY: compiler-built default ranges are disjoint and bounded.
            unsafe { fill.apply_default_ranges(source, row) };
        }
    }
}

unsafe fn fill_zero(output: *mut u8, len: usize, _word: u64) {
    // SAFETY: FillOp::apply proves the complete destination prefix is writable.
    unsafe { output.write_bytes(0, len) };
}

unsafe fn fill_uniform(output: *mut u8, len: usize, word: u64) {
    // SAFETY: FillOp::apply proves the prefix and `word` repeats one byte.
    unsafe { output.write_bytes(word as u8, len) };
}

#[cfg(target_endian = "little")]
#[inline(always)]
unsafe fn fill_repeated_scalar(output: *mut u8, len: usize, word: u64) {
    let word_bytes = len & !7;
    // SAFETY: `word_bytes <= len`; each unaligned word store is disjoint and
    // the final byte copy covers only the remaining validated suffix.
    unsafe {
        for offset in (0..word_bytes).step_by(8) {
            output.add(offset).cast::<u64>().write_unaligned(word);
        }
        let tail = len - word_bytes;
        if tail != 0 {
            std::ptr::copy_nonoverlapping(
                word.to_le_bytes().as_ptr(),
                output.add(word_bytes),
                tail,
            );
        }
    }
}

#[cfg(target_endian = "big")]
#[inline(always)]
unsafe fn fill_repeated_scalar(output: *mut u8, len: usize, word: u64) {
    let bytes = word.to_le_bytes();
    // SAFETY: FillOp::apply proves the full destination prefix; every chunk is
    // disjoint and the repeated LE byte pattern preserves output encoding.
    unsafe {
        for offset in (0..len).step_by(8) {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                output.add(offset),
                (len - offset).min(8),
            );
        }
    }
}

#[inline(always)]
unsafe fn read_u16(input: *const u8) -> usize {
    // SAFETY: caller proves a complete possibly unaligned u16 is available.
    usize::from(u16::from_le(unsafe {
        input.cast::<u16>().read_unaligned()
    }))
}

#[inline(always)]
unsafe fn read_u32(input: *const u8) -> usize {
    // SAFETY: caller proves a complete possibly unaligned u32 is available.
    u32::from_le(unsafe { input.cast::<u32>().read_unaligned() }) as usize
}

unsafe fn validate_u16_scalar(indices: *const u8, count: usize, n_cols: usize) -> bool {
    if count == 0 {
        return true;
    }
    // SAFETY: caller proves at least one complete u16 index.
    let mut previous = unsafe { read_u16(indices) };
    if previous >= n_cols {
        return false;
    }
    for element in 1..count {
        // SAFETY: `element < count` covers this complete u16 index.
        let current = unsafe { read_u16(indices.add(element << 1)) };
        if current <= previous || current >= n_cols {
            return false;
        }
        previous = current;
    }
    true
}

unsafe fn validate_u32_scalar(indices: *const u8, count: usize, n_cols: usize) -> bool {
    if count == 0 {
        return true;
    }
    // SAFETY: caller proves at least one complete u32 index.
    let mut previous = unsafe { read_u32(indices) };
    if previous >= n_cols {
        return false;
    }
    for element in 1..count {
        // SAFETY: `element < count` covers this complete u32 index.
        let current = unsafe { read_u32(indices.add(element << 2)) };
        if current <= previous || current >= n_cols {
            return false;
        }
        previous = current;
    }
    true
}

#[inline]
fn has_avx2() -> bool {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
    false
}

#[inline]
fn has_avx512bw() -> bool {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
    false
}

#[inline]
fn has_avx512f() -> bool {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        std::arch::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
    false
}

unsafe fn validate_u16_avx512_dispatch(indices: *const u8, count: usize, n_cols: usize) -> bool {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        // SAFETY: IndexOp binds this target-feature function only after
        // AVX-512F/BW detection and the caller supplies the complete extent.
        unsafe { validate_u16_avx512(indices, count, n_cols) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
    {
        // SAFETY: caller supplies the complete index extent.
        unsafe { validate_u16_scalar(indices, count, n_cols) }
    }
}

unsafe fn validate_u16_avx2_dispatch(indices: *const u8, count: usize, n_cols: usize) -> bool {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        // SAFETY: IndexOp binds this target-feature function only after AVX2
        // detection and the caller supplies the complete index extent.
        unsafe { validate_u16_avx2(indices, count, n_cols) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
    {
        // SAFETY: caller supplies the complete index extent.
        unsafe { validate_u16_scalar(indices, count, n_cols) }
    }
}

unsafe fn validate_u32_avx2_dispatch(indices: *const u8, count: usize, n_cols: usize) -> bool {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        // SAFETY: IndexOp binds this function only after AVX2 detection.
        unsafe { validate_u32_avx2(indices, count, n_cols) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
    {
        // SAFETY: caller supplies the complete index extent.
        unsafe { validate_u32_scalar(indices, count, n_cols) }
    }
}

unsafe fn validate_u32_avx512_dispatch(indices: *const u8, count: usize, n_cols: usize) -> bool {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        // SAFETY: IndexOp binds this function only after AVX-512F detection,
        // and its caller supplies the complete validated index extent.
        unsafe { validate_u32_avx512(indices, count, n_cols) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
    {
        // SAFETY: caller supplies the complete index extent.
        unsafe { validate_u32_scalar(indices, count, n_cols) }
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2")]
unsafe fn validate_u16_avx2(indices: *const u8, count: usize, n_cols: usize) -> bool {
    use std::arch::x86_64::*;

    if count == 0 {
        return true;
    }
    // SAFETY: caller proves at least one complete u16 index.
    let mut previous = unsafe { read_u16(indices) };
    if previous >= n_cols {
        return false;
    }
    let sign = _mm256_set1_epi16(i16::MIN);
    let limit = (n_cols <= u16::MAX as usize)
        .then(|| _mm256_xor_si256(_mm256_set1_epi16(n_cols as i16), sign));
    let mut element = 1usize;
    // SAFETY: the loop guard covers each unaligned 16*u16 vector load.
    unsafe {
        while element + 16 <= count {
            let values = _mm256_loadu_si256(indices.add(element << 1).cast::<__m256i>());
            let shifted = _mm256_slli_si256::<2>(values);
            let low_last = _mm_extract_epi16::<7>(_mm256_castsi256_si128(values));
            let boundaries = _mm256_setr_epi16(
                previous as i16,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                low_last as i16,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            );
            // The blend mask applies to each 128-bit lane, replacing lane 0
            // with the scalar predecessor and lane 8 with value lane 7.
            let predecessors = _mm256_blend_epi16::<1>(shifted, boundaries);
            let unsigned_values = _mm256_xor_si256(values, sign);
            let unsigned_predecessors = _mm256_xor_si256(predecessors, sign);
            if _mm256_movemask_epi8(_mm256_cmpgt_epi16(unsigned_values, unsigned_predecessors))
                != -1
            {
                return false;
            }
            if limit.is_some_and(|limit| {
                _mm256_movemask_epi8(_mm256_cmpgt_epi16(limit, unsigned_values)) != -1
            }) {
                return false;
            }
            previous = read_u16(indices.add((element + 15) << 1));
            element += 16;
        }
        while element < count {
            let current = read_u16(indices.add(element << 1));
            if current <= previous || current >= n_cols {
                return false;
            }
            previous = current;
            element += 1;
        }
    }
    true
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn validate_u16_avx512(indices: *const u8, count: usize, n_cols: usize) -> bool {
    use std::arch::x86_64::*;

    if count == 0 {
        return true;
    }
    // SAFETY: caller proves at least one complete u16 index.
    let mut previous = unsafe { read_u16(indices) };
    if previous >= n_cols {
        return false;
    }
    const PREDECESSOR_LANES: [u16; 32] = [
        0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30,
    ];
    // SAFETY: the static array contains one complete 512-bit load, and runtime
    // dispatch established the target features used by these constructors.
    let (permutation, limit) = unsafe {
        (
            _mm512_loadu_si512(PREDECESSOR_LANES.as_ptr().cast::<__m512i>()),
            (n_cols <= u16::MAX as usize).then(|| _mm512_set1_epi16(n_cols as i16)),
        )
    };
    let mut element = 1usize;
    // SAFETY: the loop guard covers each unaligned 32*u16 vector load. The
    // permutation supplies the preceding lane and mask bit zero bridges the
    // vector boundary with the prior scalar value.
    unsafe {
        while element + 32 <= count {
            let values = _mm512_loadu_si512(indices.add(element << 1).cast::<__m512i>());
            let shifted = _mm512_permutexvar_epi16(permutation, values);
            let predecessors =
                _mm512_mask_mov_epi16(shifted, 1, _mm512_set1_epi16(previous as i16));
            if _mm512_cmp_epu16_mask::<_MM_CMPINT_NLE>(values, predecessors) != u32::MAX {
                return false;
            }
            if limit.is_some_and(|limit| {
                _mm512_cmp_epu16_mask::<_MM_CMPINT_LT>(values, limit) != u32::MAX
            }) {
                return false;
            }
            previous = read_u16(indices.add((element + 31) << 1));
            element += 32;
        }
        let remaining = count - element;
        if remaining <= 6 {
            while element < count {
                let current = read_u16(indices.add(element << 1));
                if current <= previous || current >= n_cols {
                    return false;
                }
                previous = current;
                element += 1;
            }
        } else {
            let mask = (1u32 << remaining) - 1;
            let values = _mm512_maskz_loadu_epi16(mask, indices.add(element << 1).cast::<i16>());
            let shifted = _mm512_permutexvar_epi16(permutation, values);
            let predecessors =
                _mm512_mask_mov_epi16(shifted, 1, _mm512_set1_epi16(previous as i16));
            if _mm512_cmp_epu16_mask::<_MM_CMPINT_NLE>(values, predecessors) & mask != mask {
                return false;
            }
            if limit.is_some_and(|limit| {
                _mm512_cmp_epu16_mask::<_MM_CMPINT_LT>(values, limit) & mask != mask
            }) {
                return false;
            }
        }
    }
    true
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2")]
unsafe fn validate_u32_avx2(indices: *const u8, count: usize, n_cols: usize) -> bool {
    use std::arch::x86_64::*;

    if count == 0 {
        return true;
    }
    // SAFETY: caller proves at least one complete u32 index.
    let mut previous = unsafe { read_u32(indices) };
    if previous >= n_cols {
        return false;
    }
    let sign = _mm256_set1_epi32(i32::MIN);
    let permutation = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    let limit = (n_cols <= u32::MAX as usize)
        .then(|| _mm256_xor_si256(_mm256_set1_epi32(n_cols as i32), sign));
    let mut element = 1usize;
    // SAFETY: the loop guard covers each unaligned 8*u32 vector load.
    unsafe {
        while element + 8 <= count {
            let values = _mm256_loadu_si256(indices.add(element << 2).cast::<__m256i>());
            let shifted = _mm256_permutevar8x32_epi32(values, permutation);
            let predecessors = _mm256_blend_epi32::<1>(shifted, _mm256_set1_epi32(previous as i32));
            let unsigned_values = _mm256_xor_si256(values, sign);
            let unsigned_predecessors = _mm256_xor_si256(predecessors, sign);
            if _mm256_movemask_epi8(_mm256_cmpgt_epi32(unsigned_values, unsigned_predecessors))
                != -1
            {
                return false;
            }
            if limit.is_some_and(|limit| {
                _mm256_movemask_epi8(_mm256_cmpgt_epi32(limit, unsigned_values)) != -1
            }) {
                return false;
            }
            previous = _mm256_extract_epi32::<7>(values) as u32 as usize;
            element += 8;
        }
        while element < count {
            let current = read_u32(indices.add(element << 2));
            if current <= previous || current >= n_cols {
                return false;
            }
            previous = current;
            element += 1;
        }
    }
    true
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn validate_u32_avx512(indices: *const u8, count: usize, n_cols: usize) -> bool {
    use std::arch::x86_64::*;

    if count == 0 {
        return true;
    }
    // SAFETY: caller proves at least one complete u32 index.
    let mut previous = unsafe { read_u32(indices) };
    if previous >= n_cols {
        return false;
    }
    const PREDECESSOR_LANES: [u32; 16] = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    // SAFETY: the static array contains one complete vector, and runtime
    // dispatch established AVX-512F for all constructors below.
    let (permutation, limit) = unsafe {
        (
            _mm512_loadu_si512(PREDECESSOR_LANES.as_ptr().cast::<__m512i>()),
            (n_cols <= u32::MAX as usize).then(|| _mm512_set1_epi32(n_cols as i32)),
        )
    };
    let mut element = 1usize;
    // SAFETY: the loop guard covers each 16*u32 load. The permutation supplies
    // lane-local predecessors and mask bit zero bridges vector boundaries.
    unsafe {
        while element + 16 <= count {
            let values = _mm512_loadu_si512(indices.add(element << 2).cast::<__m512i>());
            let shifted = _mm512_permutexvar_epi32(permutation, values);
            let predecessors =
                _mm512_mask_mov_epi32(shifted, 1, _mm512_set1_epi32(previous as i32));
            if _mm512_cmp_epu32_mask::<_MM_CMPINT_NLE>(values, predecessors) != u16::MAX {
                return false;
            }
            if limit.is_some_and(|limit| {
                _mm512_cmp_epu32_mask::<_MM_CMPINT_LT>(values, limit) != u16::MAX
            }) {
                return false;
            }
            previous = read_u32(indices.add((element + 15) << 2));
            element += 16;
        }
        let remaining = count - element;
        if remaining <= 10 {
            while element < count {
                let current = read_u32(indices.add(element << 2));
                if current <= previous || current >= n_cols {
                    return false;
                }
                previous = current;
                element += 1;
            }
        } else {
            let mask = ((1u32 << remaining) - 1) as __mmask16;
            let values = _mm512_maskz_loadu_epi32(mask, indices.add(element << 2).cast::<i32>());
            let shifted = _mm512_permutexvar_epi32(permutation, values);
            let predecessors =
                _mm512_mask_mov_epi32(shifted, 1, _mm512_set1_epi32(previous as i32));
            if _mm512_cmp_epu32_mask::<_MM_CMPINT_NLE>(values, predecessors) & mask != mask {
                return false;
            }
            if limit.is_some_and(|limit| {
                _mm512_cmp_epu32_mask::<_MM_CMPINT_LT>(values, limit) & mask != mask
            }) {
                return false;
            }
        }
    }
    true
}

#[cfg(all(test, target_arch = "x86_64", target_endian = "little"))]
mod simd_tests {
    use super::*;

    #[test]
    fn u32_index_avx512_matches_avx2_across_vector_boundaries() {
        let Some(avx2) = IndexOp::new_avx2_for_test(StorageDType::U32) else {
            return;
        };
        let Some(avx512) = IndexOp::new_avx512_for_test(StorageDType::U32) else {
            return;
        };
        let check = |op: IndexOp, values: &[u8], count: usize, columns: usize| {
            // SAFETY: each call supplies `count` complete u32 indices.
            unsafe { op.validate(values.as_ptr(), count, columns) }
        };

        let valid = (0..35u32)
            .map(|value| value * 3)
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert!(check(avx2, &valid, 35, 128));
        assert!(check(avx512, &valid, 35, 128));

        let mut duplicate = valid.clone();
        duplicate[16 * 4..17 * 4].copy_from_slice(&(15u32 * 3).to_le_bytes());
        assert!(!check(avx2, &duplicate, 35, 128));
        assert!(!check(avx512, &duplicate, 35, 128));

        let mut out_of_bounds = valid;
        out_of_bounds[34 * 4..35 * 4].copy_from_slice(&128u32.to_le_bytes());
        assert!(!check(avx2, &out_of_bounds, 35, 128));
        assert!(!check(avx512, &out_of_bounds, 35, 128));
    }

    #[test]
    fn index_avx512_hybrid_tails_match_scalar() {
        let check = |op: IndexOp, values: &[u8], count: usize| {
            // SAFETY: every case supplies `count` complete bound-dtype indices.
            unsafe { op.validate(values.as_ptr(), count, 512) }
        };

        if let Some(avx512) = IndexOp::new_avx512_for_test(StorageDType::U16) {
            let scalar = IndexOp::new_scalar(StorageDType::U16).unwrap();
            let valid = (0..97u16)
                .map(|value| value * 3)
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>();
            for count in 0..=97 {
                assert_eq!(check(avx512, &valid, count), check(scalar, &valid, count));
                if count >= 2 {
                    let mut duplicate = valid.clone();
                    duplicate[(count - 1) * 2..count * 2]
                        .copy_from_slice(&((count as u16 - 2) * 3).to_le_bytes());
                    assert!(!check(avx512, &duplicate, count));
                }
                if count >= 1 {
                    let mut out_of_bounds = valid.clone();
                    out_of_bounds[(count - 1) * 2..count * 2]
                        .copy_from_slice(&512u16.to_le_bytes());
                    assert!(!check(avx512, &out_of_bounds, count));
                }
            }
        }

        if let Some(avx512) = IndexOp::new_avx512_for_test(StorageDType::U32) {
            let scalar = IndexOp::new_scalar(StorageDType::U32).unwrap();
            let valid = (0..97u32)
                .map(|value| value * 3)
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            for count in 0..=97 {
                assert_eq!(check(avx512, &valid, count), check(scalar, &valid, count));
                if count >= 2 {
                    let mut duplicate = valid.clone();
                    duplicate[(count - 1) * 4..count * 4]
                        .copy_from_slice(&((count as u32 - 2) * 3).to_le_bytes());
                    assert!(!check(avx512, &duplicate, count));
                }
                if count >= 1 {
                    let mut out_of_bounds = valid.clone();
                    out_of_bounds[(count - 1) * 4..count * 4]
                        .copy_from_slice(&512u32.to_le_bytes());
                    assert!(!check(avx512, &out_of_bounds, count));
                }
            }
        }
    }
}

#[cfg(test)]
mod systematic_tests {
    use std::sync::Arc;

    use sc_compress::{CsrWriter, DType as StorageDType, Partition};

    use crate::compiler::build_default_ranges;
    use crate::convert::ConvertOp;
    use crate::dtype::{promote_kind, OutputDType, PromoteKind};
    use crate::output::{Fill, FloatCastPolicy, OutputSpec, OverflowPolicy};
    use crate::plan::{
        csr_sparse_binary_is_cheaper, CellTask, CsrMap, CsrSparseMap, CsrSparseMapEntry, DenseMap,
        DenseMapEntry, DenseMapRun, SourcePlan, UNMAPPED_TARGET, UNMAPPED_TARGET_U32,
    };
    use crate::scatter::{scatter_row_prevalidated, validate_row, FillOp, IndexOp};
    use crate::source::OutputSlot;
    use crate::{
        compile, Dataset, FeatureMap, IoMode, PlanSpec, RowRef, SessionConfig, Source, SourceId,
    };

    const STORAGE_DTYPES: [StorageDType; 8] = [
        StorageDType::I16,
        StorageDType::I32,
        StorageDType::I64,
        StorageDType::U16,
        StorageDType::U32,
        StorageDType::U64,
        StorageDType::F32,
        StorageDType::F64,
    ];
    const OUTPUT_DTYPES: [OutputDType; 8] = [
        OutputDType::I16,
        OutputDType::I32,
        OutputDType::I64,
        OutputDType::U16,
        OutputDType::U32,
        OutputDType::U64,
        OutputDType::F32,
        OutputDType::F64,
    ];
    const LENGTHS: [usize; 12] = [0, 1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 65];

    #[test]
    fn every_conversion_matches_generic_across_all_scatter_representations() {
        let mut cases = 0usize;
        for src in STORAGE_DTYPES {
            for dst in OUTPUT_DTYPES {
                let Some(kind) = promote_kind(src, dst) else {
                    continue;
                };
                let policies: &[OverflowPolicy] = if kind == PromoteKind::CheckedSign {
                    &[
                        OverflowPolicy::Error,
                        OverflowPolicy::UseFill,
                        OverflowPolicy::UseValue(sentinel(dst)),
                        OverflowPolicy::Unchecked,
                    ]
                } else {
                    &[OverflowPolicy::Error]
                };
                for policy in policies {
                    for len in LENGTHS {
                        check_edge(src, dst, kind, policy.clone(), len);
                        cases += 1;
                    }
                    check_sparse_edge(src, dst, kind, policy.clone());
                }
            }
        }
        assert_eq!(cases, 768);
    }

    fn check_sparse_edge(
        src: StorageDType,
        dst: OutputDType,
        kind: PromoteKind,
        policy: OverflowPolicy,
    ) {
        const N_COLS: usize = 1024;
        const NNZ: usize = 512;
        let output = output_spec(1, dst, policy.clone(), kind);
        let invalid = kind == PromoteKind::CheckedSign && !matches!(policy, OverflowPolicy::Error);
        let data = values(src, NNZ, invalid);
        let fill = FillOp::new(&output.fill().encode()[..dst.size()]);
        let convert = ConvertOp::resolve(src, &output).unwrap();
        let mut dense_targets = vec![UNMAPPED_TARGET_U32; N_COLS];
        dense_targets[0] = 0;
        for index_dtype in [StorageDType::U16, StorageDType::U32] {
            let columns = (0..NNZ)
                .map(|index| index * N_COLS / NNZ)
                .collect::<Vec<_>>();
            let indices = encode_indices(&columns, index_dtype);
            let task = task(0..data.len(), Some(0..indices.len()));
            let dense: Arc<[u32]> = Arc::from(dense_targets.clone());
            let dense_source = SourcePlan {
                feature_map: Some(CsrMap::Packed32(Arc::clone(&dense))),
                ..source(
                    N_COLS,
                    src,
                    IndexOp::new(index_dtype),
                    None,
                    false,
                    Default::default(),
                    convert,
                )
            };
            let sparse_source = SourcePlan {
                feature_map: Some(CsrMap::Packed32(dense)),
                csr_sparse_map: Some(CsrSparseMap::Packed32(Arc::from([0u64]))),
                ..dense_source.clone()
            };
            assert!(csr_sparse_binary_is_cheaper(1, NNZ));
            assert_same(
                &sparse_source,
                &dense_source,
                &task,
                &data,
                &indices,
                dst.size(),
                fill,
            );
        }
    }

    fn check_edge(
        src: StorageDType,
        dst: OutputDType,
        kind: PromoteKind,
        policy: OverflowPolicy,
        len: usize,
    ) {
        let output = output_spec(len.saturating_add(3), dst, policy.clone(), kind);
        let invalid = kind == PromoteKind::CheckedSign && !matches!(policy, OverflowPolicy::Error);
        let input = values(src, len, invalid);
        let specialized = ConvertOp::resolve(src, &output).unwrap();
        let mut generic = specialized;
        generic.force_generic_for_test();
        let fill = FillOp::new(&output.fill().encode()[..dst.size()]);

        check_dense_identity(src, dst, len, &input, specialized, generic, fill);
        check_dense_maps(src, dst, len, &input, specialized, generic, fill);
        check_csr(
            src,
            dst,
            len,
            &input,
            specialized,
            generic,
            fill,
            StorageDType::U16,
        );
        check_csr(
            src,
            dst,
            len,
            &input,
            specialized,
            generic,
            fill,
            StorageDType::U32,
        );
    }

    fn check_dense_identity(
        src: StorageDType,
        dst: OutputDType,
        len: usize,
        input: &[u8],
        specialized: ConvertOp,
        generic: ConvertOp,
        fill: FillOp,
    ) {
        let task = task(0..input.len(), None);
        let specialized = source(len, src, None, None, false, Default::default(), specialized);
        let generic = SourcePlan {
            convert: generic,
            ..specialized.clone()
        };
        assert_same(
            &specialized,
            &generic,
            &task,
            input,
            &[],
            len * dst.size(),
            fill,
        );
    }

    fn check_dense_maps(
        src: StorageDType,
        dst: OutputDType,
        len: usize,
        input: &[u8],
        specialized: ConvertOp,
        generic: ConvertOp,
        fill: FillOp,
    ) {
        let src_size = src.size();
        let dst_size = dst.size();
        let mapped = len.div_ceil(2);
        let output_cols = mapped.saturating_add(3);
        let targets = (0..len)
            .map(|column| (column % 2 == 0).then_some(column / 2))
            .collect::<Vec<_>>();
        let defaults = build_default_ranges(Some(&targets), output_cols, dst_size).unwrap();
        let packed = DenseMap::Packed32 {
            entries: Arc::from(
                targets
                    .iter()
                    .enumerate()
                    .filter_map(|(column, target)| {
                        target.map(|target| {
                            u64::from((column * src_size) as u32)
                                | (u64::from((target * dst_size) as u32) << 32)
                        })
                    })
                    .collect::<Vec<_>>(),
            ),
            covers_output: false,
        };
        let wide = DenseMap::Wide {
            entries: Arc::from(
                targets
                    .iter()
                    .enumerate()
                    .filter_map(|(column, target)| {
                        target.map(|target| DenseMapEntry {
                            source_byte: column * src_size,
                            target_byte: target * dst_size,
                        })
                    })
                    .collect::<Vec<_>>(),
            ),
            covers_output: false,
        };
        let gather = DenseMap::Gather32 {
            source_offsets: Arc::from(
                (0..len)
                    .step_by(2)
                    .map(|column| i32::try_from(column * src_size).unwrap())
                    .collect::<Vec<_>>(),
            ),
            target_byte: 0,
            covers_output: true,
        };
        let run_count = len / 2;
        let runs = DenseMap::Runs {
            entries: if run_count == 0 {
                Default::default()
            } else {
                Arc::from([DenseMapRun {
                    source_byte: 0,
                    target_byte: 0,
                    count: run_count,
                }])
            },
            covers_output: run_count == mapped,
        };
        let task = task(0..input.len(), None);
        for (map, columns, ranges) in [
            (packed, output_cols, Arc::clone(&defaults)),
            (wide, output_cols, Arc::clone(&defaults)),
            (gather, mapped, Default::default()),
            (runs, run_count, Default::default()),
        ] {
            let specialized_source = source(len, src, None, Some(map), false, ranges, specialized);
            let generic_source = SourcePlan {
                convert: generic,
                ..specialized_source.clone()
            };
            assert_same(
                &specialized_source,
                &generic_source,
                &task,
                input,
                &[],
                columns * dst_size,
                fill,
            );
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "test matrix dimensions are explicit"
    )]
    fn check_csr(
        src: StorageDType,
        dst: OutputDType,
        len: usize,
        input: &[u8],
        specialized: ConvertOp,
        generic: ConvertOp,
        fill: FillOp,
        index_dtype: StorageDType,
    ) {
        let src_size = src.size();
        let dst_size = dst.size();
        let count = len.div_ceil(2);
        let data = input[..count * src_size].to_vec();
        let columns = (0..len).step_by(2).collect::<Vec<_>>();
        let indices = encode_indices(&columns, index_dtype);
        let task = task(0..data.len(), Some(0..indices.len()));
        let identity = source(
            len,
            src,
            IndexOp::new(index_dtype),
            None,
            false,
            Default::default(),
            specialized,
        );
        let generic_identity = SourcePlan {
            convert: generic,
            ..identity.clone()
        };
        assert_same(
            &identity,
            &generic_identity,
            &task,
            &data,
            &indices,
            len * dst_size,
            fill,
        );

        let mapped_cols = len.div_ceil(3);
        let output_cols = mapped_cols.saturating_add(3);
        let mut targets = vec![UNMAPPED_TARGET_U32; len];
        for (target, column) in (0..len).step_by(3).enumerate() {
            targets[column] = u32::try_from(target * dst_size).unwrap();
        }
        let logical_targets = targets
            .iter()
            .map(|target| (*target != UNMAPPED_TARGET_U32).then_some(*target as usize / dst_size))
            .collect::<Vec<_>>();
        let defaults = build_default_ranges(Some(&logical_targets), output_cols, dst_size).unwrap();
        let mapped = source(
            len,
            src,
            IndexOp::new(index_dtype),
            None,
            false,
            defaults,
            specialized,
        );
        let mapped = SourcePlan {
            feature_map: Some(CsrMap::Packed32(Arc::from(targets))),
            ..mapped
        };
        let generic_mapped = SourcePlan {
            convert: generic,
            ..mapped.clone()
        };
        assert_same(
            &mapped,
            &generic_mapped,
            &task,
            &data,
            &indices,
            output_cols * dst_size,
            fill,
        );

        let mut hybrid_dense = vec![UNMAPPED_TARGET_U32; len];
        if len != 0 {
            hybrid_dense[0] = 0;
        }
        let sparse_packed: Arc<[u64]> = if len == 0 {
            Default::default()
        } else {
            Arc::from([0u64])
        };
        let hybrid_packed = SourcePlan {
            feature_map: Some(CsrMap::Packed32(Arc::from(hybrid_dense.clone()))),
            csr_sparse_map: Some(CsrSparseMap::Packed32(sparse_packed)),
            default_ranges: Default::default(),
            ..identity.clone()
        };
        let dense_packed = SourcePlan {
            feature_map: Some(CsrMap::Packed32(Arc::from(hybrid_dense))),
            ..identity.clone()
        };
        assert_same(
            &hybrid_packed,
            &dense_packed,
            &task,
            &data,
            &indices,
            dst_size,
            fill,
        );

        let mut hybrid_wide = vec![UNMAPPED_TARGET; len];
        if len != 0 {
            hybrid_wide[0] = 0;
        }
        let sparse_wide: Arc<[CsrSparseMapEntry]> = if len == 0 {
            Default::default()
        } else {
            Arc::from([CsrSparseMapEntry {
                source_column: 0,
                target_byte: 0,
            }])
        };
        let hybrid_wide_source = SourcePlan {
            feature_map: Some(CsrMap::Wide(Arc::from(hybrid_wide.clone()))),
            csr_sparse_map: Some(CsrSparseMap::Wide(sparse_wide)),
            default_ranges: Default::default(),
            ..identity.clone()
        };
        let dense_wide = SourcePlan {
            feature_map: Some(CsrMap::Wide(Arc::from(hybrid_wide))),
            ..identity
        };
        assert_same(
            &hybrid_wide_source,
            &dense_wide,
            &task,
            &data,
            &indices,
            dst_size,
            fill,
        );
    }

    #[test]
    fn csr_sparse_cost_model_covers_measured_boundaries() {
        assert!(!csr_sparse_binary_is_cheaper(328, 16_384));
        assert!(!csr_sparse_binary_is_cheaper(328, 6_554));
        assert!(csr_sparse_binary_is_cheaper(66, 16_384));
        assert!(!csr_sparse_binary_is_cheaper(66, 6_554));
        assert!(!csr_sparse_binary_is_cheaper(66, 3_277));
        assert!(!csr_sparse_binary_is_cheaper(66, 1_639));
        assert!(!csr_sparse_binary_is_cheaper(66, 328));
        assert!(csr_sparse_binary_is_cheaper(33, 3_277));
        assert!(!csr_sparse_binary_is_cheaper(33, 1_639));
        assert!(!csr_sparse_binary_is_cheaper(33, 328));
    }

    #[test]
    fn compiler_builds_sparse_csr_sidecar_only_when_it_can_win() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("csr-sparse-sidecar");
        CsrWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
            .write(&[0u64, 1], &[0u32], &[3f32], [1, 32_768])
            .unwrap();

        let mut sparse_targets = vec![None; 32_768];
        for (target, source) in (0..32_768).step_by(1_000).enumerate() {
            sparse_targets[source] = Some(target);
        }
        let sparse_plan = compile(PlanSpec::new(
            vec![Source::new(0, Dataset::open(&path).unwrap())
                .feature_map(FeatureMap::new(sparse_targets).unwrap())],
            vec![RowRef::new(SourceId::new(0), 0)],
            OutputSpec::new(33, OutputDType::F32, Fill::F32(0.0)).unwrap(),
            1,
            1,
        ))
        .unwrap();
        assert!(matches!(
            sparse_plan.inner.source_plans[0].csr_sparse_map,
            Some(CsrSparseMap::Packed32(ref entries)) if entries.len() == 33
        ));

        let dense_targets = (0..32_768)
            .map(|source| (source % 2 == 0).then_some(source / 2))
            .collect::<Vec<_>>();
        let dense_plan = compile(PlanSpec::new(
            vec![Source::new(0, Dataset::open(&path).unwrap())
                .feature_map(FeatureMap::new(dense_targets).unwrap())],
            vec![RowRef::new(SourceId::new(0), 0)],
            OutputSpec::new(16_384, OutputDType::F32, Fill::F32(0.0)).unwrap(),
            1,
            1,
        ))
        .unwrap();
        assert!(dense_plan.inner.source_plans[0].csr_sparse_map.is_none());

        let mut session = sparse_plan
            .open(SessionConfig {
                worker_count: 1,
                initialize_workers: 1,
                initialize_inflight_io_ops: 1,
                io_mode: IoMode::Blocking,
                ..SessionConfig::default()
            })
            .unwrap();
        let batch = session.next_batch().unwrap().unwrap();
        let row = batch.row_as::<f32>(0).unwrap();
        assert_eq!(row[0], 3.0);
        assert!(row[1..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn compiler_folds_explicit_identity_maps_into_none_fastpaths() {
        let temporary = tempfile::tempdir().unwrap();
        let dense_path = temporary.path().join("dense-identity-map");
        sc_compress::DenseWriter::new(
            &dense_path,
            Partition::fixed_cells(1),
            Partition::fixed_cells(1),
        )
        .write(&[1u16, 2, 3, 4], [1, 4])
        .unwrap();
        let identity = FeatureMap::new((0..4).map(Some)).unwrap();
        let dense = compile(PlanSpec::new(
            vec![Source::new(0, Dataset::open(&dense_path).unwrap()).feature_map(identity.clone())],
            vec![RowRef::new(SourceId::new(0), 0)],
            OutputSpec::new(4, OutputDType::U16, Fill::U16(0)).unwrap(),
            1,
            1,
        ))
        .unwrap();
        let dense_source = &dense.inner.source_plans[0];
        assert!(dense_source.feature_map.is_none());
        assert!(dense_source.csr_sparse_map.is_none());
        assert!(dense_source.dense_map.is_none());

        let csr_path = temporary.path().join("csr-identity-map");
        CsrWriter::new(
            &csr_path,
            Partition::fixed_cells(1),
            Partition::fixed_cells(1),
        )
        .write(&[0u64, 2], &[0u32, 3], &[1u16, 4], [1, 4])
        .unwrap();
        let csr = compile(PlanSpec::new(
            vec![Source::new(0, Dataset::open(&csr_path).unwrap()).feature_map(identity)],
            vec![RowRef::new(SourceId::new(0), 0)],
            OutputSpec::new(4, OutputDType::U16, Fill::U16(0)).unwrap(),
            1,
            1,
        ))
        .unwrap();
        let csr_source = &csr.inner.source_plans[0];
        assert!(csr_source.feature_map.is_none());
        assert!(csr_source.csr_sparse_map.is_none());
        assert!(csr_source.dense_map.is_none());
    }

    fn source(
        n_cols: usize,
        dtype: StorageDType,
        index: Option<IndexOp>,
        dense_map: Option<DenseMap>,
        dense_fill_whole: bool,
        default_ranges: Arc<[crate::plan::OutputRange]>,
        convert: ConvertOp,
    ) -> SourcePlan {
        SourcePlan {
            n_cols,
            value_dtype: dtype,
            index,
            feature_map: None,
            csr_sparse_map: None,
            dense_map,
            dense_fill_whole,
            default_ranges,
            convert,
        }
    }

    fn assert_same(
        specialized: &SourcePlan,
        generic: &SourcePlan,
        task: &CellTask,
        data: &[u8],
        indices: &[u8],
        row_bytes: usize,
        fill: FillOp,
    ) {
        if specialized.requires_runtime_validation() {
            validate_row(specialized, task, data, indices).unwrap();
            validate_row(generic, task, data, indices).unwrap();
        }
        let mut actual = vec![0xA5; row_bytes];
        let mut expected = vec![0x5A; row_bytes];
        unsafe {
            // SAFETY: validation above or compiler-style infallible extents cover
            // these immutable inputs and distinct uniquely owned outputs.
            scatter_row_prevalidated(
                specialized,
                task,
                data,
                indices,
                &mut actual,
                row_bytes,
                fill,
            )
            .unwrap();
            // SAFETY: the generic operator uses the same validated extents.
            scatter_row_prevalidated(generic, task, data, indices, &mut expected, row_bytes, fill)
                .unwrap();
        }
        assert_eq!(actual, expected);
    }

    fn task(data: std::ops::Range<usize>, indices: Option<std::ops::Range<usize>>) -> CellTask {
        CellTask::new(OutputSlot::new(0).unwrap(), data, indices).unwrap()
    }

    fn output_spec(
        n_cols: usize,
        dtype: OutputDType,
        overflow: OverflowPolicy,
        kind: PromoteKind,
    ) -> OutputSpec {
        let mut output = OutputSpec::new(n_cols, dtype, zero(dtype))
            .unwrap()
            .overflow(overflow)
            .unwrap();
        if kind == PromoteKind::RoundingToFloat {
            output = output.float_cast(FloatCastPolicy::AllowRounding);
        }
        output
    }

    fn zero(dtype: OutputDType) -> Fill {
        match dtype {
            OutputDType::I16 => Fill::I16(0),
            OutputDType::I32 => Fill::I32(0),
            OutputDType::I64 => Fill::I64(0),
            OutputDType::U16 => Fill::U16(0),
            OutputDType::U32 => Fill::U32(0),
            OutputDType::U64 => Fill::U64(0),
            OutputDType::F32 => Fill::F32(0.0),
            OutputDType::F64 => Fill::F64(0.0),
        }
    }

    fn sentinel(dtype: OutputDType) -> Fill {
        match dtype {
            OutputDType::I16 => Fill::I16(7),
            OutputDType::I32 => Fill::I32(7),
            OutputDType::I64 => Fill::I64(7),
            OutputDType::U16 => Fill::U16(7),
            OutputDType::U32 => Fill::U32(7),
            OutputDType::U64 => Fill::U64(7),
            OutputDType::F32 => Fill::F32(7.0),
            OutputDType::F64 => Fill::F64(7.0),
        }
    }

    fn values(dtype: StorageDType, count: usize, invalid: bool) -> Vec<u8> {
        let mut output = Vec::with_capacity(count * dtype.size());
        for index in 0..count {
            match dtype {
                StorageDType::I16 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        -3i16
                    } else {
                        index as i16 % 251
                    })
                    .to_le_bytes(),
                ),
                StorageDType::I32 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        -3i32
                    } else {
                        index as i32 % 65_521
                    })
                    .to_le_bytes(),
                ),
                StorageDType::I64 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        -3i64
                    } else {
                        index as i64 * 17
                    })
                    .to_le_bytes(),
                ),
                StorageDType::U16 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        u16::MAX
                    } else {
                        index as u16 % 251
                    })
                    .to_le_bytes(),
                ),
                StorageDType::U32 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        u32::MAX
                    } else {
                        index as u32 * 17
                    })
                    .to_le_bytes(),
                ),
                StorageDType::U64 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        u64::MAX
                    } else {
                        index as u64 * 17
                    })
                    .to_le_bytes(),
                ),
                StorageDType::F32 => {
                    output.extend_from_slice(&((index as f32 - 17.0) * 0.25).to_le_bytes())
                }
                StorageDType::F64 => {
                    output.extend_from_slice(&((index as f64 - 17.0) * 0.25).to_le_bytes())
                }
            }
        }
        output
    }

    fn encode_indices(columns: &[usize], dtype: StorageDType) -> Vec<u8> {
        let mut output = Vec::with_capacity(columns.len() * dtype.size());
        for &column in columns {
            match dtype {
                StorageDType::U16 => output.extend_from_slice(&(column as u16).to_le_bytes()),
                StorageDType::U32 => output.extend_from_slice(&(column as u32).to_le_bytes()),
                _ => unreachable!(),
            }
        }
        output
    }
}
