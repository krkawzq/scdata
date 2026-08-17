//! Dense / CSR initialization and scatter using pre-bound output kernels.

use sc_compress::DType as StorageDType;

use crate::plan::{CsrMap, DenseMap, SourcePlan, UNMAPPED_TARGET, UNMAPPED_TARGET_U32};
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
