//! C-style runtime dispatch for storage → output numeric promotion kernels.

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
mod simd;

use sc_compress::DType as StorageDType;

use crate::dtype::{promote_kind, OutputDType, PromoteKind};
use crate::output::{FloatCastPolicy, OutputSpec, OverflowPolicy};
use crate::plan::{CsrMap, DenseMap, DenseMapEntry};
use crate::{Error, Result};

/// Bound conversion operator resolved once per source at compile time.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvertOp {
    pub(crate) src: StorageDType,
    pub(crate) dst: OutputDType,
    pub(crate) kind: PromoteKind,
    /// When true, range failure returns [`Error::Conversion`].
    pub(crate) fail_on_overflow: bool,
    /// When true (and not fail_on_overflow), write `fallback` instead of erroring.
    pub(crate) write_fallback: bool,
    pub(crate) fallback: [u8; 8],
    pub(crate) src_size: u8,
    pub(crate) dst_size: u8,
    pub(crate) src_shift: u8,
    pub(crate) dst_shift: u8,
    convert_1: Convert1Fn,
    convert_slice: ConvertSliceFn,
    convert_map_wide: ConvertMapFn,
    convert_map_packed: ConvertPackedMapFn,
    convert_map_gather32: ConvertGather32Fn,
    convert_csr_u16: Option<ConvertCsrFn>,
    convert_csr_u32: Option<ConvertCsrFn>,
    validate_slice: ValidateSliceFn,
}

type Convert1Fn = fn(input: &[u8], output: &mut [u8], op: &ConvertOp) -> Result<()>;
type ConvertSliceFn =
    unsafe fn(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()>;
type ConvertMapFn = unsafe fn(
    input: *const u8,
    output: *mut u8,
    entries: &[DenseMapEntry],
    op: &ConvertOp,
) -> Result<()>;
type ConvertPackedMapFn =
    unsafe fn(input: *const u8, output: *mut u8, entries: &[u64], op: &ConvertOp) -> Result<()>;
type ConvertGather32Fn = unsafe fn(
    input: *const u8,
    output: *mut u8,
    source_offsets: &[i32],
    target_byte: usize,
    op: &ConvertOp,
) -> Result<()>;
type ConvertCsrFn = unsafe fn(
    values: *const u8,
    indices: *const u8,
    output: *mut u8,
    count: usize,
    map: Option<&CsrMap>,
);
type ValidateSliceFn = unsafe fn(input: *const u8, count: usize, op: &ConvertOp) -> bool;

impl ConvertOp {
    pub(crate) fn resolve(src: StorageDType, output: &OutputSpec) -> Result<Self> {
        let kind = promote_kind(src, output.dtype).ok_or_else(|| {
            Error::Promote(format!(
                "cannot promote storage dtype {src} to output dtype {}",
                output.dtype
            ))
        })?;
        if kind == PromoteKind::RoundingToFloat && output.float_cast == FloatCastPolicy::ExactOnly {
            return Err(Error::Promote(format!(
                "storage dtype {src} to output dtype {} may round; select FloatCastPolicy::AllowRounding explicitly",
                output.dtype
            )));
        }
        let convert_1 = dispatch_fn(src, output.dtype).ok_or_else(|| {
            Error::Promote(format!(
                "missing convert kernel for {src} → {}",
                output.dtype
            ))
        })?;
        let checked_sign = kind == PromoteKind::CheckedSign;
        let (fail_on_overflow, write_fallback) = match &output.overflow {
            OverflowPolicy::Error => (checked_sign, false),
            OverflowPolicy::UseFill | OverflowPolicy::UseValue(_) => (false, checked_sign),
            OverflowPolicy::Unchecked => (false, false),
        };
        let convert_slice = dispatch_slice_fn(src, output.dtype, write_fallback);
        let convert_map_wide = dispatch_map_fn(src, output.dtype, write_fallback);
        let convert_map_packed = dispatch_packed_map_fn(src, output.dtype, write_fallback);
        let convert_map_gather32 = dispatch_gather32_fn(src, output.dtype, write_fallback);
        let convert_csr_u16 = dispatch_csr_fn::<2>(src, output.dtype, write_fallback);
        let convert_csr_u32 = dispatch_csr_fn::<4>(src, output.dtype, write_fallback);
        let validate_slice = dispatch_validate_slice_fn(src, output.dtype, fail_on_overflow);
        let src_size = src.size() as u8;
        let dst_size = output.dtype.size() as u8;
        Ok(Self {
            src,
            dst: output.dtype,
            kind,
            fail_on_overflow,
            write_fallback,
            fallback: output.fallback_bytes(),
            src_size,
            dst_size,
            src_shift: src_size.trailing_zeros() as u8,
            dst_shift: dst_size.trailing_zeros() as u8,
            convert_1,
            convert_slice,
            convert_map_wide,
            convert_map_packed,
            convert_map_gather32,
            convert_csr_u16,
            convert_csr_u32,
            validate_slice,
        })
    }

    #[cfg(test)]
    pub(crate) fn force_scalar_for_test(&mut self) {
        self.convert_slice = scalar_slice;
    }

    #[cfg(all(test, target_arch = "x86_64", target_endian = "little"))]
    pub(crate) fn force_sse2_for_test(&mut self) {
        if let Some(kernel) = simd::dispatch_sse2(self.src, self.dst) {
            self.convert_slice = kernel;
        }
    }

    #[cfg(all(test, target_arch = "x86_64", target_endian = "little"))]
    pub(crate) fn force_avx2_for_test(&mut self) {
        if let Some(kernel) = simd::dispatch_avx2(self.src, self.dst) {
            self.convert_slice = kernel;
        }
    }

    #[cfg(all(test, target_arch = "x86_64", target_endian = "little"))]
    pub(crate) fn force_avx512_for_test(&mut self) {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            if let Some(kernel) = simd::dispatch_avx512(self.src, self.dst) {
                self.convert_slice = kernel;
            }
        }
    }

    /// Validate the only conversion class that can fail during execution.
    #[inline]
    pub(crate) fn validate_one(&self, input: &[u8]) -> Result<()> {
        if !self.fail_on_overflow || self.kind != PromoteKind::CheckedSign {
            return Ok(());
        }
        use OutputDType as O;
        use StorageDType as S;
        let valid = match (self.src, self.dst) {
            (S::I16, O::U16 | O::U32) => load_i16(input) >= 0,
            (S::I32, O::U32) => load_i32(input) >= 0,
            (S::U16, O::I16) => load_u16(input) <= i16::MAX as u16,
            (S::U32, O::I32) => load_u32(input) <= i32::MAX as u32,
            _ => true,
        };
        if valid {
            Ok(())
        } else {
            Err(Error::Conversion(format!(
                "value cannot convert {} → {} without overflow",
                self.src, self.dst
            )))
        }
    }

    pub(crate) fn can_fail(&self) -> bool {
        self.fail_on_overflow && self.kind == PromoteKind::CheckedSign
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.src == self.dst.to_storage()
    }

    pub(crate) fn dense_gather_min_entries(&self) -> Option<usize> {
        if self.src_size == 4 && self.dst_size == 4 {
            return Some(16);
        }
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        if self.src == StorageDType::I32
            && self.dst == OutputDType::F64
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
        {
            return Some(8);
        }
        None
    }

    /// Validate a contiguous source range before entering the commit phase.
    #[inline]
    pub(crate) fn validate_slice(&self, input: &[u8]) -> Result<()> {
        if !self.can_fail() {
            return Ok(());
        }
        let src_size = usize::from(self.src_size);
        if input.len() & (src_size - 1) != 0 {
            return Err(Error::Invariant(
                "conversion validation input is not element-aligned".into(),
            ));
        }
        // SAFETY: the alignment check proves that `input` contains `count`
        // complete elements. The validator only reads that immutable range.
        let valid =
            unsafe { (self.validate_slice)(input.as_ptr(), input.len() >> self.src_shift, self) };
        if valid {
            Ok(())
        } else {
            Err(Error::Conversion(format!(
                "value cannot convert {} → {} without overflow",
                self.src, self.dst
            )))
        }
    }

    /// Convert a contiguous row after the caller has completed range checks.
    #[cfg(test)]
    #[inline]
    pub(crate) fn convert_slice_prevalidated(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let src_size = usize::from(self.src_size);
        if input.len() & (src_size - 1) != 0 {
            return Err(Error::Invariant(
                "prevalidated conversion input is not element-aligned".into(),
            ));
        }
        let count = input.len() >> self.src_shift;
        let output_bytes = count
            .checked_shl(u32::from(self.dst_shift))
            .ok_or_else(|| Error::Invariant("conversion output byte count overflow".into()))?;
        if output.len() != output_bytes {
            return Err(Error::Invariant(
                "prevalidated conversion output length is inconsistent".into(),
            ));
        }
        // SAFETY: exact input/output byte lengths above prove that the bound
        // kernel can access `count` source and destination elements. The
        // buffers are distinct decoded/output allocations, so they do not
        // overlap. Checked-sign inputs were validated before commit.
        unsafe { (self.convert_slice)(input.as_ptr(), output.as_mut_ptr(), count, self) }
    }

    #[inline(always)]
    pub(crate) unsafe fn convert_slice_unchecked(
        &self,
        input: *const u8,
        output: *mut u8,
        count: usize,
    ) -> Result<()> {
        // SAFETY: caller proves the raw buffers contain `count` complete,
        // non-overlapping source/destination elements and conversion policy was
        // validated before commit.
        unsafe { (self.convert_slice)(input, output, count, self) }
    }

    #[inline(always)]
    pub(crate) unsafe fn convert_one_prevalidated(
        &self,
        input: *const u8,
        output: *mut u8,
    ) -> Result<()> {
        // SAFETY: the caller guarantees one complete source and destination
        // element. The slices are exact and do not outlive this kernel call.
        let (input, output) = unsafe {
            (
                std::slice::from_raw_parts(input, usize::from(self.src_size)),
                std::slice::from_raw_parts_mut(output, usize::from(self.dst_size)),
            )
        };
        (self.convert_1)(input, output, self)
    }

    #[inline(always)]
    pub(crate) unsafe fn convert_map_prevalidated(
        &self,
        input: *const u8,
        output: *mut u8,
        entries: &DenseMap,
    ) -> Result<()> {
        match entries {
            DenseMap::Packed32 { entries, .. } => {
                // SAFETY: the caller proves every packed compiler-built offset
                // addresses one complete element in distinct allocations.
                unsafe { (self.convert_map_packed)(input, output, entries, self) }
            }
            DenseMap::Gather32 {
                source_offsets,
                target_byte,
                ..
            } => {
                // SAFETY: compiler-built signed offsets address complete source
                // elements and map in order to one contiguous target range.
                unsafe {
                    (self.convert_map_gather32)(
                        input,
                        output,
                        source_offsets,
                        *target_byte as usize,
                        self,
                    )
                }
            }
            DenseMap::Wide { entries, .. } => {
                // SAFETY: the same proof covers the wide-offset representation.
                unsafe { (self.convert_map_wide)(input, output, entries, self) }
            }
            DenseMap::Runs { entries, .. } => {
                for run in entries.iter() {
                    // SAFETY: compiler-built runs contain contiguous, complete,
                    // non-overlapping source and destination element ranges.
                    unsafe {
                        (self.convert_slice)(
                            input.add(run.source_byte),
                            output.add(run.target_byte),
                            run.count,
                            self,
                        )?;
                    }
                }
                Ok(())
            }
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn convert_csr_prevalidated(
        &self,
        values: *const u8,
        indices: *const u8,
        output: *mut u8,
        count: usize,
        index_size: usize,
        map: Option<&CsrMap>,
    ) -> bool {
        let kernel = match index_size {
            2 => self.convert_csr_u16,
            4 => self.convert_csr_u32,
            _ => None,
        };
        let Some(kernel) = kernel else {
            return false;
        };
        // SAFETY: validation proved both source arrays contain `count`
        // elements, all indices/map targets are in range, and output is unique.
        unsafe { kernel(values, indices, output, count, map) };
        true
    }
}

fn dispatch_csr_fn<const INDEX_BYTES: usize>(
    src: StorageDType,
    dst: OutputDType,
    write_fallback: bool,
) -> Option<ConvertCsrFn> {
    if write_fallback {
        return None;
    }
    use OutputDType as O;
    use StorageDType as S;
    Some(match (src, dst) {
        (S::I16 | S::U16, O::I16 | O::U16) => csr_copy::<INDEX_BYTES, 2>,
        (S::I32 | S::U32, O::I32 | O::U32) | (S::F32, O::F32) => csr_copy::<INDEX_BYTES, 4>,
        (S::F64, O::F64) => csr_copy::<INDEX_BYTES, 8>,
        (S::I16, O::I32) => csr_i16_i32::<INDEX_BYTES>,
        (S::I16, O::U32) => csr_i16_u32::<INDEX_BYTES>,
        (S::U16, O::I32) => csr_u16_i32::<INDEX_BYTES>,
        (S::U16, O::U32) => csr_u16_u32::<INDEX_BYTES>,
        (S::I16, O::F32) => csr_i16_f32::<INDEX_BYTES>,
        (S::U16, O::F32) => csr_u16_f32::<INDEX_BYTES>,
        (S::I32, O::F32) => csr_i32_f32::<INDEX_BYTES>,
        (S::U32, O::F32) => csr_u32_f32::<INDEX_BYTES>,
        (S::I16, O::F64) => csr_i16_f64::<INDEX_BYTES>,
        (S::U16, O::F64) => csr_u16_f64::<INDEX_BYTES>,
        (S::I32, O::F64) => csr_i32_f64::<INDEX_BYTES>,
        (S::U32, O::F64) => csr_u32_f64::<INDEX_BYTES>,
        (S::F32, O::F64) => csr_f32_f64::<INDEX_BYTES>,
        _ => return None,
    })
}

fn dispatch_map_fn(src: StorageDType, dst: OutputDType, write_fallback: bool) -> ConvertMapFn {
    if write_fallback {
        return scalar_map;
    }
    use OutputDType as O;
    use StorageDType as S;
    match (src, dst) {
        (S::I16 | S::U16, O::I16 | O::U16) => mapped_copy::<2>,
        (S::I32 | S::U32, O::I32 | O::U32) | (S::F32, O::F32) => mapped_copy::<4>,
        (S::F64, O::F64) => mapped_copy::<8>,
        (S::I16, O::I32) => mapped_i16_i32,
        (S::I16, O::U32) => mapped_i16_u32,
        (S::U16, O::I32) => mapped_u16_i32,
        (S::U16, O::U32) => mapped_u16_u32,
        (S::I16, O::F32) => mapped_i16_f32,
        (S::U16, O::F32) => mapped_u16_f32,
        (S::I32, O::F32) => mapped_i32_f32,
        (S::U32, O::F32) => mapped_u32_f32,
        (S::I16, O::F64) => mapped_i16_f64,
        (S::U16, O::F64) => mapped_u16_f64,
        (S::I32, O::F64) => mapped_i32_f64,
        (S::U32, O::F64) => mapped_u32_f64,
        (S::F32, O::F64) => mapped_f32_f64,
        _ => scalar_map,
    }
}

fn dispatch_packed_map_fn(
    src: StorageDType,
    dst: OutputDType,
    write_fallback: bool,
) -> ConvertPackedMapFn {
    if write_fallback {
        return scalar_packed_map;
    }
    use OutputDType as O;
    use StorageDType as S;
    match (src, dst) {
        (S::I16 | S::U16, O::I16 | O::U16) => packed_copy::<2>,
        (S::I32 | S::U32, O::I32 | O::U32) | (S::F32, O::F32) => packed_copy::<4>,
        (S::F64, O::F64) => packed_copy::<8>,
        (S::I16, O::I32) => packed_i16_i32,
        (S::I16, O::U32) => packed_i16_u32,
        (S::U16, O::I32) => packed_u16_i32,
        (S::U16, O::U32) => packed_u16_u32,
        (S::I16, O::F32) => packed_i16_f32,
        (S::U16, O::F32) => packed_u16_f32,
        (S::I32, O::F32) => packed_i32_f32,
        (S::U32, O::F32) => packed_u32_f32,
        (S::I16, O::F64) => packed_i16_f64,
        (S::U16, O::F64) => packed_u16_f64,
        (S::I32, O::F64) => packed_i32_f64,
        (S::U32, O::F64) => packed_u32_f64,
        (S::F32, O::F64) => packed_f32_f64,
        _ => scalar_packed_map,
    }
}

fn dispatch_gather32_fn(
    src: StorageDType,
    dst: OutputDType,
    write_fallback: bool,
) -> ConvertGather32Fn {
    if write_fallback {
        return scalar_gather32;
    }
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if std::arch::is_x86_feature_detected!("avx512f") {
        if let Some(kernel) = simd::dispatch_gather32_avx512(src, dst) {
            return kernel;
        }
    }
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        if let Some(kernel) = simd::dispatch_gather32_avx2(src, dst) {
            return kernel;
        }
    }
    use OutputDType as O;
    use StorageDType as S;
    match (src, dst) {
        (S::I32 | S::U32, O::I32 | O::U32) | (S::F32, O::F32) => gather32_copy,
        _ => scalar_gather32,
    }
}

fn dispatch_validate_slice_fn(
    src: StorageDType,
    dst: OutputDType,
    fail_on_overflow: bool,
) -> ValidateSliceFn {
    if fail_on_overflow {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            if let Some(kernel) = simd::dispatch_validate_avx512(src, dst) {
                return kernel;
            }
        }
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            if let Some(kernel) = simd::dispatch_validate_avx2(src, dst) {
                return kernel;
            }
        }
    }
    scalar_validate_slice
}

fn dispatch_slice_fn(src: StorageDType, dst: OutputDType, write_fallback: bool) -> ConvertSliceFn {
    if write_fallback {
        return scalar_slice;
    }

    use OutputDType as O;
    use StorageDType as S;
    if matches!(
        (src, dst),
        (S::I16, O::I16 | O::U16)
            | (S::I32, O::I32 | O::U32)
            | (S::U16, O::U16 | O::I16)
            | (S::U32, O::U32 | O::I32)
            | (S::F32, O::F32)
            | (S::F64, O::F64)
    ) {
        return copy_slice;
    }

    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
    {
        if let Some(kernel) = simd::dispatch_avx512(src, dst) {
            return kernel;
        }
    }
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        if let Some(kernel) = simd::dispatch_avx2(src, dst) {
            return kernel;
        }
    }
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if let Some(kernel) = simd::dispatch_sse2(src, dst) {
        return kernel;
    }
    scalar_slice
}

unsafe fn copy_slice(
    input: *const u8,
    output: *mut u8,
    count: usize,
    op: &ConvertOp,
) -> Result<()> {
    // The caller's allocation proof implies this mathematical product fits
    // usize; no real allocation can contain an overflowed element extent.
    let bytes = count << op.src_shift;
    // SAFETY: `convert_slice_prevalidated` proves both allocations contain
    // `bytes` accessible bytes and that decoded input and output do not overlap.
    unsafe { std::ptr::copy_nonoverlapping(input, output, bytes) };
    Ok(())
}

unsafe fn scalar_slice(
    input: *const u8,
    output: *mut u8,
    count: usize,
    op: &ConvertOp,
) -> Result<()> {
    let src_size = usize::from(op.src_size);
    let dst_size = usize::from(op.dst_size);
    for index in 0..count {
        // SAFETY: the caller proves `count * {src,dst}_size` bytes are valid;
        // each iteration constructs one disjoint, exactly sized element view.
        let (source, destination) = unsafe {
            (
                std::slice::from_raw_parts(input.add(index << op.src_shift), src_size),
                std::slice::from_raw_parts_mut(output.add(index << op.dst_shift), dst_size),
            )
        };
        (op.convert_1)(source, destination, op)?;
    }
    Ok(())
}

unsafe fn scalar_validate_slice(input: *const u8, count: usize, op: &ConvertOp) -> bool {
    // All currently fallible conversions are same-width signedness edges. OR
    // reduction turns per-element range branches into one sign-bit test.
    if op.src_size == 2 {
        let mut bits = 0u16;
        for index in 0..count {
            // SAFETY: the caller proves `count` complete two-byte elements.
            bits |= u16::from_le(unsafe { input.add(index << 1).cast::<u16>().read_unaligned() });
        }
        bits & 0x8000 == 0
    } else {
        debug_assert_eq!(op.src_size, 4);
        let mut bits = 0u32;
        for index in 0..count {
            // SAFETY: the caller proves `count` complete four-byte elements.
            bits |= u32::from_le(unsafe { input.add(index << 2).cast::<u32>().read_unaligned() });
        }
        bits & 0x8000_0000 == 0
    }
}

unsafe fn scalar_map(
    input: *const u8,
    output: *mut u8,
    entries: &[DenseMapEntry],
    op: &ConvertOp,
) -> Result<()> {
    for entry in entries {
        // SAFETY: caller proves all precompiled entry offsets point at complete
        // source/destination elements and mapped conversions were validated.
        let (source, destination) = unsafe {
            (
                std::slice::from_raw_parts(input.add(entry.source_byte), usize::from(op.src_size)),
                std::slice::from_raw_parts_mut(
                    output.add(entry.target_byte),
                    usize::from(op.dst_size),
                ),
            )
        };
        (op.convert_1)(source, destination, op)?;
    }
    Ok(())
}

#[inline(always)]
fn unpack_dense_offsets(entry: u64) -> (usize, usize) {
    (entry as u32 as usize, (entry >> 32) as u32 as usize)
}

unsafe fn scalar_packed_map(
    input: *const u8,
    output: *mut u8,
    entries: &[u64],
    op: &ConvertOp,
) -> Result<()> {
    for &entry in entries {
        let (source_byte, target_byte) = unpack_dense_offsets(entry);
        // SAFETY: caller proves both packed byte offsets point at one complete
        // source/destination element in non-overlapping allocations.
        let (source, destination) = unsafe {
            (
                std::slice::from_raw_parts(input.add(source_byte), usize::from(op.src_size)),
                std::slice::from_raw_parts_mut(output.add(target_byte), usize::from(op.dst_size)),
            )
        };
        (op.convert_1)(source, destination, op)?;
    }
    Ok(())
}

unsafe fn scalar_gather32(
    input: *const u8,
    output: *mut u8,
    source_offsets: &[i32],
    target_byte: usize,
    op: &ConvertOp,
) -> Result<()> {
    for (index, &source_byte) in source_offsets.iter().enumerate() {
        // SAFETY: compiler-built offsets are nonnegative and point at complete
        // source elements; the target run contains one element per offset.
        unsafe {
            op.convert_one_prevalidated(
                input.add(source_byte as usize),
                output.add(target_byte + (index << op.dst_shift)),
            )?;
        }
    }
    Ok(())
}

unsafe fn gather32_copy(
    input: *const u8,
    output: *mut u8,
    source_offsets: &[i32],
    target_byte: usize,
    _op: &ConvertOp,
) -> Result<()> {
    for (index, &source_byte) in source_offsets.iter().enumerate() {
        // SAFETY: compiler-built offsets cover one complete u32 in distinct
        // allocations and the destination run has the same element count.
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.add(source_byte as usize),
                output.add(target_byte + index * 4),
                4,
            );
        }
    }
    Ok(())
}

unsafe fn mapped_copy<const BYTES: usize>(
    input: *const u8,
    output: *mut u8,
    entries: &[DenseMapEntry],
    _op: &ConvertOp,
) -> Result<()> {
    for entry in entries {
        // SAFETY: compiler-built offsets and the caller contract prove both
        // ranges contain BYTES accessible, non-overlapping bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.add(entry.source_byte),
                output.add(entry.target_byte),
                BYTES,
            );
        }
    }
    Ok(())
}

unsafe fn packed_copy<const BYTES: usize>(
    input: *const u8,
    output: *mut u8,
    entries: &[u64],
    _op: &ConvertOp,
) -> Result<()> {
    for &entry in entries {
        let (source_byte, target_byte) = unpack_dense_offsets(entry);
        // SAFETY: compiler-packed offsets and the caller contract prove both
        // complete BYTES-wide ranges are valid and non-overlapping.
        unsafe {
            std::ptr::copy_nonoverlapping(input.add(source_byte), output.add(target_byte), BYTES);
        }
    }
    Ok(())
}

macro_rules! mapped_kernel {
    ($name:ident, $load:ident, $store:ident, $convert:expr) => {
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            entries: &[DenseMapEntry],
            _op: &ConvertOp,
        ) -> Result<()> {
            let convert = $convert;
            for entry in entries {
                // SAFETY: compiler-built byte offsets point at one complete
                // source and destination element in non-overlapping buffers.
                unsafe {
                    let value = $load(input.add(entry.source_byte));
                    $store(output.add(entry.target_byte), convert(value));
                }
            }
            Ok(())
        }
    };
}

macro_rules! packed_kernel {
    ($name:ident, $load:ident, $store:ident, $convert:expr) => {
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            entries: &[u64],
            _op: &ConvertOp,
        ) -> Result<()> {
            let convert = $convert;
            for &entry in entries {
                let (source_byte, target_byte) = unpack_dense_offsets(entry);
                // SAFETY: compiler-packed offsets point at one complete source
                // and destination element in non-overlapping buffers.
                unsafe {
                    let value = $load(input.add(source_byte));
                    $store(output.add(target_byte), convert(value));
                }
            }
            Ok(())
        }
    };
}

mapped_kernel!(mapped_i16_i32, load_i16_ptr, store_i32_ptr, |value: i16| {
    i32::from(value)
});
mapped_kernel!(mapped_i16_u32, load_i16_ptr, store_u32_ptr, |value: i16| {
    value as u32
});
mapped_kernel!(mapped_u16_i32, load_u16_ptr, store_i32_ptr, |value: u16| {
    i32::from(value)
});
mapped_kernel!(mapped_u16_u32, load_u16_ptr, store_u32_ptr, |value: u16| {
    u32::from(value)
});
mapped_kernel!(mapped_i16_f32, load_i16_ptr, store_f32_ptr, |value: i16| {
    f32::from(value)
});
mapped_kernel!(mapped_u16_f32, load_u16_ptr, store_f32_ptr, |value: u16| {
    f32::from(value)
});
mapped_kernel!(mapped_i32_f32, load_i32_ptr, store_f32_ptr, |value: i32| {
    value as f32
});
mapped_kernel!(mapped_u32_f32, load_u32_ptr, store_f32_ptr, |value: u32| {
    value as f32
});
mapped_kernel!(mapped_i16_f64, load_i16_ptr, store_f64_ptr, |value: i16| {
    f64::from(value)
});
mapped_kernel!(mapped_u16_f64, load_u16_ptr, store_f64_ptr, |value: u16| {
    f64::from(value)
});
mapped_kernel!(mapped_i32_f64, load_i32_ptr, store_f64_ptr, |value: i32| {
    f64::from(value)
});
mapped_kernel!(mapped_u32_f64, load_u32_ptr, store_f64_ptr, |value: u32| {
    f64::from(value)
});
mapped_kernel!(mapped_f32_f64, load_f32_ptr, store_f64_ptr, |value: f32| {
    f64::from(value)
});

packed_kernel!(packed_i16_i32, load_i16_ptr, store_i32_ptr, |value: i16| {
    i32::from(value)
});
packed_kernel!(packed_i16_u32, load_i16_ptr, store_u32_ptr, |value: i16| {
    value as u32
});
packed_kernel!(packed_u16_i32, load_u16_ptr, store_i32_ptr, |value: u16| {
    i32::from(value)
});
packed_kernel!(packed_u16_u32, load_u16_ptr, store_u32_ptr, |value: u16| {
    u32::from(value)
});
packed_kernel!(packed_i16_f32, load_i16_ptr, store_f32_ptr, |value: i16| {
    f32::from(value)
});
packed_kernel!(packed_u16_f32, load_u16_ptr, store_f32_ptr, |value: u16| {
    f32::from(value)
});
packed_kernel!(packed_i32_f32, load_i32_ptr, store_f32_ptr, |value: i32| {
    value as f32
});
packed_kernel!(packed_u32_f32, load_u32_ptr, store_f32_ptr, |value: u32| {
    value as f32
});
packed_kernel!(packed_i16_f64, load_i16_ptr, store_f64_ptr, |value: i16| {
    f64::from(value)
});
packed_kernel!(packed_u16_f64, load_u16_ptr, store_f64_ptr, |value: u16| {
    f64::from(value)
});
packed_kernel!(packed_i32_f64, load_i32_ptr, store_f64_ptr, |value: i32| {
    f64::from(value)
});
packed_kernel!(packed_u32_f64, load_u32_ptr, store_f64_ptr, |value: u32| {
    f64::from(value)
});
packed_kernel!(packed_f32_f64, load_f32_ptr, store_f64_ptr, |value: f32| {
    f64::from(value)
});

const CSR_MAP_IDENTITY: u8 = 0;
const CSR_MAP_PACKED: u8 = 1;
const CSR_MAP_WIDE: u8 = 2;

#[inline(always)]
unsafe fn csr_target<const INDEX_BYTES: usize, const DST_BYTES: usize, const MAP_KIND: u8>(
    indices: *const u8,
    element: usize,
    packed_map: *const u32,
    wide_map: *const usize,
) -> usize {
    let col = if INDEX_BYTES == 2 {
        // SAFETY: caller proves one complete u16 index at this element.
        usize::from(u16::from_le(unsafe {
            indices.add(element * 2).cast::<u16>().read_unaligned()
        }))
    } else {
        // SAFETY: dispatch only instantiates INDEX_BYTES=4 here and the caller
        // proves one complete u32 index at this element.
        u32::from_le(unsafe { indices.add(element * 4).cast::<u32>().read_unaligned() }) as usize
    };
    if MAP_KIND == CSR_MAP_IDENTITY {
        col * DST_BYTES
    } else if MAP_KIND == CSR_MAP_PACKED {
        // SAFETY: structural validation proved col is within the packed map.
        let target = unsafe { *packed_map.add(col) };
        if target == u32::MAX {
            usize::MAX
        } else {
            target as usize
        }
    } else {
        debug_assert_eq!(MAP_KIND, CSR_MAP_WIDE);
        // SAFETY: structural validation proved col is within the wide map.
        unsafe { *wide_map.add(col) }
    }
}

macro_rules! dispatch_csr_map {
    ($kernel:ident, $values:expr, $indices:expr, $output:expr, $count:expr, $map:expr) => {{
        // SAFETY: the wrapper's prevalidation contract covers the selected map
        // extent. Each specialization receives only the pointer it can read.
        unsafe {
            match $map {
                None => $kernel::<INDEX_BYTES, CSR_MAP_IDENTITY>(
                    $values,
                    $indices,
                    $output,
                    $count,
                    std::ptr::null(),
                    std::ptr::null(),
                ),
                Some(CsrMap::Packed32(entries)) => $kernel::<INDEX_BYTES, CSR_MAP_PACKED>(
                    $values,
                    $indices,
                    $output,
                    $count,
                    entries.as_ptr(),
                    std::ptr::null(),
                ),
                Some(CsrMap::Wide(entries)) => $kernel::<INDEX_BYTES, CSR_MAP_WIDE>(
                    $values,
                    $indices,
                    $output,
                    $count,
                    std::ptr::null(),
                    entries.as_ptr(),
                ),
            }
        }
    }};
}

unsafe fn csr_copy<const INDEX_BYTES: usize, const BYTES: usize>(
    values: *const u8,
    indices: *const u8,
    output: *mut u8,
    count: usize,
    map: Option<&CsrMap>,
) {
    #[inline(always)]
    unsafe fn kernel<const INDEX_BYTES: usize, const MAP_KIND: u8, const BYTES: usize>(
        values: *const u8,
        indices: *const u8,
        output: *mut u8,
        count: usize,
        packed_map: *const u32,
        wide_map: *const usize,
    ) {
        for element in 0..count {
            // SAFETY: caller proved all CSR indices and mapped byte targets valid.
            let target = unsafe {
                csr_target::<INDEX_BYTES, BYTES, MAP_KIND>(indices, element, packed_map, wide_map)
            };
            if target != usize::MAX {
                // SAFETY: element and target address one complete non-overlapping
                // value in the validated input/output allocations.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        values.add(element * BYTES),
                        output.add(target),
                        BYTES,
                    );
                }
            }
        }
    }

    // SAFETY: the wrapper's prevalidation contract covers the selected map
    // extent. Each specialization receives only the pointer it can read.
    unsafe {
        match map {
            None => kernel::<INDEX_BYTES, CSR_MAP_IDENTITY, BYTES>(
                values,
                indices,
                output,
                count,
                std::ptr::null(),
                std::ptr::null(),
            ),
            Some(CsrMap::Packed32(entries)) => kernel::<INDEX_BYTES, CSR_MAP_PACKED, BYTES>(
                values,
                indices,
                output,
                count,
                entries.as_ptr(),
                std::ptr::null(),
            ),
            Some(CsrMap::Wide(entries)) => kernel::<INDEX_BYTES, CSR_MAP_WIDE, BYTES>(
                values,
                indices,
                output,
                count,
                std::ptr::null(),
                entries.as_ptr(),
            ),
        }
    }
}

macro_rules! csr_kernel {
    ($name:ident, $src_bytes:expr, $dst_bytes:expr, $load:ident, $store:ident, $convert:expr) => {
        unsafe fn $name<const INDEX_BYTES: usize>(
            values: *const u8,
            indices: *const u8,
            output: *mut u8,
            count: usize,
            map: Option<&CsrMap>,
        ) {
            #[inline(always)]
            unsafe fn kernel<const INDEX_BYTES: usize, const MAP_KIND: u8>(
                values: *const u8,
                indices: *const u8,
                output: *mut u8,
                count: usize,
                packed_map: *const u32,
                wide_map: *const usize,
            ) {
                let convert = $convert;
                // SAFETY: caller proved all `count` CSR indices, source
                // elements, selected map targets, and destination elements are
                // valid. Every iteration touches one complete value.
                for element in 0..count {
                    // SAFETY: the kernel contract proves this element's index
                    // and selected map entry are present and in range.
                    let target = unsafe {
                        csr_target::<INDEX_BYTES, $dst_bytes, MAP_KIND>(
                            indices, element, packed_map, wide_map,
                        )
                    };
                    if target != usize::MAX {
                        // SAFETY: the caller's extent proof covers these
                        // complete source and destination elements.
                        unsafe {
                            let value = $load(values.add(element * $src_bytes));
                            $store(output.add(target), convert(value));
                        }
                    }
                }
            }

            dispatch_csr_map!(kernel, values, indices, output, count, map);
        }
    };
}

csr_kernel!(csr_i16_i32, 2, 4, load_i16_ptr, store_i32_ptr, |v: i16| {
    i32::from(v)
});
csr_kernel!(csr_i16_u32, 2, 4, load_i16_ptr, store_u32_ptr, |v: i16| v
    as u32);
csr_kernel!(csr_u16_i32, 2, 4, load_u16_ptr, store_i32_ptr, |v: u16| {
    i32::from(v)
});
csr_kernel!(csr_u16_u32, 2, 4, load_u16_ptr, store_u32_ptr, |v: u16| {
    u32::from(v)
});
csr_kernel!(csr_i16_f32, 2, 4, load_i16_ptr, store_f32_ptr, |v: i16| {
    f32::from(v)
});
csr_kernel!(csr_u16_f32, 2, 4, load_u16_ptr, store_f32_ptr, |v: u16| {
    f32::from(v)
});
csr_kernel!(csr_i32_f32, 4, 4, load_i32_ptr, store_f32_ptr, |v: i32| v
    as f32);
csr_kernel!(csr_u32_f32, 4, 4, load_u32_ptr, store_f32_ptr, |v: u32| v
    as f32);
csr_kernel!(csr_i16_f64, 2, 8, load_i16_ptr, store_f64_ptr, |v: i16| {
    f64::from(v)
});
csr_kernel!(csr_u16_f64, 2, 8, load_u16_ptr, store_f64_ptr, |v: u16| {
    f64::from(v)
});
csr_kernel!(csr_i32_f64, 4, 8, load_i32_ptr, store_f64_ptr, |v: i32| {
    f64::from(v)
});
csr_kernel!(csr_u32_f64, 4, 8, load_u32_ptr, store_f64_ptr, |v: u32| {
    f64::from(v)
});
csr_kernel!(csr_f32_f64, 4, 8, load_f32_ptr, store_f64_ptr, |v: f32| {
    f64::from(v)
});

fn dispatch_fn(src: StorageDType, dst: OutputDType) -> Option<Convert1Fn> {
    use OutputDType as O;
    use StorageDType as S;
    Some(match (src, dst) {
        (S::I16, O::I16) => kernel_i16_i16,
        (S::I16, O::I32) => kernel_i16_i32,
        (S::I16, O::U16) => kernel_i16_u16,
        (S::I16, O::U32) => kernel_i16_u32,
        (S::I16, O::F32) => kernel_i16_f32,
        (S::I16, O::F64) => kernel_i16_f64,
        (S::I32, O::I32) => kernel_i32_i32,
        (S::I32, O::U32) => kernel_i32_u32,
        (S::I32, O::F32) => kernel_i32_f32,
        (S::I32, O::F64) => kernel_i32_f64,
        (S::U16, O::U16) => kernel_u16_u16,
        (S::U16, O::U32) => kernel_u16_u32,
        (S::U16, O::I16) => kernel_u16_i16,
        (S::U16, O::I32) => kernel_u16_i32,
        (S::U16, O::F32) => kernel_u16_f32,
        (S::U16, O::F64) => kernel_u16_f64,
        (S::U32, O::U32) => kernel_u32_u32,
        (S::U32, O::I32) => kernel_u32_i32,
        (S::U32, O::F32) => kernel_u32_f32,
        (S::U32, O::F64) => kernel_u32_f64,
        (S::F32, O::F32) => kernel_f32_f32,
        (S::F32, O::F64) => kernel_f32_f64,
        (S::F64, O::F64) => kernel_f64_f64,
        _ => return None,
    })
}

fn load_i16(input: &[u8]) -> i16 {
    i16::from_le_bytes(input[..2].try_into().unwrap())
}
fn load_i32(input: &[u8]) -> i32 {
    i32::from_le_bytes(input[..4].try_into().unwrap())
}
fn load_u16(input: &[u8]) -> u16 {
    u16::from_le_bytes(input[..2].try_into().unwrap())
}
fn load_u32(input: &[u8]) -> u32 {
    u32::from_le_bytes(input[..4].try_into().unwrap())
}
fn load_f32(input: &[u8]) -> f32 {
    f32::from_le_bytes(input[..4].try_into().unwrap())
}
fn store_i16(output: &mut [u8], value: i16) {
    output[..2].copy_from_slice(&value.to_le_bytes());
}
fn store_i32(output: &mut [u8], value: i32) {
    output[..4].copy_from_slice(&value.to_le_bytes());
}
fn store_u16(output: &mut [u8], value: u16) {
    output[..2].copy_from_slice(&value.to_le_bytes());
}
fn store_u32(output: &mut [u8], value: u32) {
    output[..4].copy_from_slice(&value.to_le_bytes());
}
fn store_f32(output: &mut [u8], value: f32) {
    output[..4].copy_from_slice(&value.to_le_bytes());
}
fn store_f64(output: &mut [u8], value: f64) {
    output[..8].copy_from_slice(&value.to_le_bytes());
}

#[inline(always)]
unsafe fn load_i16_ptr(input: *const u8) -> i16 {
    // SAFETY: caller guarantees one complete possibly unaligned i16 element.
    i16::from_le(unsafe { input.cast::<i16>().read_unaligned() })
}

#[inline(always)]
unsafe fn load_i32_ptr(input: *const u8) -> i32 {
    // SAFETY: caller guarantees one complete possibly unaligned i32 element.
    i32::from_le(unsafe { input.cast::<i32>().read_unaligned() })
}

#[inline(always)]
unsafe fn load_u16_ptr(input: *const u8) -> u16 {
    // SAFETY: caller guarantees one complete possibly unaligned u16 element.
    u16::from_le(unsafe { input.cast::<u16>().read_unaligned() })
}

#[inline(always)]
unsafe fn load_u32_ptr(input: *const u8) -> u32 {
    // SAFETY: caller guarantees one complete possibly unaligned u32 element.
    u32::from_le(unsafe { input.cast::<u32>().read_unaligned() })
}

#[inline(always)]
unsafe fn load_f32_ptr(input: *const u8) -> f32 {
    // SAFETY: caller guarantees one complete possibly unaligned f32 bit pattern.
    f32::from_bits(u32::from_le(unsafe {
        input.cast::<u32>().read_unaligned()
    }))
}

#[inline(always)]
unsafe fn store_i32_ptr(output: *mut u8, value: i32) {
    // SAFETY: caller guarantees one complete possibly unaligned i32 destination.
    unsafe { output.cast::<i32>().write_unaligned(value.to_le()) };
}

#[inline(always)]
unsafe fn store_u32_ptr(output: *mut u8, value: u32) {
    // SAFETY: caller guarantees one complete possibly unaligned u32 destination.
    unsafe { output.cast::<u32>().write_unaligned(value.to_le()) };
}

#[inline(always)]
unsafe fn store_f32_ptr(output: *mut u8, value: f32) {
    // SAFETY: caller guarantees one complete possibly unaligned f32 destination.
    unsafe {
        output
            .cast::<u32>()
            .write_unaligned(value.to_bits().to_le())
    };
}

#[inline(always)]
unsafe fn store_f64_ptr(output: *mut u8, value: f64) {
    // SAFETY: caller guarantees one complete possibly unaligned f64 destination.
    unsafe {
        output
            .cast::<u64>()
            .write_unaligned(value.to_bits().to_le())
    };
}

fn write_fallback(output: &mut [u8], op: &ConvertOp) {
    let n = op.dst_size as usize;
    output[..n].copy_from_slice(&op.fallback[..n]);
}

fn kernel_i16_i16(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    output[..2].copy_from_slice(&input[..2]);
    Ok(())
}
fn kernel_i32_i32(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    output[..4].copy_from_slice(&input[..4]);
    Ok(())
}
fn kernel_u16_u16(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    output[..2].copy_from_slice(&input[..2]);
    Ok(())
}
fn kernel_u32_u32(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    output[..4].copy_from_slice(&input[..4]);
    Ok(())
}
fn kernel_f32_f32(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    output[..4].copy_from_slice(&input[..4]);
    Ok(())
}
fn kernel_f64_f64(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    output[..8].copy_from_slice(&input[..8]);
    Ok(())
}

fn kernel_i16_i32(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_i32(output, i32::from(load_i16(input)));
    Ok(())
}
fn kernel_u16_u32(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_u32(output, u32::from(load_u16(input)));
    Ok(())
}
fn kernel_u16_i32(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_i32(output, i32::from(load_u16(input)));
    Ok(())
}
fn kernel_f32_f64(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_f64(output, f64::from(load_f32(input)));
    Ok(())
}

fn kernel_i16_u16(input: &[u8], output: &mut [u8], op: &ConvertOp) -> Result<()> {
    let v = load_i16(input);
    if v < 0 && op.write_fallback {
        write_fallback(output, op);
        return Ok(());
    }
    store_u16(output, v as u16);
    Ok(())
}

fn kernel_i16_u32(input: &[u8], output: &mut [u8], op: &ConvertOp) -> Result<()> {
    let v = load_i16(input);
    if v < 0 && op.write_fallback {
        write_fallback(output, op);
        return Ok(());
    }
    store_u32(output, v as u32);
    Ok(())
}

fn kernel_i32_u32(input: &[u8], output: &mut [u8], op: &ConvertOp) -> Result<()> {
    let v = load_i32(input);
    if v < 0 && op.write_fallback {
        write_fallback(output, op);
        return Ok(());
    }
    store_u32(output, v as u32);
    Ok(())
}

fn kernel_u16_i16(input: &[u8], output: &mut [u8], op: &ConvertOp) -> Result<()> {
    let v = load_u16(input);
    if v > i16::MAX as u16 && op.write_fallback {
        write_fallback(output, op);
        return Ok(());
    }
    store_i16(output, v as i16);
    Ok(())
}

fn kernel_u32_i32(input: &[u8], output: &mut [u8], op: &ConvertOp) -> Result<()> {
    let v = load_u32(input);
    if v > i32::MAX as u32 && op.write_fallback {
        write_fallback(output, op);
        return Ok(());
    }
    store_i32(output, v as i32);
    Ok(())
}

fn kernel_i16_f32(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_f32(output, f32::from(load_i16(input)));
    Ok(())
}
fn kernel_i16_f64(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_f64(output, f64::from(load_i16(input)));
    Ok(())
}
fn kernel_i32_f32(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_f32(output, load_i32(input) as f32);
    Ok(())
}
fn kernel_i32_f64(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_f64(output, f64::from(load_i32(input)));
    Ok(())
}
fn kernel_u16_f32(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_f32(output, f32::from(load_u16(input)));
    Ok(())
}
fn kernel_u16_f64(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_f64(output, f64::from(load_u16(input)));
    Ok(())
}
fn kernel_u32_f32(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_f32(output, load_u32(input) as f32);
    Ok(())
}
fn kernel_u32_f64(input: &[u8], output: &mut [u8], _op: &ConvertOp) -> Result<()> {
    store_f64(output, f64::from(load_u32(input)));
    Ok(())
}
