use std::arch::x86_64::*;

use sc_compress::DType as StorageDType;

use super::{scalar_slice, ConvertGather32Fn, ConvertOp, ConvertSliceFn, ValidateSliceFn};
use crate::dtype::OutputDType;
use crate::Result;

pub(super) fn dispatch_avx512(src: StorageDType, dst: OutputDType) -> Option<ConvertSliceFn> {
    use OutputDType as O;
    use StorageDType as S;
    Some(match (src, dst) {
        (S::I16, O::I32 | O::U32) => avx512_i16_i32,
        (S::I16, O::I64 | O::U64) => avx512_i16_i64,
        (S::U16, O::I32 | O::U32) => avx512_u16_i32,
        (S::U16, O::I64 | O::U64) => avx512_u16_i64,
        (S::I32, O::I64 | O::U64) => avx512_i32_i64,
        (S::U32, O::I64 | O::U64) => avx512_u32_i64,
        (S::I16, O::F32) => avx512_i16_f32,
        (S::U16, O::F32) => avx512_u16_f32,
        (S::I32, O::F32) => avx512_i32_f32,
        (S::U32, O::F32) => avx512_u32_f32,
        (S::I16, O::F64) => avx512_i16_f64,
        (S::U16, O::F64) => avx512_u16_f64,
        (S::I32, O::F64) => avx512_i32_f64,
        (S::U32, O::F64) => avx512_u32_f64,
        (S::F32, O::F64) => avx512_f32_f64,
        (S::I64, O::F64) if std::arch::is_x86_feature_detected!("avx512dq") => avx512_i64_f64,
        (S::U64, O::F64) if std::arch::is_x86_feature_detected!("avx512dq") => avx512_u64_f64,
        _ => return None,
    })
}

pub(super) fn dispatch_avx2(src: StorageDType, dst: OutputDType) -> Option<ConvertSliceFn> {
    use OutputDType as O;
    use StorageDType as S;
    Some(match (src, dst) {
        (S::I16, O::I32 | O::U32) => i16_i32,
        (S::I16, O::I64 | O::U64) => i16_i64,
        (S::U16, O::I32 | O::U32) => u16_i32,
        (S::U16, O::I64 | O::U64) => u16_i64,
        (S::I32, O::I64 | O::U64) => i32_i64,
        (S::U32, O::I64 | O::U64) => u32_i64,
        (S::I16, O::F32) => i16_f32,
        (S::U16, O::F32) => u16_f32,
        (S::I32, O::F32) => i32_f32,
        (S::U32, O::F32) => u32_f32,
        (S::I16, O::F64) => i16_f64,
        (S::U16, O::F64) => u16_f64,
        (S::I32, O::F64) => i32_f64,
        (S::U32, O::F64) => u32_f64,
        (S::F32, O::F64) => f32_f64,
        _ => return None,
    })
}

pub(super) fn dispatch_sse2(src: StorageDType, dst: OutputDType) -> Option<ConvertSliceFn> {
    use OutputDType as O;
    use StorageDType as S;
    Some(match (src, dst) {
        (S::I16, O::I32 | O::U32) => sse2_i16_i32,
        (S::U16, O::I32 | O::U32) => sse2_u16_i32,
        (S::I16, O::F32) => sse2_i16_f32,
        (S::U16, O::F32) => sse2_u16_f32,
        (S::I32, O::F32) => sse2_i32_f32,
        (S::I16, O::F64) => sse2_i16_f64,
        (S::U16, O::F64) => sse2_u16_f64,
        (S::I32, O::F64) => sse2_i32_f64,
        (S::F32, O::F64) => sse2_f32_f64,
        _ => return None,
    })
}

pub(super) fn dispatch_gather32_avx512(
    src: StorageDType,
    dst: OutputDType,
) -> Option<ConvertGather32Fn> {
    use OutputDType as O;
    use StorageDType as S;
    Some(match (src, dst) {
        (S::I32 | S::U32, O::I32 | O::U32) | (S::F32, O::F32) => gather32_copy_avx512,
        (S::I32, O::F32) => gather32_i32_f32_avx512,
        (S::U32, O::F32) => gather32_u32_f32_avx512,
        (S::I32, O::F64) if std::arch::is_x86_feature_detected!("avx2") => gather32_i32_f64_avx512,
        _ => return None,
    })
}

pub(super) fn dispatch_gather32_avx2(
    src: StorageDType,
    dst: OutputDType,
) -> Option<ConvertGather32Fn> {
    use OutputDType as O;
    use StorageDType as S;
    Some(match (src, dst) {
        (S::I32 | S::U32, O::I32 | O::U32) | (S::F32, O::F32) => gather32_copy_avx2,
        (S::I32, O::F32) => gather32_i32_f32_avx2,
        (S::U32, O::F32) => gather32_u32_f32_avx2,
        _ => return None,
    })
}

#[target_feature(enable = "avx512f")]
unsafe fn gather32_copy_avx512(
    input: *const u8,
    output: *mut u8,
    source_offsets: &[i32],
    target_byte: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 16;
    let vector_count = source_offsets.len() / LANES * LANES;
    // SAFETY: every signed compiler offset names one complete u32 and the
    // target is a disjoint contiguous run with the same element count.
    unsafe {
        let target = output.add(target_byte);
        for index in (0..vector_count).step_by(LANES) {
            let offsets = _mm512_loadu_si512(source_offsets.as_ptr().add(index).cast::<__m512i>());
            let values = _mm512_i32gather_epi32::<1>(offsets, input.cast::<i32>());
            _mm512_storeu_si512(target.add(index * 4).cast::<__m512i>(), values);
        }
        for (index, &source_byte) in source_offsets[vector_count..].iter().enumerate() {
            target
                .add((vector_count + index) * 4)
                .cast::<u32>()
                .write_unaligned(
                    input
                        .add(source_byte as usize)
                        .cast::<u32>()
                        .read_unaligned(),
                );
        }
    }
    Ok(())
}

#[target_feature(enable = "avx512f")]
unsafe fn gather32_i32_f32_avx512(
    input: *const u8,
    output: *mut u8,
    source_offsets: &[i32],
    target_byte: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 16;
    let vector_count = source_offsets.len() / LANES * LANES;
    // SAFETY: the compiler and caller establish all gather and contiguous-store
    // extents; signed conversion matches Rust's round-to-nearest cast.
    unsafe {
        let target = output.add(target_byte);
        for index in (0..vector_count).step_by(LANES) {
            let offsets = _mm512_loadu_si512(source_offsets.as_ptr().add(index).cast::<__m512i>());
            let values = _mm512_i32gather_epi32::<1>(offsets, input.cast::<i32>());
            _mm512_storeu_ps(
                target.add(index * 4).cast::<f32>(),
                _mm512_cvtepi32_ps(values),
            );
        }
        for (index, &source_byte) in source_offsets[vector_count..].iter().enumerate() {
            let value = input
                .add(source_byte as usize)
                .cast::<i32>()
                .read_unaligned();
            target
                .add((vector_count + index) * 4)
                .cast::<u32>()
                .write_unaligned((value as f32).to_bits());
        }
    }
    Ok(())
}

#[target_feature(enable = "avx512f")]
unsafe fn gather32_u32_f32_avx512(
    input: *const u8,
    output: *mut u8,
    source_offsets: &[i32],
    target_byte: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 16;
    let vector_count = source_offsets.len() / LANES * LANES;
    // SAFETY: the same compiler proof covers all gathers/stores and the native
    // unsigned conversion has Rust's f32 rounding semantics.
    unsafe {
        let target = output.add(target_byte);
        for index in (0..vector_count).step_by(LANES) {
            let offsets = _mm512_loadu_si512(source_offsets.as_ptr().add(index).cast::<__m512i>());
            let values = _mm512_i32gather_epi32::<1>(offsets, input.cast::<i32>());
            _mm512_storeu_ps(
                target.add(index * 4).cast::<f32>(),
                _mm512_cvtepu32_ps(values),
            );
        }
        for (index, &source_byte) in source_offsets[vector_count..].iter().enumerate() {
            let value = input
                .add(source_byte as usize)
                .cast::<u32>()
                .read_unaligned();
            target
                .add((vector_count + index) * 4)
                .cast::<u32>()
                .write_unaligned((value as f32).to_bits());
        }
    }
    Ok(())
}

#[target_feature(enable = "avx2,avx512f")]
unsafe fn gather32_i32_f64_avx512(
    input: *const u8,
    output: *mut u8,
    source_offsets: &[i32],
    target_byte: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 8;
    let vector_count = source_offsets.len() / LANES * LANES;
    // SAFETY: compiler-built offsets name complete i32 elements and the target
    // is a disjoint contiguous f64 run. Runtime dispatch checked AVX2/AVX-512F.
    unsafe {
        let target = output.add(target_byte);
        for index in (0..vector_count).step_by(LANES) {
            let offsets = _mm256_loadu_si256(source_offsets.as_ptr().add(index).cast::<__m256i>());
            let values = _mm256_i32gather_epi32::<1>(input.cast::<i32>(), offsets);
            _mm512_storeu_pd(
                target.add(index * 8).cast::<f64>(),
                _mm512_cvtepi32_pd(values),
            );
        }
        for (index, &source_byte) in source_offsets[vector_count..].iter().enumerate() {
            let value = input
                .add(source_byte as usize)
                .cast::<i32>()
                .read_unaligned();
            target
                .add((vector_count + index) * 8)
                .cast::<f64>()
                .write_unaligned(f64::from(value));
        }
    }
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn gather32_copy_avx2(
    input: *const u8,
    output: *mut u8,
    source_offsets: &[i32],
    target_byte: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 8;
    let vector_count = source_offsets.len() / LANES * LANES;
    // SAFETY: all gather indices and contiguous destination elements were
    // compiler-validated before this bound kernel was selected.
    unsafe {
        let target = output.add(target_byte);
        for index in (0..vector_count).step_by(LANES) {
            let offsets = _mm256_loadu_si256(source_offsets.as_ptr().add(index).cast::<__m256i>());
            let values = _mm256_i32gather_epi32::<1>(input.cast::<i32>(), offsets);
            _mm256_storeu_si256(target.add(index * 4).cast::<__m256i>(), values);
        }
        for (index, &source_byte) in source_offsets[vector_count..].iter().enumerate() {
            target
                .add((vector_count + index) * 4)
                .cast::<u32>()
                .write_unaligned(
                    input
                        .add(source_byte as usize)
                        .cast::<u32>()
                        .read_unaligned(),
                );
        }
    }
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn gather32_i32_f32_avx2(
    input: *const u8,
    output: *mut u8,
    source_offsets: &[i32],
    target_byte: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 8;
    let vector_count = source_offsets.len() / LANES * LANES;
    // SAFETY: all gathers and stores stay within the compiler-sealed mapping.
    unsafe {
        let target = output.add(target_byte);
        for index in (0..vector_count).step_by(LANES) {
            let offsets = _mm256_loadu_si256(source_offsets.as_ptr().add(index).cast::<__m256i>());
            let values = _mm256_i32gather_epi32::<1>(input.cast::<i32>(), offsets);
            _mm256_storeu_ps(
                target.add(index * 4).cast::<f32>(),
                _mm256_cvtepi32_ps(values),
            );
        }
        for (index, &source_byte) in source_offsets[vector_count..].iter().enumerate() {
            let value = input
                .add(source_byte as usize)
                .cast::<i32>()
                .read_unaligned();
            target
                .add((vector_count + index) * 4)
                .cast::<u32>()
                .write_unaligned((value as f32).to_bits());
        }
    }
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn gather32_u32_f32_avx2(
    input: *const u8,
    output: *mut u8,
    source_offsets: &[i32],
    target_byte: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 8;
    let vector_count = source_offsets.len() / LANES * LANES;
    let correction = _mm256_set1_ps(4_294_967_296.0);
    // SAFETY: all accesses use the compiler-sealed mapping. Signed conversion
    // plus a 2^32 correction implements the complete u32 domain.
    unsafe {
        let target = output.add(target_byte);
        for index in (0..vector_count).step_by(LANES) {
            let offsets = _mm256_loadu_si256(source_offsets.as_ptr().add(index).cast::<__m256i>());
            let values = _mm256_i32gather_epi32::<1>(input.cast::<i32>(), offsets);
            let signed = _mm256_cvtepi32_ps(values);
            let negative = _mm256_srai_epi32::<31>(values);
            let add = _mm256_and_ps(_mm256_castsi256_ps(negative), correction);
            _mm256_storeu_ps(
                target.add(index * 4).cast::<f32>(),
                _mm256_add_ps(signed, add),
            );
        }
        for (index, &source_byte) in source_offsets[vector_count..].iter().enumerate() {
            let value = input
                .add(source_byte as usize)
                .cast::<u32>()
                .read_unaligned();
            target
                .add((vector_count + index) * 4)
                .cast::<u32>()
                .write_unaligned((value as f32).to_bits());
        }
    }
    Ok(())
}

pub(super) fn dispatch_validate_avx2(
    src: StorageDType,
    dst: OutputDType,
) -> Option<ValidateSliceFn> {
    use OutputDType as O;
    use StorageDType as S;
    match (src, dst) {
        (S::I16, O::U16 | O::U32 | O::U64) | (S::U16, O::I16) => Some(validate_sign_i16),
        (S::I32, O::U32 | O::U64) | (S::U32, O::I32) => Some(validate_sign_i32),
        (S::I64, O::U64) | (S::U64, O::I64) => Some(validate_sign_i64),
        _ => None,
    }
}

pub(super) fn dispatch_validate_avx512(
    src: StorageDType,
    dst: OutputDType,
) -> Option<ValidateSliceFn> {
    use OutputDType as O;
    use StorageDType as S;
    match (src, dst) {
        (S::I16, O::U16 | O::U32 | O::U64) | (S::U16, O::I16) => Some(validate_sign_i16_avx512),
        (S::I32, O::U32 | O::U64) | (S::U32, O::I32) => Some(validate_sign_i32_avx512),
        (S::I64, O::U64) | (S::U64, O::I64) => Some(validate_sign_i64_avx512),
        _ => None,
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn validate_sign_i16_avx512(input: *const u8, count: usize, op: &ConvertOp) -> bool {
    const LANES: usize = 32;
    let vector_count = count / LANES * LANES;
    // SAFETY: runtime dispatch checked AVX-512F/BW and the entry wrapper proves
    // the full input extent. OR reduction preserves every source sign bit.
    unsafe {
        let mut sign_bits = _mm512_setzero_si512();
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm512_loadu_si512(input.add(index * 2).cast::<__m512i>());
            sign_bits = _mm512_or_si512(sign_bits, values);
        }
        if _mm512_cmp_epi16_mask::<_MM_CMPINT_LT>(sign_bits, _mm512_setzero_si512()) != 0 {
            return false;
        }
        scalar_validate_slice_tail(input, vector_count, count, op)
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn validate_sign_i32_avx512(input: *const u8, count: usize, op: &ConvertOp) -> bool {
    const LANES: usize = 16;
    let vector_count = count / LANES * LANES;
    // SAFETY: runtime dispatch checked AVX-512F and the entry wrapper proves
    // the full input extent. OR reduction preserves every source sign bit.
    unsafe {
        let mut sign_bits = _mm512_setzero_si512();
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm512_loadu_si512(input.add(index * 4).cast::<__m512i>());
            sign_bits = _mm512_or_si512(sign_bits, values);
        }
        if _mm512_cmp_epi32_mask::<_MM_CMPINT_LT>(sign_bits, _mm512_setzero_si512()) != 0 {
            return false;
        }
        scalar_validate_slice_tail(input, vector_count, count, op)
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn validate_sign_i64_avx512(input: *const u8, count: usize, op: &ConvertOp) -> bool {
    const LANES: usize = 8;
    let vector_count = count / LANES * LANES;
    // SAFETY: runtime dispatch checked AVX-512F and the entry wrapper proves
    // the full input extent. OR reduction preserves every source sign bit.
    unsafe {
        let mut sign_bits = _mm512_setzero_si512();
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm512_loadu_si512(input.add(index * 8).cast::<__m512i>());
            sign_bits = _mm512_or_si512(sign_bits, values);
        }
        if _mm512_cmp_epi64_mask::<_MM_CMPINT_LT>(sign_bits, _mm512_setzero_si512()) != 0 {
            return false;
        }
        scalar_validate_slice_tail(input, vector_count, count, op)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn validate_sign_i16(input: *const u8, count: usize, op: &ConvertOp) -> bool {
    let lanes = 16;
    let vector_count = count / lanes * lanes;
    let mut sign_bits = _mm256_setzero_si256();
    // SAFETY: runtime dispatch checked AVX2 and the entry wrapper proves the
    // complete input extent. Checked same-width signedness conversions are
    // valid exactly when bit 15 is clear for every source element.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm256_loadu_si256(input.add(index * 2).cast::<__m256i>());
            sign_bits = _mm256_or_si256(sign_bits, values);
        }
        if (_mm256_movemask_epi8(sign_bits) as u32 & 0xaaaa_aaaa) != 0 {
            return false;
        }
        scalar_validate_slice_tail(input, vector_count, count, op)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn validate_sign_i32(input: *const u8, count: usize, op: &ConvertOp) -> bool {
    let lanes = 8;
    let vector_count = count / lanes * lanes;
    let mut sign_bits = _mm256_setzero_si256();
    // SAFETY: runtime dispatch checked AVX2 and the entry wrapper proves the
    // complete input extent. Checked same-width signedness conversions are
    // valid exactly when bit 31 is clear for every source element.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm256_loadu_si256(input.add(index * 4).cast::<__m256i>());
            sign_bits = _mm256_or_si256(sign_bits, values);
        }
        if _mm256_movemask_ps(_mm256_castsi256_ps(sign_bits)) != 0 {
            return false;
        }
        scalar_validate_slice_tail(input, vector_count, count, op)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn validate_sign_i64(input: *const u8, count: usize, op: &ConvertOp) -> bool {
    const LANES: usize = 4;
    let vector_count = count / LANES * LANES;
    let mut sign_bits = _mm256_setzero_si256();
    // SAFETY: runtime dispatch checked AVX2 and the entry wrapper proves the
    // complete input extent. Checked 64-bit signedness conversions are valid
    // exactly when bit 63 is clear for every source element.
    unsafe {
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm256_loadu_si256(input.add(index * 8).cast::<__m256i>());
            sign_bits = _mm256_or_si256(sign_bits, values);
        }
        if _mm256_movemask_pd(_mm256_castsi256_pd(sign_bits)) != 0 {
            return false;
        }
        scalar_validate_slice_tail(input, vector_count, count, op)
    }
}

#[inline]
unsafe fn scalar_validate_slice_tail(
    input: *const u8,
    done: usize,
    count: usize,
    op: &ConvertOp,
) -> bool {
    match op.src_size {
        2 => {
            let mut bits = 0u16;
            for index in done..count {
                // SAFETY: the vector prefix consumed `done <= count` complete
                // elements and this reads one complete two-byte tail element.
                bits |=
                    u16::from_le(unsafe { input.add(index << 1).cast::<u16>().read_unaligned() });
            }
            bits & 0x8000 == 0
        }
        4 => {
            let mut bits = 0u32;
            for index in done..count {
                // SAFETY: the same extent proof covers each four-byte tail read.
                bits |=
                    u32::from_le(unsafe { input.add(index << 2).cast::<u32>().read_unaligned() });
            }
            bits & 0x8000_0000 == 0
        }
        8 => {
            let mut bits = 0u64;
            for index in done..count {
                // SAFETY: the same extent proof covers each eight-byte tail read.
                bits |=
                    u64::from_le(unsafe { input.add(index << 3).cast::<u64>().read_unaligned() });
            }
            bits & 0x8000_0000_0000_0000 == 0
        }
        _ => {
            debug_assert!(false, "checked conversion has unsupported source width");
            false
        }
    }
}

macro_rules! finish_scalar {
    ($input:expr, $output:expr, $done:expr, $count:expr, $op:expr) => {{
        let remaining = $count - $done;
        if remaining != 0 {
            let input_offset = $done << $op.src_shift;
            let output_offset = $done << $op.dst_shift;
            // SAFETY: the vector prefix consumed exactly `done` elements, so
            // these pointers and `remaining` describe the validated tail.
            unsafe {
                scalar_slice(
                    $input.add(input_offset),
                    $output.add(output_offset),
                    remaining,
                    $op,
                )?;
            }
        }
    }};
}

macro_rules! avx512_int_to_i64 {
    ($name:ident, $src_bytes:expr, $load:ident, $extend:ident) => {
        #[target_feature(enable = "avx2,avx512f,avx512bw")]
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            count: usize,
            op: &ConvertOp,
        ) -> Result<()> {
            const LANES: usize = 8;
            let vector_count = count / LANES * LANES;
            // SAFETY: runtime dispatch proved the required vector features and
            // the entry contract covers every eight-element load and store.
            unsafe {
                for index in (0..vector_count).step_by(LANES) {
                    let values = $load(input.add(index * $src_bytes).cast());
                    _mm512_storeu_si512(output.add(index * 8).cast::<__m512i>(), $extend(values));
                }
            }
            finish_scalar!(input, output, vector_count, count, op);
            Ok(())
        }
    };
}

avx512_int_to_i64!(avx512_i16_i64, 2, _mm_loadu_si128, _mm512_cvtepi16_epi64);
avx512_int_to_i64!(avx512_u16_i64, 2, _mm_loadu_si128, _mm512_cvtepu16_epi64);
avx512_int_to_i64!(avx512_i32_i64, 4, _mm256_loadu_si256, _mm512_cvtepi32_epi64);
avx512_int_to_i64!(avx512_u32_i64, 4, _mm256_loadu_si256, _mm512_cvtepu32_epi64);

macro_rules! avx512_i16_to_i32 {
    ($name:ident, $extend:ident) => {
        #[target_feature(enable = "avx2,avx512f,avx512bw")]
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            count: usize,
            _op: &ConvertOp,
        ) -> Result<()> {
            const LANES: usize = 16;
            let vector_count = count / LANES * LANES;
            // SAFETY: runtime dispatch proved AVX-512F/BW and AVX2 support;
            // the entry contract covers every 16-element unaligned load/store.
            unsafe {
                for index in (0..vector_count).step_by(LANES) {
                    let values = _mm256_loadu_si256(input.add(index * 2).cast::<__m256i>());
                    let wide = $extend(values);
                    _mm512_storeu_si512(output.add(index * 4).cast::<__m512i>(), wide);
                }
                let remaining = count - vector_count;
                if remaining != 0 {
                    let input_mask = (1u32 << remaining) - 1;
                    let output_mask = input_mask as __mmask16;
                    let packed = _mm512_maskz_loadu_epi16(
                        input_mask,
                        input.add(vector_count * 2).cast::<i16>(),
                    );
                    let values = _mm512_castsi512_si256(packed);
                    _mm512_mask_storeu_epi32(
                        output.add(vector_count * 4).cast::<i32>(),
                        output_mask,
                        $extend(values),
                    );
                }
            }
            Ok(())
        }
    };
}

macro_rules! avx512_i16_to_f32 {
    ($name:ident, $extend:ident) => {
        #[target_feature(enable = "avx2,avx512f,avx512bw")]
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            count: usize,
            _op: &ConvertOp,
        ) -> Result<()> {
            const LANES: usize = 16;
            let vector_count = count / LANES * LANES;
            // SAFETY: dispatch and the entry extent proof cover each 16-lane
            // integer expansion, conversion, and unaligned f32 store.
            unsafe {
                for index in (0..vector_count).step_by(LANES) {
                    let values = _mm256_loadu_si256(input.add(index * 2).cast::<__m256i>());
                    let wide = $extend(values);
                    _mm512_storeu_ps(
                        output.add(index * 4).cast::<f32>(),
                        _mm512_cvtepi32_ps(wide),
                    );
                }
                let remaining = count - vector_count;
                if remaining != 0 {
                    let input_mask = (1u32 << remaining) - 1;
                    let output_mask = input_mask as __mmask16;
                    let packed = _mm512_maskz_loadu_epi16(
                        input_mask,
                        input.add(vector_count * 2).cast::<i16>(),
                    );
                    let values = _mm512_castsi512_si256(packed);
                    let wide = $extend(values);
                    _mm512_mask_storeu_ps(
                        output.add(vector_count * 4).cast::<f32>(),
                        output_mask,
                        _mm512_cvtepi32_ps(wide),
                    );
                }
            }
            Ok(())
        }
    };
}

avx512_i16_to_i32!(avx512_i16_i32, _mm512_cvtepi16_epi32);
avx512_i16_to_i32!(avx512_u16_i32, _mm512_cvtepu16_epi32);
avx512_i16_to_f32!(avx512_i16_f32, _mm512_cvtepi16_epi32);
avx512_i16_to_f32!(avx512_u16_f32, _mm512_cvtepu16_epi32);

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn avx512_i32_f32(
    input: *const u8,
    output: *mut u8,
    count: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 16;
    let vector_count = count / LANES * LANES;
    // SAFETY: dispatch proved AVX-512F/BW and every vector access lies within
    // the caller-validated 16-element source and destination windows.
    unsafe {
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm512_loadu_si512(input.add(index * 4).cast::<__m512i>());
            _mm512_storeu_ps(
                output.add(index * 4).cast::<f32>(),
                _mm512_cvtepi32_ps(values),
            );
        }
        let remaining = count - vector_count;
        if remaining != 0 {
            let mask = ((1u32 << remaining) - 1) as __mmask16;
            let values = _mm512_maskz_loadu_epi32(mask, input.add(vector_count * 4).cast::<i32>());
            _mm512_mask_storeu_ps(
                output.add(vector_count * 4).cast::<f32>(),
                mask,
                _mm512_cvtepi32_ps(values),
            );
        }
    }
    Ok(())
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn avx512_u32_f32(
    input: *const u8,
    output: *mut u8,
    count: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 16;
    let vector_count = count / LANES * LANES;
    // SAFETY: dispatch proved AVX-512F/BW; native unsigned conversion has the
    // same IEEE-754 round-to-nearest behavior as the scalar Rust cast.
    unsafe {
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm512_loadu_si512(input.add(index * 4).cast::<__m512i>());
            _mm512_storeu_ps(
                output.add(index * 4).cast::<f32>(),
                _mm512_cvtepu32_ps(values),
            );
        }
        let remaining = count - vector_count;
        if remaining != 0 {
            let mask = ((1u32 << remaining) - 1) as __mmask16;
            let values = _mm512_maskz_loadu_epi32(mask, input.add(vector_count * 4).cast::<i32>());
            _mm512_mask_storeu_ps(
                output.add(vector_count * 4).cast::<f32>(),
                mask,
                _mm512_cvtepu32_ps(values),
            );
        }
    }
    Ok(())
}

macro_rules! avx512_i16_to_f64 {
    ($name:ident, $extend:ident) => {
        #[target_feature(enable = "avx2,avx512f,avx512bw")]
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            count: usize,
            _op: &ConvertOp,
        ) -> Result<()> {
            const LANES: usize = 8;
            let vector_count = count / LANES * LANES;
            // SAFETY: each iteration consumes exactly 8 validated i16/u16
            // elements and produces 8 f64 elements under the dispatched ISA.
            unsafe {
                for index in (0..vector_count).step_by(LANES) {
                    let values = _mm_loadu_si128(input.add(index * 2).cast::<__m128i>());
                    let wide = $extend(values);
                    _mm512_storeu_pd(
                        output.add(index * 8).cast::<f64>(),
                        _mm512_cvtepi32_pd(wide),
                    );
                }
                let remaining = count - vector_count;
                if remaining != 0 {
                    let input_mask = (1u32 << remaining) - 1;
                    let output_mask = input_mask as __mmask8;
                    let packed = _mm512_maskz_loadu_epi16(
                        input_mask,
                        input.add(vector_count * 2).cast::<i16>(),
                    );
                    let values = _mm256_castsi256_si128(_mm512_castsi512_si256(packed));
                    _mm512_mask_storeu_pd(
                        output.add(vector_count * 8).cast::<f64>(),
                        output_mask,
                        _mm512_cvtepi32_pd($extend(values)),
                    );
                }
            }
            Ok(())
        }
    };
}

avx512_i16_to_f64!(avx512_i16_f64, _mm256_cvtepi16_epi32);
avx512_i16_to_f64!(avx512_u16_f64, _mm256_cvtepu16_epi32);

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn avx512_i32_f64(
    input: *const u8,
    output: *mut u8,
    count: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 8;
    let vector_count = count / LANES * LANES;
    // SAFETY: every 8-lane load, conversion, and store is contained in the
    // validated raw buffers and dispatch proved AVX-512F/BW.
    unsafe {
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm256_loadu_si256(input.add(index * 4).cast::<__m256i>());
            _mm512_storeu_pd(
                output.add(index * 8).cast::<f64>(),
                _mm512_cvtepi32_pd(values),
            );
        }
        let remaining = count - vector_count;
        if remaining != 0 {
            let input_mask = ((1u32 << remaining) - 1) as __mmask16;
            let output_mask = input_mask as __mmask8;
            let packed =
                _mm512_maskz_loadu_epi32(input_mask, input.add(vector_count * 4).cast::<i32>());
            _mm512_mask_storeu_pd(
                output.add(vector_count * 8).cast::<f64>(),
                output_mask,
                _mm512_cvtepi32_pd(_mm512_castsi512_si256(packed)),
            );
        }
    }
    Ok(())
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn avx512_u32_f64(
    input: *const u8,
    output: *mut u8,
    count: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 8;
    let vector_count = count / LANES * LANES;
    // SAFETY: native unsigned conversion is exact for the full u32 domain and
    // all accesses lie within the caller-proved extents.
    unsafe {
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm256_loadu_si256(input.add(index * 4).cast::<__m256i>());
            _mm512_storeu_pd(
                output.add(index * 8).cast::<f64>(),
                _mm512_cvtepu32_pd(values),
            );
        }
        let remaining = count - vector_count;
        if remaining != 0 {
            let input_mask = ((1u32 << remaining) - 1) as __mmask16;
            let output_mask = input_mask as __mmask8;
            let packed =
                _mm512_maskz_loadu_epi32(input_mask, input.add(vector_count * 4).cast::<i32>());
            _mm512_mask_storeu_pd(
                output.add(vector_count * 8).cast::<f64>(),
                output_mask,
                _mm512_cvtepu32_pd(_mm512_castsi512_si256(packed)),
            );
        }
    }
    Ok(())
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn avx512_f32_f64(
    input: *const u8,
    output: *mut u8,
    count: usize,
    _op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 8;
    let vector_count = count / LANES * LANES;
    // SAFETY: all 8-lane loads/stores lie within the validated buffers and f32
    // values are represented exactly by the resulting f64 values.
    unsafe {
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm256_loadu_ps(input.add(index * 4).cast::<f32>());
            _mm512_storeu_pd(output.add(index * 8).cast::<f64>(), _mm512_cvtps_pd(values));
        }
        let remaining = count - vector_count;
        if remaining != 0 {
            let input_mask = ((1u32 << remaining) - 1) as __mmask16;
            let output_mask = input_mask as __mmask8;
            let packed =
                _mm512_maskz_loadu_epi32(input_mask, input.add(vector_count * 4).cast::<i32>());
            let values = _mm256_castsi256_ps(_mm512_castsi512_si256(packed));
            _mm512_mask_storeu_pd(
                output.add(vector_count * 8).cast::<f64>(),
                output_mask,
                _mm512_cvtps_pd(values),
            );
        }
    }
    Ok(())
}

macro_rules! avx512_i64_to_f64 {
    ($name:ident, $convert:ident) => {
        #[target_feature(enable = "avx512f,avx512dq")]
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            count: usize,
            op: &ConvertOp,
        ) -> Result<()> {
            const LANES: usize = 8;
            let vector_count = count / LANES * LANES;
            // SAFETY: runtime dispatch checked AVX-512F/DQ and every vector
            // load/store lies within the wrapper-validated eight-byte elements.
            unsafe {
                for index in (0..vector_count).step_by(LANES) {
                    let values = _mm512_loadu_si512(input.add(index * 8).cast::<__m512i>());
                    _mm512_storeu_pd(output.add(index * 8).cast::<f64>(), $convert(values));
                }
            }
            finish_scalar!(input, output, vector_count, count, op);
            Ok(())
        }
    };
}

avx512_i64_to_f64!(avx512_i64_f64, _mm512_cvtepi64_pd);
avx512_i64_to_f64!(avx512_u64_f64, _mm512_cvtepu64_pd);

macro_rules! avx2_int_to_i64 {
    ($name:ident, $src_bytes:expr, $load:ident, $extend:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            count: usize,
            op: &ConvertOp,
        ) -> Result<()> {
            const LANES: usize = 4;
            let vector_count = count / LANES * LANES;
            // SAFETY: runtime dispatch checked AVX2 and the wrapper proves each
            // four-element source window and four-element i64/u64 destination.
            unsafe {
                for index in (0..vector_count).step_by(LANES) {
                    let values = $load(input.add(index * $src_bytes).cast());
                    _mm256_storeu_si256(output.add(index * 8).cast::<__m256i>(), $extend(values));
                }
            }
            finish_scalar!(input, output, vector_count, count, op);
            Ok(())
        }
    };
}

avx2_int_to_i64!(i16_i64, 2, _mm_loadl_epi64, _mm256_cvtepi16_epi64);
avx2_int_to_i64!(u16_i64, 2, _mm_loadl_epi64, _mm256_cvtepu16_epi64);
avx2_int_to_i64!(i32_i64, 4, _mm_loadu_si128, _mm256_cvtepi32_epi64);
avx2_int_to_i64!(u32_i64, 4, _mm_loadu_si128, _mm256_cvtepu32_epi64);

#[target_feature(enable = "avx2")]
unsafe fn i16_i32(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 8;
    let vector_count = count / lanes * lanes;
    // SAFETY: dispatch checked AVX2. The entry wrapper proves both byte ranges;
    // every load reads 8 i16 values and every store writes 8 i32/u32 values.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm_loadu_si128(input.add(index * 2).cast::<__m128i>());
            let wide = _mm256_cvtepi16_epi32(values);
            _mm256_storeu_si256(output.add(index * 4).cast::<__m256i>(), wide);
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn u16_i32(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 8;
    let vector_count = count / lanes * lanes;
    // SAFETY: dispatch checked AVX2 and the wrapper proves the complete source
    // and destination extents for these 8-lane unaligned operations.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm_loadu_si128(input.add(index * 2).cast::<__m128i>());
            let wide = _mm256_cvtepu16_epi32(values);
            _mm256_storeu_si256(output.add(index * 4).cast::<__m256i>(), wide);
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn i16_f32(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 8;
    let vector_count = count / lanes * lanes;
    // SAFETY: dispatch checked AVX2; each iteration stays inside the validated
    // 8*i16 input and 8*f32 output regions.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm_loadu_si128(input.add(index * 2).cast::<__m128i>());
            let wide = _mm256_cvtepi16_epi32(values);
            _mm256_storeu_ps(
                output.add(index * 4).cast::<f32>(),
                _mm256_cvtepi32_ps(wide),
            );
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn u16_f32(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 8;
    let vector_count = count / lanes * lanes;
    // SAFETY: dispatch checked AVX2; each iteration stays inside the validated
    // 8*u16 input and 8*f32 output regions.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm_loadu_si128(input.add(index * 2).cast::<__m128i>());
            let wide = _mm256_cvtepu16_epi32(values);
            _mm256_storeu_ps(
                output.add(index * 4).cast::<f32>(),
                _mm256_cvtepi32_ps(wide),
            );
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn i32_f32(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 8;
    let vector_count = count / lanes * lanes;
    // SAFETY: dispatch checked AVX2 and all 8-lane unaligned accesses fit the
    // validated input/output lengths.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm256_loadu_si256(input.add(index * 4).cast::<__m256i>());
            _mm256_storeu_ps(
                output.add(index * 4).cast::<f32>(),
                _mm256_cvtepi32_ps(values),
            );
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn u32_f32(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 8;
    let vector_count = count / lanes * lanes;
    let correction = _mm256_set1_ps(4_294_967_296.0);
    // SAFETY: dispatch checked AVX2 and all 8-lane unaligned accesses fit the
    // validated ranges. Signed conversion plus a 2^32 correction implements
    // the full u32 domain with IEEE-754 round-to-nearest semantics.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm256_loadu_si256(input.add(index * 4).cast::<__m256i>());
            let signed = _mm256_cvtepi32_ps(values);
            let negative = _mm256_srai_epi32::<31>(values);
            let add = _mm256_and_ps(_mm256_castsi256_ps(negative), correction);
            _mm256_storeu_ps(
                output.add(index * 4).cast::<f32>(),
                _mm256_add_ps(signed, add),
            );
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn i16_f64(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 4;
    let vector_count = count / lanes * lanes;
    // SAFETY: each 64-bit load covers 4 i16 values and each AVX store covers
    // their 4 f64 outputs within the wrapper-validated regions.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm_loadl_epi64(input.add(index * 2).cast::<__m128i>());
            let wide = _mm_cvtepi16_epi32(values);
            _mm256_storeu_pd(
                output.add(index * 8).cast::<f64>(),
                _mm256_cvtepi32_pd(wide),
            );
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn u16_f64(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 4;
    let vector_count = count / lanes * lanes;
    // SAFETY: each 64-bit load covers 4 u16 values and each AVX store covers
    // their 4 f64 outputs within the wrapper-validated regions.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm_loadl_epi64(input.add(index * 2).cast::<__m128i>());
            let wide = _mm_cvtepu16_epi32(values);
            _mm256_storeu_pd(
                output.add(index * 8).cast::<f64>(),
                _mm256_cvtepi32_pd(wide),
            );
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn i32_f64(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 4;
    let vector_count = count / lanes * lanes;
    // SAFETY: every unaligned 128-bit load and 256-bit store stays inside the
    // validated 4-element source and destination windows.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm_loadu_si128(input.add(index * 4).cast::<__m128i>());
            _mm256_storeu_pd(
                output.add(index * 8).cast::<f64>(),
                _mm256_cvtepi32_pd(values),
            );
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn u32_f64(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 4;
    let vector_count = count / lanes * lanes;
    let low_mask = _mm_set1_epi32(i32::MAX);
    let high_scale = _mm256_set1_pd(2_147_483_648.0);
    // SAFETY: every unaligned load/store is within the validated regions.
    // Splitting off bit 31 keeps both integer-to-f64 conversions exact.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm_loadu_si128(input.add(index * 4).cast::<__m128i>());
            let low = _mm_and_si128(values, low_mask);
            let high = _mm_srli_epi32::<31>(values);
            let base = _mm256_cvtepi32_pd(low);
            let add = _mm256_mul_pd(_mm256_cvtepi32_pd(high), high_scale);
            _mm256_storeu_pd(
                output.add(index * 8).cast::<f64>(),
                _mm256_add_pd(base, add),
            );
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn f32_f64(input: *const u8, output: *mut u8, count: usize, op: &ConvertOp) -> Result<()> {
    let lanes = 4;
    let vector_count = count / lanes * lanes;
    // SAFETY: every unaligned 4*f32 load and 4*f64 store stays inside the
    // wrapper-validated buffers; AVX preserves all f32 values exactly in f64.
    unsafe {
        for index in (0..vector_count).step_by(lanes) {
            let values = _mm_loadu_ps(input.add(index * 4).cast::<f32>());
            _mm256_storeu_pd(output.add(index * 8).cast::<f64>(), _mm256_cvtps_pd(values));
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

macro_rules! sse2_i16_to_i32 {
    ($name:ident, $signed:expr) => {
        #[target_feature(enable = "sse2")]
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            count: usize,
            op: &ConvertOp,
        ) -> Result<()> {
            const LANES: usize = 8;
            let vector_count = count / LANES * LANES;
            let zero = _mm_setzero_si128();
            // SAFETY: x86_64 guarantees SSE2, and each two-store expansion is
            // contained in the caller-validated 8-element windows.
            unsafe {
                for index in (0..vector_count).step_by(LANES) {
                    let values = _mm_loadu_si128(input.add(index * 2).cast::<__m128i>());
                    let high = if $signed {
                        _mm_cmpgt_epi16(zero, values)
                    } else {
                        zero
                    };
                    let low_wide = _mm_unpacklo_epi16(values, high);
                    let high_wide = _mm_unpackhi_epi16(values, high);
                    _mm_storeu_si128(output.add(index * 4).cast::<__m128i>(), low_wide);
                    _mm_storeu_si128(output.add((index + 4) * 4).cast::<__m128i>(), high_wide);
                }
            }
            finish_scalar!(input, output, vector_count, count, op);
            Ok(())
        }
    };
}

macro_rules! sse2_i16_to_f32 {
    ($name:ident, $signed:expr) => {
        #[target_feature(enable = "sse2")]
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            count: usize,
            op: &ConvertOp,
        ) -> Result<()> {
            const LANES: usize = 8;
            let vector_count = count / LANES * LANES;
            let zero = _mm_setzero_si128();
            // SAFETY: SSE2 is baseline on x86_64; the validated buffers contain
            // every 8-lane load and both 4-lane f32 stores.
            unsafe {
                for index in (0..vector_count).step_by(LANES) {
                    let values = _mm_loadu_si128(input.add(index * 2).cast::<__m128i>());
                    let high = if $signed {
                        _mm_cmpgt_epi16(zero, values)
                    } else {
                        zero
                    };
                    let low_wide = _mm_unpacklo_epi16(values, high);
                    let high_wide = _mm_unpackhi_epi16(values, high);
                    _mm_storeu_ps(
                        output.add(index * 4).cast::<f32>(),
                        _mm_cvtepi32_ps(low_wide),
                    );
                    _mm_storeu_ps(
                        output.add((index + 4) * 4).cast::<f32>(),
                        _mm_cvtepi32_ps(high_wide),
                    );
                }
            }
            finish_scalar!(input, output, vector_count, count, op);
            Ok(())
        }
    };
}

sse2_i16_to_i32!(sse2_i16_i32, true);
sse2_i16_to_i32!(sse2_u16_i32, false);
sse2_i16_to_f32!(sse2_i16_f32, true);
sse2_i16_to_f32!(sse2_u16_f32, false);

#[target_feature(enable = "sse2")]
unsafe fn sse2_i32_f32(
    input: *const u8,
    output: *mut u8,
    count: usize,
    op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 4;
    let vector_count = count / LANES * LANES;
    // SAFETY: x86_64 guarantees SSE2 and all 4-lane operations stay within the
    // raw extents established by the conversion entry point.
    unsafe {
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm_loadu_si128(input.add(index * 4).cast::<__m128i>());
            _mm_storeu_ps(output.add(index * 4).cast::<f32>(), _mm_cvtepi32_ps(values));
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

macro_rules! sse2_i16_to_f64 {
    ($name:ident, $signed:expr) => {
        #[target_feature(enable = "sse2")]
        unsafe fn $name(
            input: *const u8,
            output: *mut u8,
            count: usize,
            op: &ConvertOp,
        ) -> Result<()> {
            const LANES: usize = 4;
            let vector_count = count / LANES * LANES;
            let zero = _mm_setzero_si128();
            // SAFETY: one 64-bit load provides four complete source elements;
            // the two f64 stores cover their validated destination extent.
            unsafe {
                for index in (0..vector_count).step_by(LANES) {
                    let values = _mm_loadl_epi64(input.add(index * 2).cast::<__m128i>());
                    let high = if $signed {
                        _mm_cmpgt_epi16(zero, values)
                    } else {
                        zero
                    };
                    let wide = _mm_unpacklo_epi16(values, high);
                    _mm_storeu_pd(output.add(index * 8).cast::<f64>(), _mm_cvtepi32_pd(wide));
                    let upper = _mm_srli_si128::<8>(wide);
                    _mm_storeu_pd(
                        output.add((index + 2) * 8).cast::<f64>(),
                        _mm_cvtepi32_pd(upper),
                    );
                }
            }
            finish_scalar!(input, output, vector_count, count, op);
            Ok(())
        }
    };
}

sse2_i16_to_f64!(sse2_i16_f64, true);
sse2_i16_to_f64!(sse2_u16_f64, false);

#[target_feature(enable = "sse2")]
unsafe fn sse2_i32_f64(
    input: *const u8,
    output: *mut u8,
    count: usize,
    op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 2;
    let vector_count = count / LANES * LANES;
    // SAFETY: each 64-bit load and 128-bit store stays within the validated
    // two-element source/destination windows.
    unsafe {
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm_loadl_epi64(input.add(index * 4).cast::<__m128i>());
            _mm_storeu_pd(output.add(index * 8).cast::<f64>(), _mm_cvtepi32_pd(values));
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[target_feature(enable = "sse2")]
unsafe fn sse2_f32_f64(
    input: *const u8,
    output: *mut u8,
    count: usize,
    op: &ConvertOp,
) -> Result<()> {
    const LANES: usize = 2;
    let vector_count = count / LANES * LANES;
    // SAFETY: each 2-element load/conversion/store is covered by the validated
    // buffers and SSE2 represents every f32 exactly as f64.
    unsafe {
        for index in (0..vector_count).step_by(LANES) {
            let values = _mm_castsi128_ps(_mm_loadl_epi64(input.add(index * 4).cast::<__m128i>()));
            _mm_storeu_pd(output.add(index * 8).cast::<f64>(), _mm_cvtps_pd(values));
        }
    }
    finish_scalar!(input, output, vector_count, count, op);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fill, OutputSpec};

    fn checked_u32_i32() -> super::super::ConvertOp {
        super::super::ConvertOp::resolve(
            StorageDType::U32,
            &OutputSpec::new(1, OutputDType::I32, Fill::I32(0)).unwrap(),
        )
        .unwrap()
    }

    fn checked_u16_i16() -> super::super::ConvertOp {
        super::super::ConvertOp::resolve(
            StorageDType::U16,
            &OutputSpec::new(1, OutputDType::I16, Fill::I16(0)).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn forced_avx512_sign_validation_matches_avx2_for_vectors_and_tails() {
        if !std::arch::is_x86_feature_detected!("avx512f")
            || !std::arch::is_x86_feature_detected!("avx512bw")
            || !std::arch::is_x86_feature_detected!("avx2")
        {
            return;
        }
        let op = checked_u32_i32();
        let mut avx2 = op;
        avx2.validate_slice = dispatch_validate_avx2(op.src, op.dst).unwrap();
        let mut avx512 = op;
        avx512.validate_slice = dispatch_validate_avx512(op.src, op.dst).unwrap();

        let valid_values = [0u32, 127, 128, 32_767, 32_768, 16_777_217, i32::MAX as u32];
        let valid = (0..67)
            .flat_map(|index| valid_values[index % valid_values.len()].to_le_bytes())
            .collect::<Vec<_>>();
        assert!(avx2.validate_slice(&valid).is_ok());
        assert!(avx512.validate_slice(&valid).is_ok());
        for index in [15, 16, 31, 32, 66] {
            let mut invalid = valid.clone();
            invalid[index * 4..(index + 1) * 4]
                .copy_from_slice(&(i32::MAX as u32 + 1).to_le_bytes());
            assert!(avx2.validate_slice(&invalid).is_err());
            assert!(avx512.validate_slice(&invalid).is_err());
        }

        let op = checked_u16_i16();
        let mut avx2 = op;
        avx2.validate_slice = dispatch_validate_avx2(op.src, op.dst).unwrap();
        let mut avx512 = op;
        avx512.validate_slice = dispatch_validate_avx512(op.src, op.dst).unwrap();
        let valid_values = [0u16, 127, 128, 255, 256, i16::MAX as u16];
        let valid = (0..67)
            .flat_map(|index| valid_values[index % valid_values.len()].to_le_bytes())
            .collect::<Vec<_>>();
        assert!(avx2.validate_slice(&valid).is_ok());
        assert!(avx512.validate_slice(&valid).is_ok());
        for index in [15, 16, 31, 32, 66] {
            let mut invalid = valid.clone();
            invalid[index * 2..(index + 1) * 2]
                .copy_from_slice(&(i16::MAX as u16 + 1).to_le_bytes());
            assert!(avx2.validate_slice(&invalid).is_err());
            assert!(avx512.validate_slice(&invalid).is_err());
        }
    }

    #[test]
    #[ignore = "manual release-mode sign-validation benchmark"]
    fn benchmark_avx512_sign_validation_against_avx2() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        if !std::arch::is_x86_feature_detected!("avx512f")
            || !std::arch::is_x86_feature_detected!("avx512bw")
            || !std::arch::is_x86_feature_detected!("avx2")
        {
            return;
        }
        fn best_of(mut run: impl FnMut(), rounds: usize) -> Duration {
            (0..rounds)
                .map(|_| {
                    let started = Instant::now();
                    run();
                    started.elapsed()
                })
                .min()
                .unwrap()
        }

        const COUNT: usize = 32 * 1_024;
        const ITERATIONS: usize = 5_000;
        let input = (0..COUNT as u32)
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let op = checked_u32_i32();
        let mut avx2 = op;
        avx2.validate_slice = dispatch_validate_avx2(op.src, op.dst).unwrap();
        let mut avx512 = op;
        avx512.validate_slice = dispatch_validate_avx512(op.src, op.dst).unwrap();
        let measure = |op: &super::super::ConvertOp| {
            best_of(
                || {
                    for _ in 0..ITERATIONS {
                        op.validate_slice(black_box(&input)).unwrap();
                        black_box(());
                    }
                },
                5,
            )
        };
        let avx512_time = measure(&avx512);
        let avx2_time = measure(&avx2);
        eprintln!(
            "u32 sign validation: AVX2 {:.2} GiB/s, AVX-512 {:.2} GiB/s, {:.2}x",
            COUNT as f64 * 4.0 * ITERATIONS as f64
                / avx2_time.as_secs_f64()
                / (1024.0 * 1024.0 * 1024.0),
            COUNT as f64 * 4.0 * ITERATIONS as f64
                / avx512_time.as_secs_f64()
                / (1024.0 * 1024.0 * 1024.0),
            avx2_time.as_secs_f64() / avx512_time.as_secs_f64(),
        );
    }
}
