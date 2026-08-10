//! Bit shuffle and unshuffle filters for Blosc-compatible payloads.
//!
//! After byte shuffle, bitshuffle additionally transposes the bits within
//! each byte, so each bit-plane of the data becomes contiguous. This works
//! best on low-entropy binary data (e.g. float arrays with mostly zero
//! mantissa bits). The implementation follows the classic three-stage
//! transpose: byte transpose within elements, bit transpose within bytes,
//! then transpose the resulting bit rows.
//! # Unsafe kernel boundary
//!
//! Every unsafe kernel operates on a complete-element prefix of
//! `size * elem_size` bytes. Its input, output, and temporary pointers must each
//! cover that prefix and must not overlap. SIMD kernels additionally require
//! the target feature on their function attribute. The checked entry points
//! validate these invariants before dispatch.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::ptr;

/// Transpose an 8x8 bit array packed into a single u64 (`x`); `t` is scratch.
#[inline(always)]
fn trans_bit_8x8(x: &mut u64, t: &mut u64) {
    *t = (*x ^ (*x >> 7)) & 0x00AA00AA00AA00AA;
    *x = *x ^ *t ^ (*t << 7);
    *t = (*x ^ (*x >> 14)) & 0x0000CCCC0000CCCC;
    *x = *x ^ *t ^ (*t << 14);
    *t = (*x ^ (*x >> 28)) & 0x00000000F0F0F0F0;
    *x = *x ^ *t ^ (*t << 28);
}

/// Transpose a matrix of `elem_bytes`-wide elements, `lda` rows by `ldb` cols.
unsafe fn trans_elem(
    in_b: *const u8,
    out_b: *mut u8,
    lda: usize,
    ldb: usize,
    elem_bytes: usize,
) -> i64 {
    for ii in 0..lda {
        for jj in 0..ldb {
            let src = (ii * ldb + jj) * elem_bytes;
            let dst = (jj * lda + ii) * elem_bytes;
            ptr::copy_nonoverlapping(in_b.add(src), out_b.add(dst), elem_bytes);
        }
    }
    (lda * ldb * elem_bytes) as i64
}

/// Transpose bytes within elements, starting partway through the input.
unsafe fn trans_byte_elem_remainder(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
    start: usize,
) -> i64 {
    debug_assert!(start.is_multiple_of(8));
    if size > start {
        let mut ii = start;
        while ii + 7 < size {
            for jj in 0..elem_size {
                for kk in 0..8 {
                    *out_b.add(jj * size + ii + kk) =
                        *in_b.add(ii * elem_size + kk * elem_size + jj);
                }
            }
            ii += 8;
        }
        let mut ii = size - size % 8;
        while ii < size {
            for jj in 0..elem_size {
                *out_b.add(jj * size + ii) = *in_b.add(ii * elem_size + jj);
            }
            ii += 1;
        }
    }
    (size * elem_size) as i64
}

/// Transpose bytes within elements (scalar).
unsafe fn trans_byte_elem_scal(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    trans_byte_elem_remainder(in_b, out_b, size, elem_size, 0)
}

/// Transpose bits within bytes, starting at byte `start_byte`.
#[allow(clippy::too_many_arguments)]
unsafe fn trans_bit_byte_remainder(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
    start_byte: usize,
) -> i64 {
    debug_assert!((size * elem_size).is_multiple_of(8));
    debug_assert!(start_byte.is_multiple_of(8));

    let nbyte = elem_size * size;
    let nbyte_bitrow = nbyte / 8;

    let mut t: u64 = 0;
    for ii in (start_byte / 8)..nbyte_bitrow {
        let mut x = ptr::read_unaligned(in_b.add(ii * 8) as *const u64);
        trans_bit_8x8(&mut x, &mut t);
        for kk in 0..8 {
            *out_b.add(kk * nbyte_bitrow + ii) = x as u8;
            x >>= 8;
        }
    }
    (size * elem_size) as i64
}

/// Transpose bits within bytes (scalar).
#[cfg(not(target_arch = "x86_64"))]
unsafe fn trans_bit_byte_scal(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    trans_bit_byte_remainder(in_b, out_b, size, elem_size, 0)
}

/// Transpose rows of shuffled bits (size/8 bytes) within groups of 8.
unsafe fn trans_bitrow_eight(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    debug_assert!(size.is_multiple_of(8));
    let nbyte_bitrow = size / 8;
    trans_elem(in_b, out_b, 8, elem_size, nbyte_bitrow)
}

/// Scalar bitshuffle: byte transpose, bit transpose, bit-row transpose.
#[cfg(not(target_arch = "x86_64"))]
unsafe fn trans_bit_elem_scal(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
    tmp_buf: *mut u8,
) -> i64 {
    debug_assert!(size.is_multiple_of(8));
    trans_byte_elem_scal(in_b, out_b, size, elem_size);
    trans_bit_byte_scal(out_b, tmp_buf, size, elem_size);
    trans_bitrow_eight(tmp_buf, out_b, size, elem_size)
}

/// For data organized into a row per bit (8*elem_size rows), transpose the
/// bytes (scalar).
#[cfg(not(target_arch = "x86_64"))]
unsafe fn trans_byte_bitrow_scal(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    debug_assert!(size.is_multiple_of(8));
    let nbyte_row = size / 8;
    for jj in 0..elem_size {
        for ii in 0..nbyte_row {
            for kk in 0..8 {
                *out_b.add(ii * 8 * elem_size + jj * 8 + kk) =
                    *in_b.add((jj * 8 + kk) * nbyte_row + ii);
            }
        }
    }
    (size * elem_size) as i64
}

/// Shuffle bits within the bytes of eight-element blocks (scalar).
unsafe fn shuffle_bit_eightelem_scal(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    debug_assert!(size.is_multiple_of(8));
    let nbyte = elem_size * size;
    let mut t: u64 = 0;
    let mut jj = 0usize;
    while jj < 8 * elem_size {
        let mut ii = 0usize;
        while ii + 8 * elem_size - 1 < nbyte {
            let mut x = ptr::read_unaligned(in_b.add(ii + jj) as *const u64);
            trans_bit_8x8(&mut x, &mut t);
            for kk in 0..8 {
                let out_index = ii + jj / 8 + kk * elem_size;
                *out_b.add(out_index) = x as u8;
                x >>= 8;
            }
            ii += 8 * elem_size;
        }
        jj += 8;
    }
    (size * elem_size) as i64
}

/// Scalar bit-unshuffle.
#[cfg(not(target_arch = "x86_64"))]
unsafe fn untrans_bit_elem_scal(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
    tmp_buf: *mut u8,
) -> i64 {
    debug_assert!(size.is_multiple_of(8));
    trans_byte_bitrow_scal(in_b, tmp_buf, size, elem_size);
    shuffle_bit_eightelem_scal(tmp_buf, out_b, size, elem_size)
}

// SSE2 kernels.

/// Transpose bytes within elements for 16-bit elements (16 elements at once).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn trans_byte_elem_sse_16(in_b: *const u8, out_b: *mut u8, size: usize) -> i64 {
    use std::arch::x86_64::*;
    let mut ii = 0usize;
    while ii + 15 < size {
        let a0 = _mm_loadu_si128(in_b.add(2 * ii) as *const __m128i);
        let b0 = _mm_loadu_si128(in_b.add(2 * ii + 16) as *const __m128i);
        let a1 = _mm_unpacklo_epi8(a0, b0);
        let b1 = _mm_unpackhi_epi8(a0, b0);
        let a0 = _mm_unpacklo_epi8(a1, b1);
        let b0 = _mm_unpackhi_epi8(a1, b1);
        let a1 = _mm_unpacklo_epi8(a0, b0);
        let b1 = _mm_unpackhi_epi8(a0, b0);
        let a0 = _mm_unpacklo_epi8(a1, b1);
        let b0 = _mm_unpackhi_epi8(a1, b1);
        _mm_storeu_si128(out_b.add(ii) as *mut __m128i, a0);
        _mm_storeu_si128(out_b.add(size + ii) as *mut __m128i, b0);
        ii += 16;
    }
    trans_byte_elem_remainder(in_b, out_b, size, 2, size - size % 16)
}

/// Transpose bytes within elements for 32-bit elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn trans_byte_elem_sse_32(in_b: *const u8, out_b: *mut u8, size: usize) -> i64 {
    use std::arch::x86_64::*;
    let mut ii = 0usize;
    while ii + 15 < size {
        let a0 = _mm_loadu_si128(in_b.add(4 * ii) as *const __m128i);
        let b0 = _mm_loadu_si128(in_b.add(4 * ii + 16) as *const __m128i);
        let c0 = _mm_loadu_si128(in_b.add(4 * ii + 32) as *const __m128i);
        let d0 = _mm_loadu_si128(in_b.add(4 * ii + 48) as *const __m128i);
        let a1 = _mm_unpacklo_epi8(a0, b0);
        let b1 = _mm_unpackhi_epi8(a0, b0);
        let c1 = _mm_unpacklo_epi8(c0, d0);
        let d1 = _mm_unpackhi_epi8(c0, d0);
        let a0 = _mm_unpacklo_epi8(a1, b1);
        let b0 = _mm_unpackhi_epi8(a1, b1);
        let c0 = _mm_unpacklo_epi8(c1, d1);
        let d0 = _mm_unpackhi_epi8(c1, d1);
        let a1 = _mm_unpacklo_epi8(a0, b0);
        let b1 = _mm_unpackhi_epi8(a0, b0);
        let c1 = _mm_unpacklo_epi8(c0, d0);
        let d1 = _mm_unpackhi_epi8(c0, d0);
        let a0 = _mm_unpacklo_epi64(a1, c1);
        let b0 = _mm_unpackhi_epi64(a1, c1);
        let c0 = _mm_unpacklo_epi64(b1, d1);
        let d0 = _mm_unpackhi_epi64(b1, d1);
        _mm_storeu_si128(out_b.add(ii) as *mut __m128i, a0);
        _mm_storeu_si128(out_b.add(size + ii) as *mut __m128i, b0);
        _mm_storeu_si128(out_b.add(2 * size + ii) as *mut __m128i, c0);
        _mm_storeu_si128(out_b.add(3 * size + ii) as *mut __m128i, d0);
        ii += 16;
    }
    trans_byte_elem_remainder(in_b, out_b, size, 4, size - size % 16)
}

/// Transpose bytes within elements for 64-bit elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn trans_byte_elem_sse_64(in_b: *const u8, out_b: *mut u8, size: usize) -> i64 {
    use std::arch::x86_64::*;
    let mut ii = 0usize;
    while ii + 15 < size {
        let a0 = _mm_loadu_si128(in_b.add(8 * ii) as *const __m128i);
        let b0 = _mm_loadu_si128(in_b.add(8 * ii + 16) as *const __m128i);
        let c0 = _mm_loadu_si128(in_b.add(8 * ii + 32) as *const __m128i);
        let d0 = _mm_loadu_si128(in_b.add(8 * ii + 48) as *const __m128i);
        let e0 = _mm_loadu_si128(in_b.add(8 * ii + 64) as *const __m128i);
        let f0 = _mm_loadu_si128(in_b.add(8 * ii + 80) as *const __m128i);
        let g0 = _mm_loadu_si128(in_b.add(8 * ii + 96) as *const __m128i);
        let h0 = _mm_loadu_si128(in_b.add(8 * ii + 112) as *const __m128i);
        let a1 = _mm_unpacklo_epi8(a0, b0);
        let b1 = _mm_unpackhi_epi8(a0, b0);
        let c1 = _mm_unpacklo_epi8(c0, d0);
        let d1 = _mm_unpackhi_epi8(c0, d0);
        let e1 = _mm_unpacklo_epi8(e0, f0);
        let f1 = _mm_unpackhi_epi8(e0, f0);
        let g1 = _mm_unpacklo_epi8(g0, h0);
        let h1 = _mm_unpackhi_epi8(g0, h0);
        let a0 = _mm_unpacklo_epi8(a1, b1);
        let b0 = _mm_unpackhi_epi8(a1, b1);
        let c0 = _mm_unpacklo_epi8(c1, d1);
        let d0 = _mm_unpackhi_epi8(c1, d1);
        let e0 = _mm_unpacklo_epi8(e1, f1);
        let f0 = _mm_unpackhi_epi8(e1, f1);
        let g0 = _mm_unpacklo_epi8(g1, h1);
        let h0 = _mm_unpackhi_epi8(g1, h1);
        let a1 = _mm_unpacklo_epi32(a0, c0);
        let b1 = _mm_unpackhi_epi32(a0, c0);
        let c1 = _mm_unpacklo_epi32(b0, d0);
        let d1 = _mm_unpackhi_epi32(b0, d0);
        let e1 = _mm_unpacklo_epi32(e0, g0);
        let f1 = _mm_unpackhi_epi32(e0, g0);
        let g1 = _mm_unpacklo_epi32(f0, h0);
        let h1 = _mm_unpackhi_epi32(f0, h0);
        let a0 = _mm_unpacklo_epi64(a1, e1);
        let b0 = _mm_unpackhi_epi64(a1, e1);
        let c0 = _mm_unpacklo_epi64(b1, f1);
        let d0 = _mm_unpackhi_epi64(b1, f1);
        let e0 = _mm_unpacklo_epi64(c1, g1);
        let f0 = _mm_unpackhi_epi64(c1, g1);
        let g0 = _mm_unpacklo_epi64(d1, h1);
        let h0 = _mm_unpackhi_epi64(d1, h1);
        _mm_storeu_si128(out_b.add(ii) as *mut __m128i, a0);
        _mm_storeu_si128(out_b.add(size + ii) as *mut __m128i, b0);
        _mm_storeu_si128(out_b.add(2 * size + ii) as *mut __m128i, c0);
        _mm_storeu_si128(out_b.add(3 * size + ii) as *mut __m128i, d0);
        _mm_storeu_si128(out_b.add(4 * size + ii) as *mut __m128i, e0);
        _mm_storeu_si128(out_b.add(5 * size + ii) as *mut __m128i, f0);
        _mm_storeu_si128(out_b.add(6 * size + ii) as *mut __m128i, g0);
        _mm_storeu_si128(out_b.add(7 * size + ii) as *mut __m128i, h0);
        ii += 16;
    }
    trans_byte_elem_remainder(in_b, out_b, size, 8, size - size % 16)
}

/// Transpose bytes within elements using the best SSE2 algorithm available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn trans_byte_elem_sse2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
    tmp_buf: *mut u8,
) -> i64 {
    match elem_size {
        1 => {
            ptr::copy_nonoverlapping(in_b, out_b, size);
            return (size) as i64;
        }
        2 => return trans_byte_elem_sse_16(in_b, out_b, size),
        4 => return trans_byte_elem_sse_32(in_b, out_b, size),
        8 => return trans_byte_elem_sse_64(in_b, out_b, size),
        _ => {}
    }

    // Odd element sizes are faster with the scalar algorithm.
    if !elem_size.is_multiple_of(4) {
        return trans_byte_elem_scal(in_b, out_b, size, elem_size);
    }

    // Multiple of a power of two: transpose hierarchically.
    if elem_size.is_multiple_of(8) {
        let nchunk_elem = elem_size / 8;
        trans_elem(in_b, out_b, size, nchunk_elem, 8);
        trans_byte_elem_sse_64(out_b, tmp_buf, size * nchunk_elem);
        trans_elem(tmp_buf, out_b, 8, nchunk_elem, size);
    } else {
        let nchunk_elem = elem_size / 4;
        trans_elem(in_b, out_b, size, nchunk_elem, 4);
        trans_byte_elem_sse_32(out_b, tmp_buf, size * nchunk_elem);
        trans_elem(tmp_buf, out_b, 4, nchunk_elem, size);
    }
    (size * elem_size) as i64
}

/// Transpose bits within bytes (16 bytes at a time, 16-bit movemask).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn trans_bit_byte_sse2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    use std::arch::x86_64::*;
    let nbyte = elem_size * size;
    debug_assert!(nbyte.is_multiple_of(8));

    let mut ii = 0usize;
    while ii + 15 < nbyte {
        let mut xmm = _mm_loadu_si128(in_b.add(ii) as *const __m128i);
        for kk in 0..8 {
            let bt = _mm_movemask_epi8(xmm) as u16;
            xmm = _mm_slli_epi16(xmm, 1);
            let out_ptr = out_b.add(((7 - kk) * nbyte + ii) / 8) as *mut u16;
            ptr::write_unaligned(out_ptr, bt);
        }
        ii += 16;
    }
    trans_bit_byte_remainder(in_b, out_b, size, elem_size, nbyte - nbyte % 16)
}

/// SSE2 bitshuffle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn trans_bit_elem_sse2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
    tmp_buf: *mut u8,
) -> i64 {
    debug_assert!(size.is_multiple_of(8));
    trans_byte_elem_sse2(in_b, out_b, size, elem_size, tmp_buf);
    trans_bit_byte_sse2(out_b, tmp_buf, size, elem_size);
    trans_bitrow_eight(tmp_buf, out_b, size, elem_size)
}

/// For data organized into a row per bit, transpose the bytes (8 rows at a
/// time, 16 bytes per row).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn trans_byte_bitrow_sse2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    use std::arch::x86_64::*;
    let nrows = 8 * elem_size;
    let nbyte_row = size / 8;
    debug_assert!(size.is_multiple_of(8));

    let mut ii = 0usize;
    while ii + 7 < nrows {
        let mut jj = 0usize;
        while jj + 15 < nbyte_row {
            let a0 = _mm_loadu_si128(in_b.add(ii * nbyte_row + jj) as *const __m128i);
            let b0 = _mm_loadu_si128(in_b.add((ii + 1) * nbyte_row + jj) as *const __m128i);
            let c0 = _mm_loadu_si128(in_b.add((ii + 2) * nbyte_row + jj) as *const __m128i);
            let d0 = _mm_loadu_si128(in_b.add((ii + 3) * nbyte_row + jj) as *const __m128i);
            let e0 = _mm_loadu_si128(in_b.add((ii + 4) * nbyte_row + jj) as *const __m128i);
            let f0 = _mm_loadu_si128(in_b.add((ii + 5) * nbyte_row + jj) as *const __m128i);
            let g0 = _mm_loadu_si128(in_b.add((ii + 6) * nbyte_row + jj) as *const __m128i);
            let h0 = _mm_loadu_si128(in_b.add((ii + 7) * nbyte_row + jj) as *const __m128i);

            let a1 = _mm_unpacklo_epi8(a0, b0);
            let b1 = _mm_unpacklo_epi8(c0, d0);
            let c1 = _mm_unpacklo_epi8(e0, f0);
            let d1 = _mm_unpacklo_epi8(g0, h0);
            let e1 = _mm_unpackhi_epi8(a0, b0);
            let f1 = _mm_unpackhi_epi8(c0, d0);
            let g1 = _mm_unpackhi_epi8(e0, f0);
            let h1 = _mm_unpackhi_epi8(g0, h0);

            let a0 = _mm_unpacklo_epi16(a1, b1);
            let b0 = _mm_unpacklo_epi16(c1, d1);
            let c0 = _mm_unpackhi_epi16(a1, b1);
            let d0 = _mm_unpackhi_epi16(c1, d1);
            let e0 = _mm_unpacklo_epi16(e1, f1);
            let f0 = _mm_unpacklo_epi16(g1, h1);
            let g0 = _mm_unpackhi_epi16(e1, f1);
            let h0 = _mm_unpackhi_epi16(g1, h1);

            let a1 = _mm_unpacklo_epi32(a0, b0);
            let b1 = _mm_unpackhi_epi32(a0, b0);
            let c1 = _mm_unpacklo_epi32(c0, d0);
            let d1 = _mm_unpackhi_epi32(c0, d0);
            let e1 = _mm_unpacklo_epi32(e0, f0);
            let f1 = _mm_unpackhi_epi32(e0, f0);
            let g1 = _mm_unpacklo_epi32(g0, h0);
            let h1 = _mm_unpackhi_epi32(g0, h0);

            // Store the low 8 bytes of each register, then the high 8.
            let out_base = out_b.add(jj * nrows + ii);
            for (k, r) in [a1, b1, c1, d1, e1, f1, g1, h1].into_iter().enumerate() {
                _mm_storel_epi64(out_base.add((2 * k) * nrows) as *mut __m128i, r);
                let hi = _mm_unpackhi_epi64(r, r);
                _mm_storel_epi64(out_base.add((2 * k + 1) * nrows) as *mut __m128i, hi);
            }
            jj += 16;
        }
        // Remaining columns, byte by byte.
        let mut jj = nbyte_row - nbyte_row % 16;
        while jj < nbyte_row {
            for k in 0..8 {
                *out_b.add(jj * nrows + ii + k) = *in_b.add((ii + k) * nbyte_row + jj);
            }
            jj += 1;
        }
        ii += 8;
    }
    (size * elem_size) as i64
}

/// Shuffle bits within the bytes of eight-element blocks (16-bit movemask).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn shuffle_bit_eightelem_sse2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    use std::arch::x86_64::*;
    debug_assert!(size.is_multiple_of(8));
    let nbyte = elem_size * size;

    if !elem_size.is_multiple_of(2) {
        return shuffle_bit_eightelem_scal(in_b, out_b, size, elem_size);
    }

    let mut ii = 0usize;
    while ii + 8 * elem_size - 1 < nbyte {
        let mut jj = 0usize;
        while jj + 15 < 8 * elem_size {
            let mut xmm = _mm_loadu_si128(in_b.add(ii + jj) as *const __m128i);
            for kk in 0..8 {
                let bt = _mm_movemask_epi8(xmm) as u16;
                xmm = _mm_slli_epi16(xmm, 1);
                let ind = ii + jj / 8 + (7 - kk) * elem_size;
                ptr::write_unaligned(out_b.add(ind) as *mut u16, bt);
            }
            jj += 16;
        }
        ii += 8 * elem_size;
    }
    (size * elem_size) as i64
}

/// SSE2 bit-unshuffle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn untrans_bit_elem_sse2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
    tmp_buf: *mut u8,
) -> i64 {
    debug_assert!(size.is_multiple_of(8));
    trans_byte_bitrow_sse2(in_b, tmp_buf, size, elem_size);
    shuffle_bit_eightelem_sse2(tmp_buf, out_b, size, elem_size)
}

// AVX2 kernels.

/// Transpose bits within bytes (32 bytes at a time, 32-bit movemask).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn trans_bit_byte_avx2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    use std::arch::x86_64::*;
    let nbyte = elem_size * size;

    let mut ii = 0usize;
    while ii + 31 < nbyte {
        let mut ymm = _mm256_loadu_si256(in_b.add(ii) as *const __m256i);
        for kk in 0..8 {
            let bt = _mm256_movemask_epi8(ymm) as u32;
            ymm = _mm256_slli_epi16(ymm, 1);
            let out_ptr = out_b.add(((7 - kk) * nbyte + ii) / 8) as *mut u32;
            ptr::write_unaligned(out_ptr, bt);
        }
        ii += 32;
    }
    trans_bit_byte_remainder(in_b, out_b, size, elem_size, nbyte - nbyte % 32)
}

/// AVX2 bitshuffle (byte transpose via SSE2, bit transpose via AVX2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn trans_bit_elem_avx2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
    tmp_buf: *mut u8,
) -> i64 {
    debug_assert!(size.is_multiple_of(8));
    trans_byte_elem_sse2(in_b, out_b, size, elem_size, tmp_buf);
    trans_bit_byte_avx2(out_b, tmp_buf, size, elem_size);
    trans_bitrow_eight(tmp_buf, out_b, size, elem_size)
}

/// For data organized into a row per bit, transpose the bytes (AVX2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn trans_byte_bitrow_avx2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    use std::arch::x86_64::*;
    let nrows = 8 * elem_size;
    let nbyte_row = size / 8;
    debug_assert!(size.is_multiple_of(8));

    if !elem_size.is_multiple_of(4) {
        return trans_byte_bitrow_sse2(in_b, out_b, size, elem_size);
    }

    let mut jj = 0usize;
    while jj + 31 < nbyte_row {
        let mut ii = 0usize;
        while ii + 3 < elem_size {
            let mut ymm_storeage = [[_mm256_setzero_si256(); 4]; 8];
            for hh in 0..4 {
                let mut ymm_0 = [_mm256_setzero_si256(); 8];
                for kk in 0..8 {
                    ymm_0[kk] = _mm256_loadu_si256(
                        in_b.add((ii * 8 + hh * 8 + kk) * nbyte_row + jj) as *const __m256i,
                    );
                }
                let mut ymm_1 = [_mm256_setzero_si256(); 8];
                for kk in 0..4 {
                    ymm_1[kk] = _mm256_unpacklo_epi8(ymm_0[kk * 2], ymm_0[kk * 2 + 1]);
                    ymm_1[kk + 4] = _mm256_unpackhi_epi8(ymm_0[kk * 2], ymm_0[kk * 2 + 1]);
                }
                for kk in 0..2 {
                    for mm in 0..2 {
                        ymm_0[kk * 4 + mm] = _mm256_unpacklo_epi16(
                            ymm_1[kk * 4 + mm * 2],
                            ymm_1[kk * 4 + mm * 2 + 1],
                        );
                        ymm_0[kk * 4 + mm + 2] = _mm256_unpackhi_epi16(
                            ymm_1[kk * 4 + mm * 2],
                            ymm_1[kk * 4 + mm * 2 + 1],
                        );
                    }
                }
                for kk in 0..4 {
                    ymm_1[kk * 2] = _mm256_unpacklo_epi32(ymm_0[kk * 2], ymm_0[kk * 2 + 1]);
                    ymm_1[kk * 2 + 1] = _mm256_unpackhi_epi32(ymm_0[kk * 2], ymm_0[kk * 2 + 1]);
                }
                for kk in 0..8 {
                    ymm_storeage[kk][hh] = ymm_1[kk];
                }
            }
            for mm in 0..8 {
                let mut ymm_0 = [_mm256_setzero_si256(); 4];
                ymm_0.copy_from_slice(&ymm_storeage[mm]);
                let ymm_1_0 = _mm256_unpacklo_epi64(ymm_0[0], ymm_0[1]);
                let ymm_1_1 = _mm256_unpacklo_epi64(ymm_0[2], ymm_0[3]);
                let ymm_1_2 = _mm256_unpackhi_epi64(ymm_0[0], ymm_0[1]);
                let ymm_1_3 = _mm256_unpackhi_epi64(ymm_0[2], ymm_0[3]);
                let ymm_0_0 = _mm256_permute2x128_si256(ymm_1_0, ymm_1_1, 32);
                let ymm_0_1 = _mm256_permute2x128_si256(ymm_1_2, ymm_1_3, 32);
                let ymm_0_2 = _mm256_permute2x128_si256(ymm_1_0, ymm_1_1, 49);
                let ymm_0_3 = _mm256_permute2x128_si256(ymm_1_2, ymm_1_3, 49);
                _mm256_storeu_si256(
                    out_b.add((jj + mm * 2) * nrows + ii * 8) as *mut __m256i,
                    ymm_0_0,
                );
                _mm256_storeu_si256(
                    out_b.add((jj + mm * 2 + 1) * nrows + ii * 8) as *mut __m256i,
                    ymm_0_1,
                );
                _mm256_storeu_si256(
                    out_b.add((jj + mm * 2 + 16) * nrows + ii * 8) as *mut __m256i,
                    ymm_0_2,
                );
                _mm256_storeu_si256(
                    out_b.add((jj + mm * 2 + 17) * nrows + ii * 8) as *mut __m256i,
                    ymm_0_3,
                );
            }
            ii += 4;
        }
        jj += 32;
    }
    // Remaining columns, byte by byte.
    let mut jj = nbyte_row - nbyte_row % 32;
    while jj < nbyte_row {
        for ii in 0..nrows {
            *out_b.add(jj * nrows + ii) = *in_b.add(ii * nbyte_row + jj);
        }
        jj += 1;
    }
    (size * elem_size) as i64
}

/// Shuffle bits within the bytes of eight-element blocks (AVX2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn shuffle_bit_eightelem_avx2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
) -> i64 {
    use std::arch::x86_64::*;
    debug_assert!(size.is_multiple_of(8));
    let nbyte = elem_size * size;

    if !elem_size.is_multiple_of(4) {
        return shuffle_bit_eightelem_sse2(in_b, out_b, size, elem_size);
    }

    let mut jj = 0usize;
    while jj + 31 < 8 * elem_size {
        let mut ii = 0usize;
        while ii + 8 * elem_size - 1 < nbyte {
            let mut ymm = _mm256_loadu_si256(in_b.add(ii + jj) as *const __m256i);
            for kk in 0..8 {
                let bt = _mm256_movemask_epi8(ymm) as u32;
                ymm = _mm256_slli_epi16(ymm, 1);
                let ind = ii + jj / 8 + (7 - kk) * elem_size;
                ptr::write_unaligned(out_b.add(ind) as *mut u32, bt);
            }
            ii += 8 * elem_size;
        }
        jj += 32;
    }
    (size * elem_size) as i64
}

/// AVX2 bit-unshuffle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn untrans_bit_elem_avx2(
    in_b: *const u8,
    out_b: *mut u8,
    size: usize,
    elem_size: usize,
    tmp_buf: *mut u8,
) -> i64 {
    debug_assert!(size.is_multiple_of(8));
    trans_byte_bitrow_avx2(in_b, tmp_buf, size, elem_size);
    shuffle_bit_eightelem_avx2(tmp_buf, out_b, size, elem_size)
}

// Checked and unchecked dispatch entry points.

/// Bitshuffle a block. Returns the number of processed bytes, or a negative value
/// on error. `tmp` must be at least `blocksize` bytes.
#[cfg(test)]
pub(crate) fn bitshuffle(
    typesize: usize,
    blocksize: usize,
    src: &[u8],
    dest: &mut [u8],
    tmp: &mut [u8],
) -> i32 {
    if typesize == 0
        || blocksize > i32::MAX as usize
        || src.len() < blocksize
        || dest.len() < blocksize
        || tmp.len() < blocksize
    {
        return -1;
    }
    // SAFETY: the checks above establish the unchecked entry point's complete
    // buffer, size, and element-width contract.
    unsafe { bitshuffle_unchecked(typesize, blocksize, src, dest, tmp) }
}

/// Bitshuffle after the caller has validated all buffer lengths.
///
/// # Safety
///
/// `typesize` must be non-zero, `blocksize` must fit in `i32`, and `src`,
/// `dest`, and `tmp` must each contain at least `blocksize` bytes. The three
/// buffers must not overlap.
pub(crate) unsafe fn bitshuffle_unchecked(
    typesize: usize,
    blocksize: usize,
    src: &[u8],
    dest: &mut [u8],
    tmp: &mut [u8],
) -> i32 {
    let size = blocksize / typesize;
    if size.is_multiple_of(8) {
        // SAFETY: the caller guarantees that all three buffers cover
        // `blocksize`; `size * typesize <= blocksize`, and the dispatchers
        // access only that complete-element prefix.
        let rc = unsafe { dispatch_trans_bit_elem(typesize, size, src, dest, tmp) };
        // Copy the leftovers.
        let offset = size * typesize;
        dest[offset..blocksize].copy_from_slice(&src[offset..blocksize]);
        rc as i32
    } else {
        dest[..blocksize].copy_from_slice(&src[..blocksize]);
        blocksize as i32
    }
}

/// Bit-unshuffle a block. Returns the number of processed bytes, or a negative
/// value on error. `tmp` must be at least `blocksize` bytes.
#[cfg(test)]
pub(crate) fn bitunshuffle(
    typesize: usize,
    blocksize: usize,
    src: &[u8],
    dest: &mut [u8],
    tmp: &mut [u8],
) -> i32 {
    if typesize == 0
        || blocksize > i32::MAX as usize
        || src.len() < blocksize
        || dest.len() < blocksize
        || tmp.len() < blocksize
    {
        return -1;
    }
    // SAFETY: the checks above establish the unchecked entry point's complete
    // buffer, size, and element-width contract.
    unsafe { bitunshuffle_unchecked(typesize, blocksize, src, dest, tmp) }
}

/// Bit-unshuffle after the caller has validated all buffer lengths.
///
/// # Safety
///
/// `typesize` must be non-zero, `blocksize` must fit in `i32`, and `src`,
/// `dest`, and `tmp` must each contain at least `blocksize` bytes. The three
/// buffers must not overlap.
pub(crate) unsafe fn bitunshuffle_unchecked(
    typesize: usize,
    blocksize: usize,
    src: &[u8],
    dest: &mut [u8],
    tmp: &mut [u8],
) -> i32 {
    let size = blocksize / typesize;
    if size.is_multiple_of(8) {
        // SAFETY: the caller guarantees that all three buffers cover
        // `blocksize`; `size * typesize <= blocksize`, and the dispatchers
        // access only that complete-element prefix.
        let rc = unsafe { dispatch_untrans_bit_elem(typesize, size, src, dest, tmp) };
        let offset = size * typesize;
        dest[offset..blocksize].copy_from_slice(&src[offset..blocksize]);
        rc as i32
    } else {
        dest[..blocksize].copy_from_slice(&src[..blocksize]);
        blocksize as i32
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn dispatch_trans_bit_elem(
    typesize: usize,
    size: usize,
    src: &[u8],
    dest: &mut [u8],
    tmp: &mut [u8],
) -> i64 {
    let in_b = src.as_ptr();
    let out_b = dest.as_mut_ptr();
    let tmp_b = tmp.as_mut_ptr();
    if std::arch::is_x86_feature_detected!("avx2") {
        trans_bit_elem_avx2(in_b, out_b, size, typesize, tmp_b)
    } else {
        trans_bit_elem_sse2(in_b, out_b, size, typesize, tmp_b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn dispatch_trans_bit_elem(
    typesize: usize,
    size: usize,
    src: &[u8],
    dest: &mut [u8],
    tmp: &mut [u8],
) -> i64 {
    trans_bit_elem_scal(
        src.as_ptr(),
        dest.as_mut_ptr(),
        size,
        typesize,
        tmp.as_mut_ptr(),
    )
}

#[cfg(target_arch = "x86_64")]
unsafe fn dispatch_untrans_bit_elem(
    typesize: usize,
    size: usize,
    src: &[u8],
    dest: &mut [u8],
    tmp: &mut [u8],
) -> i64 {
    let in_b = src.as_ptr();
    let out_b = dest.as_mut_ptr();
    let tmp_b = tmp.as_mut_ptr();
    if std::arch::is_x86_feature_detected!("avx2") {
        untrans_bit_elem_avx2(in_b, out_b, size, typesize, tmp_b)
    } else {
        untrans_bit_elem_sse2(in_b, out_b, size, typesize, tmp_b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn dispatch_untrans_bit_elem(
    typesize: usize,
    size: usize,
    src: &[u8],
    dest: &mut [u8],
    tmp: &mut [u8],
) -> i64 {
    untrans_bit_elem_scal(
        src.as_ptr(),
        dest.as_mut_ptr(),
        size,
        typesize,
        tmp.as_mut_ptr(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterned(blocksize: usize) -> Vec<u8> {
        (0..blocksize)
            .map(|i| ((i * 13 + (i >> 3)) & 0xFF) as u8)
            .collect()
    }

    fn roundtrip(typesize: usize, size: usize) {
        let blocksize = typesize * size;
        let src = patterned(blocksize);
        let mut shuffled = vec![0u8; blocksize];
        let mut tmp = vec![0u8; blocksize];
        let mut back = vec![0u8; blocksize];
        let rc = bitshuffle(typesize, blocksize, &src, &mut shuffled, &mut tmp);
        assert!(rc >= 0);
        let rc = bitunshuffle(typesize, blocksize, &shuffled, &mut back, &mut tmp);
        assert!(rc >= 0);
        assert_eq!(back, src, "typesize={typesize} size={size}");
    }

    #[test]
    fn roundtrip_multiple_of_eight() {
        for typesize in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32] {
            for size in [8usize, 16, 24, 40, 64, 128, 1024] {
                roundtrip(typesize, size);
            }
        }
    }

    #[test]
    fn not_multiple_of_eight_is_identity() {
        // When the element count is not a multiple of 8 the filter must
        // copy the block verbatim.
        let typesize = 4usize;
        let size = 10usize; // not a multiple of 8
        let blocksize = typesize * size;
        let src = patterned(blocksize);
        let mut out = vec![0u8; blocksize];
        let mut tmp = vec![0u8; blocksize];
        let rc = bitshuffle(typesize, blocksize, &src, &mut out, &mut tmp);
        assert_eq!(rc, blocksize as i32);
        assert_eq!(out, src);
        let rc = bitunshuffle(typesize, blocksize, &src, &mut out, &mut tmp);
        assert_eq!(rc, blocksize as i32);
        assert_eq!(out, src);
    }

    #[test]
    fn safe_entries_reject_invalid_buffer_contracts() {
        let source = [0u8; 8];
        let mut destination = [0u8; 8];
        let mut temporary = [0u8; 8];
        assert_eq!(
            bitshuffle(0, 8, &source, &mut destination, &mut temporary),
            -1
        );
        assert_eq!(
            bitshuffle(1, 9, &source, &mut destination, &mut temporary),
            -1
        );
        assert_eq!(
            bitunshuffle(1, 9, &source, &mut destination, &mut temporary),
            -1
        );
    }

    #[test]
    fn leftover_bytes_copied() {
        // blocksize not a multiple of typesize: trailing bytes are copied
        // verbatim by the entry point.
        let typesize = 8usize;
        let size = 16usize;
        let extra = 5usize;
        let blocksize = typesize * size + extra;
        let src = patterned(blocksize);
        let mut shuffled = vec![0u8; blocksize];
        let mut tmp = vec![0u8; blocksize];
        let mut back = vec![0u8; blocksize];
        bitshuffle(typesize, blocksize, &src, &mut shuffled, &mut tmp);
        assert_eq!(
            &shuffled[typesize * size..],
            &src[typesize * size..],
            "leftover bytes must not be shuffled"
        );
        bitunshuffle(typesize, blocksize, &shuffled, &mut back, &mut tmp);
        assert_eq!(back, src);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_matches_scalar() {
        use std::arch::is_x86_feature_detected;
        for typesize in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32] {
            for size in [8usize, 16, 24, 40, 64, 128, 1024] {
                let blocksize = typesize * size;
                let src = patterned(blocksize);
                let mut tmp = vec![0u8; blocksize];
                // SAFETY: the test allocates distinct `blocksize`-byte buffers
                // and only calls kernels whose CPU feature was detected.
                unsafe {
                    if is_x86_feature_detected!("sse2") {
                        let mut sse_out = vec![0u8; blocksize];
                        trans_bit_elem_sse2(
                            src.as_ptr(),
                            sse_out.as_mut_ptr(),
                            size,
                            typesize,
                            tmp.as_mut_ptr(),
                        );
                        if is_x86_feature_detected!("avx2") {
                            let mut avx_out = vec![0u8; blocksize];
                            trans_bit_elem_avx2(
                                src.as_ptr(),
                                avx_out.as_mut_ptr(),
                                size,
                                typesize,
                                tmp.as_mut_ptr(),
                            );
                            assert_eq!(
                                avx_out, sse_out,
                                "avx2 vs sse2 typesize={typesize} size={size}"
                            );
                        }
                    }
                }
            }
        }
    }
}
