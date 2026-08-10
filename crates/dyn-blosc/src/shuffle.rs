//! Byte shuffle and unshuffle filters for Blosc-compatible payloads.
//!
//! Shuffle transposes a typed array so that all first bytes of every element
//! are contiguous, then all second bytes, etc. This groups equal-significance
//! bytes together and makes the data much more compressible for typical
//! numeric payloads. Bit shuffle additionally transposes bits within bytes.
//!
//! SSE2 and AVX2 implementations are selected at runtime; a scalar generic
//! implementation covers every other case (and the vector tails).
//! # Unsafe kernel boundary
//!
//! All unsafe kernels require a non-zero element size, non-overlapping buffers
//! of at least `blocksize` bytes, and a vectorizable prefix that is both bounded
//! by the block and element-aligned. Runtime CPU detection guards every SIMD
//! entry point.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

/// Generic (non-vectorized) shuffle; also used by the vectorized paths to
/// handle the elements that do not fit a whole number of vectors.
#[inline]
unsafe fn shuffle_generic_inline_unchecked(
    typesize: usize,
    vectorizable_blocksize: usize,
    blocksize: usize,
    src: &[u8],
    dest: &mut [u8],
) {
    debug_assert!(typesize != 0);
    debug_assert!(vectorizable_blocksize <= blocksize);
    debug_assert!(vectorizable_blocksize.is_multiple_of(typesize));
    debug_assert!(src.len() >= blocksize);
    debug_assert!(dest.len() >= blocksize);
    let neblock_quot = blocksize / typesize;
    let neblock_rem = blocksize % typesize;
    let vectorizable_elements = vectorizable_blocksize / typesize;

    // SAFETY: the function contract and assertions above prove every computed
    // source and destination index is below `blocksize`. The mutable and
    // immutable slice borrows guarantee non-overlap.
    unsafe {
        let source = src.as_ptr();
        let destination = dest.as_mut_ptr();
        for j in 0..typesize {
            for i in vectorizable_elements..neblock_quot {
                *destination.add(j * neblock_quot + i) = *source.add(i * typesize + j);
            }
        }
        if neblock_rem != 0 {
            let remainder_start = blocksize - neblock_rem;
            std::ptr::copy_nonoverlapping(
                source.add(remainder_start),
                destination.add(remainder_start),
                neblock_rem,
            );
        }
    }
}

/// Generic (non-vectorized) unshuffle.
#[inline]
unsafe fn unshuffle_generic_inline_unchecked(
    typesize: usize,
    vectorizable_blocksize: usize,
    blocksize: usize,
    src: &[u8],
    dest: &mut [u8],
) {
    debug_assert!(typesize != 0);
    debug_assert!(vectorizable_blocksize <= blocksize);
    debug_assert!(vectorizable_blocksize.is_multiple_of(typesize));
    debug_assert!(src.len() >= blocksize);
    debug_assert!(dest.len() >= blocksize);
    let neblock_quot = blocksize / typesize;
    let neblock_rem = blocksize % typesize;
    let vectorizable_elements = vectorizable_blocksize / typesize;

    // SAFETY: the function contract and assertions above prove every computed
    // source and destination index is below `blocksize`. The mutable and
    // immutable slice borrows guarantee non-overlap.
    unsafe {
        let source = src.as_ptr();
        let destination = dest.as_mut_ptr();
        for i in vectorizable_elements..neblock_quot {
            for j in 0..typesize {
                *destination.add(i * typesize + j) = *source.add(j * neblock_quot + i);
            }
        }
        if neblock_rem != 0 {
            let remainder_start = blocksize - neblock_rem;
            std::ptr::copy_nonoverlapping(
                source.add(remainder_start),
                destination.add(remainder_start),
                neblock_rem,
            );
        }
    }
}

/// Shuffle a block, dispatching to the best implementation for this CPU.
///
/// # Safety
///
/// `typesize` must be non-zero, and `src` and `dest` must each contain at
/// least `blocksize` bytes. The buffers must not overlap.
pub(crate) unsafe fn shuffle_unchecked(
    typesize: usize,
    blocksize: usize,
    src: &[u8],
    dest: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the caller guarantees the buffer and element-size invariants;
    // runtime detection guarantees the selected target feature.
    unsafe {
        if std::arch::is_x86_feature_detected!("avx2") {
            return shuffle_avx2(typesize, blocksize, src, dest);
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            return shuffle_sse2(typesize, blocksize, src, dest);
        }
    }
    shuffle_generic_inline_unchecked(typesize, 0, blocksize, src, dest);
}

/// Unshuffle a block, dispatching to the best implementation for this CPU.
///
/// # Safety
///
/// `typesize` must be non-zero, and `src` and `dest` must each contain at
/// least `blocksize` bytes. The buffers must not overlap.
pub(crate) unsafe fn unshuffle_unchecked(
    typesize: usize,
    blocksize: usize,
    src: &[u8],
    dest: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the caller guarantees the buffer and element-size invariants;
    // runtime detection guarantees the selected target feature.
    unsafe {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unshuffle_avx2(typesize, blocksize, src, dest);
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            return unshuffle_sse2(typesize, blocksize, src, dest);
        }
    }
    unshuffle_generic_inline_unchecked(typesize, 0, blocksize, src, dest);
}

/// Scalar shuffle for small blocks or non-x86 targets.
#[cfg(test)]
pub(crate) fn shuffle_generic(typesize: usize, blocksize: usize, src: &[u8], dest: &mut [u8]) {
    assert!(typesize != 0, "typesize must be non-zero");
    assert!(src.len() >= blocksize, "source is shorter than blocksize");
    assert!(
        dest.len() >= blocksize,
        "destination is shorter than blocksize"
    );
    // SAFETY: the assertions establish the unchecked kernel's preconditions,
    // and distinct slice borrows cannot overlap.
    unsafe { shuffle_generic_inline_unchecked(typesize, 0, blocksize, src, dest) };
}

/// Scalar unshuffle.
#[cfg(test)]
pub(crate) fn unshuffle_generic(typesize: usize, blocksize: usize, src: &[u8], dest: &mut [u8]) {
    assert!(typesize != 0, "typesize must be non-zero");
    assert!(src.len() >= blocksize, "source is shorter than blocksize");
    assert!(
        dest.len() >= blocksize,
        "destination is shorter than blocksize"
    );
    // SAFETY: the assertions establish the unchecked kernel's preconditions,
    // and distinct slice borrows cannot overlap.
    unsafe { unshuffle_generic_inline_unchecked(typesize, 0, blocksize, src, dest) };
}

// SSE2 kernels.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn shuffle_sse2(typesize: usize, blocksize: usize, src: &[u8], dest: &mut [u8]) {
    use std::arch::x86_64::*;

    const VEC: usize = 16;
    let vectorized_chunk_size = typesize * VEC;
    if blocksize < vectorized_chunk_size {
        return shuffle_generic_inline_unchecked(typesize, 0, blocksize, src, dest);
    }
    let vectorizable_bytes = blocksize - (blocksize % vectorized_chunk_size);
    let vectorizable_elements = vectorizable_bytes / typesize;
    let total_elements = blocksize / typesize;

    match typesize {
        2 => {
            for j in (0..vectorizable_elements).step_by(VEC) {
                let mut xmm0 = [_mm_setzero_si128(); 2];
                let mut xmm1 = [_mm_setzero_si128(); 2];
                for k in 0..2 {
                    xmm0[k] = _mm_loadu_si128(src.as_ptr().add(j * 2 + k * VEC) as *const __m128i);
                    xmm0[k] = _mm_shufflelo_epi16(xmm0[k], 0xd8);
                    xmm0[k] = _mm_shufflehi_epi16(xmm0[k], 0xd8);
                    xmm0[k] = _mm_shuffle_epi32(xmm0[k], 0xd8);
                    xmm1[k] = _mm_shuffle_epi32(xmm0[k], 0x4e);
                    xmm0[k] = _mm_unpacklo_epi8(xmm0[k], xmm1[k]);
                    xmm0[k] = _mm_shuffle_epi32(xmm0[k], 0xd8);
                    xmm1[k] = _mm_shuffle_epi32(xmm0[k], 0x4e);
                    xmm0[k] = _mm_unpacklo_epi16(xmm0[k], xmm1[k]);
                    xmm0[k] = _mm_shuffle_epi32(xmm0[k], 0xd8);
                }
                xmm1[0] = _mm_unpacklo_epi64(xmm0[0], xmm0[1]);
                xmm1[1] = _mm_unpackhi_epi64(xmm0[0], xmm0[1]);
                for k in 0..2 {
                    _mm_storeu_si128(
                        dest.as_mut_ptr().add(j + k * total_elements) as *mut __m128i,
                        xmm1[k],
                    );
                }
            }
        }
        4 => {
            for i in (0..vectorizable_elements).step_by(VEC) {
                let mut xmm0 = [_mm_setzero_si128(); 4];
                let mut xmm1 = [_mm_setzero_si128(); 4];
                for j in 0..4 {
                    xmm0[j] = _mm_loadu_si128(src.as_ptr().add(i * 4 + j * VEC) as *const __m128i);
                    xmm1[j] = _mm_shuffle_epi32(xmm0[j], 0xd8);
                    xmm0[j] = _mm_shuffle_epi32(xmm0[j], 0x8d);
                    xmm0[j] = _mm_unpacklo_epi8(xmm1[j], xmm0[j]);
                    xmm1[j] = _mm_shuffle_epi32(xmm0[j], 0x4e);
                    xmm0[j] = _mm_unpacklo_epi16(xmm0[j], xmm1[j]);
                }
                for j in 0..2 {
                    xmm1[j * 2] = _mm_unpacklo_epi32(xmm0[j * 2], xmm0[j * 2 + 1]);
                    xmm1[j * 2 + 1] = _mm_unpackhi_epi32(xmm0[j * 2], xmm0[j * 2 + 1]);
                }
                for j in 0..2 {
                    xmm0[j * 2] = _mm_unpacklo_epi64(xmm1[j], xmm1[j + 2]);
                    xmm0[j * 2 + 1] = _mm_unpackhi_epi64(xmm1[j], xmm1[j + 2]);
                }
                for j in 0..4 {
                    _mm_storeu_si128(
                        dest.as_mut_ptr().add(i + j * total_elements) as *mut __m128i,
                        xmm0[j],
                    );
                }
            }
        }
        8 => {
            for j in (0..vectorizable_elements).step_by(VEC) {
                let mut xmm0 = [_mm_setzero_si128(); 8];
                let mut xmm1 = [_mm_setzero_si128(); 8];
                for k in 0..8 {
                    xmm0[k] = _mm_loadu_si128(src.as_ptr().add(j * 8 + k * VEC) as *const __m128i);
                    xmm1[k] = _mm_shuffle_epi32(xmm0[k], 0x4e);
                    xmm1[k] = _mm_unpacklo_epi8(xmm0[k], xmm1[k]);
                }
                for (k, l) in (0..4).zip((0..8).step_by(2)) {
                    xmm0[k * 2] = _mm_unpacklo_epi16(xmm1[l], xmm1[l + 1]);
                    xmm0[k * 2 + 1] = _mm_unpackhi_epi16(xmm1[l], xmm1[l + 1]);
                }
                for k in 0..4 {
                    let l = if k < 2 { k } else { k + 2 };
                    xmm1[k * 2] = _mm_unpacklo_epi32(xmm0[l], xmm0[l + 2]);
                    xmm1[k * 2 + 1] = _mm_unpackhi_epi32(xmm0[l], xmm0[l + 2]);
                }
                for k in 0..4 {
                    xmm0[k * 2] = _mm_unpacklo_epi64(xmm1[k], xmm1[k + 4]);
                    xmm0[k * 2 + 1] = _mm_unpackhi_epi64(xmm1[k], xmm1[k + 4]);
                }
                for k in 0..8 {
                    _mm_storeu_si128(
                        dest.as_mut_ptr().add(j + k * total_elements) as *mut __m128i,
                        xmm0[k],
                    );
                }
            }
        }
        16 => {
            for j in (0..vectorizable_elements).step_by(VEC) {
                let mut xmm0 = [_mm_setzero_si128(); 16];
                let mut xmm1 = [_mm_setzero_si128(); 16];
                for k in 0..16 {
                    xmm0[k] = _mm_loadu_si128(src.as_ptr().add(j * 16 + k * VEC) as *const __m128i);
                }
                for (k, l) in (0..8).zip((0..16).step_by(2)) {
                    xmm1[k * 2] = _mm_unpacklo_epi8(xmm0[l], xmm0[l + 1]);
                    xmm1[k * 2 + 1] = _mm_unpackhi_epi8(xmm0[l], xmm0[l + 1]);
                }
                for k in 0..8 {
                    let l = (k / 2) * 4 + k % 2;
                    xmm0[k * 2] = _mm_unpacklo_epi16(xmm1[l], xmm1[l + 2]);
                    xmm0[k * 2 + 1] = _mm_unpackhi_epi16(xmm1[l], xmm1[l + 2]);
                }
                for k in 0..8 {
                    let l = (k / 4) * 8 + k % 4;
                    xmm1[k * 2] = _mm_unpacklo_epi32(xmm0[l], xmm0[l + 4]);
                    xmm1[k * 2 + 1] = _mm_unpackhi_epi32(xmm0[l], xmm0[l + 4]);
                }
                for k in 0..8 {
                    xmm0[k * 2] = _mm_unpacklo_epi64(xmm1[k], xmm1[k + 8]);
                    xmm0[k * 2 + 1] = _mm_unpackhi_epi64(xmm1[k], xmm1[k + 8]);
                }
                for k in 0..16 {
                    _mm_storeu_si128(
                        dest.as_mut_ptr().add(j + k * total_elements) as *mut __m128i,
                        xmm0[k],
                    );
                }
            }
        }
        _ => {
            if typesize > VEC {
                shuffle16_tiled_sse2(typesize, vectorizable_elements, total_elements, src, dest);
            } else {
                return shuffle_generic_inline_unchecked(typesize, 0, blocksize, src, dest);
            }
        }
    }

    if vectorizable_bytes < blocksize {
        shuffle_generic_inline_unchecked(typesize, vectorizable_bytes, blocksize, src, dest);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn unshuffle_sse2(typesize: usize, blocksize: usize, src: &[u8], dest: &mut [u8]) {
    use std::arch::x86_64::*;

    const VEC: usize = 16;
    let vectorized_chunk_size = typesize * VEC;
    if blocksize < vectorized_chunk_size {
        return unshuffle_generic_inline_unchecked(typesize, 0, blocksize, src, dest);
    }
    let vectorizable_bytes = blocksize - (blocksize % vectorized_chunk_size);
    let vectorizable_elements = vectorizable_bytes / typesize;
    let total_elements = blocksize / typesize;

    match typesize {
        2 => {
            for i in (0..vectorizable_elements).step_by(VEC) {
                let mut xmm0 = [_mm_setzero_si128(); 2];
                let mut xmm1 = [_mm_setzero_si128(); 2];
                for j in 0..2 {
                    xmm0[j] =
                        _mm_loadu_si128(src.as_ptr().add(i + j * total_elements) as *const __m128i);
                }
                xmm1[0] = _mm_unpacklo_epi8(xmm0[0], xmm0[1]);
                xmm1[1] = _mm_unpackhi_epi8(xmm0[0], xmm0[1]);
                _mm_storeu_si128(dest.as_mut_ptr().add(i * 2) as *mut __m128i, xmm1[0]);
                _mm_storeu_si128(dest.as_mut_ptr().add(i * 2 + VEC) as *mut __m128i, xmm1[1]);
            }
        }
        4 => {
            for i in (0..vectorizable_elements).step_by(VEC) {
                let mut xmm0 = [_mm_setzero_si128(); 4];
                let mut xmm1 = [_mm_setzero_si128(); 4];
                for j in 0..4 {
                    xmm0[j] =
                        _mm_loadu_si128(src.as_ptr().add(i + j * total_elements) as *const __m128i);
                }
                for j in 0..2 {
                    xmm1[j] = _mm_unpacklo_epi8(xmm0[j * 2], xmm0[j * 2 + 1]);
                    xmm1[2 + j] = _mm_unpackhi_epi8(xmm0[j * 2], xmm0[j * 2 + 1]);
                }
                for j in 0..2 {
                    xmm0[j] = _mm_unpacklo_epi16(xmm1[j * 2], xmm1[j * 2 + 1]);
                    xmm0[2 + j] = _mm_unpackhi_epi16(xmm1[j * 2], xmm1[j * 2 + 1]);
                }
                _mm_storeu_si128(dest.as_mut_ptr().add(i * 4) as *mut __m128i, xmm0[0]);
                _mm_storeu_si128(dest.as_mut_ptr().add(i * 4 + VEC) as *mut __m128i, xmm0[2]);
                _mm_storeu_si128(
                    dest.as_mut_ptr().add(i * 4 + 2 * VEC) as *mut __m128i,
                    xmm0[1],
                );
                _mm_storeu_si128(
                    dest.as_mut_ptr().add(i * 4 + 3 * VEC) as *mut __m128i,
                    xmm0[3],
                );
            }
        }
        8 => {
            for i in (0..vectorizable_elements).step_by(VEC) {
                let mut xmm0 = [_mm_setzero_si128(); 8];
                let mut xmm1 = [_mm_setzero_si128(); 8];
                for j in 0..8 {
                    xmm0[j] =
                        _mm_loadu_si128(src.as_ptr().add(i + j * total_elements) as *const __m128i);
                }
                for j in 0..4 {
                    xmm1[j] = _mm_unpacklo_epi8(xmm0[j * 2], xmm0[j * 2 + 1]);
                    xmm1[4 + j] = _mm_unpackhi_epi8(xmm0[j * 2], xmm0[j * 2 + 1]);
                }
                for j in 0..4 {
                    xmm0[j] = _mm_unpacklo_epi16(xmm1[j * 2], xmm1[j * 2 + 1]);
                    xmm0[4 + j] = _mm_unpackhi_epi16(xmm1[j * 2], xmm1[j * 2 + 1]);
                }
                for j in 0..4 {
                    xmm1[j] = _mm_unpacklo_epi32(xmm0[j * 2], xmm0[j * 2 + 1]);
                    xmm1[4 + j] = _mm_unpackhi_epi32(xmm0[j * 2], xmm0[j * 2 + 1]);
                }
                for (k, idx) in [0usize, 4, 2, 6, 1, 5, 3, 7].into_iter().enumerate() {
                    _mm_storeu_si128(
                        dest.as_mut_ptr().add(i * 8 + k * VEC) as *mut __m128i,
                        xmm1[idx],
                    );
                }
            }
        }
        16 => {
            for i in (0..vectorizable_elements).step_by(VEC) {
                let mut xmm1 = [_mm_setzero_si128(); 16];
                let mut xmm2 = [_mm_setzero_si128(); 16];
                for j in 0..16 {
                    xmm1[j] =
                        _mm_loadu_si128(src.as_ptr().add(i + j * total_elements) as *const __m128i);
                }
                for j in 0..8 {
                    xmm2[j] = _mm_unpacklo_epi8(xmm1[j * 2], xmm1[j * 2 + 1]);
                    xmm2[8 + j] = _mm_unpackhi_epi8(xmm1[j * 2], xmm1[j * 2 + 1]);
                }
                for j in 0..8 {
                    xmm1[j] = _mm_unpacklo_epi16(xmm2[j * 2], xmm2[j * 2 + 1]);
                    xmm1[8 + j] = _mm_unpackhi_epi16(xmm2[j * 2], xmm2[j * 2 + 1]);
                }
                for j in 0..8 {
                    xmm2[j] = _mm_unpacklo_epi32(xmm1[j * 2], xmm1[j * 2 + 1]);
                    xmm2[8 + j] = _mm_unpackhi_epi32(xmm1[j * 2], xmm1[j * 2 + 1]);
                }
                for j in 0..8 {
                    xmm1[j] = _mm_unpacklo_epi64(xmm2[j * 2], xmm2[j * 2 + 1]);
                    xmm1[8 + j] = _mm_unpackhi_epi64(xmm2[j * 2], xmm2[j * 2 + 1]);
                }
                for (k, idx) in [0usize, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
                    .into_iter()
                    .enumerate()
                {
                    _mm_storeu_si128(
                        dest.as_mut_ptr().add(i * 16 + k * VEC) as *mut __m128i,
                        xmm1[idx],
                    );
                }
            }
        }
        _ => {
            if typesize > VEC {
                unshuffle16_tiled_sse2(typesize, vectorizable_elements, total_elements, src, dest);
            } else {
                return unshuffle_generic_inline_unchecked(typesize, 0, blocksize, src, dest);
            }
        }
    }

    if vectorizable_bytes < blocksize {
        unshuffle_generic_inline_unchecked(typesize, vectorizable_bytes, blocksize, src, dest);
    }
}

/// Tiled shuffle for types larger than one vector: process the type in
/// 16-byte slabs, vectorizing 16 elements at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn shuffle16_tiled_sse2(
    typesize: usize,
    vectorizable_elements: usize,
    total_elements: usize,
    src: &[u8],
    dest: &mut [u8],
) {
    use std::arch::x86_64::*;

    const VEC: usize = 16;
    let vecs_per_el_rem = typesize % VEC;

    for j in (0..vectorizable_elements).step_by(VEC) {
        let mut offset_into_type = 0usize;
        while offset_into_type < typesize {
            let mut xmm0 = [_mm_setzero_si128(); 16];
            let mut xmm1 = [_mm_setzero_si128(); 16];
            let src_with_offset = src.as_ptr().add(offset_into_type);
            for k in 0..16 {
                xmm0[k] =
                    _mm_loadu_si128(src_with_offset.add((j + k) * typesize) as *const __m128i);
            }
            for (k, l) in (0..8).zip((0..16).step_by(2)) {
                xmm1[k * 2] = _mm_unpacklo_epi8(xmm0[l], xmm0[l + 1]);
                xmm1[k * 2 + 1] = _mm_unpackhi_epi8(xmm0[l], xmm0[l + 1]);
            }
            for k in 0..8 {
                let l = (k / 2) * 4 + k % 2;
                xmm0[k * 2] = _mm_unpacklo_epi16(xmm1[l], xmm1[l + 2]);
                xmm0[k * 2 + 1] = _mm_unpackhi_epi16(xmm1[l], xmm1[l + 2]);
            }
            for k in 0..8 {
                let l = (k / 4) * 8 + k % 4;
                xmm1[k * 2] = _mm_unpacklo_epi32(xmm0[l], xmm0[l + 4]);
                xmm1[k * 2 + 1] = _mm_unpackhi_epi32(xmm0[l], xmm0[l + 4]);
            }
            for k in 0..8 {
                xmm0[k * 2] = _mm_unpacklo_epi64(xmm1[k], xmm1[k + 8]);
                xmm0[k * 2 + 1] = _mm_unpackhi_epi64(xmm1[k], xmm1[k + 8]);
            }
            let dest_for_j = dest.as_mut_ptr().add(j);
            for k in 0..16 {
                _mm_storeu_si128(
                    dest_for_j.add(total_elements * (offset_into_type + k)) as *mut __m128i,
                    xmm0[k],
                );
            }
            offset_into_type += if offset_into_type == 0 && vecs_per_el_rem > 0 {
                vecs_per_el_rem
            } else {
                VEC
            };
        }
    }
}

/// Tiled unshuffle for types larger than one vector. The loops are inverted
/// compared to the shuffle variant to optimize cache utilization.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn unshuffle16_tiled_sse2(
    typesize: usize,
    vectorizable_elements: usize,
    total_elements: usize,
    src: &[u8],
    dest: &mut [u8],
) {
    use std::arch::x86_64::*;

    const VEC: usize = 16;
    let vecs_per_el_rem = typesize % VEC;

    let mut offset_into_type = 0usize;
    while offset_into_type < typesize {
        for i in (0..vectorizable_elements).step_by(VEC) {
            let mut xmm1 = [_mm_setzero_si128(); 16];
            let mut xmm2 = [_mm_setzero_si128(); 16];
            let src_for_i = src.as_ptr().add(i);
            for j in 0..16 {
                xmm1[j] = _mm_loadu_si128(
                    src_for_i.add(total_elements * (offset_into_type + j)) as *const __m128i
                );
            }
            for j in 0..8 {
                xmm2[j] = _mm_unpacklo_epi8(xmm1[j * 2], xmm1[j * 2 + 1]);
                xmm2[8 + j] = _mm_unpackhi_epi8(xmm1[j * 2], xmm1[j * 2 + 1]);
            }
            for j in 0..8 {
                xmm1[j] = _mm_unpacklo_epi16(xmm2[j * 2], xmm2[j * 2 + 1]);
                xmm1[8 + j] = _mm_unpackhi_epi16(xmm2[j * 2], xmm2[j * 2 + 1]);
            }
            for j in 0..8 {
                xmm2[j] = _mm_unpacklo_epi32(xmm1[j * 2], xmm1[j * 2 + 1]);
                xmm2[8 + j] = _mm_unpackhi_epi32(xmm1[j * 2], xmm1[j * 2 + 1]);
            }
            for j in 0..8 {
                xmm1[j] = _mm_unpacklo_epi64(xmm2[j * 2], xmm2[j * 2 + 1]);
                xmm1[8 + j] = _mm_unpackhi_epi64(xmm2[j * 2], xmm2[j * 2 + 1]);
            }
            let dest_with_offset = dest.as_mut_ptr().add(offset_into_type);
            for (k, idx) in [0usize, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
                .into_iter()
                .enumerate()
            {
                _mm_storeu_si128(
                    dest_with_offset.add((i + k) * typesize) as *mut __m128i,
                    xmm1[idx],
                );
            }
        }
        offset_into_type += if offset_into_type == 0 && vecs_per_el_rem > 0 {
            vecs_per_el_rem
        } else {
            VEC
        };
    }
}

// AVX2 kernels.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn shuffle_avx2(typesize: usize, blocksize: usize, src: &[u8], dest: &mut [u8]) {
    use std::arch::x86_64::*;

    const VEC: usize = 32;
    let vectorized_chunk_size = typesize * VEC;
    if blocksize < vectorized_chunk_size {
        return shuffle_sse2(typesize, blocksize, src, dest);
    }
    let vectorizable_bytes = blocksize - (blocksize % vectorized_chunk_size);
    let vectorizable_elements = vectorizable_bytes / typesize;
    let total_elements = blocksize / typesize;

    match typesize {
        2 => {
            // `_mm256_set_epi8` arguments are ordered most-significant first.
            let shmask = _mm256_set_epi8(
                0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04,
                0x02, 0x00, 0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01, 0x0e, 0x0c, 0x0a, 0x08,
                0x06, 0x04, 0x02, 0x00,
            );
            for j in (0..vectorizable_elements).step_by(VEC) {
                let mut ymm0 = [_mm256_setzero_si256(); 2];
                let mut ymm1 = [_mm256_setzero_si256(); 2];
                for k in 0..2 {
                    ymm0[k] =
                        _mm256_loadu_si256(src.as_ptr().add(j * 2 + k * VEC) as *const __m256i);
                    ymm1[k] = _mm256_shuffle_epi8(ymm0[k], shmask);
                }
                ymm0[0] = _mm256_permute4x64_epi64(ymm1[0], 0xd8);
                ymm0[1] = _mm256_permute4x64_epi64(ymm1[1], 0x8d);
                ymm1[0] = _mm256_blend_epi32(ymm0[0], ymm0[1], 0xf0);
                ymm0[1] = _mm256_blend_epi32(ymm0[0], ymm0[1], 0x0f);
                ymm1[1] = _mm256_permute4x64_epi64(ymm0[1], 0x4e);
                for k in 0..2 {
                    _mm256_storeu_si256(
                        dest.as_mut_ptr().add(j + k * total_elements) as *mut __m256i,
                        ymm1[k],
                    );
                }
            }
        }
        4 => {
            let mask = _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
            for i in (0..vectorizable_elements).step_by(VEC) {
                let mut ymm0 = [_mm256_setzero_si256(); 4];
                let mut ymm1 = [_mm256_setzero_si256(); 4];
                for j in 0..4 {
                    ymm0[j] =
                        _mm256_loadu_si256(src.as_ptr().add(i * 4 + j * VEC) as *const __m256i);
                    ymm1[j] = _mm256_shuffle_epi32(ymm0[j], 0xd8);
                    ymm0[j] = _mm256_shuffle_epi32(ymm0[j], 0x8d);
                    ymm0[j] = _mm256_unpacklo_epi8(ymm1[j], ymm0[j]);
                    ymm1[j] = _mm256_shuffle_epi32(ymm0[j], 0x4e);
                    ymm0[j] = _mm256_unpacklo_epi16(ymm0[j], ymm1[j]);
                }
                for j in 0..2 {
                    ymm1[j * 2] = _mm256_unpacklo_epi32(ymm0[j * 2], ymm0[j * 2 + 1]);
                    ymm1[j * 2 + 1] = _mm256_unpackhi_epi32(ymm0[j * 2], ymm0[j * 2 + 1]);
                }
                for j in 0..2 {
                    ymm0[j * 2] = _mm256_unpacklo_epi64(ymm1[j], ymm1[j + 2]);
                    ymm0[j * 2 + 1] = _mm256_unpackhi_epi64(ymm1[j], ymm1[j + 2]);
                }
                for j in 0..4 {
                    ymm0[j] = _mm256_permutevar8x32_epi32(ymm0[j], mask);
                }
                for j in 0..4 {
                    _mm256_storeu_si256(
                        dest.as_mut_ptr().add(i + j * total_elements) as *mut __m256i,
                        ymm0[j],
                    );
                }
            }
        }
        8 => {
            for j in (0..vectorizable_elements).step_by(VEC) {
                let mut ymm0 = [_mm256_setzero_si256(); 8];
                let mut ymm1 = [_mm256_setzero_si256(); 8];
                for k in 0..8 {
                    ymm0[k] =
                        _mm256_loadu_si256(src.as_ptr().add(j * 8 + k * VEC) as *const __m256i);
                    ymm1[k] = _mm256_shuffle_epi32(ymm0[k], 0x4e);
                    ymm1[k] = _mm256_unpacklo_epi8(ymm0[k], ymm1[k]);
                }
                for (k, l) in (0..4).zip((0..8).step_by(2)) {
                    ymm0[k * 2] = _mm256_unpacklo_epi16(ymm1[l], ymm1[l + 1]);
                    ymm0[k * 2 + 1] = _mm256_unpackhi_epi16(ymm1[l], ymm1[l + 1]);
                }
                for k in 0..4 {
                    let l = if k < 2 { k } else { k + 2 };
                    ymm1[k * 2] = _mm256_unpacklo_epi32(ymm0[l], ymm0[l + 2]);
                    ymm1[k * 2 + 1] = _mm256_unpackhi_epi32(ymm0[l], ymm0[l + 2]);
                }
                for k in 0..4 {
                    ymm0[k * 2] = _mm256_unpacklo_epi64(ymm1[k], ymm1[k + 4]);
                    ymm0[k * 2 + 1] = _mm256_unpackhi_epi64(ymm1[k], ymm1[k + 4]);
                }
                for k in 0..8 {
                    ymm1[k] = _mm256_permute4x64_epi64(ymm0[k], 0x72);
                    ymm0[k] = _mm256_permute4x64_epi64(ymm0[k], 0xd8);
                    ymm0[k] = _mm256_unpacklo_epi16(ymm0[k], ymm1[k]);
                }
                for k in 0..8 {
                    _mm256_storeu_si256(
                        dest.as_mut_ptr().add(j + k * total_elements) as *mut __m256i,
                        ymm0[k],
                    );
                }
            }
        }
        16 => {
            let shmask = _mm256_set_epi8(
                0x0f, 0x07, 0x0e, 0x06, 0x0d, 0x05, 0x0c, 0x04, 0x0b, 0x03, 0x0a, 0x02, 0x09, 0x01,
                0x08, 0x00, 0x0f, 0x07, 0x0e, 0x06, 0x0d, 0x05, 0x0c, 0x04, 0x0b, 0x03, 0x0a, 0x02,
                0x09, 0x01, 0x08, 0x00,
            );
            for j in (0..vectorizable_elements).step_by(VEC) {
                let mut ymm0 = [_mm256_setzero_si256(); 16];
                let mut ymm1 = [_mm256_setzero_si256(); 16];
                for k in 0..16 {
                    ymm0[k] =
                        _mm256_loadu_si256(src.as_ptr().add(j * 16 + k * VEC) as *const __m256i);
                }
                for (k, l) in (0..8).zip((0..16).step_by(2)) {
                    ymm1[k * 2] = _mm256_unpacklo_epi8(ymm0[l], ymm0[l + 1]);
                    ymm1[k * 2 + 1] = _mm256_unpackhi_epi8(ymm0[l], ymm0[l + 1]);
                }
                for k in 0..8 {
                    let l = (k / 2) * 4 + k % 2;
                    ymm0[k * 2] = _mm256_unpacklo_epi16(ymm1[l], ymm1[l + 2]);
                    ymm0[k * 2 + 1] = _mm256_unpackhi_epi16(ymm1[l], ymm1[l + 2]);
                }
                for k in 0..8 {
                    let l = (k / 4) * 8 + k % 4;
                    ymm1[k * 2] = _mm256_unpacklo_epi32(ymm0[l], ymm0[l + 4]);
                    ymm1[k * 2 + 1] = _mm256_unpackhi_epi32(ymm0[l], ymm0[l + 4]);
                }
                for k in 0..8 {
                    ymm0[k * 2] = _mm256_unpacklo_epi64(ymm1[k], ymm1[k + 8]);
                    ymm0[k * 2 + 1] = _mm256_unpackhi_epi64(ymm1[k], ymm1[k + 8]);
                }
                for k in 0..16 {
                    ymm0[k] = _mm256_permute4x64_epi64(ymm0[k], 0xd8);
                    ymm0[k] = _mm256_shuffle_epi8(ymm0[k], shmask);
                }
                for k in 0..16 {
                    _mm256_storeu_si256(
                        dest.as_mut_ptr().add(j + k * total_elements) as *mut __m256i,
                        ymm0[k],
                    );
                }
            }
        }
        _ => {
            if typesize > 16 {
                shuffle16_tiled_avx2(typesize, vectorizable_elements, total_elements, src, dest);
            } else {
                return shuffle_generic_inline_unchecked(typesize, 0, blocksize, src, dest);
            }
        }
    }

    if vectorizable_bytes < blocksize {
        shuffle_generic_inline_unchecked(typesize, vectorizable_bytes, blocksize, src, dest);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unshuffle_avx2(typesize: usize, blocksize: usize, src: &[u8], dest: &mut [u8]) {
    use std::arch::x86_64::*;

    const VEC: usize = 32;
    let vectorized_chunk_size = typesize * VEC;
    if blocksize < vectorized_chunk_size {
        return unshuffle_sse2(typesize, blocksize, src, dest);
    }
    let vectorizable_bytes = blocksize - (blocksize % vectorized_chunk_size);
    let vectorizable_elements = vectorizable_bytes / typesize;
    let total_elements = blocksize / typesize;

    match typesize {
        2 => {
            for i in (0..vectorizable_elements).step_by(VEC) {
                let mut ymm0 = [_mm256_setzero_si256(); 2];
                let mut ymm1 = [_mm256_setzero_si256(); 2];
                for j in 0..2 {
                    ymm0[j] = _mm256_loadu_si256(
                        src.as_ptr().add(i + j * total_elements) as *const __m256i
                    );
                }
                for j in 0..2 {
                    ymm0[j] = _mm256_permute4x64_epi64(ymm0[j], 0xd8);
                }
                ymm1[0] = _mm256_unpacklo_epi8(ymm0[0], ymm0[1]);
                ymm1[1] = _mm256_unpackhi_epi8(ymm0[0], ymm0[1]);
                _mm256_storeu_si256(dest.as_mut_ptr().add(i * 2) as *mut __m256i, ymm1[0]);
                _mm256_storeu_si256(dest.as_mut_ptr().add(i * 2 + VEC) as *mut __m256i, ymm1[1]);
            }
        }
        4 => {
            for i in (0..vectorizable_elements).step_by(VEC) {
                let mut ymm0 = [_mm256_setzero_si256(); 4];
                let mut ymm1 = [_mm256_setzero_si256(); 4];
                for j in 0..4 {
                    ymm0[j] = _mm256_loadu_si256(
                        src.as_ptr().add(i + j * total_elements) as *const __m256i
                    );
                }
                for j in 0..2 {
                    ymm1[j] = _mm256_unpacklo_epi8(ymm0[j * 2], ymm0[j * 2 + 1]);
                    ymm1[2 + j] = _mm256_unpackhi_epi8(ymm0[j * 2], ymm0[j * 2 + 1]);
                }
                for j in 0..2 {
                    ymm0[j] = _mm256_unpacklo_epi16(ymm1[j * 2], ymm1[j * 2 + 1]);
                    ymm0[2 + j] = _mm256_unpackhi_epi16(ymm1[j * 2], ymm1[j * 2 + 1]);
                }
                ymm1[0] = _mm256_permute2x128_si256(ymm0[0], ymm0[2], 0x20);
                ymm1[1] = _mm256_permute2x128_si256(ymm0[1], ymm0[3], 0x20);
                ymm1[2] = _mm256_permute2x128_si256(ymm0[0], ymm0[2], 0x31);
                ymm1[3] = _mm256_permute2x128_si256(ymm0[1], ymm0[3], 0x31);
                for j in 0..4 {
                    _mm256_storeu_si256(
                        dest.as_mut_ptr().add(i * 4 + j * VEC) as *mut __m256i,
                        ymm1[j],
                    );
                }
            }
        }
        8 => {
            for i in (0..vectorizable_elements).step_by(VEC) {
                let mut ymm0 = [_mm256_setzero_si256(); 8];
                let mut ymm1 = [_mm256_setzero_si256(); 8];
                for j in 0..8 {
                    ymm0[j] = _mm256_loadu_si256(
                        src.as_ptr().add(i + j * total_elements) as *const __m256i
                    );
                }
                for j in 0..4 {
                    ymm1[j] = _mm256_unpacklo_epi8(ymm0[j * 2], ymm0[j * 2 + 1]);
                    ymm1[4 + j] = _mm256_unpackhi_epi8(ymm0[j * 2], ymm0[j * 2 + 1]);
                }
                for j in 0..4 {
                    ymm0[j] = _mm256_unpacklo_epi16(ymm1[j * 2], ymm1[j * 2 + 1]);
                    ymm0[4 + j] = _mm256_unpackhi_epi16(ymm1[j * 2], ymm1[j * 2 + 1]);
                }
                for j in 0..8 {
                    ymm0[j] = _mm256_permute4x64_epi64(ymm0[j], 0xd8);
                }
                for j in 0..4 {
                    ymm1[j] = _mm256_unpacklo_epi32(ymm0[j * 2], ymm0[j * 2 + 1]);
                    ymm1[4 + j] = _mm256_unpackhi_epi32(ymm0[j * 2], ymm0[j * 2 + 1]);
                }
                for (k, idx) in [0usize, 2, 1, 3, 4, 6, 5, 7].into_iter().enumerate() {
                    _mm256_storeu_si256(
                        dest.as_mut_ptr().add(i * 8 + k * VEC) as *mut __m256i,
                        ymm1[idx],
                    );
                }
            }
        }
        16 => {
            for i in (0..vectorizable_elements).step_by(VEC) {
                let mut ymm0 = [_mm256_setzero_si256(); 16];
                let mut ymm1 = [_mm256_setzero_si256(); 16];
                for j in 0..16 {
                    ymm0[j] = _mm256_loadu_si256(
                        src.as_ptr().add(i + j * total_elements) as *const __m256i
                    );
                }
                for j in 0..8 {
                    ymm1[j] = _mm256_unpacklo_epi8(ymm0[j * 2], ymm0[j * 2 + 1]);
                    ymm1[8 + j] = _mm256_unpackhi_epi8(ymm0[j * 2], ymm0[j * 2 + 1]);
                }
                for j in 0..8 {
                    ymm0[j] = _mm256_unpacklo_epi16(ymm1[j * 2], ymm1[j * 2 + 1]);
                    ymm0[8 + j] = _mm256_unpackhi_epi16(ymm1[j * 2], ymm1[j * 2 + 1]);
                }
                for j in 0..8 {
                    ymm1[j] = _mm256_unpacklo_epi32(ymm0[j * 2], ymm0[j * 2 + 1]);
                    ymm1[8 + j] = _mm256_unpackhi_epi32(ymm0[j * 2], ymm0[j * 2 + 1]);
                }
                for j in 0..8 {
                    ymm0[j] = _mm256_unpacklo_epi64(ymm1[j * 2], ymm1[j * 2 + 1]);
                    ymm0[8 + j] = _mm256_unpackhi_epi64(ymm1[j * 2], ymm1[j * 2 + 1]);
                }
                for j in 0..8 {
                    ymm1[j] = _mm256_permute2x128_si256(ymm0[j], ymm0[j + 8], 0x20);
                    ymm1[j + 8] = _mm256_permute2x128_si256(ymm0[j], ymm0[j + 8], 0x31);
                }
                for (k, idx) in [0usize, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15]
                    .into_iter()
                    .enumerate()
                {
                    _mm256_storeu_si256(
                        dest.as_mut_ptr().add(i * 16 + k * VEC) as *mut __m256i,
                        ymm1[idx],
                    );
                }
            }
        }
        _ => {
            if typesize > 16 {
                unshuffle16_tiled_avx2(typesize, vectorizable_elements, total_elements, src, dest);
            } else {
                return unshuffle_generic_inline_unchecked(typesize, 0, blocksize, src, dest);
            }
        }
    }

    if vectorizable_bytes < blocksize {
        unshuffle_generic_inline_unchecked(typesize, vectorizable_bytes, blocksize, src, dest);
    }
}

/// Tiled AVX2 shuffle for types larger than 16 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn shuffle16_tiled_avx2(
    typesize: usize,
    vectorizable_elements: usize,
    total_elements: usize,
    src: &[u8],
    dest: &mut [u8],
) {
    use std::arch::x86_64::*;

    const VEC: usize = 32;
    let vecs_per_el_rem = typesize % 16;

    let shmask = _mm256_set_epi8(
        0x0f, 0x07, 0x0e, 0x06, 0x0d, 0x05, 0x0c, 0x04, 0x0b, 0x03, 0x0a, 0x02, 0x09, 0x01, 0x08,
        0x00, 0x0f, 0x07, 0x0e, 0x06, 0x0d, 0x05, 0x0c, 0x04, 0x0b, 0x03, 0x0a, 0x02, 0x09, 0x01,
        0x08, 0x00,
    );

    for j in (0..vectorizable_elements).step_by(VEC) {
        let mut offset_into_type = 0usize;
        while offset_into_type < typesize {
            let mut ymm0 = [_mm256_setzero_si256(); 16];
            let mut ymm1 = [_mm256_setzero_si256(); 16];
            let src_with_offset = src.as_ptr().add(offset_into_type);
            for k in 0..16 {
                ymm0[k] = _mm256_loadu2_m128i(
                    src_with_offset.add((j + 2 * k + 1) * typesize) as *const __m128i,
                    src_with_offset.add((j + 2 * k) * typesize) as *const __m128i,
                );
            }
            for (k, l) in (0..8).zip((0..16).step_by(2)) {
                ymm1[k * 2] = _mm256_unpacklo_epi8(ymm0[l], ymm0[l + 1]);
                ymm1[k * 2 + 1] = _mm256_unpackhi_epi8(ymm0[l], ymm0[l + 1]);
            }
            for k in 0..8 {
                let l = (k / 2) * 4 + k % 2;
                ymm0[k * 2] = _mm256_unpacklo_epi16(ymm1[l], ymm1[l + 2]);
                ymm0[k * 2 + 1] = _mm256_unpackhi_epi16(ymm1[l], ymm1[l + 2]);
            }
            for k in 0..8 {
                let l = (k / 4) * 8 + k % 4;
                ymm1[k * 2] = _mm256_unpacklo_epi32(ymm0[l], ymm0[l + 4]);
                ymm1[k * 2 + 1] = _mm256_unpackhi_epi32(ymm0[l], ymm0[l + 4]);
            }
            for k in 0..8 {
                ymm0[k * 2] = _mm256_unpacklo_epi64(ymm1[k], ymm1[k + 8]);
                ymm0[k * 2 + 1] = _mm256_unpackhi_epi64(ymm1[k], ymm1[k + 8]);
            }
            for k in 0..16 {
                ymm0[k] = _mm256_permute4x64_epi64(ymm0[k], 0xd8);
                ymm0[k] = _mm256_shuffle_epi8(ymm0[k], shmask);
            }
            let dest_for_j = dest.as_mut_ptr().add(j);
            for k in 0..16 {
                _mm256_storeu_si256(
                    dest_for_j.add(total_elements * (offset_into_type + k)) as *mut __m256i,
                    ymm0[k],
                );
            }
            offset_into_type += if offset_into_type == 0 && vecs_per_el_rem > 0 {
                vecs_per_el_rem
            } else {
                16
            };
        }
    }
}

/// Tiled AVX2 unshuffle for types larger than 16 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unshuffle16_tiled_avx2(
    typesize: usize,
    vectorizable_elements: usize,
    total_elements: usize,
    src: &[u8],
    dest: &mut [u8],
) {
    use std::arch::x86_64::*;

    const VEC: usize = 32;
    let vecs_per_el_rem = typesize % 16;

    let mut offset_into_type = 0usize;
    while offset_into_type < typesize {
        for i in (0..vectorizable_elements).step_by(VEC) {
            let mut ymm0 = [_mm256_setzero_si256(); 16];
            let mut ymm1 = [_mm256_setzero_si256(); 16];
            let src_for_i = src.as_ptr().add(i);
            for j in 0..16 {
                ymm0[j] = _mm256_loadu_si256(
                    src_for_i.add(total_elements * (offset_into_type + j)) as *const __m256i,
                );
            }
            for j in 0..8 {
                ymm1[j] = _mm256_unpacklo_epi8(ymm0[j * 2], ymm0[j * 2 + 1]);
                ymm1[8 + j] = _mm256_unpackhi_epi8(ymm0[j * 2], ymm0[j * 2 + 1]);
            }
            for j in 0..8 {
                ymm0[j] = _mm256_unpacklo_epi16(ymm1[j * 2], ymm1[j * 2 + 1]);
                ymm0[8 + j] = _mm256_unpackhi_epi16(ymm1[j * 2], ymm1[j * 2 + 1]);
            }
            for j in 0..8 {
                ymm1[j] = _mm256_unpacklo_epi32(ymm0[j * 2], ymm0[j * 2 + 1]);
                ymm1[8 + j] = _mm256_unpackhi_epi32(ymm0[j * 2], ymm0[j * 2 + 1]);
            }
            for j in 0..8 {
                ymm0[j] = _mm256_unpacklo_epi64(ymm1[j * 2], ymm1[j * 2 + 1]);
                ymm0[8 + j] = _mm256_unpackhi_epi64(ymm1[j * 2], ymm1[j * 2 + 1]);
            }
            for j in 0..8 {
                ymm1[j] = _mm256_permute2x128_si256(ymm0[j], ymm0[j + 8], 0x20);
                ymm1[j + 8] = _mm256_permute2x128_si256(ymm0[j], ymm0[j + 8], 0x31);
            }
            let dest_with_offset = dest.as_mut_ptr().add(offset_into_type);
            // Each ymm1[j] holds two adjacent 16-byte elements (lo, hi).
            for (k, idx) in [0usize, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15]
                .into_iter()
                .enumerate()
            {
                _mm256_storeu2_m128i(
                    dest_with_offset.add((i + 2 * k + 1) * typesize) as *mut __m128i,
                    dest_with_offset.add((i + 2 * k) * typesize) as *mut __m128i,
                    ymm1[idx],
                );
            }
        }
        offset_into_type += if offset_into_type == 0 && vecs_per_el_rem > 0 {
            vecs_per_el_rem
        } else {
            16
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference shuffle: dest[j * nelem + i] = src[i * typesize + j].
    fn check_layout(typesize: usize, blocksize: usize, src: &[u8], dest: &[u8]) {
        let nelem = blocksize / typesize;
        for j in 0..typesize {
            for i in 0..nelem {
                assert_eq!(
                    dest[j * nelem + i],
                    src[i * typesize + j],
                    "typesize={typesize} blocksize={blocksize} j={j} i={i}"
                );
            }
        }
        // Leftover bytes are copied verbatim.
        let rem = blocksize % typesize;
        if rem > 0 {
            assert_eq!(
                &dest[blocksize - rem..blocksize],
                &src[blocksize - rem..blocksize]
            );
        }
    }

    fn patterned(blocksize: usize) -> Vec<u8> {
        (0..blocksize)
            .map(|i| ((i * 31 + i / 7) & 0xFF) as u8)
            .collect()
    }

    #[test]
    fn generic_matches_reference_layout() {
        for typesize in [1usize, 2, 3, 4, 5, 7, 8, 12, 16, 17, 24, 32, 48, 64] {
            for blocksize in [
                typesize,
                typesize + 1,
                typesize * 3,
                typesize * 16,
                typesize * 16 + 5,
                typesize * 33 + 7,
                4096,
            ] {
                let src = patterned(blocksize);
                let mut dest = vec![0u8; blocksize];
                let mut back = vec![0u8; blocksize];
                shuffle_generic(typesize, blocksize, &src, &mut dest);
                check_layout(typesize, blocksize, &src, &dest);
                unshuffle_generic(typesize, blocksize, &dest, &mut back);
                assert_eq!(back, src);
            }
        }
    }

    #[test]
    fn roundtrip_all_typesizes() {
        for typesize in [1usize, 2, 3, 4, 5, 7, 8, 12, 16, 17, 24, 32, 48, 64] {
            for blocksize in [typesize, typesize + 1, 100, 1000, 4096, 100_000] {
                let src = patterned(blocksize);
                let mut shuffled = vec![0u8; blocksize];
                let mut back = vec![0u8; blocksize];
                // SAFETY: all buffers have exactly `blocksize` bytes,
                // `typesize` is non-zero, and distinct Vecs cannot overlap.
                unsafe {
                    shuffle_unchecked(typesize, blocksize, &src, &mut shuffled);
                    unshuffle_unchecked(typesize, blocksize, &shuffled, &mut back);
                }
                assert_eq!(back, src, "typesize={typesize} blocksize={blocksize}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_matches_generic() {
        for typesize in [1usize, 2, 3, 4, 5, 7, 8, 12, 16, 17, 24, 32, 48, 64] {
            for blocksize in [typesize, typesize + 1, 100, 1000, 4096, 100_000] {
                let src = patterned(blocksize);
                let mut generic = vec![0u8; blocksize];
                let mut sse2 = vec![0u8; blocksize];
                let mut avx2 = vec![0u8; blocksize];
                let mut back = vec![0u8; blocksize];
                shuffle_generic(typesize, blocksize, &src, &mut generic);
                // SAFETY: the test allocates distinct `blocksize`-byte buffers
                // and only calls kernels whose CPU feature was detected.
                unsafe {
                    if std::arch::is_x86_feature_detected!("sse2") {
                        shuffle_sse2(typesize, blocksize, &src, &mut sse2);
                        unshuffle_sse2(typesize, blocksize, &sse2, &mut back);
                        assert_eq!(
                            sse2, generic,
                            "sse2 shuffle typesize={typesize} blocksize={blocksize}"
                        );
                        assert_eq!(
                            back, src,
                            "sse2 roundtrip typesize={typesize} blocksize={blocksize}"
                        );
                    }
                    if std::arch::is_x86_feature_detected!("avx2") {
                        shuffle_avx2(typesize, blocksize, &src, &mut avx2);
                        unshuffle_avx2(typesize, blocksize, &avx2, &mut back);
                        assert_eq!(
                            avx2, generic,
                            "avx2 shuffle typesize={typesize} blocksize={blocksize}"
                        );
                        assert_eq!(
                            back, src,
                            "avx2 roundtrip typesize={typesize} blocksize={blocksize}"
                        );
                    }
                }
            }
        }
    }
}
