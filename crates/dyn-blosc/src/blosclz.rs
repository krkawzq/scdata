//! BloscLZ-compatible FastLZ-derived LZ77 compression.
//!
//! The wire format is byte-for-byte compatible with c-blosc's `blosclz.c`
//! (version 2.5.1): a control byte stream where values 0-31 encode a literal
//! run of 1-32 bytes and values 32-255 encode a back-reference match.

use crate::error::{Error, Result};

const MAX_COPY: u32 = 32;
const MAX_DISTANCE: u32 = 8191;
const MAX_FARDISTANCE: u32 = 65535 + MAX_DISTANCE - 1;
const HASH_LOG2: u32 = 12;

const CRATIO_THRESHOLD: [f64; 10] = [0.0, 2.0, 1.5, 1.2, 1.2, 1.2, 1.2, 1.15, 1.1, 1.0];
const HASHLOG_TABLE: [u32; 10] = [0, 12, 13, 14, 14, 14, 14, 14, 14, 14];

#[derive(Debug, Default)]
pub(crate) struct Workspace {
    probe: Vec<u16>,
    hash: Vec<u32>,
}

impl Workspace {
    pub(crate) fn prepare(&mut self, level: usize) -> Result<()> {
        let hash_log = HASHLOG_TABLE.get(level).copied().ok_or_else(|| {
            Error::InvalidArgument(format!("BloscLZ level {level} is outside 0..=9"))
        })?;
        resize_table(&mut self.probe, 1usize << HASH_LOG2)?;
        resize_table(&mut self.hash, 1usize << hash_log)?;
        self.probe.fill(0);
        self.hash.fill(0);
        Ok(())
    }
}

fn resize_table<T: Copy + Default>(table: &mut Vec<T>, len: usize) -> Result<()> {
    if table.len() < len {
        let additional = len - table.len();
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::AllocationFailed { bytes: usize::MAX })?;
        table
            .try_reserve_exact(additional)
            .map_err(|_| Error::AllocationFailed { bytes })?;
        table.resize(len, T::default());
    }
    Ok(())
}

#[inline(always)]
fn hash_function(v: u32, h: u32) -> usize {
    // A zero-bit hash addresses the only table slot without shifting by the
    // integer width.
    if h == 0 {
        return 0;
    }
    (v.wrapping_mul(2654435761) >> (32 - h)) as usize
}

#[inline(always)]
/// # Safety
///
/// `off..off + 4` must be in bounds for `buf`.
unsafe fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([
        *buf.get_unchecked(off),
        *buf.get_unchecked(off + 1),
        *buf.get_unchecked(off + 2),
        *buf.get_unchecked(off + 3),
    ])
}

/// Estimate the compression ratio of `ibase[..maxlen]` using a cheap probe.
/// Returns input_bytes / output_bytes (>= 1 means compressible).
///
/// # Safety
///
/// `maxlen <= ibase.len()`, and the probe parameters must leave four readable
/// bytes at every candidate and reference position.
unsafe fn get_cratio(
    ibase: &[u8],
    maxlen: usize,
    minlen: usize,
    ipshift: usize,
    htab: &mut [u16],
) -> f64 {
    let hashlen = 1usize << HASH_LOG2;
    debug_assert!(htab.len() >= hashlen);
    // Bound probe cost independently of the input length.
    let limit = maxlen.min(hashlen);
    // A probe shorter than 12 bytes has no complete match candidate.
    let ip_bound = limit.saturating_sub(1);
    let ip_limit = limit.saturating_sub(12);

    let mut ip = 0usize;
    let mut oc: i64 = 0;
    let mut copy: u32 = 4;
    // We start with a literal copy: marker byte + 4 bytes.
    oc += 5;

    while ip < ip_limit {
        let anchor = ip;

        let seq = read_u32(ibase, ip);
        let hval = hash_function(seq, HASH_LOG2);
        let mut ref_ = htab[hval] as usize;
        let mut distance = anchor - ref_;
        htab[hval] = anchor as u16;

        if distance == 0 || distance >= MAX_FARDISTANCE as usize {
            // literal
            oc += 1;
            ip = anchor + 1;
            copy += 1;
            if copy == MAX_COPY {
                copy = 0;
                oc += 1;
            }
            continue;
        }

        if read_u32(ibase, ref_) == seq {
            ref_ += 4;
        } else {
            // literal
            oc += 1;
            ip = anchor + 1;
            copy += 1;
            if copy == MAX_COPY {
                copy = 0;
                oc += 1;
            }
            continue;
        }

        ip = anchor + 4;
        distance -= 1;

        // zero biased distance means a run
        ip = if distance == 0 {
            get_run(ibase, ip, ip_bound, ref_)
        } else {
            get_match(ibase, ip, ip_bound, ref_)
        };

        ip -= ipshift;
        let len = ip - anchor;
        if len < minlen {
            // literal
            oc += 1;
            ip = anchor + 1;
            copy += 1;
            if copy == MAX_COPY {
                copy = 0;
                oc += 1;
            }
            continue;
        }

        if copy == 0 {
            oc -= 1;
        }
        copy = 0;

        if distance < MAX_DISTANCE as usize {
            if len >= 7 {
                oc += ((len - 7) / 255 + 1) as i64;
            }
            oc += 2;
        } else {
            if len >= 7 {
                oc += ((len - 7) / 255 + 1) as i64;
            }
            oc += 4;
        }

        // Update the hash at the match boundary.
        let seq = read_u32(ibase, ip);
        let hval = hash_function(seq, HASH_LOG2);
        htab[hval] = ip as u16;
        ip += 2;
        // Assuming literal copy.
        oc += 1;
    }

    let ic = ip as f64;
    ic / oc as f64
}

/// Return the index just past the run of `buf[ip-1]` starting at `ref`.
///
/// # Safety
///
/// `0 < ip <= ip_bound < buf.len()`, `ref_ < ip`, and the matching region from
/// `ref_` through `ip_bound` must lie in `buf`.
#[inline]
unsafe fn get_run(buf: &[u8], mut ip: usize, ip_bound: usize, mut ref_: usize) -> usize {
    let x = *buf.get_unchecked(ip - 1);
    while ip + 8 < ip_bound {
        let mut same = true;
        for k in 0..8 {
            if *buf.get_unchecked(ref_ + k) != x {
                same = false;
                break;
            }
        }
        if !same {
            while *buf.get_unchecked(ref_) == x {
                ip += 1;
                ref_ += 1;
            }
            return ip;
        }
        ip += 8;
        ref_ += 8;
    }
    while ip < ip_bound && *buf.get_unchecked(ref_) == x {
        ip += 1;
        ref_ += 1;
    }
    ip
}

/// Return the index just past the match between `ip` and `ref`.
///
/// # Safety
///
/// `ip <= ip_bound < buf.len()`, `ref_ < ip`, and the compared regions through
/// `ip_bound` must lie in `buf`.
#[cfg(not(target_feature = "sse2"))]
#[inline]
unsafe fn get_match(buf: &[u8], mut ip: usize, ip_bound: usize, mut ref_: usize) -> usize {
    while ip + 8 < ip_bound {
        let mut same = true;
        for k in 0..8 {
            if *buf.get_unchecked(ref_ + k) != *buf.get_unchecked(ip + k) {
                same = false;
                break;
            }
        }
        if !same {
            while *buf.get_unchecked(ref_) == *buf.get_unchecked(ip) {
                ip += 1;
                ref_ += 1;
            }
            return ip + 1;
        }
        ip += 8;
        ref_ += 8;
    }
    while ip < ip_bound && *buf.get_unchecked(ref_) == *buf.get_unchecked(ip) {
        ip += 1;
        ref_ += 1;
    }
    if ip < ip_bound {
        ip += 1;
    }
    ip
}

/// SSE2 16-byte match finder.
///
/// # Safety
///
/// The scalar match-finder invariants apply, and the target must support SSE2.
#[cfg(target_feature = "sse2")]
#[inline]
unsafe fn get_match(buf: &[u8], mut ip: usize, ip_bound: usize, mut ref_: usize) -> usize {
    use std::arch::x86_64::*;
    while ip + 16 < ip_bound {
        let a = _mm_loadu_si128(buf.as_ptr().add(ip) as *const __m128i);
        let b = _mm_loadu_si128(buf.as_ptr().add(ref_) as *const __m128i);
        let cmp = _mm_cmpeq_epi32(a, b);
        if _mm_movemask_epi8(cmp) != 0xFFFF {
            while *buf.get_unchecked(ref_) == *buf.get_unchecked(ip) {
                ip += 1;
                ref_ += 1;
            }
            return ip + 1;
        }
        ip += 16;
        ref_ += 16;
    }
    while ip < ip_bound && *buf.get_unchecked(ref_) == *buf.get_unchecked(ip) {
        ip += 1;
        ref_ += 1;
    }
    if ip < ip_bound {
        ip += 1;
    }
    ip
}

/// Compress `input` into `output`.
///
/// Returns the compressed size, or 0 when the data is incompressible or the
/// output buffer is too small (callers then store the block raw).
#[cfg(test)]
pub(crate) fn blosclz_compress(
    input: &[u8],
    output: &mut [u8],
    clevel: i32,
    split_block: bool,
) -> usize {
    let mut workspace = Workspace::default();
    let clevel = clevel.clamp(0, 9) as usize;
    if workspace.prepare(clevel).is_err() {
        return 0;
    }
    blosclz_compress_with_workspace(input, output, clevel, split_block, &mut workspace)
}

pub(crate) fn blosclz_compress_with_workspace(
    input: &[u8],
    output: &mut [u8],
    clevel: usize,
    split_block: bool,
    workspace: &mut Workspace,
) -> usize {
    debug_assert!(clevel <= 9);
    let clevel = clevel.clamp(0, 9);
    let length = input.len();
    let maxout = output.len();

    // Input and output buffers cannot be smaller than 16 and 66 bytes.
    if length < 16 || maxout < 66 {
        return 0;
    }

    let mut op;
    // SAFETY: `length >= 16` and `maxout >= 66`. The loop bounds keep every
    // unchecked input read below `length`; every output write is preceded by a
    // capacity check against `maxout`; hash references always point to an
    // earlier input position.
    unsafe {
        // Entropy probing: check 1/4 of the buffer to estimate the ratio.
        let maxlen = length / 4;
        let shift = length - maxlen;
        let cratio = get_cratio(&input[shift..], maxlen, 3, 3, &mut workspace.probe);
        if cratio < CRATIO_THRESHOLD[clevel] {
            return 0;
        }

        // When going back in a match (shift), compression properties change;
        // 3 works best with bitshuffle/small typesizes and low-entropy data.
        let (ipshift, minlen) = if !split_block || cratio < 4.0 {
            (3usize, 3usize)
        } else {
            (4usize, 4usize)
        };
        let hashlog = HASHLOG_TABLE[clevel];
        let htab = &mut workspace.hash[..1usize << hashlog];

        let ip_bound = length - 1;
        let ip_limit = length - 12;

        let mut ip;
        let mut copy: u32 = 4;
        let mut hval: usize;

        // Start with a literal copy.
        *output.get_unchecked_mut(0) = (MAX_COPY - 1) as u8;
        output
            .get_unchecked_mut(1..5)
            .copy_from_slice(input.get_unchecked(0..4));
        op = 5;
        ip = 4;

        macro_rules! literal {
            () => {{
                if op + 2 > maxout {
                    return 0;
                }
                *output.get_unchecked_mut(op) = *input.get_unchecked(ip);
                op += 1;
                ip += 1;
                copy += 1;
                if copy == MAX_COPY {
                    copy = 0;
                    *output.get_unchecked_mut(op) = (MAX_COPY - 1) as u8;
                    op += 1;
                }
            }};
        }

        while ip < ip_limit {
            let anchor = ip;

            let seq = read_u32(input, ip);
            hval = hash_function(seq, hashlog);
            let mut ref_ = htab[hval] as usize;
            let distance = anchor - ref_;
            htab[hval] = anchor as u32;

            if distance == 0 || distance >= MAX_FARDISTANCE as usize {
                literal!();
                continue;
            }

            if read_u32(input, ref_) == seq {
                ref_ += 4;
            } else {
                literal!();
                continue;
            }

            ip = anchor + 4;
            let distance = distance - 1;

            // Zero biased distance means a run.
            ip = if distance == 0 {
                get_run(input, ip, ip_bound, ref_)
            } else {
                get_match(input, ip, ip_bound, ref_)
            };

            // Length is biased: '1' means a match of 3 bytes.
            ip -= ipshift;
            let len = ip - anchor;

            // Short matches are expensive to decompress; skip them.
            if len < minlen || (len <= 5 && distance >= MAX_DISTANCE as usize) {
                ip = anchor;
                literal!();
                continue;
            }

            // Patch the literal count written so far.
            if copy != 0 {
                *output.get_unchecked_mut(op - copy as usize - 1) = (copy - 1) as u8;
            } else {
                op -= 1;
            }
            copy = 0;

            if distance < MAX_DISTANCE as usize {
                if len < 7 {
                    if op + 2 > maxout {
                        return 0;
                    }
                    *output.get_unchecked_mut(op) = ((len << 5) | (distance >> 8)) as u8;
                    *output.get_unchecked_mut(op + 1) = (distance & 255) as u8;
                    op += 2;
                } else {
                    if op + 1 > maxout {
                        return 0;
                    }
                    *output.get_unchecked_mut(op) = ((7 << 5) | (distance >> 8)) as u8;
                    op += 1;
                    let mut l = len - 7;
                    while l >= 255 {
                        if op + 1 > maxout {
                            return 0;
                        }
                        *output.get_unchecked_mut(op) = 255;
                        op += 1;
                        l -= 255;
                    }
                    if op + 2 > maxout {
                        return 0;
                    }
                    *output.get_unchecked_mut(op) = l as u8;
                    *output.get_unchecked_mut(op + 1) = (distance & 255) as u8;
                    op += 2;
                }
            } else {
                // Far match.
                let distance = distance - MAX_DISTANCE as usize;
                if len < 7 {
                    if op + 4 > maxout {
                        return 0;
                    }
                    *output.get_unchecked_mut(op) = ((len << 5) + 31) as u8;
                    *output.get_unchecked_mut(op + 1) = 255;
                    *output.get_unchecked_mut(op + 2) = (distance >> 8) as u8;
                    *output.get_unchecked_mut(op + 3) = (distance & 255) as u8;
                    op += 4;
                } else {
                    if op + 1 > maxout {
                        return 0;
                    }
                    *output.get_unchecked_mut(op) = ((7 << 5) + 31) as u8;
                    op += 1;
                    let mut l = len - 7;
                    while l >= 255 {
                        if op + 1 > maxout {
                            return 0;
                        }
                        *output.get_unchecked_mut(op) = 255;
                        op += 1;
                        l -= 255;
                    }
                    if op + 4 > maxout {
                        return 0;
                    }
                    *output.get_unchecked_mut(op) = l as u8;
                    *output.get_unchecked_mut(op + 1) = 255;
                    *output.get_unchecked_mut(op + 2) = (distance >> 8) as u8;
                    *output.get_unchecked_mut(op + 3) = (distance & 255) as u8;
                    op += 4;
                }
            }

            // Update the hash at the match boundary.
            let seq = read_u32(input, ip);
            hval = hash_function(seq, hashlog);
            htab[hval] = ip as u32;
            ip += 1;
            if clevel == 9 {
                // A second hash helps on some data at max clevel only.
                let seq = seq >> 8;
                hval = hash_function(seq, hashlog);
                htab[hval] = ip as u32;
                ip += 1;
            } else {
                ip += 1;
            }

            if op + 1 > maxout {
                return 0;
            }
            // Assuming a literal copy follows.
            *output.get_unchecked_mut(op) = (MAX_COPY - 1) as u8;
            op += 1;
        }

        // Left-over as literal copy.
        while ip <= ip_bound {
            if op + 2 > maxout {
                return 0;
            }
            *output.get_unchecked_mut(op) = *input.get_unchecked(ip);
            op += 1;
            ip += 1;
            copy += 1;
            if copy == MAX_COPY {
                copy = 0;
                *output.get_unchecked_mut(op) = (MAX_COPY - 1) as u8;
                op += 1;
            }
        }

        if copy != 0 {
            *output.get_unchecked_mut(op - copy as usize - 1) = (copy - 1) as u8;
        } else {
            op -= 1;
        }
    }

    // Marker bit distinguishing blosclz streams.
    output[0] |= 1 << 5;
    op
}

/// Copy `len` bytes from `from` to `op` in `buf`, handling overlap the way
/// LZ77 requires (repeating the pattern when the source overlaps the output).
///
/// # Safety
///
/// `from < op`, `op + len <= buf.len()`, and bytes before `op` through the
/// repeated match period must already be initialized.
#[inline]
unsafe fn copy_match(buf: &mut [u8], mut op: usize, from: usize, mut len: usize) -> usize {
    let mut initialized = op - from;
    while len > 0 {
        let chunk = initialized.min(len);
        buf.copy_within(from..from + chunk, op);
        op += chunk;
        len -= chunk;
        initialized += chunk;
    }
    op
}

/// Decompress `input` into `output`.
///
/// Returns the decompressed size, or 0 on corrupted input or an output
/// buffer that is too small.
pub(crate) fn blosclz_decompress(input: &[u8], output: &mut [u8]) -> usize {
    let length = input.len();
    let maxout = output.len();
    if length == 0 {
        return 0;
    }

    let ip_limit = length;
    let op_limit = maxout;

    let mut ip = 1usize;
    let mut op = 0usize;
    let mut ctrl = (input[0] & 31) as u32;

    // SAFETY: every unchecked input access is dominated by an `ip_limit`
    // check. Every unchecked output access is dominated by an `op_limit`
    // check, and match references are verified to lie in initialized output.
    unsafe {
        loop {
            if ctrl >= 32 {
                // Match.
                let mut len = (ctrl >> 5) as u64 - 1;
                let mut ofs = (ctrl & 31) << 8;

                if len == 6 {
                    loop {
                        if ip + 1 >= ip_limit {
                            return 0;
                        }
                        let code = *input.get_unchecked(ip) as u64;
                        ip += 1;
                        len = match len.checked_add(code) {
                            Some(len) => len,
                            None => return 0,
                        };
                        if code != 255 {
                            break;
                        }
                    }
                } else if ip + 1 >= ip_limit {
                    return 0;
                }
                let code = *input.get_unchecked(ip) as u64;
                ip += 1;
                len += 3;
                // A reference before the initialized output marks corrupt input.
                let mut ref_ = match op.checked_sub(ofs as usize + code as usize) {
                    Some(r) => r,
                    None => return 0,
                };

                // 16-bit distance match.
                if code == 255 && ofs == (31 << 8) {
                    if ip + 1 >= ip_limit {
                        return 0;
                    }
                    ofs = ((*input.get_unchecked(ip) as u32) << 8)
                        | *input.get_unchecked(ip + 1) as u32;
                    ip += 2;
                    match op.checked_sub(ofs as usize + MAX_DISTANCE as usize) {
                        Some(r) => ref_ = r,
                        None => return 0,
                    }
                }

                let len = match usize::try_from(len) {
                    Ok(len) => len,
                    Err(_) => return 0,
                };
                let match_end = match op.checked_add(len) {
                    Some(end) if end <= op_limit => end,
                    _ => return 0,
                };
                if ref_ < 1 {
                    return 0;
                }
                if ip >= ip_limit {
                    break;
                }
                ctrl = *input.get_unchecked(ip) as u32;
                ip += 1;

                ref_ -= 1;
                if ref_ + 1 == op {
                    // Optimized copy for a run.
                    let b = *output.get_unchecked(ref_);
                    output.get_unchecked_mut(op..match_end).fill(b);
                    op = match_end;
                } else if ref_.checked_add(len).is_some_and(|end| end <= op) {
                    output.copy_within(ref_..ref_ + len, op);
                    op = match_end;
                } else if ref_ >= op {
                    return 0;
                } else {
                    op = copy_match(output, op, ref_, len);
                }
            } else {
                // Literal.
                ctrl += 1;
                if op + ctrl as usize > op_limit {
                    return 0;
                }
                if ip + ctrl as usize > ip_limit {
                    return 0;
                }
                output
                    .get_unchecked_mut(op..op + ctrl as usize)
                    .copy_from_slice(input.get_unchecked(ip..ip + ctrl as usize));
                op += ctrl as usize;
                ip += ctrl as usize;
                if ip >= ip_limit {
                    break;
                }
                ctrl = *input.get_unchecked(ip) as u32;
                ip += 1;
            }
        }
    }

    op
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_rejects_out_of_range_levels() {
        assert!(matches!(
            Workspace::default().prepare(10),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn overlapping_match_copy_matches_lz77_semantics() {
        for distance in 1..=64 {
            for len in 1..=256 {
                let prefix = 96;
                let from = prefix - distance;
                let mut expected: Vec<u8> =
                    (0..prefix + len).map(|index| (index * 37) as u8).collect();
                let mut actual = expected.clone();
                for offset in 0..len {
                    expected[prefix + offset] = expected[from + offset];
                }
                // SAFETY: `from < prefix`, the destination has `len` bytes,
                // and the initialized source window grows with the output.
                let end = unsafe { copy_match(&mut actual, prefix, from, len) };
                assert_eq!(end, prefix + len);
                assert_eq!(actual, expected, "distance={distance} len={len}");
            }
        }
    }

    fn roundtrip(data: &[u8], clevel: i32, split: bool) {
        let mut out = vec![0u8; data.len() + data.len() / 8 + 256];
        let n = blosclz_compress(data, &mut out, clevel, split);
        if n == 0 {
            // Incompressible: fine, caller stores raw.
            return;
        }
        let mut dec = vec![0u8; data.len()];
        let m = blosclz_decompress(&out[..n], &mut dec);
        let first_difference = dec[..m.min(data.len())]
            .iter()
            .zip(data)
            .position(|(actual, expected)| actual != expected);
        assert_eq!(
            m,
            data.len(),
            "clevel={clevel} split={split} len={} encoded={n} first_difference={first_difference:?}",
            data.len()
        );
        assert_eq!(&dec[..m], data);
    }

    #[test]
    fn roundtrip_long_overlapping_matches() {
        let pattern: Vec<u8> = (0..512).map(|index| (index * 7) as u8).collect();
        let mut data = Vec::with_capacity(4096);
        for _ in 0..8 {
            data.extend_from_slice(&pattern);
        }
        roundtrip(&data, 5, false);
    }

    #[test]
    fn roundtrip_shuffled_float_bytes() {
        let values: Vec<[u8; 4]> = (0..32_768)
            .map(|index| (((index as f32) * 0.017).sin()).to_le_bytes())
            .collect();
        let mut shuffled = Vec::with_capacity(values.len() * 4);
        for byte_index in 0..4 {
            shuffled.extend(values.iter().map(|value| value[byte_index]));
        }
        for level in [1, 5, 9] {
            roundtrip(&shuffled, level, false);
            roundtrip(&shuffled, level, true);
        }
    }

    #[test]
    fn roundtrip_patterns_all_levels() {
        // Sequential f32-like bytes: highly compressible with runs and matches.
        let mut seq = Vec::with_capacity(64 * 1024);
        for i in 0..16_384u32 {
            seq.extend_from_slice(&i.to_le_bytes());
        }
        // Constant data: pure runs.
        let const_data = vec![0xABu8; 100_000];
        // Random data: expected incompressible.
        let mut rng = 0x12345678u64;
        let mut rand = Vec::with_capacity(100_000);
        for _ in 0..100_000 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rand.push((rng >> 33) as u8);
        }
        for clevel in 0..10 {
            for split in [false, true] {
                roundtrip(&seq, clevel, split);
                roundtrip(&const_data, clevel, split);
                roundtrip(&rand, clevel, split);
            }
        }
    }

    #[test]
    fn roundtrip_small_and_edge_sizes() {
        for len in [
            0usize, 1, 15, 16, 17, 31, 32, 33, 65, 66, 100, 255, 256, 1024,
        ] {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            for clevel in [1, 5, 9] {
                roundtrip(&data, clevel, false);
                roundtrip(&data, clevel, true);
            }
        }
    }

    #[test]
    fn roundtrip_large_block() {
        // 8 MiB of structured data.
        let mut data = Vec::with_capacity(8 << 20);
        let mut x = 0u64;
        for _ in 0..(8 << 20) / 8 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.extend_from_slice(&(x & 0x00FF_FFFF_FFFF_FFFF).to_le_bytes());
        }
        roundtrip(&data, 5, true);
        roundtrip(&data, 9, false);
    }

    #[test]
    fn decompress_never_panics_on_garbage() {
        // Feeding arbitrary bytes must return 0 or a bounded result, never panic.
        let mut rng = 0xDEADBEEF_CAFEBABEu64;
        let mut garbage = Vec::with_capacity(4096);
        for _ in 0..4096 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            garbage.push((rng >> 33) as u8);
        }
        let mut out = vec![0u8; 8192];
        for start in 0..garbage.len().saturating_sub(64) {
            let _ = blosclz_decompress(&garbage[start..start + 64], &mut out);
        }
        let _ = blosclz_decompress(&garbage, &mut out);
        let _ = blosclz_decompress(&[], &mut out);
    }

    #[test]
    fn decompress_rejects_overflowing_lengths() {
        // A match whose length extension would overflow must be rejected,
        // not wrap around and write out of bounds.
        let mut input = vec![0u8; 1024];
        // ctrl byte: len-1 == 6 (triggers the extension loop), ofs_hi == 0
        input[0] = 224; // 224 >> 5 == 7 -> len = 6; 224 & 31 == 0 -> ofs = 0
        for b in &mut input[1..501] {
            *b = 255;
        }
        input[501] = 0; // extension terminator
        input[502] = 0; // distance low byte
        let mut out = vec![0u8; 16];
        let n = blosclz_decompress(&input, &mut out);
        // len would be 6 + 500*255 + 3; the reference is before the output
        // start, so the stream is rejected.
        assert_eq!(n, 0);
    }
}
