//! Shared helpers for kernel modules.

use crate::error::{Error, Result};
use crate::parallel;

use std::mem::{ManuallyDrop, MaybeUninit};

/// Default rows per parallel job — small enough for dynamic balance, large
/// enough to amortize dispatch cost.
pub(crate) const ROW_JOB: usize = 64;

pub(crate) fn checked_mul(count: usize, size: usize, context: &str) -> Result<usize> {
    count
        .checked_mul(size)
        .ok_or_else(|| Error::invalid_argument(format!("{context} size overflow")))
}

pub(crate) fn zeroed(len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)?;
    out.resize(len, 0);
    Ok(out)
}

/// Allocate byte storage without paying to initialize bytes that a kernel will
/// overwrite completely.
pub(crate) fn uninit_bytes(len: usize) -> Result<Vec<MaybeUninit<u8>>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)?;
    // SAFETY: every bit pattern, including uninitialized storage, is valid for
    // `MaybeUninit<u8>`. Callers still cannot observe the bytes as `u8` until
    // they have initialized the complete allocation.
    unsafe { out.set_len(len) };
    Ok(out)
}

/// Convert a fully initialized byte allocation to its ordinary `Vec<u8>` form.
///
/// # Safety
///
/// Every element in `bytes` must have been initialized. The allocation must
/// still have the length and capacity produced by [`uninit_bytes`].
pub(crate) unsafe fn assume_init_bytes(bytes: Vec<MaybeUninit<u8>>) -> Vec<u8> {
    let mut bytes = ManuallyDrop::new(bytes);
    let pointer = bytes.as_mut_ptr().cast::<u8>();
    let len = bytes.len();
    let capacity = bytes.capacity();
    // SAFETY: the caller guarantees all `len` bytes are initialized. `u8` and
    // `MaybeUninit<u8>` have identical size/alignment, and `ManuallyDrop`
    // transfers the original allocation to this `Vec` exactly once.
    unsafe { Vec::from_raw_parts(pointer, len, capacity) }
}

pub(crate) fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::invalid_argument(format!("{context} exceeds usize")))
}

/// Partition `[0, n_rows)` into jobs of size `ROW_JOB` and process in parallel.
///
/// Each job receives a half-open row range plus an exclusive mutable destination
/// buffer for those rows (safe disjoint writes without `unsafe`).
pub(crate) fn par_for_row_blocks<T, F>(
    threads: usize,
    n_rows: usize,
    row_len: usize,
    output: &mut [T],
    work: F,
) -> Result<()>
where
    T: Send,
    F: Fn(usize, usize, &mut [T]) -> Result<()> + Sync,
{
    if n_rows == 0 {
        return Ok(());
    }
    let expected = checked_mul(n_rows, row_len, "par row blocks")?;
    if output.len() != expected {
        return Err(Error::invalid_argument(
            "output length does not match n_rows × row_bytes",
        ));
    }
    let job_count = n_rows.div_ceil(ROW_JOB);
    let mut remaining = output;
    let mut row_start = 0usize;
    parallel::try_for_each_stream(
        threads.max(1),
        job_count,
        |emit| {
            while row_start < n_rows {
                let row_end = (row_start + ROW_JOB).min(n_rows);
                let block_len = (row_end - row_start) * row_len;
                let tail = std::mem::take(&mut remaining);
                let (block, tail) = tail.split_at_mut(block_len);
                remaining = tail;
                emit((row_start, row_end, block))?;
                row_start = row_end;
            }
            if !remaining.is_empty() {
                return Err(Error::invalid_argument(
                    "row-block producer did not consume the full output",
                ));
            }
            Ok(())
        },
        |(start, end, block)| work(start, end, block),
    )
}

/// Copy one matrix element without slice construction or bounds checks.
///
/// # Safety
///
/// `src` and `dst` must each be valid for `elem_size` initialized/readable or
/// writable bytes respectively, and the two ranges must not overlap.
#[inline(always)]
pub(crate) unsafe fn copy_elem_unchecked(dst: *mut u8, src: *const u8, elem_size: usize) {
    // SAFETY: the caller provides the complete pointer validity, length, and
    // non-overlap invariants required by `copy_nonoverlapping`.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, elem_size) };
}

/// Read a CSR column index (u16/u32 LE) at element position `pos` without a
/// bounds check.
///
/// # Safety
///
/// `index_size` must be 2 or 4 and `pos * index_size + index_size` must not
/// exceed `indices.len()`.
#[inline(always)]
pub(crate) unsafe fn read_index_unchecked(indices: &[u8], pos: usize, index_size: usize) -> u64 {
    debug_assert!(matches!(index_size, 2 | 4));
    debug_assert!(pos
        .checked_mul(index_size)
        .and_then(|offset| offset.checked_add(index_size))
        .is_some_and(|end| end <= indices.len()));
    let offset = pos * index_size;
    match index_size {
        2 => {
            // SAFETY: the caller guarantees that two initialized bytes begin
            // at `offset`; `read_unaligned` imposes no alignment requirement.
            let value = unsafe { indices.as_ptr().add(offset).cast::<u16>().read_unaligned() };
            u64::from(u16::from_le(value))
        }
        4 => {
            // SAFETY: the caller guarantees that four initialized bytes begin
            // at `offset`; `read_unaligned` imposes no alignment requirement.
            let value = unsafe { indices.as_ptr().add(offset).cast::<u32>().read_unaligned() };
            u64::from(u32::from_le(value))
        }
        _ => unreachable!("CSR index size is 2 or 4"),
    }
}

/// Write a CSR column index after the output layout and value width have been
/// validated at the kernel boundary.
///
/// # Safety
///
/// `index_size` must be 2 or 4, `out.add(pos * index_size)` must be valid for a
/// complete index, and `value` must fit the selected integer width.
#[inline(always)]
pub(crate) unsafe fn write_index_unchecked(
    out: *mut u8,
    pos: usize,
    index_size: usize,
    value: u64,
) {
    debug_assert!(matches!(index_size, 2 | 4));
    debug_assert!(index_size == 4 || value <= u64::from(u16::MAX));
    debug_assert!(value <= u64::from(u32::MAX));
    let offset = pos * index_size;
    match index_size {
        2 => {
            // SAFETY: the caller guarantees a complete writable u16 slot at
            // `offset`; unaligned output is explicitly supported.
            unsafe {
                out.add(offset)
                    .cast::<u16>()
                    .write_unaligned((value as u16).to_le());
            }
        }
        4 => {
            // SAFETY: the caller guarantees a complete writable u32 slot at
            // `offset`; unaligned output is explicitly supported.
            unsafe {
                out.add(offset)
                    .cast::<u32>()
                    .write_unaligned((value as u32).to_le());
            }
        }
        _ => unreachable!("CSR index size is 2 or 4"),
    }
}
