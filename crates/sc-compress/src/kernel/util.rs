//! Shared helpers for kernel modules.

use crate::error::{Error, Result};
use crate::parallel;

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

pub(crate) fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::invalid_argument(format!("{context} exceeds usize")))
}

/// Partition `[0, n_rows)` into jobs of size `ROW_JOB` and process in parallel.
///
/// Each job receives a half-open row range plus an exclusive mutable destination
/// buffer for those rows (safe disjoint writes without `unsafe`).
pub(crate) fn par_for_row_blocks<F>(
    threads: usize,
    n_rows: usize,
    row_bytes: usize,
    output: &mut [u8],
    work: F,
) -> Result<()>
where
    F: Fn(usize, usize, &mut [u8]) -> Result<()> + Sync,
{
    if n_rows == 0 {
        return Ok(());
    }
    let expected = checked_mul(n_rows, row_bytes, "par row blocks")?;
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
                let block_bytes = (row_end - row_start) * row_bytes;
                let tail = std::mem::take(&mut remaining);
                let (block, tail) = tail.split_at_mut(block_bytes);
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

/// Copy `elem_size` bytes at `src` into `dst`.
#[inline(always)]
pub(crate) fn copy_elem(dst: &mut [u8], src: &[u8], elem_size: usize) {
    assert_eq!(dst.len(), elem_size, "destination element size mismatch");
    assert_eq!(src.len(), elem_size, "source element size mismatch");
    // SAFETY: the exact-length assertions prove both pointers are valid for
    // `elem_size` bytes. The slices are distinct inputs to this kernel helper,
    // so their ranges do not overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), elem_size);
    }
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

/// Write a CSR column index as u16/u32 LE.
#[inline(always)]
pub(crate) fn write_index(out: &mut [u8], pos: usize, index_size: usize, value: u64) -> Result<()> {
    let offset = pos
        .checked_mul(index_size)
        .ok_or_else(|| Error::invalid_argument("CSR output index offset overflow"))?;
    let end = offset
        .checked_add(index_size)
        .ok_or_else(|| Error::invalid_argument("CSR output index end overflow"))?;
    if end > out.len() {
        return Err(Error::invalid_argument(
            "CSR output index position is out of bounds",
        ));
    }
    match index_size {
        2 => {
            let value = u16::try_from(value)
                .map_err(|_| Error::invalid_argument("CSR index does not fit u16"))?
                .to_le();
            // SAFETY: the `end <= out.len()` check proves two writable bytes;
            // `write_unaligned` imposes no alignment requirement.
            unsafe {
                out.as_mut_ptr()
                    .add(offset)
                    .cast::<u16>()
                    .write_unaligned(value);
            }
        }
        4 => {
            let value = u32::try_from(value)
                .map_err(|_| Error::invalid_argument("CSR index does not fit u32"))?
                .to_le();
            // SAFETY: the `end <= out.len()` check proves four writable bytes;
            // `write_unaligned` imposes no alignment requirement.
            unsafe {
                out.as_mut_ptr()
                    .add(offset)
                    .cast::<u32>()
                    .write_unaligned(value);
            }
        }
        _ => unreachable!("CSR index size is 2 or 4"),
    }
    Ok(())
}
