//! Dense gather / slice kernels with row-parallel execution.

use crate::array::DenseArray;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::select::NormalizedAxis;

use super::util::{
    assume_init_bytes, checked_mul, par_for_row_blocks, uninit_bytes, usize_from_u64,
};

/// Select arbitrary rows and columns from a dense row-major buffer.
pub fn dense_select(
    values: &[u8],
    n_rows: usize,
    n_cols: usize,
    dtype: DType,
    rows: &NormalizedAxis,
    cols: &NormalizedAxis,
    threads: usize,
) -> Result<DenseArray> {
    crate::parallel::validate_threads(threads)?;
    let n_rows_u64 = u64::try_from(n_rows)
        .map_err(|_| Error::invalid_argument("dense row count exceeds u64"))?;
    let n_cols_u64 = u64::try_from(n_cols)
        .map_err(|_| Error::invalid_argument("dense column count exceeds u64"))?;
    rows.validate(n_rows_u64)?;
    cols.validate(n_cols_u64)?;
    let elem = dtype.size();
    let expected = checked_mul(
        n_rows,
        checked_mul(n_cols, elem, "dense row")?,
        "dense buffer",
    )?;
    if values.len() != expected {
        return Err(Error::invalid_argument(format!(
            "dense buffer length {} does not match {n_rows}×{n_cols}×{elem}",
            values.len()
        )));
    }

    let out_rows = usize_from_u64(rows.len(), "selected row count")?;
    let out_cols = usize_from_u64(cols.len(), "selected col count")?;
    let row_bytes = checked_mul(n_cols, elem, "dense source row")?;
    let out_row_bytes = checked_mul(out_cols, elem, "dense output row")?;
    let out_len = checked_mul(out_rows, out_row_bytes, "dense output")?;
    let mut output = uninit_bytes(out_len)?;

    if out_rows == 0 || out_cols == 0 {
        // SAFETY: a zero-sized output contains no uninitialized elements.
        let output = unsafe { assume_init_bytes(output) };
        return DenseArray::from_bytes([out_rows, out_cols], dtype, output);
    }

    // Fast path: contiguous rows + full-width columns → single memcpy.
    if let (Some(row_range), Some(col_range)) = (rows.as_range(), cols.as_range()) {
        let row_start = usize_from_u64(row_range.start, "row start")?;
        let col_start = usize_from_u64(col_range.start, "col start")?;
        let col_end = usize_from_u64(col_range.end, "col end")?;

        if col_start == 0 && col_end == n_cols {
            let src_start = checked_mul(row_start, row_bytes, "row block start")?;
            let src_end = src_start
                .checked_add(out_len)
                .ok_or_else(|| Error::invalid_argument("dense full-row copy overflow"))?;
            let source = values
                .get(src_start..src_end)
                .ok_or_else(|| Error::invalid_argument("dense row block out of bounds"))?;
            // SAFETY: `source.len() == out_len == output.len()`, the source
            // matrix and new output allocation do not overlap, and the copy
            // initializes every destination byte exactly once.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr(),
                    output.as_mut_ptr().cast::<u8>(),
                    out_len,
                );
            }
            // SAFETY: the full-row copy above initialized all `out_len` bytes.
            let output = unsafe { assume_init_bytes(output) };
            return DenseArray::from_bytes([out_rows, out_cols], dtype, output);
        }

        let col_bytes = checked_mul(col_end - col_start, elem, "col strip")?;
        let col_byte_start = checked_mul(col_start, elem, "col byte start")?;
        debug_assert_eq!(col_bytes, out_row_bytes);
        par_for_row_blocks(
            threads,
            out_rows,
            out_row_bytes,
            &mut output,
            |job_start, job_end, block| {
                for local_row in job_start..job_end {
                    let src_row = row_start + local_row;
                    let src_off = src_row
                        .checked_mul(row_bytes)
                        .and_then(|base| base.checked_add(col_byte_start))
                        .ok_or_else(|| {
                            Error::invalid_argument("dense col-strip offset overflow")
                        })?;
                    let src = values
                        .get(src_off..src_off + col_bytes)
                        .ok_or_else(|| Error::invalid_argument("dense col strip out of bounds"))?;
                    let dst_off = (local_row - job_start) * out_row_bytes;
                    debug_assert!(dst_off + col_bytes <= block.len());
                    // SAFETY: the checked source range and exact row-block
                    // partition prove both ranges have `col_bytes` elements.
                    // Every selected row owns a disjoint destination row.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr(),
                            block.as_mut_ptr().cast::<u8>().add(dst_off),
                            col_bytes,
                        );
                    }
                }
                Ok(())
            },
        )?;
        // SAFETY: every output row was filled with one complete column strip.
        let output = unsafe { assume_init_bytes(output) };
        return DenseArray::from_bytes([out_rows, out_cols], dtype, output);
    }

    // Gathered rows with a contiguous column strip need no materialized row index vector.
    if let Some(col_range) = cols.as_range() {
        let col_start = usize_from_u64(col_range.start, "col start")?;
        let col_end = usize_from_u64(col_range.end, "col end")?;
        let col_bytes = checked_mul(col_end - col_start, elem, "col strip")?;
        let col_byte_start = checked_mul(col_start, elem, "col byte start")?;
        debug_assert_eq!(col_bytes, out_row_bytes);
        par_for_row_blocks(
            threads,
            out_rows,
            out_row_bytes,
            &mut output,
            |job_start, job_end, block| {
                for local_row in job_start..job_end {
                    let src_row = axis_position(rows, local_row, "row position")?;
                    let src_off = checked_mul(src_row, row_bytes, "gather row offset")?
                        .checked_add(col_byte_start)
                        .ok_or_else(|| {
                            Error::invalid_argument("dense gather row offset overflow")
                        })?;
                    let src = values
                        .get(src_off..src_off + col_bytes)
                        .ok_or_else(|| Error::invalid_argument("dense gather row out of bounds"))?;
                    let dst_off = (local_row - job_start) * out_row_bytes;
                    debug_assert!(dst_off + col_bytes <= block.len());
                    // SAFETY: the normalized row selection and checked source
                    // range prove `src` is valid; row-block partitioning gives
                    // this worker the entire disjoint destination row.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr(),
                            block.as_mut_ptr().cast::<u8>().add(dst_off),
                            col_bytes,
                        );
                    }
                }
                Ok(())
            },
        )?;
        // SAFETY: every gathered row was filled with one complete column strip.
        let output = unsafe { assume_init_bytes(output) };
        return DenseArray::from_bytes([out_rows, out_cols], dtype, output);
    }

    let mut col_byte_offsets = Vec::new();
    col_byte_offsets.try_reserve_exact(out_cols)?;
    for local_col in 0..out_cols {
        let col = cols.nth(
            u64::try_from(local_col)
                .map_err(|_| Error::invalid_argument("selected column index exceeds u64"))?,
        )?;
        col_byte_offsets.push(checked_mul(
            usize_from_u64(col, "col position")?,
            elem,
            "gather col byte offset",
        )?);
    }

    // Element-wise column gather; rows are resolved directly from their normalized form.
    par_for_row_blocks(
        threads,
        out_rows,
        out_row_bytes,
        &mut output,
        |job_start, job_end, block| {
            for local_row in job_start..job_end {
                let src_row = axis_position(rows, local_row, "row position")?;
                let row_base = checked_mul(src_row, row_bytes, "gather row base")?;
                let dst_row_off = (local_row - job_start) * out_row_bytes;
                for (local_col, &col_byte_offset) in col_byte_offsets.iter().enumerate() {
                    let src_off = row_base
                        .checked_add(col_byte_offset)
                        .ok_or_else(|| Error::invalid_argument("dense gather offset overflow"))?;
                    let dst_off = dst_row_off + checked_mul(local_col, elem, "gather dst col")?;
                    debug_assert!(src_off + elem <= values.len());
                    debug_assert!(dst_off + elem <= block.len());
                    // SAFETY: axis validation and the checked row/column byte
                    // calculations prove both ranges are in bounds. Each
                    // `(local_row, local_col)` owns a distinct output element,
                    // and the source matrix does not alias the output buffer.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            values.as_ptr().add(src_off),
                            block.as_mut_ptr().cast::<u8>().add(dst_off),
                            elem,
                        );
                    }
                }
            }
            Ok(())
        },
    )?;

    // SAFETY: the nested row/column loops initialized every output element,
    // and `par_for_row_blocks` returned only after all workers completed.
    let output = unsafe { assume_init_bytes(output) };
    DenseArray::from_bytes([out_rows, out_cols], dtype, output)
}

#[inline]
fn axis_position(axis: &NormalizedAxis, local: usize, context: &str) -> Result<usize> {
    let position = axis.nth(
        u64::try_from(local)
            .map_err(|_| Error::invalid_argument(format!("{context} exceeds u64")))?,
    )?;
    usize_from_u64(position, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::select::AxisIndex;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn contiguous_block() {
        let values: Vec<f32> = (0..20).map(|v| v as f32).collect();
        let bytes = f32_bytes(&values);
        let rows = AxisIndex::range(1, 4).normalize(5).unwrap();
        let cols = AxisIndex::range(1, 3).normalize(4).unwrap();
        let out = dense_select(&bytes, 5, 4, DType::F32, &rows, &cols, 2).unwrap();
        assert_eq!(out.shape(), [3, 2]);
        let expected = f32_bytes(&[5.0, 6.0, 9.0, 10.0, 13.0, 14.0]);
        assert_eq!(out.values(), expected);
    }

    #[test]
    fn fancy_rows_and_cols() {
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let bytes = f32_bytes(&values);
        let rows = AxisIndex::positions([2, 0]).normalize(3).unwrap();
        let cols = AxisIndex::positions([3, 1]).normalize(4).unwrap();
        let out = dense_select(&bytes, 3, 4, DType::F32, &rows, &cols, 4).unwrap();
        assert_eq!(out.values(), f32_bytes(&[11.0, 9.0, 3.0, 1.0]));
    }

    #[test]
    fn strided_rows_match_expanded_positions() {
        let values: Vec<f32> = (0..20).map(|v| v as f32).collect();
        let bytes = f32_bytes(&values);
        let strided = AxisIndex::strided(0, 5, 2).normalize(5).unwrap();
        let expanded = AxisIndex::positions([0, 2, 4]).normalize(5).unwrap();
        let cols = AxisIndex::range(1, 3).normalize(4).unwrap();
        let from_stride = dense_select(&bytes, 5, 4, DType::F32, &strided, &cols, 2).unwrap();
        let from_gather = dense_select(&bytes, 5, 4, DType::F32, &expanded, &cols, 2).unwrap();
        assert_eq!(from_stride.values(), from_gather.values());
        assert_eq!(
            from_stride.values(),
            f32_bytes(&[1.0, 2.0, 9.0, 10.0, 17.0, 18.0])
        );
    }
}
