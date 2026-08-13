//! CSR row-gather, column-filter, and densify kernels.
#![expect(
    clippy::too_many_arguments,
    reason = "low-level CSR kernels keep independent layout buffers and dtypes explicit"
)]

use std::mem::MaybeUninit;

use crate::array::{CsrArray, DenseArray};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::limits::ReadLimits;
use crate::parallel;
use crate::select::NormalizedAxis;

use super::util::{
    assume_init_bytes, checked_mul, copy_elem_unchecked, par_for_row_blocks, read_index_unchecked,
    uninit_bytes, usize_from_u64, write_index_unchecked, zeroed, ROW_JOB,
};

/// Column remap: `src_col -> Some(dst_col)` or drop.
#[derive(Debug, Clone)]
pub struct CsrColMap {
    storage: CsrColMapStorage,
    n_out_cols: usize,
}

#[derive(Debug, Clone)]
enum CsrColMapStorage {
    Affine { start: u64, end: u64 },
    Dense(Vec<u32>),
    Sparse(Vec<(u64, u32)>),
}

impl CsrColMap {
    pub fn n_out_cols(&self) -> usize {
        self.n_out_cols
    }

    #[inline(always)]
    pub fn map(&self, src_col: u64) -> Option<u32> {
        match &self.storage {
            CsrColMapStorage::Affine { start, end } => {
                if src_col < *start || src_col >= *end {
                    None
                } else {
                    u32::try_from(src_col - start).ok()
                }
            }
            CsrColMapStorage::Dense(map) => {
                let destination = *map.get(usize::try_from(src_col).ok()?)?;
                (destination != u32::MAX).then_some(destination)
            }
            CsrColMapStorage::Sparse(map) => map
                .binary_search_by_key(&src_col, |&(source, _)| source)
                .ok()
                .map(|index| map[index].1),
        }
    }

    /// Map a source column already validated against the map's source width.
    ///
    /// # Safety
    ///
    /// `src_col` must be smaller than the `n_cols` used to construct this map.
    #[inline(always)]
    unsafe fn map_unchecked(&self, src_col: u64) -> Option<u32> {
        match &self.storage {
            CsrColMapStorage::Affine { start, end } => {
                if src_col < *start || src_col >= *end {
                    None
                } else {
                    u32::try_from(src_col - start).ok()
                }
            }
            CsrColMapStorage::Dense(map) => {
                debug_assert!((src_col as usize) < map.len());
                // SAFETY: the caller guarantees `src_col` is inside the source
                // width, which is exactly the dense map length.
                let destination = unsafe { *map.get_unchecked(src_col as usize) };
                (destination != u32::MAX).then_some(destination)
            }
            CsrColMapStorage::Sparse(map) => map
                .binary_search_by_key(&src_col, |&(source, _)| source)
                .ok()
                .map(|index| {
                    // SAFETY: `binary_search_by_key` returned an index into the
                    // same immutable slice.
                    unsafe { map.get_unchecked(index).1 }
                }),
        }
    }

    /// Visit mapped entries from one canonical CSR row.
    ///
    /// # Safety
    ///
    /// `start..end` must delimit a validated, strictly increasing packed CSR
    /// row in `indices`, and `index_size` must match its u16/u32 representation.
    #[inline]
    unsafe fn visit_row_unchecked(
        &self,
        indices: &[u8],
        start: usize,
        end: usize,
        index_size: usize,
        mut visit: impl FnMut(usize, u32),
    ) {
        if let CsrColMapStorage::Sparse(map) = &self.storage {
            let mut position = start;
            let mut selected = 0usize;
            while position < end && selected < map.len() {
                // SAFETY: the caller guarantees `position` remains inside one
                // complete packed CSR row.
                let source = unsafe { read_index_unchecked(indices, position, index_size) };
                let candidate = map[selected].0;
                if source < candidate {
                    position += 1;
                } else if source > candidate {
                    selected += 1;
                } else {
                    visit(position, map[selected].1);
                    position += 1;
                    selected += 1;
                }
            }
            return;
        }

        for position in start..end {
            // SAFETY: the caller guarantees this is a complete packed row and
            // every source column is inside the map's source width.
            let source = unsafe { read_index_unchecked(indices, position, index_size) };
            // SAFETY: source-column validity is part of the same caller
            // invariant stated above.
            if let Some(destination) = unsafe { self.map_unchecked(source) } {
                visit(position, destination);
            }
        }
    }
}

pub(crate) struct GatherColumns {
    by_source: Vec<(u64, u32)>,
    unique_sources: usize,
    destinations_are_ordered: bool,
}

impl GatherColumns {
    pub(crate) fn new(n_cols: usize, positions: &[u64]) -> Result<Self> {
        let mut by_source = Vec::new();
        by_source.try_reserve_exact(positions.len())?;
        for (destination, &source) in positions.iter().enumerate() {
            if usize_from_u64(source, "col position")? >= n_cols {
                return Err(Error::invalid_argument("column position out of bounds"));
            }
            by_source.push((
                source,
                u32::try_from(destination)
                    .map_err(|_| Error::invalid_argument("selected columns exceed u32"))?,
            ));
        }
        by_source.sort_unstable();
        let unique_sources = if by_source.is_empty() {
            0
        } else {
            1 + by_source
                .windows(2)
                .filter(|pair| pair[0].0 != pair[1].0)
                .count()
        };
        let destinations_are_ordered = by_source.windows(2).all(|pair| pair[0].1 < pair[1].1);
        Ok(Self {
            by_source,
            unique_sources,
            destinations_are_ordered,
        })
    }

    #[inline]
    pub(crate) fn destinations(&self, source: u64) -> &[(u64, u32)] {
        let start = self
            .by_source
            .partition_point(|&(candidate, _)| candidate < source);
        let end =
            self.by_source[start..].partition_point(|&(candidate, _)| candidate == source) + start;
        &self.by_source[start..end]
    }

    pub(crate) fn unique_sources(&self) -> usize {
        self.unique_sources
    }

    pub(crate) fn destinations_are_ordered(&self) -> bool {
        self.destinations_are_ordered
    }

    pub(crate) fn prefer_binary_search(&self, row_nnz: usize) -> bool {
        prefer_index_binary_search(self.unique_sources(), row_nnz)
    }

    pub(crate) fn count_hits(
        &self,
        indices: &[u8],
        start: usize,
        end: usize,
        index_size: usize,
    ) -> Result<usize> {
        let nnz = end.saturating_sub(start);
        if self.prefer_binary_search(nnz) {
            let mut kept = 0usize;
            let mut index = 0usize;
            while index < self.by_source.len() {
                let source = self.by_source[index].0;
                let found = equal_index(indices, start, end, index_size, source);
                let mut copies = 0usize;
                while index < self.by_source.len() && self.by_source[index].0 == source {
                    copies += 1;
                    index += 1;
                }
                if found.is_some() {
                    kept = kept
                        .checked_add(copies)
                        .ok_or_else(|| Error::invalid_argument("CSR selected nnz overflow"))?;
                }
            }
            Ok(kept)
        } else {
            let mut kept = 0usize;
            let mut position = start;
            let mut query = 0usize;
            while position < end && query < self.by_source.len() {
                // SAFETY: callers validate that `start..end` is one canonical
                // packed CSR row, so `position` addresses a complete index.
                let source = unsafe { read_index_unchecked(indices, position, index_size) };
                let selected = self.by_source[query].0;
                if source < selected {
                    position += 1;
                } else if source > selected {
                    query += 1;
                } else {
                    let mut group_end = query + 1;
                    while group_end < self.by_source.len() && self.by_source[group_end].0 == source
                    {
                        group_end += 1;
                    }
                    kept = kept
                        .checked_add(group_end - query)
                        .ok_or_else(|| Error::invalid_argument("CSR selected nnz overflow"))?;
                    position += 1;
                    query = group_end;
                }
            }
            Ok(kept)
        }
    }

    pub(crate) fn collect_hits(
        &self,
        indices: &[u8],
        start: usize,
        end: usize,
        index_size: usize,
        out: &mut Vec<(u32, usize)>,
    ) -> Result<()> {
        let nnz = end.saturating_sub(start);
        if self.prefer_binary_search(nnz) {
            let mut index = 0usize;
            while index < self.by_source.len() {
                let source = self.by_source[index].0;
                let found = equal_index(indices, start, end, index_size, source);
                while index < self.by_source.len() && self.by_source[index].0 == source {
                    if let Some(position) = found {
                        out.push((self.by_source[index].1, position));
                    }
                    index += 1;
                }
            }
        } else {
            let mut position = start;
            let mut query = 0usize;
            while position < end && query < self.by_source.len() {
                // SAFETY: callers validate that `start..end` is one canonical
                // packed CSR row, so `position` addresses a complete index.
                let source = unsafe { read_index_unchecked(indices, position, index_size) };
                let selected = self.by_source[query].0;
                if source < selected {
                    position += 1;
                } else if source > selected {
                    query += 1;
                } else {
                    let mut group_end = query + 1;
                    while group_end < self.by_source.len() && self.by_source[group_end].0 == source
                    {
                        group_end += 1;
                    }
                    for &(_, destination) in &self.by_source[query..group_end] {
                        out.push((destination, position));
                    }
                    position += 1;
                    query = group_end;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn resident_bytes(&self) -> Result<usize> {
        self.by_source
            .len()
            .checked_mul(std::mem::size_of::<(u64, u32)>())
            .ok_or_else(|| Error::invalid_argument("column gather map size overflow"))
    }
}

/// Build a column map from a normalized column selection.
pub fn build_col_map(n_cols: usize, cols: &NormalizedAxis) -> Result<CsrColMap> {
    match cols {
        NormalizedAxis::Contiguous { start, end } => {
            let start = usize_from_u64(*start, "col start")?;
            let end = usize_from_u64(*end, "col end")?;
            if end > n_cols {
                return Err(Error::invalid_argument("column range exceeds n_cols"));
            }
            Ok(CsrColMap {
                storage: CsrColMapStorage::Affine {
                    start: start as u64,
                    end: end as u64,
                },
                n_out_cols: end - start,
            })
        }
        NormalizedAxis::Gather { .. } | NormalizedAxis::Strided { .. } => {
            let owned = match cols {
                NormalizedAxis::Strided { .. } => Some(cols.to_positions()),
                _ => None,
            };
            let positions = match cols {
                NormalizedAxis::Gather { positions } => positions.as_slice(),
                NormalizedAxis::Strided { .. } => owned.as_ref().map_or(&[][..], Vec::as_slice),
                NormalizedAxis::Contiguous { .. } => {
                    unreachable!("contiguous columns use affine map")
                }
            };
            let mut pairs = Vec::new();
            pairs.try_reserve_exact(positions.len())?;
            for (destination, &source) in positions.iter().enumerate() {
                if usize_from_u64(source, "col position")? >= n_cols {
                    return Err(Error::invalid_argument("column position out of bounds"));
                }
                let destination = u32::try_from(destination)
                    .map_err(|_| Error::invalid_argument("column map exceeds u32"))?;
                pairs.push((source, destination));
            }
            pairs.sort_unstable_by_key(|&(source, _)| source);
            if pairs.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err(Error::invalid_argument(
                    "column map cannot represent duplicate source columns",
                ));
            }
            let dense_limit = positions.len().saturating_mul(8).max(1024);
            let storage = if n_cols <= dense_limit && positions.len() < u32::MAX as usize {
                let mut map = Vec::new();
                map.try_reserve_exact(n_cols)?;
                map.resize(n_cols, u32::MAX);
                for &(source, destination) in &pairs {
                    map[source as usize] = destination;
                }
                CsrColMapStorage::Dense(map)
            } else {
                CsrColMapStorage::Sparse(pairs)
            };
            Ok(CsrColMap {
                storage,
                n_out_cols: positions.len(),
            })
        }
    }
}

/// Gather / slice CSR rows; column count unchanged.
pub fn csr_select_rows(
    indptr: &[u64],
    indices: &[u8],
    data: &[u8],
    n_rows: usize,
    n_cols: usize,
    index_dtype: DType,
    value_dtype: DType,
    rows: &NormalizedAxis,
    threads: usize,
) -> Result<CsrArray> {
    let index_size = index_dtype.size();
    let value_size = value_dtype.size();
    validate_csr_layout(indptr, indices, data, n_rows, index_size, value_size)?;

    if let Some(range) = rows.as_range() {
        let start = usize_from_u64(range.start, "row start")?;
        let end = usize_from_u64(range.end, "row end")?;
        return csr_slice_rows(
            indptr,
            indices,
            data,
            start,
            end,
            n_cols,
            index_dtype,
            value_dtype,
        );
    }

    let out_rows = usize_from_u64(rows.len(), "selected row count")?;
    if out_rows == 0 {
        return CsrArray::empty([0, n_cols], index_dtype, value_dtype);
    }

    // Pass 1: nnz per selected row with disjoint mutable count blocks.
    let mut row_nnz = zeroed_u64(out_rows)?;
    count_row_nnz(threads, out_rows, &mut row_nnz, |local| {
        let src_row = usize_from_u64(rows.nth(local as u64)?, "row position")?;
        if src_row >= n_rows {
            return Err(Error::invalid_argument("row position out of bounds"));
        }
        Ok(indptr[src_row + 1] - indptr[src_row])
    })?;

    let mut out_indptr = indptr_buffer(out_rows)?;
    out_indptr.push(0u64);
    for &nnz in &row_nnz {
        let next = out_indptr
            .last()
            .copied()
            .unwrap()
            .checked_add(nnz)
            .ok_or_else(|| Error::invalid_argument("CSR gather nnz overflow"))?;
        out_indptr.push(next);
    }
    let total_nnz = *out_indptr.last().unwrap();
    let mut out_indices = uninit_bytes(checked_mul(
        usize_from_u64(total_nnz, "nnz")?,
        index_size,
        "gather indices",
    )?)?;
    let mut out_data = uninit_bytes(checked_mul(
        usize_from_u64(total_nnz, "nnz")?,
        value_size,
        "gather data",
    )?)?;

    // Pass 2: copy disjoint row segments via indptr-guided stream of jobs.
    copy_csr_rows_parallel(
        threads,
        out_rows,
        &out_indptr,
        &mut out_indices,
        &mut out_data,
        index_size,
        value_size,
        |local, dst_i, dst_d| {
            let src_row = usize_from_u64(rows.nth(local as u64)?, "row position")?;
            let a = usize_from_u64(indptr[src_row], "indptr")?;
            let b = usize_from_u64(indptr[src_row + 1], "indptr")?;
            let n = b - a;
            let index_bytes = n * index_size;
            let data_bytes = n * value_size;
            if dst_i.len() != index_bytes || dst_d.len() != data_bytes {
                return Err(Error::invalid_argument(
                    "CSR gathered row output length mismatch",
                ));
            }
            if n == 0 {
                return Ok(());
            }
            // SAFETY: validated `indptr` bounds make both source ranges valid;
            // the output prefix sum produced exact destination row lengths.
            // Inputs and newly allocated outputs cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    indices.as_ptr().add(a * index_size),
                    dst_i.as_mut_ptr().cast::<u8>(),
                    index_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(a * value_size),
                    dst_d.as_mut_ptr().cast::<u8>(),
                    data_bytes,
                );
            }
            Ok(())
        },
    )?;

    // SAFETY: every row callback checked its exact destination lengths and
    // initialized both disjoint payload slices before the workers joined.
    let out_indices = unsafe { assume_init_bytes(out_indices) };
    // SAFETY: the same row coverage proof applies to the data allocation.
    let out_data = unsafe { assume_init_bytes(out_data) };

    Ok(CsrArray::from_parts_validated(
        [out_rows, n_cols],
        index_dtype,
        value_dtype,
        out_indptr,
        out_indices,
        out_data,
    ))
}

fn csr_slice_rows(
    indptr: &[u64],
    indices: &[u8],
    data: &[u8],
    start: usize,
    end: usize,
    n_cols: usize,
    index_dtype: DType,
    value_dtype: DType,
) -> Result<CsrArray> {
    let index_size = index_dtype.size();
    let value_size = value_dtype.size();
    let out_rows = end - start;
    if out_rows == 0 {
        return CsrArray::empty([0, n_cols], index_dtype, value_dtype);
    }
    let nnz_start = indptr[start];
    let nnz_end = indptr[end];
    let base = nnz_start;
    let mut out_indptr = indptr_buffer(out_rows)?;
    for &offset in &indptr[start..=end] {
        out_indptr.push(offset - base);
    }
    let i0 = usize_from_u64(nnz_start, "nnz start")? * index_size;
    let i1 = usize_from_u64(nnz_end, "nnz end")? * index_size;
    let d0 = usize_from_u64(nnz_start, "nnz start")? * value_size;
    let d1 = usize_from_u64(nnz_end, "nnz end")? * value_size;
    Ok(CsrArray::from_parts_validated(
        [out_rows, n_cols],
        index_dtype,
        value_dtype,
        out_indptr,
        copy_slice(&indices[i0..i1])?,
        copy_slice(&data[d0..d1])?,
    ))
}

/// Filter CSR columns via a column map; remaps indices to `0..n_out_cols`.
pub(crate) fn csr_filter_cols(
    indptr: &[u64],
    indices: &[u8],
    data: &[u8],
    n_rows: usize,
    n_cols: usize,
    index_dtype: DType,
    value_dtype: DType,
    cols: &NormalizedAxis,
    threads: usize,
    limits: Option<(ReadLimits, usize)>,
) -> Result<CsrArray> {
    let source_index_size = index_dtype.size();
    let value_size = value_dtype.size();
    validate_csr_layout(indptr, indices, data, n_rows, source_index_size, value_size)?;
    let n_out_cols = usize_from_u64(cols.len(), "selected columns")?;
    let out_index_dtype = output_index_dtype(n_out_cols)?;
    let out_index_size = out_index_dtype.size();

    if n_rows == 0 || n_out_cols == 0 {
        return CsrArray::empty([n_rows, n_out_cols], out_index_dtype, value_dtype);
    }

    if let Some(range) = cols.as_range() {
        if range.start == 0 && range.end == n_cols as u64 {
            return Ok(CsrArray::from_parts_validated(
                [n_rows, n_cols],
                index_dtype,
                value_dtype,
                copy_slice(indptr)?,
                copy_slice(indices)?,
                copy_slice(data)?,
            ));
        }
    }

    let gather = match cols {
        NormalizedAxis::Gather { positions } => Some(GatherColumns::new(n_cols, positions)?),
        NormalizedAxis::Strided { .. } => Some(GatherColumns::new(n_cols, &cols.to_positions())?),
        NormalizedAxis::Contiguous { .. } => None,
    };
    let col_range = cols.as_range().map(|range| (range.start, range.end));

    let mut row_nnz = zeroed_u64(n_rows)?;
    count_row_nnz(threads, n_rows, &mut row_nnz, |row| {
        let start = usize_from_u64(indptr[row], "indptr")?;
        let end = usize_from_u64(indptr[row + 1], "indptr")?;
        if let Some((first_col, past_last_col)) = col_range {
            let first = lower_bound_index(indices, start, end, source_index_size, first_col);
            let past_last =
                lower_bound_index(indices, first, end, source_index_size, past_last_col);
            return Ok((past_last - first) as u64);
        }
        let gather = gather
            .as_ref()
            .expect("non-contiguous selection has gather lookup");
        u64::try_from(gather.count_hits(indices, start, end, source_index_size)?)
            .map_err(|_| Error::invalid_argument("CSR selected nnz overflow"))
    })?;

    let mut out_indptr = indptr_buffer(n_rows)?;
    out_indptr.push(0u64);
    for &nnz in &row_nnz {
        let next = out_indptr
            .last()
            .copied()
            .unwrap()
            .checked_add(nnz)
            .ok_or_else(|| Error::invalid_argument("CSR filter nnz overflow"))?;
        out_indptr.push(next);
    }
    let total_nnz = *out_indptr.last().unwrap();
    let out_indices_len = checked_mul(
        usize_from_u64(total_nnz, "nnz")?,
        out_index_size,
        "filter indices",
    )?;
    let out_data_len = checked_mul(usize_from_u64(total_nnz, "nnz")?, value_size, "filter data")?;
    check_filter_resident_limit(
        limits,
        indptr,
        indices,
        data,
        &row_nnz,
        &out_indptr,
        gather.as_ref(),
        out_indices_len,
        out_data_len,
        threads,
    )?;
    let mut out_indices = uninit_bytes(out_indices_len)?;
    let mut out_data = uninit_bytes(out_data_len)?;

    if let Some((first_col, past_last_col)) = col_range {
        copy_csr_rows_parallel(
            threads,
            n_rows,
            &out_indptr,
            &mut out_indices,
            &mut out_data,
            out_index_size,
            value_size,
            |row, dst_i, dst_d| {
                let start = usize_from_u64(indptr[row], "indptr")?;
                let end = usize_from_u64(indptr[row + 1], "indptr")?;
                let first = lower_bound_index(indices, start, end, source_index_size, first_col);
                let past_last =
                    lower_bound_index(indices, first, end, source_index_size, past_last_col);
                let selected = past_last - first;
                let expected_indices = selected * out_index_size;
                let expected_data = selected * value_size;
                if dst_i.len() != expected_indices || dst_d.len() != expected_data {
                    return Err(Error::invalid_argument(
                        "CSR range-filter output length mismatch",
                    ));
                }
                for (cursor, position) in (first..past_last).enumerate() {
                    // SAFETY: both binary-search results stay inside this
                    // validated CSR row's packed index range. The count pass
                    // sized `dst_i` for exactly these entries, and subtracting
                    // `first_col` yields an index representable by the selected
                    // output dtype.
                    unsafe {
                        let source = read_index_unchecked(indices, position, source_index_size);
                        write_index_unchecked(
                            dst_i.as_mut_ptr().cast::<u8>(),
                            cursor,
                            out_index_size,
                            source - first_col,
                        );
                    }
                }
                // SAFETY: the exact-length check above matches the validated
                // contiguous source range, and the buffers do not overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr().add(first * value_size),
                        dst_d.as_mut_ptr().cast::<u8>(),
                        expected_data,
                    );
                }
                Ok(())
            },
        )?;
    } else {
        let gather = gather
            .as_ref()
            .expect("non-contiguous selection has gather lookup");
        copy_csr_rows_parallel_init(
            threads,
            n_rows,
            &out_indptr,
            &mut out_indices,
            &mut out_data,
            out_index_size,
            value_size,
            Vec::<(u32, usize)>::new,
            |row, dst_i, dst_d, entries| {
                let start = usize_from_u64(indptr[row], "indptr")?;
                let end = usize_from_u64(indptr[row + 1], "indptr")?;
                let required = dst_i.len() / out_index_size;
                entries.clear();
                if entries.capacity() < required {
                    entries.try_reserve_exact(required)?;
                }
                gather.collect_hits(indices, start, end, source_index_size, entries)?;
                if !gather.destinations_are_ordered() {
                    entries.sort_unstable_by_key(|&(destination, _)| destination);
                }
                if entries.len() != required || dst_d.len() != required * value_size {
                    return Err(Error::invalid_argument(
                        "CSR gather-filter output length mismatch",
                    ));
                }
                for (cursor, &(destination, source_position)) in entries.iter().enumerate() {
                    let src_offset = source_position * value_size;
                    let dst_offset = cursor * value_size;
                    debug_assert!(src_offset + value_size <= data.len());
                    debug_assert!(dst_offset + value_size <= dst_d.len());
                    // SAFETY: `destination` came from a validated output
                    // position and therefore fits `out_index_size`. The count
                    // pass made `dst_i`/`dst_d` exact, `source_position` lies in
                    // this validated row, and input/output allocations do not
                    // overlap.
                    unsafe {
                        write_index_unchecked(
                            dst_i.as_mut_ptr().cast::<u8>(),
                            cursor,
                            out_index_size,
                            u64::from(destination),
                        );
                        copy_elem_unchecked(
                            dst_d.as_mut_ptr().cast::<u8>().add(dst_offset),
                            data.as_ptr().add(src_offset),
                            value_size,
                        );
                    }
                }
                Ok(())
            },
        )?;
    }

    // SAFETY: the count pass partitioned both allocations into exact row
    // slices; every successful range/gather callback checked and initialized
    // its complete slices before the worker pool joined.
    let out_indices = unsafe { assume_init_bytes(out_indices) };
    // SAFETY: the same exact row coverage applies to every data byte.
    let out_data = unsafe { assume_init_bytes(out_data) };

    Ok(CsrArray::from_parts_validated(
        [n_rows, n_out_cols],
        out_index_dtype,
        value_dtype,
        out_indptr,
        out_indices,
        out_data,
    ))
}

fn check_filter_resident_limit(
    limits: Option<(ReadLimits, usize)>,
    indptr: &[u64],
    indices: &[u8],
    data: &[u8],
    row_nnz: &[u64],
    out_indptr: &[u64],
    gather: Option<&GatherColumns>,
    out_indices_len: usize,
    out_data_len: usize,
    threads: usize,
) -> Result<()> {
    let Some((limits, additional_resident)) = limits else {
        return Ok(());
    };
    let u64_size = std::mem::size_of::<u64>();
    let tuple_size = std::mem::size_of::<(u32, usize)>();
    let indptr_bytes = checked_mul(indptr.len(), u64_size, "CSR input indptr")?;
    let row_nnz_bytes = checked_mul(row_nnz.len(), u64_size, "CSR row counts")?;
    let out_indptr_bytes = checked_mul(out_indptr.len(), u64_size, "CSR output indptr")?;
    let gather_bytes = gather.map_or(Ok(0), |gather| {
        checked_mul(
            gather.by_source.len(),
            std::mem::size_of::<(u64, u32)>(),
            "CSR column gather map",
        )
    })?;
    let scratch_entries = top_worker_row_nnz(row_nnz, threads)?;
    let scratch_bytes = checked_mul(scratch_entries, tuple_size, "CSR gather row scratch")?;
    limits.check_decoded_sum(
        [
            indptr_bytes,
            indices.len(),
            data.len(),
            row_nnz_bytes,
            out_indptr_bytes,
            gather_bytes,
            out_indices_len,
            out_data_len,
            scratch_bytes,
            additional_resident,
        ],
        "CSR column selection resident output",
    )?;
    Ok(())
}

fn top_worker_row_nnz(row_nnz: &[u64], threads: usize) -> Result<usize> {
    let worker_count = threads.min(row_nnz.len());
    if worker_count == 0 {
        return Ok(0);
    }
    let mut largest = Vec::new();
    largest.try_reserve_exact(worker_count)?;
    for &nnz in row_nnz {
        let position = largest.partition_point(|&candidate| candidate >= nnz);
        if position < worker_count {
            largest.insert(position, nnz);
            if largest.len() > worker_count {
                largest.pop();
            }
        }
    }
    largest.into_iter().try_fold(0usize, |total, nnz| {
        total
            .checked_add(usize_from_u64(nnz, "CSR gather row scratch")?)
            .ok_or_else(|| Error::invalid_argument("CSR gather row scratch overflow"))
    })
}

/// Densify a full CSR matrix.
pub fn csr_to_dense(
    indptr: &[u64],
    indices: &[u8],
    data: &[u8],
    n_rows: usize,
    n_cols: usize,
    index_dtype: DType,
    value_dtype: DType,
    threads: usize,
) -> Result<DenseArray> {
    let cols = NormalizedAxis::Contiguous {
        start: 0,
        end: n_cols as u64,
    };
    csr_to_dense_selected_cols(
        indptr,
        indices,
        data,
        n_rows,
        n_cols,
        index_dtype,
        value_dtype,
        &cols,
        threads,
    )
}

/// Densify CSR into selected columns only (scatter values into dense rows).
pub fn csr_to_dense_selected_cols(
    indptr: &[u64],
    indices: &[u8],
    data: &[u8],
    n_rows: usize,
    n_cols: usize,
    index_dtype: DType,
    value_dtype: DType,
    cols: &NormalizedAxis,
    threads: usize,
) -> Result<DenseArray> {
    let index_size = index_dtype.size();
    let value_size = value_dtype.size();
    validate_csr_layout(indptr, indices, data, n_rows, index_size, value_size)?;

    let out_cols = usize_from_u64(cols.len(), "selected cols")?;
    let row_bytes = checked_mul(out_cols, value_size, "dense row")?;
    let out_len = checked_mul(n_rows, row_bytes, "dense output")?;
    let mut output = zeroed(out_len)?;
    if n_rows == 0 || out_cols == 0 {
        return DenseArray::from_bytes([n_rows, out_cols], value_dtype, output);
    }

    match cols {
        NormalizedAxis::Contiguous { start, end } => {
            let start = usize_from_u64(*start, "col start")?;
            let end = usize_from_u64(*end, "col end")?;
            par_for_row_blocks(
                threads,
                n_rows,
                row_bytes,
                &mut output,
                |job_start, job_end, block| {
                    for row in job_start..job_end {
                        let a = usize_from_u64(indptr[row], "indptr")?;
                        let b = usize_from_u64(indptr[row + 1], "indptr")?;
                        let dst_row_off = (row - job_start) * row_bytes;
                        debug_assert!(dst_row_off + row_bytes <= block.len());
                        let first = if start == 0 {
                            a
                        } else {
                            lower_bound_index(indices, a, b, index_size, start as u64)
                        };
                        let past_last = if end == n_cols {
                            b
                        } else {
                            lower_bound_index(indices, first, b, index_size, end as u64)
                        };
                        for pos in first..past_last {
                            // SAFETY: the lower bounds constrain `pos` to
                            // this validated row and `col` to `[start, end)`.
                            // Thus both byte offsets address complete,
                            // non-overlapping source/destination elements.
                            unsafe {
                                let col = read_index_unchecked(indices, pos, index_size) as usize;
                                let dst_offset = dst_row_off + (col - start) * value_size;
                                let src_offset = pos * value_size;
                                copy_elem_unchecked(
                                    block.as_mut_ptr().add(dst_offset),
                                    data.as_ptr().add(src_offset),
                                    value_size,
                                );
                            }
                        }
                    }
                    Ok(())
                },
            )?;
            DenseArray::from_bytes([n_rows, out_cols], value_dtype, output)
        }
        NormalizedAxis::Gather { .. } | NormalizedAxis::Strided { .. } => {
            let owned = match cols {
                NormalizedAxis::Strided { .. } => Some(cols.to_positions()),
                _ => None,
            };
            let positions = match cols {
                NormalizedAxis::Gather { positions } => positions.as_slice(),
                NormalizedAxis::Strided { .. } => owned.as_ref().map_or(&[][..], Vec::as_slice),
                NormalizedAxis::Contiguous { .. } => {
                    unreachable!("contiguous columns densify via range scan")
                }
            };
            let unique = {
                let mut sorted = Vec::new();
                sorted.try_reserve_exact(positions.len())?;
                sorted.extend_from_slice(positions);
                sorted.sort_unstable();
                sorted.windows(2).all(|w| w[0] != w[1])
            };
            if unique {
                let col_map = build_col_map(n_cols, cols)?;
                par_for_row_blocks(
                    threads,
                    n_rows,
                    row_bytes,
                    &mut output,
                    |job_start, job_end, block| {
                        for row in job_start..job_end {
                            let a = usize_from_u64(indptr[row], "indptr")?;
                            let b = usize_from_u64(indptr[row + 1], "indptr")?;
                            let dst_row_off = (row - job_start) * row_bytes;
                            debug_assert!(dst_row_off + row_bytes <= block.len());
                            let copy_mapped = |pos: usize, dst_col: u32| {
                                let dst_offset = dst_row_off + dst_col as usize * value_size;
                                let src_offset = pos * value_size;
                                // SAFETY: the validated CSR layout proves
                                // `src_offset` is a full source element;
                                // `col_map` returns only output columns and
                                // row-block partitioning makes `dst_offset`
                                // exclusive to this worker.
                                unsafe {
                                    copy_elem_unchecked(
                                        block.as_mut_ptr().add(dst_offset),
                                        data.as_ptr().add(src_offset),
                                        value_size,
                                    );
                                }
                            };
                            // SAFETY: `validate_csr_layout` and the owning
                            // `CsrArray` contract establish that `a..b` is one
                            // strictly increasing packed row.
                            unsafe {
                                col_map.visit_row_unchecked(indices, a, b, index_size, copy_mapped);
                            }
                        }
                        Ok(())
                    },
                )?;
            } else {
                let gather = GatherColumns::new(n_cols, positions)?;
                par_for_row_blocks(
                    threads,
                    n_rows,
                    row_bytes,
                    &mut output,
                    |job_start, job_end, block| {
                        for row in job_start..job_end {
                            let a = usize_from_u64(indptr[row], "indptr")?;
                            let b = usize_from_u64(indptr[row + 1], "indptr")?;
                            let dst_row_off = (row - job_start) * row_bytes;
                            debug_assert!(dst_row_off + row_bytes <= block.len());
                            for pos in a..b {
                                // SAFETY: `a..b` is a validated CSR row and the
                                // packed index width matches `index_size`.
                                let col = unsafe { read_index_unchecked(indices, pos, index_size) };
                                for &(_, dst_col) in gather.destinations(col) {
                                    let dst_offset = dst_row_off + dst_col as usize * value_size;
                                    let src_offset = pos * value_size;
                                    // SAFETY: the validated source position and
                                    // gathered destination column each address
                                    // one complete element in distinct buffers.
                                    unsafe {
                                        copy_elem_unchecked(
                                            block.as_mut_ptr().add(dst_offset),
                                            data.as_ptr().add(src_offset),
                                            value_size,
                                        );
                                    }
                                }
                            }
                        }
                        Ok(())
                    },
                )?;
            }
            DenseArray::from_bytes([n_rows, out_cols], value_dtype, output)
        }
    }
}

fn count_row_nnz<F>(threads: usize, n_rows: usize, row_nnz: &mut [u64], count: F) -> Result<()>
where
    F: Fn(usize) -> Result<u64> + Sync,
{
    if row_nnz.len() != n_rows {
        return Err(Error::invalid_argument("row count output length mismatch"));
    }
    if n_rows == 0 {
        return Ok(());
    }
    let job_count = n_rows.div_ceil(ROW_JOB);
    let mut remaining = row_nnz;
    let mut row_start = 0usize;
    parallel::try_for_each_stream(
        threads.max(1),
        job_count,
        |emit| {
            while row_start < n_rows {
                let row_end = (row_start + ROW_JOB).min(n_rows);
                let tail = std::mem::take(&mut remaining);
                let (block, tail) = tail.split_at_mut(row_end - row_start);
                remaining = tail;
                emit((row_start, block))?;
                row_start = row_end;
            }
            Ok(())
        },
        |(start, block)| {
            for (offset, slot) in block.iter_mut().enumerate() {
                *slot = count(start + offset)?;
            }
            Ok(())
        },
    )
}

/// Emit per-row jobs with exclusive index/data destination slices.
fn copy_csr_rows_parallel<F>(
    threads: usize,
    n_rows: usize,
    out_indptr: &[u64],
    out_indices: &mut [MaybeUninit<u8>],
    out_data: &mut [MaybeUninit<u8>],
    index_size: usize,
    value_size: usize,
    fill: F,
) -> Result<()>
where
    F: Fn(usize, &mut [MaybeUninit<u8>], &mut [MaybeUninit<u8>]) -> Result<()> + Sync,
{
    copy_csr_rows_parallel_init(
        threads,
        n_rows,
        out_indptr,
        out_indices,
        out_data,
        index_size,
        value_size,
        || (),
        |row, dst_i, dst_d, ()| fill(row, dst_i, dst_d),
    )
}

/// Variant of [`copy_csr_rows_parallel`] with reusable worker-local scratch.
fn copy_csr_rows_parallel_init<S, I, F>(
    threads: usize,
    n_rows: usize,
    out_indptr: &[u64],
    out_indices: &mut [MaybeUninit<u8>],
    out_data: &mut [MaybeUninit<u8>],
    index_size: usize,
    value_size: usize,
    initialize: I,
    fill: F,
) -> Result<()>
where
    S: Send,
    I: Fn() -> S + Sync,
    F: Fn(usize, &mut [MaybeUninit<u8>], &mut [MaybeUninit<u8>], &mut S) -> Result<()> + Sync,
{
    if n_rows == 0 {
        return Ok(());
    }
    let job_count = n_rows.div_ceil(ROW_JOB);
    let mut indices_remaining = out_indices;
    let mut data_remaining = out_data;
    let mut row_start = 0usize;

    parallel::try_for_each_stream_init(
        threads.max(1),
        job_count,
        |emit| {
            while row_start < n_rows {
                let row_end = (row_start + ROW_JOB).min(n_rows);
                let nnz_start = out_indptr[row_start];
                let nnz_end = out_indptr[row_end];
                let n_nnz = usize_from_u64(nnz_end - nnz_start, "job nnz")?;
                let i_bytes = n_nnz * index_size;
                let d_bytes = n_nnz * value_size;

                let i_tail = std::mem::take(&mut indices_remaining);
                let (i_block, i_tail) = i_tail.split_at_mut(i_bytes);
                indices_remaining = i_tail;

                let d_tail = std::mem::take(&mut data_remaining);
                let (d_block, d_tail) = d_tail.split_at_mut(d_bytes);
                data_remaining = d_tail;

                emit((row_start, row_end, i_block, d_block))?;
                row_start = row_end;
            }
            if !indices_remaining.is_empty() || !data_remaining.is_empty() {
                return Err(Error::invalid_argument(
                    "CSR row producer did not consume the full payload",
                ));
            }
            Ok(())
        },
        initialize,
        |(job_start, job_end, i_block, d_block), state| {
            let mut i_rest = i_block;
            let mut d_rest = d_block;
            for row in job_start..job_end {
                let n = usize_from_u64(out_indptr[row + 1] - out_indptr[row], "row nnz")?;
                let i_bytes = n * index_size;
                let d_bytes = n * value_size;
                let (dst_i, i_tail) = i_rest.split_at_mut(i_bytes);
                let (dst_d, d_tail) = d_rest.split_at_mut(d_bytes);
                i_rest = i_tail;
                d_rest = d_tail;
                fill(row, dst_i, dst_d, state)?;
            }
            Ok(())
        },
    )
}

fn validate_csr_layout(
    indptr: &[u64],
    indices: &[u8],
    data: &[u8],
    n_rows: usize,
    index_size: usize,
    value_size: usize,
) -> Result<()> {
    if indptr.len() != n_rows + 1 {
        return Err(Error::invalid_argument("CSR indptr length mismatch"));
    }
    let nnz = indptr.last().copied().unwrap_or(0);
    let i_len = checked_mul(usize_from_u64(nnz, "nnz")?, index_size, "indices")?;
    let d_len = checked_mul(usize_from_u64(nnz, "nnz")?, value_size, "data")?;
    if indices.len() != i_len || data.len() != d_len {
        return Err(Error::invalid_argument("CSR payload length mismatch"));
    }
    Ok(())
}

pub(crate) fn output_index_dtype(n_cols: usize) -> Result<DType> {
    let n_cols = n_cols as u64;
    if n_cols <= u64::from(u16::MAX) + 1 {
        Ok(DType::U16)
    } else if n_cols <= u64::from(u32::MAX) + 1 {
        Ok(DType::U32)
    } else {
        Err(Error::invalid_argument(
            "selected column count exceeds CSR u32 index capacity",
        ))
    }
}

#[inline]
fn lower_bound_index(
    indices: &[u8],
    mut start: usize,
    mut end: usize,
    index_size: usize,
    target: u64,
) -> usize {
    assert!(matches!(index_size, 2 | 4));
    assert!(
        end.checked_mul(index_size)
            .is_some_and(|bytes| bytes <= indices.len()),
        "CSR binary-search range is out of bounds"
    );
    while start < end {
        let middle = start + (end - start) / 2;
        // SAFETY: `middle < end` and the entry check above proves every index
        // in `start..end` has a complete packed representation.
        if unsafe { read_index_unchecked(indices, middle, index_size) } < target {
            start = middle + 1;
        } else {
            end = middle;
        }
    }
    start
}

fn equal_index(
    indices: &[u8],
    start: usize,
    end: usize,
    index_size: usize,
    target: u64,
) -> Option<usize> {
    let found = lower_bound_index(indices, start, end, index_size, target);
    if found < end {
        // SAFETY: `found` is inside the validated packed row passed to lower_bound.
        let value = unsafe { read_index_unchecked(indices, found, index_size) };
        if value == target {
            return Some(found);
        }
    }
    None
}

fn prefer_index_binary_search(n_queries: usize, haystack: usize) -> bool {
    if haystack < 32 || n_queries == 0 {
        return false;
    }
    let log2 = (usize::BITS - haystack.leading_zeros()) as usize;
    n_queries.saturating_mul(log2.saturating_mul(3).saturating_add(8)) < haystack
}

fn zeroed_u64(len: usize) -> Result<Vec<u64>> {
    let mut output = Vec::new();
    output.try_reserve_exact(len)?;
    output.resize(len, 0);
    Ok(output)
}

fn indptr_buffer(n_rows: usize) -> Result<Vec<u64>> {
    let capacity = n_rows
        .checked_add(1)
        .ok_or_else(|| Error::invalid_argument("CSR indptr length overflow"))?;
    let mut output = Vec::new();
    output.try_reserve_exact(capacity)?;
    Ok(output)
}

fn copy_slice<T: Copy>(source: &[T]) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output.try_reserve_exact(source.len())?;
    output.extend_from_slice(source);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::select::AxisIndex;

    fn u16_idx(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f32_data(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn gather_rows_and_filter_cols() {
        let indptr = vec![0, 2, 3, 4];
        let indices = u16_idx(&[0, 2, 1, 3]);
        let data = f32_data(&[1.0, 3.0, 2.0, 4.0]);

        let rows = AxisIndex::positions([2, 0]).normalize(3).unwrap();
        let gathered = csr_select_rows(
            &indptr,
            &indices,
            &data,
            3,
            4,
            DType::U16,
            DType::F32,
            &rows,
            2,
        )
        .unwrap();
        assert_eq!(gathered.indptr(), &[0, 1, 3]);
        assert_eq!(gathered.indices(), u16_idx(&[3, 0, 2]));

        let cols = AxisIndex::positions([2, 0]).normalize(4).unwrap();
        let filtered = csr_filter_cols(
            gathered.indptr(),
            gathered.indices(),
            gathered.data(),
            gathered.n_rows(),
            gathered.n_cols(),
            gathered.index_dtype(),
            gathered.value_dtype(),
            &cols,
            2,
            None,
        )
        .unwrap();
        assert_eq!(filtered.shape(), [2, 2]);
        assert_eq!(filtered.indptr(), &[0, 0, 2]);
    }

    #[test]
    fn strided_rows_match_expanded_positions() {
        let indptr = vec![0, 2, 3, 4];
        let indices = u16_idx(&[0, 2, 1, 3]);
        let data = f32_data(&[1.0, 3.0, 2.0, 4.0]);
        let strided = AxisIndex::strided(0, 3, 2).normalize(3).unwrap();
        let expanded = AxisIndex::positions([0, 2]).normalize(3).unwrap();
        let from_stride = csr_select_rows(
            &indptr,
            &indices,
            &data,
            3,
            4,
            DType::U16,
            DType::F32,
            &strided,
            2,
        )
        .unwrap();
        let from_gather = csr_select_rows(
            &indptr,
            &indices,
            &data,
            3,
            4,
            DType::U16,
            DType::F32,
            &expanded,
            2,
        )
        .unwrap();
        assert_eq!(from_stride.indptr(), from_gather.indptr());
        assert_eq!(from_stride.indices(), from_gather.indices());
        assert_eq!(from_stride.data(), from_gather.data());
    }

    #[test]
    fn densify_selected() {
        let indptr = vec![0, 2, 3];
        let indices = u16_idx(&[0, 2, 1]);
        let data = f32_data(&[1.0, 3.0, 2.0]);
        let cols = AxisIndex::range(0, 3).normalize(3).unwrap();
        let dense = csr_to_dense_selected_cols(
            &indptr,
            &indices,
            &data,
            2,
            3,
            DType::U16,
            DType::F32,
            &cols,
            2,
        )
        .unwrap();
        assert_eq!(dense.shape(), [2, 3]);
        assert_eq!(dense.values(), f32_data(&[1.0, 0.0, 3.0, 0.0, 2.0, 0.0]));
    }

    #[test]
    fn densify_sparse_column_map_preserves_requested_order() {
        let indptr = vec![0, 3];
        let indices = u16_idx(&[0, 2_048, 4_095]);
        let data = f32_data(&[1.0, 2.0, 3.0]);
        let cols = AxisIndex::positions([4_095, 0]).normalize(4_096).unwrap();
        let dense = csr_to_dense_selected_cols(
            &indptr,
            &indices,
            &data,
            1,
            4_096,
            DType::U16,
            DType::F32,
            &cols,
            2,
        )
        .unwrap();
        assert_eq!(dense.shape(), [1, 2]);
        assert_eq!(dense.values(), f32_data(&[3.0, 1.0]));
    }

    #[test]
    fn sparse_gather_preserves_order_and_duplicates_canonically() {
        let indptr = vec![0, 2];
        let indices = u16_idx(&[0, 2]);
        let data = f32_data(&[1.0, 3.0]);
        let cols = AxisIndex::positions([2, 0, 2]).normalize(3).unwrap();
        let selected = csr_filter_cols(
            &indptr,
            &indices,
            &data,
            1,
            3,
            DType::U16,
            DType::F32,
            &cols,
            2,
            None,
        )
        .unwrap();
        assert_eq!(selected.shape(), [1, 3]);
        assert_eq!(selected.indptr(), &[0, 3]);
        assert_eq!(selected.indices(), u16_idx(&[0, 1, 2]));
        assert_eq!(selected.data(), f32_data(&[3.0, 1.0, 3.0]));
    }

    #[test]
    fn sparse_gather_widens_output_indices_before_writing() {
        let indptr = vec![0, 1];
        let indices = u16_idx(&[1]);
        let data = f32_data(&[7.0]);
        let mut positions = (2u64..65_538).collect::<Vec<_>>();
        positions.push(1);
        let cols = AxisIndex::positions(positions).normalize(70_000).unwrap();
        let selected = csr_filter_cols(
            &indptr,
            &indices,
            &data,
            1,
            70_000,
            DType::U16,
            DType::F32,
            &cols,
            2,
            None,
        )
        .unwrap();
        assert_eq!(selected.index_dtype(), DType::U32);
        assert_eq!(selected.indices(), 65_536u32.to_le_bytes().as_slice());
        assert_eq!(selected.data(), f32_data(&[7.0]));
    }
}
