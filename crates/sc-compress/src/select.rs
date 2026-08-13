//! Axis selection for on-demand row/column loading.
//!
//! Supports the single-cell access patterns that matter in practice:
//! contiguous batches, strided slices, fancy integer indices, and boolean masks
//! (converted to indices up-front).

use std::ops::Range;

use crate::error::{Error, Result};

/// One-dimensional selection along rows or columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisIndex {
    /// Keep every element on this axis.
    All,
    /// Half-open contiguous range `[start, end)`.
    Range { start: u64, end: u64 },
    /// Python-resolved strided slice `(start, stop, step)` with a non-zero step.
    /// Negative steps accept the `-1` stop sentinel returned by `slice.indices()`.
    Strided { start: i64, stop: i64, step: i64 },
    /// Explicit positions in output order (duplicates allowed).
    Positions(Vec<u64>),
}

impl AxisIndex {
    pub fn all() -> Self {
        Self::All
    }

    pub fn range(start: u64, end: u64) -> Self {
        Self::Range { start, end }
    }

    pub fn strided(start: i64, stop: i64, step: i64) -> Self {
        Self::Strided { start, stop, step }
    }

    pub fn positions(positions: impl IntoIterator<Item = u64>) -> Self {
        Self::Positions(positions.into_iter().collect())
    }

    /// Boolean mask → positions of `true` entries.
    pub fn from_mask(mask: &[bool]) -> Self {
        let positions = mask
            .iter()
            .enumerate()
            .filter_map(|(index, keep)| keep.then_some(index as u64))
            .collect();
        Self::Positions(positions)
    }

    pub fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }

    pub fn is_contiguous(&self) -> bool {
        match self {
            Self::All | Self::Range { .. } => true,
            Self::Strided { step, .. } => *step == 1,
            Self::Positions(positions) => is_contiguous_positions(positions),
        }
    }

    /// Normalize against an axis length, resolving `All` and validating bounds.
    pub fn normalize(self, axis_len: u64) -> Result<NormalizedAxis> {
        match self {
            Self::All => Ok(NormalizedAxis::Contiguous {
                start: 0,
                end: axis_len,
            }),
            Self::Range { start, end } => {
                if start > end || end > axis_len {
                    return Err(Error::invalid_argument(format!(
                        "axis range [{start}, {end}) outside 0..{axis_len}"
                    )));
                }
                Ok(NormalizedAxis::Contiguous { start, end })
            }
            Self::Strided { start, stop, step } => {
                if step == 0 {
                    return Err(Error::invalid_argument("slice step must not be zero"));
                }
                validate_resolved_slice(start, stop, step, axis_len)?;
                if step == 1 && start <= stop {
                    return Ok(NormalizedAxis::Contiguous {
                        start: start as u64,
                        end: stop as u64,
                    });
                }
                let len = strided_len(start, stop, step)?;
                if len == 0 {
                    return Ok(NormalizedAxis::Contiguous { start: 0, end: 0 });
                }
                let first = u64::try_from(start).map_err(|_| {
                    Error::invalid_argument("strided slice produced a negative index")
                })?;
                if len == 1 {
                    return Ok(NormalizedAxis::Contiguous {
                        start: first,
                        end: first + 1,
                    });
                }
                Ok(NormalizedAxis::Strided {
                    start: first,
                    step,
                    len,
                })
            }
            Self::Positions(positions) => {
                for (i, &pos) in positions.iter().enumerate() {
                    if pos >= axis_len {
                        return Err(Error::invalid_argument(format!(
                            "axis position[{i}]={pos} outside 0..{axis_len}"
                        )));
                    }
                }
                if is_contiguous_positions(&positions) {
                    let start = positions.first().copied().unwrap_or(0);
                    let end = positions.last().map(|last| last + 1).unwrap_or(start);
                    return Ok(NormalizedAxis::Contiguous { start, end });
                }
                Ok(NormalizedAxis::Gather { positions })
            }
        }
    }
}

/// Bounds-checked axis selection ready for kernels / planners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizedAxis {
    Contiguous {
        start: u64,
        end: u64,
    },
    /// Arithmetic sequence `start + i * step` for `i in 0..len`.
    ///
    /// `step` is never `0` or `1`; `step == 1` normalizes to [`Self::Contiguous`].
    Strided {
        start: u64,
        step: i64,
        len: u64,
    },
    Gather {
        positions: Vec<u64>,
    },
}

/// One arithmetic run of selected positions: `source + i * source_step` writes
/// to output slots `destination + i` for `i in 0..count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AxisRun {
    pub source: u64,
    pub destination: u64,
    pub count: u64,
    pub source_step: i64,
}

impl AxisRun {
    pub(crate) fn nth(self, index: u64) -> Result<u64> {
        strided_nth(self.source, self.source_step, index)
    }

    /// How many terms of this run, starting at `self.source`, stay inside `[lo, hi)`.
    pub(crate) fn prefix_in_cell_range(self, lo: u64, hi: u64) -> Result<u64> {
        if self.count == 0 || self.source < lo || self.source >= hi {
            return Ok(0);
        }
        if self.source_step > 0 {
            let step = self.source_step as u64;
            let last = (hi - 1 - self.source) / step;
            Ok((last + 1).min(self.count))
        } else if self.source_step < 0 {
            let step = self.source_step.unsigned_abs();
            let last = (self.source - lo) / step;
            Ok((last + 1).min(self.count))
        } else {
            Err(Error::invalid_argument("axis run step must not be zero"))
        }
    }
}

impl NormalizedAxis {
    pub fn len(&self) -> u64 {
        match self {
            Self::Contiguous { start, end } => end - start,
            Self::Strided { len, .. } => *len,
            Self::Gather { positions } => positions.len() as u64,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_contiguous(&self) -> bool {
        matches!(self, Self::Contiguous { .. })
    }

    pub fn validate(&self, axis_len: u64) -> Result<()> {
        match self {
            Self::Contiguous { start, end } => {
                if start > end || *end > axis_len {
                    return Err(Error::invalid_argument(format!(
                        "normalized axis range [{start}, {end}) outside 0..{axis_len}"
                    )));
                }
            }
            Self::Strided { start, step, len } => {
                if *step == 0 {
                    return Err(Error::invalid_argument(
                        "strided axis step must not be zero",
                    ));
                }
                if *len == 0 {
                    return Ok(());
                }
                let last = strided_nth(*start, *step, *len - 1)?;
                if *start >= axis_len || last >= axis_len {
                    return Err(Error::invalid_argument(format!(
                        "strided axis [{start} + i*{step}; i<={}] outside 0..{axis_len}",
                        *len - 1
                    )));
                }
            }
            Self::Gather { positions } => {
                if let Some((index, position)) = positions
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, position)| *position >= axis_len)
                {
                    return Err(Error::invalid_argument(format!(
                        "normalized axis position[{index}]={position} outside 0..{axis_len}"
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn as_range(&self) -> Option<Range<u64>> {
        match self {
            Self::Contiguous { start, end } => Some(*start..*end),
            Self::Strided { .. } | Self::Gather { .. } => None,
        }
    }

    pub fn positions(&self) -> Option<&[u64]> {
        match self {
            Self::Gather { positions } => Some(positions),
            Self::Contiguous { .. } | Self::Strided { .. } => None,
        }
    }

    /// Bounding half-open range covering every selected position.
    pub fn bounding_range(&self) -> Range<u64> {
        match self {
            Self::Contiguous { start, end } => *start..*end,
            Self::Strided { start, step, len } => {
                if *len == 0 {
                    0..0
                } else {
                    let last = strided_nth(*start, *step, *len - 1).unwrap_or(*start);
                    let lo = (*start).min(last);
                    let hi = (*start).max(last);
                    lo..hi + 1
                }
            }
            Self::Gather { positions } => {
                if positions.is_empty() {
                    0..0
                } else {
                    let mut lo = positions[0];
                    let mut hi = positions[0];
                    for &p in &positions[1..] {
                        lo = lo.min(p);
                        hi = hi.max(p);
                    }
                    lo..hi + 1
                }
            }
        }
    }

    pub fn nth(&self, index: u64) -> Result<u64> {
        match self {
            Self::Contiguous { start, end } => {
                let position = start
                    .checked_add(index)
                    .ok_or_else(|| Error::invalid_argument("axis position overflow"))?;
                if position >= *end {
                    return Err(Error::invalid_argument("axis position is out of bounds"));
                }
                Ok(position)
            }
            Self::Strided { start, step, len } => {
                if index >= *len {
                    return Err(Error::invalid_argument("axis position is out of bounds"));
                }
                strided_nth(*start, *step, index)
            }
            Self::Gather { positions } => {
                let index = usize::try_from(index)
                    .map_err(|_| Error::invalid_argument("axis position exceeds usize"))?;
                positions
                    .get(index)
                    .copied()
                    .ok_or_else(|| Error::invalid_argument("axis position is out of bounds"))
            }
        }
    }

    /// Materialize explicit positions (for gather kernels).
    pub fn to_positions(&self) -> Vec<u64> {
        match self {
            Self::Contiguous { start, end } => (*start..*end).collect(),
            Self::Strided { start, step, len } => (0..*len)
                .filter_map(|index| strided_nth(*start, *step, index).ok())
                .collect(),
            Self::Gather { positions } => positions.clone(),
        }
    }

    /// Visit coalesced arithmetic runs covering this axis in output order.
    pub(crate) fn visit_runs<F>(&self, mut visit: F) -> Result<()>
    where
        F: FnMut(AxisRun) -> Result<()>,
    {
        match self {
            Self::Contiguous { start, end } => {
                let count = *end - *start;
                if count > 0 {
                    visit(AxisRun {
                        source: *start,
                        destination: 0,
                        count,
                        source_step: 1,
                    })?;
                }
                Ok(())
            }
            Self::Strided { start, step, len } => {
                if *len > 0 {
                    visit(AxisRun {
                        source: *start,
                        destination: 0,
                        count: *len,
                        source_step: *step,
                    })?;
                }
                Ok(())
            }
            Self::Gather { positions } => visit_gather_runs(positions, visit),
        }
    }
}

/// Split an arithmetic run at chunk-file boundaries.
pub(crate) fn visit_run_chunks<F>(
    run: AxisRun,
    chunk_of: impl Fn(u64) -> Result<usize>,
    cell_range: impl Fn(usize) -> Result<(u64, u64)>,
    mut visit: F,
) -> Result<()>
where
    F: FnMut(usize, AxisRun) -> Result<()>,
{
    let mut done = 0u64;
    while done < run.count {
        let source = run.nth(done)?;
        let chunk = chunk_of(source)?;
        let (lo, hi) = cell_range(chunk)?;
        let sub = AxisRun {
            source,
            destination: run
                .destination
                .checked_add(done)
                .ok_or_else(|| Error::invalid_argument("axis run destination overflow"))?,
            count: run.count - done,
            source_step: run.source_step,
        };
        let take = sub.prefix_in_cell_range(lo, hi)?;
        if take == 0 {
            return Err(Error::invalid_argument(
                "axis run did not intersect its owning chunk",
            ));
        }
        visit(
            chunk,
            AxisRun {
                source,
                destination: sub.destination,
                count: take,
                source_step: run.source_step,
            },
        )?;
        done = done
            .checked_add(take)
            .ok_or_else(|| Error::invalid_argument("axis run split overflow"))?;
    }
    Ok(())
}

/// Full matrix selection on both axes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub rows: AxisIndex,
    pub cols: AxisIndex,
}

impl Selection {
    pub fn new(rows: AxisIndex, cols: AxisIndex) -> Self {
        Self { rows, cols }
    }

    pub fn all() -> Self {
        Self {
            rows: AxisIndex::All,
            cols: AxisIndex::All,
        }
    }

    pub fn rows_only(rows: AxisIndex) -> Self {
        Self {
            rows,
            cols: AxisIndex::All,
        }
    }

    pub fn normalize(self, n_rows: u64, n_cols: u64) -> Result<NormalizedSelection> {
        Ok(NormalizedSelection {
            rows: self.rows.normalize(n_rows)?,
            cols: self.cols.normalize(n_cols)?,
        })
    }
}

/// Bounds-checked row/column selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedSelection {
    pub rows: NormalizedAxis,
    pub cols: NormalizedAxis,
}

impl NormalizedSelection {
    pub fn shape(&self) -> [u64; 2] {
        [self.rows.len(), self.cols.len()]
    }

    pub fn n_rows(&self) -> u64 {
        self.rows.len()
    }

    pub fn n_cols(&self) -> u64 {
        self.cols.len()
    }
}

/// How CSR materialization should present selected columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CsrOutput {
    /// Keep CSR layout with remapped column indices in `0..n_selected_cols`.
    #[default]
    Sparse,
    /// Densify into a row-major dense buffer.
    Dense,
}

fn is_contiguous_positions(positions: &[u64]) -> bool {
    positions
        .windows(2)
        .all(|pair| pair[0].checked_add(1) == Some(pair[1]))
}

fn validate_resolved_slice(start: i64, stop: i64, step: i64, axis_len: u64) -> Result<()> {
    let axis_len = i64::try_from(axis_len)
        .map_err(|_| Error::invalid_argument("strided slices require axis_len <= i64::MAX"))?;
    let valid = if step > 0 {
        start >= 0 && stop >= 0 && start <= axis_len && stop <= axis_len
    } else if axis_len == 0 {
        start == -1 && stop == -1
    } else {
        start >= 0 && start < axis_len && stop >= -1 && stop < axis_len
    };
    if !valid {
        return Err(Error::invalid_argument(format!(
            "resolved slice ({start}, {stop}, {step}) is outside axis length {axis_len}"
        )));
    }
    Ok(())
}

fn strided_len(start: i64, stop: i64, step: i64) -> Result<u64> {
    let count = if step > 0 && start < stop {
        ((i128::from(stop) - i128::from(start) - 1) / i128::from(step) + 1) as u128
    } else if step < 0 && start > stop {
        let step_abs = i128::from(step).unsigned_abs();
        ((i128::from(start) - i128::from(stop) - 1) as u128 / step_abs) + 1
    } else {
        0
    };
    u64::try_from(count).map_err(|_| Error::invalid_argument("strided slice length exceeds u64"))
}

pub(crate) fn strided_nth(start: u64, step: i64, index: u64) -> Result<u64> {
    try_strided_nth(start, step, index)
        .ok_or_else(|| Error::invalid_argument("strided slice produced a negative index"))
}

fn try_strided_nth(start: u64, step: i64, index: u64) -> Option<u64> {
    let value = i128::from(start).checked_add(i128::from(step).checked_mul(i128::from(index))?)?;
    u64::try_from(value).ok()
}

fn visit_gather_runs<F>(positions: &[u64], mut visit: F) -> Result<()>
where
    F: FnMut(AxisRun) -> Result<()>,
{
    let mut index = 0usize;
    while index < positions.len() {
        let source = positions[index];
        let destination = u64::try_from(index)
            .map_err(|_| Error::invalid_argument("gather run destination exceeds u64"))?;
        if index + 1 >= positions.len() {
            visit(AxisRun {
                source,
                destination,
                count: 1,
                source_step: 1,
            })?;
            break;
        }
        let step = i128::from(positions[index + 1]) - i128::from(source);
        if step == 0 {
            visit(AxisRun {
                source,
                destination,
                count: 1,
                source_step: 1,
            })?;
            index += 1;
            continue;
        }
        let source_step = i64::try_from(step)
            .map_err(|_| Error::invalid_argument("gather run step exceeds i64"))?;
        let mut count = 2u64;
        while index + (count as usize) < positions.len() {
            let Some(expected) = try_strided_nth(source, source_step, count) else {
                break;
            };
            if positions[index + (count as usize)] != expected {
                break;
            }
            count += 1;
        }
        visit(AxisRun {
            source,
            destination,
            count,
            source_step,
        })?;
        index += usize::try_from(count)
            .map_err(|_| Error::invalid_argument("gather run length exceeds usize"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_and_all_normalize() {
        assert_eq!(
            AxisIndex::All.normalize(10).unwrap(),
            NormalizedAxis::Contiguous { start: 0, end: 10 }
        );
        assert_eq!(
            AxisIndex::range(2, 7).normalize(10).unwrap(),
            NormalizedAxis::Contiguous { start: 2, end: 7 }
        );
    }

    #[test]
    fn contiguous_positions_collapse() {
        let axis = AxisIndex::positions([3, 4, 5]);
        assert_eq!(
            axis.normalize(10).unwrap(),
            NormalizedAxis::Contiguous { start: 3, end: 6 }
        );
    }

    #[test]
    fn gather_positions_preserve_order() {
        let axis = AxisIndex::positions([5, 1, 5, 0]);
        match axis.normalize(10).unwrap() {
            NormalizedAxis::Gather { positions } => assert_eq!(positions, vec![5, 1, 5, 0]),
            other => panic!("expected gather, got {other:?}"),
        }
    }

    #[test]
    fn mask_to_positions() {
        let axis = AxisIndex::from_mask(&[false, true, false, true, true]);
        match axis.normalize(5).unwrap() {
            NormalizedAxis::Gather { positions } => assert_eq!(positions, vec![1, 3, 4]),
            other => panic!("expected gather, got {other:?}"),
        }
    }

    #[test]
    fn out_of_bounds_rejected() {
        assert!(AxisIndex::range(0, 11).normalize(10).is_err());
        assert!(AxisIndex::positions([10]).normalize(10).is_err());
    }

    #[test]
    fn positive_and_negative_strides_stay_compact() {
        assert_eq!(
            AxisIndex::strided(1, 9, 3).normalize(10).unwrap(),
            NormalizedAxis::Strided {
                start: 1,
                step: 3,
                len: 3
            }
        );
        assert_eq!(
            AxisIndex::strided(9, -1, -2).normalize(10).unwrap(),
            NormalizedAxis::Strided {
                start: 9,
                step: -2,
                len: 5
            }
        );
        assert_eq!(
            AxisIndex::strided(1, 9, -1).normalize(10).unwrap(),
            NormalizedAxis::Contiguous { start: 0, end: 0 }
        );
        let strided = AxisIndex::strided(1, 9, 3).normalize(10).unwrap();
        assert_eq!(strided.nth(0).unwrap(), 1);
        assert_eq!(strided.nth(1).unwrap(), 4);
        assert_eq!(strided.nth(2).unwrap(), 7);
    }

    #[test]
    fn minimum_step_is_supported_without_negation_overflow() {
        assert_eq!(
            AxisIndex::strided(4, -1, i64::MIN).normalize(5).unwrap(),
            NormalizedAxis::Contiguous { start: 4, end: 5 }
        );
    }

    #[test]
    fn gather_positions_coalesce_into_arithmetic_runs() {
        let axis = AxisIndex::positions([0, 2, 4, 6, 9, 10])
            .normalize(12)
            .unwrap();
        let mut runs = Vec::new();
        axis.visit_runs(|run| {
            runs.push(run);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            runs,
            vec![
                AxisRun {
                    source: 0,
                    destination: 0,
                    count: 4,
                    source_step: 2
                },
                AxisRun {
                    source: 9,
                    destination: 4,
                    count: 2,
                    source_step: 1
                }
            ]
        );
    }
}
