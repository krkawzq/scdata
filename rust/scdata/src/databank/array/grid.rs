use crate::databank::error::{DataBankError, DataBankResult};

use super::dtype::DType;
use super::spec::{ArrayGridSpec, EdgeChunkLayout};
use super::storage::Chunk;

#[derive(Debug, Clone)]
pub enum ArrayGrid {
    Regular {
        chunk_shape: Vec<usize>,
        grid_shape: Vec<usize>,
        edge: EdgeChunkLayout,
    },
    Rectilinear {
        axes: Vec<RectilinearAxis>,
        grid_shape: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
pub struct RectilinearAxis {
    /// Monotonic boundaries, length = chunks_on_axis + 1.
    pub boundaries: Vec<usize>,
}

impl ArrayGrid {
    pub fn from_spec(shape: &[usize], spec: ArrayGridSpec) -> DataBankResult<Self> {
        match spec {
            ArrayGridSpec::Regular { chunk_shape, edge } => {
                validate_regular_grid_shape(shape, &chunk_shape)?;
                let mut grid_shape = Vec::with_capacity(shape.len());
                for (&dim, &chunk) in shape.iter().zip(chunk_shape.iter()) {
                    grid_shape.push(div_ceil(dim, chunk));
                }
                Ok(Self::Regular {
                    chunk_shape,
                    grid_shape,
                    edge,
                })
            }
            ArrayGridSpec::Rectilinear { axes } => {
                if axes.len() != shape.len() {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "rectilinear axis count {} does not match shape rank {}",
                        axes.len(),
                        shape.len()
                    )));
                }
                let mut parsed_axes = Vec::with_capacity(axes.len());
                let mut grid_shape = Vec::with_capacity(axes.len());
                for (axis_index, (boundaries, &dim)) in axes.into_iter().zip(shape).enumerate() {
                    validate_rectilinear_boundaries(axis_index, &boundaries, dim)?;
                    grid_shape.push(boundaries.len() - 1);
                    parsed_axes.push(RectilinearAxis { boundaries });
                }
                Ok(Self::Rectilinear {
                    axes: parsed_axes,
                    grid_shape,
                })
            }
        }
    }

    pub fn grid_shape(&self) -> &[usize] {
        match self {
            Self::Regular { grid_shape, .. } | Self::Rectilinear { grid_shape, .. } => grid_shape,
        }
    }

    pub fn regular_chunk_shape(&self) -> Option<&[usize]> {
        match self {
            Self::Regular { chunk_shape, .. } => Some(chunk_shape),
            Self::Rectilinear { .. } => None,
        }
    }

    pub fn num_chunks(&self) -> DataBankResult<usize> {
        product(self.grid_shape())
            .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk count overflow".to_string()))
    }

    #[allow(dead_code)]
    pub fn logical_chunk_index(&self, coords: &[usize]) -> DataBankResult<usize> {
        logical_chunk_index(coords, self.grid_shape())
    }

    pub fn chunk_coords(&self, chunk_index: usize) -> DataBankResult<Vec<usize>> {
        unravel_chunk_index(chunk_index, self.grid_shape())
    }

    pub fn decoded_extent_for_chunk(
        &self,
        shape: &[usize],
        chunk_index: usize,
    ) -> DataBankResult<Vec<usize>> {
        let coords = self.chunk_coords(chunk_index)?;
        match self {
            Self::Regular {
                chunk_shape, edge, ..
            } => match edge {
                EdgeChunkLayout::Padded => Ok(chunk_shape.clone()),
                EdgeChunkLayout::Cropped => regular_logical_extent(shape, chunk_shape, &coords),
            },
            Self::Rectilinear { axes, .. } => axes
                .iter()
                .zip(coords.iter())
                .map(|(axis, &coord)| {
                    axis.boundaries
                        .get(coord + 1)
                        .zip(axis.boundaries.get(coord))
                        .map(|(&end, &start)| end - start)
                        .ok_or_else(|| invalid_chunk_index(chunk_index))
                })
                .collect(),
        }
    }

    pub fn decoded_bytes_for_chunk(
        &self,
        shape: &[usize],
        dtype: DType,
        chunk_index: usize,
    ) -> DataBankResult<usize> {
        let elements =
            product(&self.decoded_extent_for_chunk(shape, chunk_index)?).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("chunk element count overflow".to_string())
            })?;
        elements
            .checked_mul(dtype.item_size())
            .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk byte size overflow".to_string()))
    }

    pub fn for_each_1d_range<F>(
        &self,
        shape: &[usize],
        dtype: DType,
        chunks: &[Chunk],
        start: usize,
        end: usize,
        push: F,
    ) -> DataBankResult<()>
    where
        F: FnMut(RangePiece) -> DataBankResult<()>,
    {
        validate_1d_range(shape, start, end, "1D range planning")?;
        if start == end {
            return Ok(());
        }

        match self {
            Self::Regular { chunk_shape, .. } => {
                let [chunk_len] = chunk_shape.as_slice() else {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "1D range planning requires 1D chunk shape, got {chunk_shape:?}"
                    )));
                };
                self.for_each_regular_1d_range(
                    shape[0], dtype, chunks, *chunk_len, start, end, push,
                )
            }
            Self::Rectilinear { axes, .. } => {
                let [axis] = axes.as_slice() else {
                    return Err(DataBankError::InvalidArrayMeta(
                        "1D range planning requires a 1D rectilinear grid".to_string(),
                    ));
                };
                self.for_each_rectilinear_1d_range(axis, dtype, chunks, start, end, push)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn for_each_regular_1d_range<F>(
        &self,
        len: usize,
        dtype: DType,
        chunks: &[Chunk],
        chunk_len: usize,
        start: usize,
        end: usize,
        mut push: F,
    ) -> DataBankResult<()>
    where
        F: FnMut(RangePiece) -> DataBankResult<()>,
    {
        let item_size = dtype.item_size();
        let mut pos = start;
        while pos < end {
            let chunk_index = pos / chunk_len;
            let chunk_start = chunk_index.checked_mul(chunk_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("1D chunk start overflow".to_string())
            })?;
            let logical_chunk_len = chunk_len.min(len - chunk_start);
            let in_chunk = pos - chunk_start;
            let elements = (end - pos).min(logical_chunk_len - in_chunk);
            let byte_start = in_chunk.checked_mul(item_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("range byte start overflow".to_string())
            })?;
            let byte_len = elements.checked_mul(item_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("range byte length overflow".to_string())
            })?;
            let byte_end = byte_start.checked_add(byte_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("range byte end overflow".to_string())
            })?;
            validate_range_inside_decoded_chunk(chunks, chunk_index, byte_end)?;
            push(RangePiece {
                chunk_index,
                byte_start,
                byte_end,
                elements,
            })?;
            pos += elements;
        }
        Ok(())
    }

    fn for_each_rectilinear_1d_range<F>(
        &self,
        axis: &RectilinearAxis,
        dtype: DType,
        chunks: &[Chunk],
        start: usize,
        end: usize,
        mut push: F,
    ) -> DataBankResult<()>
    where
        F: FnMut(RangePiece) -> DataBankResult<()>,
    {
        let item_size = dtype.item_size();
        let mut chunk_index = rectilinear_chunk_for_pos(axis, start)?;
        let mut pos = start;
        while pos < end {
            let chunk_start = axis.boundaries[chunk_index];
            let chunk_end = axis.boundaries[chunk_index + 1];
            let in_chunk = pos - chunk_start;
            let elements = (end.min(chunk_end)) - pos;
            let byte_start = in_chunk.checked_mul(item_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("rectilinear byte start overflow".to_string())
            })?;
            let byte_len = elements.checked_mul(item_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("rectilinear byte length overflow".to_string())
            })?;
            let byte_end = byte_start.checked_add(byte_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("rectilinear byte end overflow".to_string())
            })?;
            validate_range_inside_decoded_chunk(chunks, chunk_index, byte_end)?;
            push(RangePiece {
                chunk_index,
                byte_start,
                byte_end,
                elements,
            })?;
            pos += elements;
            chunk_index += usize::from(pos < end);
        }
        Ok(())
    }

    pub fn range_piece_count_1d(
        &self,
        shape: &[usize],
        start: usize,
        end: usize,
    ) -> DataBankResult<usize> {
        validate_1d_range(shape, start, end, "1D range piece count")?;
        if start == end {
            return Ok(0);
        }
        match self {
            Self::Regular { chunk_shape, .. } => {
                let [chunk_len] = chunk_shape.as_slice() else {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "1D range piece count requires 1D chunk shape, got {chunk_shape:?}"
                    )));
                };
                Ok(fixed_range_piece_count(start, end, *chunk_len))
            }
            Self::Rectilinear { axes, .. } => {
                let [axis] = axes.as_slice() else {
                    return Err(DataBankError::InvalidArrayMeta(
                        "1D range piece count requires a 1D rectilinear grid".to_string(),
                    ));
                };
                let first = rectilinear_chunk_for_pos(axis, start)?;
                let last = rectilinear_chunk_for_pos(axis, end - 1)?;
                Ok(last - first + 1)
            }
        }
    }

    pub fn physical_row_width_2d(
        &self,
        shape: &[usize],
        chunk_row: usize,
        chunk_col: usize,
    ) -> DataBankResult<usize> {
        match self {
            Self::Regular {
                chunk_shape, edge, ..
            } => {
                let [_, chunk_cols] = chunk_shape.as_slice() else {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "Dense2D requires 2D chunk shape, got {chunk_shape:?}"
                    )));
                };
                match edge {
                    EdgeChunkLayout::Padded => Ok(*chunk_cols),
                    EdgeChunkLayout::Cropped => {
                        let extent =
                            regular_logical_extent(shape, chunk_shape, &[chunk_row, chunk_col])?;
                        Ok(extent[1])
                    }
                }
            }
            Self::Rectilinear { .. } => Err(DataBankError::InvalidArrayMeta(
                "Dense2D does not support rectilinear chunk grids".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RangePiece {
    pub chunk_index: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    pub elements: usize,
}

#[allow(dead_code)]
pub fn logical_chunk_index(coords: &[usize], grid_shape: &[usize]) -> DataBankResult<usize> {
    if coords.len() != grid_shape.len() {
        return Err(DataBankError::InvalidArrayMeta(
            "chunk coord dimensionality mismatch".to_string(),
        ));
    }

    let mut index = 0usize;
    let mut stride = 1usize;
    for (&coord, &dim) in coords.iter().rev().zip(grid_shape.iter().rev()) {
        if coord >= dim {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "chunk coord {coord} is out of range for dim {dim}"
            )));
        }
        index = index
            .checked_add(coord.checked_mul(stride).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("chunk index overflow".to_string())
            })?)
            .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk index overflow".to_string()))?;
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk stride overflow".to_string()))?;
    }
    Ok(index)
}

pub(super) fn validate_array_shape(shape: &[usize]) -> DataBankResult<()> {
    if shape.is_empty() {
        return Err(DataBankError::InvalidArrayMeta(
            "shape must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_regular_grid_shape(shape: &[usize], chunk_shape: &[usize]) -> DataBankResult<()> {
    if shape.len() != chunk_shape.len() {
        return Err(DataBankError::InvalidArrayMeta(
            "shape/chunk_shape dimensionality mismatch".to_string(),
        ));
    }
    if chunk_shape.contains(&0) {
        return Err(DataBankError::InvalidArrayMeta(
            "chunk_shape entries must be nonzero".to_string(),
        ));
    }
    Ok(())
}

fn validate_rectilinear_boundaries(
    axis_index: usize,
    boundaries: &[usize],
    dim: usize,
) -> DataBankResult<()> {
    if boundaries.len() < 2 {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "rectilinear axis {axis_index} needs at least two boundaries"
        )));
    }
    if boundaries[0] != 0 {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "rectilinear axis {axis_index} must start at 0"
        )));
    }
    if *boundaries.last().unwrap() != dim {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "rectilinear axis {axis_index} final boundary is {}, expected {dim}",
            boundaries.last().unwrap()
        )));
    }
    for pair in boundaries.windows(2) {
        if pair[0] > pair[1] {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "rectilinear axis {axis_index} boundaries must be monotonic"
            )));
        }
    }
    Ok(())
}

fn validate_1d_range(
    shape: &[usize],
    start: usize,
    end: usize,
    context: &'static str,
) -> DataBankResult<()> {
    let [len] = shape else {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "{context} requires 1D array, got shape {shape:?}"
        )));
    };
    if start > end || end > *len {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "invalid 1D range [{start}, {end}) for length {len}"
        )));
    }
    Ok(())
}

fn validate_range_inside_decoded_chunk(
    chunks: &[Chunk],
    chunk_index: usize,
    byte_end: usize,
) -> DataBankResult<()> {
    let Some(chunk) = chunks.get(chunk_index) else {
        return Err(invalid_chunk_index(chunk_index));
    };
    if byte_end > chunk.decoded_bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "planned byte end {byte_end} exceeds decoded chunk size {} for chunk {chunk_index}",
            chunk.decoded_bytes
        )));
    }
    Ok(())
}

fn regular_logical_extent(
    shape: &[usize],
    chunk_shape: &[usize],
    coords: &[usize],
) -> DataBankResult<Vec<usize>> {
    if shape.len() != chunk_shape.len() || shape.len() != coords.len() {
        return Err(DataBankError::InvalidArrayMeta(
            "regular extent dimensionality mismatch".to_string(),
        ));
    }
    let mut extent = Vec::with_capacity(shape.len());
    for ((&dim, &chunk), &coord) in shape.iter().zip(chunk_shape).zip(coords) {
        let start = coord.checked_mul(chunk).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("regular chunk start overflow".to_string())
        })?;
        if start >= dim && dim != 0 {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "regular chunk coord {coord} starts past dim {dim}"
            )));
        }
        extent.push(chunk.min(dim.saturating_sub(start)));
    }
    Ok(extent)
}

fn unravel_chunk_index(chunk_index: usize, grid_shape: &[usize]) -> DataBankResult<Vec<usize>> {
    let count = product(grid_shape)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk count overflow".to_string()))?;
    if chunk_index >= count {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "invalid chunk index {chunk_index}"
        )));
    }
    let mut rem = chunk_index;
    let mut coords = vec![0usize; grid_shape.len()];
    for axis in (0..grid_shape.len()).rev() {
        let dim = grid_shape[axis];
        if dim == 0 {
            return Err(DataBankError::InvalidArrayMeta(
                "grid shape contains zero".to_string(),
            ));
        }
        coords[axis] = rem % dim;
        rem /= dim;
    }
    Ok(coords)
}

fn rectilinear_chunk_for_pos(axis: &RectilinearAxis, pos: usize) -> DataBankResult<usize> {
    let final_boundary = *axis.boundaries.last().unwrap_or(&0);
    if pos >= final_boundary {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "position {pos} is out of rectilinear axis range {final_boundary}"
        )));
    }
    let upper = axis.boundaries.partition_point(|&boundary| boundary <= pos);
    Ok(upper.saturating_sub(1))
}

pub(super) fn fixed_range_piece_count(start: usize, end: usize, chunk_len: usize) -> usize {
    debug_assert!(chunk_len > 0);
    if start == end {
        return 0;
    }
    debug_assert!(start < end);
    (end - 1) / chunk_len - start / chunk_len + 1
}

fn product(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
}

fn div_ceil(n: usize, d: usize) -> usize {
    n / d + usize::from(n % d != 0)
}

fn invalid_chunk_index(index: usize) -> DataBankError {
    DataBankError::InvalidArrayMeta(format!("invalid chunk index {index}"))
}
