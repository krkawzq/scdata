use std::path::PathBuf;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::codecs::{codec_pipeline_from_zarr_v2_json_str, SharedCodec};
use crate::databank::{
    ArrayCodecSpec, ArrayGridSpec, ArrayOrder, ArraySpec, ChunkSourceSpec, ChunkSpec, DType,
    EdgeChunkLayout,
};

pub(crate) fn build_array_spec(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    store_path: &str,
) -> PyResult<ArraySpec> {
    let shape: Vec<usize> = data.getattr("shape")?.extract()?;
    let chunk_shape: Vec<usize> = data.getattr("chunk_shape")?.extract()?;
    if shape.len() != chunk_shape.len() {
        return Err(PyValueError::new_err(format!(
            "shape rank {} != chunk_shape rank {}",
            shape.len(),
            chunk_shape.len()
        )));
    }
    let dtype = super::dtype::extract_dtype(&data.getattr("dtype")?)?;
    let codec = build_codec(py, &data.getattr("codec")?)?;
    let grid = build_grid_spec(data, &shape, chunk_shape)?;
    let decoded_bytes = chunk_decoded_bytes(&shape, dtype, &grid)?;
    let chunks = build_chunks(data, store_path, &decoded_bytes)?;

    Ok(ArraySpec {
        shape,
        dtype,
        order: ArrayOrder::C,
        codec,
        grid,
        chunks,
    })
}

pub(crate) fn extract_u64_vec(obj: &Bound<'_, PyAny>, context: &str) -> PyResult<Vec<u64>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
        let slice = array.as_slice().map_err(|_| {
            PyValueError::new_err(format!("{context} must be a contiguous 1D uint64 array"))
        })?;
        return Ok(slice.to_vec());
    }
    obj.extract::<Vec<u64>>()
        .map_err(|err| PyValueError::new_err(format!("{context}: {err}")))
}

pub(crate) fn u64_to_usize(value: u64, context: &str) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("{context} value {value} does not fit in usize"))
    })
}

pub(crate) fn build_shared_codec(
    py: Python<'_>,
    codec: &Bound<'_, PyAny>,
) -> PyResult<Option<SharedCodec>> {
    match build_codec(py, codec)? {
        ArrayCodecSpec::Uncompressed => Ok(None),
        ArrayCodecSpec::ZarrV2Json {
            filters,
            compressor,
        } => codec_pipeline_from_zarr_v2_json_str(filters.as_deref(), compressor.as_deref())
            .map(Some)
            .map_err(|err| PyValueError::new_err(err.to_string())),
        other => Err(PyValueError::new_err(format!(
            "unsupported index codec metadata: {other:?}"
        ))),
    }
}

fn build_grid_spec(
    data: &Bound<'_, PyAny>,
    shape: &[usize],
    chunk_shape: Vec<usize>,
) -> PyResult<ArrayGridSpec> {
    let axes = optional_chunk_boundaries(data)?;
    if let Some(axes) = axes {
        validate_rectilinear_axes(shape, &axes)?;
        return Ok(ArrayGridSpec::Rectilinear { axes });
    }
    validate_regular_grid(shape, &chunk_shape)?;
    Ok(ArrayGridSpec::Regular {
        chunk_shape,
        edge: extract_edge_layout(data)?,
    })
}

fn optional_chunk_boundaries(data: &Bound<'_, PyAny>) -> PyResult<Option<Vec<Vec<usize>>>> {
    match data.getattr("chunk_boundaries") {
        Ok(value) => {
            let axes: Vec<Vec<usize>> = value.extract()?;
            if axes.is_empty() {
                Ok(None)
            } else {
                Ok(Some(axes))
            }
        }
        Err(_) => Ok(None),
    }
}

fn extract_edge_layout(data: &Bound<'_, PyAny>) -> PyResult<EdgeChunkLayout> {
    let Ok(value) = data.getattr("edge") else {
        return Ok(EdgeChunkLayout::Padded);
    };
    let text: String = value.extract()?;
    match text.to_ascii_lowercase().as_str() {
        "padded" => Ok(EdgeChunkLayout::Padded),
        "cropped" => Ok(EdgeChunkLayout::Cropped),
        other => Err(PyValueError::new_err(format!(
            "unknown edge layout {other:?}; use 'cropped' or 'padded'"
        ))),
    }
}

fn build_chunks(
    data: &Bound<'_, PyAny>,
    store_path: &str,
    decoded_bytes: &[usize],
) -> PyResult<Vec<ChunkSpec>> {
    let store_kind: String = data.getattr("store_kind")?.extract()?;
    let ranges = match store_kind.as_str() {
        "file" => file_ranges(data, store_path)?,
        "dir" => directory_ranges(data, store_path)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown store_kind {other:?} (expected 'file' or 'dir')"
            )))
        }
    };
    if ranges.len() != decoded_bytes.len() {
        return Err(PyValueError::new_err(format!(
            "chunk source count {} != chunk grid size {}",
            ranges.len(),
            decoded_bytes.len()
        )));
    }
    Ok(ranges
        .into_iter()
        .zip(decoded_bytes)
        .map(|((path, offset, len), &decoded_bytes)| ChunkSpec {
            source: ChunkSourceSpec::File { path, offset, len },
            decoded_bytes,
        })
        .collect())
}

fn file_ranges(data: &Bound<'_, PyAny>, store_path: &str) -> PyResult<Vec<(PathBuf, u64, usize)>> {
    let payload_path: String = data.getattr("payload_path")?.extract()?;
    let payload_file_path = optional_string_attr(data, "payload_file_path")?;
    let path = if payload_file_path.is_empty() {
        PathBuf::from(store_path).join(payload_path)
    } else {
        PathBuf::from(payload_file_path)
    };
    let ranges = match extract_ranges_from_offset_arrays(data)? {
        Some(ranges) => ranges,
        None => extract_ranges_from_chunks(&data.getattr("chunks")?)?,
    };
    Ok(ranges
        .into_iter()
        .map(|(offset, len)| (path.clone(), offset, len))
        .collect())
}

fn directory_ranges(
    data: &Bound<'_, PyAny>,
    store_path: &str,
) -> PyResult<Vec<(PathBuf, u64, usize)>> {
    let chunk_paths: Vec<String> = data.getattr("chunk_paths")?.extract()?;
    let n = chunk_paths.len();
    let chunk_file_paths = match data.getattr("chunk_file_paths") {
        Ok(paths) => {
            let values: Vec<String> = paths.extract()?;
            if values.is_empty() {
                None
            } else {
                Some(values)
            }
        }
        Err(_) => None,
    };
    let chunk_file_count = chunk_file_paths.as_ref().map_or(0, Vec::len);
    if chunk_file_count != 0 && chunk_file_count != n {
        return Err(PyValueError::new_err(format!(
            "chunk_file_paths count {chunk_file_count} != chunk_paths count {n}"
        )));
    }
    let chunk_offsets =
        extract_optional_u64_vec_attr(data, "chunk_offsets")?.unwrap_or_else(|| vec![0; n]);
    let chunk_lengths = extract_u64_vec(&data.getattr("chunk_lengths")?, "chunk_lengths")?;
    if chunk_offsets.len() != n {
        return Err(PyValueError::new_err(format!(
            "chunk_offsets count {} != chunk_paths count {n}",
            chunk_offsets.len()
        )));
    }
    if chunk_lengths.len() != n {
        return Err(PyValueError::new_err(format!(
            "chunk_lengths count {} != chunk_paths count {n}",
            chunk_lengths.len()
        )));
    }

    let store_root = PathBuf::from(store_path);
    let mut out = Vec::with_capacity(n);
    for (i, rel) in chunk_paths.into_iter().enumerate() {
        let path = if let Some(paths) = &chunk_file_paths {
            PathBuf::from(&paths[i])
        } else {
            store_root.join(rel)
        };
        out.push((
            path,
            chunk_offsets[i],
            u64_to_usize(chunk_lengths[i], "chunk_lengths")?,
        ));
    }
    Ok(out)
}

fn extract_ranges_from_offset_arrays(
    data: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<(u64, usize)>>> {
    let Some(offsets) = extract_optional_u64_vec_attr(data, "chunk_offsets")? else {
        return Ok(None);
    };
    let lengths = extract_u64_vec(&data.getattr("chunk_lengths")?, "chunk_lengths")?;
    if offsets.len() != lengths.len() {
        return Err(PyValueError::new_err(format!(
            "chunk_offsets length {} != chunk_lengths length {}",
            offsets.len(),
            lengths.len()
        )));
    }
    offsets
        .into_iter()
        .zip(lengths)
        .map(|(offset, len)| Ok((offset, u64_to_usize(len, "chunk_lengths")?)))
        .collect::<PyResult<_>>()
        .map(Some)
}

fn extract_ranges_from_chunks(chunks: &Bound<'_, PyAny>) -> PyResult<Vec<(u64, usize)>> {
    let n = chunks.len()?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let item = chunks.get_item(i)?;
        let offset: u64 = item.getattr("offset")?.extract()?;
        let len: usize = item.getattr("length")?.extract()?;
        out.push((offset, len));
    }
    Ok(out)
}

fn extract_optional_u64_vec_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<Vec<u64>>> {
    match obj.getattr(name) {
        Ok(value) => Ok(Some(extract_u64_vec(&value, name)?)),
        Err(_) => Ok(None),
    }
}

fn optional_string_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    match obj.getattr(name) {
        Ok(value) => value.extract(),
        Err(_) => Ok(String::new()),
    }
}

fn build_codec(py: Python<'_>, codec: &Bound<'_, PyAny>) -> PyResult<ArrayCodecSpec> {
    let is_uncompressed: bool = codec.getattr("is_uncompressed")?.extract()?;
    if is_uncompressed {
        return Ok(ArrayCodecSpec::Uncompressed);
    }

    let pair = codec.getattr("to_zarr")?.call0()?;
    let filters_obj = pair.get_item(0)?;
    let compressor_obj = pair.get_item(1)?;

    let json = py.import("json")?;
    let dumps = json.getattr("dumps")?;
    let filters = json_opt(&dumps, &filters_obj)?;
    let compressor = json_opt(&dumps, &compressor_obj)?;
    Ok(ArrayCodecSpec::ZarrV2Json {
        filters,
        compressor,
    })
}

fn json_opt(dumps: &Bound<'_, PyAny>, obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if obj.is_none() {
        return Ok(None);
    }
    Ok(Some(dumps.call1((obj.clone().unbind(),))?.extract()?))
}

fn validate_regular_grid(shape: &[usize], chunk_shape: &[usize]) -> PyResult<()> {
    if shape.is_empty() {
        return Err(PyValueError::new_err("array shape must not be empty"));
    }
    if shape.len() != chunk_shape.len() {
        return Err(PyValueError::new_err(format!(
            "shape rank {} != chunk_shape rank {}",
            shape.len(),
            chunk_shape.len()
        )));
    }
    for (axis, &chunk) in chunk_shape.iter().enumerate() {
        if chunk == 0 {
            return Err(PyValueError::new_err(format!(
                "chunk_shape[{axis}] must be greater than 0"
            )));
        }
    }
    Ok(())
}

fn validate_rectilinear_axes(shape: &[usize], axes: &[Vec<usize>]) -> PyResult<()> {
    if axes.len() != shape.len() {
        return Err(PyValueError::new_err(format!(
            "chunk_boundaries rank {} != shape rank {}",
            axes.len(),
            shape.len()
        )));
    }
    for (axis, (bounds, &dim)) in axes.iter().zip(shape).enumerate() {
        if bounds.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "chunk_boundaries[{axis}] must contain at least two entries"
            )));
        }
        if bounds.first().copied() != Some(0) {
            return Err(PyValueError::new_err(format!(
                "chunk_boundaries[{axis}] must start at 0"
            )));
        }
        if bounds.last().copied() != Some(dim) {
            return Err(PyValueError::new_err(format!(
                "chunk_boundaries[{axis}] final boundary must equal shape {dim}"
            )));
        }
        if bounds.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PyValueError::new_err(format!(
                "chunk_boundaries[{axis}] must be strictly increasing"
            )));
        }
    }
    Ok(())
}

fn chunk_decoded_bytes(
    shape: &[usize],
    dtype: DType,
    grid: &ArrayGridSpec,
) -> PyResult<Vec<usize>> {
    let grid_shape = grid_shape(shape, grid)?;
    let count = grid_shape.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value)
            .ok_or_else(|| PyValueError::new_err("chunk grid size overflow"))
    })?;
    let mut out = Vec::with_capacity(count);
    for chunk_index in 0..count {
        let coords = chunk_coords(chunk_index, &grid_shape);
        let elements = match grid {
            ArrayGridSpec::Regular { chunk_shape, edge } => {
                regular_chunk_elements(shape, chunk_shape, *edge, &coords)?
            }
            ArrayGridSpec::Rectilinear { axes } => rectilinear_chunk_elements(axes, &coords)?,
        };
        out.push(elements.checked_mul(dtype.item_size()).ok_or_else(|| {
            PyValueError::new_err(format!("chunk {chunk_index} decoded byte size overflow"))
        })?);
    }
    Ok(out)
}

fn grid_shape(shape: &[usize], grid: &ArrayGridSpec) -> PyResult<Vec<usize>> {
    match grid {
        ArrayGridSpec::Regular { chunk_shape, .. } => Ok(shape
            .iter()
            .zip(chunk_shape)
            .map(|(&dim, &chunk)| div_ceil(dim, chunk))
            .collect()),
        ArrayGridSpec::Rectilinear { axes } => Ok(axes.iter().map(|axis| axis.len() - 1).collect()),
    }
}

fn regular_chunk_elements(
    shape: &[usize],
    chunk_shape: &[usize],
    edge: EdgeChunkLayout,
    coords: &[usize],
) -> PyResult<usize> {
    shape.iter().zip(chunk_shape).zip(coords).try_fold(
        1usize,
        |elements, ((&dim, &chunk), &coord)| {
            let extent = match edge {
                EdgeChunkLayout::Padded => chunk,
                EdgeChunkLayout::Cropped => {
                    let start = coord.checked_mul(chunk).ok_or_else(|| {
                        PyValueError::new_err("chunk coordinate multiplication overflow")
                    })?;
                    dim.saturating_sub(start).min(chunk)
                }
            };
            elements
                .checked_mul(extent)
                .ok_or_else(|| PyValueError::new_err("chunk element count overflow"))
        },
    )
}

fn rectilinear_chunk_elements(axes: &[Vec<usize>], coords: &[usize]) -> PyResult<usize> {
    axes.iter()
        .zip(coords)
        .try_fold(1usize, |elements, (bounds, &coord)| {
            let extent = bounds[coord + 1] - bounds[coord];
            elements
                .checked_mul(extent)
                .ok_or_else(|| PyValueError::new_err("chunk element count overflow"))
        })
}

fn chunk_coords(mut index: usize, grid_shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; grid_shape.len()];
    for axis in (0..grid_shape.len()).rev() {
        let dim = grid_shape[axis];
        coords[axis] = index % dim;
        index /= dim;
    }
    coords
}

fn div_ceil(n: usize, d: usize) -> usize {
    n / d + usize::from(n % d != 0)
}
