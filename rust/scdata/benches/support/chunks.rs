//! Synthetic chunk fixtures for DataBank and access benchmarks.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use _scdata::databank::{
    ArrayCodecSpec, ArrayGridSpec, ArrayOrder, ArraySpec, ChunkSourceSpec, ChunkSpec, DType,
    Dense1DSpec, Dense2DSpec, EdgeChunkLayout, SparseCsrSpec,
};

use super::codecs::crc32_encode;

#[derive(Debug, Clone, Copy)]
pub struct ChunkLocation {
    pub offset: u64,
    pub len: usize,
}

#[derive(Debug, Clone)]
pub struct WrittenChunks {
    pub path: PathBuf,
    pub locations: Vec<ChunkLocation>,
}

pub fn gene_names(genes: usize) -> Vec<String> {
    (0..genes).map(|idx| format!("gene-{idx}")).collect()
}

pub fn dense2d_u32_spec(
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
    chunks: Vec<Arc<[u8]>>,
    codec: ArrayCodecSpec,
    edge: EdgeChunkLayout,
) -> Dense2DSpec {
    Dense2DSpec {
        gene_names: gene_names(genes),
        data: regular_memory_array_spec(
            vec![cells, genes],
            vec![chunk_rows, chunk_cols],
            DType::U32,
            chunks,
            codec,
            edge,
        ),
    }
}

pub fn dense1d_u32_spec(
    cells: usize,
    genes: usize,
    chunk_len: usize,
    chunks: Vec<Arc<[u8]>>,
    codec: ArrayCodecSpec,
) -> Dense1DSpec {
    Dense1DSpec {
        gene_names: gene_names(genes),
        data: regular_memory_array_spec(
            vec![cells * genes],
            vec![chunk_len],
            DType::U32,
            chunks,
            codec,
            EdgeChunkLayout::Cropped,
        ),
    }
}

pub fn dense1d_u32_rectilinear_spec(
    cells: usize,
    genes: usize,
    boundaries: Vec<usize>,
    chunks: Vec<Arc<[u8]>>,
    codec: ArrayCodecSpec,
) -> Dense1DSpec {
    Dense1DSpec {
        gene_names: gene_names(genes),
        data: rectilinear_memory_array_spec(
            vec![cells * genes],
            vec![boundaries],
            DType::U32,
            chunks,
            codec,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sparse_csr_u32_f32_spec(
    cells: usize,
    genes: usize,
    indptr: Vec<u64>,
    index_chunk_len: usize,
    data_chunk_len: usize,
    indices: Vec<Arc<[u8]>>,
    data: Vec<Arc<[u8]>>,
    codec: ArrayCodecSpec,
) -> SparseCsrSpec {
    let nnz = indptr.last().copied().unwrap_or(0) as usize;
    SparseCsrSpec {
        gene_names: gene_names(genes),
        indptr,
        indices: regular_memory_array_spec(
            vec![nnz],
            vec![index_chunk_len],
            DType::U32,
            indices,
            codec.clone(),
            EdgeChunkLayout::Cropped,
        ),
        data: regular_memory_array_spec(
            vec![nnz],
            vec![data_chunk_len],
            DType::F32,
            data,
            codec,
            EdgeChunkLayout::Cropped,
        ),
        index_dtype: DType::U32,
        num_cells: cells,
        num_genes: genes,
    }
}

pub fn regular_memory_array_spec(
    shape: Vec<usize>,
    chunk_shape: Vec<usize>,
    dtype: DType,
    chunks: Vec<Arc<[u8]>>,
    codec: ArrayCodecSpec,
    edge: EdgeChunkLayout,
) -> ArraySpec {
    let sources = chunks
        .into_iter()
        .map(|bytes| ChunkSourceSpec::Memory { bytes })
        .collect::<Vec<_>>();
    regular_array_spec(shape, chunk_shape, dtype, codec, edge, sources)
}

pub fn regular_file_array_spec(
    shape: Vec<usize>,
    chunk_shape: Vec<usize>,
    dtype: DType,
    written: &WrittenChunks,
    codec: ArrayCodecSpec,
    edge: EdgeChunkLayout,
) -> ArraySpec {
    let sources = written
        .locations
        .iter()
        .map(|location| ChunkSourceSpec::File {
            path: written.path.clone(),
            offset: location.offset,
            len: location.len,
        })
        .collect::<Vec<_>>();
    regular_array_spec(shape, chunk_shape, dtype, codec, edge, sources)
}

pub fn rectilinear_memory_array_spec(
    shape: Vec<usize>,
    axes: Vec<Vec<usize>>,
    dtype: DType,
    chunks: Vec<Arc<[u8]>>,
    codec: ArrayCodecSpec,
) -> ArraySpec {
    let chunks = chunks
        .into_iter()
        .map(|bytes| ChunkSpec {
            decoded_bytes: bytes.len(),
            source: ChunkSourceSpec::Memory { bytes },
        })
        .collect();
    ArraySpec {
        shape,
        dtype,
        order: ArrayOrder::C,
        codec,
        grid: ArrayGridSpec::Rectilinear { axes },
        chunks,
    }
}

fn regular_array_spec(
    shape: Vec<usize>,
    chunk_shape: Vec<usize>,
    dtype: DType,
    codec: ArrayCodecSpec,
    edge: EdgeChunkLayout,
    sources: Vec<ChunkSourceSpec>,
) -> ArraySpec {
    let grid_shape = shape
        .iter()
        .zip(chunk_shape.iter())
        .map(|(&dim, &chunk)| dim.div_ceil(chunk))
        .collect::<Vec<_>>();
    let expected_chunks = grid_shape.iter().product::<usize>();
    assert_eq!(sources.len(), expected_chunks, "chunk count mismatch");

    let chunks = sources
        .into_iter()
        .enumerate()
        .map(|(chunk_index, source)| ChunkSpec {
            source,
            decoded_bytes: regular_chunk_decoded_bytes(
                &shape,
                &chunk_shape,
                &grid_shape,
                dtype,
                edge,
                chunk_index,
            ),
        })
        .collect();

    ArraySpec {
        shape,
        dtype,
        order: ArrayOrder::C,
        codec,
        grid: ArrayGridSpec::Regular { chunk_shape, edge },
        chunks,
    }
}

fn regular_chunk_decoded_bytes(
    shape: &[usize],
    chunk_shape: &[usize],
    grid_shape: &[usize],
    dtype: DType,
    edge: EdgeChunkLayout,
    mut chunk_index: usize,
) -> usize {
    let mut coords = vec![0; grid_shape.len()];
    for axis in (0..grid_shape.len()).rev() {
        coords[axis] = chunk_index % grid_shape[axis];
        chunk_index /= grid_shape[axis];
    }
    let elements = coords
        .iter()
        .enumerate()
        .map(|(axis, &coord)| match edge {
            EdgeChunkLayout::Padded => chunk_shape[axis],
            EdgeChunkLayout::Cropped => {
                let start = coord * chunk_shape[axis];
                (shape[axis] - start).min(chunk_shape[axis])
            }
        })
        .product::<usize>();
    elements * dtype.item_size()
}

pub fn make_dense_u32_chunks(
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) -> Vec<Arc<[u8]>> {
    let row_chunks = cells.div_ceil(chunk_rows);
    let col_chunks = genes.div_ceil(chunk_cols);
    let mut chunks = Vec::with_capacity(row_chunks * col_chunks);
    for row_chunk in 0..row_chunks {
        for col_chunk in 0..col_chunks {
            let mut bytes =
                Vec::with_capacity(chunk_rows * chunk_cols * std::mem::size_of::<u32>());
            for row_in_chunk in 0..chunk_rows {
                let cell = row_chunk * chunk_rows + row_in_chunk;
                for col_in_chunk in 0..chunk_cols {
                    let gene = col_chunk * chunk_cols + col_in_chunk;
                    let value = if cell < cells && gene < genes {
                        ((cell as u32) << 16) ^ gene as u32
                    } else {
                        0
                    };
                    bytes.extend_from_slice(&value.to_ne_bytes());
                }
            }
            chunks.push(bytes.into());
        }
    }
    chunks
}

pub fn make_dense_u32_chunks_lz4(
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) -> Vec<Arc<[u8]>> {
    make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols)
        .into_iter()
        .map(|chunk| lz4_flex::block::compress_prepend_size(&chunk).into())
        .collect()
}

pub fn make_dense1d_u32_chunks(cells: usize, genes: usize, chunk_len: usize) -> Vec<Arc<[u8]>> {
    let total = cells * genes;
    let mut chunks = Vec::with_capacity(total.div_ceil(chunk_len));
    let mut bytes = Vec::with_capacity(chunk_len * std::mem::size_of::<u32>());
    for idx in 0..total {
        let cell = idx / genes;
        let gene = idx % genes;
        let value = ((cell as u32) << 16) ^ gene as u32;
        bytes.extend_from_slice(&value.to_ne_bytes());
        if (idx + 1) % chunk_len == 0 {
            chunks.push(std::mem::take(&mut bytes).into());
            bytes = Vec::with_capacity(chunk_len * std::mem::size_of::<u32>());
        }
    }
    if !bytes.is_empty() {
        chunks.push(bytes.into());
    }
    chunks
}

pub fn make_dense1d_u32_variable_chunks(
    cells: usize,
    genes: usize,
    boundaries: &[usize],
) -> Vec<Arc<[u8]>> {
    let total = cells * genes;
    assert_eq!(boundaries.first().copied(), Some(0));
    assert_eq!(boundaries.last().copied(), Some(total));
    boundaries
        .windows(2)
        .map(|window| {
            let mut bytes =
                Vec::with_capacity((window[1] - window[0]) * std::mem::size_of::<u32>());
            for idx in window[0]..window[1] {
                let cell = idx / genes;
                let gene = idx % genes;
                let value = ((cell as u32) << 16) ^ gene as u32;
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            bytes.into()
        })
        .collect()
}

pub type CsrU32F32Chunks = (Vec<u64>, Vec<Arc<[u8]>>, Vec<Arc<[u8]>>);

pub fn make_csr_u32_f32_chunks(
    cells: usize,
    genes: usize,
    nnz_per_cell: usize,
    chunk_len: usize,
) -> CsrU32F32Chunks {
    let (indptr, indices, data) = make_csr_values(cells, genes, nnz_per_cell);
    (
        indptr,
        encode_u32_chunks_raw(&indices, chunk_len),
        encode_f32_chunks_raw(&data, chunk_len),
    )
}

pub fn make_csr_u32_f32_chunks_lz4(
    cells: usize,
    genes: usize,
    nnz_per_cell: usize,
    chunk_len: usize,
) -> CsrU32F32Chunks {
    let (indptr, indices, data) = make_csr_values(cells, genes, nnz_per_cell);
    (
        indptr,
        encode_u32_chunks_lz4(&indices, chunk_len),
        encode_f32_chunks_lz4(&data, chunk_len),
    )
}

pub fn make_csr_u32_f32_chunks_crc32(
    cells: usize,
    genes: usize,
    nnz_per_cell: usize,
    chunk_len: usize,
) -> CsrU32F32Chunks {
    let (indptr, indices, data) = make_csr_values(cells, genes, nnz_per_cell);
    (
        indptr,
        encode_u32_chunks_crc32(&indices, chunk_len),
        encode_f32_chunks_crc32(&data, chunk_len),
    )
}

fn make_csr_values(
    cells: usize,
    genes: usize,
    nnz_per_cell: usize,
) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let nnz = cells * nnz_per_cell;
    let mut indptr = Vec::with_capacity(cells + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    indptr.push(0);
    for cell in 0..cells {
        for k in 0..nnz_per_cell {
            indices.push(((cell * nnz_per_cell + k) % genes) as u32);
            data.push(cell as f32 + k as f32 * 0.1);
        }
        indptr.push(indptr.last().copied().unwrap() + nnz_per_cell as u64);
    }
    (indptr, indices, data)
}

pub fn encode_u32_chunks_raw(values: &[u32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    values
        .chunks(chunk_len)
        .map(|chunk| {
            let mut bytes = Vec::with_capacity(std::mem::size_of_val(chunk));
            for value in chunk {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            bytes.into()
        })
        .collect()
}

pub fn encode_f32_chunks_raw(values: &[f32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    values
        .chunks(chunk_len)
        .map(|chunk| {
            let mut bytes = Vec::with_capacity(std::mem::size_of_val(chunk));
            for value in chunk {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            bytes.into()
        })
        .collect()
}

pub fn encode_u32_chunks_lz4(values: &[u32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    encode_u32_chunks_raw(values, chunk_len)
        .into_iter()
        .map(|chunk| lz4_flex::block::compress_prepend_size(&chunk).into())
        .collect()
}

pub fn encode_f32_chunks_lz4(values: &[f32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    encode_f32_chunks_raw(values, chunk_len)
        .into_iter()
        .map(|chunk| lz4_flex::block::compress_prepend_size(&chunk).into())
        .collect()
}

pub fn encode_u32_chunks_crc32(values: &[u32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    encode_u32_chunks_raw(values, chunk_len)
        .into_iter()
        .map(|chunk| crc32_encode(&chunk).into())
        .collect()
}

pub fn encode_f32_chunks_crc32(values: &[f32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    encode_f32_chunks_raw(values, chunk_len)
        .into_iter()
        .map(|chunk| crc32_encode(&chunk).into())
        .collect()
}

pub fn write_chunks_file(label: &str, chunks: &[Arc<[u8]>]) -> WrittenChunks {
    let path = super::bench_data_dir().join(format!("{label}-{}.bin", std::process::id()));
    write_chunks_to_path(path, chunks)
}

pub fn write_chunks_to_path(path: impl AsRef<Path>, chunks: &[Arc<[u8]>]) -> WrittenChunks {
    let path = path.as_ref().to_path_buf();
    let mut file = std::fs::File::create(&path).expect("create chunk file");
    let mut offset = 0u64;
    let mut locations = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        use std::io::Write;
        file.write_all(chunk).expect("write chunk");
        locations.push(ChunkLocation {
            offset,
            len: chunk.len(),
        });
        offset += chunk.len() as u64;
    }
    file.sync_all().expect("sync chunk file");
    WrittenChunks { path, locations }
}
