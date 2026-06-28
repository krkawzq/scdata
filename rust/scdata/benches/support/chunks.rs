//! Dense / CSR / directory chunk generators and file writers shared by the
//! `modules`, `stress`, and `fullchain` benches.
//!
//! Values are deterministic so checksums are stable across runs. Dense chunks
//! encode `(cell, gene)` pairs as `((cell as u32) << 16) ^ gene as u32`; CSR
//! rows place `nnz_per_cell` entries per cell with indices cycling through the
//! gene axis.

use std::io::{Cursor, Write};
use std::path::Path;
use std::sync::Arc;

use _scdata::databank::{DirectoryChunkLocationMeta, FileChunkLocation};

use super::codecs::crc32_encode;

/// Dense2D chunks in C order: outer loop over row-chunks, inner over col-chunks,
/// each chunk row-major `[chunk_rows, chunk_cols]` of `u32`.
pub fn make_dense_u32_chunks(
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) -> Vec<Arc<[u8]>> {
    let row_chunks = cells / chunk_rows;
    let col_chunks = genes / chunk_cols;
    let mut chunks = Vec::with_capacity(row_chunks * col_chunks);

    for row_chunk in 0..row_chunks {
        for col_chunk in 0..col_chunks {
            let mut bytes =
                Vec::with_capacity(chunk_rows * chunk_cols * std::mem::size_of::<u32>());
            for row_in_chunk in 0..chunk_rows {
                let cell = row_chunk * chunk_rows + row_in_chunk;
                for col_in_chunk in 0..chunk_cols {
                    let gene = col_chunk * chunk_cols + col_in_chunk;
                    let value = ((cell as u32) << 16) ^ gene as u32;
                    bytes.extend_from_slice(&value.to_ne_bytes());
                }
            }
            chunks.push(Arc::from(bytes.into_boxed_slice()));
        }
    }

    chunks
}

pub fn make_dense_u32_chunks_zstd(
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
    level: i32,
) -> Vec<Arc<[u8]>> {
    make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols)
        .into_iter()
        .map(|raw| {
            let encoded = zstd::encode_all(Cursor::new(&raw[..]), level).expect("zstd encode");
            Arc::from(encoded.into_boxed_slice())
        })
        .collect()
}

/// Cell-major 1D layout: one logical `[cells * genes]` array split into
/// `chunk_len`-element chunks.
pub fn make_dense1d_u32_chunks(cells: usize, genes: usize, chunk_len: usize) -> Vec<Arc<[u8]>> {
    let total = cells * genes;
    let mut chunks = Vec::with_capacity(total.div_ceil(chunk_len));
    let mut bytes = Vec::with_capacity(chunk_len * std::mem::size_of::<u32>());
    for idx in 0..total {
        let cell = idx / genes;
        let gene = idx % genes;
        let value = ((cell as u32) << 16) ^ gene as u32;
        bytes.extend_from_slice(&value.to_ne_bytes());
        if bytes.len() == chunk_len * std::mem::size_of::<u32>() {
            chunks.push(Arc::from(bytes.clone().into_boxed_slice()));
            bytes.clear();
        }
    }
    if !bytes.is_empty() {
        chunks.push(Arc::from(bytes.into_boxed_slice()));
    }
    chunks
}

/// Single-chunk CSR (indices + data each in one chunk). Used by the
/// `modules`/`stress` single-chunk CSR benches.
pub fn make_csr_u32_f32_chunks(
    cells: usize,
    genes: usize,
    nnz_per_cell: usize,
) -> (Vec<u64>, Arc<[u8]>, Arc<[u8]>) {
    let nnz = cells * nnz_per_cell;
    let mut indptr = Vec::with_capacity(cells + 1);
    let mut indices_bytes = Vec::with_capacity(nnz * std::mem::size_of::<u32>());
    let mut data_bytes = Vec::with_capacity(nnz * std::mem::size_of::<f32>());
    indptr.push(0);
    for cell in 0..cells {
        for k in 0..nnz_per_cell {
            let gene = ((cell * nnz_per_cell + k) % genes) as u32;
            let value = cell as f32 + k as f32 * 0.1;
            indices_bytes.extend_from_slice(&gene.to_ne_bytes());
            data_bytes.extend_from_slice(&value.to_ne_bytes());
        }
        indptr.push(indptr.last().copied().unwrap() + nnz_per_cell as u64);
    }
    (
        indptr,
        Arc::from(indices_bytes.into_boxed_slice()),
        Arc::from(data_bytes.into_boxed_slice()),
    )
}

/// Chunked CSR with raw (uncompressed) `u32`/`f32` chunks of `chunk_len`
/// elements each. Used by the fullchain bench.
pub fn make_csr_u32_f32_chunked_raw(
    cells: usize,
    genes: usize,
    nnz_per_cell: usize,
    chunk_len: usize,
) -> (Vec<u64>, Vec<Arc<[u8]>>, Vec<Arc<[u8]>>) {
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
    (
        indptr,
        encode_u32_chunks_raw(&indices, chunk_len),
        encode_f32_chunks_raw(&data, chunk_len),
    )
}

/// Chunked CSR with lz4-compressed `u32`/`f32` chunks (size-prefixed).
pub fn make_csr_u32_f32_chunks_lz4(
    cells: usize,
    genes: usize,
    nnz_per_cell: usize,
    chunk_len: usize,
) -> (Vec<u64>, Vec<Arc<[u8]>>, Vec<Arc<[u8]>>) {
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
    (
        indptr,
        encode_u32_chunks_lz4(&indices, chunk_len),
        encode_f32_chunks_lz4(&data, chunk_len),
    )
}

pub fn encode_u32_chunks_raw(values: &[u32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    values
        .chunks(chunk_len)
        .map(|chunk| {
            let mut raw = Vec::with_capacity(std::mem::size_of_val(chunk));
            for value in chunk {
                raw.extend_from_slice(&value.to_ne_bytes());
            }
            Arc::from(raw.into_boxed_slice())
        })
        .collect()
}

pub fn encode_f32_chunks_raw(values: &[f32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    values
        .chunks(chunk_len)
        .map(|chunk| {
            let mut raw = Vec::with_capacity(std::mem::size_of_val(chunk));
            for value in chunk {
                raw.extend_from_slice(&value.to_ne_bytes());
            }
            Arc::from(raw.into_boxed_slice())
        })
        .collect()
}

pub fn encode_u32_chunks_lz4(values: &[u32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    values
        .chunks(chunk_len)
        .map(|chunk| {
            let mut raw = Vec::with_capacity(std::mem::size_of_val(chunk));
            for value in chunk {
                raw.extend_from_slice(&value.to_ne_bytes());
            }
            Arc::from(lz4_flex::block::compress_prepend_size(&raw).into_boxed_slice())
        })
        .collect()
}

pub fn encode_f32_chunks_lz4(values: &[f32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    values
        .chunks(chunk_len)
        .map(|chunk| {
            let mut raw = Vec::with_capacity(std::mem::size_of_val(chunk));
            for value in chunk {
                raw.extend_from_slice(&value.to_ne_bytes());
            }
            Arc::from(lz4_flex::block::compress_prepend_size(&raw).into_boxed_slice())
        })
        .collect()
}

/// CRC32-prefixed raw chunks, for fullchain `crc32` codec runs.
pub fn encode_u32_chunks_crc32(values: &[u32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    values
        .chunks(chunk_len)
        .map(|chunk| {
            let mut raw = Vec::with_capacity(std::mem::size_of_val(chunk));
            for value in chunk {
                raw.extend_from_slice(&value.to_ne_bytes());
            }
            Arc::from(crc32_encode(&raw).into_boxed_slice())
        })
        .collect()
}

pub fn encode_f32_chunks_crc32(values: &[f32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    values
        .chunks(chunk_len)
        .map(|chunk| {
            let mut raw = Vec::with_capacity(std::mem::size_of_val(chunk));
            for value in chunk {
                raw.extend_from_slice(&value.to_ne_bytes());
            }
            Arc::from(crc32_encode(&raw).into_boxed_slice())
        })
        .collect()
}

/// Write `chunks` to a single file and return `(path, locations)` with
/// `(offset, len)` per chunk in submission order.
pub fn write_chunks_file(
    label: &str,
    chunks: &[Arc<[u8]>],
) -> (std::path::PathBuf, Vec<FileChunkLocation>) {
    let path = super::bench_data_dir().join(format!("{label}-{}.bin", std::process::id()));
    let mut file = std::fs::File::create(&path).expect("create chunk file");
    let mut offset = 0u64;
    let mut locations = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        file.write_all(chunk).expect("write chunk");
        locations.push(FileChunkLocation {
            offset,
            len: chunk.len(),
        });
        offset += chunk.len() as u64;
    }
    file.sync_all().expect("sync chunk file");
    (path, locations)
}

pub fn write_dense_u32_file(
    path: &Path,
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) -> Vec<FileChunkLocation> {
    let chunks = make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols);
    let mut file = std::fs::File::create(path).expect("create dense file");
    let mut offset = 0u64;
    let mut locations = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        file.write_all(chunk).expect("write dense chunk");
        locations.push(FileChunkLocation {
            offset,
            len: chunk.len(),
        });
        offset += chunk.len() as u64;
    }
    file.sync_all().expect("sync dense file");
    locations
}

pub fn write_dense_u32_directory(
    dir: &Path,
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) -> Vec<DirectoryChunkLocationMeta> {
    let chunks = make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols);
    let mut locations = Vec::with_capacity(chunks.len());
    for (idx, chunk) in chunks.iter().enumerate() {
        let path = dir.join(format!("chunk-{idx}.bin"));
        std::fs::write(&path, chunk).expect("write dir chunk");
        locations.push(DirectoryChunkLocationMeta {
            path,
            len: chunk.len(),
        });
    }
    locations
}
