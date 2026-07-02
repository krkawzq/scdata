use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};

use super::super::array::{Array, ArrayGrid, Chunk, ChunkSource, EdgeChunkLayout, RegisteredFile};
use super::super::dataset::{Dataset, Dense2DDataset, SparseCsrDataset};
use crate::access::{ScheduledAccessConfig, SliceSpec};
use crate::codecs::{ChunkCodec, CodecError, CodecResult, SharedCodec, UncompressedCodec};
use crate::databank::{
    ArrayCodecSpec, ArrayGridSpec, ArrayOrder, ArraySpec, ChunkSourceSpec, ChunkSpec, DType,
    DataBank, DataBankConfig, Dense1DSpec, Dense2DSpec, MissingGenePolicy, PrefetchedBatch,
    ProjectedSparseDataGroupStrategy, ScheduledPrefetchConfig, SparseCsrSpec,
};
use crate::iopool::{BaseIoConfig, IoConfig, ThreadedConfig};
#[cfg(feature = "profile")]
use crate::profile::{ProfileMetricId, ProfileSnapshot};

#[cfg(feature = "profile")]
use super::super::profile::{test_metrics as databank_profile_metrics, DataBankProfile};

static FILE_SEQ: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "profile")]
fn profile_metric(snapshot: &ProfileSnapshot, metric: ProfileMetricId) -> u64 {
    snapshot.metric_value(metric).unwrap_or(0)
}

#[derive(Debug, Clone, Copy)]
struct TestChunkLocation {
    offset: u64,
    len: usize,
}

fn parallel_config() -> DataBankConfig {
    let mut config = DataBankConfig::default();
    config.fill_config.parallel = true;
    config.fill_config.num_workers = 2;
    config.fill_config.min_parallel_rows = 1;
    config.fill_config.min_parallel_bytes = 1;
    config
}

fn write_chunk_file(chunks: Vec<Arc<[u8]>>) -> (PathBuf, Vec<TestChunkLocation>) {
    let mut bytes = Vec::new();
    let mut locations = Vec::new();
    for chunk in &chunks {
        let offset = bytes.len();
        bytes.extend_from_slice(chunk);
        locations.push(TestChunkLocation {
            offset: offset as u64,
            len: chunk.len(),
        });
    }
    let seq = FILE_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("scdata-prefetch-{}-{seq}", std::process::id()));
    std::fs::write(&path, &bytes).expect("write temp chunk file");
    (path, locations)
}

fn memory_array_spec(
    shape: Vec<usize>,
    chunk_shape: Vec<usize>,
    dtype: DType,
    chunks: Vec<Arc<[u8]>>,
) -> ArraySpec {
    regular_array_spec(
        shape,
        chunk_shape,
        dtype,
        EdgeChunkLayout::Padded,
        chunks
            .into_iter()
            .map(|bytes| ChunkSourceSpec::Memory { bytes })
            .collect(),
    )
}

fn file_array_spec(
    shape: Vec<usize>,
    chunk_shape: Vec<usize>,
    dtype: DType,
    path: PathBuf,
    locations: Vec<TestChunkLocation>,
    edge: EdgeChunkLayout,
) -> ArraySpec {
    regular_array_spec(
        shape,
        chunk_shape,
        dtype,
        edge,
        locations
            .into_iter()
            .map(|location| ChunkSourceSpec::File {
                path: path.clone(),
                offset: location.offset,
                len: location.len,
            })
            .collect(),
    )
}

fn regular_array_spec(
    shape: Vec<usize>,
    chunk_shape: Vec<usize>,
    dtype: DType,
    edge: EdgeChunkLayout,
    sources: Vec<ChunkSourceSpec>,
) -> ArraySpec {
    let grid_shape = shape
        .iter()
        .zip(chunk_shape.iter())
        .map(|(&dim, &chunk)| dim.div_ceil(chunk))
        .collect::<Vec<_>>();
    let expected_chunks = grid_shape.iter().product::<usize>();
    assert_eq!(sources.len(), expected_chunks);
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
        codec: ArrayCodecSpec::Uncompressed,
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

fn arc_u32_bytes(values: &[u32]) -> Arc<[u8]> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes.into()
}

fn arc_u64_bytes(values: &[u64]) -> Arc<[u8]> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes.into()
}

fn arc_u16_bytes(values: &[u16]) -> Arc<[u8]> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes.into()
}

fn arc_f32_bytes(values: &[f32]) -> Arc<[u8]> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes.into()
}

/// Encode ``values`` as little-endian bytes, zero-padded to ``chunk_len``
/// elements — standard zarr edge-chunk layout (every decoded chunk is
/// exactly ``chunk_len`` elements; trailing padding is never read because
/// the CSR indptr bounds the logical range).
fn padded_u32_bytes(values: &[u32], chunk_len: usize) -> Arc<[u8]> {
    let mut bytes = vec![0u8; chunk_len * std::mem::size_of::<u32>()];
    for (i, value) in values.iter().enumerate() {
        let offset = i * std::mem::size_of::<u32>();
        bytes[offset..offset + std::mem::size_of::<u32>()].copy_from_slice(&value.to_ne_bytes());
    }
    bytes.into()
}

#[test]
fn memory_none_group_copies_raw_slices_directly() {
    let codec: SharedCodec = Arc::new(UncompressedCodec);
    let bytes = b"abcdefghijkl";
    let slice = SliceSpec::from_triples(vec![0, 2, 5, 3, 8, 11]).expect("slice spec");

    let sliced = super::load_memory_group(bytes, &codec, bytes.len(), false, &slice)
        .expect("slice raw none codec");

    assert_eq!(sliced, b"cdeijk");
}

#[test]
fn memory_none_group_preserves_sparse_slice_gaps() {
    let codec: SharedCodec = Arc::new(UncompressedCodec);
    let bytes = b"abcdefghijkl";
    let slice = SliceSpec::from_triples(vec![2, 2, 5]).expect("slice spec");

    let sliced = super::load_memory_group(bytes, &codec, bytes.len(), false, &slice)
        .expect("slice raw none codec");

    assert_eq!(sliced, b"\0\0cde");
}

#[test]
fn memory_none_group_preserves_size_mismatch_error() {
    let codec: SharedCodec = Arc::new(UncompressedCodec);

    let err = super::load_memory_group(b"abc", &codec, 4, false, &SliceSpec::Full)
        .expect_err("size mismatch should be surfaced");

    assert!(matches!(
        err,
        crate::databank::DataBankError::Codec(CodecError::SizeMismatch {
            expected: 4,
            actual: 3,
            ..
        })
    ));
}

#[test]
fn zero_length_file_chunk_reads_as_zero_fill_without_opening_path() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let (path, locations) = write_chunk_file(vec![arc_u32_bytes(&[11, 22])]);
    let missing_path = path.with_extension("missing");
    assert!(!missing_path.exists());

    let id = bank
        .register_dense_2d(Dense2DSpec {
            gene_names: vec!["g0".to_string(), "g1".to_string()],
            data: regular_array_spec(
                vec![2, 2],
                vec![1, 2],
                DType::U32,
                EdgeChunkLayout::Padded,
                vec![
                    ChunkSourceSpec::File {
                        path,
                        offset: locations[0].offset,
                        len: locations[0].len,
                    },
                    ChunkSourceSpec::File {
                        path: missing_path,
                        offset: 0,
                        len: 0,
                    },
                ],
            ),
        })
        .expect("register dense with missing chunk");

    let mut out = vec![99u32; 4];
    bank.access_cells(id, &[0, 1], &mut out, None)
        .expect("access dense with missing chunk");

    assert_eq!(out, vec![11, 22, 0, 0]);
}

#[test]
fn dense_1d_memory_fast_path_rejects_oversized_chunk() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = bank
        .register_dense_1d(Dense1DSpec {
            gene_names: vec!["g0".to_string(), "g1".to_string()],
            data: memory_array_spec(
                vec![6],
                vec![4],
                DType::U32,
                vec![arc_u32_bytes(&[1, 2, 3, 4, 999]), arc_u32_bytes(&[5, 6])],
            ),
        })
        .expect("register dense 1d");

    let mut out = vec![0u32; 2];
    let err = bank
        .access_cells(id, &[0], &mut out, None)
        .expect_err("oversized chunk should fail");

    assert!(matches!(
        err,
        crate::databank::DataBankError::Codec(CodecError::SizeMismatch {
            expected: 16,
            actual: 20,
            ..
        })
    ));
}

#[test]
fn empty_gene_name_is_rejected_at_registration() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let err = bank
        .register_dense_2d(Dense2DSpec {
            gene_names: vec![String::new()],
            data: memory_array_spec(
                vec![1, 1],
                vec![1, 1],
                DType::U32,
                vec![arc_u32_bytes(&[1])],
            ),
        })
        .expect_err("empty gene name should fail");

    assert!(matches!(
        err,
        crate::databank::DataBankError::InvalidArrayMeta(message)
            if message.contains("must not be empty")
    ));
}

#[cfg(feature = "profile")]
#[test]
fn databank_profile_records_facade_coverage_and_reset_keeps_recording() {
    let profiler = DataBankProfile::enabled("databank-profile-test");
    let round = profiler.runtime().start_owned();
    let mut bank =
        DataBank::new_with_profiler(DataBankConfig::default(), profiler).expect("databank");

    assert!(bank.profile().is_recording());

    let id = register_dense_2d_memory(&mut bank, 3, 4, 2, 2);

    let mut out = vec![0u32; 8];
    bank.access_cells(id, &[0, 2], &mut out, None)
        .expect("access cells");

    let genes = ["g1", "missing"];
    let mut projected = vec![0u32; 4];
    bank.access_cells_by_gene_names(
        id,
        &[1, 2],
        &genes,
        &mut projected,
        None,
        MissingGenePolicy::Zero,
    )
    .expect("projected access");

    bank.prefetch_cells(id, &[0, 1]).expect("direct prefetch");

    let scheduled = bank
        .prefetch_cells_scheduled::<u32, _>(
            id,
            vec![vec![0usize], vec![1usize]],
            ScheduledPrefetchConfig::default(),
        )
        .expect("scheduled prefetch")
        .map(|batch| batch.map(|batch| batch.buffer))
        .collect::<Result<Vec<_>, _>>()
        .expect("collect scheduled prefetch");
    assert_eq!(scheduled.len(), 2);

    let snapshot = bank.profile_snapshot_and_reset();
    assert_eq!(
        profile_metric(&snapshot, databank_profile_metrics::REGISTER_CALLS),
        1
    );
    assert_eq!(
        profile_metric(&snapshot, databank_profile_metrics::ACCESS_CALLS),
        2
    );
    assert_eq!(
        profile_metric(
            &snapshot,
            databank_profile_metrics::ACCESS_BY_GENE_NAMES_CALLS
        ),
        1
    );
    assert_eq!(
        profile_metric(&snapshot, databank_profile_metrics::ACCESS_OUTPUT_ELEMENTS),
        12
    );
    assert_eq!(
        profile_metric(&snapshot, databank_profile_metrics::PREFETCH_CALLS),
        1
    );
    assert_eq!(
        profile_metric(&snapshot, databank_profile_metrics::SCHEDULED_CALLS),
        1
    );
    assert!(bank.profile().is_recording());
    assert_eq!(
        profile_metric(
            &bank.profile_snapshot(),
            databank_profile_metrics::ACCESS_CALLS
        ),
        0
    );

    drop(bank);
    round.end();
}

#[test]
fn unregister_failure_unloads_dataset() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_file(&mut bank, 2, 2, 1);
    let file_id = match bank.registry.get(id).expect("dataset") {
        Dataset::Dense2D(dataset) => match &dataset.data.chunks[0].source {
            ChunkSource::File { file, .. } => file.id,
            _ => panic!("expected file-backed dense dataset"),
        },
        _ => panic!("expected dense dataset"),
    };
    bank.io_pool
        .unregister_file(file_id)
        .expect("manual unregister");

    let err = bank
        .unregister(id)
        .expect_err("second unregister should fail");

    assert!(matches!(err, crate::databank::DataBankError::Io(_)));
    assert!(matches!(
        bank.dataset_genes(id),
        Err(crate::databank::DataBankError::DatasetUnloaded(unloaded)) if unloaded == id
    ));
}

#[test]
fn memory_dense_2d_identity_scatter_handles_partial_column_chunks() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_memory(&mut bank, 5, 5, 2, 3);
    let cells = vec![4, 0, 3];
    let expected = expected_dense_rows(&cells, 5);

    let mut out = vec![0u32; cells.len() * 5];
    bank.access_cells(id, &cells, &mut out, None)
        .expect("memory dense access");
    assert_eq!(out, expected);

    let batches: Vec<Vec<usize>> = vec![vec![4, 0], vec![3]];
    let prefetch = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches.clone(), ScheduledPrefetchConfig::default())
        .expect("memory dense prefetch");
    let collected: Vec<Vec<u32>> = prefetch
        .map(|batch| batch.expect("prefetch batch").buffer)
        .collect();
    let expected_batches: Vec<Vec<u32>> = batches
        .iter()
        .map(|batch| expected_dense_rows(batch, 5))
        .collect();
    assert_eq!(collected, expected_batches);
}

#[test]
fn csr_batch_scatter_preserves_requested_cell_order_across_grouped_chunks() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = bank
        .register_sparse_csr(SparseCsrSpec {
            gene_names: (0..5).map(|idx| format!("g{idx}")).collect(),
            indptr: vec![0, 2, 5, 6],
            indices: u32_array_spec(vec![
                padded_u32_bytes(&[1, 3, 0, 4], 4),
                padded_u32_bytes(&[2, 3], 4),
            ]),
            data: u32_array_spec(vec![
                padded_u32_bytes(&[10, 30, 100, 400], 4),
                padded_u32_bytes(&[200, 3000], 4),
            ]),
            index_dtype: DType::U32,
            num_cells: 3,
            num_genes: 5,
        })
        .expect("register CSR");

    let mut checked = vec![0u32; 10];
    bank.access_cells(id, &[1, 0], &mut checked, None)
        .expect("checked CSR access");
    assert_eq!(checked, vec![100, 0, 200, 0, 400, 0, 10, 0, 30, 0]);

    let mut unchecked = vec![0u32; 10];
    unsafe {
        bank.access_cells_unchecked(id, &[1, 0], &mut unchecked, None)
            .expect("unchecked CSR access");
    }
    assert_eq!(unchecked, checked);
}

#[test]
fn compressed_csr_repeated_chunk_batch_decodes_each_chunk_once_and_preserves_order() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let (id, decodes) = register_counted_csr_memory(&mut bank, 8, 16, 2, 8);
    let cells = vec![3, 0, 1, 3];

    let out = bank
        .access_cells_alloc::<u32>(id, &cells)
        .expect("counted CSR access");

    assert_eq!(out, expected_counted_csr_rows(&cells, 16, 2));
    assert_eq!(
        decodes.load(Ordering::SeqCst),
        2,
        "one repeated index chunk and one repeated data chunk should decode once each"
    );
}

#[test]
fn compressed_csr_scattered_batch_decodes_only_touched_chunks_and_preserves_order() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let (id, decodes) = register_counted_csr_memory(&mut bank, 8, 16, 2, 2);
    let cells = vec![7, 0, 5, 3];

    let out = bank
        .access_cells_alloc::<u32>(id, &cells)
        .expect("counted CSR access");

    assert_eq!(out, expected_counted_csr_rows(&cells, 16, 2));
    assert_eq!(
        decodes.load(Ordering::SeqCst),
        cells.len() * 2,
        "scattered single-hit cells should decode one index and one data chunk per touched cell"
    );
}

#[test]
fn memory_csr_direct_handles_mismatched_chunk_boundaries_and_index_width() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = bank
        .register_sparse_csr(SparseCsrSpec {
            gene_names: (0..6).map(|idx| format!("g{idx}")).collect(),
            indptr: vec![0, 3, 6],
            indices: memory_array_spec(
                vec![6],
                vec![3],
                DType::U64,
                vec![arc_u64_bytes(&[0, 2, 5]), arc_u64_bytes(&[1, 3, 4])],
            ),
            data: memory_array_spec(
                vec![6],
                vec![2],
                DType::F32,
                vec![
                    arc_f32_bytes(&[1.0, 2.0]),
                    arc_f32_bytes(&[5.0, 10.0]),
                    arc_f32_bytes(&[30.0, 40.0]),
                ],
            ),
            index_dtype: DType::U64,
            num_cells: 2,
            num_genes: 6,
        })
        .expect("register CSR");

    let checked = bank
        .access_cells_alloc::<f32>(id, &[1, 0])
        .expect("checked CSR access");
    assert_eq!(
        checked,
        vec![
            0.0, 10.0, 0.0, 30.0, 40.0, 0.0, // cell 1
            1.0, 0.0, 2.0, 0.0, 0.0, 5.0, // cell 0
        ]
    );

    let mut unchecked = vec![0.0f32; checked.len()];
    unsafe {
        bank.access_cells_unchecked(id, &[1, 0], &mut unchecked, None)
            .expect("unchecked CSR access");
    }
    assert_eq!(unchecked, checked);
}

#[test]
fn access_cells_owned_matches_access_cells_dense_2d() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_file(&mut bank, 4, 4, 2);

    let cells = vec![3, 0, 2];
    let mut borrowed = vec![0u32; cells.len() * 4];
    bank.access_cells(id, &cells, &mut borrowed, None)
        .expect("checked dense access");
    let owned = bank
        .access_cells_alloc::<u32>(id, &cells)
        .expect("owned dense access");
    assert_eq!(owned, borrowed);
}

#[test]
fn access_cells_by_gene_names_reorders_dense_and_zero_fills_missing() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_memory(&mut bank, 4, 5, 2, 2);
    let cells = vec![2, 0];
    let genes = ["g3", "missing", "g1"];
    let mut out = vec![99u32; cells.len() * genes.len()];
    let mut names = vec![crate::databank::GeneNameView::empty(); genes.len()];

    bank.access_cells_by_gene_names(
        id,
        &cells,
        &genes,
        &mut out,
        Some(&mut names),
        MissingGenePolicy::Zero,
    )
    .expect("projected dense access");

    assert_eq!(out, vec![203, 0, 201, 3, 0, 1]);
    assert!(!names[0].is_empty());
    assert!(names[1].is_empty());
    assert!(!names[2].is_empty());
}

#[test]
fn projected_dense_segments_keep_only_selected_source_ranges() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_memory(&mut bank, 1, 8, 1, 8);
    let dataset = bank.registry.get(id).expect("dataset");
    let dense = match dataset {
        Dataset::Dense2D(dense) => dense,
        _ => panic!("expected dense 2d dataset"),
    };
    let genes = ["g6", "g1", "g2"];
    let gene_axis = super::GeneAxisPlan::requested(dataset, &genes, MissingGenePolicy::Zero)
        .expect("gene projection");
    let projection = match gene_axis {
        super::GeneAxisPlan::Requested(projection) => projection,
        super::GeneAxisPlan::DatasetOrder => panic!("subset should remain projected"),
    };

    let segments =
        super::plan::plan_dense_2d_selected_sources(dense, &[0], &projection.selected_sources)
            .expect("selected dense plan");

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].output_col_start, 1);
    assert_eq!(segments[0].output_cols, 2);
    assert_eq!(segments[0].source.start, 4);
    assert_eq!(segments[0].source.end, 12);
    assert_eq!(segments[1].output_col_start, 6);
    assert_eq!(segments[1].output_cols, 1);
    assert_eq!(segments[1].source.start, 24);
    assert_eq!(segments[1].source.end, 28);

    let out = bank
        .access_cells_owned_by_gene_names::<u32, _>(id, &[0], &genes, MissingGenePolicy::Zero)
        .expect("projected dense access");
    assert_eq!(out, vec![6, 1, 2]);
}

#[test]
fn projected_dense_1d_planning_visits_only_selected_ranges() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = bank
        .register_dense_1d(Dense1DSpec {
            gene_names: (0..8).map(|g| format!("g{g}")).collect(),
            data: memory_array_spec(
                vec![8],
                vec![4],
                DType::U32,
                vec![arc_u32_bytes(&[0, 1, 2, 3]), arc_u32_bytes(&[4, 5, 6, 7])],
            ),
        })
        .expect("register dense 1d");
    let dataset = bank.registry.get(id).expect("dataset");
    let dense = match dataset {
        Dataset::Dense1D(dense) => dense,
        _ => panic!("expected dense 1d dataset"),
    };
    let genes = ["g6", "g1", "g2"];
    let gene_axis = super::GeneAxisPlan::requested(dataset, &genes, MissingGenePolicy::Zero)
        .expect("gene projection");
    let projection = match gene_axis {
        super::GeneAxisPlan::Requested(projection) => projection,
        super::GeneAxisPlan::DatasetOrder => panic!("subset should remain projected"),
    };

    let segments =
        super::plan::plan_dense_1d_selected_sources(dense, &[0], &projection.selected_sources)
            .expect("selected dense 1d plan");

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].output_col_start, 1);
    assert_eq!(segments[0].output_cols, 2);
    assert_eq!(segments[0].source.start, 4);
    assert_eq!(segments[0].source.end, 12);
    assert_eq!(segments[1].output_col_start, 6);
    assert_eq!(segments[1].output_cols, 1);
    assert_eq!(segments[1].source.start, 8);
    assert_eq!(segments[1].source.end, 12);

    let out = bank
        .access_cells_owned_by_gene_names::<u32, _>(id, &[0], &genes, MissingGenePolicy::Zero)
        .expect("projected dense 1d access");
    assert_eq!(out, vec![6, 1, 2]);
}

#[test]
fn projected_dense_2d_planning_visits_only_selected_chunk_columns() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_memory(&mut bank, 1, 8, 1, 2);
    let dataset = bank.registry.get(id).expect("dataset");
    let dense = match dataset {
        Dataset::Dense2D(dense) => dense,
        _ => panic!("expected dense 2d dataset"),
    };
    let genes = ["g1", "g6"];
    let gene_axis = super::GeneAxisPlan::requested(dataset, &genes, MissingGenePolicy::Zero)
        .expect("gene projection");
    let projection = match gene_axis {
        super::GeneAxisPlan::Requested(projection) => projection,
        super::GeneAxisPlan::DatasetOrder => panic!("subset should remain projected"),
    };

    let segments =
        super::plan::plan_dense_2d_selected_sources(dense, &[0], &projection.selected_sources)
            .expect("selected dense plan");

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].output_col_start, 1);
    assert_eq!(segments[0].output_cols, 1);
    assert_eq!(segments[0].source.start, 4);
    assert_eq!(segments[0].source.end, 8);
    assert_eq!(segments[1].output_col_start, 6);
    assert_eq!(segments[1].output_cols, 1);
    assert_eq!(segments[1].source.start, 0);
    assert_eq!(segments[1].source.end, 4);
}

#[test]
fn requested_full_gene_order_collapses_to_dataset_order() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_memory(&mut bank, 2, 4, 1, 2);
    let dataset = bank.registry.get(id).expect("dataset");
    let genes = ["g0", "g1", "g2", "g3"];

    let gene_axis = super::GeneAxisPlan::requested(dataset, &genes, MissingGenePolicy::Zero)
        .expect("gene projection");

    assert!(matches!(gene_axis, super::GeneAxisPlan::DatasetOrder));
}

#[test]
fn projected_gene_runs_detect_contiguous_output_ranges() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_memory(&mut bank, 1, 8, 1, 8);
    let dataset = bank.registry.get(id).expect("dataset");
    let genes = ["g1", "g2", "g6"];
    let gene_axis = super::GeneAxisPlan::requested(dataset, &genes, MissingGenePolicy::Zero)
        .expect("gene projection");
    let projection = match gene_axis {
        super::GeneAxisPlan::Requested(projection) => projection,
        super::GeneAxisPlan::DatasetOrder => panic!("subset should remain projected"),
    };

    assert_eq!(projection.contiguous_output_for_source_run(1, 2), Some(0));
    assert_eq!(projection.contiguous_output_for_source_run(6, 1), Some(2));
    assert_eq!(projection.contiguous_output_for_source_run(2, 2), None);
}

#[test]
fn access_cells_by_gene_names_can_error_on_missing_gene() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_memory(&mut bank, 2, 3, 1, 2);
    let mut out = vec![0u32; 1];

    let err = bank
        .access_cells_by_gene_names(
            id,
            &[0],
            &["missing"],
            &mut out,
            None,
            MissingGenePolicy::Error,
        )
        .expect_err("missing gene should error");

    assert!(matches!(
        err,
        crate::databank::DataBankError::GeneNameNotFound { .. }
    ));
}

#[test]
fn access_cells_by_gene_names_rejects_duplicate_dataset_gene_names() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = bank
        .register_dense_2d(Dense2DSpec {
            gene_names: vec!["g0".to_string(), "g0".to_string()],
            data: memory_array_spec(
                vec![1, 2],
                vec![1, 2],
                DType::U32,
                vec![arc_u32_bytes(&[1, 2])],
            ),
        })
        .expect("register dense with duplicate gene names");
    let mut out = vec![0u32; 1];

    let err = bank
        .access_cells_by_gene_names(id, &[0], &["g0"], &mut out, None, MissingGenePolicy::Zero)
        .expect_err("duplicate dataset gene names should fail");

    assert!(matches!(
        err,
        crate::databank::DataBankError::InvalidArrayMeta(message)
            if message.contains("duplicate dataset gene name: g0")
    ));
}

#[test]
fn access_cells_by_gene_names_reorders_csr() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_csr_file(&mut bank);
    let cells = vec![1, 0, 2];
    let genes = ["g4", "g1", "missing", "g2"];

    let out = bank
        .access_cells_owned_by_gene_names::<u32, _>(id, &cells, &genes, MissingGenePolicy::Zero)
        .expect("projected CSR access");

    assert_eq!(
        out,
        vec![
            400, 0, 0, 200, // cell 1
            0, 10, 0, 0, // cell 0
            0, 0, 0, 0, // cell 2 has only g3
        ]
    );
}

#[test]
fn scheduled_prefetch_dense_2d_matches_direct_access() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_file(&mut bank, 4, 4, 2);

    let batches: Vec<Vec<usize>> = vec![vec![0, 1], vec![3, 2, 0], vec![1]];
    let expected: Vec<Vec<u32>> = batches
        .iter()
        .map(|cells| {
            bank.access_cells_owned::<u32>(id, cells)
                .expect("ground truth")
        })
        .collect();

    let prefetch = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches.clone(), ScheduledPrefetchConfig::default())
        .expect("build prefetcher");
    assert_eq!(
        prefetch.prefetch_step(),
        ScheduledPrefetchConfig::default().prefetch_step
    );

    let mut collected = Vec::new();
    for result in prefetch {
        let batch: PrefetchedBatch<u32> = result.expect("prefetch batch");
        assert_eq!(batch.num_genes, 4);
        collected.push(batch.buffer);
    }
    assert_eq!(collected, expected);
}

#[test]
fn scheduled_prefetch_waits_when_iopool_queue_is_full() {
    let mut config = DataBankConfig {
        io_config: IoConfig::Threaded(ThreadedConfig {
            base: BaseIoConfig {
                max_in_flight: 1,
                queue_capacity: 1,
                priority_levels: 3,
                queue_shards: 1,
                assume_non_overlapping_reads: false,
            },
            num_workers: 1,
            cpus: None,
        }),
        ..DataBankConfig::default()
    };
    config.access_config.queue_capacity = 8;
    config.fill_config.parallel = true;
    config.fill_config.num_workers = 2;
    config.fill_config.min_parallel_rows = 1;
    config.fill_config.min_parallel_bytes = 1;

    let mut bank = DataBank::new(config).expect("databank");
    let id = register_dense_2d_file(&mut bank, 12, 4, 1);
    let batches: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];
    let expected: Vec<Vec<u32>> = batches
        .iter()
        .map(|cells| expected_dense_rows(cells, 4))
        .collect();
    let prefetch_config = ScheduledPrefetchConfig {
        prefetch_step: 4,
        access: ScheduledAccessConfig {
            prefetch_step: 8,
            decode_ahead_steps: 8,
            ready_ahead_steps: 4,
        },
        ..ScheduledPrefetchConfig::default()
    };

    let collected: Vec<Vec<u32>> = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches, prefetch_config)
        .expect("scheduled prefetch")
        .map(|batch| batch.expect("prefetch batch").buffer)
        .collect();
    assert_eq!(collected, expected);
}

#[test]
fn scheduled_multi_prefetch_casts_bf16_and_f32_to_f32() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let genes = vec!["g0".to_string(), "g1".to_string()];
    let bf16_id = bank
        .register_dense_2d(Dense2DSpec {
            gene_names: genes.clone(),
            data: memory_array_spec(
                vec![2, 2],
                vec![1, 2],
                DType::BF16,
                vec![
                    arc_u16_bytes(&[0x3F80, 0x4000]), // 1.0, 2.0
                    arc_u16_bytes(&[0x4040, 0x4080]), // 3.0, 4.0
                ],
            ),
        })
        .expect("register bf16 dense 2d");
    let f32_id = bank
        .register_dense_2d(Dense2DSpec {
            gene_names: genes,
            data: memory_array_spec(
                vec![2, 2],
                vec![1, 2],
                DType::F32,
                vec![arc_f32_bytes(&[10.0, 20.0]), arc_f32_bytes(&[30.0, 40.0])],
            ),
        })
        .expect("register f32 dense 2d");

    let batches = vec![
        super::MultiBatchCells::new(vec![(0, vec![0]), (1, vec![1]), (0, vec![1])]),
        super::MultiBatchCells::new(vec![(1, vec![0]), (0, vec![0])]),
    ];
    let mut prefetch = bank
        .prefetch_cells_scheduled_multi::<f32, _>(
            &[bf16_id, f32_id],
            batches,
            ScheduledPrefetchConfig::default(),
        )
        .expect("multi prefetch");

    let first = prefetch.next().expect("first batch").expect("first ok");
    assert_eq!(first.cells, vec![0, 1, 1]);
    assert_eq!(first.num_genes, 2);
    assert_eq!(first.buffer, vec![1.0, 2.0, 30.0, 40.0, 3.0, 4.0]);

    let second = prefetch.next().expect("second batch").expect("second ok");
    assert_eq!(second.cells, vec![0, 0]);
    assert_eq!(second.num_genes, 2);
    assert_eq!(second.buffer, vec![10.0, 20.0, 1.0, 2.0]);
    assert!(prefetch.next().is_none());
}

#[test]
fn scheduled_multi_prefetch_coalesces_repeated_dataset_parts() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let (id, decodes) = register_counted_csr_memory(&mut bank, 8, 16, 2, 16);
    let cells = vec![3, 0, 1, 3];
    let batches = vec![super::MultiBatchCells::new(vec![
        (0, vec![3]),
        (0, vec![0, 1]),
        (0, vec![3]),
    ])];

    let mut prefetch = bank
        .prefetch_cells_scheduled_multi::<u32, _>(
            &[id],
            batches,
            ScheduledPrefetchConfig::default(),
        )
        .expect("multi prefetch");

    let batch = prefetch.next().expect("batch").expect("batch ok");
    assert_eq!(batch.cells, cells);
    assert_eq!(batch.num_genes, 16);
    assert_eq!(batch.buffer, expected_counted_csr_rows(&cells, 16, 2));
    assert!(prefetch.next().is_none());
    assert_eq!(
        decodes.load(Ordering::SeqCst),
        2,
        "repeated dataset parts in one multi batch should share the same index and data chunks"
    );
}

#[test]
fn plan_batch_multi_uses_single_fast_path_for_dataset_zero_sequential_rows() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_memory(&mut bank, 4, 5, 2, 3);
    let dataset = bank.registry.get_arc(id).expect("dataset");
    let datasets = vec![dataset];
    let gene_axes = super::MultiGeneAxisPlan::dataset_order(&datasets).expect("gene axes");
    let batch = super::MultiBatchCells::new(vec![(0, vec![2]), (0, vec![0, 1]), (0, vec![3])]);

    let (plan, _items) = super::plan_batch_multi(
        &datasets,
        batch,
        &gene_axes,
        ProjectedSparseDataGroupStrategy::SelectedOnly,
    )
    .expect("plan batch");

    match plan {
        super::BatchPlan::Single {
            dataset_idx,
            cells,
            plan,
        } => {
            assert_eq!(dataset_idx, 0);
            assert_eq!(cells, vec![2, 0, 1, 3]);
            assert!(matches!(plan, super::SingleDatasetPlan::Dense { .. }));
        }
        super::BatchPlan::Multi(_) => {
            panic!("dataset 0-only batch should use the single-dataset plan path");
        }
    }
}

#[test]
fn plan_batch_multi_uses_single_fast_path_for_nonzero_single_dataset() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id0 = register_dense_2d_memory(&mut bank, 4, 5, 2, 3);
    let id1 = register_dense_2d_memory(&mut bank, 4, 5, 2, 3);
    let datasets = vec![
        bank.registry.get_arc(id0).expect("dataset 0"),
        bank.registry.get_arc(id1).expect("dataset 1"),
    ];
    let gene_axes = super::MultiGeneAxisPlan::dataset_order(&datasets).expect("gene axes");
    let batch = super::MultiBatchCells::new(vec![(1, vec![2, 0])]);

    let (plan, _items) = super::plan_batch_multi(
        &datasets,
        batch,
        &gene_axes,
        ProjectedSparseDataGroupStrategy::SelectedOnly,
    )
    .expect("plan batch");

    match plan {
        super::BatchPlan::Single {
            dataset_idx, cells, ..
        } => {
            assert_eq!(dataset_idx, 1);
            assert_eq!(cells, vec![2, 0]);
        }
        _ => panic!("dataset 1-only batch should use the single-dataset plan path"),
    }
}

#[test]
fn scheduled_multi_prefetch_rejects_float_to_int_output() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = bank
        .register_dense_2d(Dense2DSpec {
            gene_names: vec!["g0".to_string()],
            data: memory_array_spec(
                vec![1, 1],
                vec![1, 1],
                DType::F32,
                vec![arc_f32_bytes(&[1.0])],
            ),
        })
        .expect("register f32 dense 2d");

    let err = match bank.prefetch_cells_scheduled_multi::<i32, _>(
        &[id],
        vec![super::MultiBatchCells::new(vec![(0, vec![0])])],
        ScheduledPrefetchConfig::default(),
    ) {
        Ok(_) => panic!("float to int should be rejected"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        crate::databank::DataBankError::CannotCast {
            src: DType::F32,
            dst: DType::I32,
            ..
        }
    ));
}

#[test]
fn unregister_defers_file_release_for_retained_prefetch_dataset() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_file(&mut bank, 4, 4, 2);
    let dataset = bank.registry.get_arc(id).expect("dataset arc");
    let batches: Vec<Vec<usize>> = vec![vec![0, 1], vec![2, 3]];
    let expected: Vec<Vec<u32>> = batches
        .iter()
        .map(|cells| expected_dense_rows(cells, 4))
        .collect();

    bank.unregister(id)
        .expect("unregister should retire retained dataset");
    let prefetch = super::prefetch_cells_scheduled::<u32, _>(
        &bank.access,
        Arc::clone(&bank.compute),
        Arc::clone(&dataset),
        batches.into_iter(),
        ScheduledPrefetchConfig::default(),
        bank.config.native_config.clone(),
        bank.native_scheduled_io(),
    )
    .expect("prefetch retained dataset");
    let collected: Vec<Vec<u32>> = prefetch
        .map(|batch| batch.expect("prefetch batch").buffer)
        .collect();
    assert_eq!(collected, expected);

    drop(dataset);
    bank.cleanup_retired().expect("cleanup retired dataset");
}

#[test]
fn scheduled_prefetch_emits_in_order_when_later_response_finishes_first() {
    let mut config = DataBankConfig::default();
    config.fill_config.parallel = true;
    config.fill_config.num_workers = 3;
    let mut bank = DataBank::new(config).expect("databank");
    let (started_tx, started_rx) = mpsc::channel();
    let (decoded_tx, decoded_rx) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let codec: SharedCodec = Arc::new(BlockingFirstChunkCodec {
        blocked: std::sync::atomic::AtomicBool::new(false),
        started: started_tx,
        decoded: decoded_tx,
        release: Arc::clone(&release),
    });
    let id = register_dense_2d_memory_with_codec(&mut bank, 2, 4, codec);
    let batches = vec![vec![0], vec![1]];
    let mut prefetch = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches.clone(), ScheduledPrefetchConfig::default())
        .expect("scheduled prefetch");

    started_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("first response blocked");
    let decoded_first = decoded_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("later response decoded");
    assert_eq!(decoded_first, dense_value(1, 0));
    {
        let (lock, cvar) = &*release;
        *lock.lock().expect("release lock") = true;
        cvar.notify_all();
    }

    let first = prefetch.next().expect("first item").expect("first batch");
    let second = prefetch.next().expect("second item").expect("second batch");
    assert_eq!(first.cells, batches[0]);
    assert_eq!(first.buffer, expected_dense_rows(&batches[0], 4));
    assert_eq!(second.cells, batches[1]);
    assert_eq!(second.buffer, expected_dense_rows(&batches[1], 4));
    assert!(prefetch.next().is_none());
}

#[test]
fn scheduled_prefetch_csr_matches_direct_access() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_csr_file(&mut bank);

    let batches: Vec<Vec<usize>> = vec![vec![1, 0], vec![2], vec![0, 1, 2]];
    let expected: Vec<Vec<u32>> = batches
        .iter()
        .map(|cells| {
            bank.access_cells_owned::<u32>(id, cells)
                .expect("ground truth")
        })
        .collect();

    let prefetch = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches.clone(), ScheduledPrefetchConfig::default())
        .expect("build prefetcher");

    let mut collected = Vec::new();
    for result in prefetch {
        let batch: PrefetchedBatch<u32> = result.expect("prefetch batch");
        assert_eq!(batch.cells, batches[collected.len()]);
        assert_eq!(batch.num_genes, 5);
        collected.push(batch.buffer);
    }
    assert_eq!(collected, expected);
}

#[test]
fn scheduled_prefetch_by_gene_names_matches_direct_csr_access() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_csr_file(&mut bank);
    let batches: Vec<Vec<usize>> = vec![vec![1, 0], vec![2]];
    let genes = ["g4", "g1", "missing", "g2"];
    let expected: Vec<Vec<u32>> = batches
        .iter()
        .map(|cells| {
            bank.access_cells_owned_by_gene_names::<u32, _>(
                id,
                cells,
                &genes,
                MissingGenePolicy::Zero,
            )
            .expect("ground truth")
        })
        .collect();

    let prefetch = bank
        .prefetch_cells_scheduled_by_gene_names::<u32, _, _>(
            id,
            batches.clone(),
            &genes,
            MissingGenePolicy::Zero,
            ScheduledPrefetchConfig::default(),
        )
        .expect("build projected prefetcher");
    assert_eq!(prefetch.gene_names().len(), genes.len());

    let mut collected = Vec::new();
    for result in prefetch {
        let batch = result.expect("prefetch batch");
        assert_eq!(batch.cells, batches[collected.len()]);
        assert_eq!(batch.num_genes, genes.len());
        collected.push(batch.buffer);
    }
    assert_eq!(collected, expected);
}

#[test]
fn scheduled_prefetch_memory_csr_matches_direct_access() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = bank
        .register_sparse_csr(SparseCsrSpec {
            gene_names: (0..5).map(|idx| format!("g{idx}")).collect(),
            indptr: vec![0, 2, 5, 6],
            indices: u32_array_spec(vec![
                padded_u32_bytes(&[1, 3, 0, 4], 4),
                padded_u32_bytes(&[2, 3], 4),
            ]),
            data: u32_array_spec(vec![
                padded_u32_bytes(&[10, 30, 100, 400], 4),
                padded_u32_bytes(&[200, 3000], 4),
            ]),
            index_dtype: DType::U32,
            num_cells: 3,
            num_genes: 5,
        })
        .expect("register CSR");

    let batches: Vec<Vec<usize>> = vec![vec![2], vec![1, 0], Vec::new()];
    let expected: Vec<Vec<u32>> = batches
        .iter()
        .map(|cells| {
            bank.access_cells_alloc::<u32>(id, cells)
                .expect("ground truth")
        })
        .collect();

    let prefetch = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches.clone(), ScheduledPrefetchConfig::default())
        .expect("build prefetcher");

    let mut collected = Vec::new();
    for result in prefetch {
        let batch = result.expect("prefetch batch");
        assert_eq!(batch.cells, batches[collected.len()]);
        assert_eq!(batch.num_genes, 5);
        collected.push(batch.buffer);
    }
    assert_eq!(collected, expected);
}

#[test]
fn databank_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DataBank>();
}

#[test]
fn parallel_csr_scatter_matches_sequential_checked_and_unchecked() {
    let mut seq_bank = DataBank::new(DataBankConfig::default()).expect("seq databank");
    let seq_id = register_csr_file(&mut seq_bank);
    let mut par_config = DataBankConfig::default();
    par_config.fill_config.parallel = true;
    par_config.fill_config.num_workers = 2;
    par_config.fill_config.min_parallel_rows = 1;
    par_config.fill_config.min_parallel_bytes = 1;
    let mut par_bank = DataBank::new(par_config).expect("parallel databank");
    let par_id = register_csr_file(&mut par_bank);
    let cells = vec![1, 0, 2, 1, 0, 2];

    let expected = seq_bank
        .access_cells_alloc::<u32>(seq_id, &cells)
        .expect("sequential checked");
    let checked = par_bank
        .access_cells_alloc::<u32>(par_id, &cells)
        .expect("parallel checked");
    assert_eq!(checked, expected);

    let mut unchecked = vec![0u32; cells.len() * 5];
    unsafe {
        par_bank
            .access_cells_unchecked(par_id, &cells, &mut unchecked, None)
            .expect("parallel unchecked");
    }
    assert_eq!(unchecked, expected);
}

#[test]
fn scheduled_parallel_csr_scatter_matches_direct_access() {
    let mut config = DataBankConfig::default();
    config.fill_config.parallel = true;
    config.fill_config.num_workers = 2;
    config.fill_config.min_parallel_rows = 1;
    config.fill_config.min_parallel_bytes = 1;
    let mut bank = DataBank::new(config).expect("databank");
    let id = register_csr_file(&mut bank);
    let batches: Vec<Vec<usize>> = vec![vec![1, 0, 2], vec![2, 1, 0]];
    let expected: Vec<Vec<u32>> = batches
        .iter()
        .map(|cells| {
            bank.access_cells_alloc::<u32>(id, cells)
                .expect("direct access")
        })
        .collect();

    let prefetch = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches.clone(), ScheduledPrefetchConfig::default())
        .expect("scheduled prefetch");
    let collected: Vec<Vec<u32>> = prefetch
        .map(|batch| batch.expect("prefetch batch").buffer)
        .collect();
    assert_eq!(collected, expected);
}

#[test]
fn parallel_dense_memory_scatter_matches_expected_for_full_and_projected() {
    let mut bank = DataBank::new(parallel_config()).expect("databank");
    let id = register_dense_2d_memory(&mut bank, 8, 7, 2, 3);
    let cells = vec![7, 0, 4, 3, 1];

    let full = bank
        .access_cells_alloc::<u32>(id, &cells)
        .expect("parallel dense access");
    assert_eq!(full, expected_dense_rows(&cells, 7));

    let genes = vec!["g6", "g2", "missing", "g0"];
    let projected = bank
        .access_cells_owned_by_gene_names::<u32, _>(id, &cells, &genes, MissingGenePolicy::Zero)
        .expect("parallel projected dense access");
    let expected: Vec<u32> = cells
        .iter()
        .flat_map(|&cell| {
            [
                dense_value(cell, 6),
                dense_value(cell, 2),
                0,
                dense_value(cell, 0),
            ]
        })
        .collect();
    assert_eq!(projected, expected);
}

#[test]
fn scheduled_parallel_dense_file_scatter_matches_direct_access() {
    let mut bank = DataBank::new(parallel_config()).expect("databank");
    let id = register_dense_2d_file(&mut bank, 8, 6, 2);
    let batches: Vec<Vec<usize>> = vec![vec![7, 0, 3], vec![2, 5, 1]];
    let expected: Vec<Vec<u32>> = batches
        .iter()
        .map(|cells| {
            bank.access_cells_alloc::<u32>(id, cells)
                .expect("direct dense access")
        })
        .collect();

    let prefetch = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches.clone(), ScheduledPrefetchConfig::default())
        .expect("scheduled dense prefetch");
    let collected: Vec<Vec<u32>> = prefetch
        .map(|batch| batch.expect("prefetch batch").buffer)
        .collect();
    assert_eq!(collected, expected);
}

#[test]
fn scheduled_prefetch_single_worker_parallel_scatter_does_not_deadlock() {
    let mut config = DataBankConfig::default();
    config.fill_config.parallel = true;
    config.fill_config.num_workers = 1;
    config.fill_config.min_parallel_rows = 1;
    config.fill_config.min_parallel_bytes = 1;
    let mut bank = DataBank::new(config).expect("databank");
    let id = register_dense_2d_file(&mut bank, 6, 4, 1);
    let batches: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![3, 4, 5]];
    let expected: Vec<Vec<u32>> = batches
        .iter()
        .map(|cells| {
            bank.access_cells_alloc::<u32>(id, cells)
                .expect("direct access")
        })
        .collect();

    let prefetch = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches, ScheduledPrefetchConfig::default())
        .expect("scheduled prefetch");
    let collected: Vec<Vec<u32>> = prefetch
        .map(|batch| batch.expect("prefetch batch").buffer)
        .collect();
    assert_eq!(collected, expected);
}

#[test]
fn parallel_projected_csr_scatter_reads_only_selected_groups() {
    let mut seq_bank = DataBank::new(DataBankConfig::default()).expect("seq databank");
    let seq_id = register_csr_file(&mut seq_bank);
    let mut par_bank = DataBank::new(parallel_config()).expect("parallel databank");
    let par_id = register_csr_file(&mut par_bank);
    let cells = vec![1, 0, 2, 1, 0];
    let genes = vec!["g1"];

    let expected = seq_bank
        .access_cells_owned_by_gene_names::<u32, _>(seq_id, &cells, &genes, MissingGenePolicy::Zero)
        .expect("sequential projected CSR");
    let checked = par_bank
        .access_cells_owned_by_gene_names::<u32, _>(par_id, &cells, &genes, MissingGenePolicy::Zero)
        .expect("parallel projected CSR");
    assert_eq!(checked, expected);
    assert_eq!(checked, vec![0, 10, 0, 0, 10]);

    let batches: Vec<Vec<usize>> = vec![vec![1, 0], vec![2, 1, 0]];
    let expected_batches: Vec<Vec<u32>> = batches
        .iter()
        .map(|batch| {
            par_bank
                .access_cells_owned_by_gene_names::<u32, _>(
                    par_id,
                    batch,
                    &genes,
                    MissingGenePolicy::Zero,
                )
                .expect("direct projected CSR")
        })
        .collect();
    let prefetch = par_bank
        .prefetch_cells_scheduled_by_gene_names::<u32, _, _>(
            par_id,
            batches.clone(),
            &genes,
            MissingGenePolicy::Zero,
            ScheduledPrefetchConfig::default(),
        )
        .expect("scheduled projected CSR");
    let collected: Vec<Vec<u32>> = prefetch
        .map(|batch| batch.expect("prefetch batch").buffer)
        .collect();
    assert_eq!(collected, expected_batches);
}

#[test]
fn scheduled_projected_csr_read_all_data_groups_matches_selected_only() {
    let mut bank = DataBank::new(parallel_config()).expect("databank");
    let id = register_csr_file(&mut bank);
    let genes = vec!["g1", "g3"];
    let batches: Vec<Vec<usize>> = vec![vec![1, 0], vec![2, 1, 0]];
    let expected_batches: Vec<Vec<u32>> = batches
        .iter()
        .map(|batch| {
            bank.access_cells_owned_by_gene_names::<u32, _>(
                id,
                batch,
                &genes,
                MissingGenePolicy::Zero,
            )
            .expect("direct projected CSR")
        })
        .collect();

    let selected_config = ScheduledPrefetchConfig {
        projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy::SelectedOnly,
        ..ScheduledPrefetchConfig::default()
    };
    let selected: Vec<Vec<u32>> = bank
        .prefetch_cells_scheduled_by_gene_names::<u32, _, _>(
            id,
            batches.clone(),
            &genes,
            MissingGenePolicy::Zero,
            selected_config,
        )
        .expect("scheduled selected-only projected CSR")
        .map(|batch| batch.expect("prefetch batch").buffer)
        .collect();
    assert_eq!(selected, expected_batches);

    let read_all_config = ScheduledPrefetchConfig {
        projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy::ReadAll,
        ..ScheduledPrefetchConfig::default()
    };
    let read_all: Vec<Vec<u32>> = bank
        .prefetch_cells_scheduled_by_gene_names::<u32, _, _>(
            id,
            batches,
            &genes,
            MissingGenePolicy::Zero,
            read_all_config,
        )
        .expect("scheduled read-all projected CSR")
        .map(|batch| batch.expect("prefetch batch").buffer)
        .collect();
    assert_eq!(read_all, expected_batches);
}

#[test]
fn projected_csr_mixed_file_memory_data_groups_matches_direct_and_prefetch() {
    let mut seq_bank = DataBank::new(DataBankConfig::default()).expect("seq databank");
    let seq_id = register_csr_mixed_file_memory_data(&mut seq_bank);
    let mut par_bank = DataBank::new(parallel_config()).expect("parallel databank");
    let par_id = register_csr_mixed_file_memory_data(&mut par_bank);
    let cells = vec![1, 0, 2, 1, 0];
    let genes = vec!["g1", "g3"];

    let expected = seq_bank
        .access_cells_owned_by_gene_names::<u32, _>(seq_id, &cells, &genes, MissingGenePolicy::Zero)
        .expect("sequential mixed projected CSR");
    let checked = par_bank
        .access_cells_owned_by_gene_names::<u32, _>(par_id, &cells, &genes, MissingGenePolicy::Zero)
        .expect("parallel mixed projected CSR");
    assert_eq!(checked, expected);
    assert_eq!(checked, vec![0, 0, 10, 30, 0, 3000, 0, 0, 10, 30]);

    let batches: Vec<Vec<usize>> = vec![vec![1, 0], vec![2, 1, 0]];
    let expected_batches: Vec<Vec<u32>> = batches
        .iter()
        .map(|batch| {
            par_bank
                .access_cells_owned_by_gene_names::<u32, _>(
                    par_id,
                    batch,
                    &genes,
                    MissingGenePolicy::Zero,
                )
                .expect("direct mixed projected CSR")
        })
        .collect();
    let prefetch = par_bank
        .prefetch_cells_scheduled_by_gene_names::<u32, _, _>(
            par_id,
            batches.clone(),
            &genes,
            MissingGenePolicy::Zero,
            ScheduledPrefetchConfig::default(),
        )
        .expect("scheduled mixed projected CSR");
    let collected: Vec<Vec<u32>> = prefetch
        .map(|batch| batch.expect("prefetch batch").buffer)
        .collect();
    assert_eq!(collected, expected_batches);
}

#[test]
fn shared_databank_allows_concurrent_callers() {
    let mut config = DataBankConfig::default();
    config.fill_config.parallel = true;
    config.fill_config.num_workers = 2;
    config.fill_config.min_parallel_rows = 1;
    config.fill_config.min_parallel_bytes = 1;
    let mut bank = DataBank::new(config).expect("databank");
    let id = register_csr_file(&mut bank);
    let bank = Arc::new(bank);
    let cells = vec![1, 0, 2, 1];
    let expected = bank
        .access_cells_alloc::<u32>(id, &cells)
        .expect("expected");

    let mut threads = Vec::new();
    for _ in 0..4 {
        let bank = Arc::clone(&bank);
        let cells = cells.clone();
        let expected = expected.clone();
        threads.push(std::thread::spawn(move || {
            let out = bank.access_cells_alloc::<u32>(id, &cells).expect("access");
            assert_eq!(out, expected);
        }));
    }
    for thread in threads {
        thread.join().expect("caller thread");
    }
}

#[test]
fn scheduled_prefetch_surfaces_plan_error_after_draining_good_batches() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_file(&mut bank, 4, 4, 2);

    // First batch is valid, second references an out-of-range cell.
    let batches: Vec<Vec<usize>> = vec![vec![0, 1], vec![99]];
    let prefetch = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches, ScheduledPrefetchConfig::default())
        .expect("build prefetcher");

    let mut iter = prefetch;
    let first = iter.next().expect("first batch").expect("first ok");
    assert_eq!(first.cells, vec![0, 1]);
    let err = iter.next().expect("error item").expect_err("plan error");
    assert!(matches!(
        err,
        crate::databank::DataBankError::CellIndexOutOfRange { cell: 99, .. }
    ));
    assert!(iter.next().is_none());
}

#[test]
fn scheduled_prefetch_exposes_first_ordered_error_only() {
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = register_dense_2d_file(&mut bank, 4, 4, 2);

    let batches: Vec<Vec<usize>> = vec![vec![0, 1], vec![99], vec![2, 3]];
    let mut iter = bank
        .prefetch_cells_scheduled::<u32, _>(id, batches, ScheduledPrefetchConfig::default())
        .expect("scheduled prefetch");

    let first = iter.next().expect("first batch").expect("first ok");
    assert_eq!(first.cells, vec![0, 1]);
    let err = iter.next().expect("error item").expect_err("first error");
    assert!(matches!(
        err,
        crate::databank::DataBankError::CellIndexOutOfRange { cell: 99, .. }
    ));
    assert!(iter.next().is_none());
}

#[derive(Clone)]
struct CountingCodec {
    decodes: Arc<AtomicUsize>,
}

impl fmt::Debug for CountingCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingCodec").finish_non_exhaustive()
    }
}

impl crate::codecs::sealed::Sealed for CountingCodec {}

impl ChunkCodec for CountingCodec {
    fn name(&self) -> &str {
        "counting"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        self.decodes.fetch_add(1, Ordering::SeqCst);
        if let Some(expected_size) = expected_size {
            assert_eq!(encoded.len(), expected_size);
        }
        Ok(encoded.to_vec())
    }

    fn decoded_size_hint(
        &self,
        _encoded: &[u8],
        expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        Ok(expected_size)
    }
}

struct BlockingFirstChunkCodec {
    blocked: std::sync::atomic::AtomicBool,
    started: mpsc::Sender<()>,
    decoded: mpsc::Sender<u32>,
    release: Arc<(Mutex<bool>, Condvar)>,
}

impl fmt::Debug for BlockingFirstChunkCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockingFirstChunkCodec")
            .finish_non_exhaustive()
    }
}

impl crate::codecs::sealed::Sealed for BlockingFirstChunkCodec {}

impl ChunkCodec for BlockingFirstChunkCodec {
    fn name(&self) -> &str {
        "blocking-first"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        if let Some(expected_size) = expected_size {
            assert_eq!(encoded.len(), expected_size);
        }
        let first = encoded
            .get(..std::mem::size_of::<u32>())
            .map(|bytes| {
                let mut word = [0u8; std::mem::size_of::<u32>()];
                word.copy_from_slice(bytes);
                u32::from_ne_bytes(word)
            })
            .unwrap_or(0);

        if first == 0 && !self.blocked.swap(true, Ordering::SeqCst) {
            self.started.send(()).expect("started");
            let (lock, cvar) = &*self.release;
            let mut released = lock.lock().expect("release lock");
            while !*released {
                released = cvar.wait(released).expect("release wait");
            }
        }
        self.decoded.send(first).expect("decoded");
        Ok(encoded.to_vec())
    }

    fn decoded_size_hint(
        &self,
        _encoded: &[u8],
        expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        Ok(expected_size)
    }
}

fn register_counted_csr_memory(
    bank: &mut DataBank,
    num_cells: usize,
    num_genes: usize,
    nnz_per_cell: usize,
    chunk_len: usize,
) -> (crate::databank::DatasetId, Arc<AtomicUsize>) {
    let nnz = num_cells * nnz_per_cell;
    let mut indptr = Vec::with_capacity(num_cells + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    indptr.push(0);
    for cell in 0..num_cells {
        for k in 0..nnz_per_cell {
            indices.push(counted_csr_gene(cell, k, num_genes));
            data.push(counted_csr_value(cell, k));
        }
        indptr.push(indptr.last().copied().unwrap() + nnz_per_cell as u64);
    }

    let decodes = Arc::new(AtomicUsize::new(0));
    let codec: SharedCodec = Arc::new(CountingCodec {
        decodes: Arc::clone(&decodes),
    });
    let gene_names = (0..num_genes)
        .map(|idx| format!("g{idx}"))
        .collect::<Vec<_>>();
    let genes = bank.interner.intern_dataset(&gene_names);
    let indices_chunks = chunk_u32_values(&indices, chunk_len);
    let data_chunks = chunk_u32_values(&data, chunk_len);
    let dataset = SparseCsrDataset {
        genes,
        indptr,
        indices: counted_memory_array(
            nnz,
            chunk_len,
            DType::U32,
            Arc::clone(&codec),
            indices_chunks,
        ),
        data: counted_memory_array(nnz, chunk_len, DType::U32, codec, data_chunks),
        index_dtype: DType::U32,
        num_cells,
        num_genes,
    };
    let id = bank
        .registry
        .register(Dataset::SparseCsr(dataset))
        .expect("register counted CSR");
    (id, decodes)
}

fn chunk_u32_values(values: &[u32], chunk_len: usize) -> Vec<Arc<[u8]>> {
    // Edge chunks padded to ``chunk_len`` (zero fill value) — standard zarr
    // edge-chunk layout: every decoded chunk is exactly ``chunk_len``
    // elements.
    values
        .chunks(chunk_len)
        .map(|chunk| {
            let mut bytes = vec![0u8; chunk_len * std::mem::size_of::<u32>()];
            for (i, value) in chunk.iter().enumerate() {
                let offset = i * std::mem::size_of::<u32>();
                bytes[offset..offset + std::mem::size_of::<u32>()]
                    .copy_from_slice(&value.to_ne_bytes());
            }
            Arc::from(bytes.into_boxed_slice())
        })
        .collect()
}

fn counted_memory_array(
    len: usize,
    chunk_len: usize,
    dtype: DType,
    codec: SharedCodec,
    raw_chunks: Vec<Arc<[u8]>>,
) -> Array {
    let grid_shape = vec![len.div_ceil(chunk_len)];
    let decoded_bytes = chunk_len * dtype.item_size();
    Array {
        shape: vec![len],
        dtype,
        codec,
        grid: ArrayGrid::Regular {
            chunk_shape: vec![chunk_len],
            grid_shape,
            edge: EdgeChunkLayout::Padded,
        },
        chunks: raw_chunks
            .into_iter()
            .map(|bytes| Chunk {
                source: ChunkSource::Memory {
                    bytes,
                    decoded: false,
                },
                decoded_bytes,
            })
            .collect(),
        files: Vec::new(),
    }
}

fn counted_decoded_memory_array(
    len: usize,
    chunk_len: usize,
    dtype: DType,
    raw_chunks: Vec<Arc<[u8]>>,
) -> (Array, Arc<AtomicUsize>) {
    let decodes = Arc::new(AtomicUsize::new(0));
    let codec: SharedCodec = Arc::new(CountingCodec {
        decodes: Arc::clone(&decodes),
    });
    let mut array = counted_memory_array(len, chunk_len, dtype, codec, raw_chunks);
    for chunk in &mut array.chunks {
        let ChunkSource::Memory { decoded, .. } = &mut chunk.source else {
            unreachable!("counted_memory_array only creates memory chunks");
        };
        *decoded = true;
    }
    (array, decodes)
}

#[test]
fn decoded_single_memory_chunk_accepts_non_identity_codec_without_decode() {
    let bytes = arc_u32_bytes(&[1, 2, 3, 4]);
    let (array, decodes) = counted_decoded_memory_array(4, 4, DType::U32, vec![Arc::clone(&bytes)]);

    let raw = super::single_memory_identity_chunk_bytes(&array)
        .expect("single decoded chunk")
        .expect("direct memory chunk");

    assert_eq!(raw, bytes.as_ref());
    assert_eq!(decodes.load(Ordering::SeqCst), 0);
}

#[test]
fn decoded_memory_1d_chunks_accept_non_identity_codec_without_decode() {
    let chunk0 = arc_u32_bytes(&[1, 2, 3, 4]);
    let chunk1 = arc_u32_bytes(&[5, 6, 0, 0]);
    let (array, decodes) = counted_decoded_memory_array(
        6,
        4,
        DType::U32,
        vec![Arc::clone(&chunk0), Arc::clone(&chunk1)],
    );

    let chunks = super::MemoryIdentity1DChunks::from_array(&array)
        .expect("decoded memory chunks")
        .expect("direct 1D chunks");

    assert_eq!(chunks.chunk_bytes(0).expect("chunk 0"), chunk0.as_ref());
    assert_eq!(chunks.chunk_bytes(1).expect("chunk 1"), chunk1.as_ref());
    assert_eq!(decodes.load(Ordering::SeqCst), 0);
}

fn expected_counted_csr_rows(cells: &[usize], num_genes: usize, nnz_per_cell: usize) -> Vec<u32> {
    let mut out = vec![0u32; cells.len() * num_genes];
    for (row, &cell) in cells.iter().enumerate() {
        let row_offset = row * num_genes;
        for k in 0..nnz_per_cell {
            let gene = counted_csr_gene(cell, k, num_genes) as usize;
            out[row_offset + gene] = counted_csr_value(cell, k);
        }
    }
    out
}

fn counted_csr_gene(cell: usize, k: usize, num_genes: usize) -> u32 {
    ((cell * 3 + k * 5) % num_genes) as u32
}

fn counted_csr_value(cell: usize, k: usize) -> u32 {
    (cell as u32) * 1000 + k as u32 + 1
}

fn register_dense_2d_file(
    bank: &mut DataBank,
    num_cells: usize,
    num_genes: usize,
    chunk_rows: usize,
) -> crate::databank::DatasetId {
    // value(cell, gene) = cell * 100 + gene
    let chunk_cols = num_genes;
    let grid_rows = num_cells.div_ceil(chunk_rows);
    let mut chunks = Vec::new();
    for chunk_row in 0..grid_rows {
        // Padded to full chunk_rows*chunk_cols (zero fill value) to match
        // standard zarr edge-chunk layout.
        let mut bytes = vec![0u8; chunk_rows * chunk_cols * std::mem::size_of::<u32>()];
        for local_row in 0..chunk_rows {
            let cell = chunk_row * chunk_rows + local_row;
            if cell >= num_cells {
                break;
            }
            for gene in 0..num_genes {
                let offset = (local_row * chunk_cols + gene) * std::mem::size_of::<u32>();
                bytes[offset..offset + std::mem::size_of::<u32>()]
                    .copy_from_slice(&((cell * 100 + gene) as u32).to_ne_bytes());
            }
        }
        chunks.push(Arc::from(bytes.into_boxed_slice()));
    }
    let (path, locations) = write_chunk_file(chunks);
    let id = bank
        .register_dense_2d(Dense2DSpec {
            gene_names: (0..num_genes).map(|g| format!("g{g}")).collect(),
            data: file_array_spec(
                vec![num_cells, num_genes],
                vec![chunk_rows, chunk_cols],
                DType::U32,
                path,
                locations,
                EdgeChunkLayout::Padded,
            ),
        })
        .expect("register dense 2d file");
    id
}

fn register_dense_2d_memory(
    bank: &mut DataBank,
    num_cells: usize,
    num_genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) -> crate::databank::DatasetId {
    let grid_rows = num_cells.div_ceil(chunk_rows);
    let grid_cols = num_genes.div_ceil(chunk_cols);
    let mut chunks = Vec::with_capacity(grid_rows * grid_cols);
    for chunk_row in 0..grid_rows {
        let row_start = chunk_row * chunk_rows;
        for chunk_col in 0..grid_cols {
            let col_start = chunk_col * chunk_cols;
            // Chunks are stored padded to the full ``chunk_shape`` with a
            // zero fill value, matching standard zarr edge-chunk layout
            // (the decoded buffer is always chunk_rows*chunk_cols elements).
            let mut bytes = vec![0u8; chunk_rows * chunk_cols * std::mem::size_of::<u32>()];
            for local_row in 0..chunk_rows {
                let cell = row_start + local_row;
                if cell >= num_cells {
                    break;
                }
                for local_col in 0..chunk_cols {
                    let gene = col_start + local_col;
                    if gene >= num_genes {
                        break;
                    }
                    let offset = (local_row * chunk_cols + local_col) * std::mem::size_of::<u32>();
                    bytes[offset..offset + std::mem::size_of::<u32>()]
                        .copy_from_slice(&dense_value(cell, gene).to_ne_bytes());
                }
            }
            chunks.push(Arc::from(bytes.into_boxed_slice()));
        }
    }

    bank.register_dense_2d(Dense2DSpec {
        gene_names: (0..num_genes).map(|g| format!("g{g}")).collect(),
        data: memory_array_spec(
            vec![num_cells, num_genes],
            vec![chunk_rows, chunk_cols],
            DType::U32,
            chunks,
        ),
    })
    .expect("register dense 2d memory")
}

fn register_dense_2d_memory_with_codec(
    bank: &mut DataBank,
    num_cells: usize,
    num_genes: usize,
    codec: SharedCodec,
) -> crate::databank::DatasetId {
    let chunk_rows = 1;
    let chunk_cols = num_genes;
    let mut chunks = Vec::with_capacity(num_cells);
    for cell in 0..num_cells {
        let mut bytes = Vec::with_capacity(num_genes * std::mem::size_of::<u32>());
        for gene in 0..num_genes {
            bytes.extend_from_slice(&dense_value(cell, gene).to_ne_bytes());
        }
        chunks.push(Chunk {
            source: ChunkSource::Memory {
                bytes: Arc::from(bytes.into_boxed_slice()),
                decoded: false,
            },
            decoded_bytes: chunk_cols * std::mem::size_of::<u32>(),
        });
    }
    let gene_names = (0..num_genes)
        .map(|gene| format!("g{gene}"))
        .collect::<Vec<_>>();
    let genes = bank.interner.intern_dataset(&gene_names);
    let dataset = Dense2DDataset {
        genes,
        data: Array {
            shape: vec![num_cells, num_genes],
            dtype: DType::U32,
            codec,
            grid: ArrayGrid::Regular {
                chunk_shape: vec![chunk_rows, chunk_cols],
                grid_shape: vec![num_cells, 1],
                edge: EdgeChunkLayout::Padded,
            },
            chunks,
            files: Vec::new(),
        },
        num_cells,
        num_genes,
    };
    bank.registry
        .register(Dataset::Dense2D(dataset))
        .expect("register dense 2d memory with codec")
}

fn expected_dense_rows(cells: &[usize], num_genes: usize) -> Vec<u32> {
    cells
        .iter()
        .flat_map(|&cell| (0..num_genes).map(move |gene| dense_value(cell, gene)))
        .collect()
}

fn dense_value(cell: usize, gene: usize) -> u32 {
    (cell * 100 + gene) as u32
}

fn register_csr_file(bank: &mut DataBank) -> crate::databank::DatasetId {
    let (indices_path, indices_locations) =
        write_chunk_file(vec![arc_u32_bytes(&[1, 3, 0, 4]), arc_u32_bytes(&[2, 3])]);
    let (data_path, data_locations) = write_chunk_file(vec![
        arc_u32_bytes(&[10, 30, 100, 400]),
        arc_u32_bytes(&[200, 3000]),
    ]);
    bank.register_sparse_csr(SparseCsrSpec {
        gene_names: (0..5).map(|idx| format!("g{idx}")).collect(),
        indptr: vec![0, 2, 5, 6],
        indices: file_array_spec(
            vec![6],
            vec![4],
            DType::U32,
            indices_path,
            indices_locations,
            EdgeChunkLayout::Cropped,
        ),
        data: file_array_spec(
            vec![6],
            vec![4],
            DType::U32,
            data_path,
            data_locations,
            EdgeChunkLayout::Cropped,
        ),
        index_dtype: DType::U32,
        num_cells: 3,
        num_genes: 5,
    })
    .expect("register CSR file")
}

fn register_csr_mixed_file_memory_data(bank: &mut DataBank) -> crate::databank::DatasetId {
    let (data_path, data_locations) = write_chunk_file(vec![arc_u32_bytes(&[10, 30, 100, 400])]);
    let file_id = bank
        .io_pool
        .register_readonly_file(&data_path)
        .expect("register data file");
    let file = RegisteredFile::new(file_id).expect("registered file");
    let gene_names = (0..5).map(|idx| format!("g{idx}")).collect::<Vec<_>>();
    let genes = bank.interner.intern_dataset(&gene_names);
    let index_chunks = vec![
        Chunk {
            source: ChunkSource::Memory {
                bytes: arc_u32_bytes(&[1, 3, 0, 4]),
                decoded: false,
            },
            decoded_bytes: 4 * std::mem::size_of::<u32>(),
        },
        Chunk {
            source: ChunkSource::Memory {
                bytes: arc_u32_bytes(&[2, 3]),
                decoded: false,
            },
            decoded_bytes: 2 * std::mem::size_of::<u32>(),
        },
    ];
    let data_chunks = vec![
        Chunk {
            source: ChunkSource::File {
                file,
                offset: data_locations[0].offset,
                len: data_locations[0].len,
            },
            decoded_bytes: 4 * std::mem::size_of::<u32>(),
        },
        Chunk {
            source: ChunkSource::Memory {
                bytes: arc_u32_bytes(&[200, 3000]),
                decoded: false,
            },
            decoded_bytes: 2 * std::mem::size_of::<u32>(),
        },
    ];
    let grid = ArrayGrid::Regular {
        chunk_shape: vec![4],
        grid_shape: vec![2],
        edge: EdgeChunkLayout::Cropped,
    };
    let dataset = SparseCsrDataset {
        genes,
        indptr: vec![0, 2, 5, 6],
        indices: Array {
            shape: vec![6],
            dtype: DType::U32,
            codec: Arc::new(UncompressedCodec),
            grid: grid.clone(),
            chunks: index_chunks,
            files: Vec::new(),
        },
        data: Array {
            shape: vec![6],
            dtype: DType::U32,
            codec: Arc::new(UncompressedCodec),
            grid,
            chunks: data_chunks,
            files: vec![file],
        },
        index_dtype: DType::U32,
        num_cells: 3,
        num_genes: 5,
    };
    bank.registry
        .register(Dataset::SparseCsr(dataset))
        .expect("register mixed CSR")
}

fn u32_array_spec(chunks: Vec<Arc<[u8]>>) -> ArraySpec {
    memory_array_spec(vec![6], vec![4], DType::U32, chunks)
}
