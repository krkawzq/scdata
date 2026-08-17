use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use sc_compress::{ByteStore, CsrWriter, DenseMatrix, DenseWriter, Partition};

use crate::{
    compile, Dataset, Error, FeatureMap, Fill, IoMergeOptions, IoMergePolicy, IoMode, OutputDType,
    OutputSpec, OverflowPolicy, PlanConfig, PlanSpec, RowRef, SessionConfig, Source, SourceId,
};

fn blocking(workers: usize) -> SessionConfig {
    SessionConfig {
        worker_count: workers,
        initialize_workers: workers,
        initialize_inflight_io_ops: workers,
        io_mode: IoMode::Blocking,
        ..SessionConfig::default()
    }
}

#[test]
fn dense_mapping_promotion_and_ring_reuse_preserve_order() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense");
    let values: Vec<u16> = (0..24).collect();
    DenseWriter::new(&path, Partition::fixed_cells(4), Partition::fixed_cells(2))
        .write(&values, [8, 3])
        .unwrap();
    let dataset = Dataset::open(&path).unwrap();
    let mapping = FeatureMap::new([Some(2), None, Some(0)]).unwrap();
    let source = Source::new(7, dataset).feature_map(mapping);
    let rows = [3, 0, 3, 7, 1, 6, 2, 5, 4];
    let row_refs = rows
        .into_iter()
        .map(|row| RowRef::new(SourceId::new(7), row))
        .collect();
    let output = OutputSpec::new(4, OutputDType::U32, Fill::U32(9)).unwrap();
    let config = PlanConfig {
        ..PlanConfig::default()
    };
    let plan = compile(PlanSpec::new(vec![source], row_refs, output, 2, 3).config(config)).unwrap();
    assert!(plan.stats().block_jobs < rows.len());
    assert_eq!(plan.batch_count(), 5);

    let mut session = plan.open(blocking(3)).unwrap();
    let mut observed = Vec::new();
    while let Some(batch) = session.next_batch().unwrap() {
        for row in 0..batch.rows() {
            observed.push(batch.row_as::<u32>(row).unwrap().to_vec());
        }
    }
    let expected = rows
        .into_iter()
        .map(|row| {
            let base = row as u32 * 3;
            vec![base + 2, 9, base, 9]
        })
        .collect::<Vec<_>>();
    assert_eq!(observed, expected);
}

#[test]
fn dense_mapping_compacts_contiguous_columns_into_one_run() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-map-run");
    DenseWriter::new(
        &path,
        Partition::fixed_cells(1024),
        Partition::fixed_cells(16),
    )
    .write(&[1u16, 2, 3, 4, 5], [1, 5])
    .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap())
        .feature_map(FeatureMap::new([Some(1), Some(2), Some(3), Some(4), None]).unwrap());
    let plan = compile(PlanSpec::new(
        vec![source],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(6, OutputDType::U16, Fill::U16(99)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    let crate::plan::DenseMap::Runs {
        entries,
        covers_output,
    } = plan.inner.source_plans[0].dense_map.as_ref().unwrap()
    else {
        panic!("contiguous dense mapping was not compacted into runs");
    };
    assert!(!covers_output);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].count, 4);
    assert!(plan.inner.source_plans[0].dense_fill_whole);
    assert!(plan.inner.source_plans[0].default_ranges.is_empty());

    let mut session = plan.open(blocking(1)).unwrap();
    assert_eq!(
        session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<u16>(0)
            .unwrap(),
        &[99, 1, 2, 3, 4, 99]
    );
}

#[test]
fn dense_mapping_uses_range_fill_when_saved_writes_cover_gap_overhead() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-map-range-fill");
    let values = (0..256u16).collect::<Vec<_>>();
    DenseWriter::new(
        &path,
        Partition::fixed_cells(1024),
        Partition::fixed_cells(16),
    )
    .write(&values, [1, 256])
    .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap())
        .feature_map(FeatureMap::new((0..256).map(Some)).unwrap());
    let plan = compile(PlanSpec::new(
        vec![source],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(300, OutputDType::U16, Fill::U16(99)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    let source = &plan.inner.source_plans[0];
    assert!(!source.dense_fill_whole);
    assert_eq!(
        source.default_ranges.as_ref(),
        [crate::plan::OutputRange {
            offset: 256 * 2,
            len: 44 * 2,
        }]
    );

    let mut session = plan.open(blocking(1)).unwrap();
    let batch = session.next_batch().unwrap().unwrap();
    let row = batch.row_as::<u16>(0).unwrap();
    assert_eq!(&row[..256], values);
    assert!(row[256..].iter().all(|value| *value == 99));
}

#[test]
fn dense_mapping_compacts_sparse_sources_into_one_gather_run() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-map-gather");
    let values = (0..64).map(|value| value as f32).collect::<Vec<_>>();
    DenseWriter::new(
        &path,
        Partition::fixed_cells(1024),
        Partition::fixed_cells(16),
    )
    .write(&values, [1, 64])
    .unwrap();
    let targets = (0..64)
        .map(|source| (source % 2 == 0).then_some(source / 2))
        .collect::<Vec<_>>();
    let source = Source::new(0, Dataset::open(&path).unwrap())
        .feature_map(FeatureMap::new(targets).unwrap());
    let plan = compile(PlanSpec::new(
        vec![source],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(32, OutputDType::F32, Fill::F32(-1.0)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    let crate::plan::DenseMap::Gather32 {
        source_offsets,
        target_byte,
        covers_output,
    } = plan.inner.source_plans[0].dense_map.as_ref().unwrap()
    else {
        panic!(
            "sparse source columns with contiguous targets were not compacted into a gather run"
        );
    };
    assert!(*covers_output);
    assert_eq!(*target_byte, 0);
    assert_eq!(
        source_offsets.as_ref(),
        &(0..64)
            .step_by(2)
            .map(|source| source * 4)
            .collect::<Vec<_>>()
    );

    let mut session = plan.open(blocking(1)).unwrap();
    assert_eq!(
        session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<f32>(0)
            .unwrap(),
        &(0..64)
            .step_by(2)
            .map(|value| value as f32)
            .collect::<Vec<_>>()
    );
}

#[test]
fn dense_mapping_gathers_i32_into_contiguous_f64_targets() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-map-gather-i32-f64");
    let values = (0..16).map(|value| value * 17 - 80).collect::<Vec<i32>>();
    DenseWriter::new(
        &path,
        Partition::fixed_cells(1024),
        Partition::fixed_cells(16),
    )
    .write(&values, [1, 16])
    .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap()).feature_map(
        FeatureMap::new(
            (0..16)
                .map(|column| (column % 2 == 0).then_some(column / 2))
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    );
    let plan = compile(PlanSpec::new(
        vec![source],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(8, OutputDType::F64, Fill::F64(-1.0)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    let fast_gather = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
    let fast_gather = false;
    assert_eq!(
        matches!(
            plan.inner.source_plans[0].dense_map,
            Some(crate::plan::DenseMap::Gather32 { .. })
        ),
        fast_gather
    );

    let mut session = plan.open(blocking(1)).unwrap();
    assert_eq!(
        session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<f64>(0)
            .unwrap(),
        &values
            .iter()
            .step_by(2)
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>()
    );
}

#[test]
fn dense_mapping_plans_widening_64_bit_gather_kernels() {
    fn sparse_even_targets() -> FeatureMap {
        FeatureMap::new(
            (0..64)
                .map(|column| (column % 2 == 0).then_some(column / 2))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    let temporary = tempfile::tempdir().unwrap();
    let i32_path = temporary.path().join("dense-map-gather-i32-i64");
    let i32_values = (0..64)
        .map(|value| value * 1_000_003 - 31_000_000)
        .collect::<Vec<i32>>();
    DenseWriter::new(
        &i32_path,
        Partition::fixed_cells(1024),
        Partition::fixed_cells(16),
    )
    .write(&i32_values, [1, 64])
    .unwrap();
    let i32_plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&i32_path).unwrap()).feature_map(sparse_even_targets())],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(32, OutputDType::I64, Fill::I64(-1)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    let i32_fast_gather = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
    let i32_fast_gather = false;
    assert_eq!(
        matches!(
            i32_plan.inner.source_plans[0].dense_map,
            Some(crate::plan::DenseMap::Gather32 { .. })
        ),
        i32_fast_gather
    );
    let mut i32_session = i32_plan.open(blocking(1)).unwrap();
    assert_eq!(
        i32_session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<i64>(0)
            .unwrap(),
        &i32_values
            .iter()
            .step_by(2)
            .map(|&value| i64::from(value))
            .collect::<Vec<_>>()
    );

    let u64_path = temporary.path().join("dense-map-gather-u64-f64");
    let u64_values = (0..64)
        .map(|value| (value as u64).wrapping_mul(1_000_000_000_000_003))
        .collect::<Vec<_>>();
    DenseWriter::new(
        &u64_path,
        Partition::fixed_cells(1024),
        Partition::fixed_cells(16),
    )
    .write(&u64_values, [1, 64])
    .unwrap();
    let u64_plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&u64_path).unwrap()).feature_map(sparse_even_targets())],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(32, OutputDType::F64, Fill::F64(-1.0))
            .unwrap()
            .float_cast(crate::FloatCastPolicy::AllowRounding),
        1,
        2,
    ))
    .unwrap();
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    let u64_fast_gather = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq");
    #[cfg(not(all(target_arch = "x86_64", target_endian = "little")))]
    let u64_fast_gather = false;
    assert_eq!(
        matches!(
            u64_plan.inner.source_plans[0].dense_map,
            Some(crate::plan::DenseMap::Gather32 { .. })
        ),
        u64_fast_gather
    );
    let mut u64_session = u64_plan.open(blocking(1)).unwrap();
    assert_eq!(
        u64_session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<f64>(0)
            .unwrap(),
        &u64_values
            .iter()
            .step_by(2)
            .map(|&value| value as f64)
            .collect::<Vec<_>>()
    );
}

#[test]
fn dense_gather_checked_sign_preserves_overflow_policy() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-map-gather-overflow");
    let mut values = (0..64u32).collect::<Vec<_>>();
    values[20] = u32::MAX;
    DenseWriter::new(
        &path,
        Partition::fixed_cells(1024),
        Partition::fixed_cells(16),
    )
    .write(&values, [1, 64])
    .unwrap();
    let source = || {
        Source::new(0, Dataset::open(&path).unwrap()).feature_map(
            FeatureMap::new(
                (0..64)
                    .map(|column| (column % 2 == 0).then_some(column / 2))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        )
    };

    let plan = compile(PlanSpec::new(
        vec![source()],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(32, OutputDType::I32, Fill::I32(-1)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    assert!(matches!(
        plan.inner.source_plans[0].dense_map,
        Some(crate::plan::DenseMap::Gather32 { .. })
    ));
    let mut session = plan.open(blocking(1)).unwrap();
    assert!(matches!(session.next_batch(), Err(Error::Session(_))));

    let output = OutputSpec::new(32, OutputDType::I32, Fill::I32(-1))
        .unwrap()
        .overflow(OverflowPolicy::UseValue(Fill::I32(-99)))
        .unwrap();
    let plan = compile(PlanSpec::new(
        vec![source()],
        vec![RowRef::new(SourceId::new(0), 0)],
        output,
        1,
        2,
    ))
    .unwrap();
    assert!(matches!(
        plan.inner.source_plans[0].dense_map,
        Some(crate::plan::DenseMap::Gather32 { .. })
    ));
    let mut session = plan.open(blocking(1)).unwrap();
    assert_eq!(
        session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<i32>(0)
            .unwrap(),
        &(0..64)
            .step_by(2)
            .map(|value| if value == 20 { -99 } else { value })
            .collect::<Vec<_>>()
    );
}

#[test]
fn contiguous_identity_block_uses_one_static_cache_residency_and_one_ring_slot() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-direct-block");
    let values = (0..128u16).collect::<Vec<_>>();
    DenseWriter::new(&path, Partition::fixed_cells(4), Partition::fixed_cells(4))
        .write(&values, [4, 32])
        .unwrap();
    let rows = (0..4)
        .map(|row| RowRef::new(SourceId::new(0), row))
        .collect();
    let plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        rows,
        OutputSpec::new(32, OutputDType::U16, Fill::U16(0)).unwrap(),
        4,
        8,
    ))
    .unwrap();
    assert_eq!(plan.batch_count(), 1);
    assert_eq!(plan.inner.ring_slots, 1);
    assert_eq!(plan.stats().output_ring_bytes, 4 * 64);
    assert_eq!(plan.stats().unique_cache_objects, 1);
    assert_eq!(plan.stats().residency_loads, 1);
    assert_eq!(plan.stats().initialize_io_tasks, 1);

    let mut session = plan.open(blocking(2)).unwrap();
    let batch = session.next_batch().unwrap().unwrap();
    for row in 0..4 {
        assert_eq!(
            batch.row_as::<u16>(row).unwrap(),
            &values[row * 32..(row + 1) * 32]
        );
    }
}

#[test]
fn adjacent_cache_objects_share_one_physical_read_without_sharing_residency() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-adjacent-blocks");
    let values = (0..512u16).collect::<Vec<_>>();
    DenseWriter::new(&path, Partition::fixed_cells(8), Partition::fixed_cells(1))
        .write(&values, [8, 64])
        .unwrap();
    let source_id = SourceId::new(0);
    let source = Source::new(source_id, Dataset::open(&path).unwrap());
    let rows = [7, 0, 6, 1, 5, 2, 4, 3]
        .into_iter()
        .map(|row| RowRef::new(source_id, row))
        .collect();
    let output = OutputSpec::new(64, OutputDType::U16, Fill::U16(0)).unwrap();
    let config = PlanConfig {
        io_merge: IoMergeOptions {
            initialize_parallelism_hint: 1,
            regular_io_parallelism_hint: 1,
            min_tasks_per_worker: 1,
            ..IoMergeOptions::default()
        },
        ..PlanConfig::default()
    };
    let plan = compile(PlanSpec::new(vec![source], rows, output, 8, 2).config(config)).unwrap();
    assert_eq!(plan.stats().block_jobs, 8);
    assert_eq!(plan.stats().jobs, 1);
    assert_eq!(plan.stats().unique_cache_objects, 8);
    assert_eq!(plan.stats().data_io_ops, 1);
}

#[test]
fn shuffled_physical_blocks_compile_exactly_one_job_per_logical_batch() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-batch-completions");
    DenseWriter::new(&path, Partition::fixed_cells(8), Partition::fixed_cells(1))
        .write(&(0..8u16).collect::<Vec<_>>(), [8, 1])
        .unwrap();
    let source_id = SourceId::new(0);
    let requested = [0u16, 4, 1, 5, 2, 6, 3, 7];
    let rows = requested
        .into_iter()
        .map(|row| RowRef::new(source_id, u64::from(row)))
        .collect();
    let config = PlanConfig {
        ..PlanConfig::default()
    };
    let plan = compile(
        PlanSpec::new(
            vec![Source::new(source_id, Dataset::open(&path).unwrap())],
            rows,
            OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap(),
            2,
            5,
        )
        .config(config),
    )
    .unwrap();
    assert_eq!(plan.inner.static_plan.jobs.len(), 4);
    assert!(plan
        .inner
        .static_plan
        .jobs
        .iter()
        .enumerate()
        .all(|(batch, job)| job.batch_id == batch as u64));
    let mut session = plan.open(blocking(2)).unwrap();
    let mut observed = Vec::new();
    while let Some(batch) = session.next_batch().unwrap() {
        for row in 0..batch.rows() {
            observed.push(batch.row_as::<u16>(row).unwrap()[0]);
        }
    }
    assert_eq!(observed, requested);
}

#[test]
fn cache_objects_are_not_merged_into_cross_block_runtime_jobs() {
    assert_eq!(
        PlanConfig::default().io_merge.max_coalesced_io_bytes,
        32 * 1024 * 1024
    );

    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-coalesced-io-limit");
    let values = (0..512u16).collect::<Vec<_>>();
    DenseWriter::new(&path, Partition::fixed_cells(8), Partition::fixed_cells(1))
        .write(&values, [8, 64])
        .unwrap();
    let source_id = SourceId::new(0);
    let source = Source::new(source_id, Dataset::open(&path).unwrap());
    let rows = (0..8).map(|row| RowRef::new(source_id, row)).collect();
    let output = OutputSpec::new(64, OutputDType::U16, Fill::U16(0)).unwrap();
    let config = PlanConfig {
        io_merge: IoMergeOptions {
            max_coalesced_io_bytes: 1,
            ..IoMergeOptions::default()
        },
        ..PlanConfig::default()
    };
    let plan = compile(PlanSpec::new(vec![source], rows, output, 8, 2).config(config)).unwrap();
    assert_eq!(plan.stats().block_jobs, 8);
    assert_eq!(plan.stats().jobs, 1);
    assert_eq!(plan.stats().data_io_ops as usize, plan.stats().block_jobs);
}

#[test]
fn decoded_cache_replaces_per_worker_decoded_scratch() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("decoded-soft-target");
    let values = (0..128u16).collect::<Vec<_>>();
    DenseWriter::new(&path, Partition::fixed_cells(4), Partition::fixed_cells(1))
        .write(&values, [4, 32])
        .unwrap();
    let rows = (0..4)
        .map(|row| RowRef::new(SourceId::new(0), row))
        .collect();
    let config = PlanConfig::default();
    let plan = compile(
        PlanSpec::new(
            vec![Source::new(0, Dataset::open(&path).unwrap())],
            rows,
            OutputSpec::new(32, OutputDType::U32, Fill::U32(0)).unwrap(),
            4,
            2,
        )
        .config(config),
    )
    .unwrap();
    assert_eq!(plan.stats().block_jobs, 4);
    assert_eq!(plan.stats().jobs, 1);
    assert_eq!(plan.stats().maximum_decoded_bytes_per_job, 256);
    assert_eq!(
        plan.stats().cache_arena_bytes,
        PlanConfig::default().cache_capacity_bytes
    );

    let mut session_config = blocking(1);
    session_config.max_decoded_bytes_per_worker = 128;
    session_config.max_total_decoded_bytes = 128;
    let mut session = plan.open(session_config).unwrap();
    let batch = session.next_batch().unwrap().unwrap();
    for row in 0..4 {
        let expected = values[row * 32..(row + 1) * 32]
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        assert_eq!(batch.row_as::<u32>(row).unwrap(), expected);
    }
}

#[test]
fn dense_plan_compacts_used_blocks_across_many_chunks() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-many-chunks");
    let values: Vec<u16> = (0..40).collect();
    DenseWriter::new(&path, Partition::fixed_cells(4), Partition::fixed_cells(1))
        .write(&values, [40, 1])
        .unwrap();
    let source_id = SourceId::new(9);
    let source = Source::new(source_id, Dataset::open(&path).unwrap());
    let selected = (0..10)
        .map(|chunk| RowRef::new(source_id, chunk * 4))
        .collect::<Vec<_>>();
    let output = OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap();
    let config = PlanConfig {
        compile_io_concurrency: 4,
        ..PlanConfig::default()
    };
    let plan = compile(PlanSpec::new(vec![source], selected, output, 2, 8).config(config)).unwrap();
    let mut session = plan.open(blocking(2)).unwrap();
    let mut observed = Vec::new();
    while let Some(batch) = session.next_batch().unwrap() {
        for row in 0..batch.rows() {
            observed.push(batch.row_as::<u16>(row).unwrap()[0]);
        }
    }
    assert_eq!(observed, (0..10).map(|chunk| chunk * 4).collect::<Vec<_>>());
}

#[test]
fn csr_empty_rows_and_scatter_are_committed_as_complete_batches() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("csr");
    CsrWriter::new(&path, Partition::fixed_cells(4), Partition::fixed_cells(2))
        .write(
            &[0u64, 2, 2, 4, 5],
            &[0u32, 3, 1, 4, 2],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0],
            [4, 5],
        )
        .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap());
    let rows = [1, 2, 0, 3]
        .into_iter()
        .map(|row| RowRef::new(SourceId::new(0), row))
        .collect();
    let output = OutputSpec::new(5, OutputDType::F32, Fill::F32(0.0)).unwrap();
    let plan = compile(PlanSpec::new(vec![source], rows, output, 2, 3)).unwrap();
    let mut session = plan.open(blocking(2)).unwrap();
    let mut rows_out = Vec::new();
    while let Some(batch) = session.next_batch().unwrap() {
        for row in 0..batch.rows() {
            rows_out.push(batch.row_as::<f32>(row).unwrap().to_vec());
        }
    }
    assert_eq!(
        rows_out,
        vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0, 0.0, 4.0],
            vec![1.0, 0.0, 0.0, 2.0, 0.0],
            vec![0.0, 0.0, 5.0, 0.0, 0.0],
        ]
    );
}

#[test]
fn csr_compact_mapping_uses_byte_targets_and_skips_dropped_conversion_checks() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("csr-mapped");
    CsrWriter::new(
        &path,
        Partition::fixed_cells(1024),
        Partition::fixed_cells(16),
    )
    .write(&[0u64, 3, 3], &[0u32, 1, 2], &[5i16, -1, 7], [2, 3])
    .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap())
        .feature_map(FeatureMap::new([Some(2), None, Some(0)]).unwrap());
    let plan = compile(PlanSpec::new(
        vec![source],
        vec![
            RowRef::new(SourceId::new(0), 0),
            RowRef::new(SourceId::new(0), 1),
        ],
        OutputSpec::new(3, OutputDType::U32, Fill::U32(99)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    assert_eq!(
        plan.inner.source_plans[0].default_ranges.as_ref(),
        [crate::plan::OutputRange { offset: 4, len: 4 }]
    );
    let mut session = plan.open(blocking(1)).unwrap();
    let mut rows = Vec::new();
    while let Some(batch) = session.next_batch().unwrap() {
        rows.push(batch.row_as::<u32>(0).unwrap().to_vec());
    }
    assert_eq!(rows, vec![vec![7, 99, 5], vec![0, 99, 0]],);
}

#[test]
fn empty_plan_finishes_without_workers_or_output_allocation() {
    let output = OutputSpec::new(3, OutputDType::F32, Fill::F32(0.0)).unwrap();
    let plan = compile(PlanSpec::new(vec![], vec![], output, 4, 2)).unwrap();
    assert!(plan.is_empty());
    assert_eq!(plan.stats().output_ring_bytes, 0);
    let mut session = plan.open(blocking(1)).unwrap();
    assert!(session.next_batch().unwrap().is_none());
}

#[test]
fn session_rejects_zero_io_uring_queue_depth() {
    let output = OutputSpec::new(1, OutputDType::F32, Fill::F32(0.0)).unwrap();
    let plan = compile(PlanSpec::new(vec![], vec![], output, 1, 2)).unwrap();
    for io_mode in [
        IoMode::Uring { queue_depth: 0 },
        IoMode::Auto { queue_depth: 0 },
    ] {
        assert!(matches!(
            plan.open(SessionConfig {
                worker_count: 1,
                io_mode,
                ..SessionConfig::default()
            }),
            Err(Error::InvalidConfig(_))
        ));
    }
}

#[test]
fn feature_map_rejects_duplicate_targets() {
    let error = FeatureMap::new([Some(1), Some(1)]).unwrap_err();
    assert!(matches!(error, Error::InvalidInput(_)));
}

#[test]
fn whole_key_plan_rejects_explicit_uring_and_auto_uses_blocking() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense");
    DenseWriter::new(&path, Partition::fixed_cells(2), Partition::fixed_cells(1))
        .write(&[1u16, 2, 3, 4], [2, 2])
        .unwrap();
    let memory = Arc::new(MemoryStore::from_directory(&path, &["meta.json", "data/0"]));
    let matrix = DenseMatrix::from_store(memory).unwrap();
    let source = Source::new(1, Dataset::from_dense(matrix));
    let output = OutputSpec::new(2, OutputDType::U16, Fill::U16(0)).unwrap();
    let plan = compile(PlanSpec::new(
        vec![source],
        vec![RowRef::new(SourceId::new(1), 0)],
        output,
        1,
        2,
    ))
    .unwrap();
    let explicit = SessionConfig {
        worker_count: 1,
        io_mode: IoMode::Uring { queue_depth: 2 },
        ..SessionConfig::default()
    };
    assert!(matches!(plan.open(explicit), Err(Error::Unsupported(_))));

    let mut auto = plan
        .open(SessionConfig {
            worker_count: 1,
            io_mode: IoMode::Auto { queue_depth: 2 },
            ..SessionConfig::default()
        })
        .unwrap();
    assert_eq!(auto.stats().actual_io_mode, IoMode::Blocking);
    assert_eq!(
        auto.next_batch().unwrap().unwrap().row(0).unwrap(),
        &[1, 0, 2, 0]
    );
}

#[cfg(all(feature = "uring", target_os = "linux"))]
#[test]
fn uring_executes_positioned_static_graph_and_preserves_batches() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-uring");
    let values: Vec<u32> = (0..64).collect();
    DenseWriter::new(&path, Partition::fixed_cells(4), Partition::fixed_cells(1))
        .write(&values, [16, 4])
        .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap());
    let rows = (0..16)
        .rev()
        .map(|row| RowRef::new(SourceId::new(0), row))
        .collect();
    let output = OutputSpec::new(4, OutputDType::U32, Fill::U32(0)).unwrap();
    let plan = compile(
        PlanSpec::new(vec![source], rows, output, 2, 4).config(PlanConfig {
            cache_capacity_bytes: 128,
            ..PlanConfig::default()
        }),
    )
    .unwrap();
    let mut session = plan
        .open(SessionConfig {
            worker_count: 1,
            io_mode: IoMode::Uring { queue_depth: 8 },
            max_inflight_jobs_per_worker: 8,
            ..SessionConfig::default()
        })
        .unwrap();
    let mut first_values = Vec::new();
    while let Some(batch) = session.next_batch().unwrap() {
        for row in 0..batch.rows() {
            first_values.push(batch.row_as::<u32>(row).unwrap()[0]);
        }
    }
    assert_eq!(
        first_values,
        (0..16).rev().map(|row| row * 4).collect::<Vec<_>>()
    );
    let stats = session.stats();
    assert_eq!(stats.actual_io_mode, IoMode::Uring { queue_depth: 8 });
    #[cfg(feature = "profile")]
    {
        assert!(stats.uring_submitted_read_sqes >= 1);
        assert_eq!(stats.uring_submitted_read_sqes, stats.uring_cqes);
        assert!(stats.peak_inflight_read_ops > 1);
        assert!(stats.io_wait_nanoseconds > 0);
        assert!(stats.decode_nanoseconds > 0);
        assert!(stats.scatter_nanoseconds > 0);
        assert!(stats.completion_nanoseconds > 0);
    }
}

#[cfg(all(feature = "uring", target_os = "linux"))]
#[test]
fn uring_multi_completion_failure_drains_before_shutdown() {
    const CHILD_ENV: &str = "SC_LOAD_TEST_URING_MULTI_CQE_ERROR";
    const TEST_NAME: &str = "tests::uring_multi_completion_failure_drains_before_shutdown";

    if std::env::var_os(CHILD_ENV).is_none() {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(CHILD_ENV, "1")
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "io_uring failure child exited with {status}"
                );
                return;
            }
            if std::time::Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("io_uring failure cleanup did not terminate");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-uring-failure");
    let values: Vec<u32> = (0..4096).collect();
    DenseWriter::new(&path, Partition::fixed_cells(4), Partition::fixed_cells(1))
        .write(&values, [1024, 4])
        .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap());
    let rows = (0..1024)
        .map(|row| RowRef::new(SourceId::new(0), row))
        .collect();
    let output = OutputSpec::new(4, OutputDType::U32, Fill::U32(0)).unwrap();
    let plan = compile(
        PlanSpec::new(vec![source], rows, output, 2, 32).config(PlanConfig {
            cache_capacity_bytes: 256,
            ..PlanConfig::default()
        }),
    )
    .unwrap();
    let server = plan
        .open_shared(
            SessionConfig {
                worker_count: 1,
                initialize_workers: 1,
                initialize_inflight_io_ops: 1,
                io_mode: IoMode::Uring { queue_depth: 8 },
                max_inflight_jobs_per_worker: 8,
                ..SessionConfig::default()
            },
            crate::SharedConfig::new(1).unwrap(),
        )
        .unwrap();
    let descriptor = server.attach_fd().unwrap();
    let consumer = std::thread::spawn(move || {
        use std::os::fd::AsFd;

        let mut client = crate::SharedClient::attach(descriptor.as_fd(), 0).unwrap();
        loop {
            match client.next_batch() {
                Ok(Some(batch)) => batch.release().unwrap(),
                result => return result,
            }
        }
    });
    assert!(server.run().is_err());
    assert!(consumer.join().unwrap().is_err());
}

#[test]
fn compiled_plan_pins_store_generation_and_sessions_own_their_rings() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("generation");
    DenseWriter::new(&path, Partition::fixed_cells(2), Partition::fixed_cells(1))
        .write(&[1u16, 2, 3, 4], [2, 2])
        .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap());
    let output = OutputSpec::new(2, OutputDType::U16, Fill::U16(0)).unwrap();
    let plan = compile(PlanSpec::new(
        vec![source],
        vec![
            RowRef::new(SourceId::new(0), 0),
            RowRef::new(SourceId::new(0), 1),
        ],
        output,
        1,
        2,
    ))
    .unwrap();

    DenseWriter::new(&path, Partition::fixed_cells(2), Partition::fixed_cells(1))
        .write(&[9u16, 10, 11, 12], [2, 2])
        .unwrap();

    let mut first = plan.open(blocking(1)).unwrap();
    let mut second = plan.open(blocking(2)).unwrap();
    for session in [&mut first, &mut second] {
        assert_eq!(
            session.next_batch().unwrap().unwrap().row(0).unwrap(),
            &[1, 0, 2, 0]
        );
        assert_eq!(
            session.next_batch().unwrap().unwrap().row(0).unwrap(),
            &[3, 0, 4, 0]
        );
        assert!(session.next_batch().unwrap().is_none());
    }
}

#[test]
fn cancellation_is_terminal_and_wakes_consumer() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("cancel");
    let values: Vec<u32> = (0..4096).collect();
    DenseWriter::new(&path, Partition::fixed_cells(16), Partition::fixed_cells(1))
        .write(&values, [1024, 4])
        .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap());
    let rows = (0..1024)
        .map(|row| RowRef::new(SourceId::new(0), row))
        .collect();
    let output = OutputSpec::new(4, OutputDType::U32, Fill::U32(0)).unwrap();
    let plan = compile(PlanSpec::new(vec![source], rows, output, 8, 4)).unwrap();
    let mut session = plan.open(blocking(2)).unwrap();
    let cancellation = session.cancellation_handle();
    std::thread::spawn(move || cancellation.cancel())
        .join()
        .unwrap();
    assert!(matches!(session.next_batch(), Err(Error::Cancelled)));
    assert_eq!(session.state(), crate::SessionState::Cancelled);
}

#[test]
fn batch_ready_publication_is_linearized_with_consumer_sleep() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("consumer-wakeup");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[7u16, 9], [1, 2])
        .unwrap();
    let plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(2, OutputDType::U16, Fill::U16(0)).unwrap(),
        1,
        1,
    ))
    .unwrap();

    for _ in 0..256 {
        let mut session = plan.open(blocking(2)).unwrap();
        let batch = session.next_batch().unwrap().unwrap();
        assert_eq!(batch.row_as::<u16>(0).unwrap(), &[7, 9]);
    }
}

#[test]
fn stored_zip_is_positioned_while_deflated_zip_materializes_whole_keys() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("zip-source");
    DenseWriter::new(&root, Partition::fixed_cells(2), Partition::fixed_cells(1))
        .write(&[1u16, 2, 3, 4], [2, 2])
        .unwrap();

    for method in [
        zip::CompressionMethod::Stored,
        zip::CompressionMethod::Deflated,
    ] {
        let archive = temporary.path().join(format!("{method:?}.zip"));
        let file = std::fs::File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default().compression_method(method);
        for key in ["meta.json", "data/0"] {
            writer.start_file(format!("assay/{key}"), options).unwrap();
            writer
                .write_all(&std::fs::read(root.join(key)).unwrap())
                .unwrap();
        }
        writer.finish().unwrap();

        let dataset = Dataset::open(crate::StoreLocation::zip(&archive, "assay")).unwrap();
        let output = OutputSpec::new(2, OutputDType::U16, Fill::U16(0)).unwrap();
        let plan = compile(PlanSpec::new(
            vec![Source::new(0, dataset)],
            vec![RowRef::new(SourceId::new(0), 1)],
            output,
            1,
            2,
        ))
        .unwrap();
        let config = if method == zip::CompressionMethod::Stored
            && cfg!(all(feature = "uring", target_os = "linux"))
        {
            SessionConfig {
                worker_count: 1,
                io_mode: IoMode::Uring { queue_depth: 2 },
                ..SessionConfig::default()
            }
        } else {
            SessionConfig {
                worker_count: 1,
                io_mode: IoMode::Auto { queue_depth: 2 },
                ..SessionConfig::default()
            }
        };
        let mut session = plan.open(config).unwrap();
        if method == zip::CompressionMethod::Deflated {
            assert_eq!(session.stats().actual_io_mode, IoMode::Blocking);
        }
        assert_eq!(
            session.next_batch().unwrap().unwrap().row(0).unwrap(),
            &[3, 0, 4, 0]
        );
    }
}

#[test]
fn promote_rejects_narrowing_at_compile_time() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("narrow");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[1i32, 2], [1, 2])
        .unwrap();
    let err = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(2, OutputDType::I16, Fill::I16(0)).unwrap(),
        1,
        2,
    ));
    assert!(matches!(err, Err(Error::Promote(_))));
}

#[test]
fn signed_to_unsigned_error_policy_fails_session() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("neg");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[-3i16, 4], [1, 2])
        .unwrap();
    let output = OutputSpec::new(2, OutputDType::U16, Fill::U16(0)).unwrap();
    let plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        vec![RowRef::new(SourceId::new(0), 0)],
        output,
        1,
        2,
    ))
    .unwrap();
    let mut session = plan.open(blocking(1)).unwrap();
    assert!(matches!(session.next_batch(), Err(Error::Session(_))));
    assert_eq!(session.state(), crate::SessionState::Failed);
}

#[test]
fn signed_to_unsigned_use_fill_writes_fill() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("neg-fill");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[-3i16, 4], [1, 2])
        .unwrap();
    let output = OutputSpec::new(2, OutputDType::U16, Fill::U16(99))
        .unwrap()
        .overflow(OverflowPolicy::UseFill)
        .unwrap();
    let plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        vec![RowRef::new(SourceId::new(0), 0)],
        output,
        1,
        2,
    ))
    .unwrap();
    let mut session = plan.open(blocking(1)).unwrap();
    let batch = session.next_batch().unwrap().unwrap();
    assert_eq!(batch.row_as::<u16>(0).unwrap(), &[99, 4]);
}

#[test]
fn signed_to_unsigned_use_value_writes_sentinel() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("neg-sentinel");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[-3i16, 4], [1, 2])
        .unwrap();
    let output = OutputSpec::new(2, OutputDType::U16, Fill::U16(0))
        .unwrap()
        .overflow(OverflowPolicy::UseValue(Fill::U16(777)))
        .unwrap();
    let plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        vec![RowRef::new(SourceId::new(0), 0)],
        output,
        1,
        2,
    ))
    .unwrap();
    let mut session = plan.open(blocking(1)).unwrap();
    let batch = session.next_batch().unwrap().unwrap();
    assert_eq!(batch.row_as::<u16>(0).unwrap(), &[777, 4]);
}

#[test]
fn int_to_float_promotion_is_allowed() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("i2f");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[1i16, 2], [1, 2])
        .unwrap();
    let plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(2, OutputDType::F32, Fill::F32(0.0)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    let mut session = plan.open(blocking(1)).unwrap();
    let batch = session.next_batch().unwrap().unwrap();
    assert_eq!(batch.row_as::<f32>(0).unwrap(), &[1.0, 2.0]);
}

#[test]
fn potentially_rounding_integer_to_float_requires_explicit_policy() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("rounding-i2f");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[16_777_217i32], [1, 1])
        .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap());
    let rows = vec![RowRef::new(SourceId::new(0), 0)];
    let exact = compile(PlanSpec::new(
        vec![source.clone()],
        rows.clone(),
        OutputSpec::new(1, OutputDType::F32, Fill::F32(0.0)).unwrap(),
        1,
        2,
    ));
    assert!(matches!(exact, Err(Error::Promote(_))));

    let output = OutputSpec::new(1, OutputDType::F32, Fill::F32(0.0))
        .unwrap()
        .float_cast(crate::FloatCastPolicy::AllowRounding);
    let plan = compile(PlanSpec::new(vec![source], rows, output, 1, 2)).unwrap();
    let mut session = plan.open(blocking(1)).unwrap();
    assert_eq!(
        session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<f32>(0)
            .unwrap(),
        &[16_777_216.0]
    );
}

#[test]
fn int64_and_uint64_promotion_rules_are_explicit() {
    use crate::{promote_kind, PromoteKind, StorageDType};

    assert_eq!(
        promote_kind(StorageDType::I64, OutputDType::I64),
        Some(PromoteKind::Lossless)
    );
    assert_eq!(
        promote_kind(StorageDType::U64, OutputDType::U64),
        Some(PromoteKind::Lossless)
    );
    assert_eq!(
        promote_kind(StorageDType::I32, OutputDType::I64),
        Some(PromoteKind::Lossless)
    );
    assert_eq!(
        promote_kind(StorageDType::U32, OutputDType::I64),
        Some(PromoteKind::Lossless)
    );
    assert_eq!(
        promote_kind(StorageDType::I64, OutputDType::U64),
        Some(PromoteKind::CheckedSign)
    );
    assert_eq!(
        promote_kind(StorageDType::U64, OutputDType::I64),
        Some(PromoteKind::CheckedSign)
    );
    assert_eq!(
        promote_kind(StorageDType::I64, OutputDType::F64),
        Some(PromoteKind::RoundingToFloat)
    );
    assert_eq!(promote_kind(StorageDType::U64, OutputDType::F32), None);
}

#[test]
fn padded_batch_storage_is_initialized_and_not_reported_as_compact() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("padded-output");
    DenseWriter::new(&path, Partition::fixed_cells(2), Partition::fixed_cells(1))
        .write(&[1u16, 2, 3, 4], [2, 2])
        .unwrap();
    let plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        vec![
            RowRef::new(SourceId::new(0), 0),
            RowRef::new(SourceId::new(0), 1),
        ],
        OutputSpec::new(2, OutputDType::U16, Fill::U16(9)).unwrap(),
        2,
        2,
    ))
    .unwrap();
    let mut session = plan.open(blocking(2)).unwrap();
    let batch = session.next_batch().unwrap().unwrap();
    assert_eq!(batch.row_stride_bytes(), 64);
    assert!(matches!(
        batch.as_slice::<u16>(),
        Err(Error::Unsupported(_))
    ));
    let padded = batch.as_padded_slice::<u16>().unwrap();
    assert_eq!(&padded[..2], &[1, 2]);
    assert!(padded[2..32].iter().all(|value| *value == 0));
    assert_eq!(&padded[32..34], &[3, 4]);
    assert!(padded[34..64].iter().all(|value| *value == 0));
}

#[test]
fn output_spec_is_revalidated_at_compile_boundary() {
    let mut output = OutputSpec::new(1, OutputDType::F32, Fill::F32(0.0)).unwrap();
    output.overflow = OverflowPolicy::UseValue(Fill::U16(7));
    let result = compile(PlanSpec::new(vec![], vec![], output, 1, 2));
    assert!(matches!(result, Err(Error::InvalidInput(_))));
}

#[test]
fn feature_map_output_bounds_are_checked_by_the_plan() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("map-bounds");
    DenseWriter::new(
        &path,
        Partition::fixed_cells(1024),
        Partition::fixed_cells(16),
    )
    .write(&[1u16], [1, 1])
    .unwrap();
    let source = Source::new(0, Dataset::open(&path).unwrap())
        .feature_map(FeatureMap::new([Some(1)]).unwrap());
    let result = compile(PlanSpec::new(
        vec![source],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap(),
        1,
        2,
    ));
    assert!(matches!(result, Err(Error::InvalidInput(_))));
}

#[test]
fn csr_indices_must_be_strictly_increasing() {
    use crate::convert::ConvertOp;
    use crate::plan::{CellTask, SourcePlan};
    use crate::source::OutputSlot;

    let output = OutputSpec::new(3, OutputDType::U16, Fill::U16(0)).unwrap();
    let source = SourcePlan {
        n_cols: 3,
        value_dtype: sc_compress::DType::U16,
        index: crate::scatter::IndexOp::new(sc_compress::DType::U16),
        feature_map: None,
        csr_sparse_map: None,
        dense_map: None,
        dense_fill_whole: false,
        default_ranges: Default::default(),
        convert: ConvertOp::resolve(sc_compress::DType::U16, &output).unwrap(),
    };
    let task = CellTask::new(OutputSlot::new(0).unwrap(), 0..4, Some(0..4)).unwrap();
    let data = [1u16, 2]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let duplicate_indices = [1u16, 1]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    assert!(matches!(
        crate::scatter::validate_row(&source, &task, &data, &duplicate_indices),
        Err(Error::Decode(_))
    ));
}

#[test]
fn wide_mapping_fallback_preserves_dense_and_csr_semantics() {
    use std::sync::Arc;

    use crate::convert::ConvertOp;
    use crate::plan::{CellTask, CsrMap, DenseMap, DenseMapEntry, SourcePlan, UNMAPPED_TARGET};
    use crate::source::OutputSlot;

    let output = OutputSpec::new(2, OutputDType::U16, Fill::U16(99)).unwrap();
    let task = CellTask::new(OutputSlot::new(0).unwrap(), 0..6, None).unwrap();
    let data = [5u16, 6, 7]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let dense = SourcePlan {
        n_cols: 3,
        value_dtype: sc_compress::DType::U16,
        index: None,
        feature_map: None,
        csr_sparse_map: None,
        dense_map: Some(DenseMap::Wide {
            entries: Arc::from([
                DenseMapEntry {
                    source_byte: 0,
                    target_byte: 2,
                },
                DenseMapEntry {
                    source_byte: 4,
                    target_byte: 0,
                },
            ]),
            covers_output: true,
        }),
        dense_fill_whole: false,
        default_ranges: Default::default(),
        convert: ConvertOp::resolve(sc_compress::DType::U16, &output).unwrap(),
    };
    crate::scatter::validate_row(&dense, &task, &data, &[]).unwrap();
    let mut row = [0u8; 4];
    // SAFETY: validation just established the exact dense task and buffer
    // invariants, and `row` uniquely owns the complete logical output.
    unsafe {
        crate::scatter::scatter_row_prevalidated(
            &dense,
            &task,
            &data,
            &[],
            &mut row,
            4,
            crate::scatter::FillOp::new(&99u16.to_le_bytes()),
        )
        .unwrap();
    }
    assert_eq!(
        row.chunks_exact(2)
            .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>(),
        [7, 5]
    );

    let csr_task = CellTask::new(OutputSlot::new(0).unwrap(), 0..4, Some(0..4)).unwrap();
    let csr_data = [5u16, 7]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let csr_indices = [0u16, 2]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let csr = SourcePlan {
        n_cols: 3,
        value_dtype: sc_compress::DType::U16,
        index: crate::scatter::IndexOp::new(sc_compress::DType::U16),
        feature_map: Some(CsrMap::Wide(Arc::from([2, UNMAPPED_TARGET, 0]))),
        csr_sparse_map: None,
        dense_map: None,
        dense_fill_whole: false,
        default_ranges: Default::default(),
        convert: ConvertOp::resolve(sc_compress::DType::U16, &output).unwrap(),
    };
    crate::scatter::validate_row(&csr, &csr_task, &csr_data, &csr_indices).unwrap();
    row.fill(0);
    // SAFETY: validation established the CSR structure, mapping, and unique
    // output extent for these exact buffers.
    unsafe {
        crate::scatter::scatter_row_prevalidated(
            &csr,
            &csr_task,
            &csr_data,
            &csr_indices,
            &mut row,
            4,
            crate::scatter::FillOp::new(&99u16.to_le_bytes()),
        )
        .unwrap();
    }
    assert_eq!(
        row.chunks_exact(2)
            .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>(),
        [7, 5]
    );
}

#[test]
fn mapped_validation_only_checks_values_with_a_destination() {
    use std::sync::Arc;

    use crate::convert::ConvertOp;
    use crate::plan::{CellTask, CsrMap, DenseMap, DenseMapEntry, SourcePlan, UNMAPPED_TARGET};
    use crate::source::OutputSlot;

    let output = OutputSpec::new(2, OutputDType::I16, Fill::I16(0)).unwrap();
    let convert = ConvertOp::resolve(sc_compress::DType::U16, &output).unwrap();
    let dense = SourcePlan {
        n_cols: 3,
        value_dtype: sc_compress::DType::U16,
        index: None,
        feature_map: None,
        csr_sparse_map: None,
        dense_map: Some(DenseMap::Wide {
            entries: Arc::from([
                DenseMapEntry {
                    source_byte: 0,
                    target_byte: 0,
                },
                DenseMapEntry {
                    source_byte: 4,
                    target_byte: 2,
                },
            ]),
            covers_output: true,
        }),
        dense_fill_whole: false,
        default_ranges: Default::default(),
        convert,
    };
    let dense_task = CellTask::new(OutputSlot::new(0).unwrap(), 0..6, None).unwrap();
    let unselected_overflow = [5u16, 40_000, 7]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    crate::scatter::validate_row(&dense, &dense_task, &unselected_overflow, &[]).unwrap();

    let csr = SourcePlan {
        n_cols: 3,
        value_dtype: sc_compress::DType::U16,
        index: crate::scatter::IndexOp::new(sc_compress::DType::U16),
        feature_map: Some(CsrMap::Wide(Arc::from([0, UNMAPPED_TARGET, 2]))),
        csr_sparse_map: None,
        dense_map: None,
        dense_fill_whole: false,
        default_ranges: Default::default(),
        convert,
    };
    let csr_task = CellTask::new(OutputSlot::new(0).unwrap(), 0..6, Some(0..6)).unwrap();
    let indices = [0u16, 1, 2]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    crate::scatter::validate_row(&csr, &csr_task, &unselected_overflow, &indices).unwrap();

    let selected_overflow = [40_000u16, 6, 7]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    assert!(matches!(
        crate::scatter::validate_row(&dense, &dense_task, &selected_overflow, &[]),
        Err(Error::Conversion(_))
    ));
    assert!(matches!(
        crate::scatter::validate_row(&csr, &csr_task, &selected_overflow, &indices),
        Err(Error::Conversion(_))
    ));
}

#[test]
fn packed_csr_fallback_conversion_selects_map_once_per_row() {
    use crate::convert::ConvertOp;
    use crate::plan::{CellTask, CsrMap, SourcePlan, UNMAPPED_TARGET_U32};
    use crate::source::OutputSlot;

    let output = OutputSpec::new(2, OutputDType::I16, Fill::I16(-1))
        .unwrap()
        .overflow(OverflowPolicy::UseValue(Fill::I16(7)))
        .unwrap();
    let source = SourcePlan {
        n_cols: 3,
        value_dtype: sc_compress::DType::U16,
        index: crate::scatter::IndexOp::new(sc_compress::DType::U16),
        feature_map: Some(CsrMap::Packed32(Arc::from([2, 0, UNMAPPED_TARGET_U32]))),
        csr_sparse_map: None,
        dense_map: None,
        dense_fill_whole: false,
        default_ranges: Default::default(),
        convert: ConvertOp::resolve(sc_compress::DType::U16, &output).unwrap(),
    };
    let task = CellTask::new(OutputSlot::new(0).unwrap(), 0..6, Some(0..6)).unwrap();
    let values = [5u16, 40_000, 9]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let indices = [0u16, 1, 2]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    crate::scatter::validate_row(&source, &task, &values, &indices).unwrap();
    let mut row = [0u8; 4];
    // SAFETY: validation established the exact CSR extents, packed targets,
    // conversion policy, and unique output row used below.
    unsafe {
        crate::scatter::scatter_row_prevalidated(
            &source,
            &task,
            &values,
            &indices,
            &mut row,
            4,
            crate::scatter::FillOp::new(&(-1i16).to_le_bytes()),
        )
        .unwrap();
    }
    assert_eq!(
        row.chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>(),
        [7, 5]
    );
}

#[test]
fn rejects_non_finite_merge_threshold() {
    let output = OutputSpec::new(1, OutputDType::F32, Fill::F32(0.0)).unwrap();
    let config = PlanConfig {
        io_merge: IoMergeOptions {
            policy: IoMergePolicy::CostAware,
            io_bandwidth_bytes_per_second: f64::INFINITY,
            ..IoMergeOptions::default()
        },
        ..PlanConfig::default()
    };
    let error = match compile(PlanSpec::new(vec![], vec![], output, 1, 2).config(config)) {
        Ok(_) => panic!("overflowed merge threshold must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, crate::Error::InvalidConfig(_)));
}

#[test]
fn rejects_zero_coalesced_io_limit() {
    let output = OutputSpec::new(1, OutputDType::F32, Fill::F32(0.0)).unwrap();
    let config = PlanConfig {
        io_merge: IoMergeOptions {
            max_coalesced_io_bytes: 0,
            ..IoMergeOptions::default()
        },
        ..PlanConfig::default()
    };
    let error = match compile(PlanSpec::new(vec![], vec![], output, 1, 2).config(config)) {
        Ok(_) => panic!("zero coalesced I/O limit must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, crate::Error::InvalidConfig(_)));
}

#[test]
fn bulk_conversion_kernels_cover_extremes_and_scalar_tails() {
    use sc_compress::DType as StorageDType;

    use crate::convert::ConvertOp;
    use crate::FloatCastPolicy;

    fn output(dtype: OutputDType) -> OutputSpec {
        let fill = match dtype {
            OutputDType::I16 => Fill::I16(0),
            OutputDType::I32 => Fill::I32(0),
            OutputDType::I64 => Fill::I64(0),
            OutputDType::U16 => Fill::U16(0),
            OutputDType::U32 => Fill::U32(0),
            OutputDType::U64 => Fill::U64(0),
            OutputDType::F32 => Fill::F32(0.0),
            OutputDType::F64 => Fill::F64(0.0),
        };
        OutputSpec::new(1, dtype, fill)
            .unwrap()
            .float_cast(FloatCastPolicy::AllowRounding)
            .overflow(OverflowPolicy::Unchecked)
            .unwrap()
    }

    fn check(src: StorageDType, dst: OutputDType, input: Vec<u8>, expected: Vec<u8>) {
        fn run(op: &ConvertOp, input: &[u8], expected: &[u8], tier: &str) {
            let mut actual = vec![0u8; expected.len()];
            op.convert_slice_prevalidated(input, &mut actual).unwrap();
            assert_eq!(actual, expected, "{tier} conversion mismatch");
        }

        let op = ConvertOp::resolve(src, &output(dst)).unwrap();
        let src_size = src.size();
        let dst_size = dst.size();
        let elements = input.len() / src_size;
        assert_eq!(expected.len(), elements * dst_size);
        for count in 0..=elements {
            let input = &input[..count * src_size];
            let expected = &expected[..count * dst_size];
            run(&op, input, expected, "runtime-selected");

            let mut scalar = op;
            scalar.force_scalar_for_test();
            run(&scalar, input, expected, "scalar");

            #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
            {
                let mut sse2 = op;
                sse2.force_sse2_for_test();
                run(&sse2, input, expected, "SSE2");
                if std::arch::is_x86_feature_detected!("avx2") {
                    let mut avx2 = op;
                    avx2.force_avx2_for_test();
                    run(&avx2, input, expected, "AVX2");
                }
                if std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                {
                    let mut avx512 = op;
                    avx512.force_avx512_for_test();
                    run(&avx512, input, expected, "AVX-512");
                }
            }
        }
    }

    let i16_values = [
        i16::MIN,
        -32_001,
        -1,
        0,
        1,
        255,
        256,
        i16::MAX,
        -17,
        17,
        -2_048,
        2_048,
        -9,
        9,
        -300,
        300,
        -12_345,
        12_345,
        7,
    ];
    let i16_input = i16_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    check(
        StorageDType::I16,
        OutputDType::I32,
        i16_input.clone(),
        i16_values
            .iter()
            .flat_map(|value| i32::from(*value).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::I16,
        OutputDType::U32,
        i16_input.clone(),
        i16_values
            .iter()
            .flat_map(|value| (*value as u32).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::I16,
        OutputDType::I64,
        i16_input.clone(),
        i16_values
            .iter()
            .flat_map(|value| i64::from(*value).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::I16,
        OutputDType::U64,
        i16_input.clone(),
        i16_values
            .iter()
            .flat_map(|value| (*value as u64).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::I16,
        OutputDType::F32,
        i16_input.clone(),
        i16_values
            .iter()
            .flat_map(|value| (*value as f32).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::I16,
        OutputDType::F64,
        i16_input,
        i16_values
            .iter()
            .flat_map(|value| f64::from(*value).to_le_bytes())
            .collect(),
    );

    let u16_values = [
        0u16,
        1,
        255,
        256,
        i16::MAX as u16,
        i16::MAX as u16 + 1,
        u16::MAX,
        17,
        2_048,
        9,
        300,
        12_345,
        50_000,
        42,
        7,
        31,
        63,
    ];
    let u16_input = u16_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    check(
        StorageDType::U16,
        OutputDType::I32,
        u16_input.clone(),
        u16_values
            .iter()
            .flat_map(|value| i32::from(*value).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::U16,
        OutputDType::U32,
        u16_input.clone(),
        u16_values
            .iter()
            .flat_map(|value| u32::from(*value).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::U16,
        OutputDType::I64,
        u16_input.clone(),
        u16_values
            .iter()
            .flat_map(|value| i64::from(*value).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::U16,
        OutputDType::U64,
        u16_input.clone(),
        u16_values
            .iter()
            .flat_map(|value| u64::from(*value).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::U16,
        OutputDType::F32,
        u16_input.clone(),
        u16_values
            .iter()
            .flat_map(|value| (*value as f32).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::U16,
        OutputDType::F64,
        u16_input,
        u16_values
            .iter()
            .flat_map(|value| f64::from(*value).to_le_bytes())
            .collect(),
    );

    let i32_values = [
        i32::MIN,
        -16_777_217,
        -16_777_216,
        -1,
        0,
        1,
        16_777_216,
        16_777_217,
        i32::MAX,
        -7,
        7,
    ];
    let i32_input = i32_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    check(
        StorageDType::I32,
        OutputDType::I64,
        i32_input.clone(),
        i32_values
            .iter()
            .flat_map(|value| i64::from(*value).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::I32,
        OutputDType::U64,
        i32_input.clone(),
        i32_values
            .iter()
            .flat_map(|value| (*value as u64).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::I32,
        OutputDType::F32,
        i32_input.clone(),
        i32_values
            .iter()
            .flat_map(|value| (*value as f32).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::I32,
        OutputDType::F64,
        i32_input,
        i32_values
            .iter()
            .flat_map(|value| f64::from(*value).to_le_bytes())
            .collect(),
    );

    let u32_values = [
        0u32,
        1,
        16_777_216,
        16_777_217,
        i32::MAX as u32,
        i32::MAX as u32 + 1,
        0x8000_0001,
        0xffff_ff00,
        u32::MAX,
        7,
        42,
    ];
    let u32_input = u32_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    check(
        StorageDType::U32,
        OutputDType::I64,
        u32_input.clone(),
        u32_values
            .iter()
            .flat_map(|value| i64::from(*value).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::U32,
        OutputDType::U64,
        u32_input.clone(),
        u32_values
            .iter()
            .flat_map(|value| u64::from(*value).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::U32,
        OutputDType::F32,
        u32_input.clone(),
        u32_values
            .iter()
            .flat_map(|value| (*value as f32).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::U32,
        OutputDType::F64,
        u32_input,
        u32_values
            .iter()
            .flat_map(|value| f64::from(*value).to_le_bytes())
            .collect(),
    );

    let i64_values = [
        i64::MIN,
        -(1i64 << 53) - 1,
        -1,
        0,
        1,
        (1i64 << 53) + 1,
        i64::MAX,
        7,
        42,
    ];
    let i64_input = i64_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    check(
        StorageDType::I64,
        OutputDType::U64,
        i64_input.clone(),
        i64_values
            .iter()
            .flat_map(|value| (*value as u64).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::I64,
        OutputDType::F64,
        i64_input,
        i64_values
            .iter()
            .flat_map(|value| (*value as f64).to_le_bytes())
            .collect(),
    );

    let u64_values = [
        0u64,
        1,
        (1u64 << 53) + 1,
        i64::MAX as u64,
        i64::MAX as u64 + 1,
        u64::MAX,
        7,
        42,
    ];
    let u64_input = u64_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    check(
        StorageDType::U64,
        OutputDType::I64,
        u64_input.clone(),
        u64_values
            .iter()
            .flat_map(|value| (*value as i64).to_le_bytes())
            .collect(),
    );
    check(
        StorageDType::U64,
        OutputDType::F64,
        u64_input,
        u64_values
            .iter()
            .flat_map(|value| (*value as f64).to_le_bytes())
            .collect(),
    );

    let f32_values = [
        f32::NEG_INFINITY,
        -f32::MAX,
        -1.5,
        -0.0,
        0.0,
        1.5,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::INFINITY,
    ];
    check(
        StorageDType::F32,
        OutputDType::F64,
        f32_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        f32_values
            .iter()
            .flat_map(|value| f64::from(*value).to_le_bytes())
            .collect(),
    );
}

#[test]
fn contiguous_target_gather_kernels_cover_conversion_and_tail() {
    use sc_compress::DType as StorageDType;

    use crate::convert::ConvertOp;

    fn check(src: StorageDType, input: Vec<u8>, expected: Vec<u8>) {
        const COUNT: usize = 35;
        let output = OutputSpec::new(COUNT + 2, OutputDType::F32, Fill::F32(0.0))
            .unwrap()
            .float_cast(crate::FloatCastPolicy::AllowRounding);
        let op = ConvertOp::resolve(src, &output).unwrap();
        let map = crate::plan::DenseMap::Gather32 {
            source_offsets: std::sync::Arc::from(
                (0..COUNT)
                    .map(|index| (index * 2 * src.size()) as i32)
                    .collect::<Vec<_>>(),
            ),
            target_byte: 4,
            covers_output: false,
        };
        let mut actual = vec![0xa5; (COUNT + 2) * 4];
        // SAFETY: every source offset selects one complete even-positioned
        // element and the target run covers exactly COUNT disjoint f32 values.
        unsafe {
            op.convert_map_prevalidated(input.as_ptr(), actual.as_mut_ptr(), &map)
                .unwrap();
        }
        assert_eq!(&actual[4..4 + COUNT * 4], expected);
        assert_eq!(&actual[..4], &[0xa5; 4]);
        assert_eq!(&actual[4 + COUNT * 4..], &[0xa5; 4]);
    }

    let i32_values = (0..70)
        .map(|value| value * 97 - 3_000)
        .collect::<Vec<i32>>();
    check(
        StorageDType::I32,
        i32_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        i32_values
            .iter()
            .step_by(2)
            .flat_map(|value| (*value as f32).to_le_bytes())
            .collect(),
    );

    let output = OutputSpec::new(37, OutputDType::F64, Fill::F64(0.0)).unwrap();
    let op = ConvertOp::resolve(StorageDType::I32, &output).unwrap();
    let map = crate::plan::DenseMap::Gather32 {
        source_offsets: std::sync::Arc::from(
            (0..35)
                .map(|index| (index * 2 * StorageDType::I32.size()) as i32)
                .collect::<Vec<_>>(),
        ),
        target_byte: 8,
        covers_output: false,
    };
    let input = i32_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let expected = i32_values
        .iter()
        .step_by(2)
        .flat_map(|value| f64::from(*value).to_le_bytes())
        .collect::<Vec<_>>();
    let mut actual = vec![0xa5; 37 * 8];
    // SAFETY: the map selects 35 complete i32 values and one contiguous,
    // disjoint f64 target run between two guard elements.
    unsafe {
        op.convert_map_prevalidated(input.as_ptr(), actual.as_mut_ptr(), &map)
            .unwrap();
    }
    assert_eq!(&actual[8..8 + 35 * 8], expected);
    assert_eq!(&actual[..8], &[0xa5; 8]);
    assert_eq!(&actual[8 + 35 * 8..], &[0xa5; 8]);

    let u32_values = (0..70)
        .map(|value| (value as u32).wrapping_mul(123_456_789))
        .collect::<Vec<_>>();
    check(
        StorageDType::U32,
        u32_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        u32_values
            .iter()
            .step_by(2)
            .flat_map(|value| (*value as f32).to_le_bytes())
            .collect(),
    );
}

#[test]
fn bulk_checked_sign_validation_covers_vectors_and_tails() {
    use sc_compress::DType as StorageDType;

    use crate::convert::ConvertOp;

    fn validator(src: StorageDType, dst: OutputDType) -> ConvertOp {
        let fill = match dst {
            OutputDType::I16 => Fill::I16(0),
            OutputDType::I32 => Fill::I32(0),
            OutputDType::I64 => Fill::I64(0),
            OutputDType::U16 => Fill::U16(0),
            OutputDType::U32 => Fill::U32(0),
            OutputDType::U64 => Fill::U64(0),
            _ => unreachable!(),
        };
        ConvertOp::resolve(src, &OutputSpec::new(1, dst, fill).unwrap()).unwrap()
    }

    let valid_i16 = (0..19i16).flat_map(i16::to_le_bytes).collect::<Vec<_>>();
    let mut invalid_i16_vector = valid_i16.clone();
    invalid_i16_vector[3 * 2..4 * 2].copy_from_slice(&(-1i16).to_le_bytes());
    let mut invalid_i16_tail = valid_i16.clone();
    invalid_i16_tail[18 * 2..19 * 2].copy_from_slice(&(-1i16).to_le_bytes());
    let i16_u32 = validator(StorageDType::I16, OutputDType::U32);
    assert!(i16_u32.validate_slice(&valid_i16).is_ok());
    assert!(i16_u32.validate_slice(&invalid_i16_vector).is_err());
    assert!(i16_u32.validate_slice(&invalid_i16_tail).is_err());
    let i16_u64 = validator(StorageDType::I16, OutputDType::U64);
    assert!(i16_u64.validate_slice(&valid_i16).is_ok());
    assert!(i16_u64.validate_slice(&invalid_i16_vector).is_err());
    assert!(i16_u64.validate_slice(&invalid_i16_tail).is_err());

    let valid_u16 = (0..19u16).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
    let mut invalid_u16 = valid_u16.clone();
    invalid_u16[16 * 2..17 * 2].copy_from_slice(&(i16::MAX as u16 + 1).to_le_bytes());
    let u16_i16 = validator(StorageDType::U16, OutputDType::I16);
    assert!(u16_i16.validate_slice(&valid_u16).is_ok());
    assert!(u16_i16.validate_slice(&invalid_u16).is_err());

    let valid_i32 = (0..11i32).flat_map(i32::to_le_bytes).collect::<Vec<_>>();
    let mut invalid_i32 = valid_i32.clone();
    invalid_i32[8 * 4..9 * 4].copy_from_slice(&(-1i32).to_le_bytes());
    let i32_u32 = validator(StorageDType::I32, OutputDType::U32);
    assert!(i32_u32.validate_slice(&valid_i32).is_ok());
    assert!(i32_u32.validate_slice(&invalid_i32).is_err());
    let i32_u64 = validator(StorageDType::I32, OutputDType::U64);
    assert!(i32_u64.validate_slice(&valid_i32).is_ok());
    assert!(i32_u64.validate_slice(&invalid_i32).is_err());

    let valid_u32 = (0..11u32).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    let mut invalid_u32 = valid_u32.clone();
    invalid_u32[2 * 4..3 * 4].copy_from_slice(&(i32::MAX as u32 + 1).to_le_bytes());
    let u32_i32 = validator(StorageDType::U32, OutputDType::I32);
    assert!(u32_i32.validate_slice(&valid_u32).is_ok());
    assert!(u32_i32.validate_slice(&invalid_u32).is_err());

    let valid_i64 = (0..9i64).flat_map(i64::to_le_bytes).collect::<Vec<_>>();
    let mut invalid_i64 = valid_i64.clone();
    invalid_i64[8 * 8..9 * 8].copy_from_slice(&(-1i64).to_le_bytes());
    let i64_u64 = validator(StorageDType::I64, OutputDType::U64);
    assert!(i64_u64.validate_slice(&valid_i64).is_ok());
    assert!(i64_u64.validate_slice(&invalid_i64).is_err());

    let valid_u64 = (0..9u64).flat_map(u64::to_le_bytes).collect::<Vec<_>>();
    let mut invalid_u64 = valid_u64.clone();
    invalid_u64[3 * 8..4 * 8].copy_from_slice(&(i64::MAX as u64 + 1).to_le_bytes());
    let u64_i64 = validator(StorageDType::U64, OutputDType::I64);
    assert!(u64_i64.validate_slice(&valid_u64).is_ok());
    assert!(u64_i64.validate_slice(&invalid_u64).is_err());
}

#[test]
fn csr_index_simd_validation_covers_unsigned_vectors_and_boundaries() {
    fn check(op: crate::scatter::IndexOp, values: &[u8], count: usize, n_cols: usize) -> bool {
        // SAFETY: each caller encodes exactly `count` complete indices of the
        // dtype bound into `op`.
        unsafe { op.validate(values.as_ptr(), count, n_cols) }
    }

    let u16_op = crate::scatter::IndexOp::new(sc_compress::DType::U16).unwrap();
    let valid_u16 = (0..67u16)
        .map(|value| value * 3)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    assert!(check(u16_op, &valid_u16, 67, 256));
    let mut duplicate_u16 = valid_u16.clone();
    duplicate_u16[32 * 2..33 * 2].copy_from_slice(&(31u16 * 3).to_le_bytes());
    assert!(!check(u16_op, &duplicate_u16, 67, 256));
    let mut out_of_bounds_u16 = valid_u16.clone();
    out_of_bounds_u16[64 * 2..65 * 2].copy_from_slice(&256u16.to_le_bytes());
    assert!(!check(u16_op, &out_of_bounds_u16, 65, 256));

    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if let Some(op) = crate::scatter::IndexOp::new_avx2_for_test(sc_compress::DType::U16) {
            assert!(check(op, &valid_u16, 67, 256));
            assert!(!check(op, &duplicate_u16, 67, 256));
            assert!(!check(op, &out_of_bounds_u16, 65, 256));
        }
        if let Some(op) = crate::scatter::IndexOp::new_avx512_for_test(sc_compress::DType::U16) {
            assert!(check(op, &valid_u16, 67, 256));
            assert!(!check(op, &duplicate_u16, 67, 256));
            assert!(!check(op, &out_of_bounds_u16, 65, 256));
        }
    }

    let step = u32::MAX / 16;
    let valid_u32 = (0..=16u32)
        .map(|value| value * step)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let u32_op = crate::scatter::IndexOp::new(sc_compress::DType::U32).unwrap();
    assert!(check(u32_op, &valid_u32, 17, u32::MAX as usize));
    let mut duplicate_u32 = valid_u32.clone();
    duplicate_u32[8 * 4..9 * 4].copy_from_slice(&(7 * step).to_le_bytes());
    assert!(!check(u32_op, &duplicate_u32, 17, u32::MAX as usize));
    let mut out_of_bounds_u32 = valid_u32.clone();
    out_of_bounds_u32[16 * 4..17 * 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(!check(u32_op, &out_of_bounds_u32, 17, u32::MAX as usize));

    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if let Some(op) = crate::scatter::IndexOp::new_avx2_for_test(sc_compress::DType::U32) {
            assert!(check(op, &valid_u32, 17, u32::MAX as usize));
            assert!(!check(op, &duplicate_u32, 17, u32::MAX as usize));
            assert!(!check(op, &out_of_bounds_u32, 17, u32::MAX as usize));
        }
        if let Some(op) = crate::scatter::IndexOp::new_avx512_for_test(sc_compress::DType::U32) {
            assert!(check(op, &valid_u32, 17, u32::MAX as usize));
            assert!(!check(op, &duplicate_u32, 17, u32::MAX as usize));
            assert!(!check(op, &out_of_bounds_u32, 17, u32::MAX as usize));
        }
    }
}

#[test]
fn unsigned_to_signed_overflow_can_use_sentinel() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("u2i");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[40_000u16, 10], [1, 2])
        .unwrap();
    let output = OutputSpec::new(2, OutputDType::I16, Fill::I16(-1))
        .unwrap()
        .overflow(OverflowPolicy::UseValue(Fill::I16(-99)))
        .unwrap();
    let plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        vec![RowRef::new(SourceId::new(0), 0)],
        output,
        1,
        2,
    ))
    .unwrap();
    let mut session = plan.open(blocking(1)).unwrap();
    let batch = session.next_batch().unwrap().unwrap();
    assert_eq!(batch.row_as::<i16>(0).unwrap(), &[-99, 10]);
}

#[derive(Default)]
struct MemoryStore {
    values: HashMap<String, Vec<u8>>,
    efficient_ranges: bool,
    reads: std::sync::atomic::AtomicUsize,
    read_into_calls: std::sync::atomic::AtomicUsize,
}

impl MemoryStore {
    fn from_directory(root: &std::path::Path, keys: &[&str]) -> Self {
        Self {
            values: keys
                .iter()
                .map(|key| ((*key).to_string(), std::fs::read(root.join(key)).unwrap()))
                .collect(),
            ..Self::default()
        }
    }

    fn with_efficient_ranges(mut self) -> Self {
        self.efficient_ranges = true;
        self
    }

    fn read_count(&self) -> usize {
        self.reads.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn reset_read_count(&self) {
        self.reads.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn read_into_count(&self) -> usize {
        self.read_into_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl ByteStore for MemoryStore {
    fn len(&self, key: &str) -> sc_compress::Result<u64> {
        self.values
            .get(key)
            .map(|value| value.len() as u64)
            .ok_or_else(|| sc_compress::Error::NotFound { key: key.into() })
    }

    fn read_range(&self, key: &str, offset: u64, len: usize) -> sc_compress::Result<Vec<u8>> {
        self.reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let value = self
            .values
            .get(key)
            .ok_or_else(|| sc_compress::Error::NotFound { key: key.into() })?;
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(value.len());
        let end = start.saturating_add(len).min(value.len());
        Ok(value[start..end].to_vec())
    }

    fn read_range_into(
        &self,
        key: &str,
        offset: u64,
        len: usize,
        output: &mut Vec<u8>,
    ) -> sc_compress::Result<()> {
        self.reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.read_into_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let value = self
            .values
            .get(key)
            .ok_or_else(|| sc_compress::Error::NotFound { key: key.into() })?;
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(value.len());
        let end = start.saturating_add(len).min(value.len());
        output.clear();
        output.extend_from_slice(&value[start..end]);
        Ok(())
    }

    fn exists(&self, key: &str) -> sc_compress::Result<bool> {
        Ok(self.values.contains_key(key))
    }

    fn supports_efficient_range_reads(&self, key: &str) -> sc_compress::Result<bool> {
        if !self.values.contains_key(key) {
            return Err(sc_compress::Error::NotFound { key: key.into() });
        }
        Ok(self.efficient_ranges)
    }
}

#[test]
fn key_backends_separate_range_reads_from_bounded_whole_key_reuse() {
    fn compile_memory(store: &Arc<MemoryStore>, retained_bytes: usize) -> crate::Plan {
        let erased: Arc<dyn ByteStore> = store.clone();
        let matrix = DenseMatrix::from_store(erased).unwrap();
        store.reset_read_count();
        let mut config = PlanConfig::default();
        config.limits.max_retained_whole_key_bytes = retained_bytes;
        compile(
            PlanSpec::new(
                vec![Source::new(0, Dataset::from_dense(matrix))],
                vec![RowRef::new(SourceId::new(0), 0)],
                OutputSpec::new(2, OutputDType::U16, Fill::U16(0)).unwrap(),
                1,
                2,
            )
            .config(config),
        )
        .unwrap()
    }

    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("key-source");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[11u16, 12], [1, 2])
        .unwrap();

    let cached_store = Arc::new(MemoryStore::from_directory(&path, &["meta.json", "data/0"]));
    let cached_plan = compile_memory(&cached_store, 1024 * 1024);
    let cached_compile_reads = cached_store.read_count();
    assert!(cached_compile_reads > 0);
    assert!(cached_plan.stats().retained_whole_key_bytes > 0);
    assert!(cached_plan.inner.sources.iter().any(|source| matches!(
        source,
        crate::plan::ReadSource::WholeKey {
            cached: Some(_),
            ..
        }
    )));
    let mut cached_session = cached_plan.open(blocking(1)).unwrap();
    assert_eq!(
        cached_session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<u16>(0)
            .unwrap(),
        &[11, 12]
    );
    assert_eq!(cached_store.read_count(), cached_compile_reads);

    let uncached_store = Arc::new(MemoryStore::from_directory(&path, &["meta.json", "data/0"]));
    let uncached_plan = compile_memory(&uncached_store, 0);
    let uncached_compile_reads = uncached_store.read_count();
    assert_eq!(uncached_plan.stats().retained_whole_key_bytes, 0);
    assert!(uncached_plan.inner.sources.iter().any(|source| matches!(
        source,
        crate::plan::ReadSource::WholeKey { cached: None, .. }
    )));
    let mut uncached_session = uncached_plan.open(blocking(1)).unwrap();
    assert_eq!(
        uncached_session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<u16>(0)
            .unwrap(),
        &[11, 12]
    );
    assert!(uncached_store.read_count() > uncached_compile_reads);

    let range_store = Arc::new(
        MemoryStore::from_directory(&path, &["meta.json", "data/0"]).with_efficient_ranges(),
    );
    let range_plan = compile_memory(&range_store, 1024 * 1024);
    let range_compile_reads = range_store.read_count();
    let range_compile_read_into = range_store.read_into_count();
    assert_eq!(range_plan.stats().retained_whole_key_bytes, 0);
    assert!(range_plan
        .inner
        .sources
        .iter()
        .any(|source| matches!(source, crate::plan::ReadSource::RangeKey { .. })));
    let mut range_session = range_plan.open(blocking(1)).unwrap();
    assert_eq!(
        range_session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<u16>(0)
            .unwrap(),
        &[11, 12]
    );
    assert!(range_store.read_count() > range_compile_reads);
    assert!(range_store.read_into_count() > range_compile_read_into);
}
