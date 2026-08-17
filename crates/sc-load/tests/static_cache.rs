mod common;

use sc_compress::{DenseWriter, Partition};
use sc_load::{
    compile, Dataset, Error, Fill, IoMergeOptions, IoMergePolicy, IoMode, OutputDType, OutputSpec,
    PlanConfig, PlanSpec, ResourceLimits, RowRef, SessionConfig, Source, SourceId,
};

use common::drain_rows;

fn dense_rows(values: &[u16], rows: usize) -> (tempfile::TempDir, Dataset) {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(values, [rows as u64, 1])
        .unwrap();
    let dataset = Dataset::open(&path).unwrap();
    (temporary, dataset)
}

#[test]
fn decoded_cache_reloads_static_extents_without_coupling_to_ring_slots() {
    let (_temporary, dataset) = dense_rows(&[10, 20, 30, 40], 4);
    let source_id = SourceId::new(0);
    let rows = [0, 1, 2, 0, 3, 0, 1]
        .into_iter()
        .map(|row| RowRef::new(source_id, row))
        .collect();
    let config = PlanConfig {
        cache_capacity_bytes: 128,
        ..PlanConfig::default()
    };
    let plan = compile(
        PlanSpec::new(
            vec![Source::new(source_id, dataset)],
            rows,
            OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap(),
            1,
            1,
        )
        .config(config),
    )
    .unwrap();

    assert_eq!(plan.prefetch_step(), 1);
    assert_eq!(plan.stats().output_ring_slots, 1);
    assert_eq!(plan.stats().cache_capacity_bytes, 128);
    assert!(plan.stats().cache_horizon_max_batches > plan.prefetch_step());
    assert!(plan.stats().residency_reloads > 0);
    assert!(plan.stats().cache_reference_hits > 0);
    assert_eq!(
        drain_rows::<u16>(&plan, 4).0,
        vec![
            vec![10],
            vec![20],
            vec![30],
            vec![10],
            vec![40],
            vec![10],
            vec![20]
        ]
    );
}

#[test]
fn one_batch_working_set_larger_than_cache_is_rejected_at_compile_time() {
    let (_temporary, dataset) = dense_rows(&[1, 2], 2);
    let source_id = SourceId::new(0);
    let config = PlanConfig {
        cache_capacity_bytes: 64,
        ..PlanConfig::default()
    };
    let result = compile(
        PlanSpec::new(
            vec![Source::new(source_id, dataset)],
            vec![RowRef::new(source_id, 0), RowRef::new(source_id, 1)],
            OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap(),
            2,
            1,
        )
        .config(config),
    );
    assert!(matches!(result, Err(Error::ResourceLimit(message)) if message.contains("batch 0")));
}

#[test]
fn io_merge_policies_preserve_output_and_publish_each_block_independently() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("merge-dense");
    DenseWriter::new(&path, Partition::fixed_cells(4), Partition::fixed_cells(1))
        .write(&(0..256u16).collect::<Vec<_>>(), [4, 64])
        .unwrap();
    let dataset = Dataset::open(&path).unwrap();
    let source_id = SourceId::new(0);
    let rows = (0..4)
        .map(|row| RowRef::new(source_id, row))
        .collect::<Vec<_>>();
    let compile_with = |io_merge| {
        compile(
            PlanSpec::new(
                vec![Source::new(source_id, dataset.clone())],
                rows.clone(),
                OutputSpec::new(64, OutputDType::U16, Fill::U16(0)).unwrap(),
                4,
                1,
            )
            .config(PlanConfig {
                io_merge,
                ..PlanConfig::default()
            }),
        )
        .unwrap()
    };
    let off = compile_with(IoMergeOptions {
        policy: IoMergePolicy::Off,
        ..IoMergeOptions::default()
    });
    let adjacent = compile_with(IoMergeOptions {
        initialize_parallelism_hint: 1,
        regular_io_parallelism_hint: 1,
        min_tasks_per_worker: 1,
        ..IoMergeOptions::default()
    });
    let cost = compile_with(IoMergeOptions {
        policy: IoMergePolicy::CostAware,
        max_io_gap_bytes: 1024,
        max_io_amplification_ratio: 2.0,
        io_bandwidth_bytes_per_second: 1_000_000_000.0,
        io_operations_per_second: 1.0,
        io_merge_delta_bytes: 0,
        initialize_parallelism_hint: 1,
        regular_io_parallelism_hint: 1,
        min_tasks_per_worker: 1,
        ..IoMergeOptions::default()
    });

    let expected = (0..4)
        .map(|row| {
            ((row * 64)..((row + 1) * 64))
                .map(|value| value as u16)
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    for plan in [&off, &adjacent, &cost] {
        assert_eq!(drain_rows::<u16>(plan, 2).0, expected);
        assert_eq!(plan.stats().independent_block_loads, 4);
        assert_eq!(plan.stats().dependency_edges, 4);
    }
    assert_eq!(off.stats().fused_io_tasks, 4);
    assert_eq!(adjacent.stats().fused_io_tasks, 1);
    assert_eq!(cost.stats().fused_io_tasks, 1);
    assert_eq!(adjacent.stats().predicted_io_ops_saved, 3);
}

#[test]
fn static_plan_resource_limits_bound_jobs_and_retained_arenas() {
    let (_temporary, dataset) = dense_rows(&[1, 2], 2);
    let source_id = SourceId::new(0);
    let output = || OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap();
    let rows = || vec![RowRef::new(source_id, 0), RowRef::new(source_id, 1)];
    let compile_with_limits = |limits| {
        compile(
            PlanSpec::new(
                vec![Source::new(source_id, dataset.clone())],
                rows(),
                output(),
                2,
                1,
            )
            .config(PlanConfig {
                cache_capacity_bytes: 128,
                limits,
                ..PlanConfig::default()
            }),
        )
    };

    let cells = compile_with_limits(ResourceLimits {
        max_cells_per_job: 1,
        ..ResourceLimits::default()
    });
    assert!(matches!(cells, Err(Error::ResourceLimit(message)) if message.contains("cells")));

    let blocks = compile_with_limits(ResourceLimits {
        max_blocks_per_job: 1,
        ..ResourceLimits::default()
    });
    assert!(matches!(blocks, Err(Error::ResourceLimit(message)) if message.contains("objects")));

    let decoded = compile_with_limits(ResourceLimits {
        max_decoded_bytes_per_job: 2,
        ..ResourceLimits::default()
    });
    assert!(
        matches!(decoded, Err(Error::ResourceLimit(message)) if message.contains("decoded bytes"))
    );

    let arena = compile_with_limits(ResourceLimits {
        max_compile_arena_bytes: 1,
        ..ResourceLimits::default()
    });
    assert!(
        matches!(arena, Err(Error::ResourceLimit(message)) if message.contains("compile arena"))
    );
}

#[test]
fn session_limits_cover_regular_staging() {
    let (_temporary, dataset) = dense_rows(&[1, 2], 2);
    let source_id = SourceId::new(0);
    let make_plan = |cache_capacity_bytes| {
        compile(
            PlanSpec::new(
                vec![Source::new(source_id, dataset.clone())],
                vec![RowRef::new(source_id, 0), RowRef::new(source_id, 1)],
                OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap(),
                1,
                1,
            )
            .config(PlanConfig {
                cache_capacity_bytes,
                ..PlanConfig::default()
            }),
        )
        .unwrap()
    };

    let staging_error = match make_plan(64).open(SessionConfig {
        worker_count: 1,
        initialize_workers: 1,
        initialize_inflight_io_ops: 1,
        initialize_inflight_encoded_bytes: 1,
        io_mode: IoMode::Blocking,
        max_inflight_encoded_bytes_per_worker: 1,
        max_total_inflight_encoded_bytes: 1,
        max_decoded_bytes_per_worker: 64,
        max_total_decoded_bytes: 64,
        ..SessionConfig::default()
    }) {
        Ok(_) => panic!("staging limit should reject the session"),
        Err(error) => error,
    };
    assert!(
        matches!(staging_error, Error::ResourceLimit(message) if message.contains("regular I/O task"))
    );
}

#[test]
fn session_limits_cover_decode_workspace_upper_bound() {
    let values = (0..64u16).collect::<Vec<_>>();
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("decode-workspace");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&values, [1, 64])
        .unwrap();
    let source_id = SourceId::new(0);
    let plan = compile(PlanSpec::new(
        vec![Source::new(source_id, Dataset::open(&path).unwrap())],
        vec![RowRef::new(source_id, 0)],
        OutputSpec::new(64, OutputDType::U16, Fill::U16(0)).unwrap(),
        1,
        1,
    ))
    .unwrap();
    let error = match plan.open(SessionConfig {
        worker_count: 1,
        initialize_workers: 1,
        initialize_inflight_io_ops: 1,
        io_mode: IoMode::Blocking,
        max_decoded_bytes_per_worker: 127,
        max_total_decoded_bytes: 127,
        ..SessionConfig::default()
    }) {
        Ok(_) => panic!("DecodeWorkspace limit should reject the session"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::ResourceLimit(message) if message.contains("workspace bytes")));
}

#[test]
fn static_cache_generations_preserve_randomized_access_order() {
    let values = (0..32u16).collect::<Vec<_>>();
    let (_temporary, dataset) = dense_rows(&values, values.len());
    let source_id = SourceId::new(0);
    let mut state = 0x9e37_79b9u32;
    let requested = (0..96)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            usize::try_from(state % values.len() as u32).unwrap()
        })
        .collect::<Vec<_>>();
    let expected = requested
        .iter()
        .map(|&row| vec![values[row]])
        .collect::<Vec<_>>();

    for batch_size in [1, 2, 4] {
        for ring_slots in [1, 3] {
            let rows = requested
                .iter()
                .map(|&row| RowRef::new(source_id, row as u64))
                .collect();
            let plan = compile(
                PlanSpec::new(
                    vec![Source::new(source_id, dataset.clone())],
                    rows,
                    OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap(),
                    batch_size,
                    ring_slots,
                )
                .config(PlanConfig {
                    cache_capacity_bytes: batch_size * 64,
                    ..PlanConfig::default()
                }),
            )
            .unwrap();
            for workers in [1, 4] {
                assert_eq!(drain_rows::<u16>(&plan, workers).0, expected);
            }
        }
    }
}

#[test]
fn parallel_chunk_preload_preserves_logical_row_order() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("parallel-preload");
    let values = (0..256u16).collect::<Vec<_>>();
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&values, [values.len() as u64, 1])
        .unwrap();
    let source_id = SourceId::new(0);
    let rows = (0..values.len())
        .rev()
        .map(|row| RowRef::new(source_id, row as u64))
        .collect();
    let plan = compile(
        PlanSpec::new(
            vec![Source::new(source_id, Dataset::open(&path).unwrap())],
            rows,
            OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap(),
            32,
            4,
        )
        .config(PlanConfig {
            compile_io_concurrency: 4,
            ..PlanConfig::default()
        }),
    )
    .unwrap();
    let expected = values
        .iter()
        .rev()
        .map(|&value| vec![value])
        .collect::<Vec<_>>();
    assert_eq!(drain_rows::<u16>(&plan, 4).0, expected);
}

#[test]
fn cost_aware_soft_merge_limit_allows_one_indivisible_block() {
    let values = (0..64u16).collect::<Vec<_>>();
    let (_temporary, dataset) = dense_rows(&values, 64);
    let source_id = SourceId::new(0);
    let plan = compile(
        PlanSpec::new(
            vec![Source::new(source_id, dataset)],
            vec![RowRef::new(source_id, 0)],
            OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap(),
            1,
            1,
        )
        .config(PlanConfig {
            io_merge: IoMergeOptions {
                policy: IoMergePolicy::CostAware,
                max_coalesced_io_bytes: 1,
                ..IoMergeOptions::default()
            },
            ..PlanConfig::default()
        }),
    )
    .unwrap();
    assert_eq!(plan.stats().independent_block_loads, 1);
    assert_eq!(plan.stats().fused_io_tasks, 1);
}

#[test]
fn cost_aware_partition_uses_the_exact_global_parallel_task_floor() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("cost-aware-floor");
    DenseWriter::new(&path, Partition::fixed_cells(4), Partition::fixed_cells(1))
        .write(&(0..256u16).collect::<Vec<_>>(), [4, 64])
        .unwrap();
    let source_id = SourceId::new(0);
    let rows = (0..4).map(|row| RowRef::new(source_id, row)).collect();
    let plan = compile(
        PlanSpec::new(
            vec![Source::new(source_id, Dataset::open(&path).unwrap())],
            rows,
            OutputSpec::new(64, OutputDType::U16, Fill::U16(0)).unwrap(),
            4,
            1,
        )
        .config(PlanConfig {
            io_merge: IoMergeOptions {
                policy: IoMergePolicy::CostAware,
                max_io_gap_bytes: 1024,
                max_io_amplification_ratio: 2.0,
                io_bandwidth_bytes_per_second: 1_000_000_000.0,
                io_operations_per_second: 1.0,
                io_merge_delta_bytes: 0,
                initialize_parallelism_hint: 1,
                min_tasks_per_worker: 2,
                ..IoMergeOptions::default()
            },
            ..PlanConfig::default()
        }),
    )
    .unwrap();
    assert_eq!(plan.stats().independent_block_loads, 4);
    assert_eq!(plan.stats().fused_io_tasks, 2);
}
