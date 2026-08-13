mod common;

use sc_compress::{CsrWriter, DenseWriter, Kind, Partition};
use sc_load::{
    compile, Dataset, FeatureMap, Fill, IoMode, OutputDType, OutputSpec, PlanSpec, RowRef, Source,
    SourceId, StorageDType,
};

use common::drain_rows;

#[test]
fn dense_plan_preserves_mapping_order_and_reusable_sessions() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense");
    let values = (0..18u16).collect::<Vec<_>>();
    DenseWriter::new(&path, Partition::fixed_cells(3), Partition::fixed_cells(1))
        .write(&values, [6, 3])
        .unwrap();

    let dataset = Dataset::open(&path).unwrap();
    assert_eq!(dataset.kind(), Kind::Dense);
    assert_eq!(dataset.shape(), [6, 3]);
    assert_eq!(dataset.n_rows(), 6);
    assert_eq!(dataset.n_cols(), 3);
    assert_eq!(dataset.dtype(), StorageDType::U16);

    let source_id = SourceId::new(7);
    let source = Source::new(source_id, dataset)
        .feature_map(FeatureMap::new([Some(2), None, Some(0)]).unwrap());
    let requested = [3u64, 0, 3, 5, 1, 4, 2, 5, 0];
    let rows = requested
        .into_iter()
        .map(|row| RowRef::new(source_id, row))
        .collect();
    let output = OutputSpec::new(4, OutputDType::U32, Fill::U32(9)).unwrap();
    let plan = compile(PlanSpec::new(vec![source], rows, output.clone(), 2, 2)).unwrap();

    assert_eq!(plan.batch_size(), 2);
    assert_eq!(plan.batch_count(), 5);
    assert_eq!(plan.prefetch_step(), 2);
    assert_eq!(plan.output_spec(), &output);
    assert_eq!(plan.row_stride_bytes(), 64);
    assert_eq!(plan.stats().input_rows, requested.len());
    assert!(plan.stats().jobs > 0);

    let expected = requested
        .into_iter()
        .map(|row| {
            let base = u32::try_from(row * 3).unwrap();
            vec![base + 2, 9, base, 9]
        })
        .collect::<Vec<_>>();
    let (first, first_stats) = drain_rows::<u32>(&plan, 2);
    let (second, second_stats) = drain_rows::<u32>(&plan, 1);

    assert_eq!(first, expected);
    assert_eq!(second, expected);
    assert_eq!(first_stats.requested_io_mode, IoMode::Blocking);
    assert_eq!(first_stats.actual_io_mode, IoMode::Blocking);
    assert_eq!(first_stats.worker_count, 2);
    assert_eq!(second_stats.worker_count, 1);
}

#[test]
fn csr_plan_scatter_handles_empty_rows_mapping_and_promotion() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("csr");
    CsrWriter::new(&path, Partition::fixed_cells(4), Partition::fixed_cells(2))
        .write(
            &[0u64, 2, 2, 4, 5],
            &[0u32, 3, 1, 4, 2],
            &[1i16, 2, 3, 4, -5],
            [4, 5],
        )
        .unwrap();

    let dataset = Dataset::open(&path).unwrap();
    assert_eq!(dataset.kind(), Kind::Csr);
    assert_eq!(dataset.shape(), [4, 5]);
    assert_eq!(dataset.dtype(), StorageDType::I16);

    let source_id = SourceId::new(11);
    let source = Source::new(source_id, dataset)
        .feature_map(FeatureMap::new([Some(4), Some(1), None, Some(0), Some(3)]).unwrap());
    let rows = [1u64, 2, 0, 3]
        .into_iter()
        .map(|row| RowRef::new(source_id, row))
        .collect();
    let output = OutputSpec::new(6, OutputDType::I32, Fill::I32(-9)).unwrap();
    let plan = compile(PlanSpec::new(vec![source], rows, output, 3, 2)).unwrap();

    assert_eq!(plan.batch_count(), 2);
    assert!(plan.stats().data_io_ops > 0);
    assert!(plan.stats().indices_io_ops > 0);

    let (observed, stats) = drain_rows::<i32>(&plan, 2);
    assert_eq!(
        observed,
        vec![
            vec![-9, -9, -9, -9, -9, -9],
            vec![-9, 3, -9, 4, -9, -9],
            vec![2, -9, -9, -9, 1, -9],
            vec![-9, -9, -9, -9, -9, -9],
        ]
    );
    assert_eq!(stats.actual_io_mode, IoMode::Blocking);
}

#[test]
fn compiled_plan_pins_generation_while_new_plans_see_replacement() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("generation");
    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[1u16, 2, 3, 4], [2, 2])
        .unwrap();

    let rows = vec![
        RowRef::new(SourceId::new(0), 0),
        RowRef::new(SourceId::new(0), 1),
    ];
    let output = OutputSpec::new(2, OutputDType::U16, Fill::U16(0)).unwrap();
    let original_plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        rows.clone(),
        output.clone(),
        1,
        2,
    ))
    .unwrap();

    DenseWriter::new(&path, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .write(&[11u16, 12, 13, 14], [2, 2])
        .unwrap();

    let replacement_plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        rows,
        output,
        1,
        2,
    ))
    .unwrap();

    assert_eq!(
        drain_rows::<u16>(&original_plan, 1).0,
        vec![vec![1, 2], vec![3, 4]]
    );
    assert_eq!(
        drain_rows::<u16>(&original_plan, 2).0,
        vec![vec![1, 2], vec![3, 4]]
    );
    assert_eq!(
        drain_rows::<u16>(&replacement_plan, 1).0,
        vec![vec![11, 12], vec![13, 14]]
    );
}

#[test]
fn int64_and_uint64_storage_preserve_full_precision() {
    let temporary = tempfile::tempdir().unwrap();

    let dense_path = temporary.path().join("dense-i64");
    let dense_values = [i64::MIN + 1, -(1i64 << 53) - 1, (1i64 << 53) + 1, i64::MAX];
    DenseWriter::new(
        &dense_path,
        Partition::fixed_cells(1),
        Partition::fixed_cells(1),
    )
    .write(&dense_values, [2, 2])
    .unwrap();
    let dense_dataset = Dataset::open(&dense_path).unwrap();
    assert_eq!(dense_dataset.dtype(), StorageDType::I64);
    let dense_source = SourceId::new(0);
    let dense_plan = compile(PlanSpec::new(
        vec![Source::new(dense_source, dense_dataset)],
        vec![RowRef::new(dense_source, 0), RowRef::new(dense_source, 1)],
        OutputSpec::new(2, OutputDType::I64, Fill::I64(0)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    assert_eq!(
        drain_rows::<i64>(&dense_plan, 1).0,
        dense_values
            .chunks_exact(2)
            .map(<[i64]>::to_vec)
            .collect::<Vec<_>>()
    );

    let csr_path = temporary.path().join("csr-u64");
    let csr_values = [0u64, (1u64 << 53) + 1, (1u64 << 63) + 1, u64::MAX];
    CsrWriter::new(
        &csr_path,
        Partition::fixed_cells(1),
        Partition::fixed_cells(1),
    )
    .write(&[0u64, 2, 4], &[0u32, 1, 0, 1], &csr_values, [2, 2])
    .unwrap();
    let csr_dataset = Dataset::open(&csr_path).unwrap();
    assert_eq!(csr_dataset.dtype(), StorageDType::U64);
    let csr_source = SourceId::new(0);
    let csr_plan = compile(PlanSpec::new(
        vec![Source::new(csr_source, csr_dataset)],
        vec![RowRef::new(csr_source, 0), RowRef::new(csr_source, 1)],
        OutputSpec::new(2, OutputDType::U64, Fill::U64(0)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    assert_eq!(
        drain_rows::<u64>(&csr_plan, 1).0,
        csr_values
            .chunks_exact(2)
            .map(<[u64]>::to_vec)
            .collect::<Vec<_>>()
    );
}

#[test]
fn widening_to_64_bit_outputs_handles_dense_maps_and_csr_scatter() {
    let temporary = tempfile::tempdir().unwrap();

    let dense_path = temporary.path().join("dense-i32-map");
    let dense_values = [i32::MIN, -1, i32::MAX, 7, -9, 1 << 30];
    DenseWriter::new(
        &dense_path,
        Partition::fixed_cells(1),
        Partition::fixed_cells(1),
    )
    .write(&dense_values, [2, 3])
    .unwrap();
    let dense_source_id = SourceId::new(21);
    let dense_source = Source::new(dense_source_id, Dataset::open(&dense_path).unwrap())
        .feature_map(FeatureMap::new([Some(2), Some(0), Some(3)]).unwrap());
    let dense_plan = compile(PlanSpec::new(
        vec![dense_source],
        vec![
            RowRef::new(dense_source_id, 1),
            RowRef::new(dense_source_id, 0),
        ],
        OutputSpec::new(5, OutputDType::I64, Fill::I64(-17)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    assert_eq!(
        drain_rows::<i64>(&dense_plan, 1).0,
        vec![
            vec![-9, -17, 7, 1 << 30, -17],
            vec![-1, -17, i64::from(i32::MIN), i64::from(i32::MAX), -17],
        ]
    );

    let csr_path = temporary.path().join("csr-u32-map");
    let csr_values = [0u32, 3, u32::MAX, (1u32 << 31) + 1, 17];
    CsrWriter::new(
        &csr_path,
        Partition::fixed_cells(1),
        Partition::fixed_cells(1),
    )
    .write(&[0u64, 3, 5], &[0u32, 2, 3, 1, 3], &csr_values, [2, 4])
    .unwrap();
    let csr_source_id = SourceId::new(22);
    let csr_source = Source::new(csr_source_id, Dataset::open(&csr_path).unwrap())
        .feature_map(FeatureMap::new([Some(3), Some(1), None, Some(0)]).unwrap());
    let csr_plan = compile(PlanSpec::new(
        vec![csr_source],
        vec![RowRef::new(csr_source_id, 0), RowRef::new(csr_source_id, 1)],
        OutputSpec::new(5, OutputDType::U64, Fill::U64(99)).unwrap(),
        1,
        2,
    ))
    .unwrap();
    assert_eq!(
        drain_rows::<u64>(&csr_plan, 1).0,
        vec![
            vec![u64::from(u32::MAX), 99, 99, 0, 99],
            vec![17, u64::from((1u32 << 31) + 1), 99, 99, 99],
        ]
    );
}
