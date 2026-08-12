use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use sc_compress::{
    open_csr, open_csr_with_limits, open_dense, open_dense_with_limits, AxisIndex, ByteStore,
    CsrMatrix, CsrOutput, CsrWriter, DenseMatrix, DenseWriter, Partition, ReadLimits, Selection,
    StoreLocation, ZipStore,
};
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

#[test]
fn dense_typed_roundtrip_and_range() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("dense");
    let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    DenseWriter::new(&root, Partition::fixed_cells(2), Partition::fixed_cells(16))
        .threads(4)
        .write(&values, [6, 4])
        .unwrap();

    let matrix = open_dense_with_limits(&root, ReadLimits::default().threads(4)).unwrap();
    assert_eq!(matrix.shape(), [6, 4]);
    assert_eq!(matrix.decode_all().unwrap(), f32_bytes(&values));
    assert_eq!(matrix.decode_rows(2..5).unwrap(), f32_bytes(&values[8..20]));
}

#[test]
fn csr_typed_roundtrip_canonicalizes_rows() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("csr");
    CsrWriter::new(&root, Partition::fixed_cells(1), Partition::fixed_cells(1))
        .threads(4)
        .write(&[0u64, 2, 3], &[2u32, 0, 1], &[20i32, 0, 10], [2, 3])
        .unwrap();

    let matrix = open_csr_with_limits(&root, ReadLimits::default().threads(4)).unwrap();
    let (indices, data) = matrix.decode_all().unwrap();
    assert_eq!(u16_values(&indices), vec![0, 2, 1]);
    assert_eq!(i32_values(&data), vec![0, 20, 10]);
    let (indices, data) = matrix.decode_rows(1..2).unwrap();
    assert_eq!(u16_values(&indices), vec![1]);
    assert_eq!(i32_values(&data), vec![10]);
}

#[test]
fn store_selection_preserves_fancy_order_and_duplicates() {
    let temp = tempfile::tempdir().unwrap();
    let dense_root = temp.path().join("dense-select");
    let dense_values = (0u16..16).collect::<Vec<_>>();
    DenseWriter::new(&dense_root, Partition::fixed_cells(2), Partition::fixed_cells(16))
        .write(&dense_values, [8, 2])
        .unwrap();
    let dense = open_dense(&dense_root).unwrap();
    let selected = dense
        .select(Selection::new(
            AxisIndex::positions([7, 0, 7]),
            AxisIndex::positions([1, 0]),
        ))
        .unwrap();
    assert_eq!(selected.shape(), [3, 2]);
    assert_eq!(u16_values(selected.values()), vec![15, 14, 1, 0, 15, 14]);
    let strided = dense
        .select(Selection::new(
            AxisIndex::strided(6, -1, -2),
            AxisIndex::range(0, 2),
        ))
        .unwrap();
    assert_eq!(u16_values(strided.values()), vec![12, 13, 8, 9, 4, 5, 0, 1]);

    let csr_root = temp.path().join("csr-select");
    let indptr = (0u64..=8).collect::<Vec<_>>();
    let indices = (0u32..8).map(|row| row % 3).collect::<Vec<_>>();
    let values = (10u16..18).collect::<Vec<_>>();
    CsrWriter::new(&csr_root, Partition::fixed_cells(2), Partition::fixed_cells(16))
        .write(&indptr, &indices, &values, [8, 3])
        .unwrap();
    let csr = open_csr(&csr_root).unwrap();
    let selected = csr
        .select(
            Selection::new(
                AxisIndex::positions([7, 0, 7]),
                AxisIndex::positions([1, 0, 1]),
            ),
            CsrOutput::Sparse,
        )
        .unwrap();
    let sc_compress::SelectedArray::Csr(selected) = selected else {
        panic!("expected CSR selection");
    };
    assert_eq!(selected.shape(), [3, 3]);
    assert_eq!(selected.indptr(), &[0, 2, 3, 5]);
    assert_eq!(u16_values(selected.indices()), vec![0, 2, 1, 0, 2]);
    assert_eq!(u16_values(selected.data()), vec![17, 17, 10, 17, 17]);

    let strided = csr
        .select(
            Selection::rows_only(AxisIndex::strided(6, -1, -2)),
            CsrOutput::Sparse,
        )
        .unwrap();
    let sc_compress::SelectedArray::Csr(strided) = strided else {
        panic!("expected CSR selection");
    };
    assert_eq!(strided.indptr(), &[0, 1, 2, 3, 4]);
    assert_eq!(u16_values(strided.indices()), vec![0, 1, 2, 0]);
    assert_eq!(u16_values(strided.data()), vec![16, 14, 12, 10]);
}

#[test]
fn store_selection_supports_range_stride_mask_and_empty_axes() {
    let temp = tempfile::tempdir().unwrap();
    let dense_root = temp.path().join("dense-axis-forms");
    let dense_values = (0u16..16).collect::<Vec<_>>();
    DenseWriter::new(&dense_root, Partition::fixed_cells(4), Partition::fixed_cells(1))
        .write(&dense_values, [4, 4])
        .unwrap();
    let dense = open_dense(&dense_root).unwrap();
    let selected = dense
        .select(Selection::new(
            AxisIndex::from_mask(&[true, false, true, false]),
            AxisIndex::strided(3, -1, -2),
        ))
        .unwrap();
    assert_eq!(selected.shape(), [2, 2]);
    assert_eq!(u16_values(selected.values()), vec![3, 1, 11, 9]);
    let empty = dense
        .select(Selection::new(
            AxisIndex::positions([]),
            AxisIndex::range(1, 3),
        ))
        .unwrap();
    assert_eq!(empty.shape(), [0, 2]);

    let csr_root = temp.path().join("csr-axis-forms");
    CsrWriter::new(&csr_root, Partition::fixed_cells(4), Partition::fixed_cells(1))
        .write(
            &[0u64, 1, 2, 3, 4],
            &[0u32, 1, 2, 3],
            &[10u16, 11, 12, 13],
            [4, 4],
        )
        .unwrap();
    let csr = open_csr(&csr_root).unwrap();
    let selected = csr
        .select(
            Selection::new(
                AxisIndex::from_mask(&[true, false, true, false]),
                AxisIndex::strided(3, -1, -1),
            ),
            CsrOutput::Sparse,
        )
        .unwrap();
    let sc_compress::SelectedArray::Csr(selected) = selected else {
        panic!("expected CSR selection");
    };
    assert_eq!(selected.shape(), [2, 4]);
    assert_eq!(selected.indptr(), &[0, 1, 2]);
    assert_eq!(u16_values(selected.indices()), vec![3, 1]);
    assert_eq!(u16_values(selected.data()), vec![10, 12]);
}

#[test]
fn store_csr_2d_selection_matches_in_memory_kernel_across_axis_forms() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("csr-selection-matrix");
    let shape = [8u64, 9u64];
    let indptr = vec![0u64, 3, 3, 5, 6, 9, 10, 12, 15];
    let indices = vec![0u32, 3, 8, 1, 7, 4, 0, 2, 8, 6, 1, 5, 0, 4, 8];
    let data = (0..indices.len())
        .map(|value| u16::try_from(value * 7 + 3).unwrap())
        .collect::<Vec<_>>();
    CsrWriter::new(&root, Partition::fixed_cells(2), Partition::fixed_cells(1))
        .write(&indptr, &indices, &data, shape)
        .unwrap();
    let store = open_csr(&root).unwrap();
    let in_memory = sc_compress::CsrArray::from_parts(
        [shape[0] as usize, shape[1] as usize],
        sc_compress::DType::U32,
        sc_compress::DType::U16,
        indptr,
        indices
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        data.iter().flat_map(|value| value.to_le_bytes()).collect(),
    )
    .unwrap();

    let row_axes = [
        AxisIndex::range(0, 8),
        AxisIndex::range(2, 7),
        AxisIndex::positions([7, 0, 7, 3]),
        AxisIndex::strided(7, -1, -2),
        AxisIndex::from_mask(&[true, false, true, false, false, true, false, true]),
    ];
    let column_axes = [
        AxisIndex::range(0, 1),
        AxisIndex::range(2, 7),
        AxisIndex::positions([8, 0, 8, 3]),
        AxisIndex::strided(8, -1, -3),
        AxisIndex::positions([]),
    ];
    for rows in &row_axes {
        for cols in &column_axes {
            for output in [CsrOutput::Sparse, CsrOutput::Dense] {
                let selection = Selection::new(rows.clone(), cols.clone());
                let expected = in_memory.select(selection.clone(), output, 3).unwrap();
                let actual = store.select(selection, output).unwrap();
                assert_eq!(actual, expected, "rows={rows:?}, cols={cols:?}, {output:?}");
            }
        }
    }
}

#[test]
fn csr_sorted_rows_stay_aligned_and_duplicates_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("sorted");
    CsrWriter::new(&root, Partition::fixed_cells(1024), Partition::fixed_cells(16))
        .write(&[0u64, 2], &[0u32, 2], &[4i32, 5], [1, 3])
        .unwrap();

    let (indices, data) = open_csr(&root).unwrap().decode_all().unwrap();
    assert_eq!(u16_values(&indices), vec![0, 2]);
    assert_eq!(i32_values(&data), vec![4, 5]);

    assert!(CsrWriter::new(temp.path().join("duplicates"), Partition::fixed_cells(1024), Partition::fixed_cells(16))
        .write(&[0u64, 2], &[1u32, 1], &[4i32, 5], [1, 3])
        .is_err());
}

#[test]
fn opened_directory_matrices_support_concurrent_decodes() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DenseMatrix>();
    assert_send_sync::<CsrMatrix>();
    assert_send_sync::<ZipStore>();

    let temp = tempfile::tempdir().unwrap();
    let dense_root = temp.path().join("dense-concurrent");
    let rows = 32usize;
    let cols = 8usize;
    let dense_values = (0..rows * cols)
        .map(|value| u16::try_from(value).unwrap())
        .collect::<Vec<_>>();
    DenseWriter::new(&dense_root, Partition::fixed_cells(4), Partition::fixed_cells(16))
        .threads(4)
        .write(&dense_values, [rows as u64, cols as u64])
        .unwrap();
    let dense =
        Arc::new(open_dense_with_limits(&dense_root, ReadLimits::default().threads(2)).unwrap());
    let expected_dense = u16_bytes(&dense_values[4 * cols..20 * cols]);

    let csr_root = temp.path().join("csr-concurrent");
    let indptr = (0..=rows as u64).collect::<Vec<_>>();
    let csr_indices = (0..rows as u32).collect::<Vec<_>>();
    let csr_values = (0..rows as u16)
        .map(|value| value + 100)
        .collect::<Vec<_>>();
    CsrWriter::new(&csr_root, Partition::fixed_cells(4), Partition::fixed_cells(16))
        .threads(4)
        .write(
            &indptr,
            &csr_indices,
            &csr_values,
            [rows as u64, rows as u64],
        )
        .unwrap();
    let csr = Arc::new(open_csr_with_limits(&csr_root, ReadLimits::default().threads(2)).unwrap());
    let expected_indices = u16_bytes(&(4u16..20).collect::<Vec<_>>());
    let expected_values = u16_bytes(&csr_values[4..20]);

    std::thread::scope(|scope| {
        for _ in 0..8 {
            let dense = Arc::clone(&dense);
            let csr = Arc::clone(&csr);
            let expected_dense = &expected_dense;
            let expected_indices = &expected_indices;
            let expected_values = &expected_values;
            scope.spawn(move || {
                for _ in 0..4 {
                    assert_eq!(dense.decode_rows(4..20).unwrap(), *expected_dense);
                    let (indices, values) = csr.decode_rows(4..20).unwrap();
                    assert_eq!(indices, *expected_indices);
                    assert_eq!(values, *expected_values);
                }
            });
        }
    });
}

#[test]
fn dense_reads_from_stored_and_deflated_zip_entries() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("dense");
    let values = vec![1u16, 2, 3, 4];
    DenseWriter::new(&root, Partition::fixed_cells(1024), Partition::fixed_cells(16)).write(&values, [2, 2]).unwrap();

    for method in [
        zip::CompressionMethod::Stored,
        zip::CompressionMethod::Deflated,
    ] {
        let zip_path = temp.path().join(format!("{method:?}.zip"));
        zip_dense(&root, &zip_path, "assay", method);
        let location = StoreLocation::zip(&zip_path, "assay");
        let store = location.open().unwrap();
        assert_eq!(
            store.supports_efficient_range_reads("data/0").unwrap(),
            method == zip::CompressionMethod::Stored
        );
        let encoded = std::fs::read(root.join("data/0")).unwrap();
        assert_eq!(store.read_range("data/0", 3, 7).unwrap(), encoded[3..10]);
        assert!(store
            .read_range("data/0", u64::try_from(encoded.len()).unwrap(), 0)
            .unwrap()
            .is_empty());
        assert_eq!(
            open_dense(location).unwrap().decode_all().unwrap(),
            u16_bytes(&values)
        );
    }
}

#[test]
fn zip_store_supports_concurrent_deflated_reads() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("dense-concurrent");
    let values = (0..8_192u32).collect::<Vec<_>>();
    DenseWriter::new(&root, Partition::fixed_cells(1024), Partition::fixed_cells(16))
        .threads(4)
        .write(&values, [128, 64])
        .unwrap();
    let zip_path = temp.path().join("concurrent-deflated.zip");
    zip_dense(&root, &zip_path, "assay", zip::CompressionMethod::Deflated);

    let expected = std::fs::read(root.join("data/0")).unwrap();
    let store = Arc::new(ZipStore::open(&zip_path, "assay").unwrap());
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let store = Arc::clone(&store);
            let expected = &expected;
            scope.spawn(move || {
                for _ in 0..8 {
                    assert_eq!(
                        store.read("data/0").unwrap().as_slice(),
                        expected.as_slice()
                    );
                }
            });
        }
    });
}

#[test]
fn csr_reads_from_deflated_zip_entries() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("csr");
    CsrWriter::new(&root, Partition::fixed_cells(1024), Partition::fixed_cells(16))
        .write(&[0u64, 1, 2], &[0u32, 1], &[1f32, 2.0], [2, 2])
        .unwrap();
    let zip_path = temp.path().join("csr-deflated.zip");
    zip_keys(
        &root,
        &zip_path,
        "assay",
        &["meta.json", "indptr", "indices/0", "data/0"],
        zip::CompressionMethod::Deflated,
    );
    let location = StoreLocation::zip(&zip_path, "assay");
    assert!(!location
        .open()
        .unwrap()
        .supports_efficient_range_reads("indices/0")
        .unwrap());
    let (indices, data) = open_csr(location).unwrap().decode_all().unwrap();
    assert_eq!(u16_values(&indices), vec![0, 1]);
    assert_eq!(f32_values(&data), vec![1.0, 2.0]);
}

#[test]
fn zip_full_reads_validate_entry_checksum() {
    let temp = tempfile::tempdir().unwrap();
    let zip_path = temp.path().join("corrupt.zip");
    let payload = b"\xde\xad\xbe\xef unique zip payload \x01\x02\x03";
    let file = File::create(&zip_path).unwrap();
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file("assay/value", options).unwrap();
    zip.write_all(payload).unwrap();
    zip.finish().unwrap();

    let mut archive = std::fs::read(&zip_path).unwrap();
    let matches = archive
        .windows(payload.len())
        .enumerate()
        .filter_map(|(offset, bytes)| (bytes == payload).then_some(offset))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1);
    archive[matches[0]] ^= 0xff;
    std::fs::write(&zip_path, archive).unwrap();

    let store = ZipStore::open(&zip_path, "assay").unwrap();
    assert!(store.read("value").is_err());
    assert!(store.read_range("value", 0, payload.len()).is_err());
}

fn zip_dense(
    root: &std::path::Path,
    archive: &std::path::Path,
    prefix: &str,
    method: zip::CompressionMethod,
) {
    zip_keys(root, archive, prefix, &["meta.json", "data/0"], method);
}

fn zip_keys(
    root: &std::path::Path,
    archive: &std::path::Path,
    prefix: &str,
    keys: &[&str],
    method: zip::CompressionMethod,
) {
    let file = File::create(archive).unwrap();
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(method);
    for &key in keys {
        zip.start_file(format!("{prefix}/{key}"), options).unwrap();
        zip.write_all(&std::fs::read(root.join(key)).unwrap())
            .unwrap();
    }
    zip.finish().unwrap();
}

#[test]
fn dense_and_csr_support_2d_on_demand_select() {
    let temp = tempfile::tempdir().unwrap();

    // Dense: 6×4 matrix, select fancy rows + column strip.
    let dense_root = temp.path().join("dense-select");
    let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
    DenseWriter::new(&dense_root, Partition::fixed_cells(2), Partition::fixed_cells(16))
        .threads(4)
        .write(&values, [6, 4])
        .unwrap();
    let dense = open_dense_with_limits(&dense_root, ReadLimits::default().threads(4)).unwrap();
    let selected = dense
        .select(Selection::new(
            AxisIndex::positions([5, 1, 5]),
            AxisIndex::range(1, 3),
        ))
        .unwrap();
    assert_eq!(selected.shape(), [3, 2]);
    // row5: 20,21,22,23 → cols 1,2 → 21,22
    // row1: 4,5,6,7     → cols 1,2 → 5,6
    // row5 again
    assert_eq!(
        f32_values(selected.values()),
        vec![21.0, 22.0, 5.0, 6.0, 21.0, 22.0]
    );

    // CSR: densify a gene subset for a mini-batch of cells.
    let csr_root = temp.path().join("csr-select");
    CsrWriter::new(&csr_root, Partition::fixed_cells(1), Partition::fixed_cells(16))
        .threads(4)
        .write(
            &[0u64, 2, 3, 5],
            &[0u32, 2, 1, 0, 3],
            &[1.0f32, 3.0, 2.0, 4.0, 5.0],
            [3, 4],
        )
        .unwrap();
    let csr = open_csr_with_limits(&csr_root, ReadLimits::default().threads(4)).unwrap();
    let batch = csr
        .select(
            Selection::new(AxisIndex::positions([2, 0]), AxisIndex::positions([3, 0])),
            CsrOutput::Dense,
        )
        .unwrap();
    let dense_batch = batch.into_dense().unwrap();
    assert_eq!(dense_batch.shape(), [2, 2]);
    // src row2: (0,4)(3,5) → cols 3,0 → 5,4
    // src row0: (0,1)(2,3) → cols 3,0 → 0,1
    assert_eq!(f32_values(dense_batch.values()), vec![5.0, 4.0, 0.0, 1.0]);
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u16_values(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect()
}

fn i32_values(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}
