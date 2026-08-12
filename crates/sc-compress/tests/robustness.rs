use std::os::unix::fs::symlink;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sc_compress::{
    open_csr, open_dense, open_dense_with_limits, AxisIndex, BloscOptions, ByteStore, ByteStoreMut,
    Compressor, CsrMatrix, CsrWriter, DenseMatrix, DenseWriter, DirectoryStore, Error, Partition,
    ReadLimits, Result, Selection, ShuffleMode,
};
use serde_json::{json, Value};

fn write_json(store: &mut DirectoryStore, value: &Value) {
    store
        .write("meta.json", &serde_json::to_vec_pretty(value).unwrap())
        .unwrap();
}

fn dyn_blosc_meta(path: &str, dtype: &str) -> Value {
    json!({
        "path": path,
        "dtype": dtype,
        "compressor": {
            "id": "dyn-blosc",
            "codec": "lz4",
            "clevel": 5,
            "shuffle": "bytes",
            "split_blocks": false
        }
    })
}

fn csr_meta(shape: [u64; 2], nnz: u64) -> Value {
    json!({
        "format": "sc-compress",
        "version": 1,
        "kind": "csr",
        "shape": shape,
        "nnz": nnz,
        "partition": {
            "chunk": {"strategy": "fixed_cells", "n": 2},
            "block": {"strategy": "fixed_cells", "n": 1}
        },
        "indptr": {
            "path": "indptr",
            "dtype": "u64",
            "compressor": {"id": "none"}
        },
        "indices": dyn_blosc_meta("indices", "u16"),
        "data": dyn_blosc_meta("data", "f32"),
        "chunks": {"offsets": [0]}
    })
}

fn dense_meta(version: u32, offsets: Vec<u64>) -> Value {
    json!({
        "format": "sc-compress",
        "version": version,
        "kind": "dense",
        "shape": [1, 1],
        "partition": {
            "chunk": {"strategy": "fixed_cells", "n": 1},
            "block": {"strategy": "fixed_cells", "n": 1}
        },
        "data": {
            "path": "data",
            "dtype": "u16",
            "compressor": {
                "id": "blosc1",
                "codec": "lz4",
                "clevel": 5,
                "shuffle": "bytes",
                "split_blocks": false,
                "block_size": 2
            }
        },
        "chunks": {"offsets": offsets}
    })
}

#[test]
fn opened_reader_remains_on_one_generation() {
    let temp = tempfile::tempdir().unwrap();
    let matrix_dir = temp.path().join("matrix");
    let original = vec![1u16, 2, 3, 4];
    let replacement = vec![11u16, 12, 13, 14];
    DenseWriter::new(&matrix_dir, Partition::fixed_cells(1), Partition::fixed_cells(16))
        .write(&original, [2, 2])
        .unwrap();

    let opened = open_dense(&matrix_dir).unwrap();
    DenseWriter::new(&matrix_dir, Partition::fixed_cells(1), Partition::fixed_cells(16))
        .write(&replacement, [2, 2])
        .unwrap();

    assert_eq!(opened.decode_all().unwrap(), values_as_bytes(&original));
    assert_eq!(
        open_dense(&matrix_dir).unwrap().decode_all().unwrap(),
        values_as_bytes(&replacement)
    );
    assert!(has_retired_generation(temp.path()));
    drop(opened);
    assert!(!has_retired_generation(temp.path()));
}

#[test]
fn cross_process_generation_reader_helper() {
    let Ok(matrix_dir) = std::env::var("SC_COMPRESS_READER_MATRIX") else {
        return;
    };
    let ready = std::path::PathBuf::from(std::env::var("SC_COMPRESS_READER_READY").unwrap());
    let release = std::path::PathBuf::from(std::env::var("SC_COMPRESS_READER_RELEASE").unwrap());
    let matrix = open_dense(matrix_dir).unwrap();
    std::fs::write(&ready, b"ready").unwrap();
    wait_for_path(&release);
    assert_eq!(
        matrix.decode_all().unwrap(),
        values_as_bytes(&[1u16, 2, 3, 4])
    );
}

#[test]
fn cross_process_reader_survives_replacement() {
    let temp = tempfile::tempdir().unwrap();
    let matrix_dir = temp.path().join("matrix");
    let ready = temp.path().join("ready");
    let release = temp.path().join("release");
    DenseWriter::new(&matrix_dir, Partition::fixed_cells(1), Partition::fixed_cells(16))
        .write(&[1u16, 2, 3, 4], [2, 2])
        .unwrap();

    let mut child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("cross_process_generation_reader_helper")
        .arg("--nocapture")
        .env("SC_COMPRESS_READER_MATRIX", &matrix_dir)
        .env("SC_COMPRESS_READER_READY", &ready)
        .env("SC_COMPRESS_READER_RELEASE", &release)
        .spawn()
        .unwrap();
    wait_for_path(&ready);

    DenseWriter::new(&matrix_dir, Partition::fixed_cells(1), Partition::fixed_cells(16))
        .write(&[11u16, 12, 13, 14], [2, 2])
        .unwrap();
    assert!(has_retired_generation(temp.path()));
    std::fs::write(&release, b"release").unwrap();
    assert!(child.wait().unwrap().success());
    let deadline = Instant::now() + Duration::from_secs(5);
    while has_retired_generation(temp.path()) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(!has_retired_generation(temp.path()));
}

#[test]
fn directory_store_does_not_follow_key_symlinks() {
    let temp = tempfile::tempdir().unwrap();
    let outside = temp.path().join("outside");
    std::fs::write(&outside, b"secret").unwrap();
    let root = temp.path().join("store");
    let store = DirectoryStore::create(&root).unwrap();
    symlink(&outside, root.join("link")).unwrap();
    assert!(store.read("link").is_err());

    let outside_dir = temp.path().join("outside-dir");
    std::fs::create_dir(&outside_dir).unwrap();
    std::fs::write(outside_dir.join("secret"), b"secret").unwrap();
    symlink(&outside_dir, root.join("nested")).unwrap();
    assert!(store.read("nested/secret").is_err());
}

#[test]
fn directory_store_rejects_escaping_keys() {
    let temp = tempfile::tempdir().unwrap();
    let mut store = DirectoryStore::create(temp.path().join("store")).unwrap();
    for key in [
        "",
        "/absolute",
        "../outside",
        "a/../outside",
        "./value",
        "a//b",
    ] {
        assert!(store.read(key).is_err(), "read accepted {key:?}");
        assert!(
            store.write(key, b"value").is_err(),
            "write accepted {key:?}"
        );
    }
    store.write("a..b", b"value").unwrap();
    assert_eq!(store.read("a..b").unwrap(), b"value");
}

#[test]
fn metadata_version_and_chunk_coverage_are_validated() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("store");
    let mut store = DirectoryStore::create(&root).unwrap();
    write_json(&mut store, &dense_meta(2, vec![0]));
    assert!(open_dense(&root).is_err());

    write_json(&mut store, &dense_meta(1, vec![]));
    assert!(open_dense(&root).is_err());

    let mut escaping = dense_meta(1, vec![0]);
    escaping["data"]["path"] = json!("../outside");
    write_json(&mut store, &escaping);
    assert!(open_dense(&root).is_err());
}

#[test]
fn writer_preserves_unrelated_nonempty_directory() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("not-a-store");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("sentinel"), b"keep").unwrap();

    assert!(DenseWriter::new(&root, Partition::fixed_cells(1024), Partition::fixed_cells(16)).write(&[1u16], [1, 1]).is_err());
    assert_eq!(std::fs::read(root.join("sentinel")).unwrap(), b"keep");
}

#[test]
fn invalid_worker_counts_do_not_touch_writer_targets() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("sentinel");
    std::fs::write(&target, b"keep").unwrap();

    assert!(DenseWriter::new(&target, Partition::fixed_cells(1024), Partition::fixed_cells(16))
        .threads(0)
        .write(&[1u16], [1, 1])
        .is_err());
    assert!(CsrWriter::new(&target, Partition::fixed_cells(1024), Partition::fixed_cells(16))
        .threads(0)
        .write(&[0u64, 1], &[0u32], &[1f32], [1, 1])
        .is_err());
    assert_eq!(std::fs::read(&target).unwrap(), b"keep");
}

#[test]
fn descending_indptr_is_rejected_while_opening() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("store");
    let mut store = DirectoryStore::create(&root).unwrap();
    write_json(&mut store, &csr_meta([2, 3], 1));
    let indptr = [0u64, 2, 1]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    store.write("indptr", &indptr).unwrap();
    assert!(open_csr(&root).is_err());
}

#[test]
fn configured_limits_bound_metadata_and_decoded_working_set() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("matrix");
    let values = vec![1u16; 8];
    DenseWriter::new(&root, Partition::fixed_cells(1024), Partition::fixed_cells(16)).write(&values, [4, 2]).unwrap();

    assert!(
        open_dense_with_limits(&root, ReadLimits::default().maximum_metadata_size(1),).is_err()
    );
    let matrix = open_dense_with_limits(
        &root,
        ReadLimits::default().maximum_decoded_size(values.len() * 2 - 1),
    )
    .unwrap();
    assert!(matrix.decode_all().is_err());

    let matrix =
        open_dense_with_limits(&root, ReadLimits::default().maximum_encoded_size(1)).unwrap();
    assert!(matrix.decode_all().is_err());
}

#[test]
fn huge_dense_output_returns_an_allocation_error() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("matrix");
    let mut store = DirectoryStore::create(&root).unwrap();
    let mut meta = dense_meta(1, vec![0]);
    meta["shape"] = json!([u64::MAX / 2, 1]);
    write_json(&mut store, &meta);

    let matrix = open_dense_with_limits(&root, ReadLimits::unlimited()).unwrap();
    assert!(matches!(matrix.decode_all(), Err(Error::Allocation(_))));
}

#[test]
fn csr_decode_limit_counts_resident_indptr() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("matrix");
    let indptr = (0..=16u64).map(|row| row * 2).collect::<Vec<_>>();
    let indices = (0..32u32).collect::<Vec<_>>();
    let data = (0..32u32).map(|value| value as f32).collect::<Vec<_>>();
    let options = BloscOptions::default()
        .compression_level(0)
        .shuffle(ShuffleMode::None);
    CsrWriter::new(&root, Partition::fixed_cells(1024), Partition::fixed_cells(16))
        .compressor(Compressor::dyn_blosc(options))
        .write(&indptr, &indices, &data, [16, 32])
        .unwrap();

    let matrix = CsrMatrix::from_store_with_limits(
        Arc::new(DirectoryStore::open(&root).unwrap()),
        ReadLimits::default().maximum_decoded_size(420),
    )
    .unwrap();
    let error = matrix.decode_all().unwrap_err();
    assert!(matches!(
        error,
        Error::CorruptData { context, .. } if context == "csr selected resident output"
    ));
}

#[test]
fn malformed_dyn_blosc_prefix_returns_error() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("matrix");
    CsrWriter::new(&root, Partition::fixed_cells(2), Partition::fixed_cells(1))
        .write(&[0u64, 1, 2], &[0u32, 1], &[1f32, 2.0], [2, 2])
        .unwrap();

    let mut store = DirectoryStore::open(&root).unwrap();
    let mut encoded = store.read("indices/0").unwrap();
    encoded.truncate(dyn_blosc::HEADER_LEN);
    store.write("indices/0", &encoded).unwrap();
    assert!(open_csr(&root).unwrap().decode_all().is_err());
}

#[test]
fn decoded_csr_indices_are_checked_against_shape() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("matrix");
    let options = BloscOptions::default()
        .compression_level(0)
        .shuffle(ShuffleMode::None);
    CsrWriter::new(&root, Partition::fixed_cells(1024), Partition::fixed_cells(16))
        .compressor(Compressor::dyn_blosc(options))
        .write(&[0u64, 1], &[1u32], &[1f32], [1, 3])
        .unwrap();

    let mut store = DirectoryStore::open(&root).unwrap();
    let mut encoded = store.read("indices/0").unwrap();
    let raw = encoded
        .get_mut(dyn_blosc::HEADER_LEN..dyn_blosc::HEADER_LEN + 2)
        .expect("compression level zero must use the raw representation");
    raw.copy_from_slice(&3u16.to_le_bytes());
    store.write("indices/0", &encoded).unwrap();

    let ranges = Arc::new(Mutex::new(Vec::new()));
    let matrix = CsrMatrix::from_store(Arc::new(CountingStore {
        inner: DirectoryStore::open(&root).unwrap(),
        ranges: Arc::clone(&ranges),
    }))
    .unwrap();
    ranges.lock().unwrap().clear();
    assert!(matrix.decode_all().is_err());
    assert!(ranges
        .lock()
        .unwrap()
        .iter()
        .all(|(key, _, _)| key != "data/0"));
    ranges.lock().unwrap().clear();
    assert!(matrix
        .select(
            Selection::new(AxisIndex::All, AxisIndex::range(0, 1)),
            sc_compress::CsrOutput::Sparse,
        )
        .is_err());
    assert!(ranges
        .lock()
        .unwrap()
        .iter()
        .all(|(key, _, _)| key != "data/0"));
}

#[test]
fn public_partition_parameters_return_errors_instead_of_panicking() {
    let temp = tempfile::tempdir().unwrap();
    assert!(DenseWriter::new(temp.path().join("dense-zero"), Partition::fixed_cells(0), Partition::fixed_cells(16))
        .write(&[1u16], [1, 1])
        .is_err());
    assert!(DenseWriter::new(temp.path().join("dense-huge"), Partition::fixed_cells(1), Partition::fixed_cells(16))
        .write::<u16>(&[], [u64::MAX, 0])
        .is_err());
    assert!(CsrWriter::new(temp.path().join("csr-zero"), Partition::fixed_cells(0), Partition::fixed_cells(16))
        .write(&[0u64, 0], &[] as &[u32], &[] as &[f32], [1, 1])
        .is_err());
}

#[derive(Clone)]
struct CountingStore {
    inner: DirectoryStore,
    ranges: Arc<Mutex<Vec<(String, u64, usize)>>>,
}

impl ByteStore for CountingStore {
    fn len(&self, key: &str) -> Result<u64> {
        self.inner.len(key)
    }

    fn read_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.ranges
            .lock()
            .unwrap()
            .push((key.to_string(), offset, len));
        self.inner.read_range(key, offset, len)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        self.inner.exists(key)
    }

    fn supports_efficient_range_reads(&self, key: &str) -> Result<bool> {
        self.inner.supports_efficient_range_reads(key)
    }
}

#[test]
fn fancy_row_loading_does_not_decode_the_global_bounding_span() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("fancy-chunks");
    let values = (0u16..8).collect::<Vec<_>>();
    DenseWriter::new(&root, Partition::fixed_cells(2), Partition::fixed_cells(16))
        .write(&values, [8, 1])
        .unwrap();
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let store = CountingStore {
        inner: DirectoryStore::open(&root).unwrap(),
        ranges: Arc::clone(&ranges),
    };
    let matrix = DenseMatrix::from_store(Arc::new(store)).unwrap();
    ranges.lock().unwrap().clear();

    let selected = matrix
        .select(Selection::rows_only(AxisIndex::positions([0, 7])))
        .unwrap();
    assert_eq!(selected.values(), [0u16, 7].map(u16::to_le_bytes).concat());
    let touched = ranges
        .lock()
        .unwrap()
        .iter()
        .filter(|(key, _, _)| key.starts_with("data/"))
        .map(|(key, _, _)| key.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        touched,
        ["data/0".to_string(), "data/3".to_string()]
            .into_iter()
            .collect()
    );
}

#[test]
fn raw_dense_2d_selection_coalesces_small_ranges() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("raw-dense-select");
    let rows = 64usize;
    let cols = 64usize;
    let values = (0..rows * cols)
        .map(|value| u16::try_from(value).unwrap())
        .collect::<Vec<_>>();
    let options = BloscOptions::default()
        .compression_level(0)
        .shuffle(ShuffleMode::None);
    DenseWriter::new(&root, Partition::fixed_cells(rows as u64), Partition::fixed_cells(1))
        .compressor(Compressor::blosc1(options, 1))
        .write(&values, [rows as u64, cols as u64])
        .unwrap();
    let encoded_len = std::fs::metadata(root.join("data/0")).unwrap().len() as usize;
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let matrix = DenseMatrix::from_store(Arc::new(CountingStore {
        inner: DirectoryStore::open(&root).unwrap(),
        ranges: Arc::clone(&ranges),
    }))
    .unwrap();
    ranges.lock().unwrap().clear();

    let selected = matrix
        .select(Selection::new(
            AxisIndex::positions((0..rows as u64).step_by(8)),
            AxisIndex::positions((0..cols as u64).step_by(4)),
        ))
        .unwrap();
    assert_eq!(selected.shape(), [8, 16]);
    let data_ranges = ranges
        .lock()
        .unwrap()
        .iter()
        .filter(|(key, _, _)| key == "data/0")
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(data_ranges.len(), 2);
    assert!(data_ranges.iter().all(|(_, _, len)| *len < encoded_len));
}

#[derive(Clone)]
struct ConcurrentReadStore {
    inner: DirectoryStore,
    active: Arc<AtomicUsize>,
    maximum: Arc<AtomicUsize>,
}

impl ByteStore for ConcurrentReadStore {
    fn len(&self, key: &str) -> Result<u64> {
        self.inner.len(key)
    }

    fn read_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let current = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.maximum.fetch_max(current, Ordering::SeqCst);
        let _guard = ActiveReadGuard(&self.active);
        std::thread::sleep(Duration::from_millis(2));
        self.inner.read_range(key, offset, len)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        self.inner.exists(key)
    }

    fn supports_efficient_range_reads(&self, key: &str) -> Result<bool> {
        self.inner.supports_efficient_range_reads(key)
    }
}

struct ActiveReadGuard<'a>(&'a AtomicUsize);

impl Drop for ActiveReadGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[test]
fn parallel_dense_decode_obeys_the_aggregate_memory_budget() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("matrix");
    let rows = 64usize;
    let cols = 32usize;
    // One chunk with many independently compressed row blocks verifies that
    // concurrency is native to the block scheduler rather than chunk workers.
    let chunk_rows = rows;
    let values = (0..rows * cols)
        .map(|value| u16::try_from(value % 251).unwrap())
        .collect::<Vec<_>>();
    DenseWriter::new(&root, Partition::fixed_cells(chunk_rows as u64), Partition::fixed_cells(1))
        .threads(4)
        .write(&values, [rows as u64, cols as u64])
        .unwrap();

    let chunk_bytes = chunk_rows * cols * std::mem::size_of::<u16>();
    let selected_rows = rows - 1;
    let chunk_count = rows / chunk_rows;
    let bounds = (0..chunk_count)
        .map(|id| {
            let encoded = usize::try_from(
                std::fs::metadata(root.join(format!("data/{id}")))
                    .unwrap()
                    .len(),
            )
            .unwrap();
            (encoded * 4).max(encoded * 3 + chunk_bytes * 3)
        })
        .collect::<Vec<_>>();
    let output_bytes = selected_rows * cols * std::mem::size_of::<u16>();
    let exclusive_capacity = *bounds.iter().min().unwrap();
    let parallel_capacity = *bounds.iter().max().unwrap() * 4;

    let run = |capacity: usize| {
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let matrix = DenseMatrix::from_store_with_limits(
            Arc::new(ConcurrentReadStore {
                inner: DirectoryStore::open(&root).unwrap(),
                active: Arc::clone(&active),
                maximum: Arc::clone(&maximum),
            }),
            ReadLimits::default()
                .threads(4)
                .maximum_decoded_size(output_bytes + capacity),
        )
        .unwrap();
        maximum.store(0, Ordering::SeqCst);
        let decoded = matrix.decode_rows(0..selected_rows as u64).unwrap();
        assert_eq!(decoded, values_as_bytes(&values[..selected_rows * cols]));
        assert_eq!(active.load(Ordering::SeqCst), 0);
        maximum.load(Ordering::SeqCst)
    };

    // Block-level accounting can safely admit several blocks under a budget
    // that previously admitted only one whole chunk.
    assert!(run(exclusive_capacity) > 1);
    assert!(run(parallel_capacity) > 1);
}

#[test]
fn parallel_csr_decode_obeys_the_aggregate_memory_budget() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("matrix");
    let rows = 64usize;
    let chunk_rows = rows;
    let indptr = (0..=rows as u64).collect::<Vec<_>>();
    let indices = (0..rows as u32).collect::<Vec<_>>();
    let values = (0..rows as u16).collect::<Vec<_>>();
    CsrWriter::new(&root, Partition::fixed_cells(chunk_rows as u64), Partition::fixed_cells(1))
        .threads(4)
        .write(&indptr, &indices, &values, [rows as u64, rows as u64])
        .unwrap();

    let selected_rows = rows - 1;
    let chunk_array_bytes = chunk_rows * std::mem::size_of::<u16>();
    let chunk_count = rows / chunk_rows;
    let mut bounds = Vec::new();
    for directory in ["indices", "data"] {
        for id in 0..chunk_count {
            let encoded = usize::try_from(
                std::fs::metadata(root.join(format!("{directory}/{id}")))
                    .unwrap()
                    .len(),
            )
            .unwrap();
            bounds.push((encoded * 4).max(encoded * 3 + chunk_array_bytes * 3));
        }
    }
    let resident = indptr.len() * std::mem::size_of::<u64>()
        + (selected_rows + 1) * std::mem::size_of::<u64>()
        + selected_rows * std::mem::size_of::<u16>() * 2;

    let run = |capacity: usize| {
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let matrix = CsrMatrix::from_store_with_limits(
            Arc::new(ConcurrentReadStore {
                inner: DirectoryStore::open(&root).unwrap(),
                active: Arc::clone(&active),
                maximum: Arc::clone(&maximum),
            }),
            ReadLimits::default()
                .threads(4)
                .maximum_decoded_size(resident + capacity),
        )
        .unwrap();
        maximum.store(0, Ordering::SeqCst);
        let (decoded_indices, decoded_values) =
            matrix.decode_rows(0..selected_rows as u64).unwrap();
        assert_eq!(decoded_indices, values_as_bytes(&values[..selected_rows]));
        assert_eq!(decoded_values, values_as_bytes(&values[..selected_rows]));
        assert_eq!(active.load(Ordering::SeqCst), 0);
        maximum.load(Ordering::SeqCst)
    };

    let exclusive_capacity = *bounds.iter().max().unwrap();
    let parallel_capacity = exclusive_capacity * 4;
    assert!(run(exclusive_capacity) > 1);
    assert!(run(parallel_capacity) > 1);
}

#[derive(Clone)]
struct ReplayingStore {
    inner: DirectoryStore,
    ranges: Arc<Mutex<Vec<(String, u64, usize)>>>,
}

impl ByteStore for ReplayingStore {
    fn len(&self, key: &str) -> Result<u64> {
        self.inner.len(key)
    }

    fn read_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.ranges
            .lock()
            .unwrap()
            .push((key.to_string(), offset, len));
        self.inner.read_range(key, offset, len)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        self.inner.exists(key)
    }

    fn supports_efficient_range_reads(&self, key: &str) -> Result<bool> {
        let _ = self.inner.len(key)?;
        Ok(false)
    }
}

#[test]
fn dense_row_selection_reads_only_intersecting_blocks() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("matrix");
    let values = vec![7u16; 256 * 256];
    DenseWriter::new(&root, Partition::fixed_cells(256), Partition::fixed_cells(2))
        .write(&values, [256, 256])
        .unwrap();

    let inner = DirectoryStore::open(&root).unwrap();
    let encoded_len = usize::try_from(inner.len("data/0").unwrap()).unwrap();
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let matrix = DenseMatrix::from_store_with_limits(
        Arc::new(CountingStore {
            inner,
            ranges: Arc::clone(&ranges),
        }),
        ReadLimits::default().maximum_decoded_size(8192),
    )
    .unwrap();
    ranges.lock().unwrap().clear();

    assert_eq!(
        matrix.decode_rows(10..11).unwrap(),
        values_as_bytes(&[7u16; 256])
    );
    let range_guard = ranges.lock().unwrap();
    assert!(range_guard.iter().any(|(key, _, _)| key == "data/0"));
    assert!(range_guard
        .iter()
        .all(|(key, _, len)| key != "data/0" || *len < encoded_len));
    drop(range_guard);

    ranges.lock().unwrap().clear();
    let selected = matrix
        .select(Selection::rows_only(AxisIndex::positions([0, 255])))
        .unwrap();
    assert_eq!(selected.shape(), [2, 256]);
    assert_eq!(
        touched_blocks(&root, "data/0", &ranges.lock().unwrap()),
        [0, 127].into_iter().collect()
    );

    let limited_ranges = Arc::new(Mutex::new(Vec::new()));
    let limited = DenseMatrix::from_store_with_limits(
        Arc::new(CountingStore {
            inner: DirectoryStore::open(&root).unwrap(),
            ranges: Arc::clone(&limited_ranges),
        }),
        ReadLimits::default().maximum_decoded_size(1024),
    )
    .unwrap();
    limited_ranges.lock().unwrap().clear();
    assert!(limited.decode_rows(10..11).is_err());
    let limited_ranges = limited_ranges.lock().unwrap();
    assert!(limited_ranges
        .iter()
        .all(|(key, _, len)| { key != "data/0" || *len == dyn_blosc::HEADER_LEN }));
    drop(limited_ranges);

    let replay_ranges = Arc::new(Mutex::new(Vec::new()));
    let selected_bytes = 256 * std::mem::size_of::<u16>();
    let replay_limit = selected_bytes.checked_add(encoded_len).unwrap() - 1;
    let replay_limited = DenseMatrix::from_store_with_limits(
        Arc::new(ReplayingStore {
            inner: DirectoryStore::open(&root).unwrap(),
            ranges: Arc::clone(&replay_ranges),
        }),
        ReadLimits::default().maximum_decoded_size(replay_limit),
    )
    .unwrap();
    replay_ranges.lock().unwrap().clear();
    assert!(replay_limited.decode_rows(10..11).is_err());
    assert!(replay_ranges
        .lock()
        .unwrap()
        .iter()
        .all(|(key, _, _)| key != "data/0"));

    let block_ranges = Arc::new(Mutex::new(Vec::new()));
    let block_limited = DenseMatrix::from_store_with_limits(
        Arc::new(CountingStore {
            inner: DirectoryStore::open(&root).unwrap(),
            ranges: Arc::clone(&block_ranges),
        }),
        ReadLimits::default().maximum_block_count(0),
    )
    .unwrap();
    block_ranges.lock().unwrap().clear();
    assert!(block_limited.decode_rows(10..11).is_err());
    let block_ranges = block_ranges.lock().unwrap();
    assert!(block_ranges
        .iter()
        .all(|(key, _, len)| { key != "data/0" || *len == dyn_blosc::HEADER_LEN }));
}

#[test]
fn full_dense_chunk_is_staged_once_instead_of_reading_each_block() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("full-dense-staging");
    let rows = 64usize;
    let cols = 64usize;
    let values = (0..rows * cols)
        .map(|value| u16::try_from(value % 251).unwrap())
        .collect::<Vec<_>>();
    DenseWriter::new(&root, Partition::fixed_cells(rows as u64), Partition::fixed_cells(1))
        .write(&values, [rows as u64, cols as u64])
        .unwrap();
    let encoded_len = std::fs::metadata(root.join("data/0")).unwrap().len() as usize;
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let matrix = DenseMatrix::from_store(Arc::new(CountingStore {
        inner: DirectoryStore::open(&root).unwrap(),
        ranges: Arc::clone(&ranges),
    }))
    .unwrap();
    ranges.lock().unwrap().clear();

    assert_eq!(matrix.decode_all().unwrap(), values_as_bytes(&values));
    let data_ranges = ranges
        .lock()
        .unwrap()
        .iter()
        .filter(|(key, _, _)| key == "data/0")
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        data_ranges
            .iter()
            .filter(|(_, offset, len)| *offset == 0 && *len == encoded_len)
            .count(),
        1
    );
    assert!(data_ranges.len() <= 3);
}

#[test]
fn csr_row_selection_reads_only_intersecting_blocks() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("matrix");
    let indptr = (0..=64u64).collect::<Vec<_>>();
    let indices = (0..64u32).collect::<Vec<_>>();
    let data = (0..64u32)
        .map(|value| value as f32 + 0.25)
        .collect::<Vec<_>>();
    CsrWriter::new(&root, Partition::fixed_cells(64), Partition::fixed_cells(1))
        .write(&indptr, &indices, &data, [64, 64])
        .unwrap();

    let inner = DirectoryStore::open(&root).unwrap();
    let encoded_indices_len = usize::try_from(inner.len("indices/0").unwrap()).unwrap();
    let encoded_data_len = usize::try_from(inner.len("data/0").unwrap()).unwrap();
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let matrix = CsrMatrix::from_store(Arc::new(CountingStore {
        inner,
        ranges: Arc::clone(&ranges),
    }))
    .unwrap();
    ranges.lock().unwrap().clear();

    matrix.decode_rows(10..11).unwrap();
    let range_guard = ranges.lock().unwrap();
    assert!(range_guard.iter().any(|(key, _, _)| key == "indices/0"));
    assert!(range_guard.iter().any(|(key, _, _)| key == "data/0"));
    assert!(range_guard.iter().all(|(key, _, len)| {
        (key != "indices/0" || *len < encoded_indices_len)
            && (key != "data/0" || *len < encoded_data_len)
    }));
    drop(range_guard);

    ranges.lock().unwrap().clear();
    let selected = matrix
        .select(
            Selection::rows_only(AxisIndex::positions([0, 63])),
            sc_compress::CsrOutput::Sparse,
        )
        .unwrap();
    assert_eq!(selected.shape(), [2, 64]);
    let ranges = ranges.lock().unwrap();
    assert_eq!(
        touched_blocks(&root, "indices/0", &ranges),
        [0, 63].into_iter().collect()
    );
    assert_eq!(
        touched_blocks(&root, "data/0", &ranges),
        [0, 63].into_iter().collect()
    );
}

#[test]
fn csr_column_selection_reads_only_data_blocks_with_matches() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("csr-column-blocks");
    let rows = 64usize;
    let indptr = (0..=rows as u64).collect::<Vec<_>>();
    let indices = (0..rows as u32).collect::<Vec<_>>();
    let data = (0..rows as u16).map(|value| value + 7).collect::<Vec<_>>();
    CsrWriter::new(&root, Partition::fixed_cells(rows as u64), Partition::fixed_cells(1))
        .write(&indptr, &indices, &data, [rows as u64, rows as u64])
        .unwrap();

    let ranges = Arc::new(Mutex::new(Vec::new()));
    let matrix = CsrMatrix::from_store(Arc::new(CountingStore {
        inner: DirectoryStore::open(&root).unwrap(),
        ranges: Arc::clone(&ranges),
    }))
    .unwrap();
    ranges.lock().unwrap().clear();

    let selected = matrix
        .select(
            Selection::new(AxisIndex::All, AxisIndex::range(0, 1)),
            sc_compress::CsrOutput::Sparse,
        )
        .unwrap();
    let sc_compress::SelectedArray::Csr(selected) = selected else {
        panic!("expected CSR selection");
    };
    assert_eq!(selected.shape(), [rows, 1]);
    assert_eq!(selected.nnz(), 1);
    assert_eq!(selected.data(), 7u16.to_le_bytes());
    assert_eq!(
        touched_blocks(&root, "data/0", &ranges.lock().unwrap()),
        [0].into_iter().collect()
    );

    ranges.lock().unwrap().clear();
    let selected = matrix
        .select(
            Selection::new(AxisIndex::All, AxisIndex::positions([63, 0, 63])),
            sc_compress::CsrOutput::Dense,
        )
        .unwrap();
    let sc_compress::SelectedArray::Dense(selected) = selected else {
        panic!("expected dense selection");
    };
    assert_eq!(selected.shape(), [rows, 3]);
    assert_eq!(
        touched_blocks(&root, "data/0", &ranges.lock().unwrap()),
        [0, 63].into_iter().collect()
    );
}

fn touched_blocks(
    root: &std::path::Path,
    key: &str,
    ranges: &[(String, u64, usize)],
) -> std::collections::BTreeSet<usize> {
    let encoded = std::fs::read(root.join(key)).unwrap();
    let decoder = dyn_blosc::Decoder::from_encoded(&encoded).unwrap();
    decoder
        .blocks()
        .filter(|block| {
            let encoded = block.encoded_range();
            ranges.iter().any(|(candidate, offset, len)| {
                candidate == key
                    && usize::try_from(*offset).unwrap() == encoded.start
                    && *len == encoded.len()
            })
        })
        .map(|block| block.index())
        .collect()
}

fn values_as_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn has_retired_generation(parent: &std::path::Path) -> bool {
    std::fs::read_dir(parent).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".sc-compress-staging-")
    })
}

fn wait_for_path(path: &std::path::Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !path.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(path.exists(), "timed out waiting for {}", path.display());
}
