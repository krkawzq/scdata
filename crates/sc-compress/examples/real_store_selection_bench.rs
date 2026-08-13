use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sc_compress::{
    AxisIndex, ByteStore, CsrMatrix, CsrOutput, DirectoryStore, PositionedValue, ReadLimits,
    Result, Selection, ZipStore,
};

#[derive(Default)]
struct ReadStats {
    calls: AtomicU64,
    bytes: AtomicU64,
}

impl ReadStats {
    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> (u64, u64) {
        (
            self.calls.load(Ordering::Relaxed),
            self.bytes.load(Ordering::Relaxed),
        )
    }
}

struct InstrumentedStore {
    inner: Arc<dyn ByteStore>,
    positioned: bool,
    stats: Arc<ReadStats>,
}

fn open_store(path: &str) -> Arc<dyn ByteStore> {
    if Path::new(path).is_dir() {
        Arc::new(DirectoryStore::open(path).unwrap())
    } else {
        Arc::new(ZipStore::open(path, "X").unwrap())
    }
}

impl ByteStore for InstrumentedStore {
    fn len(&self, key: &str) -> Result<u64> {
        self.inner.len(key)
    }

    fn read_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes.fetch_add(len as u64, Ordering::Relaxed);
        self.inner.read_range(key, offset, len)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        self.inner.exists(key)
    }

    fn supports_efficient_range_reads(&self, key: &str) -> Result<bool> {
        self.inner.supports_efficient_range_reads(key)
    }

    fn open_positioned(&self, key: &str) -> Result<Option<PositionedValue>> {
        if self.positioned {
            self.inner.open_positioned(key)
        } else {
            Ok(None)
        }
    }
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn timed_select(
    matrix: &CsrMatrix,
    selection: &Selection,
    output: CsrOutput,
) -> (Duration, sc_compress::SelectedArray) {
    let start = Instant::now();
    let selected = matrix.select(selection.clone(), output).unwrap();
    (start.elapsed(), selected)
}

fn bench_case(
    name: &str,
    positioned: &CsrMatrix,
    positioned_stats: &ReadStats,
    range: &CsrMatrix,
    range_stats: &ReadStats,
    selection: Selection,
    output: CsrOutput,
) {
    let positioned_result = positioned.select(selection.clone(), output).unwrap();
    let range_result = range.select(selection.clone(), output).unwrap();
    assert_eq!(positioned_result, range_result);
    black_box(positioned_result);
    black_box(range_result);

    let mut positioned_samples = Vec::with_capacity(7);
    let mut range_samples = Vec::with_capacity(7);
    for round in 0..7 {
        if round % 2 == 0 {
            let (elapsed, selected) = timed_select(positioned, &selection, output);
            positioned_samples.push(elapsed);
            black_box(selected);
            let (elapsed, selected) = timed_select(range, &selection, output);
            range_samples.push(elapsed);
            black_box(selected);
        } else {
            let (elapsed, selected) = timed_select(range, &selection, output);
            range_samples.push(elapsed);
            black_box(selected);
            let (elapsed, selected) = timed_select(positioned, &selection, output);
            positioned_samples.push(elapsed);
            black_box(selected);
        }
    }

    positioned_stats.reset();
    black_box(positioned.select(selection.clone(), output).unwrap());
    let positioned_io = positioned_stats.snapshot();
    range_stats.reset();
    black_box(range.select(selection, output).unwrap());
    let range_io = range_stats.snapshot();

    let positioned_median = median(positioned_samples);
    let range_median = median(range_samples);
    let speedup = range_median.as_secs_f64() / positioned_median.as_secs_f64();
    println!(
        "{name} positioned={positioned_median:?} range={range_median:?} speedup={speedup:.3}x \
         range_api_calls={}->{} range_api_bytes={}->{}",
        range_io.0, positioned_io.0, range_io.1, positioned_io.1,
    );
}

fn evenly_spaced(length: u64, count: usize) -> Vec<u64> {
    let count = count.min(length as usize).max(1);
    (0..count)
        .map(|index| (index as u64).saturating_mul(length) / count as u64)
        .collect()
}

fn main() {
    let paths = std::env::args().skip(1).collect::<Vec<_>>();
    if paths.is_empty() {
        eprintln!("usage: real_store_selection_bench <archive.scc.zip> [...]");
        std::process::exit(2);
    }

    for path in paths {
        let positioned_stats = Arc::new(ReadStats::default());
        let range_stats = Arc::new(ReadStats::default());
        let limits = ReadLimits::default().threads(4);
        let positioned = CsrMatrix::from_store_with_limits(
            Arc::new(InstrumentedStore {
                inner: open_store(&path),
                positioned: true,
                stats: Arc::clone(&positioned_stats),
            }),
            limits,
        )
        .unwrap();
        let range = CsrMatrix::from_store_with_limits(
            Arc::new(InstrumentedStore {
                inner: open_store(&path),
                positioned: false,
                stats: Arc::clone(&range_stats),
            }),
            limits,
        )
        .unwrap();
        assert_eq!(positioned.shape(), range.shape());
        assert_eq!(positioned.nnz(), range.nnz());
        let [n_rows, n_cols] = positioned.shape();
        println!(
            "archive={path} shape={n_rows}x{n_cols} nnz={}",
            positioned.nnz()
        );

        let rows = evenly_spaced(n_rows, 256);
        bench_case(
            "sparse_rows",
            &positioned,
            &positioned_stats,
            &range,
            &range_stats,
            Selection::rows_only(AxisIndex::positions(rows)),
            CsrOutput::Sparse,
        );

        let width = 1_024u64.min(n_cols);
        let start = (n_cols - width) / 2;
        bench_case(
            "contiguous_columns",
            &positioned,
            &positioned_stats,
            &range,
            &range_stats,
            Selection::new(AxisIndex::All, AxisIndex::range(start, start + width)),
            CsrOutput::Sparse,
        );

        bench_case(
            "gathered_columns",
            &positioned,
            &positioned_stats,
            &range,
            &range_stats,
            Selection::new(
                AxisIndex::All,
                AxisIndex::positions(evenly_spaced(n_cols, 128)),
            ),
            CsrOutput::Sparse,
        );

        bench_case(
            "dense_2d",
            &positioned,
            &positioned_stats,
            &range,
            &range_stats,
            Selection::new(
                AxisIndex::positions(evenly_spaced(n_rows, 512)),
                AxisIndex::positions(evenly_spaced(n_cols, 128)),
            ),
            CsrOutput::Dense,
        );
    }
}
