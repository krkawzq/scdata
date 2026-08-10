use std::hint::black_box;
use std::time::{Duration, Instant};

use sc_compress::{AxisIndex, CsrArray, CsrOutput, DType, DenseArray, Selection};

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let dense_rows = 10_000usize;
    let dense_cols = 1_024usize;
    let dense = DenseArray::from_bytes(
        [dense_rows, dense_cols],
        DType::F32,
        vec![0; dense_rows * dense_cols * DType::F32.size()],
    )
    .unwrap();
    let selected_rows = (0..dense_rows as u64).step_by(2).collect::<Vec<_>>();
    let selected_cols = (0..256u64).map(|column| column * 3 + 1).collect::<Vec<_>>();
    let dense_selection = Selection::new(
        AxisIndex::positions(selected_rows),
        AxisIndex::positions(selected_cols),
    );

    for threads in [1, 4] {
        let mut samples = Vec::new();
        for _ in 0..7 {
            let start = Instant::now();
            black_box(dense.select(dense_selection.clone(), threads).unwrap());
            samples.push(start.elapsed());
        }
        println!(
            "dense_gather threads={threads} median={:?}",
            median(samples)
        );
    }

    let csr_rows = 20_000usize;
    let csr_cols = 4_096usize;
    let nnz_per_row = 128usize;
    let mut indptr = Vec::with_capacity(csr_rows + 1);
    for row in 0..=csr_rows {
        indptr.push((row * nnz_per_row) as u64);
    }
    let row_indices = (0..nnz_per_row as u16)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let mut indices = Vec::with_capacity(csr_rows * row_indices.len());
    for _ in 0..csr_rows {
        indices.extend_from_slice(&row_indices);
    }
    let data = vec![0; csr_rows * nnz_per_row * DType::F32.size()];
    let csr = CsrArray::from_parts(
        [csr_rows, csr_cols],
        DType::U16,
        DType::F32,
        indptr,
        indices,
        data,
    )
    .unwrap();
    let contiguous_cols = AxisIndex::range(32, 96).normalize(csr_cols as u64).unwrap();

    for threads in [1, 4] {
        let mut samples = Vec::new();
        for _ in 0..7 {
            let input = csr.clone();
            let start = Instant::now();
            black_box(
                input
                    .select_columns(&contiguous_cols, CsrOutput::Sparse, threads)
                    .unwrap(),
            );
            samples.push(start.elapsed());
        }
        println!(
            "csr_col_range threads={threads} median={:?}",
            median(samples)
        );
    }
}
