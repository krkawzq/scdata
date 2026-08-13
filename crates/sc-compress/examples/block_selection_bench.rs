use std::hint::black_box;
use std::time::{Duration, Instant};

use sc_compress::{
    open_csr_with_limits, open_dense_with_limits, AxisIndex, CsrOutput, CsrWriter, DenseWriter,
    Partition, ReadLimits, SelectedArray, Selection,
};

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn measure(mut operation: impl FnMut()) -> Duration {
    operation();
    median(
        (0..7)
            .map(|_| {
                let start = Instant::now();
                operation();
                start.elapsed()
            })
            .collect(),
    )
}

fn main() {
    let temp = tempfile::tempdir().unwrap();
    let rows = 8_192usize;
    let cols = 256usize;
    let selected_rows = (0..rows as u64).step_by(64).collect::<Vec<_>>();
    let limits = ReadLimits::default().threads(4);

    let dense_root = temp.path().join("dense");
    let dense_values = (0..rows * cols)
        .map(|value| u16::try_from(value % 251).unwrap())
        .collect::<Vec<_>>();
    DenseWriter::new(
        &dense_root,
        Partition::fixed_cells(4_096),
        Partition::fixed_cells(8),
    )
    .threads(4)
    .write(&dense_values, [rows as u64, cols as u64])
    .unwrap();
    let dense = open_dense_with_limits(&dense_root, limits).unwrap();
    let selection = Selection::rows_only(AxisIndex::positions(selected_rows.clone()));
    let direct_dense = measure(|| {
        black_box(dense.select(selection.clone()).unwrap());
    });
    let bounding_dense = measure(|| {
        let decoded = dense
            .decode_rows(selected_rows[0]..selected_rows[selected_rows.len() - 1] + 1)
            .unwrap();
        let first = selected_rows[0] as usize;
        let mut output = Vec::with_capacity(selected_rows.len() * cols * 2);
        for &row in &selected_rows {
            let local = row as usize - first;
            output.extend_from_slice(&decoded[local * cols * 2..(local + 1) * cols * 2]);
        }
        black_box(output);
    });
    println!("dense_block_scatter median={direct_dense:?}");
    println!("dense_bounding_window median={bounding_dense:?}");

    let selected_cols = (0..cols as u64).step_by(4).collect::<Vec<_>>();
    let selection_2d = Selection::new(
        AxisIndex::positions(selected_rows.clone()),
        AxisIndex::positions(selected_cols.clone()),
    );
    let direct_dense_2d = measure(|| {
        black_box(dense.select(selection_2d.clone()).unwrap());
    });
    let bounding_dense_2d = measure(|| {
        let decoded = dense
            .decode_rows(selected_rows[0]..selected_rows[selected_rows.len() - 1] + 1)
            .unwrap();
        let first = selected_rows[0] as usize;
        let mut output = Vec::with_capacity(selected_rows.len() * selected_cols.len() * 2);
        for &row in &selected_rows {
            let row_base = (row as usize - first) * cols * 2;
            for &col in &selected_cols {
                let source = row_base + col as usize * 2;
                output.extend_from_slice(&decoded[source..source + 2]);
            }
        }
        black_box(output);
    });
    println!("dense_2d_block_scatter median={direct_dense_2d:?}");
    println!("dense_2d_bounding_window median={bounding_dense_2d:?}");

    let csr_root = temp.path().join("csr");
    let nnz_per_row = 64usize;
    let indptr = (0..=rows)
        .map(|row| (row * nnz_per_row) as u64)
        .collect::<Vec<_>>();
    let mut indices = Vec::with_capacity(rows * nnz_per_row);
    let mut values = Vec::with_capacity(rows * nnz_per_row);
    for row in 0..rows {
        for col in 0..nnz_per_row {
            indices.push(col as u32);
            values.push(((row + col) % 251) as u16);
        }
    }
    CsrWriter::new(
        &csr_root,
        Partition::fixed_cells(4_096),
        Partition::fixed_cells(8),
    )
    .threads(4)
    .write(&indptr, &indices, &values, [rows as u64, cols as u64])
    .unwrap();
    let csr = open_csr_with_limits(&csr_root, limits).unwrap();
    let selection = Selection::rows_only(AxisIndex::positions(selected_rows.clone()));
    let direct_csr = measure(|| {
        let result = csr.select(selection.clone(), CsrOutput::Sparse).unwrap();
        let SelectedArray::Csr(result) = result else {
            unreachable!();
        };
        black_box(result);
    });
    let bounding_csr = measure(|| {
        let first = selected_rows[0] as usize;
        let past_last = selected_rows[selected_rows.len() - 1] as usize + 1;
        let (decoded_indices, decoded_data) =
            csr.decode_rows(first as u64..past_last as u64).unwrap();
        let nnz_base = csr.indptr()[first] as usize;
        let mut output_indices = Vec::with_capacity(selected_rows.len() * nnz_per_row * 2);
        let mut output_data = Vec::with_capacity(selected_rows.len() * nnz_per_row * 2);
        for &row in &selected_rows {
            let row = row as usize;
            let start = csr.indptr()[row] as usize - nnz_base;
            let end = csr.indptr()[row + 1] as usize - nnz_base;
            output_indices.extend_from_slice(&decoded_indices[start * 2..end * 2]);
            output_data.extend_from_slice(&decoded_data[start * 2..end * 2]);
        }
        black_box((output_indices, output_data));
    });
    println!("csr_block_scatter median={direct_csr:?}");
    println!("csr_bounding_window median={bounding_csr:?}");

    let wide_column_selection = Selection::new(AxisIndex::All, AxisIndex::range(8, 56));
    let direct_csr_wide_columns = measure(|| {
        black_box(
            csr.select(wide_column_selection.clone(), CsrOutput::Sparse)
                .unwrap(),
        );
    });
    println!("csr_wide_column_range median={direct_csr_wide_columns:?}");

    let sparse_cols_root = temp.path().join("csr-sparse-columns");
    let sparse_col_indptr = (0..=rows)
        .map(|row| (row * nnz_per_row) as u64)
        .collect::<Vec<_>>();
    let mut sparse_col_indices = Vec::with_capacity(rows * nnz_per_row);
    let mut sparse_col_values = Vec::with_capacity(rows * nnz_per_row);
    for row in 0..rows {
        let first_col = if row % 64 == 0 { 0 } else { 64 };
        for offset in 0..nnz_per_row {
            sparse_col_indices.push(u32::try_from(first_col + offset).unwrap());
            sparse_col_values.push((row * nnz_per_row + offset) as f64);
        }
    }
    CsrWriter::new(
        &sparse_cols_root,
        Partition::fixed_cells(rows as u64),
        Partition::fixed_cells(1),
    )
    .threads(4)
    .write(
        &sparse_col_indptr,
        &sparse_col_indices,
        &sparse_col_values,
        [rows as u64, cols as u64],
    )
    .unwrap();
    let sparse_cols = open_csr_with_limits(&sparse_cols_root, limits).unwrap();
    let sparse_col_selection = Selection::new(AxisIndex::All, AxisIndex::range(0, 1));
    let direct_sparse_cols = measure(|| {
        black_box(
            sparse_cols
                .select(sparse_col_selection.clone(), CsrOutput::Sparse)
                .unwrap(),
        );
    });
    let full_sparse_col_rows = measure(|| {
        black_box(sparse_cols.decode_rows(0..rows as u64).unwrap());
    });
    println!("csr_sparse_col_block_scatter median={direct_sparse_cols:?}");
    println!("csr_sparse_col_full_rows median={full_sparse_col_rows:?}");
}
