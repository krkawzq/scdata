//! On-demand selection against store-backed dense / CSR matrices.
//!
//! Dense selections decode intersecting blocks directly into their final 2-D
//! output. CSR column selections decode and validate the selected index blocks
//! first, then load only data blocks containing values mapped to the output.

use crate::array::{CsrArray, DenseArray, SelectedArray};
use crate::csr::CsrMatrix;
use crate::dense::DenseMatrix;
use crate::error::{Error, Result};
use crate::matrix::Matrix;
use crate::select::{AxisIndex, CsrOutput, NormalizedAxis, NormalizedSelection, Selection};

impl DenseMatrix {
    /// Select arbitrary rows and columns into an owned dense array.
    pub fn select(&self, selection: Selection) -> Result<DenseArray> {
        let normalized = selection.normalize(self.n_rows(), self.n_cols())?;
        self.select_normalized(&normalized)
    }

    pub fn select_normalized(&self, selection: &NormalizedSelection) -> Result<DenseArray> {
        selection.rows.validate(self.n_rows())?;
        selection.cols.validate(self.n_cols())?;
        let n_rows = usize_from_u64(selection.rows.len(), "selected rows")?;
        let n_cols = usize_from_u64(selection.cols.len(), "selected columns")?;
        DenseArray::from_bytes(
            [n_rows, n_cols],
            self.dtype(),
            self.decode_selection(&selection.rows, &selection.cols)?,
        )
    }

    /// Gather rows by explicit positions (order-preserving, duplicates allowed).
    pub fn gather_rows(&self, rows: &[u64]) -> Result<DenseArray> {
        self.select(Selection::rows_only(AxisIndex::positions(
            rows.iter().copied(),
        )))
    }

    /// Decode a contiguous row range into a dense array.
    pub fn load_rows(&self, start: u64, end: u64) -> Result<DenseArray> {
        self.select(Selection::rows_only(AxisIndex::range(start, end)))
    }
}

impl CsrMatrix {
    /// Select rows/columns. Column selection stays CSR unless `output` is dense.
    pub fn select(&self, selection: Selection, output: CsrOutput) -> Result<SelectedArray> {
        let normalized = selection.normalize(self.n_rows(), self.n_cols())?;
        self.select_normalized(&normalized, output)
    }

    pub fn select_normalized(
        &self,
        selection: &NormalizedSelection,
        output: CsrOutput,
    ) -> Result<SelectedArray> {
        selection.rows.validate(self.n_rows())?;
        selection.cols.validate(self.n_cols())?;
        let full_columns = selection
            .cols
            .as_range()
            .is_some_and(|range| range.start == 0 && range.end == self.n_cols());
        if !full_columns {
            return self.decode_selection(&selection.rows, &selection.cols, output);
        }
        let threads = self.limits().thread_count();
        let rows_materialized = self.materialize_rows(&selection.rows)?;
        let source_indptr_bytes = checked_mul(
            self.indptr().len(),
            std::mem::size_of::<u64>(),
            "CSR source indptr",
        )?;
        rows_materialized.select_columns_with_limits(
            &selection.cols,
            output,
            threads,
            self.limits(),
            source_indptr_bytes,
        )
    }

    pub fn gather_rows(&self, rows: &[u64], output: CsrOutput) -> Result<SelectedArray> {
        self.select(
            Selection::rows_only(AxisIndex::positions(rows.iter().copied())),
            output,
        )
    }

    pub fn load_rows(&self, start: u64, end: u64, output: CsrOutput) -> Result<SelectedArray> {
        self.select(Selection::rows_only(AxisIndex::range(start, end)), output)
    }

    fn materialize_rows(&self, rows: &NormalizedAxis) -> Result<CsrArray> {
        let index_dtype = self.index_dtype();
        let value_dtype = self.value_dtype();
        let n_cols = usize_from_u64(self.n_cols(), "n_cols")?;
        let n_rows = usize_from_u64(rows.len(), "selected rows")?;
        let (indptr, indices, data) = self.decode_selected_rows(rows)?;
        Ok(CsrArray::from_parts_validated(
            [n_rows, n_cols],
            index_dtype,
            value_dtype,
            indptr,
            indices,
            data,
        ))
    }
}

impl Matrix {
    /// Select from either dense or CSR store.
    ///
    /// Dense always returns [`SelectedArray::Dense`]. CSR respects `csr_output`.
    pub fn select(&self, selection: Selection, csr_output: CsrOutput) -> Result<SelectedArray> {
        match self {
            Self::Dense(matrix) => Ok(SelectedArray::Dense(matrix.select(selection)?)),
            Self::Csr(matrix) => matrix.select(selection, csr_output),
        }
    }
}

fn checked_mul(count: usize, size: usize, context: &str) -> Result<usize> {
    count
        .checked_mul(size)
        .ok_or_else(|| Error::invalid_argument(format!("{context} size overflow")))
}

fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::invalid_argument(format!("{context} exceeds usize")))
}
