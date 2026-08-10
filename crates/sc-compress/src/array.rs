//! In-memory dense and CSR matrix containers.
//!
//! These are the Rust-native counterparts of the Python `ScDense` / `ScCsr`
//! types. Payload storage is raw little-endian bytes (ready for NumPy views).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::kernel;
use crate::limits::ReadLimits;
use crate::select::{CsrOutput, NormalizedAxis, Selection};

/// Owned row-major dense matrix (`n_rows × n_cols` elements).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseArray {
    shape: [usize; 2],
    dtype: DType,
    /// C-contiguous little-endian values, length `n_rows * n_cols * dtype.size()`.
    values: Vec<u8>,
}

impl DenseArray {
    pub fn from_bytes(shape: [usize; 2], dtype: DType, values: Vec<u8>) -> Result<Self> {
        validate_value_dtype(dtype)?;
        let expected = shape[0]
            .checked_mul(shape[1])
            .and_then(|n| n.checked_mul(dtype.size()))
            .ok_or_else(|| Error::invalid_argument("dense byte length overflow"))?;
        if values.len() != expected {
            return Err(Error::invalid_argument(format!(
                "dense values length {} does not match shape {:?} dtype {}",
                values.len(),
                shape,
                dtype
            )));
        }
        Ok(Self {
            shape,
            dtype,
            values,
        })
    }

    pub fn zeros(shape: [usize; 2], dtype: DType) -> Result<Self> {
        validate_value_dtype(dtype)?;
        let len = shape[0]
            .checked_mul(shape[1])
            .and_then(|n| n.checked_mul(dtype.size()))
            .ok_or_else(|| Error::invalid_argument("dense zeros size overflow"))?;
        let mut values = Vec::new();
        values.try_reserve_exact(len)?;
        values.resize(len, 0);
        Ok(Self {
            shape,
            dtype,
            values,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn n_rows(&self) -> usize {
        self.shape[0]
    }

    pub fn n_cols(&self) -> usize {
        self.shape[1]
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn values(&self) -> &[u8] {
        &self.values
    }

    pub fn into_values(self) -> Vec<u8> {
        self.values
    }

    pub fn as_parts(&self) -> ([usize; 2], DType, &[u8]) {
        (self.shape, self.dtype, &self.values)
    }

    pub fn into_parts(self) -> ([usize; 2], DType, Vec<u8>) {
        (self.shape, self.dtype, self.values)
    }

    /// Select rows/columns with high-speed kernels (returns a new dense array).
    pub fn select(&self, selection: Selection, threads: usize) -> Result<Self> {
        let n_rows = u64::try_from(self.n_rows())
            .map_err(|_| Error::invalid_argument("dense row count exceeds u64"))?;
        let n_cols = u64::try_from(self.n_cols())
            .map_err(|_| Error::invalid_argument("dense col count exceeds u64"))?;
        let normalized = selection.normalize(n_rows, n_cols)?;
        self.select_normalized(&normalized.rows, &normalized.cols, threads)
    }

    pub fn select_normalized(
        &self,
        rows: &NormalizedAxis,
        cols: &NormalizedAxis,
        threads: usize,
    ) -> Result<Self> {
        crate::parallel::validate_threads(threads)?;
        rows.validate(self.n_rows() as u64)?;
        cols.validate(self.n_cols() as u64)?;
        kernel::dense_select(
            &self.values,
            self.n_rows(),
            self.n_cols(),
            self.dtype,
            rows,
            cols,
            threads,
        )
    }
}

/// Owned CSR matrix (`indptr` + packed column indices + values).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrArray {
    shape: [usize; 2],
    index_dtype: DType,
    value_dtype: DType,
    indptr: Vec<u64>,
    indices: Vec<u8>,
    data: Vec<u8>,
}

impl CsrArray {
    pub fn from_parts(
        shape: [usize; 2],
        index_dtype: DType,
        value_dtype: DType,
        indptr: Vec<u64>,
        indices: Vec<u8>,
        data: Vec<u8>,
    ) -> Result<Self> {
        if !index_dtype.is_csr_index() {
            return Err(Error::invalid_argument(format!(
                "CSR index dtype must be u16 or u32, got {index_dtype}"
            )));
        }
        validate_value_dtype(value_dtype)?;
        let expected_indptr_len = shape[0]
            .checked_add(1)
            .ok_or_else(|| Error::invalid_argument("CSR indptr length overflow"))?;
        if indptr.len() != expected_indptr_len {
            return Err(Error::invalid_argument(format!(
                "CSR indptr length {} does not match n_rows+1={}",
                indptr.len(),
                expected_indptr_len
            )));
        }
        if indptr.first().copied() != Some(0) {
            return Err(Error::invalid_argument("CSR indptr[0] must be 0"));
        }
        let nnz = *indptr.last().unwrap_or(&0);
        let index_bytes = checked_mul(nnz, index_dtype.size(), "CSR indices")?;
        let data_bytes = checked_mul(nnz, value_dtype.size(), "CSR data")?;
        if indices.len() != index_bytes {
            return Err(Error::invalid_argument(format!(
                "CSR indices length {} does not match nnz*{index_size}={index_bytes}",
                indices.len(),
                index_size = index_dtype.size()
            )));
        }
        if data.len() != data_bytes {
            return Err(Error::invalid_argument(format!(
                "CSR data length {} does not match nnz*{data_size}={data_bytes}",
                data.len(),
                data_size = value_dtype.size()
            )));
        }
        for window in indptr.windows(2) {
            if window[1] < window[0] {
                return Err(Error::invalid_argument("CSR indptr must be non-decreasing"));
            }
        }
        validate_canonical_indices(&indptr, &indices, index_dtype.size(), shape[1])?;
        Ok(Self {
            shape,
            index_dtype,
            value_dtype,
            indptr,
            indices,
            data,
        })
    }

    pub(crate) fn from_parts_validated(
        shape: [usize; 2],
        index_dtype: DType,
        value_dtype: DType,
        indptr: Vec<u64>,
        indices: Vec<u8>,
        data: Vec<u8>,
    ) -> Self {
        debug_assert!(index_dtype.is_csr_index());
        debug_assert!(value_dtype.is_matrix_value());
        debug_assert_eq!(indptr.len(), shape[0] + 1);
        debug_assert_eq!(indptr.first(), Some(&0));
        Self {
            shape,
            index_dtype,
            value_dtype,
            indptr,
            indices,
            data,
        }
    }

    pub fn empty(shape: [usize; 2], index_dtype: DType, value_dtype: DType) -> Result<Self> {
        if !index_dtype.is_csr_index() {
            return Err(Error::invalid_argument(format!(
                "CSR index dtype must be u16 or u32, got {index_dtype}"
            )));
        }
        validate_value_dtype(value_dtype)?;
        let indptr_len = shape[0]
            .checked_add(1)
            .ok_or_else(|| Error::invalid_argument("CSR indptr length overflow"))?;
        let mut indptr = Vec::new();
        indptr.try_reserve_exact(indptr_len)?;
        indptr.resize(indptr_len, 0);
        Ok(Self {
            shape,
            index_dtype,
            value_dtype,
            indptr,
            indices: Vec::new(),
            data: Vec::new(),
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn n_rows(&self) -> usize {
        self.shape[0]
    }

    pub fn n_cols(&self) -> usize {
        self.shape[1]
    }

    pub fn nnz(&self) -> u64 {
        self.indptr.last().copied().unwrap_or(0)
    }

    pub fn index_dtype(&self) -> DType {
        self.index_dtype
    }

    pub fn value_dtype(&self) -> DType {
        self.value_dtype
    }

    pub fn indptr(&self) -> &[u64] {
        &self.indptr
    }

    pub fn indices(&self) -> &[u8] {
        &self.indices
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn into_parts(self) -> (Vec<u64>, Vec<u8>, Vec<u8>, [usize; 2], DType, DType) {
        (
            self.indptr,
            self.indices,
            self.data,
            self.shape,
            self.index_dtype,
            self.value_dtype,
        )
    }

    /// Select rows/columns. Column selection keeps CSR unless `output` is dense.
    pub fn select(
        &self,
        selection: Selection,
        output: CsrOutput,
        threads: usize,
    ) -> Result<SelectedArray> {
        let n_rows = u64::try_from(self.n_rows())
            .map_err(|_| Error::invalid_argument("CSR row count exceeds u64"))?;
        let n_cols = u64::try_from(self.n_cols())
            .map_err(|_| Error::invalid_argument("CSR col count exceeds u64"))?;
        let normalized = selection.normalize(n_rows, n_cols)?;
        self.select_normalized(&normalized.rows, &normalized.cols, output, threads)
    }

    pub fn select_normalized(
        &self,
        rows: &NormalizedAxis,
        cols: &NormalizedAxis,
        output: CsrOutput,
        threads: usize,
    ) -> Result<SelectedArray> {
        crate::parallel::validate_threads(threads)?;
        rows.validate(self.n_rows() as u64)?;
        cols.validate(self.n_cols() as u64)?;
        let row_selected = kernel::csr_select_rows(
            &self.indptr,
            &self.indices,
            &self.data,
            self.n_rows(),
            self.n_cols(),
            self.index_dtype,
            self.value_dtype,
            rows,
            threads,
        )?;
        row_selected.select_columns(cols, output, threads)
    }

    /// Consume this array and select columns without first cloning all rows.
    pub fn select_columns(
        self,
        cols: &NormalizedAxis,
        output: CsrOutput,
        threads: usize,
    ) -> Result<SelectedArray> {
        self.select_columns_impl(cols, output, threads, None)
    }

    pub(crate) fn select_columns_with_limits(
        self,
        cols: &NormalizedAxis,
        output: CsrOutput,
        threads: usize,
        limits: ReadLimits,
        additional_resident: usize,
    ) -> Result<SelectedArray> {
        self.select_columns_impl(cols, output, threads, Some((limits, additional_resident)))
    }

    fn select_columns_impl(
        self,
        cols: &NormalizedAxis,
        output: CsrOutput,
        threads: usize,
        limits: Option<(ReadLimits, usize)>,
    ) -> Result<SelectedArray> {
        crate::parallel::validate_threads(threads)?;
        cols.validate(self.n_cols() as u64)?;
        if let Some(range) = cols.as_range() {
            if range.start == 0 && range.end == self.n_cols() as u64 {
                return match output {
                    CsrOutput::Sparse => Ok(SelectedArray::Csr(self)),
                    CsrOutput::Dense => {
                        self.check_dense_resident_limit(self.n_cols(), limits)?;
                        Ok(SelectedArray::Dense(self.to_dense(threads)?))
                    }
                };
            }
        }

        match output {
            CsrOutput::Sparse => {
                let filtered = kernel::csr_filter_cols(
                    &self.indptr,
                    &self.indices,
                    &self.data,
                    self.n_rows(),
                    self.n_cols(),
                    self.index_dtype,
                    self.value_dtype,
                    cols,
                    threads,
                    limits,
                )?;
                Ok(SelectedArray::Csr(filtered))
            }
            CsrOutput::Dense => {
                let n_out_cols = usize::try_from(cols.len())
                    .map_err(|_| Error::invalid_argument("selected column count exceeds usize"))?;
                self.check_dense_resident_limit(n_out_cols, limits)?;
                let dense = kernel::csr_to_dense_selected_cols(
                    &self.indptr,
                    &self.indices,
                    &self.data,
                    self.n_rows(),
                    self.n_cols(),
                    self.index_dtype,
                    self.value_dtype,
                    cols,
                    threads,
                )?;
                Ok(SelectedArray::Dense(dense))
            }
        }
    }

    fn check_dense_resident_limit(
        &self,
        n_out_cols: usize,
        limits: Option<(ReadLimits, usize)>,
    ) -> Result<()> {
        let Some((limits, additional_resident)) = limits else {
            return Ok(());
        };
        let indptr_bytes = self
            .indptr
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| Error::invalid_argument("CSR indptr byte length overflow"))?;
        let output_bytes = self
            .n_rows()
            .checked_mul(n_out_cols)
            .and_then(|elements| elements.checked_mul(self.value_dtype.size()))
            .ok_or_else(|| Error::invalid_argument("dense selection byte length overflow"))?;
        limits.check_decoded_sum(
            [
                indptr_bytes,
                self.indices.len(),
                self.data.len(),
                output_bytes,
                additional_resident,
            ],
            "CSR dense selection resident output",
        )?;
        Ok(())
    }

    /// Densify the full matrix into row-major bytes.
    pub fn to_dense(&self, threads: usize) -> Result<DenseArray> {
        kernel::csr_to_dense(
            &self.indptr,
            &self.indices,
            &self.data,
            self.n_rows(),
            self.n_cols(),
            self.index_dtype,
            self.value_dtype,
            threads,
        )
    }
}

/// Result of a selection that may stay sparse or densify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectedArray {
    Dense(DenseArray),
    Csr(CsrArray),
}

impl SelectedArray {
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::Dense(a) => a.shape(),
            Self::Csr(a) => a.shape(),
        }
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    pub fn is_csr(&self) -> bool {
        matches!(self, Self::Csr(_))
    }

    pub fn into_dense(self) -> Result<DenseArray> {
        match self {
            Self::Dense(a) => Ok(a),
            Self::Csr(a) => a.to_dense(1),
        }
    }
}

fn checked_mul(count: u64, size: usize, context: &str) -> Result<usize> {
    let size = u64::try_from(size)
        .map_err(|_| Error::invalid_argument(format!("{context} element size exceeds u64")))?;
    let bytes = count
        .checked_mul(size)
        .ok_or_else(|| Error::invalid_argument(format!("{context} byte length overflow")))?;
    usize::try_from(bytes)
        .map_err(|_| Error::invalid_argument(format!("{context} byte length exceeds usize")))
}

fn validate_value_dtype(dtype: DType) -> Result<()> {
    if !dtype.is_matrix_value() {
        return Err(Error::invalid_argument(format!(
            "matrix value dtype must be u16, u32, i16, i32, f32, or f64, got {dtype}"
        )));
    }
    Ok(())
}

fn validate_canonical_indices(
    indptr: &[u64],
    indices: &[u8],
    index_size: usize,
    n_cols: usize,
) -> Result<()> {
    let n_cols = n_cols as u64;
    for (row, bounds) in indptr.windows(2).enumerate() {
        let start = usize::try_from(bounds[0])
            .map_err(|_| Error::invalid_argument("CSR row start exceeds usize"))?;
        let end = usize::try_from(bounds[1])
            .map_err(|_| Error::invalid_argument("CSR row end exceeds usize"))?;
        let mut previous = None;
        for position in start..end {
            let offset = position
                .checked_mul(index_size)
                .ok_or_else(|| Error::invalid_argument("CSR index offset overflow"))?;
            let index = match index_size {
                2 => u64::from(u16::from_le_bytes([indices[offset], indices[offset + 1]])),
                4 => u64::from(u32::from_le_bytes([
                    indices[offset],
                    indices[offset + 1],
                    indices[offset + 2],
                    indices[offset + 3],
                ])),
                _ => unreachable!("validated CSR index dtype is u16 or u32"),
            };
            if index >= n_cols {
                return Err(Error::invalid_argument(format!(
                    "CSR row {row} index {index} is outside 0..{n_cols}"
                )));
            }
            if previous.is_some_and(|value| value >= index) {
                return Err(Error::invalid_argument(format!(
                    "CSR row {row} indices must be strictly increasing"
                )));
            }
            previous = Some(index);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_indices(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn arrays_reject_non_value_dtype_and_noncanonical_csr() {
        assert!(DenseArray::from_bytes([0, 0], DType::U64, vec![]).is_err());
        assert!(CsrArray::from_parts(
            [1, 3],
            DType::U16,
            DType::F32,
            vec![0, 2],
            u16_indices(&[2, 1]),
            vec![0; 8],
        )
        .is_err());
        assert!(CsrArray::from_parts(
            [1, 3],
            DType::U16,
            DType::F32,
            vec![0, 1],
            u16_indices(&[3]),
            vec![0; 4],
        )
        .is_err());
    }

    #[test]
    fn normalized_selection_and_worker_count_are_validated() {
        let dense = DenseArray::zeros([2, 2], DType::F32).unwrap();
        let invalid_rows = NormalizedAxis::Contiguous { start: 0, end: 3 };
        let cols = NormalizedAxis::Contiguous { start: 0, end: 2 };
        assert!(dense.select_normalized(&invalid_rows, &cols, 1).is_err());
        assert!(dense.select_normalized(&cols, &cols, 0).is_err());
    }

    #[test]
    fn csr_column_selection_limits_the_complete_resident_working_set() {
        let make_csr = || {
            CsrArray::from_parts(
                [1, 3],
                DType::U16,
                DType::F32,
                vec![0, 2],
                u16_indices(&[0, 2]),
                vec![0; 8],
            )
            .unwrap()
        };
        let full = NormalizedAxis::Contiguous { start: 0, end: 3 };
        let limits = ReadLimits::default().maximum_decoded_size(39);
        assert!(make_csr()
            .select_columns_with_limits(&full, CsrOutput::Dense, 1, limits, 0)
            .is_err());

        let gather = NormalizedAxis::Gather {
            positions: vec![2, 0],
        };
        let limits = ReadLimits::default().maximum_decoded_size(64);
        assert!(make_csr()
            .select_columns_with_limits(&gather, CsrOutput::Sparse, 1, limits, 0)
            .is_err());
    }
}
