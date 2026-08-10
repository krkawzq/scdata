//! Typed numeric helpers for writers.

use crate::dtype::DType;
use crate::error::{Error, Result};

mod private {
    pub trait Sealed {}
}

/// Supported dense / CSR payload element type.
pub trait MatrixValue: private::Sealed + Copy + Send + Sync + 'static {
    const DTYPE: DType;

    fn append_le(self, out: &mut Vec<u8>);
}

macro_rules! impl_matrix_value {
    ($ty:ty, $dtype:expr) => {
        impl private::Sealed for $ty {}

        impl MatrixValue for $ty {
            const DTYPE: DType = $dtype;

            #[inline]
            fn append_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}

impl_matrix_value!(u16, DType::U16);
impl_matrix_value!(u32, DType::U32);
impl_matrix_value!(i16, DType::I16);
impl_matrix_value!(i32, DType::I32);
impl_matrix_value!(f32, DType::F32);
impl_matrix_value!(f64, DType::F64);

/// Encode into a reusable buffer, replacing its previous contents.
pub(crate) fn encode_matrix_values_into<V: MatrixValue>(
    values: &[V],
    out: &mut Vec<u8>,
) -> Result<()> {
    let capacity = values
        .len()
        .checked_mul(V::DTYPE.size())
        .ok_or_else(|| Error::invalid_argument("matrix byte length overflow"))?;
    out.clear();
    out.try_reserve_exact(capacity)?;
    for &value in values {
        value.append_le(out);
    }
    Ok(())
}

/// Integer input accepted for CSR `indptr` / column indices.
pub trait IntegerIndex: Copy + Send + Sync + 'static {
    fn try_as_nonneg_u64(self) -> Result<u64>;
}

macro_rules! impl_unsigned_index {
    ($ty:ty) => {
        impl IntegerIndex for $ty {
            #[inline]
            fn try_as_nonneg_u64(self) -> Result<u64> {
                Ok(u64::from(self))
            }
        }
    };
}

impl_unsigned_index!(u8);
impl_unsigned_index!(u16);
impl_unsigned_index!(u32);
impl_unsigned_index!(u64);

macro_rules! impl_signed_index {
    ($ty:ty) => {
        impl IntegerIndex for $ty {
            #[inline]
            fn try_as_nonneg_u64(self) -> Result<u64> {
                u64::try_from(self).map_err(|_| {
                    Error::invalid_argument(format!("negative integer index/offset {self}"))
                })
            }
        }
    };
}

impl_signed_index!(i8);
impl_signed_index!(i16);
impl_signed_index!(i32);
impl_signed_index!(i64);

/// Choose `u16` / `u32` storage from an OR-reduced non-negative index mask.
pub fn csr_index_dtype_from_or_mask(acc: u64) -> Result<DType> {
    if acc <= u64::from(u16::MAX) {
        Ok(DType::U16)
    } else if acc <= u64::from(u32::MAX) {
        Ok(DType::U32)
    } else {
        Err(Error::invalid_argument(
            "csr column index exceeds uint32 range",
        ))
    }
}

/// Promote integer `indptr` values to `u64`, rejecting negatives.
pub fn promote_indptr<P: IntegerIndex>(indptr: &[P]) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    out.try_reserve_exact(indptr.len())?;
    for (position, &value) in indptr.iter().enumerate() {
        out.push(
            value
                .try_as_nonneg_u64()
                .map_err(|error| Error::invalid_argument(format!("indptr[{position}]: {error}")))?,
        );
    }
    Ok(out)
}

/// Promote column indices, OR-reduce for width selection, and check `0..n_cols`.
pub fn promote_csr_indices<I: IntegerIndex>(indices: &[I], n_cols: u64) -> Result<(u64, Vec<u64>)> {
    let mut acc = 0u64;
    let mut out = Vec::new();
    out.try_reserve_exact(indices.len())?;
    for (position, &value) in indices.iter().enumerate() {
        let index = value
            .try_as_nonneg_u64()
            .map_err(|error| Error::invalid_argument(format!("indices[{position}]: {error}")))?;
        if index >= n_cols {
            return Err(Error::invalid_argument(format!(
                "csr index at position {position} is {index}, outside 0..{n_cols}"
            )));
        }
        acc |= index;
        out.push(index);
    }
    Ok((acc, out))
}

/// Encode normalized CSR indices into a reusable little-endian byte buffer.
pub(crate) fn encode_csr_indices_into(
    indices: &[u64],
    dtype: DType,
    out: &mut Vec<u8>,
) -> Result<()> {
    let capacity = indices
        .len()
        .checked_mul(dtype.size())
        .ok_or_else(|| Error::invalid_argument("csr index byte length overflow"))?;
    out.clear();
    out.try_reserve_exact(capacity)?;
    match dtype {
        DType::U16 => {
            for &value in indices {
                let value = u16::try_from(value).map_err(|_| {
                    Error::invalid_argument("csr index does not fit declared u16 dtype")
                })?;
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        DType::U32 => {
            for &value in indices {
                let value = u32::try_from(value).map_err(|_| {
                    Error::invalid_argument("csr index does not fit declared u32 dtype")
                })?;
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        _ => {
            return Err(Error::invalid_argument(format!(
                "csr index dtype must be u16 or u32, got `{dtype}`"
            )));
        }
    }
    Ok(())
}
