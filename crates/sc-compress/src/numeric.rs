//! Typed numeric helpers for writers.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::kernel::write_index_unchecked;

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
impl_matrix_value!(u64, DType::U64);
impl_matrix_value!(i16, DType::I16);
impl_matrix_value!(i32, DType::I32);
impl_matrix_value!(i64, DType::I64);
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
    #[cfg(target_endian = "little")]
    {
        debug_assert_eq!(std::mem::size_of::<V>(), V::DTYPE.size());
        // SAFETY: MatrixValue is sealed to plain primitive numerics, so their
        // initialized object representation is a valid byte slice. On a
        // little-endian target that representation is already the wire format.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), capacity) };
        out.extend_from_slice(bytes);
    }
    #[cfg(target_endian = "big")]
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
    let (index_size, maximum) = match dtype {
        DType::U16 => (DType::U16.size(), u64::from(u16::MAX)),
        DType::U32 => (DType::U32.size(), u64::from(u32::MAX)),
        _ => {
            return Err(Error::invalid_argument(format!(
                "csr index dtype must be u16 or u32, got `{dtype}`"
            )));
        }
    };
    if indices.iter().any(|&value| value > maximum) {
        return Err(Error::invalid_argument(format!(
            "csr index does not fit declared {dtype} dtype"
        )));
    }
    let capacity = indices
        .len()
        .checked_mul(index_size)
        .ok_or_else(|| Error::invalid_argument("csr index byte length overflow"))?;
    out.clear();
    out.try_reserve_exact(capacity)?;
    let destination = out.as_mut_ptr();
    for (position, &value) in indices.iter().enumerate() {
        // SAFETY: the pre-scan proved `value` fits `index_size`; the checked
        // capacity calculation and exact reservation provide one complete,
        // disjoint output slot for every `position`.
        unsafe {
            write_index_unchecked(destination, position, index_size, value);
        }
    }
    // SAFETY: the loop initialized every one of the `capacity` reserved bytes;
    // no safe reference to the spare allocation existed while it was written.
    unsafe { out.set_len(capacity) };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_index_encoding_writes_exact_little_endian_widths() {
        let mut output = Vec::new();
        encode_csr_indices_into(&[0, 1, u16::MAX as u64], DType::U16, &mut output).unwrap();
        assert_eq!(
            output,
            [0u16, 1, u16::MAX]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>()
        );

        encode_csr_indices_into(
            &[0, u16::MAX as u64 + 1, u32::MAX as u64],
            DType::U32,
            &mut output,
        )
        .unwrap();
        assert_eq!(
            output,
            [0u32, u16::MAX as u32 + 1, u32::MAX]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn csr_index_encoding_rejects_values_wider_than_declared_dtype() {
        let mut output = Vec::new();
        assert!(encode_csr_indices_into(&[u16::MAX as u64 + 1], DType::U16, &mut output).is_err());
        assert!(encode_csr_indices_into(&[u32::MAX as u64 + 1], DType::U32, &mut output).is_err());
    }
}

#[cfg(test)]
mod benchmarks {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use super::{encode_matrix_values_into, MatrixValue};

    fn encode_generic<V: MatrixValue>(values: &[V], out: &mut Vec<u8>) {
        out.clear();
        out.reserve(values.len() * V::DTYPE.size());
        for &value in values {
            value.append_le(out);
        }
    }

    fn best_of(mut run: impl FnMut(), rounds: usize) -> Duration {
        (0..rounds)
            .map(|_| {
                let started = Instant::now();
                run();
                started.elapsed()
            })
            .min()
            .unwrap()
    }

    #[test]
    #[ignore = "manual release-mode 64-bit encoding benchmark"]
    fn benchmark_64_bit_bulk_encoding() {
        let count = 256 * 1024;
        let iterations = 256;
        let values = (0..count)
            .map(|index| {
                (index as i64)
                    .wrapping_mul(1_000_000_007)
                    .wrapping_sub(1i64 << 54)
            })
            .collect::<Vec<_>>();
        let mut specialized = Vec::new();
        let mut generic = Vec::new();
        encode_matrix_values_into(&values, &mut specialized).unwrap();
        encode_generic(&values, &mut generic);
        assert_eq!(specialized, generic);

        let specialized_time = best_of(
            || {
                for _ in 0..iterations {
                    encode_matrix_values_into(black_box(&values), black_box(&mut specialized))
                        .unwrap();
                }
            },
            5,
        );
        let generic_time = best_of(
            || {
                for _ in 0..iterations {
                    encode_generic(black_box(&values), black_box(&mut generic));
                }
            },
            5,
        );
        let gib = count as f64 * 8.0 * iterations as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "64-bit encoding: bulk={:.2} GiB/s generic={:.2} GiB/s speedup={:.2}x",
            gib / specialized_time.as_secs_f64(),
            gib / generic_time.as_secs_f64(),
            generic_time.as_secs_f64() / specialized_time.as_secs_f64(),
        );
    }
}
