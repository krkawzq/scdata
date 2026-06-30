use crate::databank::error::{DataBankError, DataBankResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    pub fn item_size(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    pub fn is_csr_index(self) -> bool {
        matches!(self, Self::I32 | Self::U32 | Self::I64 | Self::U64)
    }

    /// Integer dtypes (signed and unsigned, any width).
    pub fn is_int(self) -> bool {
        matches!(
            self,
            Self::U8
                | Self::I8
                | Self::U16
                | Self::I16
                | Self::U32
                | Self::I32
                | Self::U64
                | Self::I64
        )
    }

    /// Floating-point dtypes, including the two 16-bit formats.
    pub fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    /// 16-bit floating formats (f16 and bfloat16).
    pub fn is_half(self) -> bool {
        matches!(self, Self::F16 | Self::BF16)
    }

    /// Whether values of `self` may be numerically cast to `dst`.
    ///
    /// Cast policy: every direction is permitted **except** float→int.
    /// That single exclusion keeps the rule trivial (no rank table) while
    /// covering the realistic single-cell cases:
    ///
    /// * float→float (including precision loss: `f64→f32`, `f32→f16`,
    ///   `f16↔bf16`) — round-to-nearest-even via the `half` crate, matching
    ///   numpy `astype`.
    /// * int→float (any width, including `f16`/`bf16`) — numeric conversion.
    /// * int→int (including narrowing like `u32→u8`) — Rust `as` truncation.
    /// * float→int — **rejected** (`CannotCast`).
    pub fn can_cast_to(self, dst: DType) -> bool {
        !(self.is_float() && dst.is_int())
    }
}

pub trait DataValue: sealed::Sealed + Copy + Send + Sync + 'static {
    const DTYPE: DType;
    fn zero() -> Self;

    /// Numerically cast `src_bytes` (laid out as `src_dtype`, native-endian)
    /// into `dst`.
    ///
    /// `dst.len()` must equal `src_bytes.len() / src_dtype.item_size()`.  When
    /// `src_dtype == Self::DTYPE` this is a zero-overhead byte copy (the same
    /// `copy_from_slice` the scatter path already used); otherwise each source
    /// element is read, converted to `Self`, and written.  Float→int is
    /// rejected here as well as at the dtype gate, so a stray call cannot
    /// silently truncate.
    fn cast_slice_from(src_bytes: &[u8], src_dtype: DType, dst: &mut [Self]) -> DataBankResult<()>;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! impl_data_value {
    ($ty:ty, $dtype:expr, $zero:expr) => {
        impl sealed::Sealed for $ty {}

        impl DataValue for $ty {
            const DTYPE: DType = $dtype;

            fn zero() -> Self {
                $zero
            }

            fn cast_slice_from(
                src_bytes: &[u8],
                src_dtype: DType,
                dst: &mut [Self],
            ) -> DataBankResult<()> {
                cast_slice_into_native::<$ty>(src_bytes, src_dtype, dst)
            }
        }
    };
}

impl_data_value!(u8, DType::U8, 0);
impl_data_value!(i8, DType::I8, 0);
impl_data_value!(u16, DType::U16, 0);
impl_data_value!(i16, DType::I16, 0);
impl_data_value!(u32, DType::U32, 0);
impl_data_value!(i32, DType::I32, 0);
impl_data_value!(u64, DType::U64, 0);
impl_data_value!(i64, DType::I64, 0);
impl_data_value!(f32, DType::F32, 0.0);
impl_data_value!(f64, DType::F64, 0.0);

/// Opaque native-endian half-precision payload.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct F16Bits(pub u16);

impl sealed::Sealed for F16Bits {}

impl DataValue for F16Bits {
    const DTYPE: DType = DType::F16;

    fn zero() -> Self {
        Self(0)
    }

    fn cast_slice_from(src_bytes: &[u8], src_dtype: DType, dst: &mut [Self]) -> DataBankResult<()> {
        cast_slice_into_half(src_bytes, src_dtype, dst, HalfKind::F16)
    }
}

/// Opaque native-endian bfloat16 payload.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Bf16Bits(pub u16);

impl sealed::Sealed for Bf16Bits {}

impl DataValue for Bf16Bits {
    const DTYPE: DType = DType::BF16;

    fn zero() -> Self {
        Self(0)
    }

    fn cast_slice_from(src_bytes: &[u8], src_dtype: DType, dst: &mut [Self]) -> DataBankResult<()> {
        cast_slice_into_half(src_bytes, src_dtype, dst, HalfKind::Bf16)
    }
}

// ---------------------------------------------------------------------------
// Cast engine
// ---------------------------------------------------------------------------

/// Read one native-endian value of a primitive type from `bytes`.
///
/// `DataValue` types are either plain numeric primitives or `#[repr(transparent)]`
/// newtypes over `u16`, so a single `read_value` per source dtype feeds every
/// `cast_slice_from` arm.  Returns `None` for `F16`/`BF16` — those go through
/// `half` in the float arms below, not through this integer-only helper.
#[inline]
fn read_int_value(bytes: &[u8], dtype: DType) -> DataBankResult<i128> {
    Ok(match dtype {
        DType::U8 => i128::from(u8::from_ne_bytes(bytes[..1].try_into().unwrap())),
        DType::I8 => i128::from(i8::from_ne_bytes(bytes[..1].try_into().unwrap())),
        DType::U16 => i128::from(u16::from_ne_bytes(bytes[..2].try_into().unwrap())),
        DType::I16 => i128::from(i16::from_ne_bytes(bytes[..2].try_into().unwrap())),
        DType::U32 => i128::from(u32::from_ne_bytes(bytes[..4].try_into().unwrap())),
        DType::I32 => i128::from(i32::from_ne_bytes(bytes[..4].try_into().unwrap())),
        DType::U64 => i128::from(u64::from_ne_bytes(bytes[..8].try_into().unwrap())),
        DType::I64 => i128::from(i64::from_ne_bytes(bytes[..8].try_into().unwrap())),
        // Float sources are handled by the float arms directly, not here.
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {
            return Err(DataBankError::CannotCast {
                src: dtype,
                dst: DType::U8,
                reason: "read_int_value called with float source",
            })
        }
    })
}

/// Read one native-endian float value as `f64`, regardless of source float dtype.
#[inline]
fn read_float_value(bytes: &[u8], dtype: DType) -> DataBankResult<f64> {
    Ok(match dtype {
        DType::F32 => f64::from(f32::from_ne_bytes(bytes[..4].try_into().unwrap())),
        DType::F64 => f64::from_ne_bytes(bytes[..8].try_into().unwrap()),
        DType::F16 => {
            let bits = u16::from_ne_bytes(bytes[..2].try_into().unwrap());
            half::f16::from_bits(bits).to_f64()
        }
        DType::BF16 => {
            let bits = u16::from_ne_bytes(bytes[..2].try_into().unwrap());
            half::bf16::from_bits(bits).to_f64()
        }
        DType::U8
        | DType::I8
        | DType::U16
        | DType::I16
        | DType::U32
        | DType::I32
        | DType::U64
        | DType::I64 => {
            return Err(DataBankError::CannotCast {
                src: dtype,
                dst: DType::F64,
                reason: "read_float_value called with int source",
            })
        }
    })
}

/// Which 16-bit float format `cast_slice_into_half` is producing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HalfKind {
    F16,
    Bf16,
}

impl HalfKind {
    #[inline]
    fn dtype(self) -> DType {
        match self {
            HalfKind::F16 => DType::F16,
            HalfKind::Bf16 => DType::BF16,
        }
    }

    /// Round an `f64` down to this half format's bit pattern.
    #[inline]
    fn round_f64_to_bits(self, v: f64) -> u16 {
        match self {
            HalfKind::F16 => half::f16::from_f64(v).to_bits(),
            HalfKind::Bf16 => half::bf16::from_f64(v).to_bits(),
        }
    }
}

/// Cast `src_bytes` (laid out as `src_dtype`) into a half-precision `dst`.
///
/// Mirrors `cast_slice_into_native` but writes `F16Bits`/`Bf16Bits` via the
/// `half` crate.  The identical-dtype fast path is a raw byte copy (both
/// newtypes are `#[repr(transparent)]` over `u16`).
#[inline]
fn cast_slice_into_half<T: HalfBits>(
    src_bytes: &[u8],
    src_dtype: DType,
    dst: &mut [T],
    kind: HalfKind,
) -> DataBankResult<()> {
    let dst_dtype = kind.dtype();
    let src_size = src_dtype.item_size();
    if src_size == 0 {
        return Err(DataBankError::InvalidArrayMeta(
            "cast source dtype has zero item size".to_string(),
        ));
    }
    if src_bytes.len() % src_size != 0 {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "cast source byte length {} is not a multiple of item size {src_size}",
            src_bytes.len()
        )));
    }
    let count = src_bytes.len() / src_size;
    if dst.len() != count {
        return Err(DataBankError::BufferSizeMismatch {
            expected: count,
            actual: dst.len(),
        });
    }

    // Fast path: identical dtype → raw byte copy.
    if src_dtype == dst_dtype {
        let expected = count.checked_mul(dst_dtype.item_size()).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("cast byte length overflow".to_string())
        })?;
        // SAFETY: the half newtypes are `#[repr(transparent)]` over `u16`, so
        // their byte layout is the native-endian `u16` pattern; reinterpreting
        // the slice is sound.  Buffers are caller-owned and do not alias.
        let dst_bytes =
            unsafe { std::slice::from_raw_parts_mut(T::as_mut_byte_ptr(dst), expected) };
        dst_bytes.copy_from_slice(src_bytes);
        return Ok(());
    }

    // dst is a half float, so float→int cannot occur; an int source goes
    // through the int→float path, a float source through float→float.
    for i in 0..count {
        let elem = &src_bytes[i * src_size..(i + 1) * src_size];
        let bits = if src_dtype.is_int() {
            let v = read_int_value(elem, src_dtype)?;
            kind.round_f64_to_bits(v as f64)
        } else {
            let v = read_float_value(elem, src_dtype)?;
            kind.round_f64_to_bits(v)
        };
        dst[i].set_bits(bits);
    }
    Ok(())
}

/// Abstraction over `F16Bits`/`Bf16Bits` so `cast_slice_into_half` can write
/// either without a second monomorphisation dimension.  Only the element type
/// implements it; callers pass `&mut [F16Bits]` / `&mut [Bf16Bits]`.
pub trait HalfBits: Sized {
    fn as_mut_byte_ptr(slice: &mut [Self]) -> *mut u8;
    fn set_bits(&mut self, bits: u16);
}

impl HalfBits for F16Bits {
    #[inline]
    fn as_mut_byte_ptr(slice: &mut [Self]) -> *mut u8 {
        slice.as_mut_ptr() as *mut u8
    }
    #[inline]
    fn set_bits(&mut self, bits: u16) {
        self.0 = bits;
    }
}

impl HalfBits for Bf16Bits {
    #[inline]
    fn as_mut_byte_ptr(slice: &mut [Self]) -> *mut u8 {
        slice.as_mut_ptr() as *mut u8
    }
    #[inline]
    fn set_bits(&mut self, bits: u16) {
        self.0 = bits;
    }
}

#[inline]
fn cast_slice_into_native<T>(
    src_bytes: &[u8],
    src_dtype: DType,
    dst: &mut [T],
) -> DataBankResult<()>
where
    T: NumCastTarget,
{
    let src_size = src_dtype.item_size();
    if src_size == 0 {
        return Err(DataBankError::InvalidArrayMeta(
            "cast source dtype has zero item size".to_string(),
        ));
    }
    if src_bytes.len() % src_size != 0 {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "cast source byte length {} is not a multiple of item size {src_size}",
            src_bytes.len()
        )));
    }
    let count = src_bytes.len() / src_size;
    if dst.len() != count {
        return Err(DataBankError::BufferSizeMismatch {
            expected: count,
            actual: dst.len(),
        });
    }

    // Fast path: identical dtype → raw byte copy, zero overhead, exactly the
    // old `copy_ne_bytes_to_values` behaviour.
    if src_dtype == T::DTYPE {
        let expected = count.checked_mul(T::DTYPE.item_size()).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("cast byte length overflow".to_string())
        })?;
        // SAFETY: `T` is one of the native numeric primitives; its bit layout
        // is the native-endian byte pattern, so reinterpreting the slice is
        // sound.  `src_bytes` and `dst` do not alias (caller-owned buffers).
        let dst_bytes =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, expected) };
        dst_bytes.copy_from_slice(src_bytes);
        return Ok(());
    }

    // Float→int is rejected by policy even if reached here.
    if src_dtype.is_float() && T::DTYPE.is_int() {
        return Err(DataBankError::CannotCast {
            src: src_dtype,
            dst: T::DTYPE,
            reason: "float-to-int cast is not permitted",
        });
    }

    for i in 0..count {
        let elem = &src_bytes[i * src_size..(i + 1) * src_size];
        if src_dtype.is_int() {
            let v = read_int_value(elem, src_dtype)?;
            dst[i] = T::from_i128(v);
        } else {
            let v = read_float_value(elem, src_dtype)?;
            dst[i] = T::from_f64(v);
        }
    }
    Ok(())
}

/// Helper trait so `cast_slice_into_native` can write into a generic `&mut [T]`
/// without monomorphising 12×12 conversion arms by hand.  Implemented once per
/// native numeric type; the two half-precision newtypes have bespoke impls.
pub trait NumCastTarget: Sized + Copy + Send + Sync + 'static {
    const DTYPE: DType;
    fn from_i128(v: i128) -> Self;
    fn from_f64(v: f64) -> Self;
}

macro_rules! impl_num_cast_target {
    ($ty:ty, $dtype:expr) => {
        impl NumCastTarget for $ty {
            const DTYPE: DType = $dtype;
            #[inline]
            fn from_i128(v: i128) -> Self {
                v as $ty
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                v as $ty
            }
        }
    };
}

impl_num_cast_target!(u8, DType::U8);
impl_num_cast_target!(i8, DType::I8);
impl_num_cast_target!(u16, DType::U16);
impl_num_cast_target!(i16, DType::I16);
impl_num_cast_target!(u32, DType::U32);
impl_num_cast_target!(i32, DType::I32);
impl_num_cast_target!(u64, DType::U64);
impl_num_cast_target!(i64, DType::I64);
impl_num_cast_target!(f32, DType::F32);
impl_num_cast_target!(f64, DType::F64);
