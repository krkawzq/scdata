//! Output and storage numeric types, plus compile-time promotion rules.

use std::fmt;
use std::str::FromStr;

use crate::{Error, Result};

/// Runtime storage payload type (sc-compress matrix values).
pub use sc_compress::DType as StorageDType;

/// Supported output element types for a session batch ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputDType {
    I16,
    I32,
    I64,
    U16,
    U32,
    U64,
    F32,
    F64,
}

impl OutputDType {
    pub const ALL: [Self; 8] = [
        Self::I16,
        Self::I32,
        Self::I64,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::F32,
        Self::F64,
    ];

    pub const fn size(self) -> usize {
        match self {
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    pub const fn from_storage(dtype: StorageDType) -> Option<Self> {
        match dtype {
            StorageDType::I16 => Some(Self::I16),
            StorageDType::I32 => Some(Self::I32),
            StorageDType::I64 => Some(Self::I64),
            StorageDType::U16 => Some(Self::U16),
            StorageDType::U32 => Some(Self::U32),
            StorageDType::U64 => Some(Self::U64),
            StorageDType::F32 => Some(Self::F32),
            StorageDType::F64 => Some(Self::F64),
        }
    }

    pub const fn to_storage(self) -> StorageDType {
        match self {
            Self::I16 => StorageDType::I16,
            Self::I32 => StorageDType::I32,
            Self::I64 => StorageDType::I64,
            Self::U16 => StorageDType::U16,
            Self::U32 => StorageDType::U32,
            Self::U64 => StorageDType::U64,
            Self::F32 => StorageDType::F32,
            Self::F64 => StorageDType::F64,
        }
    }
}

impl fmt::Display for OutputDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OutputDType {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        match name {
            "i16" | "int16" => Ok(Self::I16),
            "i32" | "int32" => Ok(Self::I32),
            "i64" | "int64" => Ok(Self::I64),
            "u16" | "uint16" => Ok(Self::U16),
            "u32" | "uint32" => Ok(Self::U32),
            "u64" | "uint64" => Ok(Self::U64),
            "f32" | "float32" => Ok(Self::F32),
            "f64" | "float64" => Ok(Self::F64),
            other => Err(Error::InvalidInput(format!(
                "unknown output dtype `{other}`"
            ))),
        }
    }
}

/// Compile-time classification of a storage → output conversion edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromoteKind {
    /// Bit-identical or pure widen with no runtime range risk.
    Lossless,
    /// Signedness / range change that may fail at runtime.
    CheckedSign,
    /// Integer → float with an exact result for every source value.
    ExactToFloat,
    /// Integer → float that may round finite integer values.
    RoundingToFloat,
}

/// Whether `src` may be promoted to `dst` at compile time (width never narrows).
pub fn promote_kind(src: StorageDType, dst: OutputDType) -> Option<PromoteKind> {
    use OutputDType as O;
    use StorageDType as S;

    // Identity
    if matches!(
        (src, dst),
        (S::I16, O::I16)
            | (S::I32, O::I32)
            | (S::I64, O::I64)
            | (S::U16, O::U16)
            | (S::U32, O::U32)
            | (S::U64, O::U64)
            | (S::F32, O::F32)
            | (S::F64, O::F64)
    ) {
        return Some(PromoteKind::Lossless);
    }

    // Integer same-signedness widen and unsigned-to-signed widens that are
    // always in range.
    if matches!(
        (src, dst),
        (S::I16, O::I32 | O::I64)
            | (S::I32, O::I64)
            | (S::U16, O::U32 | O::U64 | O::I32 | O::I64)
            | (S::U32, O::U64 | O::I64)
    ) {
        return Some(PromoteKind::Lossless);
    }

    // Float widen
    if matches!((src, dst), (S::F32, O::F64)) {
        return Some(PromoteKind::Lossless);
    }

    // Every 16-bit integer is exactly representable by f32. Every 16/32-bit
    // integer is exactly representable by f64.
    if matches!(
        (src, dst),
        (S::I16 | S::U16, O::F32) | (S::I16 | S::I32 | S::U16 | S::U32, O::F64)
    ) {
        return Some(PromoteKind::ExactToFloat);
    }

    // f32 has only 24 bits of integer precision; f64 has 53 bits.
    if matches!(
        (src, dst),
        (S::I32 | S::U32, O::F32) | (S::I64 | S::U64, O::F64)
    ) {
        return Some(PromoteKind::RoundingToFloat);
    }

    // Signedness changes needing runtime checks (same width or widen only)
    if matches!(
        (src, dst),
        (S::I16, O::U16 | O::U32 | O::U64)
            | (S::I32, O::U32 | O::U64)
            | (S::I64, O::U64)
            | (S::U16, O::I16)
            | (S::U32, O::I32)
            | (S::U64, O::I64)
    ) {
        return Some(PromoteKind::CheckedSign);
    }

    None
}

mod sealed {
    pub trait Sealed {}
}

/// Numeric types that can be used by typed batch views.
///
/// This trait is sealed because the safe byte-to-value view relies on its
/// implementations having the exact size, alignment, and validity rules of
/// the corresponding primitive numeric dtype.
pub trait OutputValue: sealed::Sealed + Copy + Send + Sync + 'static {
    const DTYPE: OutputDType;
}

macro_rules! impl_output_value {
    ($ty:ty, $dtype:expr) => {
        impl sealed::Sealed for $ty {}

        impl OutputValue for $ty {
            const DTYPE: OutputDType = $dtype;
        }
    };
}

impl_output_value!(i16, OutputDType::I16);
impl_output_value!(i32, OutputDType::I32);
impl_output_value!(i64, OutputDType::I64);
impl_output_value!(u16, OutputDType::U16);
impl_output_value!(u32, OutputDType::U32);
impl_output_value!(u64, OutputDType::U64);
impl_output_value!(f32, OutputDType::F32);
impl_output_value!(f64, OutputDType::F64);
