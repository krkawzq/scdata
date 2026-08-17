//! Output specification: dtype, fill, and overflow handling policy.

use crate::dtype::OutputDType;
use crate::{Error, Result};

/// Default value for output columns that have no mapped source feature.
///
/// CSR structural absence remains zero; it is distinct from an unmapped
/// output column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fill {
    I16(i16),
    I32(i32),
    I64(i64),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
}

impl Fill {
    pub const fn dtype(self) -> OutputDType {
        match self {
            Self::I16(_) => OutputDType::I16,
            Self::I32(_) => OutputDType::I32,
            Self::I64(_) => OutputDType::I64,
            Self::U16(_) => OutputDType::U16,
            Self::U32(_) => OutputDType::U32,
            Self::U64(_) => OutputDType::U64,
            Self::F32(_) => OutputDType::F32,
            Self::F64(_) => OutputDType::F64,
        }
    }

    pub(crate) fn write_le(self, output: &mut [u8]) {
        match self {
            Self::I16(value) => output[..2].copy_from_slice(&value.to_le_bytes()),
            Self::I32(value) => output[..4].copy_from_slice(&value.to_le_bytes()),
            Self::I64(value) => output[..8].copy_from_slice(&value.to_le_bytes()),
            Self::U16(value) => output[..2].copy_from_slice(&value.to_le_bytes()),
            Self::U32(value) => output[..4].copy_from_slice(&value.to_le_bytes()),
            Self::U64(value) => output[..8].copy_from_slice(&value.to_le_bytes()),
            Self::F32(value) => output[..4].copy_from_slice(&value.to_le_bytes()),
            Self::F64(value) => output[..8].copy_from_slice(&value.to_le_bytes()),
        }
    }

    pub(crate) fn encode(self) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        self.write_le(&mut bytes);
        bytes
    }
}

/// Runtime policy when a checked signedness / range conversion fails.
///
/// The fill used for unmapped features is always [`OutputSpec::fill`]. Overflow
/// handling is independent and configurable.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum OverflowPolicy {
    /// Fail the job with [`Error::Conversion`].
    #[default]
    Error,
    /// Write [`OutputSpec::fill`] for the failing element and continue.
    UseFill,
    /// Write a separately specified sentinel (must match the output dtype).
    UseValue(Fill),
    /// Skip range checks and use Rust `as` casts.
    Unchecked,
}

/// Policy for integer-to-float edges that are not exact for every source value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FloatCastPolicy {
    /// Reject a potentially rounding conversion while compiling the plan.
    #[default]
    ExactOnly,
    /// Use the deterministic IEEE-754 rounding performed by Rust's numeric cast.
    AllowRounding,
}

/// Batch output ring specification.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputSpec {
    pub(crate) n_cols: usize,
    pub(crate) dtype: OutputDType,
    /// Value written for every column that has no mapped source feature.
    /// CSR structural absence within a mapped feature remains zero.
    pub(crate) fill: Fill,
    pub(crate) overflow: OverflowPolicy,
    pub(crate) float_cast: FloatCastPolicy,
}

impl OutputSpec {
    pub fn new(n_cols: usize, dtype: OutputDType, fill: Fill) -> Result<Self> {
        if fill.dtype() != dtype {
            return Err(Error::InvalidInput(format!(
                "fill dtype {} does not match output dtype {dtype}",
                fill.dtype()
            )));
        }
        Ok(Self {
            n_cols,
            dtype,
            fill,
            overflow: OverflowPolicy::default(),
            float_cast: FloatCastPolicy::default(),
        })
    }

    pub const fn n_cols(&self) -> usize {
        self.n_cols
    }

    pub const fn dtype(&self) -> OutputDType {
        self.dtype
    }

    pub const fn fill(&self) -> Fill {
        self.fill
    }

    pub fn overflow_policy(&self) -> &OverflowPolicy {
        &self.overflow
    }

    pub const fn float_cast_policy(&self) -> FloatCastPolicy {
        self.float_cast
    }

    pub fn overflow(mut self, overflow: OverflowPolicy) -> Result<Self> {
        if let OverflowPolicy::UseValue(value) = &overflow {
            if value.dtype() != self.dtype {
                return Err(Error::InvalidInput(format!(
                    "overflow sentinel dtype {} does not match output dtype {}",
                    value.dtype(),
                    self.dtype
                )));
            }
        }
        self.overflow = overflow;
        Ok(self)
    }

    #[must_use]
    pub fn float_cast(mut self, policy: FloatCastPolicy) -> Self {
        self.float_cast = policy;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.fill.dtype() != self.dtype {
            return Err(Error::InvalidInput(format!(
                "fill dtype {} does not match output dtype {}",
                self.fill.dtype(),
                self.dtype
            )));
        }
        if let OverflowPolicy::UseValue(value) = &self.overflow {
            if value.dtype() != self.dtype {
                return Err(Error::InvalidInput(format!(
                    "overflow sentinel dtype {} does not match output dtype {}",
                    value.dtype(),
                    self.dtype
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn fallback_bytes(&self) -> [u8; 8] {
        match &self.overflow {
            OverflowPolicy::UseValue(value) => value.encode(),
            OverflowPolicy::Error | OverflowPolicy::UseFill | OverflowPolicy::Unchecked => {
                self.fill.encode()
            }
        }
    }
}
