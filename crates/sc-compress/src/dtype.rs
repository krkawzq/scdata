use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// On-disk element type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    U16,
    U32,
    U64,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl DType {
    pub const fn size(self) -> usize {
        match self {
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    pub const fn is_csr_index(self) -> bool {
        matches!(self, Self::U16 | Self::U32)
    }

    pub const fn is_matrix_value(self) -> bool {
        true
    }
}

impl fmt::Display for DType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for DType {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        match name {
            "u16" | "uint16" => Ok(Self::U16),
            "u32" | "uint32" => Ok(Self::U32),
            "u64" | "uint64" => Ok(Self::U64),
            "i16" | "int16" => Ok(Self::I16),
            "i32" | "int32" => Ok(Self::I32),
            "i64" | "int64" => Ok(Self::I64),
            "f32" | "float32" => Ok(Self::F32),
            "f64" | "float64" => Ok(Self::F64),
            other => Err(Error::invalid_argument(format!("unknown dtype `{other}`"))),
        }
    }
}
