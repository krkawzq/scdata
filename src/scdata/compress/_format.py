"""On-disk format constants and NumPy dtype helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "INDEX_DTYPES",
    "STORAGE_INDEX_DTYPES",
    "STORAGE_VALUE_DTYPES",
    "VALUE_DTYPES",
    "dtype_from_storage",
    "is_index_dtype",
    "is_value_dtype",
]

FORMAT_NAME = "sc-compress"
FORMAT_VERSION = 1
STORAGE_VALUE_DTYPES: tuple[str, ...] = (
    "u16",
    "u32",
    "u64",
    "i16",
    "i32",
    "i64",
    "f32",
    "f64",
)
STORAGE_INDEX_DTYPES: tuple[str, ...] = ("u16", "u32")

_DTYPE_BY_NAME = {
    "u16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32),
    "u64": np.dtype(np.uint64),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}

VALUE_DTYPES: frozenset[np.dtype[np.generic]] = frozenset(
    _DTYPE_BY_NAME[name] for name in STORAGE_VALUE_DTYPES
)
INDEX_DTYPES: frozenset[np.dtype[np.generic]] = frozenset(
    _DTYPE_BY_NAME[name] for name in STORAGE_INDEX_DTYPES
)


def dtype_from_storage(name: str) -> np.dtype[np.generic]:
    """Map an on-disk dtype name (for example ``"f32"``) to a NumPy dtype."""
    return _DTYPE_BY_NAME[name]


def _native_dtype(dtype: Any) -> np.dtype[np.generic] | None:
    try:
        resolved = np.dtype(dtype)
    except (TypeError, ValueError):
        return None
    if resolved.byteorder not in ("=", "|"):
        resolved = resolved.newbyteorder("=")
    return resolved


def is_value_dtype(dtype: Any) -> bool:
    """Return whether ``dtype`` is a supported matrix value type.

    Byte order is ignored; writers normalize endianness before entering Rust.
    """
    return _native_dtype(dtype) in VALUE_DTYPES


def is_index_dtype(dtype: Any) -> bool:
    """Return whether ``dtype`` is a supported on-disk index type."""
    return _native_dtype(dtype) in INDEX_DTYPES
