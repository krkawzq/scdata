"""On-disk format constants and NumPy dtype helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from scdata._core import (
    FORMAT_NAME,
    FORMAT_VERSION,
    INDEX_DTYPES as _STORAGE_INDEX_DTYPES,
    VALUE_DTYPES as _STORAGE_VALUE_DTYPES,
)

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

STORAGE_VALUE_DTYPES: tuple[str, ...] = tuple(_STORAGE_VALUE_DTYPES)
STORAGE_INDEX_DTYPES: tuple[str, ...] = tuple(_STORAGE_INDEX_DTYPES)

_DTYPE_BY_NAME = {
    "u16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}

# NumPy dtypes accepted for matrix values / CSR indices.
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
