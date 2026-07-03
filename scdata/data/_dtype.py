"""DType normalization for scdata's Python and Rust-facing APIs."""

from __future__ import annotations

import sys
from enum import Enum, EnumMeta
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import DTypeLike as NumpyDTypeLike

__all__ = [
    "DataError",
    "DtypeParseError",
    "DType",
    "DTypeLike",
    "normalize_dtype",
]


class DataError(ValueError):
    """Base for metadata parsing errors raised by :mod:`scdata.data`.

    The io layer catches :class:`DataError` and re-raises it as
    :class:`~scdata.io.StoreError` so :func:`~scdata.io.launch` exposes a single
    error type to callers.
    """


class DtypeParseError(DataError):
    """Raised when a dtype-like value cannot be mapped to a scdata type."""


class _DTypeMeta(EnumMeta):
    def __call__(cls, value: object, names: object | None = None, *args: object, **kwargs: Any):
        if names is None and not args and not kwargs:
            if isinstance(value, np.ndarray):
                return cls._missing_(value)
            return super().__call__(value)
        return super().__call__(value, names, *args, **kwargs)


class DType(Enum, metaclass=_DTypeMeta):
    """Element dtype, mirrors the Rust ``DType`` enum (``databank/array.rs``).

    ``DType(value)`` is the canonical public normalization path.  Besides the
    enum values (``"f32"``) it accepts enum names (``"F32"``), NumPy dtype-like
    values (``np.float32``, ``np.dtype("float32")``), NumPy arrays, zarr v2
    dtype strings (``"<f4"``, ``"|u1"``), and zarr v3 names
    (``"float32"``, ``"uint64"``, ``"bfloat16"``).
    """

    U8 = "u8"
    I8 = "i8"
    U16 = "u16"
    I16 = "i16"
    U32 = "u32"
    I32 = "i32"
    U64 = "u64"
    I64 = "i64"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"

    @classmethod
    def _missing_(cls, value: object) -> "DType":
        return _coerce_dtype(value)

    @property
    def item_size(self) -> int:
        match self:
            case DType.U8 | DType.I8:
                return 1
            case DType.U16 | DType.I16 | DType.F16 | DType.BF16:
                return 2
            case DType.U32 | DType.I32 | DType.F32:
                return 4
            case DType.U64 | DType.I64 | DType.F64:
                return 8

    @property
    def is_csr_index(self) -> bool:
        """Whether this dtype is valid for CSR ``indices`` (Rust ``is_csr_index``)."""
        return self in (DType.I32, DType.U32, DType.I64, DType.U64)

    @classmethod
    def parse(cls, dtype: object) -> "DType":
        """Parse a NumPy/zarr dtype field into a :class:`DType`.

        Kept for compatibility.  New code should prefer ``DType(dtype)`` or
        :func:`normalize_dtype`.
        """
        return cls(dtype)

    @classmethod
    def from_numpy(cls, dtype: Any) -> "DType":
        """Map a numpy dtype-like object to a :class:`DType`.

        Kept for compatibility.  New code should prefer ``DType(dtype)``.
        """
        return cls(dtype)


DTypeLike: TypeAlias = DType | str | NumpyDTypeLike | np.ndarray[Any, Any]


def normalize_dtype(dtype: DTypeLike | None, *, allow_none: bool = False) -> DType | None:
    """Normalize a public dtype-like value to :class:`DType`.

    ``allow_none=True`` is useful for optional output dtype parameters where
    ``None`` means "use the stored dataset dtype".
    """
    if dtype is None:
        if allow_none:
            return None
        raise DtypeParseError("dtype is None")
    return DType(dtype)


def _coerce_dtype(dtype: object) -> DType:
    if isinstance(dtype, DType):
        return dtype
    if dtype is None:
        raise DtypeParseError("dtype is None")
    if isinstance(dtype, str):
        return _from_dtype_string(dtype)
    if isinstance(dtype, list):
        return _decode_base_dtype(_extract_base_dtype(dtype))
    if isinstance(dtype, np.ndarray):
        return _from_numpy_dtype(dtype.dtype)
    try:
        return _from_numpy_dtype(np.dtype(dtype))
    except DtypeParseError:
        raise
    except (TypeError, ValueError) as err:
        raise DtypeParseError(f"unsupported dtype-like value: {dtype!r}") from err


def _from_dtype_string(value: str) -> DType:
    text = value.strip()
    if not text:
        raise DtypeParseError("empty dtype string")

    folded = text.lower()
    if folded in _NAME_DTYPE_MAP:
        return _NAME_DTYPE_MAP[folded]
    if folded in _V3_DTYPE_NAME_MAP:
        return _V3_DTYPE_NAME_MAP[folded]
    if _looks_like_zarr_base_dtype(text):
        return _decode_base_dtype(text)

    try:
        return _from_numpy_dtype(np.dtype(text))
    except DtypeParseError:
        raise
    except (TypeError, ValueError) as err:
        raise DtypeParseError(f"unsupported dtype {value!r}") from err


def _extract_base_dtype(dtype: object) -> str:
    """Return the base type string from a zarr dtype field."""
    if isinstance(dtype, str):
        return dtype
    # zarr represents structured arrays as [base_str, [(field, dtype), ...]].
    if isinstance(dtype, list) and dtype:
        first = dtype[0]
        if isinstance(first, str):
            return first
    text = str(dtype).strip()
    if not text:
        raise DtypeParseError(f"empty dtype descriptor: {dtype!r}")
    return text


def _looks_like_zarr_base_dtype(text: str) -> bool:
    if text[0] in "<>|=":
        return True
    return text.lower() in _BASE_DTYPE_MAP


def _decode_base_dtype(base: str) -> DType:
    text = base.strip()
    if not text:
        raise DtypeParseError("empty base dtype string")

    # Strip endianness prefix.  scdata stores are little-endian on disk; we
    # accept '<' (little), '=' (native on our write targets), and '|' (not
    # applicable / byte).  Big-endian input is rejected because Rust decodes
    # bytes exactly as written.
    if text[0] in "<>=":
        prefix = text[0]
        body = text[1:]
        if prefix == ">":
            raise DtypeParseError(
                f"big-endian dtype {base!r} is unsupported (scdata stores are little-endian)"
            )
    elif text[0] == "|":
        body = text[1:]
    else:
        body = text

    body = body.strip().lower()
    if not body:
        raise DtypeParseError(f"missing type code in {base!r}")

    if body not in _BASE_DTYPE_MAP:
        raise DtypeParseError(f"unsupported dtype {base!r} (body {body!r})")
    return _BASE_DTYPE_MAP[body]


def _from_numpy_dtype(dtype: np.dtype[Any]) -> DType:
    np_dtype = np.dtype(dtype)
    if np_dtype.byteorder == ">" or (np_dtype.byteorder == "=" and sys.byteorder == "big"):
        raise DtypeParseError(
            f"big-endian dtype {np_dtype!r} is unsupported (scdata stores are little-endian)"
        )

    kind = np_dtype.kind
    size = np_dtype.itemsize
    match (kind, size):
        case ("u", 1):
            return DType.U8
        case ("i", 1):
            return DType.I8
        case ("u", 2):
            return DType.U16
        case ("i", 2):
            return DType.I16
        case ("u", 4):
            return DType.U32
        case ("i", 4):
            return DType.I32
        case ("u", 8):
            return DType.U64
        case ("i", 8):
            return DType.I64
        case ("f", 2):
            # numpy has no native bf16; a 2-byte float is f16 here.
            return DType.F16
        case ("f", 4):
            return DType.F32
        case ("f", 8):
            return DType.F64
        case _:
            raise DtypeParseError(f"unsupported numpy dtype: {np_dtype}")


_NAME_DTYPE_MAP: dict[str, DType] = {}
for _dtype in DType:
    _NAME_DTYPE_MAP[_dtype.value] = _dtype
    _NAME_DTYPE_MAP[_dtype.name.lower()] = _dtype

_BASE_DTYPE_MAP: dict[str, DType] = {
    "u1": DType.U8,
    "i1": DType.I8,
    "u2": DType.U16,
    "i2": DType.I16,
    "u4": DType.U32,
    "i4": DType.I32,
    "u8": DType.U64,
    "i8": DType.I64,
    "f2": DType.F16,
    "f4": DType.F32,
    "f8": DType.F64,
    # numcodecs/bfloat extensions used by some single-cell stores.
    "bf2": DType.BF16,
}

_V3_DTYPE_NAME_MAP: dict[str, DType] = {
    "int8": DType.I8,
    "uint8": DType.U8,
    "int16": DType.I16,
    "uint16": DType.U16,
    "int32": DType.I32,
    "uint32": DType.U32,
    "int64": DType.I64,
    "uint64": DType.U64,
    "float16": DType.F16,
    "float32": DType.F32,
    "float64": DType.F64,
    "bfloat16": DType.BF16,
}

del _dtype
