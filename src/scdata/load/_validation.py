"""Shared validation for the public Python layer."""

from __future__ import annotations

import math
import operator
import sys
from collections.abc import Iterable
from numbers import Real
from typing import Any, SupportsIndex, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1
_I64_MAX = (1 << 63) - 1

_DTYPE_NAMES = {
    np.dtype(np.int16): "i16",
    np.dtype(np.int32): "i32",
    np.dtype(np.uint16): "u16",
    np.dtype(np.uint32): "u32",
    np.dtype(np.float32): "f32",
    np.dtype(np.float64): "f64",
}
_CORE_OUTPUT_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}
_CORE_STORAGE_DTYPES = {
    **_CORE_OUTPUT_DTYPES,
    "u64": np.dtype(np.uint64),
}


def as_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = sys.maxsize,
) -> int:
    """Return an exact integer, rejecting booleans and lossy coercions."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {parsed}")
    return parsed


def as_float(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite floating-point value") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if positive and parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def normalize_dtype(value: DTypeLike) -> tuple[np.dtype[Any], str]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _CORE_OUTPUT_DTYPES:
            return _CORE_OUTPUT_DTYPES[normalized], normalized
        value = normalized
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"unsupported output dtype {value!r}") from error
    dtype = dtype.newbyteorder("=")
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        supported = ", ".join(item.name for item in _DTYPE_NAMES)
        raise TypeError(f"unsupported output dtype {dtype}; expected one of {supported}")
    return dtype, name


def dtype_from_core(name: str) -> np.dtype[Any]:
    try:
        return _CORE_STORAGE_DTYPES[name]
    except KeyError as error:
        raise RuntimeError(f"Rust core returned unknown dtype {name!r}") from error


def normalize_fill(value: object, dtype: np.dtype[Any], name: str) -> int | float:
    if isinstance(value, np.generic):
        value = value.item()
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return as_int(value, name, minimum=int(info.min), maximum=int(info.max))
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name}={value!r} overflows {dtype.name}") from error
    if not math.isfinite(parsed) and bool(np.isfinite(cast(Any, value))):
        raise ValueError(f"{name}={value!r} overflows {dtype.name}")
    if math.isfinite(parsed) and abs(parsed) > float(np.finfo(dtype).max):
        raise ValueError(f"{name}={parsed!r} overflows {dtype.name}")
    converted = np.asarray(parsed, dtype=dtype).item()
    if math.isfinite(parsed) and not math.isfinite(converted):
        raise ValueError(f"{name}={parsed!r} overflows {dtype.name}")
    return converted


def normalize_feature_map(
    feature_map: Iterable[int | None],
    n_features: int,
) -> NDArray[np.int64]:
    try:
        values = list(feature_map)
    except TypeError as error:
        raise TypeError("feature_map must be a one-dimensional iterable") from error
    if len(values) != n_features:
        raise ValueError(
            f"feature_map has length {len(values)}, but the dataset has {n_features} columns"
        )
    normalized = np.empty(n_features, dtype=np.int64)
    seen: set[int] = set()
    for source_column, value in enumerate(values):
        if value is None:
            normalized[source_column] = -1
            continue
        target = as_int(
            value,
            f"feature_map[{source_column}]",
            minimum=-1,
            maximum=_I64_MAX,
        )
        if target >= 0 and target in seen:
            raise ValueError(f"feature_map contains duplicate output target {target}")
        if target >= 0:
            seen.add(target)
        normalized[source_column] = target
    return np.ascontiguousarray(normalized)


def normalize_rows(
    rows: object,
    *,
    default_source_id: int | None,
) -> tuple[NDArray[np.uint32] | None, NDArray[np.uint64]]:
    if isinstance(rows, np.ndarray):
        return _normalize_row_array(rows, default_source_id)

    source_ids: list[int] | None = None
    row_indices: list[int] = []
    try:
        iterator = iter(cast(Iterable[Any], rows))
    except TypeError as error:
        raise TypeError("rows must be an iterable of row indices, RowRef, or pairs") from error
    for position, item in enumerate(iterator):
        if hasattr(item, "source_id") and hasattr(item, "row"):
            source_id = item.source_id
            row = item.row
            explicit_source = True
        else:
            try:
                operator.index(item)
            except TypeError:
                try:
                    source_id, row = item
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"rows[{position}] must be an integer, RowRef, or a 2-item pair"
                    ) from error
                explicit_source = True
            else:
                if default_source_id is None:
                    raise TypeError(
                        "plain row indices require exactly one source; use (source_id, row) pairs"
                    )
                source_id = default_source_id
                row = item
                explicit_source = False
        normalized_source = as_int(
            source_id,
            f"rows[{position}].source_id",
            maximum=_U32_MAX,
        )
        if explicit_source and source_ids is None:
            if row_indices:
                assert default_source_id is not None
                source_ids = [default_source_id] * len(row_indices)
            else:
                source_ids = []
        if source_ids is not None:
            source_ids.append(normalized_source)
        row_indices.append(as_int(row, f"rows[{position}].row", maximum=_U64_MAX))
    return (
        None
        if source_ids is None and default_source_id is not None
        else np.ascontiguousarray(source_ids or [], dtype=np.uint32),
        np.ascontiguousarray(row_indices, dtype=np.uint64),
    )


def _normalize_row_array(
    rows: NDArray[Any],
    default_source_id: int | None,
) -> tuple[NDArray[np.uint32] | None, NDArray[np.uint64]]:
    if rows.ndim == 1:
        if rows.dtype.kind not in "iu":
            raise TypeError("rows array must have an integer dtype")
        if rows.size == 0:
            return (
                None if default_source_id is not None else np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint64),
            )
        if default_source_id is None:
            raise ValueError(
                "a 1D rows array requires exactly one source; use shape (n, 2) for multiple sources"
            )
        if rows.dtype.kind == "i" and bool(np.any(rows < 0)):
            raise ValueError("rows array must not contain negative values")
        if bool(np.any(rows > _U64_MAX)):
            raise ValueError("row index exceeds uint64")
        return (
            None,
            np.ascontiguousarray(rows, dtype=np.uint64),
        )
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError(
            f"rows array must be one-dimensional or have shape (n, 2), got {rows.shape}"
        )
    if rows.dtype.kind not in "iu":
        raise TypeError("rows array must have an integer dtype")
    if rows.dtype.kind == "i" and rows.size and bool(np.any(rows < 0)):
        raise ValueError("rows array must not contain negative values")
    source_column = rows[:, 0]
    if source_column.size and bool(np.any(source_column > _U32_MAX)):
        raise ValueError("row source id exceeds uint32")
    row_column = rows[:, 1]
    if row_column.size and bool(np.any(row_column > _U64_MAX)):
        raise ValueError("row index exceeds uint64")
    return (
        np.ascontiguousarray(source_column, dtype=np.uint32),
        np.ascontiguousarray(row_column, dtype=np.uint64),
    )
