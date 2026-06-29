"""Shared input coercion helpers for :mod:`scdata.data`."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from numpy.typing import NDArray

_INTP_MAX = int(np.iinfo(np.intp).max)
_U64_MAX = int(np.iinfo(np.uint64).max)


def _coerce_index_int(value: Any, context: str) -> int:
    """Coerce one Python index with strict integer semantics."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{context} must be an integer, got bool")
    try:
        parsed = operator.index(value)
    except TypeError as err:
        raise TypeError(f"{context} must be an integer, got {value!r}") from err
    if parsed < 0:
        raise ValueError(f"{context} must be non-negative, got {parsed}")
    if parsed > _INTP_MAX:
        raise ValueError(f"{context} must fit in numpy intp, got {parsed}")
    return int(parsed)


def _coerce_u64_int(value: Any, context: str) -> int:
    """Coerce one non-negative integer that must fit in ``uint64``."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{context} must be an integer, got bool")
    try:
        parsed = operator.index(value)
    except TypeError as err:
        raise TypeError(f"{context} must be an integer, got {value!r}") from err
    if parsed < 0:
        raise ValueError(f"{context} must be non-negative, got {parsed}")
    if parsed > _U64_MAX:
        raise ValueError(f"{context} must fit in uint64, got {parsed}")
    return int(parsed)


def _as_cell_index(value: Any, name: str) -> NDArray[np.intp]:
    """Coerce a 1D cell-index iterable into contiguous ``intp``."""
    arr = _as_strict_integer_array(value, name=name, signed=True, max_value=_INTP_MAX)
    if arr.dtype != np.dtype(np.intp):
        arr = arr.astype(np.intp, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _as_u64_array(value: Any, name: str) -> NDArray[np.uint64]:
    """Coerce a 1D non-negative integer iterable into contiguous ``uint64``."""
    arr = _as_strict_integer_array(value, name=name, signed=False, max_value=_U64_MAX)
    if arr.dtype != np.dtype(np.uint64):
        arr = arr.astype(np.uint64, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _as_gene_names(value: Any, name: str = "gene_names") -> tuple[str, ...]:
    """Normalize gene-name input without splitting a single string."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must contain strings, got bytes")
    try:
        values = tuple(value)
    except TypeError as err:
        raise TypeError(f"{name} must be a string or iterable of strings") from err
    for i, item in enumerate(values):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{i}] must be a string, got {type(item).__name__}")
    return values


def _as_strict_integer_array(
    value: Any,
    *,
    name: str,
    signed: bool,
    max_value: int,
) -> NDArray[np.integer[Any]]:
    """Coerce ``value`` to a 1D integer array without float/string truncation."""
    if isinstance(value, np.ndarray):
        return _coerce_numpy_integer_array(
            value,
            name=name,
            signed=signed,
            max_value=max_value,
        )
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a 1D iterable of integers, got {type(value).__name__}")
    try:
        iterator = iter(value)
    except TypeError as err:
        raise TypeError(f"{name} must be a 1D iterable of integers") from err
    parser = _coerce_index_int if signed else _coerce_u64_int
    parsed = [parser(item, f"{name}[{i}]") for i, item in enumerate(iterator)]
    dtype = np.intp if signed else np.uint64
    return np.asarray(parsed, dtype=dtype)


def _coerce_numpy_integer_array(
    arr: NDArray[Any],
    *,
    name: str,
    signed: bool,
    max_value: int,
) -> NDArray[np.integer[Any]]:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.ndim}D")
    kind = arr.dtype.kind
    if kind == "b":
        raise TypeError(f"{name} must contain integers, got bool")
    if kind == "O":
        parser = _coerce_index_int if signed else _coerce_u64_int
        parsed = [parser(item, f"{name}[{i}]") for i, item in enumerate(arr.tolist())]
        dtype = np.intp if signed else np.uint64
        return np.asarray(parsed, dtype=dtype)
    if kind not in ("i", "u"):
        raise TypeError(f"{name} must contain integers, got dtype {arr.dtype}")
    if arr.size:
        if kind == "i" and int(arr.min()) < 0:
            raise ValueError(f"{name} values must be non-negative")
        max_seen = int(arr.max())
        if max_seen > max_value:
            target = "numpy intp" if signed else "uint64"
            raise ValueError(f"{name} values must fit in {target}, got {max_seen}")
    return arr
