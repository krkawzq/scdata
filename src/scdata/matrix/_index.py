"""Normalize NumPy/Python indexing keys into Rust axis descriptors."""

from __future__ import annotations

from operator import index
from typing import Any

import numpy as np

from scdata.exceptions import _invalid_argument

__all__ = ["AxisSpec", "normalize_axis", "normalize_key"]


class AxisSpec:
    """Rust-facing axis descriptor: kind + payload."""

    __slots__ = ("kind", "payload", "out_len")

    def __init__(self, kind: str, payload: Any, out_len: int) -> None:
        self.kind = kind
        self.payload = payload
        self.out_len = int(out_len)


def normalize_key(
    key: Any,
    n_rows: int,
    n_cols: int,
) -> tuple[AxisSpec, AxisSpec, bool]:
    """Normalize ``obj[key]`` into ``(rows, cols, drop_row_dim)``.

    Supports:
    - ``obj[i]`` / ``obj[i:j]`` / fancy rows  → columns = all
    - ``obj[rows, cols]`` 2-D indexing
    - bool masks, integer arrays, slices (incl. step), ``:``, ``...``
    """
    if isinstance(key, tuple):
        if len(key) == 0:
            return (
                AxisSpec("all", None, n_rows),
                AxisSpec("all", None, n_cols),
                False,
            )
        if len(key) == 1:
            rows, drop = _normalize_one(key[0], n_rows, name="row")
            return rows, AxisSpec("all", None, n_cols), drop
        if len(key) == 2:
            rows, drop_r = _normalize_one(key[0], n_rows, name="row")
            cols, drop_c = _normalize_one(key[1], n_cols, name="col")
            return rows, cols, drop_r and drop_c
        raise IndexError(f"too many indices for 2-D matrix: {len(key)}")

    rows, drop = _normalize_one(key, n_rows, name="row")
    return rows, AxisSpec("all", None, n_cols), drop


def normalize_axis(key: Any, axis_len: int, *, name: str = "axis") -> AxisSpec:
    spec, _ = _normalize_one(key, axis_len, name=name)
    return spec


def _normalize_one(key: Any, axis_len: int, *, name: str) -> tuple[AxisSpec, bool]:
    """Return ``(spec, is_scalar)``."""
    if key is Ellipsis or key is None:
        return AxisSpec("all", None, axis_len), False

    if isinstance(key, slice):
        start, stop, step = key.indices(axis_len)
        if step == 1:
            return AxisSpec("range", (start, stop), max(0, stop - start)), False
        positions = np.arange(start, stop, step, dtype=np.uint64)
        return AxisSpec("positions", positions, int(positions.size)), False

    if isinstance(key, (bool, np.bool_)):
        raise TypeError(f"{name} index must not be a bare bool")

    # Boolean mask
    if isinstance(key, np.ndarray) and key.dtype == np.bool_:
        if key.ndim != 1:
            _invalid_argument(f"{name} boolean mask must be 1-D, got shape {key.shape}")
        if key.shape[0] != axis_len:
            _invalid_argument(
                f"{name} boolean mask length {key.shape[0]} does not match axis {axis_len}"
            )
        positions = np.flatnonzero(key).astype(np.uint64, copy=False)
        return AxisSpec("positions", np.ascontiguousarray(positions), int(positions.size)), False

    # Integer array / sequence
    if isinstance(key, np.ndarray) or isinstance(key, (list, tuple)):
        arr = np.asarray(key)
        if arr.dtype == np.bool_:
            return _normalize_one(arr, axis_len, name=name)
        if arr.ndim == 0:
            return _normalize_scalar(int(arr), axis_len, name=name)
        if arr.ndim != 1:
            _invalid_argument(f"{name} fancy index must be 1-D, got shape {arr.shape}")
        if arr.size == 0:
            return AxisSpec("positions", np.empty(0, dtype=np.uint64), 0), False
        if arr.dtype.kind not in "iu":
            _invalid_argument(f"{name} fancy index must be integer, got dtype {arr.dtype}")
        positions = arr.astype(np.int64, copy=False)
        # Resolve negatives.
        neg = positions < 0
        if neg.any():
            positions = positions.copy()
            positions[neg] += axis_len
        if (positions < 0).any() or (positions >= axis_len).any():
            bad = int(positions[(positions < 0) | (positions >= axis_len)][0])
            raise IndexError(f"{name} index {bad} out of range for axis of size {axis_len}")
        out = np.ascontiguousarray(positions.astype(np.uint64, copy=False))
        return AxisSpec("positions", out, int(out.size)), False

    # Scalar integer
    try:
        scalar = index(key)
    except TypeError as exc:
        raise TypeError(
            f"{name} index must be int, slice, array, or bool mask, got {type(key).__name__}"
        ) from exc
    return _normalize_scalar(scalar, axis_len, name=name)


def _normalize_scalar(value: int, axis_len: int, *, name: str) -> tuple[AxisSpec, bool]:
    if value < 0:
        value += axis_len
    if value < 0 or value >= axis_len:
        raise IndexError(f"{name} index {value} out of range for axis of size {axis_len}")
    return AxisSpec("range", (value, value + 1), 1), True


def rust_payload(spec: AxisSpec) -> Any:
    """Payload accepted by ``scdata._core`` select APIs."""
    if spec.kind == "all":
        return None
    return spec.payload
