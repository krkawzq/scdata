"""Shared call-site checks for the public Python layer.

Validation raises built-in exceptions: ``TypeError`` for the wrong type,
``ValueError`` for an illegal value. Operational failures from the native
core stay on :class:`scdata.exceptions.Error`.
"""

from __future__ import annotations

import math
import operator
import sys
from numbers import Real
from os import PathLike, fspath
from pathlib import Path
from typing import SupportsIndex, cast

import numpy as np

_UINT64_MAX = (1 << 64) - 1
_UINTP_MAX = int(np.iinfo(np.uintp).max)


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
        if parsed > maximum and maximum == _UINTP_MAX:
            raise ValueError(f"{name} exceeds the platform limit {maximum}")
        if parsed > maximum and maximum == _UINT64_MAX:
            raise ValueError(f"{name} exceeds uint64 ({maximum})")
        if parsed < minimum:
            qualifier = "positive" if minimum == 1 else f">= {minimum}"
            raise ValueError(f"{name} must be {qualifier}, got {parsed}")
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


def ensure_path(path: str | PathLike[str]) -> Path:
    try:
        raw_path = fspath(path)
    except TypeError as error:
        raise TypeError(
            f"path must be str or os.PathLike[str], got {type(path).__name__}"
        ) from error
    if not isinstance(raw_path, str):
        raise TypeError("path must resolve to str, not bytes")
    if "\0" in raw_path:
        raise ValueError("path must not contain NUL characters")
    return Path(raw_path)
