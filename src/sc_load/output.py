"""Output matrix dtype, fill, and conversion policy."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from numpy.typing import DTypeLike

from sc_load._validation import as_int, normalize_dtype, normalize_fill

OverflowPolicy = Literal["error", "use_fill", "use_value", "unchecked"]

__all__ = ["OutputSpec", "OverflowPolicy"]


class OutputSpec:
    """Dense output layout and explicit numeric-conversion behavior."""

    __slots__ = (
        "_allow_float_rounding",
        "_dtype",
        "_dtype_name",
        "_fill",
        "_n_cols",
        "_overflow",
        "_overflow_value",
    )

    def __init__(
        self,
        n_cols: int,
        dtype: DTypeLike,
        *,
        fill: object = 0,
        overflow: OverflowPolicy = "error",
        overflow_value: object | None = None,
        allow_float_rounding: bool = False,
    ) -> None:
        normalized_cols = as_int(n_cols, "n_cols")
        normalized_dtype, dtype_name = normalize_dtype(dtype)
        normalized_fill = normalize_fill(fill, normalized_dtype, "fill")
        if not isinstance(overflow, str):
            raise TypeError("overflow must be a string")
        normalized_overflow = overflow.strip().lower()
        if normalized_overflow not in {"error", "use_fill", "use_value", "unchecked"}:
            raise ValueError("overflow must be 'error', 'use_fill', 'use_value', or 'unchecked'")
        if normalized_overflow == "use_value":
            if overflow_value is None:
                raise ValueError("overflow_value is required when overflow='use_value'")
            normalized_overflow_value = normalize_fill(
                overflow_value, normalized_dtype, "overflow_value"
            )
        else:
            if overflow_value is not None:
                raise ValueError("overflow_value is only valid when overflow='use_value'")
            normalized_overflow_value = None
        if not isinstance(allow_float_rounding, bool):
            raise TypeError("allow_float_rounding must be bool")
        self._n_cols = normalized_cols
        self._dtype = normalized_dtype
        self._dtype_name = dtype_name
        self._fill = normalized_fill
        self._overflow = normalized_overflow
        self._overflow_value = normalized_overflow_value
        self._allow_float_rounding = allow_float_rounding

    @property
    def n_cols(self) -> int:
        return self._n_cols

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def fill(self) -> int | float:
        return self._fill

    @property
    def overflow(self) -> OverflowPolicy:
        return cast(OverflowPolicy, self._overflow)

    @property
    def overflow_value(self) -> int | float | None:
        return self._overflow_value

    @property
    def allow_float_rounding(self) -> bool:
        return self._allow_float_rounding

    @property
    def row_nbytes(self) -> int:
        """Logical byte size of one dense output row."""
        return self._n_cols * self._dtype.itemsize

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary suitable for logging or serialization."""
        return {
            "n_cols": self._n_cols,
            "dtype": self._dtype.name,
            "fill": self._fill,
            "overflow": self._overflow,
            "overflow_value": self._overflow_value,
            "allow_float_rounding": self._allow_float_rounding,
        }

    def _to_core(self) -> dict[str, object]:
        return {
            "n_cols": self._n_cols,
            "dtype": self._dtype_name,
            "fill": self._fill,
            "overflow": self._overflow,
            "overflow_value": self._overflow_value,
            "allow_float_rounding": self._allow_float_rounding,
        }

    def __repr__(self) -> str:
        extra = ""
        if self._overflow_value is not None:
            extra = f", overflow_value={self._overflow_value!r}"
        return (
            f"OutputSpec(n_cols={self._n_cols}, dtype={self._dtype.name!r}, "
            f"fill={self._fill!r}, overflow={self._overflow!r}{extra}, "
            f"allow_float_rounding={self._allow_float_rounding})"
        )
