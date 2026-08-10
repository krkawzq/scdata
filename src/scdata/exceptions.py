"""Public exception hierarchy and private Rust-error translation."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from scdata import _core

__all__ = [
    "AllocationError",
    "CancelledError",
    "ConversionError",
    "DecodeError",
    "InternalError",
    "InvalidConfigError",
    "InvalidDatasetError",
    "InvalidInputError",
    "IoError",
    "PromotionError",
    "ResourceLimitError",
    "ScdataError",
    "SessionError",
    "StalePlanError",
    "UnsupportedError",
    "WorkerPanicError",
]


class ScdataError(Exception):
    """Base class for failures reported by the Rust execution core."""

    kind = "unknown"


class InvalidConfigError(ScdataError):
    kind = "invalid_config"


class InvalidInputError(ScdataError):
    kind = "invalid_input"


class InvalidDatasetError(ScdataError):
    kind = "invalid_dataset"


class ResourceLimitError(ScdataError):
    kind = "resource_limit"


class StalePlanError(ScdataError):
    kind = "stale_plan"


class UnsupportedError(ScdataError):
    kind = "unsupported"


class IoError(ScdataError):
    kind = "io"


class DecodeError(ScdataError):
    kind = "decode"


class PromotionError(ScdataError):
    kind = "promotion"


class ConversionError(ScdataError):
    kind = "conversion"


class CancelledError(ScdataError):
    kind = "cancelled"


class SessionError(ScdataError):
    kind = "session"


class WorkerPanicError(ScdataError):
    kind = "worker_panic"


class AllocationError(ScdataError):
    kind = "allocation"


class InternalError(ScdataError):
    kind = "internal"


_ERROR_TYPES: dict[str, type[ScdataError]] = {
    error_type.kind: error_type
    for error_type in (
        InvalidConfigError,
        InvalidInputError,
        InvalidDatasetError,
        ResourceLimitError,
        StalePlanError,
        UnsupportedError,
        IoError,
        DecodeError,
        PromotionError,
        ConversionError,
        CancelledError,
        SessionError,
        WorkerPanicError,
        AllocationError,
        InternalError,
    )
}

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _call_core(function: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
    try:
        return function(*args, **kwargs)
    except _core.CoreError as error:
        kind = getattr(error, "kind", "internal")
        error_type = _ERROR_TYPES.get(kind, ScdataError)
        translated = error_type(str(error))
        translated.kind = kind
        raise translated from None
