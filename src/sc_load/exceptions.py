"""Public exception hierarchy and private Rust-error translation."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from sc_load import _core

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
    "ScLoadError",
    "SessionError",
    "StalePlanError",
    "UnsupportedError",
    "WorkerPanicError",
]


class ScLoadError(Exception):
    """Base class for failures reported by the Rust execution core."""

    kind = "unknown"


class InvalidConfigError(ScLoadError):
    kind = "invalid_config"


class InvalidInputError(ScLoadError):
    kind = "invalid_input"


class InvalidDatasetError(ScLoadError):
    kind = "invalid_dataset"


class ResourceLimitError(ScLoadError):
    kind = "resource_limit"


class StalePlanError(ScLoadError):
    kind = "stale_plan"


class UnsupportedError(ScLoadError):
    kind = "unsupported"


class IoError(ScLoadError):
    kind = "io"


class DecodeError(ScLoadError):
    kind = "decode"


class PromotionError(ScLoadError):
    kind = "promotion"


class ConversionError(ScLoadError):
    kind = "conversion"


class CancelledError(ScLoadError):
    kind = "cancelled"


class SessionError(ScLoadError):
    kind = "session"


class WorkerPanicError(ScLoadError):
    kind = "worker_panic"


class AllocationError(ScLoadError):
    kind = "allocation"


class InternalError(ScLoadError):
    kind = "internal"


_ERROR_TYPES: dict[str, type[ScLoadError]] = {
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
        error_type = _ERROR_TYPES.get(kind, ScLoadError)
        translated = error_type(str(error))
        translated.kind = kind
        raise translated from None
