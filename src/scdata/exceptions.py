"""Unified exception hierarchy for ``scdata``."""

from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn, ParamSpec, TypeVar

from scdata import _core

__all__ = [
    "AllocationError",
    "CancelledError",
    "CodecError",
    "ConversionError",
    "CorruptDataError",
    "DecodeError",
    "Error",
    "InternalError",
    "InvalidArgumentError",
    "InvalidConfigError",
    "InvalidDatasetError",
    "InvalidInputError",
    "InvalidMetaError",
    "IoError",
    "JsonError",
    "NotFoundError",
    "PathError",
    "PerformanceWarning",
    "PromotionError",
    "ResourceLimitError",
    "SessionError",
    "StalePlanError",
    "UnsupportedError",
    "Warning",
    "WorkerPanicError",
    "ZipError",
    "error_kind",
]


class Error(Exception):
    """Base class for failures reported by scdata."""

    kind = "unknown"


class Warning(UserWarning):
    """Base class for actionable, non-fatal scdata feedback."""


class PerformanceWarning(Warning):
    """A valid operation selected a substantially less efficient path."""


class InvalidArgumentError(Error):
    kind = "invalid_argument"


class InvalidInputError(Error):
    kind = "invalid_input"


class InvalidConfigError(Error):
    kind = "invalid_config"


class InvalidDatasetError(Error):
    kind = "invalid_dataset"


class InvalidMetaError(Error):
    kind = "invalid_meta"


class ResourceLimitError(Error):
    kind = "resource_limit"


class StalePlanError(Error):
    kind = "stale_plan"


class UnsupportedError(Error):
    kind = "unsupported"


class IoError(Error):
    kind = "io"


class JsonError(Error):
    kind = "json"


class CodecError(Error):
    kind = "codec"


class ZipError(Error):
    kind = "zip"


class DecodeError(Error):
    kind = "decode"


class PromotionError(Error):
    kind = "promotion"


class ConversionError(Error):
    kind = "conversion"


class CancelledError(Error):
    kind = "cancelled"


class SessionError(Error):
    kind = "session"


class WorkerPanicError(Error):
    kind = "worker_panic"


class AllocationError(Error):
    kind = "allocation"


class InternalError(Error):
    kind = "internal"


class NotFoundError(Error):
    kind = "not_found"


class CorruptDataError(Error):
    kind = "corrupt_data"


class PathError(Error):
    kind = "path"


_ERROR_TYPES: dict[str, type[Error]] = {
    error_type.kind: error_type
    for error_type in (
        InvalidArgumentError,
        InvalidInputError,
        InvalidConfigError,
        InvalidDatasetError,
        InvalidMetaError,
        ResourceLimitError,
        StalePlanError,
        UnsupportedError,
        IoError,
        JsonError,
        CodecError,
        ZipError,
        DecodeError,
        PromotionError,
        ConversionError,
        CancelledError,
        SessionError,
        WorkerPanicError,
        AllocationError,
        InternalError,
        NotFoundError,
        CorruptDataError,
        PathError,
    )
}


def error_kind(exc: BaseException) -> str | None:
    """Return the stable machine-readable kind attached by the core, if any."""
    kind = getattr(exc, "kind", None)
    return kind if isinstance(kind, str) else None


def _invalid_argument(message: str) -> NoReturn:
    err = InvalidArgumentError(message)
    err.kind = "invalid_argument"
    raise err


_P = ParamSpec("_P")
_T = TypeVar("_T")


def _call_core(function: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
    try:
        return function(*args, **kwargs)
    except _core.CoreError as error:
        kind = getattr(error, "kind", "internal")
        error_type = _ERROR_TYPES.get(kind, Error)
        translated = error_type(str(error))
        translated.kind = kind
        raise translated from None
