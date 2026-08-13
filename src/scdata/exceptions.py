"""Public exception types for ``scdata``.

Call-site validation uses built-in exceptions:

* ``TypeError`` — wrong Python type
* ``ValueError`` — right type, illegal value
* ``IndexError`` — out-of-range or malformed index
* ``FileNotFoundError`` / ``FileExistsError`` / ``NotADirectoryError`` —
  filesystem problems

Operational failures from the native core use the ``Error`` hierarchy
below (decode, conversion, cancellation, corrupt data, and so on).
``except scdata.Error`` catches those runtime failures, not argument
mistakes.

``InvalidArgumentError`` remains importable for compatibility. New
validation code raises the matching built-in exception instead.
"""

from __future__ import annotations

from scdata._core import (
    AllocationError,
    CancelledError,
    CodecError,
    ConversionError,
    CorruptDataError,
    DecodeError,
    Error,
    InternalError,
    InvalidArgumentError,
    InvalidConfigError,
    InvalidDatasetError,
    InvalidInputError,
    InvalidMetaError,
    IoError,
    JsonError,
    NotFoundError,
    PathError,
    PerformanceWarning,
    PromotionError,
    ResourceLimitError,
    SessionError,
    StalePlanError,
    UnsupportedError,
    Warning,
    WorkerPanicError,
    ZipError,
)

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


def error_kind(exc: BaseException) -> str | None:
    """Return the stable machine-readable kind attached by the core, if any."""
    kind = getattr(exc, "kind", None)
    return kind if isinstance(kind, str) else None
