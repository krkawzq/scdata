"""Exception and warning hierarchy for ``sc_compress``."""

from __future__ import annotations

from typing import NoReturn

from sc_compress._core import (
    AllocationError,
    CodecError,
    CorruptDataError,
    InvalidArgumentError,
    InvalidMetaError,
    IoError,
    JsonError,
    NotFoundError,
    PathError,
    ScCompressError,
    ZipError,
)

__all__ = [
    "AllocationError",
    "CodecError",
    "CorruptDataError",
    "InvalidArgumentError",
    "InvalidMetaError",
    "IoError",
    "JsonError",
    "NotFoundError",
    "PathError",
    "PerformanceWarning",
    "ScCompressError",
    "ScCompressWarning",
    "ZipError",
    "error_kind",
]


class ScCompressWarning(UserWarning):
    """Base class for actionable, non-fatal sc-compress feedback."""


class PerformanceWarning(ScCompressWarning):
    """A valid operation selected a substantially less efficient path."""


def error_kind(exc: BaseException) -> str | None:
    """Return the stable machine-readable kind attached by the core, if any."""
    kind = getattr(exc, "kind", None)
    if isinstance(kind, str):
        return kind
    mapping = (
        (InvalidArgumentError, "invalid_argument"),
        (InvalidMetaError, "invalid_meta"),
        (CorruptDataError, "corrupt_data"),
        (NotFoundError, "not_found"),
        (IoError, "io"),
        (JsonError, "json"),
        (CodecError, "codec"),
        (ZipError, "zip"),
        (AllocationError, "allocation"),
        (PathError, "path"),
        (ScCompressError, "error"),
    )
    for cls, name in mapping:
        if isinstance(exc, cls):
            return name
    return None


def _invalid_argument(message: str) -> NoReturn:
    err = InvalidArgumentError(message)
    err.kind = "invalid_argument"
    raise err
