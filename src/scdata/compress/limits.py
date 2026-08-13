"""Resource limits for opening and decoding untrusted stores."""

from __future__ import annotations

from dataclasses import dataclass, replace

from scdata._core import (
    DEFAULT_MAXIMUM_BLOCK_COUNT,
    DEFAULT_MAXIMUM_DECODED_SIZE,
    DEFAULT_MAXIMUM_ENCODED_SIZE,
    DEFAULT_MAXIMUM_METADATA_SIZE,
    DEFAULT_N_WORKERS,
)
from scdata.compress._validate import _UINTP_MAX, as_int
from scdata.exceptions import _invalid_argument

__all__ = ["DEFAULT_N_WORKERS", "DEFAULT_READ_LIMITS", "ReadLimits"]


@dataclass(frozen=True, slots=True)
class ReadLimits:
    """Upper bounds applied while opening and decoding a store.

    The object is immutable and safe to reuse. Keyword overrides on
    :func:`scdata.open_store` are applied on top of this object.
    """

    max_metadata_size: int = DEFAULT_MAXIMUM_METADATA_SIZE
    max_encoded_size: int = DEFAULT_MAXIMUM_ENCODED_SIZE
    max_decoded_size: int = DEFAULT_MAXIMUM_DECODED_SIZE
    max_block_count: int = DEFAULT_MAXIMUM_BLOCK_COUNT
    n_workers: int = DEFAULT_N_WORKERS

    def __post_init__(self) -> None:
        for name in (
            "max_metadata_size",
            "max_encoded_size",
            "max_decoded_size",
            "max_block_count",
        ):
            object.__setattr__(
                self,
                name,
                as_int(getattr(self, name), name=name, maximum=_UINTP_MAX),
            )
        object.__setattr__(
            self,
            "n_workers",
            as_int(self.n_workers, name="n_workers", minimum=1, maximum=_UINTP_MAX),
        )

    def with_overrides(
        self,
        *,
        max_metadata_size: object | None = None,
        max_encoded_size: object | None = None,
        max_decoded_size: object | None = None,
        max_block_count: object | None = None,
        n_workers: object | None = None,
    ) -> ReadLimits:
        """Return a copy with only the non-``None`` values replaced."""
        changes = {
            name: value
            for name, value in (
                ("max_metadata_size", max_metadata_size),
                ("max_encoded_size", max_encoded_size),
                ("max_decoded_size", max_decoded_size),
                ("max_block_count", max_block_count),
                ("n_workers", n_workers),
            )
            if value is not None
        }
        return replace(self, **changes)


DEFAULT_READ_LIMITS = ReadLimits()


def resolve_read_limits(
    limits: ReadLimits | None,
    *,
    max_metadata_size: object | None = None,
    max_encoded_size: object | None = None,
    max_decoded_size: object | None = None,
    max_block_count: object | None = None,
    n_workers: object | None = None,
) -> ReadLimits:
    if limits is None:
        limits = DEFAULT_READ_LIMITS
    elif not isinstance(limits, ReadLimits):
        _invalid_argument(f"limits must be ReadLimits or None, got {type(limits).__name__}")
    return limits.with_overrides(
        max_metadata_size=max_metadata_size,
        max_encoded_size=max_encoded_size,
        max_decoded_size=max_decoded_size,
        max_block_count=max_block_count,
        n_workers=n_workers,
    )
