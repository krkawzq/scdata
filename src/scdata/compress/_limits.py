"""Resource limits for opening and decoding untrusted stores."""

from __future__ import annotations

from dataclasses import dataclass, replace

from scdata.compress._validate import _UINTP_MAX, as_int

__all__ = ["DEFAULT_READ_LIMITS", "ReadLimits"]

_DEFAULT_MAXIMUM_METADATA_SIZE = 1024 * 1024
_DEFAULT_MAXIMUM_ENCODED_SIZE = 1024 * 1024 * 1024
_DEFAULT_MAXIMUM_DECODED_SIZE = 1024 * 1024 * 1024
_DEFAULT_MAXIMUM_BLOCK_COUNT = 1_000_000


@dataclass(frozen=True, slots=True)
class ReadLimits:
    """Upper bounds applied while opening and decoding a store.

    The object is immutable and safe to reuse. Keyword overrides on
    :func:`scdata.open` are applied on top of this object.
    """

    max_metadata_size: int = _DEFAULT_MAXIMUM_METADATA_SIZE
    max_encoded_size: int = _DEFAULT_MAXIMUM_ENCODED_SIZE
    max_decoded_size: int = _DEFAULT_MAXIMUM_DECODED_SIZE
    max_block_count: int = _DEFAULT_MAXIMUM_BLOCK_COUNT
    num_workers: int = 1

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
            "num_workers",
            as_int(self.num_workers, name="num_workers", minimum=1, maximum=_UINTP_MAX),
        )

    def with_overrides(
        self,
        *,
        max_metadata_size: object | None = None,
        max_encoded_size: object | None = None,
        max_decoded_size: object | None = None,
        max_block_count: object | None = None,
        num_workers: object | None = None,
    ) -> ReadLimits:
        """Return a copy with only the non-``None`` values replaced."""
        changes = {
            name: value
            for name, value in (
                ("max_metadata_size", max_metadata_size),
                ("max_encoded_size", max_encoded_size),
                ("max_decoded_size", max_decoded_size),
                ("max_block_count", max_block_count),
                ("num_workers", num_workers),
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
    num_workers: object | None = None,
) -> ReadLimits:
    if limits is None:
        limits = DEFAULT_READ_LIMITS
    elif not isinstance(limits, ReadLimits):
        raise TypeError(f"limits must be ReadLimits or None, got {type(limits).__name__}")
    return limits.with_overrides(
        max_metadata_size=max_metadata_size,
        max_encoded_size=max_encoded_size,
        max_decoded_size=max_decoded_size,
        max_block_count=max_block_count,
        num_workers=num_workers,
    )
