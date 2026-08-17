"""Reusable write/partition options for directory, ZIP, and AnnData writers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

from scdata.compress._codec import Codec, as_codec
from scdata.compress._validate import (
    DEFAULT_BLOCK_BUDGET,
    DEFAULT_CHUNK_BUDGET,
    _UINTP_MAX,
    ResolvedPartition,
    as_int,
    resolve_write_partitions,
)

__all__ = [
    "DEFAULT_BLOCK_BUDGET",
    "DEFAULT_CHUNK_BUDGET",
    "DEFAULT_WRITE_OPTIONS",
    "WriteOptions",
    "resolve_write_options",
]

PartitionPolicy = Literal["cells", "budget"]


@dataclass(frozen=True)
class WriteOptions:
    """Chunk/block partition knobs shared by all sc-compress writers.

    ``policy='cells'`` maps to Rust ``Partition::FixedCells``; ``policy='budget'``
    maps to ``Partition::BytesBudget`` for CSR. For dense matrices, Python lowers
    ``budget`` to ``fixed_cells`` that meet or slightly exceed the byte target
    using ``row_bytes = n_cols * dtype.itemsize`` (Rust still receives cells).

    ``codec`` configures dense or CSR matrix data through one representation-
    independent Blosc policy. ``indptr_codec`` applies only to CSR row pointers;
    dense writers intentionally do not consume it so the same immutable options
    object can be reused for mixed matrix collections such as AnnData.
    """

    chunk_policy: PartitionPolicy = "budget"
    block_policy: PartitionPolicy = "budget"
    chunk_cells: int | None = None
    block_cells: int | None = None
    chunk_budget: int | None = DEFAULT_CHUNK_BUDGET
    block_budget: int | None = DEFAULT_BLOCK_BUDGET
    codec: Codec | None = None
    indptr_codec: Codec | None = None
    num_workers: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "num_workers",
            as_int(self.num_workers, name="num_workers", minimum=1, maximum=_UINTP_MAX),
        )
        if self.codec is not None and not isinstance(self.codec, Codec):
            raise TypeError(f"codec must be Codec or None, got {type(self.codec).__name__}")
        if self.indptr_codec is not None and not isinstance(self.indptr_codec, Codec):
            raise TypeError(
                f"indptr_codec must be Codec or None, got {type(self.indptr_codec).__name__}"
            )

    def with_overrides(
        self,
        *,
        chunk_policy: PartitionPolicy | None = None,
        block_policy: PartitionPolicy | None = None,
        chunk_cells: int | None = None,
        block_cells: int | None = None,
        chunk_budget: int | None = None,
        block_budget: int | None = None,
        codec: Codec | str | Mapping[str, Any] | None = None,
        indptr_codec: Codec | str | Mapping[str, Any] | None = None,
        num_workers: int | None = None,
    ) -> WriteOptions:
        """Return a copy with only the non-``None`` values replaced."""
        changes = {
            name: value
            for name, value in (
                ("chunk_policy", chunk_policy),
                ("block_policy", block_policy),
                ("chunk_cells", chunk_cells),
                ("block_cells", block_cells),
                ("chunk_budget", chunk_budget),
                ("block_budget", block_budget),
                ("num_workers", num_workers),
            )
            if value is not None
        }
        if codec is not None:
            changes["codec"] = as_codec(codec, name="codec")
        if indptr_codec is not None:
            changes["indptr_codec"] = as_codec(indptr_codec, name="indptr_codec")
        return replace(self, **changes)

    def resolved_codec(self) -> Codec:
        """Validated matrix-data codec shared by dense and CSR writers."""
        codec = Codec.blosc() if self.codec is None else self.codec
        if codec.algorithm != "blosc":
            raise ValueError(f"SCC matrix data requires Codec.blosc(), got {codec.algorithm!r}")
        return codec

    def resolved_indptr_codec(self) -> Codec:
        """Codec for CSR row pointers; dense writers do not consume this option."""
        if self.indptr_codec is None:
            return Codec.zstd()
        return self.indptr_codec

    def resolve(
        self,
        *,
        dense: bool = False,
        row_bytes: int | None = None,
    ) -> tuple[ResolvedPartition, ResolvedPartition]:
        """Normalize policies/defaults and lower dense budgets to fixed cells."""
        return resolve_write_partitions(
            chunk_policy=self.chunk_policy,
            block_policy=self.block_policy,
            chunk_cells=self.chunk_cells,
            block_cells=self.block_cells,
            chunk_budget=self.chunk_budget,
            block_budget=self.block_budget,
            dense=dense,
            row_bytes=row_bytes,
        )


DEFAULT_WRITE_OPTIONS = WriteOptions()


def resolve_write_options(
    options: WriteOptions | None,
    *,
    chunk_policy: PartitionPolicy | None = None,
    block_policy: PartitionPolicy | None = None,
    chunk_cells: int | None = None,
    block_cells: int | None = None,
    chunk_budget: int | None = None,
    block_budget: int | None = None,
    codec: Codec | str | Mapping[str, Any] | None = None,
    indptr_codec: Codec | str | Mapping[str, Any] | None = None,
    num_workers: int | None = None,
) -> WriteOptions:
    """Apply optional keyword overrides on top of ``options`` or the defaults."""
    if options is None:
        options = DEFAULT_WRITE_OPTIONS
    elif not isinstance(options, WriteOptions):
        raise TypeError(f"options must be WriteOptions or None, got {type(options).__name__}")
    return options.with_overrides(
        chunk_policy=chunk_policy,
        block_policy=block_policy,
        chunk_cells=chunk_cells,
        block_cells=block_cells,
        chunk_budget=chunk_budget,
        block_budget=block_budget,
        codec=codec,
        indptr_codec=indptr_codec,
        num_workers=num_workers,
    )
