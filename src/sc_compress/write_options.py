"""Reusable write/partition options for directory, ZIP, and AnnData writers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from sc_compress._core import DEFAULT_N_WORKERS
from sc_compress._validate import (
    _UINTP_MAX,
    ResolvedPartition,
    as_int,
    resolve_write_partitions,
)
from sc_compress.exceptions import _invalid_argument

__all__ = ["DEFAULT_WRITE_OPTIONS", "WriteOptions", "resolve_write_options"]

PartitionPolicy = Literal["cells", "budget"]


@dataclass(frozen=True, slots=True)
class WriteOptions:
    """Chunk/block partition knobs shared by all sc-compress writers.

    ``policy='cells'`` maps to Rust ``Partition::FixedCells``; ``policy='budget'``
    maps to ``Partition::BytesBudget``. Dense matrices only support ``cells``.
    """

    chunk_policy: PartitionPolicy = "cells"
    block_policy: PartitionPolicy = "cells"
    chunk_cells: int | None = None
    block_cells: int | None = None
    chunk_budget: int | None = None
    block_budget: int | None = None
    n_workers: int = DEFAULT_N_WORKERS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_workers",
            as_int(self.n_workers, name="n_workers", minimum=1, maximum=_UINTP_MAX),
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
        n_workers: int | None = None,
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
                ("n_workers", n_workers),
            )
            if value is not None
        }
        return replace(self, **changes)

    def resolve(self, *, dense: bool = False) -> tuple[ResolvedPartition, ResolvedPartition]:
        """Normalize policies/defaults and enforce dense vs CSR constraints."""
        return resolve_write_partitions(
            chunk_policy=self.chunk_policy,
            block_policy=self.block_policy,
            chunk_cells=self.chunk_cells,
            block_cells=self.block_cells,
            chunk_budget=self.chunk_budget,
            block_budget=self.block_budget,
            dense=dense,
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
    n_workers: int | None = None,
) -> WriteOptions:
    """Apply optional keyword overrides on top of ``options`` or the defaults."""
    if options is None:
        options = DEFAULT_WRITE_OPTIONS
    elif not isinstance(options, WriteOptions):
        _invalid_argument(f"options must be WriteOptions or None, got {type(options).__name__}")
    return options.with_overrides(
        chunk_policy=chunk_policy,
        block_policy=block_policy,
        chunk_cells=chunk_cells,
        block_cells=block_cells,
        chunk_budget=chunk_budget,
        block_budget=block_budget,
        n_workers=n_workers,
    )
