"""Scheduled prefetch iterator types for the DataBank."""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
from numpy.typing import NDArray

from scdata.data._coerce import _as_gene_names
from scdata.data._cell import CellBatch

__all__ = ["PrefetchIterator"]


class PrefetchIterator:
    """Iterator yielding decoded :class:`~scdata.data._cell.CellBatch` objects.

    Wraps any iterable of ``(cells, data, num_genes)`` tuples — in particular
    the Rust ``_PrefetchCells`` pyclass, whose ``__next__`` returns exactly
    that tuple form.  Each tuple is lifted into a decoded :class:`CellBatch`,
    the output type of the prefetch pipeline.  Staying duck-typed keeps this
    class in the data layer (no import of the Rust extension); the execution
    layer constructs it with the Rust producer as the source.

    Two observability attributes are forwarded (as plain values) from the Rust
    producer when available: :attr:`resolved_strategy` (``"blosc_lz4_fast"`` if
    the fast path engaged, else ``"generic"``) and :attr:`fallback_reason`
    (why the fast path fell back to generic when requested but not engaged, or
    ``None``).
    """

    __slots__ = ("_inner", "_gene_names", "_resolved_strategy", "_fallback_reason")

    def __init__(
        self,
        inner: Iterable[tuple[NDArray[np.intp], NDArray[np.generic], int]],
        gene_names: Iterable[str] | None = None,
        resolved_strategy: str | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        # Bind the iterator protocol once; ``inner`` may be a Rust pyclass that
        # is itself an iterator (``__iter__`` returns self) or any Python
        # iterable.
        self._inner: Iterator[tuple[NDArray[np.intp], NDArray[np.generic], int]] = iter(inner)
        self._gene_names = _as_gene_names(gene_names) if gene_names is not None else None
        self._resolved_strategy = resolved_strategy
        self._fallback_reason = fallback_reason

    def __iter__(self) -> "PrefetchIterator":
        return self

    def __next__(self) -> CellBatch:
        cells, data, num_genes = next(self._inner)
        return CellBatch.from_array(
            cells=cells,
            data=data,
            num_genes=num_genes,
            gene_names=self._gene_names,
        )

    @property
    def resolved_strategy(self) -> str | None:
        """Short name of the access strategy this session runs.

        ``"blosc_lz4_fast"`` if the fast Blosc-LZ4 path engaged, ``"generic"``
        for the standard access-scheduler path, or ``None`` if the underlying
        producer does not expose it.
        """
        return self._resolved_strategy

    @property
    def fallback_reason(self) -> str | None:
        """Why the fast path fell back to generic, when requested but not engaged.

        ``None`` when the fast path is active, or when fast mode was not
        requested (``fast_mode='disabled'``), or when the underlying producer
        does not expose it.
        """
        return self._fallback_reason
