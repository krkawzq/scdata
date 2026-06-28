"""Pythonic wrappers around the private Rust DataBank extension classes.

The Rust extension (:mod:`scdata._scdata`) exposes only private pyclasses
(``_DataBank``, ``_PrefetchCells``, ``_DataBankConfig``, ...).  This module
wraps every one of them into a public Python class so callers never touch the
Rust layer directly and there are no shared-clone side effects on nested
config mutation.

Write-through config wrappers
-----------------------------
Each config wrapper owns a Rust instance and, when nested inside another
wrapper, registers a *sync-back* callback.  Mutating a leaf field (e.g.
``cfg.access_config.keep_decoded = True``) writes straight through to the
underlying Rust ``_DataBankConfig`` — the ``access_config`` sub-wrapper holds
a reference to its parent and pushes its own Rust instance back into the
parent on every mutation, so there is no stale clone.

The PyObject reflection (``getattr`` / ``json.dumps`` / dtype dispatch) stays
in Rust for speed; these wrappers are thin attribute façades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional

from numpy.typing import NDArray

from ._scdata import (
    _AccessConfig,
    _AccessCpuConfig,
    _BaseIoConfig,
    _DataBank,
    _DataBankConfig,
    _DatasetId,
    _DecodePoolConfig,
    _FillConfig,
    _IoConfig,
    _MissingGenePolicy,
    _PrefetchCells,
    _ScheduledAccessConfig,
    _ScheduledPrefetchConfig,
    _ThreadedConfig,
    _UringConfig,
    DataBankError,
)

__all__ = [
    "ScDataBank",
    "PrefetchedBatch",
    "DatasetId",
    "DataBankError",
    "MissingGenePolicy",
    "DataBankConfig",
    "IoConfig",
    "UringConfig",
    "ThreadedConfig",
    "BaseIoConfig",
    "DecodePoolConfig",
    "AccessConfig",
    "AccessCpuConfig",
    "FillConfig",
    "ScheduledAccessConfig",
    "ScheduledPrefetchConfig",
]


# ===========================================================================
# Public simple wrappers (value / enum types)
# ===========================================================================


class DatasetId:
    """Opaque ``(slot, generation)`` handle returned by ``register_*``.

    A thin wrapper over the Rust ``_DatasetId`` value object.  Immutable and
    hashable, so it can be used as a dict key or set member.
    """

    __slots__ = ("_rust",)

    def __init__(self, rust: _DatasetId) -> None:
        self._rust = rust

    @property
    def slot(self) -> int:
        return self._rust.slot

    @property
    def generation(self) -> int:
        return self._rust.generation

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DatasetId) and self._rust == other._rust

    def __hash__(self) -> int:
        return hash(self._rust)

    def __repr__(self) -> str:
        return f"DatasetId(slot={self.slot}, generation={self.generation})"


class MissingGenePolicy:
    """Policy for gene names absent from the dataset: ``ZERO`` or ``ERROR``."""

    __slots__ = ("_rust",)

    def __init__(self, rust: _MissingGenePolicy) -> None:
        self._rust = rust

    @classmethod
    def ZERO(cls) -> "MissingGenePolicy":
        return cls(_MissingGenePolicy.ZERO)

    @classmethod
    def ERROR(cls) -> "MissingGenePolicy":
        return cls(_MissingGenePolicy.ERROR)

    def __repr__(self) -> str:
        return "MissingGenePolicy.ZERO" if self._rust is _MissingGenePolicy.ZERO else "MissingGenePolicy.ERROR"


# ===========================================================================
# Write-through config wrappers
# ===========================================================================
#
# A ``_Sync`` is a zero-arg callback the wrapper invokes after mutating its
# Rust instance, so the parent wrapper can pull the updated child back in.
# Leaf wrappers (top-level configs) have ``_sync = None``.


_Sync = Callable[[Any], None]


class BaseIoConfig:
    """Shared IO backend settings (max in-flight, priority levels, ...)."""

    __slots__ = ("_rust", "_sync")

    def __init__(self, rust: Optional[_BaseIoConfig] = None, _sync: Optional[_Sync] = None) -> None:
        self._rust = rust if rust is not None else _BaseIoConfig()
        self._sync = _sync

    @property
    def max_in_flight(self) -> int:
        return self._rust.max_in_flight

    @max_in_flight.setter
    def max_in_flight(self, value: int) -> None:
        self._rust.max_in_flight = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def priority_levels(self) -> int:
        return self._rust.priority_levels

    @priority_levels.setter
    def priority_levels(self, value: int) -> None:
        self._rust.priority_levels = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def queue_shards(self) -> int:
        return self._rust.queue_shards

    @queue_shards.setter
    def queue_shards(self, value: int) -> None:
        self._rust.queue_shards = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def assume_non_overlapping_reads(self) -> bool:
        return self._rust.assume_non_overlapping_reads

    @assume_non_overlapping_reads.setter
    def assume_non_overlapping_reads(self, value: bool) -> None:
        self._rust.assume_non_overlapping_reads = value
        if self._sync is not None:
            self._sync(self._rust)

    def __repr__(self) -> str:
        return f"BaseIoConfig(max_in_flight={self.max_in_flight}, queue_shards={self.queue_shards})"


class UringConfig:
    """io_uring backend settings."""

    __slots__ = ("_rust", "_sync")

    def __init__(self, rust: Optional[_UringConfig] = None, _sync: Optional[_Sync] = None) -> None:
        self._rust = rust if rust is not None else _UringConfig()
        self._sync = _sync

    @property
    def base(self) -> BaseIoConfig:
        def _sync_base(v: Any) -> None:
            self._rust.base = v
            if self._sync is not None:
                self._sync(self._rust)
        return BaseIoConfig(self._rust.base, _sync=_sync_base)

    @base.setter
    def base(self, value: BaseIoConfig) -> None:
        self._rust.base = value._rust
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def entries(self) -> int:
        return self._rust.entries

    @entries.setter
    def entries(self, value: int) -> None:
        self._rust.entries = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def drivers(self) -> int:
        return self._rust.drivers

    @drivers.setter
    def drivers(self, value: int) -> None:
        self._rust.drivers = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def iowq_bounded_workers(self) -> int:
        return self._rust.iowq_bounded_workers

    @iowq_bounded_workers.setter
    def iowq_bounded_workers(self, value: int) -> None:
        self._rust.iowq_bounded_workers = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def iowq_unbounded_workers(self) -> int:
        return self._rust.iowq_unbounded_workers

    @iowq_unbounded_workers.setter
    def iowq_unbounded_workers(self, value: int) -> None:
        self._rust.iowq_unbounded_workers = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def registered_files(self) -> int:
        return self._rust.registered_files

    @registered_files.setter
    def registered_files(self, value: int) -> None:
        self._rust.registered_files = value
        if self._sync is not None:
            self._sync(self._rust)

    def __repr__(self) -> str:
        return f"UringConfig(entries={self.entries}, drivers={self.drivers})"


class ThreadedConfig:
    """Thread-pool pread/pwrite backend settings."""

    __slots__ = ("_rust", "_sync")

    def __init__(self, rust: Optional[_ThreadedConfig] = None, _sync: Optional[_Sync] = None) -> None:
        self._rust = rust if rust is not None else _ThreadedConfig()
        self._sync = _sync

    @property
    def base(self) -> BaseIoConfig:
        def _sync_base(v: Any) -> None:
            self._rust.base = v
            if self._sync is not None:
                self._sync(self._rust)
        return BaseIoConfig(self._rust.base, _sync=_sync_base)

    @base.setter
    def base(self, value: BaseIoConfig) -> None:
        self._rust.base = value._rust
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def num_workers(self) -> int:
        return self._rust.num_workers

    @num_workers.setter
    def num_workers(self, value: int) -> None:
        self._rust.num_workers = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def cpus(self) -> Optional[list[int]]:
        return self._rust.cpus

    @cpus.setter
    def cpus(self, value: Optional[list[int]]) -> None:
        self._rust.cpus = value
        if self._sync is not None:
            self._sync(self._rust)

    def __repr__(self) -> str:
        return f"ThreadedConfig(num_workers={self.num_workers})"


class IoConfig:
    """IO backend selection.

    Construct with :meth:`uring` or :meth:`threaded`; ``kind`` reports which
    backend is active.
    """

    __slots__ = ("_rust", "_sync")

    def __init__(self, rust: Optional[_IoConfig] = None, _sync: Optional[_Sync] = None) -> None:
        self._rust = rust if rust is not None else _IoConfig()
        self._sync = _sync

    @classmethod
    def uring(cls, config: Optional[UringConfig] = None) -> "IoConfig":
        return cls(_IoConfig.uring(config._rust if config is not None else None))

    @classmethod
    def threaded(cls, config: Optional[ThreadedConfig] = None) -> "IoConfig":
        return cls(_IoConfig.threaded(config._rust if config is not None else None))

    @property
    def kind(self) -> str:
        return self._rust.kind

    def __repr__(self) -> str:
        return f"IoConfig({self.kind})"


class DecodePoolConfig:
    """Decode worker pool settings."""

    __slots__ = ("_rust", "_sync")

    def __init__(self, rust: Optional[_DecodePoolConfig] = None, _sync: Optional[_Sync] = None) -> None:
        self._rust = rust if rust is not None else _DecodePoolConfig()
        self._sync = _sync

    @property
    def num_workers(self) -> int:
        return self._rust.num_workers

    @num_workers.setter
    def num_workers(self, value: int) -> None:
        self._rust.num_workers = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def queue_capacity(self) -> int:
        return self._rust.queue_capacity

    @queue_capacity.setter
    def queue_capacity(self, value: int) -> None:
        self._rust.queue_capacity = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def cpus(self) -> Optional[list[int]]:
        return self._rust.cpus

    @cpus.setter
    def cpus(self, value: Optional[list[int]]) -> None:
        self._rust.cpus = value
        if self._sync is not None:
            self._sync(self._rust)

    def __repr__(self) -> str:
        return f"DecodePoolConfig(num_workers={self.num_workers})"


class AccessCpuConfig:
    """Access-side CPU materialization pool settings."""

    __slots__ = ("_rust", "_sync")

    def __init__(self, rust: Optional[_AccessCpuConfig] = None, _sync: Optional[_Sync] = None) -> None:
        self._rust = rust if rust is not None else _AccessCpuConfig()
        self._sync = _sync

    @property
    def num_workers(self) -> int:
        return self._rust.num_workers

    @num_workers.setter
    def num_workers(self, value: int) -> None:
        self._rust.num_workers = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def queue_capacity(self) -> int:
        return self._rust.queue_capacity

    @queue_capacity.setter
    def queue_capacity(self, value: int) -> None:
        self._rust.queue_capacity = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def cpus(self) -> Optional[list[int]]:
        return self._rust.cpus

    @cpus.setter
    def cpus(self, value: Optional[list[int]]) -> None:
        self._rust.cpus = value
        if self._sync is not None:
            self._sync(self._rust)

    def __repr__(self) -> str:
        return f"AccessCpuConfig(num_workers={self.num_workers})"


class AccessConfig:
    """Access scheduler settings (cache, memory budget, shards, ...)."""

    __slots__ = ("_rust", "_sync")

    def __init__(self, rust: Optional[_AccessConfig] = None, _sync: Optional[_Sync] = None) -> None:
        self._rust = rust if rust is not None else _AccessConfig()
        self._sync = _sync

    @property
    def queue_capacity(self) -> int:
        return self._rust.queue_capacity

    @queue_capacity.setter
    def queue_capacity(self, value: int) -> None:
        self._rust.queue_capacity = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def scheduler_shards(self) -> int:
        return self._rust.scheduler_shards

    @scheduler_shards.setter
    def scheduler_shards(self, value: int) -> None:
        self._rust.scheduler_shards = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def cache_capacity_bytes(self) -> int:
        return self._rust.cache_capacity_bytes

    @cache_capacity_bytes.setter
    def cache_capacity_bytes(self, value: int) -> None:
        self._rust.cache_capacity_bytes = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def memory_budget_bytes(self) -> int:
        return self._rust.memory_budget_bytes

    @memory_budget_bytes.setter
    def memory_budget_bytes(self, value: int) -> None:
        self._rust.memory_budget_bytes = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def default_io_priority(self) -> int:
        return self._rust.default_io_priority

    @default_io_priority.setter
    def default_io_priority(self, value: int) -> None:
        self._rust.default_io_priority = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def keep_decoded(self) -> bool:
        return self._rust.keep_decoded

    @keep_decoded.setter
    def keep_decoded(self, value: bool) -> None:
        self._rust.keep_decoded = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def cpu(self) -> AccessCpuConfig:
        def _sync_cpu(v: Any) -> None:
            self._rust.cpu = v
            if self._sync is not None:
                self._sync(self._rust)
        return AccessCpuConfig(self._rust.cpu, _sync=_sync_cpu)

    @cpu.setter
    def cpu(self, value: AccessCpuConfig) -> None:
        self._rust.cpu = value._rust
        if self._sync is not None:
            self._sync(self._rust)

    def __repr__(self) -> str:
        return (
            f"AccessConfig(shards={self.scheduler_shards}, "
            f"cache={self.cache_capacity_bytes}, keep_decoded={self.keep_decoded})"
        )


class FillConfig:
    """Compute / fill pool settings."""

    __slots__ = ("_rust", "_sync")

    def __init__(self, rust: Optional[_FillConfig] = None, _sync: Optional[_Sync] = None) -> None:
        self._rust = rust if rust is not None else _FillConfig()
        self._sync = _sync

    @property
    def parallel(self) -> bool:
        return self._rust.parallel

    @parallel.setter
    def parallel(self, value: bool) -> None:
        self._rust.parallel = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def num_workers(self) -> int:
        return self._rust.num_workers

    @num_workers.setter
    def num_workers(self, value: int) -> None:
        self._rust.num_workers = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def queue_capacity(self) -> int:
        return self._rust.queue_capacity

    @queue_capacity.setter
    def queue_capacity(self, value: int) -> None:
        self._rust.queue_capacity = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def min_parallel_rows(self) -> int:
        return self._rust.min_parallel_rows

    @min_parallel_rows.setter
    def min_parallel_rows(self, value: int) -> None:
        self._rust.min_parallel_rows = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def min_parallel_bytes(self) -> int:
        return self._rust.min_parallel_bytes

    @min_parallel_bytes.setter
    def min_parallel_bytes(self, value: int) -> None:
        self._rust.min_parallel_bytes = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def cpus(self) -> Optional[list[int]]:
        return self._rust.cpus

    @cpus.setter
    def cpus(self, value: Optional[list[int]]) -> None:
        self._rust.cpus = value
        if self._sync is not None:
            self._sync(self._rust)

    def __repr__(self) -> str:
        return f"FillConfig(parallel={self.parallel}, num_workers={self.num_workers})"


class DataBankConfig:
    """Top-level DataBank configuration.

    All nested sub-configs write through to the underlying Rust
    ``_DataBankConfig``, so mutations like
    ``cfg.access_config.keep_decoded = True`` are immediately visible to a
    ``ScDataBank`` built from ``cfg``.

    Write-through works because each sub-config wrapper captures a closure that
    pushes its own Rust instance back into this top-level config on every
    mutation — the Rust nested getters return clones, so we must reassign the
    whole sub-config after editing it.
    """

    __slots__ = ("_rust",)

    def __init__(self, rust: Optional[_DataBankConfig] = None) -> None:
        self._rust = rust if rust is not None else _DataBankConfig()

    @property
    def io_config(self) -> IoConfig:
        # Each access returns a fresh sub-wrapper wrapping a clone of the
        # current sub-config; the sync closure writes that clone back here.
        return IoConfig(
            self._rust.io_config,
            _sync=lambda v: setattr(self._rust, "io_config", v),
        )

    @io_config.setter
    def io_config(self, value: IoConfig) -> None:
        self._rust.io_config = value._rust

    @property
    def decode_config(self) -> DecodePoolConfig:
        return DecodePoolConfig(
            self._rust.decode_config,
            _sync=lambda v: setattr(self._rust, "decode_config", v),
        )

    @decode_config.setter
    def decode_config(self, value: DecodePoolConfig) -> None:
        self._rust.decode_config = value._rust

    @property
    def access_config(self) -> AccessConfig:
        return AccessConfig(
            self._rust.access_config,
            _sync=lambda v: setattr(self._rust, "access_config", v),
        )

    @access_config.setter
    def access_config(self, value: AccessConfig) -> None:
        self._rust.access_config = value._rust

    @property
    def fill_config(self) -> FillConfig:
        return FillConfig(
            self._rust.fill_config,
            _sync=lambda v: setattr(self._rust, "fill_config", v),
        )

    @fill_config.setter
    def fill_config(self, value: FillConfig) -> None:
        self._rust.fill_config = value._rust

    def __repr__(self) -> str:
        return "DataBankConfig(...)"


class ScheduledAccessConfig:
    """Look-ahead distances for scheduled access."""

    __slots__ = ("_rust", "_sync")

    def __init__(self, rust: Optional[_ScheduledAccessConfig] = None, _sync: Optional[_Sync] = None) -> None:
        self._rust = rust if rust is not None else _ScheduledAccessConfig()
        self._sync = _sync

    @property
    def prefetch_step(self) -> int:
        return self._rust.prefetch_step

    @prefetch_step.setter
    def prefetch_step(self, value: int) -> None:
        self._rust.prefetch_step = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def decode_ahead_steps(self) -> int:
        return self._rust.decode_ahead_steps

    @decode_ahead_steps.setter
    def decode_ahead_steps(self, value: int) -> None:
        self._rust.decode_ahead_steps = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def ready_ahead_steps(self) -> int:
        return self._rust.ready_ahead_steps

    @ready_ahead_steps.setter
    def ready_ahead_steps(self, value: int) -> None:
        self._rust.ready_ahead_steps = value
        if self._sync is not None:
            self._sync(self._rust)

    def __repr__(self) -> str:
        return (
            f"ScheduledAccessConfig(prefetch_step={self.prefetch_step}, "
            f"decode_ahead_steps={self.decode_ahead_steps})"
        )


class ScheduledPrefetchConfig:
    """Per-call settings for scheduled DataBank cell prefetch."""

    __slots__ = ("_rust", "_sync")

    def __init__(self, rust: Optional[_ScheduledPrefetchConfig] = None, _sync: Optional[_Sync] = None) -> None:
        self._rust = rust if rust is not None else _ScheduledPrefetchConfig()
        self._sync = _sync

    @property
    def prefetch_step(self) -> int:
        return self._rust.prefetch_step

    @prefetch_step.setter
    def prefetch_step(self, value: int) -> None:
        self._rust.prefetch_step = value
        if self._sync is not None:
            self._sync(self._rust)

    @property
    def access(self) -> ScheduledAccessConfig:
        def _sync_access(v: Any) -> None:
            self._rust.access = v
            if self._sync is not None:
                self._sync(self._rust)
        return ScheduledAccessConfig(self._rust.access, _sync=_sync_access)

    @access.setter
    def access(self, value: ScheduledAccessConfig) -> None:
        self._rust.access = value._rust
        if self._sync is not None:
            self._sync(self._rust)

    def __repr__(self) -> str:
        return f"ScheduledPrefetchConfig(prefetch_step={self.prefetch_step})"


# ===========================================================================
# Prefetched batch + iterator
# ===========================================================================


@dataclass(frozen=True)
class PrefetchedBatch:
    """One decoded batch from :meth:`ScDataBank.prefetch_cells`.

    Attributes:
        cells: 1D numpy array of cell indices in this batch.
        data: 1D row-major numpy array of shape ``[len(cells) * num_genes]``.
            For ``f16`` data the dtype is ``float16``; for ``bf16`` it is
            ``uint16`` holding the raw bfloat16 bit pattern.
        num_genes: Number of gene columns per cell.
    """

    cells: NDArray[Any]
    data: NDArray[Any]
    num_genes: int


class _PrefetchIter:
    """Iterator wrapping ``_PrefetchCells`` tuples into ``PrefetchedBatch``."""

    __slots__ = ("_inner",)

    def __init__(self, inner: _PrefetchCells) -> None:
        self._inner = inner

    def __iter__(self) -> Iterator[PrefetchedBatch]:
        return self

    def __next__(self) -> PrefetchedBatch:
        cells, data, num_genes = next(self._inner)
        return PrefetchedBatch(cells=cells, data=data, num_genes=num_genes)


# ===========================================================================
# ScDataBank
# ===========================================================================


class ScDataBank:
    """Single-cell DataBank: registers parsed datasets and serves cell access.

    A Pythonic wrapper around the Rust ``_DataBank``.  Constructed with an
    optional :class:`DataBankConfig`; the Rust core owns its IO / decode /
    access / compute pools and tears them down when this object is dropped.

    Args:
        config: Optional :class:`DataBankConfig`.  Defaults to a sensible
            thread-pool configuration when omitted.
    """

    __slots__ = ("_inner",)

    def __init__(self, config: Optional[DataBankConfig] = None) -> None:
        rust_config = config._rust if config is not None else None
        self._inner = _DataBank(rust_config)

    def register_dense(self, ds: Any, store_path: str) -> DatasetId:
        """Register a dense dataset parsed by ``scdata.read``.

        ``store_path`` is the filesystem path to the ``.zarr`` directory holding
        the payload files; the dataset's ``payload_path`` is a key relative to
        it.  ZIP stores are not supported yet.
        """
        return DatasetId(self._inner.register_dense(ds, store_path))

    def register_sparse_csr(self, ds: Any, store_path: str) -> DatasetId:
        """Register a CSR sparse dataset parsed by ``scdata.read``."""
        return DatasetId(self._inner.register_sparse_csr(ds, store_path))

    def unregister(self, id: DatasetId) -> None:
        """Unregister a dataset, releasing its file handles and gene refs."""
        self._inner.unregister(id._rust)

    def dataset_genes(self, id: DatasetId) -> list[str]:
        """Gene names for ``id``, in column order matching access results."""
        return self._inner.dataset_genes(id._rust)

    def dataset_num_cells(self, id: DatasetId) -> int:
        """Number of cells (rows) in the registered dataset."""
        return self._inner.dataset_num_cells(id._rust)

    def dataset_num_genes(self, id: DatasetId) -> int:
        """Number of genes (columns) in the registered dataset."""
        return self._inner.dataset_num_genes(id._rust)

    def dataset_dtype(self, id: DatasetId) -> Any:
        """Stored value dtype of the registered dataset (a ``DType``)."""
        return self._inner.dataset_dtype(id._rust)

    def access_cells(
        self,
        id: DatasetId,
        cells: Iterable[int],
        dtype: Optional[Any] = None,
    ) -> NDArray[Any]:
        """Read cells into a 1D numpy array of shape ``[len(cells) * num_genes]``.

        Args:
            id: Dataset handle from :meth:`register_dense` / :meth:`register_sparse_csr`.
            cells: Cell indices into the dataset (any order, subset, repeats).
            dtype: Optional scdata ``DType``; when omitted it is inferred from
                the dataset's stored value dtype.

        Returns:
            Row-major numpy array: cell ``i``'s genes occupy
            ``out[i*num_genes : (i+1)*num_genes]``.  ``f16`` returns
            ``float16``; ``bf16`` returns ``uint16`` bit patterns.
        """
        return self._inner.access_cells(id._rust, list(cells), dtype)

    def access_cells_by_gene_names(
        self,
        id: DatasetId,
        cells: Iterable[int],
        gene_names: list[str],
        missing: Optional[MissingGenePolicy] = None,
        dtype: Optional[Any] = None,
    ) -> NDArray[Any]:
        """Read cells projected onto a subset of gene names.

        Returns a 1D numpy array of shape ``[len(cells) * len(gene_names)]``.
        ``missing`` controls genes absent from the dataset: ``ZERO`` (default)
        fills a zero column, ``ERROR`` raises.  Genes are returned in the
        requested order.
        """
        return self._inner.access_cells_by_gene_names(
            id._rust, list(cells), gene_names,
            missing._rust if missing is not None else None, dtype,
        )

    def prefetch_cells(
        self,
        id: DatasetId,
        batches: Iterable[list[int]],
        config: Optional[ScheduledPrefetchConfig] = None,
    ) -> Iterator[PrefetchedBatch]:
        """Stream cell batches through a scheduled prefetch iterator.

        Args:
            id: Dataset handle.
            batches: Iterable of cell-index lists.
            config: Optional :class:`ScheduledPrefetchConfig` tuning the
                ring-buffer depth and access-layer look-ahead.

        Returns:
            Iterator of :class:`PrefetchedBatch`.
        """
        inner = self._inner.prefetch_cells(id._rust, batches, config._rust if config is not None else None)
        return _PrefetchIter(inner)

    def prefetch_cells_by_gene_names(
        self,
        id: DatasetId,
        batches: Iterable[list[int]],
        gene_names: list[str],
        missing: Optional[MissingGenePolicy] = None,
        config: Optional[ScheduledPrefetchConfig] = None,
    ) -> Iterator[PrefetchedBatch]:
        """Like :meth:`prefetch_cells` but each batch is projected onto ``gene_names``."""
        inner = self._inner.prefetch_cells_by_gene_names(
            id._rust, batches, gene_names,
            missing._rust if missing is not None else None,
            config._rust if config is not None else None,
        )
        return _PrefetchIter(inner)

    def __repr__(self) -> str:
        return "ScDataBank(scdata-rust)"
