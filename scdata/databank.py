"""Pythonic wrappers around the private Rust DataBank extension classes.

The Rust extension (:mod:`scdata._scdata`) exposes only private pyclasses
(``_DataBank``, ``_PrefetchCells``, ``_DataBankConfig``, ...).  This module
wraps every one of them into a public Python class so callers never touch the
Rust layer directly.

Config dataclasses
------------------
All config types (``DataBankConfig``, ``IoConfig``, ``AccessConfig``, ...) are
**pure** Python :func:`dataclasses.dataclass` objects with sensible defaults
mirroring the Rust constructors.  They carry no Rust conversion logic of their
own — they support keyword-argument construction, equality comparison,
:func:`dataclasses.replace`, and :func:`dataclasses.asdict` like any dataclass.

Config is **not** a live view into a running ``ScDataBank``.  A ``ScDataBank``
deep-copies its config at construction time; mutations to the config after that
point do not affect the bank.  This is intentional and matches standard
dataclass semantics.

Rust conversion is centralized in :func:`_config_to_rust`, which walks any
config tree reflectively (field-by-field, recursing into nested configs) and
builds the matching Rust instances in one place.  ``ScDataBank.__init__`` and
the prefetch entry points call it automatically — callers never invoke it.

Flat, dynamic construction
--------------------------
Every config inherits :meth:`make` / :meth:`update` from ``_Config``, which
route flat kwargs onto nested fields so deep attribute paths are optional::

    cfg = DataBankConfig.make(
        backend="uring",                     # → io_config.backend
        entries=256,                         # → io_config.uring_config.entries
        cache_capacity_bytes=512 * 1024**2,  # → access_config.cache_capacity_bytes
        decode__num_workers=16,              # disambiguated by a ``__`` path
    )
    cfg.update(fill__num_workers=8)          # in-place, chainable

Routing rules (see :func:`_apply_dynamic`):

* a bare key matching a direct field is set on this config;
* a bare key matching exactly one nested sub-tree's leaf field is routed there
  (``entries`` → ``io_config.uring_config.entries``);
* a key ambiguous across sub-trees must be qualified with a ``__`` path
  (``decode__num_workers``, ``io__uring__base__max_in_flight``) — path segments
  accept a field's short form (``io`` for ``io_config``);
* a whole sub-config may be passed for a direct field
  (``DataBankConfig.make(io_config=IoConfig.uring(entries=256))``).

``IoConfig.uring(...)`` / ``IoConfig.threaded(...)`` remain as convenience
factories that pin ``backend`` and route the rest of their kwargs onto the
chosen backend's config.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Iterable, Iterator, Literal, TypeVar, get_type_hints

from .data._cell import CellAccess, CellBatch, CellData, _as_cell_index
from .data._dataset import Dataset, DenseDataset, DType, SparseDataset
from .data._prefetch import PrefetchBatches, PrefetchIterator
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
    _ScheduledAccessConfig,
    _ScheduledPrefetchConfig,
    _ThreadedConfig,
    _UringConfig,
    DataBankError,
)

__all__ = [
    "ScDataBank",
    "CellAccess",
    "CellBatch",
    "CellData",
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
    """Policy for gene names absent from the dataset: ``ZERO`` or ``ERROR``.

    Use the class attributes directly: ``MissingGenePolicy.ZERO`` or
    ``MissingGenePolicy.ERROR``.
    """

    __slots__ = ("_rust",)

    ZERO: MissingGenePolicy
    ERROR: MissingGenePolicy

    def __init__(self, rust: _MissingGenePolicy) -> None:
        self._rust = rust

    def __repr__(self) -> str:
        # The Rust ``_MissingGenePolicy`` already renders
        # "MissingGenePolicy.ZERO" / "MissingGenePolicy.ERROR" via its own
        # ``__repr__``, so reuse it instead of identity-comparing classattrs —
        # a policy rebuilt via ``_MissingGenePolicy("zero")`` is a fresh
        # instance that would fail an ``is`` check and misrender as ERROR.
        return repr(self._rust)


# Module-level singletons mirroring the Rust enum variants.
MissingGenePolicy.ZERO = MissingGenePolicy(_MissingGenePolicy.ZERO)
MissingGenePolicy.ERROR = MissingGenePolicy(_MissingGenePolicy.ERROR)


# ===========================================================================
# Config dataclasses
# ===========================================================================
# Every config is a pure `@dataclass` with defaults matching the Rust
# `_XxxConfig()` constructors.  They carry **no** `_to_rust()` of their own:
# `_config_to_rust` below walks any config tree reflectively and builds the
# matching Rust instances in one place, so adding a field only means adding it
# to the dataclass (and the Rust pyclass) — no per-class conversion to maintain.
#
# `_Config` adds two pythonic helpers shared by every config:
#   * `Cfg.make(**kwargs)`   — construct from flat kwargs, dynamically routed
#                              to the right nested field.
#   * `cfg.update(**kwargs)` — mutate in place the same way (chainable).
# Routing is implemented in `_apply_dynamic` (see its docstring for the rules).


_C = TypeVar("_C", bound="_Config")


class _Config:
    """Mixin base for config dataclasses: flat-kwargs construction / update.

    Subclasses are plain ``@dataclass`` types; this only contributes
    :meth:`make` and :meth:`update`.  Rust conversion lives in
    :func:`_config_to_rust`, not on the dataclasses.
    """

    @classmethod
    def make(cls: type[_C], **kwargs: Any) -> _C:
        """Build a config from flat kwargs, dynamically routed to nested fields.

        See the module docstring for the routing rules.
        """
        cfg = cls()
        _apply_dynamic(cfg, kwargs)
        return cfg

    def update(self: _C, **kwargs: Any) -> _C:
        """Mutate this config in place from flat kwargs (chainable).

        See the module docstring for the routing rules.
        """
        _apply_dynamic(self, kwargs)
        return self


@dataclass
class BaseIoConfig(_Config):
    """Shared IO backend settings (max in-flight, priority levels, ...)."""

    max_in_flight: int = 1024
    priority_levels: int = 3
    queue_shards: int = 1
    assume_non_overlapping_reads: bool = False


@dataclass
class UringConfig(_Config):
    """io_uring backend settings."""

    entries: int = 256
    drivers: int = 1
    iowq_bounded_workers: int = 0
    iowq_unbounded_workers: int = 0
    registered_files: int = 4096
    base: BaseIoConfig = field(default_factory=BaseIoConfig)


@dataclass
class ThreadedConfig(_Config):
    """Thread-pool pread/pwrite backend settings."""

    num_workers: int = 8
    cpus: list[int] | None = None
    base: BaseIoConfig = field(default_factory=BaseIoConfig)


@dataclass
class IoConfig(_Config):
    """IO backend selection.

    Construct with :meth:`uring` / :meth:`threaded` (they pin ``backend`` and
    route their kwargs onto the chosen backend's config), or :meth:`make` with
    ``backend=``.  ``kind`` reports which backend is active.  The default
    backend is ``"threaded"``.
    """

    backend: Literal["uring", "threaded"] = "threaded"
    uring_config: UringConfig = field(default_factory=UringConfig)
    threaded_config: ThreadedConfig = field(default_factory=ThreadedConfig)

    @classmethod
    def uring(cls, config: UringConfig | None = None, **kwargs: Any) -> "IoConfig":
        """Create an IoConfig with the uring backend.

        ``config`` optionally supplies a whole :class:`UringConfig`; extra
        kwargs are routed onto it (e.g. ``entries=256``,
        ``base__max_in_flight=2048``).
        """
        config = UringConfig() if config is None else config
        _apply_dynamic(config, kwargs)
        return cls(backend="uring", uring_config=config)

    @classmethod
    def threaded(cls, config: ThreadedConfig | None = None, **kwargs: Any) -> "IoConfig":
        """Create an IoConfig with the threaded backend.

        ``config`` optionally supplies a whole :class:`ThreadedConfig`; extra
        kwargs are routed onto it (e.g. ``num_workers=16``).
        """
        config = ThreadedConfig() if config is None else config
        _apply_dynamic(config, kwargs)
        return cls(backend="threaded", threaded_config=config)

    @property
    def kind(self) -> str:
        return self.backend


@dataclass
class DecodePoolConfig(_Config):
    """Decode worker pool settings."""

    num_workers: int = 4
    queue_capacity: int = 1024
    cpus: list[int] | None = None


@dataclass
class AccessCpuConfig(_Config):
    """Access-side CPU materialization pool settings."""

    num_workers: int = 4
    queue_capacity: int = 1024
    cpus: list[int] | None = None


@dataclass
class AccessConfig(_Config):
    """Access scheduler settings (cache, memory budget, shards, ...)."""

    queue_capacity: int = 1024
    scheduler_shards: int = 1
    cache_capacity_bytes: int = 268435456
    memory_budget_bytes: int = 536870912
    default_io_priority: int = 0
    keep_decoded: bool = False
    cpu: AccessCpuConfig = field(default_factory=AccessCpuConfig)


@dataclass
class FillConfig(_Config):
    """Compute / fill pool settings."""

    parallel: bool = True
    num_workers: int = 4
    queue_capacity: int = 1024
    min_parallel_rows: int = 16
    min_parallel_bytes: int = 1048576
    cpus: list[int] | None = None


@dataclass
class DataBankConfig(_Config):
    """Top-level DataBank configuration.

    All nested sub-configs are plain dataclass attributes.  The config is
    deep-copied into the Rust core when passed to :class:`ScDataBank`; after
    construction, mutating ``cfg`` does **not** affect a running bank.

    Flat construction via :meth:`make` routes kwargs to nested fields, e.g.::

        DataBankConfig.make(
            io_config=IoConfig.uring(entries=256),   # whole sub-config
            backend="uring",                         # → io_config.backend
            entries=256,                             # → io_config.uring_config.entries
            decode__num_workers=16,                  # disambiguated by path
            cache_capacity_bytes=512 * 1024 ** 2,    # → access_config
        )
    """

    io_config: IoConfig = field(default_factory=IoConfig)
    decode_config: DecodePoolConfig = field(default_factory=DecodePoolConfig)
    access_config: AccessConfig = field(default_factory=AccessConfig)
    fill_config: FillConfig = field(default_factory=FillConfig)


@dataclass
class ScheduledAccessConfig(_Config):
    """Look-ahead distances for scheduled access."""

    prefetch_step: int = 2
    decode_ahead_steps: int = 1
    ready_ahead_steps: int = 0


@dataclass
class ScheduledPrefetchConfig(_Config):
    """Per-call settings for scheduled DataBank cell prefetch."""

    prefetch_step: int = 2
    access: ScheduledAccessConfig = field(default_factory=ScheduledAccessConfig)


# ---------------------------------------------------------------------------
# Reflective Rust conversion
# ---------------------------------------------------------------------------
#
# All config dataclasses, used to detect nested config values while walking a
# tree.  IoConfig is included; it is special-cased in `_config_to_rust` because
# the Rust `_IoConfig` is built via its `uring` / `threaded` static factories,
# not field-by-field assignment.

_CONFIG_CLASSES: frozenset[type] = frozenset(
    {
        BaseIoConfig,
        UringConfig,
        ThreadedConfig,
        IoConfig,
        DecodePoolConfig,
        AccessCpuConfig,
        AccessConfig,
        FillConfig,
        DataBankConfig,
        ScheduledAccessConfig,
        ScheduledPrefetchConfig,
    }
)

# Python config class → Rust pyclass, for the reflective converter.  IoConfig
# is absent on purpose (see `_config_to_rust`).
_RUST_CONFIG_TYPES: dict[type, type] = {
    BaseIoConfig: _BaseIoConfig,
    UringConfig: _UringConfig,
    ThreadedConfig: _ThreadedConfig,
    DecodePoolConfig: _DecodePoolConfig,
    AccessCpuConfig: _AccessCpuConfig,
    AccessConfig: _AccessConfig,
    FillConfig: _FillConfig,
    DataBankConfig: _DataBankConfig,
    ScheduledAccessConfig: _ScheduledAccessConfig,
    ScheduledPrefetchConfig: _ScheduledPrefetchConfig,
}


def _config_to_rust(config: Any) -> Any:
    """Recursively build the Rust counterpart of a config tree.

    Walks the dataclass fields reflectively; nested config values are
    converted recursively.  ``IoConfig`` is special-cased because the Rust
    ``_IoConfig`` is built via its ``uring`` / ``threaded`` static factories
    rather than field-by-field assignment.  This is the single place that
    knows how to turn a Python config into a Rust one.
    """
    cls = type(config)
    if cls is IoConfig:
        if config.backend == "uring":
            return _IoConfig.uring(_config_to_rust(config.uring_config))
        return _IoConfig.threaded(_config_to_rust(config.threaded_config))
    rust_cls = _RUST_CONFIG_TYPES.get(cls)
    if rust_cls is None:
        raise TypeError(f"not a config type: {cls.__name__}")
    r = rust_cls()
    for f in fields(config):
        value = getattr(config, f.name)
        if type(value) in _CONFIG_CLASSES:
            value = _config_to_rust(value)
        setattr(r, f.name, value)
    return r


# ---------------------------------------------------------------------------
# Dynamic flat-kwargs routing (used by _Config.make / _Config.update)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _config_type_hints(cls: Any) -> dict[str, Any]:
    """Resolved type hints for ``cls`` (cached; annotations are strings)."""
    return get_type_hints(cls)


@lru_cache(maxsize=None)
def _config_subfields(cls: Any) -> tuple[tuple[str, type], ...]:
    """``(field_name, config_type)`` for each field whose type is a config."""
    hints = _config_type_hints(cls)
    return tuple(
        (f.name, hints[f.name])
        for f in fields(cls)
        if hints.get(f.name) in _CONFIG_CLASSES
    )


@lru_cache(maxsize=None)
def _config_leaf_names(cls: Any) -> frozenset[str]:
    """All field names reachable from ``cls`` (its own + nested config leaves)."""
    names: set[str] = {f.name for f in fields(cls)}
    for _fname, sub in _config_subfields(cls):
        names |= _config_leaf_names(sub)
    return frozenset(names)


def _resolve_path_field(cls: Any, segment: str) -> str:
    """Match a ``__`` path segment to a config field name.

    Accepts the literal field name (``io_config``) or its short form with a
    trailing ``_config`` stripped (``io``), so paths stay readable:
    ``io__uring__entries`` rather than ``io_config__uring_config__entries``.
    """
    names = {f.name for f in fields(cls)}
    if segment in names:
        return segment
    candidate = f"{segment}_config"
    if candidate in names:
        return candidate
    raise TypeError(f"{cls.__name__} has no field matching path segment {segment!r}")


def _apply_dynamic(config: Any, kwargs: dict[str, Any]) -> Any:
    """Route ``kwargs`` onto ``config`` by direct field, unique leaf, or path.

    For each key:

    * a ``__``-separated path (``io__uring__entries``) walks the field tree,
      resolving each segment via :func:`_resolve_path_field`;
    * a bare key matching a direct field of ``config`` is set on ``config``;
    * a bare key matching exactly one nested sub-tree's leaf field is routed
      there (``entries`` → ``io_config.uring_config.entries``);
    * a bare key matching no leaf raises ``TypeError``;
    * a bare key matching several sub-trees raises ``TypeError`` telling the
      caller to qualify it with a ``__`` path.
    """
    cls = type(config)
    direct = {f.name for f in fields(cls)}
    subfields = dict(_config_subfields(cls))
    for key, value in kwargs.items():
        if "__" in key:
            segments = key.split("__")
            obj: Any = config
            obj_cls = cls
            for seg in segments[:-1]:
                fname = _resolve_path_field(obj_cls, seg)
                obj = getattr(obj, fname)
                obj_cls = type(obj)
            setattr(obj, _resolve_path_field(obj_cls, segments[-1]), value)
            continue
        if key in direct:
            setattr(config, key, value)
            continue
        hits = [
            fname for fname, sub in subfields.items() if key in _config_leaf_names(sub)
        ]
        if len(hits) == 1:
            # Recurse so a leaf living deeper than the immediate child (e.g.
            # ``entries`` on ``io_config.uring_config``) lands on the right
            # field instead of being attached as a stray attribute.
            _apply_dynamic(getattr(config, hits[0]), {key: value})
        elif not hits:
            raise TypeError(f"{cls.__name__} has no field {key!r}")
        else:
            raise TypeError(
                f"{cls.__name__}: {key!r} is ambiguous across {hits}; "
                f"qualify with a path like {hits[0]}__{key}"
            )
    return config


# ===========================================================================
# Cell access / batch carriers + iterator
# ===========================================================================
#
# ``CellAccess`` (access + prefetch input), ``CellData`` (single-call decoded
# output), ``CellBatch`` (prefetch decoded output) and ``PrefetchIterator``
# live in the data layer (:mod:`scdata.data._cell` / :mod:`scdata.data._prefetch`)
# so they can be reused without pulling in the Rust extension.  They are
# re-exported here for callers that import them from :mod:`scdata.databank`.


# ===========================================================================
# ScDataBank
# ===========================================================================


class ScDataBank:
    """Single-cell DataBank: registers parsed datasets and serves cell access.

    A Pythonic wrapper around the Rust ``_DataBank``.  Constructed with an
    optional :class:`DataBankConfig`; the Rust core owns its IO / decode /
    access / compute pools and tears them down when this object is dropped.

    The config is deep-copied into the Rust core at construction time.
    Mutations to the config after ``ScDataBank(...)`` returns do **not**
    affect the running bank.  To change bank settings, construct a new
    ``ScDataBank`` with the updated config.

    Args:
        config: Optional :class:`DataBankConfig`.  Defaults to a sensible
            thread-pool configuration when omitted.
    """

    __slots__ = ("_inner", "_registered_count", "_meta_cache")

    # Declared non-Optional so the access methods below type-check.  ``close``
    # and ``__exit__`` invalidate the slot by storing ``None`` (each assignment
    # suppressed with ``# type: ignore[assignment]``); that ``None`` is exactly
    # what makes post-close calls raise ``AttributeError``, as documented on
    # :meth:`close`.
    _inner: _DataBank

    def __init__(self, config: DataBankConfig | None = None) -> None:
        if config is None:
            config = DataBankConfig()
        self._inner = _DataBank(_config_to_rust(config))
        self._registered_count = 0
        # Per-dataset ``(gene_names, num_genes)`` cache, keyed by DatasetId.
        # Access results carry num_genes / gene_names, and ``access_cells``
        # needs num_genes to validate shape before Rust returns the payload —
        # caching avoids a Rust round-trip on every call.
        self._meta_cache: dict[DatasetId, tuple[list[str], int]] = {}

    # -- registration --------------------------------------------------------

    def register(self, dataset: Dataset) -> DatasetId:
        """Register a parsed dataset and return its handle.

        The dataset's :attr:`~scdata.data.DenseDataset.store_root` supplies the
        filesystem path the Rust databank opens, so a dataset produced by
        :func:`scdata.io.launch` carries everything the bank needs.  Dense and
        CSR-sparse datasets dispatch to the matching Rust entry point by type.

        For datasets assembled without ``store_root`` (e.g. by hand in tests),
        use :meth:`register_dense` / :meth:`register_sparse_csr` and pass the
        store path explicitly.
        """
        if not isinstance(dataset, (DenseDataset, SparseDataset)):
            raise TypeError(f"unsupported dataset type: {type(dataset).__name__}")
        store_root = dataset.store_root
        if not store_root:
            raise ValueError(
                "dataset has no store_root; pass it to register_dense / "
                "register_sparse_csr, or build it via scdata.io.launch"
            )
        return self._register(dataset, store_root)

    def register_dense(self, ds: DenseDataset, store_path: str) -> DatasetId:
        """Register a dense dataset parsed by :func:`scdata.io.launch`.

        ``store_path`` is the filesystem path to the zarr directory or
        ``.zarr.zip`` archive.  Datasets returned by :func:`scdata.io.launch`
        already carry resolved source files; manually-built datasets use
        ``store_path`` as the root for their logical payload/chunk keys.
        """
        rust_id = self._inner.register_dense(ds, store_path)
        did = DatasetId(rust_id)
        self._registered_count += 1
        return did

    def register_sparse_csr(self, ds: SparseDataset, store_path: str) -> DatasetId:
        """Register a CSR sparse dataset parsed by :func:`scdata.io.launch`."""
        rust_id = self._inner.register_sparse_csr(ds, store_path)
        did = DatasetId(rust_id)
        self._registered_count += 1
        return did

    def _register(self, ds: Dataset, store_path: str) -> DatasetId:
        """Register a dataset by type, dispatching to the matching Rust entry point."""
        if isinstance(ds, DenseDataset):
            return self.register_dense(ds, store_path)
        if isinstance(ds, SparseDataset):
            return self.register_sparse_csr(ds, store_path)
        raise TypeError(f"unsupported dataset type: {type(ds).__name__}")

    def unregister(self, id: DatasetId) -> None:
        """Unregister a dataset, releasing its file handles and gene refs."""
        self._inner.unregister(id._rust)
        self._meta_cache.pop(id, None)
        self._registered_count = max(0, self._registered_count - 1)

    # -- queries -------------------------------------------------------------

    def dataset_genes(self, id: DatasetId) -> list[str]:
        """Gene names for ``id``, in column order matching access results."""
        return self._meta(id)[0]

    def dataset_num_cells(self, id: DatasetId) -> int:
        """Number of cells (rows) in the registered dataset."""
        return self._inner.dataset_num_cells(id._rust)

    def dataset_num_genes(self, id: DatasetId) -> int:
        """Number of genes (columns) in the registered dataset."""
        return self._meta(id)[1]

    def dataset_dtype(self, id: DatasetId) -> DType:
        """Stored value dtype of the registered dataset (a ``DType``)."""
        return self._inner.dataset_dtype(id._rust)

    def _meta(self, id: DatasetId) -> tuple[list[str], int]:
        """Cached ``(gene_names, num_genes)`` for ``id``, fetched on first use."""
        cached = self._meta_cache.get(id)
        if cached is None:
            genes = self._inner.dataset_genes(id._rust)
            num_genes = self._inner.dataset_num_genes(id._rust)
            cached = (genes, num_genes)
            self._meta_cache[id] = cached
        return cached

    # -- cell access ---------------------------------------------------------

    def access_cells(
        self,
        id: DatasetId,
        cells: Iterable[int],
        dtype: DType | None = None,
    ) -> CellData:
        """Read cells and return a decoded :class:`CellData`.

        Args:
            id: Dataset handle from :meth:`register` / :meth:`register_dense` /
                :meth:`register_sparse_csr`.
            cells: Cell indices into the dataset (any order, subset, repeats).
            dtype: Optional scdata ``DType``; when omitted it is inferred from
                the dataset's stored value dtype.

        Returns:
            A :class:`CellData` whose :attr:`~CellData.data` is a row-major
            1D array of shape ``[len(cells) * num_genes]`` (cell ``i``'s genes
            occupy ``data[i*num_genes : (i+1)*num_genes]``), with
            :attr:`~CellData.num_genes` and :attr:`~CellData.gene_names`
            populated.  ``np.asarray(result)`` returns the raw 1D array, and
            :attr:`~CellData.matrix` is a zero-copy ``[num_cells, num_genes]``
            view.  ``f16`` returns ``float16``; ``bf16`` returns ``uint16`` bit
            patterns.
        """
        return self.access_cells_fast(id, cells, dtype=dtype)

    def access_cells_fast(
        self,
        id: DatasetId,
        cells: Iterable[int],
        dtype: DType | None = None,
    ) -> CellData:
        """Read cells through the public numpy-array fast path."""
        cell_arr = _as_cell_index(cells, "cells")
        gene_names, num_genes = self._meta(id)
        fast = getattr(self._inner, "access_cells_array", None)
        if fast is not None:
            data = fast(id._rust, cell_arr, dtype)
        else:
            data = self._inner.access_cells(id._rust, cell_arr.tolist(), dtype)
        return CellData.from_array(
            cells=cell_arr,
            data=data,
            num_genes=num_genes,
            gene_names=tuple(gene_names),
        )

    def access_cells_by_gene_names(
        self,
        id: DatasetId,
        cells: Iterable[int],
        gene_names: Iterable[str],
        missing: MissingGenePolicy | None = None,
        dtype: DType | None = None,
    ) -> CellData:
        """Read cells projected onto a subset of gene names.

        Args:
            id: Dataset handle.
            cells: Cell indices into the dataset (any order, subset, repeats).
            gene_names: Gene names to project each cell onto, in the requested
                order.  ``missing`` controls genes absent from the dataset:
                ``ZERO`` (default) fills a zero column, ``ERROR`` raises.
            missing: :class:`MissingGenePolicy` for absent gene names.
            dtype: Optional scdata ``DType``; when omitted it is inferred from
                the dataset's stored value dtype.

        Returns:
            A :class:`CellData` with :attr:`~CellData.data` of shape
            ``[len(cells) * len(gene_names)]`` and :attr:`~CellData.gene_names`
            set to the requested projection.
        """
        return self.access_cells_by_gene_names_fast(
            id,
            cells,
            gene_names,
            missing=missing,
            dtype=dtype,
        )

    def access_cells_by_gene_names_fast(
        self,
        id: DatasetId,
        cells: Iterable[int],
        gene_names: Iterable[str],
        missing: MissingGenePolicy | None = None,
        dtype: DType | None = None,
    ) -> CellData:
        """Read projected cells through the public numpy-array fast path."""
        cell_arr = _as_cell_index(cells, "cells")
        names = tuple(gene_names)
        fast = getattr(self._inner, "access_cells_by_gene_names_array", None)
        if fast is not None:
            data = fast(
                id._rust,
                cell_arr,
                list(names),
                missing._rust if missing is not None else None,
                dtype,
            )
        else:
            data = self._inner.access_cells_by_gene_names(
                id._rust,
                cell_arr.tolist(),
                list(names),
                missing._rust if missing is not None else None,
                dtype,
            )
        return CellData.from_array(
            cells=cell_arr,
            data=data,
            num_genes=len(names),
            gene_names=names,
        )

    # -- prefetch ------------------------------------------------------------

    def prefetch_cells(
        self,
        id: DatasetId,
        batches: Iterable[CellAccess | Iterable[int]],
        config: ScheduledPrefetchConfig | None = None,
    ) -> Iterator[CellBatch]:
        """Stream cell batches through a scheduled prefetch iterator.

        Args:
            id: Dataset handle.
            batches: Iterable of batch inputs.  Each element may be a
                :class:`CellAccess` or any 1D cell-index iterable; it is
                normalized to a :class:`CellAccess` up front (see
                :class:`~scdata.data._prefetch.PrefetchBatches` for why this is
                not lazy).
            config: Optional :class:`ScheduledPrefetchConfig` tuning the
                ring-buffer depth and access-layer look-ahead.

        Returns:
            Iterator of decoded :class:`CellBatch` objects (each with
            ``cells`` / ``data`` / ``num_genes`` populated).
        """
        return self.prefetch_cells_fast(id, batches, config=config)

    def prefetch_cells_fast(
        self,
        id: DatasetId,
        batches: Iterable[CellAccess | Iterable[int]],
        config: ScheduledPrefetchConfig | None = None,
    ) -> Iterator[CellBatch]:
        """Stream batches through the public numpy-array fast path."""
        request = PrefetchBatches.from_iterable(batches)
        rust_config = _config_to_rust(config) if config is not None else None
        fast = getattr(self._inner, "prefetch_cells_arrays", None)
        if fast is not None:
            inner = fast(id._rust, request.batch_cell_arrays(), rust_config)
        else:
            inner = self._inner.prefetch_cells(id._rust, request.batch_cell_lists(), rust_config)
        return PrefetchIterator(inner)

    def prefetch_cells_by_gene_names(
        self,
        id: DatasetId,
        batches: Iterable[CellAccess | Iterable[int]],
        gene_names: Iterable[str],
        missing: MissingGenePolicy | None = None,
        config: ScheduledPrefetchConfig | None = None,
    ) -> Iterator[CellBatch]:
        """Like :meth:`prefetch_cells` but each batch is projected onto ``gene_names``.

        ``gene_names`` applies to every batch in the stream (the Rust entry
        point takes one shared projection).  ``missing`` controls genes absent
        from the dataset: ``ZERO`` (default) fills a zero column, ``ERROR``
        raises.
        """
        return self.prefetch_cells_by_gene_names_fast(
            id,
            batches,
            gene_names,
            missing=missing,
            config=config,
        )

    def prefetch_cells_by_gene_names_fast(
        self,
        id: DatasetId,
        batches: Iterable[CellAccess | Iterable[int]],
        gene_names: Iterable[str],
        missing: MissingGenePolicy | None = None,
        config: ScheduledPrefetchConfig | None = None,
    ) -> Iterator[CellBatch]:
        """Stream projected batches through the public numpy-array fast path."""
        request = PrefetchBatches.from_iterable(batches, gene_names=gene_names)
        names = list(request.gene_names) if request.gene_names is not None else []
        rust_missing = missing._rust if missing is not None else None
        rust_config = _config_to_rust(config) if config is not None else None
        fast = getattr(self._inner, "prefetch_cells_by_gene_names_arrays", None)
        if fast is not None:
            inner = fast(id._rust, request.batch_cell_arrays(), names, rust_missing, rust_config)
        else:
            inner = self._inner.prefetch_cells_by_gene_names(
                id._rust,
                request.batch_cell_lists(),
                names,
                rust_missing,
                rust_config,
            )
        return PrefetchIterator(inner)

    def __repr__(self) -> str:
        return f"ScDataBank(registered={self._registered_count})"

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "ScDataBank":
        return self

    def __exit__(self, *exc: object) -> None:
        # Drop the Rust core explicitly so its IO / decode / access / compute
        # thread pools are torn down deterministically on scope exit or
        # exception, rather than waiting on garbage collection.
        self._inner = None  # type: ignore[assignment]

    def close(self) -> None:
        """Release the Rust core and its thread pools immediately.

        Safe to call more than once.  After ``close``, any further call on
        this bank raises :class:`AttributeError`.
        """
        self._inner = None  # type: ignore[assignment]
