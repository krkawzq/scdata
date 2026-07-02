"""Pythonic wrappers around the private Rust DataBank extension classes.

The Rust extension (:mod:`scdata._scdata`) exposes only private pyclasses
(``_DataBank``, ``_PrefetchCells``, ``_DataBankConfig``, ...).  This module
wraps every one of them into a public Python class, so callers never touch the
Rust layer directly.

All config types (``DataBankConfig``, ``IoConfig``, ``AccessConfig``, ...) are
pure Python :func:`dataclasses.dataclass` objects with Python-owned defaults.
Those Python defaults are what get sent to Rust whenever a caller omits an
explicit config.  They carry no Rust conversion logic of their own — they
support keyword-argument construction, equality comparison,
:func:`dataclasses.replace`, and :func:`dataclasses.asdict` like any dataclass.
A config is **not** a live view into a running ``ScDataBank``: the bank
deep-copies its config at construction time, and later mutations do not affect
it.  Rust conversion is centralized in :func:`_config_to_rust`, which walks a
config tree field-by-field and builds the matching Rust instances; it is
called automatically by ``ScDataBank.__init__`` and the prefetch entry points.

Every config also inherits :meth:`make` / :meth:`from_dict` / :meth:`update`
from ``_Config``.  ``make`` accepts either flat kwargs, a mapping, or both, and
routes values onto nested fields so deep attribute paths are optional::

    cfg = DataBankConfig.make(
        backend="uring",                     # → io_config.backend
        entries=256,                         # → io_config.uring_config.entries
        cache_capacity_bytes=512 * 1024**2,  # → access_config.cache_capacity_bytes
        decode__num_workers=16,              # disambiguated by a ``__`` path
    )
    cfg = DataBankConfig.from_dict({
        "io": {"backend": "uring", "uring": {"entries": 256}},
        "access": {"cache_capacity_bytes": 512 * 1024**2},
    })
    cfg.update(fill__num_workers=8)          # in-place, chainable

Routing rules (see :func:`_apply_dynamic`):

* a bare key matching a direct field or short ``*_config`` form is set on this
  config, and nested config fields accept either config objects or mappings;
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

from collections.abc import Iterable as IterableABC, Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass, field, fields
from functools import lru_cache
from os import PathLike, fspath
from typing import (
    Any,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    TypeVar,
    cast,
    get_type_hints,
    overload,
)

from .data._cell import CellAccess, CellBatch, CellData, _as_cell_index
from .data._coerce import _as_gene_names
from .data._dataset import DataError, Dataset, DatasetCollection, DenseDataset, DType, SparseDataset
from .data._index import CellIndexPlan
from .data._prefetch import PrefetchIterator
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
    _NativeAccessConfig,
    _NativeBloscConfig,
    _NativeLoadCoalesceConfig,
    _NativeLoadConfig,
    _PrefetchPlan,
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
    "NativeAccessConfig",
    "NativeLoadConfig",
    "NativeLoadCoalesceConfig",
    "NativeBloscConfig",
    "NativeMode",
    "ScheduledAccessConfig",
    "ScheduledPrefetchConfig",
    "ProjectedSparseDataGroupStrategy",
    "ProfileSnapshot",
]


ProfileSnapshot = dict[str, Any]
ProjectedSparseDataGroupStrategy = Literal["selected_only", "read_all"]
NativeMode = Literal["disabled", "auto", "force"]


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


def _coerce_missing_policy(
    missing: MissingGenePolicy | Literal["zero", "error", "raise", "strict"] | str | None,
) -> MissingGenePolicy | None:
    """Normalize a public missing-gene policy value."""
    if missing is None or isinstance(missing, MissingGenePolicy):
        return missing
    text = str(missing).strip().lower()
    if text in ("zero", "zeros"):
        return MissingGenePolicy.ZERO
    if text in ("error", "raise", "strict"):
        return MissingGenePolicy.ERROR
    raise ValueError(
        "missing must be MissingGenePolicy.ZERO, MissingGenePolicy.ERROR, "
        f"'zero', or 'error'; got {missing!r}"
    )


def _coerce_dtype(dtype: DType | str | None) -> DType | None:
    """Normalize a public dtype value."""
    if dtype is None or isinstance(dtype, DType):
        return dtype
    text = str(dtype).strip()
    folded = text.lower()
    for candidate in DType:
        if folded == candidate.value or text.upper() == candidate.name:
            return candidate
    return DType.parse(dtype)


_FLOAT_DTYPE_RANK = {
    DType.F16: 1,
    DType.BF16: 1,
    DType.F32: 2,
    DType.F64: 3,
}

_INT_DTYPE_RANK = {
    DType.U8: (1, False),
    DType.I8: (1, True),
    DType.U16: (2, False),
    DType.I16: (2, True),
    DType.U32: (4, False),
    DType.I32: (4, True),
    DType.U64: (8, False),
    DType.I64: (8, True),
}


def _is_float_dtype(dtype: DType) -> bool:
    return dtype in _FLOAT_DTYPE_RANK


def _auto_output_dtype(dtypes: Iterable[DType]) -> DType:
    values = tuple(dtypes)
    if not values:
        raise ValueError("cannot infer output dtype from an empty dataset list")
    floats = [dtype for dtype in values if _is_float_dtype(dtype)]
    if floats:
        if DType.F64 in floats:
            return DType.F64
        if DType.F32 in floats:
            return DType.F32
        if DType.F16 in floats and DType.BF16 in floats:
            return DType.F32
        return floats[0]

    max_size = max(_INT_DTYPE_RANK[dtype][0] for dtype in values)
    same_width = [dtype for dtype in values if _INT_DTYPE_RANK[dtype][0] == max_size]
    signed = [dtype for dtype in same_width if _INT_DTYPE_RANK[dtype][1]]
    return signed[0] if signed else same_width[0]


def _coerce_prefetch_dtype(
    dtype: DType | Literal["auto"] | str | None, ids: Iterable[DatasetId], bank: "ScDataBank"
) -> DType:
    if dtype is None or (isinstance(dtype, str) and dtype.strip().lower() == "auto"):
        return _auto_output_dtype(bank.dataset_dtype(id) for id in ids)
    value = _coerce_dtype(dtype)
    if value is None:
        raise ValueError("dtype must not resolve to None")
    return value


# ===========================================================================
# Config dataclasses
# ===========================================================================
# Every config is a pure `@dataclass` with Python-owned defaults.  They carry
# **no** `_to_rust()` of their own:
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

    __slots__ = ()

    @classmethod
    def make(cls: type[_C], data: Mapping[str, Any] | None = None, /, **kwargs: Any) -> _C:
        """Build a config from a mapping and/or flat kwargs.

        See the module docstring for the routing rules.
        """
        values = _merge_mapping_and_kwargs(cls, data, kwargs)
        cfg = cls()
        _apply_dynamic(cfg, values)
        return cfg

    @classmethod
    def from_dict(
        cls: type[_C],
        data: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> _C:
        """Alias for :meth:`make`, for explicit dict-based construction."""
        return cls.make(data, **kwargs)

    def update(
        self: _C,
        data: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> _C:
        """Mutate this config in place from a mapping and/or flat kwargs.

        See the module docstring for the routing rules.
        """
        values = _merge_mapping_and_kwargs(type(self), data, kwargs)
        _apply_dynamic(self, values)
        return self

    def validate(self) -> None:
        """Validate this config through the Rust binding."""
        _config_to_rust(self).validate()


@dataclass(slots=True)
class BaseIoConfig(_Config):
    """Shared IO backend settings (queue capacity, max in-flight, ...)."""

    max_in_flight: int = 768
    queue_capacity: int = 4096
    priority_levels: int = 3
    queue_shards: int = 8
    assume_non_overlapping_reads: bool = False


@dataclass(slots=True)
class UringConfig(_Config):
    """io_uring backend settings."""

    entries: int = 256
    drivers: int = 8
    iowq_bounded_workers: int = 0
    iowq_unbounded_workers: int = 0
    registered_files: int = 4096
    base: BaseIoConfig = field(default_factory=BaseIoConfig)

    def __post_init__(self) -> None:
        self.base = _coerce_config_value(self.base, BaseIoConfig, "base")


@dataclass(slots=True)
class ThreadedConfig(_Config):
    """Thread-pool pread/pwrite backend settings."""

    num_workers: int = 24
    cpus: list[int] | None = None
    base: BaseIoConfig = field(default_factory=BaseIoConfig)

    def __post_init__(self) -> None:
        self.base = _coerce_config_value(self.base, BaseIoConfig, "base")


@dataclass(slots=True)
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

    def __post_init__(self) -> None:
        self.uring_config = _coerce_config_value(
            self.uring_config,
            UringConfig,
            "uring_config",
        )
        self.threaded_config = _coerce_config_value(
            self.threaded_config,
            ThreadedConfig,
            "threaded_config",
        )
        if self.backend not in ("uring", "threaded"):
            raise ValueError(
                f"IoConfig.backend must be 'uring' or 'threaded', got {self.backend!r}"
            )

    @classmethod
    def uring(
        cls,
        config: UringConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "IoConfig":
        """Create an IoConfig with the uring backend.

        ``config`` optionally supplies a whole :class:`UringConfig`; extra
        kwargs are routed onto it (e.g. ``entries=256``,
        ``base__max_in_flight=2048``).
        """
        config = (
            UringConfig()
            if config is None
            else _coerce_config_value(
                config,
                UringConfig,
                "config",
            )
        )
        _apply_dynamic(config, kwargs)
        return cls(backend="uring", uring_config=config)

    @classmethod
    def threaded(
        cls,
        config: ThreadedConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "IoConfig":
        """Create an IoConfig with the threaded backend.

        ``config`` optionally supplies a whole :class:`ThreadedConfig`; extra
        kwargs are routed onto it (e.g. ``num_workers=16``).
        """
        config = (
            ThreadedConfig()
            if config is None
            else _coerce_config_value(
                config,
                ThreadedConfig,
                "config",
            )
        )
        _apply_dynamic(config, kwargs)
        return cls(backend="threaded", threaded_config=config)

    @property
    def kind(self) -> str:
        return self.backend


@dataclass(slots=True)
class DecodePoolConfig(_Config):
    """Decode worker pool settings."""

    num_workers: int = 24
    queue_capacity: int = 1024
    cpus: list[int] | None = None


@dataclass(slots=True)
class AccessCpuConfig(_Config):
    """Access-side CPU materialization pool settings."""

    num_workers: int = 12
    queue_capacity: int = 1024
    cpus: list[int] | None = None


@dataclass(slots=True)
class AccessConfig(_Config):
    """Access scheduler settings (cache, memory budget, shards, ...)."""

    queue_capacity: int = 1024
    scheduler_shards: int = 8
    # Conservative defaults so a freshly-constructed bank does not risk OOM on
    # a typical dev box (16-32 GiB). These are upper bounds, not preallocated;
    # scale them up explicitly on a big-memory training host.
    cache_capacity_bytes: int = 4 * 1024**3
    memory_budget_bytes: int = 8 * 1024**3
    default_io_priority: int = 1
    keep_decoded: bool = False
    cpu: AccessCpuConfig = field(default_factory=AccessCpuConfig)

    def __post_init__(self) -> None:
        self.cpu = _coerce_config_value(self.cpu, AccessCpuConfig, "cpu")


@dataclass(slots=True)
class FillConfig(_Config):
    """Compute / fill pool settings."""

    parallel: bool = True
    num_workers: int = 12
    queue_capacity: int = 1024
    min_parallel_rows: int = 16
    min_parallel_bytes: int = 1048576
    cpus: list[int] | None = None


@dataclass(slots=True)
class NativeLoadCoalesceConfig(_Config):
    """Native byte-range coalescing policy."""

    max_window_us: int = 50
    max_merged_len: int = 1024 * 1024
    max_gap_bytes: int = 16 * 1024
    max_waste_ratio: float = 0.10
    min_children: int = 2


@dataclass(slots=True)
class NativeLoadConfig(_Config):
    """Native byte-range load scheduler settings."""

    scheduler_workers: int = 1
    io_workers: int = 4
    coalesce: NativeLoadCoalesceConfig = field(default_factory=NativeLoadCoalesceConfig)

    def __post_init__(self) -> None:
        self.coalesce = _coerce_config_value(
            self.coalesce,
            NativeLoadCoalesceConfig,
            "coalesce",
        )


@dataclass(slots=True)
class NativeBloscConfig(_Config):
    """Native Blosc-LZ4 block handling settings."""

    preload_block_tables: bool = True
    full_unshuffle_threshold: float = 0.75
    max_block_size: int = 2 * 1024 * 1024


@dataclass(slots=True)
class NativeAccessConfig(_Config):
    """Experimental Blosc-LZ4 native access path settings."""

    enabled: bool = False
    fallback_to_generic: bool = True
    fused_workers: int = 4
    request_prefetch_batches: int = 8
    request_prefetch_blocks: int = 4096
    memory_budget_bytes: int = 8 * 1024**3
    response_queue_bytes_soft_limit: int = 4 * 1024**3
    response_queue_bytes_hard_limit: int = 6 * 1024**3
    load: NativeLoadConfig = field(default_factory=NativeLoadConfig)
    blosc: NativeBloscConfig = field(default_factory=NativeBloscConfig)

    def __post_init__(self) -> None:
        self.load = _coerce_config_value(self.load, NativeLoadConfig, "load")
        self.blosc = _coerce_config_value(self.blosc, NativeBloscConfig, "blosc")


@dataclass(slots=True)
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
    native_config: NativeAccessConfig = field(default_factory=NativeAccessConfig)

    def __post_init__(self) -> None:
        self.io_config = _coerce_config_value(self.io_config, IoConfig, "io_config")
        self.decode_config = _coerce_config_value(
            self.decode_config,
            DecodePoolConfig,
            "decode_config",
        )
        self.access_config = _coerce_config_value(
            self.access_config,
            AccessConfig,
            "access_config",
        )
        self.fill_config = _coerce_config_value(self.fill_config, FillConfig, "fill_config")
        self.native_config = _coerce_config_value(
            self.native_config,
            NativeAccessConfig,
            "native_config",
        )


@dataclass(slots=True)
class ScheduledAccessConfig(_Config):
    """Look-ahead distances for scheduled access."""

    prefetch_step: int = 16
    decode_ahead_steps: int = 8
    ready_ahead_steps: int = 4


@dataclass(slots=True)
class ScheduledPrefetchConfig(_Config):
    """Per-call settings for scheduled DataBank cell prefetch."""

    prefetch_step: int = 8
    access: ScheduledAccessConfig = field(default_factory=ScheduledAccessConfig)
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy = "selected_only"
    native_mode: NativeMode = "disabled"

    def __post_init__(self) -> None:
        self.access = _coerce_config_value(self.access, ScheduledAccessConfig, "access")
        allowed = {"selected_only", "read_all", "selected", "selected-only", "read-all", "all"}
        if self.projected_sparse_data_strategy not in allowed:
            raise ValueError(
                "projected_sparse_data_strategy must be 'selected_only' or 'read_all', "
                f"got {self.projected_sparse_data_strategy!r}"
            )
        native_allowed = {"disabled", "auto", "force", "off", "on", "true", "false", "forced"}
        if self.native_mode not in native_allowed:
            raise ValueError(
                "native_mode must be 'disabled', 'auto', or 'force', "
                f"got {self.native_mode!r}"
            )


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
        NativeLoadCoalesceConfig,
        NativeLoadConfig,
        NativeBloscConfig,
        NativeAccessConfig,
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
    NativeLoadCoalesceConfig: _NativeLoadCoalesceConfig,
    NativeLoadConfig: _NativeLoadConfig,
    NativeBloscConfig: _NativeBloscConfig,
    NativeAccessConfig: _NativeAccessConfig,
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
        if config.backend == "threaded":
            return _IoConfig.threaded(_config_to_rust(config.threaded_config))
        raise ValueError(f"IoConfig.backend must be 'uring' or 'threaded', got {config.backend!r}")
    rust_cls = _RUST_CONFIG_TYPES.get(cls)
    if rust_cls is None:
        raise TypeError(f"not a config type: {cls.__name__}")
    r = rust_cls()
    for f in fields(config):
        value = getattr(config, f.name)
        if type(value) in _CONFIG_CLASSES:
            value = _config_to_rust(value)
        if not hasattr(r, f.name):
            if (
                cls is ScheduledPrefetchConfig
                and f.name == "projected_sparse_data_strategy"
                and value in {"selected_only", "selected", "selected-only"}
            ):
                continue
            raise AttributeError(
                f"Rust extension {type(r).__name__} has no config field {f.name!r}; "
                "rebuild scdata._scdata"
            )
        setattr(r, f.name, value)
    return r


def _config_from_rust(config: Any) -> Any:
    """Convert a Rust config pyclass back into the public Python dataclass."""
    if isinstance(config, _DataBankConfig):
        return DataBankConfig(
            io_config=_config_from_rust(config.io_config),
            decode_config=_config_from_rust(config.decode_config),
            access_config=_config_from_rust(config.access_config),
            fill_config=_config_from_rust(config.fill_config),
            native_config=_config_from_rust(config.native_config),
        )
    if isinstance(config, _IoConfig):
        if config.kind == "uring":
            uring = config.uring_config()
            if uring is None:
                raise ValueError("Rust IoConfig kind is 'uring' but uring_config() returned None")
            return IoConfig(backend="uring", uring_config=_config_from_rust(uring))
        if config.kind == "threaded":
            threaded = config.threaded_config()
            if threaded is None:
                raise ValueError(
                    "Rust IoConfig kind is 'threaded' but threaded_config() returned None"
                )
            return IoConfig(backend="threaded", threaded_config=_config_from_rust(threaded))
        raise ValueError(f"unknown Rust IoConfig kind {config.kind!r}")
    if isinstance(config, _BaseIoConfig):
        return BaseIoConfig(
            max_in_flight=config.max_in_flight,
            queue_capacity=config.queue_capacity,
            priority_levels=config.priority_levels,
            queue_shards=config.queue_shards,
            assume_non_overlapping_reads=config.assume_non_overlapping_reads,
        )
    if isinstance(config, _UringConfig):
        return UringConfig(
            entries=config.entries,
            drivers=config.drivers,
            iowq_bounded_workers=config.iowq_bounded_workers,
            iowq_unbounded_workers=config.iowq_unbounded_workers,
            registered_files=config.registered_files,
            base=_config_from_rust(config.base),
        )
    if isinstance(config, _ThreadedConfig):
        return ThreadedConfig(
            num_workers=config.num_workers,
            cpus=config.cpus,
            base=_config_from_rust(config.base),
        )
    if isinstance(config, _DecodePoolConfig):
        return DecodePoolConfig(
            num_workers=config.num_workers,
            queue_capacity=config.queue_capacity,
            cpus=config.cpus,
        )
    if isinstance(config, _AccessCpuConfig):
        return AccessCpuConfig(
            num_workers=config.num_workers,
            queue_capacity=config.queue_capacity,
            cpus=config.cpus,
        )
    if isinstance(config, _AccessConfig):
        return AccessConfig(
            queue_capacity=config.queue_capacity,
            scheduler_shards=config.scheduler_shards,
            cache_capacity_bytes=config.cache_capacity_bytes,
            memory_budget_bytes=config.memory_budget_bytes,
            default_io_priority=config.default_io_priority,
            keep_decoded=config.keep_decoded,
            cpu=_config_from_rust(config.cpu),
        )
    if isinstance(config, _FillConfig):
        return FillConfig(
            parallel=config.parallel,
            num_workers=config.num_workers,
            queue_capacity=config.queue_capacity,
            min_parallel_rows=config.min_parallel_rows,
            min_parallel_bytes=config.min_parallel_bytes,
            cpus=config.cpus,
        )
    if isinstance(config, _NativeLoadCoalesceConfig):
        return NativeLoadCoalesceConfig(
            max_window_us=config.max_window_us,
            max_merged_len=config.max_merged_len,
            max_gap_bytes=config.max_gap_bytes,
            max_waste_ratio=config.max_waste_ratio,
            min_children=config.min_children,
        )
    if isinstance(config, _NativeLoadConfig):
        return NativeLoadConfig(
            scheduler_workers=config.scheduler_workers,
            io_workers=config.io_workers,
            coalesce=_config_from_rust(config.coalesce),
        )
    if isinstance(config, _NativeBloscConfig):
        return NativeBloscConfig(
            preload_block_tables=config.preload_block_tables,
            full_unshuffle_threshold=config.full_unshuffle_threshold,
            max_block_size=config.max_block_size,
        )
    if isinstance(config, _NativeAccessConfig):
        return NativeAccessConfig(
            enabled=config.enabled,
            fallback_to_generic=config.fallback_to_generic,
            fused_workers=config.fused_workers,
            request_prefetch_batches=config.request_prefetch_batches,
            request_prefetch_blocks=config.request_prefetch_blocks,
            memory_budget_bytes=config.memory_budget_bytes,
            response_queue_bytes_soft_limit=config.response_queue_bytes_soft_limit,
            response_queue_bytes_hard_limit=config.response_queue_bytes_hard_limit,
            load=_config_from_rust(config.load),
            blosc=_config_from_rust(config.blosc),
        )
    if isinstance(config, _ScheduledAccessConfig):
        return ScheduledAccessConfig(
            prefetch_step=config.prefetch_step,
            decode_ahead_steps=config.decode_ahead_steps,
            ready_ahead_steps=config.ready_ahead_steps,
        )
    if isinstance(config, _ScheduledPrefetchConfig):
        return ScheduledPrefetchConfig(
            prefetch_step=config.prefetch_step,
            access=_config_from_rust(config.access),
            projected_sparse_data_strategy=cast(
                ProjectedSparseDataGroupStrategy,
                getattr(config, "projected_sparse_data_strategy", "selected_only"),
            ),
            native_mode=cast(NativeMode, getattr(config, "native_mode", "disabled")),
        )
    raise TypeError(f"not a Rust config object: {type(config).__name__}")


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
        (f.name, hints[f.name]) for f in fields(cls) if hints.get(f.name) in _CONFIG_CLASSES
    )


@lru_cache(maxsize=None)
def _config_leaf_names(cls: Any) -> frozenset[str]:
    """All field names reachable from ``cls`` (its own + nested config leaves)."""
    names: set[str] = {f.name for f in fields(cls)}
    for _fname, sub in _config_subfields(cls):
        names |= _config_leaf_names(sub)
    return frozenset(names)


@lru_cache(maxsize=None)
def _config_leaf_paths(cls: Any, leaf_name: str) -> tuple[tuple[str, ...], ...]:
    """All field paths from ``cls`` to a reachable field named ``leaf_name``."""
    paths: list[tuple[str, ...]] = []
    direct = {f.name for f in fields(cls)}
    if leaf_name in direct:
        paths.append((leaf_name,))
    for fname, sub in _config_subfields(cls):
        for subpath in _config_leaf_paths(sub, leaf_name):
            paths.append((fname, *subpath))
    return tuple(paths)


def _merge_mapping_and_kwargs(
    cls: Any,
    data: Mapping[str, Any] | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge optional mapping input with kwargs, validating mapping shape."""
    if data is None:
        values: dict[str, Any] = {}
    elif isinstance(data, MappingABC):
        values = dict(data)
    else:
        raise TypeError(f"{cls.__name__}.make() expected a mapping, got {type(data).__name__}")
    values.update(kwargs)
    return values


def _coerce_config_value(value: Any, target_cls: type[_C], field_name: str) -> _C:
    """Coerce nested config mappings into their dataclass config object."""
    if isinstance(value, target_cls):
        return value
    if isinstance(value, MappingABC):
        return target_cls.from_dict(value)
    raise TypeError(
        f"{field_name} must be {target_cls.__name__} or a mapping, got {type(value).__name__}"
    )


def _resolve_field_or_none(cls: Any, segment: str) -> str | None:
    """Resolve a direct field name or short ``*_config`` alias, if present."""
    names = {f.name for f in fields(cls)}
    if segment in names:
        return segment
    candidate = f"{segment}_config"
    if candidate in names:
        return candidate
    return None


def _resolve_path_field(cls: Any, segment: str) -> str:
    """Match a ``__`` path segment to a config field name.

    Accepts the literal field name (``io_config``) or its short form with a
    trailing ``_config`` stripped (``io``), so paths stay readable:
    ``io__uring__entries`` rather than ``io_config__uring_config__entries``.
    """
    resolved = _resolve_field_or_none(cls, segment)
    if resolved is not None:
        return resolved
    raise TypeError(f"{cls.__name__} has no field matching path segment {segment!r}")


def _coerce_config_assignment(cls: Any, field_name: str, value: Any) -> Any:
    """Coerce a value assigned to ``cls.field_name`` when it is a config field."""
    hint = _config_type_hints(cls).get(field_name)
    if hint in _CONFIG_CLASSES:
        return _coerce_config_value(value, hint, field_name)
    return value


def _assign_config_field(config: Any, field_name: str, value: Any) -> None:
    """Assign a config field with nested-dict coercion and local validation."""
    setattr(config, field_name, _coerce_config_assignment(type(config), field_name, value))
    _validate_config_object(config)


def _path_hint(path: Iterable[str]) -> str:
    """Render a config path using the public short-form convention."""
    return "__".join(part.removesuffix("_config") for part in path)


def _set_config_path(config: Any, path: tuple[str, ...], value: Any) -> None:
    """Set ``value`` at a resolved config field path."""
    obj = config
    for field_name in path[:-1]:
        obj = getattr(obj, field_name)
        if type(obj) not in _CONFIG_CLASSES:
            raise TypeError(f"{type(obj).__name__} at {_path_hint(path)} is not a config object")
    _assign_config_field(obj, path[-1], value)


def _validate_config_object(config: Any) -> None:
    """Run lightweight Python-side validation for dynamically updated configs."""
    if isinstance(config, IoConfig):
        config.__post_init__()


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
    for key, value in kwargs.items():
        if "__" in key:
            segments = key.split("__")
            obj: Any = config
            obj_cls = cls
            for seg in segments[:-1]:
                fname = _resolve_path_field(obj_cls, seg)
                obj = getattr(obj, fname)
                obj_cls = type(obj)
            _assign_config_field(obj, _resolve_path_field(obj_cls, segments[-1]), value)
            continue
        field_name = _resolve_field_or_none(cls, key)
        if field_name in direct:
            _assign_config_field(config, field_name, value)
            continue
        hits = [path for path in _config_leaf_paths(cls, key) if len(path) > 1]
        if len(hits) == 1:
            _set_config_path(config, hits[0], value)
        elif not hits:
            raise TypeError(f"{cls.__name__} has no field {key!r}")
        else:
            suggestions = ", ".join(_path_hint(path) for path in hits)
            raise TypeError(
                f"{cls.__name__}: {key!r} is ambiguous across {suggestions}; "
                f"qualify with a path like {_path_hint(hits[0])}"
            )
    return config


def _coerce_prefetch_config(
    config: ScheduledPrefetchConfig | Mapping[str, Any] | None,
) -> ScheduledPrefetchConfig:
    """Normalize optional prefetch config input before Rust conversion."""
    if config is None:
        return ScheduledPrefetchConfig()
    if isinstance(config, ScheduledPrefetchConfig):
        return config
    if isinstance(config, MappingABC):
        return ScheduledPrefetchConfig.from_dict(config)
    raise TypeError(
        f"config must be ScheduledPrefetchConfig, a mapping, or None; got {type(config).__name__}"
    )


def _coerce_access_config(
    config: ScheduledAccessConfig | Mapping[str, Any] | None,
) -> ScheduledAccessConfig:
    """Normalize optional per-access config input before Rust conversion."""
    if config is None:
        return ScheduledAccessConfig()
    if isinstance(config, ScheduledAccessConfig):
        return config
    if isinstance(config, MappingABC):
        return ScheduledAccessConfig.from_dict(config)
    raise TypeError(
        f"access_config must be ScheduledAccessConfig, a mapping, or None; "
        f"got {type(config).__name__}"
    )


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

    _inner: _DataBank | None

    def __init__(self, config: DataBankConfig | Mapping[str, Any] | None = None) -> None:
        if config is None:
            config = DataBankConfig()
        elif isinstance(config, MappingABC):
            config = DataBankConfig.from_dict(config)
        self._inner = _DataBank(_config_to_rust(config))
        self._registered_count = 0
        # Per-dataset ``(gene_names, num_genes)`` cache, keyed by DatasetId.
        # Load results carry num_genes / gene_names, and ``load``
        # needs num_genes to validate shape before Rust returns the payload —
        # caching avoids a Rust round-trip on every call.
        self._meta_cache: dict[DatasetId, tuple[list[str], int]] = {}

    def _core(self) -> _DataBank:
        inner = self._inner
        if inner is None:
            raise RuntimeError("ScDataBank is closed")
        return inner

    @property
    def is_closed(self) -> bool:
        """Whether this bank has already released its Rust core."""
        return self._inner is None

    @property
    def config(self) -> DataBankConfig:
        """A Python dataclass copy of the running Rust bank config."""
        return _config_from_rust(self._core().config())

    def profile_snapshot(self) -> ProfileSnapshot:
        """Return a structured snapshot of DataBank profiling metrics."""
        return self._core().profile_snapshot()

    def profile_snapshot_and_reset(self) -> ProfileSnapshot:
        """Return profiling metrics and reset the active counters."""
        return self._core().profile_snapshot_and_reset()

    def reset_profile(self) -> None:
        """Reset DataBank profiling metrics."""
        self._core().reset_profile()

    # -- registration --------------------------------------------------------

    @overload
    def register(self, dataset: Dataset) -> DatasetId: ...

    @overload
    def register(self, dataset: Iterable[Dataset]) -> list[DatasetId]: ...

    def register(self, dataset: Dataset | Iterable[Dataset]) -> DatasetId | list[DatasetId]:
        """Register parsed dataset metadata and return databank handles.

        The dataset's :attr:`~scdata.data.DenseDataset.store_root` supplies the
        filesystem path the Rust databank opens, so a dataset produced by
        :func:`scdata.io.launch` carries everything the bank needs.  Dense and
        CSR-sparse datasets dispatch to the matching Rust entry point by type.
        Passing an iterable of datasets registers them in order and returns a
        ``list[DatasetId]``.

        For datasets assembled without ``store_root`` (e.g. by hand in tests),
        use :meth:`register_dense` / :meth:`register_sparse_csr` and pass the
        store path explicitly.
        """
        if not isinstance(dataset, (DenseDataset, SparseDataset)):
            if isinstance(dataset, IterableABC) and not isinstance(
                dataset, (str, bytes, bytearray)
            ):
                return self._register_many(dataset)
            raise TypeError(f"unsupported dataset type: {type(dataset).__name__}")
        return self._register_from_dataset(dataset)

    def _register_many(self, datasets: Iterable[Dataset]) -> list[DatasetId]:
        registered: list[DatasetId] = []
        try:
            for dataset in datasets:
                registered.append(self._register_from_dataset(dataset))
        except Exception:
            for did in reversed(registered):
                self.unregister(did)
            raise
        return registered

    def _register_from_dataset(self, dataset: Dataset) -> DatasetId:
        if not isinstance(dataset, (DenseDataset, SparseDataset)):
            raise TypeError(f"unsupported dataset type: {type(dataset).__name__}")
        store_root = dataset.store_root
        if not store_root:
            raise ValueError(
                "dataset has no store_root; pass it to register_dense / "
                "register_sparse_csr, or build it via scdata.io.launch"
            )
        return self._register(dataset, store_root)

    def register_dense(self, ds: DenseDataset, store_path: str | PathLike[str]) -> DatasetId:
        """Register a dense dataset parsed by :func:`scdata.io.launch`.

        ``store_path`` is the filesystem path to the zarr directory or
        ``.zarr.zip`` archive.  Datasets returned by :func:`scdata.io.launch`
        already carry resolved source files; manually-built datasets use
        ``store_path`` as the root for their logical payload/chunk keys.
        """
        rust_id = self._core().register_dense(ds, fspath(store_path))
        did = DatasetId(rust_id)
        self._registered_count += 1
        return did

    def register_sparse_csr(self, ds: SparseDataset, store_path: str | PathLike[str]) -> DatasetId:
        """Register a CSR sparse dataset parsed by :func:`scdata.io.launch`."""
        rust_id = self._core().register_sparse_csr(ds, fspath(store_path))
        did = DatasetId(rust_id)
        self._registered_count += 1
        return did

    def _register(self, ds: Dataset, store_path: str | PathLike[str]) -> DatasetId:
        """Register a dataset by type, dispatching to the matching Rust entry point."""
        if isinstance(ds, DenseDataset):
            return self.register_dense(ds, store_path)
        if isinstance(ds, SparseDataset):
            return self.register_sparse_csr(ds, store_path)
        raise TypeError(f"unsupported dataset type: {type(ds).__name__}")

    def register_all(self, datasets: DatasetCollection) -> dict[str, DatasetId]:
        """Register ``X`` and all layers from a :class:`DatasetCollection`.

        Returns a mapping keyed by matrix path: ``"X"`` and
        ``"layers/<name>"``.  If any layer fails to register, all datasets
        registered by this call are unregistered before re-raising.
        """
        if not isinstance(datasets, DatasetCollection):
            raise TypeError(f"unsupported dataset collection type: {type(datasets).__name__}")
        registered: dict[str, DatasetId] = {}
        try:
            for key, dataset in datasets.items():
                registered[key] = self.register(dataset)
        except Exception:
            for did in reversed(tuple(registered.values())):
                self.unregister(did)
            raise
        return registered

    def unregister_all(self, ids: Mapping[str, DatasetId] | Iterable[DatasetId]) -> None:
        """Unregister multiple dataset ids, ignoring no ids."""
        values = ids.values() if isinstance(ids, Mapping) else ids
        for did in tuple(values):
            self.unregister(did)

    def unregister(self, id: DatasetId) -> None:
        """Unregister a dataset, releasing its file handles and gene refs."""
        self._core().unregister(id._rust)
        self._meta_cache.pop(id, None)
        self._registered_count = max(0, self._registered_count - 1)

    # -- queries -------------------------------------------------------------

    def dataset_genes(self, id: DatasetId) -> list[str]:
        """Gene names for ``id``, in column order matching access results."""
        return self._meta(id)[0]

    def dataset_num_cells(self, id: DatasetId) -> int:
        """Number of cells (rows) in the registered dataset."""
        return self._core().dataset_num_cells(id._rust)

    def dataset_num_genes(self, id: DatasetId) -> int:
        """Number of genes (columns) in the registered dataset."""
        return self._meta(id)[1]

    def dataset_dtype(self, id: DatasetId) -> DType:
        """Stored value dtype of the registered dataset (a ``DType``)."""
        return self._core().dataset_dtype(id._rust)

    def _meta(self, id: DatasetId) -> tuple[list[str], int]:
        """Cached ``(gene_names, num_genes)`` for ``id``, fetched on first use."""
        cached = self._meta_cache.get(id)
        if cached is None:
            core = self._core()
            genes = core.dataset_genes(id._rust)
            num_genes = core.dataset_num_genes(id._rust)
            cached = (genes, num_genes)
            self._meta_cache[id] = cached
        return cached

    # -- cell access ---------------------------------------------------------

    def load(
        self,
        id: DatasetId,
        cells: CellAccess | Iterable[int],
        genes: Iterable[str] | None = None,
        *,
        missing: MissingGenePolicy
        | Literal["zero", "error", "raise", "strict"]
        | str
        | None = None,
        dtype: DType | Literal["auto"] | str | None = None,
        access_config: ScheduledAccessConfig | Mapping[str, Any] | None = None,
    ) -> CellData:
        """Load cells from one dataset, optionally projected onto a gene subset.

        ``genes=None`` returns all genes in dataset order.  Passing ``genes``
        dispatches to the Rust projection path and returns columns in the
        requested order.  ``cells`` may also be a :class:`CellAccess`; when
        ``genes`` is omitted, its ``gene_names`` are used.

        For a batch mixing cells from several datasets, use
        :meth:`load_multi` instead.
        """
        if not isinstance(id, DatasetId):
            raise TypeError(
                "load() takes a single DatasetId; for a sequence of dataset "
                "ids use load_multi(ids, batch, ...)"
            )
        if isinstance(cells, CellAccess):
            if genes is None:
                genes = cells.gene_names
            cells = cells.cells
        cell_iter = cast(Iterable[int], cells)
        dtype_value = (
            None
            if isinstance(dtype, str) and dtype.strip().lower() == "auto"
            else _coerce_dtype(dtype)  # type: ignore[arg-type]
        )
        if genes is None:
            return self._load_all_genes(
                id, cell_iter, dtype=dtype_value, access_config=access_config
            )
        return self._load_genes(
            id,
            cell_iter,
            genes,
            missing=_coerce_missing_policy(missing),
            dtype=dtype_value,
            access_config=access_config,
        )

    def load_multi(
        self,
        ids: SequenceABC[DatasetId],
        batch: Iterable[tuple[int, CellAccess | Iterable[int]]],
        genes: Iterable[str] | None = None,
        *,
        missing: MissingGenePolicy
        | Literal["zero", "error", "raise", "strict"]
        | str
        | None = None,
        dtype: DType | Literal["auto"] | str | None = None,
        access_config: ScheduledAccessConfig | Mapping[str, Any] | None = None,
    ) -> CellData:
        """Load one multi-dataset batch of cells, optionally projected onto ``genes``.

        ``ids`` is a sequence of :class:`DatasetId`.  ``batch`` is one multi-part
        batch of ``(dataset_idx, cells)`` pairs, where ``dataset_idx`` indexes
        into ``ids``; the result rows follow the input part order.  This is the
        one-shot equivalent of :meth:`prefetch_multi` for a single batch.
        ``dtype=None``/``"auto"`` selects one output dtype across all datasets.
        """
        prefetch_config = None
        if access_config is not None:
            prefetch_config = ScheduledPrefetchConfig(
                prefetch_step=1,
                access=_coerce_access_config(access_config),
            )
        iterator = self.prefetch_multi(
            ids,
            [batch],
            genes=genes,
            missing=missing,
            dtype=dtype,
            config=prefetch_config,
        )
        try:
            decoded = next(iterator)
        except StopIteration as exc:
            raise DataError("multi-dataset load produced no batch") from exc
        return CellData.from_array(
            cells=decoded.cells,
            data=decoded.data,
            num_genes=decoded.num_genes,
            gene_names=decoded.var_names,
        )

    def _load_all_genes(
        self,
        id: DatasetId,
        cells: Iterable[int],
        *,
        dtype: DType | None = None,
        access_config: ScheduledAccessConfig | Mapping[str, Any] | None = None,
    ) -> CellData:
        cell_arr = _as_cell_index(cells, "cells")
        gene_names, num_genes = self._meta(id)
        core = self._core()
        rust_config = _config_to_rust(_coerce_access_config(access_config))
        data = core.access_cells_array(id._rust, cell_arr, dtype, rust_config)
        return CellData.from_array(
            cells=cell_arr,
            data=data,
            num_genes=num_genes,
            gene_names=tuple(gene_names),
        )

    def _load_genes(
        self,
        id: DatasetId,
        cells: Iterable[int],
        genes: Iterable[str],
        *,
        missing: MissingGenePolicy | None = None,
        dtype: DType | None = None,
        access_config: ScheduledAccessConfig | Mapping[str, Any] | None = None,
    ) -> CellData:
        cell_arr = _as_cell_index(cells, "cells")
        names = _as_gene_names(genes, "genes")
        core = self._core()
        rust_missing = missing._rust if missing is not None else None
        rust_config = _config_to_rust(_coerce_access_config(access_config))
        data = core.access_cells_by_gene_names_array(
            id._rust,
            cell_arr,
            list(names),
            rust_missing,
            dtype,
            rust_config,
        )
        return CellData.from_array(
            cells=cell_arr,
            data=data,
            num_genes=len(names),
            gene_names=names,
        )

    # -- prefetch ------------------------------------------------------------

    def prefetch_cells_cache(self, id: DatasetId, cells: Iterable[int]) -> None:
        """Warm the Rust access cache for ``cells`` without returning data."""
        cell_arr = _as_cell_index(cells, "cells")
        self._core().prefetch_cells_cache_array(id._rust, cell_arr)

    def prefetch(
        self,
        id: DatasetId,
        batches: Iterable[CellAccess | Iterable[int]],
        genes: Iterable[str] | None = None,
        *,
        missing: MissingGenePolicy
        | Literal["zero", "error", "raise", "strict"]
        | str
        | None = None,
        dtype: DType | Literal["auto"] | str | None = None,
        config: ScheduledPrefetchConfig | Mapping[str, Any] | None = None,
    ) -> Iterator[CellBatch]:
        """Stream decoded cell batches from one dataset, optionally onto ``genes``.

        For cross-dataset batches where one batch may mix cells from several
        datasets, use :meth:`prefetch_multi` instead.
        """
        if not isinstance(id, DatasetId):
            raise TypeError(
                "prefetch() takes a single DatasetId; for a sequence of dataset "
                "ids use prefetch_multi(ids, batches, ...)"
            )
        ids = (id,)
        stored_dtype = self.dataset_dtype(id)
        if dtype is None or (isinstance(dtype, str) and dtype.strip().lower() == "auto"):
            dtype_value = stored_dtype
        else:
            dtype_value = _coerce_prefetch_dtype(dtype, ids, self)
        names = _as_gene_names(genes, "genes") if genes is not None else None
        rust_missing = _coerce_missing_policy(missing)
        config = _coerce_prefetch_config(config)

        if dtype_value == stored_dtype:
            if names is None:
                return self._prefetch_all_genes(id, batches, config=config)
            return self._prefetch_genes(
                id,
                batches,
                names,
                missing=rust_missing,
                config=config,
            )

        # dtype differs from the stored one: plan as a single-dataset multi-style
        # request so the Rust projection path can cast on output.
        plan = _PrefetchPlan.single(batches)
        rust_missing_value = rust_missing._rust if rust_missing is not None else None
        rust_config = _config_to_rust(config)
        inner = self._core().prefetch_cells(
            [id._rust],
            plan,
            dtype_value,
            list(names) if names is not None else None,
            rust_missing_value,
            rust_config,
        )
        return PrefetchIterator(inner, gene_names=tuple(inner.gene_names))

    def prefetch_multi(
        self,
        ids: SequenceABC[DatasetId],
        batches: Iterable[Iterable[tuple[int, CellAccess | Iterable[int]]]],
        genes: Iterable[str] | None = None,
        *,
        missing: MissingGenePolicy
        | Literal["zero", "error", "raise", "strict"]
        | str
        | None = None,
        dtype: DType | Literal["auto"] | str | None = None,
        config: ScheduledPrefetchConfig | Mapping[str, Any] | None = None,
    ) -> Iterator[CellBatch]:
        """Stream decoded cell batches mixing cells from several datasets.

        ``ids`` is a sequence of :class:`DatasetId`.  ``batches`` yields one
        batch per torch batch; each batch is an iterable of
        ``(dataset_idx, cells)`` parts where ``dataset_idx`` indexes into
        ``ids``.  ``dtype=None``/``"auto"`` picks one output dtype across all
        datasets; an explicit ``dtype`` casts every batch to it.
        """
        ids = self._coerce_prefetch_ids(ids)
        dtype_value = _coerce_prefetch_dtype(dtype, ids, self)
        names = _as_gene_names(genes, "genes") if genes is not None else None
        rust_missing = _coerce_missing_policy(missing)
        config = _coerce_prefetch_config(config)
        plan = _PrefetchPlan.multi(batches)
        rust_missing_value = rust_missing._rust if rust_missing is not None else None
        rust_config = _config_to_rust(config)
        inner = self._core().prefetch_cells(
            [dataset_id._rust for dataset_id in ids],
            plan,
            dtype_value,
            list(names) if names is not None else None,
            rust_missing_value,
            rust_config,
        )
        return PrefetchIterator(inner, gene_names=tuple(inner.gene_names))

    def prefetch_indexed(
        self,
        ids: SequenceABC[DatasetId],
        plan: CellIndexPlan,
        genes: Iterable[str] | None = None,
        *,
        missing: MissingGenePolicy
        | Literal["zero", "error", "raise", "strict"]
        | str
        | None = None,
        dtype: DType | Literal["auto"] | str | None = None,
        config: ScheduledPrefetchConfig | Mapping[str, Any] | None = None,
    ) -> Iterator[CellBatch]:
        """Stream batches from a numeric :class:`~scdata.data.CellIndexPlan`.

        ``plan.dataset_index`` selects entries from ``ids`` and
        ``plan.cell_index`` holds local cell ids.  This avoids Python
        per-batch ``(dataset_idx, cells)`` object materialization on the
        scheduled path.
        """
        if not isinstance(plan, CellIndexPlan):
            raise TypeError(f"plan must be a CellIndexPlan, got {type(plan).__name__}")
        ids = self._coerce_prefetch_ids(ids)
        dtype_value = _coerce_prefetch_dtype(dtype, ids, self)
        names = _as_gene_names(genes, "genes") if genes is not None else None
        rust_missing = _coerce_missing_policy(missing)
        config = _coerce_prefetch_config(config)
        rust_plan = _PrefetchPlan.indexed(plan.dataset_index, plan.cell_index, plan.batch_size)
        rust_missing_value = rust_missing._rust if rust_missing is not None else None
        rust_config = _config_to_rust(config)
        inner = self._core().prefetch_cells(
            [dataset_id._rust for dataset_id in ids],
            rust_plan,
            dtype_value,
            list(names) if names is not None else None,
            rust_missing_value,
            rust_config,
        )
        return PrefetchIterator(inner, gene_names=tuple(inner.gene_names))

    @staticmethod
    def _coerce_prefetch_ids(id: DatasetId | SequenceABC[DatasetId]) -> tuple[DatasetId, ...]:
        if isinstance(id, DatasetId):
            return (id,)
        if not isinstance(id, SequenceABC) or isinstance(id, (str, bytes, bytearray)):
            raise TypeError("id must be a DatasetId or a sequence of DatasetId values")
        ids = tuple(id)
        if not ids:
            raise ValueError("id sequence must not be empty")
        for idx, dataset_id in enumerate(ids):
            if not isinstance(dataset_id, DatasetId):
                raise TypeError(f"id[{idx}] must be a DatasetId, got {type(dataset_id).__name__}")
        return ids

    def _prefetch_all_genes(
        self,
        id: DatasetId,
        batches: Iterable[CellAccess | Iterable[int]],
        *,
        config: ScheduledPrefetchConfig | Mapping[str, Any] | None = None,
    ) -> Iterator[CellBatch]:
        config = _coerce_prefetch_config(config)
        rust_config = _config_to_rust(config)
        plan = _PrefetchPlan.single(batches)
        inner = self._core().prefetch_cells_raw(id._rust, plan, rust_config)
        return PrefetchIterator(inner, gene_names=tuple(inner.gene_names))

    def _prefetch_genes(
        self,
        id: DatasetId,
        batches: Iterable[CellAccess | Iterable[int]],
        genes: Iterable[str],
        *,
        missing: MissingGenePolicy | None = None,
        config: ScheduledPrefetchConfig | Mapping[str, Any] | None = None,
    ) -> Iterator[CellBatch]:
        names = tuple(_as_gene_names(genes, "genes"))
        rust_missing = missing._rust if missing is not None else None
        config = _coerce_prefetch_config(config)
        rust_config = _config_to_rust(config)
        plan = _PrefetchPlan.single(batches)
        inner = self._core().prefetch_cells_by_gene_names_raw(
            id._rust,
            plan,
            list(names),
            rust_missing,
            rust_config,
        )
        return PrefetchIterator(inner, gene_names=tuple(inner.gene_names))

    def __repr__(self) -> str:
        state = "closed" if self.is_closed else "open"
        return f"ScDataBank(state={state!r}, registered={self._registered_count})"

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "ScDataBank":
        return self

    def __exit__(self, *exc: object) -> None:
        # Drop the Rust core explicitly so its IO / decode / access / compute
        # thread pools are torn down deterministically on scope exit or
        # exception, rather than waiting on garbage collection.
        self._inner = None

    def close(self) -> None:
        """Release the Rust core and its thread pools immediately.

        Safe to call more than once.  After ``close``, any further call on
        this bank raises :class:`RuntimeError`.
        """
        self._inner = None
