"""Private PyO3 boundary for the public ``scdata`` package.

Public Python classes live in pure Python. This module exposes:

* an exception / warning hierarchy re-exported as ``scdata.exceptions``
* frozen opaque handles (``_Store``, ``_Dataset``, ``_Plan``, ``_Session``,
  and the Linux shared-ring types)
* function-style operations over those handles

Handles have no public constructor and no instance methods. Call the
matching ``store_*`` / ``dataset_*`` / ``plan_*`` / ``session_*`` /
``shared_*`` functions instead.

Shared-ring exports (``plan_open_shared``, ``_Shared*``, ``shared_*``,
``DEFAULT_MAX_SHARED_CONTROL_BYTES``) exist only on Linux with lock-free
64-bit atomics.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, ClassVar, Final, Literal, Mapping, TypedDict, final

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Scalar aliases
# ---------------------------------------------------------------------------

StorageDTypeName = Literal["i16", "i32", "i64", "u16", "u32", "u64", "f32", "f64"]
"""On-disk / output element tag used by the native core (not NumPy names)."""

MatrixKind = Literal["dense", "csr"]
AxisKind = Literal["all", "range", "positions"]
"""Normalized axis descriptor consumed by select / gather APIs.

``all``
    Entire axis. Payload is ignored (pass ``None``).
``range``
    Half-open ``(start, end)`` uint pair.
``positions``
    C-contiguous 1-D ``uint64`` index array.
"""
AxisPayload = tuple[int, int] | NDArray[np.uint64] | None
CsrOutput = Literal["sparse", "dense", "csr"]
"""CSR materialization mode. ``csr`` is accepted as an alias of ``sparse``."""
PartitionPolicy = Literal["cells", "fixed_cells", "budget", "bytes_budget"]
"""Chunk / block partition strategy.

``cells`` / ``fixed_cells``
    ``n`` is a cell (row) count.
``budget`` / ``bytes_budget``
    ``n`` is a byte budget.
"""
IoModeName = Literal["auto", "blocking", "uring"]
SessionStateName = Literal["running", "failed", "cancelled", "finished"]
OverflowPolicyName = Literal["error", "use_fill", "use_value", "unchecked"]
CompressorDict = Mapping[str, Any]
"""Private JSON-shaped SCC compressor stored in ``meta.json``."""

DenseSelectResult = tuple[Literal["dense"], NDArray[Any]]
"""Tagged dense gather: ``("dense", values)`` with shape ``(n_rows, n_cols)``."""
CsrSelectResult = tuple[
    Literal["csr"],
    NDArray[Any],
    NDArray[Any],
    NDArray[np.uint64],
    tuple[int, int],
]
"""Tagged CSR gather: ``("csr", indices, data, indptr, (n_rows, n_cols))``."""
SelectResult = DenseSelectResult | CsrSelectResult

# ---------------------------------------------------------------------------
# Metadata / config mappings returned or consumed by the native functions
# ---------------------------------------------------------------------------

class StoreMeta(TypedDict):
    """Snapshot from :func:`store_meta`."""

    kind: MatrixKind
    shape: tuple[int, int]
    value_dtype: StorageDTypeName
    index_dtype: StorageDTypeName | None
    nnz: int | None
    max_metadata_size: int
    max_encoded_size: int
    max_decoded_size: int
    max_block_count: int
    num_workers: int
    compressor: dict[str, Any]
    indptr_compressor: dict[str, Any] | None

class DatasetMeta(TypedDict):
    """Snapshot from :func:`dataset_meta`."""

    kind: MatrixKind
    shape: tuple[int, int]
    n_rows: int
    n_cols: int
    dtype: StorageDTypeName

class PlanMeta(TypedDict):
    """Snapshot from :func:`plan_meta`."""

    batch_size: int
    batch_count: int
    prefetch_step: int
    cache_capacity_bytes: int
    n_cols: int
    dtype: StorageDTypeName
    row_stride_bytes: int
    is_empty: bool

class PlanStatsDict(TypedDict):
    """Compiler cost-model snapshot from :func:`plan_stats`.

    Builds with the ``profile`` feature may add extra nanosecond counters
    (for example ``compile_resolve_ns`` and ``compile_finalize_ns``).
    """

    input_rows: int
    block_jobs: int
    jobs: int
    data_io_ops: int
    indices_io_ops: int
    predicted_physical_bytes: int
    gap_bytes: int
    max_encoded_bytes_per_side: int
    max_decoded_bytes_per_job: int
    arena_bytes: int
    compile_working_set_bytes: int
    retained_whole_key_bytes: int
    output_ring_bytes: int
    compile_time_io_bytes: int
    compile_time_io_ops: int
    predicted_io_seconds: float
    cache_capacity_bytes: int
    cache_arena_bytes: int
    cache_alignment_loss_bytes: int
    unique_cache_objects: int
    residency_loads: int
    residency_reloads: int
    cache_reference_hits: int
    cache_reference_misses: int
    cache_capacity_stalls: int
    cache_fragmentation_stalls: int
    cache_horizon_max_batches: int
    initialize_io_tasks: int
    executable_tasks: int
    dependency_edges: int
    independent_block_loads: int
    fused_io_tasks: int
    predicted_io_ops_saved: int
    io_payload_bytes: int
    io_span_bytes: int
    io_read_amplification: float
    max_decode_ops_per_io_task: int
    max_decoded_bytes_per_io_task: int
    initialize_fused_io_tasks: int
    regular_fused_io_tasks: int

class SessionMeta(TypedDict):
    """Lifecycle snapshot from :func:`session_meta`."""

    closed: bool
    exhausted: bool
    state: SessionStateName

class RuntimeStatsDict(TypedDict):
    """Worker / I/O snapshot from :func:`session_stats`.

    Builds with the ``profile`` feature may add read, decode, scatter, and
    per-worker counters.
    """

    requested_io_mode: IoModeName
    requested_queue_depth: int
    actual_io_mode: IoModeName
    actual_queue_depth: int
    num_workers: int
    max_inflight_jobs_per_worker: int
    max_inflight_encoded_bytes_per_worker: int
    max_decoded_bytes_per_worker: int
    state: SessionStateName

class SharedServerMeta(TypedDict):
    """Snapshot from :func:`shared_server_meta`."""

    world_size: int
    n_rows: int
    n_cols: int
    batch_size: int
    batch_count: int
    row_stride_bytes: int
    dtype: StorageDTypeName
    state: SessionStateName

class SharedClientMeta(TypedDict):
    """Snapshot from :func:`shared_client_meta`."""

    rank: int
    world_size: int
    n_rows: int
    n_cols: int
    batch_size: int
    batch_count: int
    dtype: StorageDTypeName
    closed: bool
    exhausted: bool
    next_logical_batch: int | None

class OutputSpecDict(TypedDict):
    """Normalized output spec consumed by :func:`plan_compile`.

    Built by ``OutputSpec._to_core()``. ``overflow_value`` is required when
    ``overflow == "use_value"``.
    """

    n_cols: int
    dtype: StorageDTypeName
    fill: int | float
    overflow: OverflowPolicyName
    overflow_value: int | float | None
    allow_float_rounding: bool

class PlanConfigDict(TypedDict):
    """Flattened plan config consumed by :func:`plan_compile`.

    Built by ``PlanConfig._to_core()``. Resource-limit fields are inlined
    (they are not nested under a ``limits`` key).
    """

    compile_io_concurrency: int
    io_merge: IoMergeConfigDict
    cache_capacity_bytes: int
    cache_alignment: int
    cache_fragmentation_slack_bytes: int
    max_output_buffer_bytes: int
    max_compile_arena_bytes: int
    max_compile_working_set_bytes: int
    max_retained_whole_key_bytes: int
    max_blocks_per_job: int
    max_cells_per_job: int
    max_encoded_bytes_per_side: int
    max_decoded_bytes_per_job: int

class IoMergeConfigDict(TypedDict):
    policy: Literal["off", "adjacent", "cost"]
    max_coalesced_io_bytes: int
    max_io_gap_bytes: int
    max_io_amplification_ratio: float
    max_decode_ops_per_io_task: int
    max_decoded_bytes_per_io_task: int
    max_encoded_staging_bytes_per_task: int
    io_bandwidth_bytes_per_second: float
    io_operations_per_second: float
    io_merge_delta_bytes: int
    initialize_parallelism_hint: int
    regular_io_parallelism_hint: int
    min_tasks_per_worker: int

class SessionConfigDict(TypedDict):
    """Normalized session config consumed by :func:`plan_open` / :func:`plan_open_shared`.

    Built by ``SessionConfig._to_core()``.
    """

    num_workers: int
    initialize_workers: int
    initialize_inflight_io_ops: int
    initialize_inflight_encoded_bytes: int
    io_mode: IoModeName
    queue_depth: int
    max_inflight_jobs_per_worker: int
    max_inflight_encoded_bytes_per_worker: int
    max_decoded_bytes_per_worker: int
    max_total_inflight_io_ops: int
    max_total_inflight_encoded_bytes: int
    max_total_decoded_bytes: int

# ---------------------------------------------------------------------------
# Exceptions / warnings
#
# Runtime ``__module__`` is ``scdata.exceptions``. Call-site validation in
# Python uses built-ins (``TypeError`` / ``ValueError`` / ``IndexError``);
# these types cover operational failures from the native core.
# Compress ``InvalidArgument`` is raised as built-in ``ValueError``, not
# :class:`InvalidArgumentError`.
# ---------------------------------------------------------------------------

class Error(Exception):
    """Base class for native operational failures.

    ``except scdata.Error`` catches decode / conversion / cancellation /
    corrupt-data failures, not argument mistakes raised as built-ins.
    """

    kind: ClassVar[str]
    """Stable machine-readable tag (``"unknown"`` on the base class)."""

    def __init__(self, *args: object) -> None: ...

class InvalidArgumentError(Error):
    """Illegal native argument. New Python validation raises ``ValueError``."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class InvalidInputError(Error):
    """Malformed compile / session input after Python normalization."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class InvalidConfigError(Error):
    """Plan or session configuration rejected by the native runtime."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class InvalidDatasetError(Error):
    """Registered dataset is inconsistent with the compile request."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class InvalidMetaError(Error):
    """On-disk ``meta.json`` is structurally invalid."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class ResourceLimitError(Error):
    """A compile- or runtime resource ceiling would be exceeded."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class StalePlanError(Error):
    """Plan no longer matches the stores it was compiled against."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class UnsupportedError(Error):
    """Requested feature is unavailable on this platform or build."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class IoError(Error):
    """Filesystem or block-device I/O failed."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class JsonError(Error):
    """JSON encode / decode of metadata failed."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class CodecError(Error):
    """Compressor / decompressor (dyn-blosc or sibling codec) failed."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class ZipError(Error):
    """ZIP archive read / write failed."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class DecodeError(Error):
    """Encoded payload could not be decoded into the expected layout."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class PromotionError(Error):
    """Numeric promotion into the output dtype failed."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class ConversionError(Error):
    """Value conversion into the output dtype failed."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class CancelledError(Error):
    """Session or shared producer was cancelled."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class SessionError(Error):
    """Session lifecycle or worker coordination failed."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class WorkerPanicError(Error):
    """A native worker thread panicked."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class AllocationError(Error):
    """Native allocation failed or an element-count overflowed."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class InternalError(Error):
    """Native invariant was violated (should not happen on valid input)."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class NotFoundError(Error):
    """Requested store path, ZIP prefix, or object does not exist."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class CorruptDataError(Error):
    """On-disk bytes are structurally inconsistent."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class PathError(Error):
    """Store path is invalid for the requested operation."""

    kind: ClassVar[str]

    def __init__(self, *args: object) -> None: ...

class Warning(UserWarning):
    """Base warning issued by the native core."""

    def __init__(self, *args: object) -> None: ...

class PerformanceWarning(Warning):
    """A configuration or workload is expected to perform poorly."""

    def __init__(self, *args: object) -> None: ...

# ---------------------------------------------------------------------------
# Opaque frozen handles
#
# None of these types is constructible from Python
# (``TypeError: No constructor defined``). They expose no instance methods;
# every operation is a module-level function that takes the handle.
# ---------------------------------------------------------------------------

@final
class _Store:
    """Opened SCC directory or ZIP-resident matrix.

    Construct with :func:`store_open`. Do not instantiate directly.

    Associated functions
    --------------------
    :func:`store_meta`
        Kind, shape, dtypes, limits, compressor.
    :func:`store_indptr`
        CSR row pointers, or ``None`` for dense.
    :func:`store_decode_dense_rows`
        Materialize a dense half-open row range.
    :func:`store_decode_csr_rows`
        Materialize a CSR half-open row range.
    :func:`store_select`
        Axis gather (``all`` / ``range`` / ``positions``).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Raise ``TypeError``; use :func:`store_open`."""

@final
class _Dataset:
    """Opened SCC matrix registered as a prefetch source.

    Construct with :func:`dataset_open`. Do not instantiate directly.

    Associated functions
    --------------------
    :func:`dataset_meta`
        Kind, shape, ``n_rows`` / ``n_cols``, storage dtype.
    :func:`plan_compile`
        Compile this handle together with sibling sources.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Raise ``TypeError``; use :func:`dataset_open`."""

@final
class _Plan:
    """Immutable compiled prefetch plan.

    Construct with :func:`plan_compile`. Do not instantiate directly.

    Associated functions
    --------------------
    :func:`plan_meta`
        Batch geometry and output dtype.
    :func:`plan_stats`
        Compiler cost-model snapshot.
    :func:`plan_open`
        Start one process-local execution session.
    :func:`plan_open_shared`
        Start a Linux shared-ring producer (multi-rank).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Raise ``TypeError``; use :func:`plan_compile`."""

@final
class _Session:
    """One process-local execution of a compiled plan.

    Construct with :func:`plan_open`. The handle is pinned to the creating
    process; using it after ``fork`` raises ``ValueError``.

    Associated functions
    --------------------
    :func:`session_next`
        Wait for the next compact owned batch, or ``None`` at EOF.
    :func:`session_cancel`
        Request cooperative cancellation.
    :func:`session_close`
        Cancel unfinished work and release the output ring.
    :func:`session_meta`
        ``closed`` / ``exhausted`` / ``state``.
    :func:`session_stats`
        Worker / I/O snapshot (retained after close / EOF).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Raise ``TypeError``; use :func:`plan_open`."""

@final
class _SharedServer:
    """Linux shared-ring producer for one compiled plan.

    Construct with :func:`plan_open_shared`. The handle is pinned to the
    creating process.

    Associated functions
    --------------------
    :func:`shared_run`
        Block until every rank has drained or the producer fails.
    :func:`shared_cancel`
        Wake clients and stop unfinished production.
    :func:`shared_duplicate_fd`
        Duplicate the attachable memfd for a rank / child process.
    :func:`shared_server_meta`
        World size, batch geometry, dtype, and producer state.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Raise ``TypeError``; use :func:`plan_open_shared`."""

@final
class _SharedClient:
    """Rank-local consumer attached to a :class:`_SharedServer` ring.

    Construct with :func:`shared_attach` **after** forking. The handle is
    pinned to the attaching process.

    Associated functions
    --------------------
    :func:`shared_next`
        Next batch as a read-only zero-copy view (leases a ring generation).
    :func:`shared_next_copy`
        Next batch copied into compact writable NumPy-owned memory.
    :func:`shared_read`
        Drain remaining expected rows into one compact matrix.
    :func:`shared_close`
        Release the client; an incomplete rank cancels the producer.
    :func:`shared_client_meta`
        Rank, geometry, ``closed`` / ``exhausted``, next logical batch.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Raise ``TypeError``; use :func:`shared_attach`."""

@final
class _SharedBatch:
    """Owner of one shared-ring generation lease.

    Not constructed by Python. Appears as the NumPy *base* of arrays
    returned by :func:`shared_next`. Dropping every derived view releases
    the generation and unblocks the producer.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Raise ``TypeError``; this type is created only by :func:`shared_next`."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_SHARED_CONTROL_BYTES: Final[int] = 67_108_864
"""Default shared-ring control-block ceiling (64 MiB).

Passed through ``max_control_bytes`` of :func:`plan_open_shared` unless the
Python wrapper overrides it. Linux-only export.
"""

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def store_open(
    path: str | PathLike[str],
    *,
    zip_prefix: str | None,
    max_metadata_size: int,
    max_encoded_size: int,
    max_decoded_size: int,
    max_block_count: int,
    num_workers: int,
) -> _Store:
    """Open a directory store or a store inside a ZIP archive.

    Parameters
    ----------
    path
        Directory, or ZIP file containing one or more SCC prefixes.
    zip_prefix
        Prefix inside the archive, or ``None`` for a directory store.
    max_metadata_size, max_encoded_size, max_decoded_size, max_block_count
        Hard decode ceilings forwarded to ``ReadLimits``.
    num_workers
        Decode thread count; must be ``>= 1``.
    """

def store_meta(store: _Store) -> StoreMeta:
    """Return kind, shape, dtypes, resource limits, and compressor objects."""

def store_indptr(store: _Store) -> NDArray[np.uint64] | None:
    """Return the CSR row-pointer array, or ``None`` when ``store`` is dense."""

def store_decode_dense_rows(store: _Store, start: int, end: int) -> NDArray[Any]:
    """Decode the half-open dense row range ``[start, end)``.

    ``store`` must be dense. ``end`` must not exceed the row count.
    """

def store_decode_csr_rows(
    store: _Store,
    start: int,
    end: int,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.uint64]]:
    """Decode the half-open CSR row range ``[start, end)``.

    Returns ``(indices, data, local_indptr)``. ``local_indptr`` is ``uint64``
    and rebased so the first selected row starts at 0. ``store`` must be CSR.
    """

def store_select(
    store: _Store,
    row_kind: AxisKind | str,
    row_payload: AxisPayload | Any,
    col_kind: AxisKind | str,
    col_payload: AxisPayload | Any,
    *,
    csr_output: CsrOutput | str,
) -> SelectResult:
    """Gather ``store`` along normalized row / column axes.

    Dense stores always return :data:`DenseSelectResult`. CSR stores return
    :data:`CsrSelectResult` when ``csr_output`` is ``"sparse"`` / ``"csr"``,
    and :data:`DenseSelectResult` when it is ``"dense"``.
    """

# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_dense(
    path: str | PathLike[str],
    values: NDArray[Any],
    *,
    chunk_policy: PartitionPolicy | str,
    chunk_n: int,
    block_policy: PartitionPolicy | str,
    block_n: int,
    num_workers: int,
    compressor: CompressorDict,
) -> None:
    """Write a C-contiguous 2-D NumPy matrix of a supported value dtype.

    Supported dtypes: ``i16``, ``i32``, ``i64``, ``u16``, ``u32``, ``u64``,
    ``f32``, ``f64``. ``compressor`` is the private storage representation
    produced by the public ``Codec`` policy. ``num_workers`` must be ``>= 1``.
    ``chunk_n`` / ``block_n`` must be non-zero.
    """

def write_csr(
    path: str | PathLike[str],
    indptr: NDArray[Any],
    indices: NDArray[Any],
    data: NDArray[Any],
    n_rows: int,
    n_cols: int,
    *,
    chunk_policy: PartitionPolicy | str,
    chunk_n: int,
    block_policy: PartitionPolicy | str,
    block_n: int,
    num_workers: int,
    compressor: CompressorDict,
    indptr_compressor: CompressorDict,
) -> None:
    """Write CSR arrays. ``indptr`` / ``indices`` must already be contiguous ``uint64``.

    ``data`` is a supported value dtype (same set as :func:`write_dense`).
    ``compressor`` encodes the indices/data payload; ``indptr_compressor``
    encodes the row pointers. Both are JSON-shaped mappings.
    """

# ---------------------------------------------------------------------------
# In-memory gather (independent of on-disk stores)
# ---------------------------------------------------------------------------

def matrix_dense_select(
    values: NDArray[Any],
    row_kind: AxisKind | str,
    row_payload: AxisPayload | Any,
    col_kind: AxisKind | str,
    col_payload: AxisPayload | Any,
    *,
    num_workers: int,
) -> NDArray[Any]:
    """Gather a C-contiguous 2-D dense array along normalized axes.

    Returns an owned ``(n_selected_rows, n_selected_cols)`` array of the
    same dtype. ``num_workers`` must be ``>= 1``.
    """

def matrix_csr_select(
    indptr: NDArray[Any],
    indices: NDArray[Any],
    data: NDArray[Any],
    n_rows: int,
    n_cols: int,
    row_kind: AxisKind | str,
    row_payload: AxisPayload | Any,
    col_kind: AxisKind | str,
    col_payload: AxisPayload | Any,
    *,
    csr_output: CsrOutput | str,
    num_workers: int,
) -> SelectResult:
    """Gather in-memory CSR buffers along normalized axes.

    ``indptr`` is copied as ``uint64``. ``indices`` must be C-contiguous
    ``uint16`` or ``uint32``. ``csr_output`` selects a tagged dense or CSR
    result. ``num_workers`` must be ``>= 1``.
    """

def matrix_csr_to_dense(
    indptr: NDArray[Any],
    indices: NDArray[Any],
    data: NDArray[Any],
    n_rows: int,
    n_cols: int,
    *,
    num_workers: int,
) -> NDArray[Any]:
    """Convert in-memory CSR buffers to a dense owned 2-D array.

    Same buffer constraints as :func:`matrix_csr_select`. Missing entries
    become the additive zero of ``data``'s dtype.
    """

# ---------------------------------------------------------------------------
# Dataset / plan / session
# ---------------------------------------------------------------------------

def dataset_open(
    path: str | PathLike[str],
    *,
    zip_prefix: str | None,
    max_metadata_size: int,
    max_encoded_size: int,
    max_decoded_size: int,
    max_block_count: int,
    num_workers: int,
) -> _Dataset:
    """Open an SCC matrix as a prefetch source.

    Arguments match :func:`store_open`. The returned handle is what
    :func:`plan_compile` consumes; it does not decode row payloads by itself.
    """

def dataset_meta(dataset: _Dataset) -> DatasetMeta:
    """Return kind, shape, and storage dtype for a registered dataset."""

def plan_compile(
    datasets: list[_Dataset],
    source_ids: list[int],
    feature_maps: list[NDArray[np.int64] | None],
    row_source_ids: NDArray[np.uint32] | None,
    row_indices: NDArray[np.uint64],
    output: OutputSpecDict,
    batch_size: int,
    prefetch_step: int,
    config: PlanConfigDict,
) -> _Plan:
    """Compile ordered row requests into an immutable execution plan.

    ``datasets``, ``source_ids``, and ``feature_maps`` must have equal
    length. Each feature map is ``None`` (identity) or a C-contiguous 1-D
    ``int64`` array (``-1`` drops a source column).

    When ``row_source_ids`` is ``None``, exactly one source must be
    registered and every ``row_indices`` entry refers to that source.
    Otherwise ``row_source_ids`` must be a C-contiguous ``uint32`` array of
    the same length as ``row_indices``.
    """

def plan_meta(plan: _Plan) -> PlanMeta:
    """Return batch geometry, output dtype, row stride, and emptiness."""

def plan_stats(plan: _Plan) -> PlanStatsDict:
    """Return the compiler cost-model snapshot captured at compile time."""

def plan_open(plan: _Plan, config: SessionConfigDict) -> _Session:
    """Start one independent process-local execution session."""

def plan_open_shared(
    plan: _Plan,
    config: SessionConfigDict,
    world_size: int,
    max_control_bytes: int,
) -> _SharedServer:
    """Start a Linux shared-ring producer with ``world_size`` ranks.

    Logical batches are assigned round-robin. ``max_control_bytes`` caps the
    control-block allocation (see :data:`DEFAULT_MAX_SHARED_CONTROL_BYTES`).
    """

def session_next(session: _Session) -> NDArray[Any] | None:
    """Wait for the next compact owned batch, or ``None`` at EOF.

    Each array owns its storage and remains valid after the output-ring
    slot is reused. Calling after :func:`session_close` (and before EOF)
    raises ``ValueError``.
    """

def session_cancel(session: _Session) -> None:
    """Request cooperative cancellation; blocked consumers are woken."""

def session_close(session: _Session) -> None:
    """Cancel unfinished work, join workers, and release the output ring.

    Safe to call more than once. Statistics remain readable afterwards.
    """

def session_meta(session: _Session) -> SessionMeta:
    """Return ``closed``, ``exhausted``, and the current ``state``."""

def session_stats(session: _Session) -> RuntimeStatsDict:
    """Return the latest worker / I/O snapshot.

    After EOF or close the last captured snapshot is returned. Raises
    ``ValueError`` if no snapshot is available.
    """

# ---------------------------------------------------------------------------
# Shared ring (Linux)
# ---------------------------------------------------------------------------

def shared_run(server: _SharedServer) -> None:
    """Run the producer to completion.

    Consumes the server: a second call raises ``ValueError``. Blocks until
    every rank has drained or the producer fails / is cancelled.
    """

def shared_cancel(server: _SharedServer) -> None:
    """Cancel production and wake rank clients.

    No-op if called from a process other than the one that opened ``server``.
    """

def shared_server_meta(server: _SharedServer) -> SharedServerMeta:
    """Return world size, batch geometry, dtype, and producer state."""

def shared_duplicate_fd(server: _SharedServer) -> int:
    """Duplicate the attachable memfd and return a raw file descriptor.

    The caller owns the returned fd and must ``os.close`` it (or pass it to
    :func:`shared_attach`, which duplicates it again).
    """

def shared_attach(fd: int, rank: int) -> _SharedClient:
    """Attach a rank-local client to a duplicated shared-ring fd.

    ``fd`` must be non-negative. Attach **after** forking; the returned
    client is pinned to the attaching process. ``fd`` remains owned by the
    caller.
    """

def shared_next(client: _SharedClient) -> NDArray[Any] | None:
    """Return the next batch as a read-only zero-copy view, or ``None`` at EOF.

    The array's *base* is a :class:`_SharedBatch` lease. Retaining the view
    keeps the ring generation alive and applies backpressure. Requires a
    little-endian target.
    """

def shared_next_copy(client: _SharedClient) -> NDArray[Any] | None:
    """Return the next batch copied into compact writable memory, or ``None``.

    The copy does not lease a ring generation. Requires a little-endian target.
    """

def shared_read(client: _SharedClient, expected_rows: int) -> tuple[NDArray[Any], int]:
    """Drain the client into one compact owned matrix.

    ``expected_rows`` is the remaining rank-local row count (must not exceed
    the dataset row count). Returns ``(values, n_batches)``. An incomplete
    rank cancels the producer. Requires a little-endian target.
    """

def shared_close(client: _SharedClient) -> None:
    """Release the client. An incomplete attached rank cancels the producer.

    No-op if called from a process other than the one that attached ``client``.
    """

def shared_client_meta(client: _SharedClient) -> SharedClientMeta:
    """Return rank, geometry, lifecycle flags, and the next logical batch."""
