"""Process-safe distributed consumption of one compiled prefetch plan."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import multiprocessing.reduction
import os
import threading
from types import TracebackType
from typing import Any, Self, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from scdata import _core
from scdata.exceptions import CancelledError, UnsupportedError
from scdata.load._config import DEFAULT_MAX_CONTROL_BYTES, PlanConfig, SessionConfig
from scdata.load._dataset import Dataset, RowRef
from scdata.load._output import OutputSpec
from scdata.load._stats import SessionState
from scdata.load._validation import as_int, dtype_from_core

if TYPE_CHECKING:
    from scdata.load._plan import Plan

__all__ = ["DistributedIterator", "DistributedSession", "distributed_prefetch"]


@dataclass(frozen=True, slots=True)
class _DistributedMetadata:
    world_size: int
    n_rows: int
    n_cols: int
    batch_size: int
    batch_count: int
    dtype_name: str


_DISTRIBUTED_DTYPE_NAMES = frozenset({"i16", "i32", "i64", "u16", "u32", "u64", "f32", "f64"})


def _normalize_distributed_metadata(metadata: object) -> _DistributedMetadata:
    if not isinstance(metadata, _DistributedMetadata):
        raise TypeError("distributed metadata has an invalid type")
    world_size = as_int(
        metadata.world_size,
        "metadata.world_size",
        minimum=1,
        maximum=(1 << 32) - 1,
    )
    n_rows = as_int(metadata.n_rows, "metadata.n_rows")
    n_cols = as_int(metadata.n_cols, "metadata.n_cols")
    batch_size = as_int(metadata.batch_size, "metadata.batch_size", minimum=1)
    batch_count = as_int(metadata.batch_count, "metadata.batch_count")
    expected_batches = (n_rows + batch_size - 1) // batch_size
    if batch_count != expected_batches:
        raise ValueError(
            f"metadata.batch_count={batch_count} does not match "
            f"n_rows={n_rows} and batch_size={batch_size}"
        )
    dtype_name = metadata.dtype_name
    if not isinstance(dtype_name, str) or dtype_name not in _DISTRIBUTED_DTYPE_NAMES:
        raise ValueError(f"distributed metadata has unsupported dtype {dtype_name!r}")
    return _DistributedMetadata(
        world_size=world_size,
        n_rows=n_rows,
        n_cols=n_cols,
        batch_size=batch_size,
        batch_count=batch_count,
        dtype_name=dtype_name,
    )


class _ProducerRunner:
    __slots__ = ("error", "inner", "lock")

    def __init__(self, inner: _core._SharedServer) -> None:
        self.inner = inner
        self.error: BaseException | None = None
        self.lock = threading.Lock()

    def run(self) -> None:
        try:
            _core.shared_run(self.inner)
        except BaseException as error:
            with self.lock:
                self.error = error

    def get_error(self) -> BaseException | None:
        with self.lock:
            return self.error


class DistributedSession:
    """A shared producer and a set of rank-local, process-transferable iterators.

    The producer runs in one background thread and performs storage I/O and
    decoding only once. Logical batches are assigned round-robin by rank. Use
    :meth:`rank` before starting worker processes and pass each returned
    :class:`DistributedIterator` to exactly one process.
    """

    __slots__ = (
        "_closed",
        "_handles",
        "_issued_ranks",
        "_metadata",
        "_owner_pid",
        "_plan",
        "_runner",
        "_thread",
    )

    def __init__(
        self,
        plan: Plan,
        world_size: int,
        config: SessionConfig | None = None,
        *,
        max_control_bytes: int = DEFAULT_MAX_CONTROL_BYTES,
    ) -> None:
        if not hasattr(_core, "shared_attach") or not hasattr(_core, "plan_open_shared"):
            raise UnsupportedError(
                "distributed prefetch requires Linux with lock-free 64-bit atomics"
            )
        normalized_world_size = as_int(world_size, "world_size", minimum=1, maximum=(1 << 32) - 1)
        if config is None:
            config = SessionConfig()
        elif not isinstance(config, SessionConfig):
            raise TypeError("config must be a SessionConfig instance")
        max_control_bytes = as_int(
            max_control_bytes,
            "max_control_bytes",
            minimum=1,
        )
        inner = _core.plan_open_shared(
            plan._inner,
            config._to_core(),
            normalized_world_size,
            max_control_bytes,
        )
        server_meta = _core.shared_server_meta(inner)
        self._plan = plan
        self._metadata = _DistributedMetadata(
            world_size=server_meta["world_size"],
            n_rows=server_meta["n_rows"],
            n_cols=server_meta["n_cols"],
            batch_size=server_meta["batch_size"],
            batch_count=server_meta["batch_count"],
            dtype_name=server_meta["dtype"],
        )
        self._runner = _ProducerRunner(inner)
        self._thread = threading.Thread(
            target=self._runner.run,
            name="sc-load-distributed-producer",
            daemon=False,
        )
        self._handles: list[DistributedIterator] = []
        self._issued_ranks: set[int] = set()
        self._closed = False
        self._owner_pid = os.getpid()
        try:
            self._thread.start()
        except BaseException:
            _core.shared_cancel(inner)
            raise

    @property
    def plan(self) -> Plan:
        return self._plan

    @property
    def world_size(self) -> int:
        return self._metadata.world_size

    @property
    def state(self) -> SessionState:
        return _core.shared_server_meta(self._runner.inner)["state"]

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def finished(self) -> bool:
        return not self._thread.is_alive() and self._runner.get_error() is None

    def rank(self, rank: int, *, copy: bool = True) -> DistributedIterator:
        """Create the sole iterator for ``rank``.

        ``copy=True`` (the default) yields compact writable NumPy-owned arrays
        and cannot stall on Python reference lifetimes. ``copy=False`` yields
        read-only zero-copy shared-ring views; retaining such a view keeps its
        ring generation leased and intentionally applies backpressure.
        """
        self._ensure_owner_process()
        if self._closed:
            raise ValueError("distributed session is closed")
        normalized_rank = as_int(
            rank,
            "rank",
            minimum=0,
            maximum=self.world_size - 1,
        )
        if not isinstance(copy, bool):
            raise TypeError("copy must be a bool")
        if normalized_rank in self._issued_ranks:
            raise ValueError(f"rank {normalized_rank} already has an iterator")
        handle = self._new_iterator(normalized_rank, copy)
        self._issued_ranks.add(normalized_rank)
        self._handles.append(handle)
        return handle

    def ranks(self, *, copy: bool = True) -> tuple[DistributedIterator, ...]:
        """Create one iterator for every not-yet-issued rank atomically."""
        self._ensure_owner_process()
        if self._closed:
            raise ValueError("distributed session is closed")
        if not isinstance(copy, bool):
            raise TypeError("copy must be a bool")
        pending = tuple(rank for rank in range(self.world_size) if rank not in self._issued_ranks)
        created: list[DistributedIterator] = []
        try:
            for rank in pending:
                created.append(self._new_iterator(rank, copy))
        except BaseException:
            for handle in created:
                try:
                    handle.close()
                except BaseException:
                    pass
            raise
        self._issued_ranks.update(pending)
        self._handles.extend(created)
        return tuple(created)

    def _new_iterator(self, rank: int, copy: bool) -> DistributedIterator:
        descriptor = _core.shared_duplicate_fd(self._runner.inner)
        try:
            return DistributedIterator(descriptor, rank, self._metadata, copy=copy)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    def wait(self, timeout: float | None = None) -> None:
        """Wait for all rank leases to drain and re-raise producer failures."""
        self._ensure_owner_process()
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be a non-negative finite number or None")
            timeout = float(timeout)
            if not np.isfinite(timeout) or timeout < 0:
                raise ValueError("timeout must be a non-negative finite number")
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("distributed producer did not finish before timeout")
        error = self._runner.get_error()
        if error is not None:
            raise error

    def cancel(self) -> None:
        """Cancel producer work and wake rank clients and ACK waits."""
        self._ensure_owner_process()
        if not self._thread.is_alive():
            return
        _core.shared_cancel(self._runner.inner)

    def close(self) -> None:
        """Cancel unfinished work, close parent descriptors, and join producer."""
        if os.getpid() != self._owner_pid:
            return
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        if self._thread.is_alive():
            try:
                self.cancel()
            except BaseException as error:
                errors.append(error)
        handles, self._handles = self._handles, []
        for handle in handles:
            try:
                handle.close()
            except BaseException as error:
                errors.append(error)
        try:
            self._thread.join()
        except BaseException as error:
            errors.append(error)
        if errors:
            raise errors[0]

    def _ensure_owner_process(self) -> None:
        current_pid = os.getpid()
        if current_pid != self._owner_pid:
            raise RuntimeError(
                f"distributed session was opened in process {self._owner_pid}, "
                f"but is being used in process {current_pid}"
            )

    def info(self) -> dict[str, object]:
        return {
            "state": self.state,
            "closed": self.closed,
            "finished": self.finished,
            "world_size": self.world_size,
            "issued_ranks": tuple(sorted(self._issued_ranks)),
            "shape": self._plan.shape,
            "dtype": self._plan.dtype.name,
        }

    def __enter__(self) -> Self:
        self._ensure_owner_process()
        if self._closed:
            raise ValueError("distributed session is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        existing_error = self._runner.get_error()
        self.close()
        if exc_type is None:
            error = existing_error or self._runner.get_error()
            if error is not None and not isinstance(error, CancelledError):
                raise error

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def __repr__(self) -> str:
        return (
            f"DistributedSession(state={self.state!r}, world_size={self.world_size}, "
            f"issued_ranks={len(self._issued_ranks)}, closed={self.closed})"
        )


class DistributedIterator(Iterator[NDArray[Any]]):
    """A rank-local iterator that can be transferred with multiprocessing.

    Attach happens lazily in the process that first consumes the iterator. Do
    not pickle or fork an iterator after consumption has begun.
    """

    __slots__ = (
        "_batch_count",
        "_batches_yielded",
        "_client",
        "_closed",
        "_consume_lock",
        "_copy",
        "_descriptor",
        "_dtype",
        "_exhausted",
        "_lock_pid",
        "_metadata",
        "_n_rows",
        "_owner_pid",
        "_rank",
        "_resource_lock",
        "_rows_yielded",
    )

    def __init__(
        self,
        descriptor: int,
        rank: int,
        metadata: _DistributedMetadata,
        *,
        copy: bool,
    ) -> None:
        if hasattr(self, "_descriptor"):
            raise RuntimeError("distributed iterator is already initialized")
        normalized_descriptor = as_int(
            descriptor,
            "descriptor",
            maximum=(1 << 31) - 1,
        )
        normalized_metadata = _normalize_distributed_metadata(metadata)
        normalized_rank = as_int(
            rank,
            "rank",
            maximum=normalized_metadata.world_size - 1,
        )
        if not isinstance(copy, bool):
            raise TypeError("copy must be a bool")
        self._descriptor = normalized_descriptor
        self._rank = normalized_rank
        self._metadata = normalized_metadata
        self._copy = copy
        self._dtype = dtype_from_core(normalized_metadata.dtype_name)
        if normalized_rank >= normalized_metadata.batch_count:
            self._batch_count = 0
            self._n_rows = 0
        else:
            self._batch_count = (
                normalized_metadata.batch_count - 1 - normalized_rank
            ) // normalized_metadata.world_size + 1
            self._n_rows = self._batch_count * normalized_metadata.batch_size
            final_logical = normalized_metadata.batch_count - 1
            if final_logical % normalized_metadata.world_size == normalized_rank:
                final_rows = (
                    normalized_metadata.n_rows - final_logical * normalized_metadata.batch_size
                )
                self._n_rows -= normalized_metadata.batch_size - final_rows
        self._client: _core._SharedClient | None = None
        self._owner_pid: int | None = None
        self._lock_pid = os.getpid()
        self._resource_lock = threading.Lock()
        self._consume_lock = threading.Lock()
        self._batches_yielded = 0
        self._rows_yielded = 0
        self._exhausted = False
        self._closed = False

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._metadata.world_size

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def n_cols(self) -> int:
        return self._metadata.n_cols

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_rows, self.n_cols

    @property
    def batch_count(self) -> int:
        return self._batch_count

    @property
    def batches_yielded(self) -> int:
        return self._batches_yielded

    @property
    def rows_yielded(self) -> int:
        return self._rows_yielded

    @property
    def rows_remaining(self) -> int:
        return self.n_rows - self._rows_yielded

    @property
    def progress(self) -> float:
        return 1.0 if self.n_rows == 0 else self._rows_yielded / self.n_rows

    @property
    def copy(self) -> bool:
        return self._copy

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def next_batch(self, *, copy: bool | None = None) -> NDArray[Any] | None:
        """Return this rank's next batch, or ``None`` at rank-local EOF."""
        self._ensure_consumption_process()
        self._refresh_locks_after_fork()
        with self._consume_lock:
            return self._next_batch(copy=copy)

    def _next_batch(self, *, copy: bool | None) -> NDArray[Any] | None:
        if self._exhausted:
            return None
        if self._closed:
            raise ValueError("distributed iterator is closed")
        if copy is None:
            resolved_copy = self._copy
        elif isinstance(copy, bool):
            resolved_copy = copy
        else:
            raise TypeError("copy must be a bool or None")
        client = self._ensure_client()
        try:
            next_fn = _core.shared_next_copy if resolved_copy else _core.shared_next
            batch = next_fn(client)
        except BaseException:
            self.close()
            raise
        if batch is None:
            self._exhausted = True
            self._closed = True
            self._close_resources()
            return None
        rows = int(batch.shape[0])
        next_rows = self._rows_yielded + rows
        if rows <= 0 or next_rows > self.n_rows:
            self.close()
            raise RuntimeError("shared core returned a batch outside the rank-local plan")
        self._batches_yielded += 1
        self._rows_yielded = next_rows
        return batch

    def read(self) -> NDArray[Any]:
        """Materialize all remaining rank-local rows into compact owned memory."""
        self._ensure_consumption_process()
        self._refresh_locks_after_fork()
        with self._consume_lock:
            return self._read()

    def _read(self) -> NDArray[Any]:
        remaining = self.rows_remaining
        if self._exhausted:
            return np.empty((0, self.n_cols), dtype=self._dtype)
        if self._closed:
            raise ValueError("distributed iterator is closed")
        client = self._ensure_client()
        try:
            output, batches = _core.shared_read(client, remaining)
        except BaseException:
            self.close()
            raise
        expected_batches = self._batch_count - self._batches_yielded
        if (
            output.shape != (remaining, self.n_cols)
            or output.dtype != self._dtype
            or not output.flags.owndata
            or not output.flags.c_contiguous
            or not output.flags.writeable
            or batches != expected_batches
        ):
            self.close()
            raise RuntimeError("shared core returned output outside the rank-local plan")
        self._batches_yielded = self._batch_count
        self._rows_yielded = self._n_rows
        self._exhausted = True
        self._closed = True
        self._close_resources()
        return output

    def close(self) -> None:
        """Close this rank; incomplete attached ranks cancel the shared session."""
        if self._closed:
            return
        self._closed = True
        self._close_resources()

    def _close_resources(self) -> None:
        current_pid = os.getpid()
        if self._owner_pid is not None and self._owner_pid != current_pid:
            client, self._client = self._client, None
            descriptor, self._descriptor = self._descriptor, -1
            if descriptor >= 0:
                os.close(descriptor)
            del client
            return
        self._refresh_locks_after_fork()
        with self._resource_lock:
            client, self._client = self._client, None
            descriptor, self._descriptor = self._descriptor, -1
        errors: list[BaseException] = []
        if client is not None:
            try:
                _core.shared_close(client)
            except BaseException as error:
                errors.append(error)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def _ensure_consumption_process(self) -> None:
        current_pid = os.getpid()
        if self._owner_pid is not None and self._owner_pid != current_pid:
            raise RuntimeError(
                f"distributed iterator was attached in process {self._owner_pid}, "
                f"but is being used in process {current_pid}; attach after forking"
            )

    def _refresh_locks_after_fork(self) -> None:
        current_pid = os.getpid()
        if self._lock_pid != current_pid:
            self._resource_lock = threading.Lock()
            self._consume_lock = threading.Lock()
            self._lock_pid = current_pid

    def _validate_attached_client(self, client: _core._SharedClient) -> None:
        meta = _core.shared_client_meta(client)
        actual = (
            meta["rank"],
            meta["world_size"],
            meta["n_rows"],
            meta["n_cols"],
            meta["batch_size"],
            meta["batch_count"],
            meta["dtype"],
        )
        expected = (
            self.rank,
            self.world_size,
            self._metadata.n_rows,
            self.n_cols,
            self._metadata.batch_size,
            self.batch_count,
            self._metadata.dtype_name,
        )
        if actual != expected:
            raise RuntimeError("shared descriptor metadata does not match the transferred iterator")

    def _ensure_client(self) -> _core._SharedClient:
        current_pid = os.getpid()
        if self._owner_pid is not None and self._owner_pid != current_pid:
            raise RuntimeError(
                f"distributed iterator was attached in process {self._owner_pid}, "
                f"but is being used in process {current_pid}; attach after forking"
            )
        client = self._client
        if client is not None:
            return client
        self._refresh_locks_after_fork()
        with self._resource_lock:
            if self._closed:
                raise ValueError("distributed iterator is closed")
            if self._client is not None:
                return self._client
            if self._descriptor < 0:
                raise ValueError("distributed iterator has no live descriptor")
            self._owner_pid = current_pid
            try:
                client = _core.shared_attach(self._descriptor, self.rank)
            except BaseException:
                self._owner_pid = None
                raise
            descriptor, self._descriptor = self._descriptor, -1
            try:
                os.close(descriptor)
                self._validate_attached_client(client)
            except BaseException:
                self._closed = True
                try:
                    _core.shared_close(client)
                except BaseException:
                    pass
                raise
            self._client = client
            return client

    def __getstate__(self) -> tuple[object, int, _DistributedMetadata, bool]:
        self._ensure_consumption_process()
        self._refresh_locks_after_fork()
        with self._resource_lock:
            if self._client is not None or self._owner_pid is not None:
                raise TypeError(
                    "cannot transfer a distributed iterator after consumption has begun"
                )
            if self._closed or self._descriptor < 0:
                raise TypeError("cannot transfer a closed distributed iterator")
            return (
                multiprocessing.reduction.DupFd(self._descriptor),
                self._rank,
                self._metadata,
                self._copy,
            )

    def __setstate__(
        self,
        state: tuple[object, int, _DistributedMetadata, bool],
    ) -> None:
        if hasattr(self, "_descriptor"):
            raise TypeError("distributed iterator state has already been restored")
        descriptor, rank, metadata, copy = state
        detach = getattr(descriptor, "detach", None)
        if detach is None:
            raise TypeError("invalid multiprocessing file-descriptor transfer")
        transferred = detach()
        if isinstance(transferred, bool) or not isinstance(transferred, int) or transferred < 0:
            raise TypeError("invalid multiprocessing file descriptor")
        try:
            self.__init__(transferred, rank, metadata, copy=copy)
        except BaseException:
            try:
                os.close(transferred)
            except OSError:
                pass
            raise

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> NDArray[Any]:
        batch = self.next_batch()
        if batch is None:
            raise StopIteration
        return batch

    def __len__(self) -> int:
        return self.batch_count

    def __enter__(self) -> Self:
        self._ensure_client()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def __repr__(self) -> str:
        return (
            f"DistributedIterator(rank={self.rank}/{self.world_size}, "
            f"batches={self.batches_yielded}/{self.batch_count}, "
            f"copy={self.copy}, closed={self.closed}, exhausted={self.exhausted})"
        )


def distributed_prefetch(
    datasets: Dataset | Iterable[Dataset],
    rows: Iterable[int | RowRef | tuple[int, int]] | NDArray[np.integer[Any]],
    *,
    world_size: int,
    output: OutputSpec | None = None,
    batch_size: int = 256,
    prefetch_step: int = 8,
    plan_config: PlanConfig | None = None,
    config: SessionConfig | None = None,
    max_control_bytes: int = DEFAULT_MAX_CONTROL_BYTES,
) -> DistributedSession:
    """Compile a plan and start its explicit multi-rank shared producer."""
    from scdata.load._plan import compile

    plan = compile(
        datasets,
        rows,
        output=output,
        batch_size=batch_size,
        prefetch_step=prefetch_step,
        config=plan_config,
    )
    return plan.open_distributed(
        world_size,
        config,
        max_control_bytes=max_control_bytes,
    )
