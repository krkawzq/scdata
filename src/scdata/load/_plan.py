"""High-level plan compilation, prefetch helpers, and session lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import threading
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from scdata import _core
from scdata.exceptions import InternalError
from scdata.load._config import DEFAULT_MAX_CONTROL_BYTES, IoMergeConfig, PlanConfig, SessionConfig
from scdata.load._config import ResourceLimits
from scdata.load._dataset import Dataset, RowRef
from scdata.compress._limits import ReadLimits
from scdata.load._output import OutputSpec
from scdata.load._location import resolve_matrix_location
from scdata.load._stats import PlanStats, RuntimeStats, SessionState
from scdata.load._validation import as_int, normalize_rows

if TYPE_CHECKING:
    from typing_extensions import Self

    from scdata.load._distributed import DistributedSession

__all__ = ["Plan", "Prefetch", "Session", "compile", "prefetch"]

_U32_MAX = (1 << 32) - 1
_PLAN_MAGIC = b"SCPLAN01"
_PLAN_HEADER = struct.Struct("<8sHHQQQ32s")
_PLAN_FORMAT_MAJOR = 1
_PLAN_FORMAT_MINOR = 0
_MAX_PLAN_IMAGE_BYTES = 2 * 1024 * 1024 * 1024


class Plan:
    """An immutable, reusable prefetch plan."""

    __slots__ = ("_bind_lock", "_inner", "_meta", "_output", "_stats", "_template")

    def __init__(
        self,
        inner: _core._Plan | None,
        output: OutputSpec,
        *,
        template: dict[str, Any],
        meta: Mapping[str, Any] | None = None,
        stats: Mapping[str, Any] | None = None,
    ) -> None:
        self._inner = inner
        self._output = output
        self._template = template
        self._bind_lock = threading.Lock()
        if inner is not None:
            self._meta = _core.plan_meta(inner)
            self._stats = PlanStats._from_mapping(_core.plan_stats(inner))
        else:
            if meta is None or stats is None:
                raise ValueError("lazy Plan requires encoded metadata and statistics")
            self._meta = dict(meta)
            self._stats = PlanStats._from_mapping(stats)

    @property
    def batch_size(self) -> int:
        return int(self._meta["batch_size"])

    @property
    def batch_count(self) -> int:
        return int(self._meta["batch_count"])

    @property
    def prefetch_step(self) -> int:
        """Number of output-ring slots compiled into the plan."""
        return int(self._meta["prefetch_step"])

    @property
    def cache_capacity_bytes(self) -> int:
        return int(self._meta["cache_capacity_bytes"])

    @property
    def n_rows(self) -> int:
        return self._stats.input_rows

    @property
    def n_cols(self) -> int:
        return int(self._meta["n_cols"])

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_rows, self.n_cols

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._output.dtype

    @property
    def output(self) -> OutputSpec:
        return self._output

    @property
    def row_stride_bytes(self) -> int:
        return int(self._meta["row_stride_bytes"])

    @property
    def nbytes(self) -> int:
        """Logical byte size of the fully materialized output matrix."""
        return self.n_rows * self._output.row_nbytes

    @property
    def is_empty(self) -> bool:
        return bool(self._meta["is_empty"])

    @property
    def stats(self) -> PlanStats:
        return self._stats

    def open(self, config: SessionConfig | None = None) -> Session:
        """Start one independent execution session."""
        resolved = _resolve_session_config(config)
        inner = _core.plan_open(self._require_inner(), resolved._to_core())
        return Session(inner, self)

    def open_distributed(
        self,
        world_size: int,
        config: SessionConfig | None = None,
        *,
        max_control_bytes: int = DEFAULT_MAX_CONTROL_BYTES,
    ) -> DistributedSession:
        """Start one shared producer with a process-transferable iterator per rank."""
        from scdata.load._distributed import DistributedSession

        return DistributedSession(
            self,
            world_size,
            config,
            max_control_bytes=max_control_bytes,
        )

    def prefetch(self, config: SessionConfig | None = None) -> Prefetch:
        """Return an inspectable lazy iterator for this compiled plan."""
        return Prefetch(self, config)

    def iter_batches(self, config: SessionConfig | None = None) -> Iterator[NDArray[Any]]:
        """Execute the plan and yield compact NumPy-owned batches."""
        with self.open(config) as session:
            yield from session

    def read(self, config: SessionConfig | None = None) -> NDArray[Any]:
        """Execute the plan and materialize its full output matrix."""
        with self.open(config) as session:
            return session.read()

    @property
    def bound(self) -> bool:
        """Whether source paths have been opened and verified."""
        return self._inner is not None

    def bind(
        self,
        *,
        sources: Mapping[int, str | os.PathLike[str]] | None = None,
        verify: Literal["strict"] = "strict",
    ) -> Plan:
        """Open and strictly verify serialized sources, then compile the native plan."""
        self._require_inner(sources=sources, verify=verify)
        return self

    def dumps(self) -> bytes:
        """Serialize the relocatable plan template without runtime pointers or file handles."""
        return _encode_plan_image(self._template, self._meta, self._stats, self._output)

    @classmethod
    def loads(
        cls,
        blob: bytes | bytearray | memoryview,
        *,
        sources: Mapping[int, str | os.PathLike[str]] | None = None,
        verify: Literal["strict"] = "strict",
        lazy: bool = True,
        _base_dir: Path | None = None,
    ) -> Plan:
        """Parse a bounded plan image; source I/O is deferred when ``lazy=True``."""
        template, meta, stats, output = _decode_plan_image(blob, base_dir=_base_dir)
        plan = cls(None, output, template=template, meta=meta, stats=stats)
        if sources is not None:
            plan._apply_source_overrides(sources)
        if not lazy:
            plan.bind(verify=verify)
        elif verify != "strict":
            raise ValueError("verify must be 'strict'")
        return plan

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        relative_sources: bool = False,
    ) -> None:
        """Atomically save this plan image."""
        target = Path(os.fspath(path)).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        template = _copy_template(self._template)
        if relative_sources:
            for source in template["sources"]:
                source["path"] = os.path.relpath(source["path"], target.parent)
        image = _encode_plan_image(template, self._meta, self._stats, self._output)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(image)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        sources: Mapping[int, str | os.PathLike[str]] | None = None,
        verify: Literal["strict"] = "strict",
        lazy: bool = True,
    ) -> Plan:
        """Load a plan image, resolving relative sources against its directory."""
        source = Path(os.fspath(path)).resolve()
        return cls.loads(
            source.read_bytes(),
            sources=sources,
            verify=verify,
            lazy=lazy,
            _base_dir=source.parent,
        )

    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[bytes]]:
        del protocol
        return _restore_plan_v1, (self.dumps(),)

    def _apply_source_overrides(
        self, sources: Mapping[int, str | os.PathLike[str]]
    ) -> None:
        if self._inner is not None:
            raise ValueError("cannot override sources after Plan binding")
        if not isinstance(sources, Mapping):
            raise TypeError("sources must be a mapping from source id to path")
        for source_id, path in sources.items():
            normalized = as_int(source_id, "source id", maximum=_U32_MAX)
            try:
                source = self._template["sources"][normalized]
            except IndexError as error:
                raise ValueError(f"source id {normalized} is not registered") from error
            source["path"] = os.fspath(path)

    def _require_inner(
        self,
        *,
        sources: Mapping[int, str | os.PathLike[str]] | None = None,
        verify: Literal["strict"] = "strict",
    ) -> _core._Plan:
        if verify != "strict":
            raise ValueError("verify must be 'strict'")
        if self._inner is not None:
            if sources is not None:
                raise ValueError("cannot override sources after Plan binding")
            return self._inner
        with self._bind_lock:
            if self._inner is not None:
                return self._inner
            if sources is not None:
                self._apply_source_overrides(sources)
            datasets = [_dataset_from_source_spec(source) for source in self._template["sources"]]
            for source, dataset in zip(self._template["sources"], datasets):
                manifest = _source_manifest(source["path"], source["key"])
                if manifest != source["manifest"]:
                    raise ValueError(
                        f"stale plan source {source['source_id']}: storage manifest changed"
                    )
                observed = (dataset.kind, list(dataset.shape), dataset.dtype.name)
                expected = (source["kind"], source["shape"], source["dtype"])
                if observed != expected:
                    raise ValueError(
                        f"stale plan source {source['source_id']}: expected {expected}, "
                        f"observed {observed}"
                    )
            config = _plan_config_from_dict(self._template["config"])
            inner = _core.plan_compile(
                [dataset._require_inner() for dataset in datasets],
                list(range(len(datasets))),
                [dataset._feature_map_array for dataset in datasets],
                self._template["row_source_ids"],
                self._template["row_indices"],
                self._output._to_core(),
                self.batch_size,
                self.prefetch_step,
                config._to_core(),
            )
            self._inner = inner
            self._meta = _core.plan_meta(inner)
            self._stats = PlanStats._from_mapping(_core.plan_stats(inner))
            return inner

    def info(self) -> dict[str, object]:
        """Return compact, serialization-friendly plan diagnostics."""
        return {
            "shape": self.shape,
            "dtype": self.dtype.name,
            "nbytes": self.nbytes,
            "batch_size": self.batch_size,
            "batch_count": self.batch_count,
            "prefetch_step": self.prefetch_step,
            "cache_capacity_bytes": self.cache_capacity_bytes,
            "row_stride_bytes": self.row_stride_bytes,
            "stats": self._stats.as_dict(),
        }

    def __len__(self) -> int:
        return self.batch_count

    def __repr__(self) -> str:
        return (
            f"Plan(shape={self.shape!r}, batches={self.batch_count}, "
            f"batch_size={self.batch_size}, dtype={self.dtype.name!r}, "
            f"cache={self.cache_capacity_bytes}, prefetch_step={self.prefetch_step})"
        )


class Session(Iterator[NDArray[Any]]):
    """One independent execution of a plan.

    Each returned array owns compact NumPy storage and remains stable after the
    underlying output-ring slot is reused.
    """

    __slots__ = ("_inner", "_plan", "_rows_yielded")

    def __init__(self, inner: _core._Session, plan: Plan) -> None:
        self._inner = inner
        self._plan = plan
        self._rows_yielded = 0

    @property
    def plan(self) -> Plan:
        return self._plan

    @property
    def closed(self) -> bool:
        return bool(_core.session_meta(self._inner)["closed"])

    @property
    def exhausted(self) -> bool:
        return bool(_core.session_meta(self._inner)["exhausted"])

    @property
    def rows_yielded(self) -> int:
        return self._rows_yielded

    @property
    def rows_remaining(self) -> int:
        return self._plan.n_rows - self._rows_yielded

    @property
    def progress(self) -> float:
        """Fraction of planned rows already returned, in the range ``0..1``."""
        if self._plan.n_rows == 0:
            return 1.0 if self.exhausted else 0.0
        return self._rows_yielded / self._plan.n_rows

    @property
    def state(self) -> SessionState:
        return _core.session_meta(self._inner)["state"]

    @property
    def stats(self) -> RuntimeStats:
        return RuntimeStats._from_mapping(_core.session_stats(self._inner))

    def next_batch(self) -> NDArray[Any] | None:
        """Wait for and return the next compact batch, or ``None`` at EOF."""
        if self.closed and not self.exhausted:
            raise ValueError("session is closed")
        batch = _core.session_next(self._inner)
        if batch is not None:
            rows_yielded = self._rows_yielded + batch.shape[0]
            if rows_yielded > self._plan.n_rows:
                raise InternalError("core returned more rows than the compiled plan")
            self._rows_yielded = rows_yielded
        return batch

    def cancel(self) -> None:
        """Request cooperative cancellation; blocked consumers are woken."""
        _core.session_cancel(self._inner)

    def close(self) -> None:
        """Cancel unfinished work, join workers, and release the output ring."""
        _core.session_close(self._inner)

    def read(self) -> NDArray[Any]:
        """Materialize all remaining batches into one compact matrix."""
        remaining = self._plan.n_rows - self._rows_yielded
        first = self.next_batch()
        if first is None:
            if remaining != 0:
                raise InternalError(
                    f"core returned {self._rows_yielded} rows for a {self._plan.n_rows}-row plan"
                )
            return np.empty((0, self._plan.n_cols), dtype=self._plan.dtype)
        if first.shape[0] == remaining:
            if self.next_batch() is not None:
                raise InternalError("core returned more rows than the compiled plan")
            return first

        output = np.empty((remaining, self._plan.n_cols), dtype=self._plan.dtype)
        offset = first.shape[0]
        output[:offset] = first
        for batch in self:
            stop = offset + batch.shape[0]
            output[offset:stop] = batch
            offset = stop
        if offset != remaining:
            raise InternalError(f"core returned {offset} remaining rows, expected {remaining}")
        return output

    def info(self) -> dict[str, object]:
        """Return the current lifecycle and row-progress snapshot."""
        return {
            "state": self.state,
            "closed": self.closed,
            "exhausted": self.exhausted,
            "rows_yielded": self.rows_yielded,
            "rows_remaining": self.rows_remaining,
            "progress": self.progress,
        }

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> NDArray[Any]:
        batch = self.next_batch()
        if batch is None:
            raise StopIteration
        return batch

    def __enter__(self) -> Self:
        if self.closed:
            raise ValueError("session is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Session(state={self.state!r}, rows={self.rows_yielded}/{self._plan.n_rows}, "
            f"closed={self.closed}, exhausted={self.exhausted})"
        )


class Prefetch(Iterator[NDArray[Any]]):
    """Lazy batch iterator over a compiled plan.

    Iteration opens a session on first use. The session is closed at EOF, on
    :meth:`close`, or when leaving a ``with`` block.
    """

    __slots__ = (
        "_cancelled",
        "_closed",
        "_config",
        "_exhausted",
        "_opening",
        "_plan",
        "_session",
    )

    def __init__(self, plan: Plan, config: SessionConfig | None = None) -> None:
        self._plan = plan
        self._config = _resolve_session_config(config)
        self._session: Session | None = None
        self._cancelled = False
        self._closed = False
        self._exhausted = False
        self._opening = False

    @property
    def plan(self) -> Plan:
        return self._plan

    @property
    def config(self) -> SessionConfig:
        return self._config

    @property
    def closed(self) -> bool:
        return self._closed or (self._session is not None and self._session.closed)

    @property
    def exhausted(self) -> bool:
        return self._exhausted or (self._session is not None and self._session.exhausted)

    @property
    def state(self) -> SessionState | Literal["unopened", "opening", "closed"]:
        if self._session is not None:
            return self._session.state
        if self._cancelled:
            return "cancelled"
        if self._closed:
            return "closed"
        if self._opening:
            return "opening"
        return "unopened"

    @property
    def rows_yielded(self) -> int:
        return 0 if self._session is None else self._session.rows_yielded

    @property
    def rows_remaining(self) -> int:
        return self._plan.n_rows - self.rows_yielded

    @property
    def progress(self) -> float:
        """Fraction of planned rows already returned, in the range ``0..1``."""
        if self._plan.n_rows == 0:
            return 1.0 if self.exhausted else 0.0
        return self.rows_yielded / self._plan.n_rows

    @property
    def stats(self) -> RuntimeStats | None:
        if self._session is None:
            return None
        return self._session.stats

    def next_batch(self) -> NDArray[Any] | None:
        if self.exhausted:
            return None
        session = self._ensure_session()
        try:
            batch = session.next_batch()
        except BaseException:
            self._close_after_failure()
            raise
        if batch is None:
            self._exhausted = True
            self.close()
        return batch

    def cancel(self) -> None:
        """Cancel active work, or prevent an unopened iterator from starting."""
        if self.closed:
            return
        self._cancelled = True
        if self._session is None:
            self._closed = True
            return
        self._session.cancel()

    def close(self) -> None:
        """Permanently close the iterator and release an active session."""
        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            self._session.close()

    def read(self) -> NDArray[Any]:
        """Materialize all remaining rows into one compact matrix."""
        if self.exhausted:
            return np.empty((0, self._plan.n_cols), dtype=self._plan.dtype)
        session = self._ensure_session()
        try:
            return session.read()
        finally:
            self._exhausted = session.exhausted
            self.close()

    def _ensure_session(self) -> Session:
        if self._closed:
            raise ValueError("prefetch is closed")
        if self._session is None:
            if self._opening:
                raise RuntimeError("prefetch is already opening a session in another thread")
            self._opening = True
            try:
                session = self._plan.open(self._config)
            finally:
                self._opening = False
            self._session = session
            if self._closed:
                session.close()
                raise ValueError("prefetch was closed while opening its session")
        elif self._session.closed:
            self._closed = True
            self._exhausted = self._session.exhausted
            raise ValueError("prefetch is closed")
        return self._session

    def _close_after_failure(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def info(self) -> dict[str, object]:
        """Return the current lifecycle and row-progress snapshot."""
        return {
            "state": self.state,
            "closed": self.closed,
            "exhausted": self.exhausted,
            "rows_yielded": self.rows_yielded,
            "rows_remaining": self.rows_remaining,
            "progress": self.progress,
        }

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> NDArray[Any]:
        batch = self.next_batch()
        if batch is None:
            raise StopIteration
        return batch

    def __enter__(self) -> Self:
        self._ensure_session()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Prefetch(state={self.state!r}, rows={self.rows_yielded}/{self._plan.n_rows}, "
            f"closed={self.closed}, exhausted={self.exhausted}, plan={self._plan!r})"
        )


def compile(
    datasets: Dataset | Iterable[Dataset],
    rows: Iterable[int | RowRef | tuple[int, int]] | NDArray[np.integer[Any]],
    *,
    output: OutputSpec | None = None,
    batch_size: int = 256,
    prefetch_step: int = 8,
    config: PlanConfig | None = None,
) -> Plan:
    """Compile ordered row requests into a reusable execution plan.

    ``datasets`` is a single :class:`Dataset` or a collection. Collection
    order defines ``source_id`` values ``0..n-1``. A single dataset accepts
    ordinary row indices; multiple datasets use :class:`RowRef` or
    ``(source_id, row)`` pairs. Each dataset's ``feature_map`` is forwarded to
    the Rust compiler as-is; build one with
    :func:`~scdata.load.build_feature_map` or
    :meth:`~scdata.load.Dataset.with_aligned_features`.
    """
    dataset_list = _normalize_datasets(datasets)
    for dataset in dataset_list:
        dataset._require_inner()
    source_ids = list(range(len(dataset_list)))

    if output is None:
        output = _infer_output(dataset_list)
    elif not isinstance(output, OutputSpec):
        raise TypeError("output must be an OutputSpec or None")
    _validate_datasets_for_output(dataset_list, output)

    normalized_batch_size = as_int(batch_size, "batch_size", minimum=1)
    normalized_prefetch_step = as_int(
        prefetch_step,
        "prefetch_step",
        minimum=1,
        maximum=_U32_MAX,
    )
    if config is None:
        config = PlanConfig()
    elif not isinstance(config, PlanConfig):
        raise TypeError("config must be a PlanConfig instance")

    default_source_id = 0 if len(dataset_list) == 1 else None
    row_source_ids, row_indices = normalize_rows(rows, default_source_id=default_source_id)
    _validate_rows_for_datasets(dataset_list, row_source_ids, row_indices)
    inner = _core.plan_compile(
        [dataset._require_inner() for dataset in dataset_list],
        source_ids,
        [dataset._feature_map_array for dataset in dataset_list],
        row_source_ids,
        row_indices,
        output._to_core(),
        normalized_batch_size,
        normalized_prefetch_step,
        config._to_core(),
    )
    template = {
        "sources": [
            _source_spec(source_id, dataset)
            for source_id, dataset in enumerate(dataset_list)
        ],
        "row_source_ids": None if row_source_ids is None else row_source_ids.copy(),
        "row_indices": row_indices.copy(),
        "config": asdict(config),
    }
    return Plan(inner, output, template=template)


def prefetch(
    datasets: Dataset | Iterable[Dataset],
    rows: Iterable[int | RowRef | tuple[int, int]] | NDArray[np.integer[Any]],
    *,
    output: OutputSpec | None = None,
    batch_size: int = 256,
    prefetch_step: int = 8,
    plan_config: PlanConfig | None = None,
    config: SessionConfig | None = None,
) -> Prefetch:
    """Compile ``datasets``/``rows`` and return a lazy batch iterator."""
    plan = compile(
        datasets,
        rows,
        output=output,
        batch_size=batch_size,
        prefetch_step=prefetch_step,
        config=plan_config,
    )
    return plan.prefetch(config)


def _source_spec(source_id: int, dataset: Dataset) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "path": os.fspath(dataset.path.resolve()),
        "key": dataset.key,
        "kind": dataset.kind,
        "shape": list(dataset.shape),
        "dtype": dataset.dtype.name,
        "feature_map": None if dataset.feature_map is None else list(dataset.feature_map),
        "limits": asdict(dataset.limits),
        "manifest": _source_manifest(dataset.path, dataset.key),
    }


def _source_manifest(path: str | os.PathLike[str], key: str) -> dict[str, Any]:
    location = resolve_matrix_location(path, key)
    if location.kind == "dir":
        meta = location.open_path / "meta.json"
        manifest: dict[str, Any] = {
            "kind": "dir",
            "meta_sha256": hashlib.sha256(meta.read_bytes()).hexdigest(),
        }
        indptr = location.open_path / "indptr"
        if indptr.is_file():
            size = indptr.stat().st_size
            manifest["indptr_size"] = size
            if size <= 64 * 1024 * 1024:
                manifest["indptr_sha256"] = hashlib.sha256(indptr.read_bytes()).hexdigest()
        return manifest
    prefix = location.zip_prefix or ""
    with zipfile.ZipFile(location.open_path) as archive:
        entries = {}
        for name in ("meta.json", "indptr"):
            logical = f"{prefix}/{name}" if prefix else name
            try:
                info = archive.getinfo(logical)
            except KeyError:
                continue
            entries[name] = {
                "crc32": info.CRC,
                "compressed_size": info.compress_size,
                "size": info.file_size,
            }
    return {"kind": "zip", "entries": entries}


def _dataset_from_source_spec(source: Mapping[str, Any]) -> Dataset:
    return Dataset(
        source["path"],
        key=source["key"],
        feature_map=source["feature_map"],
        feature_names=None,
        obs_names=None,
        limits=ReadLimits(**source["limits"]),
    )


def _plan_config_from_dict(values: Mapping[str, Any]) -> PlanConfig:
    config = dict(values)
    config["limits"] = ResourceLimits(**config["limits"])
    config["io_merge"] = IoMergeConfig(**config["io_merge"])
    return PlanConfig(**config)


def _copy_template(template: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sources": [dict(source) for source in template["sources"]],
        "row_source_ids": (
            None
            if template["row_source_ids"] is None
            else np.asarray(template["row_source_ids"], dtype=np.uint32).copy()
        ),
        "row_indices": np.asarray(template["row_indices"], dtype=np.uint64).copy(),
        "config": json.loads(json.dumps(template["config"])),
    }


def _encode_plan_image(
    template: Mapping[str, Any],
    meta: Mapping[str, Any],
    stats: PlanStats,
    output: OutputSpec,
) -> bytes:
    source_ids = template["row_source_ids"]
    source_bytes = (
        b""
        if source_ids is None
        else np.asarray(source_ids, dtype="<u4", order="C").tobytes()
    )
    row_bytes = np.asarray(template["row_indices"], dtype="<u8", order="C").tobytes()
    stats_values = stats.as_dict()
    profile = stats_values.pop("profile")
    stats_values.update(profile)
    document = {
        "sources": template["sources"],
        "config": template["config"],
        "meta": dict(meta),
        "stats": stats_values,
        "output": output.as_dict(),
        "has_row_source_ids": source_ids is not None,
    }
    metadata = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    body = metadata + source_bytes + row_bytes
    total = _PLAN_HEADER.size + len(body)
    if total > _MAX_PLAN_IMAGE_BYTES:
        raise ValueError(f"plan image requires {total} bytes, limit is {_MAX_PLAN_IMAGE_BYTES}")
    header = _PLAN_HEADER.pack(
        _PLAN_MAGIC,
        _PLAN_FORMAT_MAJOR,
        _PLAN_FORMAT_MINOR,
        len(metadata),
        len(source_bytes),
        len(row_bytes),
        hashlib.sha256(body).digest(),
    )
    return header + body


def _decode_plan_image(
    blob: bytes | bytearray | memoryview,
    *,
    base_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], OutputSpec]:
    try:
        view = memoryview(blob).cast("B")
    except TypeError as error:
        raise TypeError("plan image must be bytes-like") from error
    if len(view) > _MAX_PLAN_IMAGE_BYTES:
        raise ValueError("plan image exceeds the configured size limit")
    if len(view) < _PLAN_HEADER.size:
        raise ValueError("plan image is shorter than its header")
    magic, major, minor, metadata_len, source_len, row_len, digest = _PLAN_HEADER.unpack(
        view[: _PLAN_HEADER.size]
    )
    if magic != _PLAN_MAGIC:
        raise ValueError("invalid plan image magic")
    if major != _PLAN_FORMAT_MAJOR or minor > _PLAN_FORMAT_MINOR:
        raise ValueError(f"unsupported plan image version {major}.{minor}")
    body_len = metadata_len + source_len + row_len
    if body_len != len(view) - _PLAN_HEADER.size:
        raise ValueError("plan image section lengths do not match total length")
    body = view[_PLAN_HEADER.size :]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("plan image checksum mismatch")
    metadata_end = metadata_len
    source_end = metadata_end + source_len
    try:
        document = json.loads(bytes(body[:metadata_end]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid plan metadata JSON") from error
    if not isinstance(document, dict):
        raise ValueError("plan metadata must be a JSON object")
    row_payload = body[source_end:]
    if row_len % 8:
        raise ValueError("row-index section is not aligned to uint64")
    row_indices = np.frombuffer(row_payload, dtype="<u8").astype(np.uint64, copy=True)
    has_source_ids = document.get("has_row_source_ids")
    if has_source_ids:
        if source_len % 4:
            raise ValueError("row-source section is not aligned to uint32")
        row_source_ids = np.frombuffer(
            body[metadata_end:source_end], dtype="<u4"
        ).astype(np.uint32, copy=True)
        if row_source_ids.size != row_indices.size:
            raise ValueError("row-source and row-index sections have different lengths")
    elif source_len != 0:
        raise ValueError("plan has row-source bytes without the corresponding flag")
    else:
        row_source_ids = None
    sources = document.get("sources")
    if not isinstance(sources, list):
        raise ValueError("plan sources must be a list")
    normalized_sources = []
    for expected_id, source in enumerate(sources):
        if not isinstance(source, dict) or source.get("source_id") != expected_id:
            raise ValueError("plan source IDs must be contiguous and ordered")
        normalized = dict(source)
        path = Path(os.fspath(normalized["path"]))
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        normalized["path"] = os.fspath(path)
        normalized_sources.append(normalized)
    output_values = document["output"]
    output = OutputSpec(
        output_values["n_cols"],
        output_values["dtype"],
        fill=output_values["fill"],
        overflow=output_values["overflow"],
        overflow_value=output_values["overflow_value"],
        allow_float_rounding=output_values["allow_float_rounding"],
    )
    template = {
        "sources": normalized_sources,
        "row_source_ids": row_source_ids,
        "row_indices": row_indices,
        "config": document["config"],
    }
    return template, document["meta"], document["stats"], output


def _restore_plan_v1(blob: bytes) -> Plan:
    return Plan.loads(blob, lazy=True)


def _normalize_datasets(datasets: Dataset | Iterable[Dataset]) -> list[Dataset]:
    if isinstance(datasets, Dataset):
        return [datasets]
    try:
        dataset_list = list(datasets)
    except TypeError as error:
        raise TypeError("datasets must be a Dataset or iterable of Dataset objects") from error
    for index, dataset in enumerate(dataset_list):
        if not isinstance(dataset, Dataset):
            raise TypeError(f"datasets[{index}] must be a Dataset instance")
    return dataset_list


def _infer_output(datasets: list[Dataset]) -> OutputSpec:
    if len(datasets) != 1 or not _has_identity_feature_map(datasets[0]):
        raise ValueError("output is required for multiple or feature-mapped datasets")
    dataset = datasets[0]
    return OutputSpec(dataset.n_cols, dataset.dtype)


def _validate_datasets_for_output(datasets: list[Dataset], output: OutputSpec) -> None:
    for source_id, dataset in enumerate(datasets):
        if dataset.feature_map is None:
            if dataset.n_cols != output.n_cols:
                raise ValueError(
                    f"dataset {source_id} has {dataset.n_cols} columns, "
                    f"but output has {output.n_cols}; an explicit feature_map is required"
                )
            continue
        invalid = next(
            (
                target
                for target in dataset.feature_map
                if target is not None and target >= output.n_cols
            ),
            None,
        )
        if invalid is not None:
            raise ValueError(
                f"dataset {source_id} maps a feature to output column {invalid}, "
                f"but output has {output.n_cols} columns"
            )


def _has_identity_feature_map(dataset: Dataset) -> bool:
    mapping = dataset.feature_map
    return mapping is None or all(target == column for column, target in enumerate(mapping))


def _validate_rows_for_datasets(
    datasets: list[Dataset],
    source_ids: NDArray[np.uint32] | None,
    row_indices: NDArray[np.uint64],
) -> None:
    if row_indices.size == 0:
        return
    if source_ids is None:
        row_limit = datasets[0].n_rows
        if int(row_indices.max()) >= row_limit:
            position = int(np.flatnonzero(row_indices >= row_limit)[0])
            raise ValueError(
                f"rows[{position}].row={int(row_indices[position])} is outside "
                f"source 0 with {row_limit} rows"
            )
        return

    source_count = len(datasets)
    maximum_source = int(source_ids.max())
    if maximum_source >= source_count:
        position = int(np.flatnonzero(source_ids >= source_count)[0])
        available = (
            "no datasets are registered"
            if source_count == 0
            else (f"source_id must be in [0, {source_count - 1}]")
        )
        raise ValueError(
            f"rows[{position}].source_id={int(source_ids[position])} is invalid; {available}"
        )

    if int(row_indices.max()) < min(dataset.n_rows for dataset in datasets):
        return

    maximum_rows = np.zeros(source_count, dtype=np.uint64)
    present = np.zeros(source_count, dtype=np.bool_)
    np.maximum.at(maximum_rows, source_ids, row_indices)
    present[source_ids] = True
    for source_id, dataset in enumerate(datasets):
        if not present[source_id] or int(maximum_rows[source_id]) < dataset.n_rows:
            continue
        invalid = (source_ids == source_id) & (row_indices >= dataset.n_rows)
        position = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"rows[{position}].row={int(row_indices[position])} is outside "
            f"source {source_id} with {dataset.n_rows} rows"
        )


def _resolve_session_config(config: SessionConfig | None) -> SessionConfig:
    if config is None:
        return SessionConfig()
    if not isinstance(config, SessionConfig):
        raise TypeError("config must be a SessionConfig instance")
    return config
