"""Dataset registration over AnnData ``.scc`` / ``.scc.zip`` matrices."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

from scdata import _core
from scdata.compress._csr import ScCsr
from scdata.compress._dense import ScDense
from scdata.compress._limits import ReadLimits, resolve_read_limits
from scdata.load._location import (
    normalize_matrix_key,
    read_feature_names,
    read_obs_names,
    resolve_matrix_location,
)
from scdata.load._names import NameSequence, build_feature_map, locate_names
from scdata.load._validation import as_int, dtype_from_core, normalize_feature_map

DatasetKind = Literal["dense", "csr"]
_PICKLE_VERSION = 1

__all__ = ["Dataset", "DatasetKind", "RowRef", "register"]


def _coerce_read_limits(limits: object) -> ReadLimits:
    try:
        return ReadLimits(
            max_metadata_size=limits.max_metadata_size,  # type: ignore[attr-defined]
            max_encoded_size=limits.max_encoded_size,  # type: ignore[attr-defined]
            max_decoded_size=limits.max_decoded_size,  # type: ignore[attr-defined]
            max_block_count=limits.max_block_count,  # type: ignore[attr-defined]
            num_workers=limits.num_workers,  # type: ignore[attr-defined]
        )
    except AttributeError as error:
        raise TypeError("limits must be scdata.ReadLimits") from error


def _as_read_limits(limits: object | None) -> ReadLimits | None:
    if limits is None or isinstance(limits, ReadLimits):
        return limits
    return _coerce_read_limits(limits)


def _resolve_names(
    value: Iterable[str] | Literal["auto"] | None,
    *,
    argument: str,
    auto: Any,
) -> tuple[str, ...] | None:
    if isinstance(value, str):
        if value != "auto":
            raise TypeError(
                f"{argument} must be 'auto', None, or an iterable of names; "
                "wrap a single name in a list"
            )
        return auto()
    if isinstance(value, bytes):
        raise TypeError(f"{argument} must be an iterable of names, not bytes")
    if value is None:
        return None
    try:
        return tuple(str(name) for name in value)
    except TypeError as error:
        raise TypeError(
            f"{argument} must be 'auto', None, or an iterable of names"
        ) from error


class Dataset:
    """An immutable handle to one sc-compress matrix inside an scc container.

    Construct with :func:`register`. Gene alignment is opt-in via
    :meth:`with_feature_map` / :meth:`with_aligned_features` or
    :func:`~scdata.load.build_feature_map`.
    """

    __slots__ = (
        "_feature_map",
        "_feature_map_array",
        "_feature_names",
        "_inner",
        "_meta",
        "_key",
        "_limits",
        "_obs_names",
        "_path",
        "_zip_prefix",
    )

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        key: str = "X",
        feature_map: Iterable[int | None] | None = None,
        feature_names: Iterable[str] | Literal["auto"] | None = "auto",
        obs_names: Iterable[str] | Literal["auto"] | None = "auto",
        limits: object | None = None,
        max_metadata_size: object | None = None,
        max_encoded_size: object | None = None,
        max_decoded_size: object | None = None,
        max_block_count: object | None = None,
        num_workers: object | None = None,
        _inner: _core._Dataset | None = None,
        _location_key: str | None = None,
        _location_zip_prefix: str | None = None,
    ) -> None:
        limits = resolve_read_limits(
            _as_read_limits(limits),
            max_metadata_size=max_metadata_size,
            max_encoded_size=max_encoded_size,
            max_decoded_size=max_decoded_size,
            max_block_count=max_block_count,
            num_workers=num_workers,
        )

        if _inner is None:
            location = resolve_matrix_location(path, key)
            inner = _core.dataset_open(
                location.open_path,
                zip_prefix=location.zip_prefix,
                max_metadata_size=limits.max_metadata_size,
                max_encoded_size=limits.max_encoded_size,
                max_decoded_size=limits.max_decoded_size,
                max_block_count=limits.max_block_count,
                num_workers=limits.num_workers,
            )
            container = location.container
            resolved_key = location.key
            zip_prefix = location.zip_prefix
        else:
            try:
                container = Path(os.fspath(path))
            except TypeError as error:
                raise TypeError("path must be str or os.PathLike[str]") from error
            inner = _inner
            resolved_key = _location_key if _location_key is not None else normalize_matrix_key(key)
            zip_prefix = _location_zip_prefix

        resolved_feature_names = _resolve_names(
            feature_names,
            argument="feature_names",
            auto=lambda: read_feature_names(container, resolved_key),
        )
        resolved_obs_names = _resolve_names(
            obs_names,
            argument="obs_names",
            auto=lambda: read_obs_names(container, resolved_key),
        )

        meta = _core.dataset_meta(inner)
        n_cols = int(meta["n_cols"])
        n_rows = int(meta["n_rows"])
        if feature_map is None:
            map_tuple = None
            map_array = None
        else:
            map_array = normalize_feature_map(feature_map, n_cols)
            map_array.flags.writeable = False
            map_tuple = tuple(None if value == -1 else int(value) for value in map_array)

        if resolved_feature_names is not None and len(resolved_feature_names) != n_cols:
            raise ValueError(
                f"feature_names has length {len(resolved_feature_names)}, "
                f"but the matrix has {n_cols} columns"
            )
        if resolved_obs_names is not None and len(resolved_obs_names) != n_rows:
            raise ValueError(
                f"obs_names has length {len(resolved_obs_names)}, "
                f"but the matrix has {n_rows} rows"
            )

        self._inner: _core._Dataset | None = inner
        self._meta = meta
        self._path = container
        self._key = resolved_key
        self._zip_prefix = zip_prefix
        self._limits = limits
        self._feature_names = resolved_feature_names
        self._obs_names = resolved_obs_names
        self._feature_map = map_tuple
        self._feature_map_array = map_array

    @property
    def path(self) -> Path:
        return self._path

    @property
    def key(self) -> str:
        return self._key

    @property
    def zip_prefix(self) -> str | None:
        return self._zip_prefix

    @property
    def limits(self) -> ReadLimits:
        return self._limits

    @property
    def num_workers(self) -> int:
        return self._limits.num_workers

    @property
    def kind(self) -> DatasetKind:
        return self._meta["kind"]

    @property
    def shape(self) -> tuple[int, int]:
        return self._meta["shape"]

    @property
    def n_rows(self) -> int:
        return int(self._meta["n_rows"])

    @property
    def n_cols(self) -> int:
        return int(self._meta["n_cols"])

    @property
    def n_obs(self) -> int:
        """Alias of :attr:`n_rows`."""
        return self.n_rows

    @property
    def n_vars(self) -> int:
        """Alias of :attr:`n_cols` (features or embedding width)."""
        return self.n_cols

    @property
    def dtype(self) -> np.dtype[Any]:
        return dtype_from_core(self._meta["dtype"])

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        return self._feature_names

    @property
    def var_names(self) -> tuple[str, ...] | None:
        """Alias of :attr:`feature_names`."""
        return self._feature_names

    @property
    def obs_names(self) -> tuple[str, ...] | None:
        return self._obs_names

    @property
    def feature_map(self) -> tuple[int | None, ...] | None:
        return self._feature_map

    @property
    def n_mapped_features(self) -> int:
        """Number of source features copied into each output row."""
        if self._feature_map is None:
            return self.n_cols
        return sum(target is not None for target in self._feature_map)

    @property
    def n_dropped_features(self) -> int:
        """Number of source features omitted by ``feature_map``."""
        return self.n_cols - self.n_mapped_features

    @property
    def closed(self) -> bool:
        return self._inner is None

    def with_feature_map(self, feature_map: Iterable[int | None] | None) -> Dataset:
        """Return a new handle sharing the same store with a different map."""
        return Dataset(
            self._path,
            key=self._key,
            feature_map=feature_map,
            feature_names=self._feature_names,
            obs_names=self._obs_names,
            limits=self._limits,
            _inner=self._require_inner(),
            _location_key=self._key,
            _location_zip_prefix=self._zip_prefix,
        )

    def with_aligned_features(self, target_names: NameSequence) -> Dataset:
        """Return a handle mapped onto ``target_names`` by exact string identity."""
        if self._feature_names is None:
            raise ValueError(
                "dataset has no feature_names; pass them to register() "
                "or build a map with scdata.load.build_feature_map"
            )
        return self.with_feature_map(build_feature_map(self._feature_names, target_names))

    def rows_for(
        self,
        names: NameSequence,
        *,
        missing: Literal["error", "drop"] = "error",
    ) -> NDArray[np.uint64]:
        """Locate observation names and return row indices for :func:`compile`."""
        if self._obs_names is None:
            raise ValueError(
                "dataset has no obs_names; pass them to register() "
                "or look them up with scdata.load.locate_names"
            )
        return locate_names(self._obs_names, names, missing=missing)

    def close(self) -> None:
        """Drop the native reader. Metadata and names remain available."""
        self._inner = None

    def _require_inner(self) -> _core._Dataset:
        if self._inner is None:
            raise ValueError("I/O operation on closed Dataset")
        return self._inner

    def __len__(self) -> int:
        return self.n_rows

    def __enter__(self) -> Self:
        self._require_inner()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "version": _PICKLE_VERSION,
            "path": os.fspath(self._path),
            "key": self._key,
            "feature_map": self._feature_map,
            "feature_names": self._feature_names,
            "obs_names": self._obs_names,
            "limits": self._limits,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("Dataset pickle state must be a dict")
        version = state.get("version")
        if version != _PICKLE_VERSION:
            raise ValueError(f"unsupported Dataset pickle version {version!r}")
        Dataset.__init__(
            self,
            state["path"],
            key=state["key"],
            feature_map=state["feature_map"],
            feature_names=state["feature_names"],
            obs_names=state["obs_names"],
            limits=state["limits"],
        )

    def info(self) -> dict[str, object]:
        """Return compact, serialization-friendly dataset diagnostics."""
        return {
            "path": str(self._path),
            "key": self._key,
            "zip_prefix": self._zip_prefix,
            "kind": self.kind,
            "shape": self.shape,
            "dtype": self.dtype.name,
            "num_workers": self._limits.num_workers,
            "closed": self.closed,
            "obs_names_loaded": self._obs_names is not None,
            "feature_names_loaded": self._feature_names is not None,
            "mapped_features": self.n_mapped_features,
            "dropped_features": self.n_dropped_features,
        }

    def __repr__(self) -> str:
        if self.closed:
            return (
                f"Dataset(path={str(self._path)!r}, key={self._key!r}, closed=True)"
            )
        mapped = (
            "identity"
            if self._feature_map is None
            else f"{self.n_mapped_features}/{self.n_cols} mapped"
        )
        return (
            f"Dataset(kind={self.kind!r}, shape={self.shape!r}, dtype={self.dtype.name!r}, "
            f"path={str(self._path)!r}, key={self._key!r}, feature_map={mapped!r})"
        )


@dataclass(frozen=True, slots=True)
class RowRef:
    """One ordered ``(source_id, row)`` request."""

    source_id: int
    row: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            as_int(self.source_id, "source_id", maximum=(1 << 32) - 1),
        )
        object.__setattr__(self, "row", as_int(self.row, "row", maximum=(1 << 64) - 1))


def register(
    source: str | os.PathLike[str] | ScDense | ScCsr,
    *,
    key: str = "X",
    feature_map: Iterable[int | None] | None = None,
    feature_names: Iterable[str] | Literal["auto"] | None = "auto",
    obs_names: Iterable[str] | Literal["auto"] | None = "auto",
    limits: object | None = None,
    max_metadata_size: object | None = None,
    max_encoded_size: object | None = None,
    max_decoded_size: object | None = None,
    max_block_count: object | None = None,
    num_workers: object | None = None,
) -> Dataset:
    """Open one SCC matrix as a prefetch source.

    ``source`` may be an AnnData ``.scc`` / ``.scc.zip`` container, a bare
    store directory, or an already opened :class:`~scdata.compress.ScDense` /
    :class:`~scdata.compress.ScCsr`. Keyword resource overrides apply on top of
    ``limits`` or the source handle's limits.

    ``feature_names='auto'`` loads ``var`` / ``raw/var`` for expression keys.
    ``obs_names='auto'`` loads container-level ``obs`` for cell-aligned keys.
    """
    if isinstance(source, (ScDense, ScCsr)):
        resolved_key = source.zip_prefix if source.zip_prefix is not None else key
        return Dataset(
            source.path,
            key=resolved_key,
            feature_map=feature_map,
            feature_names=feature_names,
            obs_names=obs_names,
            limits=source.limits if limits is None else limits,
            max_metadata_size=max_metadata_size,
            max_encoded_size=max_encoded_size,
            max_decoded_size=max_decoded_size,
            max_block_count=max_block_count,
            num_workers=num_workers,
        )
    return Dataset(
        source,
        key=key,
        feature_map=feature_map,
        feature_names=feature_names,
        obs_names=obs_names,
        limits=limits,
        max_metadata_size=max_metadata_size,
        max_encoded_size=max_encoded_size,
        max_decoded_size=max_decoded_size,
        max_block_count=max_block_count,
        num_workers=num_workers,
    )
