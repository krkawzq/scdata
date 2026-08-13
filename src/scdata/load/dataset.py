"""Dataset registration over AnnData ``.scc`` / ``.scc.zip`` matrices."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from scdata import _core
from scdata.load._validation import as_int, dtype_from_core, normalize_feature_map
from scdata.compress.limits import ReadLimits
from scdata.exceptions import _call_core
from scdata.load.scc import normalize_matrix_key, read_feature_names, resolve_matrix_location

DatasetKind = Literal["dense", "csr"]

__all__ = ["Dataset", "DatasetKind", "RowRef", "register"]


def _coerce_read_limits(limits: object | None) -> ReadLimits:
    if limits is None:
        return ReadLimits()
    if isinstance(limits, ReadLimits):
        return limits
    try:
        return ReadLimits(
            max_metadata_size=limits.max_metadata_size,  # type: ignore[attr-defined]
            max_encoded_size=limits.max_encoded_size,  # type: ignore[attr-defined]
            max_decoded_size=limits.max_decoded_size,  # type: ignore[attr-defined]
            max_block_count=limits.max_block_count,  # type: ignore[attr-defined]
        )
    except AttributeError as error:
        raise TypeError("limits must be scdata.ReadLimits") from error


class Dataset:
    """An immutable handle to one sc-compress matrix inside an scc container.

    Construct with :func:`register`. The optional ``feature_map`` is supplied by
    the caller; this type does not perform gene alignment.
    """

    __slots__ = (
        "_feature_map",
        "_feature_map_array",
        "_feature_names",
        "_inner",
        "_meta",
        "_key",
        "_limits",
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
        limits: object | None = None,
        _inner: _core._Dataset | None = None,
        _location_key: str | None = None,
        _location_zip_prefix: str | None = None,
    ) -> None:
        limits = _coerce_read_limits(limits)

        if _inner is None:
            location = resolve_matrix_location(path, key)
            inner = _call_core(
                _core.dataset_open,
                location.open_path,
                zip_prefix=location.zip_prefix,
                maximum_metadata_size=limits.max_metadata_size,
                maximum_encoded_size=limits.max_encoded_size,
                maximum_decoded_size=limits.max_decoded_size,
                maximum_block_count=limits.max_block_count,
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

        if isinstance(feature_names, str):
            if feature_names != "auto":
                raise TypeError(
                    "feature_names must be 'auto', None, or an iterable of names; "
                    "wrap a single name in a list"
                )
            resolved_names = read_feature_names(container, resolved_key)
        elif isinstance(feature_names, bytes):
            raise TypeError("feature_names must be an iterable of names, not bytes")
        elif feature_names is None:
            resolved_names = None
        else:
            try:
                resolved_names = tuple(str(name) for name in feature_names)
            except TypeError as error:
                raise TypeError(
                    "feature_names must be 'auto', None, or an iterable of names"
                ) from error

        meta = _call_core(_core.dataset_meta, inner)
        n_cols = int(meta["n_cols"])
        if feature_map is None:
            map_tuple = None
            map_array = None
        else:
            map_array = normalize_feature_map(feature_map, n_cols)
            map_array.flags.writeable = False
            map_tuple = tuple(None if value == -1 else int(value) for value in map_array)

        if resolved_names is not None and len(resolved_names) != n_cols:
            raise ValueError(
                f"feature_names has length {len(resolved_names)}, "
                f"but the matrix has {n_cols} columns"
            )

        self._inner = inner
        self._meta = meta
        self._path = container
        self._key = resolved_key
        self._zip_prefix = zip_prefix
        self._limits = limits
        self._feature_names = resolved_names
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
    def dtype(self) -> np.dtype[Any]:
        return dtype_from_core(self._meta["dtype"])

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        return self._feature_names

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

    def with_feature_map(self, feature_map: Iterable[int | None] | None) -> Dataset:
        """Return a new handle sharing the same store with a different map."""
        return Dataset(
            self._path,
            key=self._key,
            feature_map=feature_map,
            feature_names=self._feature_names,
            limits=self._limits,
            _inner=self._inner,
            _location_key=self._key,
            _location_zip_prefix=self._zip_prefix,
        )

    def __len__(self) -> int:
        return self.n_rows

    def info(self) -> dict[str, object]:
        """Return compact, serialization-friendly dataset diagnostics."""
        return {
            "path": str(self._path),
            "key": self._key,
            "zip_prefix": self._zip_prefix,
            "kind": self.kind,
            "shape": self.shape,
            "dtype": self.dtype.name,
            "feature_names_loaded": self._feature_names is not None,
            "mapped_features": self.n_mapped_features,
            "dropped_features": self.n_dropped_features,
        }

    def __repr__(self) -> str:
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
    path: str | os.PathLike[str],
    *,
    key: str = "X",
    feature_map: Iterable[int | None] | None = None,
    feature_names: Iterable[str] | Literal["auto"] | None = "auto",
    limits: object | None = None,
) -> Dataset:
    """Open one matrix inside an AnnData ``.scc`` / ``.scc.zip`` container.

    ``feature_map`` must be built by the caller when projection is needed.
    Expression keys expose ``feature_names`` from ``var`` / ``raw/var`` for
    convenience; embedding keys leave them as ``None``. Pass
    ``feature_names=None`` to skip metadata discovery, or an iterable to supply
    names directly.

    ``limits`` accepts :class:`scdata.ReadLimits`.
    """
    return Dataset(
        path,
        key=key,
        feature_map=feature_map,
        feature_names=feature_names,
        limits=limits,
    )
