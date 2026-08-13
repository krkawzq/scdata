"""Python helpers for string-identity name alignment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["as_str_tuple", "build_feature_map", "locate_names"]

NameSequence = Any
MissingPolicy = Literal["error", "drop"]


def as_str_tuple(names: NameSequence, *, argument: str = "names") -> tuple[str, ...]:
    """Coerce a 1-D name sequence to ``tuple[str, ...]``.

    Accepts lists, tuples, NumPy 1-D arrays, and pandas ``Index`` / ``Series``.
    A bare ``str`` / ``bytes`` is rejected so a single name is not iterated
    character-wise.
    """
    if names is None:
        raise TypeError(f"{argument} must be a 1-D sequence of names")
    if isinstance(names, (str, bytes)):
        raise TypeError(
            f"{argument} must be a 1-D sequence of names; wrap a single name in a list"
        )
    if isinstance(names, np.ndarray):
        if names.ndim != 1:
            raise ValueError(f"{argument} must be 1-D, got shape {names.shape}")
        return tuple(map(str, names.tolist()))
    ndim = getattr(names, "ndim", None)
    if isinstance(ndim, int) and ndim != 1:
        raise ValueError(f"{argument} must be 1-D, got ndim={ndim}")
    try:
        iterator = iter(names)
    except TypeError as error:
        raise TypeError(f"{argument} must be a 1-D sequence of names") from error
    return tuple(str(name) for name in iterator)


def build_feature_map(
    source_names: NameSequence,
    target_names: NameSequence,
) -> tuple[int | None, ...]:
    """Map source feature names onto target columns by exact string identity.

    The returned tuple has length ``len(source_names)``. ``result[i]`` is the
    first index of ``source_names[i]`` in ``target_names``, or ``None`` when the
    name is absent. Each target column is claimed at most once: later source
    columns that repeat an already-mapped name are dropped. Duplicate target
    names keep the first occurrence.
    """
    source = as_str_tuple(source_names, argument="source_names")
    target = as_str_tuple(target_names, argument="target_names")
    target_index: dict[str, int] = {}
    for position, name in enumerate(target):
        if name not in target_index:
            target_index[name] = position
    taken = [False] * len(target)
    result: list[int | None] = [None] * len(source)
    for source_column, name in enumerate(source):
        target_column = target_index.get(name)
        if target_column is None or taken[target_column]:
            continue
        taken[target_column] = True
        result[source_column] = target_column
    return tuple(result)


def locate_names(
    names: NameSequence,
    requested: NameSequence,
    *,
    missing: MissingPolicy = "error",
) -> NDArray[np.uint64]:
    """Return positions of ``requested`` inside ``names`` (first match wins).

    ``missing='error'`` (default) raises ``ValueError`` on the first absent
    name. ``missing='drop'`` omits absent names and preserves request order.
    """
    if missing not in {"error", "drop"}:
        raise ValueError("missing must be 'error' or 'drop'")
    catalog = as_str_tuple(names, argument="names")
    query = as_str_tuple(requested, argument="requested")
    index: dict[str, int] = {}
    for position, name in enumerate(catalog):
        if name not in index:
            index[name] = position
    if missing == "error":
        out = np.empty(len(query), dtype=np.uint64)
        for position, name in enumerate(query):
            try:
                out[position] = index[name]
            except KeyError:
                raise ValueError(f"name {name!r} is not present") from None
        return out
    found = [index[name] for name in query if name in index]
    return np.asarray(found, dtype=np.uint64)
