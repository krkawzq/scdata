"""High-level corpus entry point: paths to bank to loader in one object.

:class:`Corpus` is the one-stop entry for training: give it a list of
``.zarr.zip`` stores (or any path :func:`~scdata.io.launch` accepts) and it
parses, registers, gene-aligns, and builds a torch-style
:class:`~scdata.data.ScDataLoader` over them.  When it creates the
:class:`~scdata.databank.ScDataBank` it owns it, and it exposes
:attr:`gene_names` / :attr:`missing` chosen so that mixed-file batches share a
single column layout.

Three entry points cover the control spectrum:

* ``Corpus(paths)`` — owns the bank (most convenient; the ``with`` form
  releases it on exit);
* :meth:`Corpus.from_bank` — reuse an existing bank the caller owns;
* :meth:`scdata.data.ScDataLoader.from_paths` — just give me a loader.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, replace
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, cast

from scdata.data._collate import stitch_dense_collate
from scdata.data._dataloader import ScDataBatch, ScDataLoader
from scdata.data._dataset import DatasetCollection, DenseDataset, SparseDataset
from scdata.data._index import CellIndexDataset
from scdata.data._stats import _bank_config_summary
from scdata.databank import (
    DataBankConfig,
    DatasetId,
    MissingGenePolicy,
    ScheduledPrefetchConfig,
    ScDataBank,
    _coerce_missing_policy,
)
from scdata.io import launch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from scdata.data._dataset import DType
    from scdata.data._stats import BankConfigSummary

__all__ = ["Corpus"]

_GeneAlignment = Literal["strict", "union", "intersection", "none"]


@dataclass(frozen=True, slots=True)
class _CorpusEntry:
    dataset: DenseDataset | SparseDataset
    matrix_key: str


class Corpus:
    """A registered multi-dataset corpus ready to build training loaders.

    Args:
        paths: Dataset sources in ``file_id`` order.  Each item may be a store
            path, ``(path, matrix_key)``, a mapping, or an already-constructed
            :class:`DenseDataset` / :class:`SparseDataset`.
        bank: Optional pre-built bank.  When ``None`` (default) a bank is
            created from ``bank_config`` and owned by this Corpus (closed on
            :meth:`close`); when supplied, the caller owns it and ``close``
            is a no-op for the bank.
        bank_config: Config for the self-owned bank (ignored with a warning
            when ``bank`` is given).
        layer / matrix: Forwarded to :func:`~scdata.io.launch` to select a
            non-default matrix (``X`` by default).
        gene_alignment: How to reconcile per-dataset gene sets:

            * ``"strict"`` (default) — every dataset must have the same gene
              names in the same order, else :class:`ValueError`;
            * ``"union"`` — project onto the union (first-seen order), with
              ``missing=ZERO`` so absent genes are zero-filled;
            * ``"intersection"`` — project onto the shared subset;
            * ``"none"`` — no projection (``genes=None``); mixed-file batches
              are the caller's responsibility.

        missing: Missing-gene policy for the projection.  ``None`` (default)
            is derived from ``gene_alignment`` (``ZERO`` for ``"union"``,
            ``None`` otherwise); an explicit value overrides.
    """

    __slots__ = (
        "_bank",
        "_owns_bank",
        "_bank_config",
        "_dataset_ids",
        "_matrix_keys",
        "_gene_names",
        "_missing",
        "_cells_per_file",
    )

    def __init__(
        self,
        paths: "Iterable[str | PathLike[str] | tuple[str | PathLike[str], str] | Mapping[str, Any] | DenseDataset | SparseDataset]",
        *,
        bank: ScDataBank | None = None,
        bank_config: "DataBankConfig | Mapping[str, Any] | None" = None,
        layer: str | None = None,
        matrix: str | None = None,
        gene_alignment: _GeneAlignment = "strict",
        missing: "MissingGenePolicy | str | None" = None,
    ) -> None:
        entries = _resolve_corpus_entries(paths, layer=layer, matrix=matrix)
        if not entries:
            raise ValueError("paths must be a non-empty iterable of store paths")
        if gene_alignment not in ("strict", "union", "intersection", "none"):
            raise ValueError(
                "gene_alignment must be 'strict', 'union', 'intersection', or 'none'; "
                f"got {gene_alignment!r}"
            )

        owns_bank = bank is None
        if bank is None:
            bank = ScDataBank(bank_config)
            self._bank_config: DataBankConfig | None = _normalize_bank_config(bank_config)
        else:
            if bank_config is not None:
                warnings.warn(
                    "bank_config is ignored when an explicit bank is provided",
                    stacklevel=2,
                )
            self._bank_config = bank.config
        self._bank: ScDataBank = bank
        self._owns_bank = owns_bank

        # Parse, register, and gene-align.  If any of this raises after the bank
        # was created, tear the bank down deterministically — ``__exit__`` won't
        # fire because ``__init__`` never finished, so an owned bank would
        # otherwise leak its Rust thread pools to the GC.
        # Parse, register, and gene-align.  If any of this raises after datasets
        # were registered, roll the registration back so neither an owned nor an
        # external bank is left with dangling datasets / file handles.  An owned
        # bank is then closed; an external bank is left open for the caller.
        registered: list[DatasetId] = []
        try:
            datasets = [entry.dataset for entry in entries]
            registered = bank.register(datasets)
            # register(Iterable[Dataset]) -> list[DatasetId], in input order.
            self._dataset_ids: tuple[DatasetId, ...] = tuple(registered)
            self._matrix_keys: tuple[str, ...] = tuple(entry.matrix_key for entry in entries)
            self._cells_per_file: tuple[int, ...] = tuple(
                bank.dataset_num_cells(did) for did in self._dataset_ids
            )
            self._gene_names, self._missing = self._resolve_gene_alignment(gene_alignment, missing)
        except Exception:
            if registered and not bank.is_closed:
                bank.unregister_all(registered)
            if owns_bank and not bank.is_closed:
                bank.close()
            raise

    # -- properties ----------------------------------------------------------

    @property
    def bank(self) -> ScDataBank:
        """The underlying :class:`~scdata.databank.ScDataBank`."""
        return self._bank

    @property
    def owns_bank(self) -> bool:
        """Whether this Corpus owns (and will close) the bank."""
        return self._owns_bank

    @property
    def dataset_ids(self) -> tuple[DatasetId, ...]:
        """Registered dataset handles, in ``file_id`` order."""
        return self._dataset_ids

    @property
    def matrix_keys(self) -> tuple[str, ...]:
        """Selected source matrix key for each ``file_id`` (for example ``"X"``)."""
        return self._matrix_keys

    @property
    def gene_names(self) -> tuple[str, ...] | None:
        """The shared gene list batches are projected onto (``None`` for ``"none"``)."""
        return self._gene_names

    @property
    def missing(self) -> "MissingGenePolicy | None":
        """The missing-gene policy applied to the projection."""
        return self._missing

    @property
    def num_cells(self) -> int:
        """Total cells across all datasets."""
        return sum(self._cells_per_file)

    @property
    def num_files(self) -> int:
        """Number of registered datasets."""
        return len(self._dataset_ids)

    @property
    def num_genes(self) -> int:
        """Number of gene columns in the projection (first dataset's for ``"none"``)."""
        if self._gene_names is not None:
            return len(self._gene_names)
        if self._dataset_ids:
            return self._bank.dataset_num_genes(self._dataset_ids[0])
        return 0

    @property
    def cells_per_file(self) -> tuple[int, ...]:
        """Per-dataset cell counts, in ``file_id`` order."""
        return self._cells_per_file

    @property
    def bank_config_summary(self) -> "BankConfigSummary | None":
        """Structural summary of the bank config for loader stats.

        The summary is built from the running bank config.  Self-owned corpora
        use the config passed to ``Corpus``; externally-owned corpora read a
        dataclass copy back from :attr:`ScDataBank.config`.
        """
        return _bank_config_summary(self._bank_config, len(self._dataset_ids))

    # -- factories -----------------------------------------------------------

    def cell_dataset(self) -> CellIndexDataset:
        """A fresh flat-index dataset mapping a global index to ``(file_id, cell_id)``."""
        return CellIndexDataset(self._cells_per_file)

    def loader(
        self,
        *,
        batch_size: int = 1024,
        shuffle: bool = True,
        drop_last: bool = False,
        sampler: Any = None,
        batch_sampler: Any = None,
        out_dtype: "DType | str | None" = None,
        prefetch_config: "ScheduledPrefetchConfig | Mapping[str, Any] | None" = None,
        collate_fn: "Callable[[ScDataBatch], Any] | None" = None,
        collect_stats: bool = True,
        **torch_kwargs: Any,
    ) -> ScDataLoader:
        """Build a :class:`~scdata.data.ScDataLoader` over this corpus.

        Defaults are tuned for training: :func:`~scdata.data._collate.stitch_dense_collate`
        (a single dense ``[B, G]`` tensor), ``collect_stats=True`` (so
        :meth:`~scdata.data.ScDataLoader.stats` works out of the box), and
        the gene alignment already baked into :attr:`gene_names` /
        :attr:`missing` so mixed-file batches always share columns.

        ``batch_size`` / ``shuffle`` / ``drop_last`` / ``sampler`` /
        ``batch_sampler`` are passed through to torch's DataLoader; the rest
        of torch's options go in ``**torch_kwargs`` (the same restrictions as
        :class:`~scdata.data.ScDataLoader` apply — no worker processes).
        """
        return ScDataLoader(
            self._bank,
            self.cell_dataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            batch_sampler=batch_sampler,
            dataset_ids=list(self._dataset_ids),
            genes=self._gene_names,
            missing=self._missing,
            out_dtype=out_dtype,
            prefetch_config=prefetch_config,
            collate_fn=collate_fn if collate_fn is not None else stitch_dense_collate,
            collect_stats=collect_stats,
            bank_config_summary=self.bank_config_summary,
            **torch_kwargs,
        )

    @classmethod
    def from_bank(
        cls,
        bank: ScDataBank,
        paths: "Iterable[str | PathLike[str] | tuple[str | PathLike[str], str] | Mapping[str, Any] | DenseDataset | SparseDataset]",
        *,
        layer: str | None = None,
        matrix: str | None = None,
        gene_alignment: _GeneAlignment = "strict",
        missing: "MissingGenePolicy | str | None" = None,
    ) -> "Corpus":
        """Build a Corpus reusing an existing ``bank`` the caller owns."""
        return cls(
            paths,
            bank=bank,
            layer=layer,
            matrix=matrix,
            gene_alignment=gene_alignment,
            missing=missing,
        )

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "Corpus":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the bank if this Corpus owns it; otherwise a no-op.

        Registered datasets are not unregistered individually — closing the
        bank tears down its Rust core and everything it owns.
        """
        if self._owns_bank and not self._bank.is_closed:
            self._bank.close()

    def __repr__(self) -> str:
        return (
            f"Corpus(num_files={len(self._dataset_ids)}, num_cells={self.num_cells}, "
            f"num_genes={self.num_genes}, owns_bank={self._owns_bank})"
        )

    # -- gene alignment ------------------------------------------------------

    def _resolve_gene_alignment(
        self,
        gene_alignment: _GeneAlignment,
        missing: "MissingGenePolicy | str | None",
    ) -> tuple[tuple[str, ...] | None, "MissingGenePolicy | None"]:
        """Pick the shared gene list and missing policy for ``gene_alignment``."""
        bank = self._bank
        per_file = [tuple(bank.dataset_genes(did)) for did in self._dataset_ids]
        explicit = _coerce_missing_policy(missing) if missing is not None else None

        if gene_alignment == "none":
            return None, explicit

        if gene_alignment == "strict":
            first = per_file[0]
            for i, genes in enumerate(per_file[1:], 1):
                if genes != first:
                    raise ValueError(
                        "gene_alignment='strict' requires identical gene names in the "
                        f"same order across all datasets; dataset 0 and {i} differ "
                        f"({len(first)} genes vs {len(genes)})"
                    )
            return first, explicit

        if gene_alignment == "union":
            seen: dict[str, None] = {}
            for genes in per_file:
                for name in genes:
                    if name not in seen:
                        seen[name] = None
            policy = explicit if explicit is not None else MissingGenePolicy.ZERO
            return tuple(seen), policy

        # gene_alignment == "intersection"
        if len(per_file) == 1:
            return per_file[0], explicit
        common = set(per_file[0])
        for genes in per_file[1:]:
            common &= set(genes)
        inter = tuple(name for name in per_file[0] if name in common)
        # Genes in the intersection exist in every dataset — no fill needed.
        return inter, explicit


def _normalize_bank_config(
    bank_config: "DataBankConfig | Mapping[str, Any] | None",
) -> DataBankConfig:
    """Normalize the ``bank_config`` argument into a :class:`DataBankConfig`.

    Mirrors :meth:`ScDataBank.__init__` so the stored config matches the one
    the Rust core was built from (for the :class:`BankConfigSummary`).
    """
    if bank_config is None:
        return DataBankConfig()
    if isinstance(bank_config, DataBankConfig):
        return bank_config
    if isinstance(bank_config, MappingABC):
        return DataBankConfig.from_dict(bank_config)
    raise TypeError(
        f"bank_config must be DataBankConfig, a mapping, or None; got {type(bank_config).__name__}"
    )


def _resolve_corpus_entries(
    paths: "Iterable[str | PathLike[str] | tuple[str | PathLike[str], str] | Mapping[str, Any] | DenseDataset | SparseDataset]",
    *,
    layer: str | None,
    matrix: str | None,
) -> tuple[_CorpusEntry, ...]:
    default_key = _resolve_matrix_key(layer=layer, matrix=matrix)
    return tuple(_coerce_corpus_item(item, default_key) for item in paths)


def _coerce_corpus_item(item: object, default_key: str) -> _CorpusEntry:
    if isinstance(item, (DenseDataset, SparseDataset)):
        return _CorpusEntry(item, default_key)
    if isinstance(item, MappingABC):
        return _coerce_mapping_item(item, default_key)
    if isinstance(item, tuple):
        return _coerce_tuple_item(item)
    return _coerce_path_item(item, default_key)


def _coerce_tuple_item(item: tuple[object, ...]) -> _CorpusEntry:
    if len(item) != 2:
        raise TypeError("tuple corpus entries must be (path, matrix_key)")
    path, matrix = item
    matrix_key = _resolve_matrix_key(layer=None, matrix=_coerce_selector(matrix, "matrix_key"))
    return _coerce_path_item(path, matrix_key)


def _coerce_mapping_item(item: MappingABC[str, object], default_key: str) -> _CorpusEntry:
    source_keys = [key for key in ("path", "dataset", "collection") if key in item]
    if len(source_keys) != 1:
        raise TypeError("mapping corpus entries must contain exactly one of: path, dataset, collection")
    matrix_key = _matrix_key_from_mapping(item, default_key)

    source_key = source_keys[0]
    if source_key == "path":
        return _coerce_path_item(item["path"], matrix_key)
    if source_key == "collection":
        collection = item["collection"]
        if not isinstance(collection, DatasetCollection):
            raise TypeError(
                f"collection must be DatasetCollection, got {type(collection).__name__}"
            )
        return _CorpusEntry(collection[matrix_key], matrix_key)

    dataset = item["dataset"]
    if not isinstance(dataset, (DenseDataset, SparseDataset)):
        raise TypeError(f"dataset must be DenseDataset or SparseDataset, got {type(dataset).__name__}")
    if "store_root" in item and item["store_root"] is not None:
        dataset = _with_store_root(dataset, item["store_root"])
    return _CorpusEntry(dataset, matrix_key)


def _coerce_path_item(path: object, matrix_key: str) -> _CorpusEntry:
    store_path = _coerce_str_path(path, "corpus path entries")
    return _CorpusEntry(launch(store_path, matrix=matrix_key), matrix_key)


def _with_store_root(
    dataset: DenseDataset | SparseDataset,
    store_root: object,
) -> DenseDataset | SparseDataset:
    root = _coerce_str_path(store_root, "store_root")
    return cast(DenseDataset | SparseDataset, replace(dataset, store_root=root))


def _coerce_str_path(value: object, context: str) -> str:
    if not isinstance(value, (str, PathLike)):
        raise TypeError(f"{context} must be str or PathLike, got {type(value).__name__}")
    path = os.fspath(value)
    if not isinstance(path, str):
        raise TypeError(f"{context} must resolve to a str path, got {type(path).__name__}")
    return path


def _matrix_key_from_mapping(item: MappingABC[str, object], default_key: str) -> str:
    layer = item.get("layer")
    selectors = [
        value
        for key, value in (
            ("matrix", item.get("matrix")),
            ("matrix_key", item.get("matrix_key")),
            ("X", item.get("X")),
        )
        if key in item and value is not None
    ]
    if layer is not None and selectors:
        raise ValueError("mapping corpus entries must pass either layer or a matrix key, not both")
    if len(selectors) > 1:
        raise ValueError("mapping corpus entries must pass only one matrix selector")
    if layer is not None:
        return _resolve_matrix_key(layer=_coerce_selector(layer, "layer"), matrix=None)
    if selectors:
        return _resolve_matrix_key(layer=None, matrix=_coerce_selector(selectors[0], "matrix_key"))
    return default_key


def _coerce_selector(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _resolve_matrix_key(*, layer: str | None, matrix: str | None) -> str:
    if layer is not None and matrix is not None:
        raise ValueError("pass either layer= or matrix=, not both")
    if layer is not None:
        return _layer_matrix_key(layer)
    if matrix is None:
        return "X"
    key = matrix.strip("/")
    if key == "X":
        return "X"
    if key in ("raw", "raw/X"):
        return "raw/X"
    if "/" not in key:
        return _layer_matrix_key(key)
    if key.startswith("layers/"):
        name = key[len("layers/") :]
        if name and "/" not in name:
            return key
    raise ValueError(
        f"unsupported matrix key {matrix!r}; expected 'X', 'layers/<name>', or 'raw/X'"
    )


def _layer_matrix_key(layer: str) -> str:
    name = str(layer)
    if not name or "/" in name:
        raise ValueError(f"layer names must be non-empty direct children, got {layer!r}")
    return f"layers/{name}"
