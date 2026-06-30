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
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal

from scdata.data._collate import stitch_dense_collate
from scdata.data._dataloader import ScDataBatch, ScDataLoader
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


class Corpus:
    """A registered multi-dataset corpus ready to build training loaders.

    Args:
        paths: Store paths (``.zarr.zip`` archive or ``.zarr`` directory) in
            ``file_id`` order — the order sampler indices map to.
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
        "_gene_names",
        "_missing",
        "_cells_per_file",
    )

    def __init__(
        self,
        paths: "Iterable[str | PathLike[str]]",
        *,
        bank: ScDataBank | None = None,
        bank_config: "DataBankConfig | Mapping[str, Any] | None" = None,
        layer: str | None = None,
        matrix: str | None = None,
        gene_alignment: _GeneAlignment = "strict",
        missing: "MissingGenePolicy | str | None" = None,
    ) -> None:
        path_list = [os.fspath(p) for p in paths]
        if not path_list:
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
            datasets = [launch(p, layer=layer, matrix=matrix) for p in path_list]
            registered = bank.register(datasets)
            # register(Iterable[Dataset]) -> list[DatasetId], in input order.
            self._dataset_ids: tuple[DatasetId, ...] = tuple(registered)
            self._cells_per_file: tuple[int, ...] = tuple(
                bank.dataset_num_cells(did) for did in self._dataset_ids
            )
            self._gene_names, self._missing = self._resolve_gene_alignment(
                gene_alignment, missing
            )
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
        """Structural summary of the bank config, for stats/tune tooling.

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
        paths: "Iterable[str | PathLike[str]]",
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
        "bank_config must be DataBankConfig, a mapping, or None; "
        f"got {type(bank_config).__name__}"
    )
