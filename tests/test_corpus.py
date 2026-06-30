"""End-to-end tests for the high-level :class:`~scdata.corpus.Corpus` entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
pytest.importorskip("zarr")

from scdata import Corpus, MissingGenePolicy, ScDataBank  # noqa: E402
from scdata.io import write_zarr  # noqa: E402


def _write_store(
    tmp_path: Path,
    name: str,
    shape: tuple[int, int],
    gene_names: list[str],
    np_dtype: Any = np.float32,
) -> Path:
    """Write a small dense2d ``.zarr.zip`` store with the given gene names."""
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    x = rng.random(shape, dtype=np_dtype).astype(np_dtype)
    adata = ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(shape[0])]),
        var=pd.DataFrame(index=gene_names),
    )
    return write_zarr(
        adata,
        tmp_path / f"{name}.zarr.zip",
        format="dense2d",
        chunk_size=(2, min(3, shape[1])),
        store="zip",
    )


def _two_identical_gene_stores(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    genes = ["g0", "g1", "g2", "g3"]
    return _write_store(tmp_path, "a", (5, 4), genes), _write_store(tmp_path, "b", (3, 4), genes), genes


def test_corpus_strict_alignment_success(tmp_path: Path) -> None:
    p0, p1, genes = _two_identical_gene_stores(tmp_path)
    with Corpus([p0, p1]) as corpus:
        assert corpus.num_files == 2
        assert corpus.num_cells == 8  # 5 + 3
        assert corpus.num_genes == 4
        assert corpus.gene_names == tuple(genes)
        # strict alignment -> no missing-gene policy needed
        assert corpus.missing is None
        assert corpus.owns_bank is True


def test_corpus_strict_alignment_rejects_mismatched_order(tmp_path: Path) -> None:
    p0 = _write_store(tmp_path, "a", (3, 2), ["g0", "g1"])
    p1 = _write_store(tmp_path, "b", (3, 2), ["g1", "g0"])  # same set, wrong order
    with pytest.raises(ValueError, match="strict"):
        Corpus([p0, p1])


def test_corpus_union_alignment(tmp_path: Path) -> None:
    p0 = _write_store(tmp_path, "a", (3, 2), ["g0", "g1"])
    p1 = _write_store(tmp_path, "b", (3, 2), ["g1", "g2"])
    with Corpus([p0, p1], gene_alignment="union") as corpus:
        assert corpus.gene_names == ("g0", "g1", "g2")
        assert corpus.missing is MissingGenePolicy.ZERO
        assert corpus.num_genes == 3


def test_corpus_union_alignment_explicit_missing_override(tmp_path: Path) -> None:
    p0 = _write_store(tmp_path, "a", (3, 2), ["g0", "g1"])
    p1 = _write_store(tmp_path, "b", (3, 2), ["g1", "g2"])
    with Corpus([p0, p1], gene_alignment="union", missing="error") as corpus:
        assert corpus.gene_names == ("g0", "g1", "g2")
        assert corpus.missing is MissingGenePolicy.ERROR


def test_corpus_intersection_alignment(tmp_path: Path) -> None:
    p0 = _write_store(tmp_path, "a", (3, 3), ["g0", "g1", "g2"])
    p1 = _write_store(tmp_path, "b", (3, 3), ["g1", "g2", "g3"])
    with Corpus([p0, p1], gene_alignment="intersection") as corpus:
        assert corpus.gene_names == ("g1", "g2")
        assert corpus.missing is None  # intersection genes exist everywhere
        assert corpus.num_genes == 2


def test_corpus_empty_intersection_yields_zero_gene_batches(tmp_path: Path) -> None:
    p0 = _write_store(tmp_path, "a", (2, 1), ["g0"])
    p1 = _write_store(tmp_path, "b", (2, 1), ["g1"])
    with Corpus([p0, p1], gene_alignment="intersection") as corpus:
        assert corpus.gene_names == ()
        assert corpus.num_genes == 0
        batch = next(iter(corpus.loader(batch_size=3, shuffle=False)))
        assert batch["x"].shape == (3, 0)
        assert batch["gene_names"] == ()


def test_corpus_none_alignment(tmp_path: Path) -> None:
    p0, p1, _ = _two_identical_gene_stores(tmp_path)
    with Corpus([p0, p1], gene_alignment="none") as corpus:
        assert corpus.gene_names is None
        assert corpus.missing is None


def test_corpus_rejects_empty_paths() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Corpus([])


def test_corpus_rejects_bad_alignment_mode(tmp_path: Path) -> None:
    p0 = _write_store(tmp_path, "a", (3, 2), ["g0", "g1"])
    with pytest.raises(ValueError, match="gene_alignment"):
        Corpus([p0], gene_alignment="bogus")  # type: ignore[arg-type]


def test_corpus_loader_iterates_dense_batches(tmp_path: Path) -> None:
    p0, p1, genes = _two_identical_gene_stores(tmp_path)
    with Corpus([p0, p1]) as corpus:
        loader = corpus.loader(batch_size=3, shuffle=False, drop_last=False)
        batches = list(loader)
        # 8 cells / batch_size 3 -> ceil = 3 batches (3, 3, 2)
        assert len(batches) == 3
        shapes = [b["x"].shape[0] for b in batches]
        assert shapes == [3, 3, 2]
        assert all(b["x"].shape[1] == 4 for b in batches)
        assert all(b["gene_names"] == tuple(genes) for b in batches)
        # file_ids cover both files across the epoch
        all_files = np.concatenate([b["file_ids"].numpy() for b in batches])
        assert set(all_files.tolist()) == {0, 1}
        assert len(all_files) == 8


def test_corpus_loader_default_collate_is_stitch(tmp_path: Path) -> None:
    import torch

    p0, _p1, _genes = _two_identical_gene_stores(tmp_path)
    with Corpus([p0]) as corpus:
        loader = corpus.loader(batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert isinstance(batch["x"], torch.Tensor)
        assert batch["x"].shape == (2, 4)


def test_corpus_close_releases_owned_bank(tmp_path: Path) -> None:
    p0, _p1, _genes = _two_identical_gene_stores(tmp_path)
    corpus = Corpus([p0])
    assert corpus.owns_bank is True
    assert corpus.bank.is_closed is False
    corpus.close()
    assert corpus.bank.is_closed is True
    # close is idempotent
    corpus.close()


def test_corpus_context_manager_closes_on_exit(tmp_path: Path) -> None:
    p0, _p1, _genes = _two_identical_gene_stores(tmp_path)
    with Corpus([p0]) as corpus:
        assert corpus.bank.is_closed is False
    assert corpus.bank.is_closed is True


def test_corpus_from_bank_does_not_own_bank(tmp_path: Path) -> None:
    p0, p1, _genes = _two_identical_gene_stores(tmp_path)
    bank = ScDataBank()
    try:
        corpus = Corpus.from_bank(bank, [p0, p1])
        assert corpus.owns_bank is False
        assert corpus.bank is bank
        assert bank.is_closed is False
        corpus.close()
        # external bank must NOT be closed by Corpus
        assert bank.is_closed is False
    finally:
        bank.close()


def test_corpus_bank_config_ignored_with_warning(tmp_path: Path) -> None:
    p0, _p1, _genes = _two_identical_gene_stores(tmp_path)
    bank = ScDataBank()
    try:
        with pytest.warns(UserWarning, match="bank_config is ignored"):
            Corpus([p0], bank=bank, bank_config={"decode__num_workers": 1})
    finally:
        bank.close()


def test_corpus_bank_config_summary_present_when_owned(tmp_path: Path) -> None:
    p0, _p1, _genes = _two_identical_gene_stores(tmp_path)
    with Corpus([p0]) as corpus:
        summary = corpus.bank_config_summary
        assert summary is not None
        assert summary.registered_datasets == 1
        assert summary.io_backend in ("uring", "threaded")
        assert summary.decode_workers > 0


def test_corpus_bank_config_summary_present_when_external_bank(tmp_path: Path) -> None:
    p0, _p1, _genes = _two_identical_gene_stores(tmp_path)
    bank = ScDataBank()
    try:
        corpus = Corpus.from_bank(bank, [p0])
        summary = corpus.bank_config_summary
        assert summary is not None
        assert summary.registered_datasets == 1
        assert summary.io_backend in ("uring", "threaded")
    finally:
        bank.close()


def test_corpus_loader_stats_collection(tmp_path: Path) -> None:
    p0, _p1, _genes = _two_identical_gene_stores(tmp_path)
    with Corpus([p0]) as corpus:
        loader = corpus.loader(batch_size=2, shuffle=False)
        assert loader.sc_collect_stats is True
        for _ in loader:
            pass
        stats = loader.stats(reset=False)
        assert stats.batches_seen == 3  # ceil(5 / 2)
        assert stats.cells_seen == 5
        assert stats.wait_p99_ms >= 0.0
        assert stats.throughput_cells_per_s >= 0.0


def test_sc_dataloader_from_paths_classmethod(tmp_path: Path) -> None:
    p0, p1, _genes = _two_identical_gene_stores(tmp_path)
    from scdata.data import ScDataLoader

    bank = ScDataBank()
    try:
        loader = ScDataLoader.from_paths(bank, [p0, p1], batch_size=4, shuffle=False)
        batches = list(loader)
        assert len(batches) == 2  # 8 cells / 4
        assert all(b["x"].shape == (4, 4) for b in batches)
        assert bank.is_closed is False
    finally:
        bank.close()


def test_corpus_init_failure_releases_owned_bank(tmp_path: Path) -> None:
    p0 = _write_store(tmp_path, "a", (3, 2), ["g0", "g1"])
    bad = tmp_path / "missing.zarr.zip"
    with pytest.raises(Exception):
        Corpus([p0, bad])
    # Resources were freed (the owned bank was torn down in __init__'s except),
    # so a fresh Corpus over the good store still works.
    with Corpus([p0]) as corpus:
        assert corpus.num_cells == 3


def test_corpus_init_failure_keeps_external_bank_open(tmp_path: Path) -> None:
    p0 = _write_store(tmp_path, "a", (3, 2), ["g0", "g1"])
    bad = tmp_path / "missing.zarr.zip"
    bank = ScDataBank()
    try:
        with pytest.raises(Exception):
            Corpus.from_bank(bank, [p0, bad])
        # owns_bank=False -> the external bank must NOT be closed on failure.
        assert bank.is_closed is False
    finally:
        bank.close()


def test_corpus_init_failure_rolls_back_external_bank_registration(tmp_path: Path) -> None:
    """register() succeeds but strict gene-alignment fails: external bank must
    not keep the dangling datasets (otherwise it leaks handles and pollutes
    state for the caller who still holds the bank)."""
    p0 = _write_store(tmp_path, "a", (3, 2), ["g0", "g1"])
    p1 = _write_store(tmp_path, "b", (3, 2), ["g1", "g0"])  # same genes, swapped order
    bank = ScDataBank()
    try:
        with pytest.raises(ValueError, match="strict"):
            Corpus.from_bank(bank, [p0, p1])
        assert bank.is_closed is False
        # Both datasets were registered before alignment failed; they must be
        # rolled back so the external bank is left clean.
        assert bank._registered_count == 0
        # The bank is still usable for a fresh registration.
        with Corpus.from_bank(bank, [p0]) as corpus:
            assert corpus.num_cells == 3
            assert bank._registered_count == 1
    finally:
        bank.close()
