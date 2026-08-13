from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

import scdata.compress as scc


def test_pack_list_and_open_zipfile(tmp_path: Path) -> None:
    store_dir = tmp_path / "dense"
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    scc.write_dense(store_dir, values)

    archive = tmp_path / "matrices.zip"
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_STORED) as zf:
        scc.zip.pack(zf, "assay", store_dir)

    assert scc.zip.list_stores(archive) == ["assay"]

    with zipfile.ZipFile(archive, mode="r") as zf:
        store = scc.open_store(zf)
        np.testing.assert_array_equal(store.read(), values)

    store = scc.open_store(archive)
    assert store.zip_prefix == "assay"
    np.testing.assert_array_equal(store.read_rows(0, 1), values[:1])


def test_write_dense_zip_deflated_and_root_prefix(tmp_path: Path) -> None:
    archive = tmp_path / "root.zip"
    values = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    with pytest.warns(scc.PerformanceWarning, match="range reads"):
        scc.zip.write_dense(
            archive,
            "",
            values,
            n_workers=2,
            compression=zipfile.ZIP_DEFLATED,
        )
    assert scc.zip.list_stores(archive) == [""]
    store = scc.open_store(archive, zip_prefix="", n_workers=2)
    assert store.n_workers == 2
    np.testing.assert_array_equal(store.read(), values)


def test_write_csr_zip_roundtrip(tmp_path: Path) -> None:
    sparse = pytest.importorskip("scipy.sparse")
    archive = tmp_path / "csr.zip"
    csr = sparse.csr_matrix(
        (
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int64),
        ),
        shape=(2, 2),
    )
    with zipfile.ZipFile(archive, mode="w") as zf:
        scc.zip.write_csr(zf, "x", csr)

    out = scc.open_store(archive, zip_prefix="x").read()
    np.testing.assert_array_equal(out.toarray(), csr.toarray())


def test_pack_rejects_duplicate_prefix(tmp_path: Path) -> None:
    store_dir = tmp_path / "dense"
    scc.write_dense(store_dir, np.zeros((2, 2), dtype=np.float32))
    archive = tmp_path / "dup.zip"
    scc.zip.pack(archive, "a", store_dir)
    with pytest.raises(scc.InvalidArgumentError):
        scc.zip.pack(archive, "a", store_dir)


def test_open_lists_prefixes_when_archive_is_ambiguous(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    scc.write_dense(first, np.ones((1, 1), dtype=np.uint16))
    scc.write_dense(second, np.zeros((1, 1), dtype=np.uint16))
    archive = tmp_path / "many.zip"
    scc.zip.pack(archive, "a", first)
    scc.zip.pack(archive, "b", second)

    with pytest.raises(scc.InvalidArgumentError, match="available: 'a', 'b'"):
        scc.open_store(archive)
    np.testing.assert_array_equal(scc.open_store(archive, zip_prefix="b").read(), [[0]])


def test_generic_write_zip_dispatches_dense_input(tmp_path: Path) -> None:
    archive = tmp_path / "generic.zip"
    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    scc.zip.write(archive, "matrix", values)
    np.testing.assert_array_equal(scc.open_store(archive).read(), values)


def test_pack_preflights_all_collisions_before_writing(tmp_path: Path) -> None:
    store_dir = tmp_path / "dense"
    scc.write_dense(store_dir, np.arange(4, dtype=np.float32).reshape(2, 2))
    relative_files = sorted(
        path.relative_to(store_dir).as_posix() for path in store_dir.rglob("*") if path.is_file()
    )
    collision = next(name for name in reversed(relative_files) if name != "meta.json")
    archive = tmp_path / "collision.zip"
    with zipfile.ZipFile(archive, mode="w") as zf:
        zf.writestr(f"x/{collision}", b"already here")
    with zipfile.ZipFile(archive) as zf:
        before = zf.namelist()

    with pytest.raises(scc.InvalidArgumentError, match="target member"):
        scc.zip.pack(archive, "x", store_dir)

    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == before


def test_list_stores_ignores_unsafe_member_prefixes(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, mode="w") as zf:
        zf.writestr("bad\\prefix/meta.json", "{}")
        zf.writestr("a/./meta.json", "{}")

    assert scc.zip.list_stores(archive) == []


def test_pack_rejects_existing_descendant_before_writing(tmp_path: Path) -> None:
    store_dir = tmp_path / "dense"
    scc.write_dense(store_dir, np.ones((1, 1), dtype=np.float32))
    archive = tmp_path / "descendant.zip"
    with zipfile.ZipFile(archive, mode="w") as zf:
        zf.writestr("x/meta.json/child", b"collision")
    with zipfile.ZipFile(archive) as zf:
        before = zf.namelist()

    with pytest.raises(scc.InvalidArgumentError, match="target member"):
        scc.zip.pack(archive, "x", store_dir)

    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == before


@pytest.mark.parametrize(
    ("compression", "compresslevel", "message"),
    [(999, None, "unsupported ZIP compression"), (zipfile.ZIP_DEFLATED, 100, "between")],
)
def test_pack_validates_compression_before_creating_archive(
    tmp_path: Path,
    compression: int,
    compresslevel: int | None,
    message: str,
) -> None:
    store_dir = tmp_path / "dense"
    scc.write_dense(store_dir, np.ones((1, 1), dtype=np.float32))
    archive = tmp_path / "invalid.zip"

    with pytest.raises(scc.InvalidArgumentError, match=message):
        scc.zip.pack(
            archive,
            "x",
            store_dir,
            compression=compression,
            compresslevel=compresslevel,
        )

    assert not archive.exists()


def test_zip_writer_validates_archive_options_before_materializing_matrix(tmp_path: Path) -> None:
    class UnreadableMatrix:
        def __array__(self) -> np.ndarray:
            raise AssertionError("matrix must not be materialized")

    archive = tmp_path / "invalid-writer.zip"
    with pytest.raises(scc.InvalidArgumentError, match="unsupported ZIP compression"):
        scc.zip.write_dense(archive, "x", UnreadableMatrix(), compression=999)
    assert not archive.exists()

    archive.write_text("not a zip", encoding="utf-8")
    with pytest.raises(scc.InvalidArgumentError, match="not a ZIP file"):
        scc.zip.write_dense(archive, "x", UnreadableMatrix())
    assert archive.read_text(encoding="utf-8") == "not a zip"


def test_open_store_rejects_zipfile_still_open_for_append(tmp_path: Path) -> None:
    store_dir = tmp_path / "dense"
    scc.write_dense(store_dir, np.ones((1, 1), dtype=np.float32))
    archive = tmp_path / "append.zip"
    scc.zip.pack(archive, "x", store_dir)

    with zipfile.ZipFile(archive, mode="a") as zf:
        with pytest.raises(scc.InvalidArgumentError, match="mode 'r'"):
            scc.open_store(zf)


def test_pack_rejects_non_regular_store_files(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO files are unavailable on this platform")
    store_dir = tmp_path / "dense"
    scc.write_dense(store_dir, np.ones((1, 1), dtype=np.float32))
    os.mkfifo(store_dir / "pipe")

    with pytest.raises(scc.InvalidArgumentError, match="non-regular file"):
        scc.zip.pack(tmp_path / "fifo.zip", "x", store_dir)


def test_pack_rejects_oversized_member_names_before_creating_archive(tmp_path: Path) -> None:
    store_dir = tmp_path / "dense"
    scc.write_dense(store_dir, np.ones((1, 1), dtype=np.float32))
    archive = tmp_path / "long-name.zip"

    with pytest.raises(scc.InvalidArgumentError, match="maximum is 65535"):
        scc.zip.pack(archive, "x" * 65_536, store_dir)
    assert not archive.exists()
