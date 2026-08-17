from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scdata.compress as scc
from scdata.compress import Codec
from scdata.compress._codec import _codec_from_wire, _codec_to_wire


def test_unified_codec_factories_and_parse() -> None:
    assert Codec.blosc() == Codec(
        algorithm="blosc",
        backend="lz4",
        level=5,
        shuffle="none",
        split_blocks=False,
    )
    assert Codec.zstd() == Codec(algorithm="zstd", level=3)
    assert Codec.parse("blosc") == Codec.blosc()
    assert Codec.parse({"algorithm": "zstd", "level": 1}) == Codec.zstd(1)


def test_codec_config_is_strict() -> None:
    with pytest.raises(ValueError, match="unknown codec option"):
        Codec.parse({"algorithm": "zstd", "typo_level": 1})
    with pytest.raises(TypeError, match="split_blocks must be bool"):
        Codec.parse({"algorithm": "blosc", "split_blocks": "false"})
    with pytest.raises(ValueError, match="unknown codec option"):
        Codec.parse({"id": "dyn-blosc"})
    with pytest.raises(ValueError, match="only valid for the blosc codec"):
        Codec(algorithm="zstd", shuffle="bytes")
    with pytest.raises(ValueError, match="not valid"):
        Codec(algorithm="none", level=1)
    with pytest.raises(ValueError, match="level"):
        Codec.blosc(level=10)


def test_codec_storage_dispatch_is_private_and_representation_aware() -> None:
    codec = Codec.blosc(backend="zstd", level=7, shuffle="bits", split_blocks=True)
    dense = _codec_to_wire(codec, role="dense")
    csr = _codec_to_wire(codec, role="csr")
    assert dense == {
        "id": "blosc1",
        "codec": "zstd",
        "clevel": 7,
        "shuffle": "bits",
        "split_blocks": True,
        "block_size": 1,
    }
    assert csr == {
        "id": "dyn-blosc",
        "codec": "zstd",
        "clevel": 7,
        "shuffle": "bits",
        "split_blocks": True,
    }
    assert _codec_from_wire(dense) == codec
    assert _codec_from_wire(csr) == codec


def test_write_options_accepts_unified_codec_spec() -> None:
    options = scc.WriteOptions(
        codec=Codec.parse(
            {
                "algorithm": "blosc",
                "backend": "lz4",
                "shuffle": "bytes",
            }
        ),
        indptr_codec=Codec.zstd(),
    )
    assert options.resolved_codec() == Codec.blosc(shuffle="bytes")
    assert options.resolved_indptr_codec() == Codec.zstd()
    with pytest.raises(TypeError, match="codec must be Codec"):
        scc.WriteOptions(codec="blosc")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="matrix data requires"):
        scc.WriteOptions(codec=Codec.zstd()).resolved_codec()


def test_dense_block_size_is_derived_only_from_partition(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    root = tmp_path / "dense"
    options = scc.WriteOptions(
        chunk_policy="cells",
        chunk_cells=3,
        block_policy="cells",
        block_cells=2,
        codec=Codec.blosc(backend="zstd", level=7, shuffle="bits"),
    )
    scc.write_dense(root, values, options=options)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert meta["data"]["compressor"]["id"] == "blosc1"
    assert meta["data"]["compressor"]["block_size"] == 2 * 4 * values.dtype.itemsize
    with scc.open_store(root) as store:
        assert store.codec == options.codec
        assert store.info().codec == options.codec
        assert store.info().indptr_codec is None
        np.testing.assert_array_equal(store.read(), values)


def test_csr_codec_and_indptr_codec_roundtrip(tmp_path: Path) -> None:
    sparse = pytest.importorskip("scipy.sparse")
    values = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    root = tmp_path / "csr"
    codec = Codec.blosc(backend="zlib", level=9, shuffle="bytes", split_blocks=True)
    indptr_codec = Codec.zstd(1)
    scc.write_csr(
        root,
        sparse.csr_matrix(values),
        codec=codec,
        indptr_codec=indptr_codec,
    )
    with scc.open_store(root) as store:
        assert store.codec == codec
        assert store.indptr_codec == indptr_codec
        assert store.info().codec == codec
        assert store.info().indptr_codec == indptr_codec
        np.testing.assert_array_equal(store.read().toarray(), values)


def test_irrelevant_and_unsupported_codec_options_are_rejected(tmp_path: Path) -> None:
    values = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="indptr_codec applies only to CSR"):
        scc.write(tmp_path / "dense-indptr", values, indptr_codec=Codec.none())
    with pytest.raises(ValueError, match="matrix data requires"):
        scc.write_dense(tmp_path / "dense-zstd", values, codec=Codec.zstd())


def test_zip_direct_codec_override_roundtrip(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    archive = tmp_path / "stores.zip"
    codec = Codec.blosc(backend="zstd", level=8, shuffle="bits", split_blocks=True)
    scc.zip.write_dense(archive, "dense", values, codec=codec)
    with scc.open_store(archive, zip_prefix="dense") as store:
        assert store.codec == codec
        np.testing.assert_array_equal(store.read(), values)
