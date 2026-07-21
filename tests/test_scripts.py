"""Lightweight unit tests for the standalone operational scripts.

These tests deliberately use tiny temporary archives and fakes.  They do not
open any FFPE/cellxgene data or run a benchmark workload.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def load_script(path: Path):
    """Load a standalone script without requiring ``scripts`` to be a package."""
    module_name = f"_test_script_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts():
    modules = {
        path.stem: load_script(path)
        for path in (
            SCRIPTS_DIR / "audit_gene_alignment.py",
            SCRIPTS_DIR / "group_by_gene_hash.py",
            SCRIPTS_DIR / "merge_group.py",
            SCRIPTS_DIR / "recompress_zarrzip_blocksize.py",
            SCRIPTS_DIR / "recompress_zarrzip_batch.py",
            SCRIPTS_DIR / "convert_10x_dirs_to_zarrzip.py",
            SCRIPTS_DIR / "bench_access.py",
            EXAMPLES_DIR / "batch_sequential_access.py",
            EXAMPLES_DIR / "batch_random_access.py",
        )
    }
    # The batch script imports this name lazily when it executes a task.
    sys.modules["recompress_zarrzip_blocksize"] = modules["recompress_zarrzip_blocksize"]
    return modules


def v3_blosc_meta(blocksize: int, *, codec_names: list[str] | None = None) -> dict[str, object]:
    names = codec_names or ["bytes", "blosc"]
    codecs: list[dict[str, object]] = []
    for name in names:
        if name == "bytes":
            codecs.append({"name": "bytes", "configuration": {"endian": "little"}})
        elif name == "blosc":
            codecs.append(
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "lz4",
                        "clevel": 5,
                        "shuffle": "shuffle",
                        "typesize": 1,
                        "blocksize": blocksize,
                    },
                }
            )
        else:
            codecs.append({"name": name, "configuration": {}})
    return {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": "uint8",
        "shape": [1],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": codecs,
        "attributes": {},
    }


def write_zip(
    path: Path,
    entries: dict[str, bytes | dict[str, object]],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        for key, value in entries.items():
            payload = json.dumps(value).encode() if isinstance(value, dict) else value
            zf.writestr(key, payload)


def test_audit_reads_complete_categorical_index_and_dry_run(scripts, monkeypatch, tmp_path, capsys):
    audit = scripts["audit_gene_alignment"]
    names = pd.CategoricalIndex(["gene_a", "gene_b", "gene_c"])
    io_module = types.ModuleType("scdata.io")
    io_module.read_var_names = lambda path, raw=False: names
    monkeypatch.setitem(sys.modules, "scdata.io", io_module)

    info = audit.read_gene_info(tmp_path / "sample.zarr.zip")
    assert info.n_genes == 3
    assert info.first3 == ("gene_a", "gene_b", "gene_c")

    store = tmp_path / "sample.zarr.zip"
    store.touch()
    output = tmp_path / "would-not-be-written.tsv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "--root", str(tmp_path), "--dry-run", "--output-tsv", str(output)],
    )
    assert audit.main() == 0
    assert not output.exists()
    assert "no stores were opened" in capsys.readouterr().out


def test_merge_guards_unsupported_slots_label_conflicts_and_var(scripts):
    merge = scripts["merge_group"]
    empty = types.SimpleNamespace(
        layers={}, obsm={}, varm={}, obsp={}, varp={}, raw=None, uns={"scdata_source": {}}
    )
    merge.validate_mergeable_adata(empty, Path("one.zarr.zip"))
    empty.layers["counts"] = object()
    with pytest.raises(ValueError, match="layers"):
        merge.validate_mergeable_adata(empty, Path("one.zarr.zip"))

    obs_source = types.SimpleNamespace(obs=pd.DataFrame({"value": [1]}, index=["AAAC"]))
    labels = merge.FileLabels("sample", "", "raw", "FHR", "donor")
    obs = merge.add_labels_and_validate_obs(obs_source, labels, Path("one.zarr.zip"))
    assert list(obs.columns) == ["value", *merge.LABEL_COLUMNS]
    assert obs.index.tolist() == ["sample_AAAC"]

    obs_source.obs["sample_id"] = "already-present"
    with pytest.raises(ValueError, match="sample_id"):
        merge.add_labels_and_validate_obs(obs_source, labels, Path("one.zarr.zip"))
    assert merge.strict_var_equal(pd.DataFrame(index=["a"]), pd.DataFrame(index=["a"]))
    assert not merge.strict_var_equal(pd.DataFrame(index=["a"]), pd.DataFrame(index=["b"]))


def test_merge_memory_estimate_and_budget_override(scripts, monkeypatch):
    merge = scripts["merge_group"]
    infos = [
        merge.SourceMatrixInfo(Path("a.zarr.zip"), 10, 100, 1000, "u16"),
        merge.SourceMatrixInfo(Path("b.zarr.zip"), 20, 100, 2000, "u32"),
    ]

    peak, final_csr, largest_source = merge._estimate_merge_peak_bytes(infos)

    assert peak > final_csr > largest_source > 0
    assert merge._memory_budget_bytes(2.5) == int(2.5 * 1024**3)
    monkeypatch.setattr(merge, "_detected_memory_limit_bytes", lambda: 10 * 1024**3)
    assert merge._memory_budget_bytes(0) == 8 * 1024**3


def test_merge_limit_zero_is_noop(scripts, tmp_path, monkeypatch, capsys):
    merge = scripts["merge_group"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge",
            "--group",
            "1",
            "--groups-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "output"),
            "--limit",
            "0",
        ],
    )
    assert merge.main() == 0
    assert not (tmp_path / "output").exists()
    assert "no-op" in capsys.readouterr().out


def test_recompress_batches_materialize_missing_destination_with_verify(scripts, monkeypatch, tmp_path):
    block = scripts["recompress_zarrzip_blocksize"]
    batch = scripts["recompress_zarrzip_batch"]
    source = tmp_path / "source.zarr.zip"
    write_zip(
        source,
        {
            "X/data/zarr.json": v3_blosc_meta(65536),
            "layers/counts/zarr.json": v3_blosc_meta(0),
        },
    )
    assert block.blosc_array_blocksizes(source) == {
        "X/data/zarr.json": 65536,
        "layers/counts/zarr.json": 0,
    }

    calls: list[dict[str, object]] = []

    def fake_convert(src, dst, **kwargs):
        calls.append(kwargs)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"published")
        return types.SimpleNamespace(
            blosc_arrays=1,
            blosc_chunks=0,
            copied_entries=2,
            recompressed_bytes_in=0,
            recompressed_bytes_out=0,
        )

    monkeypatch.setattr(block, "convert_store", fake_convert)
    destination = tmp_path / "nested" / "target.zarr.zip"
    result = batch.recompress_one(
        batch.BatchTask(str(source), str(destination)), blocksize=65536, overwrite=False
    )
    assert result.status == "converted"
    assert destination.read_bytes() == b"published"
    assert calls == [{"target_blocksize": 65536, "overwrite": False, "verify": True}]


def test_recompress_batch_uses_atomic_verify_path_for_in_place_work(scripts, monkeypatch, tmp_path):
    block = scripts["recompress_zarrzip_blocksize"]
    batch = scripts["recompress_zarrzip_batch"]
    source = tmp_path / "in-place.zarr.zip"
    write_zip(source, {"X/data/zarr.json": v3_blosc_meta(0)})
    calls: list[dict[str, object]] = []

    def fake_convert(src, dst, **kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            blosc_arrays=1,
            blosc_chunks=0,
            copied_entries=1,
            recompressed_bytes_in=0,
            recompressed_bytes_out=0,
        )

    monkeypatch.setattr(block, "convert_store", fake_convert)
    result = batch.recompress_one(
        batch.BatchTask(str(source), str(source)), blocksize=65536, overwrite=False
    )
    assert result.status == "converted"
    assert calls == [{"target_blocksize": 65536, "overwrite": True, "verify": True}]


def test_recompress_batch_skips_only_complete_destination_at_target(scripts, monkeypatch, tmp_path):
    block = scripts["recompress_zarrzip_blocksize"]
    batch = scripts["recompress_zarrzip_batch"]
    source = tmp_path / "source.zarr.zip"
    destination = tmp_path / "destination.zarr.zip"
    entries = ("X/data/zarr.json", "layers/counts/zarr.json")
    write_zip(source, {key: v3_blosc_meta(0) for key in entries})
    write_zip(destination, {key: v3_blosc_meta(65536) for key in entries})
    monkeypatch.setattr(
        block,
        "convert_store",
        lambda *args, **kwargs: pytest.fail("complete destination must be skipped"),
    )

    result = batch.recompress_one(
        batch.BatchTask(str(source), str(destination)), blocksize=65536, overwrite=False
    )
    assert result.status == "skipped"
    assert "destination Blosc arrays" in result.message


def test_recompress_batch_does_not_skip_deflated_destination(scripts, monkeypatch, tmp_path):
    block = scripts["recompress_zarrzip_blocksize"]
    batch = scripts["recompress_zarrzip_batch"]
    source = tmp_path / "source.zarr.zip"
    destination = tmp_path / "destination.zarr.zip"
    entries = ("X/data/zarr.json", "layers/counts/zarr.json")
    write_zip(source, {key: v3_blosc_meta(0) for key in entries})
    write_zip(
        destination,
        {key: v3_blosc_meta(65536) for key in entries},
        compression=zipfile.ZIP_DEFLATED,
    )
    monkeypatch.setattr(
        block,
        "convert_store",
        lambda *args, **kwargs: pytest.fail("deflated destination must not be skipped"),
    )

    result = batch.recompress_one(
        batch.BatchTask(str(source), str(destination)), blocksize=65536, overwrite=False
    )
    assert result.status == "failed"
    assert str(destination) in result.message
    assert "X/data/zarr.json" in result.message
    assert "ZIP_STORED" in result.message


def test_recompress_batch_does_not_skip_incomplete_or_corrupt_destination(scripts, tmp_path):
    batch = scripts["recompress_zarrzip_batch"]
    source = tmp_path / "source.zarr.zip"
    entries = ("X/data/zarr.json", "layers/counts/zarr.json")
    write_zip(source, {key: v3_blosc_meta(0) for key in entries})

    incomplete = tmp_path / "incomplete.zarr.zip"
    write_zip(incomplete, {"X/data/zarr.json": v3_blosc_meta(65536)})
    incomplete_result = batch.recompress_one(
        batch.BatchTask(str(source), str(incomplete)), blocksize=65536, overwrite=False
    )
    assert incomplete_result.status == "failed"
    assert "incomplete" in incomplete_result.message

    corrupt = tmp_path / "corrupt.zarr.zip"
    corrupt.write_bytes(b"not a ZIP archive")
    corrupt_result = batch.recompress_one(
        batch.BatchTask(str(source), str(corrupt)), blocksize=65536, overwrite=False
    )
    assert corrupt_result.status == "failed"
    assert "BadZipFile" in corrupt_result.message


def test_recompress_rejects_unsupported_blosc_pipeline_with_array_path(scripts, tmp_path):
    block = scripts["recompress_zarrzip_blocksize"]
    source = tmp_path / "unsupported.zarr.zip"
    write_zip(
        source,
        {"X/data/zarr.json": v3_blosc_meta(0, codec_names=["bytes", "blosc", "zstd"])},
    )

    with pytest.raises(ValueError, match=r"X/data/zarr\.json: unsupported Blosc codec pipeline"):
        block.blosc_array_blocksizes(source)


def test_recompress_verify_checks_each_changed_chunk(scripts, monkeypatch, tmp_path):
    numcodecs = pytest.importorskip("numcodecs")
    block = scripts["recompress_zarrzip_blocksize"]
    source = tmp_path / "source.zarr.zip"
    target = tmp_path / "target.zarr.zip"
    params = block.BloscParams(cname="lz4", clevel=5, shuffle=1, typesize=1)
    source_chunks = {
        "X/data/c/0": bytes(params.encoder(0).encode(b"first chunk")),
        "X/data/c/1": bytes(params.encoder(0).encode(b"second chunk")),
    }
    write_zip(source, {"X/data/zarr.json": v3_blosc_meta(0), **source_chunks})
    block.convert_store(
        source,
        target,
        target_blocksize=65536,
        overwrite=False,
        verify=False,
    )

    with zipfile.ZipFile(target) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    entries["X/data/c/1"] = bytes(params.encoder(65536).encode(b"different bytes"))
    write_zip(target, entries)
    fake_io = types.ModuleType("scdata.io")
    fake_io.launch = lambda path: types.SimpleNamespace(num_cells=1, num_genes=1)
    monkeypatch.setitem(sys.modules, "scdata.io", fake_io)

    with pytest.raises(RuntimeError, match=r"X/data/c/1.*decode mismatch"):
        block._verify_store(target, source, {"X/data/": params}, 65536)
    assert numcodecs is not None


def test_10x_verifies_staged_zip_before_replacing_existing_target(scripts, monkeypatch, tmp_path):
    converter = scripts["convert_10x_dirs_to_zarrzip"]
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    target = tmp_path / "target.zarr.zip"
    target.write_bytes(b"known-good-target")
    task = converter.ConvertTask(source_dir=str(source_dir), target_zip=str(target))
    adata = types.SimpleNamespace(
        n_obs=2,
        n_vars=3,
        X=types.SimpleNamespace(nnz=4),
    )
    monkeypatch.setattr(converter, "read_10x_directory", lambda *args, **kwargs: adata)

    def fake_zip_directory(source, staged):
        staged.write_bytes(b"unverified-new-target")

    monkeypatch.setattr(converter, "zip_directory_stored", fake_zip_directory)
    fake_io = types.ModuleType("scdata.io")

    def fake_write_zarr(adata, path, **kwargs):
        path.mkdir(parents=True, exist_ok=True)

    staged_paths: list[Path] = []

    def fail_launch(path):
        staged_paths.append(path)
        raise RuntimeError("staged archive rejected")

    fake_io.write_zarr = fake_write_zarr
    fake_io.launch = fail_launch
    monkeypatch.setitem(sys.modules, "scdata.io", fake_io)

    result = converter.convert_one_task(
        task,
        chunk_size=1,
        compressor="blosc.lz4.level5",
        data_dtype="auto",
        var_names="symbol",
        make_var_names_unique=True,
        sample_metadata="none",
        obs_sample_id=False,
        overwrite=True,
        verify=True,
        keep_zarr=False,
        keep_failed_zarr=False,
    )
    assert result.status == "failed"
    assert staged_paths == [target.parent / f".{target.name}.tmp"]
    assert target.read_bytes() == b"known-good-target"


def test_10x_build_tasks_rejects_duplicate_manifest_entries(scripts, tmp_path):
    converter = scripts["convert_10x_dirs_to_zarrzip"]
    source = tmp_path / "sample" / "raw_feature_bc_matrix"
    source.mkdir(parents=True)

    with pytest.raises(ValueError, match="duplicate source directory"):
        converter.build_tasks(
            [source, source],
            input_root=tmp_path,
            output_root=tmp_path / "output",
            drop_matrix_dir=True,
        )


def test_10x_legacy_genes_tsv_fallback(scripts, tmp_path):
    converter = scripts["convert_10x_dirs_to_zarrzip"]
    legacy = tmp_path / "genes.tsv.gz"
    legacy.touch()
    assert converter.require_existing_any(tmp_path, "features.tsv", "genes.tsv") == legacy


def benchmark_sample(bench, *, batch_latencies: list[float], peak_rss_kib: int | None = 2048):
    return bench._sample(
        mode="unscheduled",
        cells=16,
        batches=len(batch_latencies),
        parts=1,
        bytes_read=128,
        checksum=7,
        seconds=2.0,
        warmup_batches=3,
        output_dtype="u16",
        resolved_strategy=None,
        fallback_reason=None,
        batch_latencies=batch_latencies,
        peak_rss_kib=peak_rss_kib,
    )


def test_benchmark_machine_info_reports_cpu_visibility(scripts, monkeypatch):
    bench = scripts["bench_access"]
    monkeypatch.setattr(bench.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(bench.os, "sched_getaffinity", lambda pid: {1, 3, 5, 7})

    info = bench.machine_info()

    assert info["cpu_count"] == 64
    assert info["affinity_cpu_count"] == 4


def test_benchmark_stored_dtype_is_common_across_access_paths(scripts):
    bench = scripts["bench_access"]
    args = types.SimpleNamespace(dtype="stored")
    u16 = types.SimpleNamespace(dtype="u16")
    u32 = types.SimpleNamespace(dtype="u32")

    assert bench.resolve_dtype(args, [u16]) == "u16"
    assert bench.resolve_dtype(args, [u16, u32]) == "u32"
    assert bench.resolve_dtype(types.SimpleNamespace(dtype="float32"), [u16, u32]) == "f32"


def test_benchmark_percentile_and_empty_latency_contract(scripts):
    bench = scripts["bench_access"]
    assert bench.percentile([], 50) is None
    assert bench.percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert bench.percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        bench.percentile([1.0], 101)

    assert bench.latency_metrics([]) == {
        "first_measured_batch_seconds": None,
        "batch_latency_p50_seconds": None,
        "batch_latency_p95_seconds": None,
    }


def test_benchmark_sample_records_latency_and_peak_rss_fields(scripts):
    sample = benchmark_sample(
        scripts["bench_access"], batch_latencies=[0.1, 0.2, 0.4], peak_rss_kib=3072
    )

    assert sample["first_measured_batch_seconds"] == pytest.approx(0.1)
    assert sample["batch_latency_p50_seconds"] == pytest.approx(0.2)
    assert sample["batch_latency_p95_seconds"] == pytest.approx(0.38)
    assert sample["process_peak_rss_kib"] == 3072


def test_benchmark_aggregate_handles_nullable_metrics(scripts):
    bench = scripts["bench_access"]
    measured = benchmark_sample(bench, batch_latencies=[1.0, 2.0, 3.0, 4.0], peak_rss_kib=1024)
    empty = benchmark_sample(bench, batch_latencies=[], peak_rss_kib=2048)
    summary = bench.aggregate_summary([{"results": [measured]}, {"results": [empty]}])[
        "unscheduled"
    ]

    assert summary["runs"] == 2
    assert summary["first_measured_batch_seconds"] == {
        "mean": 1.0,
        "stdev": 0.0,
        "min": 1.0,
        "max": 1.0,
    }
    assert summary["batch_latency_p50_seconds"]["mean"] == pytest.approx(2.5)
    assert summary["batch_latency_p95_seconds"]["mean"] == pytest.approx(3.85)
    assert summary["process_peak_rss_kib"] == {
        "mean": 1536.0,
        "stdev": pytest.approx(724.0773439350247),
        "min": 1024.0,
        "max": 2048.0,
    }

    empty_only = bench.aggregate_summary([{"results": [empty]}])["unscheduled"]
    for metric in (
        "first_measured_batch_seconds",
        "batch_latency_p50_seconds",
        "batch_latency_p95_seconds",
    ):
        assert empty_only[metric] == {
            "mean": None,
            "stdev": None,
            "min": None,
            "max": None,
        }


def test_unscheduled_warmup_executes_loads_without_counting_them(scripts, monkeypatch):
    bench = scripts["bench_access"]

    class Progress:
        def update(self, count=1):
            pass

        def close(self):
            pass

    monkeypatch.setattr(bench, "tqdm", lambda **kwargs: Progress())

    class Output:
        data = np.array([1], dtype=np.uint16)

    class Bank:
        def __init__(self):
            self.loads = []

        def load(self, dataset_id, cells, **kwargs):
            self.loads.append((dataset_id, tuple(cells)))
            return Output()

    args = types.SimpleNamespace(gene_mode="native", dtype="stored", batch_size=2, quiet=True)
    bank = Bank()
    sample = bench.bench_unscheduled_once(
        bank,
        ids=["dataset"],
        order=np.arange(4, dtype=np.int64),
        offsets=np.array([0, 4], dtype=np.int64),
        args=args,
        warmup=1,
    )

    assert bank.loads == [("dataset", (0, 1)), ("dataset", (2, 3))]
    assert sample["batches"] == 1
    assert sample["cells"] == 2
    assert sample["bytes"] == 2
    assert sample["checksum"] == 1
