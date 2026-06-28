# scdata

<p align="center">
  <strong>High-performance storage, compression, and loading for single-cell data.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-≥3.10-blue?logo=python&logoColor=white"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust" src="https://img.shields.io/badge/Rust-core-dea584?logo=rust&logoColor=white"></a>
  <a href="https://pyo3.rs/"><img alt="PyO3" src="https://img.shields.io/badge/PyO3-bindings-d11141"></a>
  <a href="https://github.com/zarr-developers/zarr-specs"><img alt="Zarr v3" src="https://img.shields.io/badge/Zarr-v3-purple"></a>
  <a href="https://docs.anndata.dev/"><img alt="anndata" src="https://img.shields.io/badge/anndata-interop-1f77b4"></a>
  <img alt="status" src="https://img.shields.io/badge/status-WIP-orange">
</p>

---

`scdata` is a Python package — backed by a Rust core — for storing, compressing,
and loading single-cell data at scale. It is designed to replace the stock
`anndata` / `zarr` read path in training pipelines, with a target throughput of
**~20 GiB/s decode bandwidth** and **~32k cells/s** end-to-end access.

> Performance figures and methodology live in [`docs/benchmark-report.md`](docs/benchmark-report.md).

## ✨ Features

| | What you get |
|---|---|
| ⚡ **Direct chunk access** | Python parses only zarr v3 metadata; the Rust core opens each chunk by `(file, offset, len)` and decodes it. Directory stores and `ZIP_STORED` archives share one code path. |
| 🧩 **Unified matrix abstraction** | Three dataset kinds — `Dense1D`, `Dense2D`, `SparseCsr` — all registered under a single `DatasetId`. The Rust core stays decoupled from layer names. |
| 🗂️ **`layers` support** | `launch_all` / `register_all` parse both `X` and every `layers/<name>` matrix into the same `DenseDataset` / `SparseDataset` types, one `DatasetId` per matrix. |
| 🔮 **Scheduled prefetch** | `prefetch_cells_scheduled` streams cell batches through a ring buffer of decoded results, feeding `ScDataLoader` for PyTorch training loops. |
| 🔢 **Full numeric dtypes** | `u8 / i8 / u16 / i16 / u32 / i32 / u64 / i64 / f32 / f64`, plus `f16` → numpy `float16` and `bf16` → `uint16` raw bits. |
| 🚇 **Zero-copy return** | The Rust-allocated `Vec<T>` is handed straight to numpy via `IntoPyArray` — the decoded buffer is never copied on the return path. |
| 🧠 **Tunable runtime** | io_uring or threaded IO pool, sharded access scheduler with LRU cache and memory budget, multi-threaded decode / compute pools — all configurable from Python. |

### Pipeline at a glance

```
anndata.AnnData ──write_zarr──▶ scdata zarr v3 store
                                      │
                                      │  launch() / launch_all()   (metadata only)
                                      ▼
                          ScDataBank.register()  ──▶ DatasetId
                                      │
                   ┌──────────────────┼───────────────────┐
                   ▼                  ▼                   ▼
            bank.load()      bank.prefetch()        ScDataLoader
           (random access)  (streaming batches)    (training loop)
                   └──────────────────┴───────────────────┘
                                      │
                                      ▼
                              numpy / torch tensors
```

### Public API surface

```python
# IO — zarr v3 read/write, anndata interop, store launch
from scdata import write_zarr, read_zarr, launch, launch_all, AnnDataZarrZipConverter

# Data — dataset metadata, cell carriers, training loader (pure Python)
from scdata import (
    DenseDataset, SparseDataset, DatasetCollection,
    CellData, CellBatch, CellAccess,
    ScDataLoader, DType, CodecPipeline,
)

# DataBank — Rust-backed access facade + config dataclasses
from scdata import (
    ScDataBank, DataBankConfig, IoConfig, AccessConfig, FillConfig,
    ScheduledPrefetchConfig, MissingGenePolicy,
)

# Example: register a store and stream cell batches
bank = ScDataBank()
did = bank.register(launch("pbmc.zarr.zip"))          # metadata-only parse
batch = bank.load(did, [0, 1, 2])                      # → CellData, row-major [3 * n_genes]
bank.unregister(did); bank.close()
```

## 📦 Installation

`scdata` builds its Rust extension via [`maturin`](https://www.maturin.rs/). From the
repository root:

```sh
labpon                                # enable network proxy on the dev box
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv sync --extra dev                   # python deps (maturin, pytest, anndata, zarr, ...)
uv run maturin develop --uv           # build the Rust extension into .venv
```

Editable install alternative:

```sh
uv pip install -e .
```

Smoke test:

```sh
uv run python -c "import scdata; print(scdata.kernel_name(), scdata.kernel_version())"
```

**Requirements:** Python ≥ 3.10, numpy ≥ 2.2, numcodecs ≥ 0.13. `anndata` / `zarr` are
only needed for interop and round-trip validation (`pip install -e ".[anndata]"`).

## 🛠️ Development

### Common commands

```sh
# Python
uv run pytest -q                              # test suite
uv run ruff check scdata tests                # lint

# Rust
cargo check  --manifest-path rust/scdata/Cargo.toml
cargo test   --manifest-path rust/scdata/Cargo.toml
cargo bench  --manifest-path rust/scdata/Cargo.toml   # modules / stress / iopool / fullchain
```

### Repository layout

```
scdata/                Python package
├── io/                zarr v3 read/write, anndata interop, launch
├── data/              Dataset / CellData / ScDataLoader / Prefetch  (pure Python)
├── databank.py        ScDataBank + config dataclasses  (Rust wrapper layer)
└── _scdata.abi3.so    built Rust extension
rust/scdata/src/       Rust core
├── databank/          registration, access facade, batch, scheduling plan
├── access/            sharded scheduler, LRU cache, CPU pool, memory budget
├── codecs/            zstd/lz4/blosc pipeline, decode pool, registry
├── iopool/            io_uring / threaded IO backends
└── pybind.rs          PyO3 bindings (the only Python-reflection layer)
docs/                  benchmark report
examples/              usage notebooks
tests/                 pytest suite
scripts/               compression / numcodecs export benchmarks
```

### Configuring the runtime

`ScDataBank(config)` accepts a tree of Pythonic dataclasses; flat kwargs are routed
to nested fields via `DataBankConfig.make(...)`:

```python
from scdata import ScDataBank, DataBankConfig

cfg = DataBankConfig.make(
    backend="uring",                     # → io_config.backend
    entries=256,                         # → io_config.uring_config.entries
    cache_capacity_bytes=512 * 1024**2,  # → access_config.cache_capacity_bytes
    decode__num_workers=16,              # disambiguated by a ``__`` path
)
bank = ScDataBank(cfg)
```

Tunable surfaces cover the IO backend (io_uring / threaded), decode pool, access
scheduler (shards / cache / memory budget), compute/fill pool, and scheduled-prefetch
depth. See the `scdata.databank` module docstring for the full field reference.

### Benchmarks

Rust bench harnesses and reproducible commands are documented in
[`docs/benchmark-report.md`](docs/benchmark-report.md), including codec bandwidth,
scheduler saturation, databank no-block stress tests, and the scheduled-prefetch
full chain.

## 📄 License

Internal project — no open-source license assigned yet.
