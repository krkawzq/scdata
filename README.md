# scdata

<p align="center">
  <strong>Read single-cell data off disk as fast as if it were in memory.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12–3.15-blue?logo=python&logoColor=white"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust" src="https://img.shields.io/badge/Rust-core-dea584?logo=rust&logoColor=white"></a>
  <a href="https://github.com/zarr-developers/zarr-specs"><img alt="Zarr v3" src="https://img.shields.io/badge/Zarr-v3-purple"></a>
  <img alt="status" src="https://img.shields.io/badge/status-WIP-orange">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

---

`scdata` is an IO toolkit built for one workload: **random access of single
cells out of massive, fully-compressed datasets.**

A single cell is just a small gene-expression vector (a few thousand floats),
yet training samples millions of them at random. Stock readers are built for
sequential scans — they pull and decompress whole chunks just to hand you a
few cells. `scdata` rebuilds the read path around the cell: data is stored as
CSR + Blosc/LZ4 (and other codecs) inside a portable `zarr.zip`, and on read
only the **64 KiB compressed blocks** a cell actually touches are fetched and
decompressed — never the whole chunk. The result is disk-resident data that
responds like an in-memory array, at a target of tens of thousands of
cells/s end-to-end.

## ✨ Features

- **Fast as memory.** Random cell access from compressed on-disk stores at
  near in-memory latency — pipelined IO → decode → scatter with caching,
  inflight dedup, coalescing, and a unified memory budget.
- **Fully compressed.** Sparse matrices stay CSR; chunks are compressed with
  Blosc/LZ4 (zstd, lz4, gzip, bz2, lzma … also supported). Stored as a single
  `zarr.zip` archive that `scdata.io.read_zarr` can open. Stock
  `anndata.read_zarr` cannot accept a `.zarr.zip` path directly; pass a
  `zarr.storage.ZipStore` instead.
- **On-demand 64 KiB blocks.** Reads and decompression are pushed down to the
  Blosc compressed block — only the blocks covering the requested cells are
  touched, so a random batch never pays for a whole chunk.
- **Native fast path.** For Blosc-LZ4 stores, a fused block-level pipeline
  (partial IO → partial decode → multi-consumer scatter) runs side-by-side
  with the generic chunk-level path and is selected automatically when every
  dataset in a stream is Blosc.
- **Rust core, Python API.** IO (`io_uring` / threaded `pread`), codecs, and
  scheduling live in a Rust extension (~55k lines); you drive it from a small
  Python API. A pure-Python fallback keeps the store format readable even
  before the extension is built.

## 📦 Installation

`scdata` builds its Rust extension with [`maturin`](https://www.maturin.rs/):

```sh
uv sync --extra dev
uv run maturin develop --uv          # editable build
# or
uv pip install -e .
```

**Requirements:** Python 3.12–3.15, numpy ≥ 2.2, numcodecs ≥ 0.13. Install
`anndata==0.13.2` / `zarr>=3.1.6,<3.3` only for conversion and round-trip
validation (`uv pip install -e ".[anndata]"`).

> The native fast path requires `io_uring`, so it is only available on
> Linux. On other platforms `scdata` falls back to the threaded `pread`
> backend automatically.

## 🚀 Usage

### 1. Convert AnnData → compressed `zarr.zip`

Any input `anndata` can read (`.h5ad`, `.zarr`, `.loom`, `.csv`, `.mtx`, …)
becomes a scdata store: sparse `X` written as CSR, chunks compressed with
Blosc-LZ4, default 64 KiB block size.

```python
from scdata import AnnDataZarrZipConverter

# writes dataset.zarr.zip next to the source (same stem, .zarr.zip suffix)
AnnDataZarrZipConverter()("path/to/dataset.h5ad")
```

For a `dataset.zarr.zip`, use `scdata.io.read_zarr(dataset_path)` to load an
AnnData object, or pass `zarr.storage.ZipStore(dataset_path, mode="r")` to
stock AnnData. `launch` / `launch_all` are direct-to-Rust chunk mappers, not
general zarr readers: arrays using `sharding_indexed` or another unknown codec
must be rewritten with `AnnDataZarrZipConverter` before they can be launched.

### 2. Random cell access, on demand

```python
from pathlib import Path
from scdata import ScDataBank, DataBankConfig, launch_all

bank = ScDataBank(DataBankConfig.make(backend="threaded"))
ds = launch_all(Path("dataset.zarr.zip"))["X"]   # or "raw/X", "layers/counts"
id = bank.register(ds)

genes = bank.dataset_genes(id)[:4096]
cell = bank.load(id, cells=[0, 1, 2, 3], genes=genes, missing="zero")
print(cell.data.shape)                            # (4 * 4096,) — 1-D row-major
print(cell.matrix.shape)                          # (4, 4096) — zero-copy view

bank.close()
```

`CellBatch.data` is a 1-D row-major array of length `num_cells * num_genes`;
cell `i`'s genes occupy `data[i*num_genes : (i+1)*num_genes]`. The `.matrix`
property reshapes it to `(num_cells, num_genes)` — zero-copy when `data` is
contiguous.

### 3. Stream training batches (random order, many datasets)

Build a `CellIndexPlan` describing which cell comes from which dataset, and
stream dense batches ready for the GPU. `fast_mode="force"` turns on the
native 64 KiB-block fast path; `fast_mode="auto"` (the default) enables it
when every registered dataset is Blosc-LZ4, and falls back to the generic
path otherwise.

```python
import numpy as np
from pathlib import Path
from scdata import (
    ScDataBank, DataBankConfig, launch_all,
    CellIndexPlan, ScheduledAccessConfig, ScheduledPrefetchConfig,
)

paths = [Path(f"ds{i}.zarr.zip") for i in range(8)]
bank = ScDataBank(DataBankConfig.make(
    backend="threaded",
    io__threaded__num_workers=16,
    fast__enabled=True,                 # enable the native fast path
    fast__fused_workers=4,
))
ids = [bank.register(launch_all(p)["X"]) for p in paths]
genes = bank.dataset_genes(ids[0])[:4096]

# a shuffled plan: (dataset_index, cell_index) for every cell, in batches of 128
counts = [bank.dataset_num_cells(i) for i in ids]
offsets = np.concatenate(([0], np.cumsum(counts)))
order = np.arange(offsets[-1])
np.random.default_rng(0).shuffle(order)
ds_idx = np.searchsorted(offsets[1:], order, side="right").astype(np.uint16)
cell_idx = (order - offsets[ds_idx]).astype(np.uint32)
plan = CellIndexPlan(ds_idx, cell_idx, 128)

config = ScheduledPrefetchConfig(
    prefetch_step=512,
    access=ScheduledAccessConfig(
        prefetch_step=512, decode_ahead_steps=512, ready_ahead_steps=512,
    ),
    fast_mode="force",
)

for batch in bank.prefetch_indexed(ids, plan, genes=genes, missing="zero", config=config):
    x = batch.matrix      # (128, 4096) ndarray — feed to your model

bank.close()
```

See `examples/` for runnable random- and sequential-access benchmarks, and
`scripts/` for batch conversion, recompression, and gene-alignment auditing.

## 🧠 How it works

Two read paths coexist under one API:

| | Generic path | Native fast path |
|---|---|---|
| **Granularity** | chunk | 64 KiB Blosc block |
| **Codecs** | any (blosc, zstd, lz4, gzip, bz2, lzma, …) | Blosc-LZ4 only |
| **IO** | positioned chunk read | block-range read, coalesced |
| **Decode** | whole chunk | touched blocks only, partial prefix decode |
| **Scatter** | slice after decode | block decode once → multi-consumer scatter |
| **Selection** | always available | `fast_mode="auto"` when all datasets Blosc; `"force"` to require |

When a stream is scheduled, the bank decides once at spawn time which path
each batch takes (`auto` falls back safely; `force` hard-fails if the
Blosc-LZ4 contract is not met — it never silently degrades). Caching,
inflight dedup, and a unified memory budget keep a random batch from paying
twice for a block, while `io_uring` (or threaded `pread`) and a separate
decode pool overlap IO with CPU work.

### Configuration

Every knob — IO backend, decode/access CPU pools, cache and memory budget,
native fast-path workers, scheduled lookahead — lives on a single
`DataBankConfig`. Construct it from nested dataclasses, a mapping, or
flat dotted kwargs:

```python
from scdata import DataBankConfig, IoConfig

# three equivalent ways to ask for io_uring with 256 queue entries
DataBankConfig.make(io_config=IoConfig.uring(entries=256))
DataBankConfig.make(backend="uring", entries=256)
DataBankConfig.make(io__uring__entries=256)
```

Inspect the resolved config with `bank.config`, or dump and reload it with
`DataBankConfig.from_dict(dict(bank.config))`.

### Profiling

`scdata` ships an always-on, low-overhead metric framework (compiled out
entirely when the `profile` Cargo feature is off). Each layer — IO pool,
decode pool, access scheduler, native executor, scheduled pipeline — reports
its own counters and timers:

```python
snap = bank.profile_snapshot_and_reset()   # dict of per-layer metrics
print(snap["label"], snap["elapsed_ms"])
```

Its granularity is controlled by `SCDATA_PROFILE*` environment variables read
once at startup, so profiling branches never sit on the hot path.

## 📄 License

Licensed under the [MIT License](LICENSE).
