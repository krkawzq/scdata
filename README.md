# scdata

<p align="center">
  <strong>Read single-cell data off disk as fast as if it were in memory.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-≥3.10-blue?logo=python&logoColor=white"></a>
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
  dedup, and a unified memory budget.
- **Fully compressed.** Sparse matrices stay CSR; chunks are compressed with
  Blosc/LZ4 (zstd, lz4, gzip, … also supported). Stored as a single
  `zarr.zip` archive that stock `anndata.read_zarr` can still open.
- **On-demand 64 KiB blocks.** Reads and decompression are pushed down to the
  Blosc compressed block — only the blocks covering the requested cells are
  touched, so a random batch never pays for a whole chunk.
- **Rust core, Python API.** IO (`io_uring` / threaded `pread`), codecs, and
  scheduling live in a Rust extension; you drive it from a small Python API.

## 📦 Installation

`scdata` builds its Rust extension with [`maturin`](https://www.maturin.rs/):

```sh
uv sync --extra dev
uv run maturin develop --uv          # editable build
# or
uv pip install -e .
```

**Requirements:** Python ≥ 3.10, numpy ≥ 2.2, numcodecs ≥ 0.13. Install
`anndata` / `zarr` only for conversion and round-trip validation
(`uv pip install -e ".[anndata]"`).

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

### 2. Random cell access, on demand

```python
from pathlib import Path
from scdata import ScDataBank, DataBankConfig, launch_all

bank = ScDataBank(DataBankConfig.make(backend="threaded"))
ds = launch_all(Path("dataset.zarr.zip"))["X"]   # or "raw/X", "layers/counts"
id = bank.register(ds)

genes = bank.dataset_genes(id)[:4096]
cell = bank.load(id, cells=[0, 1, 2, 3], genes=genes, missing="zero")
print(cell.data.shape)                            # (4, 4096)

bank.close()
```

### 3. Stream training batches (random order, many datasets)

Build a `CellIndexPlan` describing which cell comes from which dataset, and
stream dense batches ready for the GPU. `fast_mode="force"` turns on the
native 64 KiB-block fast path.

```python
import numpy as np
from pathlib import Path
from scdata import (
    ScDataBank, DataBankConfig, launch_all,
    CellIndexPlan, ScheduledAccessConfig, ScheduledPrefetchConfig,
)

paths = [Path(f"ds{i}.zarr.zip") for i in range(8)]
bank = ScDataBank(DataBankConfig.make(backend="threaded", io__threaded__num_workers=16))
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
    x = batch.data        # (128, 4096) dense ndarray
    # ...feed to your model

bank.close()
```

See `examples/` for runnable random- and sequential-access benchmarks.

## 📄 License

Licensed under the [MIT License](LICENSE).
