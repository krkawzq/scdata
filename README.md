# scdata

<p align="center">
  <strong>High-performance storage, compression, and loading for single-cell data.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-≥3.10-blue?logo=python&logoColor=white"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust" src="https://img.shields.io/badge/Rust-core-dea584?logo=rust&logoColor=white"></a>
  <a href="https://pyo3.rs/"><img alt="PyO3" src="https://img.shields.io/badge/PyO3-bindings-d11141"></a>
  <a href="https://github.com/zarr-developers/zarr-specs"><img alt="Zarr v3" src="https://img.shields.io/badge/Zarr-v3-purple"></a>
  <img alt="status" src="https://img.shields.io/badge/status-WIP-orange">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

---

`scdata` solves one engineering problem: **random access of a huge number of
small vectors out of massive compressed file stores.**

Single-cell data is exactly this shape — every cell is a small gene-expression
vector (a few thousand floats), datasets hold millions of cells, and training
samples them at random. Stock `anndata` / `zarr` reads are built for sequential
scans, not for millions of tiny random reads against a compressed archive.
`scdata` is a Python package backed by a Rust core that rebuilds the entire read
path — IO, decompression, scheduling, and single-cell layout — for this
workload, with a target of **~20 GiB/s decode bandwidth** and **~32k cells/s**
end-to-end access on GPFS.

## ✨ Features

### ⚡ IO — batched random reads against huge files

- **io_uring backend.** SQE/CQE batches amortize per-read syscall cost;
  `IOSQE_ASYNC` plus kernel io-wq give true async reads; fixed-file registration
  skips per-IO fd lookup. A threaded `pread` backend falls back off-Linux.
- **Byte-range planning.** A cell's vector (a few KB) is sliced out of a large
  chunk (tens of MB) — only the needed byte range is read, never the whole chunk.
  Dense rows, 1D spans, and variable-length CSR rows all plan into minimal ranges.
- **Two-level dedup.** Concurrent requests for the same chunk share one IO (keyed
  by `ChunkKey`); the same chunk+codec shares one decode (keyed by `DecodeKey`).
  A million-cell sample never re-reads or re-decodes a chunk it already has.
- **Sharded `QueueCore`.** Priority queues, per-file ordering, and read
  coalescing live behind a sharded queue keyed by byte range, so dedup holds
  while lock contention stays low.

### 🧠 Scheduling — a lock-free, memory-bounded pipeline

- **Sharded scheduler.** Each shard is a single current-thread tokio runtime with
  `Rc<RefCell>` state — fully lock-free within a shard, parallel across shards,
  routed by a splitmix64 hash of the chunk key.
- **IO → decode → scatter, pipelined.** The three stages run on independent
  thread pools; different chunks run fully in parallel, while the same chunk's
  IO and decode stay ordered.
- **Dual-form LRU cache (raw + decoded).** A hit skips IO; a decoded hit skips
  IO *and* decode. Pinned entries cannot be evicted while a caller holds them.
- **Unified memory budget.** Cache, in-flight, and staging bytes share one
  budget; over-budget triggers three-stage eviction (cache → staging → blocking
  backpressure) instead of OOM.
- **Three-stage scheduled prefetch.** Independent look-ahead depths for IO →
  decode → ready, drained through a ring buffer; dropping the consumer cancels
  in-flight work rather than leaking it.

### 🧬 Single-cell native layout

- **Three matrix kinds.** `Dense1D`, `Dense2D`, and `SparseCsr`, all registered
  under one `DatasetId`; `launch_all` / `register_all` expose `X` and every
  `layers/<name>` matrix as independent datasets.
- **Rectilinear cell-aligned CSR.** Chunk boundaries align to cell boundaries, so
  a random cell's sparse row lands in a single chunk — minimizing random IO on
  variable-length rows, the decisive cost for sparse single-cell access.
- **Gene interning.** Identical gene names across datasets share one `Arc<str>`;
  a `repr(C)` pointer/len view exposes them to Python with zero allocation.
- **Parallel scatter/fill.** A compute pool assembles decoded bytes into the
  output matrix (with CSR deserialization), triggered above a row/byte threshold.

### Rust core architecture

```
                         pybind.rs
                     (PyO3 reflection layer)
                              │
                              ▼
                     ┌──────────────────┐
                     │     DataBank     │
                     │  facade: regis-  │
                     │  tration, gene   │
                     │  interner, ac-   │
                     │  cess entry pts  │
                     └────────┬─────────┘
              ┌───────────────┼───────────────┐
              │               │               │
   ┌──────────▼───────┐ ┌─────▼──────────┐ ┌──▼────────────────┐
   │  DatasetRegistry │ │ AccessScheduler│ │ DataBankCompute   │
   │                  │ │ (sharded,      │ │   Pool            │
   │ Dense1D/Dense2D/ │ │  lock-free)    │ │ (scatter / fill)  │
   │ SparseCsr        │ │                │ │                   │
   │                  │ │ each shard =   │ │ • dense row       │
   │ • Array / Chunk  │ │  single cur-   │ │   scatter         │
   │ • byte-range     │ │  rent_thread   │ │ • CSR deserialize │
   │   chunk locations│ │  tokio rt +    │ │ • gene projection │
   │   (FileOffset /  │ │  Rc<RefCell>   │ │                   │
   │    Dir / Memory) │ │                │ │ triggered above   │
   │                  │ │ routed by      │ │ rows/bytes        │
   │ • ArrayGrid      │ │ chunk hash     │ │ threshold         │
   │   Regular /      │ │                │ └───────────────────┘
   │   Rectilinear    │ └───────┬────────┘
   └──────────────────┘         │
                       ┌────────┴────────┐
                       │  per-shard      │
                       │                 │
                       │ • LRU cache     │ ┌──────────────────┐
                       │   (raw +        │▶│  MemoryBudget    │
                       │    decoded,     │ │                  │
                       │    pinned)      │ │ cache + inflight │
                       │                 │ │ + staging        │
                       │ • inflight      │ │                  │
                       │   dedup         │ │ 3-stage eviction │
                       │   (IO + decode) │ │ → backpressure   │
                       │                 │ └──────────────────┘
                       │ • 3-stage       │
                       │   scheduled     │
                       │   prefetch      │
                       │   (ring buffer) │
                       └────────┬────────┘
                                │
                     ┌──────────┴──────────┐
                     │  IoBackend /        │
                     │  DecodeBackend      │  (trait adapters —
                     │  (boxed async       │   access core
                     │   futures)          │   stays backend-                     
                     └──────────┬──────────┘   agnostic)
                                │
             ┌──────────────────┴──────────────────────┐
             ▼                                         ▼
   ┌──────────────────────┐                ┌──────────────────────┐
   │  IoPool              │                │  DecodePool          │
   │                      │  encoded bytes │                      │
   │  io_uring backend:   │◀──────────────▶│  dedicated OS        │
   │  SQE/CQE batches,    │  (Arc<[u8]>)   │  threads             │
   │  IOSQE_ASYNC +       │                │                      │
   │  kernel io-wq,       │                │  zstd / lz4 / blosc  │
   │  fixed files         │                │  / gzip / bz2 / xz   │
   │                      │                │  / crc32 / identity  │
   │  threaded pread      │                │                      │
   │  fallback            │                │  bounded channel +   │
   │                      │                │  backpressure        │
   │  sharded QueueCore:  │                │                      │
   │  priority queues,    │                │  zero-alloc decode:  │
   │  read coalescing,    │                │  set_len, buffer     │
   │  per-file ordering   │                │  reuse, spare ping-  │
   └──────────┬───────────┘                │  pong, codec cache   │
              │                            └──────────────────────┘
              ▼  positioned reads by (file, offset, len)
        disk / zarr v3 store / ZIP_STORED archive
```

The `DataBank` facade owns a `DatasetRegistry`, a sharded `AccessScheduler`, and
a `DataBankComputePool`. A cell request is **planned** into byte ranges against
registered chunks, the scheduler **reads** each chunk via `IoBackend`, **decodes**
it via `DecodeBackend`, and the compute pool **scatters** the decoded bytes into
the output matrix. All four — IO, decode, scheduling, scatter — are independent
pools running concurrently, with cache hits, dedup, and the memory budget keeping
random access of small vectors from ever stalling on a single chunk.

## 📦 Installation

`scdata` builds its Rust extension via [`maturin`](https://www.maturin.rs/). From the
repository root:

```sh
uv sync --extra dev
uv run maturin develop --uv
```

Editable install alternative:

```sh
uv pip install -e .
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

## 📄 License

Licensed under the [MIT License](LICENSE).
