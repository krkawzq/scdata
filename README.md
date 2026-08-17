# scdata-toolkit

PyPI name `scdata-toolkit`, import name `scdata`. One Python package for on-disk SCC matrices (`ScDense` / `ScCsr`), store I/O,
AnnData `.scc` / `.scc.zip` containers, and prefetch planning.

Rust stays split (`dyn-blosc`, `sc-compress`, `sc-load`). The public Python
classes live in `src/scdata`; `scdata._core` is a private function-style
extension that only holds opaque handles and hot kernels.

## Development install

```sh
uv sync --extra dev
uv run maturin develop --release --strip --locked
```

Python 3.9 or newer and NumPy 2.0 or newer are required.

## Example

```python
import numpy as np
import scdata

# Open one matrix inside an AnnData scc container (directory or ZIP).
dataset = scdata.register("sample.scc.zip", key="X")

# Optional: align genes onto a target name list by exact string identity.
dataset = dataset.with_aligned_features(["g2", "g0", "extra", "gX"])
# Equivalent: dataset.with_feature_map(scdata.load.build_feature_map(source, target))
output = scdata.load.OutputSpec(4, np.float32, fill=-1)

rows = dataset.rows_for(["c7", "c2", "c11"])  # needs obs_names
for batch in scdata.prefetch(
    dataset,
    rows,
    output=output,
    batch_size=256,
    prefetch_step=8,
    config=scdata.SessionConfig(io_mode="auto"),
):
    # `batch` is compact, NumPy-owned, and remains valid after the Rust ring
    # slot has been released and reused.
    consume(batch)

# For smaller results, compile once and materialize.
plan = scdata.compile(dataset, rows, output=output)
matrix = plan.read()
```

`prefetch_step` is exactly the output-ring slot count. Decoded prefetch is
independent and uses `PlanConfig(cache_capacity_bytes=...)`. Compatible cache
loads are fused after residency assignment according to nested
`IoMergeConfig(policy="off" | "adjacent" | "cost")`.

Compiled requests can be moved without serializing native handles or runtime
pointers:

```python
plan.save("train.scplan", relative_sources=True)
lazy = scdata.Plan.load("train.scplan")  # no source I/O yet
lazy.bind(sources={0: "/new/mount/sample.scc"})
```

`Plan.dumps/loads` and pickle use the same bounded, checksummed plan image.
Source manifests are verified when a lazy plan first binds.

`register()` accepts `.scc` directories and `.scc.zip` archives. Discover
available matrices with `scdata.load.list_keys(path)` (`"X"`, `"layers/<name>"`,
`"raw/X"`, `"obsm/<name>"`, …), then pass `key=`. Expression keys expose
`feature_names` / `var_names` from `var` / `raw/var`; embedding keys leave
`feature_names` as `None`. Cell-aligned keys expose `obs_names` from container
`obs`. Build a gene map with `scdata.load.build_feature_map(source, target)`
(exact string identity; pandas `Index` accepted) or
`dataset.with_aligned_features(target_names)`. `limits=` accepts
`scdata.ReadLimits`. `Dataset` is pickleable (re-opens the store in the
receiving process) and supports `close()` / `with register(...) as dataset`.

Multiple datasets use collection order as `source_id`:

```python
a = scdata.register("a.scc", key="X")
b = scdata.register("b.scc", key="X")
target = a.feature_names  # or any ordered gene list
a = a.with_aligned_features(target)
b = b.with_aligned_features(target)
plan = scdata.compile(
    [a, b],
    [(0, 10), (1, 3), (0, 11)],
    output=scdata.load.OutputSpec(len(target), np.float32, fill=0),
)
```

With one identity-mapped dataset, `compile()` / `prefetch()` can infer the
output width and dtype. Multi-dataset or feature-mapped plans require an
explicit `OutputSpec`.

## Distributed loading

`Plan.open_distributed()` starts one producer in the parent process and returns
one process-transferable iterator per rank. Storage reads and decoding happen
once; logical batches are assigned round-robin across ranks.

```python
import multiprocessing as mp

import scdata


def consume_rank(loader: scdata.load.DistributedIterator) -> None:
    with loader:
        for batch in loader:
            train_step(batch)


if __name__ == "__main__":
    context = mp.get_context("spawn")
    with plan.open_distributed(
        world_size=4,
        config=scdata.SessionConfig(num_workers=8, io_mode="auto"),
    ) as distributed:
        processes = [
            context.Process(target=consume_rank, args=(loader,))
            for loader in distributed.ranks()
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"rank process exited with {process.exitcode}")
        distributed.wait()
```

The default `copy=True` yields compact, writable, NumPy-owned arrays and is the
safe choice for training loops. `distributed.rank(rank, copy=False)` is an
explicit read-only zero-copy mode: every retained array keeps one physical ring
generation leased, so release or drop views promptly. Holding enough views to
block the caller's own next batch raises `InvalidInputError` instead of
deadlocking. A rank iterator attaches lazily in its final process, may be
transferred only before consumption starts, and is the sole live owner of that
rank. Closing an incomplete attached iterator cancels the whole distributed
session so the producer and other ranks do not hang. `ranks()` creates its
remaining handles atomically: descriptor or allocation failure leaves every
rank retryable and closes handles created by that failed call.

Potentially rounding `i32/u32 -> f32` and `i64/u64 -> f64` conversion requires
`allow_float_rounding=True`. Signedness/range failures are controlled by the
output overflow policy: `error`, `use_fill`, `use_value`, or `unchecked`.

## Ownership and cancellation

`Plan` is immutable and reusable. `Plan.open()` creates a fresh `Session`.
`prefetch()` returns a `Prefetch` iterator that opens a session on first use.
Blocking dataset open, plan compilation, batch waiting, and worker shutdown all
release the Python GIL. Standard-session batches are copied once into compact
NumPy storage and never alias a reusable ring slot; the distributed path's
explicit `copy=False` exception follows the lease rules described above.

Sessions and prefetch handles are context managers; `cancel()` wakes blocked
consumers, while `close()` cancels unfinished work, joins workers, and releases
the output ring. Both plan and session statistics are immutable typed snapshots
with `as_dict()` for logging and serialization.

The Rust execution model and lower-level invariants are documented in
[`crates/sc-load/README.md`](crates/sc-load/README.md).

## Publishing

CI builds and tests on every push/PR to `main`. Pushing a `v*` tag (for example
`v0.2.0`) builds Linux manylinux wheels (`x86_64`, `aarch64`) plus sdists for
`scdata-toolkit`, then uploads it to PyPI.

1. Create a PyPI API token with upload access to `scdata-toolkit`.
2. Add repository secret `PYPI_API_TOKEN`.
3. Keep the package version aligned with the Cargo workspace version.
4. Tag and push when ready:

```sh
git tag v0.2.0
git push origin v0.2.0
```
