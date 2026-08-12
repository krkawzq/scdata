# sc-load

`sc-load` compiles an ordered list of `(source, row)` accesses into immutable,
block-granular jobs over `sc-compress` stores. A compiled plan can be reused;
each session owns a separate cache-line-aligned output ring, completion
counters, consumer cursor, error state, and worker set.

The session worker owns each job end to end:

```text
physical read(s) -> block decode -> structure validation -> dense/csr scatter
                 -> release batch completion
```

There is no separate public I/O pool. `Blocking` performs bounded positioned or
whole-key reads in the algorithm workers. On Linux, the `uring` feature enables
one independent ring per worker and keeps multiple jobs in flight. `Auto`
selects io_uring only when every source is positioned and every worker ring can
be created before the session begins; Linux FUSE sources select `Blocking`
because their request path does not benefit reliably from per-worker rings.

## Precision model

- **Storage** payload dtypes match `sc-compress` matrix values:
  `i16 | i32 | u16 | u32 | f32 | f64`.
- **Output** is an explicit [`OutputDType`] of the same set.
- Only **promotions** are allowed at compile time (no width narrowing, no
  float→int). Integer→float is exact by default: `i16/u16→f32` and every
  supported integer→`f64` edge are accepted; `i32/u32→f32` requires explicit
  [`FloatCastPolicy::AllowRounding`].
- Signedness changes (`i↔u`) run optional runtime checks controlled by
  [`OverflowPolicy`]:
  - `Error` (default) — fail the session
  - `UseFill` — write [`OutputSpec::fill`]
  - `UseValue(Fill)` — write a separate sentinel
  - `Unchecked` — Rust `as` casts, no checks

Unmapped feature columns always receive `fill` (independent of overflow policy).
On little-endian x86-64, contiguous promotion kernels bind AVX-512, AVX2, or
the x86-64 SSE2 baseline once when a source plan is compiled. Checked-sign and
canonical CSR-index validation use AVX2 when available. Other targets retain
the same scalar semantics, and every vector kernel has an exact scalar tail.

## Example

```rust,no_run
use sc_load::{
    compile, Dataset, Fill, OutputDType, OutputSpec, PlanSpec, RowRef,
    SessionConfig, Source, SourceId,
};

# fn run() -> sc_load::Result<()> {
let dataset = Dataset::open("matrix.sc-compress")?;
let source = Source::new(0, dataset);
let rows = vec![
    RowRef::new(SourceId::new(0), 7),
    RowRef::new(SourceId::new(0), 2),
];
let output = OutputSpec::new(30_000, OutputDType::F32, Fill::F32(0.0))?;
let plan = compile(PlanSpec::new(vec![source], rows, output, 256, 8))?;

let mut session = plan.open(SessionConfig::default())?;
while let Some(batch) = session.next_batch()? {
    // `batch` keeps its physical ring slot leased. Finish any asynchronous
    // reader (for example an H2D copy) before dropping or releasing it.
    for row in 0..batch.rows() {
        consume_row(batch.row_as::<f32>(row)?);
    }
}
# Ok(())
# }
# fn consume_row(_: &[f32]) {}
```

[`FeatureMap`] supports feature selection/permutation and rejects duplicate
output targets; target bounds are checked against the final [`OutputSpec`] by
the compiler. An explicit identity map is canonicalized to the identity fast
path. Mapped dense sources store only surviving `(source_byte, target_byte)`
entries; ordinary offsets pack both `u32` values into one `u64`, with a wide
fallback for rows above 4 GiB. CSR target-byte maps similarly use `u32` plus a
sentinel when possible and fall back to `usize`, without narrowing the supported
size boundary. CSR rows must use canonical, strictly increasing unique column
indices. Rows in the output ring are 64-byte aligned and padded; `row_as`
returns logical values, while `as_padded_slice` explicitly includes initialized
padding. Compile and session configurations carry hard output, arena,
working-set payload, per-job and per-worker decoded, encoded, queue-depth, and
aggregate resident-buffer limits. The compile payload includes decoder block
indexes, lookup tables, and bounded prefix/whole-key parse inputs. The default
aggregate compile working-set limit is 40 GiB; callers may still override it.
Chunk metadata reads are parallelized for large grids using
`PlanConfig::compile_io_concurrency` (available CPU count, capped at 32, by
default) and remain subject to the aggregate compile limit. Small grids keep the
single-pass lazy path so local in-memory compilation does not pay a second row
scan.

Within the configured coalescing window, physically overlapping or adjacent
block reads are consolidated in physical order without adding read
amplification. Consolidation follows the configured bandwidth/IOPS balance and
stops before `PlanConfig::max_coalesced_io_bytes` (32 MiB by default), while the
explicit per-job encoded/decoded resource limits remain hard bounds. The
coalescing limit does not split an indivisible source block or whole key.

Execution keeps all data-dependent CSR and fallible-conversion checks in a safe
validation phase. Compiler-sealed infallible dense ranges do not pay a duplicate
row scan. Only after those invariants hold does the private commit phase use
unchecked indexing and raw pointers. Each such block states the local `SAFETY`
invariant; invalid external data is rejected before any completion count is
published. A single-cell dense identity block decodes directly into its unique
ring row, eliminating decoded scratch and the subsequent full-row copy. Other
dense identity rows skip default filling; fresh zero-filled sparse rows reuse
the anonymous mapping's initial zero state. Row padding is initialized once and
never rewritten. Job completion
updates are grouped by contiguous ring batch to reduce atomic cache-line
traffic. Each immutable `CellTask` occupies 32 bytes. Consumers are
notified only when at least one batch actually becomes ready.

## Profiling

Profiling is compile-time opt-in with the `profile` feature. It adds plan-phase
timers and 256-byte, cache-line-exclusive per-worker shards for I/O, decode,
validation, scatter, completion, scheduler contention, and io_uring telemetry.
Cell resolution, candidate compaction, and same-block grouping are one fused
pass and are reported together as `compile_resolve_ns`; the retained
`compile_same_block_ns` schema field is zero.

Without `--features profile`, the timer fields, worker shards, telemetry
atomics, in-flight aggregate counters, and their update instructions are not
compiled. `RuntimeStats` then contains only configuration, selected I/O mode,
and session state. Applications and custom benchmarks that need detailed
counters must compile with `--features profile` and retrieve plan/session
statistics through the public API.

## Shared-ring benchmark

The built-in benchmark compares ordinary single-consumer delivery with the
distributed shared-ring path:

```sh
SC_LOAD_BENCH_MODE=both cargo bench -p sc-load --bench shared_ring
```

`SC_LOAD_BENCH_ROWS`, `SC_LOAD_BENCH_COLUMNS`, `SC_LOAD_BENCH_BATCH_SIZE`,
`SC_LOAD_BENCH_PREFETCH_STEP`, `SC_LOAD_BENCH_WORLD_SIZE`, and
`SC_LOAD_BENCH_ITERATIONS` override its workload defaults.

## Distributed shared ring

On Linux targets with lock-free 64-bit atomics, `Plan::open_shared` runs one
ordinary execution session against a sealed `memfd` mapping. Logical batches
are assigned round-robin (`logical_batch % world_size`) and published through
per-rank futexes. Each rank has one live owner, while every returned
`SharedBatch` owns its generation lease and may outlive the `SharedClient` that
requested it.

The producer can publish up to the plan's existing ring capacity without
waiting for the globally oldest generation, so an idle rank does not impose a
mailbox-wide head-of-line stall. Ring reuse remains bounded and globally safe:
release generations are committed in order, and rank resume cursors are
advanced before a slot can be reused. Retaining more same-rank batches than the
ring can hold returns an explicit error rather than waiting on the caller's own
lease.

The control region is size-limited and page-aligned. The data mapping is
read-only in consumers, the file size is sealed, layout arithmetic and header
fields are validated at attachment, and process ownership rejects cross-fork
use of producer, client, and batch handles as well as duplicate rank consumers.
Ownership records combine PID with the Linux process start time so PID reuse
cannot keep a dead producer or rank owner apparently alive. Cancellation wakes
worker waits and both futex directions. Dropping an incomplete client cancels
the shared session; timed state checks also detect vanished or zombie processes
instead of waiting forever.
