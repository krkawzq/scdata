# scdata Python binding

The public API lives in `src/scdata`. This crate exposes only the private
`scdata._core` PyO3 handles used to open datasets, compile immutable plans, and
drive independent execution sessions. Sources, output policies, and configs
cross the boundary as normalized arrays and scalar dictionaries rather than
Rust-backed Python domain objects.

Every blocking Rust operation releases the GIL. Standard session batches are
copied into compact NumPy-owned arrays before the output-ring lease is released.
The distributed binding additionally exposes process-transferable rank handles.
Its public Python layer copies directly from a leased generation into the final
NumPy-owned array by default; `read()` drains the remaining rank into one final
allocation without a Python call or temporary view per batch. An explicit
zero-copy mode returns a read-only NumPy view whose base object owns the
shared-ring generation lease.
Transfer rank handles before their first attachment. An attached client or a
zero-copy array must not cross `fork`; use the default owned copies for data that
needs to outlive or move beyond its consuming process. Bulk `ranks()` creation
is all-or-nothing so a partial descriptor-allocation failure does not consume
rank identities or retain inaccessible handles.
