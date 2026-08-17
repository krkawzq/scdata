//! Optional decoded-scatter profiling entry point.
//!
//! Select a suite with `SC_LOAD_SCATTER_PROFILE`: `real` (default), `all`,
//! `fastpaths`, `gather`, `csr-init`, `dense-init`, `csr-sparse`, `csr-hybrid`,
//! or `identity`.

fn main() {
    sc_load::run_scatter_profile_from_env();
}
