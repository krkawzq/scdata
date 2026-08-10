//! High-speed dense / CSR scatter and gather kernels.
//!
//! All hot paths are pure Rust (no NumPy/SciPy). Row-parallel work uses the
//! crate's bounded worker pool with fine-grained job chunks for dynamic load
//! balancing (similar to OpenMP `schedule(dynamic)`).

mod csr;
mod dense;
mod util;

pub use csr::{
    build_col_map, csr_select_rows, csr_to_dense, csr_to_dense_selected_cols, CsrColMap,
};
pub(crate) use csr::{csr_filter_cols, output_index_dtype, GatherColumns};
pub use dense::dense_select;
pub(crate) use util::{read_index_unchecked, write_index};
