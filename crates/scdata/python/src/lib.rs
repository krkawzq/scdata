//! Private PyO3 boundary for the public pure-Python `scdata` package.

mod config;
mod dataset;
mod error;
mod output;
mod plan;
mod session;
#[cfg(all(target_os = "linux", target_has_atomic = "64"))]
mod shared;
mod stats;

use pyo3::prelude::*;
use scdata::ReadLimits;

use crate::config::{plan_config_defaults, session_config_defaults};
use crate::dataset::{open_dataset, PyDataset};
use crate::plan::{compile_plan, PyPlan};
use crate::session::PySession;

const OUTPUT_DTYPES: [&str; 6] = ["i16", "i32", "u16", "u32", "f32", "f64"];

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("OUTPUT_DTYPES", OUTPUT_DTYPES)?;
    let limits = ReadLimits::default();
    module.add("DEFAULT_MAXIMUM_METADATA_SIZE", limits.metadata_size())?;
    module.add("DEFAULT_MAXIMUM_ENCODED_SIZE", limits.encoded_size())?;
    module.add("DEFAULT_MAXIMUM_DECODED_SIZE", limits.decoded_size())?;
    module.add("DEFAULT_MAXIMUM_BLOCK_COUNT", limits.block_count())?;
    error::register(module)?;
    module.add_class::<PyDataset>()?;
    module.add_class::<PyPlan>()?;
    module.add_class::<PySession>()?;
    #[cfg(all(target_os = "linux", target_has_atomic = "64"))]
    shared::register(module)?;
    module.add_function(wrap_pyfunction!(plan_config_defaults, module)?)?;
    module.add_function(wrap_pyfunction!(session_config_defaults, module)?)?;
    module.add_function(wrap_pyfunction!(open_dataset, module)?)?;
    module.add_function(wrap_pyfunction!(compile_plan, module)?)?;
    Ok(())
}
