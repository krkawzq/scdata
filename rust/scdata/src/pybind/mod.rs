//! PyO3 bindings for the Rust scdata core.
//!
//! The binding layer is intentionally thin: Python-facing classes collect
//! values, then build the current Rust `databank` specs directly. No legacy
//! `*Meta` compatibility surface is kept here.

mod arrays;
mod config;
mod databank;
mod dispatch;
mod dtype;
mod errors;
mod ids;
mod index;
mod prefetch;
mod profile;
mod zip;

use pyo3::prelude::*;

pub(crate) use errors::DataBankError;

/// Register the Python-facing names exposed by this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    databank::register(m)?;
    ids::register(m)?;
    config::register(m)?;
    prefetch::register(m)?;
    zip::register(m)?;
    index::register(m)?;
    m.add("DataBankError", m.py().get_type::<DataBankError>())?;
    Ok(())
}
