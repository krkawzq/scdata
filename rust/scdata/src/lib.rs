#[cfg(feature = "python-extension")]
use pyo3::prelude::*;

pub mod access;
pub mod codecs;
pub mod databank;
pub mod iopool;
pub mod profile;

#[doc(hidden)]
#[cfg(feature = "python-extension")]
pub mod pybind;

#[cfg(feature = "python-extension")]
#[pyfunction]
fn kernel_name() -> &'static str {
    "scdata-rust"
}

#[cfg(feature = "python-extension")]
#[pyfunction]
fn kernel_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(feature = "python-extension")]
#[pymodule]
fn _scdata(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(kernel_name, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_version, m)?)?;
    pybind::register(m)?;
    Ok(())
}
