use pyo3::prelude::*;

pub mod access;
pub mod codecs;
pub mod databank;
pub mod iopool;

mod pybind;

#[pyfunction]
fn kernel_name() -> &'static str {
    "scdata-rust"
}

#[pyfunction]
fn kernel_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _scdata(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(kernel_name, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_version, m)?)?;
    pybind::register(m)?;
    Ok(())
}
