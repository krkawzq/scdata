//! A single structured error type for the private extension boundary.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use sc_load::Error as RustError;

create_exception!(_core, CoreError, PyException);

pub(crate) fn error_kind(error: &RustError) -> &'static str {
    match error {
        RustError::InvalidConfig(_) => "invalid_config",
        RustError::InvalidInput(_) => "invalid_input",
        RustError::InvalidDataset(_) => "invalid_dataset",
        RustError::ResourceLimit(_) => "resource_limit",
        RustError::StalePlan(_) => "stale_plan",
        RustError::Unsupported(_) => "unsupported",
        RustError::Io { .. } => "io",
        RustError::Decode(_) => "decode",
        RustError::Promote(_) => "promotion",
        RustError::Conversion(_) => "conversion",
        RustError::Cancelled => "cancelled",
        RustError::Session(_) => "session",
        RustError::WorkerPanic => "worker_panic",
        RustError::Allocation(_) => "allocation",
        RustError::Invariant(_) => "internal",
    }
}

pub(crate) fn from_rust(error: RustError) -> PyErr {
    attach_kind(CoreError::new_err(error.to_string()), error_kind(&error))
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> PyErr {
    attach_kind(CoreError::new_err(message.into()), "invalid_input")
}

fn attach_kind(error: PyErr, kind: &'static str) -> PyErr {
    Python::with_gil(|py| {
        let _result = error.value(py).setattr("kind", kind);
        error
    })
}

pub(crate) trait ResultExt<T> {
    fn map_sc(self) -> PyResult<T>;
}

impl<T> ResultExt<T> for Result<T, RustError> {
    fn map_sc(self) -> PyResult<T> {
        self.map_err(from_rust)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("CoreError", module.py().get_type::<CoreError>())
}
