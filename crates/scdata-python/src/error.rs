//! A single structured error type for the private extension boundary.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_core, CoreError, PyException);

pub(crate) fn compress_kind(error: &sc_compress::Error) -> &'static str {
    match error {
        sc_compress::Error::Io(_) => "io",
        sc_compress::Error::Json(_) => "json",
        sc_compress::Error::DynBlosc(_) => "codec",
        sc_compress::Error::Zip(_) => "zip",
        sc_compress::Error::Allocation(_) => "allocation",
        sc_compress::Error::NotFound { .. } => "not_found",
        sc_compress::Error::InvalidArgument(_) => "invalid_argument",
        sc_compress::Error::InvalidMeta(_) => "invalid_meta",
        sc_compress::Error::CorruptData { .. } => "corrupt_data",
        sc_compress::Error::Path { .. } => "path",
    }
}

pub(crate) fn load_kind(error: &sc_load::Error) -> &'static str {
    match error {
        sc_load::Error::InvalidConfig(_) => "invalid_config",
        sc_load::Error::InvalidInput(_) => "invalid_input",
        sc_load::Error::InvalidDataset(_) => "invalid_dataset",
        sc_load::Error::ResourceLimit(_) => "resource_limit",
        sc_load::Error::StalePlan(_) => "stale_plan",
        sc_load::Error::Unsupported(_) => "unsupported",
        sc_load::Error::Io { .. } => "io",
        sc_load::Error::Decode(_) => "decode",
        sc_load::Error::Promote(_) => "promotion",
        sc_load::Error::Conversion(_) => "conversion",
        sc_load::Error::Cancelled => "cancelled",
        sc_load::Error::Session(_) => "session",
        sc_load::Error::WorkerPanic => "worker_panic",
        sc_load::Error::Allocation(_) => "allocation",
        sc_load::Error::Invariant(_) => "internal",
    }
}

pub(crate) fn from_compress(error: sc_compress::Error) -> PyErr {
    attach_kind(CoreError::new_err(error.to_string()), compress_kind(&error))
}

pub(crate) fn from_load(error: sc_load::Error) -> PyErr {
    attach_kind(CoreError::new_err(error.to_string()), load_kind(&error))
}

pub(crate) fn from_rust(error: sc_load::Error) -> PyErr {
    from_load(error)
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> PyErr {
    attach_kind(CoreError::new_err(message.into()), "invalid_argument")
}

pub(crate) fn invalid_input(message: impl Into<String>) -> PyErr {
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

impl<T> ResultExt<T> for Result<T, sc_compress::Error> {
    fn map_sc(self) -> PyResult<T> {
        self.map_err(from_compress)
    }
}

impl<T> ResultExt<T> for Result<T, sc_load::Error> {
    fn map_sc(self) -> PyResult<T> {
        self.map_err(from_load)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("CoreError", module.py().get_type::<CoreError>())
}
