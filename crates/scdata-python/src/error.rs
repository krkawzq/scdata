//! Python exception types for the public `scdata.exceptions` hierarchy.

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::PyTypeInfo;

create_exception!(_core, Error, PyException);
create_exception!(_core, Warning, PyUserWarning);
create_exception!(_core, PerformanceWarning, Warning);

macro_rules! define_errors {
    ($(($cls:ident, $kind:literal),)+) => {
        $(create_exception!(_core, $cls, Error);)+

        fn typed_err(message: impl Into<String>, kind: &'static str) -> PyErr {
            let message = message.into();
            match kind {
                $($kind => $cls::new_err(message),)+
                _ => Error::new_err(message),
            }
        }

        fn register_errors(module: &Bound<'_, PyModule>) -> PyResult<()> {
            export_type::<Error>(module, "Error", Some("unknown"))?;
            $(export_type::<$cls>(module, stringify!($cls), Some($kind))?;)+
            Ok(())
        }
    };
}

define_errors! {
    (InvalidArgumentError, "invalid_argument"),
    (InvalidInputError, "invalid_input"),
    (InvalidConfigError, "invalid_config"),
    (InvalidDatasetError, "invalid_dataset"),
    (InvalidMetaError, "invalid_meta"),
    (ResourceLimitError, "resource_limit"),
    (StalePlanError, "stale_plan"),
    (UnsupportedError, "unsupported"),
    (IoError, "io"),
    (JsonError, "json"),
    (CodecError, "codec"),
    (ZipError, "zip"),
    (DecodeError, "decode"),
    (PromotionError, "promotion"),
    (ConversionError, "conversion"),
    (CancelledError, "cancelled"),
    (SessionError, "session"),
    (WorkerPanicError, "worker_panic"),
    (AllocationError, "allocation"),
    (InternalError, "internal"),
    (NotFoundError, "not_found"),
    (CorruptDataError, "corrupt_data"),
    (PathError, "path"),
}

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
    if matches!(error, sc_compress::Error::InvalidArgument(_)) {
        return PyValueError::new_err(error.to_string());
    }
    typed_err(error.to_string(), compress_kind(&error))
}

pub(crate) fn from_load(error: sc_load::Error) -> PyErr {
    typed_err(error.to_string(), load_kind(&error))
}

pub(crate) fn from_rust(error: sc_load::Error) -> PyErr {
    from_load(error)
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

pub(crate) fn invalid_input(message: impl Into<String>) -> PyErr {
    typed_err(message, "invalid_input")
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

fn export_type<T>(module: &Bound<'_, PyModule>, name: &str, kind: Option<&str>) -> PyResult<()>
where
    T: PyTypeInfo,
{
    let cls: Bound<'_, PyType> = module.py().get_type::<T>();
    cls.setattr("__module__", "scdata.exceptions")?;
    if let Some(kind) = kind {
        cls.setattr("kind", kind)?;
    }
    module.add(name, cls)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_errors(module)?;
    export_type::<Warning>(module, "Warning", None)?;
    export_type::<PerformanceWarning>(module, "PerformanceWarning", None)?;
    Ok(())
}
