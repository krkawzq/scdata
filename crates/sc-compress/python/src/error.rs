//! Structured exception hierarchy mirroring `sc_compress::Error`.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use sc_compress::Error as RustError;

create_exception!(_core, ScCompressError, PyException);
create_exception!(_core, IoError, ScCompressError);
create_exception!(_core, JsonError, ScCompressError);
create_exception!(_core, CodecError, ScCompressError);
create_exception!(_core, ZipError, ScCompressError);
create_exception!(_core, AllocationError, ScCompressError);
create_exception!(_core, NotFoundError, ScCompressError);
create_exception!(_core, InvalidArgumentError, ScCompressError);
create_exception!(_core, InvalidMetaError, ScCompressError);
create_exception!(_core, CorruptDataError, ScCompressError);
create_exception!(_core, PathError, ScCompressError);

/// Stable machine-readable kind string for a Rust error.
pub(crate) fn error_kind(err: &RustError) -> &'static str {
    match err {
        RustError::Io(_) => "io",
        RustError::Json(_) => "json",
        RustError::DynBlosc(_) => "codec",
        RustError::Zip(_) => "zip",
        RustError::Allocation(_) => "allocation",
        RustError::NotFound { .. } => "not_found",
        RustError::InvalidArgument(_) => "invalid_argument",
        RustError::InvalidMeta(_) => "invalid_meta",
        RustError::CorruptData { .. } => "corrupt_data",
        RustError::Path { .. } => "path",
    }
}

pub(crate) fn from_rust(err: RustError) -> PyErr {
    let kind = error_kind(&err);
    let message = err.to_string();
    let py_err = match &err {
        RustError::Io(_) => IoError::new_err(message),
        RustError::Json(_) => JsonError::new_err(message),
        RustError::DynBlosc(_) => CodecError::new_err(message),
        RustError::Zip(_) => ZipError::new_err(message),
        RustError::Allocation(_) => AllocationError::new_err(message),
        RustError::NotFound { .. } => NotFoundError::new_err(message),
        RustError::InvalidArgument(_) => InvalidArgumentError::new_err(message),
        RustError::InvalidMeta(_) => InvalidMetaError::new_err(message),
        RustError::CorruptData { .. } => CorruptDataError::new_err(message),
        RustError::Path { .. } => PathError::new_err(message),
    };
    attach_kind(py_err, kind)
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> PyErr {
    attach_kind(
        InvalidArgumentError::new_err(message.into()),
        "invalid_argument",
    )
}

fn attach_kind(err: PyErr, kind: &'static str) -> PyErr {
    Python::with_gil(|py| {
        let _set_kind_result = err.value(py).setattr("kind", kind);
        err
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ScCompressError", m.py().get_type::<ScCompressError>())?;
    m.add("IoError", m.py().get_type::<IoError>())?;
    m.add("JsonError", m.py().get_type::<JsonError>())?;
    m.add("CodecError", m.py().get_type::<CodecError>())?;
    m.add("ZipError", m.py().get_type::<ZipError>())?;
    m.add("AllocationError", m.py().get_type::<AllocationError>())?;
    m.add("NotFoundError", m.py().get_type::<NotFoundError>())?;
    m.add(
        "InvalidArgumentError",
        m.py().get_type::<InvalidArgumentError>(),
    )?;
    m.add("InvalidMetaError", m.py().get_type::<InvalidMetaError>())?;
    m.add("CorruptDataError", m.py().get_type::<CorruptDataError>())?;
    m.add("PathError", m.py().get_type::<PathError>())?;
    Ok(())
}
