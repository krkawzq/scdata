use pyo3::create_exception;
use pyo3::prelude::*;

use crate::databank::DataBankError as RustDataBankError;

create_exception!(_scdata, DataBankError, pyo3::exceptions::PyRuntimeError);

impl From<RustDataBankError> for PyErr {
    fn from(err: RustDataBankError) -> Self {
        DataBankError::new_err(err.to_string())
    }
}
