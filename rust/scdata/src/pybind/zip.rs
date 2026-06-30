use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_zip_stored_offsets, m)?)?;
    Ok(())
}

#[pyfunction]
fn _zip_stored_offsets(path: String, header_offsets: Vec<u64>) -> PyResult<Vec<u64>> {
    let mut file = File::open(&path)
        .map_err(|err| PyOSError::new_err(format!("cannot open zip archive {path}: {err}")))?;
    let mut out = Vec::with_capacity(header_offsets.len());
    let mut header = [0u8; 30];
    for offset in header_offsets {
        file.seek(SeekFrom::Start(offset)).map_err(|err| {
            PyOSError::new_err(format!("cannot seek zip local header at {offset}: {err}"))
        })?;
        file.read_exact(&mut header).map_err(|err| {
            PyOSError::new_err(format!("cannot read zip local header at {offset}: {err}"))
        })?;
        if &header[..4] != b"PK\x03\x04" {
            return Err(PyValueError::new_err(format!(
                "invalid zip local header at {offset}"
            )));
        }
        let filename_len = u16::from_le_bytes([header[26], header[27]]) as u64;
        let extra_len = u16::from_le_bytes([header[28], header[29]]) as u64;
        out.push(offset + 30 + filename_len + extra_len);
    }
    Ok(out)
}
