//! Python session lifecycle and owned NumPy batch conversion.

use numpy::{Element, PyArray1};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use sc_load::{
    Batch, CancellationHandle, Error, OutputDType, OutputValue, RuntimeStats, Session, SessionState,
};

use crate::error::{from_rust, invalid_input as invalid_argument};
use crate::stats::{runtime_stats_to_dict, session_state_name};

struct SessionSlot {
    session: Option<Session>,
    last_stats: Option<RuntimeStats>,
    last_state: SessionState,
    exhausted: bool,
}

#[pyclass(name = "_Session", module = "scdata._core", frozen)]
pub(crate) struct PySession {
    slot: Mutex<SessionSlot>,
    cancellation: Mutex<Option<CancellationHandle>>,
}

impl PySession {
    pub(crate) fn new(session: Session) -> Self {
        let cancellation = session.cancellation_handle();
        let last_state = session.state();
        Self {
            slot: Mutex::new(SessionSlot {
                session: Some(session),
                last_stats: None,
                last_state,
                exhausted: false,
            }),
            cancellation: Mutex::new(Some(cancellation)),
        }
    }

    fn cancellation_handle(&self) -> Option<CancellationHandle> {
        self.cancellation.lock().as_ref().cloned()
    }

    fn next_owned(&self) -> Result<Option<OwnedBatch>, NextError> {
        enum Outcome {
            Batch(OwnedBatch),
            End,
            Error(Error),
        }

        let mut slot = self.slot.lock();
        let Some(session) = slot.session.as_mut() else {
            if slot.exhausted {
                return Ok(None);
            }
            return Err(NextError::Closed);
        };
        let outcome = match session.next_batch().map_err(NextError::Rust)? {
            Some(batch) => {
                let copied = OwnedBatch::copy(&batch);
                drop(batch);
                match copied {
                    Ok(batch) => Outcome::Batch(batch),
                    Err(error) => Outcome::Error(error),
                }
            }
            None => Outcome::End,
        };
        match outcome {
            Outcome::Batch(batch) => Ok(Some(batch)),
            Outcome::Error(error) => {
                drop(slot);
                if let Some(cancellation) = self.cancellation_handle() {
                    cancellation.cancel();
                }
                Err(NextError::Rust(error))
            }
            Outcome::End => {
                if let Some((stats, state)) = slot
                    .session
                    .as_ref()
                    .map(|session| (session.stats(), session.state()))
                {
                    slot.last_stats = Some(stats);
                    slot.last_state = state;
                }
                slot.exhausted = true;
                let finished = slot.session.take();
                drop(slot);
                let cancellation = self.cancellation.lock().take();
                drop(finished);
                drop(cancellation);
                Ok(None)
            }
        }
    }
}

#[pyfunction]
pub(crate) fn session_next<'py>(
    py: Python<'py>,
    session: &PySession,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let batch = py.allow_threads(|| session.next_owned());
    match batch {
        Ok(Some(batch)) => batch.into_python(py).map(Some),
        Ok(None) => Ok(None),
        Err(NextError::Rust(error)) => Err(from_rust(error)),
        Err(NextError::Closed) => Err(invalid_argument("session is closed")),
    }
}

#[pyfunction]
pub(crate) fn session_cancel(session: &PySession) {
    if let Some(cancellation) = session.cancellation_handle() {
        cancellation.cancel();
    }
}

#[pyfunction]
pub(crate) fn session_close(py: Python<'_>, session: &PySession) {
    let cancellation = session.cancellation.lock().take();
    if let Some(cancellation) = cancellation.as_ref() {
        cancellation.cancel();
    }
    let dropped = {
        let mut slot = session.slot.lock();
        if let Some((stats, state)) = slot
            .session
            .as_ref()
            .map(|session| (session.stats(), session.state()))
        {
            slot.last_stats = Some(stats);
            slot.last_state = state;
        }
        slot.session.take()
    };
    py.allow_threads(move || drop((dropped, cancellation)));
}

#[pyfunction]
pub(crate) fn session_meta<'py>(
    py: Python<'py>,
    session: &PySession,
) -> PyResult<Bound<'py, PyDict>> {
    let slot = session.slot.lock();
    let values = PyDict::new(py);
    values.set_item("closed", slot.session.is_none())?;
    values.set_item("exhausted", slot.exhausted)?;
    let state = slot
        .session
        .as_ref()
        .map(Session::state)
        .unwrap_or(slot.last_state);
    values.set_item("state", session_state_name(state))?;
    Ok(values)
}

#[pyfunction]
pub(crate) fn session_stats<'py>(
    py: Python<'py>,
    session: &PySession,
) -> PyResult<Bound<'py, PyDict>> {
    let stats = {
        let slot = session.slot.lock();
        slot.session
            .as_ref()
            .map(Session::stats)
            .or_else(|| slot.last_stats.clone())
    }
    .ok_or_else(|| invalid_argument("session statistics are unavailable"))?;
    runtime_stats_to_dict(py, &stats)
}

impl Drop for PySession {
    fn drop(&mut self) {
        let cancellation = self.cancellation.get_mut().take();
        if let Some(cancellation) = cancellation.as_ref() {
            cancellation.cancel();
        }
        let session = self.slot.get_mut().session.take();
        drop((session, cancellation));
    }
}

enum NextError {
    Rust(Error),
    Closed,
}

struct OwnedBatch {
    rows: usize,
    cols: usize,
    values: BatchValues,
}

enum BatchValues {
    I16(Vec<i16>),
    I32(Vec<i32>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl OwnedBatch {
    fn copy(batch: &Batch<'_>) -> Result<Self, Error> {
        let rows = batch.rows();
        let cols = batch.n_cols();
        let values = match batch.dtype() {
            OutputDType::I16 => BatchValues::I16(copy_values::<i16>(batch)?),
            OutputDType::I32 => BatchValues::I32(copy_values::<i32>(batch)?),
            OutputDType::U16 => BatchValues::U16(copy_values::<u16>(batch)?),
            OutputDType::U32 => BatchValues::U32(copy_values::<u32>(batch)?),
            OutputDType::F32 => BatchValues::F32(copy_values::<f32>(batch)?),
            OutputDType::F64 => BatchValues::F64(copy_values::<f64>(batch)?),
        };
        Ok(Self { rows, cols, values })
    }

    fn into_python<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.values {
            BatchValues::I16(values) => vec_to_array(py, values, self.rows, self.cols),
            BatchValues::I32(values) => vec_to_array(py, values, self.rows, self.cols),
            BatchValues::U16(values) => vec_to_array(py, values, self.rows, self.cols),
            BatchValues::U32(values) => vec_to_array(py, values, self.rows, self.cols),
            BatchValues::F32(values) => vec_to_array(py, values, self.rows, self.cols),
            BatchValues::F64(values) => vec_to_array(py, values, self.rows, self.cols),
        }
    }
}

fn copy_values<T>(batch: &Batch<'_>) -> Result<Vec<T>, Error>
where
    T: OutputValue + Element,
{
    let count = batch
        .rows()
        .checked_mul(batch.n_cols())
        .ok_or_else(|| Error::Allocation("batch element count overflow".into()))?;
    let mut values = Vec::new();
    values.try_reserve_exact(count)?;
    let row_bytes = batch
        .n_cols()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::Allocation("batch row byte count overflow".into()))?;
    if batch.row_stride_bytes() == row_bytes {
        values.extend_from_slice(batch.as_slice::<T>()?);
    } else {
        for row in 0..batch.rows() {
            values.extend_from_slice(batch.row_as::<T>(row)?);
        }
    }
    Ok(values)
}

fn vec_to_array<'py, T: Element>(
    py: Python<'py>,
    values: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    PyArray1::from_vec(py, values).call_method1("reshape", (rows, cols))
}
