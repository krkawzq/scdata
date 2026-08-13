//! Python session lifecycle and owned NumPy batch conversion.

use std::mem::ManuallyDrop;

use numpy::{Element, PyArray1, PyArrayMethods};
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
    slot: ManuallyDrop<Mutex<SessionSlot>>,
    cancellation: ManuallyDrop<Mutex<Option<CancellationHandle>>>,
    process_id: u32,
}

impl PySession {
    pub(crate) fn new(session: Session) -> Self {
        let cancellation = session.cancellation_handle();
        let last_state = session.state();
        Self {
            slot: ManuallyDrop::new(Mutex::new(SessionSlot {
                session: Some(session),
                last_stats: None,
                last_state,
                exhausted: false,
            })),
            cancellation: ManuallyDrop::new(Mutex::new(Some(cancellation))),
            process_id: std::process::id(),
        }
    }

    fn ensure_process(&self) -> PyResult<()> {
        let current = std::process::id();
        if current != self.process_id {
            return Err(invalid_argument(format!(
                "session was opened in process {}, but is being used in process {current}",
                self.process_id
            )));
        }
        Ok(())
    }

    fn cancellation_handle(&self) -> Option<CancellationHandle> {
        self.cancellation.lock().as_ref().cloned()
    }

    fn cancel(&self) {
        if let Some(cancellation) = self.cancellation_handle() {
            cancellation.cancel();
        }
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
        let outcome = match session.next_batch() {
            Ok(Some(batch)) => {
                let copied = OwnedBatch::copy(&batch);
                drop(batch);
                match copied {
                    Ok(batch) => Outcome::Batch(batch),
                    Err(error) => Outcome::Error(error),
                }
            }
            Ok(None) => Outcome::End,
            Err(error) => Outcome::Error(error),
        };
        match outcome {
            Outcome::Batch(batch) => Ok(Some(batch)),
            Outcome::Error(error) => {
                drop(slot);
                self.cancel();
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
    session.ensure_process()?;
    let batch = py.allow_threads(|| session.next_owned());
    match batch {
        Ok(Some(batch)) => match batch.into_python(py) {
            Ok(batch) => Ok(Some(batch)),
            Err(error) => {
                session.cancel();
                Err(error)
            }
        },
        Ok(None) => Ok(None),
        Err(NextError::Rust(error)) => Err(from_rust(error)),
        Err(NextError::Closed) => Err(invalid_argument("session is closed")),
    }
}

#[pyfunction]
pub(crate) fn session_cancel(session: &PySession) -> PyResult<()> {
    session.ensure_process()?;
    session.cancel();
    Ok(())
}

#[pyfunction]
pub(crate) fn session_close(py: Python<'_>, session: &PySession) -> PyResult<()> {
    session.ensure_process()?;
    py.allow_threads(|| {
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
        drop((dropped, cancellation));
    });
    Ok(())
}

#[pyfunction]
pub(crate) fn session_meta<'py>(
    py: Python<'py>,
    session: &PySession,
) -> PyResult<Bound<'py, PyDict>> {
    session.ensure_process()?;
    let (closed, exhausted, state) = py.allow_threads(|| {
        let slot = session.slot.lock();
        let state = slot
            .session
            .as_ref()
            .map(Session::state)
            .unwrap_or(slot.last_state);
        (slot.session.is_none(), slot.exhausted, state)
    });
    let values = PyDict::new(py);
    values.set_item("closed", closed)?;
    values.set_item("exhausted", exhausted)?;
    values.set_item("state", session_state_name(state))?;
    Ok(values)
}

#[pyfunction]
pub(crate) fn session_stats<'py>(
    py: Python<'py>,
    session: &PySession,
) -> PyResult<Bound<'py, PyDict>> {
    session.ensure_process()?;
    let stats = py
        .allow_threads(|| {
            let slot = session.slot.lock();
            slot.session
                .as_ref()
                .map(Session::stats)
                .or_else(|| slot.last_stats.clone())
        })
        .ok_or_else(|| invalid_argument("session statistics are unavailable"))?;
    runtime_stats_to_dict(py, &stats)
}

impl Drop for PySession {
    fn drop(&mut self) {
        if std::process::id() != self.process_id {
            return;
        }
        let cancellation = self.cancellation.get_mut().take();
        if let Some(cancellation) = cancellation.as_ref() {
            cancellation.cancel();
        }
        let session = self.slot.get_mut().session.take();
        drop((session, cancellation));
        // SAFETY: only the creating process destroys these fields. A post-fork
        // child leaves the copied thread handles untouched until process exit.
        unsafe {
            ManuallyDrop::drop(&mut self.slot);
            ManuallyDrop::drop(&mut self.cancellation);
        }
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
    I64(Vec<i64>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
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
            OutputDType::I64 => BatchValues::I64(copy_values::<i64>(batch)?),
            OutputDType::U16 => BatchValues::U16(copy_values::<u16>(batch)?),
            OutputDType::U32 => BatchValues::U32(copy_values::<u32>(batch)?),
            OutputDType::U64 => BatchValues::U64(copy_values::<u64>(batch)?),
            OutputDType::F32 => BatchValues::F32(copy_values::<f32>(batch)?),
            OutputDType::F64 => BatchValues::F64(copy_values::<f64>(batch)?),
        };
        Ok(Self { rows, cols, values })
    }

    fn into_python<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.values {
            BatchValues::I16(values) => vec_to_array(py, values, self.rows, self.cols),
            BatchValues::I32(values) => vec_to_array(py, values, self.rows, self.cols),
            BatchValues::I64(values) => vec_to_array(py, values, self.rows, self.cols),
            BatchValues::U16(values) => vec_to_array(py, values, self.rows, self.cols),
            BatchValues::U32(values) => vec_to_array(py, values, self.rows, self.cols),
            BatchValues::U64(values) => vec_to_array(py, values, self.rows, self.cols),
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
    PyArray1::from_vec(py, values)
        .reshape([rows, cols])
        .map(|array| array.into_any())
}
