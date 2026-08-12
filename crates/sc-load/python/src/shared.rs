//! Linux shared-ring bindings. NumPy arrays borrow read-only ring generations.

use std::os::fd::{AsFd, BorrowedFd, IntoRawFd, OwnedFd};
use std::sync::atomic::{AtomicU8, Ordering};

use numpy::ndarray::{ArrayView2, ShapeBuilder};
use numpy::{Element, PyArray2, PyArrayMethods};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use sc_load::{
    Error, OutputDType, SessionState, SharedBatch, SharedCancellationHandle, SharedClient,
    SharedClientCancellationHandle, SharedServer,
};

use crate::error::{from_rust, invalid_argument, ResultExt};
use crate::stats::session_state_name;

const RUNNING: u8 = 0;
const FAILED: u8 = 1;
const CANCELLED: u8 = 2;
const FINISHED: u8 = 3;
const COPY_WITHOUT_GIL_THRESHOLD: usize = 64 * 1024;

#[derive(Clone, Copy)]
struct SharedMetadata {
    world_size: usize,
    n_rows: usize,
    n_cols: usize,
    batch_size: usize,
    batch_count: usize,
    row_stride: usize,
    dtype: OutputDType,
}

#[pyclass(name = "_SharedServer", module = "sc_load._core")]
pub(crate) struct PySharedServer {
    server: Mutex<Option<SharedServer>>,
    cancellation: Mutex<Option<SharedCancellationHandle>>,
    descriptor: OwnedFd,
    metadata: SharedMetadata,
    final_state: AtomicU8,
    process_id: u32,
}

impl PySharedServer {
    pub(crate) fn new(server: SharedServer) -> Result<Self, Error> {
        let descriptor = server.attach_fd()?;
        let cancellation = server.cancellation_handle();
        let final_state = state_code(cancellation.state());
        let metadata = SharedMetadata {
            world_size: server.world_size(),
            n_rows: server.n_rows(),
            n_cols: server.n_cols(),
            batch_size: server.batch_size(),
            batch_count: server.batch_count(),
            row_stride: server.row_stride_bytes(),
            dtype: server.dtype(),
        };
        Ok(Self {
            server: Mutex::new(Some(server)),
            cancellation: Mutex::new(Some(cancellation)),
            descriptor,
            metadata,
            final_state: AtomicU8::new(final_state),
            process_id: std::process::id(),
        })
    }

    fn cancellation_handle(&self) -> Option<SharedCancellationHandle> {
        self.cancellation.lock().as_ref().cloned()
    }

    fn ensure_process(&self) -> PyResult<()> {
        let current = std::process::id();
        if current != self.process_id {
            return Err(invalid_argument(format!(
                "shared producer was opened in process {}, but is being used in process {current}",
                self.process_id
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PySharedServer {
    fn run(&self, py: Python<'_>) -> PyResult<()> {
        self.ensure_process()?;
        let server = self
            .server
            .lock()
            .take()
            .ok_or_else(|| invalid_argument("shared producer has already been run"))?;
        let result = py.allow_threads(move || server.run());
        let state = match &result {
            Ok(()) => SessionState::Finished,
            Err(Error::Cancelled) => SessionState::Cancelled,
            Err(_) => SessionState::Failed,
        };
        self.final_state.store(state_code(state), Ordering::Release);
        let cancellation = self.cancellation.lock().take();
        drop(cancellation);
        result.map_sc()
    }

    fn cancel(&self) {
        if std::process::id() != self.process_id {
            return;
        }
        if let Some(cancellation) = self.cancellation_handle() {
            cancellation.cancel();
        }
    }

    fn duplicate_fd(&self) -> PyResult<i32> {
        self.ensure_process()?;
        let descriptor = self
            .descriptor
            .as_fd()
            .try_clone_to_owned()
            .map_err(|error| from_rust(error.into()))?;
        Ok(descriptor.into_raw_fd())
    }

    #[getter]
    fn world_size(&self) -> usize {
        self.metadata.world_size
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.metadata.n_rows
    }

    #[getter]
    fn n_cols(&self) -> usize {
        self.metadata.n_cols
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.metadata.batch_size
    }

    #[getter]
    fn batch_count(&self) -> usize {
        self.metadata.batch_count
    }

    #[getter]
    fn row_stride_bytes(&self) -> usize {
        self.metadata.row_stride
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.metadata.dtype.as_str()
    }

    #[getter]
    fn state(&self) -> &'static str {
        if std::process::id() != self.process_id {
            return state_name(self.final_state.load(Ordering::Acquire));
        }
        if let Some(cancellation) = self.cancellation_handle() {
            session_state_name(cancellation.state())
        } else {
            state_name(self.final_state.load(Ordering::Acquire))
        }
    }
}

impl Drop for PySharedServer {
    fn drop(&mut self) {
        if std::process::id() != self.process_id {
            return;
        }
        let cancellation = self.cancellation.get_mut().take();
        if let Some(cancellation) = cancellation.as_ref() {
            cancellation.cancel();
        }
        let server = self.server.get_mut().take();
        drop((server, cancellation));
    }
}

struct SharedClientSlot {
    client: Option<SharedClient>,
    exhausted: bool,
}

#[pyclass(name = "_SharedClient", module = "sc_load._core")]
pub(crate) struct PySharedClient {
    slot: Mutex<SharedClientSlot>,
    cancellation: SharedClientCancellationHandle,
    metadata: SharedMetadata,
    rank: usize,
    rank_batch_count: usize,
    process_id: u32,
}

impl PySharedClient {
    fn new(client: SharedClient) -> Self {
        let cancellation = client.cancellation_handle();
        let metadata = SharedMetadata {
            world_size: client.world_size(),
            n_rows: client.n_rows(),
            n_cols: client.n_cols(),
            batch_size: client.batch_size(),
            batch_count: 0,
            row_stride: 0,
            dtype: client.dtype(),
        };
        let rank = client.rank();
        let rank_batch_count = client.batch_count();
        Self {
            slot: Mutex::new(SharedClientSlot {
                client: Some(client),
                exhausted: false,
            }),
            cancellation,
            metadata,
            rank,
            rank_batch_count,
            process_id: std::process::id(),
        }
    }

    fn ensure_process(&self) -> PyResult<()> {
        let current = std::process::id();
        if current != self.process_id {
            return Err(invalid_argument(format!(
                "shared client was attached in process {}, but is being used in process {current}; attach after forking",
                self.process_id
            )));
        }
        Ok(())
    }

    fn next_owned(&self) -> Result<Option<SharedBatch>, NextError> {
        debug_assert_eq!(std::process::id(), self.process_id);
        let mut slot = self.slot.lock();
        let Some(client) = slot.client.as_mut() else {
            return if slot.exhausted {
                Ok(None)
            } else {
                Err(NextError::Closed)
            };
        };
        match client.next_batch().map_err(NextError::Rust)? {
            Some(batch) => Ok(Some(batch)),
            None => {
                slot.exhausted = true;
                let client = slot.client.take();
                drop(slot);
                drop(client);
                Ok(None)
            }
        }
    }

    fn drain_into<T: Copy + Element>(
        &self,
        destination: usize,
        expected_rows: usize,
    ) -> Result<usize, NextError> {
        debug_assert_eq!(std::process::id(), self.process_id);
        let mut slot = self.slot.lock();
        let Some(client) = slot.client.as_mut() else {
            return if slot.exhausted && expected_rows == 0 {
                Ok(0)
            } else {
                Err(NextError::Closed)
            };
        };
        let mut rows = 0usize;
        let mut batches = 0usize;
        loop {
            let Some(batch) = client.next_batch().map_err(NextError::Rust)? else {
                slot.exhausted = true;
                let client = slot.client.take();
                drop(slot);
                drop(client);
                if rows != expected_rows {
                    return Err(NextError::Rust(Error::Invariant(format!(
                        "shared rank returned {rows} rows, expected {expected_rows}"
                    ))));
                }
                return Ok(batches);
            };
            let layout = compact_copy_layout::<T>(&batch).map_err(NextError::Rust)?;
            if layout.cols != self.metadata.n_cols || layout.rows == 0 {
                return Err(NextError::Rust(Error::Invariant(
                    "shared batch shape does not match client metadata".into(),
                )));
            }
            let next_rows = rows.checked_add(layout.rows).ok_or_else(|| {
                NextError::Rust(Error::Allocation("shared row count overflow".into()))
            })?;
            if next_rows > expected_rows {
                return Err(NextError::Rust(Error::Invariant(format!(
                    "shared rank produced {next_rows} rows, exceeding expected {expected_rows}"
                ))));
            }
            if layout.copy_bytes != 0 {
                let destination_offset = rows.checked_mul(layout.row_bytes).ok_or_else(|| {
                    NextError::Rust(Error::Allocation(
                        "shared destination byte offset overflow".into(),
                    ))
                })?;
                let destination = destination.checked_add(destination_offset).ok_or_else(|| {
                    NextError::Rust(Error::Allocation(
                        "shared destination pointer overflow".into(),
                    ))
                })?;
                // SAFETY: the NumPy allocation is unexposed while the GIL is
                // released, and the expected row count bounds every destination.
                unsafe {
                    copy_compact_rows(
                        layout.source,
                        destination,
                        layout.rows,
                        layout.row_stride,
                        layout.row_bytes,
                        layout.copy_bytes,
                    );
                }
            }
            rows = next_rows;
            batches = batches.checked_add(1).ok_or_else(|| {
                NextError::Rust(Error::Allocation("shared batch count overflow".into()))
            })?;
            drop(batch);
        }
    }
}

#[pymethods]
impl PySharedClient {
    fn next_batch<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.ensure_process()?;
        match py.allow_threads(|| self.next_owned()) {
            Ok(Some(batch)) => batch_into_array(py, batch).map(Some),
            Ok(None) => Ok(None),
            Err(NextError::Rust(error)) => Err(from_rust(error)),
            Err(NextError::Closed) => Err(invalid_argument("shared client is closed")),
        }
    }

    fn next_batch_copy<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.ensure_process()?;
        match py.allow_threads(|| self.next_owned()) {
            Ok(Some(batch)) => batch_into_owned_array(py, batch).map(Some),
            Ok(None) => Ok(None),
            Err(NextError::Rust(error)) => Err(from_rust(error)),
            Err(NextError::Closed) => Err(invalid_argument("shared client is closed")),
        }
    }

    fn read<'py>(
        &self,
        py: Python<'py>,
        expected_rows: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        self.ensure_process()?;
        if expected_rows > self.metadata.n_rows {
            return Err(invalid_argument(format!(
                "shared read expects {expected_rows} rows, exceeding dataset rows {}",
                self.metadata.n_rows
            )));
        }
        #[cfg(target_endian = "big")]
        {
            let _ = (py, expected_rows);
            return Err(from_rust(Error::Unsupported(
                "shared NumPy copies require a little-endian target".into(),
            )));
        }
        #[cfg(target_endian = "little")]
        match self.metadata.dtype {
            OutputDType::I16 => typed_read_array::<i16>(py, self, expected_rows),
            OutputDType::I32 => typed_read_array::<i32>(py, self, expected_rows),
            OutputDType::U16 => typed_read_array::<u16>(py, self, expected_rows),
            OutputDType::U32 => typed_read_array::<u32>(py, self, expected_rows),
            OutputDType::F32 => typed_read_array::<f32>(py, self, expected_rows),
            OutputDType::F64 => typed_read_array::<f64>(py, self, expected_rows),
        }
    }

    fn close(&self, py: Python<'_>) {
        if std::process::id() != self.process_id {
            return;
        }
        self.cancellation.cancel_if_incomplete();
        let client = self.slot.lock().client.take();
        py.allow_threads(move || drop(client));
    }

    #[getter]
    fn rank(&self) -> usize {
        self.rank
    }

    #[getter]
    fn world_size(&self) -> usize {
        self.metadata.world_size
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.metadata.n_rows
    }

    #[getter]
    fn n_cols(&self) -> usize {
        self.metadata.n_cols
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.metadata.batch_size
    }

    #[getter]
    fn batch_count(&self) -> usize {
        self.rank_batch_count
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.metadata.dtype.as_str()
    }

    #[getter]
    fn closed(&self) -> PyResult<bool> {
        self.ensure_process()?;
        Ok(self.slot.lock().client.is_none())
    }

    #[getter]
    fn exhausted(&self) -> PyResult<bool> {
        self.ensure_process()?;
        Ok(self.slot.lock().exhausted)
    }

    #[getter]
    fn next_logical_batch(&self) -> PyResult<Option<usize>> {
        self.ensure_process()?;
        Ok(self
            .slot
            .lock()
            .client
            .as_ref()
            .and_then(SharedClient::next_logical_batch))
    }
}

impl Drop for PySharedClient {
    fn drop(&mut self) {
        if std::process::id() != self.process_id {
            return;
        }
        let client = self.slot.get_mut().client.take();
        drop(client);
    }
}

enum NextError {
    Rust(Error),
    Closed,
}

#[pyclass(name = "_SharedBatch", module = "sc_load._core", frozen)]
struct PySharedBatch {
    batch: SharedBatch,
}

#[pymethods]
impl PySharedBatch {
    #[getter]
    fn logical_batch(&self) -> usize {
        self.batch.logical_batch()
    }

    #[getter]
    fn rows(&self) -> usize {
        self.batch.rows()
    }
}

fn batch_into_array<'py>(py: Python<'py>, batch: SharedBatch) -> PyResult<Bound<'py, PyAny>> {
    let owner = Bound::new(py, PySharedBatch { batch })?;
    let (pointer, rows, cols, row_stride, dtype) = {
        let owner_ref = owner.borrow();
        let bytes = owner_ref.batch.bytes().map_sc()?;
        (
            bytes.as_ptr(),
            owner_ref.batch.rows(),
            owner_ref.batch.n_cols(),
            owner_ref.batch.row_stride_bytes(),
            owner_ref.batch.dtype(),
        )
    };
    #[cfg(target_endian = "big")]
    {
        let _ = (pointer, rows, cols, row_stride, dtype, owner);
        return Err(from_rust(Error::Unsupported(
            "shared NumPy views require a little-endian target".into(),
        )));
    }
    #[cfg(target_endian = "little")]
    match dtype {
        OutputDType::I16 => typed_array::<i16>(owner, pointer, rows, cols, row_stride),
        OutputDType::I32 => typed_array::<i32>(owner, pointer, rows, cols, row_stride),
        OutputDType::U16 => typed_array::<u16>(owner, pointer, rows, cols, row_stride),
        OutputDType::U32 => typed_array::<u32>(owner, pointer, rows, cols, row_stride),
        OutputDType::F32 => typed_array::<f32>(owner, pointer, rows, cols, row_stride),
        OutputDType::F64 => typed_array::<f64>(owner, pointer, rows, cols, row_stride),
    }
}

fn batch_into_owned_array<'py>(py: Python<'py>, batch: SharedBatch) -> PyResult<Bound<'py, PyAny>> {
    #[cfg(target_endian = "big")]
    {
        let _ = (py, batch);
        return Err(from_rust(Error::Unsupported(
            "shared NumPy copies require a little-endian target".into(),
        )));
    }
    #[cfg(target_endian = "little")]
    match batch.dtype() {
        OutputDType::I16 => typed_owned_array::<i16>(py, batch),
        OutputDType::I32 => typed_owned_array::<i32>(py, batch),
        OutputDType::U16 => typed_owned_array::<u16>(py, batch),
        OutputDType::U32 => typed_owned_array::<u32>(py, batch),
        OutputDType::F32 => typed_owned_array::<f32>(py, batch),
        OutputDType::F64 => typed_owned_array::<f64>(py, batch),
    }
}

#[cfg(target_endian = "little")]
#[derive(Clone, Copy)]
struct CompactCopyLayout {
    source: usize,
    rows: usize,
    cols: usize,
    row_stride: usize,
    row_bytes: usize,
    copy_bytes: usize,
}

#[cfg(target_endian = "little")]
fn compact_copy_layout<T: Element>(batch: &SharedBatch) -> Result<CompactCopyLayout, Error> {
    let rows = batch.rows();
    let cols = batch.n_cols();
    let row_stride = batch.row_stride_bytes();
    let element_size = std::mem::size_of::<T>();
    if element_size == 0 {
        return Err(Error::Invariant(
            "shared output dtype has zero-sized elements".into(),
        ));
    }
    let row_bytes = cols
        .checked_mul(element_size)
        .ok_or_else(|| Error::Allocation("shared row byte count overflow".into()))?;
    if row_stride < row_bytes {
        return Err(Error::Invariant(format!(
            "shared row stride {row_stride} is smaller than {row_bytes} payload bytes"
        )));
    }
    let copy_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::Allocation("shared batch byte count overflow".into()))?;
    let bytes = batch.bytes()?;
    let required = rows
        .checked_mul(row_stride)
        .ok_or_else(|| Error::Allocation("shared strided batch byte count overflow".into()))?;
    if bytes.len() < required {
        return Err(Error::Invariant(format!(
            "shared batch exposes {} bytes, expected at least {required}",
            bytes.len()
        )));
    }
    Ok(CompactCopyLayout {
        source: bytes.as_ptr() as usize,
        rows,
        cols,
        row_stride,
        row_bytes,
        copy_bytes,
    })
}

#[cfg(target_endian = "little")]
fn typed_owned_array<'py, T: Copy + Element>(
    py: Python<'py>,
    batch: SharedBatch,
) -> PyResult<Bound<'py, PyAny>> {
    let layout = compact_copy_layout::<T>(&batch).map_sc()?;
    // SAFETY: all supported output dtypes are trivially copyable, and every
    // element is initialized below before the array becomes visible to Python.
    let array = unsafe { PyArray2::<T>::new(py, [layout.rows, layout.cols], false) };
    let destination = array.data().cast::<u8>() as usize;
    if layout.copy_bytes != 0 {
        if layout.copy_bytes >= COPY_WITHOUT_GIL_THRESHOLD {
            copy_compact_rows_without_gil(py, layout, destination);
        } else {
            // SAFETY: as above; small copies stay under the GIL to avoid another
            // thread-state transition for latency-sensitive small batches.
            unsafe {
                copy_compact_rows(
                    layout.source,
                    destination,
                    layout.rows,
                    layout.row_stride,
                    layout.row_bytes,
                    layout.copy_bytes,
                );
            }
        }
    }
    drop(batch);
    Ok(array.into_any())
}

#[cfg(target_endian = "little")]
#[inline(never)]
fn copy_compact_rows_without_gil(py: Python<'_>, layout: CompactCopyLayout, destination: usize) {
    py.allow_threads(move || {
        // SAFETY: the source lease and the unexposed NumPy allocation stay
        // live for this synchronous copy, and their regions never overlap.
        unsafe {
            copy_compact_rows(
                layout.source,
                destination,
                layout.rows,
                layout.row_stride,
                layout.row_bytes,
                layout.copy_bytes,
            );
        }
    });
}

#[cfg(target_endian = "little")]
fn typed_read_array<'py, T: Copy + Element>(
    py: Python<'py>,
    client: &PySharedClient,
    expected_rows: usize,
) -> PyResult<(Bound<'py, PyAny>, usize)> {
    // SAFETY: all supported output dtypes are trivially copyable. The array is
    // not exposed to Python until `drain_into` initializes every element.
    let array = unsafe { PyArray2::<T>::new(py, [expected_rows, client.metadata.n_cols], false) };
    let destination = array.data().cast::<u8>() as usize;
    let batches = match py.allow_threads(|| client.drain_into::<T>(destination, expected_rows)) {
        Ok(batches) => batches,
        Err(NextError::Rust(error)) => return Err(from_rust(error)),
        Err(NextError::Closed) => return Err(invalid_argument("shared client is closed")),
    };
    Ok((array.into_any(), batches))
}

#[cfg(target_endian = "little")]
unsafe fn copy_compact_rows(
    source: usize,
    destination: usize,
    rows: usize,
    row_stride: usize,
    row_bytes: usize,
    copy_bytes: usize,
) {
    let source = source as *const u8;
    let destination = destination as *mut u8;
    if row_stride == row_bytes {
        debug_assert_eq!(copy_bytes, rows * row_bytes);
        // SAFETY: the caller validated both allocations for `copy_bytes`.
        unsafe {
            std::ptr::copy_nonoverlapping(source, destination, copy_bytes);
        }
        return;
    }
    for row in 0..rows {
        // SAFETY: the caller validated the strided source and compact target.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.add(row * row_stride),
                destination.add(row * row_bytes),
                row_bytes,
            );
        }
    }
}

#[cfg(target_endian = "little")]
fn typed_array<'py, T: Element>(
    owner: Bound<'py, PySharedBatch>,
    pointer: *const u8,
    rows: usize,
    cols: usize,
    row_stride: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let element_size = std::mem::size_of::<T>();
    if element_size == 0
        || !row_stride.is_multiple_of(element_size)
        || pointer.align_offset(std::mem::align_of::<T>()) != 0
    {
        return Err(from_rust(Error::Invariant(
            "shared NumPy view has an incompatible element layout".into(),
        )));
    }
    let shape = (rows, cols).strides((row_stride / element_size, 1));
    // SAFETY: `SharedBatch` owns a generation lease and its mapping. The array
    // base is that Python owner, so the pointer remains live and immutable until
    // every derived NumPy view is dropped. Shape and strides stay in bounds of
    // the validated batch extent.
    let view = unsafe { ArrayView2::from_shape_ptr(shape, pointer.cast::<T>()) };
    // SAFETY: the view cannot be reallocated, and `owner` becomes the NumPy base.
    let array = unsafe { PyArray2::borrow_from_array(&view, owner.into_any()) };
    let _readonly = array.readwrite().make_nonwriteable();
    Ok(array.into_any())
}

#[pyfunction(name = "_attach_shared")]
pub(crate) fn attach_shared(py: Python<'_>, fd: i32, rank: usize) -> PyResult<PySharedClient> {
    if fd < 0 {
        return Err(invalid_argument(
            "shared file descriptor must be non-negative",
        ));
    }
    // SAFETY: Python retains ownership of `fd`; it is borrowed only long enough
    // to duplicate it before the GIL is released.
    let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
    let descriptor = borrowed
        .try_clone_to_owned()
        .map_err(|error| from_rust(error.into()))?;
    let client = py
        .allow_threads(move || SharedClient::attach(descriptor.as_fd(), rank))
        .map_sc()?;
    Ok(PySharedClient::new(client))
}

fn state_code(state: SessionState) -> u8 {
    match state {
        SessionState::Running => RUNNING,
        SessionState::Failed => FAILED,
        SessionState::Cancelled => CANCELLED,
        SessionState::Finished => FINISHED,
    }
}

fn state_name(state: u8) -> &'static str {
    match state {
        RUNNING => "running",
        FAILED => "failed",
        CANCELLED => "cancelled",
        FINISHED => "finished",
        _ => "failed",
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "DEFAULT_MAX_SHARED_CONTROL_BYTES",
        sc_load::DEFAULT_MAX_SHARED_CONTROL_BYTES,
    )?;
    module.add_class::<PySharedServer>()?;
    module.add_class::<PySharedClient>()?;
    module.add_class::<PySharedBatch>()?;
    module.add_function(wrap_pyfunction!(attach_shared, module)?)?;
    Ok(())
}
