//! Linux shared-ring bindings. NumPy arrays borrow read-only ring generations.

use std::mem::ManuallyDrop;
use std::os::fd::{AsFd, BorrowedFd, IntoRawFd, OwnedFd};
use std::sync::atomic::{AtomicU8, Ordering};

use numpy::ndarray::{ArrayView2, ShapeBuilder};
use numpy::{Element, PyArray2, PyArrayMethods};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use sc_load::{
    Error, OutputDType, SessionState, SharedBatch, SharedCancellationHandle, SharedClient,
    SharedClientCancellationHandle, SharedServer,
};

use crate::error::{from_rust, invalid_input as invalid_argument, ResultExt};
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

#[pyclass(name = "_SharedServer", module = "scdata._core", frozen)]
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

#[pyfunction]
pub(crate) fn shared_run(py: Python<'_>, server: &PySharedServer) -> PyResult<()> {
    server.ensure_process()?;
    let inner = server
        .server
        .lock()
        .take()
        .ok_or_else(|| invalid_argument("shared producer has already been run"))?;
    let result = py.allow_threads(move || inner.run());
    let state = match &result {
        Ok(()) => SessionState::Finished,
        Err(Error::Cancelled) => SessionState::Cancelled,
        Err(_) => SessionState::Failed,
    };
    server
        .final_state
        .store(state_code(state), Ordering::Release);
    let cancellation = server.cancellation.lock().take();
    drop(cancellation);
    result.map_sc()
}

#[pyfunction]
pub(crate) fn shared_cancel(server: &PySharedServer) {
    if std::process::id() != server.process_id {
        return;
    }
    if let Some(cancellation) = server.cancellation_handle() {
        cancellation.cancel();
    }
}

#[pyfunction]
pub(crate) fn shared_duplicate_fd(server: &PySharedServer) -> PyResult<i32> {
    server.ensure_process()?;
    let descriptor = server
        .descriptor
        .as_fd()
        .try_clone_to_owned()
        .map_err(|error| from_rust(error.into()))?;
    Ok(descriptor.into_raw_fd())
}

#[pyfunction]
pub(crate) fn shared_server_meta<'py>(
    py: Python<'py>,
    server: &PySharedServer,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    values.set_item("world_size", server.metadata.world_size)?;
    values.set_item("n_rows", server.metadata.n_rows)?;
    values.set_item("n_cols", server.metadata.n_cols)?;
    values.set_item("batch_size", server.metadata.batch_size)?;
    values.set_item("batch_count", server.metadata.batch_count)?;
    values.set_item("row_stride_bytes", server.metadata.row_stride)?;
    values.set_item("dtype", server.metadata.dtype.as_str())?;
    let state = if std::process::id() != server.process_id {
        state_name(server.final_state.load(Ordering::Acquire))
    } else if let Some(cancellation) = server.cancellation_handle() {
        session_state_name(cancellation.state())
    } else {
        state_name(server.final_state.load(Ordering::Acquire))
    };
    values.set_item("state", state)?;
    Ok(values)
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

#[pyclass(name = "_SharedClient", module = "scdata._core", frozen)]
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

#[pyfunction]
pub(crate) fn shared_next<'py>(
    py: Python<'py>,
    client: &PySharedClient,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    client.ensure_process()?;
    match py.allow_threads(|| client.next_owned()) {
        Ok(Some(batch)) => {
            cancel_incomplete_on_error(client, batch_into_array(py, batch)).map(Some)
        }
        Ok(None) => Ok(None),
        Err(NextError::Rust(error)) => Err(from_rust(error)),
        Err(NextError::Closed) => Err(invalid_argument("shared client is closed")),
    }
}

#[pyfunction]
pub(crate) fn shared_next_copy<'py>(
    py: Python<'py>,
    client: &PySharedClient,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    client.ensure_process()?;
    match py.allow_threads(|| client.next_owned()) {
        Ok(Some(batch)) => {
            cancel_incomplete_on_error(client, batch_into_owned_array(py, batch)).map(Some)
        }
        Ok(None) => Ok(None),
        Err(NextError::Rust(error)) => Err(from_rust(error)),
        Err(NextError::Closed) => Err(invalid_argument("shared client is closed")),
    }
}

#[pyfunction]
pub(crate) fn shared_read<'py>(
    py: Python<'py>,
    client: &PySharedClient,
    expected_rows: usize,
) -> PyResult<(Bound<'py, PyAny>, usize)> {
    client.ensure_process()?;
    if expected_rows > client.metadata.n_rows {
        return Err(invalid_argument(format!(
            "shared read expects {expected_rows} rows, exceeding dataset rows {}",
            client.metadata.n_rows
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
    match client.metadata.dtype {
        OutputDType::I16 => typed_read_array::<i16>(py, client, expected_rows),
        OutputDType::I32 => typed_read_array::<i32>(py, client, expected_rows),
        OutputDType::I64 => typed_read_array::<i64>(py, client, expected_rows),
        OutputDType::U16 => typed_read_array::<u16>(py, client, expected_rows),
        OutputDType::U32 => typed_read_array::<u32>(py, client, expected_rows),
        OutputDType::U64 => typed_read_array::<u64>(py, client, expected_rows),
        OutputDType::F32 => typed_read_array::<f32>(py, client, expected_rows),
        OutputDType::F64 => typed_read_array::<f64>(py, client, expected_rows),
    }
}

#[pyfunction]
pub(crate) fn shared_close(py: Python<'_>, client: &PySharedClient) {
    if std::process::id() != client.process_id {
        return;
    }
    py.allow_threads(|| {
        client.cancellation.cancel_if_incomplete();
        let inner = client.slot.lock().client.take();
        drop(inner);
    });
}

#[pyfunction]
pub(crate) fn shared_client_meta<'py>(
    py: Python<'py>,
    client: &PySharedClient,
) -> PyResult<Bound<'py, PyDict>> {
    client.ensure_process()?;
    let (closed, exhausted, next_logical_batch) = py.allow_threads(|| {
        let slot = client.slot.lock();
        (
            slot.client.is_none(),
            slot.exhausted,
            slot.client
                .as_ref()
                .and_then(SharedClient::next_logical_batch),
        )
    });
    let values = PyDict::new(py);
    values.set_item("rank", client.rank)?;
    values.set_item("world_size", client.metadata.world_size)?;
    values.set_item("n_rows", client.metadata.n_rows)?;
    values.set_item("n_cols", client.metadata.n_cols)?;
    values.set_item("batch_size", client.metadata.batch_size)?;
    values.set_item("batch_count", client.rank_batch_count)?;
    values.set_item("dtype", client.metadata.dtype.as_str())?;
    values.set_item("closed", closed)?;
    values.set_item("exhausted", exhausted)?;
    values.set_item("next_logical_batch", next_logical_batch)?;
    Ok(values)
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

fn cancel_incomplete_on_error<T>(client: &PySharedClient, result: PyResult<T>) -> PyResult<T> {
    if result.is_err() {
        client.cancellation.cancel_if_incomplete();
    }
    result
}

#[pyclass(name = "_SharedBatch", module = "scdata._core", frozen)]
struct PySharedBatch {
    batch: ManuallyDrop<SharedBatch>,
    process_id: u32,
}

impl Drop for PySharedBatch {
    fn drop(&mut self) {
        if std::process::id() != self.process_id {
            return;
        }
        // SAFETY: the creating process owns the only destructor for this lease.
        // A post-fork child leaves its copied lease untouched until process exit.
        unsafe { ManuallyDrop::drop(&mut self.batch) };
    }
}

fn batch_into_array<'py>(py: Python<'py>, batch: SharedBatch) -> PyResult<Bound<'py, PyAny>> {
    let owner = Bound::new(
        py,
        PySharedBatch {
            batch: ManuallyDrop::new(batch),
            process_id: std::process::id(),
        },
    )?;
    let (pointer, byte_len, rows, cols, row_stride, dtype) = {
        let owner_ref = owner.borrow();
        let bytes = owner_ref.batch.bytes().map_sc()?;
        (
            bytes.as_ptr(),
            bytes.len(),
            owner_ref.batch.rows(),
            owner_ref.batch.n_cols(),
            owner_ref.batch.row_stride_bytes(),
            owner_ref.batch.dtype(),
        )
    };
    #[cfg(target_endian = "big")]
    {
        let _ = (pointer, byte_len, rows, cols, row_stride, dtype, owner);
        return Err(from_rust(Error::Unsupported(
            "shared NumPy views require a little-endian target".into(),
        )));
    }
    #[cfg(target_endian = "little")]
    match dtype {
        OutputDType::I16 => typed_array::<i16>(owner, pointer, byte_len, rows, cols, row_stride),
        OutputDType::I32 => typed_array::<i32>(owner, pointer, byte_len, rows, cols, row_stride),
        OutputDType::I64 => typed_array::<i64>(owner, pointer, byte_len, rows, cols, row_stride),
        OutputDType::U16 => typed_array::<u16>(owner, pointer, byte_len, rows, cols, row_stride),
        OutputDType::U32 => typed_array::<u32>(owner, pointer, byte_len, rows, cols, row_stride),
        OutputDType::U64 => typed_array::<u64>(owner, pointer, byte_len, rows, cols, row_stride),
        OutputDType::F32 => typed_array::<f32>(owner, pointer, byte_len, rows, cols, row_stride),
        OutputDType::F64 => typed_array::<f64>(owner, pointer, byte_len, rows, cols, row_stride),
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
        OutputDType::I64 => typed_owned_array::<i64>(py, batch),
        OutputDType::U16 => typed_owned_array::<u16>(py, batch),
        OutputDType::U32 => typed_owned_array::<u32>(py, batch),
        OutputDType::U64 => typed_owned_array::<u64>(py, batch),
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
    let allocation_bytes = checked_array_bytes::<T>(layout.rows, layout.cols).map_sc()?;
    if allocation_bytes != layout.copy_bytes {
        return Err(from_rust(Error::Invariant(
            "shared compact byte count does not match the output shape".into(),
        )));
    }
    // SAFETY: the checked shape fits NumPy's signed dimensions, and every
    // element is initialized below before the array is returned to Python.
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
    checked_array_bytes::<T>(expected_rows, client.metadata.n_cols).map_sc()?;
    // SAFETY: the checked shape fits NumPy's signed dimensions. The array is
    // not exposed until `drain_into` has initialized every element.
    let array = unsafe { PyArray2::<T>::new(py, [expected_rows, client.metadata.n_cols], false) };
    let destination = array.data().cast::<u8>() as usize;
    let batches = match py.allow_threads(|| client.drain_into::<T>(destination, expected_rows)) {
        Ok(batches) => batches,
        Err(NextError::Rust(error)) => {
            client.cancellation.cancel_if_incomplete();
            return Err(from_rust(error));
        }
        Err(NextError::Closed) => return Err(invalid_argument("shared client is closed")),
    };
    Ok((array.into_any(), batches))
}

#[cfg(target_endian = "little")]
fn checked_array_bytes<T>(rows: usize, cols: usize) -> Result<usize, Error> {
    let element_size = std::mem::size_of::<T>();
    if element_size == 0 {
        return Err(Error::Invariant(
            "shared output dtype has zero-sized elements".into(),
        ));
    }
    if rows > isize::MAX as usize || cols > isize::MAX as usize {
        return Err(Error::Allocation(
            "shared NumPy dimension exceeds isize".into(),
        ));
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| Error::Allocation("shared element count overflow".into()))?;
    let bytes = count
        .checked_mul(element_size)
        .ok_or_else(|| Error::Allocation("shared array byte count overflow".into()))?;
    if bytes > isize::MAX as usize {
        return Err(Error::Allocation(
            "shared NumPy allocation exceeds isize".into(),
        ));
    }
    Ok(bytes)
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
    byte_len: usize,
    rows: usize,
    cols: usize,
    row_stride: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let element_size = std::mem::size_of::<T>();
    if element_size == 0 || !row_stride.is_multiple_of(element_size) {
        return Err(from_rust(Error::Invariant(
            "shared NumPy view has an incompatible element layout".into(),
        )));
    }
    checked_array_bytes::<T>(rows, cols).map_sc()?;
    let row_bytes = cols
        .checked_mul(element_size)
        .ok_or_else(|| from_rust(Error::Allocation("shared row byte count overflow".into())))?;
    if row_stride < row_bytes {
        return Err(from_rust(Error::Invariant(format!(
            "shared row stride {row_stride} is smaller than {row_bytes} payload bytes"
        ))));
    }
    let required = if rows == 0 || cols == 0 {
        0
    } else {
        (rows - 1)
            .checked_mul(row_stride)
            .and_then(|offset| offset.checked_add(row_bytes))
            .ok_or_else(|| from_rust(Error::Allocation("shared view extent overflow".into())))?
    };
    if required > byte_len {
        return Err(from_rust(Error::Invariant(format!(
            "shared batch exposes {byte_len} bytes, but the NumPy view requires {required}"
        ))));
    }
    if pointer.align_offset(std::mem::align_of::<T>()) != 0 {
        return Err(from_rust(Error::Invariant(
            "shared NumPy view has a misaligned data pointer".into(),
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

#[pyfunction]
pub(crate) fn shared_attach(py: Python<'_>, fd: i32, rank: usize) -> PyResult<PySharedClient> {
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
    module.add_function(wrap_pyfunction!(shared_run, module)?)?;
    module.add_function(wrap_pyfunction!(shared_cancel, module)?)?;
    module.add_function(wrap_pyfunction!(shared_duplicate_fd, module)?)?;
    module.add_function(wrap_pyfunction!(shared_server_meta, module)?)?;
    module.add_function(wrap_pyfunction!(shared_next, module)?)?;
    module.add_function(wrap_pyfunction!(shared_next_copy, module)?)?;
    module.add_function(wrap_pyfunction!(shared_read, module)?)?;
    module.add_function(wrap_pyfunction!(shared_close, module)?)?;
    module.add_function(wrap_pyfunction!(shared_client_meta, module)?)?;
    module.add_function(wrap_pyfunction!(shared_attach, module)?)?;
    Ok(())
}

#[cfg(all(test, target_endian = "little"))]
mod tests {
    use super::checked_array_bytes;

    #[test]
    fn numpy_array_size_checks_dimensions_and_bytes() {
        assert_eq!(checked_array_bytes::<u32>(3, 5).unwrap(), 60);
        assert!(checked_array_bytes::<u32>(0, usize::MAX).is_err());
        assert!(checked_array_bytes::<u64>(isize::MAX as usize, 2).is_err());
    }
}
