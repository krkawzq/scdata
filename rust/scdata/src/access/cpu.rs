//! CPU worker pool for access-side materialization.

use std::collections::BTreeSet;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::Arc;
use std::thread;

use tokio::sync::oneshot;

use super::error::AccessError;

const INLINE_MATERIALIZE_MAX_BYTES: usize = 16 * 1024;
const INLINE_MATERIALIZE_MAX_RANGES: usize = 4;

/// Worker-pool settings for access-side CPU work.
#[derive(Debug, Clone)]
pub struct AccessCpuConfig {
    /// Number of materialization workers. Default: 4.
    pub num_workers: usize,
    /// Bounded CPU command queue capacity. Default: 1024.
    pub queue_capacity: usize,
    /// Optional CPU affinity allow-list for workers.
    pub cpus: Option<Vec<usize>>,
}

impl Default for AccessCpuConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            queue_capacity: 1024,
            cpus: None,
        }
    }
}

impl AccessCpuConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.num_workers == 0 {
            return Err("cpu.num_workers must be greater than 0".to_string());
        }
        if self.queue_capacity == 0 {
            return Err("cpu.queue_capacity must be greater than 0".to_string());
        }
        if let Some(cpus) = &self.cpus {
            if cpus.is_empty() {
                return Err("cpu.cpus list must not be empty".to_string());
            }

            let unique = cpus.iter().copied().collect::<BTreeSet<_>>();
            if unique.len() != cpus.len() {
                return Err("cpu.cpus list contains duplicate entries".to_string());
            }
        }
        Ok(())
    }

    fn worker_count(&self) -> usize {
        self.num_workers.max(1)
    }

    fn queue_capacity(&self) -> usize {
        self.queue_capacity.max(1)
    }
}

struct AccessCpuWork {
    input: Arc<[u8]>,
    slice: Option<Vec<usize>>,
    output_len: usize,
    reply: oneshot::Sender<Result<Vec<u8>, AccessError>>,
}

/// Access-side CPU pool for copy and scatter-copy materialization.
pub(crate) struct AccessCpuPool {
    tx: Option<flume::Sender<AccessCpuWork>>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl AccessCpuPool {
    pub(crate) fn new(config: AccessCpuConfig) -> io::Result<Self> {
        config
            .validate()
            .map_err(|msg| io::Error::new(io::ErrorKind::InvalidInput, msg))?;
        let affinity_cpus = resolve_cpu_affinity(&config)?;
        let (tx, rx) = flume::bounded(config.queue_capacity());

        let mut threads = Vec::with_capacity(config.worker_count());
        for worker_idx in 0..config.worker_count() {
            let worker_rx = rx.clone();
            let cpu = if affinity_cpus.is_empty() {
                None
            } else {
                Some(affinity_cpus[worker_idx % affinity_cpus.len()])
            };

            match thread::Builder::new()
                .name(format!("access-cpu-{worker_idx}"))
                .spawn(move || {
                    if let Some(cpu) = cpu {
                        pin_current_thread(cpu);
                    }
                    worker_loop(worker_rx);
                }) {
                Ok(handle) => threads.push(handle),
                Err(err) => {
                    drop(tx);
                    for handle in threads {
                        let _ = handle.join();
                    }
                    return Err(err);
                }
            }
        }

        Ok(Self {
            tx: Some(tx),
            threads,
        })
    }

    pub(crate) async fn materialize(
        &self,
        input: Arc<[u8]>,
        slice: Option<&[usize]>,
        output_len: usize,
    ) -> Result<Vec<u8>, AccessError> {
        if should_materialize_inline(slice, output_len) {
            return materialize_prevalidated(&input, slice, output_len);
        }

        let (reply, rx) = oneshot::channel();
        let work = AccessCpuWork {
            input,
            slice: slice.map(Vec::from),
            output_len,
            reply,
        };
        let tx = self.tx.as_ref().ok_or(AccessError::Shutdown)?;
        tx.send_async(work)
            .await
            .map_err(|_| AccessError::Shutdown)?;
        rx.await.map_err(|_| AccessError::Shutdown)?
    }

    pub(crate) async fn materialize_owned(
        &self,
        input: Vec<u8>,
        slice: Option<&[usize]>,
        output_len: usize,
    ) -> Result<Vec<u8>, AccessError> {
        if slice.is_none() {
            debug_assert_eq!(input.len(), output_len);
            return Ok(input);
        }

        if should_materialize_inline(slice, output_len) {
            return materialize_prevalidated(input.as_slice(), slice, output_len);
        }

        self.materialize(Arc::from(input.into_boxed_slice()), slice, output_len)
            .await
    }
}

impl Drop for AccessCpuPool {
    fn drop(&mut self) {
        self.tx.take();
        while let Some(handle) = self.threads.pop() {
            if handle.join().is_err() {
                eprintln!("[access] CPU worker panicked during shutdown");
            }
        }
    }
}

fn worker_loop(rx: flume::Receiver<AccessCpuWork>) {
    while let Ok(work) = rx.recv() {
        complete_work(work);
    }
}

fn complete_work(work: AccessCpuWork) {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        materialize_prevalidated(&work.input, work.slice.as_deref(), work.output_len)
    }))
    .unwrap_or(Err(AccessError::CpuWorkerPanic));

    let _ = work.reply.send(result);
}

fn materialize_prevalidated(
    input: &[u8],
    slice: Option<&[usize]>,
    output_len: usize,
) -> Result<Vec<u8>, AccessError> {
    let Some(slice) = slice else {
        debug_assert_eq!(input.len(), output_len);
        return Ok(input.to_vec());
    };
    let Some(shape) = validate_scatter_shape(input.len(), slice, output_len) else {
        return Err(AccessError::InvalidSlice(
            "slice spec failed prevalidated CPU bounds check".to_string(),
        ));
    };

    if output_len == 0 {
        return Ok(Vec::new());
    }

    if slice.len() == 3 {
        let (off, start, end) = prevalidated_triple(slice);
        if off == 0 && end - start == output_len {
            return Ok(prevalidated_range(input, start, end).to_vec());
        }
    }

    if shape == ScatterShape::Sequential {
        let mut out = Vec::with_capacity(output_len);
        let out_ptr = out.as_mut_ptr();
        for triple in slice.chunks_exact(3) {
            let (off, start, end) = prevalidated_triple(triple);
            copy_prevalidated_range_to_ptr(input, out_ptr, output_len, off, start, end);
        }
        // SAFETY: `validate_scatter_shape` classified this slice as
        // sequential, so the loop above has initialized every byte in
        // 0..output_len exactly once.
        unsafe {
            out.set_len(output_len);
        }
        return Ok(out);
    }

    let mut out = vec![0; output_len];
    for triple in slice.chunks_exact(3) {
        let (off, start, end) = prevalidated_triple(triple);
        copy_prevalidated_range(input, out.as_mut_slice(), off, start, end);
    }
    Ok(out)
}

#[inline]
fn prevalidated_triple(triple: &[usize]) -> (usize, usize, usize) {
    debug_assert_eq!(triple.len(), 3);
    // SAFETY: callers pass either the whole three-element slice fast path or
    // slices yielded by `chunks_exact(3)`, so these indexes are in-bounds.
    unsafe {
        (
            *triple.get_unchecked(0),
            *triple.get_unchecked(1),
            *triple.get_unchecked(2),
        )
    }
}

#[inline]
fn prevalidated_range(input: &[u8], start: usize, end: usize) -> &[u8] {
    debug_assert!(start <= end);
    debug_assert!(end <= input.len());
    // SAFETY: `materialize_prevalidated` checks every source range before
    // entering the unchecked copy path.
    unsafe { input.get_unchecked(start..end) }
}

#[inline]
fn copy_prevalidated_range(input: &[u8], output: &mut [u8], off: usize, start: usize, end: usize) {
    debug_assert!(start <= end);
    debug_assert!(end <= input.len());
    debug_assert!(off <= output.len());
    debug_assert!(off
        .checked_add(end - start)
        .is_some_and(|out_end| out_end <= output.len()));

    let len = end - start;
    // SAFETY: `materialize_prevalidated` checks source and destination ranges
    // before calling this helper. `output` is a fresh Vec allocation, so source
    // and destination do not overlap.
    unsafe {
        ptr::copy_nonoverlapping(input.as_ptr().add(start), output.as_mut_ptr().add(off), len);
    }
}

#[inline]
fn copy_prevalidated_range_to_ptr(
    input: &[u8],
    output: *mut u8,
    output_len: usize,
    off: usize,
    start: usize,
    end: usize,
) {
    debug_assert!(start <= end);
    debug_assert!(end <= input.len());
    debug_assert!(off <= output_len);
    debug_assert!(off
        .checked_add(end - start)
        .is_some_and(|out_end| out_end <= output_len));

    let len = end - start;
    // SAFETY: validation checked both source and destination ranges. The
    // destination pointer comes from a fresh Vec allocation with capacity
    // `output_len`, so it is valid for writes within 0..output_len and cannot
    // overlap the immutable input slice.
    unsafe {
        ptr::copy_nonoverlapping(input.as_ptr().add(start), output.add(off), len);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScatterShape {
    Sequential,
    Sparse,
}

fn validate_scatter_shape(
    data_len: usize,
    slice: &[usize],
    output_len: usize,
) -> Option<ScatterShape> {
    if slice.len() % 3 != 0 {
        return None;
    }

    let mut expected_len = 0usize;
    let mut cursor = 0usize;
    let mut sequential = true;
    for triple in slice.chunks_exact(3) {
        let off = triple[0];
        let start = triple[1];
        let end = triple[2];
        if start > end || end > data_len {
            return None;
        }
        let out_end = off.checked_add(end - start)?;

        if sequential && off == cursor {
            cursor = out_end;
        } else {
            sequential = false;
        }
        expected_len = expected_len.max(out_end);
    }

    if expected_len != output_len {
        return None;
    }

    Some(if sequential && cursor == output_len {
        ScatterShape::Sequential
    } else {
        ScatterShape::Sparse
    })
}

fn should_materialize_inline(slice: Option<&[usize]>, output_len: usize) -> bool {
    match slice {
        Some(slice) => {
            output_len <= INLINE_MATERIALIZE_MAX_BYTES
                && slice.len() / 3 <= INLINE_MATERIALIZE_MAX_RANGES
        }
        None => output_len <= INLINE_MATERIALIZE_MAX_BYTES,
    }
}

fn resolve_cpu_affinity(config: &AccessCpuConfig) -> io::Result<Vec<usize>> {
    let Some(cpus) = &config.cpus else {
        return Ok(Vec::new());
    };

    let Some(core_ids) = core_affinity::get_core_ids() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "CPU affinity requested but core ids are unavailable",
        ));
    };

    let available = core_ids
        .iter()
        .map(|core_id| core_id.id)
        .collect::<BTreeSet<_>>();

    for cpu in cpus {
        if !available.contains(cpu) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("CPU id {cpu} is not available"),
            ));
        }
    }

    Ok(cpus.clone())
}

fn pin_current_thread(cpu: usize) {
    let Some(core_ids) = core_affinity::get_core_ids() else {
        return;
    };
    if let Some(core_id) = core_ids.into_iter().find(|core_id| core_id.id == cpu) {
        core_affinity::set_for_current(core_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Builder;

    #[test]
    fn config_rejects_invalid_workers() {
        let config = AccessCpuConfig {
            num_workers: 0,
            ..AccessCpuConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_rejects_duplicate_cpus() {
        let config = AccessCpuConfig {
            cpus: Some(vec![0, 0]),
            ..AccessCpuConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_without_cpus_leaves_workers_unpinned() {
        let config = AccessCpuConfig {
            cpus: None,
            ..AccessCpuConfig::default()
        };
        let cpus = resolve_cpu_affinity(&config).expect("no affinity requested");
        assert!(cpus.is_empty());
    }

    #[test]
    fn materializes_scatter_copy_on_worker() {
        let pool = AccessCpuPool::new(AccessCpuConfig {
            num_workers: 2,
            queue_capacity: 4,
            cpus: None,
        })
        .expect("pool");

        let out = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(pool.materialize(Arc::from(&b"abcdef"[..]), Some(&[0, 0, 3, 3, 2, 5]), 6))
            .expect("materialize");

        assert_eq!(out, b"abccde");
    }

    #[test]
    fn inline_policy_keeps_small_outputs_on_scheduler() {
        assert!(should_materialize_inline(Some(&[0, 0, 3, 3, 2, 5]), 6));
        assert!(should_materialize_inline(None, 1024));
        assert!(!should_materialize_inline(None, 1024 * 1024));
        assert!(!should_materialize_inline(
            Some(&[0, 0, 1024 * 1024]),
            1024 * 1024
        ));
    }
}
