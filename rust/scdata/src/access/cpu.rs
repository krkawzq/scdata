//! CPU worker pool for access-side materialization.

use std::collections::BTreeSet;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::Arc;
use std::thread;

use tokio::sync::oneshot;

use super::error::AccessError;
use super::slice::{RangeCopy, SlicePlan, SliceShape};

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

enum AccessCpuInput {
    Shared(Arc<[u8]>),
    Owned(Vec<u8>),
}

impl AccessCpuInput {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Shared(input) => input.as_ref(),
            Self::Owned(input) => input.as_slice(),
        }
    }
}

struct AccessCpuWork {
    input: AccessCpuInput,
    ranges: Option<Arc<[RangeCopy]>>,
    output_len: usize,
    shape: SliceShape,
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
        plan: SlicePlan,
    ) -> Result<Vec<u8>, AccessError> {
        if should_materialize_inline(&plan) {
            return materialize_planned(&input, plan.ranges(), plan.output_len, plan.shape);
        }

        self.materialize_on_worker(AccessCpuInput::Shared(input), plan)
            .await
    }

    async fn materialize_on_worker(
        &self,
        input: AccessCpuInput,
        plan: SlicePlan,
    ) -> Result<Vec<u8>, AccessError> {
        let (reply, rx) = oneshot::channel();
        let work = AccessCpuWork {
            input,
            ranges: plan.ranges,
            output_len: plan.output_len,
            shape: plan.shape,
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
        plan: SlicePlan,
    ) -> Result<Vec<u8>, AccessError> {
        if plan.ranges.is_none() {
            debug_assert_eq!(input.len(), plan.output_len);
            return Ok(input);
        }

        if should_materialize_inline(&plan) {
            return materialize_planned(
                input.as_slice(),
                plan.ranges(),
                plan.output_len,
                plan.shape,
            );
        }

        self.materialize_on_worker(AccessCpuInput::Owned(input), plan)
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
    let AccessCpuWork {
        input,
        ranges,
        output_len,
        shape,
        reply,
    } = work;
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        materialize_planned(input.as_slice(), ranges.as_deref(), output_len, shape)
    }))
    .unwrap_or(Err(AccessError::CpuWorkerPanic));

    let _ = reply.send(result);
}

fn materialize_planned(
    input: &[u8],
    ranges: Option<&[RangeCopy]>,
    output_len: usize,
    shape: SliceShape,
) -> Result<Vec<u8>, AccessError> {
    let Some(ranges) = ranges else {
        debug_assert_eq!(input.len(), output_len);
        return Ok(input.to_vec());
    };

    if output_len == 0 {
        return Ok(Vec::new());
    }

    if ranges.len() == 1 {
        let range = ranges[0];
        if range.dst_offset == 0 && range.len() == output_len {
            return Ok(prevalidated_range(input, range.src_start, range.src_end).to_vec());
        }
    }

    if shape == SliceShape::Sequential {
        let mut out = Vec::with_capacity(output_len);
        let out_ptr = out.as_mut_ptr();
        for range in ranges {
            copy_prevalidated_range_to_ptr(
                input,
                out_ptr,
                output_len,
                range.dst_offset,
                range.src_start,
                range.src_end,
            );
        }
        // SAFETY: the SlicePlan classified these ranges as sequential, so the
        // loop above has initialized every byte in 0..output_len exactly once.
        unsafe {
            out.set_len(output_len);
        }
        return Ok(out);
    }

    let mut out = vec![0; output_len];
    for range in ranges {
        copy_prevalidated_range(
            input,
            out.as_mut_slice(),
            range.dst_offset,
            range.src_start,
            range.src_end,
        );
    }
    Ok(out)
}

#[inline]
fn prevalidated_range(input: &[u8], start: usize, end: usize) -> &[u8] {
    debug_assert!(start <= end);
    debug_assert!(end <= input.len());
    // SAFETY: `SlicePlan` checked every source range before entering the
    // planned copy path.
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
    // SAFETY: `SlicePlan` checked source and destination ranges before calling
    // this helper. `output` is a fresh Vec allocation, so source and
    // destination do not overlap.
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

fn should_materialize_inline(plan: &SlicePlan) -> bool {
    match plan.ranges.as_ref() {
        Some(_) => {
            plan.output_len <= INLINE_MATERIALIZE_MAX_BYTES
                && plan.range_count() <= INLINE_MATERIALIZE_MAX_RANGES
        }
        None => plan.output_len <= INLINE_MATERIALIZE_MAX_BYTES,
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
    use super::super::slice::SliceSpec;
    use super::*;
    use tokio::runtime::Builder;

    fn plan(triples: &[(usize, usize, usize)], data_len: usize) -> SlicePlan {
        let mut flat = Vec::with_capacity(triples.len() * 3);
        for &(dst_offset, src_start, src_end) in triples {
            flat.extend_from_slice(&[dst_offset, src_start, src_end]);
        }
        SliceSpec::from_triples(flat)
            .expect("slice spec")
            .plan(data_len)
            .expect("slice plan")
    }

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
            .block_on(pool.materialize(Arc::from(&b"abcdef"[..]), plan(&[(0, 0, 3), (3, 2, 5)], 6)))
            .expect("materialize");

        assert_eq!(out, b"abccde");
    }

    #[test]
    fn materializes_owned_scatter_copy_on_worker() {
        let pool = AccessCpuPool::new(AccessCpuConfig {
            num_workers: 2,
            queue_capacity: 4,
            cpus: None,
        })
        .expect("pool");
        let input = (0..64 * 1024)
            .map(|idx| (idx % 251) as u8)
            .collect::<Vec<_>>();
        let ranges = [(0, 1024, 12_000), (10_976, 30_000, 42_000)];
        let mut expected = Vec::new();
        expected.extend_from_slice(&input[1024..12_000]);
        expected.extend_from_slice(&input[30_000..42_000]);

        let out = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(pool.materialize_owned(input, plan(&ranges, 64 * 1024)))
            .expect("materialize");

        assert_eq!(out, expected);
    }

    #[test]
    fn inline_policy_keeps_small_outputs_on_scheduler() {
        assert!(should_materialize_inline(&plan(&[(0, 0, 3), (3, 2, 5)], 6)));
        assert!(should_materialize_inline(
            &SliceSpec::Full.plan(1024).expect("full plan")
        ));
        assert!(!should_materialize_inline(
            &SliceSpec::Full.plan(1024 * 1024).expect("large full plan")
        ));
        assert!(!should_materialize_inline(&plan(
            &[(0, 0, 1024 * 1024)],
            1024 * 1024
        )));
    }
}
