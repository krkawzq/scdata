#[cfg(feature = "profile")]
use std::sync::atomic::Ordering;
use std::sync::Arc;
#[cfg(feature = "profile")]
use std::time::Instant;

use crate::plan::{JobSide, ReadSource};
use crate::{Error, Result};

use super::{RangeClaim, SessionInner, WorkerScratch};

pub(super) fn run_worker(inner: Arc<SessionInner>, worker_id: usize) -> Result<()> {
    let mut data_encoded = Vec::new();
    let mut indices_encoded = Vec::new();
    let mut scratch = WorkerScratch::new();
    loop {
        match inner.claim_blocking_jobs(worker_id, 8) {
            RangeClaim::Claimed(jobs) => {
                for job_idx in jobs {
                    if !inner.is_running() {
                        return Ok(());
                    }
                    let job = &inner.plan.jobs[job_idx];
                    read_side(&inner, worker_id, &job.data, &mut data_encoded)?;
                    let data = side_bytes(&inner, &job.data, &data_encoded)?;
                    let indices = if let Some(side) = &job.indices {
                        read_side(&inner, worker_id, side, &mut indices_encoded)?;
                        Some(side_bytes(&inner, side, &indices_encoded)?)
                    } else {
                        None
                    };
                    inner.decode_and_commit(job_idx, data, indices, &mut scratch, worker_id)?;
                }
            }
            RangeClaim::WindowBlocked => inner.wait_for_window(worker_id),
            RangeClaim::Exhausted | RangeClaim::Stopped => return Ok(()),
        }
    }
}

fn read_side(
    inner: &SessionInner,
    worker_id: usize,
    side: &JobSide,
    output: &mut Vec<u8>,
) -> Result<()> {
    #[cfg(feature = "profile")]
    let started = Instant::now();
    let result = read_side_inner(inner, worker_id, side, output);
    #[cfg(feature = "profile")]
    inner.worker_stats(worker_id).io_wait_ns.fetch_add(
        u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    result
}

fn read_side_inner(
    inner: &SessionInner,
    _worker_id: usize,
    side: &JobSide,
    output: &mut Vec<u8>,
) -> Result<()> {
    let source = inner
        .plan
        .sources
        .get(side.source)
        .ok_or_else(|| Error::Invariant("job read source is missing".into()))?;
    match source {
        ReadSource::Empty => {
            output.clear();
            Ok(())
        }
        ReadSource::Positioned {
            file,
            base_offset,
            view_len,
        } => {
            let requested = side
                .read_range
                .end
                .checked_sub(side.read_range.start)
                .ok_or_else(|| Error::Invariant("positioned read range is reversed".into()))?;
            if side.read_range.end > *view_len {
                return Err(Error::StalePlan(format!(
                    "positioned range ends at {}, view length is {view_len}",
                    side.read_range.end
                )));
            }
            let len = usize::try_from(requested)
                .map_err(|_| Error::ResourceLimit("positioned read exceeds usize".into()))?;
            output.clear();
            output.try_reserve_exact(len)?;
            if len == 0 {
                return Ok(());
            }
            let absolute = base_offset
                .checked_add(side.read_range.start)
                .ok_or_else(|| Error::StalePlan("absolute positioned offset overflow".into()))?;
            let mut filled = 0usize;
            while filled < len {
                let offset = absolute
                    .checked_add(filled as u64)
                    .ok_or_else(|| Error::StalePlan("positioned read offset overflow".into()))?;
                let read = {
                    let remaining = len - filled;
                    let spare = &mut output.spare_capacity_mut()[..remaining];
                    match rustix::io::pread(file, spare, offset) {
                        Ok((initialized, _)) => initialized.len(),
                        Err(error) if error == rustix::io::Errno::INTR => continue,
                        Err(error) => return Err(std::io::Error::from(error).into()),
                    }
                };
                match read {
                    0 => {
                        return Err(Error::Io {
                            kind: std::io::ErrorKind::UnexpectedEof,
                            message: format!("positioned read ended at {filled} of {len} bytes"),
                        })
                    }
                    read => {
                        if read < len - filled {
                            #[cfg(feature = "profile")]
                            inner
                                .worker_stats(_worker_id)
                                .short_reads
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        filled += read;
                        // SAFETY: rustix returned an initialized prefix of the
                        // exact spare-capacity slice passed to `pread`. The
                        // vector is not reallocated between that call and this
                        // length update, and prior iterations initialized the
                        // existing prefix.
                        unsafe { output.set_len(filled) };
                    }
                }
            }
            #[cfg(feature = "profile")]
            {
                let stats = inner.worker_stats(_worker_id);
                stats.read_ops.fetch_add(1, Ordering::Relaxed);
                stats.read_bytes.fetch_add(len as u64, Ordering::Relaxed);
            }
            Ok(())
        }
        ReadSource::RangeKey {
            store,
            key,
            declared_len,
        } => {
            let start = usize::try_from(side.read_range.start)
                .map_err(|_| Error::ResourceLimit("range-key offset exceeds usize".into()))?;
            let end = usize::try_from(side.read_range.end)
                .map_err(|_| Error::ResourceLimit("range-key end exceeds usize".into()))?;
            if start > end || end > *declared_len {
                return Err(Error::StalePlan(format!(
                    "range key '{key}' request {start}..{end} exceeds declared length {declared_len}"
                )));
            }
            let len = end - start;
            let bytes = store.read_range(key, side.read_range.start, len)?;
            if bytes.len() != len {
                return Err(Error::StalePlan(format!(
                    "range key '{key}' returned {} bytes, expected {len}",
                    bytes.len()
                )));
            }
            *output = bytes;
            if len > 0 {
                #[cfg(feature = "profile")]
                {
                    let stats = inner.worker_stats(_worker_id);
                    stats.read_ops.fetch_add(1, Ordering::Relaxed);
                    stats.read_bytes.fetch_add(len as u64, Ordering::Relaxed);
                }
            }
            Ok(())
        }
        ReadSource::WholeKey {
            store,
            key,
            declared_len,
            cached,
        } => {
            if cached.is_some() {
                output.clear();
                return Ok(());
            }
            let bytes = store.read_limited(key, *declared_len)?;
            if bytes.len() != *declared_len {
                return Err(Error::StalePlan(format!(
                    "whole key '{key}' returned {} bytes, expected {declared_len}",
                    bytes.len()
                )));
            }
            *output = bytes;
            if *declared_len > 0 {
                #[cfg(feature = "profile")]
                {
                    let stats = inner.worker_stats(_worker_id);
                    stats.read_ops.fetch_add(1, Ordering::Relaxed);
                    stats
                        .read_bytes
                        .fetch_add(*declared_len as u64, Ordering::Relaxed);
                    stats.whole_keys.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        }
    }
}

fn side_bytes<'a>(inner: &'a SessionInner, side: &JobSide, fallback: &'a [u8]) -> Result<&'a [u8]> {
    let source = inner
        .plan
        .sources
        .get(side.source)
        .ok_or_else(|| Error::Invariant("job read source is missing".into()))?;
    let ReadSource::WholeKey {
        declared_len,
        cached: Some(cached),
        ..
    } = source
    else {
        return Ok(fallback);
    };
    let start = usize::try_from(side.read_range.start)
        .map_err(|_| Error::ResourceLimit("cached key offset exceeds usize".into()))?;
    let end = usize::try_from(side.read_range.end)
        .map_err(|_| Error::ResourceLimit("cached key end exceeds usize".into()))?;
    if cached.len() != *declared_len || start > end {
        return Err(Error::Invariant(
            "cached whole-key source has an invalid extent".into(),
        ));
    }
    cached
        .get(start..end)
        .ok_or_else(|| Error::StalePlan("cached whole-key range exceeds declared length".into()))
}
