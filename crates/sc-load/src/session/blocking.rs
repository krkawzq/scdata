use std::sync::Arc;

use dyn_blosc::DecodeWorkspace;

use crate::plan::ReadSource;
use crate::{Error, Result};

use super::SessionInner;

const READY_BATCH: usize = 16;

pub(super) fn run_worker(inner: Arc<SessionInner>, _worker_id: usize) -> Result<()> {
    let mut encoded = Vec::new();
    let mut workspace = DecodeWorkspace::new();
    let mut ready = Vec::new();
    ready.try_reserve_exact(READY_BATCH)?;
    while inner.ready.pop_many(&mut ready, READY_BATCH) {
        for &node in &ready {
            if !inner.is_running() {
                return Ok(());
            }
            inner.claim_ready_node(node)?;
            if inner.is_io_node(node) {
                read_and_decode(&inner, node, &mut encoded, &mut workspace, true)?;
            } else {
                inner.execute_cpu_node(node)?;
            }
            inner.finish_node(node);
        }
    }
    Ok(())
}

pub(super) fn read_and_decode(
    inner: &SessionInner,
    node: usize,
    encoded: &mut Vec<u8>,
    workspace: &mut DecodeWorkspace,
    publish_ready: bool,
) -> Result<()> {
    let task = inner.io_task(node)?;
    // SAFETY: ExecutionPlan lowering points at a frozen ReadSource arena held
    // by SessionInner.plan for the complete worker lifetime.
    let source = unsafe { task.source.as_ref() };
    if let ReadSource::WholeKey {
        declared_len,
        cached: Some(cached),
        ..
    } = source
    {
        let start = usize::try_from(task.file_offset)
            .map_err(|_| Error::StalePlan("cached whole-key offset exceeds usize".into()))?;
        let end = start
            .checked_add(task.file_len)
            .ok_or_else(|| Error::StalePlan("cached whole-key range overflow".into()))?;
        if cached.len() != *declared_len {
            return Err(Error::StalePlan(
                "cached whole-key length changed after planning".into(),
            ));
        }
        let input = cached
            .get(start..end)
            .ok_or_else(|| Error::StalePlan("cached whole-key range is invalid".into()))?;
        return inner.decode_io(node, input, workspace, publish_ready);
    }

    let read = {
        #[cfg(feature = "profile")]
        let _timer = inner.profile_io_wait();
        read_exact(inner, source, task.file_offset, task.file_len, encoded)
    };
    let (operations, bytes) = read?;
    inner.record_reads(operations, bytes);
    inner.decode_io(node, encoded, workspace, publish_ready)
}

fn read_exact(
    inner: &SessionInner,
    source: &ReadSource,
    offset: u64,
    len: usize,
    output: &mut Vec<u8>,
) -> Result<(usize, usize)> {
    match source {
        ReadSource::Empty => Err(Error::Invariant("cache load uses an empty source".into())),
        ReadSource::Positioned {
            file,
            base_offset,
            view_len,
        } => {
            let end = offset
                .checked_add(len as u64)
                .ok_or_else(|| Error::StalePlan("positioned range overflow".into()))?;
            if end > *view_len {
                return Err(Error::StalePlan("positioned range exceeds source".into()));
            }
            output.clear();
            output.try_reserve_exact(len)?;
            let absolute = base_offset
                .checked_add(offset)
                .ok_or_else(|| Error::StalePlan("absolute offset overflow".into()))?;
            let mut filled = 0usize;
            let mut operations = 0usize;
            while filled < len {
                let read_offset = absolute
                    .checked_add(filled as u64)
                    .ok_or_else(|| Error::StalePlan("read offset overflow".into()))?;
                let read = {
                    let spare = &mut output.spare_capacity_mut()[..len - filled];
                    match rustix::io::pread(file, spare, read_offset) {
                        Ok((initialized, _)) => initialized.len(),
                        Err(error) if error == rustix::io::Errno::INTR => continue,
                        Err(error) => return Err(std::io::Error::from(error).into()),
                    }
                };
                operations = operations.saturating_add(1);
                if read == 0 {
                    return Err(Error::Io {
                        kind: std::io::ErrorKind::UnexpectedEof,
                        message: format!("positioned read ended at {filled} of {len} bytes"),
                    });
                }
                if read < len - filled {
                    inner.record_short_read();
                }
                filled += read;
                // SAFETY: rustix initialized exactly the returned spare prefix;
                // no allocation occurs between pread and this length update.
                unsafe { output.set_len(filled) };
            }
            Ok((operations, filled))
        }
        ReadSource::RangeKey {
            store,
            key,
            declared_len,
        } => {
            let end = usize::try_from(offset)
                .ok()
                .and_then(|start| start.checked_add(len))
                .ok_or_else(|| Error::StalePlan("range-key extent overflow".into()))?;
            if end > *declared_len {
                return Err(Error::StalePlan(format!(
                    "range key '{key}' exceeds declared length"
                )));
            }
            store.read_range_into(key, offset, len, output)?;
            if output.len() != len {
                return Err(Error::StalePlan("range-key short read".into()));
            }
            Ok((usize::from(len != 0), len))
        }
        ReadSource::WholeKey {
            store,
            key,
            declared_len,
            cached: None,
        } => {
            if offset != 0 || len != *declared_len {
                return Err(Error::Invariant(
                    "WholeKey task must materialize the complete logical key".into(),
                ));
            }
            let current_len = usize::try_from(store.len(key)?)
                .map_err(|_| Error::ResourceLimit("whole-key length exceeds usize".into()))?;
            if current_len != *declared_len {
                return Err(Error::StalePlan(format!(
                    "whole key '{key}' has {current_len} bytes, expected {declared_len}"
                )));
            }
            store.read_range_into(key, 0, *declared_len, output)?;
            if output.len() != *declared_len {
                return Err(Error::StalePlan("whole-key short read".into()));
            }
            inner.record_whole_key();
            Ok((usize::from(*declared_len != 0), *declared_len))
        }
        ReadSource::WholeKey {
            cached: Some(_), ..
        } => unreachable!("cached WholeKey is decoded without staging copy"),
    }
}
