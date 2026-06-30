//! Portable blocking backend built from positioned pread/pwrite worker threads.

use std::io;
use std::sync::Arc;
use std::thread;

use super::{execute_work, FileTable, QueueCore, ThreadedConfig};

pub(super) fn start(
    config: ThreadedConfig,
    queues: &[Arc<QueueCore>],
    file_table: Arc<FileTable>,
) -> io::Result<Vec<thread::JoinHandle<()>>> {
    let affinity_cpus = resolve_cpu_affinity(&config)?;
    let worker_count = config.effective_worker_count();
    let queues = queues.to_vec();

    let mut handles = Vec::with_capacity(worker_count);
    for worker_idx in 0..worker_count {
        let worker_queue = Arc::clone(&queues[worker_idx % queues.len()]);
        let file_table = Arc::clone(&file_table);
        let cpu = if affinity_cpus.is_empty() {
            None
        } else {
            Some(affinity_cpus[worker_idx % affinity_cpus.len()])
        };

        match thread::Builder::new()
            .name(format!("iopool-wrk-{worker_idx}"))
            .spawn(move || {
                if let Some(cpu) = cpu {
                    core_affinity::set_for_current(cpu);
                }
                worker_loop(worker_queue, file_table);
            }) {
            Ok(handle) => handles.push(handle),
            Err(err) => {
                for queue in &queues {
                    queue.shutdown();
                }
                for handle in handles {
                    let _ = handle.join();
                }
                return Err(err);
            }
        }
    }

    Ok(handles)
}

fn worker_loop(queue: Arc<QueueCore>, file_table: Arc<FileTable>) {
    while let Some(work) = queue.pop() {
        let id = work.id;
        let operation_started = work.operation_started;
        let result = execute_work(&file_table, work.op);
        queue.complete(id, operation_started, result);
    }
}

fn resolve_cpu_affinity(config: &ThreadedConfig) -> io::Result<Vec<core_affinity::CoreId>> {
    let Some(cpus) = &config.cpus else {
        return Ok(Vec::new());
    };

    let Some(core_ids) = core_affinity::get_core_ids() else {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "CPU affinity requested but core ids are unavailable",
        ));
    };

    let mut selected = Vec::with_capacity(cpus.len());
    for cpu in cpus {
        let Some(core_id) = core_ids.iter().find(|core_id| core_id.id == *cpu) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("CPU id {cpu} is not available"),
            ));
        };
        selected.push(*core_id);
    }
    Ok(selected)
}
