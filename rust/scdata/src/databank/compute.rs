//! DataBank-side CPU worker pool for planning and scatter work.

use std::collections::BTreeSet;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::thread;

use super::config::FillConfig;
use super::error::{DataBankError, DataBankResult};

pub(crate) type ComputeJob = Box<dyn FnOnce() -> DataBankResult<()> + Send + 'static>;

struct ComputeWork {
    job: ComputeJob,
    reply: std::sync::mpsc::SyncSender<DataBankResult<()>>,
}

pub(crate) struct DataBankComputePool {
    config: FillConfig,
    tx: Option<flume::Sender<ComputeWork>>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl DataBankComputePool {
    pub(crate) fn new(config: FillConfig) -> DataBankResult<Self> {
        config.validate().map_err(DataBankError::InvalidConfig)?;
        if !config.parallel {
            return Ok(Self {
                config,
                tx: None,
                threads: Vec::new(),
            });
        }

        let affinity_cpus = resolve_cpu_affinity(&config)?;
        let (tx, rx) = flume::bounded(config.queue_capacity.max(1));
        let mut threads = Vec::with_capacity(config.num_workers);

        for worker_idx in 0..config.num_workers {
            let worker_rx = rx.clone();
            let cpu = if affinity_cpus.is_empty() {
                None
            } else {
                Some(affinity_cpus[worker_idx % affinity_cpus.len()])
            };
            match thread::Builder::new()
                .name(format!("databank-cpu-{worker_idx}"))
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
                    return Err(DataBankError::Io(err));
                }
            }
        }

        Ok(Self {
            config,
            tx: Some(tx),
            threads,
        })
    }

    pub(crate) fn should_parallelize(&self, rows: usize, bytes: usize) -> bool {
        self.tx.is_some()
            && rows >= self.config.min_parallel_rows
            && bytes >= self.config.min_parallel_bytes
    }

    pub(crate) fn worker_count(&self) -> usize {
        if self.tx.is_some() {
            self.config.num_workers.max(1)
        } else {
            1
        }
    }

    pub(crate) fn run_jobs(&self, jobs: Vec<ComputeJob>) -> DataBankResult<()> {
        let Some(tx) = self.tx.as_ref() else {
            for job in jobs {
                job()?;
            }
            return Ok(());
        };

        if jobs.len() <= 1 {
            for job in jobs {
                job()?;
            }
            return Ok(());
        }

        let mut receivers = Vec::with_capacity(jobs.len());
        for job in jobs {
            let (reply, rx) = std::sync::mpsc::sync_channel(1);
            tx.send(ComputeWork { job, reply })
                .map_err(|_| DataBankError::ComputeShutdown)?;
            receivers.push(rx);
        }

        let mut first_error = None;
        for rx in receivers {
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
                Err(_) => {
                    if first_error.is_none() {
                        first_error = Some(DataBankError::ComputeShutdown);
                    }
                }
            }
        }

        if let Some(err) = first_error {
            Err(err)
        } else {
            Ok(())
        }
    }
}

impl Drop for DataBankComputePool {
    fn drop(&mut self) {
        self.tx.take();
        while let Some(handle) = self.threads.pop() {
            let _ = handle.join();
        }
    }
}

fn worker_loop(rx: flume::Receiver<ComputeWork>) {
    while let Ok(work) = rx.recv() {
        complete_work(work);
    }
}

fn complete_work(work: ComputeWork) {
    let result = panic::catch_unwind(AssertUnwindSafe(|| (work.job)()))
        .unwrap_or(Err(DataBankError::ComputeWorkerPanic));
    let _ = work.reply.send(result);
}

fn resolve_cpu_affinity(config: &FillConfig) -> io::Result<Vec<usize>> {
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
