//! DataBank-side CPU worker pool for planning and scatter work.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::thread;

use super::config::FillConfig;
use super::error::{DataBankError, DataBankResult};

pub(crate) type ComputeJob = Box<dyn FnOnce() -> DataBankResult<()> + Send + 'static>;

struct ComputeWork {
    job: ComputeJob,
    reply: Option<std::sync::mpsc::SyncSender<DataBankResult<()>>>,
}

pub(crate) struct DataBankComputePool {
    config: FillConfig,
    request_tx: Option<flume::Sender<ComputeWork>>,
    response_tx: Option<flume::Sender<ComputeWork>>,
    wake_tx: Option<flume::Sender<()>>,
    threads: Vec<thread::JoinHandle<()>>,
}

thread_local! {
    static IN_DATABANK_WORKER: Cell<bool> = const { Cell::new(false) };
}

struct WorkerFlagGuard;

impl Drop for WorkerFlagGuard {
    fn drop(&mut self) {
        IN_DATABANK_WORKER.with(|flag| flag.set(false));
    }
}

impl DataBankComputePool {
    pub(crate) fn new(config: FillConfig) -> DataBankResult<Self> {
        config.validate().map_err(DataBankError::InvalidConfig)?;
        if !config.parallel {
            return Ok(Self {
                config,
                request_tx: None,
                response_tx: None,
                wake_tx: None,
                threads: Vec::new(),
            });
        }

        let affinity_cpus = resolve_cpu_affinity(&config)?;
        let capacity = config.queue_capacity.max(1);
        let (request_tx, request_rx) = flume::bounded(capacity);
        let (response_tx, response_rx) = flume::bounded(capacity);
        let (wake_tx, wake_rx) = flume::bounded(config.num_workers.max(1));
        let mut threads = Vec::with_capacity(config.num_workers);

        for worker_idx in 0..config.num_workers {
            let worker_request_rx = request_rx.clone();
            let worker_response_rx = response_rx.clone();
            let worker_wake_rx = wake_rx.clone();
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
                    worker_loop(worker_response_rx, worker_request_rx, worker_wake_rx);
                }) {
                Ok(handle) => threads.push(handle),
                Err(err) => {
                    drop(request_tx);
                    drop(response_tx);
                    drop(wake_tx);
                    for handle in threads {
                        let _ = handle.join();
                    }
                    return Err(DataBankError::Io(err));
                }
            }
        }

        Ok(Self {
            config,
            request_tx: Some(request_tx),
            response_tx: Some(response_tx),
            wake_tx: Some(wake_tx),
            threads,
        })
    }

    pub(crate) fn should_parallelize(&self, rows: usize, bytes: usize) -> bool {
        self.response_tx.is_some()
            && !in_databank_worker()
            && rows >= self.config.min_parallel_rows
            && bytes >= self.config.min_parallel_bytes
    }

    pub(crate) fn worker_count(&self) -> usize {
        if self.response_tx.is_some() {
            self.config.num_workers.max(1)
        } else {
            1
        }
    }

    pub(crate) fn run_jobs(&self, jobs: Vec<ComputeJob>) -> DataBankResult<()> {
        let Some(tx) = self.response_tx.as_ref() else {
            for job in jobs {
                job()?;
            }
            return Ok(());
        };

        if jobs.len() <= 1 || in_databank_worker() {
            for job in jobs {
                job()?;
            }
            return Ok(());
        }

        let wake_tx = self
            .wake_tx
            .as_ref()
            .ok_or(DataBankError::ComputeShutdown)?;

        let mut receivers = Vec::with_capacity(jobs.len());
        for job in jobs {
            let (reply, rx) = std::sync::mpsc::sync_channel(1);
            tx.send(ComputeWork {
                job,
                reply: Some(reply),
            })
            .map_err(|_| DataBankError::ComputeShutdown)?;
            wake_workers(wake_tx);
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

    pub(crate) fn submit_request(&self, job: ComputeJob) -> DataBankResult<()> {
        self.submit(job, WorkClass::Request)
    }

    pub(crate) fn submit_response(&self, job: ComputeJob) -> DataBankResult<()> {
        self.submit(job, WorkClass::Response)
    }

    fn submit(&self, job: ComputeJob, class: WorkClass) -> DataBankResult<()> {
        if in_databank_worker() {
            return job();
        }

        let Some(wake_tx) = self.wake_tx.as_ref() else {
            return job();
        };
        let tx = match class {
            WorkClass::Request => self.request_tx.as_ref(),
            WorkClass::Response => self.response_tx.as_ref(),
        }
        .ok_or(DataBankError::ComputeShutdown)?;

        tx.send(ComputeWork { job, reply: None })
            .map_err(|_| DataBankError::ComputeShutdown)?;
        wake_workers(wake_tx);
        Ok(())
    }
}

impl Drop for DataBankComputePool {
    fn drop(&mut self) {
        self.request_tx.take();
        self.response_tx.take();
        self.wake_tx.take();
        while let Some(handle) = self.threads.pop() {
            let _ = handle.join();
        }
    }
}

#[derive(Clone, Copy)]
enum WorkClass {
    Request,
    Response,
}

fn worker_loop(
    response_rx: flume::Receiver<ComputeWork>,
    request_rx: flume::Receiver<ComputeWork>,
    wake_rx: flume::Receiver<()>,
) {
    IN_DATABANK_WORKER.with(|flag| flag.set(true));
    let _guard = WorkerFlagGuard;

    loop {
        if let Ok(work) = response_rx.try_recv() {
            complete_work(work);
            continue;
        }
        if let Ok(work) = request_rx.try_recv() {
            complete_work(work);
            continue;
        }

        if response_rx.is_disconnected() && request_rx.is_disconnected() {
            break;
        }

        if wake_rx.recv().is_err() && response_rx.is_empty() && request_rx.is_empty() {
            break;
        }
    }
}

fn complete_work(work: ComputeWork) {
    let result = panic::catch_unwind(AssertUnwindSafe(|| (work.job)()))
        .unwrap_or(Err(DataBankError::ComputeWorkerPanic));
    if let Some(reply) = work.reply {
        let _ = reply.send(result);
    }
}

fn in_databank_worker() -> bool {
    IN_DATABANK_WORKER.with(Cell::get)
}

fn wake_workers(wake_tx: &flume::Sender<()>) {
    let _ = wake_tx.try_send(());
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

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::*;

    fn test_config(num_workers: usize) -> FillConfig {
        FillConfig {
            parallel: true,
            num_workers,
            queue_capacity: 16,
            min_parallel_rows: 1,
            min_parallel_bytes: 1,
            cpus: None,
        }
    }

    #[test]
    fn request_queue_is_fifo_with_single_worker() {
        let pool = DataBankComputePool::new(test_config(1)).expect("pool");
        let order = Arc::new(Mutex::new(Vec::new()));
        let (done_tx, done_rx) = mpsc::channel();

        for seq in 0..5 {
            let order = Arc::clone(&order);
            let done_tx = done_tx.clone();
            pool.submit_request(Box::new(move || {
                order.lock().expect("order").push(seq);
                done_tx.send(()).expect("done");
                Ok(())
            }))
            .expect("submit request");
        }

        for _ in 0..5 {
            done_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("request completed");
        }
        assert_eq!(*order.lock().expect("order"), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn worker_prefers_response_over_queued_request() {
        let pool = DataBankComputePool::new(test_config(1)).expect("pool");
        let order = Arc::new(Mutex::new(Vec::new()));
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();

        let order_first = Arc::clone(&order);
        pool.submit_request(Box::new(move || {
            order_first.lock().expect("order").push("first");
            started_tx.send(()).expect("started");
            release_rx.recv().expect("release");
            Ok(())
        }))
        .expect("submit first request");
        started_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("first request started");

        let order_request = Arc::clone(&order);
        let done_request = done_tx.clone();
        pool.submit_request(Box::new(move || {
            order_request.lock().expect("order").push("request");
            done_request.send(()).expect("done request");
            Ok(())
        }))
        .expect("submit second request");

        let order_response = Arc::clone(&order);
        let done_response = done_tx.clone();
        pool.submit_response(Box::new(move || {
            order_response.lock().expect("order").push("response");
            done_response.send(()).expect("done response");
            Ok(())
        }))
        .expect("submit response");

        release_tx.send(()).expect("release first request");
        for _ in 0..2 {
            done_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("queued work completed");
        }

        assert_eq!(
            *order.lock().expect("order"),
            vec!["first", "response", "request"]
        );
    }

    #[test]
    fn run_jobs_in_worker_executes_inline_without_deadlock() {
        let pool = Arc::new(DataBankComputePool::new(test_config(1)).expect("pool"));
        let order = Arc::new(Mutex::new(Vec::new()));
        let (done_tx, done_rx) = mpsc::channel();

        let pool_in_job = Arc::clone(&pool);
        let order_in_job = Arc::clone(&order);
        pool.submit_response(Box::new(move || {
            let mut jobs = Vec::new();
            for seq in 0..3 {
                let order = Arc::clone(&order_in_job);
                jobs.push(Box::new(move || {
                    order.lock().expect("order").push(seq);
                    Ok(())
                }) as ComputeJob);
            }
            pool_in_job.run_jobs(jobs)?;
            done_tx.send(()).expect("done");
            Ok(())
        }))
        .expect("submit response");

        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("response completed");
        assert_eq!(*order.lock().expect("order"), vec![0, 1, 2]);
    }

    #[test]
    fn worker_does_not_request_nested_parallel_scatter() {
        let pool = Arc::new(DataBankComputePool::new(test_config(2)).expect("pool"));
        let (done_tx, done_rx) = mpsc::channel();

        let pool_in_job = Arc::clone(&pool);
        pool.submit_response(Box::new(move || {
            done_tx
                .send(pool_in_job.should_parallelize(usize::MAX, usize::MAX))
                .expect("done");
            Ok(())
        }))
        .expect("submit response");

        let nested_parallel = done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("response completed");
        assert!(!nested_parallel);
        assert!(pool.should_parallelize(usize::MAX, usize::MAX));
    }
}
