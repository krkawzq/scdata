//! Bounded chunk-level task parallelism for sc-compress writers and readers.
//!
//! dyn-blosc block parallelism is intentionally left at `threads = 1`; this
//! module schedules whole chunk encode/decode jobs instead.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use crate::error::{Error, Result};

/// Default worker count: host `available_parallelism`, or `1` if unknown.
#[must_use]
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
}

pub(crate) fn validate_threads(threads: usize) -> Result<usize> {
    if threads == 0 {
        return Err(Error::invalid_argument("threads must be greater than zero"));
    }
    Ok(threads)
}

/// Produce jobs on the calling thread and consume them on a fixed set of
/// scoped workers.
///
/// The channel holds at most one queued job per worker. This keeps writer
/// chunk spans and encoded buffers bounded while allowing encoding and file
/// installation to overlap. Borrowed jobs are supported because every worker
/// is joined before this function returns.
pub(crate) fn try_for_each_stream<T, P, F>(
    threads: usize,
    job_upper_bound: usize,
    produce: P,
    consume: F,
) -> Result<()>
where
    T: Send,
    P: FnOnce(&mut dyn FnMut(T) -> Result<()>) -> Result<()>,
    F: Fn(T) -> Result<()> + Sync,
{
    try_for_each_stream_init(
        threads,
        job_upper_bound,
        produce,
        || (),
        move |item, ()| consume(item),
    )
}

/// Variant of [`try_for_each_stream`] with one reusable state value per worker.
///
/// This is useful for codec scratch buffers: state is created once for each
/// participating worker and reused for every job claimed by that worker.
pub(crate) fn try_for_each_stream_init<T, P, S, I, F>(
    threads: usize,
    job_upper_bound: usize,
    produce: P,
    initialize: I,
    consume: F,
) -> Result<()>
where
    T: Send,
    S: Send,
    P: FnOnce(&mut dyn FnMut(T) -> Result<()>) -> Result<()>,
    I: Fn() -> S + Sync,
    F: Fn(T, &mut S) -> Result<()> + Sync,
{
    let threads = validate_threads(threads)?;
    if threads == 1 || job_upper_bound <= 1 {
        let mut state = initialize();
        let mut emit = |item| {
            catch_unwind(AssertUnwindSafe(|| consume(item, &mut state)))
                .unwrap_or_else(|_| Err(Error::invalid_argument("chunk worker panicked")))
        };
        return produce(&mut emit);
    }

    let worker_count = threads.min(job_upper_bound);
    let cancelled = AtomicBool::new(false);
    let worker_error = Mutex::new(None);
    std::thread::scope(|scope| {
        let (sender, receiver) = mpsc::sync_channel::<T>(worker_count);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut handles = Vec::new();
        handles.try_reserve_exact(worker_count)?;

        for _ in 0..worker_count {
            let receiver = Arc::clone(&receiver);
            let cancelled = &cancelled;
            let worker_error = &worker_error;
            let initialize = &initialize;
            let consume = &consume;
            handles.push(std::thread::Builder::new().spawn_scoped(scope, move || {
                let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
                    let mut state = initialize();
                    loop {
                        if cancelled.load(Ordering::Acquire) {
                            return Ok(());
                        }
                        let item = {
                            let receiver = receiver.lock().map_err(|_| {
                                Error::invalid_argument("chunk work queue mutex poisoned")
                            })?;
                            match receiver.recv() {
                                Ok(item) => item,
                                Err(_) => return Ok(()),
                            }
                        };
                        if cancelled.load(Ordering::Acquire) {
                            return Ok(());
                        }
                        consume(item, &mut state)?;
                    }
                }));
                match outcome {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => record_worker_error(worker_error, cancelled, error),
                    Err(_) => record_worker_error(
                        worker_error,
                        cancelled,
                        Error::invalid_argument("chunk worker panicked"),
                    ),
                }
            })?);
        }

        // Only workers retain receivers. If they all stop after an error, a
        // producer blocked on the bounded channel is woken with Disconnected.
        drop(receiver);
        let mut emit = |item| {
            if cancelled.load(Ordering::Acquire) {
                return Err(Error::invalid_argument(
                    "chunk workers stopped before accepting all jobs",
                ));
            }
            sender.send(item).map_err(|_| {
                Error::invalid_argument("chunk workers stopped before accepting all jobs")
            })
        };
        let produced = produce(&mut emit);
        if produced.is_err() {
            cancelled.store(true, Ordering::Release);
        }
        drop(sender);

        for handle in handles {
            if handle.join().is_err() {
                record_worker_error(
                    &worker_error,
                    &cancelled,
                    Error::invalid_argument("chunk worker panicked"),
                );
            }
        }

        let worker_error = worker_error
            .lock()
            .map_err(|_| Error::invalid_argument("chunk worker error mutex poisoned"))?
            .take();
        if let Some(error) = worker_error {
            Err(error)
        } else {
            produced
        }
    })
}

fn record_worker_error(slot: &Mutex<Option<Error>>, cancelled: &AtomicBool, error: Error) {
    cancelled.store(true, Ordering::Release);
    let mut slot = slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if slot.is_none() {
        *slot = Some(error);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use super::*;

    #[test]
    fn zero_threads_is_rejected() {
        assert!(validate_threads(0).is_err());
        assert!(try_for_each_stream::<(), _, _>(0, 3, |_| Ok(()), |_| Ok(())).is_err());
    }

    #[test]
    fn sequential_and_parallel_agree() {
        fn run(threads: usize) -> Vec<usize> {
            let output = Mutex::new(Vec::new());
            try_for_each_stream(
                threads,
                32,
                |emit| {
                    for value in 0..32usize {
                        emit(value)?;
                    }
                    Ok(())
                },
                |value| {
                    output.lock().unwrap().push(value * value);
                    Ok(())
                },
            )
            .unwrap();
            let mut output = output.into_inner().unwrap();
            output.sort_unstable();
            output
        }

        assert_eq!(run(1), run(4));
    }

    #[test]
    fn worker_state_is_initialized_once_and_reused() {
        let initialized = AtomicUsize::new(0);
        let maximum_jobs_per_state = AtomicUsize::new(0);
        try_for_each_stream_init(
            4,
            32,
            |emit| {
                for value in 0..32usize {
                    emit(value)?;
                }
                Ok(())
            },
            || {
                initialized.fetch_add(1, Ordering::SeqCst);
                0usize
            },
            |_value, jobs| {
                *jobs += 1;
                maximum_jobs_per_state.fetch_max(*jobs, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(initialized.load(Ordering::SeqCst), 4);
        assert!(maximum_jobs_per_state.load(Ordering::SeqCst) > 1);
    }

    #[test]
    fn stream_keeps_queued_and_active_jobs_bounded() {
        struct Tracked {
            alive: Arc<AtomicUsize>,
        }

        impl Tracked {
            fn new(alive: &Arc<AtomicUsize>, maximum: &Arc<AtomicUsize>) -> Self {
                let current = alive.fetch_add(1, Ordering::SeqCst) + 1;
                maximum.fetch_max(current, Ordering::SeqCst);
                Self {
                    alive: Arc::clone(alive),
                }
            }
        }

        impl Drop for Tracked {
            fn drop(&mut self) {
                self.alive.fetch_sub(1, Ordering::SeqCst);
            }
        }

        let alive = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        try_for_each_stream(
            4,
            100,
            |emit| {
                for _ in 0..100 {
                    emit(Tracked::new(&alive, &maximum))?;
                }
                Ok(())
            },
            |_item| {
                std::thread::sleep(Duration::from_millis(1));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(alive.load(Ordering::SeqCst), 0);
        assert!(maximum.load(Ordering::SeqCst) <= 9);
    }

    #[test]
    fn worker_errors_cancel_a_full_queue() {
        let error = try_for_each_stream(
            4,
            1_000,
            |emit| {
                for value in 0..1_000usize {
                    emit(value)?;
                }
                Ok(())
            },
            |value| {
                if value == 2 {
                    Err(Error::invalid_argument("boom"))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(message) if message == "boom"));
    }

    #[test]
    fn worker_panics_are_reported_and_cancel_the_queue() {
        let error = try_for_each_stream(
            4,
            1_000,
            |emit| {
                for value in 0..1_000usize {
                    emit(value)?;
                }
                Ok(())
            },
            |value| {
                assert_ne!(value, 2, "boom");
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidArgument(message) if message == "chunk worker panicked"
        ));
    }

    #[test]
    fn sequential_worker_panics_are_reported() {
        let error =
            try_for_each_stream(1, 1, |emit| emit(()), |()| -> Result<()> { panic!("boom") })
                .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidArgument(message) if message == "chunk worker panicked"
        ));
    }
}
