use std::collections::BTreeSet;
use std::future::Future;
use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread;

use tokio::sync::oneshot;

use crate::profile::{ProfileRuntime, ProfileSnapshot};

use super::profile::{CodecProfile, CodecQueueTimer, CodecSubmitKind};
use super::runner::DecodeRunner;
use super::{CodecError, CodecResult, CodecSpec, SharedCodec};

/// Worker-pool settings for chunk decoding.
#[derive(Debug, Clone)]
pub struct DecodePoolConfig {
    /// Number of cross-chunk decode workers. Default: 4.
    pub num_workers: usize,
    /// Bounded decode command queue capacity. Default: 1024.
    pub queue_capacity: usize,
    /// Optional CPU affinity allow-list for decode workers.
    pub cpus: Option<Vec<usize>>,
}

impl Default for DecodePoolConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            queue_capacity: 1024,
            cpus: None,
        }
    }
}

impl DecodePoolConfig {
    pub fn validate(&self) -> CodecResult<()> {
        if self.num_workers == 0 {
            return Err(CodecError::InvalidConfig(
                "num_workers must be greater than 0".to_string(),
            ));
        }
        if self.queue_capacity == 0 {
            return Err(CodecError::InvalidConfig(
                "queue_capacity must be greater than 0".to_string(),
            ));
        }
        if let Some(cpus) = &self.cpus {
            if cpus.is_empty() {
                return Err(CodecError::InvalidConfig(
                    "cpus list must not be empty".to_string(),
                ));
            }

            let unique = cpus.iter().copied().collect::<BTreeSet<_>>();
            if unique.len() != cpus.len() {
                return Err(CodecError::InvalidConfig(
                    "cpus list contains duplicate entries".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn worker_count(&self) -> usize {
        self.num_workers.max(1)
    }

    pub fn queue_capacity(&self) -> usize {
        self.queue_capacity.max(1)
    }
}

/// One decode request. The output buffer belongs to this request only.
#[derive(Debug)]
pub struct DecodeRequest {
    pub codec: SharedCodec,
    pub encoded: Arc<[u8]>,
    pub expected_size: Option<usize>,
    pub output: DecodeOutput,
}

/// Output ownership strategy for a decode request.
#[derive(Debug)]
pub enum DecodeOutput {
    /// Allocate a fresh decoded output on the worker.
    Allocate,
    /// Reuse the vector's current initialized length as writable output.
    ReuseInitialized(Vec<u8>),
    /// Reuse the vector's capacity as writable output when the decoded size is known.
    ReuseCapacity(Vec<u8>),
}

impl DecodeRequest {
    pub fn new(codec: SharedCodec, encoded: impl Into<Arc<[u8]>>) -> Self {
        Self {
            codec,
            encoded: encoded.into(),
            expected_size: None,
            output: DecodeOutput::Allocate,
        }
    }

    pub fn from_spec(spec: &CodecSpec, encoded: impl Into<Arc<[u8]>>) -> Self {
        Self::new(spec.build(), encoded)
    }

    pub fn with_expected_size(mut self, expected_size: usize) -> Self {
        self.expected_size = Some(expected_size);
        self
    }

    /// Provide initialized caller-owned output memory for the worker to fill.
    ///
    /// Only the vector's current length is writable. Decoding fails instead of
    /// growing this buffer behind the caller's back.
    pub fn with_reuse_initialized_output(mut self, output: Vec<u8>) -> Self {
        self.output = DecodeOutput::ReuseInitialized(output);
        self
    }

    /// Provide caller-owned output capacity for the worker to fill.
    ///
    /// When the final decoded size is known, the vector's capacity is used as
    /// writable memory so callers can pass `Vec::with_capacity(size)`.
    pub fn with_reuse_capacity_output(mut self, output: Vec<u8>) -> Self {
        self.output = DecodeOutput::ReuseCapacity(output);
        self
    }
}

#[derive(Debug)]
struct DecodeWork {
    request: DecodeRequest,
    reply: oneshot::Sender<CodecResult<Vec<u8>>>,
    queued_at: CodecQueueTimer,
}

/// Future returned by decode submission.
#[derive(Debug)]
pub struct DecodeFuture {
    rx: Option<oneshot::Receiver<CodecResult<Vec<u8>>>>,
}

impl DecodeFuture {
    fn new(rx: oneshot::Receiver<CodecResult<Vec<u8>>>) -> Self {
        Self { rx: Some(rx) }
    }

    pub fn blocking_recv(mut self) -> CodecResult<Vec<u8>> {
        let Some(rx) = self.rx.take() else {
            return Err(CodecError::Shutdown);
        };
        match rx.blocking_recv() {
            Ok(result) => result,
            Err(_) => Err(CodecError::Shutdown),
        }
    }
}

fn decode_work(request: DecodeRequest, queued_at: CodecQueueTimer) -> (DecodeWork, DecodeFuture) {
    let (reply, rx) = oneshot::channel();
    let work = DecodeWork {
        request,
        reply,
        queued_at,
    };
    (work, DecodeFuture::new(rx))
}

impl Future for DecodeFuture {
    type Output = CodecResult<Vec<u8>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let Some(rx) = self.rx.as_mut() else {
            return Poll::Ready(Err(CodecError::Shutdown));
        };

        match Pin::new(rx).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(CodecError::Shutdown)),
        }
    }
}

/// Cross-chunk decode worker pool.
pub struct DecodePool {
    tx: Option<flume::Sender<DecodeWork>>,
    threads: Vec<thread::JoinHandle<()>>,
    profile: CodecProfile,
}

impl DecodePool {
    pub fn new(config: DecodePoolConfig) -> CodecResult<Self> {
        Self::with_codec_profile(config, CodecProfile::from_env())
    }

    pub fn with_profile(config: DecodePoolConfig, profile: ProfileRuntime) -> CodecResult<Self> {
        Self::with_codec_profile(config, CodecProfile::from_runtime(profile))
    }

    pub fn with_codec_profile(
        config: DecodePoolConfig,
        profile: CodecProfile,
    ) -> CodecResult<Self> {
        config.validate()?;
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
            let worker_profile = profile.clone();

            match thread::Builder::new()
                .name(format!("decode-wrk-{worker_idx}"))
                .spawn(move || {
                    if let Some(cpu) = cpu {
                        pin_current_thread(cpu);
                    }
                    worker_loop(worker_rx, worker_profile);
                }) {
                Ok(handle) => threads.push(handle),
                Err(err) => {
                    drop(tx);
                    for handle in threads {
                        let _ = handle.join();
                    }
                    return Err(CodecError::ThreadSpawn(err));
                }
            }
        }

        Ok(Self {
            tx: Some(tx),
            threads,
            profile,
        })
    }

    pub fn profiler(&self) -> &CodecProfile {
        &self.profile
    }

    pub fn profile(&self) -> &ProfileRuntime {
        self.profile.runtime()
    }

    pub fn profile_snapshot(&self) -> ProfileSnapshot {
        self.profile.snapshot()
    }

    pub fn profile_snapshot_and_reset(&self) -> ProfileSnapshot {
        self.profile.snapshot_and_reset()
    }

    pub fn reset_profile(&self) {
        self.profile.reset_metrics();
    }

    /// Submit a decode request, blocking only when the bounded queue is full.
    pub fn submit(&self, request: DecodeRequest) -> CodecResult<DecodeFuture> {
        let queued_at = self.profile.record_submit(CodecSubmitKind::Blocking);
        let (work, future) = decode_work(request, queued_at);
        let Some(tx) = self.tx.as_ref() else {
            self.profile.record_submit_error();
            return Err(CodecError::Shutdown);
        };
        if tx.send(work).is_err() {
            self.profile.record_submit_error();
            return Err(CodecError::Shutdown);
        }
        Ok(future)
    }

    /// Submit without waiting for queue capacity.
    pub fn try_submit(&self, request: DecodeRequest) -> CodecResult<DecodeFuture> {
        let queued_at = self.profile.record_submit(CodecSubmitKind::Try);
        let (work, future) = decode_work(request, queued_at);
        let Some(tx) = self.tx.as_ref() else {
            self.profile.record_submit_error();
            return Err(CodecError::Shutdown);
        };
        if let Err(err) = tx.try_send(work) {
            self.profile.record_submit_error();
            return Err(match err {
                flume::TrySendError::Full(_) => CodecError::QueueFull {
                    capacity: tx.capacity().unwrap_or(0),
                },
                flume::TrySendError::Disconnected(_) => CodecError::Shutdown,
            });
        }
        Ok(future)
    }

    /// Async submission for the scheduler: awaiting this only means the
    /// bounded queue accepted the command. Await the returned future for decode
    /// completion.
    pub async fn submit_async(&self, request: DecodeRequest) -> CodecResult<DecodeFuture> {
        let queued_at = self.profile.record_submit(CodecSubmitKind::Async);
        let (work, future) = decode_work(request, queued_at);
        let Some(tx) = self.tx.as_ref() else {
            self.profile.record_submit_error();
            return Err(CodecError::Shutdown);
        };

        match tx.try_send(work) {
            Ok(()) => return Ok(future),
            Err(flume::TrySendError::Full(work)) => {
                if tx.send_async(work).await.is_err() {
                    self.profile.record_submit_error();
                    return Err(CodecError::Shutdown);
                }
            }
            Err(flume::TrySendError::Disconnected(_)) => {
                self.profile.record_submit_error();
                return Err(CodecError::Shutdown);
            }
        }
        Ok(future)
    }
}

impl Drop for DecodePool {
    fn drop(&mut self) {
        self.tx.take();
        while let Some(handle) = self.threads.pop() {
            if handle.join().is_err() {
                eprintln!("[codecs] decode worker panicked during shutdown");
            }
        }
    }
}

fn worker_loop(rx: flume::Receiver<DecodeWork>, profile: CodecProfile) {
    while let Ok(work) = rx.recv() {
        // One worker is one OS thread; each chunk is decoded serially.
        complete_work(work, &profile);
    }
}

fn complete_work(work: DecodeWork, profile: &CodecProfile) {
    let queued_at = work.queued_at;
    let DecodeRequest {
        codec,
        encoded,
        expected_size,
        output,
    } = work.request;
    let encoded_bytes = encoded.len();
    let work_profile = profile.start_work();
    let mut panicked = false;
    let result = match panic::catch_unwind(AssertUnwindSafe(|| match output {
        DecodeOutput::Allocate => DecodeRunner::decode_borrowed_to_vec(
            codec.as_ref(),
            &encoded,
            Vec::new(),
            expected_size,
        ),
        DecodeOutput::ReuseInitialized(output) => {
            DecodeRunner::decode_to_initialized_vec(codec.as_ref(), &encoded, output, expected_size)
        }
        DecodeOutput::ReuseCapacity(output) => {
            DecodeRunner::decode_to_capacity_vec(codec.as_ref(), &encoded, output, expected_size)
        }
    })) {
        Ok(result) => result,
        Err(_) => {
            panicked = true;
            Err(CodecError::WorkerPanic {
                codec: codec.name().to_string(),
            })
        }
    };
    let decoded_bytes = result.as_ref().ok().map(Vec::len);
    work_profile.record(
        queued_at,
        encoded_bytes,
        decoded_bytes,
        result.is_err() && !panicked,
        panicked,
    );

    let _ = work.reply.send(result);
}

fn resolve_cpu_affinity(config: &DecodePoolConfig) -> CodecResult<Vec<usize>> {
    let Some(requested_cpus) = &config.cpus else {
        return Ok(Vec::new());
    };

    let Some(core_ids) = core_affinity::get_core_ids() else {
        return Err(CodecError::InvalidConfig(
            "CPU affinity requested but core ids are unavailable".to_string(),
        ));
    };

    let available = core_ids
        .iter()
        .map(|core_id| core_id.id)
        .collect::<BTreeSet<_>>();

    for cpu in requested_cpus {
        if !available.contains(cpu) {
            return Err(CodecError::InvalidConfig(format!(
                "CPU id {cpu} is not available"
            )));
        }
    }

    Ok(requested_cpus.clone())
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
    #[cfg(feature = "profile")]
    use crate::codecs::profile::test_metrics;
    use crate::codecs::{sealed, ChunkCodec, UncompressedCodec, UnsupportedCodec};
    #[cfg(feature = "profile")]
    use crate::profile::{ProfileMetricId, ProfileRegistry, ProfileRuntime};

    #[cfg(feature = "profile")]
    static PROFILE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    #[cfg(feature = "profile")]
    const PROFILE_ENV_KEYS: &[&str] = &[
        "SCDATA_PROFILE",
        "SCDATA_PROFILE_LABEL",
        "SCDATA_PROFILE_COMPONENTS",
        "SCDATA_PROFILE_COMPONENT_ENABLE",
        "SCDATA_PROFILE_COMPONENT_DISABLE",
        "SCDATA_PROFILE_SCOPES",
        "SCDATA_PROFILE_SCOPE_ENABLE",
        "SCDATA_PROFILE_SCOPE_DISABLE",
    ];

    fn make_pool() -> DecodePool {
        DecodePool::new(DecodePoolConfig {
            num_workers: 2,
            queue_capacity: 8,
            cpus: None,
        })
        .expect("create decode pool")
    }

    #[cfg(feature = "profile")]
    fn make_profiled_pool(profile: CodecProfile) -> DecodePool {
        DecodePool::with_codec_profile(
            DecodePoolConfig {
                num_workers: 2,
                queue_capacity: 8,
                cpus: None,
            },
            profile,
        )
        .expect("create decode pool")
    }

    #[cfg(feature = "profile")]
    fn enabled_profile(label: &'static str) -> CodecProfile {
        CodecProfile::enabled(label)
    }

    #[cfg(feature = "profile")]
    fn metric(snapshot: &ProfileSnapshot, id: ProfileMetricId) -> u64 {
        snapshot.metric_value(id).unwrap_or(0)
    }

    #[cfg(feature = "profile")]
    fn with_profile_env_enabled<T>(f: impl FnOnce() -> T) -> T {
        let _guard = PROFILE_ENV_LOCK.lock().unwrap();
        let saved = PROFILE_ENV_KEYS
            .iter()
            .map(|key| (*key, std::env::var_os(key)))
            .collect::<Vec<_>>();

        std::env::set_var("SCDATA_PROFILE", "1");
        for key in PROFILE_ENV_KEYS
            .iter()
            .copied()
            .filter(|key| *key != "SCDATA_PROFILE")
        {
            std::env::remove_var(key);
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        for (key, value) in saved {
            if let Some(value) = value {
                std::env::set_var(key, value);
            } else {
                std::env::remove_var(key);
            }
        }

        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    #[derive(Debug)]
    struct VecFastPathCodec {
        called: Arc<std::sync::atomic::AtomicBool>,
    }

    impl sealed::Sealed for VecFastPathCodec {}

    impl ChunkCodec for VecFastPathCodec {
        fn name(&self) -> &str {
            "vec-fast"
        }

        fn decode(&self, _encoded: &[u8], _expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
            panic!("decode should not be called when Allocate uses vec fastpath");
        }

        fn decode_to_vec_grow(
            &self,
            encoded: &[u8],
            mut output: Vec<u8>,
            expected_size: Option<usize>,
        ) -> CodecResult<Vec<u8>> {
            self.called.store(true, std::sync::atomic::Ordering::SeqCst);
            if let Some(expected_size) = expected_size {
                if expected_size != encoded.len() {
                    return Err(CodecError::SizeMismatch {
                        codec: self.name().to_string(),
                        expected: expected_size,
                        actual: encoded.len(),
                    });
                }
            }
            output.clear();
            output.extend_from_slice(encoded);
            Ok(output)
        }
    }

    #[cfg(feature = "profile")]
    #[derive(Debug)]
    struct PanicCodec;

    #[cfg(feature = "profile")]
    impl sealed::Sealed for PanicCodec {}

    #[cfg(feature = "profile")]
    impl ChunkCodec for PanicCodec {
        fn name(&self) -> &str {
            "panic"
        }

        fn decode(&self, _encoded: &[u8], _expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
            panic!("intentional decode panic");
        }
    }

    #[cfg(feature = "profile")]
    #[derive(Debug)]
    struct BlockingCodec {
        started: std::sync::mpsc::Sender<()>,
        release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }

    #[cfg(feature = "profile")]
    impl sealed::Sealed for BlockingCodec {}

    #[cfg(feature = "profile")]
    impl ChunkCodec for BlockingCodec {
        fn name(&self) -> &str {
            "blocking"
        }

        fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
            let _ = self.started.send(());
            let (released, condvar) = &*self.release;
            let mut released = released.lock().unwrap();
            while !*released {
                released = condvar.wait(released).unwrap();
            }
            let codec = UncompressedCodec;
            codec.decode(encoded, expected_size)
        }
    }

    #[test]
    fn config_rejects_zero_workers() {
        let config = DecodePoolConfig {
            num_workers: 0,
            ..DecodePoolConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_without_cpus_does_not_request_affinity() {
        let config = DecodePoolConfig::default();
        assert_eq!(
            resolve_cpu_affinity(&config).expect("resolve default affinity"),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn decode_pool_runs_uncompressed_requests() {
        let pool = make_pool();
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let request = DecodeRequest::new(codec, b"abcdef".to_vec()).with_expected_size(6);

        let decoded = pool
            .submit(request)
            .expect("submit")
            .blocking_recv()
            .expect("decode");

        assert_eq!(&decoded, b"abcdef");
    }

    #[cfg(feature = "profile")]
    #[test]
    fn decode_pool_records_profile_for_successful_work() {
        let profile = enabled_profile("codec-success");
        let pool = make_profiled_pool(profile.clone());
        let round = profile.start();
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let request = DecodeRequest::new(codec, b"abcdef".to_vec()).with_expected_size(6);

        let decoded = pool
            .submit(request)
            .expect("submit")
            .blocking_recv()
            .expect("decode");

        assert_eq!(&decoded, b"abcdef");
        let snapshot = pool.profiler().snapshot();
        assert_eq!(snapshot.label, "codec-success");
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_BLOCKING_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_ERRORS), 0);
        assert_eq!(metric(&snapshot, test_metrics::WORK_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::ENCODED_BYTES), 6);
        assert_eq!(metric(&snapshot, test_metrics::DECODED_BYTES), 6);
        assert_eq!(metric(&snapshot, test_metrics::WORK_ERRORS), 0);
        assert_eq!(metric(&snapshot, test_metrics::WORK_PANICS), 0);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn decode_pool_records_profile_for_async_submit() {
        let profile = enabled_profile("codec-async");
        let pool = make_profiled_pool(profile.clone());
        let round = profile.start();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("tokio runtime");
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let request = DecodeRequest::new(codec, b"abcdef".to_vec()).with_expected_size(6);

        let future = runtime
            .block_on(pool.submit_async(request))
            .expect("async submit");
        let decoded = future.blocking_recv().expect("decode");

        assert_eq!(&decoded, b"abcdef");
        let snapshot = pool.profiler().snapshot();
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_ASYNC_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::WORK_CALLS), 1);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn decode_pool_from_env_auto_records_and_reset_keeps_recording() {
        with_profile_env_enabled(|| {
            let pool = make_pool();
            assert!(pool.profile().is_recording());

            let codec: SharedCodec = Arc::new(UncompressedCodec);
            let request = DecodeRequest::new(codec, b"abcdef".to_vec()).with_expected_size(6);
            let decoded = pool
                .submit(request)
                .expect("submit")
                .blocking_recv()
                .expect("decode");
            assert_eq!(&decoded, b"abcdef");

            let first_snapshot = pool.profile_snapshot_and_reset();
            assert_eq!(metric(&first_snapshot, test_metrics::SUBMIT_CALLS), 1);
            assert_eq!(metric(&first_snapshot, test_metrics::WORK_CALLS), 1);
            assert!(pool.profile().is_recording());
            assert_eq!(
                metric(&pool.profile_snapshot(), test_metrics::SUBMIT_CALLS),
                0
            );

            let codec: SharedCodec = Arc::new(UncompressedCodec);
            let request = DecodeRequest::new(codec, b"ghijkl".to_vec()).with_expected_size(6);
            let decoded = pool
                .submit(request)
                .expect("second submit")
                .blocking_recv()
                .expect("second decode");
            assert_eq!(&decoded, b"ghijkl");
            pool.reset_profile();
            assert!(pool.profile().is_recording());
            assert_eq!(
                metric(&pool.profile_snapshot(), test_metrics::SUBMIT_CALLS),
                0
            );
        });
    }

    #[cfg(feature = "profile")]
    #[test]
    fn codec_profile_snapshot_and_reset_keeps_manual_round_recording() {
        let profile = enabled_profile("codec-manual-reset");
        let pool = make_profiled_pool(profile.clone());
        let round = profile.start();
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let request = DecodeRequest::new(codec, b"abcdef".to_vec()).with_expected_size(6);

        let decoded = pool
            .submit(request)
            .expect("submit")
            .blocking_recv()
            .expect("decode");

        assert_eq!(&decoded, b"abcdef");
        let reset = pool.profile_snapshot_and_reset();
        assert_eq!(metric(&reset, test_metrics::SUBMIT_CALLS), 1);
        assert_eq!(metric(&reset, test_metrics::WORK_CALLS), 1);
        assert!(pool.profile().is_recording());
        assert_eq!(
            metric(&pool.profile_snapshot(), test_metrics::SUBMIT_CALLS),
            0
        );

        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let request = DecodeRequest::new(codec, b"ghijkl".to_vec()).with_expected_size(6);
        let decoded = pool
            .submit(request)
            .expect("second submit")
            .blocking_recv()
            .expect("second decode");
        assert_eq!(&decoded, b"ghijkl");
        pool.reset_profile();
        assert!(pool.profile().is_recording());
        assert_eq!(
            metric(&pool.profile_snapshot(), test_metrics::SUBMIT_CALLS),
            0
        );
        round.end();
    }

    #[test]
    fn decode_pool_returns_codec_errors() {
        let pool = make_pool();
        let codec: SharedCodec = Arc::new(UnsupportedCodec::new("blosc"));
        let request = DecodeRequest::new(codec, b"payload".to_vec());

        let err = pool
            .submit(request)
            .expect("submit")
            .blocking_recv()
            .expect_err("unsupported codec");

        assert!(matches!(err, CodecError::Unsupported { codec } if codec == "blosc"));
    }

    #[test]
    fn decode_pool_allocate_uses_codec_vec_fastpath() {
        let pool = make_pool();
        let called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let codec: SharedCodec = Arc::new(VecFastPathCodec {
            called: Arc::clone(&called),
        });
        let request = DecodeRequest::new(codec, b"abcdef".to_vec()).with_expected_size(6);

        let decoded = pool
            .submit(request)
            .expect("submit")
            .blocking_recv()
            .expect("decode");

        assert_eq!(&decoded, b"abcdef");
        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[cfg(feature = "profile")]
    #[test]
    fn decode_pool_records_profile_for_codec_errors() {
        let profile = enabled_profile("codec-error");
        let pool = make_profiled_pool(profile.clone());
        let round = profile.start();
        let codec: SharedCodec = Arc::new(UnsupportedCodec::new("blosc"));
        let request = DecodeRequest::new(codec, b"payload".to_vec());

        let err = pool
            .try_submit(request)
            .expect("submit")
            .blocking_recv()
            .expect_err("unsupported codec");

        assert!(matches!(err, CodecError::Unsupported { codec } if codec == "blosc"));
        let snapshot = pool.profiler().snapshot();
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_TRY_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_ERRORS), 0);
        assert_eq!(metric(&snapshot, test_metrics::WORK_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::ENCODED_BYTES), 7);
        assert_eq!(metric(&snapshot, test_metrics::DECODED_BYTES), 0);
        assert_eq!(metric(&snapshot, test_metrics::WORK_ERRORS), 1);
        assert_eq!(metric(&snapshot, test_metrics::WORK_PANICS), 0);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn decode_pool_records_profile_for_worker_panics() {
        let profile = enabled_profile("codec-panic");
        let pool = make_profiled_pool(profile.clone());
        let round = profile.start();
        let codec: SharedCodec = Arc::new(PanicCodec);
        let request = DecodeRequest::new(codec, b"payload".to_vec());

        let err = pool
            .submit(request)
            .expect("submit")
            .blocking_recv()
            .expect_err("panic is reported");

        assert!(matches!(err, CodecError::WorkerPanic { codec } if codec == "panic"));
        let snapshot = pool.profiler().snapshot();
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_BLOCKING_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::WORK_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::ENCODED_BYTES), 7);
        assert_eq!(metric(&snapshot, test_metrics::DECODED_BYTES), 0);
        assert_eq!(metric(&snapshot, test_metrics::WORK_ERRORS), 0);
        assert_eq!(metric(&snapshot, test_metrics::WORK_PANICS), 1);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn codec_profile_registers_scopes_for_active_runtime() {
        let runtime = ProfileRuntime::enabled_lazy("codec-active-registry", ProfileRegistry::new);
        let round = runtime.start();
        let pool = make_profiled_pool(CodecProfile::from_runtime(runtime.clone()));
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let request = DecodeRequest::new(codec, b"abcdef".to_vec()).with_expected_size(6);

        let decoded = pool
            .submit(request)
            .expect("submit")
            .blocking_recv()
            .expect("decode");

        assert_eq!(&decoded, b"abcdef");
        let snapshot = runtime.snapshot();
        assert_eq!(metric(&snapshot, test_metrics::SUBMIT_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::WORK_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::DECODED_BYTES), 6);
        round.end();
    }

    #[cfg(feature = "profile")]
    #[test]
    fn decode_pool_does_not_record_queue_wait_across_rounds() {
        let profile = enabled_profile("codec-queue-round");
        let pool = DecodePool::with_codec_profile(
            DecodePoolConfig {
                num_workers: 1,
                queue_capacity: 8,
                cpus: None,
            },
            profile.clone(),
        )
        .expect("create decode pool");
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));

        let round = profile.start();
        let blocking_codec: SharedCodec = Arc::new(BlockingCodec {
            started: started_tx,
            release: Arc::clone(&release),
        });
        let first = pool
            .submit(DecodeRequest::new(blocking_codec, b"first".to_vec()).with_expected_size(5))
            .expect("submit blocking request");
        started_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("worker should start blocking request");

        let second = pool
            .submit(
                DecodeRequest::new(Arc::new(UncompressedCodec), b"second".to_vec())
                    .with_expected_size(6),
            )
            .expect("submit queued request");
        let round_one = round.end();
        assert_eq!(metric(&round_one, test_metrics::SUBMIT_CALLS), 2);

        let round = profile.start();
        {
            let (released, condvar) = &*release;
            *released.lock().unwrap() = true;
            condvar.notify_all();
        }

        assert_eq!(first.blocking_recv().expect("first decode"), b"first");
        assert_eq!(second.blocking_recv().expect("second decode"), b"second");

        let snapshot = profile.snapshot();
        assert_eq!(metric(&snapshot, test_metrics::WORK_CALLS), 1);
        assert_eq!(metric(&snapshot, test_metrics::QUEUE_WAIT_NS), 0);
        round.end();
    }

    #[test]
    fn decode_request_shares_encoded_payload() {
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let encoded: Arc<[u8]> = Arc::from(&b"payload"[..]);
        let request = DecodeRequest::new(codec, Arc::clone(&encoded));

        assert!(Arc::ptr_eq(&request.encoded, &encoded));
    }

    #[test]
    fn decode_pool_fills_caller_owned_output_buffer() {
        let pool = make_pool();
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let output = vec![0u8; 6];
        let output_ptr = output.as_ptr();
        let request = DecodeRequest::new(codec, b"abcdef".to_vec())
            .with_expected_size(6)
            .with_reuse_initialized_output(output);

        let decoded = pool
            .submit(request)
            .expect("submit")
            .blocking_recv()
            .expect("decode");

        assert_eq!(&decoded, b"abcdef");
        assert_eq!(decoded.as_ptr(), output_ptr);
    }

    #[test]
    fn decode_pool_fills_uninitialized_capacity_output_buffer() {
        let pool = make_pool();
        let codec: SharedCodec = Arc::new(UncompressedCodec);
        let output = Vec::with_capacity(6);
        let output_ptr = output.as_ptr();
        let request = DecodeRequest::new(codec, b"abcdef".to_vec())
            .with_expected_size(6)
            .with_reuse_capacity_output(output);

        let decoded = pool
            .submit(request)
            .expect("submit")
            .blocking_recv()
            .expect("decode");

        assert_eq!(&decoded, b"abcdef");
        assert_eq!(decoded.as_ptr(), output_ptr);
        assert_eq!(decoded.len(), 6);
    }
}
