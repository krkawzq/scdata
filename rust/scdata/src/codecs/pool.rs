use std::collections::BTreeSet;
use std::future::Future;
use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread;

use tokio::sync::oneshot;

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
    pub output: Option<Vec<u8>>,
}

impl DecodeRequest {
    pub fn new(codec: SharedCodec, encoded: impl Into<Arc<[u8]>>) -> Self {
        Self {
            codec,
            encoded: encoded.into(),
            expected_size: None,
            output: None,
        }
    }

    pub fn from_spec(spec: &CodecSpec, encoded: impl Into<Arc<[u8]>>) -> Self {
        Self::new(spec.build(), encoded)
    }

    pub fn with_expected_size(mut self, expected_size: usize) -> Self {
        self.expected_size = Some(expected_size);
        self
    }

    /// Provide caller-owned output memory for the worker to fill.
    ///
    /// When the final decoded size is known, the vector's capacity is used as
    /// writable memory so callers can pass `Vec::with_capacity(size)` without
    /// zero-filling. Without an exact size hint, the vector's current length is
    /// the writable view. Decoding never reallocates this buffer behind the
    /// caller's back.
    pub fn with_output_buffer(mut self, output: Vec<u8>) -> Self {
        self.output = Some(output);
        self
    }
}

#[derive(Debug)]
struct DecodeWork {
    request: DecodeRequest,
    reply: oneshot::Sender<CodecResult<Vec<u8>>>,
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

fn decode_work(request: DecodeRequest) -> (DecodeWork, DecodeFuture) {
    let (reply, rx) = oneshot::channel();
    let work = DecodeWork { request, reply };
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
}

impl DecodePool {
    pub fn new(config: DecodePoolConfig) -> CodecResult<Self> {
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

            match thread::Builder::new()
                .name(format!("decode-wrk-{worker_idx}"))
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
                    return Err(CodecError::ThreadSpawn(err));
                }
            }
        }

        Ok(Self {
            tx: Some(tx),
            threads,
        })
    }

    /// Submit a decode request, blocking only when the bounded queue is full.
    pub fn submit(&self, request: DecodeRequest) -> CodecResult<DecodeFuture> {
        let (work, future) = decode_work(request);
        let tx = self.tx.as_ref().ok_or(CodecError::Shutdown)?;
        tx.send(work).map_err(|_| CodecError::Shutdown)?;
        Ok(future)
    }

    /// Submit without waiting for queue capacity.
    pub fn try_submit(&self, request: DecodeRequest) -> CodecResult<DecodeFuture> {
        let (work, future) = decode_work(request);
        let tx = self.tx.as_ref().ok_or(CodecError::Shutdown)?;
        tx.try_send(work).map_err(|err| match err {
            flume::TrySendError::Full(_) => CodecError::QueueFull {
                capacity: tx.capacity().unwrap_or(0),
            },
            flume::TrySendError::Disconnected(_) => CodecError::Shutdown,
        })?;
        Ok(future)
    }

    /// Async submission for the scheduler: awaiting this only means the
    /// bounded queue accepted the command. Await the returned future for decode
    /// completion.
    pub async fn submit_async(&self, request: DecodeRequest) -> CodecResult<DecodeFuture> {
        let (work, future) = decode_work(request);
        let tx = self.tx.as_ref().ok_or(CodecError::Shutdown)?;
        tx.send_async(work)
            .await
            .map_err(|_| CodecError::Shutdown)?;
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

fn worker_loop(rx: flume::Receiver<DecodeWork>) {
    while let Ok(work) = rx.recv() {
        // One worker is one OS thread; each chunk is decoded serially.
        complete_work(work);
    }
}

fn complete_work(work: DecodeWork) {
    let DecodeRequest {
        codec,
        encoded,
        expected_size,
        output,
    } = work.request;
    let result = panic::catch_unwind(AssertUnwindSafe(|| match output {
        Some(output) => codec.decode_to_vec(&encoded, output, expected_size),
        None => codec.decode(&encoded, expected_size),
    }))
    .unwrap_or_else(|_| {
        Err(CodecError::WorkerPanic {
            codec: codec.name().to_string(),
        })
    });

    let _ = work.reply.send(result);
}

fn resolve_cpu_affinity(config: &DecodePoolConfig) -> CodecResult<Vec<usize>> {
    let Some(core_ids) = core_affinity::get_core_ids() else {
        if config.cpus.is_none() {
            return Ok(Vec::new());
        }
        return Err(CodecError::InvalidConfig(
            "CPU affinity requested but core ids are unavailable".to_string(),
        ));
    };

    let available = core_ids
        .iter()
        .map(|core_id| core_id.id)
        .collect::<BTreeSet<_>>();

    let cpus = match &config.cpus {
        Some(cpus) => {
            for cpu in cpus {
                if !available.contains(cpu) {
                    return Err(CodecError::InvalidConfig(format!(
                        "CPU id {cpu} is not available"
                    )));
                }
            }
            cpus.clone()
        }
        None => core_ids.into_iter().map(|core_id| core_id.id).collect(),
    };

    Ok(cpus)
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
    use crate::codecs::{UncompressedCodec, UnsupportedCodec};

    fn make_pool() -> DecodePool {
        DecodePool::new(DecodePoolConfig {
            num_workers: 2,
            queue_capacity: 8,
            cpus: None,
        })
        .expect("create decode pool")
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
            .with_output_buffer(output);

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
            .with_output_buffer(output);

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
