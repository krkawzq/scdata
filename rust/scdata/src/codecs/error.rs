use thiserror::Error;

pub type CodecResult<T> = Result<T, CodecError>;

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("codec {codec} is not implemented yet")]
    Unsupported { codec: String },

    #[error("invalid config for codec {codec}: {message}")]
    InvalidCodecConfig { codec: String, message: String },

    #[error("failed to decode {codec}: {message}")]
    Decode { codec: String, message: String },

    #[error("decoded size mismatch for {codec}: expected {expected} bytes, got {actual} bytes")]
    SizeMismatch {
        codec: String,
        expected: usize,
        actual: usize,
    },

    #[error("output buffer is too small for {codec}: need at least {required} bytes, got {capacity} bytes")]
    OutputTooSmall {
        codec: String,
        required: usize,
        capacity: usize,
    },

    #[error("invalid decode config: {0}")]
    InvalidConfig(String),

    #[error("decode queue is full at capacity {capacity}")]
    QueueFull { capacity: usize },

    #[error("decode queue is shut down")]
    Shutdown,

    #[error("decode worker panicked while running {codec}")]
    WorkerPanic { codec: String },

    #[error("failed to spawn decode worker: {0}")]
    ThreadSpawn(#[source] std::io::Error),
}
