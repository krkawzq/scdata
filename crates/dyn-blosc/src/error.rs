use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitKind {
    DecodedSize,
    BlockSize,
    BlockCount,
}

impl std::fmt::Display for LimitKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecodedSize => formatter.write_str("decoded size"),
            Self::BlockSize => formatter.write_str("block size"),
            Self::BlockCount => formatter.write_str("block count"),
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid Blosc data: {0}")]
    InvalidFormat(String),
    #[error("invalid encoder options: {0}")]
    InvalidOptions(String),
    #[error("unsupported codec: {0}")]
    UnsupportedCodec(String),
    #[error("codec failure: {0}")]
    Codec(String),
    #[error("filter failure: {0}")]
    Filter(String),
    #[error("output buffer too small: need {need}, have {have}")]
    BufferTooSmall { need: usize, have: usize },
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("{kind} {actual} exceeds configured limit {limit}")]
    LimitExceeded {
        kind: LimitKind,
        actual: usize,
        limit: usize,
    },
    #[error("encoded chunk does not match the decoder schema: {0}")]
    SchemaMismatch(String),
    #[error("memory allocation failed while reserving {bytes} bytes")]
    AllocationFailed { bytes: usize },
    #[error("codec worker thread panicked")]
    WorkerPanicked,
}

pub(crate) fn resize_zeroed(buffer: &mut Vec<u8>, len: usize) -> Result<()> {
    if buffer.len() < len {
        let additional = len - buffer.len();
        buffer
            .try_reserve_exact(additional)
            .map_err(|_| Error::AllocationFailed { bytes: len })?;
        buffer.resize(len, 0);
    }
    Ok(())
}

pub(crate) fn reserve_exact(buffer: &mut Vec<u8>, additional: usize) -> Result<()> {
    let requested = buffer.len().checked_add(additional).ok_or_else(|| {
        Error::InvalidArgument("requested allocation size overflows usize".into())
    })?;
    buffer
        .try_reserve_exact(additional)
        .map_err(|_| Error::AllocationFailed { bytes: requested })
}

pub(crate) fn vector_with_capacity<T>(capacity: usize) -> Result<Vec<T>> {
    let bytes = capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(Error::AllocationFailed { bytes: usize::MAX })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| Error::AllocationFailed { bytes })?;
    Ok(values)
}

pub(crate) fn join_workers<'scope>(
    handles: Vec<std::thread::ScopedJoinHandle<'scope, Result<()>>>,
) -> Result<()> {
    let mut result = Ok(());
    for handle in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) if result.is_ok() => result = Err(error),
            Err(_) => result = Err(Error::WorkerPanicked),
            Ok(Err(_)) => {}
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_panics_are_reported_without_unwinding_the_scope() {
        let result = std::thread::scope(|scope| {
            let handles = vec![
                scope.spawn(|| Err(Error::Codec("worker failure".into()))),
                scope.spawn(|| -> Result<()> { panic!("worker panic") }),
            ];
            join_workers(handles)
        });

        assert_eq!(result, Err(Error::WorkerPanicked));
    }
}
