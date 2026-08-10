//! Byte-keyed store backends for reading (and directory-only writing).

mod directory;
mod zip;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use directory::DirectoryStore;
pub(crate) use directory::DirectoryTransaction;
pub use zip::ZipStore;

use crate::error::{Error, Result};

/// Maximum value size accepted by the raw [`ByteStore::read`] convenience.
pub const DEFAULT_MAX_VALUE_SIZE: usize = 1024 * 1024 * 1024;

/// An immutable value exposed as a bounded view of a positioned file.
///
/// The file handle pins the underlying file generation. Callers must add
/// `base_offset` to value-relative offsets and must never read past `len`.
#[derive(Debug, Clone)]
pub struct PositionedValue {
    file: Arc<File>,
    base_offset: u64,
    len: u64,
}

impl PositionedValue {
    pub(crate) fn new(file: File, base_offset: u64, len: u64) -> Self {
        Self {
            file: Arc::new(file),
            base_offset,
            len,
        }
    }

    pub fn file(&self) -> &Arc<File> {
        &self.file
    }

    pub const fn base_offset(&self) -> u64 {
        self.base_offset
    }

    pub const fn len(&self) -> u64 {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Read-only key/value store over relative POSIX-style keys (`meta.json`, `data/0`).
///
/// Published values must remain immutable while a matrix reader is using the
/// store, because bounded decoders may issue multiple ranges for one value.
pub trait ByteStore: Send + Sync {
    fn len(&self, key: &str) -> Result<u64>;

    fn is_empty(&self, key: &str) -> Result<bool> {
        Ok(self.len(key)? == 0)
    }

    fn read(&self, key: &str) -> Result<Vec<u8>> {
        self.read_limited(key, DEFAULT_MAX_VALUE_SIZE)
    }

    fn read_limited(&self, key: &str, maximum: usize) -> Result<Vec<u8>> {
        let declared = self.len(key)?;
        let declared_usize = usize::try_from(declared)
            .map_err(|_| Error::corrupt("store value", "declared size exceeds usize"))?;
        if declared_usize > maximum {
            return Err(Error::corrupt(
                "store value",
                format!("value '{key}' has {declared_usize} bytes, limit is {maximum}"),
            ));
        }
        let bytes = self.read_range(key, 0, declared_usize)?;
        if bytes.len() != declared_usize {
            return Err(Error::corrupt(
                "store value",
                format!(
                    "value '{key}' returned {} bytes, expected {declared_usize}",
                    bytes.len()
                ),
            ));
        }
        Ok(bytes)
    }

    fn read_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>>;

    fn exists(&self, key: &str) -> Result<bool>;

    /// Report true only when a non-zero offset does not replay the value prefix.
    fn supports_efficient_range_reads(&self, key: &str) -> Result<bool> {
        validate_key(key)?;
        Ok(false)
    }

    /// Return a generation-pinned positioned view when the backend can expose
    /// this value without replaying or materializing it.
    fn open_positioned(&self, key: &str) -> Result<Option<PositionedValue>> {
        validate_key(key)?;
        Ok(None)
    }
}

/// Mutable store used by writers. Only [`DirectoryStore`] implements this.
pub trait ByteStoreMut: ByteStore {
    fn write(&mut self, key: &str, bytes: &[u8]) -> Result<()>;
}

/// Where a matrix store lives: a directory, or a prefix inside a zip archive.
#[derive(Debug, Clone)]
pub enum StoreLocation {
    Dir(PathBuf),
    Zip { archive: PathBuf, prefix: String },
}

impl StoreLocation {
    pub fn dir(path: impl Into<PathBuf>) -> Self {
        Self::Dir(path.into())
    }

    pub fn zip(archive: impl Into<PathBuf>, prefix: impl Into<String>) -> Self {
        Self::Zip {
            archive: archive.into(),
            prefix: normalize_prefix(prefix.into()),
        }
    }

    pub fn open(&self) -> Result<Arc<dyn ByteStore>> {
        match self {
            Self::Dir(path) => Ok(Arc::new(DirectoryStore::open(path)?)),
            Self::Zip { archive, prefix } => Ok(Arc::new(ZipStore::open(archive, prefix)?)),
        }
    }
}

impl From<&Path> for StoreLocation {
    fn from(path: &Path) -> Self {
        Self::dir(path)
    }
}

impl From<PathBuf> for StoreLocation {
    fn from(path: PathBuf) -> Self {
        Self::dir(path)
    }
}

impl From<&PathBuf> for StoreLocation {
    fn from(path: &PathBuf) -> Self {
        Self::dir(path)
    }
}

impl From<&str> for StoreLocation {
    fn from(path: &str) -> Self {
        Self::dir(path)
    }
}

impl From<String> for StoreLocation {
    fn from(path: String) -> Self {
        Self::dir(path)
    }
}

impl From<&String> for StoreLocation {
    fn from(path: &String) -> Self {
        Self::dir(path)
    }
}

pub(crate) fn normalize_prefix(prefix: String) -> String {
    prefix.trim_matches('/').to_string()
}

pub(crate) fn join_key(prefix: &str, key: &str) -> String {
    let key = key.trim_start_matches('/');
    if prefix.is_empty() {
        key.to_string()
    } else if key.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}/{key}")
    }
}

pub(crate) const META_FILE_NAME: &str = "meta.json";

pub fn chunk_key(dir: &str, id: u64) -> String {
    if dir.is_empty() {
        id.to_string()
    } else {
        format!("{dir}/{id}")
    }
}

pub(crate) fn validate_key(key: &str) -> Result<()> {
    if key.is_empty()
        || key.starts_with('/')
        || key
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(Error::invalid_argument(format!(
            "invalid store key `{key}`"
        )));
    }
    Ok(())
}

pub(crate) fn validate_prefix(prefix: &str) -> Result<()> {
    if prefix.is_empty() {
        return Ok(());
    }
    validate_key(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UnknownRangeCapability;

    impl ByteStore for UnknownRangeCapability {
        fn len(&self, _key: &str) -> Result<u64> {
            Ok(0)
        }

        fn read_range(&self, _key: &str, _offset: u64, _len: usize) -> Result<Vec<u8>> {
            Ok(Vec::new())
        }

        fn exists(&self, _key: &str) -> Result<bool> {
            Ok(false)
        }
    }

    #[test]
    fn range_read_capability_defaults_to_conservative() {
        assert!(!UnknownRangeCapability
            .supports_efficient_range_reads("value")
            .unwrap());
    }
}
