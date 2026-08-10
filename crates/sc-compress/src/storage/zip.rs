use std::fs::File;
use std::io::{self, copy, sink, Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use zip::{CompressionMethod, ZipArchive};

use crate::error::{Error, Result};
use crate::storage::{
    join_key, normalize_prefix, validate_key, validate_prefix, ByteStore, PositionedValue,
};

/// Read-only store rooted at an optional prefix inside a zip archive.
///
/// Entry metadata is indexed once at open. Stored ranges use positioned reads;
/// compressed full reads construct an independent archive cursor over the
/// generation-pinned file descriptor. Concurrent chunk readers therefore do
/// not contend on a single `ZipArchive` seek cursor.
pub struct ZipStore {
    file: Arc<File>,
    archive: ZipArchive<SharedFileCursor>,
    entries: Vec<Option<ZipEntry>>,
    prefix: String,
}

#[derive(Debug, Clone)]
struct ZipEntry {
    index: usize,
    size: u64,
    data_start: u64,
    compression: CompressionMethod,
    encrypted: bool,
}

impl ZipStore {
    pub fn open(archive: impl AsRef<Path>, prefix: impl Into<String>) -> Result<Self> {
        let path = archive.as_ref();
        let file = File::open(path).map_err(|error| Error::Path {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        let file_len = file.metadata()?.len();
        let file = Arc::new(file);
        let prefix = normalize_prefix(prefix.into());
        validate_prefix(&prefix)?;

        let mut archive = ZipArchive::new(SharedFileCursor::new(Arc::clone(&file), file_len))?;
        let mut entries = Vec::new();
        entries.try_reserve_exact(archive.len())?;
        entries.resize_with(archive.len(), || None);
        for (index, entry_slot) in entries.iter_mut().enumerate() {
            let name = archive
                .name_for_index(index)
                .ok_or_else(|| Error::corrupt("zip archive", "missing entry name"))?
                .to_owned();
            if !entry_is_under_prefix(&name, &prefix) {
                continue;
            }
            let entry = archive.by_index_raw(index)?;
            let data_start = entry.data_start();
            let data_end = data_start
                .checked_add(entry.compressed_size())
                .ok_or_else(|| Error::corrupt(&name, "zip entry extent overflow"))?;
            if data_end > file_len {
                return Err(Error::corrupt(
                    &name,
                    format!("zip entry data ends at {data_end}, archive has {file_len} bytes"),
                ));
            }
            if !entry.encrypted()
                && entry.compression() == CompressionMethod::Stored
                && entry.compressed_size() != entry.size()
            {
                return Err(Error::corrupt(
                    &name,
                    "stored zip entry has different encoded and decoded sizes",
                ));
            }
            *entry_slot = Some(ZipEntry {
                index,
                size: entry.size(),
                data_start,
                compression: entry.compression(),
                encrypted: entry.encrypted(),
            });
        }

        Ok(Self {
            file,
            archive,
            entries,
            prefix,
        })
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    fn entry_name(&self, key: &str) -> Result<String> {
        validate_key(key)?;
        Ok(join_key(&self.prefix, key))
    }

    fn entry(&self, key: &str) -> Result<ZipEntry> {
        let name = self.entry_name(key)?;
        let index = self
            .archive
            .index_for_name(&name)
            .ok_or_else(|| Error::not_found(key))?;
        self.entries
            .get(index)
            .and_then(Option::as_ref)
            .cloned()
            .ok_or_else(|| Error::not_found(key))
    }

    fn archive(&self) -> ZipArchive<SharedFileCursor> {
        self.archive.clone()
    }

    fn read_full_entry(&self, key: &str, entry: &ZipEntry, buffer: &mut [u8]) -> Result<()> {
        let mut archive = self.archive();
        let mut file = archive.by_index(entry.index)?;
        if file.size() != entry.size || file.compression() != entry.compression {
            return Err(Error::corrupt(
                key,
                "zip central-directory entry changed while store was open",
            ));
        }
        file.read_exact(buffer)?;
        require_eof(&mut file, key)
    }

    fn read_stored_range(
        &self,
        key: &str,
        entry: &ZipEntry,
        offset: u64,
        buffer: &mut [u8],
    ) -> Result<()> {
        let absolute = entry
            .data_start
            .checked_add(offset)
            .ok_or_else(|| Error::corrupt(key, "zip stored range offset overflow"))?;
        self.file.read_exact_at(buffer, absolute)?;
        Ok(())
    }
}

impl ByteStore for ZipStore {
    fn len(&self, key: &str) -> Result<u64> {
        Ok(self.entry(key)?.size)
    }

    fn read_limited(&self, key: &str, maximum: usize) -> Result<Vec<u8>> {
        let entry = self.entry(key)?;
        let declared = usize::try_from(entry.size)
            .map_err(|_| Error::corrupt("store value", "declared size exceeds usize"))?;
        if declared > maximum {
            return Err(Error::corrupt(
                "store value",
                format!("value '{key}' has {declared} bytes, limit is {maximum}"),
            ));
        }
        let mut buffer = zeroed_buffer(declared)?;
        self.read_full_entry(key, &entry, &mut buffer)?;
        Ok(buffer)
    }

    fn read_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let entry = self.entry(key)?;
        if offset > entry.size {
            return Err(Error::corrupt(
                "zip entry range",
                format!(
                    "offset {offset} past end of {}-byte entry '{key}'",
                    entry.size
                ),
            ));
        }
        let available = usize::try_from(entry.size - offset)
            .map_err(|_| Error::corrupt("zip entry range", "available length exceeds usize"))?;
        let to_read = len.min(available);
        let mut buffer = zeroed_buffer(to_read)?;
        if to_read == 0 {
            return Ok(buffer);
        }
        if offset == 0 && u64::try_from(to_read).ok() == Some(entry.size) {
            self.read_full_entry(key, &entry, &mut buffer)?;
        } else if entry.compression == CompressionMethod::Stored && !entry.encrypted {
            self.read_stored_range(key, &entry, offset, &mut buffer)?;
        } else {
            let mut archive = self.archive();
            let mut file = archive.by_index(entry.index)?;
            if offset > 0 {
                let skipped = copy(&mut (&mut file).take(offset), &mut sink())?;
                if skipped != offset {
                    return Err(Error::corrupt(
                        "zip entry range",
                        format!(
                            "entry '{key}' ended after skipping {skipped} of {offset} requested bytes"
                        ),
                    ));
                }
            }
            file.read_exact(&mut buffer)?;
            if u64::try_from(to_read)
                .ok()
                .and_then(|to_read| offset.checked_add(to_read))
                == Some(entry.size)
            {
                require_eof(&mut file, key)?;
            }
        }
        Ok(buffer)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let name = self.entry_name(key)?;
        Ok(self
            .archive
            .index_for_name(&name)
            .and_then(|index| self.entries.get(index))
            .is_some_and(Option::is_some))
    }

    fn supports_efficient_range_reads(&self, key: &str) -> Result<bool> {
        let entry = self.entry(key)?;
        Ok(entry.compression == CompressionMethod::Stored && !entry.encrypted)
    }

    fn open_positioned(&self, key: &str) -> Result<Option<PositionedValue>> {
        let entry = self.entry(key)?;
        if entry.compression != CompressionMethod::Stored || entry.encrypted {
            return Ok(None);
        }
        Ok(Some(PositionedValue::new(
            self.file.try_clone()?,
            entry.data_start,
            entry.size,
        )))
    }
}

fn entry_is_under_prefix(name: &str, prefix: &str) -> bool {
    prefix.is_empty()
        || name
            .strip_prefix(prefix)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn require_eof(reader: &mut impl Read, key: &str) -> Result<()> {
    let mut excess = [0u8; 1];
    if reader.read(&mut excess)? != 0 {
        return Err(Error::corrupt(
            "zip entry",
            format!("entry '{key}' exceeds its declared size"),
        ));
    }
    Ok(())
}

fn zeroed_buffer(len: usize) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(len)?;
    buffer.resize(len, 0);
    Ok(buffer)
}

/// Independent logical cursor backed by positioned reads on one immutable
/// archive generation. Clones share file contents, never seek state.
#[derive(Clone)]
struct SharedFileCursor {
    file: Arc<File>,
    position: u64,
    len: u64,
}

impl SharedFileCursor {
    const fn new(file: Arc<File>, len: u64) -> Self {
        Self {
            file,
            position: 0,
            len,
        }
    }
}

impl Read for SharedFileCursor {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let available = self.len.saturating_sub(self.position);
        let to_read = usize::try_from(available)
            .unwrap_or(usize::MAX)
            .min(buffer.len());
        if to_read == 0 {
            return Ok(0);
        }
        let read = self.file.read_at(&mut buffer[..to_read], self.position)?;
        let read_u64 = u64::try_from(read)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "zip read size overflow"))?;
        self.position = self
            .position
            .checked_add(read_u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "zip cursor overflow"))?;
        Ok(read)
    }
}

impl Seek for SharedFileCursor {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let next = match position {
            SeekFrom::Start(position) => i128::from(position),
            SeekFrom::End(delta) => i128::from(self.len) + i128::from(delta),
            SeekFrom::Current(delta) => i128::from(self.position) + i128::from(delta),
        };
        if !(0..=i128::from(u64::MAX)).contains(&next) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid zip cursor seek",
            ));
        }
        self.position = u64::try_from(next)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid zip cursor seek"))?;
        Ok(self.position)
    }
}
