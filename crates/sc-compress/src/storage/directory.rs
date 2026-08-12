use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use rustix::fs::{
    flock, mkdirat, openat, renameat, renameat_with, unlinkat, AtFlags, Dir, FlockOperation, Mode,
    OFlags, RenameFlags, CWD,
};
use rustix::io::{dup, Errno};

use crate::error::{Error, Result};
use crate::io_util::read_meta;
use crate::limits::ReadLimits;
use crate::storage::{
    validate_key, ByteStore, ByteStoreMut, PositionedValue, DEFAULT_MAX_VALUE_SIZE,
};

const DIRECTORY_MODE: Mode = Mode::from_raw_mode(0o755);
const FILE_MODE: Mode = Mode::from_raw_mode(0o644);
static WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Directory-backed store for regular files.
///
/// The root is held by an open directory descriptor, so an opened store stays
/// pinned to one generation across atomic directory replacement. Every key is
/// resolved descriptor-relatively with `O_NOFOLLOW`; path components cannot be
/// swapped to symbolic links between validation and I/O.
#[derive(Debug, Clone)]
pub struct DirectoryStore {
    root: PathBuf,
    generation: Arc<DirectoryGeneration>,
}

pub(crate) struct DirectoryTransaction {
    target: PathBuf,
    target_name: OsString,
    parent: File,
    expected_target: Option<TargetSnapshot>,
    staging: Option<tempfile::TempDir>,
    store: DirectoryStore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

#[derive(Debug)]
struct DirectoryGeneration {
    identity: FileIdentity,
    directory: File,
    lease: Option<File>,
    cleanup: Mutex<Option<PathBuf>>,
}

#[derive(Debug, Clone)]
struct TargetSnapshot {
    identity: FileIdentity,
    generation: Arc<DirectoryGeneration>,
}

impl Drop for DirectoryGeneration {
    fn drop(&mut self) {
        let _ = self.cleanup_if_unused();
    }
}

impl DirectoryGeneration {
    fn cleanup_if_unused(&mut self) -> std::io::Result<bool> {
        let cleanup = self
            .cleanup
            .get_mut()
            .map_err(|_| std::io::Error::other("directory generation cleanup mutex poisoned"))?;
        if let Some(lease) = &self.lease {
            match flock(lease, FlockOperation::NonBlockingLockExclusive) {
                Ok(()) => {}
                Err(error) if error == Errno::WOULDBLOCK => return Ok(false),
                Err(error) => return Err(io_error(error)),
            }
            if cleanup.is_none() {
                *cleanup = retired_path_for(&self.directory);
            }
        }
        let Some(path) = cleanup.as_ref() else {
            return Ok(true);
        };
        if path_identity(path)? != self.identity {
            return Err(std::io::Error::other(format!(
                "cleanup path '{}' no longer identifies the retired generation",
                path.display()
            )));
        }
        empty_directory(&self.directory)?;
        fs::remove_dir(path)?;
        *cleanup = None;
        Ok(true)
    }
}

impl DirectoryStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let requested = path.as_ref().to_path_buf();
        let root = fs::canonicalize(&requested).map_err(|error| Error::Path {
            path: requested,
            message: error.to_string(),
        })?;
        let directory = open_directory_path(&root)?;
        Self::from_directory(root, directory)
    }

    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let requested = path.as_ref().to_path_buf();
        fs::create_dir_all(&requested)?;
        Self::open(requested)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn open_at(parent: &File, name: &OsStr, root: PathBuf) -> Result<Self> {
        let descriptor = open_directory_at(parent, name).map_err(|error| {
            if error == Errno::NOENT {
                Error::not_found(root.display().to_string())
            } else {
                Error::Path {
                    path: root.clone(),
                    message: error.to_string(),
                }
            }
        })?;
        Self::from_directory(root, File::from(descriptor))
    }

    fn from_directory(root: PathBuf, directory: File) -> Result<Self> {
        let identity = identity_of(&directory).map_err(|error| Error::Path {
            path: root.clone(),
            message: error.to_string(),
        })?;
        let lease = open_generation_lease(&directory, &root)?;
        let registry = generation_registry();
        let mut registry = registry
            .lock()
            .map_err(|_| Error::corrupt("directory generations", "registry mutex poisoned"))?;
        registry.retain(|_, generation| generation.strong_count() > 0);
        let generation = match registry.get(&identity).and_then(Weak::upgrade) {
            Some(generation) => generation,
            None => {
                let generation = Arc::new(DirectoryGeneration {
                    identity,
                    directory,
                    lease,
                    cleanup: Mutex::new(None),
                });
                registry.insert(identity, Arc::downgrade(&generation));
                generation
            }
        };
        Ok(Self { root, generation })
    }

    fn identity(&self) -> Result<FileIdentity> {
        Ok(self.generation.identity)
    }

    fn parent_for_key<'a>(&self, key: &'a str, create: bool) -> Result<(File, &'a str)> {
        validate_key(key)?;
        let (parent_key, file_name) = key.rsplit_once('/').unwrap_or(("", key));
        let root_copy = dup(&self.generation.directory).map_err(io_error)?;
        let mut current = File::from(root_copy);
        if parent_key.is_empty() {
            return Ok((current, file_name));
        }

        let mut traversed = PathBuf::new();
        for part in parent_key.split('/') {
            traversed.push(part);
            let next = match open_directory_at(&current, OsStr::new(part)) {
                Ok(descriptor) => descriptor,
                Err(error) if create && error == Errno::NOENT => {
                    match mkdirat(&current, part, DIRECTORY_MODE) {
                        Ok(()) => {}
                        Err(error) if error == Errno::EXIST => {}
                        Err(error) => {
                            return Err(Error::Path {
                                path: self.root.join(&traversed),
                                message: error.to_string(),
                            });
                        }
                    }
                    open_directory_at(&current, OsStr::new(part)).map_err(|error| Error::Path {
                        path: self.root.join(&traversed),
                        message: error.to_string(),
                    })?
                }
                Err(error) if error == Errno::NOENT => return Err(Error::not_found(key)),
                Err(error) => {
                    return Err(Error::Path {
                        path: self.root.join(&traversed),
                        message: error.to_string(),
                    });
                }
            };
            current = File::from(next);
        }
        Ok((current, file_name))
    }

    fn open_value(&self, key: &str) -> Result<File> {
        let (parent, file_name) = self.parent_for_key(key, false)?;
        let descriptor = openat(
            &parent,
            file_name,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        )
        .map_err(|error| {
            if error == Errno::NOENT {
                Error::not_found(key)
            } else {
                Error::Path {
                    path: self.root.join(key),
                    message: error.to_string(),
                }
            }
        })?;
        let file = File::from(descriptor);
        let metadata = file.metadata().map_err(|error| Error::Path {
            path: self.root.join(key),
            message: error.to_string(),
        })?;
        if !metadata.is_file() {
            return Err(Error::Path {
                path: self.root.join(key),
                message: "store key is not a regular file".into(),
            });
        }
        Ok(file)
    }

    /// Install one immutable value in this generation.
    ///
    /// The operation only mutates descriptor-relative filesystem state; the
    /// store handle itself has no mutable state. Distinct keys may therefore be
    /// installed concurrently by chunk workers.
    pub(crate) fn write_value(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let (parent, file_name) = self.parent_for_key(key, true)?;
        let mut temporary = None;
        for _ in 0..128 {
            let sequence = WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let candidate = format!(".sc-compress-write-{}-{sequence}", std::process::id());
            match openat(
                &parent,
                candidate.as_str(),
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC | OFlags::NOFOLLOW,
                FILE_MODE,
            ) {
                Ok(descriptor) => {
                    temporary = Some((candidate, File::from(descriptor)));
                    break;
                }
                Err(error) if error == Errno::EXIST => continue,
                Err(error) => {
                    return Err(Error::Path {
                        path: self.root.join(key),
                        message: format!("failed to create temporary value: {error}"),
                    });
                }
            }
        }
        let (temporary_name, mut file) = temporary.ok_or_else(|| Error::Path {
            path: self.root.join(key),
            message: "could not allocate a unique temporary value name".into(),
        })?;

        if let Err(error) = file.write_all(bytes) {
            drop(file);
            let _ = unlinkat(&parent, temporary_name.as_str(), AtFlags::empty());
            return Err(Error::Path {
                path: self.root.join(key),
                message: error.to_string(),
            });
        }
        drop(file);
        if let Err(error) = renameat(&parent, temporary_name.as_str(), &parent, file_name) {
            let _ = unlinkat(&parent, temporary_name.as_str(), AtFlags::empty());
            return Err(Error::Path {
                path: self.root.join(key),
                message: format!("failed to install value atomically: {error}"),
            });
        }
        Ok(())
    }
}

impl DirectoryTransaction {
    pub(crate) fn new(target: impl AsRef<Path>) -> Result<Self> {
        let requested = std::path::absolute(target.as_ref()).map_err(|error| Error::Path {
            path: target.as_ref().to_path_buf(),
            message: error.to_string(),
        })?;
        let requested_parent = requested.parent().ok_or_else(|| Error::Path {
            path: requested.clone(),
            message: "store target must have a parent directory".into(),
        })?;
        let target_name = requested.file_name().ok_or_else(|| Error::Path {
            path: requested.clone(),
            message: "store target must name a directory".into(),
        })?;
        fs::create_dir_all(requested_parent)?;
        let parent_path = fs::canonicalize(requested_parent).map_err(|error| Error::Path {
            path: requested_parent.to_path_buf(),
            message: error.to_string(),
        })?;
        let target_name = target_name.to_os_string();
        let target = parent_path.join(&target_name);
        let parent = open_directory_path(&parent_path)?;
        let expected_target = validate_replace_target(&parent, &target_name, &target)?;
        let staging = tempfile::Builder::new()
            .prefix(".sc-compress-staging-")
            .tempdir_in(&parent_path)?;
        let store = DirectoryStore::open(staging.path())?;
        Ok(Self {
            target,
            target_name,
            parent,
            expected_target,
            staging: Some(staging),
            store,
        })
    }

    pub(crate) fn store_mut(&mut self) -> &mut DirectoryStore {
        &mut self.store
    }

    pub(crate) fn store(&self) -> &DirectoryStore {
        &self.store
    }

    pub(crate) fn commit(self) -> Result<()> {
        self.commit_impl(false)
    }

    #[cfg(test)]
    fn commit_portable(self) -> Result<()> {
        self.commit_impl(true)
    }

    fn commit_impl(mut self, force_portable: bool) -> Result<()> {
        flock(&self.parent, FlockOperation::LockExclusive).map_err(|error| Error::Path {
            path: self.target.clone(),
            message: format!("failed to lock store parent during commit: {error}"),
        })?;
        let current = validate_replace_target(&self.parent, &self.target_name, &self.target)?;
        if current.as_ref().map(|snapshot| snapshot.identity)
            != self
                .expected_target
                .as_ref()
                .map(|snapshot| snapshot.identity)
        {
            return Err(Error::invalid_argument(format!(
                "refusing to replace changed store target '{}'",
                self.target.display()
            )));
        }

        let staging = self
            .staging
            .take()
            .ok_or_else(|| Error::invalid_argument("store transaction was already committed"))?;
        let staging_name = staging
            .path()
            .file_name()
            .ok_or_else(|| Error::Path {
                path: staging.path().to_path_buf(),
                message: "staging directory has no file name".into(),
            })?
            .to_os_string();
        let Some(expected) = self.expected_target.take() else {
            self.install_new_generation(&staging_name, force_portable)?;
            drop(staging);
            return Ok(());
        };

        let old_path =
            self.replace_generation(staging, &staging_name, &expected, force_portable)?;
        let mut cleanup = expected
            .generation
            .cleanup
            .lock()
            .map_err(|_| Error::corrupt("directory generation", "cleanup mutex poisoned"))?;
        if cleanup.is_some() {
            return Err(Error::Path {
                path: old_path,
                message: "old generation already has a pending cleanup path".into(),
            });
        }
        *cleanup = Some(old_path);
        drop(cleanup);
        drop(current);
        if Arc::strong_count(&expected.generation) == 1 {
            let TargetSnapshot {
                generation,
                identity: _,
            } = expected;
            if let Ok(mut generation) = Arc::try_unwrap(generation) {
                generation
                    .cleanup_if_unused()
                    .map_err(|error| Error::Path {
                        path: self.target,
                        message: format!(
                            "new store committed but old generation cleanup failed: {error}"
                        ),
                    })?;
            }
        }
        Ok(())
    }

    fn install_new_generation(&self, staging_name: &OsStr, force_portable: bool) -> Result<()> {
        if !force_portable {
            match renameat_with(
                &self.parent,
                staging_name,
                &self.parent,
                &self.target_name,
                RenameFlags::NOREPLACE,
            ) {
                Ok(()) => return Ok(()),
                Err(error) if rename_flags_unsupported(error) => {}
                Err(error) => {
                    return Err(Error::Path {
                        path: self.target.clone(),
                        message: format!("failed to commit new store without replacement: {error}"),
                    });
                }
            }
        }

        // Reserve the absent name with mkdir, which has portable no-replace
        // semantics. Renaming the completed staging directory over our empty
        // reservation never exposes a partially populated store.
        mkdirat(&self.parent, &self.target_name, DIRECTORY_MODE).map_err(|error| Error::Path {
            path: self.target.clone(),
            message: format!("failed to reserve new store target: {error}"),
        })?;
        if let Err(error) = renameat(&self.parent, staging_name, &self.parent, &self.target_name) {
            let cleanup = unlinkat(&self.parent, &self.target_name, AtFlags::REMOVEDIR)
                .err()
                .map(|cleanup| format!("; target reservation cleanup failed: {cleanup}"))
                .unwrap_or_default();
            return Err(Error::Path {
                path: self.target.clone(),
                message: format!("failed to commit new store on this filesystem: {error}{cleanup}"),
            });
        }
        Ok(())
    }

    fn replace_generation(
        &self,
        staging: tempfile::TempDir,
        staging_name: &OsStr,
        expected: &TargetSnapshot,
        force_portable: bool,
    ) -> Result<PathBuf> {
        if !force_portable {
            match renameat_with(
                &self.parent,
                staging_name,
                &self.parent,
                &self.target_name,
                RenameFlags::EXCHANGE,
            ) {
                Ok(()) => return self.finish_exchange(staging, staging_name, expected),
                Err(error) if rename_flags_unsupported(error) => {}
                Err(error) => {
                    return Err(Error::Path {
                        path: self.target.clone(),
                        message: format!(
                            "failed to atomically exchange store generations: {error}"
                        ),
                    });
                }
            }
        }
        self.replace_generation_portable(staging, staging_name, expected)
    }

    fn finish_exchange(
        &self,
        staging: tempfile::TempDir,
        staging_name: &OsStr,
        expected: &TargetSnapshot,
    ) -> Result<PathBuf> {
        let swapped =
            DirectoryStore::open_at(&self.parent, staging_name, staging.path().to_path_buf())
                .and_then(|store| store.identity());
        if swapped.as_ref().ok() == Some(&expected.identity) {
            return Ok(staging.keep());
        }

        let restore = renameat_with(
            &self.parent,
            staging_name,
            &self.parent,
            &self.target_name,
            RenameFlags::EXCHANGE,
        );
        let observed = swapped
            .map(|identity| format!("{identity:?}"))
            .unwrap_or_else(|error| format!("unreadable: {error}"));
        match restore {
            Ok(()) => Err(Error::Path {
                path: self.target.clone(),
                message: format!(
                    "store target changed during commit; exchange was restored (observed {observed})"
                ),
            }),
            Err(restore_error) => {
                let preserved = staging.keep();
                Err(Error::Path {
                    path: self.target.clone(),
                    message: format!(
                        "store target changed during commit (observed {observed}) and restoration failed ({restore_error}); preserved swapped directory at {}",
                        preserved.display()
                    ),
                })
            }
        }
    }

    fn replace_generation_portable(
        &self,
        staging: tempfile::TempDir,
        staging_name: &OsStr,
        expected: &TargetSnapshot,
    ) -> Result<PathBuf> {
        // Without directory exchange support, two ordinary renames are the
        // strongest portable protocol. Existing readers retain the old
        // directory fd; a new opener may briefly see an empty reservation but
        // can never observe files mixed from two generations.
        let parent_path = self.target.parent().ok_or_else(|| Error::Path {
            path: self.target.clone(),
            message: "store target has no parent directory".into(),
        })?;
        let retired = tempfile::Builder::new()
            // Keep the established prefix so a reader in another process can
            // discover and clean this generation from its open directory fd.
            .prefix(".sc-compress-staging-")
            .tempdir_in(parent_path)?;
        let retired_name = retired
            .path()
            .file_name()
            .ok_or_else(|| Error::Path {
                path: retired.path().to_path_buf(),
                message: "retired directory has no file name".into(),
            })?
            .to_os_string();

        renameat(&self.parent, &self.target_name, &self.parent, &retired_name).map_err(
            |error| Error::Path {
                path: self.target.clone(),
                message: format!("failed to retire the old store generation: {error}"),
            },
        )?;

        let moved =
            DirectoryStore::open_at(&self.parent, &retired_name, retired.path().to_path_buf())
                .and_then(|store| store.identity());
        if moved.as_ref().ok() != Some(&expected.identity) {
            let restore = renameat(&self.parent, &retired_name, &self.parent, &self.target_name);
            let observed = moved
                .map(|identity| format!("{identity:?}"))
                .unwrap_or_else(|error| format!("unreadable: {error}"));
            return match restore {
                Ok(()) => Err(Error::Path {
                    path: self.target.clone(),
                    message: format!(
                        "store target changed while retiring it; restoration succeeded (observed {observed})"
                    ),
                }),
                Err(restore_error) => {
                    let preserved = retired.keep();
                    Err(Error::Path {
                        path: self.target.clone(),
                        message: format!(
                            "store target changed while retiring it (observed {observed}) and restoration failed ({restore_error}); preserved the moved directory at {}",
                            preserved.display()
                        ),
                    })
                }
            };
        }

        if let Err(reserve_error) = mkdirat(&self.parent, &self.target_name, DIRECTORY_MODE) {
            let restore = renameat(&self.parent, &retired_name, &self.parent, &self.target_name);
            return match restore {
                Ok(()) => Err(Error::Path {
                    path: self.target.clone(),
                    message: format!(
                        "failed to reserve the target for the new generation; restored the old generation: {reserve_error}"
                    ),
                }),
                Err(restore_error) => {
                    let preserved = retired.keep();
                    Err(Error::Path {
                        path: self.target.clone(),
                        message: format!(
                            "failed to reserve the target for the new generation ({reserve_error}) and restore the old generation ({restore_error}); preserved the old generation at {}",
                            preserved.display()
                        ),
                    })
                }
            };
        }

        if let Err(install_error) =
            renameat(&self.parent, staging_name, &self.parent, &self.target_name)
        {
            let restore = renameat(&self.parent, &retired_name, &self.parent, &self.target_name);
            return match restore {
                Ok(()) => Err(Error::Path {
                    path: self.target.clone(),
                    message: format!(
                        "failed to install the new store generation; restored the old generation: {install_error}"
                    ),
                }),
                Err(restore_error) => {
                    let preserved = retired.keep();
                    Err(Error::Path {
                        path: self.target.clone(),
                        message: format!(
                            "failed to install the new store generation ({install_error}) and restore the old generation ({restore_error}); preserved the old generation at {}",
                            preserved.display()
                        ),
                    })
                }
            };
        }

        drop(staging);
        Ok(retired.keep())
    }
}

fn rename_flags_unsupported(error: Errno) -> bool {
    matches!(error, Errno::INVAL | Errno::NOSYS | Errno::OPNOTSUPP)
}

impl ByteStore for DirectoryStore {
    fn len(&self, key: &str) -> Result<u64> {
        self.open_value(key)?
            .metadata()
            .map(|meta| meta.len())
            .map_err(|error| Error::Path {
                path: self.root.join(key),
                message: error.to_string(),
            })
    }

    fn read_limited(&self, key: &str, maximum: usize) -> Result<Vec<u8>> {
        let mut file = self.open_value(key)?;
        let declared = file
            .metadata()
            .map_err(|error| Error::Path {
                path: self.root.join(key),
                message: error.to_string(),
            })?
            .len();
        let declared = usize::try_from(declared)
            .map_err(|_| Error::corrupt("store value", "declared size exceeds usize"))?;
        if declared > maximum {
            return Err(Error::corrupt(
                "store value",
                format!("value '{key}' has {declared} bytes, limit is {maximum}"),
            ));
        }
        let mut buffer = zeroed_buffer(declared)?;
        file.read_exact(&mut buffer).map_err(|error| Error::Path {
            path: self.root.join(key),
            message: error.to_string(),
        })?;
        Ok(buffer)
    }

    fn read_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let file = self.open_value(key)?;
        let file_len = file
            .metadata()
            .map_err(|error| Error::Path {
                path: self.root.join(key),
                message: error.to_string(),
            })?
            .len();
        if offset > file_len {
            return Err(Error::corrupt(
                "directory store range",
                format!("offset {offset} past end of {file_len}-byte value '{key}'"),
            ));
        }
        let available = usize::try_from(file_len - offset).map_err(|_| {
            Error::corrupt("directory store range", "available length exceeds usize")
        })?;
        let to_read = len.min(available);
        let mut buffer = zeroed_buffer(to_read)?;
        if to_read == 0 {
            return Ok(buffer);
        }
        file.read_exact_at(&mut buffer, offset)
            .map_err(|error| Error::Path {
                path: self.root.join(key),
                message: error.to_string(),
            })?;
        Ok(buffer)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        match self.open_value(key) {
            Ok(_) => Ok(true),
            Err(Error::NotFound { .. }) => Ok(false),
            Err(error) => Err(error),
        }
    }

    fn supports_efficient_range_reads(&self, key: &str) -> Result<bool> {
        validate_key(key)?;
        Ok(true)
    }

    fn open_positioned(&self, key: &str) -> Result<Option<PositionedValue>> {
        let file = self.open_value(key)?;
        let len = file
            .metadata()
            .map_err(|error| Error::Path {
                path: self.root.join(key),
                message: error.to_string(),
            })?
            .len();
        Ok(Some(PositionedValue::new(file, 0, len)))
    }
}

impl ByteStoreMut for DirectoryStore {
    fn write(&mut self, key: &str, bytes: &[u8]) -> Result<()> {
        self.write_value(key, bytes)
    }
}

fn open_directory_path(path: &Path) -> Result<File> {
    let descriptor = openat(
        CWD,
        path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(|error| Error::Path {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    Ok(File::from(descriptor))
}

fn open_directory_at(
    parent: &File,
    name: &OsStr,
) -> std::result::Result<std::os::fd::OwnedFd, Errno> {
    openat(
        parent,
        name,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )
}

fn validate_replace_target(
    parent: &File,
    target_name: &OsStr,
    target: &Path,
) -> Result<Option<TargetSnapshot>> {
    let store = match DirectoryStore::open_at(parent, target_name, target.to_path_buf()) {
        Ok(store) => store,
        Err(Error::NotFound { .. }) => return Ok(None),
        Err(error) => return Err(error),
    };
    let identity = store.identity()?;
    if directory_is_empty(&store.generation.directory).map_err(|error| Error::Path {
        path: target.to_path_buf(),
        message: error.to_string(),
    })? {
        return Ok(Some(TargetSnapshot {
            identity,
            generation: Arc::clone(&store.generation),
        }));
    }
    read_meta(
        &store,
        ReadLimits::default().maximum_metadata_size(DEFAULT_MAX_VALUE_SIZE),
    )
    .map(|_| {
        Some(TargetSnapshot {
            identity,
            generation: Arc::clone(&store.generation),
        })
    })
    .map_err(|error| {
        Error::invalid_argument(format!(
            "refusing to replace non-empty directory without valid sc-compress metadata: {error}"
        ))
    })
}

fn directory_is_empty(directory: &File) -> std::io::Result<bool> {
    let mut entries = Dir::read_from(directory).map_err(io_error)?;
    while let Some(entry) = entries.read() {
        let entry = entry.map_err(io_error)?;
        let name = entry.file_name().to_bytes();
        if name != b"." && name != b".." {
            return Ok(false);
        }
    }
    Ok(true)
}

fn empty_directory(directory: &File) -> std::io::Result<()> {
    let mut entries = Dir::read_from(directory).map_err(io_error)?;
    while let Some(entry) = entries.read() {
        let entry = entry.map_err(io_error)?;
        let name = entry.file_name();
        if name.to_bytes() == b"." || name.to_bytes() == b".." {
            continue;
        }
        match open_directory_at(directory, OsStr::from_bytes(name.to_bytes())) {
            Ok(child) => {
                let child = File::from(child);
                empty_directory(&child)?;
                unlinkat(directory, name, AtFlags::REMOVEDIR).map_err(io_error)?;
            }
            Err(error) if error == Errno::NOTDIR || error == Errno::LOOP => {
                unlinkat(directory, name, AtFlags::empty()).map_err(io_error)?;
            }
            Err(error) => return Err(io_error(error)),
        }
    }
    Ok(())
}

fn identity_of(file: &File) -> std::io::Result<FileIdentity> {
    let metadata = file.metadata()?;
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

fn open_generation_lease(directory: &File, root: &Path) -> Result<Option<File>> {
    let descriptor = match openat(
        directory,
        "meta.json",
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    ) {
        Ok(descriptor) => descriptor,
        Err(error) if error == Errno::NOENT => return Ok(None),
        Err(error) => {
            return Err(Error::Path {
                path: root.join("meta.json"),
                message: format!("failed to open generation lease: {error}"),
            });
        }
    };
    let lease = File::from(descriptor);
    flock(&lease, FlockOperation::LockShared).map_err(|error| Error::Path {
        path: root.join("meta.json"),
        message: format!("failed to acquire generation lease: {error}"),
    })?;
    Ok(Some(lease))
}

fn retired_path_for(directory: &File) -> Option<PathBuf> {
    let path = fs::read_link(format!("/proc/self/fd/{}", directory.as_raw_fd())).ok()?;
    let name = path.file_name()?.to_string_lossy();
    name.starts_with(".sc-compress-staging-").then_some(path)
}

fn path_identity(path: &Path) -> std::io::Result<FileIdentity> {
    let directory = openat(
        CWD,
        path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(io_error)?;
    identity_of(&File::from(directory))
}

fn generation_registry() -> &'static Mutex<HashMap<FileIdentity, Weak<DirectoryGeneration>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<FileIdentity, Weak<DirectoryGeneration>>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn zeroed_buffer(len: usize) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(len)?;
    buffer.resize(len, 0);
    Ok(buffer)
}

fn io_error(error: Errno) -> std::io::Error {
    std::io::Error::from_raw_os_error(error.raw_os_error())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Partition;

    #[test]
    fn changed_target_is_preserved() {
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("matrix");
        fs::create_dir(&target).unwrap();

        let mut transaction = DirectoryTransaction::new(&target).unwrap();
        transaction.store_mut().write("value", b"new").unwrap();

        fs::remove_dir(&target).unwrap();
        fs::create_dir(&target).unwrap();
        fs::write(target.join("sentinel"), b"keep").unwrap();

        assert!(transaction.commit().is_err());
        assert_eq!(fs::read(target.join("sentinel")).unwrap(), b"keep");
        assert!(fs::read_dir(parent.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("staging")
        }));
    }

    #[test]
    fn newly_appeared_target_is_preserved() {
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("matrix");
        let mut transaction = DirectoryTransaction::new(&target).unwrap();
        transaction.store_mut().write("value", b"new").unwrap();

        fs::create_dir(&target).unwrap();
        fs::write(target.join("sentinel"), b"keep").unwrap();

        assert!(transaction.commit().is_err());
        assert_eq!(fs::read(target.join("sentinel")).unwrap(), b"keep");
    }

    #[test]
    fn replacing_hard_link_does_not_modify_link_target() {
        let parent = tempfile::tempdir().unwrap();
        let outside = parent.path().join("outside");
        fs::write(&outside, b"keep").unwrap();
        let root = parent.path().join("store");
        let mut store = DirectoryStore::create(&root).unwrap();
        fs::hard_link(&outside, root.join("value")).unwrap();

        store.write("value", b"replacement").unwrap();

        assert_eq!(fs::read(outside).unwrap(), b"keep");
        assert_eq!(store.read("value").unwrap(), b"replacement");
    }

    #[test]
    fn distinct_values_can_be_installed_concurrently() {
        let parent = tempfile::tempdir().unwrap();
        let store = DirectoryStore::create(parent.path().join("store")).unwrap();
        std::thread::scope(|scope| {
            for id in 0..32u64 {
                let store = store.clone();
                scope.spawn(move || {
                    store
                        .write_value(&format!("chunks/{id}"), &id.to_le_bytes())
                        .unwrap();
                });
            }
        });
        for id in 0..32u64 {
            assert_eq!(
                store.read(&format!("chunks/{id}")).unwrap(),
                id.to_le_bytes()
            );
        }
    }

    #[test]
    fn portable_commit_installs_a_new_store() {
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("matrix");
        let mut transaction = DirectoryTransaction::new(&target).unwrap();
        transaction.store_mut().write("value", b"new").unwrap();
        transaction.commit_portable().unwrap();

        assert_eq!(
            DirectoryStore::open(target).unwrap().read("value").unwrap(),
            b"new"
        );
    }

    #[test]
    fn portable_commit_keeps_the_old_generation_readable() {
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("matrix");
        crate::DenseWriter::new(&target, Partition::fixed_cells(1024), Partition::fixed_cells(16))
            .write(&[1u16, 2, 3, 4], [2, 2])
            .unwrap();
        let old = DirectoryStore::open(&target).unwrap();
        let old_chunk = old.read("data/0").unwrap();

        let mut transaction = DirectoryTransaction::new(&target).unwrap();
        transaction
            .store_mut()
            .write("replacement", b"new")
            .unwrap();
        transaction.commit_portable().unwrap();

        assert_eq!(old.read("data/0").unwrap(), old_chunk);
        assert_eq!(
            DirectoryStore::open(&target)
                .unwrap()
                .read("replacement")
                .unwrap(),
            b"new"
        );
        assert!(fs::read_dir(parent.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".sc-compress-staging-")
        }));
        drop(old);
        assert!(fs::read_dir(parent.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".sc-compress-staging-")
        }));
    }

    #[test]
    fn portable_commits_serialize_competing_replacements() {
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("matrix");
        crate::DenseWriter::new(&target, Partition::fixed_cells(1024), Partition::fixed_cells(16))
            .write(&[1u16], [1, 1])
            .unwrap();

        let mut first = DirectoryTransaction::new(&target).unwrap();
        first.store_mut().write("value", b"first").unwrap();
        let mut second = DirectoryTransaction::new(&target).unwrap();
        second.store_mut().write("value", b"second").unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let (first_result, second_result) = std::thread::scope(|scope| {
            let first_barrier = Arc::clone(&barrier);
            let first = scope.spawn(move || {
                first_barrier.wait();
                first.commit_portable()
            });
            let second_barrier = Arc::clone(&barrier);
            let second = scope.spawn(move || {
                second_barrier.wait();
                second.commit_portable()
            });
            barrier.wait();
            (first.join().unwrap(), second.join().unwrap())
        });

        assert_ne!(first_result.is_ok(), second_result.is_ok());
        let value = DirectoryStore::open(&target)
            .unwrap()
            .read("value")
            .unwrap();
        assert!(value == b"first" || value == b"second");
    }

    #[test]
    fn writer_can_replace_store_with_large_valid_metadata() {
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("matrix");
        crate::DenseWriter::new(&target, Partition::fixed_cells(1024), Partition::fixed_cells(16))
            .write(&[1u16], [1, 1])
            .unwrap();

        let metadata_path = target.join("meta.json");
        let mut metadata = fs::OpenOptions::new()
            .append(true)
            .open(&metadata_path)
            .unwrap();
        metadata
            .write_all(&vec![b' '; ReadLimits::default().metadata_size()])
            .unwrap();
        drop(metadata);
        assert!(
            fs::metadata(&metadata_path).unwrap().len()
                > u64::try_from(ReadLimits::default().metadata_size()).unwrap()
        );

        crate::DenseWriter::new(&target, Partition::fixed_cells(1024), Partition::fixed_cells(16))
            .write(&[2u16], [1, 1])
            .unwrap();
        assert_eq!(
            crate::open_dense(&target).unwrap().decode_all().unwrap(),
            2u16.to_le_bytes()
        );
    }
}
