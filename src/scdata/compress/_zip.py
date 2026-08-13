"""ZIP archive helpers for packing and writing sc-compress stores."""

from __future__ import annotations

import os
import shutil
import stat
import time
import warnings
import zipfile
from bisect import bisect_left
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from operator import index
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, SupportsIndex

from scdata.compress._validate import ensure_path
from scdata.compress._write_options import WriteOptions
from scdata.exceptions import PerformanceWarning

__all__ = [
    "list_stores",
    "pack",
    "write",
    "write_csr",
    "write_csr_arrays",
    "write_dense",
]

META_NAME = "meta.json"
Archive = str | PathLike[str] | zipfile.ZipFile


def archive_path(archive: Archive) -> Path:
    """Resolve the filesystem path behind a path-like or ``ZipFile`` object."""
    if isinstance(archive, zipfile.ZipFile):
        filename = archive.filename
        if not filename:
            raise ValueError("ZipFile must be backed by a filesystem path for sc-compress reads")
        return ensure_path(filename)
    return ensure_path(archive)


def normalize_prefix(prefix: str) -> str:
    if not isinstance(prefix, str):
        raise TypeError(f"prefix must be str, got {type(prefix).__name__}")
    if "\\" in prefix or "\0" in prefix:
        raise ValueError(f"invalid ZIP prefix {prefix!r}")
    normalized = prefix.strip("/")
    if not normalized:
        return ""
    if any(part in ("", ".", "..") for part in normalized.split("/")):
        raise ValueError(f"invalid ZIP prefix {prefix!r}")
    return normalized


def list_stores(archive: Archive) -> list[str]:
    """Return sorted prefixes that contain an sc-compress ``meta.json``.

    A root-level store is represented by ``""``. Unsafe archive member paths
    are ignored rather than advertised as selectable stores.
    """
    with _zip_reader(archive) as zf:
        prefixes: set[str] = set()
        for name in zf.namelist():
            if name.endswith("/") or not _is_safe_member_name(name):
                continue
            if name == META_NAME:
                prefixes.add("")
                continue
            suffix = "/" + META_NAME
            if not name.endswith(suffix):
                continue
            prefixes.add(name[: -len(suffix)])
        return sorted(prefixes)


def _is_safe_member_name(name: str) -> bool:
    return (
        bool(name)
        and not name.startswith("/")
        and "\\" not in name
        and "\0" not in name
        and all(part not in ("", ".", "..") for part in name.split("/"))
    )


def pack(
    archive: Archive,
    prefix: str,
    store_dir: str | PathLike[str],
    *,
    compression: int = zipfile.ZIP_STORED,
    compresslevel: int | None = None,
) -> None:
    """Copy a directory store into ``archive`` under ``prefix``.

    Path inputs are opened in append mode. Member names and collisions are
    checked before the archive is modified, and symbolic links are rejected.
    """
    _pack_impl(
        archive,
        prefix,
        ensure_path(store_dir),
        compression=compression,
        compresslevel=compresslevel,
        # pack → _pack_impl → _warn_compression; user frame is 4 up.
        warn_stacklevel=4,
    )


def write(
    archive: Archive,
    prefix: str,
    matrix: Any,
    *,
    options: WriteOptions | None = None,
    num_workers: int | None = None,
    compression: int = zipfile.ZIP_STORED,
    compresslevel: int | None = None,
) -> None:
    """Write a dense or sparse matrix into a ZIP archive."""
    from scdata.compress._io import write as write_matrix

    _write_via_directory(
        archive,
        prefix,
        lambda path: write_matrix(
            path,
            matrix,
            options=options,
            num_workers=num_workers,
            overwrite=True,
        ),
        compression=compression,
        compresslevel=compresslevel,
    )


def write_dense(
    archive: Archive,
    prefix: str,
    values: Any,
    *,
    options: WriteOptions | None = None,
    num_workers: int | None = None,
    compression: int = zipfile.ZIP_STORED,
    compresslevel: int | None = None,
) -> None:
    """Write a dense matrix into ``archive`` under ``prefix``."""
    from scdata.compress._io import write_dense as write_dense_matrix

    _write_via_directory(
        archive,
        prefix,
        lambda path: write_dense_matrix(
            path,
            values,
            options=options,
            num_workers=num_workers,
            overwrite=True,
        ),
        compression=compression,
        compresslevel=compresslevel,
    )


def write_csr(
    archive: Archive,
    prefix: str,
    matrix: Any,
    *,
    options: WriteOptions | None = None,
    num_workers: int | None = None,
    compression: int = zipfile.ZIP_STORED,
    compresslevel: int | None = None,
) -> None:
    """Write a SciPy sparse matrix into ``archive`` under ``prefix``."""
    from scdata.compress._io import write_csr as write_csr_matrix

    _write_via_directory(
        archive,
        prefix,
        lambda path: write_csr_matrix(
            path,
            matrix,
            options=options,
            num_workers=num_workers,
            overwrite=True,
        ),
        compression=compression,
        compresslevel=compresslevel,
    )


def write_csr_arrays(
    archive: Archive,
    prefix: str,
    indptr: Any,
    indices: Any,
    data: Any,
    shape: tuple[int, int] | list[int],
    *,
    options: WriteOptions | None = None,
    num_workers: int | None = None,
    compression: int = zipfile.ZIP_STORED,
    compresslevel: int | None = None,
) -> None:
    """Write explicit CSR buffers into ``archive`` under ``prefix``."""
    from scdata.compress._io import write_csr_arrays as write_csr_buffers

    _write_via_directory(
        archive,
        prefix,
        lambda path: write_csr_buffers(
            path,
            indptr,
            indices,
            data,
            shape,
            options=options,
            num_workers=num_workers,
            overwrite=True,
        ),
        compression=compression,
        compresslevel=compresslevel,
    )


def _write_via_directory(
    archive: Archive,
    prefix: str,
    writer: Callable[[Path], None],
    *,
    compression: int,
    compresslevel: int | None,
) -> None:
    prefix = normalize_prefix(prefix)
    compression, compresslevel = _validate_compression(compression, compresslevel)
    _validate_archive_writer(archive)
    with _temporary_store(archive) as temporary_store:
        writer(temporary_store)
        # write_* → _write_via_directory → _pack_impl → _warn_compression.
        _pack_impl(
            archive,
            prefix,
            temporary_store,
            compression=compression,
            compresslevel=compresslevel,
            warn_stacklevel=5,
        )


def _pack_impl(
    archive: Archive,
    prefix: str,
    root: Path,
    *,
    compression: int,
    compresslevel: int | None,
    warn_stacklevel: int,
) -> None:
    prefix_normalized = normalize_prefix(prefix)
    compression, compresslevel = _validate_compression(compression, compresslevel)
    _validate_archive_writer(archive)
    if root.is_symlink():
        raise ValueError(f"store_dir must not be a symbolic link: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"store_dir is not a directory: {root}")
    if not (root / META_NAME).is_file():
        raise FileNotFoundError(f"store_dir is missing {META_NAME}: {root}")
    _reject_archive_inside_store(archive, root)

    planned = [
        (f"{prefix_normalized}/{relative}" if prefix_normalized else relative, absolute)
        for relative, absolute in _iter_store_files(root)
    ]
    if not planned:
        raise ValueError(f"store_dir contains no files: {root}")
    for arcname, _ in planned:
        if not _is_safe_member_name(arcname):
            raise ValueError(f"store contains an unsafe ZIP member name: {arcname!r}")
        try:
            encoded_name = arcname.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(f"ZIP member name is not valid UTF-8: {arcname!r}")
        if len(encoded_name) > 65_535:
            raise ValueError(
                f"ZIP member name is {len(encoded_name)} encoded bytes; maximum is 65535"
            )

    _warn_compression(compression, stacklevel=warn_stacklevel)
    with _zip_writer(archive) as zf:
        existing = set(zf.namelist())
        existing_sorted = sorted(existing)
        collisions = sorted(
            arcname
            for arcname, _ in planned
            if _member_collides(arcname, existing, existing_sorted)
        )
        if collisions:
            preview = ", ".join(repr(name) for name in collisions[:3])
            if len(collisions) > 3:
                preview += f", ... ({len(collisions)} total)"
            raise ValueError(
                f"ZIP already contains target member(s): {preview}; choose another prefix"
            )
        for arcname, absolute in planned:
            _write_member(zf, arcname, absolute, compression, compresslevel)


@contextmanager
def _temporary_store(archive: Archive) -> Iterator[Path]:
    """Place potentially large staging data beside the destination archive."""
    if isinstance(archive, zipfile.ZipFile) and not archive.filename:
        parent = Path.cwd()
    else:
        parent = archive_path(archive).parent
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sc-compress-zip-", dir=parent) as directory:
        yield Path(directory)


def _iter_store_files(root: Path) -> Iterable[tuple[str, Path]]:
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            if (base / dirname).is_symlink():
                raise ValueError(f"store contains a symbolic-link directory: {base / dirname}")
        dirnames[:] = [name for name in dirnames if not name.startswith(".sc-compress-staging-")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            absolute = base / filename
            metadata = absolute.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"store contains a symbolic-link file: {absolute}")
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"store contains a non-regular file: {absolute}")
            yield absolute.relative_to(root).as_posix(), absolute


def _reject_archive_inside_store(archive: Archive, root: Path) -> None:
    if isinstance(archive, zipfile.ZipFile) and not archive.filename:
        return
    path = archive_path(archive).resolve(strict=False)
    if path.is_relative_to(root.resolve()):
        raise ValueError("archive path must not be inside the store being packed")


def _member_collides(
    arcname: str,
    existing: set[str],
    existing_sorted: list[str],
) -> bool:
    if arcname in existing or f"{arcname}/" in existing:
        return True
    parts = arcname.split("/")
    if any("/".join(parts[:end]) in existing for end in range(1, len(parts))):
        return True
    child_prefix = f"{arcname}/"
    position = bisect_left(existing_sorted, child_prefix)
    return position < len(existing_sorted) and existing_sorted[position].startswith(child_prefix)


def _validate_compression(
    compression: object,
    compresslevel: object | None,
) -> tuple[int, int | None]:
    if isinstance(compression, bool):
        raise TypeError("compression must be a zipfile compression constant, not bool")
    if not isinstance(compression, SupportsIndex):
        raise TypeError(
            f"compression must be a zipfile compression constant, got {compression!r}"
        )
    method = index(compression)

    supported = {
        zipfile.ZIP_STORED,
        zipfile.ZIP_DEFLATED,
        zipfile.ZIP_BZIP2,
        zipfile.ZIP_LZMA,
    }
    zip_zstandard = getattr(zipfile, "ZIP_ZSTANDARD", None)
    if zip_zstandard is not None:
        supported.add(zip_zstandard)
    if method not in supported:
        raise ValueError(f"unsupported ZIP compression method: {method}")

    checker = getattr(zipfile, "_check_compression", None)
    if checker is not None:
        try:
            checker(method)
        except (NotImplementedError, RuntimeError) as error:
            raise ValueError(str(error))

    level: int | None = None
    if compresslevel is not None:
        if isinstance(compresslevel, bool):
            raise TypeError("compresslevel must be an integer or None, not bool")
        if not isinstance(compresslevel, SupportsIndex):
            raise TypeError(f"compresslevel must be an integer or None, got {compresslevel!r}")
        level = index(compresslevel)

    if level is not None and method == zipfile.ZIP_DEFLATED and not -1 <= level <= 9:
        raise ValueError("ZIP_DEFLATED compresslevel must be between -1 and 9")
    if level is not None and method == zipfile.ZIP_BZIP2 and not 1 <= level <= 9:
        raise ValueError("ZIP_BZIP2 compresslevel must be between 1 and 9")
    if (
        level is not None
        and zip_zstandard is not None
        and method == zip_zstandard
        and not -131_072 <= level <= 22
    ):
        raise ValueError("ZIP_ZSTANDARD compresslevel must be between -131072 and 22")
    return method, level


def _warn_compression(compression: int, *, stacklevel: int) -> None:
    if compression != zipfile.ZIP_STORED:
        warnings.warn(
            "sc-compress chunks are already compressed; outer ZIP compression "
            "disables efficient range reads. Prefer zipfile.ZIP_STORED.",
            PerformanceWarning,
            stacklevel=stacklevel,
        )


def _write_member(
    zf: zipfile.ZipFile,
    arcname: str,
    absolute: Path,
    compression: int,
    compresslevel: int | None,
) -> None:
    metadata = absolute.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"store file changed while packing: {absolute}")

    flags = os.O_RDONLY
    for flag_name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= getattr(os, flag_name, 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise OSError(f"cannot safely open store file {absolute}: {error}") from error

    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (metadata.st_dev, metadata.st_ino):
            raise ValueError(f"store file changed while packing: {absolute}")

        date_time = time.localtime(opened.st_mtime)[:6]
        if not getattr(zf, "_strict_timestamps", True):
            if date_time[0] < 1980:
                date_time = (1980, 1, 1, 0, 0, 0)
            elif date_time[0] > 2107:
                date_time = (2107, 12, 31, 23, 59, 59)
        info = zipfile.ZipInfo(arcname, date_time=date_time)
        info.external_attr = (opened.st_mode & 0xFFFF) << 16
        info.file_size = opened.st_size
        info.compress_type = compression
        setattr(info, "_compresslevel", compresslevel)
        with zf.open(info, mode="w") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)


@contextmanager
def _zip_reader(archive: Archive) -> Iterator[zipfile.ZipFile]:
    if isinstance(archive, zipfile.ZipFile):
        if archive.fp is None:
            raise ValueError("ZipFile is closed")
        yield archive
        return
    with zipfile.ZipFile(ensure_path(archive), mode="r") as zf:
        yield zf


@contextmanager
def _zip_writer(archive: Archive) -> Iterator[zipfile.ZipFile]:
    if isinstance(archive, zipfile.ZipFile):
        _validate_archive_writer(archive)
        yield archive
        return
    path = ensure_path(archive)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="a", compression=zipfile.ZIP_STORED) as zf:
        yield zf


def _validate_archive_writer(archive: Archive) -> None:
    if not isinstance(archive, zipfile.ZipFile):
        path = ensure_path(archive)
        if path.name in ("", ".", ".."):
            raise ValueError(f"archive path must name a ZIP file, got {path}")
        if path.exists() or path.is_symlink():
            if path.is_dir():
                raise IsADirectoryError(f"archive path is a directory: {path}")
            if not path.is_file():
                raise ValueError(f"archive path is not a regular file: {path}")
            if not zipfile.is_zipfile(path):
                raise ValueError(f"existing archive path is not a ZIP file: {path}")
        return
    if archive.fp is None:
        raise ValueError("ZipFile is closed")
    if archive.mode not in ("w", "x", "a"):
        raise ValueError(
            f"ZipFile must be opened for writing (mode 'w', 'x', or 'a'), got {archive.mode!r}"
        )
