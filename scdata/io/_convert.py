"""High-level conversion of AnnData input into scdata ``.zarr.zip`` stores."""

from __future__ import annotations

import bz2
import gzip
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, cast

from scdata.io._anndata import (
    _DEFAULT_CHUNK_ELEMENTS,
    _DEFAULT_COMPRESSOR,
    _Compressor,
    _LayerFormat,
    write_zarr,
)
from scdata.io._launch import StoreError

if TYPE_CHECKING:
    from anndata import AnnData

_ReadFormat = Literal[
    "auto",
    "h5ad",
    "zarr",
    "loom",
    "hdf",
    "excel",
    "umi_tools",
    "csv",
    "text",
    "mtx",
]
_XFormat = Literal["auto", "dense2d", "dense1d", "sparse"]
_UNSET = object()

_READ_FORMAT_ALIASES = {
    "h5": "hdf",
    "hdf5": "hdf",
    "xlsx": "excel",
    "xls": "excel",
    "tsv": "text",
    "txt": "text",
    "tab": "text",
    "data": "text",
    "matrix_market": "mtx",
    "mtx.gz": "mtx",
    "mtx.bz2": "mtx",
}

_READ_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".zarr.zip", "zarr"),
    (".mtx.gz", "mtx"),
    (".mtx.bz2", "mtx"),
    (".csv.gz", "csv"),
    (".csv.bz2", "csv"),
    (".tsv.gz", "text"),
    (".tsv.bz2", "text"),
    (".tab.gz", "text"),
    (".tab.bz2", "text"),
    (".txt.gz", "text"),
    (".txt.bz2", "text"),
    (".data.gz", "text"),
    (".data.bz2", "text"),
    (".h5ad", "h5ad"),
    (".zarr", "zarr"),
    (".loom", "loom"),
    (".h5", "hdf"),
    (".hdf", "hdf"),
    (".hdf5", "hdf"),
    (".xlsx", "excel"),
    (".xls", "excel"),
    (".mtx", "mtx"),
    (".csv", "csv"),
    (".tsv", "text"),
    (".tab", "text"),
    (".txt", "text"),
    (".data", "text"),
)


@dataclass
class AnnDataZarrZipConverter:
    """Callable converter from AnnData-readable inputs to scdata ``.zarr.zip``.

    Reading is delegated to the public/compat readers exposed by
    ``anndata.io`` in anndata 0.12.x:
    ``read_h5ad``, ``read_zarr``, ``read_loom``, ``read_hdf``,
    ``read_excel``, ``read_umi_tools``, ``read_csv``, ``read_text``, and
    ``read_mtx``.  Writing is delegated to :func:`scdata.io.write_zarr`.

    ``format="auto"`` writes sparse ``X`` as CSR and dense ``X`` as
    ``dense1d`` so whole-cell access stays aligned by default.  Pass
    ``format="dense2d"`` for the standard anndata dense layout.  Layers are
    written with ``layer_format``; the default preserves dense vs sparse
    storage while making both registerable by scdata.

    Chunks are compressed by default using ``"blosc.lz4.level5"``.  Pass
    ``compressor=None`` to write uncompressed chunks.
    """

    smart: bool = True
    format: _XFormat = "auto"
    chunk_size: int | list[int] | tuple[int, ...] = _DEFAULT_CHUNK_ELEMENTS
    align_cells: bool = True
    layer_format: _LayerFormat = "preserve"
    compressor: _Compressor = _DEFAULT_COMPRESSOR
    output_dir: str | os.PathLike[str] | None = None
    overwrite: bool = True
    read_kwargs: Mapping[str, Any] = field(default_factory=dict)

    supported_read_formats = (
        "h5ad",
        "zarr",
        "loom",
        "hdf",
        "excel",
        "umi_tools",
        "csv",
        "text",
        "mtx",
    )
    supported_x_formats = ("dense2d", "dense1d", "sparse")

    def __call__(
        self,
        source: str | os.PathLike[str],
        target: str | os.PathLike[str] | None = None,
        *,
        read_format: _ReadFormat | str | None = None,
        format: _XFormat | None = None,
        chunk_size: int | list[int] | tuple[int, ...] | None = None,
        align_cells: bool | None = None,
        layer_format: _LayerFormat | None = None,
        compressor: Any = _UNSET,
        read_kwargs: Mapping[str, Any] | None = None,
        overwrite: bool | None = None,
    ) -> Path:
        """Convert ``source`` into a same-name ``.zarr.zip`` store.

        Parameters override constructor defaults for one call.  ``target`` is
        optional; when omitted, the converter strips the recognized input
        suffix and appends ``.zarr.zip`` in ``output_dir`` or next to source.
        """
        src = Path(os.fspath(source))
        out = Path(os.fspath(target)) if target is not None else self._default_target(src)
        self._check_output_path(src, out, self.overwrite if overwrite is None else overwrite)

        kwargs = dict(self.read_kwargs)
        if read_kwargs is not None:
            kwargs.update(read_kwargs)
        resolved_read = self._resolve_read_format(src, read_format)
        adata = self._read(src, resolved_read, kwargs)
        resolved_format = self._resolve_x_format(
            adata,
            self.format if format is None else format,
        )
        return write_zarr(
            adata,
            out,
            format=resolved_format,
            layer_format=self.layer_format if layer_format is None else layer_format,
            chunk_size=self.chunk_size if chunk_size is None else chunk_size,
            align_cells=self.align_cells if align_cells is None else align_cells,
            compressor=self.compressor if compressor is _UNSET else compressor,
            store="zip",
        )

    def _resolve_read_format(self, source: Path, read_format: _ReadFormat | str | None) -> str:
        if read_format is not None and read_format != "auto":
            return _normalize_read_format(read_format)
        if not self.smart:
            raise StoreError("read_format must be explicit when smart=False")
        detected = _detect_read_format(source)
        if detected is None:
            raise StoreError(
                f"cannot infer AnnData reader for {source}; pass read_format explicitly"
            )
        return detected

    def _resolve_x_format(
        self, adata: "AnnData", format: _XFormat
    ) -> Literal["dense2d", "dense1d", "sparse"]:
        if format != "auto":
            if format not in self.supported_x_formats:
                raise StoreError(f"unsupported output format {format!r}")
            return format
        if not self.smart:
            raise StoreError("format must be explicit when smart=False")
        if adata.X is None:
            raise StoreError("AnnData.X is None; cannot choose an output X layout")
        from scipy import sparse

        return "sparse" if sparse.issparse(adata.X) else "dense1d"

    def _read(self, source: Path, read_format: str, kwargs: dict[str, Any]) -> "AnnData":
        import anndata as ad

        io = ad.io
        if read_format == "h5ad":
            return io.read_h5ad(source, **kwargs)
        if read_format == "zarr":
            return _read_zarr_with_anndata(source, kwargs)
        if read_format == "loom":
            return io.read_loom(source, **kwargs)
        if read_format == "hdf":
            kwargs = dict(kwargs)
            if "key" not in kwargs:
                kwargs["key"] = _infer_hdf_key(source)
            return io.read_hdf(source, **kwargs)
        if read_format == "excel":
            kwargs = dict(kwargs)
            kwargs.setdefault("sheet", 0)
            return io.read_excel(source, **kwargs)
        if read_format == "umi_tools":
            return io.read_umi_tools(source, **kwargs)
        if read_format == "csv":
            return io.read_csv(source, **kwargs)
        if read_format == "text":
            kwargs = _text_kwargs_for_source(source, kwargs)
            return io.read_text(source, **kwargs)
        if read_format == "mtx":
            return io.read_mtx(source, **kwargs)
        raise StoreError(f"unsupported read_format {read_format!r}")

    def _default_target(self, source: Path) -> Path:
        base = Path(os.fspath(self.output_dir)) if self.output_dir is not None else source.parent
        return base / f"{_source_stem(source)}.zarr.zip"

    @staticmethod
    def _check_output_path(source: Path, target: Path, overwrite: bool) -> None:
        if _same_existing_path(source, target):
            raise StoreError(
                "source and target resolve to the same path; pass an explicit different target"
            )
        if target.exists() and not overwrite:
            raise StoreError(f"target already exists: {target}")


def _normalize_read_format(read_format: str) -> str:
    value = read_format.lower().replace("-", "_").replace(".", "_")
    value = _READ_FORMAT_ALIASES.get(value, value)
    if value not in AnnDataZarrZipConverter.supported_read_formats:
        raise StoreError(f"unsupported read_format {read_format!r}")
    return value


def _detect_read_format(source: Path) -> str | None:
    if source.is_dir():
        if _is_zarr_directory(source):
            return "zarr"
        if source.name.lower().endswith(".zarr"):
            return "zarr"
    name = source.name.lower()
    for suffix, read_format in _READ_SUFFIXES:
        if name.endswith(suffix):
            if read_format == "text" and _looks_like_umi_tools(source):
                return "umi_tools"
            return read_format
    return None


def _is_zarr_directory(source: Path) -> bool:
    """Return true for v2/v3 zarr directory stores, regardless of suffix."""
    return (
        (source / "zarr.json").is_file()
        or (source / ".zgroup").is_file()
        or (source / ".zmetadata").is_file()
    )


def _source_stem(source: Path) -> str:
    name = source.name
    lower = name.lower()
    for suffix, _ in _READ_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return source.stem if source.suffix else name


def _same_existing_path(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and left.resolve() == right.resolve()
    except OSError:
        return False


def _read_zarr_with_anndata(source: Path, kwargs: dict[str, Any]) -> "AnnData":
    import anndata as ad

    if source.is_file():
        from zarr.storage import ZipStore

        store = ZipStore(str(source), mode="r")
        try:
            return ad.io.read_zarr(cast(Any, store), **kwargs)
        finally:
            store.close()
    return ad.io.read_zarr(source, **kwargs)


def _infer_hdf_key(source: Path) -> str:
    import h5py

    with h5py.File(source, "r") as handle:
        datasets = [name for name, value in handle.items() if isinstance(value, h5py.Dataset)]
        keys = list(handle.keys())
    if len(datasets) == 1:
        return datasets[0]
    raise StoreError(
        f"cannot infer HDF dataset key for {source}; pass read_kwargs={{'key': ...}}. "
        f"Top-level keys: {keys}"
    )


def _text_kwargs_for_source(source: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    name = source.name.lower()
    if "delimiter" not in out and (
        name.endswith(".tsv")
        or name.endswith(".tsv.gz")
        or name.endswith(".tsv.bz2")
        or name.endswith(".tab")
        or name.endswith(".tab.gz")
        or name.endswith(".tab.bz2")
    ):
        out["delimiter"] = "\t"
    return out


def _looks_like_umi_tools(source: Path) -> bool:
    name = source.name.lower()
    if not (
        name.endswith(".tsv")
        or name.endswith(".tsv.gz")
        or name.endswith(".tsv.bz2")
        or name.endswith(".txt")
        or name.endswith(".txt.gz")
        or name.endswith(".txt.bz2")
    ):
        return False
    try:
        first = _read_first_line(source)
    except OSError:
        return False
    fields = {part.strip().lower() for part in first.split("\t")}
    return {"gene", "cell", "count"}.issubset(fields)


def _read_first_line(source: Path) -> str:
    name = source.name.lower()
    if name.endswith(".gz"):
        with gzip.open(source, "rt") as handle:
            return handle.readline()
    if name.endswith(".bz2"):
        with bz2.open(source, "rt") as handle:
            return handle.readline()
    with source.open() as handle:
        return handle.readline()
