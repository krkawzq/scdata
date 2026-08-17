"""Unified public codec configuration for SCC writers and readers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

CodecAlgorithm = Literal["blosc", "zstd", "zlib", "lz4", "none"]
BloscBackend = Literal["blosclz", "lz4", "zlib", "zstd"]
Shuffle = Literal["none", "bytes", "bits"]
_WireRole = Literal["dense", "csr", "indptr"]

_ALGORITHMS = frozenset(("blosc", "zstd", "zlib", "lz4", "none"))
_BLOSC_BACKENDS = frozenset(("blosclz", "lz4", "zlib", "zstd"))
_SHUFFLES = frozenset(("none", "bytes", "bits"))
_CONFIG_KEYS = frozenset(("algorithm", "backend", "level", "shuffle", "split_blocks"))


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    return int(value)


def _reject_option(algorithm: str, name: str, value: object | None) -> None:
    if value is not None:
        raise ValueError(f"{name} is only valid for the blosc codec, not {algorithm!r}")


@dataclass(frozen=True)
class Codec:
    """Storage-codec policy independent of the SCC matrix representation.

    ``algorithm='blosc'`` is lowered internally to the representation required
    by dense or CSR data. Storage-format variants and derived block sizes are
    intentionally not part of the public Python API.

    ``zstd``, ``zlib``, ``lz4``, and ``none`` are valid for CSR ``indptr``.
    Matrix data uses the Blosc codec because SCC dense and CSR payloads require
    their respective block-aware Blosc representations.
    """

    algorithm: CodecAlgorithm = "blosc"
    backend: BloscBackend | None = None
    level: int | None = None
    shuffle: Shuffle | None = None
    split_blocks: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, str) or self.algorithm not in _ALGORITHMS:
            raise ValueError(f"unsupported codec algorithm {self.algorithm!r}")

        if self.algorithm == "blosc":
            backend = "lz4" if self.backend is None else self.backend
            if backend not in _BLOSC_BACKENDS:
                raise ValueError(f"unsupported blosc backend {backend!r}")
            level = 5 if self.level is None else _require_int(self.level, "level")
            if not 0 <= level <= 9:
                raise ValueError(f"blosc level must be in 0..=9, got {level}")
            shuffle = "none" if self.shuffle is None else self.shuffle
            if shuffle not in _SHUFFLES:
                raise ValueError(f"shuffle must be 'none', 'bytes', or 'bits', got {shuffle!r}")
            split_blocks = False if self.split_blocks is None else self.split_blocks
            if not isinstance(split_blocks, bool):
                raise TypeError(f"split_blocks must be bool, got {type(split_blocks).__name__}")
            object.__setattr__(self, "backend", backend)
            object.__setattr__(self, "level", level)
            object.__setattr__(self, "shuffle", shuffle)
            object.__setattr__(self, "split_blocks", split_blocks)
            return

        _reject_option(self.algorithm, "backend", self.backend)
        _reject_option(self.algorithm, "shuffle", self.shuffle)
        _reject_option(self.algorithm, "split_blocks", self.split_blocks)
        if self.algorithm in ("zstd", "zlib"):
            default = 3 if self.algorithm == "zstd" else 6
            level = default if self.level is None else _require_int(self.level, "level")
            if self.algorithm == "zlib" and not 0 <= level <= 9:
                raise ValueError(f"zlib level must be in 0..=9, got {level}")
            if self.algorithm == "zstd" and not -(1 << 31) <= level < (1 << 31):
                raise ValueError(f"zstd level must fit signed 32-bit, got {level}")
            object.__setattr__(self, "level", level)
            return
        if self.level is not None:
            raise ValueError(f"level is not valid for codec {self.algorithm!r}")

    @staticmethod
    def blosc(
        backend: BloscBackend = "lz4",
        level: int = 5,
        shuffle: Shuffle = "none",
        split_blocks: bool = False,
    ) -> Codec:
        return Codec(
            algorithm="blosc",
            backend=backend,
            level=level,
            shuffle=shuffle,
            split_blocks=split_blocks,
        )

    @staticmethod
    def zstd(level: int = 3) -> Codec:
        return Codec(algorithm="zstd", level=level)

    @staticmethod
    def zlib(level: int = 6) -> Codec:
        return Codec(algorithm="zlib", level=level)

    @staticmethod
    def lz4() -> Codec:
        return Codec(algorithm="lz4")

    @staticmethod
    def none() -> Codec:
        return Codec(algorithm="none")

    @classmethod
    def parse(cls, spec: Codec | str | Mapping[str, Any]) -> Codec:
        """Parse a strict public codec specification.

        String specifications name only the algorithm. Use :class:`Codec` or a
        mapping for algorithm-specific options.
        """
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, Mapping):
            unknown = set(spec) - _CONFIG_KEYS
            if unknown:
                names = ", ".join(sorted(repr(str(name)) for name in unknown))
                raise ValueError(f"unknown codec option(s): {names}")
            if "algorithm" not in spec:
                raise ValueError("codec mapping requires 'algorithm'")
            return cls(**dict(spec))
        if not isinstance(spec, str) or not spec.strip():
            raise TypeError("codec spec must be Codec, mapping, or non-empty str")
        algorithm = spec.strip().lower()
        if algorithm not in _ALGORITHMS:
            raise ValueError(
                f"unsupported codec algorithm {spec!r}; expected one of {sorted(_ALGORITHMS)}"
            )
        return cls(algorithm=algorithm)  # type: ignore[arg-type]


def _reject_wire_unknown(payload: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(payload) - allowed
    if unknown:
        names = ", ".join(sorted(repr(str(name)) for name in unknown))
        raise ValueError(f"unknown on-disk codec field(s): {names}")


def _codec_to_wire(codec: Codec, *, role: _WireRole) -> dict[str, Any]:
    """Lower the public policy to the private on-disk compressor schema."""
    if role not in ("dense", "csr", "indptr"):
        raise ValueError(f"unknown codec role {role!r}")
    if role in ("dense", "csr") and codec.algorithm != "blosc":
        raise ValueError(f"SCC {role} matrix data requires Codec.blosc(), got {codec.algorithm!r}")
    if codec.algorithm == "blosc":
        payload: dict[str, Any] = {
            "id": "blosc1" if role == "dense" else "dyn-blosc",
            "codec": codec.backend,
            "clevel": codec.level,
            "shuffle": codec.shuffle,
            "split_blocks": codec.split_blocks,
        }
        if role == "dense":
            # The private schema requires a value; DenseWriter derives and
            # replaces it from the first-class block partition.
            payload["block_size"] = 1
        return payload
    if codec.algorithm in ("zstd", "zlib"):
        return {"id": codec.algorithm, "level": codec.level}
    return {"id": codec.algorithm}


def _codec_from_wire(payload: Mapping[str, Any]) -> Codec:
    """Collapse private storage variants into the unified public policy."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"codec payload must be a mapping, got {type(payload).__name__}")
    ident = payload.get("id")
    if not isinstance(ident, str):
        raise ValueError("codec payload missing string 'id'")
    if ident in ("dyn-blosc", "blosc1"):
        allowed = {"id", "codec", "clevel", "shuffle", "split_blocks"}
        if ident == "blosc1":
            allowed.add("block_size")
        _reject_wire_unknown(payload, allowed)
        return Codec.blosc(
            backend=payload.get("codec", "lz4"),
            level=payload.get("clevel", 5),
            shuffle=payload.get("shuffle", "none"),
            split_blocks=payload.get("split_blocks", False),
        )
    if ident in ("zstd", "zlib"):
        _reject_wire_unknown(payload, {"id", "level"})
        default = 3 if ident == "zstd" else 6
        return Codec(algorithm=ident, level=payload.get("level", default))
    if ident in ("lz4", "none"):
        _reject_wire_unknown(payload, {"id"})
        return Codec(algorithm=ident)
    raise ValueError(f"unsupported on-disk codec id {ident!r}")


def as_codec(value: Codec | str | Mapping[str, Any], *, name: str = "codec") -> Codec:
    try:
        return Codec.parse(value)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{name}: {error}") from error
