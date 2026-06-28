# scdata

`scdata` is a Python package for single-cell data storage, compression, and
loading experiments. The Python package lives in `scdata/`; the Rust core lives
in `rust/scdata/` and is built as the private extension module
`scdata._scdata`.

## Development

Install the Python development environment and build the Rust extension from
the repository root:

```sh
labpon
export PATH=/home/wangzhongqi/.local/bin:/home/wangzhongqi/.cargo/bin:$PATH
uv sync --extra dev
uv run maturin develop --uv
```

Check the Rust crate directly:

```sh
labpon && /home/wangzhongqi/.cargo/bin/cargo check --manifest-path rust/scdata/Cargo.toml
```

Smoke-test the Python import after building:

```sh
/home/wangzhongqi/.local/bin/uv run python -c "import scdata; print(scdata.kernel_name(), scdata.kernel_version())"
```
