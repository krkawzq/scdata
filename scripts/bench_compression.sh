#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/wangzhongqi/Code/Project/scdata}
cd "$PROJECT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

INPUT=${BENCH_INPUT:-/mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Project/PRISM/data/norman/03_cell_cycle/norman_raw_u2000_cell_cycle.h5ad}
OUTPUT_DIR=${BENCH_OUTPUT_DIR:-outputs/bench_compression}
SAMPLE_BYTES_LIST=${BENCH_SAMPLE_BYTES_LIST:-512MiB}
BLOCK_BYTES_LIST=${BENCH_BLOCK_BYTES_LIST:-20KiB,400KiB,1MiB,2MiB,8MiB}
PROFILE_LIST=${BENCH_PROFILE_LIST:-default}
MAX_DATASETS=${BENCH_MAX_DATASETS:-12}
MIN_SAMPLE_PER_DATASET=${BENCH_MIN_SAMPLE_PER_DATASET:-4MiB}
REPEATS=${BENCH_REPEATS:-3}
WARMUPS=${BENCH_WARMUPS:-1}
VERIFY=${BENCH_VERIFY:-all}
DECODE_ORDER=${BENCH_DECODE_ORDER:-sequential}
SEED=${BENCH_SEED:-17}
THREADS=${BENCH_THREADS:-1}
CARGO_BIN=${CARGO_BIN:-$HOME/.cargo/bin/cargo}

COMMON_ARGS=(
  --input "$INPUT"
  --sample-bytes-list "$SAMPLE_BYTES_LIST"
  --block-bytes-list "$BLOCK_BYTES_LIST"
  --profile-list "$PROFILE_LIST"
  --max-datasets "$MAX_DATASETS"
  --min-sample-per-dataset "$MIN_SAMPLE_PER_DATASET"
  --verify "$VERIFY"
  --seed "$SEED"
  --threads "$THREADS"
)

if [[ -n "${BENCH_ONLY_CODEC:-}" ]]; then
  COMMON_ARGS+=(--only-codec "$BENCH_ONLY_CODEC")
fi
if [[ -n "${BENCH_EXCLUDE_CODEC:-}" ]]; then
  COMMON_ARGS+=(--exclude-codec "$BENCH_EXCLUDE_CODEC")
fi
if [[ "${BENCH_SKIP_SLOW:-0}" == "1" ]]; then
  COMMON_ARGS+=(--skip-slow)
fi
if [[ "${BENCH_NO_BASELINE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--no-baseline)
fi

python scripts/bench_compression.py \
  "${COMMON_ARGS[@]}" \
  --repeats "$REPEATS" \
  --warmups "$WARMUPS" \
  --decode-order "$DECODE_ORDER" \
  --output-dir "$OUTPUT_DIR" \
  --sort decode

python scripts/export_numcodecs_payloads.py \
  "${COMMON_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR/rust_payloads"

"$CARGO_BIN" bench --manifest-path rust/scdata/Cargo.toml --bench codec_manifest -- \
  --manifest "$OUTPUT_DIR/rust_payloads/matrix_manifest.json" \
  --output-dir "$OUTPUT_DIR/rust_aligned" \
  --repeats "$REPEATS" \
  --warmups "$WARMUPS" \
  --verify "$VERIFY" \
  --decode-order "$DECODE_ORDER" \
  --seed "$SEED"
