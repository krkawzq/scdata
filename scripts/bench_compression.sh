#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Project/scdata}
cd "$PROJECT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

enabled() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

append_if_set() {
  local flag=$1
  local value=${2:-}
  if [[ -n "$value" ]]; then
    COMMON_ARGS+=("$flag" "$value")
  fi
}

PYTHON_BIN=${PYTHON_BIN:-python}
CARGO_BIN=${CARGO_BIN:-$HOME/.cargo/bin/cargo}

PRISM_DATA_DIR=/mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Project/PRISM/data
DEFAULT_INPUT_DIR=$PRISM_DATA_DIR/norman/03_cell_cycle
DEFAULT_INPUT=$DEFAULT_INPUT_DIR/norman_raw_u2000_cell_cycle.h5ad
INPUT=${BENCH_INPUT:-$DEFAULT_INPUT}
OUTPUT_DIR=${BENCH_OUTPUT_DIR:-outputs/bench_compression}
PAYLOAD_DIR=${BENCH_PAYLOAD_DIR:-$OUTPUT_DIR/rust_payloads}
RUST_OUTPUT_DIR=${BENCH_RUST_OUTPUT_DIR:-$OUTPUT_DIR/rust_aligned}
SAMPLE_BYTES_LIST=${BENCH_SAMPLE_BYTES_LIST:-512MiB}
BLOCK_BYTES_LIST=${BENCH_BLOCK_BYTES_LIST:-20KiB,400KiB,1MiB,2MiB,8MiB}
PROFILE_LIST=${BENCH_PROFILE_LIST:-default}
MAX_DATASETS=${BENCH_MAX_DATASETS:-12}
MIN_DATASET_BYTES=${BENCH_MIN_DATASET_BYTES:-1MiB}
MIN_SAMPLE_PER_DATASET=${BENCH_MIN_SAMPLE_PER_DATASET:-4MiB}
SELECTION=${BENCH_SELECTION:-stratified}
REPEATS=${BENCH_REPEATS:-3}
WARMUPS=${BENCH_WARMUPS:-1}
VERIFY=${BENCH_VERIFY:-all}
DECODE_ORDER=${BENCH_DECODE_ORDER:-sequential}
SEED=${BENCH_SEED:-17}
THREADS=${BENCH_THREADS:-1}
SORT=${BENCH_SORT:-decode}
CARGO_OFFLINE=${BENCH_CARGO_OFFLINE:-0}
RUN_PYTHON=${BENCH_RUN_PYTHON:-1}
RUN_EXPORT=${BENCH_RUN_EXPORT:-1}
RUN_RUST=${BENCH_RUN_RUST:-1}

COMMON_ARGS=(
  --input "$INPUT"
  --sample-bytes-list "$SAMPLE_BYTES_LIST"
  --block-bytes-list "$BLOCK_BYTES_LIST"
  --profile-list "$PROFILE_LIST"
  --max-datasets "$MAX_DATASETS"
  --min-dataset-bytes "$MIN_DATASET_BYTES"
  --min-sample-per-dataset "$MIN_SAMPLE_PER_DATASET"
  --selection "$SELECTION"
  --verify "$VERIFY"
  --seed "$SEED"
  --threads "$THREADS"
)

append_if_set --dataset "${BENCH_DATASET:-}"
append_if_set --exclude-dataset "${BENCH_EXCLUDE_DATASET:-}"
append_if_set --blosc-shuffle "${BENCH_BLOSC_SHUFFLE:-}"
append_if_set --only-codec "${BENCH_ONLY_CODEC:-}"
append_if_set --exclude-codec "${BENCH_EXCLUDE_CODEC:-}"

if enabled "${BENCH_INCLUDE_OPTIONAL:-0}"; then
  COMMON_ARGS+=(--include-optional)
fi
if enabled "${BENCH_SKIP_SLOW:-0}"; then
  COMMON_ARGS+=(--skip-slow)
fi
if enabled "${BENCH_NO_BASELINE:-0}"; then
  COMMON_ARGS+=(--no-baseline)
fi

if enabled "$RUN_PYTHON"; then
  "$PYTHON_BIN" scripts/bench_compression.py \
    "${COMMON_ARGS[@]}" \
    --repeats "$REPEATS" \
    --warmups "$WARMUPS" \
    --decode-order "$DECODE_ORDER" \
    --output-dir "$OUTPUT_DIR" \
    --sort "$SORT"
fi

if enabled "$RUN_EXPORT"; then
  "$PYTHON_BIN" scripts/export_numcodecs_payloads.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$PAYLOAD_DIR"
fi

if enabled "$RUN_RUST"; then
  CARGO_ARGS=(bench --manifest-path rust/scdata/Cargo.toml --bench codec_manifest)
  if enabled "$CARGO_OFFLINE"; then
    CARGO_ARGS+=(--offline)
  fi
  "$CARGO_BIN" "${CARGO_ARGS[@]}" -- \
    --manifest "$PAYLOAD_DIR/matrix_manifest.json" \
    --output-dir "$RUST_OUTPUT_DIR" \
    --repeats "$REPEATS" \
    --warmups "$WARMUPS" \
    --verify "$VERIFY" \
    --decode-order "$DECODE_ORDER" \
    --seed "$SEED"
fi
