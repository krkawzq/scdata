//! Codec manifest benchmark.
//!
//! Decodes numcodecs-compatible payloads across a matrix of codecs and reports
//! best / median / mean / std throughput. Two modes:
//!
//! 1. **Manifest mode** (`--manifest <path>`): runs a Python-exported manifest
//!    (single run or matrix) of real single-cell chunks. This is the
//!    apples-to-apples path against the Python `bench_compression` results.
//!
//! 2. **Synth mode** (default, no `--manifest`): generates payloads in-process
//!    via `support::data` (normalized data pipeline: dtype × distribution ×
//!    missing-rate × scale), encodes them with every codec in
//!    `support::codecs::default_codec_matrix`, and benches decode. This is the
//!    reproducible coverage sweep — no Python or external data required.
//!
//! ```sh
//! # synth sweep (default)
//! cargo bench --manifest-path rust/scdata/Cargo.toml --bench codec_manifest
//!
//! # real-data manifest from the Python exporter
//! cargo bench --bench codec_manifest -- --manifest outputs/.../matrix_manifest.json \
//!     --output-dir outputs/.../rust_aligned
//! ```

mod support;

use std::path::PathBuf;
use std::process::ExitCode;

use _scdata::codecs::CodecSpec;
use _scdata::databank::DType;
use support::codecs::default_codec_matrix;
use support::data::{DataDist, DataProfile};
use support::manifest::{
    self, DecodeOrder, ManifestBenchArgs, VerifyMode,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug)]
struct Args {
    manifest: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    repeats: usize,
    warmups: usize,
    verify: String,
    decode_order: String,
    seed: u64,
    // Synth-mode knobs (ignored when --manifest is set).
    synth_dtypes: Vec<String>,
    synth_dists: Vec<String>,
    synth_missing: Vec<u32>,
    synth_chunk_bytes: Vec<usize>,
    synth_num_chunks: usize,
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let bench_args = ManifestBenchArgs {
        repeats: args.repeats,
        warmups: args.warmups,
        verify: VerifyMode::parse(&args.verify)?,
        decode_order: DecodeOrder::parse(&args.decode_order)?,
        seed: args.seed,
    };

    if let Some(manifest_path) = &args.manifest {
        let output_dir = args
            .output_dir
            .as_ref()
            .ok_or_else(|| "--output-dir is required with --manifest".to_string())?;
        let manifest_json = serde_json::from_str::<serde_json::Value>(
            &std::fs::read_to_string(manifest_path)
                .map_err(|err| format!("read {}: {err}", manifest_path.display()))?,
        )
        .map_err(|err| format!("parse {}: {err}", manifest_path.display()))?;

        if manifest_json.get("runs").is_some() {
            let runs = manifest::load_matrix_file(manifest_path)?;
            manifest::bench_matrix(&runs, &bench_args, output_dir)
        } else {
            let run = manifest::load_manifest_file(manifest_path)?;
            manifest::bench_run(&run, &bench_args, output_dir)
        }
    } else {
        run_synth(&args, &bench_args)
    }
}

/// Synth mode: sweep dtype × distribution × missing-rate × chunk-size, encode
/// every codec, and bench decode. Writes one run per combination into
/// `output_dir/runs/<label>` plus a matrix summary at `output_dir`.
fn run_synth(args: &Args, bench_args: &ManifestBenchArgs) -> Result<(), String> {
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| support::bench_data_dir().join("codec-manifest-synth"));
    let codecs = default_codec_matrix();
    let dtypes = if args.synth_dtypes.is_empty() {
        vec!["u32".to_string(), "f32".to_string(), "u64".to_string(), "f64".to_string()]
    } else {
        args.synth_dtypes.clone()
    };
    let dists = if args.synth_dists.is_empty() {
        DataDist::ALL.iter().map(|d| d.label().to_string()).collect()
    } else {
        args.synth_dists.clone()
    };
    let missing = if args.synth_missing.is_empty() {
        vec![0, 250, 500, 950]
    } else {
        args.synth_missing.clone()
    };
    let chunk_sizes = if args.synth_chunk_bytes.is_empty() {
        vec![64 * 1024, 1024 * 1024]
    } else {
        args.synth_chunk_bytes.clone()
    };

    let mut runs = Vec::new();
    for dtype_name in &dtypes {
        let dtype = parse_dtype(dtype_name)?;
        for dist_name in &dists {
            let dist = parse_dist(dist_name)?;
            for &miss in &missing {
                for &chunk_bytes in &chunk_sizes {
                    let profile = DataProfile {
                        dtype,
                        dist,
                        missing_permille: miss,
                        chunk_bytes,
                        num_chunks: args.synth_num_chunks,
                        seed: args.seed,
                    };
                    let codec_specs: Vec<(&str, CodecSpec)> =
                        codecs.iter().map(|(name, spec)| (*name, spec.clone())).collect();
                    let run = manifest::synthesize_run(&profile.label(), &profile, &codec_specs)?;
                    runs.push(run);
                }
            }
        }
    }

    println!(
        "codec_manifest synth: {} runs × {} codecs, output_dir={}",
        runs.len(),
        codecs.len(),
        output_dir.display(),
    );
    manifest::bench_matrix(&runs, bench_args, &output_dir)
}

fn parse_dtype(name: &str) -> Result<DType, String> {
    Ok(match name {
        "u8" => DType::U8,
        "i8" => DType::I8,
        "u16" => DType::U16,
        "i16" => DType::I16,
        "u32" => DType::U32,
        "i32" => DType::I32,
        "u64" => DType::U64,
        "i64" => DType::I64,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        "f64" => DType::F64,
        other => return Err(format!("unknown dtype `{other}`")),
    })
}

fn parse_dist(name: &str) -> Result<DataDist, String> {
    Ok(match name {
        "uniform" => DataDist::Uniform,
        "counting" => DataDist::Counting,
        "constant" => DataDist::Constant,
        "low_entropy" => DataDist::LowEntropy,
        "high_entropy" => DataDist::HighEntropy,
        other => return Err(format!("unknown dist `{other}`")),
    })
}

fn parse_args() -> Result<Args, String> {
    let mut manifest = None;
    let mut output_dir = None;
    let mut repeats = 3usize;
    let mut warmups = 1usize;
    let mut verify = "all".to_string();
    let mut decode_order = "sequential".to_string();
    let mut seed = 17u64;
    let mut synth_dtypes = Vec::new();
    let mut synth_dists = Vec::new();
    let mut synth_missing = Vec::new();
    let mut synth_chunk_bytes = Vec::new();
    let mut synth_num_chunks = 8usize;

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(next_arg(&mut iter, "--manifest")?)),
            "--output-dir" => {
                output_dir = Some(PathBuf::from(next_arg(&mut iter, "--output-dir")?))
            }
            "--repeats" => repeats = parse_next(&mut iter, "--repeats")?,
            "--warmups" => warmups = parse_next(&mut iter, "--warmups")?,
            "--verify" => verify = next_arg(&mut iter, "--verify")?,
            "--decode-order" => decode_order = next_arg(&mut iter, "--decode-order")?,
            "--seed" => seed = parse_next(&mut iter, "--seed")?,
            "--synth-dtypes" => {
                synth_dtypes = next_arg(&mut iter, "--synth-dtypes")?
                    .split(',')
                    .map(str::to_string)
                    .collect()
            }
            "--synth-dists" => {
                synth_dists = next_arg(&mut iter, "--synth-dists")?
                    .split(',')
                    .map(str::to_string)
                    .collect()
            }
            "--synth-missing" => {
                synth_missing = next_arg(&mut iter, "--synth-missing")?
                    .split(',')
                    .filter_map(|value| value.parse::<u32>().ok())
                    .collect()
            }
            "--synth-chunk-bytes" => {
                synth_chunk_bytes = next_arg(&mut iter, "--synth-chunk-bytes")?
                    .split(',')
                    .filter_map(|value| value.parse::<usize>().ok())
                    .collect()
            }
            "--synth-num-chunks" => {
                synth_num_chunks = parse_next(&mut iter, "--synth-num-chunks")?
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            // `cargo bench` appends harness flags (`--bench`); ignore them.
            "--bench" | "--nocapture" | "--exact" => {}
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    Ok(Args {
        manifest,
        output_dir,
        repeats,
        warmups,
        verify,
        decode_order,
        seed,
        synth_dtypes,
        synth_dists,
        synth_missing,
        synth_chunk_bytes,
        synth_num_chunks,
    })
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    iter.next().ok_or_else(|| format!("{name} requires a value"))
}

fn parse_next<T>(iter: &mut impl Iterator<Item = String>, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    next_arg(iter, name)?
        .parse::<T>()
        .map_err(|err| format!("invalid {name}: {err}"))
}

fn print_help() {
    println!(
        "codec_manifest [--manifest <manifest.json|matrix_manifest.json>] [--output-dir <dir>]\n\
         \x20             [--repeats N] [--warmups N] [--verify all|first|none]\n\
         \x20             [--decode-order sequential|random] [--seed N]\n\
         \x20             [--synth-dtypes u32,f32,...] [--synth-dists uniform,counting,...]\n\
         \x20             [--synth-missing 0,250,500,950] [--synth-chunk-bytes 65536,1048576]\n\
         \x20             [--synth-num-chunks N]\n\
         \n\
         Without --manifest, runs the synth sweep (normalized data pipeline)."
    );
}
