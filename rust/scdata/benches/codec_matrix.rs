//! Synthetic codec matrix benchmark with DecodePool profile coverage.

mod support;

use std::process::ExitCode;
use std::sync::Arc;

use _scdata::codecs::{
    codec_from_spec, codecs_profile_registry, CodecSpec, DecodePool, DecodePoolConfig,
    DecodeRequest,
};
use _scdata::databank::DType;
use support::codecs::{decode_into_checksum, default_codec_matrix, encode_for_spec};
use support::data::{DataDist, DataProfile};
use support::{bench, bench_profiled, env_usize, profile_runtime, BenchConfig};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug, Clone)]
struct Args {
    dtypes: Vec<DType>,
    dists: Vec<DataDist>,
    missing: Vec<u32>,
    chunk_bytes: Vec<usize>,
    num_chunks: usize,
    direct: bool,
    pool: bool,
}

fn run() -> Result<(), String> {
    let config = BenchConfig::from_env();
    let args = parse_args()?;
    let codecs = default_codec_matrix();
    println!(
        "codec_matrix/profiled synth dtypes={:?} dists={:?} missing={:?} chunk_bytes={:?} num_chunks={} codecs={} direct={} pool={}",
        args.dtypes,
        args.dists,
        args.missing,
        args.chunk_bytes,
        args.num_chunks,
        codecs.len(),
        args.direct,
        args.pool
    );

    for dtype in &args.dtypes {
        for dist in &args.dists {
            for &missing in &args.missing {
                for &chunk_bytes in &args.chunk_bytes {
                    let profile = DataProfile {
                        dtype: *dtype,
                        dist: *dist,
                        missing_permille: missing,
                        chunk_bytes,
                        num_chunks: args.num_chunks,
                        seed: 17,
                    };
                    run_profile(config, &profile, &codecs, &args)?;
                }
            }
        }
    }
    Ok(())
}

fn run_profile(
    config: BenchConfig,
    profile: &DataProfile,
    codecs: &[(&'static str, CodecSpec)],
    args: &Args,
) -> Result<(), String> {
    let raw = profile.generate();
    let raw_chunks = raw
        .chunks_exact(profile.chunk_bytes)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();

    for (name, spec) in codecs {
        let encoded = raw_chunks
            .iter()
            .map(|chunk| Arc::<[u8]>::from(encode_for_spec(spec, chunk)))
            .collect::<Vec<_>>();
        let label = format!("codec_matrix/{}/{}", profile.label(), name);

        if args.direct {
            let codec = codec_from_spec(spec);
            let mut idx = 0usize;
            bench(
                config,
                &format!("{label}/direct"),
                default_iters(name),
                Some(profile.chunk_bytes),
                || {
                    let i = idx % encoded.len();
                    idx += 1;
                    decode_into_checksum(&codec, &encoded[i], profile.chunk_bytes)
                },
            );
        }

        if args.pool {
            let runtime = profile_runtime(format!("{label}/pool"), codecs_profile_registry);
            let pool = DecodePool::with_profile(
                DecodePoolConfig {
                    num_workers: env_usize("SCDATA_CODEC_POOL_WORKERS").unwrap_or(4),
                    queue_capacity: env_usize("SCDATA_CODEC_POOL_QUEUE").unwrap_or(512),
                    cpus: None,
                },
                runtime.clone(),
            )
            .map_err(|err| err.to_string())?;
            let mut idx = 0usize;
            bench_profiled(
                config,
                &format!("{label}/pool"),
                default_iters(name),
                Some(profile.chunk_bytes),
                &runtime,
                || {
                    let i = idx % encoded.len();
                    idx += 1;
                    let decoded = pool
                        .submit(
                            DecodeRequest::from_spec(spec, Arc::clone(&encoded[i]))
                                .with_expected_size(profile.chunk_bytes),
                        )
                        .expect("submit decode")
                        .blocking_recv()
                        .expect("decode");
                    decoded.len() ^ decoded[decoded.len() / 2] as usize
                },
            );
        }
    }
    Ok(())
}

fn default_iters(codec: &str) -> usize {
    match codec {
        "bz2_5" | "lzma" | "zlib1" | "gzip5" => 64,
        _ => 256,
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        dtypes: vec![DType::U32, DType::F32],
        dists: vec![
            DataDist::Counting,
            DataDist::LowEntropy,
            DataDist::HighEntropy,
        ],
        missing: vec![0, 250],
        chunk_bytes: vec![64 * 1024],
        num_chunks: env_usize("SCDATA_CODEC_NUM_CHUNKS").unwrap_or(4),
        direct: true,
        pool: true,
    };

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--dtypes" => args.dtypes = parse_dtypes(&next_arg(&mut iter, "--dtypes")?)?,
            "--dists" => args.dists = parse_dists(&next_arg(&mut iter, "--dists")?)?,
            "--missing" => args.missing = parse_csv(&next_arg(&mut iter, "--missing")?)?,
            "--chunk-bytes" => {
                args.chunk_bytes = parse_csv(&next_arg(&mut iter, "--chunk-bytes")?)?
            }
            "--num-chunks" => args.num_chunks = parse_next(&mut iter, "--num-chunks")?,
            "--direct-only" => {
                args.direct = true;
                args.pool = false;
            }
            "--pool-only" => {
                args.direct = false;
                args.pool = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(args)
}

fn parse_dtypes(value: &str) -> Result<Vec<DType>, String> {
    value
        .split(',')
        .map(|token| match token.trim().to_ascii_lowercase().as_str() {
            "u8" => Ok(DType::U8),
            "i8" => Ok(DType::I8),
            "u16" => Ok(DType::U16),
            "i16" => Ok(DType::I16),
            "u32" => Ok(DType::U32),
            "i32" => Ok(DType::I32),
            "u64" => Ok(DType::U64),
            "i64" => Ok(DType::I64),
            "f16" => Ok(DType::F16),
            "bf16" => Ok(DType::BF16),
            "f32" => Ok(DType::F32),
            "f64" => Ok(DType::F64),
            other => Err(format!("unknown dtype `{other}`")),
        })
        .collect()
}

fn parse_dists(value: &str) -> Result<Vec<DataDist>, String> {
    value
        .split(',')
        .map(|token| match token.trim().to_ascii_lowercase().as_str() {
            "uniform" => Ok(DataDist::Uniform),
            "counting" => Ok(DataDist::Counting),
            "constant" => Ok(DataDist::Constant),
            "low_entropy" => Ok(DataDist::LowEntropy),
            "high_entropy" => Ok(DataDist::HighEntropy),
            other => Err(format!("unknown dist `{other}`")),
        })
        .collect()
}

fn parse_csv<T: std::str::FromStr>(value: &str) -> Result<Vec<T>, String> {
    value
        .split(',')
        .map(|token| {
            token
                .trim()
                .parse::<T>()
                .map_err(|_| format!("invalid value `{token}`"))
        })
        .collect()
}

fn next_arg(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_next<T: std::str::FromStr>(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    next_arg(iter, flag)?
        .parse::<T>()
        .map_err(|_| format!("invalid value for {flag}"))
}

fn print_help() {
    println!(
        "cargo bench --bench codec_matrix -- [--dtypes u32,f32] [--dists counting,low_entropy] [--missing 0,250] [--chunk-bytes 65536] [--direct-only|--pool-only]"
    );
}
