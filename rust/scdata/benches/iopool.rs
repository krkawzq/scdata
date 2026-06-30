//! Parameterized IO pool benchmark with profile snapshots.

mod support;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "uring")]
use _scdata::iopool::UringConfig;
use _scdata::iopool::{BaseIoConfig, IoCommand, IoConfig, IoPool, ThreadedConfig};
use _scdata::profile::ProfileRegistry;
use support::{bench_profiled, env_usize, payload, profile_runtime, stress_mt, BenchConfig};

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
    path: Option<PathBuf>,
    file_mib: usize,
    workers: usize,
    shards: usize,
    inflight: usize,
    threads: Vec<usize>,
    read_sizes: Vec<usize>,
    duplicate_fanout: usize,
    backend: String,
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let config = BenchConfig::from_env();
    let (path, cleanup) = prepare_file(&args)?;
    let data_len = std::fs::metadata(&path)
        .map_err(|err| format!("metadata {}: {err}", path.display()))?
        .len() as usize;
    if data_len == 0 {
        return Err(format!("{} is empty", path.display()));
    }

    println!(
        "iopool/profiled path={} bytes={} backend={} workers={} shards={} inflight={} threads={:?} read_sizes={:?} duplicate_fanout={}",
        path.display(),
        data_len,
        args.backend,
        args.workers,
        args.shards,
        args.inflight,
        args.threads,
        args.read_sizes,
        args.duplicate_fanout
    );

    match args.backend.as_str() {
        "threaded" => bench_backend(
            config,
            &args,
            &path,
            data_len,
            "threaded",
            threaded_config(&args),
        )?,
        #[cfg(feature = "uring")]
        "uring" => bench_backend(config, &args, &path, data_len, "uring", uring_config(&args))?,
        #[cfg(not(feature = "uring"))]
        "uring" => return Err("io_uring backend is not compiled in".to_string()),
        "both" => {
            bench_backend(
                config,
                &args,
                &path,
                data_len,
                "threaded",
                threaded_config(&args),
            )?;
            #[cfg(feature = "uring")]
            bench_backend(config, &args, &path, data_len, "uring", uring_config(&args))?;
        }
        other => return Err(format!("unknown --backend `{other}`")),
    }

    if cleanup {
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}

fn bench_backend(
    config: BenchConfig,
    args: &Args,
    path: &Path,
    data_len: usize,
    backend: &str,
    io_config: IoConfig,
) -> Result<(), String> {
    let runtime = profile_runtime(format!("iopool-{backend}"), ProfileRegistry::new);
    let pool = match IoPool::new_with_profile(io_config, runtime.clone()) {
        Ok(pool) => Arc::new(pool),
        #[cfg(feature = "uring")]
        Err(err)
            if backend == "uring"
                && matches!(
                    err.raw_os_error(),
                    Some(libc::ENOSYS | libc::EPERM | libc::EACCES)
                ) =>
        {
            eprintln!("[iopool] skipping io_uring benchmark (unavailable): {err}");
            return Ok(());
        }
        Err(err) => return Err(format!("create {backend} iopool: {err}")),
    };
    let file = pool
        .register_readonly_file(path)
        .map_err(|err| format!("register {}: {err}", path.display()))?;

    for &read_size in &args.read_sizes {
        if read_size > data_len {
            eprintln!("[iopool] skip {read_size} byte read: file has {data_len} bytes");
            continue;
        }
        let max_offset = data_len - read_size;
        let mut counter = 0usize;
        bench_profiled(
            config,
            &format!("iopool/{backend}/read_{}", fmt_bytes(read_size)),
            2048,
            Some(read_size),
            &runtime,
            || {
                let idx = counter;
                counter += 1;
                let offset = ((idx * read_size) % (max_offset + 1)) as u64;
                let bytes = pool
                    .submit(IoCommand::read(file, offset, read_size, 0))
                    .expect("submit read")
                    .blocking_recv_read()
                    .expect("read");
                bytes.len() ^ bytes[bytes.len() / 2] as usize
            },
        );

        for &threads in &args.threads {
            let pool = Arc::clone(&pool);
            let counter = Arc::new(AtomicUsize::new(0));
            let runtime = runtime.clone();
            let round = runtime.start();
            stress_mt(
                config,
                &format!("iopool/{backend}/mt_read_{}", fmt_bytes(read_size)),
                threads,
                2048,
                Some(read_size),
                move |_| {
                    let idx = counter.fetch_add(1, Ordering::Relaxed);
                    let offset = ((idx * read_size) % (max_offset + 1)) as u64;
                    let bytes = pool
                        .submit(IoCommand::read(file, offset, read_size, 0))
                        .expect("submit mt read")
                        .blocking_recv_read()
                        .expect("mt read");
                    bytes.len() ^ bytes[bytes.len() / 2] as usize
                },
            );
            support::maybe_print_profile_snapshot(
                config,
                &format!("iopool/{backend}/mt_read_{}", fmt_bytes(read_size)),
                &round.end(),
            );
        }
    }

    if args.duplicate_fanout > 1 {
        let fanout = args.duplicate_fanout;
        let pool = Arc::clone(&pool);
        bench_profiled(
            config,
            &format!("iopool/{backend}/duplicate_read_64k_x{fanout}"),
            512,
            Some(64 * 1024 * fanout),
            &runtime,
            move || {
                let futures = (0..fanout)
                    .map(|_| {
                        pool.submit(IoCommand::read(file, 0, 64 * 1024, 0))
                            .expect("submit duplicate read")
                    })
                    .collect::<Vec<_>>();
                futures.into_iter().fold(0usize, |acc, future| {
                    let bytes = future.blocking_recv_read().expect("duplicate read");
                    acc ^ bytes.len() ^ bytes[bytes.len() / 2] as usize
                })
            },
        );
    }

    pool.unregister_file(file)
        .map_err(|err| format!("unregister {}: {err}", path.display()))?;
    Ok(())
}

fn threaded_config(args: &Args) -> IoConfig {
    IoConfig::Threaded(ThreadedConfig {
        base: base_config(args),
        num_workers: args.workers,
        cpus: None,
    })
}

#[cfg(feature = "uring")]
fn uring_config(args: &Args) -> IoConfig {
    IoConfig::Uring(UringConfig {
        base: base_config(args),
        entries: args.inflight.max(64) as u32,
        drivers: 1,
        iowq_bounded_workers: 0,
        iowq_unbounded_workers: 0,
        registered_files: 64,
    })
}

fn base_config(args: &Args) -> BaseIoConfig {
    BaseIoConfig {
        max_in_flight: args.inflight,
        priority_levels: 2,
        queue_shards: args.shards,
        assume_non_overlapping_reads: true,
    }
}

fn prepare_file(args: &Args) -> Result<(PathBuf, bool), String> {
    if let Some(path) = &args.path {
        return Ok((path.clone(), false));
    }
    let path = support::bench_data_dir().join(format!("iopool-param-{}.bin", std::process::id()));
    let bytes = args.file_mib.max(1) * 1024 * 1024;
    std::fs::write(&path, payload(bytes))
        .map_err(|err| format!("write {}: {err}", path.display()))?;
    Ok((path, true))
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        path: None,
        file_mib: env_usize("SCDATA_IOPOOL_FILE_MIB").unwrap_or(8),
        workers: env_usize("SCDATA_IOPOOL_WORKERS").unwrap_or(4),
        shards: env_usize("SCDATA_IOPOOL_SHARDS").unwrap_or(2),
        inflight: env_usize("SCDATA_IOPOOL_INFLIGHT").unwrap_or(128),
        threads: parse_csv_usize_env("SCDATA_IOPOOL_THREADS").unwrap_or_else(|| vec![2, 4, 8]),
        read_sizes: parse_csv_bytes_env("SCDATA_IOPOOL_READ_SIZES")
            .unwrap_or_else(|| vec![4 * 1024, 64 * 1024, 1024 * 1024]),
        duplicate_fanout: env_usize("SCDATA_IOPOOL_DUPLICATE_FANOUT").unwrap_or(8),
        backend: std::env::var("SCDATA_IOPOOL_BACKEND").unwrap_or_else(|_| "threaded".to_string()),
    };

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--path" => args.path = Some(PathBuf::from(next_arg(&mut iter, "--path")?)),
            "--file-mib" => args.file_mib = parse_next(&mut iter, "--file-mib")?,
            "--workers" => args.workers = parse_next(&mut iter, "--workers")?,
            "--shards" => args.shards = parse_next(&mut iter, "--shards")?,
            "--inflight" => args.inflight = parse_next(&mut iter, "--inflight")?,
            "--threads" => args.threads = parse_csv_usize(&next_arg(&mut iter, "--threads")?)?,
            "--read-sizes" => {
                args.read_sizes = parse_csv_bytes(&next_arg(&mut iter, "--read-sizes")?)?
            }
            "--duplicate-fanout" => {
                args.duplicate_fanout = parse_next(&mut iter, "--duplicate-fanout")?
            }
            "--backend" => args.backend = next_arg(&mut iter, "--backend")?,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(args)
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

fn parse_csv_usize_env(name: &str) -> Option<Vec<usize>> {
    std::env::var(name)
        .ok()
        .and_then(|value| parse_csv_usize(&value).ok())
}

fn parse_csv_bytes_env(name: &str) -> Option<Vec<usize>> {
    std::env::var(name)
        .ok()
        .and_then(|value| parse_csv_bytes(&value).ok())
}

fn parse_csv_usize(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|token| {
            token
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid usize `{token}`"))
        })
        .collect()
}

fn parse_csv_bytes(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|token| parse_bytes(token.trim()))
        .collect()
}

fn parse_bytes(value: &str) -> Result<usize, String> {
    let lower = value.to_ascii_lowercase();
    if let Some(number) = lower
        .strip_suffix("kib")
        .or_else(|| lower.strip_suffix('k'))
    {
        return number
            .parse::<usize>()
            .map(|n| n * 1024)
            .map_err(|_| format!("invalid byte size `{value}`"));
    }
    if let Some(number) = lower
        .strip_suffix("mib")
        .or_else(|| lower.strip_suffix('m'))
    {
        return number
            .parse::<usize>()
            .map(|n| n * 1024 * 1024)
            .map_err(|_| format!("invalid byte size `{value}`"));
    }
    lower
        .parse::<usize>()
        .map_err(|_| format!("invalid byte size `{value}`"))
}

fn fmt_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{}m", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{}k", bytes / 1024)
    } else {
        format!("{bytes}b")
    }
}

fn print_help() {
    println!(
        "cargo bench --bench iopool -- [--path FILE] [--backend threaded|uring|both] [--read-sizes 4k,64k,1m] [--threads 2,4,8]"
    );
}
