//! Parameterized IO-pool stress benchmark.
//!
//! With no `--path`, this target creates a deterministic file under
//! `target/bench-data` so `cargo bench --bench iopool` is a real smoke
//! benchmark. Pass `--path <FILE>` to stress an external dataset or filesystem.

mod support;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "uring")]
use _scdata::iopool::UringConfig;
use _scdata::iopool::{BaseIoConfig, IoCommand, IoConfig, IoPool, ThreadedConfig};
use support::{env_usize, payload, stress_mt, BenchConfig};

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
        "iopool parameterized benchmark path={} bytes={} backend={} workers={} shards={} inflight={} threads={:?} read_sizes={:?} duplicate_fanout={}",
        path.display(),
        data_len,
        args.backend,
        args.workers,
        args.shards,
        args.inflight,
        args.threads,
        args.read_sizes,
        args.duplicate_fanout,
    );

    match args.backend.as_str() {
        "threaded" => bench_threaded(config, &args, &path, data_len)?,
        #[cfg(feature = "uring")]
        "uring" => bench_uring(config, &args, &path, data_len)?,
        #[cfg(not(feature = "uring"))]
        "uring" => return Err("io_uring backend is not compiled in".to_string()),
        "both" => {
            bench_threaded(config, &args, &path, data_len)?;
            #[cfg(feature = "uring")]
            bench_uring(config, &args, &path, data_len)?;
        }
        other => return Err(format!("unknown --backend `{other}`")),
    }

    if cleanup {
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}

fn bench_threaded(
    config: BenchConfig,
    args: &Args,
    path: &PathBuf,
    data_len: usize,
) -> Result<(), String> {
    let pool = Arc::new(
        IoPool::new(IoConfig::Threaded(ThreadedConfig {
            base: BaseIoConfig {
                max_in_flight: args.inflight,
                priority_levels: 2,
                queue_shards: args.shards,
                assume_non_overlapping_reads: true,
            },
            num_workers: args.workers,
            cpus: None,
        }))
        .map_err(|err| format!("create threaded iopool: {err}"))?,
    );
    bench_pool(config, "threaded", pool, path, data_len, args)
}

#[cfg(feature = "uring")]
fn bench_uring(
    config: BenchConfig,
    args: &Args,
    path: &PathBuf,
    data_len: usize,
) -> Result<(), String> {
    let pool = match IoPool::new(IoConfig::Uring(UringConfig {
        base: BaseIoConfig {
            max_in_flight: args.inflight,
            priority_levels: 2,
            queue_shards: args.shards,
            assume_non_overlapping_reads: true,
        },
        entries: args.inflight.max(64) as u32,
        drivers: 1,
        iowq_bounded_workers: 0,
        iowq_unbounded_workers: 0,
        registered_files: 64,
    })) {
        Ok(pool) => Arc::new(pool),
        Err(err)
            if matches!(
                err.raw_os_error(),
                Some(libc::ENOSYS | libc::EPERM | libc::EACCES)
            ) =>
        {
            eprintln!("[iopool] skipping io_uring benchmark (unavailable): {err}");
            return Ok(());
        }
        Err(err) => return Err(format!("create uring iopool: {err}")),
    };
    bench_pool(config, "uring", pool, path, data_len, args)
}

fn bench_pool(
    config: BenchConfig,
    backend: &str,
    pool: Arc<IoPool>,
    path: &PathBuf,
    data_len: usize,
    args: &Args,
) -> Result<(), String> {
    let file = pool
        .register_readonly_file(path)
        .map_err(|err| format!("register {}: {err}", path.display()))?;

    for &read_size in &args.read_sizes {
        if read_size > data_len {
            eprintln!("[iopool] skipping {read_size} byte read: larger than file ({data_len})");
            continue;
        }
        let max_offset = data_len - read_size;
        let counter = AtomicUsize::new(0);
        support::bench(
            config,
            &format!("iopool_param/{backend}_read_{}", fmt_bytes(read_size)),
            2048,
            Some(read_size),
            || {
                let idx = counter.fetch_add(1, Ordering::Relaxed);
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
            stress_mt(
                config,
                &format!(
                    "iopool_param/{backend}_mt_read_{}/t{threads}",
                    fmt_bytes(read_size)
                ),
                threads,
                1024,
                Some(read_size),
                move |_| {
                    let idx = counter.fetch_add(1, Ordering::Relaxed);
                    let offset = ((idx * read_size) % (max_offset + 1)) as u64;
                    let bytes = pool
                        .submit(IoCommand::read(file, offset, read_size, 0))
                        .expect("submit read")
                        .blocking_recv_read()
                        .expect("read");
                    bytes.len() ^ bytes[bytes.len() / 2] as usize
                },
            );
        }
    }

    if args.duplicate_fanout > 1 {
        for &threads in &args.threads {
            let pool = Arc::clone(&pool);
            let fanout = args.duplicate_fanout;
            stress_mt(
                config,
                &format!("iopool_param/{backend}_duplicate_64k_x{fanout}/t{threads}"),
                threads,
                512,
                Some(64 * 1024 * fanout),
                move |_| {
                    let futures = (0..fanout)
                        .map(|_| {
                            pool.submit(IoCommand::read(file, 0, 64 * 1024, 0))
                                .expect("submit duplicate read")
                        })
                        .collect::<Vec<_>>();
                    futures.into_iter().fold(0usize, |acc, future| {
                        let bytes = future.blocking_recv_read().expect("read duplicate");
                        acc ^ bytes.len() ^ bytes[bytes.len() / 2] as usize
                    })
                },
            );
        }
    }

    pool.unregister_file(file)
        .map_err(|err| format!("unregister {}: {err}", path.display()))?;
    Ok(())
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
        threads: vec![2, 4, 8],
        read_sizes: vec![4 * 1024, 64 * 1024, 1024 * 1024],
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
                args.read_sizes = parse_read_sizes(&next_arg(&mut iter, "--read-sizes")?)?
            }
            "--duplicate-fanout" => {
                args.duplicate_fanout = parse_next(&mut iter, "--duplicate-fanout")?
            }
            "--backend" => args.backend = next_arg(&mut iter, "--backend")?,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--bench" | "--nocapture" | "--exact" => {}
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    if args.workers == 0 || args.shards == 0 || args.inflight == 0 {
        return Err("workers, shards, and inflight must be nonzero".to_string());
    }
    if args.threads.is_empty() || args.read_sizes.is_empty() {
        return Err("threads and read-sizes must be non-empty".to_string());
    }
    args.backend = args.backend.to_ascii_lowercase();
    Ok(args)
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{name} requires a value"))
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

fn parse_csv_usize(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|err| format!("invalid integer `{part}`: {err}"))
        })
        .collect()
}

fn parse_read_sizes(value: &str) -> Result<Vec<usize>, String> {
    value.split(',').map(parse_size).collect()
}

fn parse_size(value: &str) -> Result<usize, String> {
    let value = value.trim().to_ascii_lowercase();
    let (number, scale) = match value.as_bytes().last().copied() {
        Some(b'k') => (&value[..value.len() - 1], 1024usize),
        Some(b'm') => (&value[..value.len() - 1], 1024usize * 1024),
        _ => (value.as_str(), 1usize),
    };
    number
        .parse::<usize>()
        .map(|n| n * scale)
        .map_err(|err| format!("invalid size `{value}`: {err}"))
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
        "iopool [--path FILE] [--file-mib N] [--backend threaded|uring|both]\n\
         \x20      [--workers N] [--shards N] [--inflight N]\n\
         \x20      [--threads 2,4,8] [--read-sizes 4k,64k,1m]\n\
         \x20      [--duplicate-fanout N]\n\
         \n\
         Env aliases: SCDATA_IOPOOL_FILE_MIB, SCDATA_IOPOOL_BACKEND,\n\
         SCDATA_IOPOOL_WORKERS, SCDATA_IOPOOL_SHARDS,\n\
         SCDATA_IOPOOL_INFLIGHT, SCDATA_IOPOOL_DUPLICATE_FANOUT."
    );
}
