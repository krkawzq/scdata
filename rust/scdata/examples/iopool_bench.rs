use std::env;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[cfg(feature = "uring")]
use _scdata::iopool::UringConfig;
use _scdata::iopool::{BaseIoConfig, IoCommand, IoConfig, IoFuture, IoPool, ThreadedConfig};

#[derive(Clone, Copy)]
struct SubmitterConfig {
    file: usize,
    read_size: usize,
    priority: usize,
    max_offset: u64,
    stride: u64,
    submitters: usize,
}

#[derive(Debug)]
struct Args {
    path: PathBuf,
    backend: String,
    read_size: usize,
    iters: usize,
    depth: usize,
    priority: usize,
    submitters: usize,
    queue_shards: usize,
    entries: u32,
    drivers: usize,
    iowq_bounded_workers: u32,
    iowq_unbounded_workers: u32,
    workers: usize,
    registered_files: u32,
    assume_non_overlapping_reads: bool,
}

impl Args {
    fn parse() -> io::Result<Self> {
        let mut args = Self {
            path: PathBuf::new(),
            backend: "uring".to_string(),
            read_size: 1024 * 1024,
            iters: 4096,
            depth: 128,
            priority: 0,
            submitters: 1,
            queue_shards: 1,
            entries: 256,
            drivers: 1,
            iowq_bounded_workers: 0,
            iowq_unbounded_workers: 0,
            workers: 8,
            registered_files: 4096,
            assume_non_overlapping_reads: false,
        };

        let mut iter = env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--path" => args.path = PathBuf::from(value(&mut iter, "--path")?),
                "--backend" => args.backend = value(&mut iter, "--backend")?,
                "--read-size" => args.read_size = parse_usize(&mut iter, "--read-size")?,
                "--iters" => args.iters = parse_usize(&mut iter, "--iters")?,
                "--depth" => args.depth = parse_usize(&mut iter, "--depth")?,
                "--priority" => args.priority = parse_usize(&mut iter, "--priority")?,
                "--submitters" => args.submitters = parse_usize(&mut iter, "--submitters")?,
                "--queue-shards" => args.queue_shards = parse_usize(&mut iter, "--queue-shards")?,
                "--entries" => args.entries = parse_u32(&mut iter, "--entries")?,
                "--drivers" => args.drivers = parse_usize(&mut iter, "--drivers")?,
                "--iowq-bounded-workers" => {
                    args.iowq_bounded_workers = parse_u32(&mut iter, "--iowq-bounded-workers")?
                }
                "--iowq-unbounded-workers" => {
                    args.iowq_unbounded_workers = parse_u32(&mut iter, "--iowq-unbounded-workers")?
                }
                "--workers" => args.workers = parse_usize(&mut iter, "--workers")?,
                "--registered-files" => {
                    args.registered_files = parse_u32(&mut iter, "--registered-files")?
                }
                "--assume-non-overlapping-reads" => {
                    args.assume_non_overlapping_reads = true;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unknown argument {other}"),
                    ));
                }
            }
        }

        if args.path.as_os_str().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--path is required",
            ));
        }
        if args.read_size == 0
            || args.iters == 0
            || args.depth == 0
            || args.submitters == 0
            || args.queue_shards == 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--read-size, --iters, --depth, --submitters, and --queue-shards must be greater than 0",
            ));
        }

        Ok(args)
    }
}

fn main() -> io::Result<()> {
    if env::args_os().len() == 1 {
        print_usage();
        eprintln!("no --path supplied; skipping iopool bench");
        return Ok(());
    }

    let args = Args::parse()?;
    let pool = Arc::new(make_pool(&args)?);
    let file = pool.register_readonly_file(&args.path)?;
    let file_len = std::fs::metadata(&args.path)?.len();
    let started = Instant::now();
    let (completed, bytes) = run_submitters(Arc::clone(&pool), file, file_len, &args)?;

    let elapsed = started.elapsed();
    let seconds = elapsed.as_secs_f64();
    let mib = bytes as f64 / (1024.0 * 1024.0);
    println!(
        "backend={} read_size={} depth={} submitters={} queue_shards={} nonoverlap_reads={} entries={} drivers={} iowq_bounded={} completed={} elapsed_s={:.6} throughput_mib_s={:.2} ops_s={:.2}",
        args.backend,
        args.read_size,
        args.depth,
        args.submitters,
        args.queue_shards,
        args.assume_non_overlapping_reads,
        args.entries,
        args.drivers,
        args.iowq_bounded_workers,
        completed,
        seconds,
        mib / seconds,
        completed as f64 / seconds
    );

    Ok(())
}

fn run_submitters(
    pool: Arc<IoPool>,
    file: usize,
    file_len: u64,
    args: &Args,
) -> io::Result<(usize, usize)> {
    let max_offset = file_len - args.read_size as u64;
    let stride = args.read_size as u64;
    let mut handles = Vec::with_capacity(args.submitters);

    for submitter in 0..args.submitters {
        let pool = Arc::clone(&pool);
        let iters = split_count(args.iters, args.submitters, submitter);
        let depth = split_count(args.depth, args.submitters, submitter).max(1);
        let read_size = args.read_size;
        let priority = args.priority;
        let submitters = args.submitters;
        let config = SubmitterConfig {
            file,
            read_size,
            priority,
            max_offset,
            stride,
            submitters,
        };
        handles.push(thread::spawn(move || {
            run_submitter(pool, config, submitter, iters, depth)
        }));
    }

    let mut completed = 0usize;
    let mut bytes = 0usize;
    for handle in handles {
        let result = handle
            .join()
            .map_err(|_| io::Error::other("submitter thread panicked"))??;
        completed += result.0;
        bytes += result.1;
    }
    Ok((completed, bytes))
}

fn run_submitter(
    pool: Arc<IoPool>,
    config: SubmitterConfig,
    submitter: usize,
    iters: usize,
    depth: usize,
) -> io::Result<(usize, usize)> {
    let mut next = submitter;
    let mut submitted = 0usize;
    let mut completed = 0usize;
    let mut bytes = 0usize;
    let mut in_flight = Vec::<IoFuture>::with_capacity(depth);
    let mut cursor = 0usize;

    while submitted < iters && in_flight.len() < depth {
        in_flight.push(pool.submit(IoCommand::read(
            config.file,
            next_offset(next, config.stride, config.max_offset),
            config.read_size,
            config.priority,
        ))?);
        submitted += 1;
        next += config.submitters;
    }

    while !in_flight.is_empty() {
        let data = match try_recv_any(&mut in_flight, &mut cursor)? {
            Some(data) => data,
            None => {
                if cursor >= in_flight.len() {
                    cursor = 0;
                }
                in_flight.swap_remove(cursor).blocking_recv_read()?
            }
        };
        bytes += data.len();
        completed += 1;

        if submitted < iters {
            in_flight.push(pool.submit(IoCommand::read(
                config.file,
                next_offset(next, config.stride, config.max_offset),
                config.read_size,
                config.priority,
            ))?);
            submitted += 1;
            next += config.submitters;
        }
    }

    Ok((completed, bytes))
}

fn try_recv_any(
    in_flight: &mut Vec<IoFuture>,
    cursor: &mut usize,
) -> io::Result<Option<Arc<[u8]>>> {
    let len = in_flight.len();
    for _ in 0..len {
        if *cursor >= in_flight.len() {
            *cursor = 0;
        }
        if let Some(data) = in_flight[*cursor].try_recv_read()? {
            in_flight.swap_remove(*cursor);
            return Ok(Some(data));
        }
        *cursor += 1;
    }
    Ok(None)
}

fn split_count(total: usize, shards: usize, index: usize) -> usize {
    total / shards + usize::from(index < total % shards)
}

fn make_pool(args: &Args) -> io::Result<IoPool> {
    let per_backend_depth = args.depth.max(args.entries as usize * args.drivers.max(1));
    let base = BaseIoConfig {
        max_in_flight: per_backend_depth.saturating_mul(args.queue_shards.max(1)),
        priority_levels: args.priority + 1,
        queue_shards: args.queue_shards,
        assume_non_overlapping_reads: args.assume_non_overlapping_reads,
    };

    match args.backend.as_str() {
        "threaded" => IoPool::new(IoConfig::Threaded(ThreadedConfig {
            base,
            num_workers: args.workers,
            cpus: None,
        })),
        "uring" => {
            #[cfg(feature = "uring")]
            {
                IoPool::new(IoConfig::Uring(UringConfig {
                    base,
                    entries: args.entries,
                    drivers: args.drivers,
                    iowq_bounded_workers: args.iowq_bounded_workers,
                    iowq_unbounded_workers: args.iowq_unbounded_workers,
                    registered_files: args.registered_files,
                }))
            }
            #[cfg(not(feature = "uring"))]
            {
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "uring backend requires the 'uring' feature",
                ))
            }
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported backend {other}"),
        )),
    }
}

fn next_offset(index: usize, stride: u64, max_offset: u64) -> u64 {
    if max_offset == 0 {
        return 0;
    }
    (index as u64).wrapping_mul(stride) % (max_offset + 1)
}

fn value(iter: &mut impl Iterator<Item = String>, flag: &str) -> io::Result<String> {
    iter.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{flag} requires a value"),
        )
    })
}

fn parse_usize(iter: &mut impl Iterator<Item = String>, flag: &str) -> io::Result<usize> {
    value(iter, flag)?
        .parse::<usize>()
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, format!("{flag}: {err}")))
}

fn parse_u32(iter: &mut impl Iterator<Item = String>, flag: &str) -> io::Result<u32> {
    value(iter, flag)?
        .parse::<u32>()
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, format!("{flag}: {err}")))
}

fn print_usage() {
    eprintln!(
        "usage: iopool_bench --path FILE [--backend uring|threaded] [--read-size BYTES] [--iters N] [--depth N] [--submitters N] [--queue-shards N] [--entries N] [--drivers N] [--iowq-bounded-workers N] [--iowq-unbounded-workers N] [--workers N] [--registered-files N] [--assume-non-overlapping-reads]"
    );
}
