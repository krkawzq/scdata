use _scdata::databank::NativeLoadCoalesceConfig;
use _scdata::synthetic::{run_native_real_io_bench, NativeRealIoConfig, NativeSyntheticOrder};

fn main() {
    match parse_args().and_then(run_native_real_io_bench) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize report")
        ),
        Err(err) => {
            eprintln!("native_real_io: {err}");
            std::process::exit(2);
        }
    }
}

fn parse_args() -> Result<NativeRealIoConfig, String> {
    let mut config = NativeRealIoConfig::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let (key, value) = if let Some((key, value)) = arg.split_once('=') {
            (key.to_string(), value.to_string())
        } else {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for argument {arg}"))?;
            (arg, value)
        };
        match key.as_str() {
            "--backend" => config.backend = _scdata::synthetic::RealIoBackend::parse(&value)?,
            "--io-workers" => config.io_workers = parse_usize(&value, &key)?,
            "--uring-entries" => config.uring_entries = parse_u32(&value, &key)?,
            "--max-in-flight" => config.max_in_flight = parse_usize(&value, &key)?,
            "--queue-capacity" => config.queue_capacity = parse_usize(&value, &key)?,
            "--queue-shards" => config.queue_shards = parse_usize(&value, &key)?,
            "--data-dir" => config.data_dir = value.into(),
            "--workers" => config.workers = parse_usize(&value, &key)?,
            "--batches" => config.batches = parse_usize(&value, &key)?,
            "--warmup-batches" => config.warmup_batches = parse_usize(&value, &key)?,
            "--batch-size" => config.batch_size = parse_usize(&value, &key)?,
            "--chunks" => config.chunks = parse_usize(&value, &key)?,
            "--genes" => config.genes = parse_usize(&value, &key)?,
            "--cells-per-chunk" => config.cells_per_chunk = parse_usize(&value, &key)?,
            "--cell-bytes" => config.cell_bytes = parse_usize(&value, &key)?,
            "--block-size" => config.block_size = parse_usize(&value, &key)?,
            "--typesize" => config.typesize = parse_usize(&value, &key)?,
            "--shuffle" => config.shuffle = parse_bool(&value, &key)?,
            "--entropy-fraction" => config.entropy_fraction = parse_f32(&value, &key)?,
            "--order" => config.order = NativeSyntheticOrder::parse(&value)?,
            "--continuation-p" => config.continuation_p = parse_f64(&value, &key)?,
            "--seed" => config.seed = parse_u64(&value, &key)?,
            "--coalesce-max-gap-bytes" => {
                config.coalesce.max_gap_bytes = parse_usize(&value, &key)?
            }
            "--coalesce-max-merged-len" => {
                config.coalesce.max_merged_len = parse_usize(&value, &key)?
            }
            "--coalesce-max-waste-ratio" => {
                config.coalesce.max_waste_ratio = parse_f32(&value, &key)?
            }
            "--coalesce-min-children" => config.coalesce.min_children = parse_usize(&value, &key)?,
            "--coalesce-max-window-us" => config.coalesce.max_window_us = parse_u32(&value, &key)?,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    config.coalesce = NativeLoadCoalesceConfig {
        max_window_us: config.coalesce.max_window_us,
        max_merged_len: config.coalesce.max_merged_len,
        max_gap_bytes: config.coalesce.max_gap_bytes,
        max_waste_ratio: config.coalesce.max_waste_ratio,
        min_children: config.coalesce.min_children,
    };
    Ok(config)
}

fn parse_usize(value: &str, key: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("{key} must be usize: {err}"))
}

fn parse_u32(value: &str, key: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|err| format!("{key} must be u32: {err}"))
}

fn parse_u64(value: &str, key: &str) -> Result<u64, String> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).map_err(|err| format!("{key} must be u64: {err}"))
    } else {
        value
            .parse::<u64>()
            .map_err(|err| format!("{key} must be u64: {err}"))
    }
}

fn parse_f32(value: &str, key: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|err| format!("{key} must be f32: {err}"))
}

fn parse_f64(value: &str, key: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|err| format!("{key} must be f64: {err}"))
}

fn parse_bool(value: &str, key: &str) -> Result<bool, String> {
    match value {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!("{key} must be bool, got {other:?}")),
    }
}

fn print_help() {
    println!(
        r#"native_real_io

Blosc-LZ4 native loader IO benchmark against a real file, through a
configurable IoPool backend (threaded / io_uring). Throughput is capped by
the filesystem (e.g. GPFS), not an in-memory virtual backend.

Backend / pool:
  --backend threaded|uring        (default: threaded)
  --io-workers N                  (threaded num_workers / uring drivers; default 48)
  --uring-entries N               (SQ/CQ depth; default 1024; uring only)
  --max-in-flight N               (default 1024)
  --queue-capacity N              (default 4096)
  --queue-shards N                (default 8)

Workload:
  --data-dir PATH                 (where to write the bench chunk file; default .)
  --workers N                     (concurrent native worker threads; default 4)
  --batches N                     (timed batches; default 2048)
  --warmup-batches N              (untimed warmup; default 64)
  --batch-size N                  (cells per batch; default 128)

Chunk geometry:
  --chunks N                      (default 64)
  --cells-per-chunk N             (default 2048)
  --cell-bytes N                  (default 12288 = 12 KiB)
  --block-size N                  (blosc block size; default 196608 = 192 KiB)
  --typesize N                    (default 2)
  --shuffle true|false            (default true)
  --entropy-fraction F            (default 0.33)
  --order random|sequential|continuity  (default random)
  --continuation-p P              (default 0.0)
  --seed N                        (default 0x5eed_5eed_1234_5678)

Coalesce:
  --coalesce-max-gap-bytes N      (default 1048576)
  --coalesce-max-merged-len N     (default 8388608)
  --coalesce-max-waste-ratio F    (default 0.90)
  --coalesce-min-children N       (default 2)
  --coalesce-max-window-us N      (default 0)

Example (48-thread uring vs threaded comparison):
  native_real_io --backend uring     --io-workers 48 --data-dir /mnt/.../tmp
  native_real_io --backend threaded  --io-workers 48 --data-dir /mnt/.../tmp
"#
    )
}
