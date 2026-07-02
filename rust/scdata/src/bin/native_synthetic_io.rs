use _scdata::databank::{NativeLoadCoalesceConfig, ProjectedSparseDataGroupStrategy};
use _scdata::synthetic::{run_native_synthetic, NativeSyntheticConfig, NativeSyntheticOrder};

fn main() {
    match parse_args().and_then(run_native_synthetic) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize report")
        ),
        Err(err) => {
            eprintln!("native_synthetic_io: {err}");
            std::process::exit(2);
        }
    }
}

fn parse_args() -> Result<NativeSyntheticConfig, String> {
    let mut config = NativeSyntheticConfig::default();
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
            "--scheduled" => config.scheduled = parse_bool(&value, &key)?,
            "--batches" => config.batches = parse_usize(&value, &key)?,
            "--warmup-batches" => config.warmup_batches = parse_usize(&value, &key)?,
            "--batch-size" => config.batch_size = parse_usize(&value, &key)?,
            "--workers" => config.workers = parse_usize(&value, &key)?,
            "--fill-workers" => config.fill_workers = parse_usize(&value, &key)?,
            "--native-workers" => config.native_workers = parse_usize(&value, &key)?,
            "--io-workers" => config.io_workers = parse_usize(&value, &key)?,
            "--chunks" => config.chunks = parse_usize(&value, &key)?,
            "--genes" => config.genes = parse_usize(&value, &key)?,
            "--source-genes" => config.source_genes = parse_usize(&value, &key)?,
            "--cells-per-chunk" => config.cells_per_chunk = parse_usize(&value, &key)?,
            "--cell-bytes" => config.cell_bytes = parse_usize(&value, &key)?,
            "--block-size" => config.block_size = parse_usize(&value, &key)?,
            "--typesize" => config.typesize = parse_usize(&value, &key)?,
            "--shuffle" => config.shuffle = parse_bool(&value, &key)?,
            "--entropy-fraction" => config.entropy_fraction = parse_f32(&value, &key)?,
            "--order" => config.order = NativeSyntheticOrder::parse(&value)?,
            "--continuation-p" => config.continuation_p = parse_f64(&value, &key)?,
            "--seed" => config.seed = parse_u64(&value, &key)?,
            "--projected-sparse-data-strategy" | "--sparse-data-strategy" => {
                config.projected_sparse_data_strategy =
                    ProjectedSparseDataGroupStrategy::parse(&value)?
            }
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
            "--scheduled-prefetch-step" => {
                config.scheduled_prefetch_step = parse_usize(&value, &key)?
            }
            "--scheduled-access-prefetch-step" => {
                config.scheduled_access_prefetch_step = parse_usize(&value, &key)?
            }
            "--scheduled-decode-ahead-steps" => {
                config.scheduled_decode_ahead_steps = parse_usize(&value, &key)?
            }
            "--scheduled-ready-ahead-steps" => {
                config.scheduled_ready_ahead_steps = parse_usize(&value, &key)?
            }
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
        r#"native_synthetic_io

Profile-only Blosc-LZ4 native synthetic IO benchmark.

Common options:
  --order random|sequential|continuity
  --scheduled true|false
  --continuation-p P
  --batches N
  --warmup-batches N
  --batch-size N
  --workers N
  --fill-workers N
  --native-workers N
  --io-workers N
  --chunks N
  --genes N
  --source-genes N
  --cells-per-chunk N
  --cell-bytes N
  --block-size N
  --entropy-fraction F
  --shuffle true|false
  --projected-sparse-data-strategy selected_only|read_all
"#
    );
}
