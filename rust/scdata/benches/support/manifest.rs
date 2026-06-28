//! Codec manifest bench runner.
//!
//! Drives decode benchmarks over a set of (codec, encoded payload, chunk
//! records) either loaded from a Python-exported manifest file or synthesized
//! in-memory by [`crate::data`]. Reports best / median / mean / std decode
//! times plus MiB/s and GiB/s, and writes summary CSV + JSON.
//!
//! This is the shared engine behind the `codec_manifest` bench target; it
//! replaces the old `src/bin/codec_manifest_bench.rs` standalone binary.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Value};

use _scdata::codecs::{CodecSpec, DecodeBuffer, SharedCodec};

use super::codecs::encode_for_spec;

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyMode {
    None,
    First,
    All,
}

impl VerifyMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "none" => Ok(Self::None),
            "first" => Ok(Self::First),
            "all" => Ok(Self::All),
            other => Err(format!("invalid --verify `{other}` (none|first|all)")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeOrder {
    Sequential,
    Random,
}

impl DecodeOrder {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "sequential" => Ok(Self::Sequential),
            "random" => Ok(Self::Random),
            other => Err(format!("invalid --decode-order `{other}` (sequential|random)")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ManifestBenchArgs {
    pub repeats: usize,
    pub warmups: usize,
    pub verify: VerifyMode,
    pub decode_order: DecodeOrder,
    pub seed: u64,
}

impl Default for ManifestBenchArgs {
    fn default() -> Self {
        Self {
            repeats: 3,
            warmups: 1,
            verify: VerifyMode::All,
            decode_order: DecodeOrder::Sequential,
            seed: 17,
        }
    }
}

/// One codec under test: its display name, family, parsed spec, and the
/// encoded payload plus per-chunk records locating each chunk inside it.
#[derive(Debug, Clone)]
pub struct CodecCase {
    pub algorithm: String,
    pub family: String,
    pub spec: CodecSpec,
    pub encoded: Vec<u8>,
    pub records: Vec<Record>,
    pub notes: String,
}

/// A chunk's location in the raw payload and its encoded location in the
/// codec's `encoded` buffer.
#[derive(Debug, Clone, Copy)]
pub struct Chunk {
    pub raw_offset: usize,
    pub raw_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Record {
    pub payload_offset: usize,
    pub payload_len: usize,
}

/// A loaded run: the raw payload, its chunk map, and the codec cases.
#[derive(Debug)]
pub struct LoadedRun {
    pub run_id: String,
    pub sample_bytes: usize,
    pub block_bytes: usize,
    pub profile: String,
    pub raw: Vec<u8>,
    pub chunks: Vec<Chunk>,
    pub cases: Vec<CodecCase>,
}

/// Bench one run and write per-run CSV + JSON outputs into `output_dir`.
pub fn bench_run(run: &LoadedRun, args: &ManifestBenchArgs, output_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| format!("create {}: {err}", output_dir.display()))?;
    let raw_bytes: usize = run.chunks.iter().map(|chunk| chunk.raw_len).sum();
    println!(
        "rust decode {}: {:.2} MiB in {} chunks, {} codecs",
        run.run_id,
        raw_bytes as f64 / (1 << 20) as f64,
        run.chunks.len(),
        run.cases.len(),
    );

    let mut rows = Vec::with_capacity(run.cases.len());
    for case in &run.cases {
        let row = bench_case(case, &run.raw, &run.chunks, args)?;
        println!(
            "  {} status={} decode_best_mib_s={}",
            row.algorithm,
            row.status,
            fmt_opt(row.decode_best_mib_s),
        );
        rows.push(row);
    }
    write_summary_csv(&output_dir.join("summary.csv"), &rows, None)?;
    let result = json!({
        "run_id": run.run_id,
        "sample_bytes": run.sample_bytes,
        "block_bytes": run.block_bytes,
        "profile": run.profile,
        "implementation": "rust",
        "results": rows.iter().map(row_json).collect::<Vec<_>>(),
    });
    fs::write(
        output_dir.join("result.json"),
        serde_json::to_string_pretty(&result).map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("write result.json: {err}"))?;
    Ok(())
}

/// Bench a matrix of runs and write `matrix_summary.csv` / `matrix_runs.csv` /
/// `matrix.json` into `output_dir`.
pub fn bench_matrix(
    runs: &[LoadedRun],
    args: &ManifestBenchArgs,
    output_dir: &Path,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| format!("create {}: {err}", output_dir.display()))?;
    let mut all_rows: Vec<(BTreeMap<String, String>, BenchRow)> = Vec::new();
    let mut run_rows: Vec<Vec<String>> = Vec::new();

    for run in runs {
        let run_dir = if runs.len() == 1 {
            output_dir.to_path_buf()
        } else {
            output_dir.join("runs").join(&run.run_id)
        };
        bench_run(run, args, &run_dir)?;
        let mut rows = Vec::new();
        for case in &run.cases {
            rows.push(bench_case(case, &run.raw, &run.chunks, args)?);
        }
        let best = rows
            .iter()
            .filter(|row| row.status == "ok")
            .max_by(|left, right| {
                left.decode_best_mib_s
                    .unwrap_or(0.0)
                    .total_cmp(&right.decode_best_mib_s.unwrap_or(0.0))
            });
        for row in &rows {
            let mut prefix = BTreeMap::new();
            prefix.insert("run_id".to_string(), run.run_id.clone());
            prefix.insert("sample_bytes".to_string(), run.sample_bytes.to_string());
            prefix.insert("block_bytes".to_string(), run.block_bytes.to_string());
            prefix.insert("profile".to_string(), run.profile.clone());
            prefix.insert(
                "output_dir".to_string(),
                run_dir.to_string_lossy().into_owned(),
            );
            all_rows.push((prefix, row.clone()));
        }
        run_rows.push(vec![
            run.run_id.clone(),
            run.sample_bytes.to_string(),
            run.block_bytes.to_string(),
            run.profile.clone(),
            run_dir.to_string_lossy().into_owned(),
            best.map(|row| row.algorithm.clone()).unwrap_or_default(),
            best.and_then(|row| row.decode_best_mib_s)
                .map(|value| value.to_string())
                .unwrap_or_default(),
        ]);
    }

    write_summary_csv(
        &output_dir.join("matrix_summary.csv"),
        &[],
        Some(&all_rows),
    )?;
    write_matrix_runs_csv(&output_dir.join("matrix_runs.csv"), &run_rows)?;
    let matrix_json = json!({
        "implementation": "rust",
        "runs": runs.iter().map(|run| {
            json!({
                "run_id": run.run_id,
                "sample_bytes": run.sample_bytes,
                "block_bytes": run.block_bytes,
                "profile": run.profile,
            })
        }).collect::<Vec<_>>(),
        "summary_rows": all_rows.iter().map(|(prefix, row)| {
            let mut value = row_json(row);
            if let Some(object) = value.as_object_mut() {
                for (key, item) in prefix {
                    object.insert(key.clone(), Value::String(item.clone()));
                }
            }
            value
        }).collect::<Vec<_>>(),
    });
    fs::write(
        output_dir.join("matrix.json"),
        serde_json::to_string_pretty(&matrix_json).map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("write matrix.json: {err}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Loading: from a manifest file, or synthesized in-memory.
// ---------------------------------------------------------------------------

/// Load a single-run manifest from disk (Python-exported format).
pub fn load_manifest_file(path: &Path) -> Result<LoadedRun, String> {
    let manifest = read_json(path)?;
    load_manifest_value(&manifest, path.parent().unwrap_or(Path::new(".")))
}

/// Load a matrix manifest and every run it references.
pub fn load_matrix_file(path: &Path) -> Result<Vec<LoadedRun>, String> {
    let matrix = read_json(path)?;
    let runs = array_field(&matrix, "runs")?;
    let mut loaded = Vec::with_capacity(runs.len());
    for run in runs {
        let run_id = string_field(run, "run_id")?.to_string();
        let manifest_path = PathBuf::from(string_field(run, "manifest")?);
        loaded.push(load_manifest_file(&manifest_path)?);
        // run_id from the matrix entry takes precedence.
        if let Some(last) = loaded.last_mut() {
            last.run_id = run_id;
        }
    }
    Ok(loaded)
}

fn load_manifest_value(manifest: &Value, base_dir: &Path) -> Result<LoadedRun, String> {
    let run_id = string_field(manifest, "run_id")?.to_string();
    let sample_bytes = usize_field(manifest, "sample_bytes")?;
    let block_bytes = usize_field(manifest, "block_bytes")?;
    let profile = string_field(manifest, "profile")?.to_string();
    let raw_file = PathBuf::from(string_field(manifest, "raw_file")?);
    let raw = fs::read(&raw_file).map_err(|err| format!("read {}: {err}", raw_file.display()))?;
    let chunks = parse_chunks(array_field(manifest, "chunks")?)?;

    let mut cases = Vec::new();
    for algorithm in array_field(manifest, "algorithms")? {
        let name = string_field(algorithm, "algorithm")?.to_string();
        let family = string_field(algorithm, "family")?.to_string();
        let notes = algorithm
            .get("notes")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let status = algorithm.get("status").and_then(Value::as_str);
        let config = algorithm.get("codec_config").filter(|value| !value.is_null());

        if status != Some("ok") || config.is_none() {
            continue; // skip non-ok / configless entries
        }
        let spec = CodecSpec::from_json_value(config.unwrap()).map_err(|err| err.to_string())?;
        let encoded_file = PathBuf::from(string_field(algorithm, "encoded_file")?);
        let encoded =
            fs::read(&encoded_file).map_err(|err| format!("read {}: {err}", encoded_file.display()))?;
        let records = parse_records(array_field(algorithm, "records")?)?;
        if records.len() != chunks.len() {
            return Err(format!(
                "{name}: records/chunks length mismatch: {} != {}",
                records.len(),
                chunks.len(),
            ));
        }
        let _ = base_dir; // paths in manifests are absolute; no join needed
        cases.push(CodecCase {
            algorithm: name,
            family,
            spec,
            encoded,
            records,
            notes,
        });
    }

    Ok(LoadedRun {
        run_id,
        sample_bytes,
        block_bytes,
        profile,
        raw,
        chunks,
        cases,
    })
}

/// Synthesize a run in-memory: generate `raw` from `profile`, encode each
/// codec spec, and build chunk records. Used by the `codec_manifest` bench's
/// `--synth` mode so it can run without a Python-exported manifest.
pub fn synthesize_run(
    run_id: &str,
    profile: &super::data::DataProfile,
    codecs: &[(&str, CodecSpec)],
) -> Result<LoadedRun, String> {
    let raw = profile.generate();
    let chunk_bytes = profile.chunk_bytes;
    let num_chunks = profile.num_chunks;
    let chunks = (0..num_chunks)
        .map(|idx| Chunk {
            raw_offset: idx * chunk_bytes,
            raw_len: chunk_bytes,
        })
        .collect::<Vec<_>>();

    let mut cases = Vec::with_capacity(codecs.len());
    for (name, spec) in codecs {
        let mut encoded = Vec::new();
        let mut records = Vec::with_capacity(num_chunks);
        for chunk in &chunks {
            let raw_slice = &raw[chunk.raw_offset..chunk.raw_offset + chunk.raw_len];
            let payload = encode_for_spec(spec, raw_slice);
            records.push(Record {
                payload_offset: encoded.len(),
                payload_len: payload.len(),
            });
            encoded.extend_from_slice(&payload);
        }
        let family = spec.name().to_string();
        cases.push(CodecCase {
            algorithm: name.to_string(),
            family,
            spec: spec.clone(),
            encoded,
            records,
            notes: String::new(),
        });
    }

    Ok(LoadedRun {
        run_id: run_id.to_string(),
        sample_bytes: profile.total_bytes(),
        block_bytes: chunk_bytes,
        profile: profile.label(),
        raw,
        chunks,
        cases,
    })
}

// ---------------------------------------------------------------------------
// Core bench loop.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BenchRow {
    implementation: String,
    status: String,
    algorithm: String,
    family: String,
    raw_bytes: usize,
    compressed_bytes: Option<usize>,
    compressed_over_raw: Option<f64>,
    raw_over_compressed: Option<f64>,
    decode_best_seconds: Option<f64>,
    decode_median_seconds: Option<f64>,
    decode_mean_seconds: Option<f64>,
    decode_std_seconds: Option<f64>,
    decode_best_mib_s: Option<f64>,
    decode_median_mib_s: Option<f64>,
    decode_best_gib_s: Option<f64>,
    decode_runs_seconds: Vec<f64>,
    notes: String,
}

fn bench_case(
    case: &CodecCase,
    raw: &[u8],
    chunks: &[Chunk],
    args: &ManifestBenchArgs,
) -> Result<BenchRow, String> {
    let codec = case.spec.build();
    let raw_bytes: usize = chunks.iter().map(|chunk| chunk.raw_len).sum();
    let compressed_bytes: usize = case.encoded.len();

    let mut buffers = chunks
        .iter()
        .map(|chunk| Vec::with_capacity(chunk.raw_len))
        .collect::<Vec<_>>();

    verify_case(
        &codec,
        &case.encoded,
        raw,
        chunks,
        &case.records,
        &mut buffers,
        args.verify,
    )?;

    for idx in 0..args.warmups {
        let (_elapsed, total_out) = decode_pass(
            &codec,
            &case.encoded,
            chunks,
            &case.records,
            &mut buffers,
            args.decode_order,
            args.seed + idx as u64,
        )?;
        if total_out != raw_bytes {
            return Err(format!(
                "{}: decoded {total_out} bytes, expected {raw_bytes}",
                case.algorithm,
            ));
        }
    }

    let mut times = Vec::with_capacity(args.repeats);
    for idx in 0..args.repeats {
        let (elapsed, total_out) = decode_pass(
            &codec,
            &case.encoded,
            chunks,
            &case.records,
            &mut buffers,
            args.decode_order,
            args.seed + args.warmups as u64 + idx as u64,
        )?;
        if total_out != raw_bytes {
            return Err(format!(
                "{}: decoded {total_out} bytes, expected {raw_bytes}",
                case.algorithm,
            ));
        }
        times.push(elapsed);
    }

    Ok(ok_row(
        &case.algorithm,
        &case.family,
        raw_bytes,
        compressed_bytes,
        &case.notes,
        times,
    ))
}

fn verify_case(
    codec: &SharedCodec,
    encoded: &[u8],
    raw: &[u8],
    chunks: &[Chunk],
    records: &[Record],
    buffers: &mut [Vec<u8>],
    verify: VerifyMode,
) -> Result<(), String> {
    if verify == VerifyMode::None {
        return Ok(());
    }
    let limit = if verify == VerifyMode::First { 1 } else { chunks.len() };
    for idx in 0..limit {
        let written = decode_one(codec, &chunks[idx], &records[idx], &mut buffers[idx], encoded)?;
        let raw_end = chunks[idx]
            .raw_offset
            .checked_add(chunks[idx].raw_len)
            .ok_or_else(|| "raw slice overflow".to_string())?;
        if buffers[idx][..written] != raw[chunks[idx].raw_offset..raw_end] {
            return Err(format!("round-trip mismatch at chunk {idx}"));
        }
    }
    Ok(())
}

fn decode_pass(
    codec: &SharedCodec,
    encoded: &[u8],
    chunks: &[Chunk],
    records: &[Record],
    buffers: &mut [Vec<u8>],
    order: DecodeOrder,
    seed: u64,
) -> Result<(f64, usize), String> {
    let mut total_out = 0usize;
    let started = Instant::now();
    let indices: Vec<usize> = match order {
        DecodeOrder::Sequential => (0..chunks.len()).collect(),
        DecodeOrder::Random => shuffle_indices(chunks.len(), seed),
    };
    for idx in indices {
        total_out += decode_one(codec, &chunks[idx], &records[idx], &mut buffers[idx], encoded)?;
    }
    Ok((started.elapsed().as_secs_f64(), total_out))
}

fn decode_one(
    codec: &SharedCodec,
    chunk: &Chunk,
    record: &Record,
    output: &mut Vec<u8>,
    encoded: &[u8],
) -> Result<usize, String> {
    let payload_end = record
        .payload_offset
        .checked_add(record.payload_len)
        .ok_or_else(|| "payload slice overflow".to_string())?;
    let payload = encoded
        .get(record.payload_offset..payload_end)
        .ok_or_else(|| "payload slice out of bounds".to_string())?;
    if output.capacity() < chunk.raw_len {
        output.reserve_exact(chunk.raw_len - output.capacity());
    }
    set_len_for_decode(output, chunk.raw_len);
    match codec.decode_into(
        payload,
        DecodeBuffer::new(output.as_mut_slice()),
        Some(chunk.raw_len),
    ) {
        Ok(written) => {
            output.truncate(written);
            Ok(written)
        }
        Err(err) => {
            output.clear();
            Err(err.to_string())
        }
    }
}

fn set_len_for_decode(output: &mut Vec<u8>, len: usize) {
    debug_assert!(len <= output.capacity());
    // SAFETY: `Vec<u8>` has no drop glue. The slice is immediately passed to a
    // decoder that initializes the returned byte count, and then truncated.
    unsafe {
        output.set_len(len);
    }
}

fn shuffle_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices = (0..len).collect::<Vec<_>>();
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    for i in (1..indices.len()).rev() {
        state = splitmix64(state);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

// ---------------------------------------------------------------------------
// Statistics + output formatting.
// ---------------------------------------------------------------------------

fn ok_row(
    algorithm: &str,
    family: &str,
    raw_bytes: usize,
    compressed_bytes: usize,
    notes: &str,
    times: Vec<f64>,
) -> BenchRow {
    let best = times.iter().copied().reduce(f64::min);
    let median = median(&times);
    let mean = mean(&times);
    let std = stddev(&times, mean);
    BenchRow {
        implementation: "rust".to_string(),
        status: "ok".to_string(),
        algorithm: algorithm.to_string(),
        family: family.to_string(),
        raw_bytes,
        compressed_bytes: Some(compressed_bytes),
        compressed_over_raw: Some(compressed_bytes as f64 / raw_bytes as f64),
        raw_over_compressed: Some(raw_bytes as f64 / compressed_bytes as f64),
        decode_best_seconds: best,
        decode_median_seconds: median,
        decode_mean_seconds: mean,
        decode_std_seconds: std,
        decode_best_mib_s: best.map(|seconds| raw_bytes as f64 / seconds / (1 << 20) as f64),
        decode_median_mib_s: median.map(|seconds| raw_bytes as f64 / seconds / (1 << 20) as f64),
        decode_best_gib_s: best.map(|seconds| raw_bytes as f64 / seconds / (1 << 30) as f64),
        decode_runs_seconds: times,
        notes: notes.to_string(),
    }
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    } else {
        Some(sorted[mid])
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn stddev(values: &[f64], mean: Option<f64>) -> Option<f64> {
    if values.len() <= 1 {
        return mean.map(|_| 0.0);
    }
    let mean = mean?;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Some(variance.sqrt())
}

fn write_summary_csv(
    path: &Path,
    rows: &[BenchRow],
    matrix_rows: Option<&[(BTreeMap<String, String>, BenchRow)]>,
) -> Result<(), String> {
    let mut text = String::new();
    let prefix_headers = if matrix_rows.is_some() {
        vec!["run_id", "sample_bytes", "block_bytes", "profile", "output_dir"]
    } else {
        Vec::new()
    };
    let headers = row_headers();
    text.push_str(&prefix_headers.join(","));
    if !prefix_headers.is_empty() {
        text.push(',');
    }
    text.push_str(&headers.join(","));
    text.push('\n');

    if let Some(matrix_rows) = matrix_rows {
        for (prefix, row) in matrix_rows {
            let mut fields = Vec::new();
            for header in &prefix_headers {
                fields.push(prefix.get(*header).cloned().unwrap_or_default());
            }
            fields.extend(row_csv_values(row));
            push_csv_record(&mut text, &fields);
        }
    } else {
        for row in rows {
            push_csv_record(&mut text, &row_csv_values(row));
        }
    }
    fs::write(path, text).map_err(|err| format!("write {}: {err}", path.display()))
}

fn write_matrix_runs_csv(path: &Path, rows: &[Vec<String>]) -> Result<(), String> {
    let mut text = String::from(
        "run_id,sample_bytes,block_bytes,profile,output_dir,best_decode_algorithm,best_decode_mib_s\n",
    );
    for row in rows {
        push_csv_record(&mut text, row);
    }
    fs::write(path, text).map_err(|err| format!("write {}: {err}", path.display()))
}

fn row_headers() -> Vec<&'static str> {
    vec![
        "implementation",
        "status",
        "algorithm",
        "family",
        "raw_bytes",
        "compressed_bytes",
        "compressed_over_raw",
        "raw_over_compressed",
        "decode_best_seconds",
        "decode_median_seconds",
        "decode_mean_seconds",
        "decode_std_seconds",
        "decode_best_mib_s",
        "decode_median_mib_s",
        "decode_best_gib_s",
        "decode_runs_seconds",
        "notes",
    ]
}

fn row_csv_values(row: &BenchRow) -> Vec<String> {
    vec![
        row.implementation.clone(),
        row.status.clone(),
        row.algorithm.clone(),
        row.family.clone(),
        row.raw_bytes.to_string(),
        row.compressed_bytes
            .map(|value| value.to_string())
            .unwrap_or_default(),
        fmt_opt(row.compressed_over_raw),
        fmt_opt(row.raw_over_compressed),
        fmt_opt(row.decode_best_seconds),
        fmt_opt(row.decode_median_seconds),
        fmt_opt(row.decode_mean_seconds),
        fmt_opt(row.decode_std_seconds),
        fmt_opt(row.decode_best_mib_s),
        fmt_opt(row.decode_median_mib_s),
        fmt_opt(row.decode_best_gib_s),
        serde_json::to_string(&row.decode_runs_seconds).unwrap_or_else(|_| "[]".to_string()),
        row.notes.clone(),
    ]
}

fn row_json(row: &BenchRow) -> Value {
    json!({
        "implementation": row.implementation,
        "status": row.status,
        "algorithm": row.algorithm,
        "family": row.family,
        "raw_bytes": row.raw_bytes,
        "compressed_bytes": row.compressed_bytes,
        "compressed_over_raw": row.compressed_over_raw,
        "raw_over_compressed": row.raw_over_compressed,
        "decode_best_seconds": row.decode_best_seconds,
        "decode_median_seconds": row.decode_median_seconds,
        "decode_mean_seconds": row.decode_mean_seconds,
        "decode_std_seconds": row.decode_std_seconds,
        "decode_best_mib_s": row.decode_best_mib_s,
        "decode_median_mib_s": row.decode_median_mib_s,
        "decode_best_gib_s": row.decode_best_gib_s,
        "decode_runs_seconds": row.decode_runs_seconds,
        "notes": row.notes,
    })
}

fn push_csv_record(text: &mut String, fields: &[String]) {
    for (idx, field) in fields.iter().enumerate() {
        if idx > 0 {
            text.push(',');
        }
        push_csv_field(text, field);
    }
    text.push('\n');
}

fn push_csv_field(text: &mut String, field: &str) {
    if field.contains([',', '"', '\n', '\r']) {
        text.push('"');
        for ch in field.chars() {
            if ch == '"' {
                text.push('"');
            }
            text.push(ch);
        }
        text.push('"');
    } else {
        text.push_str(field);
    }
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// JSON field helpers.
// ---------------------------------------------------------------------------

fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    serde_json::from_str(&text).map_err(|err| format!("parse {}: {err}", path.display()))
}

fn array_field<'a>(value: &'a Value, key: &str) -> Result<&'a [Value], String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("missing array field `{key}`"))
}

fn string_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string field `{key}`"))
}

fn usize_field(value: &Value, key: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("missing usize field `{key}`"))
}

fn parse_chunks(items: &[Value]) -> Result<Vec<Chunk>, String> {
    items
        .iter()
        .map(|item| {
            Ok(Chunk {
                raw_offset: usize_field(item, "raw_offset")?,
                raw_len: usize_field(item, "raw_len")?,
            })
        })
        .collect()
}

fn parse_records(items: &[Value]) -> Result<Vec<Record>, String> {
    items
        .iter()
        .map(|item| {
            Ok(Record {
                payload_offset: usize_field(item, "payload_offset")?,
                payload_len: usize_field(item, "payload_len")?,
            })
        })
        .collect()
}
