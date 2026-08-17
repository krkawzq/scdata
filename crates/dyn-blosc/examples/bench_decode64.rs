//! Microbench: dyn-blosc LZ4 clevel=5 decode of ~64 KiB blocks.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use dyn_blosc::{BlockDecoder, Codec, DecodeWorkspace, Decoder, Encoder, Shuffle};

const N: u32 = 20_000;
const WARM: u32 = 200;
const TARGET: usize = 64 * 1024;

fn ns_loop(iters: u32, mut f: impl FnMut()) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_nanos() as f64 / f64::from(iters)
}

fn bench_payload(label: &str, src: &[u8], shuffle: Shuffle) {
    let encoder = Encoder::new()
        .codec(Codec::Lz4)
        .compression_level(5)
        .shuffle(shuffle)
        .element_size(4)
        .block_size(src.len())
        .threads(1);
    let chunk = encoder.encode(src).expect("encode");
    let decoder = Decoder::from_encoded(&chunk).expect("decoder");
    let block = decoder.block(0).expect("block 0");
    let bd = decoder.block_decoder(0).expect("block decoder");
    let payload = &chunk[block.encoded_range()];
    let mut out = vec![0u8; bd.decoded_len()];
    let mut ws = DecodeWorkspace::new();
    for _ in 0..WARM {
        bd.decode_into(payload, &mut out, &mut ws).unwrap();
    }
    let ns = ns_loop(N, || {
        bd.decode_into(payload, &mut out, &mut ws).unwrap();
    });
    let ratio = src.len() as f64 / payload.len() as f64;
    let mib_s = (src.len() as f64 / (1024.0 * 1024.0)) / (ns / 1e9);
    println!(
        "{label:28} decoded={:>6} encoded={:>6} ratio={:.2}x  {:>7.1} ns  {:>6.1} MiB/s  shuffle={shuffle:?}",
        src.len(),
        payload.len(),
        ratio,
        ns,
        mib_s
    );
}

fn bench_chunk_file(path: &PathBuf) {
    let chunk = fs::read(path).expect("read chunk");
    let decoder = Decoder::from_encoded(&chunk).expect("decoder");
    let meta = decoder.metadata();
    println!(
        "\nreal chunk {}  decoded={} encoded={} blocks={} codec={:?} shuffle={:?} elem={}",
        path.display(),
        meta.decoded_size,
        meta.encoded_size,
        meta.block_count,
        meta.codec,
        meta.shuffle,
        meta.element_size
    );
    let mut ws = DecodeWorkspace::new();
    let mut times = Vec::new();
    for i in 0..meta.block_count {
        let range = decoder.block(i).unwrap();
        let bd: BlockDecoder = decoder.block_decoder(i).unwrap();
        let payload = &chunk[range.encoded_range()];
        let mut out = vec![0u8; bd.decoded_len()];
        for _ in 0..WARM {
            bd.decode_into(payload, &mut out, &mut ws).unwrap();
        }
        let ns = ns_loop(N, || {
            bd.decode_into(payload, &mut out, &mut ws).unwrap();
        });
        times.push((bd.decoded_len(), payload.len(), ns));
    }
    times.sort_by(|a, b| a.0.cmp(&b.0));
    let mut sum_ns = 0.0;
    let mut sum_dec = 0usize;
    for (decoded, encoded, ns) in &times {
        sum_ns += ns;
        sum_dec += *decoded;
        if times.len() <= 12
            || *decoded == times[0].0
            || *decoded == times[times.len() / 2].0
            || *decoded == times[times.len() - 1].0
        {
            println!(
                "  block decoded={:>6} encoded={:>6}  {:>7.1} ns  {:>6.1} MiB/s",
                decoded,
                encoded,
                ns,
                (*decoded as f64 / (1024.0 * 1024.0)) / (ns / 1e9)
            );
        }
    }
    let mean = sum_ns / times.len() as f64;
    println!(
        "  {} blocks  mean={:.1} ns/block  total_decoded={}  implied {:.1} ns/64KiB",
        times.len(),
        mean,
        sum_dec,
        mean * (TARGET as f64 / (sum_dec as f64 / times.len() as f64))
    );
}

fn main() {
    let mut counts = vec![0u8; TARGET];
    for (i, chunk) in counts.chunks_exact_mut(4).enumerate() {
        let v = ((i % 17) + 1) as f32;
        chunk.copy_from_slice(&v.to_le_bytes());
    }
    let mut ones = vec![0u8; TARGET];
    for chunk in ones.chunks_exact_mut(4) {
        chunk.copy_from_slice(&1f32.to_le_bytes());
    }
    let mut random = vec![0u8; TARGET];
    let mut x: u32 = 0x12345678;
    for b in &mut random {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (x >> 24) as u8;
    }

    println!("synthetic 64 KiB, LZ4 clevel=5, {N} iters after {WARM} warmup");
    bench_payload("f32 counts 1..=17 +shuffle", &counts, Shuffle::Bytes);
    bench_payload("f32 counts 1..=17 noshuffle", &counts, Shuffle::None);
    bench_payload("f32 ones +shuffle", &ones, Shuffle::Bytes);
    bench_payload("f32 ones noshuffle", &ones, Shuffle::None);
    bench_payload("random bytes +shuffle", &random, Shuffle::Bytes);
    bench_payload("random bytes noshuffle", &random, Shuffle::None);

    for arg in std::env::args().skip(1) {
        bench_chunk_file(&PathBuf::from(arg));
    }
}
