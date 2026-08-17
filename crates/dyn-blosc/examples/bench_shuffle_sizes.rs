//! Shuffle vs no-shuffle at block sizes up to 1 MiB on real decoded payloads.

use std::fs;
use std::path::Path;
use std::time::Instant;

use dyn_blosc::{Codec, Decoder, Encoder, Shuffle};

const WARM: u32 = 15;
const EPOCHS: u32 = 40;
const SIZES: &[usize] = &[
    16 << 10,
    32 << 10,
    64 << 10,
    128 << 10,
    256 << 10,
    512 << 10,
    1024 << 10,
];

fn decode_all(path: &Path) -> (Vec<u8>, usize) {
    let chunk = fs::read(path).expect("read");
    let decoder = Decoder::from_encoded(&chunk).expect("decoder");
    let meta = decoder.metadata();
    let mut out = vec![0u8; meta.decoded_size];
    decoder
        .decode_into(&chunk, &mut out)
        .expect("decode chunk");
    (out, meta.element_size)
}

fn encode(src: &[u8], shuffle: Shuffle, element_size: usize, block: usize) -> Vec<u8> {
    Encoder::new()
        .codec(Codec::Lz4)
        .compression_level(5)
        .shuffle(shuffle)
        .element_size(element_size)
        .block_size(block.min(src.len()))
        .threads(1)
        .encode(src)
        .expect("encode")
}

fn time_decode(chunk: &[u8]) -> (f64, usize, usize) {
    let decoder = Decoder::from_encoded(chunk).expect("decoder");
    let meta = decoder.metadata();
    let mut out = vec![0u8; meta.decoded_size];
    for _ in 0..WARM {
        decoder.decode_into(chunk, &mut out).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..EPOCHS {
        decoder.decode_into(chunk, &mut out).unwrap();
    }
    let ns = t0.elapsed().as_nanos() as f64 / f64::from(EPOCHS);
    (ns, meta.block_count, meta.decoded_size)
}

fn bench_file(path: &Path) {
    let (raw, elem) = decode_all(path);
    println!(
        "\n{}  decoded={} elem={}  decode {} epochs",
        path.display(),
        raw.len(),
        elem,
        EPOCHS
    );
    println!(
        "{:>8}  {:>10} {:>8} {:>8} {:>10} {:>8} {:>8}  {:>8} {:>8}",
        "block", "enc+sh", "ratio", "us", "enc-nosh", "ratio", "us", "speedup", "size×"
    );
    for &block in SIZES {
        if block > raw.len() {
            continue;
        }
        let on = encode(&raw, Shuffle::Bytes, elem, block);
        let off = encode(&raw, Shuffle::None, elem, block);
        let (ns_on, n_on, dec) = time_decode(&on);
        let (ns_off, _, _) = time_decode(&off);
        let _ = n_on;
        println!(
            "{:>7}K  {:>10} {:>7.2}x {:>7.1}  {:>10} {:>7.2}x {:>7.1}  {:>7.2}x {:>7.2}x",
            block / 1024,
            on.len(),
            dec as f64 / on.len() as f64,
            ns_on / 1e3,
            off.len(),
            dec as f64 / off.len() as f64,
            ns_off / 1e3,
            ns_on / ns_off,
            off.len() as f64 / on.len() as f64
        );
    }
}

fn main() {
    for arg in std::env::args().skip(1) {
        bench_file(Path::new(&arg));
    }
}
