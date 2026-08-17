//! Re-encode real SCC blocks with and without byte shuffle, then time decode.

use std::fs;
use std::path::Path;
use std::time::Instant;

use dyn_blosc::{Codec, DecodeWorkspace, Decoder, Encoder, Shuffle};

const WARM_EPOCHS: u32 = 20;
const EPOCHS: u32 = 80;

struct Prepared {
    payload: Vec<u8>,
    decoder: Decoder,
    decoded_len: usize,
}

fn encode_block(src: &[u8], shuffle: Shuffle, element_size: usize) -> Prepared {
    let chunk = Encoder::new()
        .codec(Codec::Lz4)
        .compression_level(5)
        .shuffle(shuffle)
        .element_size(element_size)
        .block_size(src.len())
        .threads(1)
        .encode(src)
        .expect("encode");
    let decoder = Decoder::from_encoded(&chunk).expect("decoder");
    let range = decoder.block(0).expect("block");
    Prepared {
        payload: chunk[range.encoded_range()].to_vec(),
        decoder,
        decoded_len: src.len(),
    }
}

fn time_decode(blocks: &[Prepared]) -> f64 {
    let mut ws = DecodeWorkspace::new();
    let mut outs: Vec<Vec<u8>> = blocks
        .iter()
        .map(|b| vec![0u8; b.decoded_len])
        .collect();
    let run = |ws: &mut DecodeWorkspace, outs: &mut [Vec<u8>]| {
        for (block, out) in blocks.iter().zip(outs.iter_mut()) {
            block
                .decoder
                .block_decoder(0)
                .unwrap()
                .decode_into(&block.payload, out, ws)
                .unwrap();
        }
    };
    for _ in 0..WARM_EPOCHS {
        run(&mut ws, &mut outs);
    }
    let t0 = Instant::now();
    for _ in 0..EPOCHS {
        run(&mut ws, &mut outs);
    }
    t0.elapsed().as_nanos() as f64 / f64::from(EPOCHS)
}

fn bench_file(path: &Path) {
    let chunk = fs::read(path).expect("read");
    let decoder = Decoder::from_encoded(&chunk).expect("decoder");
    let meta = decoder.metadata();
    let mut raws = Vec::with_capacity(meta.block_count);
    let mut ws = DecodeWorkspace::new();
    for i in 0..meta.block_count {
        let range = decoder.block(i).unwrap();
        let bd = decoder.block_decoder(i).unwrap();
        let mut out = vec![0u8; bd.decoded_len()];
        bd.decode_into(&chunk[range.encoded_range()], &mut out, &mut ws)
            .unwrap();
        raws.push(out);
    }

    let on: Vec<Prepared> = raws
        .iter()
        .map(|src| encode_block(src, Shuffle::Bytes, meta.element_size))
        .collect();
    let off: Vec<Prepared> = raws
        .iter()
        .map(|src| encode_block(src, Shuffle::None, meta.element_size))
        .collect();

    let decoded: usize = raws.iter().map(Vec::len).sum();
    let enc_on: usize = on.iter().map(|b| b.payload.len()).sum();
    let enc_off: usize = off.iter().map(|b| b.payload.len()).sum();
    let ns_on = time_decode(&on);
    let ns_off = time_decode(&off);

    println!(
        "{}  blocks={} decoded={} elem={} orig_encoded={}",
        path.display(),
        meta.block_count,
        decoded,
        meta.element_size,
        meta.encoded_size
    );
    println!(
        "  shuffle=Bytes  encoded={:>10}  ratio={:.2}x  decode={:>8.1} us/pass  {:>6.1} ns/64KiB  {:>5.1} MiB/s",
        enc_on,
        decoded as f64 / enc_on as f64,
        ns_on / 1e3,
        ns_on * 65536.0 / decoded as f64,
        (decoded as f64 / (1024.0 * 1024.0)) / (ns_on / 1e9)
    );
    println!(
        "  shuffle=None   encoded={:>10}  ratio={:.2}x  decode={:>8.1} us/pass  {:>6.1} ns/64KiB  {:>5.1} MiB/s",
        enc_off,
        decoded as f64 / enc_off as f64,
        ns_off / 1e3,
        ns_off * 65536.0 / decoded as f64,
        (decoded as f64 / (1024.0 * 1024.0)) / (ns_off / 1e9)
    );
    println!(
        "  decode speedup without shuffle: {:.2}x    encoded size without/with: {:.2}x",
        ns_on / ns_off,
        enc_off as f64 / enc_on as f64
    );
}

fn main() {
    println!(
        "re-encode real blocks LZ4 clevel=5; decode {} epochs after {} warmup",
        EPOCHS, WARM_EPOCHS
    );
    for arg in std::env::args().skip(1) {
        bench_file(Path::new(&arg));
        println!();
    }
}
