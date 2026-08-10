//! Item-range decoding coverage, including c-blosc-compatible semantics.
//!
//! c-blosc's `getitem` extracts the whole element range from a compressed
//! buffer. These tests cover whole and randomized partial ranges, including
//! ranges that straddle block boundaries.

use dyn_blosc::{Decoder, Encoder, Shuffle};

/// Deterministic xorshift64* generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// f32-like structured data so compression actually kicks in and blocks
/// split at the default 32 KiB block size.
fn structured_f32(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    let mut x = 1.0f32;
    for _ in 0..n {
        out.extend_from_slice(&x.to_le_bytes());
        x = (x * 1.0001 + 0.001).sin();
    }
    out
}

fn encode_structured(element_size: usize, byte_count: usize, shuffle: Shuffle) -> Vec<u8> {
    Encoder::new()
        .compression_level(5)
        .shuffle(shuffle)
        .element_size(element_size)
        .encode(&structured_f32(byte_count / element_size))
        .unwrap()
}

/// Decoding the whole item range must equal a full decode.
#[test]
fn whole_item_range() {
    let encoded = encode_structured(4, 4 * 100_000, Shuffle::Bytes);
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    let whole = decoder.decode_items(&encoded, 0..100_000).unwrap();
    assert_eq!(whole, decoder.decode(&encoded).unwrap());
}

/// Randomized partial ranges must match the corresponding decompressed slice.
#[test]
fn random_item_ranges() {
    let element_size = 4usize;
    let element_count = 50_000usize;
    let source = structured_f32(element_count);
    let encoded = Encoder::new()
        .shuffle(Shuffle::Bytes)
        .element_size(element_size)
        .encode(&source)
        .unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    let full = decoder.decode(&encoded).unwrap();

    let mut rng = Rng::new(0xC0FFEE);
    for _ in 0..200 {
        let start = rng.below(element_count - 1);
        let item_count = 1 + rng.below(element_count - start);
        let got = decoder
            .decode_items(&encoded, start..start + item_count)
            .unwrap();
        let expected = &full[start * element_size..(start + item_count) * element_size];
        assert_eq!(got, expected, "start={start} item_count={item_count}");
    }
}

/// Ranges that straddle a block boundary (blocks are 32 KiB here).
#[test]
fn item_ranges_across_block_boundaries() {
    let element_size = 4usize;
    let block = 32 * 1024usize;
    let element_count = block / element_size * 3;
    let source = structured_f32(element_count);
    let encoded = Encoder::new()
        .shuffle(Shuffle::Bytes)
        .element_size(element_size)
        .block_size(block)
        .encode(&source)
        .unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    let full = decoder.decode(&encoded).unwrap();

    // Element ranges crossing every possible boundary alignment.
    let elements_per_block = block / element_size;
    for off in 0..16 {
        for span in [1usize, 7, 31, 32, 33, 100, elements_per_block] {
            let start = elements_per_block - off;
            let item_count = span.min(element_count - start);
            let got = decoder
                .decode_items(&encoded, start..start + item_count)
                .unwrap();
            let expected = &full[start * element_size..(start + item_count) * element_size];
            assert_eq!(got, expected, "off={off} span={span}");
        }
    }
}

/// Item ranges also work on raw level-zero chunks.
#[test]
fn item_range_on_raw_chunk() {
    let source = structured_f32(500);
    let encoded = Encoder::new()
        .compression_level(0)
        .shuffle(Shuffle::Bytes)
        .element_size(4)
        .encode(&source)
        .unwrap();
    let got = Decoder::from_encoded(&encoded)
        .unwrap()
        .decode_items(&encoded, 100..150)
        .unwrap();
    assert_eq!(got, &structured_f32(500)[400..600]);
}

/// Out-of-bounds requests are rejected.
#[test]
fn item_range_out_of_bounds() {
    let encoded = encode_structured(4, 4 * 1000, Shuffle::None);
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    assert!(decoder.decode_items(&encoded, 1000..1001).is_err());
    assert!(decoder.decode_items(&encoded, 0..1001).is_err());
    assert!(decoder.decode_items(&encoded, 999..1001).is_err());
}
