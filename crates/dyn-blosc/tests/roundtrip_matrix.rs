//! Port of c-blosc's `test_compress_roundtrip.c` + its CSV test matrix.
//!
//! The matrix covers 19 element type sizes x 7 element counts x {no shuffle,
//! byte shuffle}, compressing random data and verifying a perfect roundtrip.

use dyn_blosc::{Decoder, Encoder, Shuffle};

/// Deterministic xorshift64* generator (stands in for C's `rand()`).
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
    fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let v = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
    }
}

fn roundtrip_random(element_size: usize, element_count: usize, shuffle: Shuffle) {
    let byte_count = element_size * element_count;
    let mut source = vec![0u8; byte_count];
    Rng::new((element_size as u64) << 32 | element_count as u64).fill(&mut source);

    let encoded = Encoder::new()
        .compression_level(5)
        .shuffle(shuffle)
        .element_size(element_size)
        .encode(&source)
        .expect("encoding should succeed");
    let decoded = Decoder::from_encoded(&encoded)
        .and_then(|decoder| decoder.decode(&encoded))
        .expect("decoding should succeed");
    assert_eq!(
        decoded, source,
        "element_size={element_size} element_count={element_count} shuffle={shuffle:?}"
    );
}

/// Element sizes 1..8 across small, irregular, and large element counts.
#[test]
fn matrix_typesize_1_to_8() {
    for typesize in [1usize, 2, 3, 4, 5, 6, 7, 8] {
        for nelem in [7usize, 192, 1792, 500, 8000, 100_000, 702_713] {
            roundtrip_random(typesize, nelem, Shuffle::None);
            roundtrip_random(typesize, nelem, Shuffle::Bytes);
        }
    }
}

/// Element sizes 11..32.
#[test]
fn matrix_typesize_11_to_32() {
    for typesize in [11usize, 16, 22, 30, 32] {
        for nelem in [7usize, 192, 1792, 500, 8000, 100_000, 702_713] {
            roundtrip_random(typesize, nelem, Shuffle::None);
            roundtrip_random(typesize, nelem, Shuffle::Bytes);
        }
    }
}

/// Element sizes 42..80.
#[test]
fn matrix_typesize_42_to_80() {
    for typesize in [42usize, 48, 52, 53, 64, 80] {
        for nelem in [7usize, 192, 1792, 500, 8000, 100_000, 702_713] {
            roundtrip_random(typesize, nelem, Shuffle::None);
            roundtrip_random(typesize, nelem, Shuffle::Bytes);
        }
    }
}

/// Empty input remains a valid chunk.
#[test]
fn empty_input() {
    let source = Vec::new();
    let encoded = Encoder::new().element_size(1).encode(&source).unwrap();
    assert_eq!(
        Decoder::from_encoded(&encoded)
            .unwrap()
            .decode(&encoded)
            .unwrap(),
        source
    );
}

/// Same matrix, but with bitshuffle: c-blosc exercises it heavily with
/// `blosc_compress(9, BLOSC_BITSHUFFLE, ...)`; here we sweep sizes and
/// element counts in one pass.
#[test]
fn matrix_bitshuffle() {
    for typesize in [1usize, 2, 3, 4, 7, 8, 11, 16, 32, 53, 80] {
        for nelem in [7usize, 192, 1792, 500, 8000, 100_000] {
            roundtrip_random(typesize, nelem, Shuffle::Bits);
        }
    }
}
