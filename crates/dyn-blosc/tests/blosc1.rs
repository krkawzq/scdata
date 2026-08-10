use dyn_blosc::{
    BlockLayout, BloscVersion, Codec, DecodeLimits, Decoder, Encoder, Error, Header, LimitKind,
    Shuffle, BLOSC1_FORMAT_VERSION, DYN_BLOSC_FORMAT_VERSION, HEADER_LEN,
};

fn sample(len: usize) -> Vec<u8> {
    (0..len)
        .map(|index| ((index * 17 + index / 11) % 251) as u8)
        .collect()
}

#[test]
fn decoder_dispatches_blosc_versions_from_the_header() {
    let source = sample(32 * 1024);
    let blosc1 = Encoder::new()
        .version(BloscVersion::Blosc1)
        .block_size(4096)
        .encode(&source)
        .unwrap();
    let dynamic = Encoder::new()
        .version(BloscVersion::DynBlosc)
        .block_size(4096)
        .encode(&source)
        .unwrap();

    assert_eq!(blosc1[0], BLOSC1_FORMAT_VERSION);
    assert_eq!(dynamic[0], DYN_BLOSC_FORMAT_VERSION);
    let blosc1_header: [u8; HEADER_LEN] = blosc1[..HEADER_LEN].try_into().unwrap();
    let dynamic_header: [u8; HEADER_LEN] = dynamic[..HEADER_LEN].try_into().unwrap();
    assert!(matches!(
        Header::from_bytes(&blosc1_header).unwrap(),
        Header::Blosc1(_)
    ));
    assert!(matches!(
        Header::from_bytes(&dynamic_header).unwrap(),
        Header::DynBlosc(_)
    ));
    assert!(matches!(Header::parse(&blosc1).unwrap(), Header::Blosc1(_)));
    assert!(matches!(
        Header::parse(&dynamic).unwrap(),
        Header::DynBlosc(_)
    ));
    assert_eq!(
        Decoder::from_encoded(&blosc1)
            .unwrap()
            .decode(&blosc1)
            .unwrap(),
        source
    );
    assert_eq!(
        Decoder::from_encoded(&dynamic)
            .unwrap()
            .decode(&dynamic)
            .unwrap(),
        source
    );
}

#[test]
fn blosc1_metadata_and_fixed_block_ranges_are_exposed_uniformly() {
    let source = sample(10_500);
    let encoded = Encoder::new()
        .version(BloscVersion::Blosc1)
        .codec(Codec::Zstd)
        .shuffle(Shuffle::Bytes)
        .element_size(4)
        .block_size(2048)
        .encode(&source)
        .unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    let metadata = decoder.metadata();

    assert_eq!(metadata.version, BloscVersion::Blosc1);
    assert_eq!(metadata.decoded_size, source.len());
    assert_eq!(metadata.block_count, source.len().div_ceil(2048));
    assert_eq!(
        metadata.block_layout,
        BlockLayout::Fixed { block_size: 2048 }
    );
    assert_eq!(decoder.block(0).unwrap().decoded_range(), 0..2048);
    assert_eq!(
        decoder
            .block(metadata.block_count - 1)
            .unwrap()
            .decoded_range(),
        10_240..10_500
    );
    assert_eq!(
        Decoder::index_prefix_len(&encoded).unwrap(),
        HEADER_LEN + metadata.block_count * 4
    );
    assert_eq!(
        decoder.decode_bytes(&encoded, 1900..2300).unwrap(),
        source[1900..2300]
    );
}

#[test]
fn blosc1_roundtrips_codecs_filters_splits_and_leftovers() {
    let source = sample(32 * 1024 + 2048);
    for codec in [Codec::BloscLz, Codec::Lz4, Codec::Zlib, Codec::Zstd] {
        for shuffle in [Shuffle::None, Shuffle::Bytes, Shuffle::Bits] {
            for split in [false, true] {
                let encoded = Encoder::new()
                    .version(BloscVersion::Blosc1)
                    .codec(codec)
                    .shuffle(shuffle)
                    .element_size(4)
                    .split_blocks(split)
                    .block_size(8192)
                    .encode(&source)
                    .unwrap();
                let decoder = Decoder::from_encoded(&encoded).unwrap();
                assert_eq!(
                    decoder.decode(&encoded).unwrap(),
                    source,
                    "{codec:?} {shuffle:?} split={split}"
                );
            }
        }
    }
}

#[test]
fn blosc1_raw_and_empty_chunks_roundtrip() {
    let source = sample(8192);
    let raw = Encoder::new()
        .version(BloscVersion::Blosc1)
        .compression_level(0)
        .encode(&source)
        .unwrap();
    let decoder = Decoder::from_encoded(&raw).unwrap();
    assert!(decoder.metadata().is_raw);
    assert_eq!(decoder.block(0).unwrap().encoded_range().start, HEADER_LEN);
    assert_eq!(decoder.decode(&raw).unwrap(), source);

    // c-blosc retains the requested filter bits on memcpy chunks. Decoders
    // must ignore them once the raw flag is set.
    let mut c_style_raw = raw.clone();
    c_style_raw[2] |= 0x04;
    let decoder = Decoder::from_encoded(&c_style_raw).unwrap();
    assert_eq!(decoder.decode(&c_style_raw).unwrap(), source);

    // Raw Blosc1 chunks contain one semantic payload block even when the wire
    // header retains a smaller compressor block size.
    let mut retained_block_size = c_style_raw;
    retained_block_size[8..12].copy_from_slice(&256u32.to_le_bytes());
    let decoder = Decoder::from_encoded(&retained_block_size).unwrap();
    let metadata = decoder.metadata();
    assert_eq!(metadata.block_count, 1);
    assert_eq!(metadata.maximum_block_size, source.len());
    assert_eq!(
        metadata.block_layout,
        BlockLayout::Fixed {
            block_size: source.len()
        }
    );
    assert_eq!(decoder.blocks().len(), 1);
    assert!(decoder.block(1).is_none());
    assert_eq!(decoder.header().index_prefix_len().unwrap(), HEADER_LEN);
    assert_eq!(decoder.decode(&retained_block_size).unwrap(), source);
    assert!(matches!(
        Decoder::from_encoded_with_limits(
            &retained_block_size,
            DecodeLimits::unlimited().maximum_block_size(source.len() - 1)
        ),
        Err(Error::LimitExceeded {
            kind: LimitKind::BlockSize,
            actual,
            ..
        }) if actual == source.len()
    ));

    let empty = Encoder::new()
        .version(BloscVersion::Blosc1)
        .encode(&[])
        .unwrap();
    assert_eq!(empty.len(), HEADER_LEN);
    assert_eq!(
        Decoder::from_encoded(&empty)
            .unwrap()
            .decode(&empty)
            .unwrap(),
        []
    );
}

#[test]
fn blosc1_rejects_variable_partitions_and_corrupt_offsets() {
    assert!(Encoder::new()
        .version(BloscVersion::Blosc1)
        .block_lengths([512, 512])
        .encode(&sample(1024))
        .is_err());

    let mut encoded = Encoder::new()
        .version(BloscVersion::Blosc1)
        .block_size(1024)
        .encode(&sample(8192))
        .unwrap();
    encoded[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&0_i32.to_le_bytes());
    assert!(Decoder::from_encoded(&encoded).is_err());
}
