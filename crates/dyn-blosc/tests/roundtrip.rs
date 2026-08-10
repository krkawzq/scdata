use dyn_blosc::{
    BlockDescriptor, ByteSelection, Codec, DecodeWorkspace, Decoder, EncodeWorkspace, Encoder,
    Shuffle, DYN_BLOSC_FORMAT_VERSION,
};

fn sample_f32(count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|index| (index as f32).to_le_bytes())
        .collect()
}

#[test]
fn fixed_blocks_roundtrip() {
    let source = sample_f32(4096);
    let encoded = Encoder::new()
        .codec(Codec::Lz4)
        .shuffle(Shuffle::Bytes)
        .element_size(4)
        .block_size(1024)
        .encode(&source)
        .unwrap();
    assert_eq!(encoded[0], DYN_BLOSC_FORMAT_VERSION);

    let decoder = Decoder::from_encoded(&encoded).unwrap();
    assert!(decoder.metadata().block_count > 1);
    assert_eq!(decoder.decode(&encoded).unwrap(), source);
}

#[test]
fn variable_blocks_roundtrip() {
    let source = sample_f32(1000);
    let encoded = Encoder::new()
        .codec(Codec::Zstd)
        .shuffle(Shuffle::Bytes)
        .element_size(4)
        .block_lengths([400, 1200, 800, 1600])
        .encode(&source)
        .unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    assert_eq!(decoder.metadata().block_count, 4);
    assert_eq!(decoder.metadata().maximum_block_size, 1600);
    assert_eq!(decoder.decode(&encoded).unwrap(), source);
}

#[test]
fn blocks_bytes_and_items_can_be_decoded_independently() {
    let source = sample_f32(2048);
    let encoded = Encoder::new()
        .codec(Codec::Lz4)
        .shuffle(Shuffle::None)
        .element_size(4)
        .block_lengths([2048, 2048, 4096])
        .encode(&source)
        .unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    let block0 = decoder.block(0).unwrap();
    let encoded_block = &encoded[block0.encoded_range()];

    assert_eq!(
        decoder.decode_block(0, encoded_block).unwrap(),
        &source[..2048]
    );
    assert_eq!(
        decoder.decode_bytes(&encoded, 100..500).unwrap(),
        &source[100..500]
    );
    assert_eq!(
        decoder.decode_items(&encoded, 10..15).unwrap(),
        &source[40..60]
    );

    let selection = ByteSelection::contiguous(3500..4500).unwrap();
    assert_eq!(
        decoder.decode_selection(&encoded, &selection).unwrap(),
        &source[3500..4500]
    );
}

#[test]
fn level_zero_stores_raw_chunk() {
    let source = sample_f32(64);
    let encoded = Encoder::new().compression_level(0).encode(&source).unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    let metadata = decoder.metadata();
    assert!(metadata.is_raw);
    assert_eq!(metadata.decoded_size, source.len());
    assert_eq!(metadata.encoded_size, encoded.len());
    assert_eq!(decoder.header().index_prefix_len().unwrap(), 16);
    assert_eq!(decoder.decode(&encoded).unwrap(), source);
}

#[test]
fn blosclz_roundtrip() {
    let source = sample_f32(512);
    let encoded = Encoder::new()
        .codec(Codec::BloscLz)
        .shuffle(Shuffle::None)
        .element_size(4)
        .block_size(512)
        .encode(&source)
        .unwrap();
    assert_eq!(
        Decoder::from_encoded(&encoded)
            .unwrap()
            .decode(&encoded)
            .unwrap(),
        source
    );
}

#[test]
fn parallel_encoding_is_deterministic_and_parallel_decoding_agrees() {
    let source = sample_f32(20_000);
    let sequential = Encoder::new()
        .block_size(512)
        .threads(1)
        .encode(&source)
        .unwrap();
    let parallel = Encoder::new()
        .block_size(512)
        .threads(4)
        .encode(&source)
        .unwrap();
    assert_eq!(parallel, sequential);
    assert_eq!(
        Decoder::from_encoded(&parallel)
            .unwrap()
            .threads(4)
            .decode(&parallel)
            .unwrap(),
        source
    );
}

#[test]
fn uneven_variable_blocks_roundtrip_in_parallel() {
    let decoded_lengths = [32_768, 256, 512, 24_576, 128, 1024, 16_384, 64];
    let source = sample_f32(decoded_lengths.iter().sum::<usize>() / 4);
    let sequential = Encoder::new()
        .block_lengths(decoded_lengths)
        .threads(1)
        .encode(&source)
        .unwrap();
    let parallel = Encoder::new()
        .block_lengths(decoded_lengths)
        .threads(4)
        .encode(&source)
        .unwrap();

    assert_eq!(parallel, sequential);
    assert_eq!(
        Decoder::from_encoded(&parallel)
            .unwrap()
            .threads(4)
            .decode(&parallel)
            .unwrap(),
        source
    );
}

#[test]
fn prefix_schema_and_block_decode_into_match_full_decode() {
    let source = sample_f32(2048);
    let encoded = Encoder::new()
        .element_size(4)
        .block_size(1024)
        .encode(&source)
        .unwrap();
    let prefix_len = Decoder::index_prefix_len(&encoded).unwrap();
    let decoder = Decoder::from_prefix(&encoded[..prefix_len]).unwrap();
    let mut workspace = DecodeWorkspace::new();
    let mut cursor = 0;
    for block_index in 0..decoder.metadata().block_count {
        let range = decoder.block(block_index).unwrap();
        let encoded_block = &encoded[range.encoded_range()];
        let mut out = vec![0; range.decoded_len()];
        assert_eq!(
            decoder
                .decode_block_into(block_index, encoded_block, &mut out, &mut workspace)
                .unwrap(),
            range.decoded_len()
        );
        assert_eq!(out, source[cursor..cursor + range.decoded_len()]);
        cursor += range.decoded_len();
    }
    assert_eq!(cursor, source.len());
}

#[test]
fn independently_encoded_blocks_can_build_a_layout_and_chunk() {
    let source = sample_f32(2048);
    let encoder = Encoder::new()
        .codec(Codec::Zstd)
        .shuffle(Shuffle::Bytes)
        .element_size(4);
    let decoded_lengths = [1024, 2048, 5120];
    let mut payloads = Vec::new();
    let mut descriptors = Vec::new();
    let mut source_offset = 0;
    let mut workspace = EncodeWorkspace::new();
    for decoded_len in decoded_lengths {
        let mut payload = Vec::new();
        encoder
            .encode_block_into(
                &source[source_offset..source_offset + decoded_len],
                &mut payload,
                &mut workspace,
            )
            .unwrap();
        assert!(payload.len() <= encoder.maximum_encoded_block_len(decoded_len).unwrap());
        descriptors.push(BlockDescriptor::new(decoded_len, payload.len()).unwrap());
        payloads.push(payload);
        source_offset += decoded_len;
    }

    let layout = encoder.chunk_layout(&descriptors).unwrap();
    assert_eq!(layout.blocks().len(), decoded_lengths.len());
    let encoded = layout.assemble(payloads.iter().map(Vec::as_slice)).unwrap();
    let decoder = Decoder::from_layout(layout);
    assert_eq!(decoder.decode(&encoded).unwrap(), source);
}

#[test]
fn parsed_raw_and_empty_layouts_can_be_serialized_and_assembled() {
    let source = sample_f32(32);
    let raw = Encoder::new().compression_level(0).encode(&source).unwrap();
    let raw_decoder = Decoder::from_encoded(&raw).unwrap();
    let raw_layout = raw_decoder.layout();
    assert_eq!(raw_layout.blocks().len(), 1);
    let raw_block = raw_layout.block(0).unwrap();
    assert_eq!(raw_block.encoded_range(), 16..raw.len());
    assert_eq!(
        raw_layout
            .assemble(std::iter::once(&raw[raw_block.encoded_range()]))
            .unwrap(),
        raw
    );

    let empty = Encoder::new().encode(&[]).unwrap();
    let empty_decoder = Decoder::from_encoded(&empty).unwrap();
    let empty_layout = empty_decoder.layout();
    assert!(empty_layout.blocks().next().is_none());
    assert_eq!(
        empty_layout.assemble(std::iter::empty::<&[u8]>()).unwrap(),
        empty
    );
    let mut short_prefix = [0; 15];
    assert!(empty_layout.write_prefix(&mut short_prefix).is_err());
}
