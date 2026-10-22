//! AIFF files built in the test, with no external tool.

use std::num::NonZeroU32;

use cutlass_aiff::{Error, Kind, Movie};
use cutlass_core::AudioCodec;

const RATE_44100: [u8; 10] = [0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0];

fn scale(value: u32) -> NonZeroU32 {
    match NonZeroU32::new(value) {
        Some(scale) => scale,
        None => panic!("timescale is non-zero"),
    }
}

fn expect_err(result: Result<Movie, Error>, expected: Error) {
    match result {
        Err(error) if error == expected => {}
        Err(error) => panic!("expected {expected}, got {error}"),
        Ok(_) => panic!("expected {expected}"),
    }
}

#[test]
fn a_pcm_aiff_keeps_its_rate_and_samples() {
    let mut body = comm(1, 2, 16, RATE_44100);
    body.extend(ssnd(0, 0, &[1, 2, 3, 4]));
    let movie = Movie::parse(&form(b"AIFF", &body)).expect("pcm");
    assert_eq!(movie.tracks().len(), 1);
    let track = &movie.tracks()[0];
    assert_eq!(track.timescale(), scale(44_100));
    let Kind::Audio(config) = track.kind() else {
        panic!("pcm stream");
    };
    assert_eq!(config.codec, AudioCodec::Pcm);
    assert_eq!(config.sample_rate, 44_100);
    assert_eq!(config.channels, 1);
    assert_eq!(config.config_record.len(), 18);
    let samples = track.samples();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].data, [1, 2, 3, 4]);
    assert_eq!(samples[0].pts.value, 0);
    assert_eq!(samples[0].pts.timescale, scale(44_100));
    assert!(samples[0].dts.is_none());
    assert_eq!(samples[0].duration.map(|span| span.value), Some(2));
    assert!(samples[0].keyframe);
}

#[test]
fn a_24_bit_frame_is_three_bytes() {
    let mut body = comm(1, 1, 24, RATE_44100);
    body.extend(ssnd(0, 0, &[1, 2, 3]));
    let movie = Movie::parse(&form(b"AIFF", &body)).expect("wide");
    assert_eq!(
        movie.tracks()[0].samples()[0]
            .duration
            .map(|span| span.value),
        Some(1)
    );
}

#[test]
fn an_aifc_sowt_id_is_pcm() {
    let mut body = comm_aifc(1, 1, 16, RATE_44100, *b"sowt", &[]);
    body.extend(ssnd(0, 0, &[2, 1]));
    let movie = Movie::parse(&form(b"AIFC", &body)).expect("sowt");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("pcm stream");
    };
    assert_eq!(config.codec, AudioCodec::Pcm);
    assert!(config.config_record.windows(4).any(|id| id == b"sowt"));
    assert_eq!(movie.tracks()[0].samples()[0].data, [2, 1]);
}

#[test]
fn a_float_compression_id_is_pcm() {
    let mut body = comm_aifc(1, 1, 32, RATE_44100, *b"fl32", &[]);
    body.extend(ssnd(0, 0, &[1, 2, 3, 4]));
    let movie = Movie::parse(&form(b"AIFC", &body)).expect("float");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("float stream");
    };
    assert_eq!(config.codec, AudioCodec::Pcm);
}

#[test]
fn an_uncompressed_aifc_id_is_pcm() {
    let body = comm_aifc(1, 0, 16, RATE_44100, *b"NONE", &[]);
    let movie = Movie::parse(&form(b"AIFC", &body)).expect("none");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("pcm stream");
    };
    assert_eq!(config.codec, AudioCodec::Pcm);
    assert!(movie.tracks()[0].samples().is_empty());
}

#[test]
fn a_raw_id_is_pcm() {
    let body = comm_aifc(1, 0, 8, RATE_44100, *b"raw ", &[]);
    let movie = Movie::parse(&form(b"AIFC", &body)).expect("raw");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("pcm stream");
    };
    assert_eq!(config.codec, AudioCodec::Pcm);
}

#[test]
fn an_unmapped_id_keeps_its_text_and_frame_count() {
    let mut body = comm_aifc(1, 5, 16, RATE_44100, *b"ulaw", &[]);
    body.extend(ssnd(0, 0, &[1, 2]));
    let movie = Movie::parse(&form(b"AIFC", &body)).expect("ulaw");
    match movie.tracks()[0].kind() {
        Kind::Other { codec_id, .. } => assert_eq!(codec_id, "ulaw"),
        Kind::Audio(_) => panic!("unmapped id"),
    }
    let sample = &movie.tracks()[0].samples()[0];
    assert_eq!(sample.data, [1, 2]);
    assert_eq!(sample.duration.map(|span| span.value), Some(5));
    assert!(sample.keyframe);
    assert!(sample.dts.is_none());
}

#[test]
fn a_non_printable_id_is_hex() {
    let body = comm_aifc(1, 0, 16, RATE_44100, [1, 2, 3, 4], &[]);
    let movie = Movie::parse(&form(b"AIFC", &body)).expect("hex");
    match movie.tracks()[0].kind() {
        Kind::Other { codec_id, .. } => assert_eq!(codec_id, "01020304"),
        Kind::Audio(_) => panic!("unmapped id"),
    }
}

#[test]
fn the_sound_offset_skips_bytes() {
    let mut body = comm(1, 1, 16, RATE_44100);
    body.extend(ssnd(2, 0, &[9, 8, 1, 2]));
    let movie = Movie::parse(&form(b"AIFF", &body)).expect("offset");
    assert_eq!(movie.tracks()[0].samples()[0].data, [1, 2]);
}

#[test]
fn the_sound_block_size_is_not_used() {
    let mut body = comm(1, 2, 16, RATE_44100);
    body.extend(ssnd(0, 7, &[1, 2, 3, 4]));
    let movie = Movie::parse(&form(b"AIFF", &body)).expect("block");
    let sample = &movie.tracks()[0].samples()[0];
    assert_eq!(sample.data, [1, 2, 3, 4]);
    assert_eq!(sample.duration.map(|span| span.value), Some(2));
}

#[test]
fn a_missing_sound_chunk_has_no_samples() {
    let movie = Movie::parse(&form(b"AIFF", &comm(1, 4, 16, RATE_44100))).expect("header");
    assert!(movie.tracks()[0].samples().is_empty());
}

#[test]
fn an_empty_sound_chunk_has_an_empty_packet() {
    let mut body = comm(1, 0, 16, RATE_44100);
    body.extend(ssnd(0, 0, &[]));
    let movie = Movie::parse(&form(b"AIFF", &body)).expect("empty");
    let sample = &movie.tracks()[0].samples()[0];
    assert!(sample.data.is_empty());
    assert_eq!(sample.duration.map(|span| span.value), Some(0));
}

#[test]
fn a_common_chunk_after_the_sound_still_reads() {
    let mut body = ssnd(0, 0, &[1, 2]);
    body.extend(comm(1, 1, 16, RATE_44100));
    let movie = Movie::parse(&form(b"AIFF", &body)).expect("order");
    assert_eq!(movie.tracks()[0].samples()[0].data, [1, 2]);
    assert_eq!(movie.tracks()[0].timescale(), scale(44_100));
}

#[test]
fn a_version_chunk_is_skipped() {
    let mut body = chunk(b"FVER", &[0xA2, 0x80, 0x51, 0x40]);
    body.extend(comm(1, 1, 16, RATE_44100));
    body.extend(ssnd(0, 0, &[1, 2]));
    let movie = Movie::parse(&form(b"AIFF", &body)).expect("version");
    assert_eq!(movie.tracks()[0].samples()[0].data, [1, 2]);
}

#[test]
fn bytes_after_an_aiff_common_chunk_are_left_out() {
    let mut header = comm_body(1, 0, 16, RATE_44100);
    header.extend_from_slice(&[9, 8]);
    let movie = Movie::parse(&form(b"AIFF", &chunk(b"COMM", &header))).expect("tail");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("pcm stream");
    };
    assert_eq!(config.config_record.len(), 18);
}

#[test]
fn bytes_after_the_compression_name_are_left_out() {
    let mut header = comm_aifc_body(1, 0, 16, RATE_44100, *b"NONE", b"x");
    header.push(0xAB);
    let movie = Movie::parse(&form(b"AIFC", &chunk(b"COMM", &header))).expect("name");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("pcm stream");
    };
    assert_eq!(*config.config_record.last().expect("name"), b'x');
    assert!(!config.config_record.contains(&0xAB));
}

#[test]
fn a_bad_form_refuses_the_file() {
    expect_err(Movie::parse(&chunk(b"FORM", b"8SVX")), Error::Form);
    expect_err(
        Movie::parse(&[]),
        Error::Inconsistent("the file has no FORM header"),
    );
    expect_err(
        Movie::parse(&chunk(b"RIFF", b"WAVE")),
        Error::Inconsistent("the file has no FORM header"),
    );
    expect_err(
        Movie::parse(&chunk(b"FORM", b"AI")),
        Error::Inconsistent("the file has no form"),
    );
    expect_err(
        Movie::parse(&form(b"AIFF", &[])),
        Error::Inconsistent("the file has no common chunk"),
    );
}

#[test]
fn a_short_file_is_truncated() {
    assert_eq!(
        Movie::parse(b"FORM"),
        Err(Error::Truncated { at: 0, need: 8 })
    );
}

#[test]
fn bytes_after_the_form_refuse_the_file() {
    let mut bytes = form(b"AIFF", &comm(1, 0, 16, RATE_44100));
    bytes.extend(chunk(b"FORM", b"AIFF"));
    expect_err(
        Movie::parse(&bytes),
        Error::Inconsistent("the file has data after the form"),
    );
}

#[test]
fn a_second_common_chunk_refuses_the_file() {
    let mut body = comm(1, 0, 16, RATE_44100);
    body.extend(comm(1, 0, 16, RATE_44100));
    expect_err(
        Movie::parse(&form(b"AIFF", &body)),
        Error::Inconsistent("the common chunk is repeated"),
    );
}

#[test]
fn a_second_sound_chunk_refuses_the_file() {
    let mut body = comm(1, 0, 16, RATE_44100);
    body.extend(ssnd(0, 0, &[]));
    body.extend(ssnd(0, 0, &[]));
    expect_err(
        Movie::parse(&form(b"AIFF", &body)),
        Error::Inconsistent("the sound chunk is repeated"),
    );
}

#[test]
fn a_zero_rate_refuses_the_file() {
    expect_err(
        Movie::parse(&form(b"AIFF", &comm(1, 0, 16, [0; 10]))),
        Error::Inconsistent("the sample rate is zero"),
    );
}

#[test]
fn a_fractional_rate_refuses_the_file() {
    expect_err(
        Movie::parse(&form(
            b"AIFF",
            &comm(1, 0, 16, [0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 1]),
        )),
        Error::Inconsistent("the sample rate is not a whole number"),
    );
}

#[test]
fn a_negative_rate_refuses_the_file() {
    expect_err(
        Movie::parse(&form(
            b"AIFF",
            &comm(1, 0, 16, [0xC0, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]),
        )),
        Error::Inconsistent("the sample rate is not a whole number"),
    );
}

#[test]
fn a_zero_channel_count_refuses_the_file() {
    expect_err(
        Movie::parse(&form(b"AIFF", &comm(0, 0, 16, RATE_44100))),
        Error::Inconsistent("the channel count is zero"),
    );
}

#[test]
fn a_negative_channel_count_refuses_the_file() {
    expect_err(
        Movie::parse(&form(b"AIFF", &comm(-1, 0, 16, RATE_44100))),
        Error::Inconsistent("the channel count is invalid"),
    );
}

#[test]
fn a_sample_width_that_is_not_a_whole_number_of_bytes_refuses_the_file() {
    expect_err(
        Movie::parse(&form(b"AIFF", &comm(1, 0, 12, RATE_44100))),
        Error::Inconsistent("the sample width is not a whole number of bytes"),
    );
}

#[test]
fn a_zero_sample_width_refuses_the_file() {
    expect_err(
        Movie::parse(&form(b"AIFF", &comm(1, 0, 0, RATE_44100))),
        Error::Inconsistent("the sample width is zero"),
    );
}

#[test]
fn a_negative_sample_width_refuses_the_file() {
    expect_err(
        Movie::parse(&form(b"AIFF", &comm(1, 0, -8, RATE_44100))),
        Error::Inconsistent("the sample width is invalid"),
    );
}

#[test]
fn sound_data_that_does_not_match_the_frame_count_refuses_the_file() {
    let mut body = comm(1, 2, 16, RATE_44100);
    body.extend(ssnd(0, 0, &[1, 2]));
    expect_err(
        Movie::parse(&form(b"AIFF", &body)),
        Error::Inconsistent("the sound data does not match the frame count"),
    );
}

#[test]
fn a_short_common_chunk_refuses_the_file() {
    expect_err(
        Movie::parse(&form(b"AIFF", &chunk(b"COMM", &[0, 1, 0, 0]))),
        Error::Inconsistent("the common chunk is too short"),
    );
    expect_err(
        Movie::parse(&form(b"AIFC", &comm(1, 0, 16, RATE_44100))),
        Error::Inconsistent("the common chunk is too short"),
    );
}

#[test]
fn a_compression_name_that_does_not_fit_refuses_the_file() {
    let mut header = comm_body(1, 0, 16, RATE_44100);
    header.extend_from_slice(b"NONE");
    expect_err(
        Movie::parse(&form(b"AIFC", &chunk(b"COMM", &header))),
        Error::Inconsistent("the compression name does not fit"),
    );
    header.push(3);
    expect_err(
        Movie::parse(&form(b"AIFC", &chunk(b"COMM", &header))),
        Error::Inconsistent("the compression name does not fit"),
    );
}

#[test]
fn a_short_sound_chunk_refuses_the_file() {
    let mut body = comm(1, 0, 16, RATE_44100);
    body.extend(chunk(b"SSND", &[0, 0, 0, 1]));
    expect_err(
        Movie::parse(&form(b"AIFF", &body)),
        Error::Inconsistent("the sound chunk is too short"),
    );
}

#[test]
fn a_sound_offset_that_does_not_fit_refuses_the_file() {
    let mut body = comm(1, 0, 16, RATE_44100);
    body.extend(ssnd(1, 0, &[]));
    expect_err(
        Movie::parse(&form(b"AIFF", &body)),
        Error::Inconsistent("the sound offset does not fit"),
    );
}

fn form(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(kind);
    data.extend_from_slice(body);
    chunk(b"FORM", &data)
}

fn comm(channels: i16, frames: u32, bits: i16, rate: [u8; 10]) -> Vec<u8> {
    chunk(b"COMM", &comm_body(channels, frames, bits, rate))
}

fn comm_body(channels: i16, frames: u32, bits: i16, rate: [u8; 10]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&channels.to_be_bytes());
    data.extend_from_slice(&frames.to_be_bytes());
    data.extend_from_slice(&bits.to_be_bytes());
    data.extend_from_slice(&rate);
    data
}

fn comm_aifc(
    channels: i16,
    frames: u32,
    bits: i16,
    rate: [u8; 10],
    id: [u8; 4],
    name: &[u8],
) -> Vec<u8> {
    chunk(
        b"COMM",
        &comm_aifc_body(channels, frames, bits, rate, id, name),
    )
}

fn comm_aifc_body(
    channels: i16,
    frames: u32,
    bits: i16,
    rate: [u8; 10],
    id: [u8; 4],
    name: &[u8],
) -> Vec<u8> {
    let mut data = comm_body(channels, frames, bits, rate);
    data.extend_from_slice(&id);
    let name_len = u8::try_from(name.len()).expect("name fits");
    data.push(name_len);
    data.extend_from_slice(name);
    data
}

fn ssnd(offset: u32, block: u32, data: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&offset.to_be_bytes());
    body.extend_from_slice(&block.to_be_bytes());
    body.extend_from_slice(data);
    chunk(b"SSND", &body)
}

fn chunk(id: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(id);
    let size = u32::try_from(data.len()).expect("size");
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(data);
    if data.len() % 2 == 1 {
        out.push(0);
    }
    out
}
