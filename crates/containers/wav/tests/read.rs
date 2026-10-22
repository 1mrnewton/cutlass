//! WAV files built in the test, with no external tool.

use std::num::NonZeroU32;

use cutlass_core::AudioCodec;
use cutlass_wav::{Error, Kind, Movie};

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
fn a_pcm_wave_keeps_its_rate_and_samples() {
    let mut body = fmt(1, 2, 48_000, 16, &[]);
    body.extend(chunk(b"data", &[1, 2, 3, 4, 5, 6, 7, 8]));
    let movie = Movie::parse(&wave(&body)).expect("pcm");
    assert_eq!(movie.tracks().len(), 1);
    let track = &movie.tracks()[0];
    assert_eq!(track.timescale(), scale(48_000));
    let Kind::Audio(config) = track.kind() else {
        panic!("pcm stream");
    };
    assert_eq!(config.codec, AudioCodec::Pcm);
    assert_eq!(config.sample_rate, 48_000);
    assert_eq!(config.channels, 2);
    assert_eq!(config.config_record.len(), 16);
    let samples = track.samples();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].data, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(samples[0].pts.value, 0);
    assert_eq!(samples[0].pts.timescale, scale(48_000));
    assert!(samples[0].dts.is_none());
    assert_eq!(samples[0].duration.map(|span| span.value), Some(2));
    assert!(samples[0].keyframe);
}

#[test]
fn a_float_wave_is_pcm() {
    let body = fmt(3, 1, 44_100, 32, &[]);
    let movie = Movie::parse(&wave(&body)).expect("float");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("float stream");
    };
    assert_eq!(config.codec, AudioCodec::Pcm);
    assert_eq!(config.sample_rate, 44_100);
    assert!(movie.tracks()[0].samples().is_empty());
}

#[test]
fn a_24_bit_frame_is_three_bytes() {
    let mut body = fmt(1, 1, 8_000, 24, &[]);
    body.extend(chunk(b"data", &[1, 2, 3, 4, 5, 6]));
    let movie = Movie::parse(&wave(&body)).expect("wide");
    let sample = &movie.tracks()[0].samples()[0];
    assert_eq!(sample.duration.map(|span| span.value), Some(2));
    assert_eq!(sample.data.len(), 6);
}

#[test]
fn an_extensible_tag_stays_unmapped() {
    let body = fmt(0xFFFE, 2, 48_000, 16, &[1, 2, 3, 4]);
    let movie = Movie::parse(&wave(&body)).expect("extensible");
    match movie.tracks()[0].kind() {
        Kind::Other {
            codec_id,
            config_record,
        } => {
            assert_eq!(codec_id, "fffe");
            assert_eq!(&config_record[16..18], [4, 0]);
            assert_eq!(&config_record[18..], [1, 2, 3, 4]);
        }
        Kind::Audio(_) => panic!("extensible tag"),
    }
}

#[test]
fn bytes_after_the_declared_extra_are_left_out() {
    let mut header = fmt_body(1, 1, 8_000, 8, &[9]);
    header.push(0xAB);
    let movie = Movie::parse(&wave(&chunk(b"fmt ", &header))).expect("tail");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("pcm stream");
    };
    assert_eq!(config.config_record.len(), 19);
    assert_eq!(*config.config_record.last().expect("extra"), 9);
}

#[test]
fn an_unmapped_tag_counts_blocks() {
    let mut body = fmt(0x0055, 1, 16_000, 8, &[]);
    body.extend(chunk(b"data", &[1, 2, 3, 4]));
    let movie = Movie::parse(&wave(&body)).expect("mp3");
    match movie.tracks()[0].kind() {
        Kind::Other { codec_id, .. } => assert_eq!(codec_id, "0055"),
        Kind::Audio(_) => panic!("unmapped tag"),
    }
    let sample = &movie.tracks()[0].samples()[0];
    assert_eq!(sample.duration.map(|span| span.value), Some(4));
    assert!(sample.keyframe);
    assert!(sample.dts.is_none());
}

#[test]
fn a_missing_data_chunk_has_no_samples() {
    let movie = Movie::parse(&wave(&fmt(1, 1, 8_000, 8, &[]))).expect("header");
    assert!(movie.tracks()[0].samples().is_empty());
}

#[test]
fn a_list_chunk_is_skipped() {
    let mut body = chunk(b"LIST", b"INFO");
    body.extend(fmt(1, 1, 8_000, 8, &[]));
    body.extend(chunk(b"data", &[3]));
    let movie = Movie::parse(&wave(&body)).expect("list");
    assert_eq!(movie.tracks()[0].samples()[0].data, [3]);
}

#[test]
fn a_fact_chunk_is_skipped() {
    let mut body = chunk(b"fact", &4u32.to_le_bytes());
    body.extend(fmt(1, 1, 8_000, 16, &[]));
    body.extend(chunk(b"data", &[1, 2]));
    let movie = Movie::parse(&wave(&body)).expect("fact");
    assert_eq!(
        movie.tracks()[0].samples()[0]
            .duration
            .map(|span| span.value),
        Some(1)
    );
}

#[test]
fn a_format_after_the_data_still_reads() {
    let mut body = chunk(b"data", &[9, 8]);
    body.extend(fmt(1, 1, 8_000, 16, &[]));
    let movie = Movie::parse(&wave(&body)).expect("order");
    assert_eq!(movie.tracks()[0].samples()[0].data, [9, 8]);
    assert_eq!(movie.tracks()[0].timescale(), scale(8_000));
}

#[test]
fn an_odd_data_chunk_does_not_swallow_the_next_one() {
    let mut body = fmt(1, 1, 8_000, 8, &[]);
    body.extend(chunk(b"data", &[7]));
    body.extend(chunk(b"JUNK", &[1, 2, 3, 4]));
    let movie = Movie::parse(&wave(&body)).expect("pad");
    assert_eq!(movie.tracks()[0].samples()[0].data, [7]);
}

#[test]
fn a_bad_form_refuses_the_file() {
    expect_err(Movie::parse(&chunk(b"RIFF", b"AVI ")), Error::Form);
    expect_err(Movie::parse(&chunk(b"RIFF", b"RF64")), Error::Form);
    expect_err(
        Movie::parse(&[]),
        Error::Inconsistent("the file has no RIFF header"),
    );
    expect_err(
        Movie::parse(&chunk(b"RIFF", b"WAVE")),
        Error::Inconsistent("the file has no format"),
    );
}

#[test]
fn a_short_file_is_truncated() {
    assert_eq!(
        Movie::parse(b"RIFF"),
        Err(Error::Truncated { at: 0, need: 8 })
    );
}

#[test]
fn bytes_after_the_wave_refuse_the_file() {
    let mut bytes = wave(&fmt(1, 1, 8_000, 8, &[]));
    bytes.extend(chunk(b"RIFF", b"WAVE"));
    expect_err(
        Movie::parse(&bytes),
        Error::Inconsistent("the file has data after the wave"),
    );
}

#[test]
fn a_second_format_chunk_refuses_the_file() {
    let mut body = fmt(1, 1, 8_000, 8, &[]);
    body.extend(fmt(1, 1, 8_000, 8, &[]));
    expect_err(
        Movie::parse(&wave(&body)),
        Error::Inconsistent("the format chunk is repeated"),
    );
}

#[test]
fn a_second_data_chunk_refuses_the_file() {
    let mut body = fmt(1, 1, 8_000, 8, &[]);
    body.extend(chunk(b"data", &[1]));
    body.extend(chunk(b"data", &[2]));
    expect_err(
        Movie::parse(&wave(&body)),
        Error::Inconsistent("the data chunk is repeated"),
    );
}

#[test]
fn a_zero_rate_refuses_the_file() {
    expect_err(
        Movie::parse(&wave(&fmt(1, 1, 0, 8, &[]))),
        Error::Inconsistent("the sample rate is zero"),
    );
}

#[test]
fn a_zero_channel_count_refuses_the_file() {
    expect_err(
        Movie::parse(&wave(&fmt(1, 0, 8_000, 8, &[]))),
        Error::Inconsistent("the channel count is zero"),
    );
}

#[test]
fn a_sample_width_that_is_not_a_whole_number_of_bytes_refuses_the_file() {
    expect_err(
        Movie::parse(&wave(&fmt(1, 1, 8_000, 12, &[]))),
        Error::Inconsistent("the sample width is not a whole number of bytes"),
    );
}

#[test]
fn a_zero_sample_width_refuses_the_file() {
    expect_err(
        Movie::parse(&wave(&fmt(1, 1, 8_000, 0, &[]))),
        Error::Inconsistent("the sample width is zero"),
    );
}

#[test]
fn a_block_align_that_does_not_match_refuses_the_file() {
    let mut header = fmt_body(1, 2, 8_000, 16, &[]);
    header[12..14].copy_from_slice(&2u16.to_le_bytes());
    expect_err(
        Movie::parse(&wave(&chunk(b"fmt ", &header))),
        Error::Inconsistent("the block align does not match the sample"),
    );
}

#[test]
fn a_data_size_that_does_not_divide_refuses_the_file() {
    let mut body = fmt(1, 1, 8_000, 16, &[]);
    body.extend(chunk(b"data", &[1, 2, 3]));
    expect_err(
        Movie::parse(&wave(&body)),
        Error::Inconsistent("the data size is not a multiple of the block align"),
    );
}

#[test]
fn a_short_wave_header_refuses_the_file() {
    expect_err(
        Movie::parse(&wave(&chunk(b"fmt ", &[1, 0, 1, 0]))),
        Error::Inconsistent("the wave header is too short"),
    );
    let mut almost = fmt_body(1, 1, 8_000, 8, &[]);
    almost.pop();
    expect_err(
        Movie::parse(&wave(&chunk(b"fmt ", &almost))),
        Error::Inconsistent("the wave header is too short"),
    );
    let mut one_past = fmt_body(1, 1, 8_000, 8, &[]);
    one_past.push(0);
    expect_err(
        Movie::parse(&wave(&chunk(b"fmt ", &one_past))),
        Error::Inconsistent("the wave header is too short"),
    );
}

#[test]
fn a_wave_extra_that_does_not_fit_refuses_the_file() {
    let mut header = fmt_body(1, 1, 8_000, 8, &[]);
    header.extend_from_slice(&4u16.to_le_bytes());
    header.push(1);
    expect_err(
        Movie::parse(&wave(&chunk(b"fmt ", &header))),
        Error::Inconsistent("the wave extra does not fit"),
    );
}

#[test]
fn a_zero_block_align_on_an_unmapped_tag_refuses_the_data() {
    let mut body = fmt(0x0055, 1, 8_000, 0, &[]);
    body.extend(chunk(b"data", &[1]));
    expect_err(
        Movie::parse(&wave(&body)),
        Error::Inconsistent("the block align is zero"),
    );
}

fn wave(body: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"WAVE");
    data.extend_from_slice(body);
    chunk(b"RIFF", &data)
}

fn fmt(tag: u16, channels: u16, rate: u32, bits: u16, extra: &[u8]) -> Vec<u8> {
    chunk(b"fmt ", &fmt_body(tag, channels, rate, bits, extra))
}

fn fmt_body(tag: u16, channels: u16, rate: u32, bits: u16, extra: &[u8]) -> Vec<u8> {
    let width = bits / 8;
    let align = channels.saturating_mul(width);
    let byte_rate = rate.saturating_mul(u32::from(align));
    let mut data = vec![0; 16];
    data[..2].copy_from_slice(&tag.to_le_bytes());
    data[2..4].copy_from_slice(&channels.to_le_bytes());
    data[4..8].copy_from_slice(&rate.to_le_bytes());
    data[8..12].copy_from_slice(&byte_rate.to_le_bytes());
    data[12..14].copy_from_slice(&align.to_le_bytes());
    data[14..16].copy_from_slice(&bits.to_le_bytes());
    if !extra.is_empty() {
        let extra_len = u16::try_from(extra.len()).expect("extra fits");
        data.extend_from_slice(&extra_len.to_le_bytes());
        data.extend_from_slice(extra);
    }
    data
}

fn chunk(id: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(id);
    let size = u32::try_from(data.len()).expect("size");
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(data);
    if data.len() % 2 == 1 {
        out.push(0);
    }
    out
}
