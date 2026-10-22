//! FLAC files built in the test, with no external tool.

use std::num::NonZeroU32;

use cutlass_core::AudioCodec;
use cutlass_flac_file::{Error, Movie};

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
fn a_stream_keeps_its_rate_and_one_packet() {
    let mut file = marker();
    file.extend(block(true, 0, &stream_info(48_000, 1, 960)));
    file.extend([1, 2, 3]);
    let movie = Movie::parse(&file).expect("flac");
    let track = &movie.tracks()[0];
    assert_eq!(track.timescale(), scale(48_000));
    assert_eq!(track.config().codec, AudioCodec::Flac);
    assert_eq!(track.config().sample_rate, 48_000);
    assert_eq!(track.config().channels, 2);
    assert_eq!(track.config().config_record.len(), 34);
    assert_eq!(
        &track.config().config_record[10..18],
        &[0x0B, 0xB8, 0x02, 0xF0, 0x00, 0x00, 0x03, 0xC0]
    );
    let samples = track.samples();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].data, [1, 2, 3]);
    assert_eq!(samples[0].pts.value, 0);
    assert_eq!(samples[0].pts.timescale, scale(48_000));
    assert!(samples[0].dts.is_none());
    assert_eq!(samples[0].duration.map(|span| span.value), Some(960));
    assert!(samples[0].keyframe);
}

#[test]
fn a_zero_sample_count_leaves_the_duration_missing() {
    let mut file = marker();
    file.extend(block(true, 0, &stream_info(48_000, 1, 0)));
    file.extend([9]);
    let movie = Movie::parse(&file).expect("unknown length");
    assert!(movie.tracks()[0].samples()[0].duration.is_none());
    assert_eq!(movie.tracks()[0].samples()[0].data, [9]);
}

#[test]
fn a_padding_block_is_skipped() {
    let mut file = marker();
    file.extend(block(false, 0, &stream_info(8_000, 0, 4)));
    file.extend(block(true, 1, &[0, 0, 0, 0]));
    file.extend([5]);
    let movie = Movie::parse(&file).expect("padding");
    assert_eq!(movie.tracks()[0].config().channels, 1);
    assert_eq!(movie.tracks()[0].samples()[0].data, [5]);
    assert_eq!(
        movie.tracks()[0].samples()[0]
            .duration
            .map(|span| span.value),
        Some(4)
    );
}

#[test]
fn no_audio_and_a_zero_count_has_no_samples() {
    let mut file = marker();
    file.extend(block(true, 0, &stream_info(8_000, 0, 0)));
    let movie = Movie::parse(&file).expect("empty");
    assert!(movie.tracks()[0].samples().is_empty());
}

#[test]
fn a_bad_form_refuses_the_file() {
    expect_err(Movie::parse(b"RIFF"), Error::Form);
    expect_err(
        Movie::parse(&[]),
        Error::Inconsistent("the file has no stream info"),
    );
}

#[test]
fn a_short_file_is_truncated() {
    assert_eq!(
        Movie::parse(b"fLa"),
        Err(Error::Truncated { at: 0, need: 4 })
    );
}

#[test]
fn a_missing_stream_info_refuses_the_file() {
    let mut file = marker();
    file.extend(block(true, 1, &[0, 0]));
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the file has no stream info"),
    );
}

#[test]
fn a_second_stream_info_refuses_the_file() {
    let mut file = marker();
    file.extend(block(false, 0, &stream_info(8_000, 0, 0)));
    file.extend(block(true, 0, &stream_info(8_000, 0, 0)));
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the stream info is repeated"),
    );
}

#[test]
fn a_stream_info_of_the_wrong_size_refuses_the_file() {
    let mut file = marker();
    file.extend(block(true, 0, &[0; 10]));
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the stream info is the wrong size"),
    );
}

#[test]
fn a_zero_rate_refuses_the_file() {
    let mut file = marker();
    file.extend(block(true, 0, &stream_info(0, 0, 0)));
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the sample rate is zero"),
    );
}

#[test]
fn a_count_with_no_audio_refuses_the_file() {
    let mut file = marker();
    file.extend(block(true, 0, &stream_info(8_000, 0, 4)));
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the stream has no audio"),
    );
}

#[test]
fn metadata_that_does_not_end_refuses_the_file() {
    let mut file = marker();
    file.extend(block(false, 0, &stream_info(8_000, 0, 0)));
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the metadata does not end"),
    );
}

#[test]
fn an_invalid_block_type_refuses_the_file() {
    let mut file = marker();
    file.extend(block(false, 0, &stream_info(8_000, 0, 0)));
    file.extend(block(true, 127, &[]));
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the metadata block is invalid"),
    );
}

#[test]
fn a_short_block_is_truncated() {
    let mut file = marker();
    file.extend([0x80, 0x00, 0x00, 0x22]);
    assert_eq!(
        Movie::parse(&file),
        Err(Error::Truncated { at: 8, need: 34 })
    );
}

fn marker() -> Vec<u8> {
    b"fLaC".to_vec()
}

fn block(last: bool, kind: u8, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let flag = if last { 0x80 } else { 0 };
    out.push(flag | kind);
    let len = u32::try_from(body.len()).expect("len");
    out.extend_from_slice(&len.to_be_bytes()[1..]);
    out.extend_from_slice(body);
    out
}

fn stream_info(rate: u32, channels: u64, samples: u64) -> Vec<u8> {
    let mut data = vec![0; 34];
    data[0..2].copy_from_slice(&16u16.to_be_bytes());
    data[2..4].copy_from_slice(&16u16.to_be_bytes());
    let packed = (u64::from(rate) << 44) | (channels << 41) | (15 << 36) | samples;
    data[10..18].copy_from_slice(&packed.to_be_bytes());
    data
}
