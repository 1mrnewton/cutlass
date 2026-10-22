//! AVI files built in the test, with no external tool.

use std::num::NonZeroU32;

use cutlass_avi::{Error, Kind, Movie};
use cutlass_core::{AudioCodec, VideoCodec};

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
fn a_video_stream_keeps_its_codec_and_size() {
    let body = hdrl(1, &video_stream(b"MJPG", 320, 180, 1, 25, &[0x01, 0x42]));
    let movie = Movie::parse(&avi(&body)).expect("video");
    assert_eq!(movie.tracks().len(), 1);
    let track = &movie.tracks()[0];
    assert_eq!(track.number(), 0);
    assert_eq!(track.timescale(), scale(25));
    let Kind::Video(config) = track.kind() else {
        panic!("video stream");
    };
    assert_eq!(config.codec, VideoCodec::Mjpeg);
    assert_eq!(config.width, 320);
    assert_eq!(config.height, 180);
    assert_eq!(config.config_record, [0x01, 0x42]);
}

#[test]
fn a_lowercase_fourcc_matches_the_codec() {
    let body = hdrl(1, &video_stream(b"mjpg", 16, -9, 1000, 30_000, &[]));
    let movie = Movie::parse(&avi(&body)).expect("case");
    assert_eq!(movie.tracks()[0].timescale(), scale(30));
    let Kind::Video(config) = movie.tracks()[0].kind() else {
        panic!("video stream");
    };
    assert_eq!(config.codec, VideoCodec::Mjpeg);
    assert_eq!(config.height, 9);
}

#[test]
fn an_unmapped_fourcc_keeps_its_id_and_config() {
    let body = hdrl(1, &video_stream(b"YUY2", 8, 8, 1, 1, &[9]));
    let movie = Movie::parse(&avi(&body)).expect("yuy2");
    match movie.tracks()[0].kind() {
        Kind::Other {
            codec_id,
            config_record,
        } => {
            assert_eq!(codec_id, "YUY2");
            assert_eq!(config_record, &[9]);
        }
        Kind::Video(_) | Kind::Audio(_) => panic!("unmapped codec"),
    }
}

#[test]
fn uncompressed_video_keeps_a_zero_fourcc() {
    let body = hdrl(1, &video_stream(&[0, 0, 0, 0], 8, 8, 1, 1, &[]));
    let movie = Movie::parse(&avi(&body)).expect("raw");
    match movie.tracks()[0].kind() {
        Kind::Other { codec_id, .. } => assert_eq!(codec_id, "0000"),
        Kind::Video(_) | Kind::Audio(_) => panic!("uncompressed"),
    }
}

#[test]
fn a_pcm_stream_keeps_its_rate() {
    let body = hdrl(1, &audio_stream(1, 2, 48_000, &[]));
    let movie = Movie::parse(&avi(&body)).expect("pcm");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("audio stream");
    };
    assert_eq!(config.codec, AudioCodec::Pcm);
    assert_eq!(config.sample_rate, 48_000);
    assert_eq!(config.channels, 2);
    assert!(config.config_record.is_empty());
    assert_eq!(movie.tracks()[0].timescale(), scale(48_000));
}

#[test]
fn an_aac_stream_keeps_its_extra_bytes() {
    let body = hdrl(1, &audio_stream(0x00FF, 1, 44_100, &[0x12, 0x10]));
    let movie = Movie::parse(&avi(&body)).expect("aac");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("audio stream");
    };
    assert_eq!(config.codec, AudioCodec::Aac);
    assert_eq!(config.config_record, [0x12, 0x10]);
}

#[test]
fn an_extensible_wave_tag_stays_unmapped() {
    let body = hdrl(1, &audio_stream(0xFFFE, 2, 48_000, &[1, 2, 3]));
    let movie = Movie::parse(&avi(&body)).expect("extensible");
    match movie.tracks()[0].kind() {
        Kind::Other {
            codec_id,
            config_record,
        } => {
            assert_eq!(codec_id, "fffe");
            assert_eq!(config_record, &[1, 2, 3]);
        }
        Kind::Video(_) | Kind::Audio(_) => panic!("extensible tag"),
    }
}

#[test]
fn a_text_stream_keeps_its_type() {
    let stream = list(b"strl", &strh(b"txts", 1, 1));
    let movie = Movie::parse(&avi(&hdrl(1, &stream))).expect("text");
    match movie.tracks()[0].kind() {
        Kind::Other { codec_id, .. } => assert_eq!(codec_id, "txts"),
        Kind::Video(_) | Kind::Audio(_) => panic!("text stream"),
    }
}

#[test]
fn an_empty_header_has_no_streams() {
    let movie = Movie::parse(&avi(&hdrl(0, &[]))).expect("empty");
    assert!(movie.tracks().is_empty());
}

#[test]
fn junk_and_a_pad_byte_are_skipped() {
    let mut body = chunk(b"JUNK", &[0xAB]);
    body.extend(hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 10, &[])));
    let movie = Movie::parse(&avi(&body)).expect("junk");
    assert_eq!(movie.tracks()[0].timescale(), scale(10));
}

#[test]
fn a_bad_form_refuses_the_file() {
    expect_err(Movie::parse(&chunk(b"RIFF", b"WAVE")), Error::Form);
    expect_err(
        Movie::parse(&[]),
        Error::Inconsistent("the file has no RIFF header"),
    );
    expect_err(
        Movie::parse(&chunk(b"RIFF", b"AVI ")),
        Error::Inconsistent("the file has no header"),
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
fn bytes_after_the_movie_refuse_the_file() {
    let mut bytes = avi(&hdrl(0, &[]));
    bytes.extend(chunk(b"RIFF", b"AVIX"));
    expect_err(
        Movie::parse(&bytes),
        Error::Inconsistent("the file has data after the movie"),
    );
}

#[test]
fn a_stream_count_that_does_not_match_refuses_the_file() {
    expect_err(
        Movie::parse(&avi(&hdrl(2, &video_stream(b"MJPG", 4, 4, 1, 1, &[])))),
        Error::Inconsistent("the stream count does not match"),
    );
}

#[test]
fn a_zero_size_refuses_the_file() {
    expect_err(
        Movie::parse(&avi(&hdrl(1, &video_stream(b"MJPG", 0, 4, 1, 1, &[])))),
        Error::Inconsistent("the video size is invalid"),
    );
}

#[test]
fn a_zero_rate_refuses_the_file() {
    let stream = list(b"strl", &strh(b"vids", 1, 0));
    expect_err(
        Movie::parse(&avi(&hdrl(1, &stream))),
        Error::Inconsistent("the stream scale or rate is zero"),
    );
}

#[test]
fn a_zero_channel_count_refuses_the_file() {
    expect_err(
        Movie::parse(&avi(&hdrl(1, &audio_stream(1, 0, 48_000, &[])))),
        Error::Inconsistent("the channel count is zero"),
    );
}

fn avi(body: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"AVI ");
    data.extend_from_slice(body);
    chunk(b"RIFF", &data)
}

fn hdrl(streams: u32, body: &[u8]) -> Vec<u8> {
    let mut data = avih(streams);
    data.extend_from_slice(body);
    list(b"hdrl", &data)
}

fn avih(streams: u32) -> Vec<u8> {
    let mut data = vec![0; 56];
    data[24..28].copy_from_slice(&streams.to_le_bytes());
    chunk(b"avih", &data)
}

fn video_stream(
    fourcc: &[u8; 4],
    width: i32,
    height: i32,
    scale: u32,
    rate: u32,
    extra: &[u8],
) -> Vec<u8> {
    let mut body = strh(b"vids", scale, rate);
    body.extend(bitmap(width, height, fourcc, extra));
    list(b"strl", &body)
}

fn audio_stream(tag: u16, channels: u16, rate: u32, extra: &[u8]) -> Vec<u8> {
    let mut body = strh(b"auds", 1, rate);
    body.extend(wave(tag, channels, rate, extra));
    list(b"strl", &body)
}

fn strh(kind: &[u8; 4], scale: u32, rate: u32) -> Vec<u8> {
    let mut data = vec![0; 48];
    data[..4].copy_from_slice(kind);
    data[20..24].copy_from_slice(&scale.to_le_bytes());
    data[24..28].copy_from_slice(&rate.to_le_bytes());
    chunk(b"strh", &data)
}

fn bitmap(width: i32, height: i32, compression: &[u8; 4], extra: &[u8]) -> Vec<u8> {
    let mut data = vec![0; 40 + extra.len()];
    data[..4].copy_from_slice(&40u32.to_le_bytes());
    data[4..8].copy_from_slice(&width.to_le_bytes());
    data[8..12].copy_from_slice(&height.to_le_bytes());
    data[16..20].copy_from_slice(compression);
    data[40..].copy_from_slice(extra);
    chunk(b"strf", &data)
}

fn wave(tag: u16, channels: u16, rate: u32, extra: &[u8]) -> Vec<u8> {
    let mut data = vec![0; 18 + extra.len()];
    data[..2].copy_from_slice(&tag.to_le_bytes());
    data[2..4].copy_from_slice(&channels.to_le_bytes());
    data[4..8].copy_from_slice(&rate.to_le_bytes());
    let extra_len = u16::try_from(extra.len()).expect("extra fits");
    data[16..18].copy_from_slice(&extra_len.to_le_bytes());
    data[18..].copy_from_slice(extra);
    chunk(b"strf", &data)
}

fn list(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(kind);
    data.extend_from_slice(body);
    chunk(b"LIST", &data)
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
