//! MKV and WebM files built in the test, with no external tool.

use std::num::NonZeroU32;

use cutlass_core::{AudioCodec, VideoCodec};
use cutlass_mkv::{DocType, Error, Kind, Movie};

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
fn a_video_track_keeps_its_codec_and_size() {
    let movie = Movie::parse(&file(
        "matroska",
        &segment_with(Some(5_000.0), &video_track()),
    ))
    .expect("video file");
    assert_eq!(movie.doc_type(), DocType::Matroska);
    assert_eq!(movie.timescale(), scale(1_000));
    assert_eq!(movie.duration(), Some(5_000));
    assert_eq!(movie.tracks().len(), 1);
    let track = &movie.tracks()[0];
    assert_eq!(track.number(), 1);
    let Kind::Video(config) = track.kind() else {
        panic!("video track");
    };
    assert_eq!(config.codec, VideoCodec::H264);
    assert_eq!(config.width, 320);
    assert_eq!(config.height, 180);
    assert_eq!(config.config_record, [0x01, 0x42, 0x00]);
}

#[test]
fn a_webm_audio_track_keeps_its_rate() {
    let mut audio = uint_el(&[0xB5], &48_000.0f64.to_be_bytes());
    audio.extend(uint_el(&[0x9F], &uint_bytes(2)));
    // Output sampling frequency is not the rate this track reports.
    audio.extend(element(&[0x78, 0xB5], &96_000.0f64.to_be_bytes()));
    let entry = track_entry(2, 2, "A_OPUS", &[0x4F, 0x70], &element(&[0xE1], &audio));
    let movie = Movie::parse(&file("webm", &tracks(&entry))).expect("webm file");
    assert_eq!(movie.doc_type(), DocType::Webm);
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("audio track");
    };
    assert_eq!(config.codec, AudioCodec::Opus);
    assert_eq!(config.sample_rate, 48_000);
    assert_eq!(config.channels, 2);
    assert_eq!(config.config_record, [0x4F, 0x70]);
}

#[test]
fn a_missing_audio_rate_uses_the_spec_default() {
    let entry = track_entry(1, 2, "A_OPUS", &[], &[]);
    let movie = Movie::parse(&file("matroska", &tracks(&entry))).expect("defaults");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("audio track");
    };
    assert_eq!(config.sample_rate, 8_000);
    assert_eq!(config.channels, 1);
    assert!(config.config_record.is_empty());
}

#[test]
fn an_aac_profile_id_is_aac() {
    let rate = uint_el(&[0xB5], &44_100.0f32.to_be_bytes());
    let entry = track_entry(
        1,
        2,
        "A_AAC/MPEG4/LC",
        &[0x12, 0x10],
        &element(&[0xE1], &rate),
    );
    let movie = Movie::parse(&file("matroska", &tracks(&entry))).expect("aac");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("audio track");
    };
    assert_eq!(config.codec, AudioCodec::Aac);
    assert_eq!(config.sample_rate, 44_100);
    assert_eq!(config.channels, 1);
    assert_eq!(config.config_record, [0x12, 0x10]);
}

#[test]
fn an_unmapped_codec_keeps_its_id_and_config() {
    let entry = track_entry(7, 1, "V_FFV1", &[0xAB], &video_settings(16, 16));
    let movie = Movie::parse(&file("matroska", &tracks(&entry))).expect("ffv1");
    match movie.tracks()[0].kind() {
        Kind::Other {
            codec_id,
            config_record,
        } => {
            assert_eq!(codec_id, "V_FFV1");
            assert_eq!(config_record, &[0xAB]);
        }
        Kind::Video(_) | Kind::Audio(_) => panic!("unmapped codec"),
    }
    assert_eq!(movie.tracks()[0].number(), 7);
}

#[test]
fn an_empty_segment_has_no_tracks() {
    let movie = Movie::parse(&file("matroska\0", &[])).expect("empty");
    assert_eq!(movie.doc_type(), DocType::Matroska);
    assert_eq!(movie.timescale(), scale(1_000));
    assert_eq!(movie.duration(), None);
    assert!(movie.tracks().is_empty());
}

#[test]
fn a_custom_timestamp_scale_sets_the_timescale() {
    let movie = Movie::parse(&file("matroska", &info(1_000, None))).expect("microseconds");
    assert_eq!(movie.timescale(), scale(1_000_000));
}

#[test]
fn a_bad_doctype_refuses_the_file() {
    expect_err(Movie::parse(&file("mkv", &[])), Error::DocType);
    expect_err(
        Movie::parse(&element(EBML, &[])),
        Error::Inconsistent("the file has no doctype"),
    );
    expect_err(
        Movie::parse(&[]),
        Error::Inconsistent("the file has no EBML header"),
    );
}

#[test]
fn a_string_that_is_not_utf8_refuses_the_file() {
    let header = element(EBML, &element(DOC_TYPE, &[0xFF]));
    expect_err(
        Movie::parse(&header),
        Error::Inconsistent("a string is not UTF-8"),
    );
}

#[test]
fn a_high_ebml_read_version_refuses_the_file() {
    let mut header = string_el(DOC_TYPE, "matroska");
    header.extend(uint_el(EBML_READ_VERSION, &uint_bytes(2)));
    expect_err(Movie::parse(&element(EBML, &header)), Error::Version);
}

#[test]
fn a_timestamp_scale_that_does_not_divide_refuses_the_file() {
    expect_err(
        Movie::parse(&file("matroska", &info(3, None))),
        Error::Inconsistent("the timestamp scale does not divide one second"),
    );
    expect_err(
        Movie::parse(&file("matroska", &info(0, None))),
        Error::Inconsistent("the timestamp scale is zero"),
    );
}

#[test]
fn a_duration_that_is_not_a_whole_number_refuses_the_file() {
    expect_err(
        Movie::parse(&file("matroska", &info(1_000_000, Some(1.5)))),
        Error::Inconsistent("the duration is not a whole number"),
    );
    expect_err(
        Movie::parse(&file("matroska", &info(1_000_000, Some(-1.0)))),
        Error::Inconsistent("the duration is not a whole number"),
    );
}

#[test]
fn a_zero_dimension_refuses_the_file() {
    let entry = track_entry(1, 1, "V_VP8", &[], &video_settings(0, 180));
    expect_err(
        Movie::parse(&file("webm", &tracks(&entry))),
        Error::Inconsistent("video width and height must be non-zero"),
    );
}

#[test]
fn a_compressed_track_refuses_the_file() {
    let entry = element(TRACK_ENTRY, &element(CONTENT_ENCODINGS, &[0x01]));
    expect_err(
        Movie::parse(&file("matroska", &tracks(&entry))),
        Error::Inconsistent("a track is compressed or encrypted"),
    );
}

#[test]
fn a_codec_on_the_wrong_track_type_refuses_the_file() {
    let video = track_entry(1, 1, "A_OPUS", &[], &video_settings(16, 16));
    expect_err(
        Movie::parse(&file("matroska", &tracks(&video))),
        Error::Inconsistent("the codec does not match the track"),
    );
    let audio = track_entry(1, 2, "V_VP8", &[], &[]);
    expect_err(
        Movie::parse(&file("matroska", &tracks(&audio))),
        Error::Inconsistent("the codec does not match the track"),
    );
}

#[test]
fn a_second_segment_refuses_the_file() {
    let mut bytes = file("matroska", &[]);
    bytes.extend(element(SEGMENT, &[]));
    expect_err(
        Movie::parse(&bytes),
        Error::Inconsistent("the file has data after the segment"),
    );
}

#[test]
fn a_repeated_track_number_refuses_the_file() {
    let mut body = track_entry(1, 2, "A_OPUS", &[], &[]);
    body.extend(track_entry(1, 2, "A_VORBIS", &[], &[]));
    expect_err(
        Movie::parse(&file("matroska", &tracks(&body))),
        Error::Inconsistent("a track number is repeated"),
    );
}

#[test]
fn a_void_and_a_crc_are_skipped() {
    let mut body = element(VOID, &[0, 0]);
    body.extend(element(CRC32, &[1, 2, 3, 4]));
    body.extend(tracks(&track_entry(1, 2, "A_FLAC", &[9], &[])));
    let movie = Movie::parse(&file("matroska", &body)).expect("skipped");
    let Kind::Audio(config) = movie.tracks()[0].kind() else {
        panic!("audio track");
    };
    assert_eq!(config.codec, AudioCodec::Flac);
    assert_eq!(config.config_record, [9]);
}

#[test]
fn a_short_file_is_truncated() {
    assert_eq!(
        Movie::parse(&[0x1A]),
        Err(Error::Truncated { at: 1, need: 3 })
    );
}

const EBML: &[u8] = &[0x1A, 0x45, 0xDF, 0xA3];
const DOC_TYPE: &[u8] = &[0x42, 0x82];
const EBML_READ_VERSION: &[u8] = &[0x42, 0xF7];
const SEGMENT: &[u8] = &[0x18, 0x53, 0x80, 0x67];
const INFO: &[u8] = &[0x15, 0x49, 0xA9, 0x66];
const TIMESTAMP_SCALE: &[u8] = &[0x2A, 0xD7, 0xB1];
const DURATION: &[u8] = &[0x44, 0x89];
const TRACKS: &[u8] = &[0x16, 0x54, 0xAE, 0x6B];
const TRACK_ENTRY: &[u8] = &[0xAE];
const CONTENT_ENCODINGS: &[u8] = &[0x6D, 0x80];
const VOID: &[u8] = &[0xEC];
const CRC32: &[u8] = &[0xBF];

fn file(doctype: &str, segment: &[u8]) -> Vec<u8> {
    let mut out = element(EBML, &string_el(DOC_TYPE, doctype));
    out.extend(element(SEGMENT, segment));
    out
}

fn segment_with(duration: Option<f64>, tracks_body: &[u8]) -> Vec<u8> {
    let mut out = info(1_000_000, duration);
    out.extend(tracks(tracks_body));
    out
}

fn info(scale_ns: u64, duration: Option<f64>) -> Vec<u8> {
    let mut body = uint_el(TIMESTAMP_SCALE, &uint_bytes(scale_ns));
    if let Some(duration) = duration {
        body.extend(element(DURATION, &duration.to_be_bytes()));
    }
    element(INFO, &body)
}

fn tracks(entries: &[u8]) -> Vec<u8> {
    element(TRACKS, entries)
}

fn video_track() -> Vec<u8> {
    track_entry(
        1,
        1,
        "V_MPEG4/ISO/AVC",
        &[0x01, 0x42, 0x00],
        &video_settings(320, 180),
    )
}

fn video_settings(width: u64, height: u64) -> Vec<u8> {
    let mut body = uint_el(&[0xB0], &uint_bytes(width));
    body.extend(uint_el(&[0xBA], &uint_bytes(height)));
    element(&[0xE0], &body)
}

fn track_entry(number: u64, track_type: u64, codec: &str, private: &[u8], extra: &[u8]) -> Vec<u8> {
    let mut body = uint_el(&[0xD7], &uint_bytes(number));
    body.extend(uint_el(&[0x83], &uint_bytes(track_type)));
    body.extend(string_el(&[0x86], codec));
    if !private.is_empty() {
        body.extend(element(&[0x63, 0xA2], private));
    }
    body.extend_from_slice(extra);
    element(TRACK_ENTRY, &body)
}

fn string_el(id: &[u8], value: &str) -> Vec<u8> {
    element(id, value.as_bytes())
}

fn uint_el(id: &[u8], bytes: &[u8]) -> Vec<u8> {
    element(id, bytes)
}

fn uint_bytes(value: u64) -> Vec<u8> {
    if value == 0 {
        return vec![0];
    }
    let raw = value.to_be_bytes();
    let start = raw.iter().position(|byte| *byte != 0).expect("non-zero");
    raw[start..].to_vec()
}

fn element(id: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(id);
    out.extend(vint(u64::try_from(body.len()).expect("length")));
    out.extend_from_slice(body);
    out
}

fn vint(value: u64) -> Vec<u8> {
    if value < 0x7F {
        return vec![0x80 | u8::try_from(value).expect("one byte")];
    }
    if value < 0x3FFF {
        let coded = 0x4000 + value;
        return vec![
            u8::try_from(coded >> 8).expect("high"),
            u8::try_from(coded & 0xFF).expect("low"),
        ];
    }
    if value < 0x1F_FFFF {
        let coded = 0x20_0000 + value;
        return vec![
            u8::try_from(coded >> 16).expect("b0"),
            u8::try_from((coded >> 8) & 0xFF).expect("b1"),
            u8::try_from(coded & 0xFF).expect("b2"),
        ];
    }
    panic!("test value needs a wider vint");
}
