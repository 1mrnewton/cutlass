//! MKV and WebM files built in the test, with no external tool.

use std::num::NonZeroU32;

use cutlass_core::{AudioCodec, Timestamp, VideoCodec};
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

#[test]
fn a_simple_block_keeps_media_time() {
    let mut blocks = simple_block(1, 0, 0x80, &[1, 2, 3]);
    blocks.extend(simple_block(1, 33, 0x00, &[4]));
    let mut segment = tracks(&video_with_duration(33_000_000));
    segment.extend(cluster(1_000, &blocks));
    let movie = Movie::parse(&file("webm", &segment)).expect("blocks");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].data, [1, 2, 3]);
    assert_eq!(samples[0].pts, ticks(1_000));
    assert_eq!(samples[0].dts, None);
    assert_eq!(samples[0].duration, Some(ticks(33)));
    assert!(samples[0].keyframe);
    assert_eq!(samples[1].data, [4]);
    assert_eq!(samples[1].pts, ticks(1_033));
    assert_eq!(samples[1].dts, None);
    assert_eq!(samples[1].duration, Some(ticks(33)));
    assert!(!samples[1].keyframe);
}

#[test]
fn a_block_group_keyframe_follows_the_reference() {
    let mut blocks = block_group(1, -20, &[9], Some(40), false);
    blocks.extend(block_group(1, 20, &[8], Some(40), true));
    let entry = track_entry(1, 1, "V_VP8", &[], &video_settings(16, 16));
    let mut segment = tracks(&entry);
    segment.extend(cluster(100, &blocks));
    let movie = Movie::parse(&file("webm", &segment)).expect("group");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].pts, ticks(80));
    assert_eq!(samples[0].duration, Some(ticks(40)));
    assert_eq!(samples[0].dts, None);
    assert!(samples[0].keyframe);
    assert_eq!(samples[1].data, [8]);
    assert_eq!(samples[1].pts, ticks(120));
    assert!(!samples[1].keyframe);
}

#[test]
fn an_unlaced_block_prefers_the_group_duration() {
    let mut extra = default_duration(10_000_000);
    extra.extend(video_settings(8, 8));
    let entry = track_entry(1, 1, "V_VP8", &[], &extra);
    let mut segment = tracks(&entry);
    segment.extend(cluster(0, &block_group(1, 0, &[1], Some(40), false)));
    let movie = Movie::parse(&file("webm", &segment)).expect("duration");
    assert_eq!(movie.tracks()[0].samples()[0].duration, Some(ticks(40)));
}

#[test]
fn an_audio_block_is_a_keyframe() {
    let entry = track_entry(1, 2, "A_OPUS", &[], &default_duration(20_000_000));
    let mut segment = tracks(&entry);
    segment.extend(cluster(0, &simple_block(1, 0, 0x00, &[1, 2])));
    let movie = Movie::parse(&file("webm", &segment)).expect("audio");
    let sample = &movie.tracks()[0].samples()[0];
    assert!(sample.keyframe);
    assert_eq!(sample.pts, ticks(0));
    assert_eq!(sample.duration, Some(ticks(20)));
}

#[test]
fn a_sample_with_no_duration_keeps_none() {
    let entry = track_entry(1, 1, "V_VP8", &[], &video_settings(8, 8));
    let mut segment = tracks(&entry);
    segment.extend(cluster(0, &simple_block(1, 0, 0x80, &[1])));
    let movie = Movie::parse(&file("webm", &segment)).expect("undetermined");
    let sample = &movie.tracks()[0].samples()[0];
    assert_eq!(sample.duration, None);
    assert_eq!(sample.dts, None);
    assert!(sample.keyframe);
}

#[test]
fn an_invisible_or_discardable_frame_is_still_returned() {
    let entry = track_entry(1, 1, "V_VP8", &[], &video_settings(8, 8));
    let mut segment = tracks(&entry);
    segment.extend(cluster(0, &simple_block(1, 0, 0x89, &[5])));
    let movie = Movie::parse(&file("webm", &segment)).expect("flags");
    assert_eq!(movie.tracks()[0].samples()[0].data, [5]);
    assert!(movie.tracks()[0].samples()[0].keyframe);
}

#[test]
fn a_block_group_with_two_blocks_refuses_the_file() {
    let mut block = vint(1);
    block.extend(0i16.to_be_bytes());
    block.push(0x80);
    block.push(1);
    let mut group = element(BLOCK, &block);
    group.extend(element(BLOCK, &block));
    let entry = track_entry(1, 1, "V_VP8", &[], &video_settings(8, 8));
    let mut segment = tracks(&entry);
    segment.extend(cluster(0, &element(BLOCK_GROUP, &group)));
    expect_err(
        Movie::parse(&file("webm", &segment)),
        Error::Inconsistent("a block group has two blocks"),
    );
}

#[test]
fn xiph_lacing_splits_the_block() {
    let block = simple_block(1, 0, 0x82, &[1, 2, 0xAA, 0xBB, 0xCC]);
    let mut extra = default_duration(10_000_000);
    extra.extend(video_settings(16, 16));
    let mut segment = tracks(&track_entry(1, 1, "V_VP8", &[], &extra));
    segment.extend(cluster(50, &block));
    let movie = Movie::parse(&file("webm", &segment)).expect("xiph");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].data, [0xAA, 0xBB]);
    assert_eq!(samples[1].data, [0xCC]);
    assert_eq!(samples[0].pts, ticks(50));
    assert_eq!(samples[1].pts, ticks(60));
    assert!(samples[0].keyframe);
    assert!(!samples[1].keyframe);
    assert_eq!(samples[0].duration, Some(ticks(10)));
}

#[test]
fn fixed_lacing_splits_the_block() {
    let block = simple_block(1, 0, 0x84, &[1, 1, 2, 3, 4]);
    let mut extra = default_duration(10_000_000);
    extra.extend(video_settings(16, 16));
    let mut segment = tracks(&track_entry(1, 1, "V_VP9", &[], &extra));
    segment.extend(cluster(0, &block));
    let movie = Movie::parse(&file("webm", &segment)).expect("fixed");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].data, [1, 2]);
    assert_eq!(samples[1].data, [3, 4]);
    assert_eq!(samples[1].pts, ticks(10));
}

#[test]
fn ebml_lacing_splits_the_block() {
    // Sizes 2, then +1, then the remainder. The difference byte 0xC0 is +1.
    let block = simple_block(1, 0, 0x86, &[2, 0x82, 0xC0, 1, 2, 3, 4, 5, 6]);
    let mut extra = default_duration(10_000_000);
    extra.extend(video_settings(16, 16));
    let mut segment = tracks(&track_entry(1, 1, "V_VP8", &[], &extra));
    segment.extend(cluster(0, &block));
    let movie = Movie::parse(&file("webm", &segment)).expect("ebml");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0].data, [1, 2]);
    assert_eq!(samples[1].data, [3, 4, 5]);
    assert_eq!(samples[2].data, [6]);
    assert_eq!(samples[2].pts, ticks(20));
    assert!(!samples[2].keyframe);
}

#[test]
fn a_laced_block_without_a_default_duration_refuses_the_file() {
    let entry = track_entry(1, 1, "V_VP8", &[], &video_settings(8, 8));
    let mut segment = tracks(&entry);
    segment.extend(cluster(0, &simple_block(1, 0, 0x02, &[1, 2, 1, 2, 3])));
    expect_err(
        Movie::parse(&file("webm", &segment)),
        Error::Inconsistent("a laced block has no default duration"),
    );
}

#[test]
fn a_block_duration_that_does_not_match_the_lace_refuses_the_file() {
    let mut block = vint(1);
    block.extend(0i16.to_be_bytes());
    block.push(0x02);
    block.extend_from_slice(&[1, 2, 0xAA, 0xBB, 0xCC]);
    let mut group = element(BLOCK, &block);
    group.extend(uint_el(BLOCK_DURATION, &uint_bytes(15)));
    let mut extra = default_duration(10_000_000);
    extra.extend(video_settings(8, 8));
    let mut segment = tracks(&track_entry(1, 1, "V_VP8", &[], &extra));
    segment.extend(cluster(0, &element(BLOCK_GROUP, &group)));
    expect_err(
        Movie::parse(&file("webm", &segment)),
        Error::Inconsistent("the block duration does not match the lace"),
    );
}

#[test]
fn a_block_for_an_unknown_track_refuses_the_file() {
    let entry = track_entry(1, 1, "V_VP8", &[], &video_settings(8, 8));
    let mut segment = tracks(&entry);
    segment.extend(cluster(0, &simple_block(9, 0, 0x80, &[1])));
    expect_err(
        Movie::parse(&file("webm", &segment)),
        Error::Inconsistent("a block names an unknown track"),
    );
}

#[test]
fn a_default_duration_that_is_not_a_whole_number_of_ticks_refuses_the_file() {
    let entry = video_with_duration(1_500_000);
    expect_err(
        Movie::parse(&file("matroska", &tracks(&entry))),
        Error::Inconsistent("the default duration is not a whole number of timestamp units"),
    );
}

#[test]
fn a_cluster_before_the_track_list_still_reads() {
    let mut segment = cluster(5, &simple_block(1, 0, 0x80, &[7]));
    segment.extend(tracks(&video_with_duration(1_000_000)));
    let movie = Movie::parse(&file("webm", &segment)).expect("reordered");
    let sample = &movie.tracks()[0].samples()[0];
    assert_eq!(sample.data, [7]);
    assert_eq!(sample.pts, ticks(5));
    assert_eq!(sample.duration, Some(ticks(1)));
}

#[test]
fn a_cluster_without_a_timestamp_refuses_the_file() {
    let entry = track_entry(1, 1, "V_VP8", &[], &video_settings(8, 8));
    let mut segment = tracks(&entry);
    segment.extend(element(CLUSTER, &simple_block(1, 0, 0x80, &[1])));
    expect_err(
        Movie::parse(&file("webm", &segment)),
        Error::Inconsistent("a cluster has no timestamp"),
    );
}

#[test]
fn an_unknown_size_cluster_reads_to_the_end_of_the_segment() {
    let mut cluster_body = uint_el(CLUSTER_TIME, &uint_bytes(4));
    cluster_body.extend(simple_block(1, 1, 0x80, &[9]));
    let mut segment = tracks(&video_with_duration(1_000_000));
    segment.extend(element_unknown(CLUSTER, &cluster_body));
    let movie = Movie::parse(&file("matroska", &segment)).expect("unknown size");
    let sample = &movie.tracks()[0].samples()[0];
    assert_eq!(sample.data, [9]);
    assert_eq!(sample.pts, ticks(5));
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
const CLUSTER: &[u8] = &[0x1F, 0x43, 0xB6, 0x75];
const CLUSTER_TIME: &[u8] = &[0xE7];
const SIMPLE_BLOCK: &[u8] = &[0xA3];
const BLOCK_GROUP: &[u8] = &[0xA0];
const BLOCK: &[u8] = &[0xA1];
const BLOCK_DURATION: &[u8] = &[0x9B];
const REFERENCE_BLOCK: &[u8] = &[0xFB];

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

fn video_with_duration(ns: u64) -> Vec<u8> {
    let mut extra = default_duration(ns);
    extra.extend(video_settings(320, 180));
    track_entry(1, 1, "V_VP8", &[], &extra)
}

fn default_duration(ns: u64) -> Vec<u8> {
    uint_el(&[0x23, 0xE3, 0x83], &uint_bytes(ns))
}

fn simple_block(track: u64, relative: i16, flags: u8, data: &[u8]) -> Vec<u8> {
    let mut body = vint(track);
    body.extend(relative.to_be_bytes());
    body.push(flags);
    body.extend_from_slice(data);
    element(SIMPLE_BLOCK, &body)
}

fn cluster(timestamp: u64, blocks: &[u8]) -> Vec<u8> {
    let mut body = uint_el(CLUSTER_TIME, &uint_bytes(timestamp));
    body.extend_from_slice(blocks);
    element(CLUSTER, &body)
}

fn block_group(
    track: u64,
    relative: i16,
    data: &[u8],
    duration: Option<u64>,
    reference: bool,
) -> Vec<u8> {
    let mut block = vint(track);
    block.extend(relative.to_be_bytes());
    block.push(0);
    block.extend_from_slice(data);
    let mut body = element(BLOCK, &block);
    if let Some(duration) = duration {
        body.extend(uint_el(BLOCK_DURATION, &uint_bytes(duration)));
    }
    if reference {
        body.extend(element(REFERENCE_BLOCK, &[0xFF]));
    }
    element(BLOCK_GROUP, &body)
}

fn element_unknown(id: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.push(0xFF);
    out.extend_from_slice(body);
    out
}

fn ticks(value: i64) -> Timestamp {
    Timestamp::new(value, scale(1_000))
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
