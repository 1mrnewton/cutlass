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

#[test]
fn a_header_without_a_movie_has_no_samples() {
    let body = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    let movie = Movie::parse(&avi(&body)).expect("header");
    assert!(movie.tracks()[0].samples().is_empty());
}

#[test]
fn two_frames_keep_stream_time() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    let mut frames = frame(0, b"dc", &[1, 2, 3]);
    frames.extend(frame(0, b"dc", &[4]));
    let movie = Movie::parse(&avi(&with_movie(&header, &frames, None))).expect("frames");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].data, [1, 2, 3]);
    assert_eq!(samples[0].pts.value, 0);
    assert_eq!(samples[0].pts.timescale, scale(25));
    assert!(samples[0].dts.is_none());
    assert_eq!(samples[0].duration.map(|span| span.value), Some(1));
    assert!(!samples[0].keyframe);
    assert_eq!(samples[1].data, [4]);
    assert_eq!(samples[1].pts.value, 1);
    assert_eq!(samples[1].duration.map(|span| span.value), Some(1));
    assert!(samples[1].dts.is_none());
    assert!(!samples[1].keyframe);
}

#[test]
fn an_uncompressed_frame_is_a_keyframe_without_an_index() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    let movie = Movie::parse(&avi(&with_movie(&header, &frame(0, b"db", &[7]), None))).expect("db");
    let sample = &movie.tracks()[0].samples()[0];
    assert!(sample.keyframe);
    assert_eq!(sample.data, [7]);
}

#[test]
fn an_index_keyframe_bit_marks_a_compressed_frame() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    let mut frames = frame(0, b"dc", &[1, 2, 3]);
    frames.extend(frame(0, b"dc", &[4]));
    let mut index = index_entry(*b"00dc", 0x10 | 0x100, 0xFFFF_FFFF, 3).to_vec();
    index.extend(index_entry(*b"00dc", 0, 0, 1));
    let movie = Movie::parse(&avi(&with_movie(&header, &frames, Some(&index)))).expect("index");
    let samples = movie.tracks()[0].samples();
    assert!(samples[0].keyframe);
    assert!(!samples[1].keyframe);
    assert_eq!(samples[1].pts.value, 1);
}

#[test]
fn audio_bytes_are_samples() {
    let header = hdrl(1, &audio_sized(1, 2, 48_000, 4, &[]));
    let mut chunks = frame(0, b"wb", &[1, 2, 3, 4, 5, 6, 7, 8]);
    chunks.extend(frame(0, b"wb", &[9, 10, 11, 12]));
    let movie = Movie::parse(&avi(&with_movie(&header, &chunks, None))).expect("pcm");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].pts.value, 0);
    assert_eq!(samples[0].duration.map(|span| span.value), Some(2));
    assert_eq!(samples[0].pts.timescale, scale(48_000));
    assert!(samples[0].keyframe);
    assert!(samples[0].dts.is_none());
    assert_eq!(samples[1].pts.value, 2);
    assert_eq!(samples[1].duration.map(|span| span.value), Some(1));
    assert_eq!(samples[1].data, [9, 10, 11, 12]);
}

#[test]
fn an_index_does_not_clear_an_audio_keyframe() {
    let header = hdrl(1, &audio_sized(1, 1, 8_000, 1, &[]));
    let index = index_entry(*b"00wb", 0, 0, 1);
    let movie = Movie::parse(&avi(&with_movie(
        &header,
        &frame(0, b"wb", &[1]),
        Some(&index),
    )))
    .expect("flag");
    assert!(movie.tracks()[0].samples()[0].keyframe);
}

#[test]
fn dw_start_shifts_the_first_sample() {
    let mut body = stream_header(b"vids", 1, 25, 2, 0);
    body.extend(bitmap(4, 4, b"MJPG", &[]));
    let header = hdrl(1, &list(b"strl", &body));
    let movie =
        Movie::parse(&avi(&with_movie(&header, &frame(0, b"dc", &[1]), None))).expect("start");
    assert_eq!(movie.tracks()[0].samples()[0].pts.value, 2);
}

#[test]
fn a_reduced_clock_keeps_one_tick_per_frame() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1000, 30_000, &[]));
    let movie =
        Movie::parse(&avi(&with_movie(&header, &frame(0, b"db", &[1]), None))).expect("gcd");
    let sample = &movie.tracks()[0].samples()[0];
    assert_eq!(movie.tracks()[0].timescale(), scale(30));
    assert_eq!(sample.duration.map(|span| span.value), Some(1));
    assert_eq!(sample.pts.value, 0);
}

#[test]
fn an_odd_sized_chunk_stays_a_sample() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    let mut frames = frame(0, b"dc", &[9]);
    frames.extend(frame(0, b"dc", &[8, 7]));
    let movie = Movie::parse(&avi(&with_movie(&header, &frames, None))).expect("pad");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].data, [9]);
    assert_eq!(samples[1].data, [8, 7]);
    assert_eq!(samples[1].pts.value, 1);
}

#[test]
fn a_record_list_is_read() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    let frames = list(b"rec ", &frame(0, b"dc", &[3]));
    let movie = Movie::parse(&avi(&with_movie(&header, &frames, None))).expect("rec");
    assert_eq!(movie.tracks()[0].samples()[0].data, [3]);
}

#[test]
fn four_nested_record_lists_are_read() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    let movie = Movie::parse(&avi(&with_movie(
        &header,
        &nest_rec(&frame(0, b"dc", &[3]), 4),
        None,
    )))
    .expect("depth");
    assert_eq!(movie.tracks()[0].samples()[0].data, [3]);
}

#[test]
fn five_nested_record_lists_refuse_the_file() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    expect_err(
        Movie::parse(&avi(&with_movie(
            &header,
            &nest_rec(&frame(0, b"dc", &[3]), 5),
            None,
        ))),
        Error::Inconsistent("the record list is nested too deep"),
    );
}

#[test]
fn a_chunk_size_that_does_not_divide_refuses_the_file() {
    let header = hdrl(1, &audio_sized(1, 1, 8_000, 4, &[]));
    expect_err(
        Movie::parse(&avi(&with_movie(
            &header,
            &frame(0, b"wb", &[1, 2, 3]),
            None,
        ))),
        Error::Inconsistent("a chunk size is not a multiple of the sample size"),
    );
}

#[test]
fn an_index_that_does_not_match_refuses_the_file() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    let index = index_entry(*b"00dc", 0x10, 0, 9);
    expect_err(
        Movie::parse(&avi(&with_movie(
            &header,
            &frame(0, b"dc", &[1]),
            Some(&index),
        ))),
        Error::Inconsistent("the index does not match the chunks"),
    );
}

#[test]
fn an_index_of_the_wrong_length_refuses_the_file() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    expect_err(
        Movie::parse(&avi(&with_movie(
            &header,
            &frame(0, b"dc", &[1]),
            Some(&[0; 8]),
        ))),
        Error::Inconsistent("the index is the wrong size"),
    );
}

#[test]
fn a_chunk_that_names_an_unknown_stream_refuses_the_file() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    expect_err(
        Movie::parse(&avi(&with_movie(&header, &frame(1, b"dc", &[1]), None))),
        Error::Inconsistent("a chunk names an unknown stream"),
    );
}

#[test]
fn interleaved_streams_keep_their_own_samples() {
    let mut streams = video_stream(b"MJPG", 4, 4, 1, 10, &[]);
    streams.extend(audio_sized(1, 1, 1_000, 2, &[]));
    let header = hdrl(2, &streams);
    let mut chunks = frame(0, b"dc", &[9]);
    chunks.extend(frame(1, b"wb", &[1, 2, 3, 4]));
    chunks.extend(frame(0, b"dc", &[8]));
    let movie = Movie::parse(&avi(&with_movie(&header, &chunks, None))).expect("interleaved");
    let video = movie.tracks()[0].samples();
    let audio = movie.tracks()[1].samples();
    assert_eq!(video[0].data, [9]);
    assert_eq!(video[0].pts.value, 0);
    assert_eq!(video[1].data, [8]);
    assert_eq!(video[1].pts.value, 1);
    assert_eq!(audio[0].data, [1, 2, 3, 4]);
    assert_eq!(audio[0].pts.value, 0);
    assert_eq!(audio[0].duration.map(|span| span.value), Some(2));
    assert_eq!(audio[0].pts.timescale, scale(1_000));
}

#[test]
fn a_palette_chunk_is_skipped() {
    let header = hdrl(1, &video_stream(b"MJPG", 4, 4, 1, 25, &[]));
    let mut chunks = chunk(b"00pc", &[1]);
    chunks.extend(frame(0, b"dc", &[2, 3]));
    let mut index = index_entry(*b"00pc", 0, 0, 1).to_vec();
    index.extend(index_entry(*b"00dc", 0x10, 99, 2));
    let movie = Movie::parse(&avi(&with_movie(&header, &chunks, Some(&index)))).expect("palette");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].data, [2, 3]);
    assert!(samples[0].keyframe);
}

#[test]
fn a_second_movie_list_refuses_the_file() {
    let mut body = with_movie(&hdrl(0, &[]), &[], None);
    body.extend(list(b"movi", &[]));
    expect_err(
        Movie::parse(&avi(&body)),
        Error::Inconsistent("the movie list is repeated"),
    );
}

#[test]
fn a_second_index_refuses_the_file() {
    let mut body = with_movie(&hdrl(0, &[]), &[], Some(&[]));
    body.extend(chunk(b"idx1", &[]));
    expect_err(
        Movie::parse(&avi(&body)),
        Error::Inconsistent("the index is repeated"),
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
    stream_header(kind, scale, rate, 0, 0)
}

fn stream_header(kind: &[u8; 4], scale: u32, rate: u32, start: u32, sample_size: u32) -> Vec<u8> {
    let mut data = vec![0; 48];
    data[..4].copy_from_slice(kind);
    data[20..24].copy_from_slice(&scale.to_le_bytes());
    data[24..28].copy_from_slice(&rate.to_le_bytes());
    data[28..32].copy_from_slice(&start.to_le_bytes());
    data[44..48].copy_from_slice(&sample_size.to_le_bytes());
    chunk(b"strh", &data)
}

fn audio_sized(tag: u16, channels: u16, rate: u32, sample_size: u32, extra: &[u8]) -> Vec<u8> {
    let mut body = stream_header(b"auds", 1, rate, 0, sample_size);
    body.extend(wave(tag, channels, rate, extra));
    list(b"strl", &body)
}

fn with_movie(header: &[u8], frames: &[u8], index: Option<&[u8]>) -> Vec<u8> {
    let mut body = header.to_vec();
    body.extend(list(b"movi", frames));
    if let Some(index) = index {
        body.extend(chunk(b"idx1", index));
    }
    body
}

fn frame(stream: u8, suffix: &[u8; 2], data: &[u8]) -> Vec<u8> {
    let id = [b'0', b'0' + stream, suffix[0], suffix[1]];
    chunk(&id, data)
}

fn index_entry(id: [u8; 4], flags: u32, offset: u32, size: u32) -> [u8; 16] {
    let mut out = [0; 16];
    out[..4].copy_from_slice(&id);
    out[4..8].copy_from_slice(&flags.to_le_bytes());
    out[8..12].copy_from_slice(&offset.to_le_bytes());
    out[12..16].copy_from_slice(&size.to_le_bytes());
    out
}

fn nest_rec(body: &[u8], depth: u32) -> Vec<u8> {
    let mut data = body.to_vec();
    for _ in 0..depth {
        data = list(b"rec ", &data);
    }
    data
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
