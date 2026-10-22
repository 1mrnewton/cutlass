//! MP3 files built in the test, with no external tool.

use std::num::NonZeroU32;

use cutlass_core::AudioCodec;
use cutlass_mp3_file::{Error, Movie};

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
fn two_frames_keep_stream_time() {
    let mut file = frame([0xFF, 0xFB, 0x90, 0x00], 417);
    file.extend(frame([0xFF, 0xFB, 0x90, 0x00], 417));
    let movie = Movie::parse(&file).expect("frames");
    let track = &movie.tracks()[0];
    assert_eq!(track.timescale(), scale(44_100));
    assert_eq!(track.config().codec, AudioCodec::Mp3);
    assert_eq!(track.config().sample_rate, 44_100);
    assert_eq!(track.config().channels, 2);
    assert!(track.config().config_record.is_empty());
    let samples = track.samples();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].data.len(), 417);
    assert_eq!(&samples[0].data[..4], &[0xFF, 0xFB, 0x90, 0x00]);
    assert_eq!(samples[0].pts.value, 0);
    assert_eq!(samples[0].pts.timescale, scale(44_100));
    assert!(samples[0].dts.is_none());
    assert_eq!(samples[0].duration.map(|span| span.value), Some(1152));
    assert!(samples[0].keyframe);
    assert_eq!(samples[1].pts.value, 1152);
    assert_eq!(samples[1].duration.map(|span| span.value), Some(1152));
}

#[test]
fn a_bitrate_change_stays_on_the_same_clock() {
    let mut file = frame([0xFF, 0xFB, 0x90, 0x00], 417);
    file.extend(frame([0xFF, 0xFB, 0xA0, 0x00], 522));
    let movie = Movie::parse(&file).expect("vbr");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].data.len(), 417);
    assert_eq!(samples[1].data.len(), 522);
    assert_eq!(samples[1].pts.value, 1152);
}

#[test]
fn padding_adds_one_byte() {
    let movie = Movie::parse(&frame([0xFF, 0xFB, 0x92, 0x00], 418)).expect("pad");
    assert_eq!(movie.tracks()[0].samples()[0].data.len(), 418);
}

#[test]
fn mono_is_one_channel() {
    let movie = Movie::parse(&frame([0xFF, 0xFB, 0x90, 0xC0], 417)).expect("mono");
    assert_eq!(movie.tracks()[0].config().channels, 1);
}

#[test]
fn mpeg2_layer3_lasts_576_samples() {
    let movie = Movie::parse(&frame([0xFF, 0xF3, 0x80, 0x00], 208)).expect("mpeg2");
    let track = &movie.tracks()[0];
    assert_eq!(track.timescale(), scale(22_050));
    assert_eq!(
        track.samples()[0].duration.map(|span| span.value),
        Some(576)
    );
}

#[test]
fn mpeg2_layer2_lasts_1152_samples() {
    let movie = Movie::parse(&frame([0xFF, 0xF5, 0x80, 0x00], 417)).expect("layer 2");
    assert_eq!(
        movie.tracks()[0].samples()[0]
            .duration
            .map(|span| span.value),
        Some(1152)
    );
}

#[test]
fn layer1_lasts_384_samples() {
    let movie = Movie::parse(&frame([0xFF, 0xFF, 0x40, 0x00], 136)).expect("layer 1");
    assert_eq!(
        movie.tracks()[0].samples()[0]
            .duration
            .map(|span| span.value),
        Some(384)
    );
}

#[test]
fn an_id3v2_tag_is_skipped() {
    let mut file = id3(&[0xAB, 0xCD], false);
    file.extend(frame([0xFF, 0xFB, 0x90, 0x00], 417));
    let movie = Movie::parse(&file).expect("id3");
    assert_eq!(movie.tracks()[0].samples().len(), 1);
    assert_eq!(movie.tracks()[0].samples()[0].data.len(), 417);
}

#[test]
fn an_id3v2_footer_is_skipped() {
    let mut file = id3(&[], true);
    file.extend(frame([0xFF, 0xFB, 0x90, 0x00], 417));
    let movie = Movie::parse(&file).expect("footer");
    assert_eq!(movie.tracks()[0].samples().len(), 1);
}

#[test]
fn an_id3v1_tag_is_skipped() {
    let mut file = frame([0xFF, 0xFB, 0x90, 0x00], 417);
    file.extend(b"TAG");
    file.extend(vec![0; 125]);
    let movie = Movie::parse(&file).expect("tag");
    assert_eq!(movie.tracks()[0].samples().len(), 1);
    assert_eq!(movie.tracks()[0].samples()[0].data.len(), 417);
}

#[test]
fn a_frame_checksum_is_not_checked() {
    let movie = Movie::parse(&frame([0xFF, 0xFA, 0x90, 0x00], 417)).expect("crc");
    assert_eq!(movie.tracks()[0].samples()[0].data[1], 0xFA);
}

#[test]
fn a_bad_form_refuses_the_file() {
    expect_err(Movie::parse(b"RIFF"), Error::Form);
    expect_err(
        Movie::parse(&[]),
        Error::Inconsistent("the file has no frames"),
    );
}

#[test]
fn a_short_file_is_truncated() {
    assert_eq!(
        Movie::parse(&[0xFF, 0xFB]),
        Err(Error::Truncated { at: 0, need: 4 })
    );
}

#[test]
fn a_free_bitrate_refuses_the_file() {
    expect_err(
        Movie::parse(&[0xFF, 0xFB, 0x00, 0x00]),
        Error::Inconsistent("a frame has no bitrate"),
    );
}

#[test]
fn a_reserved_version_refuses_the_file() {
    expect_err(
        Movie::parse(&[0xFF, 0xEB, 0x90, 0x00]),
        Error::Inconsistent("the frame header is not audio"),
    );
}

#[test]
fn a_frame_that_does_not_match_refuses_the_file() {
    let mut file = frame([0xFF, 0xFB, 0x90, 0x00], 417);
    file.extend(frame([0xFF, 0xFB, 0x94, 0x00], 384));
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("a frame does not match the stream"),
    );
}

#[test]
fn a_truncated_frame_refuses_the_file() {
    let mut file = frame([0xFF, 0xFB, 0x90, 0x00], 10);
    file.truncate(10);
    assert_eq!(
        Movie::parse(&file),
        Err(Error::Truncated { at: 0, need: 417 })
    );
}

#[test]
fn bytes_after_the_last_frame_refuse_the_file() {
    let mut file = frame([0xFF, 0xFB, 0x90, 0x00], 417);
    file.push(0);
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the file has data after the last frame"),
    );
}

#[test]
fn a_tag_size_that_is_not_synchsafe_refuses_the_file() {
    let mut file = b"ID3\x03\x00\x00".to_vec();
    file.extend([0x80, 0, 0, 0]);
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the tag size is not synchsafe"),
    );
}

fn frame(header: [u8; 4], len: usize) -> Vec<u8> {
    let mut out = vec![0; len];
    out[..4].copy_from_slice(&header);
    out
}

fn id3(body: &[u8], footer: bool) -> Vec<u8> {
    let mut out = b"ID3\x03\x00".to_vec();
    out.push(if footer { 0x10 } else { 0 });
    let size = u32::try_from(body.len()).expect("size");
    out.push(u8::try_from((size >> 21) & 0x7F).expect("size"));
    out.push(u8::try_from((size >> 14) & 0x7F).expect("size"));
    out.push(u8::try_from((size >> 7) & 0x7F).expect("size"));
    out.push(u8::try_from(size & 0x7F).expect("size"));
    out.extend_from_slice(body);
    if footer {
        out.extend([0; 10]);
    }
    out
}
