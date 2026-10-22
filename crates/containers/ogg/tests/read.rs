//! Ogg files built in the test, with no external tool.

use std::num::NonZeroU32;

use cutlass_core::AudioCodec;
use cutlass_ogg::{Error, Movie};

const BOS: u8 = 0x02;
const CONTINUED: u8 = 0x01;
const EOS: u8 = 0x04;

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

fn vorbis_id(channels: u8, rate: u32) -> Vec<u8> {
    let mut packet = Vec::with_capacity(16);
    packet.push(1);
    packet.extend_from_slice(b"vorbis");
    packet.extend_from_slice(&0u32.to_le_bytes());
    packet.push(channels);
    packet.extend_from_slice(&rate.to_le_bytes());
    packet
}

fn opus_head(channels: u8, pre_skip: u16, input_rate: u32) -> Vec<u8> {
    let mut packet = Vec::with_capacity(19);
    packet.extend_from_slice(b"OpusHead");
    packet.push(1);
    packet.push(channels);
    packet.extend_from_slice(&pre_skip.to_le_bytes());
    packet.extend_from_slice(&input_rate.to_le_bytes());
    packet.extend_from_slice(&0i16.to_le_bytes());
    packet.push(0);
    packet
}

fn ogg_page(serial: u32, seq: u32, flags: u8, granule: i64, packets: &[&[u8]]) -> Vec<u8> {
    let mut table = Vec::new();
    let mut payload = Vec::new();
    for packet in packets {
        lace(packet, &mut table, &mut payload);
    }
    raw_page(serial, seq, flags, granule, &table, &payload)
}

fn lace(packet: &[u8], table: &mut Vec<u8>, payload: &mut Vec<u8>) {
    let mut rest = packet;
    while rest.len() >= 255 {
        table.push(255);
        payload.extend_from_slice(&rest[..255]);
        rest = &rest[255..];
    }
    let size = u8::try_from(rest.len()).expect("a segment is below 255");
    table.push(size);
    payload.extend_from_slice(rest);
}

fn raw_page(
    serial: u32,
    seq: u32,
    flags: u8,
    granule: i64,
    table: &[u8],
    payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(27 + table.len() + payload.len());
    out.extend_from_slice(b"OggS");
    out.push(0);
    out.push(flags);
    out.extend_from_slice(&granule.to_le_bytes());
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&0x11u32.to_le_bytes());
    out.push(u8::try_from(table.len()).expect("a page has at most 255 segments"));
    out.extend_from_slice(table);
    out.extend_from_slice(payload);
    out
}

fn join(pages: &[Vec<u8>]) -> Vec<u8> {
    let mut file = Vec::new();
    for page in pages {
        file.extend_from_slice(page);
    }
    file
}

#[test]
fn a_vorbis_stream_keeps_its_rate_and_one_packet() {
    let id = vorbis_id(2, 44_100);
    let file = join(&[
        ogg_page(7, 0, BOS, -1, &[&id]),
        ogg_page(7, 1, 0, 100, &[&[9, 8, 7]]),
    ]);
    let movie = Movie::parse(&file).expect("vorbis");
    let track = &movie.tracks()[0];
    assert_eq!(track.serial(), 7);
    assert_eq!(track.timescale(), scale(44_100));
    assert_eq!(track.config().codec, AudioCodec::Vorbis);
    assert_eq!(track.config().sample_rate, 44_100);
    assert_eq!(track.config().channels, 2);
    assert_eq!(track.config().config_record, id);
    let samples = track.samples();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].data, [9, 8, 7]);
    assert_eq!(samples[0].pts.value, 0);
    assert_eq!(samples[0].pts.timescale, scale(44_100));
    assert!(samples[0].dts.is_none());
    assert_eq!(samples[0].duration.map(|span| span.value), Some(100));
    assert!(samples[0].keyframe);
}

#[test]
fn the_next_page_starts_where_the_granule_ended() {
    let id = vorbis_id(1, 48_000);
    let file = join(&[
        ogg_page(1, 0, BOS, -1, &[&id]),
        ogg_page(1, 1, 0, 100, &[&[1]]),
        ogg_page(1, 2, 0, 250, &[&[2]]),
    ]);
    let movie = Movie::parse(&file).expect("span");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].pts.value, 0);
    assert_eq!(samples[0].duration.map(|span| span.value), Some(100));
    assert_eq!(samples[1].pts.value, 100);
    assert_eq!(samples[1].duration.map(|span| span.value), Some(150));
}

#[test]
fn a_page_does_not_split_its_granule() {
    let id = vorbis_id(1, 8_000);
    let file = join(&[
        ogg_page(1, 0, BOS, -1, &[&id]),
        ogg_page(1, 1, 0, 50, &[&[0xAA], &[0xBB]]),
    ]);
    let movie = Movie::parse(&file).expect("group");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].data, [0xAA]);
    assert_eq!(samples[0].pts.value, 0);
    assert!(samples[0].duration.is_none());
    assert_eq!(samples[1].data, [0xBB]);
    assert_eq!(samples[1].pts.value, 0);
    assert_eq!(samples[1].duration.map(|span| span.value), Some(50));
}

#[test]
fn a_negative_granule_does_not_move_the_clock() {
    let id = vorbis_id(1, 44_100);
    let mut comment = vec![3];
    comment.extend_from_slice(b"vorbis");
    let file = join(&[
        ogg_page(1, 0, BOS, -1, &[&id]),
        ogg_page(1, 1, 0, -1, &[&comment]),
        ogg_page(1, 2, 0, 80, &[&[7]]),
    ]);
    let movie = Movie::parse(&file).expect("comment");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].data, comment);
    assert_eq!(samples[0].pts.value, 0);
    assert!(samples[0].duration.is_none());
    assert!(samples[0].keyframe);
    assert_eq!(samples[1].data, [7]);
    assert_eq!(samples[1].pts.value, 0);
    assert_eq!(samples[1].duration.map(|span| span.value), Some(80));
}

#[test]
fn a_repeated_granule_has_no_duration() {
    let id = vorbis_id(1, 44_100);
    let file = join(&[
        ogg_page(1, 0, BOS, -1, &[&id]),
        ogg_page(1, 1, 0, 10, &[&[1]]),
        ogg_page(1, 2, 0, 10, &[&[2]]),
    ]);
    let movie = Movie::parse(&file).expect("repeat");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[1].pts.value, 10);
    assert_eq!(samples[1].duration.map(|span| span.value), Some(0));
}

#[test]
fn opus_uses_48_khz_and_does_not_apply_pre_skip() {
    let head = opus_head(1, 312, 44_100);
    let file = join(&[
        ogg_page(3, 0, BOS, -1, &[&head]),
        ogg_page(3, 1, 0, 960, &[&[0xF8, 0xFF, 0xFE]]),
    ]);
    let movie = Movie::parse(&file).expect("opus");
    let track = &movie.tracks()[0];
    assert_eq!(track.serial(), 3);
    assert_eq!(track.timescale(), scale(48_000));
    assert_eq!(track.config().codec, AudioCodec::Opus);
    assert_eq!(track.config().sample_rate, 48_000);
    assert_eq!(track.config().channels, 1);
    assert_eq!(track.config().config_record, head);
    let sample = &track.samples()[0];
    assert_eq!(sample.data, [0xF8, 0xFF, 0xFE]);
    assert_eq!(sample.pts.value, 0);
    assert_eq!(sample.duration.map(|span| span.value), Some(960));
    assert!(sample.dts.is_none());
}

#[test]
fn two_streams_keep_their_own_packets() {
    let vorbis = vorbis_id(2, 44_100);
    let opus = opus_head(1, 0, 48_000);
    let file = join(&[
        ogg_page(1, 0, BOS, -1, &[&vorbis]),
        ogg_page(2, 0, BOS, -1, &[&opus]),
        ogg_page(1, 1, 0, 100, &[&[1]]),
        ogg_page(2, 1, 0, 960, &[&[2]]),
    ]);
    let movie = Movie::parse(&file).expect("two");
    assert_eq!(movie.tracks().len(), 2);
    assert_eq!(movie.tracks()[0].serial(), 1);
    assert_eq!(movie.tracks()[0].config().codec, AudioCodec::Vorbis);
    assert_eq!(movie.tracks()[0].samples()[0].data, [1]);
    assert_eq!(movie.tracks()[1].serial(), 2);
    assert_eq!(movie.tracks()[1].config().codec, AudioCodec::Opus);
    assert_eq!(movie.tracks()[1].timescale(), scale(48_000));
    assert_eq!(movie.tracks()[1].samples()[0].data, [2]);
}

#[test]
fn an_unknown_stream_is_skipped() {
    let id = vorbis_id(1, 22_050);
    let file = join(&[
        ogg_page(9, 0, BOS, -1, &[b"not-audio"]),
        ogg_page(4, 0, BOS, -1, &[&id]),
        ogg_page(4, 1, 0, 8, &[&[5]]),
    ]);
    let movie = Movie::parse(&file).expect("skip");
    assert_eq!(movie.tracks().len(), 1);
    assert_eq!(movie.tracks()[0].serial(), 4);
    assert_eq!(movie.tracks()[0].samples()[0].data, [5]);
}

#[test]
fn a_continued_packet_is_one_sample() {
    let id = vorbis_id(1, 48_000);
    let open = raw_page(1, 1, 0, 100, &[255], &[0xAB; 255]);
    let close = raw_page(1, 2, CONTINUED, 20, &[1], &[0xCD]);
    let file = join(&[ogg_page(1, 0, BOS, -1, &[&id]), open, close]);
    let movie = Movie::parse(&file).expect("lace");
    let sample = &movie.tracks()[0].samples()[0];
    assert_eq!(sample.data.len(), 256);
    assert_eq!(sample.data[0], 0xAB);
    assert_eq!(sample.data[255], 0xCD);
    assert_eq!(sample.pts.value, 0);
    assert_eq!(sample.duration.map(|span| span.value), Some(20));
}

#[test]
fn a_full_segment_is_one_packet() {
    let id = vorbis_id(1, 44_100);
    let audio = vec![0x5A; 255];
    let file = join(&[
        ogg_page(1, 0, BOS, -1, &[&id]),
        ogg_page(1, 1, 0, 255, &[&audio]),
    ]);
    let movie = Movie::parse(&file).expect("full");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].data, audio);
    assert_eq!(samples[0].duration.map(|span| span.value), Some(255));
}

#[test]
fn a_continued_page_without_a_packet_refuses_the_file() {
    let id = vorbis_id(1, 44_100);
    let file = join(&[
        ogg_page(1, 0, BOS, -1, &[&id]),
        ogg_page(1, 1, CONTINUED, 1, &[&[1]]),
    ]);
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("a page continues a packet that has not started"),
    );
}

#[test]
fn an_open_packet_without_the_continued_flag_refuses_the_file() {
    let id = vorbis_id(1, 44_100);
    let open = raw_page(1, 1, 0, -1, &[255], &[0xAB; 255]);
    let close = raw_page(1, 2, 0, 20, &[1], &[0xCD]);
    let file = join(&[ogg_page(1, 0, BOS, -1, &[&id]), open, close]);
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("a continued packet is not continued"),
    );
}

#[test]
fn an_open_packet_at_the_end_refuses_the_file() {
    let id = vorbis_id(1, 44_100);
    let open = raw_page(1, 1, EOS, -1, &[255], &[0xAB; 255]);
    let file = join(&[ogg_page(1, 0, BOS, -1, &[&id]), open]);
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the last packet is not finished"),
    );
}

#[test]
fn a_page_sequence_starts_at_zero() {
    let id = vorbis_id(1, 44_100);
    expect_err(
        Movie::parse(&ogg_page(1, 1, BOS, -1, &[&id])),
        Error::Inconsistent("the page sequence is broken"),
    );
}

#[test]
fn a_page_sequence_increases_by_one() {
    let id = vorbis_id(1, 44_100);
    let file = join(&[
        ogg_page(1, 0, BOS, -1, &[&id]),
        ogg_page(1, 2, 0, 10, &[&[1]]),
    ]);
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the page sequence is broken"),
    );
}

#[test]
fn a_stream_without_a_start_refuses_the_file() {
    let id = vorbis_id(1, 44_100);
    expect_err(
        Movie::parse(&ogg_page(1, 0, 0, -1, &[&id])),
        Error::Inconsistent("a stream has no start"),
    );
}

#[test]
fn a_stream_that_starts_twice_refuses_the_file() {
    let id = vorbis_id(1, 44_100);
    let file = join(&[
        ogg_page(1, 0, BOS, -1, &[&id]),
        ogg_page(1, 1, BOS, -1, &[&id]),
    ]);
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("a stream starts twice"),
    );
}

#[test]
fn an_unsupported_version_refuses_the_file() {
    let id = vorbis_id(1, 44_100);
    let mut page = ogg_page(1, 0, BOS, -1, &[&id]);
    page[4] = 1;
    expect_err(
        Movie::parse(&page),
        Error::Inconsistent("the ogg version is not supported"),
    );
}

#[test]
fn a_granule_that_moves_backwards_refuses_the_file() {
    let id = vorbis_id(1, 44_100);
    let file = join(&[
        ogg_page(1, 0, BOS, -1, &[&id]),
        ogg_page(1, 1, 0, 10, &[&[1]]),
        ogg_page(1, 2, 0, 5, &[&[2]]),
    ]);
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("the granule moves backwards"),
    );
}

#[test]
fn a_bad_form_refuses_the_file() {
    expect_err(Movie::parse(b"nope"), Error::Form);
}

#[test]
fn an_empty_file_has_no_pages() {
    expect_err(
        Movie::parse(&[]),
        Error::Inconsistent("the file has no pages"),
    );
}

#[test]
fn a_short_file_is_truncated() {
    expect_err(Movie::parse(b"Ogg"), Error::Truncated { at: 0, need: 27 });
}

#[test]
fn a_short_page_is_truncated() {
    expect_err(Movie::parse(b"OggS"), Error::Truncated { at: 0, need: 27 });
}

#[test]
fn a_later_page_that_is_not_ogg_refuses_the_file() {
    let id = vorbis_id(1, 44_100);
    let mut file = ogg_page(1, 0, BOS, -1, &[&id]);
    file.extend_from_slice(b"nope");
    expect_err(
        Movie::parse(&file),
        Error::Inconsistent("a page is not ogg"),
    );
}

#[test]
fn a_short_tail_is_truncated() {
    let id = vorbis_id(1, 44_100);
    let page = ogg_page(1, 0, BOS, -1, &[&id]);
    let at = u64::try_from(page.len()).expect("the page fits in u64");
    let mut file = page;
    file.extend_from_slice(&[1, 2, 3]);
    expect_err(Movie::parse(&file), Error::Truncated { at, need: 27 });
}

#[test]
fn a_short_page_payload_is_truncated() {
    let page = raw_page(1, 0, BOS, -1, &[10], &[0; 10]);
    let mut file = page;
    file.truncate(file.len() - 10);
    expect_err(Movie::parse(&file), Error::Truncated { at: 28, need: 10 });
}

#[test]
fn a_zero_vorbis_rate_refuses_the_file() {
    let id = vorbis_id(2, 0);
    expect_err(
        Movie::parse(&ogg_page(1, 0, BOS, -1, &[&id])),
        Error::Inconsistent("the sample rate is zero"),
    );
}

#[test]
fn a_zero_channel_count_refuses_the_file() {
    let id = vorbis_id(0, 44_100);
    expect_err(
        Movie::parse(&ogg_page(1, 0, BOS, -1, &[&id])),
        Error::Inconsistent("the channel count is zero"),
    );
}

#[test]
fn opus_with_no_channels_refuses_the_file() {
    let head = opus_head(0, 0, 48_000);
    expect_err(
        Movie::parse(&ogg_page(1, 0, BOS, -1, &[&head])),
        Error::Inconsistent("the channel count is zero"),
    );
}

#[test]
fn an_unsupported_vorbis_version_refuses_the_file() {
    let mut id = vorbis_id(1, 44_100);
    id[7..11].copy_from_slice(&1u32.to_le_bytes());
    expect_err(
        Movie::parse(&ogg_page(1, 0, BOS, -1, &[&id])),
        Error::Inconsistent("the identification header is not supported"),
    );
}

#[test]
fn an_unsupported_opus_version_refuses_the_file() {
    let mut head = opus_head(1, 0, 48_000);
    head[8] = 0;
    expect_err(
        Movie::parse(&ogg_page(1, 0, BOS, -1, &[&head])),
        Error::Inconsistent("the identification header is not supported"),
    );
}

#[test]
fn a_short_vorbis_header_refuses_the_file() {
    let id = b"\x01vorbis";
    expect_err(
        Movie::parse(&ogg_page(1, 0, BOS, -1, &[id])),
        Error::Inconsistent("the identification header is too short"),
    );
}

#[test]
fn a_short_opus_header_refuses_the_file() {
    expect_err(
        Movie::parse(&ogg_page(1, 0, BOS, -1, &[b"OpusHead"])),
        Error::Inconsistent("the identification header is too short"),
    );
}

#[test]
fn a_short_opus_channel_map_refuses_the_file() {
    let mut head = opus_head(2, 0, 48_000);
    head[18] = 1;
    expect_err(
        Movie::parse(&ogg_page(1, 0, BOS, -1, &[&head])),
        Error::Inconsistent("the identification header is too short"),
    );
}

#[test]
fn a_file_with_no_supported_stream_refuses() {
    expect_err(
        Movie::parse(&ogg_page(9, 0, BOS, -1, &[b"not-audio"])),
        Error::Inconsistent("the file has no supported stream"),
    );
}
