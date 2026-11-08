//! Behavior of the demuxer on Vorbis streams built in memory.
//!
//! Headers come from [`crate::vorbis::build`], pages from
//! [`crate::page::encode`]. Files from libvorbis and FFmpeg are checked in
//! `tests/external`.

use std::io::Cursor;

use cutlass_codec::{CodecId, StreamCodec};
use cutlass_container::{Demux, Landing, Limits, OpenError, SeekTo};

use crate::OggDemux;
use crate::opus::OpusTags;
use crate::page::{FLAG_BOS, FLAG_EOS, encode, lacing_for};
use crate::vorbis::build::{audio, comment, id, setup};
use crate::vorbis::xiph_lace;

const SERIAL: u32 = 0x5EED;
const SHORT: u8 = 0;
const LONG: u8 = 1;

/// The three headers of a stereo 44.1 kHz stream with blocks 256 and 2048.
fn headers() -> [Vec<u8>; 3] {
    [
        id(2, 44_100, 8, 11),
        comment("test vendor", &["TITLE=Tone"]),
        setup(2, &[false, true]),
    ]
}

fn page_of(flags: u8, granule: i64, serial: u32, sequence: u32, packets: &[Vec<u8>]) -> Vec<u8> {
    let mut lacing = Vec::new();
    let mut body = Vec::new();
    for packet in packets {
        lacing.extend(lacing_for(packet.len()));
        body.extend_from_slice(packet);
    }
    encode(flags, granule, serial, sequence, &lacing, &body)
}

/// A file with `headers` on the first two pages, then `pages` of audio
/// packets: (flags, granule, packets).
fn file_with(headers: &[Vec<u8>; 3], pages: &[(u8, i64, Vec<Vec<u8>>)]) -> Vec<u8> {
    let mut file = page_of(FLAG_BOS, 0, SERIAL, 0, &[headers[0].clone()]);
    file.extend(page_of(
        0,
        0,
        SERIAL,
        1,
        &[headers[1].clone(), headers[2].clone()],
    ));
    for (index, (flags, granule, packets)) in pages.iter().enumerate() {
        let sequence = u32::try_from(index + 2).expect("sequence");
        file.extend(page_of(*flags, *granule, SERIAL, sequence, packets));
    }
    file
}

/// An audio packet in `mode` whose bytes name `tag`.
fn packet(mode: u8, tag: u8) -> Vec<u8> {
    let mut bytes = audio(mode, 6);
    bytes[1] = tag;
    bytes
}

fn try_open_with(bytes: &[u8], limits: Limits) -> Result<OggDemux<Cursor<Vec<u8>>>, OpenError> {
    let len = u64::try_from(bytes.len()).expect("len");
    OggDemux::open_limited(Cursor::new(bytes.to_vec()), len, limits)
}

fn open(bytes: &[u8]) -> OggDemux<Cursor<Vec<u8>>> {
    try_open_with(bytes, Limits::standard()).expect("open")
}

fn open_error(bytes: &[u8]) -> String {
    match try_open_with(bytes, Limits::standard()) {
        Ok(_) => panic!("opened"),
        Err(error) => format!("{error:?}"),
    }
}

fn read_times(demux: &mut OggDemux<Cursor<Vec<u8>>>) -> Vec<(i64, i64, u8)> {
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        let time = packet.timestamp();
        out.push((
            time.presentation().expect("pts"),
            time.duration().expect("duration"),
            packet.bytes()[1],
        ));
    }
    out
}

#[test]
fn a_vorbis_stream_opens_with_its_rate_headers_and_trim() {
    let headers = headers();
    let file = file_with(
        &headers,
        &[
            (
                0,
                1728,
                vec![
                    packet(LONG, 0),
                    packet(LONG, 1),
                    packet(SHORT, 2),
                    packet(SHORT, 3),
                ],
            ),
            (FLAG_EOS, 3000, vec![packet(LONG, 4), packet(LONG, 5)]),
        ],
    );
    let mut demux = open(&file);
    let stream = demux.streams()[0].clone();
    assert_eq!(
        stream.params().codec(),
        &StreamCodec::Known(CodecId::Vorbis)
    );
    assert_eq!(stream.params().sample_rate(), Some(44_100));
    assert_eq!(stream.params().channels(), Some(2));
    assert_eq!(
        stream.params().extradata(),
        xiph_lace([&headers[0], &headers[1], &headers[2]]).as_slice()
    );
    assert_eq!(stream.timescale().ticks_per_second(), 44_100);
    assert_eq!(stream.trim().encoder_delay(), Some(0));
    assert_eq!(stream.trim().trailing_padding(), Some(328));
    assert_eq!(stream.edits()[0].media_time(), Some(0));
    assert_eq!(stream.edits()[0].duration(), 3000);

    let info = demux.vorbis(0).expect("info").clone();
    assert_eq!(info.serial, SERIAL);
    assert_eq!(info.id.blocksizes, [256, 2048]);
    assert_eq!(info.comment.vendor, "test vendor");
    assert_eq!(info.comment.comments, ["TITLE=Tone"]);
    assert_eq!(info.modes, [false, true]);
    assert_eq!(
        (info.start, info.end, info.trailing),
        (Some(0), Some(3000), Some(328))
    );
    assert!(demux.opus(0).is_none());
    assert!(demux.tags(0).is_none());

    assert_eq!(
        read_times(&mut demux),
        vec![
            (0, 0, 0),
            (0, 1024, 1),
            (1024, 576, 2),
            (1600, 128, 3),
            (1728, 576, 4),
            (2304, 696, 5),
        ]
    );
}

#[test]
fn a_vorbis_stream_that_starts_before_zero_trims_its_lead() {
    let file = file_with(
        &headers(),
        &[
            (
                0,
                1000,
                vec![packet(LONG, 0), packet(LONG, 1), packet(LONG, 2)],
            ),
            (FLAG_EOS, 2024, vec![packet(LONG, 3)]),
        ],
    );
    let mut demux = open(&file);
    let stream = demux.streams()[0].clone();
    assert_eq!(stream.edits()[0].media_time(), Some(0));
    assert_eq!(stream.edits()[0].duration(), 2024);
    assert_eq!(stream.trim().encoder_delay(), Some(1048));
    assert_eq!(stream.trim().trailing_padding(), Some(0));
    assert_eq!(demux.vorbis(0).expect("info").start, Some(-1048));
    assert_eq!(
        read_times(&mut demux),
        vec![
            (-1048, 0, 0),
            (-1048, 1024, 1),
            (-24, 1024, 2),
            (1000, 1024, 3)
        ]
    );
}

#[test]
fn a_vorbis_stream_on_one_page_starts_at_zero() {
    let file = file_with(
        &headers(),
        &[(
            FLAG_EOS,
            1500,
            vec![packet(LONG, 0), packet(LONG, 1), packet(LONG, 2)],
        )],
    );
    let mut demux = open(&file);
    let stream = demux.streams()[0].clone();
    assert_eq!(stream.edits()[0].media_time(), Some(0));
    assert_eq!(stream.edits()[0].duration(), 1500);
    assert_eq!(stream.trim().trailing_padding(), Some(548));
    assert_eq!(
        read_times(&mut demux),
        vec![(0, 0, 0), (0, 1024, 1), (1024, 476, 2)]
    );
}

#[test]
fn a_forward_vorbis_read_gives_the_same_packets() {
    let file = file_with(
        &headers(),
        &[
            (
                0,
                2048,
                vec![packet(LONG, 0), packet(LONG, 1), packet(LONG, 2)],
            ),
            (FLAG_EOS, 3000, vec![packet(LONG, 3)]),
        ],
    );
    let mut forward = OggDemux::open_forward(Cursor::new(file.clone())).expect("forward");
    let mut random = open(&file);
    assert_eq!(forward.streams()[0].params(), random.streams()[0].params());
    assert_eq!(read_times(&mut forward), read_times(&mut random));
}

/// 30 pages of four long packets; packet `n` lasts 1024 samples after the
/// first, which lasts none.
fn long_file() -> (Vec<u8>, i64) {
    let mut pages = Vec::new();
    let mut granule = -1024;
    for page in 0..30_u8 {
        let packets: Vec<Vec<u8>> = (0..4).map(|index| packet(LONG, page * 4 + index)).collect();
        granule += 4 * 1024;
        let flags = if page == 29 { FLAG_EOS } else { 0 };
        pages.push((flags, granule, packets));
    }
    (file_with(&headers(), &pages), granule)
}

#[test]
fn a_vorbis_seek_lands_a_packet_before_the_target_less_half_a_long_block() {
    let (file, end) = long_file();
    let mut demux = open(&file);
    let all = read_times(&mut demux);
    assert_eq!(all.len(), 120);
    assert_eq!(demux.streams()[0].edits()[0].duration(), end);
    for target in [0, 1, 1023, 1024, 5000, 40_000, 77_777, end - 1025, end - 1] {
        let landing = demux
            .seek(SeekTo {
                track: 0,
                time: target,
                end: None,
            })
            .expect("seek");
        let Landing::Packet { time, preroll, .. } = landing.landing else {
            panic!("{target} landed at the end");
        };
        let aim = (target - 1024).max(0);
        assert!(time <= aim && aim < time + 1024 * 4, "{target} -> {time}");
        let expected = all
            .iter()
            .filter(|(pts, _, _)| *pts >= time && *pts < target)
            .count();
        assert_eq!(
            usize::try_from(preroll).expect("preroll"),
            expected,
            "{target}"
        );
        let next = demux.read_packet().expect("read").expect("packet");
        let pts = next.timestamp().presentation().expect("pts");
        assert_eq!(pts, time, "{target}");
        let duration = next.timestamp().duration().expect("duration");
        let reference = all
            .iter()
            .find(|(_, _, tag)| *tag == next.bytes()[1])
            .expect("packet");
        assert_eq!((pts, duration), (reference.0, reference.1), "{target}");
        // The decoder returns nothing for the landing packet, and the next
        // packet starts at or before the target.
        assert!(pts + duration <= target.max(1024), "{target}");
    }
    let past = demux
        .seek(SeekTo {
            track: 0,
            time: end,
            end: None,
        })
        .expect("past");
    assert_eq!(past.landing, Landing::End);
}

#[test]
fn a_file_with_opus_and_vorbis_reads_both() {
    let headers = headers();
    let opus_serial = 77;
    let mut opus_head = b"OpusHead".to_vec();
    opus_head.extend_from_slice(&[1, 1, 0, 0]);
    opus_head.extend_from_slice(&48_000_u32.to_le_bytes());
    opus_head.extend_from_slice(&[0, 0, 0]);
    let mut file = page_of(FLAG_BOS, 0, opus_serial, 0, &[opus_head.clone()]);
    file.extend(page_of(FLAG_BOS, 0, SERIAL, 0, &[headers[0].clone()]));
    file.extend(page_of(
        0,
        0,
        opus_serial,
        1,
        &[OpusTags::default().encode()],
    ));
    file.extend(page_of(
        0,
        0,
        SERIAL,
        1,
        &[headers[1].clone(), headers[2].clone()],
    ));
    file.extend(page_of(
        0,
        1024,
        SERIAL,
        2,
        &[packet(LONG, 0), packet(LONG, 1)],
    ));
    file.extend(page_of(FLAG_EOS, 960, opus_serial, 2, &[vec![31 << 3, 9]]));
    file.extend(page_of(FLAG_EOS, 2048, SERIAL, 3, &[packet(LONG, 2)]));
    let mut demux = open(&file);
    assert_eq!(demux.streams().len(), 2);
    assert_eq!(
        demux.streams()[0].params().codec(),
        &StreamCodec::Known(CodecId::Opus)
    );
    assert_eq!(
        demux.streams()[1].params().codec(),
        &StreamCodec::Known(CodecId::Vorbis)
    );
    assert!(demux.opus(0).is_some() && demux.vorbis(1).is_some());
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        packets.push((
            packet.stream(),
            packet.timestamp().presentation().expect("pts"),
        ));
    }
    assert_eq!(packets, vec![(1, 0), (1, 0), (0, 0), (1, 1024)]);
}

#[test]
fn the_identification_header_is_alone_on_its_page() {
    let headers = headers();
    let mut file = page_of(
        FLAG_BOS,
        0,
        SERIAL,
        0,
        &[headers[0].clone(), headers[1].clone()],
    );
    file.extend(page_of(0, 0, SERIAL, 1, &[headers[2].clone()]));
    let error = open_error(&file);
    assert!(error.contains("broken first page"), "{error}");
}

#[test]
fn a_file_without_the_setup_header_is_refused() {
    let headers = headers();
    let mut file = page_of(FLAG_BOS, 0, SERIAL, 0, &[headers[0].clone()]);
    file.extend(page_of(0, 0, SERIAL, 1, &[headers[1].clone()]));
    let error = open_error(&file);
    assert!(error.contains("Vorbis comment and setup"), "{error}");
}

#[test]
fn audio_on_the_setup_page_is_refused() {
    let headers = headers();
    let mut file = page_of(FLAG_BOS, 0, SERIAL, 0, &[headers[0].clone()]);
    file.extend(page_of(
        0,
        0,
        SERIAL,
        1,
        &[headers[1].clone(), headers[2].clone(), packet(LONG, 0)],
    ));
    file.extend(page_of(FLAG_EOS, 1024, SERIAL, 2, &[packet(LONG, 1)]));
    let error = open_error(&file);
    assert!(error.contains("does not finish its page"), "{error}");
}

#[test]
fn a_damaged_vorbis_header_is_refused() {
    let [first, second, third] = headers();
    let mut version = first.clone();
    version[7] = 3;
    let error = open_error(&file_with(&[version, second.clone(), third.clone()], &[]));
    assert!(error.contains("UnsupportedBrand"), "{error}");

    let mut broken = third.clone();
    let last = broken.len() - 1;
    broken[last] = 0;
    let error = open_error(&file_with(&[first.clone(), second.clone(), broken], &[]));
    assert!(error.contains("Vorbis setup"), "{error}");

    let error = open_error(&file_with(&[first, third, second], &[]));
    assert!(error.contains("comment header"), "{error}");
}

#[test]
fn a_setup_header_past_the_header_bound_is_refused() {
    let headers = headers();
    let file = file_with(&headers, &[(FLAG_EOS, 0, vec![packet(LONG, 0)])]);
    let limits = Limits {
        max_header_bytes: u64::try_from(headers[2].len() - 1).expect("len"),
        ..Limits::standard()
    };
    let Err(error) = try_open_with(&file, limits) else {
        panic!("opened");
    };
    assert!(matches!(error, OpenError::HostileSize { .. }), "{error:?}");
}

#[test]
fn a_long_comment_is_cut_and_the_stream_still_opens() {
    let [first, _, third] = headers();
    let long = format!("DESCRIPTION={}", "x".repeat(900));
    let second = comment("vendor", &["A=1", long.as_str()]);
    let file = file_with(
        &[first.clone(), second, third.clone()],
        &[(FLAG_EOS, 1024, vec![packet(LONG, 0), packet(LONG, 1)])],
    );
    let limits = Limits {
        max_header_bytes: 200,
        ..Limits::standard()
    };
    let demux = try_open_with(&file, limits).expect("open");
    let info = demux.vorbis(0).expect("info");
    assert!(info.comment.cut);
    assert_eq!(info.comment.comments, ["A=1"]);
    let written = info.comment.encode();
    assert_eq!(
        demux.streams()[0].params().extradata(),
        xiph_lace([&first, &written, &third]).as_slice()
    );
}

#[test]
fn extradata_past_its_bound_keeps_only_the_vendor() {
    let [first, _, third] = headers();
    let long = format!("DESCRIPTION={}", "x".repeat(900));
    let second = comment("vendor", &[long.as_str()]);
    let file = file_with(
        &[first.clone(), second, third.clone()],
        &[(FLAG_EOS, 1024, vec![packet(LONG, 0), packet(LONG, 1)])],
    );
    let limits = Limits {
        max_extradata_bytes: 300,
        ..Limits::standard()
    };
    let demux = try_open_with(&file, limits).expect("open");
    assert_eq!(demux.vorbis(0).expect("info").comment.comments, [long]);
    let vendor_only = comment("vendor", &[]);
    assert_eq!(
        demux.streams()[0].params().extradata(),
        xiph_lace([&first, &vendor_only, &third]).as_slice()
    );
    let tight = Limits {
        max_extradata_bytes: 40,
        ..Limits::standard()
    };
    let file = file_with(&headers(), &[(FLAG_EOS, 0, vec![packet(LONG, 0)])]);
    let Err(error) = try_open_with(&file, tight) else {
        panic!("opened");
    };
    assert!(matches!(error, OpenError::HostileSize { .. }), "{error:?}");
}

#[test]
fn every_cut_of_a_vorbis_file_fails_closed_or_opens() {
    let (file, _) = long_file();
    for end in (0..file.len()).step_by(7) {
        let len = u64::try_from(end).expect("len");
        if let Ok(mut demux) = OggDemux::open(Cursor::new(file[..end].to_vec()), len) {
            while let Ok(Some(_)) = demux.read_packet() {}
        }
    }
}
