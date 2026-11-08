//! Behavior of the demuxer on Vorbis streams built in memory.
//!
//! Headers come from [`crate::vorbis::build`], pages from
//! [`crate::page::encode`]. Files from libvorbis and FFmpeg are checked in
//! `tests/external`.

use std::io::Cursor;

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{
    Demux, Landing, Limits, Mux, MuxError, NewStream, OpenError, SeekTo, TrackDuration,
};

use crate::opus::OpusTags;
use crate::page::{FLAG_BOS, FLAG_EOS, encode, lacing_for, parse_header};
use crate::vorbis::build::{audio, comment, id, setup};
use crate::vorbis::xiph_lace;
use crate::{OggDemux, OggMux};

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
        let stream = usize::try_from(packet.stream()).expect("stream");
        assert_eq!(packet.timescale(), demux.streams()[stream].timescale());
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

fn vorbis_params(headers: &[Vec<u8>; 3]) -> NewStream {
    NewStream {
        params: StreamParams::try_audio(
            StreamCodec::Known(CodecId::Vorbis),
            44_100,
            2,
            xiph_lace([&headers[0], &headers[1], &headers[2]]),
        )
        .expect("params"),
        timescale: Timescale::new(44_100),
        language: None,
    }
}

fn timed(stream: u32, bytes: Vec<u8>, pts: i64, duration: i64) -> Packet {
    let time = Timestamp::try_new(Some(pts), Some(pts), Some(duration)).expect("time");
    Packet::try_new(stream, bytes, time, Timescale::new(44_100), true).expect("packet")
}

/// Modes for `count` packets, natural durations, from block flags.
fn natural(modes: &[u8]) -> Vec<i64> {
    let block = |mode: u8| if mode == LONG { 2048 } else { 256 };
    modes
        .iter()
        .enumerate()
        .map(|(index, &mode)| {
            if index == 0 {
                0
            } else {
                (block(modes[index - 1]) + block(mode)) / 4
            }
        })
        .collect()
}

/// Muxes packets in `modes` from `start`; the last lasts `last` samples,
/// or its natural duration.
fn mux_vorbis(headers: &[Vec<u8>; 3], start: i64, modes: &[u8], last: Option<i64>) -> Vec<u8> {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(vorbis_params(headers)).expect("stream");
    let durations = natural(modes);
    let mut pts = start;
    let mut sum = 0;
    for (index, (&mode, &natural)) in modes.iter().zip(&durations).enumerate() {
        let duration = if index + 1 == modes.len() {
            last.unwrap_or(natural)
        } else {
            natural
        };
        let tag = u8::try_from(index % 256).expect("tag");
        mux.write_packet(&timed(track.get(), packet(mode, tag), pts, duration))
            .expect("write");
        pts += duration;
        sum += duration;
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(44_100),
        ticks: sum,
    }])
    .expect("finish");
    mux.into_inner().into_inner()
}

/// Every page of a file: (header, lacing values).
fn page_list(bytes: &[u8]) -> Vec<(crate::page::Header, Vec<u8>)> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(header) = parse_header(&bytes[at..]) {
        let table = 27 + usize::from(header.segments);
        let lacing = bytes[at + 27..at + table].to_vec();
        let body: usize = lacing.iter().map(|&value| usize::from(value)).sum();
        out.push((header, lacing));
        at += table + body;
    }
    out
}

#[test]
fn a_muxed_vorbis_stream_reads_back_with_its_headers_times_and_trim() {
    let headers = headers();
    let modes: Vec<u8> = (0..300)
        .map(|index| if index % 7 < 3 { SHORT } else { LONG })
        .collect();
    let durations = natural(&modes);
    let bytes = mux_vorbis(&headers, 0, &modes, Some(100));
    let mut demux = open(&bytes);
    let stream = demux.streams()[0].clone();
    assert_eq!(
        stream.params().codec(),
        &StreamCodec::Known(CodecId::Vorbis)
    );
    assert_eq!(
        stream.params().extradata(),
        xiph_lace([&headers[0], &headers[1], &headers[2]]).as_slice()
    );
    assert_eq!(stream.timescale().ticks_per_second(), 44_100);
    let total: i64 = durations[..299].iter().sum::<i64>() + 100;
    assert_eq!(stream.edits()[0].media_time(), Some(0));
    assert_eq!(stream.edits()[0].duration(), total);
    assert_eq!(
        stream.trim().trailing_padding(),
        Some(u64::try_from(durations[299] - 100).expect("trim"))
    );
    let read = read_times(&mut demux);
    assert_eq!(read.len(), 300);
    let mut pts = 0;
    for (index, &(time, duration, tag)) in read.iter().enumerate() {
        let expected = if index == 299 { 100 } else { durations[index] };
        assert_eq!((time, duration), (pts, expected), "packet {index}");
        assert_eq!(tag, u8::try_from(index % 256).expect("tag"));
        pts += duration;
    }
}

#[test]
fn vorbis_headers_take_their_own_pages() {
    let headers = headers();
    let bytes = mux_vorbis(&headers, 0, &[LONG, LONG, SHORT, LONG], None);
    let pages = page_list(&bytes);
    let (bos, bos_lacing) = &pages[0];
    assert_eq!(bos.flags, FLAG_BOS);
    assert_eq!((bos.granule, bos.sequence), (0, 0));
    assert_eq!(bos_lacing, &lacing_for(headers[0].len()));
    let (setup_page, setup_lacing) = &pages[1];
    assert_eq!((setup_page.granule, setup_page.sequence), (0, 1));
    let mut expected = lacing_for(headers[1].len());
    expected.extend(lacing_for(headers[2].len()));
    assert_eq!(
        setup_lacing, &expected,
        "comment and setup, and nothing after"
    );
    let (audio, _) = &pages[2];
    assert_eq!(audio.sequence, 2);
    assert_eq!(audio.flags, FLAG_EOS);
    assert_eq!(audio.granule, 1024 + 576 + 576);
}

#[test]
fn a_comment_longer_than_a_page_spans_pages() {
    let [first, _, third] = headers();
    let picture = format!("METADATA_BLOCK_PICTURE={}", "A".repeat(70_000));
    let second = comment("vendor", &["TITLE=Long", picture.as_str()]);
    let headers = [first, second, third];
    let bytes = mux_vorbis(&headers, 0, &[LONG, LONG, LONG], None);
    let pages = page_list(&bytes);
    assert_eq!(pages[1].0.granule, -1, "a page that completes no packet");
    assert_eq!(pages[2].0.flags, crate::page::FLAG_CONTINUED);
    assert_eq!(pages[2].0.granule, 0);
    let mut demux = open(&bytes);
    let info = demux.vorbis(0).expect("info");
    assert_eq!(info.comment.comments.len(), 2);
    assert_eq!(info.comment.comments[1], picture);
    assert_eq!(read_times(&mut demux).len(), 3);
}

#[test]
fn vorbis_configs_in_three_layouts_are_accepted() {
    let headers = headers();
    let mut sixteen = Vec::new();
    let mut plain = Vec::new();
    for packet in &headers {
        sixteen.extend_from_slice(&u16::try_from(packet.len()).expect("len").to_be_bytes());
        sixteen.extend_from_slice(packet);
        plain.extend_from_slice(packet);
    }
    for config in [sixteen, plain] {
        let mut mux = OggMux::new(Cursor::new(Vec::new()));
        let mut params = vorbis_params(&headers);
        params.params =
            StreamParams::try_audio(StreamCodec::Known(CodecId::Vorbis), 44_100, 2, Vec::new())
                .expect("params");
        let track = mux.add_stream(params).expect("stream");
        mux.set_config(track, &config).expect("config");
        mux.write_packet(&timed(track.get(), packet(LONG, 0), 0, 0))
            .expect("write");
        mux.write_packet(&timed(track.get(), packet(LONG, 1), 0, 1024))
            .expect("write");
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(44_100),
            ticks: 1024,
        }])
        .expect("finish");
        let demux = open(&mux.into_inner().into_inner());
        assert_eq!(
            demux.streams()[0].params().extradata(),
            xiph_lace([&headers[0], &headers[1], &headers[2]]).as_slice()
        );
    }
}

#[test]
fn the_vorbis_muxer_refuses_what_a_reader_cannot_time() {
    let headers = headers();
    let start = |pts: i64, mode: u8, duration: i64| {
        let mut mux = OggMux::new(Cursor::new(Vec::new()));
        let track = mux.add_stream(vorbis_params(&headers)).expect("stream");
        let result = mux.write_packet(&timed(track.get(), packet(mode, 0), pts, duration));
        (mux, track, result)
    };
    let message = |result: Result<(), MuxError>| match result {
        Err(MuxError::Rejected { message }) => message,
        other => panic!("{other:?}"),
    };
    assert!(message(start(0, LONG, 1).2).contains("natural 0"));
    let (mut mux, track, first) = start(0, LONG, 0);
    first.expect("first");
    assert!(message(mux.write_packet(&timed(track.get(), vec![1, 2, 3], 0, 0))).contains("header"));
    let (mut mux, track, first) = start(0, LONG, 0);
    first.expect("first");
    assert!(
        message(mux.write_packet(&timed(track.get(), packet(SHORT, 1), 0, 577)))
            .contains("natural 576")
    );
    let (mut mux, track, first) = start(0, LONG, 0);
    first.expect("first");
    mux.write_packet(&timed(track.get(), packet(SHORT, 1), 0, 100))
        .expect("trimmed");
    assert!(
        message(mux.write_packet(&timed(track.get(), packet(SHORT, 2), 100, 128)))
            .contains("only the last packet")
    );
    let (mut mux, track, first) = start(0, LONG, 0);
    first.expect("first");
    assert!(
        message(mux.write_packet(&timed(track.get(), packet(LONG, 1), 10, 1024))).contains("gap")
    );

    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(vorbis_params(&headers)).expect("stream");
    let wrong = Packet::try_new(
        track.get(),
        packet(LONG, 0),
        Timestamp::try_new(Some(0), Some(0), Some(0)).expect("time"),
        Timescale::new(48_000),
        true,
    )
    .expect("packet");
    assert!(matches!(
        mux.write_packet(&wrong),
        Err(MuxError::TimescaleMismatch {
            packet: 48_000,
            stream: 44_100
        })
    ));

    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let mut mono = vorbis_params(&headers);
    mono.params = StreamParams::try_audio(
        StreamCodec::Known(CodecId::Vorbis),
        44_100,
        1,
        xiph_lace([&headers[0], &headers[1], &headers[2]]),
    )
    .expect("params");
    assert!(message(mux.add_stream(mono).map(|_| ())).contains("channels"));

    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let bare = || {
        let mut bare = vorbis_params(&headers);
        bare.params =
            StreamParams::try_audio(StreamCodec::Known(CodecId::Vorbis), 44_100, 2, Vec::new())
                .expect("params");
        bare
    };
    let track = mux.add_stream(bare()).expect("stream");
    assert!(message(mux.set_config(track, b"\x01vorbis")).contains("three Vorbis headers"));
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(bare()).expect("stream");
    assert!(
        message(mux.write_packet(&timed(track.get(), packet(LONG, 0), 0, 0)))
            .contains("no Vorbis id, comment, and setup headers")
    );
}

#[test]
fn a_late_vorbis_start_with_a_trimmed_end_keeps_both() {
    let headers = headers();
    let bytes = mux_vorbis(&headers, 44_100, &[LONG, LONG, LONG], Some(600));
    let mut demux = open(&bytes);
    let info = demux.vorbis(0).expect("info").clone();
    assert_eq!((info.start, info.end), (Some(44_100), Some(44_100 + 1624)));
    assert_eq!(
        read_times(&mut demux),
        vec![(44_100, 0, 0), (44_100, 1024, 1), (45_124, 600, 2)]
    );
}

#[test]
fn opus_and_vorbis_mux_into_one_file() {
    let headers = headers();
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let mut head = b"OpusHead".to_vec();
    head.extend_from_slice(&[1, 2, 0, 0]);
    head.extend_from_slice(&48_000_u32.to_le_bytes());
    head.extend_from_slice(&[0, 0, 0]);
    let opus = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(StreamCodec::Known(CodecId::Opus), 48_000, 2, head)
                .expect("params"),
            timescale: Timescale::new(48_000),
            language: None,
        })
        .expect("opus");
    let vorbis = mux.add_stream(vorbis_params(&headers)).expect("vorbis");
    let opus_packet = |pts: i64| {
        Packet::try_new(
            opus.get(),
            vec![31 << 3, 1, 2],
            Timestamp::try_new(Some(pts), Some(pts), Some(960)).expect("time"),
            Timescale::new(48_000),
            true,
        )
        .expect("packet")
    };
    mux.write_packet(&opus_packet(0)).expect("opus");
    mux.write_packet(&timed(vorbis.get(), packet(LONG, 0), 0, 0))
        .expect("vorbis");
    mux.write_packet(&opus_packet(960)).expect("opus");
    mux.write_packet(&timed(vorbis.get(), packet(LONG, 1), 0, 1024))
        .expect("vorbis");
    mux.finish(&[
        TrackDuration {
            track: opus,
            timescale: Timescale::new(48_000),
            ticks: 1920,
        },
        TrackDuration {
            track: vorbis,
            timescale: Timescale::new(44_100),
            ticks: 1024,
        },
    ])
    .expect("finish");
    let mut demux = open(&mux.into_inner().into_inner());
    assert_eq!(
        demux.streams()[0].params().codec(),
        &StreamCodec::Known(CodecId::Opus)
    );
    assert_eq!(
        demux.streams()[1].params().codec(),
        &StreamCodec::Known(CodecId::Vorbis)
    );
    let mut counts = [0, 0];
    while let Some(packet) = demux.read_packet().expect("read") {
        counts[usize::try_from(packet.stream()).expect("stream")] += 1;
        assert_eq!(
            packet.timescale(),
            demux.streams()[usize::try_from(packet.stream()).expect("stream")].timescale()
        );
    }
    assert_eq!(counts, [2, 2]);
}

#[test]
fn a_vorbis_stream_that_starts_before_zero_muxes_and_reads_back() {
    let headers = headers();
    let bytes = mux_vorbis(&headers, -1000, &[LONG; 4], None);
    let pages = page_list(&bytes);
    assert_eq!(pages.len(), 4);
    assert_eq!((pages[2].0.flags, pages[2].0.granule), (0, 2072));
    assert_eq!((pages[3].0.flags, pages[3].0.granule), (FLAG_EOS, 2072));
    assert!(pages[3].1.is_empty(), "an empty last page");
    let mut demux = open(&bytes);
    let stream = demux.streams()[0].clone();
    let info = demux.vorbis(0).expect("info").clone();
    assert_eq!(
        (info.start, info.end, info.trailing),
        (Some(-1000), Some(2072), Some(0))
    );
    assert_eq!(stream.edits()[0].media_time(), Some(0));
    assert_eq!(stream.edits()[0].duration(), 2072);
    assert_eq!(stream.trim().encoder_delay(), Some(1000));
    assert_eq!(
        read_times(&mut demux),
        vec![
            (-1000, 0, 0),
            (-1000, 1024, 1),
            (24, 1024, 2),
            (1048, 1024, 3)
        ]
    );
}

#[test]
fn a_vorbis_stream_before_zero_with_a_trimmed_end_keeps_both() {
    let headers = headers();
    let bytes = mux_vorbis(&headers, -1000, &[LONG; 4], Some(600));
    let pages = page_list(&bytes);
    assert_eq!((pages[2].0.flags, pages[2].0.granule), (0, 1048));
    assert_eq!((pages[3].0.flags, pages[3].0.granule), (FLAG_EOS, 1648));
    let mut demux = open(&bytes);
    let info = demux.vorbis(0).expect("info").clone();
    assert_eq!(
        (info.start, info.end, info.trailing),
        (Some(-1000), Some(1648), Some(424))
    );
    assert_eq!(
        read_times(&mut demux),
        vec![
            (-1000, 0, 0),
            (-1000, 1024, 1),
            (24, 1024, 2),
            (1048, 600, 3)
        ]
    );
}

#[test]
fn a_vorbis_page_that_would_end_before_zero_is_refused() {
    let headers = headers();
    let finish = |start: i64, count: usize, last: i64| {
        let mut mux = OggMux::new(Cursor::new(Vec::new()));
        let track = mux.add_stream(vorbis_params(&headers)).expect("stream");
        let durations = natural(&vec![LONG; count]);
        let mut pts = start;
        let mut sum = 0;
        for (index, &natural) in durations.iter().enumerate() {
            let duration = if index + 1 == count { last } else { natural };
            let tag = u8::try_from(index % 256).expect("tag");
            mux.write_packet(&timed(track.get(), packet(LONG, tag), pts, duration))
                .map_err(|error| error.to_string())?;
            pts += duration;
            sum += duration;
        }
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(44_100),
            ticks: sum,
        }])
        .map(|_| ())
        .map_err(|failed| failed.reason().to_string())
    };
    // Only the trimmed packet crosses 0, and it must go on a later page.
    let error = finish(-1500, 3, 600).expect_err("head ends before 0");
    assert!(
        error.to_string().contains("would end at -476, before 0"),
        "{error}"
    );
    // 255 lacing values fill the first page while its packets end before 0.
    let error = finish(-400_000, 300, 1024).expect_err("page full before 0");
    assert!(error.contains("before 0"), "{error}");
    finish(-1500, 3, 1024).expect("ends past 0 untrimmed");
}

#[test]
fn a_seek_in_a_long_variable_rate_file_takes_few_probes() {
    let headers = headers();
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(vorbis_params(&headers)).expect("stream");
    // 8,000 long blocks, 186 s, with packets from 200 to 1,800 bytes in a
    // slow swell, so bytes per sample change along the file.
    let mut pts = 0;
    let mut starts = Vec::new();
    for index in 0..8_000_usize {
        let swell = (index / 400) % 5;
        let len = 200 + swell * 300 + (index * 37) % 200;
        let mut bytes = packet(LONG, u8::try_from(index % 256).expect("tag"));
        bytes.resize(len, 0x33);
        let duration = if index == 0 { 0 } else { 1024 };
        starts.push(pts);
        mux.write_packet(&timed(track.get(), bytes, pts, duration))
            .expect("write");
        pts += duration;
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(44_100),
        ticks: pts,
    }])
    .expect("finish");
    let bytes = mux.into_inner().into_inner();
    assert!(bytes.len() > 7_000_000);
    let mut demux = open(&bytes);
    let mut worst = 0;
    let mut total = 0;
    for step in 1..60 {
        let target = pts * step / 60 + 333;
        let landing = demux
            .seek(SeekTo {
                track: 0,
                time: target,
                end: None,
            })
            .expect("seek");
        let Landing::Packet { time, .. } = landing.landing else {
            panic!("{target} reached the end");
        };
        let aim = target - 1024;
        let holding = starts[1..]
            .iter()
            .rev()
            .find(|&&start| start <= aim)
            .expect("packet");
        assert_eq!(time, *holding, "landing for {target}");
        worst = worst.max(demux.last_seek().probes);
        total += demux.last_seek().probes;
    }
    // Halving alone takes 7 probes to reach the 64 KiB window here.
    assert!(worst <= 6, "a seek took {worst} probes");
    assert!(total <= 59 * 4, "{total} probes for 59 seeks");
}
