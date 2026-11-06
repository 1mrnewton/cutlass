//! Behavior of the demuxer and muxer on files built in memory.
//!
//! Valid files come from [`OggMux`]; hostile ones are built page by page
//! with [`crate::page::encode`] or by cutting and flipping muxed bytes. The
//! independent producer and demuxer checks are in `tests/external`.

use std::io::Cursor;

use cutlass_codec::{
    CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp, UnknownCodec,
};
use cutlass_container::{
    Demux, Landing, Mux, MuxError, MuxStatus, NewStream, OpenError, ReadError, SeekError, SeekTo,
    TrackDuration, TrackId,
};

use crate::crc::page_crc;
use crate::error::OggError;
use crate::limits::PAGE_HEADER_BYTES;
use crate::mux::{FIRST_SERIAL, OggMux};
use crate::page::{FLAG_BOS, FLAG_CONTINUED, FLAG_EOS, encode, lacing_for, parse_header};
use crate::{OggDemux, OpusTags};

/// 20 ms CELT fullband, code 0.
const TOC_20MS: u8 = 31 << 3;

fn head(channels: u8, pre_skip: u16) -> Vec<u8> {
    let mut bytes = b"OpusHead".to_vec();
    bytes.push(1);
    bytes.push(channels);
    bytes.extend_from_slice(&pre_skip.to_le_bytes());
    bytes.extend_from_slice(&48_000_u32.to_le_bytes());
    bytes.extend_from_slice(&0_i16.to_le_bytes());
    if channels <= 2 {
        bytes.push(0);
    } else {
        bytes.push(1);
        bytes.push(channels);
        bytes.push(0);
        bytes.extend(0..channels);
    }
    bytes
}

fn payload(len: usize, seed: u8) -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..len)
        .map(|index| u8::try_from(index % 251).unwrap_or(0) ^ seed)
        .collect();
    if let Some(first) = bytes.first_mut() {
        *first = TOC_20MS;
    }
    bytes
}

fn packet(stream: u32, bytes: Vec<u8>, pts: i64, duration: i64) -> Packet {
    let time = Timestamp::try_new(Some(pts), Some(pts), Some(duration)).expect("time");
    Packet::try_new(stream, bytes, time, Timescale::new(48_000), true).expect("packet")
}

fn opus_stream(channels: u16, config: Vec<u8>) -> NewStream {
    NewStream {
        params: StreamParams::try_audio(
            StreamCodec::Known(CodecId::Opus),
            48_000,
            channels,
            config,
        )
        .expect("params"),
        timescale: Timescale::new(48_000),
        language: None,
    }
}

/// One stream of `sizes` 20 ms packets from `start`; the last packet lasts
/// `last` samples.
fn mux_one(pre_skip: u16, start: i64, sizes: &[usize], last: i64) -> Vec<u8> {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux
        .add_stream(opus_stream(2, head(2, pre_skip)))
        .expect("stream");
    let mut pts = start;
    let mut sum = 0;
    for (index, &size) in sizes.iter().enumerate() {
        let duration = if index + 1 == sizes.len() { last } else { 960 };
        let seed = u8::try_from(index % 256).unwrap_or(0);
        mux.write_packet(&packet(track.get(), payload(size, seed), pts, duration))
            .expect("write");
        pts += duration;
        sum += duration;
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(48_000),
        ticks: sum,
    }])
    .expect("finish");
    mux.into_inner().into_inner()
}

fn open(bytes: &[u8]) -> OggDemux<Cursor<Vec<u8>>> {
    try_open(bytes).expect("open")
}

fn try_open(bytes: &[u8]) -> Result<OggDemux<Cursor<Vec<u8>>>, OpenError> {
    let len = u64::try_from(bytes.len()).expect("len");
    OggDemux::open(Cursor::new(bytes.to_vec()), len)
}

fn read_all(demux: &mut OggDemux<Cursor<Vec<u8>>>) -> Vec<Packet> {
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        packets.push(packet);
    }
    packets
}

fn times(packets: &[Packet]) -> Vec<(i64, i64)> {
    packets
        .iter()
        .map(|packet| {
            (
                packet.timestamp().presentation().expect("pts"),
                packet.timestamp().duration().expect("duration"),
            )
        })
        .collect()
}

/// Every page of a file: (offset, length, header).
fn pages(bytes: &[u8]) -> Vec<(usize, usize, crate::page::Header)> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(header) = parse_header(&bytes[at..]) {
        let table = 27 + usize::from(header.segments);
        let body: usize = bytes[at + 27..at + table]
            .iter()
            .map(|&value| usize::from(value))
            .sum();
        out.push((at, table + body, header));
        at += table + body;
        if at >= bytes.len() {
            break;
        }
    }
    out
}

/// A hand-built stream: BOS with `head`, `OpusTags`, then audio pages.
fn crafted(head_bytes: &[u8], audio: &[(u8, i64, Vec<Vec<u8>>)]) -> Vec<u8> {
    let serial = 77;
    let mut file = encode(
        FLAG_BOS,
        0,
        serial,
        0,
        &lacing_for(head_bytes.len()),
        head_bytes,
    );
    let tags = OpusTags {
        vendor: "test".to_owned(),
        comments: vec!["TITLE=t".to_owned()],
        cut: false,
    }
    .encode();
    file.extend(encode(0, 0, serial, 1, &lacing_for(tags.len()), &tags));
    for (index, (flags, granule, packets)) in audio.iter().enumerate() {
        let mut lacing = Vec::new();
        let mut body = Vec::new();
        for bytes in packets {
            lacing.extend(lacing_for(bytes.len()));
            body.extend_from_slice(bytes);
        }
        let sequence = u32::try_from(index + 2).expect("seq");
        file.extend(encode(*flags, *granule, serial, sequence, &lacing, &body));
    }
    file
}

#[test]
fn a_muxed_stream_reads_back_with_its_bytes_times_and_trim() {
    let sizes = [1, 254, 255, 256, 510, 300, 700, 120];
    let bytes = mux_one(312, 0, &sizes, 500);
    let mut demux = open(&bytes);
    let stream = demux.streams()[0].clone();
    assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Opus));
    assert_eq!(stream.params().channels(), Some(2));
    assert_eq!(stream.params().sample_rate(), Some(48_000));
    assert_eq!(stream.params().extradata(), head(2, 312).as_slice());
    assert_eq!(stream.timescale().ticks_per_second(), 48_000);
    assert_eq!(stream.trim().encoder_delay(), Some(312));
    assert_eq!(stream.trim().trailing_padding(), Some(460));
    let end = 7 * 960 + 500;
    assert_eq!(stream.edits()[0].media_time(), Some(312));
    assert_eq!(stream.edits()[0].duration(), end - 312);
    let tags = demux.tags(0).expect("tags");
    assert_eq!(tags.vendor, "cutlass");
    assert!(tags.comments.is_empty());

    let packets = read_all(&mut demux);
    assert_eq!(packets.len(), sizes.len());
    for (index, (packet, &size)) in packets.iter().zip(&sizes).enumerate() {
        let seed = u8::try_from(index).expect("seed");
        assert_eq!(packet.bytes(), payload(size, seed).as_slice());
        assert!(packet.keyframe());
        assert_eq!(
            packet.timestamp().decode(),
            packet.timestamp().presentation()
        );
    }
    let mut expected: Vec<(i64, i64)> = (0..7).map(|index| (index * 960, 960)).collect();
    expected.push((7 * 960, 500));
    assert_eq!(times(&packets), expected);
    assert!(demux.read_packet().expect("end").is_none());
    assert!(demux.gaps().is_empty());
}

#[test]
fn a_cropped_start_keeps_granule_time_and_moves_the_edit() {
    let bytes = mux_one(3_840, 48_000, &[100; 10], 960);
    let mut demux = open(&bytes);
    let info = demux.opus(0).expect("info").clone();
    assert_eq!(info.start, Some(48_000));
    assert_eq!(info.end, Some(48_000 + 10 * 960));
    let edit = demux.streams()[0].edits()[0];
    assert_eq!(edit.media_time(), Some(48_000 + 3_840));
    assert_eq!(edit.duration(), 10 * 960 - 3_840);
    // The muxer refuses a stream shorter than its pre-skip, and the
    // demuxer does not open one written by hand.
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux
        .add_stream(opus_stream(2, head(2, 3_840)))
        .expect("stream");
    for index in 0..3 {
        mux.write_packet(&packet(
            track.get(),
            payload(100, index),
            48_000 + i64::from(index) * 960,
            960,
        ))
        .expect("write");
    }
    let refused = mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(48_000),
        ticks: 3 * 960,
    }]);
    assert!(refused.is_err());
    assert_eq!(mux.status(), MuxStatus::Failed);
    let mut short = mux_one(0, 48_000, &[100; 3], 960);
    // OpusHead's pre-skip sits at body bytes 10..12 of the first page,
    // whose lacing table has one value.
    let body = PAGE_HEADER_BYTES + 1;
    short[body + 10..body + 12].copy_from_slice(&3_840_u16.to_le_bytes());
    let checksum = page_crc(&short[..body], &short[body..body + 19]);
    short[22..26].copy_from_slice(&checksum.to_le_bytes());
    assert!(matches!(
        try_open(&short),
        Err(OpenError::Source { message }) if message.contains("pre-skip")
    ));
    let packets = read_all(&mut demux);
    assert_eq!(times(&packets)[0], (48_000, 960));
}

#[test]
fn a_cropped_start_and_an_end_trim_do_not_share_the_first_page() {
    // On one first-and-last page the demuxer would read the start as the
    // granule minus all samples, so the muxer splits off the last packet.
    let bytes = mux_one(0, 1_920, &[100; 4], 480);
    let layout = pages(&bytes);
    assert_eq!(layout.len(), 4);
    assert_eq!(layout[2].2.flags & FLAG_EOS, 0);
    assert_eq!(layout[2].2.granule, 1_920 + 3 * 960);
    assert_eq!(layout[3].2.flags & FLAG_EOS, FLAG_EOS);
    assert_eq!(layout[3].2.granule, 1_920 + 3 * 960 + 480);
    let mut demux = open(&bytes);
    assert_eq!(demux.opus(0).expect("info").start, Some(1_920));
    assert_eq!(
        times(&read_all(&mut demux)),
        vec![(1_920, 960), (2_880, 960), (3_840, 960), (4_800, 480)]
    );
    // From 0 the single page is unambiguous and stays whole.
    assert_eq!(pages(&mux_one(0, 0, &[100; 4], 480)).len(), 3);

    // One trimmed packet after 0 cannot be written.
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(opus_stream(2, head(2, 0))).expect("stream");
    mux.write_packet(&packet(track.get(), payload(100, 0), 960, 480))
        .expect("write");
    let refused = mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(48_000),
        ticks: 480,
    }]);
    assert!(refused.is_err());
    assert_eq!(mux.status(), MuxStatus::Failed);

    // Nor one that spans pages: the first page with a completed packet is
    // still the last page.
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(opus_stream(4, head(4, 0))).expect("stream");
    mux.write_packet(&packet(track.get(), payload(90_000, 0), 960, 480))
        .expect("write");
    let refused = mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(48_000),
        ticks: 480,
    }]);
    assert!(refused.is_err());
}

#[test]
fn a_packet_across_three_pages_and_255_multiples_round_trip() {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux
        .add_stream(opus_stream(4, head(4, 0)))
        .expect("four streams");
    let big = payload(140_000, 3);
    let packets = [big.clone(), payload(255 * 3, 1), vec![TOC_20MS]];
    for (index, bytes) in packets.iter().enumerate() {
        let pts = i64::try_from(index).expect("i") * 960;
        mux.write_packet(&packet(track.get(), bytes.clone(), pts, 960))
            .expect("write");
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(48_000),
        ticks: 3 * 960,
    }])
    .expect("finish");
    let bytes = mux.into_inner().into_inner();
    let layout = pages(&bytes);
    // Two header pages, then the big packet over three pages.
    assert!(layout.len() >= 5);
    assert_eq!(layout[2].2.granule, -1);
    assert_eq!(layout[3].2.granule, -1);
    assert!(layout[3].2.flags & FLAG_CONTINUED != 0);
    assert!(layout[4].2.flags & FLAG_CONTINUED != 0);
    let mut demux = open(&bytes);
    let read = read_all(&mut demux);
    assert_eq!(read.len(), 3);
    assert_eq!(read[0].bytes(), big.as_slice());
    assert_eq!(read[1].bytes().len(), 765);
    assert_eq!(read[2].bytes(), &[TOC_20MS]);
}

#[test]
fn the_mux_groups_header_pages_and_numbers_pages_per_stream() {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let left = mux.add_stream(opus_stream(2, head(2, 312))).expect("a");
    let right = mux.add_stream(opus_stream(1, Vec::new())).expect("b");
    mux.set_config(right, &head(1, 120)).expect("config");
    for index in 0..300 {
        let pts = index * 960;
        mux.write_packet(&packet(left.get(), payload(200, 1), pts, 960))
            .expect("left");
        mux.write_packet(&packet(right.get(), payload(80, 2), pts, 960))
            .expect("right");
    }
    let done = mux
        .finish(&[
            TrackDuration {
                track: left,
                timescale: Timescale::new(48_000),
                ticks: 300 * 960,
            },
            TrackDuration {
                track: right,
                timescale: Timescale::new(48_000),
                ticks: 300 * 960,
            },
        ])
        .expect("finish");
    assert!(done.is_complete());
    assert!(done.interleaved());
    let bytes = mux.into_inner().into_inner();
    let layout = pages(&bytes);
    let flags: Vec<u8> = layout.iter().take(4).map(|page| page.2.flags).collect();
    assert_eq!(flags, vec![FLAG_BOS, FLAG_BOS, 0, 0]);
    assert_eq!(layout[0].2.serial, FIRST_SERIAL);
    assert_eq!(layout[1].2.serial, FIRST_SERIAL + 1);
    assert!(layout.iter().take(4).all(|page| page.2.granule == 0));
    for serial in [FIRST_SERIAL, FIRST_SERIAL + 1] {
        let mine: Vec<_> = layout
            .iter()
            .filter(|page| page.2.serial == serial)
            .collect();
        for (index, page) in mine.iter().enumerate() {
            assert_eq!(page.2.sequence, u32::try_from(index).expect("seq"));
        }
        let last = mine.last().expect("last");
        assert_eq!(last.2.flags & FLAG_EOS, FLAG_EOS);
        assert_eq!(last.2.granule, 300 * 960);
        // Page policy: a page closes at 4 KiB or one second.
        for page in &mine[2..] {
            assert!(page.1 < 4_096 + 27 + 255 + 300);
        }
        let mut previous = 0;
        for page in &mine[2..] {
            assert!(page.2.granule > previous);
            assert_eq!(page.2.granule % 960, 0);
            assert!(page.2.granule - previous <= 48_000);
            previous = page.2.granule;
        }
    }
    let mut demux = open(&bytes);
    assert_eq!(demux.streams().len(), 2);
    assert_eq!(demux.streams()[1].trim().encoder_delay(), Some(120));
    let packets = read_all(&mut demux);
    assert_eq!(packets.iter().filter(|p| p.stream() == 0).count(), 300);
    assert_eq!(packets.iter().filter(|p| p.stream() == 1).count(), 300);
    let landing = demux
        .seek(SeekTo {
            track: 1,
            time: 150 * 960,
            end: None,
        })
        .expect("seek");
    match landing.landing {
        Landing::Packet { time, preroll, .. } => {
            assert_eq!(time, 150 * 960 - 3_840);
            assert_eq!(preroll, 4);
        }
        Landing::End => panic!("end"),
    }
}

#[test]
fn every_seek_lands_before_the_preroll_and_returns_no_stale_packet() {
    let sizes: Vec<usize> = (0..400).map(|index| 60 + (index * 37) % 400).collect();
    let bytes = mux_one(312, 0, &sizes, 960);
    let mut demux = open(&bytes);
    let all = read_all(&mut demux);
    let end = 400 * 960;
    let consumed = demux.consumed();
    demux
        .seek(SeekTo {
            track: 0,
            time: 100_000,
            end: None,
        })
        .expect("seek");
    assert_eq!(demux.consumed(), consumed, "a seek is not a packet read");
    assert_eq!(
        demux.total_read(),
        consumed + demux.last_seek().bytes,
        "seek bytes are counted apart"
    );
    for target in [
        0,
        1,
        3_839,
        3_840,
        3_841,
        100_000,
        200_123,
        end - 961,
        end - 1,
    ] {
        let landing = demux
            .seek(SeekTo {
                track: 0,
                time: target,
                end: None,
            })
            .expect("seek");
        let Landing::Packet {
            time,
            preroll,
            keyframe,
            ..
        } = landing.landing
        else {
            panic!("{target} landed at the end");
        };
        assert!(keyframe);
        assert!(!landing.index_used);
        let aim = (target - 3_840).max(0);
        assert!(time <= aim && aim < time + 960, "{target} -> {time}");
        let expected = all
            .iter()
            .filter(|packet| {
                let pts = packet.timestamp().presentation().expect("pts");
                pts >= time && pts < target
            })
            .count();
        assert_eq!(usize::try_from(preroll).expect("p"), expected, "{target}");
        let next = demux.read_packet().expect("read").expect("packet");
        assert_eq!(next.timestamp().presentation(), Some(time));
        let index = usize::try_from(time / 960).expect("index");
        assert_eq!(next.bytes(), all[index].bytes());
        assert!(landing.payload_scans >= 1);
    }
    let past = demux
        .seek(SeekTo {
            track: 0,
            time: end,
            end: None,
        })
        .expect("past");
    assert_eq!(past.landing, Landing::End);
    assert!(demux.read_packet().expect("end").is_none());
    let back = demux
        .seek(SeekTo {
            track: 0,
            time: -50,
            end: None,
        })
        .expect("before start");
    assert!(matches!(
        back.landing,
        Landing::Packet {
            ordinal: 0,
            time: 0,
            preroll: 0,
            ..
        }
    ));
    assert_eq!(read_all(&mut demux).len(), 400);
}

#[test]
fn a_failed_seek_leaves_the_read_position() {
    let bytes = mux_one(312, 0, &[100; 20], 960);
    let mut demux = open(&bytes);
    let first = demux.read_packet().expect("read").expect("first");
    assert_eq!(first.timestamp().presentation(), Some(0));
    let error = demux
        .seek(SeekTo {
            track: 5,
            time: 0,
            end: None,
        })
        .expect_err("unknown track");
    assert_eq!(error, ReadError::Seek(SeekError::UnknownTrack));
    let next = demux.read_packet().expect("read").expect("second");
    assert_eq!(next.timestamp().presentation(), Some(960));
}

#[test]
fn cancel_stops_reads_and_seeks() {
    let bytes = mux_one(312, 0, &[100; 4], 960);
    let mut demux = open(&bytes);
    demux.cancel();
    assert_eq!(demux.read_packet(), Err(ReadError::Cancelled));
    assert_eq!(
        demux.seek(SeekTo {
            track: 0,
            time: 0,
            end: None
        }),
        Err(ReadError::Cancelled)
    );
}

#[test]
fn a_forward_input_reads_the_same_packets_and_refuses_seek() {
    let bytes = mux_one(312, 0, &[100, 200, 300], 700);
    let mut random = open(&bytes);
    let expected = read_all(&mut random);
    let mut forward = OggDemux::open_forward(Cursor::new(bytes)).expect("forward");
    let stream = forward.streams()[0].clone();
    assert_eq!(stream.trim().trailing_padding(), None);
    assert_eq!(stream.edits()[0].media_time(), Some(312));
    assert_eq!(stream.edits()[0].duration(), 0);
    assert_eq!(read_all(&mut forward), expected);
    assert_eq!(
        forward.seek(SeekTo {
            track: 0,
            time: 0,
            end: None
        }),
        Err(ReadError::Seek(SeekError::ForwardOnly))
    );
}

#[test]
fn garbage_before_the_first_page_is_skipped() {
    let mut bytes = b"ID3 not an ogg page".to_vec();
    bytes.extend(mux_one(312, 0, &[100, 100], 960));
    let mut demux = open(&bytes);
    assert_eq!(demux.skipped_bytes(), 19);
    assert_eq!(read_all(&mut demux).len(), 2);
}

#[test]
fn a_bad_checksum_drops_that_page_and_marks_the_gap() {
    let bytes = mux_one(312, 0, &[1_000; 30], 960);
    let layout = pages(&bytes);
    // Corrupt the body of the second audio page.
    let (offset, len, header) = layout[3];
    assert_eq!(header.sequence, 3);
    let mut broken = bytes.clone();
    broken[offset + len - 1] ^= 0xFF;
    let mut demux = open(&broken);
    let packets = read_all(&mut demux);
    assert_eq!(demux.crc_failures(), 1);
    assert_eq!(demux.lost_pages(), 1);
    assert!(packets.len() < 30);
    assert_eq!(demux.gaps().len(), 1);
    let gap = demux.gaps()[0];
    assert_eq!(gap.track(), TrackId::new(0));
    // Every packet still has its own time.
    for packet in &packets {
        let pts = packet.timestamp().presentation().expect("pts");
        let index = usize::try_from(pts / 960).expect("index");
        assert_eq!(
            packet.bytes(),
            payload(1_000, u8::try_from(index).expect("i"))
        );
    }
}

#[test]
fn a_page_sequence_gap_drops_the_spanning_packet() {
    // Four Opus streams allow 100,000-byte packets, which span two pages.
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(opus_stream(4, head(4, 0))).expect("s");
    for index in 0..6 {
        let seed = u8::try_from(index).expect("seed");
        mux.write_packet(&packet(
            track.get(),
            payload(100_000, seed),
            i64::from(seed) * 960,
            960,
        ))
        .expect("write");
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(48_000),
        ticks: 6 * 960,
    }])
    .expect("finish");
    let bytes = mux.into_inner().into_inner();
    let layout = pages(&bytes);
    // Page 4 holds the first part of packet 1; its second part is orphaned.
    assert_eq!(layout[4].2.granule, -1);
    let (offset, len, _) = layout[4];
    let mut missing = bytes[..offset].to_vec();
    missing.extend_from_slice(&bytes[offset + len..]);
    let mut demux = open(&missing);
    let packets = read_all(&mut demux);
    assert_eq!(demux.lost_pages(), 1);
    assert_eq!(demux.dropped_pieces(), 1);
    assert_eq!(packets.len(), 5);
    for packet in &packets {
        let pts = packet.timestamp().presentation().expect("pts");
        let index = u8::try_from(pts / 960).expect("index");
        assert_eq!(packet.bytes(), payload(100_000, index));
        assert_ne!(index, 1);
    }
    assert_eq!(
        demux.gaps(),
        &[cutlass_container::Gap::try_new(TrackId::new(0), 960, 960).expect("gap")]
    );
}

#[test]
fn a_truncated_last_page_is_an_error_after_the_complete_packets() {
    let bytes = mux_one(312, 0, &[500; 20], 960);
    let cut = &bytes[..bytes.len() - 100];
    let mut demux = open(cut);
    let mut count = 0;
    let error = loop {
        match demux.read_packet() {
            Ok(Some(_)) => count += 1,
            Ok(None) => panic!("the cut page was not reported"),
            Err(error) => break error,
        }
    };
    assert!(matches!(error, ReadError::Truncated { .. }));
    assert!(count > 0 && count < 20);
    assert!(demux.read_packet().expect("after").is_none());
}

#[test]
fn a_segment_table_past_the_end_is_truncation() {
    let mut bytes = mux_one(312, 0, &[100; 3], 960);
    let layout = pages(&bytes);
    let (offset, _, _) = layout[layout.len() - 1];
    bytes.truncate(offset + 27 + 2);
    let mut demux = open(&bytes);
    assert!(matches!(
        read_all_result(&mut demux),
        Err(ReadError::Truncated { .. })
    ));
}

fn read_all_result(demux: &mut OggDemux<Cursor<Vec<u8>>>) -> Result<Vec<Packet>, ReadError> {
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet()? {
        packets.push(packet);
    }
    Ok(packets)
}

#[test]
fn a_granule_that_goes_back_is_a_typed_read_error() {
    let bytes = crafted(
        &head(2, 312),
        &[
            (0, 1920, vec![payload(10, 0), payload(10, 1)]),
            (0, 1000, vec![payload(10, 2)]),
            (FLAG_EOS, 3840, vec![payload(10, 3)]),
        ],
    );
    let mut demux = open(&bytes);
    assert_eq!(
        demux.read_packet().expect("1").expect("p").bytes().len(),
        10
    );
    assert!(demux.read_packet().expect("2").is_some());
    let error = demux.read_packet().expect_err("backwards");
    assert!(matches!(error, ReadError::Source { .. }));
    assert!(matches!(
        demux.last_error(),
        Some(OggError::GranuleBackwards { granule: 1000, .. })
    ));
}

#[test]
fn a_first_granule_below_its_samples_fails_open_unless_the_page_ends_the_stream() {
    let invalid = crafted(&head(2, 312), &[(0, 900, vec![payload(10, 0)])]);
    assert!(matches!(
        try_open(&invalid),
        Err(OpenError::Source { message }) if message.contains("first granule")
    ));
    let short = crafted(
        &head(2, 312),
        &[(FLAG_EOS, 1_400, vec![payload(10, 0), payload(10, 1)])],
    );
    let mut demux = open(&short);
    assert_eq!(demux.opus(0).expect("info").start, Some(0));
    assert_eq!(demux.streams()[0].trim().trailing_padding(), Some(520));
    assert_eq!(demux.streams()[0].edits()[0].duration(), 1_400 - 312);
    assert_eq!(times(&read_all(&mut demux)), vec![(0, 960), (960, 440)]);
    let all_skipped = crafted(&head(2, 312), &[(FLAG_EOS, 200, vec![payload(10, 0)])]);
    assert!(matches!(
        try_open(&all_skipped),
        Err(OpenError::Source { .. })
    ));
}

#[test]
fn a_missing_or_garbled_opus_head_fails_open() {
    let not_head = crafted(b"OpusTags\0\0\0\0\0\0\0\0", &[]);
    assert!(matches!(
        try_open(&not_head),
        Err(OpenError::Source { message }) if message.contains("not OpusHead")
    ));
    let short = crafted(b"OpusHead\x01\x02", &[]);
    assert!(matches!(try_open(&short), Err(OpenError::Source { .. })));
    let mut future = head(2, 312);
    future[8] = 0x20;
    assert!(matches!(
        try_open(&crafted(&future, &[])),
        Err(OpenError::UnsupportedBrand { .. })
    ));
    // OpusHead must be alone on its page.
    let mut lacing = lacing_for(19);
    lacing.extend(lacing_for(3));
    let mut body = head(2, 0);
    body.extend_from_slice(b"abc");
    let crowded = encode(FLAG_BOS, 0, 1, 0, &lacing, &body);
    assert!(matches!(try_open(&crowded), Err(OpenError::Source { .. })));
}

#[test]
fn missing_or_garbled_opus_tags_fail_open() {
    let header = head(2, 312);
    let bos = encode(FLAG_BOS, 0, 9, 0, &lacing_for(header.len()), &header);
    assert!(matches!(
        try_open(&bos),
        Err(OpenError::Source { message }) if message.contains("no OpusTags")
    ));
    let mut audio_second = bos.clone();
    let audio = payload(10, 0);
    audio_second.extend(encode(0, 960, 9, 1, &lacing_for(10), &audio));
    assert!(matches!(
        try_open(&audio_second),
        Err(OpenError::Source { message }) if message.contains("not OpusTags")
    ));
    let mut hostile = b"OpusTags".to_vec();
    hostile.extend_from_slice(&u32::MAX.to_le_bytes());
    let mut huge_vendor = bos.clone();
    huge_vendor.extend(encode(0, 0, 9, 1, &lacing_for(hostile.len()), &hostile));
    assert!(matches!(
        try_open(&huge_vendor),
        Err(OpenError::Source { .. })
    ));
    // Tags that do not finish their page.
    let tags = OpusTags::default().encode();
    let mut lacing = lacing_for(tags.len());
    lacing.extend(lacing_for(audio.len()));
    let mut body = tags;
    body.extend_from_slice(&audio);
    let mut crowded = bos;
    crowded.extend(encode(0, 960, 9, 1, &lacing, &body));
    assert!(matches!(
        try_open(&crowded),
        Err(OpenError::Source { message }) if message.contains("finish its page")
    ));
}

#[test]
fn an_unknown_mapping_is_listed_and_its_pages_skipped() {
    let vorbis = b"\x01vorbis\0\0\0\0\x02\x44\xac\0\0".to_vec();
    let mut file = encode(FLAG_BOS, 0, 5, 0, &lacing_for(vorbis.len()), &vorbis);
    let header = head(1, 312);
    file.extend(encode(
        FLAG_BOS,
        0,
        6,
        0,
        &lacing_for(header.len()),
        &header,
    ));
    file.extend(encode(0, 0, 5, 1, &[3], b"xyz"));
    let tags = OpusTags::default().encode();
    file.extend(encode(0, 0, 6, 1, &lacing_for(tags.len()), &tags));
    file.extend(encode(0, 4410, 5, 2, &[4], b"data"));
    file.extend(encode(FLAG_EOS, 960, 6, 2, &[10], &payload(10, 0)));
    let mut demux = open(&file);
    assert_eq!(demux.streams().len(), 2);
    assert_eq!(
        demux.streams()[0].params().codec(),
        &StreamCodec::Unknown(UnknownCodec::new(b"ogg:vorbis".to_vec()).expect("tag"))
    );
    assert_eq!(demux.streams()[0].params().extradata(), vorbis.as_slice());
    assert!(!demux.streams()[0].timescale().is_clock());
    let packets = read_all(&mut demux);
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].stream(), 1);
    assert_eq!(
        demux.seek(SeekTo {
            track: 0,
            time: 0,
            end: None
        }),
        Err(ReadError::Seek(SeekError::Empty))
    );
}

#[test]
fn a_second_link_is_not_returned() {
    let first = mux_one(312, 0, &[100; 10], 960);
    let mut second = OggMux::with_first_serial(Cursor::new(Vec::new()), 99);
    let track = second.add_stream(opus_stream(1, head(1, 120))).expect("b");
    second
        .write_packet(&packet(track.get(), payload(50, 9), 0, 960))
        .expect("p");
    second
        .finish(&[TrackDuration {
            track,
            timescale: Timescale::new(48_000),
            ticks: 960,
        }])
        .expect("f");
    let mut file = first.clone();
    file.extend(second.into_inner().into_inner());
    let mut demux = open(&file);
    assert!(demux.chained());
    assert_eq!(
        demux.link_end(),
        Some(u64::try_from(first.len()).expect("len"))
    );
    assert_eq!(demux.opus(0).expect("info").end, Some(9_600));
    assert_eq!(read_all(&mut demux).len(), 10);
    let mut forward = OggDemux::open_forward(Cursor::new(file)).expect("forward");
    assert_eq!(read_all(&mut forward).len(), 10);
    assert!(forward.chained());
}

#[test]
fn hostile_sizes_are_refused_before_allocation() {
    // More logical streams than the limit.
    let mut many = Vec::new();
    for serial in 0..70 {
        let header = head(1, 0);
        many.extend(encode(
            FLAG_BOS,
            0,
            serial,
            0,
            &lacing_for(header.len()),
            &header,
        ));
    }
    assert!(
        matches!(try_open(&many), Err(OpenError::Source { message }) if message.contains("streams"))
    );
    // An audio packet above 61,440 bytes per Opus stream, over two pages.
    let big = payload(70_000, 1);
    let mut bytes = crafted(&head(2, 0), &[]);
    bytes.extend(encode(0, -1, 77, 2, &[255; 255], &big[..65_025]));
    let mut lacing = vec![255; 19];
    lacing.push(130);
    lacing.push(10);
    let mut body = big[65_025..].to_vec();
    body.extend(payload(10, 1));
    bytes.extend(encode(
        FLAG_EOS | FLAG_CONTINUED,
        1_920,
        77,
        3,
        &lacing,
        &body,
    ));
    let mut demux = open(&bytes);
    assert_eq!(
        demux.read_packet(),
        Err(ReadError::HostileSize {
            declared: 70_000,
            limit: 61_440
        })
    );
    let next = demux
        .read_packet()
        .expect("read")
        .expect("after the refusal");
    assert_eq!(next.timestamp().presentation(), Some(960));
    // A duplicate serial.
    let header = head(1, 0);
    let mut twice = encode(FLAG_BOS, 0, 3, 0, &lacing_for(header.len()), &header);
    twice.extend(encode(
        FLAG_BOS,
        0,
        3,
        0,
        &lacing_for(header.len()),
        &header,
    ));
    assert_eq!(
        try_open(&twice).err(),
        Some(OpenError::DuplicateTrack { id: 3 })
    );
}

#[test]
fn empty_and_foreign_inputs_do_not_open() {
    assert!(matches!(try_open(b""), Err(OpenError::Unrecognized { .. })));
    assert!(matches!(
        try_open(b"RIFF\0\0\0\0WAVEfmt "),
        Err(OpenError::Unrecognized { .. })
    ));
    let audio = encode(0, 960, 1, 5, &[10], &payload(10, 0));
    assert!(matches!(
        try_open(&audio),
        Err(OpenError::Unrecognized { .. })
    ));
    let whole = mux_one(312, 0, &[100], 960);
    assert!(matches!(
        try_open(&whole[..40]),
        Err(OpenError::Truncated { .. })
    ));
}

#[test]
fn a_stream_with_no_audio_opens_empty() {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(opus_stream(2, head(2, 312))).expect("s");
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(48_000),
        ticks: 0,
    }])
    .expect("finish");
    let bytes = mux.into_inner().into_inner();
    let mut demux = open(&bytes);
    assert_eq!(demux.opus(0).expect("info").start, None);
    assert_eq!(demux.streams()[0].edits()[0].duration(), 0);
    assert!(read_all(&mut demux).is_empty());
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: 0,
            end: None,
        })
        .expect("seek");
    assert_eq!(landing.landing, Landing::End);
}

#[test]
fn open_reads_the_header_and_the_tail_not_the_whole_file() {
    let sizes = vec![400; 3_000];
    let bytes = mux_one(312, 0, &sizes, 960);
    let demux = open(&bytes);
    assert!(bytes.len() > 1_000_000);
    assert!(
        demux.consumed() < 200 * 1024,
        "open read {}",
        demux.consumed()
    );
}

#[test]
fn the_mux_refuses_streams_and_packets_it_cannot_store() {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let aac = StreamParams::try_audio(StreamCodec::Known(CodecId::Aac), 48_000, 2, Vec::new())
        .expect("aac");
    let error = mux
        .add_stream(NewStream {
            params: aac,
            timescale: Timescale::new(48_000),
            language: None,
        })
        .expect_err("aac");
    assert!(matches!(error, MuxError::Rejected { message } if message.contains("Opus")));
    assert_eq!(mux.status(), MuxStatus::Failed);

    let mut unknown = OggMux::new(Cursor::new(Vec::new()));
    let tag = UnknownCodec::new(b"vorbis".to_vec()).expect("tag");
    let error = unknown
        .add_stream(NewStream {
            params: StreamParams::opaque(StreamCodec::Unknown(tag.clone()), Vec::new()),
            timescale: Timescale::new(48_000),
            language: None,
        })
        .expect_err("unknown");
    assert_eq!(error, MuxError::UnknownCodec(tag));

    let mut video = OggMux::new(Cursor::new(Vec::new()));
    let params = StreamParams::video(
        StreamCodec::Known(CodecId::Opus),
        FrameSize::try_new(2, 2).expect("size"),
        Vec::new(),
    );
    assert!(
        video
            .add_stream(NewStream {
                params,
                timescale: Timescale::new(48_000),
                language: None,
            })
            .is_err()
    );

    let mut rate = OggMux::new(Cursor::new(Vec::new()));
    let mut stream = opus_stream(2, head(2, 0));
    stream.timescale = Timescale::new(44_100);
    assert_eq!(
        rate.add_stream(stream).expect_err("rate"),
        MuxError::TimescaleMismatch {
            packet: 44_100,
            stream: 48_000
        }
    );

    let mut mismatch = OggMux::new(Cursor::new(Vec::new()));
    assert!(mismatch.add_stream(opus_stream(1, head(2, 0))).is_err());
}

#[test]
fn the_mux_needs_a_config_by_the_first_packet() {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(opus_stream(2, Vec::new())).expect("s");
    let error = mux
        .write_packet(&packet(track.get(), payload(10, 0), 0, 960))
        .expect_err("no config");
    assert!(matches!(error, MuxError::Rejected { message } if message.contains("OpusHead")));
    assert_eq!(mux.status(), MuxStatus::Failed);
    let failed = mux
        .finish(&[TrackDuration {
            track,
            timescale: Timescale::new(48_000),
            ticks: 0,
        }])
        .expect_err("failed");
    assert!(!failed.is_complete());
    assert!(failed.durations().is_none());
}

#[test]
fn the_mux_refuses_late_streams_late_configs_and_bad_timing() {
    let fresh = || {
        let mut mux = OggMux::new(Cursor::new(Vec::new()));
        let track = mux.add_stream(opus_stream(2, head(2, 0))).expect("s");
        mux.write_packet(&packet(track.get(), payload(10, 0), 0, 960))
            .expect("first");
        (mux, track)
    };
    let (mut mux, _) = fresh();
    assert!(mux.add_stream(opus_stream(2, head(2, 0))).is_err());
    let (mut mux, track) = fresh();
    assert!(mux.set_config(track, &head(2, 0)).is_err());
    let (mut mux, track) = fresh();
    assert!(
        mux.write_packet(&packet(track.get(), payload(10, 0), 0, 960))
            .is_err()
    );
    let (mut mux, track) = fresh();
    assert!(
        mux.write_packet(&packet(track.get(), payload(10, 0), 1_920, 960))
            .is_err()
    );
    let (mut mux, track) = fresh();
    assert!(
        mux.write_packet(&packet(track.get(), payload(10, 0), 960, 2_000))
            .is_err()
    );
    let (mut mux, track) = fresh();
    mux.write_packet(&packet(track.get(), payload(10, 0), 960, 500))
        .expect("trimmed");
    assert!(
        mux.write_packet(&packet(track.get(), payload(10, 0), 1_460, 960))
            .is_err()
    );
    let (mut mux, track) = fresh();
    assert!(
        mux.write_packet(&packet(track.get(), Vec::new(), 960, 0))
            .is_err()
    );
    let (mut mux, track) = fresh();
    let timeless = Packet::try_new(
        track.get(),
        payload(10, 0),
        Timestamp::missing(),
        Timescale::new(48_000),
        true,
    )
    .expect("p");
    assert_eq!(mux.write_packet(&timeless), Err(MuxError::MissingTimestamp));
    let (mut mux, track) = fresh();
    let error = mux
        .finish(&[TrackDuration {
            track,
            timescale: Timescale::new(48_000),
            ticks: 1_000,
        }])
        .expect_err("mismatch");
    assert_eq!(error.reason(), &MuxError::DurationMismatch { track: 0 });
    assert_eq!(mux.status(), MuxStatus::Failed);
}

#[test]
fn a_cancelled_or_finished_mux_does_not_report_success_again() {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(opus_stream(2, head(2, 0))).expect("s");
    mux.write_packet(&packet(track.get(), payload(10, 0), 0, 960))
        .expect("p");
    mux.cancel();
    assert_eq!(mux.status(), MuxStatus::Failed);
    let failed = mux
        .finish(&[TrackDuration {
            track,
            timescale: Timescale::new(48_000),
            ticks: 960,
        }])
        .expect_err("cancelled");
    assert_eq!(failed.reason(), &MuxError::Cancelled);
    assert!(
        mux.write_packet(&packet(track.get(), payload(10, 0), 960, 960))
            .is_err()
    );

    let mut done = OggMux::new(Cursor::new(Vec::new()));
    let track = done.add_stream(opus_stream(2, head(2, 0))).expect("s");
    let durations = [TrackDuration {
        track,
        timescale: Timescale::new(48_000),
        ticks: 0,
    }];
    done.finish(&durations).expect("finish");
    assert_eq!(done.status(), MuxStatus::Finished);
    assert_eq!(
        done.finish(&durations).expect_err("again").reason(),
        &MuxError::AlreadyFinished
    );
}

#[test]
fn an_mp4_dops_config_becomes_opus_head() {
    // dOps: version 0, 2 channels, pre-skip 312, 44100 Hz, gain 0, family 0.
    let mut dops = vec![0, 2];
    dops.extend_from_slice(&312_u16.to_be_bytes());
    dops.extend_from_slice(&44_100_u32.to_be_bytes());
    dops.extend_from_slice(&0_i16.to_be_bytes());
    dops.push(0);
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let track = mux.add_stream(opus_stream(2, Vec::new())).expect("s");
    mux.set_config(track, &dops).expect("dops");
    mux.write_packet(&packet(track.get(), payload(10, 0), 0, 960))
        .expect("p");
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(48_000),
        ticks: 960,
    }])
    .expect("finish");
    let demux = open(&mux.into_inner().into_inner());
    let info = demux.opus(0).expect("info");
    assert_eq!(info.head.pre_skip, 312);
    assert_eq!(info.head.input_rate, 44_100);
    assert_eq!(&demux.streams()[0].params().extradata()[..8], b"OpusHead");
}

#[test]
fn a_write_failure_fails_the_mux() {
    struct Full;
    impl std::io::Write for Full {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut mux = OggMux::new(Full);
    let track = mux.add_stream(opus_stream(2, head(2, 0))).expect("s");
    let error = mux
        .write_packet(&packet(track.get(), payload(10, 0), 0, 960))
        .expect_err("write");
    assert!(matches!(error, MuxError::Rejected { message } if message.contains("disk full")));
    assert_eq!(mux.status(), MuxStatus::Failed);
}
