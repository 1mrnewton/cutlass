//! Round trips through this crate's muxer and demuxer, seeking, damage,
//! and hostile input. Every file is built in memory.
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

use std::io::Cursor;

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{
    Demux, Landing, Mux, NewStream, OpenError, ReadError, SeekError, SeekTo, TrackDuration,
};
use cutlass_mpegts::crc::crc32;
use cutlass_mpegts::{MuxOptions, TsDemux, TsMux};

const DELAY: i64 = 126_000;
/// libx264's SPS for High profile 1920x1080 and its PPS.
const SPS: &[u8] = &[
    0x67, 0x64, 0x00, 0x28, 0xAC, 0xD9, 0x40, 0x78, 0x02, 0x27, 0xE5, 0xC0, 0x44, 0x00, 0x00, 0x03,
    0x00, 0x04, 0x00, 0x00, 0x03, 0x00, 0xC8, 0x3C, 0x60, 0xC6, 0x58,
];
const PPS: &[u8] = &[0x68, 0xEB, 0xE3, 0xCB, 0x22, 0xC0];
/// AAC LC, 48 kHz, stereo.
const ASC: &[u8] = &[0x11, 0x90];

/// One access unit written, as it should read back.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Unit {
    track: u32,
    bytes: Vec<u8>,
    pts: i64,
    dts: i64,
    key: bool,
}

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    /// Bytes with no zero, so no start code appears by chance.
    fn body(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| (self.next() % 255 + 1) as u8).collect()
    }
}

fn nal(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0, 0, 0, 1];
    out.extend_from_slice(bytes);
    out
}

/// A picture: AUD, the parameter sets on a keyframe, then a slice with
/// `first_mb_in_slice` 0 (`0x88`: ue 0, then slice data).
fn picture(rng: &mut Rng, key: bool, len: usize) -> Vec<u8> {
    let mut out = nal(&[0x09, 0xF0]);
    if key {
        out.extend(nal(SPS));
        out.extend(nal(PPS));
    }
    let mut slice = vec![if key { 0x65 } else { 0x41 }, 0x88];
    slice.extend(rng.body(len));
    out.extend(nal(&slice));
    out
}

/// `seconds` of 25 fps H.264 (a keyframe each second, PTS one frame after
/// DTS) and 48 kHz AAC, interleaved by decode time, starting at `base`
/// ticks of the 90 kHz clock.
fn content(seconds: i64, base: i64) -> Vec<Unit> {
    let mut rng = Rng(0x1234_5678);
    let mut video: Vec<Unit> = (0..seconds * 25)
        .map(|index| {
            let key = index % 25 == 0;
            let len = if key {
                3000
            } else {
                200 + (index as usize * 37) % 900
            };
            Unit {
                track: 0,
                bytes: picture(&mut rng, key, len),
                dts: base + index * 3600,
                pts: base + index * 3600 + 3600,
                key,
            }
        })
        .collect();
    let frames = seconds * 48_000 / 1024;
    let audio: Vec<Unit> = (0..frames)
        .map(|index| Unit {
            track: 1,
            bytes: rng.body(150 + (index as usize * 13) % 200),
            pts: base + index * 1920,
            dts: base + index * 1920,
            key: true,
        })
        .collect();
    video.extend(audio);
    video.sort_by_key(|unit| (unit.dts, unit.track));
    video
}

fn mux(units: &[Unit], options: MuxOptions) -> Vec<u8> {
    mux_into(units, TsMux::with_options(Vec::new(), options))
}

fn mux_into(units: &[Unit], mut mux: TsMux<Vec<u8>>) -> Vec<u8> {
    let size = cutlass_codec::FrameSize::try_new(1920, 1080).unwrap();
    let video = mux
        .add_stream(NewStream {
            params: StreamParams::video(StreamCodec::Known(CodecId::H264), size, Vec::new()),
            timescale: Timescale::new(90_000),
            language: None,
        })
        .unwrap();
    let audio = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Aac),
                48_000,
                2,
                ASC.to_vec(),
            )
            .unwrap(),
            timescale: Timescale::new(90_000),
            language: cutlass_container::Language::new("eng").ok(),
        })
        .unwrap();
    for unit in units {
        let track = if unit.track == 0 { video } else { audio };
        let packet = Packet::try_new(
            track.get(),
            unit.bytes.clone(),
            Timestamp::try_new(Some(unit.dts), Some(unit.pts), None).unwrap(),
            Timescale::new(90_000),
            unit.key,
        )
        .unwrap();
        mux.write_packet(&packet).unwrap();
    }
    let durations: [TrackDuration; 0] = [];
    mux.finish(&durations).unwrap();
    mux.into_inner()
}

fn open(bytes: &[u8]) -> TsDemux<Cursor<Vec<u8>>> {
    TsDemux::open(Cursor::new(bytes.to_vec()), bytes.len() as u64).expect("open")
}

/// Every packet left, as `Unit`s with the demuxer's track ids mapped back
/// to 0 (video) and 1 (audio).
fn drain(demux: &mut TsDemux<Cursor<Vec<u8>>>) -> Vec<Unit> {
    let ids: Vec<u32> = demux
        .streams()
        .iter()
        .map(|stream| stream.id().get())
        .collect();
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        let track = ids.iter().position(|&id| id == packet.stream()).unwrap() as u32;
        let time = packet.timestamp();
        out.push(Unit {
            track,
            bytes: packet.bytes().to_vec(),
            pts: time.presentation().unwrap(),
            dts: time.decode().unwrap(),
            key: packet.keyframe(),
        });
    }
    out
}

/// The units of one track, in order.
fn track(units: &[Unit], track: u32) -> Vec<Unit> {
    units
        .iter()
        .filter(|unit| unit.track == track)
        .cloned()
        .collect()
}

fn shifted(units: &[Unit]) -> Vec<Unit> {
    units
        .iter()
        .map(|unit| Unit {
            pts: unit.pts + DELAY,
            dts: unit.dts + DELAY,
            ..unit.clone()
        })
        .collect()
}

#[test]
fn every_access_unit_reads_back_from_ts_and_m2ts() {
    let units = content(4, 0);
    for options in [
        MuxOptions::default(),
        MuxOptions {
            packet_size: cutlass_mpegts::packet::PacketSize::M2ts,
            ..MuxOptions::default()
        },
    ] {
        let bytes = mux(&units, options);
        let mut demux = open(&bytes);
        assert_eq!(demux.packet_size(), options.packet_size);
        let streams = demux.streams();
        assert_eq!(streams.len(), 2);
        assert_eq!(
            streams[0]
                .params()
                .size()
                .map(|size| (size.width(), size.height())),
            Some((1920, 1080))
        );
        assert_eq!(streams[1].params().sample_rate(), Some(48_000));
        assert_eq!(streams[1].params().channels(), Some(2));
        assert_eq!(streams[1].params().extradata(), ASC);
        assert_eq!(
            streams[1].language().map(|language| language.as_str()),
            Some("eng")
        );
        assert_eq!(streams[0].edits()[0].media_time(), Some(DELAY));
        let read = drain(&mut demux);
        let expected = shifted(&units);
        for which in [0, 1] {
            let got = track(&read, which);
            let want = track(&expected, which);
            assert_eq!(got.len(), want.len(), "track {which}");
            for (index, (got, want)) in got.iter().zip(&want).enumerate() {
                assert_eq!(got, want, "track {which} unit {index}");
            }
        }
        // Video ends last: its last picture is shown at 4 s and lasts a
        // frame; 187 audio frames end at 3.989 s.
        let end = DELAY + 100 * 3600 + 3600;
        assert_eq!(demux.end(), Some(end));
        assert_eq!(streams_duration(&demux), end - DELAY);
        let counters = demux.counters();
        assert_eq!(counters, cutlass_mpegts::Counters::default());
    }
}

fn streams_duration(demux: &TsDemux<Cursor<Vec<u8>>>) -> i64 {
    demux.streams()[0].edits()[0].duration()
}

#[test]
fn the_service_description_names_the_program() {
    let units = content(4, 0);
    let bytes = mux(&units, MuxOptions::default());
    let service = open(&bytes).programs()[0].service.clone().expect("an SDT");
    assert_eq!(
        (
            service.program,
            service.provider.as_str(),
            service.name.as_str()
        ),
        (1, "Cutlass", "Service01")
    );
    // An SDT every 500 ms of the 4 s, and one at the start.
    let sdts = packets(&bytes)
        .iter()
        .filter(|packet| ((u16::from(packet[1] & 0x1F) << 8) | u16::from(packet[2])) == 0x11)
        .count();
    assert!((8..=10).contains(&sdts), "{sdts} SDT packets");

    let named =
        TsMux::with_options(Vec::new(), MuxOptions::default()).with_service("Prov", "Café ☕");
    let bytes = mux_into(&units, named);
    let service = open(&bytes).programs()[0].service.clone().expect("an SDT");
    assert_eq!(
        (service.provider.as_str(), service.name.as_str()),
        ("Prov", "Café ☕")
    );
    // A name past what the descriptor holds is cut at a character boundary.
    let long = "é".repeat(100);
    let bytes = mux_into(&units, TsMux::new(Vec::new()).with_service(&long, &long));
    let service = open(&bytes).programs()[0].service.clone().expect("an SDT");
    assert_eq!(service.name, "é".repeat(62));
    let read = drain(&mut open(&bytes));
    for which in [0, 1] {
        assert!(track(&read, which) == track(&shifted(&units), which));
    }
}

#[test]
fn a_forward_input_reads_the_same_and_cannot_seek() {
    let units = content(3, 0);
    let bytes = mux(&units, MuxOptions::default());
    let mut random = open(&bytes);
    let mut forward = TsDemux::open_forward(Cursor::new(bytes.clone())).expect("forward open");
    assert_eq!(forward.streams()[0].edits()[0].duration(), 0);
    assert_eq!(drain(&mut forward), drain(&mut random));
    let error = forward
        .seek(SeekTo {
            track: 0x100,
            time: 0,
            end: None,
        })
        .unwrap_err();
    assert_eq!(error, ReadError::Seek(SeekError::ForwardOnly));
}

/// Seeks the video of `bytes` to nine targets: each lands on the last
/// keyframe not after the target, counts the preroll, and the video then
/// follows in order.
fn check_video_seeks(bytes: &[u8], expected: &[Unit]) -> TsDemux<Cursor<Vec<u8>>> {
    let keys: Vec<i64> = expected
        .iter()
        .filter(|unit| unit.track == 0 && unit.key)
        .map(|unit| unit.pts)
        .collect();
    let first_key = keys[0];
    let mut demux = open(bytes);
    let end = demux.end().unwrap();
    for target in [
        0,
        DELAY,
        first_key,
        first_key + 1,
        keys[3],
        keys[3] - 1,
        keys[5] + 40_000,
        keys[9] + 3600 * 20,
        end - 1,
    ] {
        let landing = demux
            .seek(SeekTo {
                track: 0x100,
                time: target,
                end: None,
            })
            .expect("seek");
        let want = keys
            .iter()
            .rev()
            .find(|&&key| key <= target)
            .copied()
            .unwrap_or(first_key);
        let Landing::Packet {
            time,
            keyframe,
            preroll,
            ..
        } = landing.landing
        else {
            panic!("target {target} landed at the end");
        };
        assert_eq!(time, want, "target {target}");
        assert!(keyframe);
        let first = demux.read_packet().unwrap().unwrap();
        assert_eq!(first.stream(), 0x100, "target {target}");
        assert_eq!(
            first.timestamp().presentation(),
            Some(want),
            "target {target}"
        );
        assert!(first.keyframe());
        // Preroll counts the video units from the landing shown before the
        // target.
        let video_before = expected
            .iter()
            .filter(|unit| {
                unit.track == 0 && unit.dts >= want - 3600 && unit.pts >= want && unit.pts < target
            })
            .count() as u32;
        assert_eq!(preroll, video_before, "target {target}");
        // The rest of the video follows in order.
        let rest: Vec<_> = std::iter::from_fn(|| demux.read_packet().unwrap())
            .filter(|packet| packet.stream() == 0x100)
            .map(|packet| packet.timestamp().presentation().unwrap())
            .collect();
        let want_rest: Vec<i64> = expected
            .iter()
            .filter(|unit| unit.track == 0 && unit.pts > want)
            .map(|unit| unit.pts)
            .collect();
        assert_eq!(rest, want_rest, "target {target}");
    }
    demux
}

/// Sets or clears the random access indicator of PID 0x100's packets that
/// start a PES packet and have an adaptation field to hold it.
fn mark_random_access(bytes: &mut [u8], set: bool) -> usize {
    let mut changed = 0;
    for packet in bytes.as_chunks_mut::<188>().0 {
        let pid = u16::from_be_bytes([packet[1] & 0x1F, packet[2]]);
        let start = packet[1] & 0x40 != 0;
        let adaptation = packet[3] & 0x20 != 0 && packet[4] > 0;
        if pid == 0x100 && start && adaptation {
            let before = packet[5];
            packet[5] = if set { before | 0x40 } else { before & !0x40 };
            changed += usize::from(before != packet[5]);
        }
    }
    changed
}

/// The indicator only speeds a seek up: without it, and with it on
/// pictures that are not keyframes, seeks land where they did.
#[test]
fn seeks_do_not_depend_on_the_random_access_indicator() {
    let units = content(10, 0);
    let expected = shifted(&units);
    let mut cleared = mux(&units, MuxOptions::default());
    assert!(mark_random_access(&mut cleared, false) > 0);
    check_video_seeks(&cleared, &expected);
    let mut false_marks = mux(&units, MuxOptions::default());
    assert!(mark_random_access(&mut false_marks, true) > 0);
    check_video_seeks(&false_marks, &expected);
    // Every other keyframe unmarked: a seek past an unmarked keyframe must
    // still land on it, not on the marked one before.
    let mut some_marks = mux(&units, MuxOptions::default());
    let mut keyframe = 0;
    for packet in some_marks.as_chunks_mut::<188>().0 {
        let video_start = pid(packet) == 0x100 && packet[1] & 0x40 != 0;
        if video_start && packet[3] & 0x20 != 0 && packet[4] > 0 && packet[5] & 0x40 != 0 {
            if keyframe % 2 == 1 {
                packet[5] &= !0x40;
            }
            keyframe += 1;
        }
    }
    assert!(keyframe > 4);
    check_video_seeks(&some_marks, &expected);
}

#[test]
fn seeks_land_on_the_last_keyframe_not_after_the_target() {
    let units = content(10, 0);
    let bytes = mux(&units, MuxOptions::default());
    let expected = shifted(&units);
    let mut demux = check_video_seeks(&bytes, &expected);
    let end = demux.end().unwrap();
    let past = demux
        .seek(SeekTo {
            track: 0x100,
            time: end,
            end: None,
        })
        .unwrap();
    assert_eq!(past.landing, Landing::End);
    assert!(demux.read_packet().unwrap().is_none());
    // An audio seek lands on the frame holding the target.
    let target = DELAY + 5 * 90_000 + 100;
    let landing = demux
        .seek(SeekTo {
            track: 0x101,
            time: target,
            end: None,
        })
        .unwrap();
    let Landing::Packet { time, .. } = landing.landing else {
        panic!("audio seek landed at the end");
    };
    assert!(time <= target && target < time + 1920, "{time}");
    let cost = demux.last_seek();
    assert!(cost.probes <= 64, "{cost:?}");
}

#[test]
fn a_failed_seek_keeps_the_read_position() {
    let units = content(3, 0);
    let bytes = mux(&units, MuxOptions::default());
    let mut demux = open(&bytes);
    let mut reference = open(&bytes);
    for _ in 0..40 {
        demux.read_packet().unwrap();
        reference.read_packet().unwrap();
    }
    let error = demux
        .seek(SeekTo {
            track: 7,
            time: DELAY,
            end: None,
        })
        .unwrap_err();
    assert_eq!(error, ReadError::Seek(SeekError::UnknownTrack));
    assert_eq!(drain(&mut demux), drain(&mut reference));
}

#[test]
fn a_cut_file_reads_a_prefix() {
    let units = content(3, 0);
    let bytes = mux(&units, MuxOptions::default());
    let whole = drain(&mut open(&bytes));
    for cut in [
        bytes.len() / 2,
        bytes.len() - 100,
        bytes.len() / 3 + 7,
        188 * 9 + 50,
    ] {
        let Ok(mut demux) = TsDemux::open(Cursor::new(bytes[..cut].to_vec()), cut as u64) else {
            continue;
        };
        let read = drain(&mut demux);
        for which in [0, 1] {
            let got = track(&read, which);
            let all = track(&whole, which);
            // Every unit but the last one cut is whole.
            let keep = got.len().saturating_sub(1);
            assert_eq!(got[..keep], all[..keep], "cut {cut} track {which}");
        }
    }
}

/// Splits a TS file into packets.
fn packets(bytes: &[u8]) -> Vec<&[u8]> {
    bytes.chunks(188).collect()
}

fn pid(packet: &[u8]) -> u16 {
    u16::from(packet[1] & 0x1F) << 8 | u16::from(packet[2])
}

#[test]
fn a_lost_packet_is_counted_and_drops_only_its_unit() {
    let units = content(3, 0);
    let bytes = mux(&units, MuxOptions::default());
    let all = packets(&bytes);
    // The second transport packet of a video PES packet several frames in.
    let starts: Vec<usize> = (0..all.len())
        .filter(|&index| pid(all[index]) == 0x100 && all[index][1] & 0x40 != 0)
        .collect();
    let lost = starts[30] + 1;
    assert_eq!(pid(all[lost]), 0x100);
    let damaged: Vec<u8> = all
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != lost)
        .flat_map(|(_, packet)| packet.to_vec())
        .collect();
    let mut demux = open(&damaged);
    let read = drain(&mut demux);
    let counters = demux.counters();
    assert_eq!(counters.cc_errors, 1);
    assert_eq!(counters.dropped_pes, 1);
    let video = track(&read, 0);
    let want = track(&shifted(&units), 0);
    assert_eq!(video.len(), want.len() - 1);
    assert_eq!(track(&read, 1), track(&shifted(&units), 1));
}

#[test]
fn a_new_pmt_version_is_counted_and_the_streams_stay() {
    let units = content(4, 0);
    let mut bytes = mux(&units, MuxOptions::default());
    let half = bytes.len() / 188 / 2;
    let mut changed = 0;
    for (index, packet) in bytes.as_chunks_mut::<188>().0.iter_mut().enumerate() {
        if index < half || pid(packet) != 0x1000 {
            continue;
        }
        // The PMT starts after the pointer field: version 1 from here on.
        let section = &mut packet[5..];
        let length = usize::from(section[1] & 0x0F) << 8 | usize::from(section[2]);
        section[5] = 0xC3;
        let crc = crc32(&section[..3 + length - 4]);
        section[3 + length - 4..3 + length].copy_from_slice(&crc.to_be_bytes());
        changed += 1;
    }
    assert!(changed > 10, "{changed} PMT packets");
    let mut demux = open(&bytes);
    let read = drain(&mut demux);
    assert_eq!(demux.counters().pmt_changes, 1);
    for which in [0, 1] {
        assert!(track(&read, which) == track(&shifted(&units), which));
    }
}

#[test]
fn a_204_byte_stream_reads_as_its_188_byte_packets() {
    let units = content(3, 0);
    let bytes = mux(&units, MuxOptions::default());
    // Each packet followed by 16 bytes of Reed-Solomon parity, as DVB
    // and ATSC transmissions carry them; the bytes are not checked.
    let mut fec = Vec::with_capacity(bytes.len() / 188 * 204);
    for (index, packet) in packets(&bytes).iter().enumerate() {
        fec.extend_from_slice(packet);
        fec.extend((0..16).map(|byte| (index * 7 + byte) as u8));
    }
    let mut demux = open(&fec);
    assert_eq!(drain(&mut demux), drain(&mut open(&bytes)));
    assert_eq!(demux.counters(), cutlass_mpegts::Counters::default());
    // Seeks land the same.
    let mut plain = open(&bytes);
    for target in [0, 90_000, 200_000] {
        let to = || SeekTo {
            track: 0x100,
            time: DELAY + target,
            end: None,
        };
        // The same packet, at its own byte position in each.
        let landed = |landing: Landing| match landing {
            Landing::Packet {
                ordinal,
                time,
                keyframe,
                preroll,
                ..
            } => Some((ordinal, time, keyframe, preroll)),
            _ => None,
        };
        let fec_landing = landed(demux.seek(to()).unwrap().landing);
        assert!(fec_landing.is_some());
        assert_eq!(fec_landing, landed(plain.seek(to()).unwrap().landing));
        let (fec, plain) = (demux.read_packet().unwrap(), plain.read_packet().unwrap());
        assert_eq!(
            fec.map(|packet| packet.timestamp()),
            plain.map(|packet| packet.timestamp())
        );
    }
}

#[test]
fn a_duplicate_packet_is_skipped() {
    let units = content(2, 0);
    let bytes = mux(&units, MuxOptions::default());
    let all = packets(&bytes);
    let twice = (0..all.len())
        .find(|&index| pid(all[index]) == 0x100 && all[index][1] & 0x40 == 0)
        .unwrap();
    let mut doubled = Vec::new();
    for (index, packet) in all.iter().enumerate() {
        doubled.extend_from_slice(packet);
        if index == twice {
            doubled.extend_from_slice(packet);
        }
    }
    let mut demux = open(&doubled);
    assert_eq!(drain(&mut demux), drain(&mut open(&bytes)));
    assert_eq!(demux.counters().duplicates, 1);
    assert_eq!(demux.counters().cc_errors, 0);
}

#[test]
fn garbage_between_packets_is_skipped_and_the_grid_found_again() {
    let units = content(2, 0);
    let bytes = mux(&units, MuxOptions::default());
    let mut damaged = bytes[..188 * 200].to_vec();
    // Junk that does not start with a sync byte but ends with three.
    damaged.extend_from_slice(&[0x11; 997]);
    damaged.extend_from_slice(&[0x47; 3]);
    damaged.extend_from_slice(&bytes[188 * 200..]);
    let mut demux = open(&damaged);
    let read = drain(&mut demux);
    assert!(demux.resyncs() >= 1);
    assert_eq!(demux.skipped_bytes(), 1000);
    assert_eq!(track(&read, 1).len(), track(&units, 1).len());
}

#[test]
fn timestamps_across_the_33_bit_wrap_keep_rising() {
    let wrap = 1_i64 << 33;
    let base = wrap - DELAY - 2 * 90_000;
    let units = content(4, base);
    let bytes = mux(&units, MuxOptions::default());
    let mut demux = open(&bytes);
    let read = drain(&mut demux);
    let video = track(&read, 0);
    assert!(
        video
            .windows(2)
            .all(|pair| pair[1].dts == pair[0].dts + 3600)
    );
    assert!(video.last().unwrap().dts > wrap);
    assert_eq!(video.len(), 100);
    let start = demux.start();
    assert!(start < wrap && start > wrap - 3 * 90_000, "{start}");
    assert!(demux.end().unwrap() > wrap);
    // A seek past the wrap lands past it.
    let landing = demux
        .seek(SeekTo {
            track: 0x100,
            time: wrap + 90_000,
            end: None,
        })
        .unwrap();
    let Landing::Packet { time, .. } = landing.landing else {
        panic!("landed at the end");
    };
    assert!(time > wrap && time <= wrap + 90_000, "{time}");
}

/// One transport packet of a PSI section on `pid`.
fn psi_packet(pid: u16, table: u8, id: u16, body: &[u8], continuity: u8) -> Vec<u8> {
    let length = body.len() + 9;
    let mut section = vec![
        table,
        0xB0 | (length >> 8) as u8,
        length as u8,
        (id >> 8) as u8,
        id as u8,
        0xC1,
        0,
        0,
    ];
    section.extend_from_slice(body);
    let crc = crc32(&section);
    section.extend_from_slice(&crc.to_be_bytes());
    let mut packet = vec![
        0x47,
        0x40 | (pid >> 8) as u8,
        pid as u8,
        0x10 | continuity,
        0,
    ];
    packet.extend_from_slice(&section);
    packet.resize(188, 0xFF);
    packet
}

/// The muxed program re-announced as the second of two programs; the first
/// has only a stream this crate does not frame.
fn two_programs() -> Vec<u8> {
    let bytes = mux(&content(2, 0), MuxOptions::default());
    let pat = [0x00, 0x01, 0xE0, 0x30, 0x00, 0x02, 0xE0, 0x31];
    let pmt1 = [0xE0, 0x40, 0xF0, 0x00, 0x05, 0xE0, 0x40, 0xF0, 0x00];
    let pmt2 = [
        0xE1, 0x00, 0xF0, 0x00, 0x1B, 0xE1, 0x00, 0xF0, 0x00, 0x0F, 0xE1, 0x01, 0xF0, 0x00,
    ];
    let mut out = Vec::new();
    let mut continuity = 0;
    for (index, packet) in packets(&bytes).into_iter().enumerate() {
        if index % 50 == 0 {
            out.extend(psi_packet(0, 0x00, 1, &pat, continuity));
            out.extend(psi_packet(0x30, 0x02, 1, &pmt1, continuity));
            out.extend(psi_packet(0x31, 0x02, 2, &pmt2, continuity));
            continuity = (continuity + 1) & 0x0F;
        }
        if matches!(pid(packet), 0 | 0x1000) {
            continue;
        }
        out.extend_from_slice(packet);
    }
    out
}

#[test]
fn the_first_program_with_a_framed_stream_is_chosen() {
    let bytes = two_programs();
    let mut demux = open(&bytes);
    assert_eq!(demux.program(), 2);
    let programs = demux.programs();
    assert_eq!(programs.len(), 2);
    assert_eq!(programs[0].number, 1);
    assert!(programs[0].pmt.is_some());
    assert_eq!(demux.streams().len(), 2);
    assert_eq!(track(&drain(&mut demux), 0).len(), 50);

    let only = TsDemux::open_program(Cursor::new(bytes.clone()), bytes.len() as u64, 1).unwrap();
    assert_eq!(only.program(), 1);
    assert_eq!(only.streams().len(), 1);
    assert!(matches!(
        only.streams()[0].params().codec(),
        StreamCodec::Unknown(_)
    ));
    let missing = TsDemux::open_program(Cursor::new(bytes.clone()), bytes.len() as u64, 9);
    assert!(matches!(missing, Err(OpenError::Source { .. })));
}

#[test]
fn input_that_is_not_a_transport_stream_is_refused() {
    for bytes in [Vec::new(), vec![0; 5000], b"RIFF....WAVEfmt ".repeat(100)] {
        let result = TsDemux::open(Cursor::new(bytes.clone()), bytes.len() as u64);
        assert!(
            matches!(result, Err(OpenError::Unrecognized { .. })),
            "{:?}",
            result.err()
        );
    }
    // Synced packets with no PAT.
    let mut synced = Vec::new();
    for _ in 0..100 {
        let mut packet = vec![0x47, 0x01, 0x00, 0x10];
        packet.resize(188, 0xAA);
        synced.extend(packet);
    }
    let result = TsDemux::open(Cursor::new(synced.clone()), synced.len() as u64);
    assert!(matches!(result, Err(OpenError::Source { .. })));
}

#[test]
fn hostile_packets_do_not_stall_or_grow_without_bound() {
    let good = mux(&content(1, 0), MuxOptions::default());
    let head: Vec<u8> = packets(&good)
        .into_iter()
        .filter(|packet| matches!(pid(packet), 0 | 0x1000))
        .take(2)
        .flat_map(<[u8]>::to_vec)
        .collect();
    let mut rng = Rng(77);
    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
    // Every packet on the video PID only an adaptation field of 183 bytes.
    let mut endless = head.clone();
    for index in 0..2000_u32 {
        let mut packet = vec![0x47, 0x01, 0x00, 0x20 | (index & 15) as u8, 183];
        packet.resize(188, 0xFF);
        endless.extend(packet);
    }
    cases.push(("endless adaptation", endless));
    // One video PES packet that never ends, 200k packets of payload.
    let mut unbounded = head.clone();
    unbounded.extend_from_slice(&[0x47, 0x41, 0x00, 0x10]);
    unbounded.extend_from_slice(&[0, 0, 1, 0xE0, 0, 0, 0x80, 0, 0]);
    unbounded.resize(unbounded.len() + 175, 0x55);
    for index in 1..200_000_u32 {
        let mut packet = vec![0x47, 0x01, 0x00, 0x10 | (index & 15) as u8];
        packet.resize(188, 0x55);
        unbounded.extend(packet);
    }
    cases.push(("unbounded PES", unbounded));
    // PES lengths that lie both ways, headers longer than the packet.
    let mut lies = head.clone();
    for index in 0..500_u32 {
        let mut packet = vec![0x47, 0x41, 0x01, 0x10 | (index & 15) as u8];
        let length = if index % 2 == 0 { 0xFFFF_u16 } else { 1 };
        packet.extend_from_slice(&[0, 0, 1, 0xC0]);
        packet.extend_from_slice(&length.to_be_bytes());
        packet.extend_from_slice(&[0x80, 0x80, 0xFF]);
        packet.extend(rng.body(175));
        lies.extend(packet);
    }
    cases.push(("PES length lies", lies));
    // Sections whose length runs past every cap, on the PAT PID.
    let mut sections = Vec::new();
    for index in 0..500_u32 {
        let mut packet = vec![
            0x47,
            0x40,
            0x00,
            0x10 | (index & 15) as u8,
            0,
            0x00,
            0xBF,
            0xFF,
        ];
        packet.extend(rng.body(180));
        sections.extend(packet);
    }
    cases.push(("section length overflow", sections));
    // Every PID in turn.
    let mut flood = head.clone();
    for index in 0..8192_u32 {
        let mut packet = vec![0x47, 0x40 | (index >> 8) as u8, index as u8, 0x10];
        packet.extend_from_slice(&[0, 0, 1, 0xE0, 0, 0, 0x80, 0, 0]);
        packet.extend(rng.body(175));
        flood.extend(packet);
    }
    cases.push(("PID flood", flood));
    for (name, bytes) in cases {
        let Ok(mut demux) = TsDemux::open(Cursor::new(bytes.clone()), bytes.len() as u64) else {
            continue;
        };
        let mut count = 0;
        while let Ok(Some(packet)) = demux.read_packet() {
            assert!(packet.bytes().len() <= 16 << 20, "{name}");
            count += 1;
            assert!(count < 1_000_000, "{name}");
        }
        let _ = demux.seek(SeekTo {
            track: 0x100,
            time: DELAY + 45_000,
            end: None,
        });
        eprintln!("{name}: {count} packets, {:?}", demux.counters());
    }
}

#[test]
fn the_muxer_refuses_what_it_cannot_write() {
    let mut mux = TsMux::new(Vec::new());
    let unknown = cutlass_codec::UnknownCodec::new(b"zzzz".to_vec()).unwrap();
    assert!(
        mux.add_stream(NewStream {
            params: StreamParams::opaque(StreamCodec::Unknown(unknown), Vec::new()),
            timescale: Timescale::new(90_000),
            language: None,
        })
        .is_err()
    );
    let mut mux = TsMux::new(Vec::new());
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Aac),
                48_000,
                2,
                Vec::new(),
            )
            .unwrap(),
            timescale: Timescale::new(48_000),
            language: None,
        })
        .unwrap();
    // Raw AAC without an AudioSpecificConfig cannot become ADTS.
    let packet = Packet::try_new(
        track.get(),
        vec![1, 2, 3],
        Timestamp::try_new(Some(0), Some(0), None).unwrap(),
        Timescale::new(48_000),
        true,
    )
    .unwrap();
    assert!(mux.write_packet(&packet).is_err());
}

#[test]
fn other_timescales_are_moved_to_the_90_khz_clock() {
    let mut mux = TsMux::new(Vec::new());
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Aac),
                44_100,
                2,
                vec![0x12, 0x10],
            )
            .unwrap(),
            timescale: Timescale::new(44_100),
            language: None,
        })
        .unwrap();
    for index in 0..200_i64 {
        let packet = Packet::try_new(
            track.get(),
            vec![0x21; 100],
            Timestamp::try_new(
                Some(index * 1024 - 2048),
                Some(index * 1024 - 2048),
                Some(1024),
            )
            .unwrap(),
            Timescale::new(44_100),
            true,
        )
        .unwrap();
        mux.write_packet(&packet).unwrap();
    }
    let durations: [TrackDuration; 0] = [];
    mux.finish(&durations).unwrap();
    let bytes = mux.into_inner();
    let mut demux = open(&bytes);
    let mut times = Vec::new();
    while let Some(packet) = demux.read_packet().unwrap() {
        assert_eq!(packet.bytes(), &[0x21; 100]);
        times.push(packet.timestamp().presentation().unwrap());
    }
    assert_eq!(times.len(), 200);
    for (index, time) in times.iter().enumerate() {
        let samples = index as i64 * 1024 - 2048;
        let want = DELAY + (samples * 90_000 * 2 + 44_100).div_euclid(2 * 44_100);
        assert_eq!(*time, want, "frame {index}");
    }
}

/// An encoder that reports only presentation times: pictures in decode
/// order I P B B P B B, each given a decode time equal to its presentation
/// time. Refused as it is; with derived decode times every picture reads
/// back at its presentation time, decode times rise and are never after
/// the presentation, and the audio decodes when it shows.
#[test]
fn decode_times_are_derived_from_presentation_times() {
    let mut rng = Rng(0x0BAD_F00D);
    let mut units = Vec::new();
    for group in 0..17_i64 {
        let order: &[i64] = if group == 0 { &[0] } else { &[2, 0, 1] };
        for &offset in order {
            let frame = if group == 0 {
                0
            } else {
                group * 3 - 2 + offset
            };
            let pts = frame * 3600;
            units.push(Unit {
                track: 0,
                bytes: picture(&mut rng, frame % 24 == 0, 300),
                pts,
                dts: pts,
                key: frame % 24 == 0,
            });
        }
    }
    for index in 0..80 {
        units.push(Unit {
            track: 1,
            bytes: rng.body(200),
            pts: index * 1920,
            dts: index * 1920,
            key: true,
        });
    }
    let refused = std::panic::catch_unwind(|| mux(&units, MuxOptions::default()));
    assert!(refused.is_err(), "decode times that go back are refused");

    let bytes = mux(
        &units,
        MuxOptions {
            derive_decode_times: true,
            ..MuxOptions::default()
        },
    );
    let mut demux = open(&bytes);
    let read = drain(&mut demux);
    let video = track(&read, 0);
    let sent = track(&units, 0);
    assert_eq!(video.len(), sent.len());
    let reorder = {
        let mut rbsp = Vec::new();
        cutlass_mpegts::es::bits::unescape(SPS, SPS.len(), &mut rbsp);
        cutlass_mpegts::es::h264::reorder_frames(&rbsp[1..]).expect("reorder")
    };
    assert!(reorder >= 1, "this pattern needs one picture of reorder");
    let mut sorted: Vec<i64> = sent.iter().map(|unit| unit.pts + DELAY).collect();
    sorted.sort_unstable();
    for (index, (got, want)) in video.iter().zip(&sent).enumerate() {
        assert_eq!(got.pts, want.pts + DELAY, "picture {index}");
        assert_eq!(got.bytes, want.bytes, "picture {index}");
        assert!(
            got.dts <= got.pts,
            "picture {index}: {} > {}",
            got.dts,
            got.pts
        );
        if let Some(rank) = index.checked_sub(reorder as usize) {
            assert_eq!(got.dts, sorted[rank], "picture {index}");
        }
    }
    assert!(video.windows(2).all(|pair| pair[0].dts < pair[1].dts));
    for unit in track(&read, 1) {
        assert_eq!(unit.dts, unit.pts);
    }
}

/// Audio frames that follow each other share PES packets of at most 2930
/// bytes, and still read back one by one at their own times.
#[test]
fn audio_frames_share_pes_packets() {
    let units = content(4, 0);
    let bytes = mux(&units, MuxOptions::default());
    let starts = packets(&bytes)
        .into_iter()
        .filter(|packet| pid(packet) == 0x101 && packet[1] & 0x40 != 0)
        .count();
    let frames = track(&units, 1).len();
    // Frames of 150 to 350 bytes: eight or more fit in 2930.
    assert!(
        starts * 5 < frames,
        "{starts} PES packets for {frames} frames"
    );
    let mut demux = open(&bytes);
    assert_eq!(track(&drain(&mut demux), 1), track(&shifted(&units), 1));
}
