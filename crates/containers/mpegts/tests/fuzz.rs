//! In-process random-mutation campaigns over the demuxer and the muxer.
//!
//! cargo-fuzz is not used: it is not installed here, and libfuzzer-sys
//! builds C++ through `cc`, which this crate's rules do not allow. These
//! campaigns are deterministic (a fixed xorshift generator per seed), so a
//! failure names the seed and iteration that reproduce it.
//!
//! Demux: the files from FFmpeg and GStreamer in `tests/fixtures`, the same
//! streams written again by [`TsMux`] as TS and M2TS, and a longer file of
//! this crate's whose clock wraps past 2^33, are mutated. Bytes are
//! flipped, written, inserted, deleted, cut, spliced and duplicated; whole
//! packets are dropped, repeated and swapped; header fields (PID, flags,
//! continuity, adaptation field, PCR), PES headers (length, flags,
//! timestamps) and table fields (lengths, stream types, PIDs, descriptors)
//! are hit. Half of the mutants get their table CRCs repaired so the
//! mutation reaches the PAT and PMT parsers behind the CRC. Each mutant is
//! probed, opened random-access, forward and per program, read to the end,
//! and seeked. Every call must return within a bounded number of steps and
//! packets must be no larger than the input.
//!
//! Mux: random stream setups and packet sequences over every codec the
//! muxer carries, including invalid ones. A refused call must leave the mux
//! failed. Sessions of well-formed H.264 and AAC must read back as exactly
//! the accepted packets, with times moved by the mux delay.
//!
//! `fuzz_seeds_*` run on every `cargo test`. The long campaigns are
//! `#[ignore]`; `TS_FUZZ_SECS` sets their length (default 60 s).
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

use std::io::Cursor;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use cutlass_codec::{
    CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp, UnknownCodec,
};
use cutlass_container::{
    Demux, Landing, Mux, MuxStatus, NewStream, SeekTo, TrackDuration, TrackId,
};
use cutlass_mpegts::crc::crc32;
use cutlass_mpegts::{MuxOptions, TsDemux, TsMux, probe};

const DELAY: i64 = 126_000;
const WRAP: i64 = 1 << 33;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
        }
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    fn byte(&mut self) -> u8 {
        u8::try_from(self.next() & 0xFF).unwrap()
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.byte()).collect()
    }

    /// Up to `max` random bytes.
    fn noise(&mut self, max: usize) -> Vec<u8> {
        let len = self.below(max);
        self.bytes(len)
    }

    /// Bytes with no zero, so no start code appears by chance.
    fn body(&mut self, len: usize) -> Vec<u8> {
        (0..len)
            .map(|_| u8::try_from(self.next() % 255 + 1).unwrap())
            .collect()
    }
}

const FIXTURES: [&[u8]; 8] = [
    include_bytes!("fixtures/gst_h264_aac.ts"),
    include_bytes!("fixtures/h264_aac.ts"),
    include_bytes!("fixtures/h264_eac3.m2ts"),
    include_bytes!("fixtures/h264_latm.ts"),
    include_bytes!("fixtures/h264_opus.ts"),
    include_bytes!("fixtures/hevc_ac3.ts"),
    include_bytes!("fixtures/mpeg2_mp2.ts"),
    include_bytes!("fixtures/two_programs.ts"),
];

/// libx264's SPS for High profile 1920x1080 and its PPS.
const SPS: &[u8] = &[
    0x67, 0x64, 0x00, 0x28, 0xAC, 0xD9, 0x40, 0x78, 0x02, 0x27, 0xE5, 0xC0, 0x44, 0x00, 0x00, 0x03,
    0x00, 0x04, 0x00, 0x00, 0x03, 0x00, 0xC8, 0x3C, 0x60, 0xC6, 0x58,
];
const PPS: &[u8] = &[0x68, 0xEB, 0xE3, 0xCB, 0x22, 0xC0];
/// AAC LC, 48 kHz, stereo.
const ASC: &[u8] = &[0x11, 0x90];

fn nal(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0, 0, 0, 1];
    out.extend_from_slice(bytes);
    out
}

/// A picture: AUD, the parameter sets on a keyframe, then a slice with
/// `first_mb_in_slice` 0.
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

fn avcc() -> Vec<u8> {
    let mut out = vec![1, SPS[1], SPS[2], SPS[3], 0xFF, 0xE1];
    out.extend_from_slice(&u16::try_from(SPS.len()).unwrap().to_be_bytes());
    out.extend_from_slice(SPS);
    out.push(1);
    out.extend_from_slice(&u16::try_from(PPS.len()).unwrap().to_be_bytes());
    out.extend_from_slice(PPS);
    out
}

fn video_stream(codec: StreamCodec, extradata: Vec<u8>, timescale: u32) -> NewStream {
    NewStream {
        params: StreamParams::video(codec, FrameSize::try_new(1920, 1080).unwrap(), extradata),
        timescale: Timescale::new(timescale),
        language: None,
    }
}

fn audio_stream(
    codec: StreamCodec,
    rate: u32,
    channels: u16,
    extradata: Vec<u8>,
    timescale: u32,
) -> NewStream {
    NewStream {
        params: StreamParams::try_audio(codec, rate, channels, extradata).expect("audio params"),
        timescale: Timescale::new(timescale),
        language: None,
    }
}

fn packet(
    track: u32,
    bytes: Vec<u8>,
    dts: Option<i64>,
    pts: Option<i64>,
    timescale: u32,
    key: bool,
) -> Option<Packet> {
    let time = Timestamp::try_new(dts, pts, None).ok()?;
    Packet::try_new(track, bytes, time, Timescale::new(timescale), key).ok()
}

/// A fixture written again by this crate's muxer, leaving out what the
/// muxer does not carry.
fn remux(bytes: &[u8], options: MuxOptions) -> Option<Vec<u8>> {
    let mut demux = TsDemux::open(Cursor::new(bytes.to_vec()), bytes.len() as u64).ok()?;
    let mut mux = TsMux::with_options(Vec::new(), options);
    let mut map = Vec::new();
    for stream in demux.streams() {
        let carried = matches!(
            stream.params().codec(),
            StreamCodec::Known(
                CodecId::H264
                    | CodecId::Hevc
                    | CodecId::Mpeg2
                    | CodecId::Mpeg4
                    | CodecId::Aac
                    | CodecId::Mp3
                    | CodecId::Ac3
                    | CodecId::Eac3
                    | CodecId::Opus
            )
        );
        if carried {
            let track = mux
                .add_stream(NewStream {
                    params: stream.params().clone(),
                    timescale: stream.timescale(),
                    language: stream.language().cloned(),
                })
                .expect("a carried stream");
            map.push((stream.id().get(), track));
        }
    }
    while let Some(source) = demux.read_packet().expect("fixture read") {
        let Some((_, track)) = map.iter().find(|(id, _)| *id == source.stream()) else {
            continue;
        };
        let time = source.timestamp();
        if time.presentation().is_none() && time.decode().is_none() {
            continue;
        }
        let packet = Packet::try_new(
            track.get(),
            source.bytes().to_vec(),
            time,
            source.timescale(),
            source.keyframe(),
        )
        .expect("packet");
        mux.write_packet(&packet).expect("remux write");
    }
    let durations: [TrackDuration; 0] = [];
    mux.finish(&durations).ok()?;
    Some(mux.into_inner())
}

/// Two seconds of 25 fps H.264 and 48 kHz AAC whose clock passes 2^33.
fn wrapping_file() -> Vec<u8> {
    let mut rng = Rng::new(7);
    let options = MuxOptions {
        delay: 0,
        ..MuxOptions::default()
    };
    let mut mux = TsMux::with_options(Vec::new(), options);
    let video = mux
        .add_stream(video_stream(
            StreamCodec::Known(CodecId::H264),
            Vec::new(),
            90_000,
        ))
        .unwrap();
    let audio = mux
        .add_stream(audio_stream(
            StreamCodec::Known(CodecId::Aac),
            48_000,
            2,
            ASC.to_vec(),
            90_000,
        ))
        .unwrap();
    let base = WRAP - 90_000;
    let mut units = Vec::new();
    for index in 0..50 {
        let key = index % 12 == 0;
        let dts = base + index * 3600;
        let bytes = picture(&mut rng, key, if key { 2000 } else { 300 });
        units.push((dts, video.get(), bytes, dts + 7200, key));
    }
    for index in 0..93 {
        let dts = base + index * 1920;
        units.push((dts, audio.get(), rng.body(120), dts, true));
    }
    units.sort_by_key(|unit| (unit.0, unit.1));
    for (dts, track, bytes, pts, key) in units {
        mux.write_packet(&packet(track, bytes, Some(dts), Some(pts), 90_000, key).unwrap())
            .unwrap();
    }
    let durations: [TrackDuration; 0] = [];
    mux.finish(&durations).unwrap();
    mux.into_inner()
}

fn corpus() -> Vec<Vec<u8>> {
    let mut files: Vec<Vec<u8>> = FIXTURES.iter().map(|bytes| bytes.to_vec()).collect();
    for fixture in FIXTURES {
        files.extend(remux(fixture, MuxOptions::default()));
        files.extend(remux(
            fixture,
            MuxOptions {
                packet_size: cutlass_mpegts::packet::PacketSize::M2ts,
                ..MuxOptions::default()
            },
        ));
    }
    files.push(wrapping_file());
    files
}

/// The packet stride and where the 188-byte packet starts in it.
fn layout(bytes: &[u8]) -> (usize, usize) {
    let m2ts = bytes.len() >= 196 + 4 && bytes[4] == 0x47 && bytes[196] == 0x47 && bytes[0] != 0x47;
    if m2ts { (192, 4) } else { (188, 0) }
}

/// Where the payload of the packet at `at` starts, if it has one.
fn payload_start(packet: &[u8]) -> Option<usize> {
    let control = (packet[3] >> 4) & 3;
    let mut at = 4;
    if control & 2 != 0 {
        at += 1 + usize::from(packet[4]);
    }
    (control & 1 != 0 && at < 188).then_some(at)
}

/// Offsets of the packets that start a PSI section (table 0 or 2).
fn sections(bytes: &[u8]) -> Vec<(usize, usize)> {
    let (stride, lead) = layout(bytes);
    let mut out = Vec::new();
    let mut at = lead;
    while at + 188 <= bytes.len() {
        let packet = &bytes[at..at + 188];
        if packet[0] == 0x47
            && packet[1] & 0x40 != 0
            && let Some(start) = payload_start(packet)
        {
            let section = start + 1 + usize::from(packet[start]);
            if section + 3 <= 188
                && matches!(packet[section], 0x00 | 0x02)
                && packet[section + 1] & 0x80 != 0
            {
                out.push((at, section));
            }
        }
        at += stride;
    }
    out
}

/// Rewrites the CRC of every table section that fits in its packet.
fn repair_crcs(bytes: &mut [u8]) {
    for (packet, section) in sections(bytes) {
        let start = packet + section;
        let length = usize::from(u16::from_be_bytes([
            bytes[start + 1] & 0x0F,
            bytes[start + 2],
        ]));
        let end = start + 3 + length;
        if length < 4 || end > packet + 188 {
            continue;
        }
        let crc = crc32(&bytes[start..end - 4]);
        bytes[end - 4..end].copy_from_slice(&crc.to_be_bytes());
    }
}

/// Offsets of the packets that start a PES packet.
fn pes_starts(bytes: &[u8]) -> Vec<usize> {
    let (stride, lead) = layout(bytes);
    let mut out = Vec::new();
    let mut at = lead;
    while at + 188 <= bytes.len() {
        let packet = &bytes[at..at + 188];
        if packet[0] == 0x47
            && packet[1] & 0x40 != 0
            && let Some(start) = payload_start(packet)
            && start + 14 <= 188
            && packet[start..start + 3] == [0, 0, 1]
        {
            out.push(at + start);
        }
        at += stride;
    }
    out
}

fn mutate(rng: &mut Rng, seed: &[u8], corpus: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = seed.to_vec();
    let rounds = 1 + rng.below(8);
    for _ in 0..rounds {
        let len = bytes.len();
        let (stride, lead) = layout(&bytes);
        let packets = len.saturating_sub(lead) / stride;
        let pick = |rng: &mut Rng| lead + rng.below(packets) * stride;
        match rng.below(16) {
            0 if len > 0 => {
                let at = rng.below(len);
                bytes[at] ^= 1 << rng.below(8);
            }
            1 if len > 0 => {
                let at = rng.below(len);
                bytes[at] = [0, 0xFF, 0x47, 0x80, rng.byte()][rng.below(5)];
            }
            2 => {
                let at = rng.below(len + 1);
                let count = 1 + rng.below(400);
                let insert = rng.bytes(count);
                bytes.splice(at..at, insert);
            }
            3 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(1024)).min(len);
                bytes.drain(at..end);
            }
            4 => bytes.truncate(rng.below(len + 1)),
            5 if len > 0 => {
                let other = &corpus[rng.below(corpus.len())];
                if !other.is_empty() {
                    let from = rng.below(other.len());
                    let take = (1 + rng.below(8192)).min(other.len() - from);
                    let at = rng.below(len);
                    bytes.splice(at..at, other[from..from + take].iter().copied());
                }
            }
            6 if packets > 0 => {
                // Drop, repeat or move whole packets.
                let at = pick(rng) - lead;
                let count = (1 + rng.below(4)) * stride;
                let end = (at + count).min(len);
                match rng.below(3) {
                    0 => {
                        bytes.drain(at..end);
                    }
                    1 => {
                        let copy = bytes[at..end].to_vec();
                        bytes.splice(at..at, copy);
                    }
                    _ => {
                        let copy: Vec<u8> = bytes.drain(at..end).collect();
                        let to = lead
                            + rng.below(bytes.len().saturating_sub(lead) / stride + 1) * stride
                            - lead;
                        let to = to.min(bytes.len());
                        bytes.splice(to..to, copy);
                    }
                }
            }
            7 if packets > 0 => {
                // A transport header field.
                let at = pick(rng);
                match rng.below(7) {
                    0 => bytes[at + 1] ^= 0x80, // error indicator
                    1 => bytes[at + 1] ^= 0x40, // unit start
                    2 => bytes[at + 3] = (bytes[at + 3] & 0x3F) | (rng.byte() & 0xC0), // scrambling
                    3 => bytes[at + 3] = (bytes[at + 3] & 0xCF) | (rng.byte() & 0x30), // adaptation control
                    4 => bytes[at + 3] = (bytes[at + 3] & 0xF0) | (rng.byte() & 0x0F), // continuity
                    5 => {
                        // A PID: random, or another packet's.
                        let pid = if rng.chance(50) {
                            [rng.byte() & 0x1F, rng.byte()]
                        } else {
                            let other = pick(rng);
                            [bytes[other + 1] & 0x1F, bytes[other + 2]]
                        };
                        bytes[at + 1] = (bytes[at + 1] & 0xE0) | pid[0];
                        bytes[at + 2] = pid[1];
                    }
                    _ => bytes[at] = rng.byte(), // sync byte
                }
            }
            8 if packets > 0 => {
                // The adaptation field: its length, flags, PCR.
                let at = pick(rng);
                bytes[at + 3] |= 0x20;
                match rng.below(4) {
                    0 => bytes[at + 4] = rng.byte(),
                    1 => bytes[at + 5] = rng.byte(),
                    2 => {
                        bytes[at + 4] = bytes[at + 4].max(7);
                        bytes[at + 5] |= 0x10;
                        for offset in 6..12 {
                            bytes[at + offset] = rng.byte();
                        }
                    }
                    _ => bytes[at + 5] |= 0x80, // discontinuity
                }
            }
            9 | 10 => {
                // A PES header field.
                let starts = pes_starts(&bytes);
                if let Some(&at) = starts.get(rng.below(starts.len())) {
                    match rng.below(6) {
                        0 => {
                            bytes[at + 4] = rng.byte();
                            bytes[at + 5] = rng.byte();
                        }
                        1 => bytes[at + 3] = rng.byte(), // stream id
                        2 => bytes[at + 7] = rng.byte(), // PTS and DTS flags
                        3 => bytes[at + 8] = rng.byte(), // header length
                        4 => {
                            for offset in 9..14 {
                                bytes[at + offset] = rng.byte();
                            }
                        }
                        _ => bytes[at + 9 + rng.below(5)] ^= 1 << rng.below(8),
                    }
                }
            }
            11 | 12 => {
                // A table field, its CRC repaired below most of the time.
                let found = sections(&bytes);
                if let Some(&(packet, section)) = found.get(rng.below(found.len())) {
                    let start = packet + section;
                    let room = packet + 188 - start;
                    match rng.below(5) {
                        0 => {
                            bytes[start + 1] = (bytes[start + 1] & 0xF0) | (rng.byte() & 0x0F);
                            bytes[start + 2] = rng.byte();
                        }
                        1 if room > 13 => {
                            // A PMT stream type.
                            let at = start + 12 + rng.below(room - 12);
                            bytes[at] = [
                                0x01,
                                0x02,
                                0x03,
                                0x04,
                                0x06,
                                0x0F,
                                0x11,
                                0x1B,
                                0x24,
                                0x81,
                                0x84,
                                0x87,
                                0xA1,
                                rng.byte(),
                            ][rng.below(14)];
                        }
                        2 if room > 8 => {
                            // A PID or a length in the body.
                            let at = start + 8 + rng.below(room - 8);
                            bytes[at] = rng.byte();
                        }
                        3 => bytes[start] = [0x00, 0x02, 0x42, 0xFF][rng.below(4)], // table id
                        _ => {
                            bytes[start + 5] = rng.byte(); // version, current, section numbers
                            bytes[start + 6] = rng.byte();
                        }
                    }
                }
            }
            _ => {
                if len > 0 {
                    let at = rng.below(len);
                    bytes[at] = bytes[at].wrapping_add(1);
                }
            }
        }
    }
    if rng.chance(50) {
        repair_crcs(&mut bytes);
    }
    bytes
}

/// Reads to the end and checks the invariants; returns the packets read.
fn drain(demux: &mut TsDemux<Cursor<Vec<u8>>>, input_len: usize, label: &str) -> usize {
    let ids: Vec<u32> = demux
        .streams()
        .iter()
        .map(|stream| stream.id().get())
        .collect();
    let cap = 2 * input_len + 64;
    let mut count = 0;
    for _ in 0..cap {
        match demux.read_packet() {
            Ok(Some(packet)) => {
                count += 1;
                assert!(
                    packet.bytes().len() <= input_len,
                    "{label}: packet larger than input"
                );
                assert!(
                    ids.contains(&packet.stream()),
                    "{label}: packet of an unknown track"
                );
                if let Some(duration) = packet.timestamp().duration() {
                    assert!(duration >= 0, "{label}: negative duration");
                }
            }
            Ok(None) => return count,
            Err(_) => {}
        }
    }
    panic!("{label}: read_packet did not reach the end within {cap} calls");
}

/// `intact` inputs are unmutated corpus files: a seek there must land on
/// a keyframe at or before the target, and the next packet of the track is
/// the landing one.
fn exercise_demux(rng: &mut Rng, bytes: &[u8], intact: bool, label: &str) {
    let _ = probe(&bytes[..bytes.len().min(1024)]);
    let len = bytes.len() as u64;
    let mut programs = Vec::new();
    if let Ok(mut demux) = TsDemux::open(Cursor::new(bytes.to_vec()), len) {
        programs = demux
            .programs()
            .iter()
            .map(|program| program.number)
            .collect();
        let ids: Vec<u32> = demux
            .streams()
            .iter()
            .map(|stream| stream.id().get())
            .collect();
        for _ in 0..rng.below(4) {
            let _ = demux.read_packet();
        }
        let start = demux.start();
        for _ in 0..3 {
            if ids.is_empty() {
                break;
            }
            let track = if rng.chance(90) {
                ids[rng.below(ids.len())]
            } else {
                u32::from(rng.byte())
            };
            let time = start + i64::try_from(rng.below(400_000)).unwrap() - 20_000;
            let before = demux.consumed();
            match demux.seek(SeekTo {
                track,
                time,
                end: demux.end(),
            }) {
                Ok(outcome) => {
                    if let Landing::Packet { time: landed, .. } = outcome.landing
                        && intact
                    {
                        for _ in 0..4096 {
                            match demux.read_packet() {
                                Ok(Some(next)) if next.stream() == track => {
                                    assert!(next.keyframe(), "{label}: landing keyframe");
                                    assert_eq!(
                                        next.timestamp().presentation(),
                                        Some(landed),
                                        "{label}: landing packet"
                                    );
                                    break;
                                }
                                Ok(Some(_)) => {}
                                Ok(None) | Err(_) => break,
                            }
                        }
                    }
                }
                Err(_) => {
                    assert_eq!(demux.consumed(), before, "{label}: failed seek read");
                }
            }
        }
        let _ = drain(&mut demux, bytes.len(), label);
        let _ = demux.gaps();
        let _ = demux.counters();
    }
    for number in programs.into_iter().take(4) {
        if let Ok(mut demux) = TsDemux::open_program(Cursor::new(bytes.to_vec()), len, number) {
            let _ = drain(&mut demux, bytes.len(), label);
        }
    }
    if let Ok(mut demux) = TsDemux::open_forward(Cursor::new(bytes.to_vec())) {
        let _ = drain(&mut demux, bytes.len(), label);
    }
}

fn shared_corpus() -> &'static [Vec<u8>] {
    static CORPUS: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    CORPUS.get_or_init(corpus)
}

/// Runs `iterations` mutants from `seed` and returns the slowest one's time.
fn demux_campaign(seed: u64, iterations: usize) -> Duration {
    let corpus = shared_corpus();
    let mut rng = Rng::new(seed);
    let mut slowest = Duration::ZERO;
    for iteration in 0..iterations {
        let base = &corpus[rng.below(corpus.len())];
        if rng.chance(5) {
            exercise_demux(
                &mut rng,
                base,
                true,
                &format!("seed {seed} intact {iteration}"),
            );
        }
        let mutant = mutate(&mut rng, base, corpus);
        let started = Instant::now();
        exercise_demux(
            &mut rng,
            &mutant,
            false,
            &format!("seed {seed} iteration {iteration}"),
        );
        slowest = slowest.max(started.elapsed());
    }
    slowest
}

/// What a session's track is, so packets fit it (or deliberately do not).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    H264,
    Aac,
    Other(CodecId),
}

struct SessionTrack {
    id: TrackId,
    shape: Shape,
    timescale: u32,
    clock: i64,
    keyed: bool,
}

/// One packet the session expects back, in 90 kHz ticks.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Expected {
    bytes: Vec<u8>,
    dts: i64,
    pts: i64,
    key: bool,
}

fn to_clock(ticks: i64, timescale: u32) -> i64 {
    let rate = i128::from(timescale);
    i64::try_from((i128::from(ticks) * 90_000 * 2 + rate).div_euclid(2 * rate)).unwrap() + DELAY
}

/// A random stream to add: a clean one when `clean`, else anything.
fn session_stream(rng: &mut Rng, clean: bool) -> (NewStream, Shape, u32) {
    let timescale = if clean {
        [90_000, 1000, 48_000, 30_000][rng.below(4)]
    } else {
        [90_000, 1000, 48_000, 1, 0, 25, u32::MAX][rng.below(7)]
    };
    let roll = if clean { rng.below(2) } else { rng.below(12) };
    match roll {
        0 => {
            // A configuration record makes the muxer read packets as
            // length-prefixed, so clean Annex B packets go without one.
            let extradata = match rng.below(if clean { 1 } else { 4 }) {
                0 => Vec::new(),
                1 => avcc(),
                2 => rng.noise(40),
                _ => {
                    let mut cut = avcc();
                    cut.truncate(rng.below(cut.len()));
                    cut
                }
            };
            (
                video_stream(StreamCodec::Known(CodecId::H264), extradata, timescale),
                Shape::H264,
                timescale,
            )
        }
        1 => {
            let extradata = if clean || rng.chance(60) {
                ASC.to_vec()
            } else {
                rng.noise(6)
            };
            let rate = if clean {
                48_000
            } else {
                [48_000, 44_100, 8000, 96_000][rng.below(4)]
            };
            (
                audio_stream(
                    StreamCodec::Known(CodecId::Aac),
                    rate,
                    2,
                    extradata,
                    timescale,
                ),
                Shape::Aac,
                timescale,
            )
        }
        2 => (
            video_stream(StreamCodec::Known(CodecId::Hevc), rng.noise(60), timescale),
            Shape::Other(CodecId::Hevc),
            timescale,
        ),
        3 => (
            video_stream(StreamCodec::Known(CodecId::Mpeg2), rng.noise(30), timescale),
            Shape::Other(CodecId::Mpeg2),
            timescale,
        ),
        4 => (
            audio_stream(
                StreamCodec::Known(CodecId::Mp3),
                [48_000, 22_050][rng.below(2)],
                2,
                Vec::new(),
                timescale,
            ),
            Shape::Other(CodecId::Mp3),
            timescale,
        ),
        5 => (
            audio_stream(
                StreamCodec::Known(CodecId::Ac3),
                48_000,
                2,
                Vec::new(),
                timescale,
            ),
            Shape::Other(CodecId::Ac3),
            timescale,
        ),
        6 => (
            audio_stream(
                StreamCodec::Known(CodecId::Eac3),
                48_000,
                6,
                Vec::new(),
                timescale,
            ),
            Shape::Other(CodecId::Eac3),
            timescale,
        ),
        7 => {
            let channels = u16::from(rng.byte() % 9);
            let mut head = b"OpusHead".to_vec();
            head.extend_from_slice(&[
                1,
                u8::try_from(channels).unwrap(),
                0x38,
                1,
                0x80,
                0xBB,
                0,
                0,
                0,
                0,
            ]);
            head.push(u8::from(channels > 2));
            if channels > 2 {
                head.push(u8::try_from(channels).unwrap());
                head.push(rng.byte() % 5);
                head.extend((0..channels).map(|value| u8::try_from(value).unwrap()));
            }
            if rng.chance(20) {
                head.truncate(rng.below(head.len()));
            }
            (
                audio_stream(
                    StreamCodec::Known(CodecId::Opus),
                    48_000,
                    channels.max(1),
                    head,
                    timescale,
                ),
                Shape::Other(CodecId::Opus),
                timescale,
            )
        }
        8 => (
            audio_stream(
                StreamCodec::Unknown(UnknownCodec::new(b"tag").unwrap()),
                48_000,
                2,
                Vec::new(),
                timescale,
            ),
            Shape::Other(CodecId::Pcm),
            timescale,
        ),
        9 => {
            // MPEG-4 Part 2 is carried; VP9 is refused.
            let codec = [CodecId::Mpeg4, CodecId::Vp9][rng.below(2)];
            (
                video_stream(StreamCodec::Known(codec), rng.noise(20), timescale),
                Shape::Other(codec),
                timescale,
            )
        }
        _ => (
            audio_stream(
                StreamCodec::Known(CodecId::Pcm),
                48_000,
                2,
                Vec::new(),
                timescale,
            ),
            Shape::Other(CodecId::Pcm),
            timescale,
        ),
    }
}

/// The bytes of the next packet on `track`, and whether it is a keyframe.
fn session_payload(rng: &mut Rng, track: &mut SessionTrack, clean: bool) -> (Vec<u8>, bool) {
    match track.shape {
        Shape::H264 => {
            let key = !track.keyed || rng.chance(10);
            track.keyed = true;
            if clean || rng.chance(70) {
                let len = if rng.chance(3) {
                    60_000 + rng.below(200_000)
                } else {
                    1 + rng.below(4000)
                };
                (picture(rng, key, len), key)
            } else {
                // Length-prefixed or bare bytes.
                let body = rng.noise(3000);
                let mut out = u32::try_from(body.len()).unwrap().to_be_bytes().to_vec();
                out.extend(body);
                if rng.chance(50) {
                    out.drain(..4);
                }
                (out, key)
            }
        }
        Shape::Aac => {
            let len = if clean {
                1 + rng.below(1500)
            } else {
                rng.below(9000)
            };
            (rng.bytes(len), true)
        }
        Shape::Other(_) => {
            let max = if rng.chance(2) { 70_000 } else { 2000 };
            (rng.noise(max), rng.chance(30))
        }
    }
}

/// One session: `Some(packets)` read back when it finished clean.
fn mux_session(rng: &mut Rng, label: &str) -> Option<usize> {
    let clean = rng.chance(40);
    let options = MuxOptions {
        packet_size: if rng.chance(30) {
            cutlass_mpegts::packet::PacketSize::M2ts
        } else {
            cutlass_mpegts::packet::PacketSize::Ts
        },
        ..MuxOptions::default()
    };
    let mut mux = TsMux::with_options(Vec::new(), options);
    let mut tracks: Vec<SessionTrack> = Vec::new();
    let mut expected: Vec<Vec<Expected>> = Vec::new();
    let mut failed = false;
    let wanted = 1 + rng.below(if clean { 3 } else { 5 });
    for _ in 0..wanted {
        let (stream, shape, timescale) = session_stream(rng, clean);
        match mux.add_stream(stream) {
            Ok(id) => {
                tracks.push(SessionTrack {
                    id,
                    shape,
                    timescale,
                    clock: 0,
                    keyed: false,
                });
                expected.push(Vec::new());
            }
            Err(_) => {
                assert_eq!(
                    mux.status(),
                    MuxStatus::Failed,
                    "{label}: refused stream left the mux building"
                );
                assert!(!clean, "{label}: a clean stream was refused");
                failed = true;
                break;
            }
        }
    }
    if !failed && !clean && rng.chance(10) {
        let id = tracks.first().map_or(TrackId::new(9), |track| track.id);
        if mux.set_config(id, &rng.noise(40)).is_err() {
            assert_eq!(mux.status(), MuxStatus::Failed, "{label}: refused config");
            failed = true;
        }
    }
    let writes = if failed || tracks.is_empty() {
        0
    } else {
        rng.below(120)
    };
    for _ in 0..writes {
        let index = rng.below(tracks.len());
        let track = &mut tracks[index];
        let (bytes, key) = session_payload(rng, track, clean);
        let step = i64::from(track.timescale.max(1) / 25).max(1);
        let (dts, pts) = if clean {
            track.clock += step;
            let dts = track.clock;
            let pts = if track.shape == Shape::H264 {
                dts + step * i64::try_from(rng.below(3)).unwrap()
            } else {
                dts
            };
            (Some(dts), Some(pts))
        } else {
            track.clock += i64::try_from(rng.below(5000)).unwrap() - 300;
            let dts = track.clock;
            match rng.below(10) {
                0 => (None, None),
                1 => (None, Some(dts)),
                2 => (Some(dts), Some(dts - 5)),
                3 => (Some(i64::MAX - 2), Some(i64::MAX - 1)),
                4 => (Some(-1_000_000), Some(0)),
                _ => (Some(dts), Some(dts + 10)),
            }
        };
        let target = if !clean && rng.chance(3) {
            u32::from(rng.byte())
        } else {
            track.id.get()
        };
        let scale = if !clean && rng.chance(3) {
            12_345
        } else {
            track.timescale.max(1)
        };
        let Some(packet) = packet(target, bytes.clone(), dts, pts, scale, key) else {
            continue;
        };
        match mux.write_packet(&packet) {
            Ok(()) => {
                if clean {
                    let dts = to_clock(dts.unwrap(), track.timescale);
                    let pts = to_clock(pts.unwrap(), track.timescale);
                    expected[index].push(Expected {
                        bytes,
                        dts,
                        pts,
                        key,
                    });
                }
            }
            Err(_) => {
                assert_eq!(
                    mux.status(),
                    MuxStatus::Failed,
                    "{label}: refused packet left the mux building"
                );
                assert!(!clean, "{label}: a clean packet was refused");
                failed = true;
                break;
            }
        }
    }
    let durations: Vec<TrackDuration> = Vec::new();
    let finished = mux.finish(&durations);
    assert_eq!(finished.is_ok(), !failed, "{label}: finish after a failure");
    let bytes = mux.into_inner();
    if failed {
        return None;
    }
    let mut check = Rng::new(rng.next());
    if !clean {
        exercise_demux(&mut check, &bytes, false, label);
        return None;
    }
    let mut demux =
        TsDemux::open(Cursor::new(bytes.clone()), bytes.len() as u64).expect("clean file opens");
    // The muxer gives tracks PIDs from 0x100, and the demuxer's track ids
    // are PIDs.
    let ids: Vec<u32> = tracks.iter().map(|track| 0x100 + track.id.get()).collect();
    let mut got: Vec<Vec<Expected>> = vec![Vec::new(); tracks.len()];
    while let Some(packet) = demux.read_packet().expect("clean read") {
        let slot = ids
            .iter()
            .position(|&id| id == packet.stream())
            .expect("known pid");
        let time = packet.timestamp();
        got[slot].push(Expected {
            bytes: packet.bytes().to_vec(),
            dts: time.decode().expect("dts"),
            pts: time.presentation().expect("pts"),
            key: packet.keyframe(),
        });
    }
    let counters = demux.counters();
    assert_eq!(counters.cc_errors, 0, "{label}");
    let mut read = 0;
    for (index, (want, have)) in expected.iter().zip(&got).enumerate() {
        // A track without packets is not in the file's streams.
        assert_eq!(want.len(), have.len(), "{label}: track {index} count");
        for (unit, (want, have)) in want.iter().zip(have).enumerate() {
            assert_eq!(want, have, "{label}: track {index} unit {unit}");
        }
        read += have.len();
    }
    exercise_demux(&mut check, &bytes, true, label);
    Some(read)
}

/// Sessions that finished clean, and the packets they read back.
fn mux_campaign(seed: u64, sessions: usize) -> (usize, usize) {
    let mut rng = Rng::new(seed);
    (0..sessions)
        .filter_map(|session| mux_session(&mut rng, &format!("seed {seed} session {session}")))
        .fold((0, 0), |(finished, packets), read| {
            (finished + 1, packets + read)
        })
}

#[test]
fn fuzz_seeds_demux() {
    for seed in 0..32 {
        demux_campaign(seed, 8);
    }
}

#[test]
fn fuzz_seeds_mux() {
    let mut finished = 0;
    let mut packets = 0;
    for seed in 0..32 {
        let (sessions, read) = mux_campaign(seed, 8);
        finished += sessions;
        packets += read;
    }
    assert!(finished > 0, "no clean mux session finished");
    assert!(packets > 0, "no packet was muxed and read back");
}

#[test]
fn the_corpus_reads_intact() {
    let corpus = shared_corpus();
    // Each fixture as it is, remuxed as TS and as M2TS, and the wrapping file.
    assert_eq!(corpus.len(), 8 + 2 * 8 + 1);
    let mut rng = Rng::new(1);
    for (index, file) in corpus.iter().enumerate() {
        exercise_demux(&mut rng, file, true, &format!("corpus {index}"));
    }
}

fn campaign_length() -> Duration {
    let secs = std::env::var("TS_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(60);
    Duration::from_secs(secs)
}

#[test]
#[ignore = "long campaign; run with TS_FUZZ_SECS and --ignored"]
fn fuzz_campaign_demux() {
    let length = campaign_length();
    let started = Instant::now();
    let mut seed = 1_000;
    let mut inputs = 0;
    let mut slowest = Duration::ZERO;
    while started.elapsed() < length {
        slowest = slowest.max(demux_campaign(seed, 50));
        inputs += 50;
        seed += 1;
    }
    eprintln!(
        "demux campaign: {inputs} mutants over {} seeds in {:.1} s, slowest input {:.1} ms",
        seed - 1_000,
        started.elapsed().as_secs_f64(),
        slowest.as_secs_f64() * 1000.0
    );
}

#[test]
#[ignore = "long campaign; run with TS_FUZZ_SECS and --ignored"]
fn fuzz_campaign_mux() {
    let length = campaign_length();
    let started = Instant::now();
    let mut seed = 1_000;
    let mut sessions = 0;
    let mut finished = 0;
    let mut packets = 0;
    while started.elapsed() < length {
        let (done, read) = mux_campaign(seed, 20);
        finished += done;
        packets += read;
        sessions += 20;
        seed += 1;
    }
    eprintln!(
        "mux campaign: {sessions} sessions ({finished} clean ones read back, {packets} packets) in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    assert!(packets > 0, "no packet was muxed and read back");
}
