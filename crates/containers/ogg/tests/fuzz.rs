//! In-process random-mutation campaigns over the demuxer and the muxer.
//!
//! cargo-fuzz is not used: it is not installed here, and libfuzzer-sys
//! builds C++ through `cc`, which this crate's rules do not allow. These
//! campaigns are deterministic (a fixed xorshift generator per seed), so a
//! failure names the seed and iteration that reproduce it.
//!
//! Demux: valid Opus and Vorbis files written by [`OggMux`], and Vorbis
//! files from libvorbis and FFmpeg (`tests/fixtures`), are mutated (bit flips, byte
//! writes, inserts, deletes, cuts, splices, duplicated ranges), and half of
//! the mutants get their page checksums repaired so the mutation reaches
//! the page and packet parsers behind the checksum. Each mutant is probed,
//! opened random-access and forward, read to the end, and seeked. Every
//! call must return within a bounded number of steps, packets must be
//! within the limits, and each stream's times must not go backwards.
//!
//! Mux: random Opus and Vorbis stream setups and packet sequences,
//! including invalid ones. Vorbis streams take real setup headers.
//! A refused call must leave the mux failed, and a mux that finishes must
//! produce a file this crate demuxes back to exactly the accepted packets.
//!
//! `fuzz_seeds_*` run on every `cargo test`. The long campaigns are
//! `#[ignore]`; `OGG_FUZZ_SECS` sets their length (default 60 s).
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Landing, Mux, MuxStatus, NewStream, SeekTo, TrackDuration};
use cutlass_ogg::crc::page_crc;
use cutlass_ogg::page::parse_header;
use cutlass_ogg::{OggDemux, OggMux, PAGE_HEADER_BYTES, packet_samples, probe};

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
}

/// An OpusHead for `channels`, family 0 up to stereo, family 1 above.
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

fn new_stream(channels: u16, extradata: Vec<u8>) -> NewStream {
    NewStream {
        params: StreamParams::try_audio(
            StreamCodec::Known(CodecId::Opus),
            48_000,
            channels,
            extradata,
        )
        .expect("params"),
        timescale: Timescale::new(48_000),
        language: None,
    }
}

fn packet(track: u32, bytes: Vec<u8>, pts: i64, duration: i64) -> Packet {
    packet_at(track, bytes, pts, duration, 48_000)
}

fn packet_at(track: u32, bytes: Vec<u8>, pts: i64, duration: i64, rate: u32) -> Packet {
    let time = Timestamp::try_new(Some(pts), Some(pts), Some(duration)).expect("time");
    Packet::try_new(track, bytes, time, Timescale::new(rate), true).expect("packet")
}

const VORBIS_FILES: [&[u8]; 4] = [
    include_bytes!("fixtures/stereo_44100_q4.ogg"),
    include_bytes!("fixtures/sweep_8000_mono_q0.ogg"),
    include_bytes!("fixtures/surround51_48000_q4.ogg"),
    include_bytes!("fixtures/ffmpeg_stereo_44100.ogg"),
];
const VORBIS_CHAIN: &[u8] = include_bytes!("fixtures/chained_mono_stereo.ogg");

/// A real Vorbis stream: its params, modes, block sizes, and packets.
struct VorbisSeed {
    params: StreamParams,
    rate: u32,
    modes: Vec<bool>,
    blocks: [u32; 2],
    packets: Vec<Packet>,
}

impl VorbisSeed {
    /// Bits of the mode number in an audio packet.
    fn mode_bits(&self) -> u32 {
        u32::BITS - u32::try_from(self.modes.len() - 1).unwrap().leading_zeros()
    }

    fn block(&self, mode: usize) -> u32 {
        self.blocks[usize::from(self.modes[mode])]
    }
}

fn vorbis_seeds() -> &'static [VorbisSeed] {
    static SEEDS: OnceLock<Vec<VorbisSeed>> = OnceLock::new();
    SEEDS.get_or_init(|| {
        VORBIS_FILES
            .iter()
            .map(|bytes| {
                let len = u64::try_from(bytes.len()).unwrap();
                let mut demux = OggDemux::open(Cursor::new(bytes.to_vec()), len).expect("fixture");
                let params = demux.streams()[0].params().clone();
                let info = demux.vorbis(0).expect("vorbis").clone();
                let mut packets = Vec::new();
                while let Some(packet) = demux.read_packet().expect("read") {
                    packets.push(packet);
                }
                VorbisSeed {
                    rate: info.id.sample_rate,
                    modes: info.modes,
                    blocks: info.id.blocksizes,
                    params,
                    packets,
                }
            })
            .collect()
    })
}

fn vorbis_stream(seed: &VorbisSeed, extradata: Vec<u8>) -> NewStream {
    let params = &seed.params;
    NewStream {
        params: StreamParams::try_audio(
            params.codec().clone(),
            seed.rate,
            params.channels().expect("channels"),
            extradata,
        )
        .expect("params"),
        timescale: Timescale::new(seed.rate),
        language: None,
    }
}

/// `seed`'s packets muxed again, from `start`, alone or beside an Opus
/// stream.
fn vorbis_corpus_file(rng: &mut Rng, seed: &VorbisSeed, start: i64, with_opus: bool) -> Vec<u8> {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let vorbis = mux
        .add_stream(vorbis_stream(seed, seed.params.extradata().to_vec()))
        .expect("vorbis");
    let opus = with_opus.then(|| mux.add_stream(new_stream(2, head(2, 312))).expect("opus"));
    let mut vorbis_ticks = 0;
    let mut opus_ticks = 0;
    for (index, source) in seed.packets.iter().enumerate() {
        let duration = source.timestamp().duration().expect("duration");
        let pts = start + vorbis_ticks;
        mux.write_packet(&packet_at(
            vorbis.get(),
            source.bytes().to_vec(),
            pts,
            duration,
            seed.rate,
        ))
        .expect("vorbis packet");
        vorbis_ticks += duration;
        if let Some(opus) = opus
            && index % 2 == 0
        {
            let mut bytes: Vec<u8> = (0..40 + rng.below(200)).map(|_| rng.byte()).collect();
            bytes[0] = 19 << 3;
            mux.write_packet(&packet(opus.get(), bytes, opus_ticks, 960))
                .expect("opus packet");
            opus_ticks += 960;
        }
    }
    let mut durations = vec![TrackDuration {
        track: vorbis,
        timescale: Timescale::new(seed.rate),
        ticks: vorbis_ticks,
    }];
    if let Some(opus) = opus {
        durations.push(TrackDuration {
            track: opus,
            timescale: Timescale::new(48_000),
            ticks: opus_ticks,
        });
    }
    mux.finish(&durations).expect("finish");
    mux.into_inner().into_inner()
}

/// TOC bytes for 2.5, 5, 10, 20, 40, and 60 ms frames, code 0, and a code 3
/// packet byte.
const TOCS: &[u8] = &[
    16 << 3,
    (17 << 3),
    (18 << 3),
    (19 << 3),
    (11 << 3),
    (3 << 3),
];

fn opus_payload(rng: &mut Rng, len: usize) -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..len.max(1)).map(|_| rng.byte()).collect();
    bytes[0] = TOCS[rng.below(TOCS.len())] | u8::from(rng.chance(10));
    bytes
}

/// A valid file: `tracks` streams of 20 ms packets, interleaved by time.
fn corpus_file(rng: &mut Rng, tracks: u8, packets: usize, big: bool) -> Vec<u8> {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let mut ids = Vec::new();
    for index in 0..tracks {
        let channels = [2, 1, 6, 4][usize::from(index) % 4];
        ids.push(
            mux.add_stream(new_stream(
                u16::from(channels),
                head(channels, [312, 120, 0, 3840][usize::from(index) % 4]),
            ))
            .expect("add"),
        );
    }
    let mut sums = vec![0_i64; ids.len()];
    for step in 0..packets {
        for (slot, id) in ids.iter().enumerate() {
            let size = if big && step % 7 == 3 {
                3000 + rng.below(9000)
            } else {
                40 + rng.below(400)
            };
            let mut bytes: Vec<u8> = (0..size).map(|_| rng.byte()).collect();
            bytes[0] = 19 << 3;
            let duration = if step + 1 == packets { 500 } else { 960 };
            let pts = i64::try_from(step).unwrap() * 960;
            mux.write_packet(&packet(id.get(), bytes, pts, duration))
                .expect("write");
            sums[slot] += duration;
        }
    }
    let durations: Vec<TrackDuration> = ids
        .iter()
        .zip(&sums)
        .map(|(id, &ticks)| TrackDuration {
            track: *id,
            timescale: Timescale::new(48_000),
            ticks,
        })
        .collect();
    mux.finish(&durations).expect("finish");
    mux.into_inner().into_inner()
}

fn corpus() -> Vec<Vec<u8>> {
    let mut rng = Rng::new(7);
    let one = corpus_file(&mut rng, 1, 120, false);
    let two = corpus_file(&mut rng, 2, 80, true);
    let three = corpus_file(&mut rng, 3, 30, true);
    let short = corpus_file(&mut rng, 1, 3, false);
    let mut chained = short.clone();
    chained.extend_from_slice(&corpus_file(&mut rng, 1, 5, false));
    let mut files = vec![one, two, three, short, chained];
    files.extend(VORBIS_FILES.iter().map(|bytes| bytes.to_vec()));
    files.push(VORBIS_CHAIN.to_vec());
    let seeds = vorbis_seeds();
    files.push(vorbis_corpus_file(&mut rng, &seeds[0], 0, false));
    files.push(vorbis_corpus_file(&mut rng, &seeds[1], -700, false));
    files.push(vorbis_corpus_file(&mut rng, &seeds[3], 0, true));
    files
}

/// Rewrites the checksum of every page found, so mutants get past it.
fn repair_checksums(bytes: &mut [u8]) {
    let mut at = 0;
    while at + PAGE_HEADER_BYTES <= bytes.len() {
        let Some(header) = parse_header(&bytes[at..]) else {
            at += 1;
            continue;
        };
        let table_end = at + PAGE_HEADER_BYTES + usize::from(header.segments);
        let Some(table) = bytes.get(at + PAGE_HEADER_BYTES..table_end) else {
            return;
        };
        let body_len: usize = table.iter().map(|&value| usize::from(value)).sum();
        let end = table_end + body_len;
        if end > bytes.len() {
            return;
        }
        let checksum = page_crc(&bytes[at..table_end], &bytes[table_end..end]);
        bytes[at + 22..at + 26].copy_from_slice(&checksum.to_le_bytes());
        at = end;
    }
}

fn mutate(rng: &mut Rng, seed: &[u8], corpus: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = seed.to_vec();
    let rounds = 1 + rng.below(8);
    for _ in 0..rounds {
        let len = bytes.len();
        match rng.below(9) {
            0 if len > 0 => {
                let at = rng.below(len);
                bytes[at] ^= 1 << rng.below(8);
            }
            1 if len > 0 => {
                let at = rng.below(len);
                bytes[at] = [0, 0xFF, 0x7F, 0x80, rng.byte()][rng.below(5)];
            }
            2 => {
                let at = rng.below(len + 1);
                let count = 1 + rng.below(64);
                let insert: Vec<u8> = (0..count).map(|_| rng.byte()).collect();
                bytes.splice(at..at, insert);
            }
            3 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(512)).min(len);
                bytes.drain(at..end);
            }
            4 => bytes.truncate(rng.below(len + 1)),
            5 if len > 0 => {
                // Splice a range from another corpus file.
                let other = &corpus[rng.below(corpus.len())];
                if !other.is_empty() {
                    let from = rng.below(other.len());
                    let take = (1 + rng.below(4096)).min(other.len() - from);
                    let at = rng.below(len);
                    bytes.splice(at..at, other[from..from + take].iter().copied());
                }
            }
            6 if len > 0 => {
                // Duplicate a range, like a repeated page.
                let at = rng.below(len);
                let end = (at + 1 + rng.below(8192)).min(len);
                let copy = bytes[at..end].to_vec();
                let to = rng.below(len);
                bytes.splice(to..to, copy);
            }
            7 if len > PAGE_HEADER_BYTES => {
                // Hit a header field of some page: flags, granule, serial,
                // sequence, or segment count.
                let pages: Vec<usize> = bytes
                    .windows(4)
                    .enumerate()
                    .filter(|(_, window)| *window == b"OggS")
                    .map(|(at, _)| at)
                    .collect();
                if let Some(&page) = pages.get(rng.below(pages.len())) {
                    let field = [5, 6, 7, 13, 14, 18, 26][rng.below(7)];
                    if let Some(byte) = bytes.get_mut(page + field) {
                        *byte = rng.byte();
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
        repair_checksums(&mut bytes);
    }
    bytes
}

/// Reads to the end and checks the invariants; returns the packets read.
fn drain(demux: &mut OggDemux<Cursor<Vec<u8>>>, input_len: usize, label: &str) -> usize {
    let cap = 2 * input_len + 64;
    let mut last: HashMap<u32, i64> = HashMap::new();
    let mut count = 0;
    for _ in 0..cap {
        match demux.read_packet() {
            Ok(Some(packet)) => {
                count += 1;
                assert!(
                    packet.bytes().len() <= input_len,
                    "{label}: packet larger than input"
                );
                assert!(packet.keyframe(), "{label}: keyframe");
                let pts = packet.timestamp().presentation().expect("pts");
                let duration = packet.timestamp().duration().expect("duration");
                assert!(duration >= 0, "{label}: negative duration");
                assert!(duration <= 5760, "{label}: duration {duration}");
                if let Some(&previous) = last.get(&packet.stream()) {
                    assert!(
                        pts >= previous,
                        "{label}: track {} went from {previous} to {pts}",
                        packet.stream()
                    );
                }
                last.insert(packet.stream(), pts);
            }
            Ok(None) => return count,
            Err(_) => {}
        }
    }
    panic!("{label}: read_packet did not reach the end within {cap} calls");
}

/// `intact` inputs are unmutated corpus files: a seek there must land at or
/// before the target. In a damaged file the pages before the target may be
/// lost, and the seek lands on the first packet after the hole, at its true
/// time.
fn exercise_demux(rng: &mut Rng, bytes: &[u8], intact: bool, label: &str) {
    let _ = probe(&bytes[..bytes.len().min(16)]);
    let len = u64::try_from(bytes.len()).unwrap();
    if let Ok(mut demux) = OggDemux::open(Cursor::new(bytes.to_vec()), len) {
        let streams = demux.streams().len();
        let read = rng.below(4);
        for _ in 0..read {
            let _ = demux.read_packet();
        }
        for _ in 0..3 {
            if streams == 0 {
                break;
            }
            let track = u32::try_from(rng.below(streams + 1)).unwrap();
            let time = i64::try_from(rng.below(200_000)).unwrap() - 4_000;
            let before = demux.consumed();
            match demux.seek(SeekTo {
                track,
                time,
                end: None,
            }) {
                Ok(outcome) => {
                    if let Landing::Packet { time: landed, .. } = outcome.landing {
                        let start = demux
                            .opus(track)
                            .and_then(|info| info.start)
                            .or_else(|| demux.vorbis(track).and_then(|info| info.start))
                            .expect("a landing needs a start");
                        if intact {
                            assert!(
                                landed <= time.max(start),
                                "{label}: landed {landed} past {time}"
                            );
                        }
                        // The next packet of the track is the landing one,
                        // unless the file breaks before it.
                        for _ in 0..64 {
                            match demux.read_packet() {
                                Ok(Some(next)) if next.stream() == track => {
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
    }
    if let Ok(mut demux) = OggDemux::open_forward(Cursor::new(bytes.to_vec())) {
        let _ = drain(&mut demux, bytes.len(), label);
    }
}

/// Runs `iterations` mutants from `seed` and returns the slowest one's time.
fn demux_campaign(seed: u64, iterations: usize) -> Duration {
    static CORPUS: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    let corpus = CORPUS.get_or_init(corpus);
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

/// One stream of a random mux session.
enum SessionTrack {
    Opus {
        channels: u8,
    },
    Vorbis {
        seed: &'static VorbisSeed,
        previous: Option<u32>,
    },
}

impl SessionTrack {
    const fn rate(&self) -> u32 {
        match self {
            Self::Opus { .. } => 48_000,
            Self::Vorbis { seed, .. } => seed.rate,
        }
    }
}

/// A random Opus or Vorbis stream to add, and the track it becomes.
fn session_stream(rng: &mut Rng) -> (NewStream, SessionTrack) {
    if rng.chance(40) {
        let seeds = vorbis_seeds();
        let seed = &seeds[rng.below(seeds.len())];
        let extradata = match rng.below(10) {
            0 => Vec::new(),
            1 => (0..rng.below(80)).map(|_| rng.byte()).collect(),
            2 => {
                let mut cut = seed.params.extradata().to_vec();
                cut.truncate(rng.below(cut.len()));
                cut
            }
            _ => seed.params.extradata().to_vec(),
        };
        let stream = vorbis_stream(seed, extradata);
        return (
            stream,
            SessionTrack::Vorbis {
                seed,
                previous: None,
            },
        );
    }
    let channels = u8::try_from(1 + rng.below(8)).unwrap();
    let config = match rng.below(10) {
        0 => Vec::new(),
        1 => (0..rng.below(40)).map(|_| rng.byte()).collect(),
        _ => head(channels, u16::try_from(rng.below(4000)).unwrap()),
    };
    let declared = if rng.chance(5) {
        u16::from(channels) + 1
    } else {
        u16::from(channels)
    };
    (
        new_stream(declared, config),
        SessionTrack::Opus { channels },
    )
}

/// A random packet for `track` and its natural duration.
fn session_packet(rng: &mut Rng, track: &mut SessionTrack) -> (Vec<u8>, i64) {
    match track {
        SessionTrack::Opus { .. } => {
            let size = if rng.chance(5) {
                rng.below(130_000)
            } else {
                1 + rng.below(1500)
            };
            let bytes = opus_payload(rng, size);
            let samples = i64::from(packet_samples(&bytes).unwrap_or(0));
            (bytes, samples)
        }
        SessionTrack::Vorbis { seed, previous } => {
            // Now and then a mode past the setup's, or a header packet.
            let extra = usize::from(rng.chance(3));
            let mode = rng.below(seed.modes.len() + extra);
            let bits = seed.mode_bits();
            let size = if rng.chance(3) {
                rng.below(70_000)
            } else {
                1 + rng.below(900)
            };
            let mut bytes: Vec<u8> = (0..size.max(1)).map(|_| rng.byte()).collect();
            let high = u32::from(bytes[0]) >> (1 + bits) << (1 + bits);
            let low = u32::try_from(mode).unwrap() << 1 | u32::from(rng.chance(2));
            bytes[0] = u8::try_from((high | low) & 0xFF).unwrap();
            if mode >= seed.modes.len() || bytes[0] & 1 == 1 {
                return (bytes, 0);
            }
            let block = seed.block(mode);
            let samples = previous.map_or(0, |last| i64::from(last / 4 + block / 4));
            *previous = Some(block);
            (bytes, samples)
        }
    }
}

/// One random mux session. Returns, when it finished, the Vorbis packets
/// read back.
fn mux_session(rng: &mut Rng, label: &str) -> Option<usize> {
    let mut mux = OggMux::new(Cursor::new(Vec::new()));
    let count = 1 + rng.below(3);
    let mut tracks: Vec<(cutlass_container::TrackId, SessionTrack)> = Vec::new();
    let mut ok = true;
    for _ in 0..count {
        let (stream, track) = session_stream(rng);
        match mux.add_stream(stream) {
            Ok(id) => tracks.push((id, track)),
            Err(_) => {
                ok = false;
                break;
            }
        }
    }
    if ok {
        for (id, track) in &tracks {
            if !rng.chance(15) {
                continue;
            }
            let config = match track {
                SessionTrack::Opus { channels } => {
                    head(*channels, u16::try_from(rng.below(1000)).unwrap())
                }
                SessionTrack::Vorbis { seed, .. } => seed.params.extradata().to_vec(),
            };
            if mux.set_config(*id, &config).is_err() {
                ok = false;
                break;
            }
        }
    }
    let mut accepted: Vec<Vec<Packet>> = vec![Vec::new(); tracks.len()];
    let mut next: Vec<i64> = tracks
        .iter()
        .map(|(_, track)| match track {
            SessionTrack::Opus { .. } => i64::try_from(rng.below(3)).unwrap() * 960,
            SessionTrack::Vorbis { .. } => {
                i64::try_from(rng.below(3000)).unwrap() - if rng.chance(30) { 3000 } else { 0 }
            }
        })
        .collect();
    if ok {
        let count = rng.below(60);
        for _ in 0..count {
            let slot = rng.below(tracks.len());
            let (bytes, samples) = session_packet(rng, &mut tracks[slot].1);
            let duration = match rng.below(20) {
                0 => samples / 2,
                1 => samples + 1,
                _ => samples,
            };
            let mut pts = next[slot];
            if rng.chance(3) {
                pts += 1;
            }
            if rng.chance(2) {
                pts -= 960;
            }
            let rate = tracks[slot].1.rate();
            let p = packet_at(tracks[slot].0.get(), bytes, pts, duration, rate);
            match mux.write_packet(&p) {
                Ok(()) => {
                    next[slot] = pts + duration;
                    accepted[slot].push(p);
                }
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
    }
    if !ok {
        assert_eq!(
            mux.status(),
            MuxStatus::Failed,
            "{label}: refused but not failed"
        );
    }
    let durations: Vec<TrackDuration> = tracks
        .iter()
        .zip(&accepted)
        .map(|((id, track), packets)| TrackDuration {
            track: *id,
            timescale: Timescale::new(track.rate()),
            ticks: packets
                .iter()
                .map(|p| p.timestamp().duration().unwrap())
                .sum(),
        })
        .collect();
    let finished = mux.finish(&durations);
    if !ok {
        assert!(finished.is_err(), "{label}: a failed mux reported success");
        return None;
    }
    if finished.is_err() {
        assert_eq!(mux.status(), MuxStatus::Failed, "{label}: finish failed");
        return None;
    }
    let bytes = mux.into_inner().into_inner();
    let len = u64::try_from(bytes.len()).unwrap();
    let mut demux = match OggDemux::open(Cursor::new(bytes), len) {
        Ok(demux) => demux,
        Err(error) => panic!("{label}: the muxer wrote a file the demuxer refuses: {error}"),
    };
    let mut back: Vec<Vec<Packet>> = vec![Vec::new(); tracks.len()];
    while let Some(p) = demux.read_packet().expect("read back") {
        back[usize::try_from(p.stream()).unwrap()].push(p);
    }
    for (slot, (mine, theirs)) in accepted.iter().zip(&back).enumerate() {
        assert_eq!(mine.len(), theirs.len(), "{label}: track {slot} count");
        for (left, right) in mine.iter().zip(theirs) {
            assert_eq!(left.bytes(), right.bytes(), "{label}: bytes");
            assert_eq!(left.timestamp(), right.timestamp(), "{label}: time");
        }
    }
    Some(
        tracks
            .iter()
            .zip(&back)
            .filter(|((_, track), _)| matches!(track, SessionTrack::Vorbis { .. }))
            .map(|(_, packets)| packets.len())
            .sum(),
    )
}

/// Sessions that finished, and the Vorbis packets they read back.
fn mux_campaign(seed: u64, sessions: usize) -> (usize, usize) {
    let mut rng = Rng::new(seed);
    (0..sessions)
        .filter_map(|session| mux_session(&mut rng, &format!("seed {seed} session {session}")))
        .fold((0, 0), |(finished, vorbis), packets| {
            (finished + 1, vorbis + packets)
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
    let mut vorbis = 0;
    for seed in 0..64 {
        let (sessions, packets) = mux_campaign(seed, 8);
        finished += sessions;
        vorbis += packets;
    }
    assert!(finished > 0, "no random mux session finished");
    assert!(vorbis > 0, "no Vorbis packet was muxed and read back");
}

fn campaign_length() -> Duration {
    let secs = std::env::var("OGG_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(60);
    Duration::from_secs(secs)
}

#[test]
#[ignore = "long campaign; run with OGG_FUZZ_SECS and --ignored"]
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
#[ignore = "long campaign; run with OGG_FUZZ_SECS and --ignored"]
fn fuzz_campaign_mux() {
    let length = campaign_length();
    let started = Instant::now();
    let mut seed = 1_000;
    let mut sessions = 0;
    let mut finished = 0;
    let mut vorbis = 0;
    while started.elapsed() < length {
        let (done, packets) = mux_campaign(seed, 20);
        finished += done;
        vorbis += packets;
        sessions += 20;
        seed += 1;
    }
    eprintln!(
        "mux campaign: {sessions} sessions ({finished} finished and read back, {vorbis} Vorbis packets) in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    assert!(vorbis > 0, "no Vorbis packet was muxed and read back");
}
