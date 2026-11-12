//! In-process demux and mux campaign.
//!
//! This is not libFuzzer: a seeded generator builds CAF files from chunks
//! with honest and lying sizes, mutates good files (and, when present, a
//! few of the Mac's own CAF files), and checks invariants on everything
//! that opens. The mux half writes random streams and requires that what
//! finishes reads back packet for packet. Run it with:
//! `CAF_FUZZ_SECS=1800 cargo test --release -p cutlass-caf --test external fuzz_campaign -- --ignored --nocapture`

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cutlass_caf::{CafDemux, CafMux, probe};
use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{
    Demux, Landing, Limits, Mux, NewStream, SeekTo, TrackDuration, TrackId, Trim,
};

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            usize::try_from(self.next() % u64::try_from(n).expect("n")).expect("fits")
        }
    }

    fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next().to_le_bytes()[0]).collect()
    }

    /// Up to `most - 1` random bytes.
    fn noise(&mut self, most: usize) -> Vec<u8> {
        let len = self.below(most);
        self.bytes(len)
    }
}

fn chunk(kind: &[u8; 4], size: i64, body: &[u8]) -> Vec<u8> {
    let mut out = kind.to_vec();
    out.extend(size.to_be_bytes());
    out.extend(body);
    out
}

fn honest(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    chunk(kind, i64::try_from(body.len()).expect("len"), body)
}

fn varint(value: u32, out: &mut Vec<u8>) {
    let mut groups = vec![value & 0x7F];
    let mut rest = value >> 7;
    while rest != 0 {
        groups.push(rest & 0x7F);
        rest >>= 7;
    }
    for (index, group) in groups.iter().enumerate().rev() {
        let more = if index == 0 { 0 } else { 0x80 };
        out.push(u8::try_from(*group).expect("group") | more);
    }
}

const FORMATS: &[&[u8; 4]] = &[
    b"lpcm", b"aac ", b"alac", b"flac", b"opus", b"ulaw", b"alaw", b".mp3", b"ac-3", b"ec-3",
    b"ima4", b"zzzz",
];

const RATES: &[f64] = &[
    8_000.0,
    24_000.0,
    44_100.0,
    48_000.0,
    0.0,
    -1.0,
    1e300,
    f64::NAN,
    0.5,
    4_294_967_295.0,
];

fn interesting(rng: &mut Rng) -> u32 {
    let values = [
        0,
        1,
        2,
        3,
        4,
        8,
        12,
        64,
        1024,
        4096,
        65_535,
        1 << 20,
        (1 << 20) + 1,
        u32::MAX,
    ];
    if rng.chance(3) {
        u32::try_from(rng.next() & 0xFFFF).expect("small")
    } else {
        values[rng.below(values.len())]
    }
}

fn desc(rng: &mut Rng) -> Vec<u8> {
    let format = FORMATS[rng.below(FORMATS.len())];
    let rate = RATES[rng.below(RATES.len())];
    let mut body = rate.to_bits().to_be_bytes().to_vec();
    body.extend(format);
    let channels = [1, 2, 0, 3, 64, 65][rng.below(6)];
    let (flags, bpp, fpp, bits) = match rng.below(4) {
        // Coherent PCM.
        0 => {
            let bits = [8, 16, 24, 32, 64][rng.below(5)];
            (rng.below(4) as u32, bits / 8 * channels, 1, bits)
        }
        // Variable packets.
        1 => (0, 0, [1024, 960, 4096, 0][rng.below(4)], 0),
        // Constant compressed.
        2 => (0, 34 * channels, 64, 0),
        _ => (
            interesting(rng),
            interesting(rng),
            interesting(rng),
            interesting(rng),
        ),
    };
    for field in [flags, bpp, fpp, channels, bits] {
        body.extend(field.to_be_bytes());
    }
    honest(b"desc", &body)
}

fn pakt(rng: &mut Rng) -> Vec<u8> {
    let entries = rng.below(300);
    let mut body = Vec::new();
    let lie = rng.chance(4);
    let packets = if lie {
        [i64::MAX, -1, 1 << 40, 0][rng.below(4)]
    } else {
        i64::try_from(entries).expect("entries")
    };
    body.extend(packets.to_be_bytes());
    body.extend(
        i64::try_from(rng.below(1 << 20))
            .expect("valid")
            .to_be_bytes(),
    );
    body.extend(
        u32::try_from(rng.below(5000))
            .expect("priming")
            .to_be_bytes(),
    );
    body.extend(
        u32::try_from(rng.below(5000))
            .expect("remainder")
            .to_be_bytes(),
    );
    let pairs = rng.chance(3);
    for _ in 0..entries {
        varint(u32::try_from(rng.below(600)).expect("size"), &mut body);
        if pairs {
            varint(u32::try_from(rng.below(5000)).expect("frames"), &mut body);
        }
    }
    if rng.chance(8) {
        body.extend([0xFF; 6]);
    }
    honest(b"pakt", &body)
}

fn kuki(rng: &mut Rng) -> Vec<u8> {
    let body = match rng.below(4) {
        0 => {
            // Apple's AAC cookie with a random AudioSpecificConfig.
            let asc_len = rng.below(8);
            let mut out = vec![
                0x03,
                0x80,
                0x80,
                0x80,
                u8::try_from(0x1D + asc_len).unwrap_or(0xFF),
            ];
            out.extend([
                0,
                0,
                0,
                0x04,
                0x80,
                0x80,
                0x80,
                u8::try_from(0x0F + asc_len).unwrap_or(0xFF),
            ]);
            out.extend([0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend([0x05, 0x80, 0x80, 0x80, u8::try_from(asc_len).unwrap_or(0)]);
            out.extend(rng.bytes(asc_len));
            out.extend([0x06, 0x80, 0x80, 0x80, 0x01, 0x02]);
            out
        }
        1 => {
            // dfLa with a STREAMINFO.
            let mut out = vec![0, 0, 0, 50];
            out.extend(b"dfLa");
            out.extend([0, 0, 0, 0, 0x80, 0, 0, 0x22]);
            out.extend(rng.bytes(34));
            out
        }
        2 => rng.bytes(24),
        _ => rng.noise(80),
    };
    honest(b"kuki", &body)
}

fn data(rng: &mut Rng) -> Vec<u8> {
    let len = rng.below(8_000);
    let mut body = 1_u32.to_be_bytes().to_vec();
    body.extend(rng.bytes(len));
    let size = match rng.below(6) {
        0 => -1,
        1 => i64::MAX,
        2 => i64::try_from(len).expect("len") / 2,
        _ => i64::try_from(body.len()).expect("len"),
    };
    chunk(b"data", size, &body)
}

/// A file assembled from random chunks.
fn assembled(rng: &mut Rng) -> Vec<u8> {
    let mut out = b"caff\x00\x01\x00\x00".to_vec();
    if rng.chance(30) {
        out[5] = 2;
    }
    if !rng.chance(20) {
        out.extend(desc(rng));
    }
    let count = rng.below(7);
    for _ in 0..count {
        let part = match rng.below(9) {
            0 | 1 => pakt(rng),
            2 => kuki(rng),
            3 => honest(b"chan", &{
                let len = [12, 32, 52, 13][rng.below(4)];
                rng.bytes(len)
            }),
            4 => honest(b"info", &rng.noise(64)),
            5 => data(rng),
            6 => chunk(b"free", [-5, 0, 3, i64::MAX][rng.below(4)], &[0; 3]),
            7 => desc(rng),
            _ => honest(b"uuid", &rng.noise(40)),
        };
        out.extend(part);
    }
    if !rng.chance(4) {
        out.extend(data(rng));
    }
    if rng.chance(3) {
        out.extend(pakt(rng));
    }
    out
}

/// Good files written by this crate's muxer.
pub fn corpus() -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut rng = Rng(0x5EED);
    for codec in [
        CodecId::Pcm,
        CodecId::Aac,
        CodecId::Alac,
        CodecId::Flac,
        CodecId::Opus,
    ] {
        if let Some(file) = random_mux(&mut rng, Some(codec), 40) {
            out.push(file.0);
        }
    }
    // A few of the Mac's own files, when present.
    for path in [
        "/System/Library/Components/CoreAudio.component/Contents/SharedSupport/SystemSounds/system/mic_unmute.caf",
        "/System/Library/Components/CoreAudio.component/Contents/SharedSupport/SystemSounds/system/begin_record.caf",
        "/System/Library/Components/CoreAudio.component/Contents/SharedSupport/SystemSounds/system/Lock.caf",
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            out.push(bytes);
        }
    }
    out
}

fn mutated(rng: &mut Rng, corpus: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = corpus[rng.below(corpus.len())].clone();
    for _ in 0..=rng.below(4) {
        if bytes.is_empty() {
            break;
        }
        let at = rng.below(bytes.len());
        match rng.below(7) {
            0 => bytes[at] ^= 1 << rng.below(8),
            1 => bytes[at] = [0, 0x7F, 0x80, 0xFF][rng.below(4)],
            2 => bytes.truncate(at),
            3 => {
                let end = (at + rng.below(64)).min(bytes.len());
                bytes.drain(at..end);
            }
            4 => {
                let insert = rng.noise(32);
                bytes.splice(at..at, insert);
            }
            5 => {
                // Overwrite eight bytes with an interesting 64-bit size.
                let value = [-1_i64, 0, i64::MAX, 1 << 32, 12, -12][rng.below(6)];
                let end = (at + 8).min(bytes.len());
                let len = end - at;
                bytes[at..end].copy_from_slice(&value.to_be_bytes()[..len]);
            }
            _ => {
                let other = &corpus[rng.below(corpus.len())];
                if !other.is_empty() {
                    let from = rng.below(other.len());
                    let take = rng.below(256).min(other.len() - from);
                    bytes.splice(at..at, other[from..from + take].iter().copied());
                }
            }
        }
    }
    bytes
}

/// Opens, reads and seeks, checking what must always hold.
fn demux_checks(bytes: &[u8], rng: &mut Rng, limits: Limits, stats: &mut Stats) {
    let _ = probe(&bytes[..bytes.len().min(12)]);
    let len = u64::try_from(bytes.len()).expect("len");
    if let Ok(mut demux) = CafDemux::open_limited(Cursor::new(bytes.to_vec()), len, limits) {
        stats.opened += 1;
        let stream = demux.streams()[0].clone();
        assert!(stream.timescale().ticks_per_second() > 0);
        let mut last = i64::MIN;
        for _ in 0..2_000 {
            match demux.read_packet() {
                Ok(Some(packet)) => {
                    assert!(
                        u64::try_from(packet.bytes().len()).expect("len")
                            <= limits.max_packet_bytes
                    );
                    let pts = packet.timestamp().presentation().expect("pts");
                    assert!(pts >= last, "times go back: {pts} after {last}");
                    last = pts;
                    assert!(packet.timestamp().duration().expect("duration") >= 0);
                    stats.packets += 1;
                }
                Ok(None) | Err(_) => break,
            }
        }
        for _ in 0..4 {
            let time = match rng.below(4) {
                0 => -1,
                1 => i64::MAX,
                _ => i64::try_from(rng.below(1 << 20)).expect("time"),
            };
            let before = demux.consumed();
            if let Ok(landing) = demux.seek(SeekTo {
                track: 0,
                time,
                end: None,
            }) {
                assert_eq!(demux.consumed(), before, "a seek read the file");
                assert_eq!(landing.payload_scans, 0);
                stats.seeks += 1;
                if let Landing::Packet { time: landed, .. } = landing.landing {
                    assert!(landed <= time.max(0), "landed after the target");
                }
                let _ = demux.read_packet();
            }
        }
    }
    if let Ok(mut forward) = CafDemux::open_forward_limited(Cursor::new(bytes.to_vec()), limits) {
        for _ in 0..2_000 {
            if !matches!(forward.read_packet(), Ok(Some(_))) {
                break;
            }
        }
    }
}

fn audio(codec: CodecId, rate: u32, channels: u16, extradata: Vec<u8>) -> Option<NewStream> {
    Some(NewStream {
        params: StreamParams::try_audio(StreamCodec::Known(codec), rate, channels, extradata)
            .ok()?,
        timescale: Timescale::new(rate),
        language: None,
    })
}

/// Writes a random stream; returns the file and the packets when the mux
/// finished.
fn random_mux(
    rng: &mut Rng,
    codec: Option<CodecId>,
    most: usize,
) -> Option<(Vec<u8>, Vec<Packet>)> {
    let codecs = [
        CodecId::Pcm,
        CodecId::Aac,
        CodecId::Alac,
        CodecId::Flac,
        CodecId::Opus,
        CodecId::Mp3,
        CodecId::Ac3,
        CodecId::Eac3,
        CodecId::Vorbis,
    ];
    let codec = codec.unwrap_or(codecs[rng.below(codecs.len())]);
    let rate = [8_000, 44_100, 48_000, 24_000][rng.below(4)];
    let channels = [1, 2, 3][rng.below(3)];
    let (extradata, frame_bytes, step) = match codec {
        CodecId::Pcm => {
            let coding = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10][rng.below(10)];
            let width = [0, 2, 3, 4, 4, 8, 1, 1, 1, 8, 4][usize::from(coding)];
            (
                vec![coding, u8::from(rng.chance(2)), 0],
                width * usize::from(channels),
                0,
            )
        }
        CodecId::Aac => (vec![0x12, 0x10], 0, 1024),
        CodecId::Alac => {
            let mut config = vec![0, 0, 0x10, 0, 0, 16, 40, 10, 14];
            config.push(u8::try_from(channels).expect("channels"));
            config.extend([0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xAC, 0x44]);
            (config, 0, 4096)
        }
        CodecId::Flac => {
            let mut info = vec![0x10, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0x0A, 0xC4, 0x42, 0xF0];
            info.extend([0; 20]);
            (info, 0, 4608)
        }
        CodecId::Opus => {
            let mut head = b"OpusHead".to_vec();
            head.extend([
                1,
                u8::try_from(channels).expect("channels"),
                0x38,
                1,
                0x80,
                0xBB,
                0,
                0,
                0,
                0,
                0,
            ]);
            (head, 0, 960)
        }
        _ => (Vec::new(), 0, 1152),
    };
    let stream = audio(codec, rate, channels, extradata)?;
    let mut limits = Limits::standard();
    limits.max_packet_bytes = 64 * 1024;
    let mut mux = CafMux::with_limits(Cursor::new(Vec::new()), limits);
    mux.add_stream(stream).ok()?;
    if rng.chance(3) {
        let delay = rng.below(3000) as u64;
        let padding = rng.below(3000) as u64;
        let _ = mux.set_trim(TrackId::new(0), Trim::new(Some(delay), Some(padding)));
    }
    let count = rng.below(most);
    let mut written = Vec::new();
    let mut pts = 0_i64;
    for index in 0..count {
        let (bytes, frames) = if frame_bytes > 0 {
            let frames = 1 + rng.below(700);
            (
                rng.bytes(frames * frame_bytes),
                i64::try_from(frames).expect("frames"),
            )
        } else {
            let mut frames = i64::try_from(step).expect("step");
            if index + 1 == count || rng.chance(15) {
                frames = 1 + i64::try_from(rng.below(step.max(1))).expect("frames");
            }
            let len = 1 + rng.below(500);
            (rng.bytes(len), frames)
        };
        let duration = if rng.chance(200) { frames + 1 } else { frames };
        let time = Timestamp::try_new(Some(pts), Some(pts), Some(duration)).ok()?;
        let packet = Packet::try_new(0, bytes, time, Timescale::new(rate), true).ok()?;
        if mux.write_packet(&packet).is_err() {
            return None;
        }
        pts += duration;
        written.push(packet);
    }
    let ticks = if rng.chance(30) { pts + 1 } else { pts };
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(rate),
        ticks,
    }])
    .ok()?;
    Some((mux.into_inner().into_inner(), written))
}

/// A finished mux must read back packet for packet.
fn mux_checks(rng: &mut Rng, stats: &mut Stats) {
    let Some((file, written)) = random_mux(rng, None, 60) else {
        return;
    };
    stats.muxed += 1;
    let len = u64::try_from(file.len()).expect("len");
    let mut demux = CafDemux::open(Cursor::new(file), len).expect("a finished mux opens");
    let pcm = demux.streams()[0].params().codec() == &StreamCodec::Known(CodecId::Pcm);
    let mut read = Vec::new();
    while let Some(packet) = demux.read_packet().expect("a finished mux reads") {
        read.push(packet);
    }
    let joined = |packets: &[Packet]| -> Vec<u8> {
        packets
            .iter()
            .flat_map(|packet| packet.bytes().to_vec())
            .collect()
    };
    assert!(
        joined(&read) == joined(&written),
        "payload differs after a mux"
    );
    if !pcm {
        assert_eq!(read.len(), written.len());
        for (index, (a, b)) in read.iter().zip(&written).enumerate() {
            assert_eq!(a.timestamp().presentation(), b.timestamp().presentation());
            if index + 1 < written.len() {
                assert_eq!(a.timestamp().duration(), b.timestamp().duration());
            }
        }
    }
    // A short last packet comes back whole, with the frames it lacked as
    // remainder.
    let frames: u64 = written
        .iter()
        .map(|packet| u64::try_from(packet.timestamp().duration().unwrap_or(0)).expect("frames"))
        .sum();
    let total = demux.total_frames().expect("frames");
    let remainder = demux
        .packet_table()
        .map_or(0, |header| u64::from(header.remainder_frames));
    assert!(
        total >= frames && total - frames <= remainder,
        "{total} {frames} {remainder}"
    );
    if let Some(header) = demux.packet_table() {
        let sum = header.valid_frames + u64::from(header.priming_frames) + remainder;
        assert_eq!(sum, total, "valid, priming and remainder add up");
    }
}

/// What the cases reached, to show the generator is not stuck at the door.
#[derive(Default)]
pub struct Stats {
    cases: u64,
    opened: u64,
    packets: u64,
    seeks: u64,
    muxed: u64,
}

pub fn exercise(seed: u64, corpus: &[Vec<u8>], stats: &mut Stats) {
    stats.cases += 1;
    let mut rng = Rng(seed | 1);
    let mut limits = Limits::standard();
    if rng.chance(4) {
        limits.max_packet_bytes = 1 + rng.below(4096) as u64;
        limits.max_index_entries = 1 + rng.below(100) as u32;
        limits.max_extradata_bytes = rng.below(64) as u64;
    }
    let bytes = match rng.below(5) {
        0 => {
            let len = rng.below(512);
            let mut bytes = b"caff\x00\x01\x00\x00".to_vec();
            bytes.extend(rng.bytes(len));
            bytes
        }
        1 | 2 => assembled(&mut rng),
        _ => mutated(&mut rng, corpus),
    };
    demux_checks(&bytes, &mut rng, limits, stats);
    if rng.chance(3) {
        mux_checks(&mut rng, stats);
    }
}

#[test]
fn fuzz_seeds_do_not_panic() {
    let corpus = corpus();
    let mut stats = Stats::default();
    for seed in 1_u64..2_000 {
        exercise(
            seed.wrapping_mul(0x9E37_79B9_7F4A_7C15),
            &corpus,
            &mut stats,
        );
    }
    assert!(
        stats.opened > 200 && stats.packets > 1_000 && stats.seeks > 300 && stats.muxed > 100,
        "opened {} packets {} seeks {} muxed {}",
        stats.opened,
        stats.packets,
        stats.seeks,
        stats.muxed
    );
}

#[test]
#[ignore = "demux and mux campaign; set CAF_FUZZ_SECS, default 1800"]
fn fuzz_campaign() {
    let secs = std::env::var("CAF_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1800_u64);
    let budget = Duration::from_secs(secs);
    let corpus = corpus();
    let started = Instant::now();
    let origin = std::env::var("CAF_FUZZ_SEED")
        .ok()
        .and_then(|value| u64::from_str_radix(value.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x00CA_FFEE);
    let mut seed = origin;
    let mut stats = Stats::default();
    while started.elapsed() < budget {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| exercise(seed, &corpus, &mut stats)))
        {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|text| (*text).to_owned())
                })
                .unwrap_or_default();
            panic!("seed {seed:#x} panicked: {message}");
        }
    }
    eprintln!(
        "caf fuzz: {} cases in {:.0} s from origin {origin:#x}, corpus {} files; {} opened, \
         {} packets read, {} seeks landed, {} muxes finished and read back",
        stats.cases,
        started.elapsed().as_secs_f64(),
        corpus.len(),
        stats.opened,
        stats.packets,
        stats.seeks,
        stats.muxed
    );
}
