//! In-process demux and mux campaign.
//!
//! This is not libFuzzer: a seeded generator builds files from the muxer's
//! output and from random chunk layouts, mutates them, and drives probe,
//! open, read, and seek on each, under `catch_unwind`. Run it with:
//! `AVI_FUZZ_SECS=1800 cargo test --release -p cutlass-avi --test external fuzz_campaign -- --ignored --nocapture`

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cutlass_avi::{AviDemux, AviForwardDemux, AviMux, AviMuxOptions, probe};
use cutlass_codec::{CodecId, FrameSize, StreamCodec, StreamParams, Timescale};
use cutlass_container::{Demux, Limits, Mux, NewStream, SeekTo, TrackDuration, TrackId};

use crate::support::{movie, stamp};

struct Rng {
    state: u64,
}

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        usize::try_from(self.next() % u64::try_from(n).expect("n")).unwrap_or(0)
    }

    fn byte(&mut self) -> u8 {
        u8::try_from(self.next() & 0xFF).unwrap_or(0)
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.byte()).collect()
    }

    /// A size field value: small, near a boundary, or huge.
    fn size(&mut self, around: usize) -> u32 {
        let around = u32::try_from(around).unwrap_or(u32::MAX);
        match self.below(6) {
            0 => 0,
            1 => u32::MAX - u32::try_from(self.below(16)).unwrap_or(0),
            2 => around
                .wrapping_add(u32::try_from(self.below(9)).unwrap_or(0))
                .wrapping_sub(4),
            3 => 0x8000_0000 | u32::try_from(self.below(1 << 20)).unwrap_or(0),
            4 => u32::try_from(self.below(64)).unwrap_or(0),
            _ => u32::try_from(self.next() & 0xFFFF_FFFF).unwrap_or(0),
        }
    }
}

#[test]
fn fuzz_seeds_do_not_panic() {
    for seed in 1u64..200 {
        let _ = exercise(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
}

#[test]
#[ignore = "demux and mux campaign; set AVI_FUZZ_SECS, default 1800"]
fn fuzz_campaign() {
    let secs = std::env::var("AVI_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1800u64);
    let budget = Duration::from_secs(secs);
    let started = Instant::now();
    let origin = std::env::var("AVI_FUZZ_SEED")
        .ok()
        .and_then(|value| u64::from_str_radix(value.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x00A5_1000_u64);
    let mut seed = origin;
    let mut exercised = 0u64;
    let mut slowest = Duration::ZERO;
    let (mut opened, mut packets, mut forward, mut muxed) = (0u64, 0u64, 0u64, 0u64);
    while started.elapsed() < budget {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let one = Instant::now();
        let result = catch_unwind(AssertUnwindSafe(|| exercise(seed)));
        match result {
            Ok(reach) => {
                opened += reach.opened;
                packets += reach.packets;
                forward += reach.forward;
                muxed += reach.muxed;
            }
            Err(payload) => panic!("seed {seed:#x} panicked: {payload:?}"),
        }
        slowest = slowest.max(one.elapsed());
        exercised += 1;
    }
    assert!(exercised > 0);
    eprintln!(
        "avi fuzz exercised {exercised} seeds in {secs} seconds from origin {origin:#x}, slowest {slowest:?}; {opened} opened, {packets} packets read, {forward} read forward, {muxed} mux runs finished and read back"
    );
}

/// What one seed reached.
#[derive(Default)]
struct Reach {
    opened: u64,
    packets: u64,
    forward: u64,
    muxed: u64,
}

fn exercise(seed: u64) -> Reach {
    let mut rng = Rng { state: seed | 1 };
    let mut reach = Reach::default();
    match rng.below(10) {
        0 => reach.muxed = u64::from(fuzz_mux(&mut rng)),
        _ => {
            let bytes = input(&mut rng);
            drive(&bytes, &mut rng, &mut reach);
        }
    }
    reach
}

/// A file to feed the demuxer.
fn input(rng: &mut Rng) -> Vec<u8> {
    let mut bytes = match rng.below(4) {
        0 => {
            let len = rng.below(2048) + 1;
            let mut out = b"RIFF\0\0\0\0AVI ".to_vec();
            out.extend(rng.bytes(len));
            out
        }
        1 => layout(rng),
        _ => {
            let frames = u32::try_from(rng.below(40) + 1).unwrap_or(1);
            let riff = [1 << 30, 4_096, 16_384][rng.below(3)];
            movie(frames, u32::try_from(rng.below(8) + 1).unwrap_or(1), riff).bytes
        }
    };
    let mutations = rng.below(6);
    for _ in 0..mutations {
        mutate(rng, &mut bytes);
    }
    bytes
}

/// Chunk ids the parser looks at, so random layouts reach its branches.
const IDS: [&[u8; 4]; 16] = [
    b"LIST", b"RIFF", b"avih", b"strh", b"strf", b"indx", b"idx1", b"JUNK", b"00dc", b"01wb",
    b"ix00", b"ix01", b"dmlh", b"00db", b"00pc", b"strn",
];
const FORMS: [&[u8; 4]; 6] = [b"hdrl", b"strl", b"movi", b"odml", b"rec ", b"INFO"];

fn layout(rng: &mut Rng) -> Vec<u8> {
    let mut body = Vec::new();
    for _ in 0..rng.below(24) {
        let id = IDS[rng.below(IDS.len())];
        let mut payload = if id == b"LIST" {
            let mut form = FORMS[rng.below(FORMS.len())].to_vec();
            let inner = layout_children(rng, 2);
            form.extend(inner);
            form
        } else {
            let len = rng.below(80);
            rng.bytes(len)
        };
        if rng.below(4) == 0 {
            payload.truncate(rng.below(payload.len() + 1));
        }
        chunk_into(&mut body, id, &payload, rng);
    }
    let mut out = b"RIFF".to_vec();
    out.extend((u32::try_from(body.len() + 4).unwrap_or(0)).to_le_bytes());
    out.extend(b"AVI ");
    out.extend(body);
    out
}

fn layout_children(rng: &mut Rng, depth: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for _ in 0..rng.below(6) {
        let id = IDS[rng.below(IDS.len())];
        let payload = if id == b"LIST" && depth > 0 {
            let mut form = FORMS[rng.below(FORMS.len())].to_vec();
            form.extend(layout_children(rng, depth - 1));
            form
        } else {
            let len = match id {
                b"avih" => 56,
                b"strh" => 56,
                b"strf" => 40 + rng.below(24),
                b"indx" => 24 + 16 * rng.below(4),
                _ => rng.below(64),
            };
            rng.bytes(len)
        };
        chunk_into(&mut out, id, &payload, rng);
    }
    out
}

fn chunk_into(out: &mut Vec<u8>, id: &[u8; 4], payload: &[u8], rng: &mut Rng) {
    out.extend(id);
    let size = if rng.below(8) == 0 {
        rng.size(payload.len())
    } else {
        u32::try_from(payload.len()).unwrap_or(0)
    };
    out.extend(size.to_le_bytes());
    out.extend(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

fn mutate(rng: &mut Rng, bytes: &mut Vec<u8>) {
    if bytes.is_empty() {
        return;
    }
    match rng.below(7) {
        0 => {
            let at = rng.below(bytes.len());
            bytes[at] = rng.byte();
        }
        1 => {
            let at = rng.below(bytes.len());
            bytes.truncate(at.max(1));
        }
        2 => {
            // Rewrite the size after a chunk id the parser knows.
            let id = IDS[rng.below(IDS.len())];
            if let Some(found) = bytes.windows(4).position(|window| window == id)
                && found + 8 <= bytes.len()
            {
                let size = rng.size(bytes.len() - found);
                bytes[found + 4..found + 8].copy_from_slice(&size.to_le_bytes());
            }
        }
        3 => {
            let at = rng.below(bytes.len());
            let len = rng.below(64);
            let junk = rng.bytes(len);
            bytes.splice(at..at, junk);
        }
        4 => {
            let at = rng.below(bytes.len());
            let len = rng.below(bytes.len() - at).min(4096);
            let copy = bytes[at..at + len].to_vec();
            let to = rng.below(bytes.len());
            bytes.splice(to..to, copy);
        }
        5 => {
            let at = rng.below(bytes.len());
            let end = (at + rng.below(32)).min(bytes.len());
            bytes.drain(at..end);
        }
        _ => {
            // A 32-bit field anywhere: an index offset, a count, a scale.
            if bytes.len() >= 4 {
                let at = rng.below(bytes.len() - 3);
                let value = rng.size(bytes.len());
                bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
}

fn drive(bytes: &[u8], rng: &mut Rng, reach: &mut Reach) {
    let _ = probe(&bytes[..bytes.len().min(64)]);
    let mut limits = Limits::standard();
    if rng.below(4) == 0 {
        limits.max_index_entries = u32::try_from(rng.below(64) + 1).unwrap_or(1);
        limits.max_packet_bytes = u64::try_from(rng.below(4096) + 1).unwrap_or(1);
        limits.max_header_bytes = u64::try_from(rng.below(8192) + 12).unwrap_or(12);
        limits.max_tracks = u32::try_from(rng.below(4) + 1).unwrap_or(1);
    }
    drive_forward(bytes, limits, reach);
    let len = u64::try_from(bytes.len()).expect("len");
    let Ok(mut demux) = AviDemux::open_with_limits(Cursor::new(bytes.to_vec()), len, limits) else {
        return;
    };
    reach.opened = 1;
    let tracks = u32::try_from(demux.streams().len()).unwrap_or(0);
    let mut failures = 0;
    for _ in 0..4_000 {
        if rng.below(16) == 0 && tracks > 0 {
            let track =
                u32::try_from(rng.below(usize::try_from(tracks).unwrap_or(0) + 1)).unwrap_or(0);
            let time = i64::try_from(rng.next() % 2_000).unwrap_or(0) - 50;
            let end = if rng.below(2) == 0 {
                None
            } else {
                Some(time + i64::try_from(rng.below(100)).unwrap_or(0))
            };
            let _ = demux.seek(SeekTo { track, time, end });
            continue;
        }
        match demux.read_packet() {
            Ok(Some(packet)) => {
                assert!(u32::try_from(packet.bytes().len()).is_ok());
                assert!(packet.stream() < tracks);
                assert!(packet.timestamp().duration().is_some_and(|span| span >= 0));
                reach.packets += 1;
            }
            Ok(None) => break,
            Err(_) => {
                failures += 1;
                if failures > 8 {
                    break;
                }
            }
        }
    }
    if rng.below(8) == 0 {
        demux.cancel();
        assert!(demux.read_packet().is_err());
    }
    let _ = demux.consumed();
}

/// The same input through the forward-only reader, which never seeks.
fn drive_forward(bytes: &[u8], limits: Limits, reach: &mut Reach) {
    let Ok(mut demux) = AviForwardDemux::open_with_limits(Cursor::new(bytes.to_vec()), limits)
    else {
        return;
    };
    let tracks = u32::try_from(demux.streams().len()).unwrap_or(0);
    let mut failures = 0;
    for _ in 0..4_000 {
        match demux.read_packet() {
            Ok(Some(packet)) => {
                assert!(packet.stream() < tracks);
                assert!(packet.timestamp().duration().is_some_and(|span| span >= 0));
                reach.forward += 1;
            }
            Ok(None) => break,
            Err(_) => {
                failures += 1;
                if failures > 8 {
                    break;
                }
            }
        }
    }
    assert!(demux.consumed() <= u64::try_from(bytes.len()).expect("len"));
}

/// Random packet lists into the muxer. When it finishes, the file must
/// open and give back every packet it took.
fn fuzz_mux(rng: &mut Rng) -> bool {
    let options = AviMuxOptions {
        riff_bytes: [1 << 30, 2_048, 8_192][rng.below(3)],
        super_index_entries: u32::try_from(rng.below(300) + 1).unwrap_or(1),
        header_buffer_bytes: u64::try_from(rng.below(1 << 16) + 1).unwrap_or(1),
        interleave_bytes: u64::try_from(rng.below(1 << 14)).unwrap_or(0),
    };
    let mut mux = AviMux::with_options(Cursor::new(Vec::new()), options);
    let mut streams = Vec::new();
    for _ in 0..rng.below(4) + 1 {
        let (codec, scale) = match rng.below(7) {
            0 => (CodecId::H264, 90_000),
            1 => (CodecId::Mjpeg, 25),
            2 => (CodecId::Pcm, 8_000),
            3 => (CodecId::Mp3, 44_100),
            4 => (CodecId::Aac, 48_000),
            5 => (CodecId::Opus, 48_000),
            _ => (CodecId::Vp9, 1_000),
        };
        let extra_len = rng.below(8);
        let extradata = if rng.below(2) == 0 {
            rng.bytes(extra_len)
        } else {
            Vec::new()
        };
        let params = match codec {
            CodecId::H264 | CodecId::Mjpeg | CodecId::Vp9 => StreamParams::video(
                StreamCodec::Known(codec),
                FrameSize::try_new(16, 16).expect("size"),
                extradata,
            ),
            _ => match StreamParams::try_audio(StreamCodec::Known(codec), scale, 2, extradata) {
                Ok(params) => params,
                Err(_) => continue,
            },
        };
        let timescale = Timescale::new(if rng.below(8) == 0 { scale + 1 } else { scale });
        if mux
            .add_stream(NewStream {
                params,
                timescale,
                language: None,
            })
            .is_ok()
        {
            streams.push((timescale, codec, 0i64));
        }
    }
    let count = rng.below(80);
    let mut written = 0usize;
    let mut empty = 0usize;
    // MP3 packets come back cut at frame headers, so random payloads are
    // counted in bytes, which can only shrink by the garbage dropped.
    let mut mp3_bytes = 0usize;
    for _ in 0..count {
        if streams.is_empty() {
            break;
        }
        let index = rng.below(streams.len());
        let (timescale, codec, next) = streams[index];
        let unit = match codec {
            CodecId::Pcm => i64::try_from(rng.below(4) + 1).unwrap_or(1) * 10,
            CodecId::Aac => 1024,
            CodecId::Mp3 => 1152,
            _ => 40 * i64::from(timescale.ticks_per_second()) / 1000,
        };
        let duration = if rng.below(10) == 0 {
            unit * i64::try_from(rng.below(3)).unwrap_or(0)
        } else {
            unit
        };
        let len = if codec == CodecId::Pcm {
            usize::try_from(duration).unwrap_or(0) * 4
        } else {
            rng.below(600)
        };
        let time = if rng.below(12) == 0 { next + 7 } else { next };
        let packet = stamp(
            u32::try_from(index).unwrap_or(0),
            rng.bytes(len),
            time,
            duration,
            timescale.ticks_per_second(),
            rng.below(3) == 0,
        );
        if mux.write_packet(&packet).is_err() {
            return false;
        }
        streams[index].2 = next + duration;
        if codec == CodecId::Mp3 {
            mp3_bytes += len;
        } else {
            written += 1;
            empty += usize::from(len == 0);
        }
    }
    let durations: Vec<_> = streams
        .iter()
        .enumerate()
        .map(|(index, (timescale, _, ticks))| TrackDuration {
            track: TrackId::new(u32::try_from(index).unwrap_or(0)),
            timescale: *timescale,
            ticks: *ticks,
        })
        .collect();
    if mux.finish(&durations).is_err() {
        return false;
    }
    let bytes = mux.into_writer().expect("writer").into_inner();
    let len = u64::try_from(bytes.len()).expect("len");
    let mut demux = AviDemux::open(Cursor::new(bytes), len).expect("a finished file opens");
    let mut read = 0usize;
    let mut mp3_read = 0usize;
    while let Some(packet) = demux.read_packet().expect("a finished file reads") {
        let stream = usize::try_from(packet.stream()).expect("stream");
        if streams[stream].1 == CodecId::Mp3 {
            mp3_read += packet.bytes().len();
        } else {
            read += 1;
        }
    }
    assert_eq!(read + empty, written, "every packet with bytes comes back");
    assert!(
        mp3_read <= mp3_bytes,
        "MP3 frames come from the bytes written"
    );
    true
}
