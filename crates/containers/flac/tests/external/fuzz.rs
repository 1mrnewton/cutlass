//! In-process demux and mux campaign. This is not libFuzzer.
//!
//! The short run is part of the suite. The long one is:
//! `FLAC_FUZZ_SECS=1800 cargo test --release -p cutlass-flac-container --test external -- --ignored --nocapture`

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, SeekTo, TrackDuration, TrackId};
use cutlass_flac::{FlacEncoder, Settings};
use cutlass_flac_container::{FlacDemux, FlacMux, probe};

use crate::support::{Kind, Rng, signal};

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n.max(1) as u64).unwrap_or(0)
    }
}

/// A small valid file and its frames: random depth, channels, block size,
/// and seek-table size.
fn base(rng: &mut Rng) -> (Vec<u8>, Vec<Vec<u8>>, u32) {
    let channels = [1u8, 2, 3, 6][rng.below(4)];
    let bits = [8u8, 16, 24][rng.below(3)];
    let rate = [8000u32, 44_100, 96_000][rng.below(3)];
    let mut settings = Settings::level(u8::try_from(rng.below(9)).unwrap_or(0)).expect("level");
    settings.block_size = [16u16, 192, 256, 1000, 4096][rng.below(5)];
    let frames = 200 + rng.below(6000);
    let samples = signal(
        Kind::Music,
        usize::from(channels),
        u32::from(bits),
        frames,
        rng.next(),
    );
    let mut encoder = FlacEncoder::new(rate, channels, bits, settings).expect("encoder");
    let mut encoded = encoder.encode(&samples).expect("encode");
    encoded.extend(encoder.finish().expect("finish"));
    let mut mux = FlacMux::with_seek_points(
        Cursor::new(Vec::new()),
        u32::try_from(rng.below(5)).unwrap_or(0),
    );
    let params = StreamParams::try_audio(
        StreamCodec::Known(CodecId::Flac),
        rate,
        u16::from(channels),
        encoder.stream_info().to_bytes().to_vec(),
    )
    .expect("params");
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(rate),
        language: None,
    })
    .expect("stream");
    let mut frames_out = Vec::new();
    for frame in &encoded {
        let packet = file_packet(frame.data.clone(), frame.first_sample, frame.samples, rate);
        mux.write_packet(&packet).expect("write");
        frames_out.push(frame.data.clone());
    }
    let ticks = encoded.iter().map(|f| i64::from(f.samples)).sum();
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(rate),
        ticks,
    }])
    .expect("finish");
    (mux.into_inner().into_inner(), frames_out, rate)
}

fn file_packet(data: Vec<u8>, pts: u64, duration: u32, rate: u32) -> Packet {
    let pts = i64::try_from(pts).unwrap_or(0);
    let timestamp =
        Timestamp::try_new(Some(pts), Some(pts), Some(i64::from(duration))).expect("timestamp");
    Packet::try_new(0, data, timestamp, Timescale::new(rate), true).expect("packet")
}

fn mutate(rng: &mut Rng, bytes: &mut Vec<u8>) {
    match rng.below(6) {
        0 => {
            for _ in 0..1 + rng.below(8) {
                let at = rng.below(bytes.len());
                bytes[at] ^= 1 << rng.below(8);
            }
        }
        1 => {
            let keep = rng.below(bytes.len());
            bytes.truncate(keep);
        }
        2 => {
            let at = rng.below(bytes.len());
            for _ in 0..rng.below(64) {
                bytes.insert(at, (rng.next() & 0xFF) as u8);
            }
        }
        3 => {
            // A metadata block header or the first bytes of a frame.
            let at = 4 + rng.below(bytes.len().min(64).saturating_sub(4));
            if at < bytes.len() {
                bytes[at] = (rng.next() & 0xFF) as u8;
            }
        }
        4 => {
            let from = rng.below(bytes.len());
            let len = rng.below(bytes.len() - from);
            let copy = bytes[from..from + len].to_vec();
            let at = rng.below(bytes.len());
            bytes.splice(at..at, copy);
        }
        _ => {
            let len = rng.below(4096);
            *bytes = b"fLaC".to_vec();
            bytes.extend((0..len).map(|_| (rng.next() & 0xFF) as u8));
        }
    }
}

fn exercise(seed: u64) {
    let mut rng = Rng(seed | 1);
    let (file, frames, rate) = base(&mut rng);
    let mut bytes = file;
    for _ in 0..1 + rng.below(3) {
        if bytes.is_empty() {
            break;
        }
        mutate(&mut rng, &mut bytes);
    }
    let _ = probe(&bytes);
    let len = bytes.len() as u64;
    if let Ok(mut demux) = FlacDemux::open(Cursor::new(bytes), len) {
        for round in 0..3 {
            for _ in 0..64 {
                match demux.read_packet() {
                    Ok(Some(_)) => {}
                    _ => break,
                }
            }
            let time = i64::try_from(rng.next() % 400_000).unwrap_or(0) - 1000 * round;
            let _ = demux.seek(SeekTo {
                track: 0,
                time,
                end: None,
            });
        }
        let _ = demux.consumed();
    }
    // The muxer with damaged and reordered frames.
    let mut mux = FlacMux::new(Cursor::new(Vec::new()));
    let params = StreamParams::try_audio(StreamCodec::Known(CodecId::Flac), rate, 2, Vec::new());
    if let Ok(params) = params {
        let _ = mux.add_stream(NewStream {
            params,
            timescale: Timescale::new(rate),
            language: None,
        });
    }
    let mut pts = 0u64;
    for frame in frames.iter().take(8) {
        let mut data = frame.clone();
        if rng.below(3) == 0 && !data.is_empty() {
            let at = rng.below(data.len());
            data[at] ^= 1 << rng.below(8);
        }
        let duration = 16 + u32::try_from(rng.below(5000)).unwrap_or(0);
        let _ = mux.write_packet(&file_packet(data, pts, duration, rate));
        pts += u64::from(duration);
    }
    let _ = mux.set_config(TrackId::new(0), &frames[0]);
    let _ = mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(rate),
        ticks: i64::try_from(pts).unwrap_or(0),
    }]);
}

fn campaign(budget: Duration, seeds: u64) -> u64 {
    let started = Instant::now();
    let mut seed = 0x00F1_AC00_u64;
    let mut exercised = 0u64;
    while started.elapsed() < budget && exercised < seeds {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| exercise(seed))) {
            panic!("seed {seed:#x} panicked: {payload:?}");
        }
        exercised += 1;
    }
    exercised
}

#[test]
fn a_short_demux_and_mux_campaign() {
    let exercised = campaign(Duration::from_secs(20), 2000);
    assert!(exercised > 0);
    eprintln!("flac container fuzz exercised {exercised} seeds");
}

#[test]
#[ignore = "long demux and mux campaign; set FLAC_FUZZ_SECS, default 1800"]
fn demux_and_mux_campaign() {
    let secs = std::env::var("FLAC_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1800u64);
    let exercised = campaign(Duration::from_secs(secs), u64::MAX);
    assert!(exercised > 0);
    eprintln!("flac container fuzz exercised {exercised} seeds in {secs} seconds");
}
