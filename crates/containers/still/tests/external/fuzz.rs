//! In-process demux and mux campaign.
//!
//! This is not libFuzzer. Run it with:
//! `STILL_FUZZ_SECS=1800 cargo test --release -p cutlass-still --offline --test external -- --ignored --nocapture`

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cutlass_codec::{FrameSize, StreamCodec, StreamParams};
use cutlass_container::{Demux, Input, Mux, NewStream, SeekTo, TrackDuration, TrackId};
use cutlass_still::{StillDemux, StillMux, probe, still_limits};

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

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut out = vec![0_u8; len];
        for byte in &mut out {
            *byte = u8::try_from(self.next() & 0xFF).unwrap_or(0);
        }
        out
    }
}

#[test]
#[ignore = "demux campaign; set STILL_FUZZ_SECS, default 1800"]
fn demux_and_mux_campaign() {
    let secs = std::env::var("STILL_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1800_u64);
    let budget = Duration::from_secs(secs);
    let started = Instant::now();
    let mut seed = 0x00C0_FFEE_u64;
    let mut exercised = 0_u64;
    while started.elapsed() < budget {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let result = catch_unwind(AssertUnwindSafe(|| exercise(seed)));
        if let Err(payload) = result {
            panic!("seed {seed:#x} panicked: {payload:?}");
        }
        exercised += 1;
    }
    assert!(exercised > 0);
    eprintln!("still fuzz exercised {exercised} seeds in {secs} seconds");
}

fn exercise(seed: u64) {
    let mut rng = Rng { state: seed };
    let bytes = match rng.below(5) {
        0 => {
            let len = rng.below(8 * 1024) + 1;
            rng.bytes(len)
        }
        1 => truncated_jpeg(&mut rng),
        2 => hostile_riff(),
        3 => flipped(&mut rng, jpeg()),
        _ => flipped(&mut rng, gif()),
    };
    let _ = probe(&bytes);
    let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let Ok(mut demux) = StillDemux::open(Input::random(Cursor::new(bytes), len, still_limits()))
    else {
        return;
    };
    let mut packets = Vec::new();
    for _ in 0..32 {
        match demux.read_packet() {
            Ok(Some(packet)) => packets.push(packet),
            Ok(None) => break,
            Err(_) => return,
        }
    }
    let _ = demux.seek(SeekTo {
        track: 0,
        time: 0,
        end: demux.media_end(),
    });
    if packets.is_empty() {
        return;
    }
    let Some(stream) = demux.streams().first() else {
        return;
    };
    let StreamCodec::Known(codec) = stream.params().codec().clone() else {
        return;
    };
    let mut mux = StillMux::new();
    let size = stream
        .params()
        .size()
        .unwrap_or_else(|| FrameSize::try_new(1, 1).expect("size"));
    if mux
        .add_stream(NewStream {
            params: StreamParams::video(StreamCodec::Known(codec), size, Vec::new()),
            timescale: stream.timescale(),
            language: None,
        })
        .is_err()
    {
        return;
    }
    for packet in packets.iter().take(4) {
        if mux.write_packet(packet).is_err() {
            return;
        }
    }
    let ticks = if stream.timescale().is_clock() {
        packets
            .iter()
            .take(4)
            .filter_map(|packet| packet.timestamp().duration())
            .fold(0_i64, i64::saturating_add)
    } else {
        0
    };
    let _ = mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: stream.timescale(),
        ticks,
    }]);
}

fn jpeg() -> Vec<u8> {
    vec![
        0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x00, 0xFF, 0xD9,
    ]
}

fn gif() -> Vec<u8> {
    let mut out = b"GIF89a".to_vec();
    out.extend([
        0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00,
    ]);
    out.extend([
        0x2C, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x4C, 0x01, 0x00,
        0x3B,
    ]);
    out
}

fn truncated_jpeg(rng: &mut Rng) -> Vec<u8> {
    let full = jpeg();
    let keep = rng.below(full.len());
    full[..keep].to_vec()
}

fn hostile_riff() -> Vec<u8> {
    let mut out = b"RIFF".to_vec();
    out.extend(u32::MAX.to_le_bytes());
    out.extend(b"WEBP");
    out
}

fn flipped(rng: &mut Rng, mut bytes: Vec<u8>) -> Vec<u8> {
    if bytes.is_empty() {
        return bytes;
    }
    let index = rng.below(bytes.len());
    bytes[index] ^= 0xFF;
    bytes
}
