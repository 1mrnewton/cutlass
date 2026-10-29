//! In-process demux and mux campaign.
//!
//! This is not libFuzzer. Run it with:
//! `WAV_FUZZ_SECS=1800 CARGO_TARGET_DIR=/tmp/cutlass-riff-fuzz cargo test --release -p cutlass-riff --test external fuzz_campaign -- --ignored --nocapture`

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, SeekTo, TrackDuration, TrackId};
use cutlass_riff::{WavDemux, WavMux, probe};

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
        let mut out = vec![0u8; len];
        for byte in &mut out {
            *byte = u8::try_from(self.next() & 0xFF).unwrap_or(0);
        }
        out
    }
}

#[test]
fn fuzz_seeds_do_not_panic() {
    for seed in 1u64..65 {
        exercise(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
}

#[test]
#[ignore = "demux and mux campaign; set WAV_FUZZ_SECS, default 1800"]
fn fuzz_campaign() {
    let secs = std::env::var("WAV_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1800u64);
    let budget = Duration::from_secs(secs);
    let started = Instant::now();
    let mut seed = 0x00C0_FFEE_u64;
    let mut exercised = 0u64;
    while started.elapsed() < budget {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let result = catch_unwind(AssertUnwindSafe(|| exercise(seed)));
        if let Err(payload) = result {
            panic!("seed {seed:#x} panicked: {payload:?}");
        }
        exercised += 1;
    }
    assert!(exercised > 0);
    eprintln!("wav fuzz exercised {exercised} seeds in {secs} seconds, origin {seed:#x}");
}

fn exercise(seed: u64) {
    let mut rng = Rng { state: seed };
    let bytes = match rng.below(5) {
        0 => {
            let len = rng.below(4 * 1024) + 1;
            rng.bytes(len)
        }
        1 => truncated_valid(&mut rng),
        2 => hostile_size(),
        3 => flipped_valid(&mut rng),
        _ => nested_list(),
    };
    let window = &bytes[..bytes.len().min(12)];
    let _ = probe(window);
    let len = u64::try_from(bytes.len()).unwrap_or(0);
    if let Ok(mut demux) = WavDemux::open(Cursor::new(bytes.clone()), len) {
        assert_ne!(demux.streams()[0].timescale().ticks_per_second(), 1_000_000);
        for _ in 0..8 {
            match demux.read_packet() {
                Ok(Some(packet)) => {
                    assert!(packet.bytes().len() <= 16 * 1024 * 1024);
                    assert_ne!(packet.timescale().ticks_per_second(), 1_000_000);
                }
                Ok(None) | Err(_) => break,
            }
        }
        let _ = demux.seek(SeekTo {
            track: 0,
            time: i64::try_from(rng.below(8_000)).unwrap_or(0),
            end: None,
        });
        let _ = demux.seek(SeekTo {
            track: 0,
            time: -1,
            end: None,
        });
        let _ = demux.read_packet();
    }
    let _ = WavDemux::open_forward(Cursor::new(bytes));
    exercise_mux(&mut rng);
}

fn stereo_body(frames: usize) -> Vec<u8> {
    let mut data = vec![0u8; frames * 4];
    for (index, byte) in data.iter_mut().enumerate() {
        *byte = u8::try_from(index & 0xFF).unwrap_or(0);
    }
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1u16.to_le_bytes());
    fmt.extend_from_slice(&2u16.to_le_bytes());
    fmt.extend_from_slice(&44_100u32.to_le_bytes());
    fmt.extend_from_slice(&176_400u32.to_le_bytes());
    fmt.extend_from_slice(&4u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());
    let mut body = chunk(b"fmt ", &fmt);
    body.extend(chunk(b"data", &data));
    riff(&body)
}

fn truncated_valid(rng: &mut Rng) -> Vec<u8> {
    let mut file = stereo_body(8);
    let keep = rng.below(file.len());
    file.truncate(keep);
    file
}

fn flipped_valid(rng: &mut Rng) -> Vec<u8> {
    let mut file = stereo_body(4);
    if !file.is_empty() {
        let at = rng.below(file.len());
        file[at] ^= 0xFF;
    }
    file
}

fn hostile_size() -> Vec<u8> {
    let mut file = b"RIFF".to_vec();
    file.extend_from_slice(&u32::MAX.to_le_bytes());
    file.extend_from_slice(b"WAVE");
    file.extend_from_slice(b"fmt ");
    file.extend_from_slice(&u32::MAX.to_le_bytes());
    file.extend_from_slice(&[0u8; 8]);
    file
}

fn nested_list() -> Vec<u8> {
    let mut inner = chunk(b"JUNK", &[0xFF]);
    for _ in 0..12 {
        let mut payload = b"INFO".to_vec();
        payload.extend(inner);
        inner = chunk(b"LIST", &payload);
    }
    riff(&inner)
}

fn chunk(id: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(id);
    let size = u32::try_from(payload.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(payload);
    if !payload.len().is_multiple_of(2) {
        out.push(0);
    }
    out
}

fn riff(body: &[u8]) -> Vec<u8> {
    let mut out = b"RIFF".to_vec();
    let size = u32::try_from(4 + body.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(body);
    out
}

fn exercise_mux(rng: &mut Rng) {
    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    let params = StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 8_000, 1, Vec::new())
        .expect("pcm");
    if mux
        .add_stream(NewStream {
            params,
            timescale: Timescale::new(8_000),
            language: None,
        })
        .is_err()
    {
        return;
    }
    let frames = rng.below(6);
    for index in 0..frames {
        let at = i64::try_from(index).unwrap_or(0);
        let timestamp = Timestamp::try_new(Some(at), Some(at), Some(1)).expect("time");
        let packet =
            Packet::try_new(0, vec![0, 1], timestamp, Timescale::new(8_000), true).expect("packet");
        if mux.write_packet(&packet).is_err() {
            return;
        }
    }
    let ticks = i64::try_from(frames).unwrap_or(0);
    let claimed = if rng.below(5) == 0 {
        ticks.saturating_add(3)
    } else {
        ticks
    };
    let finished = mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(8_000),
        ticks: claimed,
    }]);
    if let Ok(done) = finished {
        assert!(done.is_complete());
        let bytes = mux.into_inner().into_inner();
        let len = u64::try_from(bytes.len()).unwrap_or(0);
        if let Ok(mut demux) = WavDemux::open(Cursor::new(bytes), len) {
            let _ = demux.read_packet();
        }
    }
}
