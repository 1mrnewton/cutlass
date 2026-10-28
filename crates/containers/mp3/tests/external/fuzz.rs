//! In-process demux and mux campaign.
//!
//! This is not libFuzzer. Run it with:
//! `MP3_FUZZ_SECS=1800 cargo test --release -p cutlass-mp3-container --test external -- --ignored --nocapture`

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use crate::support::mpeg1_frame;
use cutlass_codec::{CodecId, CodecParams, Packet};
use cutlass_container::{Demux, Mux};
use cutlass_mp3_container::{MAX_ID3_BYTES, Mp3Demux, Mp3Mux, probe};

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
#[ignore = "demux campaign; set MP3_FUZZ_SECS, default 1800"]
fn demux_and_mux_campaign() {
    let secs = std::env::var("MP3_FUZZ_SECS")
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
    eprintln!("mp3 fuzz exercised {exercised} seeds in {secs} seconds");
}

fn exercise(seed: u64) {
    let mut rng = Rng { state: seed };
    let kind = rng.below(4);
    let bytes = match kind {
        0 => {
            let len = rng.below(8 * 1024) + 1;
            rng.bytes(len)
        }
        1 => truncated_valid(&mut rng),
        2 => hostile_id3(),
        _ => flipped_valid(&mut rng),
    };
    let _ = probe(&bytes);
    if let Ok(mut demux) = Mp3Demux::open(Cursor::new(bytes)) {
        for _ in 0..48 {
            match demux.next_packet() {
                Ok(Some(packet)) => {
                    assert!(packet.data.len() <= 4096);
                    assert_ne!(packet.timescale, 1_000_000);
                }
                Ok(None) | Err(_) => break,
            }
        }
        let _ = demux.seek(i64::try_from(rng.below(2_000_000)).unwrap_or(0));
        let _ = demux.seek(-1);
        let _ = demux.next_packet();
    }
    exercise_mux(&mut rng);
}

fn truncated_valid(rng: &mut Rng) -> Vec<u8> {
    let mut file = mpeg1_frame(9, 1, false, 1);
    file.extend(mpeg1_frame(9, 1, false, 2));
    let keep = rng.below(file.len());
    file.truncate(keep);
    file
}

fn hostile_id3() -> Vec<u8> {
    let mut header = vec![0u8; 10];
    header[..4].copy_from_slice(b"ID3\x04");
    header[6] = 0x0F;
    header[7] = 0x7F;
    header[8] = 0x7F;
    header[9] = 0x7F;
    let err = Mp3Demux::open(Cursor::new(header.clone())).expect_err("limit");
    assert!(matches!(
        err,
        cutlass_container::ContainerError::Limit { .. }
    ));
    assert!(header.len() < usize::try_from(MAX_ID3_BYTES).unwrap_or(usize::MAX));
    header
}

fn flipped_valid(rng: &mut Rng) -> Vec<u8> {
    let mut file = mpeg1_frame(9, 1, false, 3);
    file.extend(mpeg1_frame(9, 1, true, 4));
    let at = rng.below(file.len());
    file[at] ^= 1 << rng.below(8);
    file
}

fn exercise_mux(rng: &mut Rng) {
    let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
    let _ = mux.add_stream(CodecParams::audio(CodecId::Mp3, 48_000, 2));
    let count = rng.below(6);
    for index in 0..count {
        let frame = if rng.below(5) == 0 {
            let len = rng.below(500) + 1;
            rng.bytes(len)
        } else {
            mpeg1_frame(9, 1, index % 2 == 1, u16::try_from(index).unwrap_or(0))
        };
        let pts = i64::try_from(index).unwrap_or(0) * 1152;
        let _ = mux.write_packet(&Packet::audio(frame, pts, 1152, 48_000));
    }
    let _ = mux.finish();
}
