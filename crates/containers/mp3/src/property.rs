//! Sizes in these tests are the ISO Layer III formula written out here.
//! They are not taken from [`crate::header::encode_layer3`].

use std::io::Cursor;

use crate::api::{CodecId, CodecParams, Packet};
use crate::api::{ContainerError, Demux, Mux};
use crate::demux::Mp3Demux;
use crate::header::{MpegVersion, frame_index, parse_header, synthetic_frame};
use crate::mux::Mp3Mux;
use crate::probe::probe;

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

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn layer3_sizes_match_the_iso_formula() {
    const MPEG1: [u32; 14] = [
        32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ];
    const LSF: [u32; 14] = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
    const RATES: [u32; 3] = [44_100, 48_000, 32_000];
    let mut rng = Rng { state: 0xC0FF_EE00 };
    for _ in 0..300 {
        let generation = rng.below(3);
        let version_bits: u32 = match generation {
            0 => 0b11,
            1 => 0b10,
            _ => 0b00,
        };
        let row: &[u32] = if generation == 0 { &MPEG1 } else { &LSF };
        let slot =
            usize::try_from(rng.below(u64::try_from(row.len()).expect("len"))).expect("slot");
        let kbps = row[slot];
        let bitrate_index = u32::try_from(slot + 1).expect("index");
        let rate_index = u32::try_from(rng.below(3)).expect("rate");
        let rate_shift = u32::try_from(generation).expect("shift");
        let rate = RATES[usize::try_from(rate_index).expect("rate")] >> rate_shift;
        let padding = u32::try_from(rng.below(2)).expect("pad");
        let mono = rng.below(2) == 0;
        let mode: u32 = if mono { 0b11 } else { 0b00 };
        let raw = (0x7FF << 21)
            | (version_bits << 19)
            | (0b01 << 17)
            | (1 << 16)
            | (bitrate_index << 12)
            | (rate_index << 10)
            | (padding << 9)
            | (mode << 6);
        let header = parse_header(raw).expect("header");
        // Layer III: 144000 * kbps / sample_rate for MPEG-1, and half of that
        // for MPEG-2 and MPEG-2.5. The sample rate is already the shifted rate.
        let lsf = u32::from(generation != 0);
        let expect = kbps * 144_000 / (rate << lsf) + padding;
        assert_eq!(header.frame_bytes, Some(expect));
        assert_eq!(header.sample_rate, rate);
        assert_eq!(header.bitrate_kbps, kbps);
        assert_eq!(
            header.samples_per_frame,
            if generation == 0 { 1152 } else { 576 }
        );
        assert_eq!(header.channels, if mono { 1 } else { 2 });
    }
}

#[test]
fn random_bytes_fail_closed() {
    let mut rng = Rng { state: 0xA11C_E000 };
    for _ in 0..48 {
        let len = usize::try_from(rng.below(4_000) + 1).expect("len");
        let mut bytes = vec![0u8; len];
        for byte in &mut bytes {
            *byte = u8::try_from(rng.below(256)).expect("byte");
        }
        let _ = probe(&bytes);
        let Ok(mut demux) = Mp3Demux::open(Cursor::new(bytes)) else {
            continue;
        };
        for _ in 0..32 {
            match demux.next_packet() {
                Ok(Some(packet)) => {
                    assert_eq!(packet.timescale, demux.streams()[0].timescale);
                    assert_ne!(packet.timescale, 1_000_000);
                }
                Ok(None) | Err(_) => break,
            }
        }
        let _ = demux.seek(i64::try_from(rng.below(50_000)).expect("time"));
    }
}

#[test]
fn mux_accounts_for_every_packet_on_the_sample_clock() {
    let rates = [128u32, 96, 160, 112, 192, 64, 80, 224];
    let mut rng = Rng { state: 0x0B0B };
    let count = 16i64;
    let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
    mux.add_stream(CodecParams::audio(CodecId::Mp3, 48_000, 2))
        .expect("stream");
    for index in 0..count {
        let slot =
            usize::try_from(rng.below(u64::try_from(rates.len()).expect("n"))).expect("slot");
        let frame = synthetic_frame(
            MpegVersion::Mpeg1,
            rates[slot],
            48_000,
            2,
            u16::try_from(index).expect("index"),
        );
        mux.write_packet(&Packet::audio(frame, index * 1152, 1152, 48_000))
            .expect("write");
    }
    mux.finish().expect("finish");
    let bytes = mux.into_inner().expect("inner").into_inner();
    let mut demux = Mp3Demux::open(Cursor::new(bytes)).expect("open");
    assert_eq!(demux.duration(), Some(count * 1152));
    assert!(demux.duration_exact());
    let mut sum = 0i64;
    for index in 0..count {
        let packet = demux.next_packet().expect("pkt").expect("some");
        assert_eq!(packet.pts, sum);
        assert_eq!(packet.pts, index * 1152);
        assert_eq!(packet.duration, 1152);
        assert_eq!(packet.timescale, 48_000);
        assert!(packet.keyframe);
        sum += packet.duration;
    }
    assert!(demux.next_packet().expect("end").is_none());
    assert_eq!(sum, count * 1152);
    let landing = demux.seek(0).expect("start");
    assert_eq!(landing.pts, 0);
    let mid = demux.seek(7 * 1152).expect("mid");
    assert_eq!(mid.pts, 7 * 1152);
    assert_eq!(
        frame_index(&demux.next_packet().expect("pkt").expect("some").data),
        7
    );
}

#[test]
fn a_truncated_frame_and_a_hostile_id3_do_not_yield_a_short_packet() {
    let frame = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 0);
    let err = Mp3Demux::open(Cursor::new(frame[..frame.len() - 1].to_vec())).expect_err("short");
    assert!(matches!(
        err,
        ContainerError::Truncated { .. } | ContainerError::NoFrames
    ));
    let mut id3 = [0u8; 10];
    id3[..4].copy_from_slice(b"ID3\x04");
    id3[6] = 0x0F;
    id3[7] = 0x7F;
    id3[8] = 0x7F;
    id3[9] = 0x7F;
    assert_eq!(
        Mp3Demux::open(Cursor::new(id3.to_vec())).expect_err("id3"),
        ContainerError::Limit { what: "id3" }
    );
}
