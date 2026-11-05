//! Lifecycle, malformed input, cancel, and the decoder fuzz run.

use std::sync::atomic::Ordering;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecError, CodecId, CodecParams, Packet};
use cutlass_encode::PixelFormat;
use cutlass_encode::video::{VideoEncoder, VideoInput};

use crate::decode::Decoder;
use crate::prove::{Ivf, decode_shown};
use crate::prove_encode::{crop, encode_synthetic, synthetic};
use crate::{Vp8Decoder, Vp8Encoder, Vp8Options};

fn input(width: u32, height: u32, t: u32) -> VideoInput {
    VideoInput {
        frame: synthetic(width, height, t),
        pts: i64::from(t),
        duration: 1,
        timescale: 30,
    }
}

fn pack(picture: &crate::Picture) -> Vec<u8> {
    let mut out = Vec::new();
    crop(&picture.frame, picture.width, picture.height, &mut out);
    out
}

#[test]
fn reset_drops_references() {
    let (frames, _) = encode_synthetic(48, 32, 3, Vp8Options::default());
    let mut decoder = Decoder::new();
    decoder.decode(&frames[0]).expect("key");
    decoder.decode(&frames[1]).expect("inter");
    decoder.reset();
    assert!(matches!(
        decoder.decode(&frames[2]),
        Err(CodecError::Malformed { .. })
    ));
    assert!(decoder.decode(&frames[0]).expect("key again").is_some());
}

#[test]
fn failed_frames_leave_the_decoder_as_it_was() {
    let (frames, _) = encode_synthetic(64, 48, 4, Vp8Options::default());
    let mut clean = Decoder::new();
    let want: Vec<_> = frames
        .iter()
        .map(|f| pack(&clean.decode(f).expect("clean").expect("shown")))
        .collect();
    let mut decoder = Decoder::new();
    for (index, frame) in frames.iter().enumerate() {
        // Before each good frame, frames that must fail.
        let truncated = &frame[..frame.len().min(5)];
        let mut lying = frame.clone();
        // A first partition longer than the packet.
        lying[0] |= 0xe0;
        lying[1] = 0xff;
        lying[2] = 0xff;
        for bad in [&[][..], &[0x00][..], truncated, &lying[..]] {
            assert!(decoder.decode(bad).is_err(), "frame {index}: {bad:?}");
        }
        let got = pack(&decoder.decode(frame).expect("good").expect("shown"));
        assert_eq!(got, want[index], "frame {index}");
    }
}

#[test]
fn malformed_headers_are_rejected() {
    let (frames, _) = encode_synthetic(48, 32, 2, Vp8Options::default());
    let key = &frames[0];
    let mut decoder = Decoder::new();
    // An inter frame before any key frame.
    assert!(matches!(
        decoder.decode(&frames[1]),
        Err(CodecError::Malformed { .. })
    ));
    // A wrong start code.
    let mut bad = key.clone();
    bad[3] ^= 0xff;
    assert!(matches!(decoder.decode(&bad), Err(CodecError::Malformed { .. })));
    // A zero size.
    let mut bad = key.clone();
    bad[6] = 0;
    bad[7] = 0;
    assert!(decoder.decode(&bad).is_err());
    // Eight partitions declared with no room for their sizes.
    let mut bad = key[..12].to_vec();
    bad.extend_from_slice(&[0xff; 4]);
    assert!(decoder.decode(&bad).is_err());
    // A version above 3 is not VP8.
    let mut bad = key.clone();
    bad[0] |= 0x0e;
    assert!(decoder.decode(&bad).is_err());
    // An area over the limit, with the decoder unchanged after it.
    let mut limited = Decoder::with_max_pixels(48 * 32 - 1);
    assert!(matches!(limited.decode(key), Err(CodecError::Limit { .. })));
    let mut enough = Decoder::with_max_pixels(48 * 32);
    assert!(enough.decode(key).expect("at the limit").is_some());
    assert!(decoder.decode(key).expect("still decodes").is_some());
}

#[test]
fn size_changes_at_key_frames() {
    let (a, ra) = encode_synthetic(48, 32, 3, Vp8Options::default());
    let (b, rb) = encode_synthetic(33, 47, 3, Vp8Options::default());
    let mut decoder = Decoder::new();
    let mut got = Vec::new();
    for frame in a.iter().chain(&b) {
        got.extend(pack(&decoder.decode(frame).expect("frame").expect("shown")));
    }
    let mut want = ra;
    want.extend(rb);
    assert_eq!(got, want);
    assert_eq!(decoder.size(), (33, 47));
}

#[test]
fn decoder_contract() {
    let (frames, recon) = encode_synthetic(40, 24, 3, Vp8Options::default());
    assert!(Vp8Decoder::open(&CodecParams::video(CodecId::Vp9, 40, 24)).is_err());
    assert!(matches!(
        Vp8Decoder::open(&CodecParams::video(CodecId::Vp8, 16384, 16)),
        Err(CodecError::Limit { .. })
    ));
    let mut decoder = Vp8Decoder::open(&CodecParams::video(CodecId::Vp8, 40, 24)).expect("open");
    assert_eq!(decoder.codec(), CodecId::Vp8);
    // An empty packet is a dropped frame.
    assert!(decoder.push(&Packet::video(Vec::new(), 0, 1, 30, false)).expect("empty").is_empty());
    let mut got = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        let out = decoder
            .push(&Packet::video(frame.clone(), 7 + i as i64, 2, 60, i == 0))
            .expect("push");
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].pts, out[0].duration, out[0].timescale), (7 + i as i64, 2, 60));
        let f = &out[0].frame;
        assert_eq!((f.width, f.height), (40, 24));
        got.extend_from_slice(&f.y.samples);
        got.extend_from_slice(&f.u.samples);
        got.extend_from_slice(&f.v.samples);
    }
    assert_eq!(got, recon);
    assert!(decoder.drain().expect("drain").is_empty());
    assert!(decoder.drain().expect("drain twice").is_empty());
    decoder.reset();
    assert!(decoder.push(&Packet::video(frames[1].clone(), 0, 1, 30, false)).is_err());
    decoder.cancel_flag().store(true, Ordering::Relaxed);
    assert!(matches!(
        decoder.push(&Packet::video(frames[0].clone(), 0, 1, 30, true)),
        Err(CodecError::Cancelled)
    ));
}

#[test]
fn encoder_contract() {
    let (w, h) = (48, 32);
    assert!(Vp8Encoder::open(&CodecParams::video(CodecId::Vp9, w, h)).is_err());
    assert!(Vp8Encoder::with_options(0, h, Vp8Options::default()).is_err());
    assert!(Vp8Encoder::with_options(16384, h, Vp8Options::default()).is_err());
    let bad_parts = Vp8Options {
        partitions: 3,
        ..Vp8Options::default()
    };
    assert!(Vp8Encoder::with_options(w, h, bad_parts).is_err());
    let mut encoder = Vp8Encoder::open(&CodecParams::video(CodecId::Vp8, w, h)).expect("open");
    assert_eq!(encoder.codec(), CodecId::Vp8);
    // A picture of the wrong size fails and adds nothing.
    assert!(matches!(
        encoder.push(&input(w + 2, h, 0)),
        Err(CodecError::InvalidApi { .. })
    ));
    let first: Vec<_> = (0..4)
        .map(|t| {
            let out = encoder.push(&input(w, h, t)).expect("push");
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].pts, i64::from(t));
            out[0].clone()
        })
        .collect();
    assert!(first[0].keyframe && !first[1].keyframe);
    assert!(encoder.drain().expect("drain").is_empty());
    assert!(encoder.drain().expect("drain twice").is_empty());
    // Reset starts over with a key frame and the same bytes.
    encoder.reset();
    for (t, want) in first.iter().enumerate() {
        let out = encoder.push(&input(w, h, t as u32)).expect("again");
        assert_eq!(out[0].data, want.data, "frame {t}");
    }
    // Bitrate mode needs a clock.
    let mut rated = Vp8Encoder::with_options(
        w,
        h,
        Vp8Options {
            bitrate: 100_000,
            ..Vp8Options::default()
        },
    )
    .expect("open");
    let mut clockless = input(w, h, 0);
    clockless.duration = 0;
    assert!(rated.push(&clockless).is_err());
    assert_eq!(input(w, h, 0).frame.format, PixelFormat::Yuv420p);
    encoder.cancel_flag().store(true, Ordering::Relaxed);
    assert!(matches!(encoder.push(&input(w, h, 9)), Err(CodecError::Cancelled)));
    encoder.reset();
    assert!(
        matches!(encoder.push(&input(w, h, 9)), Err(CodecError::Cancelled)),
        "reset keeps the raised flag"
    );
}

#[test]
fn hidden_frames_produce_no_picture() {
    let (frames, _) = encode_synthetic(32, 32, 2, Vp8Options::default());
    let mut hidden = frames[1].clone();
    hidden[0] &= !0x10;
    let mut decoder = Decoder::new();
    decoder.decode(&frames[0]).expect("key");
    assert!(decoder.decode(&hidden).expect("hidden").is_none());
    let stream = Ivf {
        width: 32,
        height: 32,
        frames: vec![frames[0].clone(), hidden],
    };
    assert_eq!(decode_shown(&stream).len(), 32 * 32 * 3 / 2);
}

/// The fuzz corpus: small streams over every coding path the encoder has.
fn fuzz_corpus() -> Vec<Vec<Vec<u8>>> {
    let mut streams = Vec::new();
    for (w, h, opts) in [
        (48, 32, Vp8Options::default()),
        (
            33,
            17,
            Vp8Options {
                quantizer: 4,
                partitions: 4,
                golden_interval: 2,
                ..Vp8Options::default()
            },
        ),
        (
            16,
            16,
            Vp8Options {
                quantizer: 100,
                keyframe_interval: 2,
                ..Vp8Options::default()
            },
        ),
    ] {
        streams.push(encode_synthetic(w, h, 4, opts).0);
    }
    streams
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
}

fn mutate(rng: &mut Rng, frame: &[u8]) -> Vec<u8> {
    let mut bytes = frame.to_vec();
    for _ in 0..=rng.below(4) {
        if bytes.is_empty() {
            bytes.push(rng.byte());
            continue;
        }
        match rng.below(6) {
            0 => {
                let at = rng.below(bytes.len());
                bytes[at] ^= 1 << rng.below(8);
            }
            1 => {
                let at = rng.below(bytes.len());
                bytes[at] = rng.byte();
            }
            2 => {
                let at = rng.below(bytes.len());
                bytes.truncate(at);
            }
            3 => {
                let at = rng.below(bytes.len() + 1);
                bytes.insert(at, rng.byte());
            }
            4 => {
                // Mostly the header, where the sizes and flags are.
                let at = rng.below(bytes.len().min(24));
                bytes[at] = rng.byte();
            }
            _ => {
                let extra: Vec<u8> = (0..rng.below(32)).map(|_| rng.byte()).collect();
                bytes.extend(extra);
            }
        }
    }
    bytes
}

/// `VP8_FUZZ_SECS` (default 2) and `VP8_FUZZ_SEED`.
#[test]
fn decoder_fuzz_does_not_panic() {
    let secs = std::env::var("VP8_FUZZ_SECS")
        .ok()
        .and_then(|t| t.parse::<u64>().ok())
        .unwrap_or(2);
    let seed = std::env::var("VP8_FUZZ_SEED")
        .ok()
        .and_then(|t| t.parse::<u64>().ok())
        .unwrap_or(0x5650_3800_f22d);
    let corpus = fuzz_corpus();
    let mut rng = Rng(seed.max(1));
    let started = std::time::Instant::now();
    let limit = std::time::Duration::from_secs(secs);
    let (mut inputs, mut pictures, mut errors) = (0u64, 0u64, 0u64);
    while started.elapsed() < limit {
        let stream = &corpus[rng.below(corpus.len())];
        // Decode a good prefix, then mutated frames, then check the
        // decoder still takes a good key frame.
        let mut decoder = Decoder::with_max_pixels(1 << 16);
        let prefix = rng.below(stream.len());
        for frame in &stream[..prefix] {
            decoder.decode(frame).expect("corpus");
        }
        for at in prefix..stream.len() {
            let bytes = if rng.below(3) == 0 {
                stream[at].clone()
            } else if rng.below(16) == 0 {
                (0..rng.below(512)).map(|_| rng.byte()).collect()
            } else {
                mutate(&mut rng, &stream[at])
            };
            match decoder.decode(&bytes) {
                Ok(Some(p)) => {
                    let (w, h) = decoder.size();
                    assert!(
                        p.width == w
                            && p.height == h
                            && p.frame.mbw == (w as usize).div_ceil(16)
                            && p.frame.mbh == (h as usize).div_ceil(16),
                        "seed {seed:#x} input {inputs}"
                    );
                    let _ = crate::decode::picture_frame(&p).expect("croppable");
                    pictures += 1;
                }
                Ok(None) => {}
                Err(_) => errors += 1,
            }
            inputs += 1;
        }
        let again = decoder.decode(&stream[0]).expect("key after noise");
        assert!(again.is_some(), "seed {seed:#x} input {inputs}");
    }
    eprintln!(
        "vp8 fuzz seed {seed:#x} seconds {secs} inputs {inputs} pictures {pictures} errors {errors} findings 0"
    );
}
