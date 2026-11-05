//! In-tree mutation fuzzing, run by `cargo test` (`cargo-fuzz` is not part
//! of this workspace). Seeds are files this crate writes: lossless and
//! lossy stills with and without alpha, an ICC profile, and animations.
//! Each round mutates one seed (bit flips, bytes, cuts, splices, chunk
//! sizes) and decodes it whole and packet by packet. Every result must be a
//! picture or an error, never a panic.
//!
//! A second campaign drives the encoder with random sizes, pictures and
//! settings and decodes what it wrote.
//!
//! `WEBP_FUZZ_ROUNDS` and `WEBP_FUZZ_SEED` set the campaign; a failing
//! input is written to the scratch folder and named in the panic.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cutlass_decode::{CodecId, CodecParams, Decoder, Packet};
use cutlass_encode::{BitDepth, ColorType, Encoder, PictureInput, StillImage};

use super::animation::split;
use super::encoder::encode_animation;
use super::oracle::{scratch, synthetic_rgba};
use crate::{WebpDecoder, WebpEncoder, WebpOptions};

/// xorshift64*: small, seeded, the same on every machine.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn rounds(default: usize) -> usize {
    std::env::var("WEBP_FUZZ_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn seed() -> u64 {
    std::env::var("WEBP_FUZZ_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x5eed_cafe_u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        | 1
}

fn encode(options: WebpOptions, width: u32, height: u32, rgba: &[u8], icc: &[u8]) -> Vec<u8> {
    let mut params = CodecParams::video(CodecId::Webp, 0, 0);
    params.extra = options.to_extra();
    let mut encoder = WebpEncoder::open(&params).expect("encoder");
    let samples = rgba.iter().map(|&s| u16::from(s)).collect();
    let mut image =
        StillImage::from_samples(width, height, ColorType::Rgba, BitDepth::Eight, samples);
    image.icc_profile = icc.to_vec();
    encoder
        .push_picture(&PictureInput { image })
        .expect("encode")[0]
        .data
        .clone()
}

fn seeds() -> Vec<Vec<u8>> {
    let rgba = synthetic_rgba(21, 13);
    let opaque: Vec<u8> = rgba
        .chunks(4)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect();
    let lossy = WebpOptions {
        lossless: false,
        quality: 70,
        effort: 4,
        exact: false,
    };
    let mut out = vec![
        encode(WebpOptions::lossless(), 21, 13, &rgba, &[]),
        encode(WebpOptions::lossless(), 21, 13, &opaque, &[]),
        encode(lossy, 21, 13, &rgba, &[]),
        encode(lossy, 21, 13, &opaque, &[1, 2, 3, 4, 5, 6, 7]),
    ];
    let frames: Vec<Vec<u8>> = (0..3)
        .map(|index| {
            let mut frame = rgba.clone();
            for pixel in frame.chunks_mut(4).skip(index * 7).take(30) {
                pixel.copy_from_slice(&[250, 20, 20, 255]);
            }
            frame
        })
        .collect();
    for options in [WebpOptions::lossless(), lossy] {
        let packets = encode_animation(options, &frames, 21, 13);
        out.push(packets.iter().flat_map(|p| p.data.clone()).collect());
    }
    out
}

fn mutate(rng: &mut Rng, seed: &[u8]) -> Vec<u8> {
    let mut data = seed.to_vec();
    for _ in 0..1 + rng.below(4) {
        if data.is_empty() {
            data.push(0);
        }
        let at = rng.below(data.len());
        match rng.below(7) {
            0 => data[at] ^= 1 << rng.below(8),
            1 => data[at] = [0, 0xff, 0x7f, 0x80, 1][rng.below(5)],
            2 => data.truncate(at),
            3 => {
                let len = rng.below(16).min(data.len() - at);
                let piece = data[at..at + len].to_vec();
                let to = rng.below(data.len());
                data.splice(to..to, piece);
            }
            4 => {
                let end = (at + 4).min(data.len());
                for byte in &mut data[at..end] {
                    *byte = rng.next() as u8;
                }
            }
            5 => {
                // A chunk size: the four bytes after a known fourcc.
                let fourccs: [&[u8; 4]; 6] = [b"VP8L", b"VP8 ", b"ALPH", b"ANMF", b"VP8X", b"ICCP"];
                let fourcc = fourccs[rng.below(fourccs.len())];
                if let Some(pos) = data.windows(4).position(|w| w == fourcc)
                    && pos + 8 <= data.len()
                {
                    let size: u32 =
                        [0, 1, 7, 0xffff_ffff, 0x7fff_ffff, rng.next() as u32][rng.below(6)];
                    data[pos + 4..pos + 8].copy_from_slice(&size.to_le_bytes());
                }
            }
            _ => {
                let end = rng.below(data.len() + 1);
                data.drain(at.min(end)..end.max(at));
            }
        }
    }
    data
}

fn decode_all(data: &[u8]) -> usize {
    let open = || WebpDecoder::open(&CodecParams::video(CodecId::Webp, 0, 0)).expect("decoder");
    let mut pictures = 0;
    if let Ok(out) = open().decode_picture(&Packet::video(data.to_vec(), 0, 0, 1000, true)) {
        pictures += out.len();
    }
    let mut decoder = open();
    for (index, part) in split(data).into_iter().enumerate() {
        let packet = Packet::video(part, index as i64 * 100, 100, 1000, index == 0);
        if let Ok(out) = decoder.decode_picture(&packet) {
            pictures += out.len();
        }
    }
    pictures
}

#[test]
fn mutated_files_decode_or_fail_and_never_panic() {
    let seeds = seeds();
    let mut rng = Rng(seed());
    let rounds = rounds(4000);
    let (mut decoded, mut slowest) = (0usize, Duration::ZERO);
    for round in 0..rounds {
        let seed = &seeds[rng.below(seeds.len())];
        let data = mutate(&mut rng, seed);
        let start = Instant::now();
        let result = catch_unwind(AssertUnwindSafe(|| decode_all(&data)));
        slowest = slowest.max(start.elapsed());
        match result {
            Ok(pictures) => decoded += usize::from(pictures > 0),
            Err(_) => {
                let path = scratch("fuzz-crash").join(format!("round-{round}.webp"));
                std::fs::write(&path, &data).expect("write");
                panic!("round {round} panicked on {}", path.display());
            }
        }
    }
    println!("{rounds} rounds, {decoded} decoded something, slowest {slowest:?}");
    // The cuts and flips leave some files whole enough to decode.
    assert!(decoded > 0);
    // A few hundred bytes may declare a canvas near the 2^26-pixel limit;
    // that costs tens of milliseconds in release and seconds unoptimised.
    let budget = Duration::from_secs(if cfg!(debug_assertions) { 30 } else { 1 });
    assert!(slowest < budget, "{slowest:?}");
}

#[test]
fn random_settings_and_pictures_encode_and_decode_back() {
    let mut rng = Rng(seed() ^ 0x00e1_c0de);
    for round in 0..rounds(150) / 2 {
        let (width, height) = (1 + rng.below(40) as u32, 1 + rng.below(40) as u32);
        let kind = rng.below(4);
        let rgba: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                let v = rng.next();
                let (x, y) = (i % width, i / width);
                match kind {
                    0 => [v as u8, (v >> 8) as u8, (v >> 16) as u8, (v >> 24) as u8],
                    1 => [90, 140, 200, 255],
                    2 => [(x * 6) as u8, (y * 6) as u8, ((x + y) * 3) as u8, 255],
                    _ => [
                        (x * 9) as u8,
                        80,
                        (y * 9) as u8,
                        if (x + y) % 3 == 0 { 0 } else { 200 },
                    ],
                }
            })
            .collect();
        let lossless = rng.below(2) == 0;
        let options = WebpOptions {
            lossless,
            quality: rng.below(101) as u8,
            effort: rng.below(if lossless { 10 } else { 7 }) as u8,
            exact: rng.below(2) == 0,
        };
        let file = encode(options, width, height, &rgba, &[]);
        let picture = super::oracle::try_decode_still(&file)
            .unwrap_or_else(|error| panic!("round {round} {options:?}: {error}"));
        assert_eq!((picture.width, picture.height), (width, height));
        for (index, (a, b)) in picture.rgba.chunks(4).zip(rgba.chunks(4)).enumerate() {
            assert_eq!(a[3], b[3], "round {round} {options:?}: alpha at {index}");
            let visible = b[3] != 0 || options.exact;
            if lossless && visible {
                assert_eq!(a, b, "round {round} {options:?}: pixel {index}");
            }
        }
    }
}
