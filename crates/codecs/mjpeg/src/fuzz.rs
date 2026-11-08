//! Mutation fuzzing of the decoder and the encoder.
//!
//! The corpus covers every framing the decoder accepts: each sampling,
//! restarts, odd sizes, frames without DHT, tables carried in extradata,
//! field pairs ordered by `fiel` or an AVID comment, and format A headers.
//! A short deterministic run is part of `cargo test`. The campaign is
//! `cargo test --release -p cutlass-mjpeg fuzz_campaign -- --ignored`, with
//! `MJPEG_FUZZ_SECONDS` (default 60) and `MJPEG_FUZZ_SEED`.

use cutlass_decode::CodecParams;
use cutlass_decode::video::VideoDecoder;
use cutlass_jpeg::{Coefficients, Planes, Sampling, WriteSettings};

use crate::tests::{detailed_rgb, field_rows, params, segments, strip, with_segment};
use crate::{MAX_PACKET, MjpegDecoder, MjpegEncoder, RateMode};

struct Seed {
    name: &'static str,
    extra: Vec<u8>,
    width: u32,
    height: u32,
    packet: Vec<u8>,
}

fn jpeg(width: u32, height: u32, sampling: Sampling, quality: u8, restart: u16) -> Vec<u8> {
    let rgb = detailed_rgb(width, height, width ^ height);
    let mut planes = Planes::default();
    planes
        .from_rgb(&rgb, 3, width, height, sampling)
        .expect("planes");
    let mut coefficients = Coefficients::new();
    coefficients
        .transform(&planes.source().expect("source"))
        .expect("transform");
    let mut out = Vec::new();
    coefficients
        .write(
            &WriteSettings {
                quality,
                restart_interval: restart,
                jfif: true,
            },
            &mut out,
        )
        .expect("write");
    out
}

/// SOI, every DQT and DHT of `full`, EOI: a table-specification stream.
fn table_stream(full: &[u8]) -> Vec<u8> {
    let mut out = vec![0xff, 0xd8];
    for (marker, start, end) in segments(full) {
        if marker == 0xdb || marker == 0xc4 {
            out.extend_from_slice(&full[start..end]);
        }
    }
    out.extend_from_slice(&[0xff, 0xd9]);
    out
}

fn field_pair(width: u32, height: u32, sampling: Sampling) -> (Vec<u8>, Vec<u8>) {
    let rgb = detailed_rgb(width, height, 5);
    let (top, rows) = field_rows(&rgb, width, height, 0);
    let (mut bottom, bottom_rows) = field_rows(&rgb, width, height, 1);
    if bottom_rows < rows {
        let row = (width * 3) as usize;
        let last = bottom[bottom.len() - row..].to_vec();
        bottom.extend_from_slice(&last);
    }
    let code = |rgb: &[u8]| {
        let mut planes = Planes::default();
        planes
            .from_rgb(rgb, 3, width, rows, sampling)
            .expect("planes");
        let mut coefficients = Coefficients::new();
        coefficients
            .transform(&planes.source().expect("source"))
            .expect("transform");
        let mut out = Vec::new();
        coefficients
            .write(
                &WriteSettings {
                    quality: 80,
                    restart_interval: 1,
                    jfif: false,
                },
                &mut out,
            )
            .expect("write");
        out
    };
    (code(&top), code(&bottom))
}

fn corpus() -> Vec<Seed> {
    let plain = jpeg(32, 24, Sampling::Yuv420, 70, 2);
    let no_dht = strip(&jpeg(40, 24, Sampling::Yuv422, 60, 0), &[0xc4]);
    let full = jpeg(24, 16, Sampling::Yuv420, 85, 0);
    let abbreviated = strip(&full, &[0xc4, 0xdb]);
    let (top, bottom) = field_pair(40, 47, Sampling::Yuv420);
    let mut fields = top.clone();
    fields.extend_from_slice(&bottom);
    let mut header = vec![0, 0, 0, 0];
    header.extend_from_slice(b"mjpg");
    header.resize(40, 0);
    let mut format_a = with_segment(&top, 0xe1, &header);
    format_a.extend_from_slice(&[0, 0, 0, 0]);
    format_a.extend_from_slice(&with_segment(&bottom, 0xe1, &header));
    let mut avid = b"AVID".to_vec();
    avid.resize(16, 0);
    avid[12] = 1;
    let mut avid_pair = with_segment(&bottom, 0xfe, &avid);
    avid_pair.extend_from_slice(&top);
    vec![
        Seed {
            name: "4:2:0 with restarts",
            extra: Vec::new(),
            width: 32,
            height: 24,
            packet: plain,
        },
        Seed {
            name: "4:2:2 without DHT",
            extra: b"jpeg".to_vec(),
            width: 40,
            height: 24,
            packet: no_dht,
        },
        Seed {
            name: "4:4:4 odd size",
            extra: Vec::new(),
            width: 17,
            height: 13,
            packet: jpeg(17, 13, Sampling::Yuv444, 90, 1),
        },
        Seed {
            name: "gray",
            extra: Vec::new(),
            width: 24,
            height: 16,
            packet: jpeg(24, 16, Sampling::Gray, 75, 0),
        },
        Seed {
            name: "tables in extradata",
            extra: table_stream(&full),
            width: 24,
            height: 16,
            packet: abbreviated,
        },
        Seed {
            name: "field pair",
            extra: b"jpeg\x02\x01".to_vec(),
            width: 40,
            height: 47,
            packet: fields,
        },
        Seed {
            name: "format A field pair",
            extra: b"mjpa".to_vec(),
            width: 40,
            height: 47,
            packet: format_a,
        },
        Seed {
            name: "AVID bottom first",
            extra: Vec::new(),
            width: 40,
            height: 47,
            packet: avid_pair,
        },
    ]
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
        (self.next() >> 24) as u8
    }
}

fn mutate(rng: &mut Rng, seed: &[u8], other: &[u8]) -> Vec<u8> {
    let mut bad = seed.to_vec();
    match rng.below(8) {
        0 => {
            for _ in 0..1 + rng.below(16) {
                let at = rng.below(bad.len());
                bad[at] ^= 1 << rng.below(8);
            }
        }
        1 => bad.truncate(rng.below(bad.len())),
        2 => {
            let at = rng.below(bad.len());
            let end = (at + rng.below(64)).min(bad.len());
            for byte in &mut bad[at..end] {
                *byte = rng.byte();
            }
        }
        3 => {
            // Header fields: sizes, counts, selectors, sampling factors.
            let header = segments(seed).last().map_or(bad.len(), |last| last.2 + 16);
            let at = rng.below(header.min(bad.len()));
            bad[at] = match rng.below(4) {
                0 => 0,
                1 => 0xff,
                2 => bad[at].wrapping_add(1),
                _ => rng.byte(),
            };
        }
        4 => {
            // A segment length that over- or under-runs the segment.
            let found = segments(seed);
            if let Some(&(_, start, _)) = found.get(rng.below(found.len())) {
                let length = (rng.next() & 0xffff) as u16;
                bad[start + 2..start + 4].copy_from_slice(&length.to_be_bytes());
            }
        }
        5 => {
            let cut = rng.below(bad.len());
            bad.truncate(cut);
            let from = rng.below(other.len());
            bad.extend_from_slice(&other[from..]);
        }
        6 => {
            let at = rng.below(bad.len() + 1);
            let insert: Vec<u8> = match rng.below(3) {
                0 => vec![0xff, 0xd8],
                1 => vec![0xff, 0xc0 + (rng.byte() & 0x3f)],
                _ => (0..1 + rng.below(8)).map(|_| rng.byte()).collect(),
            };
            bad.splice(at..at, insert);
        }
        _ => {
            let at = rng.below(bad.len());
            let end = (at + 1 + rng.below(256)).min(bad.len());
            let copy = bad[at..end].to_vec();
            let to = rng.below(bad.len() + 1);
            bad.splice(to..to, copy);
        }
    }
    bad
}

/// What a decoder makes of `data`, through both outputs, or why not.
fn outputs(decoder: &mut MjpegDecoder, data: &[u8]) -> Result<Vec<u8>, String> {
    let picture = decoder.decode(data).map_err(|error| error.to_string())?;
    let (width, height) = (picture.width, picture.height);
    assert!(width > 0 && height > 0, "an accepted picture is empty");
    assert!(u64::from(width) * u64::from(height) <= 16_777_216);
    let mut rgba = Vec::new();
    picture
        .to_rgba(&mut rgba)
        .map_err(|error| error.to_string())?;
    assert_eq!(rgba.len(), (width * height * 4) as usize);
    let frame = picture.into_frame().map_err(|error| error.to_string())?;
    assert_eq!((frame.width, frame.height), (width, height));
    Ok(rgba)
}

struct Tally {
    cases: u64,
    refused: u64,
    accepted: u64,
    encoded: u64,
}

/// One case. A refused packet leaves the decoder as it was: the seed's own
/// packet decodes to what it did before.
fn case(rng: &mut Rng, seeds: &[Seed], wants: &[Vec<u8>], tally: &mut Tally) {
    tally.cases += 1;
    if rng.below(16) == 0 {
        encode_case(rng);
        tally.encoded += 1;
        return;
    }
    let pick = rng.below(seeds.len());
    let seed = &seeds[pick];
    let other = &seeds[rng.below(seeds.len())].packet;
    let bad = mutate(rng, &seed.packet, other);
    let mut decoder =
        MjpegDecoder::open(&params(seed.width, seed.height, &seed.extra)).expect("open");
    // Half the cases run on a decoder that has already decoded the seed.
    if rng.below(2) == 0 {
        assert_eq!(
            outputs(&mut decoder, &seed.packet).as_ref(),
            Ok(&wants[pick])
        );
    }
    match outputs(&mut decoder, &bad) {
        Ok(_) => {
            tally.accepted += 1;
            VideoDecoder::reset(&mut decoder);
        }
        Err(_) => tally.refused += 1,
    }
    assert_eq!(
        outputs(&mut decoder, &seed.packet).as_ref(),
        Ok(&wants[pick]),
        "{} after {} bytes",
        seed.name,
        bad.len()
    );
}

/// Random sizes, sampling, quality, restarts, and pixels: the encoder
/// codes them, and the decoder reads back that size.
fn encode_case(rng: &mut Rng) {
    let width = rng.below(70) as u32 + 1;
    let height = rng.below(70) as u32 + 1;
    let sampling = [
        Sampling::Gray,
        Sampling::Yuv444,
        Sampling::Yuv422,
        Sampling::Yuv420,
    ][rng.below(4)];
    let params = CodecParams::with_format(
        cutlass_decode::CodecId::Mjpeg,
        width,
        height,
        cutlass_decode::PixelFormat::Rgba8,
    );
    let quality = 1 + rng.below(100) as u8;
    let mut encoder = MjpegEncoder::open(&params)
        .expect("open")
        .with_sampling(sampling)
        .with_restart_interval(rng.below(4) as u16)
        .with_mode(RateMode::Quality(quality))
        .expect("quality");
    let rgba: Vec<u8> = (0..width * height * 4).map(|_| rng.byte()).collect();
    let data = encoder
        .encode_rgba(width, height, &rgba, 1, 30)
        .expect("encode");
    let mut decoder = MjpegDecoder::open(&params).expect("decoder");
    let picture = decoder.decode(&data).expect("own packet");
    assert_eq!((picture.width, picture.height), (width, height));
}

fn run(seed: u64, until: impl Fn(&Tally) -> bool) -> Tally {
    let seeds = corpus();
    let wants: Vec<Vec<u8>> = seeds
        .iter()
        .map(|seed| {
            let mut decoder =
                MjpegDecoder::open(&params(seed.width, seed.height, &seed.extra)).expect("open");
            outputs(&mut decoder, &seed.packet)
                .unwrap_or_else(|error| panic!("{}: {error}", seed.name))
        })
        .collect();
    let mut rng = Rng(seed | 1);
    let mut tally = Tally {
        cases: 0,
        refused: 0,
        accepted: 0,
        encoded: 0,
    };
    while !until(&tally) {
        case(&mut rng, &seeds, &wants, &mut tally);
    }
    tally
}

#[test]
fn mutated_packets_fail_closed_and_leave_the_decoder_as_it_was() {
    let tally = run(0x0005_eed0_fa11, |tally| tally.cases >= 3_000);
    assert!(
        tally.refused > 1_000 && tally.accepted > 50,
        "{}/{}",
        tally.refused,
        tally.accepted
    );
    assert!(tally.encoded > 100);
}

#[test]
fn an_oversized_packet_is_refused_before_it_is_read() {
    let mut decoder = MjpegDecoder::open(&params(16, 16, &[])).expect("open");
    let mut big = vec![0u8; MAX_PACKET + 1];
    big[..2].copy_from_slice(&[0xff, 0xd8]);
    assert!(decoder.decode(&big).is_err());
}

#[test]
#[ignore = "time-budgeted campaign; run by hand"]
fn fuzz_campaign() {
    let seconds: u64 = std::env::var("MJPEG_FUZZ_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(60);
    let seed: u64 = std::env::var("MJPEG_FUZZ_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0x9e37_79b9_7f4a_7c15);
    let start = std::time::Instant::now();
    let tally = run(seed, |_| start.elapsed().as_secs() >= seconds);
    eprintln!(
        "fuzz campaign: seed {seed}, {} cases in {seconds} s, {} refused, {} accepted, {} encoder cases, no panics",
        tally.cases, tally.refused, tally.accepted, tally.encoded
    );
}
