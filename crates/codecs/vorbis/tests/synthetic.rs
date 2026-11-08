//! Streams this encoder never emits: floor 0, residue 0 and 1, a lookup
//! type 2 book, and more than two modes. libvorbis and FFmpeg's native
//! decoder (lookup type 2: libvorbis only) must match this decoder at 90 dB.
//!
//! Without GStreamer's Vorbis plugin or ffmpeg the test skips.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr
)]

mod common;

use std::fs::File;
use std::path::Path;
use std::process::{Command, Stdio};

use common::{
    decode_file_with_layout, ffmpeg_decode, has_ffmpeg, has_gst_vorbis, libvorbis_decode, scratch,
    snr_db,
};
use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Mux, NewStream, TrackDuration};
use cutlass_ogg::OggMux;
use cutlass_vorbis::{Comments, Headers, Identification, VorbisDecoder};

const MIN_SNR_DB: f64 = 90.0;
const RATE: u32 = 44_100;
const SHORT: usize = 256;
const LONG: usize = 2048;

#[test]
fn unusual_setups_decode_like_libvorbis_and_ffmpeg() {
    if !has_gst_vorbis() || !has_ffmpeg() {
        eprintln!("skipped: needs GStreamer vorbisdec and ffmpeg");
        return;
    }
    let dir = scratch("synthetic");
    let cases = [
        Case {
            name: "residue 0",
            floor: Floor::One,
            residue: 0,
            lookup: Lookup::Type1,
            modes: &[false],
            sequence: &[0],
            ffmpeg: true,
        },
        Case {
            name: "residue 1",
            floor: Floor::One,
            residue: 1,
            lookup: Lookup::Type1,
            modes: &[false],
            sequence: &[0],
            ffmpeg: true,
        },
        Case {
            name: "floor 0",
            floor: Floor::Zero,
            residue: 1,
            lookup: Lookup::Type1,
            modes: &[false],
            sequence: &[0],
            ffmpeg: true,
        },
        Case {
            name: "lookup type 2",
            floor: Floor::One,
            residue: 1,
            lookup: Lookup::Type2,
            modes: &[false],
            sequence: &[0],
            ffmpeg: false,
        },
        Case {
            name: "many modes",
            floor: Floor::One,
            residue: 1,
            lookup: Lookup::Type1,
            modes: &[false, false, true, false],
            sequence: &[0, 2, 1, 2, 3],
            ffmpeg: true,
        },
    ];
    for case in cases {
        let built = build(&case);
        let file = dir.join(format!("{}.ogg", case.name.replace(' ', "_")));
        mux(&file, &built);
        check(&dir, &file, &case, built.samples);
    }
}

#[derive(Clone, Copy)]
enum Floor {
    Zero,
    One,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Lookup {
    Type1,
    Type2,
}

struct Case {
    name: &'static str,
    floor: Floor,
    residue: u32,
    lookup: Lookup,
    /// `true` when that mode uses the long block.
    modes: &'static [bool],
    /// Mode number of each audio packet. One mode is repeated so the
    /// overlap is several packets long.
    sequence: &'static [u32],
    /// FFmpeg's native decoder accepts the setup.
    ffmpeg: bool,
}

struct Built {
    config: Vec<u8>,
    audio: Vec<Vec<u8>>,
    durations: Vec<i64>,
    samples: usize,
}

fn build(case: &Case) -> Built {
    let id = Identification {
        channels: 1,
        sample_rate: RATE,
        bitrate_maximum: 0,
        bitrate_nominal: 0,
        bitrate_minimum: 0,
        blocksizes: [SHORT, LONG],
    };
    let comment = Comments {
        vendor: "Cutlass".to_string(),
        fields: Vec::new(),
    };
    let setup = setup(case);
    let headers = Headers {
        identification: id.write(),
        comment: comment.write(),
        setup,
    };
    let config = headers.join();
    let (audio, durations) = audio(case);
    let mut decoder = VorbisDecoder::from_config(&config).expect(case.name);
    let mut pcm = Vec::new();
    for packet in &audio {
        pcm.extend(decoder.decode(packet).expect(case.name));
    }
    let samples: i64 = durations.iter().sum();
    assert_eq!(pcm.len() as i64, samples, "{}", case.name);
    assert!(
        pcm.iter().any(|sample| sample.abs() > 1e-8),
        "{} decoded to silence",
        case.name
    );
    assert!(pcm.iter().all(|sample| sample.is_finite()), "{}", case.name);
    Built {
        config,
        audio,
        durations,
        samples: samples as usize,
    }
}

fn setup(case: &Case) -> Vec<u8> {
    let mut body = Pack::default();
    // Two books, or three when floor 0 needs its own.
    let books = if matches!(case.floor, Floor::Zero) {
        3
    } else {
        2
    };
    body.put(books - 1, 8);
    codebook(&mut body, 1, &[1], None);
    match case.lookup {
        Lookup::Type1 => codebook(&mut body, 2, &[2, 2, 2, 2], Some(Look::Type1)),
        Lookup::Type2 => codebook(&mut body, 2, &[1], Some(Look::Type2)),
    }
    if matches!(case.floor, Floor::Zero) {
        codebook(&mut body, 2, &[1], Some(Look::Floor));
    }
    // One time-domain transform, type 0.
    body.put(0, 6);
    body.put(0, 16);
    body.put(0, 6);
    match case.floor {
        Floor::Zero => {
            body.put(0, 16);
            body.put(2, 8);
            body.put(RATE, 16);
            body.put(64, 16);
            body.put(8, 6);
            body.put(1, 8);
            body.put(0, 4);
            body.put(2, 8);
        }
        Floor::One => {
            body.put(1, 16);
            body.put(0, 5);
            body.put(0, 2);
            body.put(8, 4);
        }
    }
    body.put(0, 6);
    body.put(case.residue, 16);
    body.put(0, 24);
    body.put(128, 24);
    body.put(15, 24);
    body.put(0, 6);
    body.put(0, 8);
    body.put(1, 3);
    body.put(0, 1);
    body.put(1, 8);
    body.put(0, 6);
    body.put(0, 16);
    body.put(0, 1);
    body.put(0, 1);
    body.put(0, 2);
    body.put(0, 8);
    body.put(0, 8);
    body.put(0, 8);
    body.put(case.modes.len() as u32 - 1, 6);
    for &long in case.modes {
        body.put(u32::from(long), 1);
        body.put(0, 16);
        body.put(0, 16);
        body.put(0, 8);
    }
    body.put(1, 1);
    let mut packet = vec![5];
    packet.extend_from_slice(b"vorbis");
    packet.extend_from_slice(&body.bytes);
    packet
}

fn audio(case: &Case) -> (Vec<Vec<u8>>, Vec<i64>) {
    let width = ilog(case.modes.len() as u32 - 1);
    let mut packets = Vec::new();
    let mut durations = Vec::new();
    let mut previous: Option<bool> = None;
    // More than one audio page. FFmpeg drops the end trim when the first
    // audio page is also the last, and a page holds 255 packets.
    let rounds = 256 / case.sequence.len() + 1;
    for _ in 0..rounds {
        for &mode in case.sequence {
            let long = case.modes[mode as usize];
            let mut body = Pack::default();
            body.put(0, 1);
            body.put(mode, width);
            if long {
                let previous_long = previous.unwrap_or(false);
                let next_long = false;
                body.put(u32::from(previous_long), 1);
                body.put(u32::from(next_long), 1);
            }
            match case.floor {
                Floor::One => {
                    body.put(1, 1);
                    body.put(200, 8);
                    body.put(200, 8);
                }
                Floor::Zero => {
                    body.put(32, 8);
                    body.put(0, 1);
                    body.put(0, 1);
                }
            }
            residue_packet(&mut body, case.lookup);
            let current = if long { LONG as i64 } else { SHORT as i64 };
            let duration = previous.map_or(0, |was_long| {
                let previous = if was_long { LONG as i64 } else { SHORT as i64 };
                previous / 4 + current / 4
            });
            packets.push(body.bytes);
            durations.push(duration);
            previous = Some(long);
        }
    }
    (packets, durations)
}

/// Eight partitions of 16 bins. The classbook is one bit. The residue
/// book is dimension 2, so each partition is eight vectors.
fn residue_packet(body: &mut Pack, lookup: Lookup) {
    for _ in 0..8 {
        body.put(0, 1);
        for _ in 0..8 {
            match lookup {
                // Entry 1 of the length-2 book: code 0b01, high bit first.
                // Its lookup-type-1 vector is [3, 1], so a format-0
                // interleave is not a format-1 copy.
                Lookup::Type1 => {
                    body.put(0, 1);
                    body.put(1, 1);
                }
                Lookup::Type2 => body.put(0, 1),
            }
        }
    }
}

enum Look {
    Type1,
    Type2,
    Floor,
}

fn codebook(body: &mut Pack, dimensions: u32, lengths: &[u32], look: Option<Look>) {
    body.put(0x0056_4342, 24);
    body.put(dimensions, 16);
    body.put(lengths.len() as u32, 24);
    body.put(0, 1);
    body.put(0, 1);
    for &len in lengths {
        body.put(len - 1, 5);
    }
    match look {
        None => body.put(0, 4),
        Some(kind) => {
            let (lookup, values) = match kind {
                Look::Type1 => (1, vec![1, 3]),
                Look::Type2 => (2, vec![1, 3]),
                Look::Floor => (1, vec![1]),
            };
            body.put(lookup, 4);
            body.put(0, 32);
            body.put(float32_pack(1.0), 32);
            body.put(1, 4);
            body.put(0, 1);
            for value in values {
                body.put(value, 2);
            }
        }
    }
}

fn mux(path: &Path, built: &Built) {
    let mut mux = OggMux::new(File::create(path).expect("create"));
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Vorbis),
                RATE,
                1,
                built.config.clone(),
            )
            .expect("params"),
            timescale: Timescale::new(RATE),
            language: None,
        })
        .expect("stream");
    let mut pts = 0_i64;
    for (packet, &duration) in built.audio.iter().zip(&built.durations) {
        let stamp = Timestamp::try_new(Some(pts), Some(pts), Some(duration)).expect("time");
        let written = Packet::try_new(
            track.get(),
            packet.clone(),
            stamp,
            Timescale::new(RATE),
            true,
        )
        .expect("packet");
        mux.write_packet(&written).expect("write");
        mux.interleave().expect("interleave");
        pts += duration;
    }
    let finished = mux
        .finish(&[TrackDuration {
            track,
            timescale: Timescale::new(RATE),
            ticks: pts,
        }])
        .expect("finish");
    assert!(finished.is_complete());
}

fn check(dir: &Path, file: &Path, case: &Case, samples: usize) {
    let bytes = std::fs::read(file).unwrap();
    let (_, _, ours) = decode_file_with_layout(&bytes);
    assert_eq!(ours.len(), samples, "{}", case.name);
    let libvorbis = libvorbis_decode(file, dir);
    let libvorbis_frames = libvorbis.len();
    assert!(
        libvorbis_frames == samples || libvorbis_frames == samples + 1,
        "{}: libvorbis {libvorbis_frames} samples, want {samples}",
        case.name
    );
    let lib_snr = snr_db(&ours, &libvorbis, 1);
    assert!(
        lib_snr >= MIN_SNR_DB,
        "{}: {lib_snr} dB against libvorbis",
        case.name
    );
    if case.ffmpeg {
        let ffmpeg = ffmpeg_decode(file, dir);
        assert_eq!(ffmpeg.len(), samples, "{}", case.name);
        let ffmpeg_snr = snr_db(&ours, &ffmpeg, 1);
        assert!(
            ffmpeg_snr >= MIN_SNR_DB,
            "{}: {ffmpeg_snr} dB against ffmpeg",
            case.name
        );
        eprintln!(
            "{}: {samples} samples, ffmpeg {ffmpeg_snr:.1} dB, libvorbis {lib_snr:.1} dB",
            case.name
        );
    } else {
        let refused = Command::new("ffmpeg")
            .stderr(Stdio::null())
            .args([
                "-v",
                "error",
                "-y",
                "-cpuflags",
                "0",
                "-c:a",
                "vorbis",
                "-i",
            ])
            .arg(file)
            .args(["-f", "f32le"])
            .arg(dir.join("refused.f32"))
            .status()
            .expect("ffmpeg runs");
        assert!(
            !refused.success(),
            "{}: ffmpeg accepted a lookup type 2 book",
            case.name
        );
        eprintln!(
            "{}: {samples} samples, libvorbis {lib_snr:.1} dB, ffmpeg refused the book",
            case.name
        );
    }
}

/// `float32_pack` (Vorbis I section 9.2.2).
fn float32_pack(value: f32) -> u32 {
    let mut magnitude = f64::from(value.abs());
    let mut exponent = 788_i32;
    while magnitude >= f64::from(1_u32 << 21) {
        magnitude /= 2.0;
        exponent += 1;
    }
    while magnitude < f64::from(1_u32 << 20) {
        magnitude *= 2.0;
        exponent -= 1;
    }
    let mantissa = magnitude.round() as u32;
    ((exponent as u32) << 21) | (mantissa & 0x001f_ffff)
}

fn ilog(value: u32) -> u32 {
    u32::BITS - value.leading_zeros()
}

#[derive(Default)]
struct Pack {
    bytes: Vec<u8>,
    n: u32,
}

impl Pack {
    fn put(&mut self, value: u32, width: u32) {
        for bit in 0..width {
            if self.n.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let last = self.bytes.len() - 1;
            if (value >> bit) & 1 == 1 {
                self.bytes[last] |= 1 << (self.n % 8);
            }
            self.n += 1;
        }
    }
}
