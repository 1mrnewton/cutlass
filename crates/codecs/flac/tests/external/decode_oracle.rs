//! Streams `flac` (libFLAC 1.5) produced, decoded here and compared with the
//! source signal, `flac -d`, and ffmpeg.

use std::path::Path;

use cutlass_flac::{ChannelAssignment, FlacDecoder, FrameHeader, Md5Check, StreamInfo, crc};

use crate::support::{
    Kind, ffmpeg_samples, flac_samples, have, path_str, run, scratch, signal, split, succeeds,
    write_wav,
};

/// One libFLAC encode to decode and compare.
pub struct Case {
    pub name: &'static str,
    pub kind: Kind,
    pub channels: usize,
    pub bits: u32,
    pub rate: u32,
    pub frames: usize,
    pub args: &'static [&'static str],
}

pub const CASES: &[Case] = &[
    Case {
        name: "s16-stereo-l0",
        kind: Kind::Music,
        channels: 2,
        bits: 16,
        rate: 44_100,
        frames: 30_000,
        args: &["-0"],
    },
    Case {
        name: "s16-stereo-l5",
        kind: Kind::Music,
        channels: 2,
        bits: 16,
        rate: 44_100,
        frames: 30_000,
        args: &["-5"],
    },
    Case {
        name: "s16-stereo-l8-exhaustive",
        kind: Kind::Music,
        channels: 2,
        bits: 16,
        rate: 48_000,
        frames: 20_000,
        args: &["-8", "-e", "-p"],
    },
    Case {
        name: "s16-correlated-mid-side",
        kind: Kind::Correlated,
        channels: 2,
        bits: 16,
        rate: 44_100,
        frames: 20_000,
        args: &["-8", "-m"],
    },
    Case {
        name: "s16-correlated-adaptive",
        kind: Kind::Correlated,
        channels: 2,
        bits: 16,
        rate: 44_100,
        frames: 20_000,
        args: &["-5", "-M"],
    },
    Case {
        name: "s16-lpc32-lax",
        kind: Kind::Music,
        channels: 2,
        bits: 16,
        rate: 44_100,
        frames: 20_000,
        args: &["--lax", "-l", "32", "-b", "8192", "-r", "15"],
    },
    Case {
        name: "s16-odd-block-1000",
        kind: Kind::Music,
        channels: 1,
        bits: 16,
        rate: 44_100,
        frames: 10_123,
        args: &["-b", "1000"],
    },
    Case {
        name: "s16-block-16",
        kind: Kind::Music,
        channels: 2,
        bits: 16,
        rate: 44_100,
        frames: 1_000,
        args: &["--lax", "-b", "16"],
    },
    Case {
        name: "s16-block-65535",
        kind: Kind::Music,
        channels: 1,
        bits: 16,
        rate: 44_100,
        frames: 140_000,
        args: &["--lax", "-b", "65535"],
    },
    Case {
        name: "s16-noise-verbatim",
        kind: Kind::Noise,
        channels: 2,
        bits: 16,
        rate: 44_100,
        frames: 9_000,
        args: &["-5"],
    },
    Case {
        name: "s16-silent-channel",
        kind: Kind::SilentFirst,
        channels: 2,
        bits: 16,
        rate: 44_100,
        frames: 9_000,
        args: &["-5", "--no-mid-side"],
    },
    Case {
        name: "s16-extremes",
        kind: Kind::Extremes,
        channels: 2,
        bits: 16,
        rate: 44_100,
        frames: 9_000,
        args: &["-8"],
    },
    Case {
        name: "s24-6ch-96k",
        kind: Kind::Music,
        channels: 6,
        bits: 24,
        rate: 96_000,
        frames: 20_000,
        args: &["-5"],
    },
    Case {
        name: "s24-stereo-l8",
        kind: Kind::Music,
        channels: 2,
        bits: 24,
        rate: 48_000,
        frames: 20_000,
        args: &["-8"],
    },
    Case {
        name: "s24-wasted-bits",
        kind: Kind::Wasted,
        channels: 2,
        bits: 24,
        rate: 48_000,
        frames: 12_000,
        args: &["-5"],
    },
    Case {
        name: "s24-extremes",
        kind: Kind::Extremes,
        channels: 2,
        bits: 24,
        rate: 48_000,
        frames: 9_000,
        args: &["-8"],
    },
    Case {
        name: "s8-mono-8k",
        kind: Kind::Music,
        channels: 1,
        bits: 8,
        rate: 8_000,
        frames: 9_000,
        args: &["-5"],
    },
    Case {
        name: "s12-mono-11025",
        kind: Kind::Music,
        channels: 1,
        bits: 12,
        rate: 11_025,
        frames: 9_000,
        args: &["-5"],
    },
    Case {
        name: "s20-stereo-192k",
        kind: Kind::Music,
        channels: 2,
        bits: 20,
        rate: 192_000,
        frames: 20_000,
        args: &["-5"],
    },
    Case {
        name: "s32-stereo",
        kind: Kind::Music,
        channels: 2,
        bits: 32,
        rate: 48_000,
        frames: 12_000,
        args: &["--lax", "-8"],
    },
    Case {
        name: "s32-extremes-side",
        kind: Kind::Extremes,
        channels: 2,
        bits: 32,
        rate: 48_000,
        frames: 9_000,
        args: &["--lax", "-8", "-m"],
    },
    Case {
        name: "s4-mono",
        kind: Kind::Music,
        channels: 1,
        bits: 4,
        rate: 8_000,
        frames: 5_000,
        args: &["--lax", "-5"],
    },
    Case {
        name: "s16-7ch",
        kind: Kind::Music,
        channels: 7,
        bits: 16,
        rate: 48_000,
        frames: 9_000,
        args: &["-5"],
    },
    Case {
        name: "s16-8ch",
        kind: Kind::Music,
        channels: 8,
        bits: 16,
        rate: 48_000,
        frames: 9_000,
        args: &["-5"],
    },
    Case {
        name: "s16-3ch-32k",
        kind: Kind::Music,
        channels: 3,
        bits: 16,
        rate: 32_000,
        frames: 9_000,
        args: &["-5"],
    },
    Case {
        name: "rate-7000-khz-code",
        kind: Kind::Music,
        channels: 1,
        bits: 16,
        rate: 7_000,
        frames: 5_000,
        args: &["--lax", "-5"],
    },
    Case {
        name: "rate-44101-hz-code",
        kind: Kind::Music,
        channels: 1,
        bits: 16,
        rate: 44_101,
        frames: 5_000,
        args: &["-5"],
    },
    Case {
        name: "rate-100010-tens-code",
        kind: Kind::Music,
        channels: 1,
        bits: 16,
        rate: 100_010,
        frames: 5_000,
        args: &["--lax", "-5"],
    },
    Case {
        name: "rate-700000-streaminfo",
        kind: Kind::Music,
        channels: 1,
        bits: 16,
        rate: 700_000,
        frames: 5_000,
        args: &["--lax", "-5"],
    },
];

/// Encodes `case` with `flac` into `dir`, returning the source and the file.
pub fn encode_case(case: &Case, dir: &Path) -> (Vec<i32>, std::path::PathBuf) {
    let source = signal(
        case.kind,
        case.channels,
        case.bits,
        case.frames,
        0x5EED ^ case.frames as u64,
    );
    let wav = dir.join(format!("{}.wav", case.name));
    let flac = dir.join(format!("{}.flac", case.name));
    write_wav(&wav, &source, case.channels, case.rate, case.bits);
    let mut args = vec!["-s", "-f"];
    args.extend_from_slice(case.args);
    args.extend_from_slice(&[path_str(&wav), "-o", path_str(&flac)]);
    run("flac", &args);
    (source, flac)
}

fn decode_all(info: StreamInfo, frames: &[Vec<u8>]) -> (Vec<i32>, FlacDecoder) {
    let mut decoder = FlacDecoder::new(Some(info));
    let mut samples = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let decoded = decoder
            .decode(frame)
            .unwrap_or_else(|error| panic!("frame {index}: {error}"));
        samples.extend_from_slice(&decoded.samples);
    }
    (samples, decoder)
}

#[test]
fn libflac_streams_decode_bit_exactly() {
    if !have("flac") || !have("ffmpeg") {
        return;
    }
    let dir = scratch("decode-oracle");
    for case in CASES {
        let (source, flac) = encode_case(case, &dir);
        let (info, frames) = split(&flac);
        assert_eq!(u32::from(info.bits), case.bits, "{}", case.name);
        let (samples, decoder) = decode_all(info, &frames);
        assert_eq!(samples.len(), source.len(), "{} length", case.name);
        assert!(
            samples == source,
            "{}: differs from the source signal",
            case.name
        );
        let (reference, bits, channels) = flac_samples(&flac);
        assert_eq!(
            (bits, channels),
            (case.bits, case.channels),
            "{}",
            case.name
        );
        assert!(samples == reference, "{}: differs from flac -d", case.name);
        let ffmpeg = ffmpeg_samples(&flac, case.bits);
        assert!(samples == ffmpeg, "{}: differs from ffmpeg", case.name);
        assert_eq!(decoder.md5_check(), Md5Check::Match, "{} md5", case.name);
        eprintln!(
            "{}: {} frames, {} samples bit-exact",
            case.name,
            frames.len(),
            samples.len()
        );
    }
}

/// Rewrites a frame header to the variable-block-size strategy at sample
/// `number` and refreshes both CRCs. The subframes are libFLAC's bytes.
fn as_variable(frame: &[u8], number: u64) -> Vec<u8> {
    let header = FrameHeader::parse(frame).expect("header");
    let mut variable = header;
    variable.variable_block = true;
    variable.number = number;
    let mut out = variable.to_bytes().expect("header bytes");
    out.extend_from_slice(&frame[header.len..frame.len() - 2]);
    let crc = crc::crc16(&out);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

#[test]
fn a_variable_block_size_stream_decodes_bit_exactly() {
    if !have("flac") || !have("ffmpeg") {
        return;
    }
    let dir = scratch("variable");
    let source = signal(Kind::Music, 2, 16, 4096 * 6, 77);
    let wav = dir.join("source.wav");
    write_wav(&wav, &source, 2, 44_100, 16);
    // Three encodes of the same signal with different block sizes. Frames
    // are taken from each in turn at sample positions all three share.
    let mut encodes = Vec::new();
    for block in ["4096", "1024", "512"] {
        let flac = dir.join(format!("b{block}.flac"));
        run(
            "flac",
            &[
                "-s",
                "-f",
                "-b",
                block,
                path_str(&wav),
                "-o",
                path_str(&flac),
            ],
        );
        encodes.push(split(&flac));
    }
    let base = encodes[0].0;
    let mut stream = Vec::new();
    let mut frames = Vec::new();
    let mut sample = 0u64;
    let mut turn = 0;
    while sample < source.len() as u64 / 2 {
        // The turn's block size, or the next smaller one whose frame grid
        // the current position lies on.
        let blocks = [4096u64, 1024, 512];
        let pick = (turn % 3..3)
            .find(|&at| sample.is_multiple_of(blocks[at]))
            .expect("512 divides every position");
        let (_, source_frames) = &encodes[pick];
        let block = blocks[pick];
        let index = usize::try_from(sample / block).expect("index");
        let frame = as_variable(&source_frames[index], sample);
        sample += u64::from(FrameHeader::parse(&frame).expect("rewritten").block_size);
        stream.extend_from_slice(&frame);
        frames.push(frame);
        turn += 1;
    }
    let mut info = base;
    info.min_block = 512;
    info.max_block = 4096;
    info.min_frame = 0;
    info.max_frame = 0;
    let (samples, decoder) = decode_all(info, &frames);
    assert!(samples == source, "variable stream differs from the source");
    assert_eq!(decoder.md5_check(), Md5Check::Match);
    // The same stream as a file, checked by the two independent decoders.
    let mut file = b"fLaC".to_vec();
    file.extend_from_slice(&[0x80, 0, 0, 34]);
    file.extend_from_slice(&info.to_bytes());
    file.extend_from_slice(&stream);
    let path = dir.join("variable.flac");
    std::fs::write(&path, &file).expect("write");
    let (ok, stderr) = succeeds("flac", &["-s", "-t", path_str(&path)]);
    assert!(ok, "flac -t refused the variable stream: {stderr}");
    assert!(
        ffmpeg_samples(&path, 16) == source,
        "ffmpeg differs on the variable stream"
    );
    assert!(
        frames
            .iter()
            .all(|frame| FrameHeader::parse(frame).expect("h").variable_block)
    );
}

#[test]
fn every_channel_assignment_is_exercised() {
    if !have("flac") {
        return;
    }
    let dir = scratch("assignments");
    let mut seen = [false; 4];
    for case in CASES.iter().filter(|case| case.channels == 2) {
        let (_, flac) = encode_case(case, &dir);
        let (_, frames) = split(&flac);
        for frame in &frames {
            let slot = match FrameHeader::parse(frame).expect("header").assignment {
                ChannelAssignment::Independent(_) => 0,
                ChannelAssignment::LeftSide => 1,
                ChannelAssignment::SideRight => 2,
                ChannelAssignment::MidSide => 3,
            };
            seen[slot] = true;
        }
    }
    assert_eq!(
        seen, [true; 4],
        "independent, left/side, side/right, mid/side"
    );
}
