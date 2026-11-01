//! Annex E on the decoded picture.
//!
//! The sample aspect and the sequence timing are carried on the picture.
//! Colour, transfer, and matrix values from ISO/IEC 23091-2 are tags on that
//! same picture. A flat picture still matches FFmpeg inside the frame.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cutlass_decode::{
    CodecId, CodecParams, ColorMatrix, ColorPrimaries, ColorRange, Packet, SampleAspect,
    SequenceTiming, Transfer, VideoDecoder, VideoOutput,
};

use crate::H264Decoder;
use crate::nal::annex_b_nal;

const FRAME: usize = 16 * 16 + 8 * 8 * 2;

#[test]
fn sample_aspect_and_timing_are_on_the_picture() {
    let square = decode(stream(Some(1), None, None, None));
    assert_eq!(
        square.sample_aspect,
        SampleAspect {
            width: 1,
            height: 1
        }
    );
    assert_eq!(square.timing, SequenceTiming::ABSENT);
    assert!(flat(&square));

    let ratio = decode(stream(Some(2), None, None, Some((1, 60, true))));
    assert_eq!(
        ratio.sample_aspect,
        SampleAspect {
            width: 12,
            height: 11
        }
    );
    assert_eq!(
        ratio.timing,
        SequenceTiming {
            num_units: 1,
            time_scale: 60,
            fixed: true,
        }
    );

    let extended = decode(stream(Some(255), Some((32, 15)), None, None));
    assert_eq!(
        extended.sample_aspect,
        SampleAspect {
            width: 32,
            height: 15
        }
    );
    assert_eq!(
        decode(stream(Some(14), None, None, None)).sample_aspect,
        SampleAspect {
            width: 4,
            height: 3
        }
    );
    assert_eq!(
        decode(stream(Some(16), None, None, None)).sample_aspect,
        SampleAspect {
            width: 2,
            height: 1
        }
    );
    assert_eq!(
        decode(stream(None, None, None, None)).sample_aspect,
        SampleAspect::UNSPECIFIED
    );
    assert_eq!(
        decode(stream(Some(17), None, None, None)).sample_aspect,
        SampleAspect::UNSPECIFIED
    );
    assert_eq!(
        decode(stream(Some(255), Some((0, 15)), None, None)).sample_aspect,
        SampleAspect::UNSPECIFIED
    );

    let tagged = stream(
        Some(2),
        None,
        Some((1, 1, 1, false)),
        Some((1001, 60_000, true)),
    );
    let picture = decode(tagged.clone());
    assert_eq!(picture.primaries, ColorPrimaries::Bt709);
    assert_eq!(picture.transfer, Transfer::Bt709);
    assert_eq!(picture.matrix, ColorMatrix::Bt709);
    assert_eq!(picture.range, ColorRange::Limited);
    assert_eq!(
        picture.timing,
        SequenceTiming {
            num_units: 1001,
            time_scale: 60_000,
            fixed: true,
        }
    );
    let full = decode(stream(None, None, Some((1, 1, 1, true)), None));
    assert_eq!(full.range, ColorRange::Full);

    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &tagged);
        assert_eq!(raw, vec![128u8; FRAME]);
        let info = ffprobe(&ffmpeg, &tagged);
        assert!(info.contains("sample_aspect_ratio=12:11"), "{info}");
        assert!(info.contains("color_primaries=bt709"), "{info}");
        assert!(info.contains("color_transfer=bt709"), "{info}");
        assert!(info.contains("color_space=bt709"), "{info}");
    }
}

#[test]
fn the_remaining_colour_idc_values_are_tags() {
    let cases = [
        (
            1,
            1,
            1,
            ColorPrimaries::Bt709,
            Transfer::Bt709,
            ColorMatrix::Bt709,
        ),
        (
            4,
            4,
            4,
            ColorPrimaries::Bt470m,
            Transfer::Bt601,
            ColorMatrix::Fcc,
        ),
        (
            5,
            5,
            5,
            ColorPrimaries::Bt601,
            Transfer::Bt601,
            ColorMatrix::Bt601,
        ),
        (
            6,
            6,
            6,
            ColorPrimaries::Bt601,
            Transfer::Bt709,
            ColorMatrix::Bt601,
        ),
        (
            7,
            7,
            7,
            ColorPrimaries::Smpte240m,
            Transfer::Smpte240m,
            ColorMatrix::Smpte240m,
        ),
        (
            8,
            8,
            8,
            ColorPrimaries::Film,
            Transfer::Linear,
            ColorMatrix::Ycgco,
        ),
        (
            9,
            9,
            9,
            ColorPrimaries::Bt2020,
            Transfer::Log100,
            ColorMatrix::Bt2020,
        ),
        (
            9,
            10,
            10,
            ColorPrimaries::Bt2020,
            Transfer::LogSqrt,
            ColorMatrix::Bt2020,
        ),
        (
            10,
            11,
            11,
            ColorPrimaries::Smpte428,
            Transfer::Iec61966_2_4,
            ColorMatrix::Smpte2085,
        ),
        (
            11,
            12,
            12,
            ColorPrimaries::Smpte431,
            Transfer::Bt1361,
            ColorMatrix::ChromaDerived,
        ),
        (
            12,
            13,
            13,
            ColorPrimaries::Smpte432,
            Transfer::Srgb,
            ColorMatrix::ChromaDerivedConstant,
        ),
        (
            9,
            14,
            14,
            ColorPrimaries::Bt2020,
            Transfer::Bt2020,
            ColorMatrix::Ictcp,
        ),
        (
            9,
            15,
            9,
            ColorPrimaries::Bt2020,
            Transfer::Bt2020,
            ColorMatrix::Bt2020,
        ),
        (
            9,
            16,
            15,
            ColorPrimaries::Bt2020,
            Transfer::Pq,
            ColorMatrix::IptC2,
        ),
        (
            9,
            17,
            16,
            ColorPrimaries::Bt2020,
            Transfer::Smpte428,
            ColorMatrix::YcgcoRe,
        ),
        (
            9,
            18,
            17,
            ColorPrimaries::Bt2020,
            Transfer::Hlg,
            ColorMatrix::YcgcoRo,
        ),
        (
            22,
            2,
            0,
            ColorPrimaries::Ebu3213,
            Transfer::Unspecified,
            ColorMatrix::Identity,
        ),
        (
            2,
            3,
            2,
            ColorPrimaries::Unspecified,
            Transfer::Unspecified,
            ColorMatrix::Unspecified,
        ),
        (
            0,
            0,
            3,
            ColorPrimaries::Unspecified,
            Transfer::Unspecified,
            ColorMatrix::Unspecified,
        ),
    ];
    for (primaries, transfer, matrix, want_p, want_t, want_m) in cases {
        let picture = decode(stream(
            None,
            None,
            Some((primaries, transfer, matrix, false)),
            None,
        ));
        assert_eq!(
            (picture.primaries, picture.transfer, picture.matrix),
            (want_p, want_t, want_m),
            "idc {primaries}/{transfer}/{matrix}"
        );
        assert!(flat(&picture), "idc {primaries}/{transfer}/{matrix}");
    }
}

fn decode(annex: Vec<u8>) -> VideoOutput {
    let mut decoder = H264Decoder::open(&CodecParams::video(CodecId::H264, 16, 16)).unwrap();
    let mut pictures = decoder.push(&Packet::video(annex, 0, 1, 30, true)).unwrap();
    pictures.extend(decoder.drain().unwrap());
    assert_eq!(pictures.len(), 1);
    pictures.remove(0)
}

fn flat(picture: &VideoOutput) -> bool {
    picture
        .picture
        .y()
        .iter()
        .chain(picture.picture.u())
        .chain(picture.picture.v())
        .all(|sample| *sample == 128)
}

fn stream(
    aspect: Option<u32>,
    extended: Option<(u32, u32)>,
    colour: Option<(u32, u32, u32, bool)>,
    timing: Option<(u32, u32, bool)>,
) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps(aspect, extended, colour, timing));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &idr()));
    out
}

fn sps(
    aspect: Option<u32>,
    extended: Option<(u32, u32)>,
    colour: Option<(u32, u32, u32, bool)>,
    timing: Option<(u32, u32, bool)>,
) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(88, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(true);
    match aspect {
        Some(idc) => {
            w.bit(true);
            w.bits(idc, 8);
            if idc == 255 {
                let (width, height) = extended.unwrap_or((1, 1));
                w.bits(width, 16);
                w.bits(height, 16);
            }
        }
        None => w.bit(false),
    }
    w.bit(false);
    match colour {
        Some((primaries, transfer, matrix, full)) => {
            w.bit(true);
            w.bits(5, 3);
            w.bit(full);
            w.bit(true);
            w.bits(primaries, 8);
            w.bits(transfer, 8);
            w.bits(matrix, 8);
        }
        None => w.bit(false),
    }
    w.bit(false);
    match timing {
        Some((units, scale, fixed)) => {
            w.bit(true);
            w.bits(units, 32);
            w.bits(scale, 32);
            w.bit(fixed);
        }
        None => w.bit(false),
    }
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.trailing();
    w.bytes
}

fn pps() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bits(0, 2);
    w.se(0);
    w.se(0);
    w.se(0);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn idr() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(2);
    w.ue(0);
    w.bits(0, 4);
    w.ue(0);
    w.bits(0, 4);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);
    w.ue(3);
    w.ue(0);
    w.se(0);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn ffmpeg_bin() -> Option<PathBuf> {
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn ffmpeg_raw(ffmpeg: &std::path::Path, annex: &[u8]) -> Vec<u8> {
    let mut child = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), annex).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn ffprobe(ffmpeg: &std::path::Path, annex: &[u8]) -> String {
    let probe = ffmpeg.with_file_name("ffprobe");
    let mut child = Command::new(probe)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-show_entries",
            "stream=sample_aspect_ratio,color_primaries,color_transfer,color_space",
            "-of",
            "default=nw=1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), annex).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

struct Writer {
    bytes: Vec<u8>,
    bit: usize,
}

impl Writer {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit: 0,
        }
    }

    fn bit(&mut self, value: bool) {
        let index = self.bit / 8;
        if index == self.bytes.len() {
            self.bytes.push(0);
        }
        if value {
            self.bytes[index] |= 1 << (7 - (self.bit % 8));
        }
        self.bit += 1;
    }

    fn bits(&mut self, value: u32, n: u32) {
        for shift in (0..n).rev() {
            self.bit(((value >> shift) & 1) == 1);
        }
    }

    fn ue(&mut self, value: u32) {
        let code = value + 1;
        let zeros = 31 - code.leading_zeros();
        for _ in 0..zeros {
            self.bit(false);
        }
        self.bits(code, zeros + 1);
    }

    fn se(&mut self, value: i32) {
        let code = if value > 0 {
            value.unsigned_abs() * 2 - 1
        } else {
            value.unsigned_abs() * 2
        };
        self.ue(code);
    }

    fn trailing(&mut self) {
        self.bit(true);
        while !self.bit.is_multiple_of(8) {
            self.bit(false);
        }
    }
}
