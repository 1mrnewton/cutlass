//! Annex A picture size.
//!
//! A picture that fits any level in Table A-1 is decoded, including one larger
//! than the level named in the SPS. A picture that fits no level is refused
//! before a frame is allocated. A flat picture matches FFmpeg inside the frame.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};

use crate::H264Decoder;
use crate::nal::annex_b_nal;
use crate::set::{
    frame_within_levels, level_max_fs, macroblocks_fit, max_coded_edge, max_coded_mbs, parse_sps,
};

struct Size {
    level: u8,
    mb_w: u32,
    mb_h: u32,
}

#[test]
fn table_a1_bounds_the_coded_picture() {
    assert_eq!(max_coded_mbs(), 139_264);
    assert_eq!(max_coded_edge(), 16_880);
    assert_eq!(level_max_fs(52), Some(36_864));
    assert_eq!(level_max_fs(60), Some(139_264));

    // 4096×2160 is 256×135 macroblocks. Level 5.2 allows it.
    assert!(macroblocks_fit(256, 135, 36_864));
    // 8704×16 is past level 5.2's edge and inside level 6.
    assert!(!macroblocks_fit(544, 1, 36_864));
    assert!(macroblocks_fit(544, 1, 139_264));
    // 8688×1088 is past level 5.2's MaxFS. Both edges still fit that level.
    assert!(!macroblocks_fit(543, 68, 36_864));
    assert!(frame_within_levels(543, 68));
    // Level 1's edge is 28 macroblocks. 480×16 is past that level and still legal.
    assert!(!macroblocks_fit(30, 1, 99));
    assert!(frame_within_levels(30, 1));

    assert!(frame_within_levels(1055, 1));
    assert!(!frame_within_levels(1056, 1));
    assert!(!frame_within_levels(1024, 137));
    assert!(frame_within_levels(1, 1054));
    assert!(!frame_within_levels(1, 1056));
}

#[test]
fn pictures_inside_annex_a_match_ffmpeg() {
    let cases = [
        Size {
            level: 10,
            mb_w: 30,
            mb_h: 1,
        },
        Size {
            level: 52,
            mb_w: 544,
            mb_h: 1,
        },
        Size {
            level: 62,
            mb_w: 1055,
            mb_h: 1,
        },
        Size {
            level: 52,
            mb_w: 256,
            mb_h: 135,
        },
        Size {
            level: 60,
            mb_w: 543,
            mb_h: 68,
        },
    ];
    let ffmpeg = ffmpeg_bin();
    for case in cases {
        let width = case.mb_w * 16;
        let height = case.mb_h * 16;
        let rbsp = sps(case.level, case.mb_w, case.mb_h, true);
        let parsed = parse_sps(&rbsp).expect("sps");
        assert_eq!(parsed.level_idc, case.level, "{width}x{height}");
        assert_eq!((parsed.width, parsed.height), (width, height));
        let annex = stream(&rbsp, case.mb_w * case.mb_h);
        let nals = crate::nal::access_unit(&annex, 0).expect("nals");
        assert_eq!(
            nals.iter().map(|nal| nal.kind).collect::<Vec<_>>(),
            vec![7, 8, 5],
            "{width}x{height}"
        );
        let picture = decode(&annex, width, height);
        assert_eq!(
            (picture.picture.width(), picture.picture.height()),
            (width, height)
        );
        assert!(flat(&picture), "{width}x{height}");
        if let Some(ffmpeg) = &ffmpeg {
            let raw = ffmpeg_raw(ffmpeg, &annex);
            let samples =
                picture.picture.y().len() + picture.picture.u().len() + picture.picture.v().len();
            assert_eq!(raw.len(), samples, "{width}x{height}");
            assert!(raw.iter().all(|sample| *sample == 128), "{width}x{height}");
        }
    }
}

#[test]
fn a_picture_past_every_level_is_refused() {
    let refused = [
        sps(62, 1056, 1, true),
        sps(62, 1024, 137, true),
        sps(62, 1, 528, false),
        sps(62, 1_000_000, 1, true),
    ];
    let mut params = CodecParams::video(CodecId::H264, 16, 16);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    for rbsp in &refused {
        let err = parse_sps(rbsp).expect_err("parse");
        assert!(
            matches!(
                err,
                CodecError::Limit {
                    what: "picture size"
                }
            ),
            "{err:?}"
        );
        let err = decoder
            .push(&Packet::video(annex_b_nal(3, 7, rbsp), 0, 1, 30, false))
            .expect_err("push");
        assert!(
            matches!(
                err,
                CodecError::Limit {
                    what: "picture size"
                }
            ),
            "{err:?}"
        );
    }
    let annex = stream(&sps(30, 1, 1, true), 1);
    let picture = decode_with(&mut decoder, &annex);
    assert_eq!(
        (picture.picture.width(), picture.picture.height()),
        (16, 16)
    );
    assert!(flat(&picture));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        assert_eq!(raw, vec![128u8; 16 * 16 + 8 * 8 * 2]);
    }
}

fn stream(rbsp: &[u8], mbs: u32) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, rbsp);
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &idr(mbs)));
    out
}

fn decode(annex: &[u8], width: u32, height: u32) -> VideoOutput {
    let mut params = CodecParams::video(CodecId::H264, width, height);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    decode_with(&mut decoder, annex)
}

fn decode_with(decoder: &mut H264Decoder, annex: &[u8]) -> VideoOutput {
    let mut pictures = decoder
        .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
        .expect("decode");
    pictures.extend(decoder.drain().expect("drain"));
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

fn sps(level: u8, width_mbs: u32, height_map: u32, frame_only: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(66, 8);
    w.bits(0, 8);
    w.bits(u32::from(level), 8);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(width_mbs - 1);
    w.ue(height_map - 1);
    w.bit(frame_only);
    if !frame_only {
        w.bit(false);
    }
    w.bit(true);
    w.bit(false);
    w.bit(false);
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

fn idr(mbs: u32) -> Vec<u8> {
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
    for _ in 0..mbs {
        w.ue(3);
        w.ue(0);
        w.se(0);
        w.bit(true);
    }
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
