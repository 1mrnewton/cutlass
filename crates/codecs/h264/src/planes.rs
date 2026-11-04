//! Separate colour planes, clause 7.4.2.1.1.
//!
//! FFmpeg refuses the flag, so the recommendation is the oracle. Each plane is
//! a monochrome slice of one picture. A following P picture copies that plane
//! from the reference.

use std::io::Write;
use std::process::{Command, Stdio};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder};

use crate::H264Decoder;
use crate::nal::annex_b_nal;
use crate::set::parse_sps;

#[test]
fn separate_colour_planes_follow_the_recommendation() {
    let intra = picture(true, &[0, 1, 2]);
    let inter = picture(false, &[0, 1, 2]);
    let mut params = CodecParams::video(CodecId::H264, 16, 16);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    let mut pictures = decoder
        .push(&Packet::video(intra.clone(), 0, 1, 30, true))
        .expect("intra");
    pictures.extend(
        decoder
            .push(&Packet::video(inter, 1, 1, 30, false))
            .expect("inter"),
    );
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2, "separate planes produced one picture");
    for (index, picture) in pictures.iter().enumerate() {
        let Some(frame) = picture.yuv444.as_ref() else {
            panic!("picture {index} dropped its 4:4:4 planes");
        };
        assert!(
            frame.y().iter().all(|sample| *sample == 128),
            "picture {index} y is not the intra plane"
        );
        assert!(
            frame.u().iter().all(|sample| *sample == 40),
            "picture {index} u is not the cb plane"
        );
        assert!(
            frame.v().iter().all(|sample| *sample == 80),
            "picture {index} v is not the cr plane"
        );
    }

    let mut missing = H264Decoder::open(&params).expect("open");
    let err = missing
        .push(&Packet::video(picture(true, &[0, 1]), 0, 1, 30, true))
        .expect_err("a picture needs all three planes");
    assert_eq!(
        err,
        CodecError::Malformed {
            at_bit: 0,
            what: "incomplete picture",
        }
    );

    let mut bad = H264Decoder::open(&params).expect("open");
    let err = bad
        .push(&Packet::video(picture_plane(3), 0, 1, 30, true))
        .expect_err("colour_plane_id 3");
    assert!(matches!(
        err,
        CodecError::Malformed {
            what: "colour plane",
            ..
        }
    ));

    let err = parse_sps(&sps(0)).expect_err("monochrome");
    assert_eq!(
        err,
        CodecError::Unsupported {
            what: "chroma format"
        }
    );

    if let Some(ffmpeg) = ffmpeg_bin() {
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
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv444p",
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("ffmpeg");
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(&intra)
            .expect("write");
        drop(child.stdin.take());
        let status = child.wait().expect("wait");
        assert!(!status.success(), "ffmpeg decoded separate colour planes");
    }
}

fn picture(intra: bool, planes: &[u8]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps(3));
    out.extend(annex_b_nal(3, 8, &pps()));
    for plane in planes {
        let kind = if intra { 5 } else { 1 };
        out.extend(annex_b_nal(3, kind, &slice(intra, *plane)));
    }
    out
}

fn picture_plane(plane: u8) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps(3));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &slice(true, plane)));
    out
}

fn sps(chroma: u32) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(244, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(chroma);
    if chroma == 3 {
        w.bit(true);
    }
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
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
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn slice(intra: bool, plane: u8) -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(if intra { 2 } else { 0 });
    w.ue(0);
    w.bits(u32::from(plane), 2);
    w.bits(u32::from(!intra), 4);
    if intra {
        w.ue(0);
    }
    w.bits(if intra { 0 } else { 2 }, 4);
    if !intra {
        w.bit(false);
        w.bit(false);
    }
    if intra {
        w.bit(false);
        w.bit(false);
    } else {
        w.bit(false);
    }
    w.se(0);
    if intra {
        if plane == 0 {
            w.ue(3);
            w.se(0);
            w.bit(true);
        } else {
            w.ue(25);
            w.align();
            let sample = if plane == 1 { 40 } else { 80 };
            w.bytes.extend(std::iter::repeat_n(sample, 256));
            w.bit = w.bytes.len() * 8;
        }
    } else {
        w.ue(1);
    }
    w.trailing();
    w.bytes
}

fn ffmpeg_bin() -> Option<std::path::PathBuf> {
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"] {
        let path = std::path::PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    None
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

    fn align(&mut self) {
        while !self.bit.is_multiple_of(8) {
            self.bit(false);
        }
    }

    fn trailing(&mut self) {
        self.bit(true);
        while !self.bit.is_multiple_of(8) {
            self.bit(false);
        }
    }
}
