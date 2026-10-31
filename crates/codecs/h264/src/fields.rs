//! Complementary field pictures.
//!
//! `frame_mbs_only_flag` is clear and `mb_adaptive_frame_field_flag` is clear.
//! A top field and a bottom field of one `frame_num` weave into one frame.
//! FFmpeg's software decoder is the oracle. Macroblock-adaptive frame-field
//! stays refused.

use std::io::Write;
use std::process::{Command, Stdio};

use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder};

use crate::H264Decoder;
use crate::nal::annex_b_nal;

#[test]
fn complementary_fields_match_ffmpeg() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    };
    let paired = access(&[top_field(), bottom_field()]);
    let frame = access(&[frame_picture()]);
    for (name, annex) in [("field pair", paired), ("frame picture", frame)] {
        let decoded = decode(&annex);
        let raw =
            ffmpeg_raw(&ffmpeg, &annex).unwrap_or_else(|| panic!("ffmpeg did not decode {name}"));
        let (y, u, v) = split_420(&raw);
        same(&format!("{name} y"), &decoded.0, &y);
        same(&format!("{name} u"), &decoded.1, &u);
        same(&format!("{name} v"), &decoded.2, &v);
    }
}

fn decode(annex: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut params = CodecParams::video(CodecId::H264, 16, 32);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    let mut pictures = decoder
        .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
        .expect("decode");
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 1);
    let picture = &pictures[0];
    (
        picture.picture.y().to_vec(),
        picture.picture.u().to_vec(),
        picture.picture.v().to_vec(),
    )
}

fn ffmpeg_raw(ffmpeg: &std::path::Path, annex: &[u8]) -> Option<Vec<u8>> {
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
            "yuv420p",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("ffmpeg");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(annex)
        .expect("write");
    drop(child.stdin.take());
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        return None;
    }
    let frame = 16 * 32 + 8 * 16 + 8 * 16;
    if output.stdout.len() < frame {
        return None;
    }
    Some(output.stdout[..frame].to_vec())
}

fn access(slices: &[Vec<u8>]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps());
    out.extend(annex_b_nal(3, 8, &pps()));
    for slice in slices {
        out.extend(annex_b_nal(3, 5, slice));
    }
    out
}

fn sps() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(66, 8);
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
    w.bit(false);
    w.bit(false);
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

fn top_field() -> Vec<u8> {
    let mut w = Writer::new();
    header(&mut w, Some(false), 0);
    intra16(&mut w);
    w.trailing();
    w.bytes
}

fn bottom_field() -> Vec<u8> {
    let mut w = Writer::new();
    header(&mut w, Some(true), 1);
    w.ue(25);
    w.align();
    w.bytes.extend(std::iter::repeat_n(40, 256));
    w.bytes.extend(std::iter::repeat_n(200, 64));
    w.bytes.extend(std::iter::repeat_n(220, 64));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn frame_picture() -> Vec<u8> {
    let mut w = Writer::new();
    header(&mut w, None, 0);
    intra16(&mut w);
    intra16(&mut w);
    w.trailing();
    w.bytes
}

fn header(w: &mut Writer, bottom: Option<bool>, poc: u32) {
    w.ue(0);
    w.ue(2);
    w.ue(0);
    w.bits(0, 4);
    match bottom {
        Some(flag) => {
            w.bit(true);
            w.bit(flag);
        }
        None => w.bit(false),
    }
    w.ue(0);
    w.bits(poc, 4);
    w.bit(false);
    w.bit(false);
    w.se(0);
}

fn intra16(w: &mut Writer) {
    w.ue(3);
    w.ue(0);
    w.se(0);
    w.bit(true);
}

fn split_420(raw: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = 16 * 32;
    let c = 8 * 16;
    (
        raw[..y].to_vec(),
        raw[y..y + c].to_vec(),
        raw[y + c..y + 2 * c].to_vec(),
    )
}

fn same(name: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{name} length");
    if let Some(index) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{name}[{index}] = {} want {}", got[index], want[index]);
    }
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
