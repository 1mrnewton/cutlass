//! 4:2:0 pictures at 9 through 14 bits.
//!
//! libx264 on this machine emits 8-bit and 10-bit only, so these streams are
//! written here. FFmpeg's software decoder is the picture oracle at 9, 12, and
//! 14 bits. It refuses 11 and 13, and the recommendation is the oracle there.
//! The test skips the FFmpeg comparison when `ffmpeg` is not installed.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};

use crate::H264Decoder;
use crate::nal::annex_b_nal;

const WIDTH: usize = 16;
const HEIGHT: usize = 16;

#[derive(Clone, Copy)]
enum Picture {
    /// I_16x16 DC, no residual. Every sample is the midpoint.
    Flat,
    /// I_16x16 DC with one +1 in the luma DC block.
    Dc,
    /// I_PCM. The samples are the codes this test wrote.
    Pcm,
}

#[test]
fn bit_depths_9_through_14_reconstruct_420() {
    let ffmpeg = ffmpeg_bin();
    for depth in [9u32, 11, 12, 13, 14] {
        for picture in [Picture::Flat, Picture::Dc, Picture::Pcm] {
            let annex = annex(depth, picture);
            let decoded = decode_one(&annex, depth);
            let (y, u, v) = planes(&decoded, depth as u8);
            let shift = depth - 8;
            assert_eq!(
                decoded.picture.y()[0],
                (y[0] >> shift) as u8,
                "depth {depth} dropped the top 8 bits"
            );
            match picture {
                Picture::Flat => {
                    let mid = 1u16 << (depth - 1);
                    assert!(y.iter().all(|sample| *sample == mid), "depth {depth} luma");
                    assert!(u.iter().all(|sample| *sample == mid), "depth {depth} cb");
                    assert!(v.iter().all(|sample| *sample == mid), "depth {depth} cr");
                }
                Picture::Dc => {
                    let mid = 1u16 << (depth - 1);
                    assert!(
                        y.iter().any(|sample| *sample != mid),
                        "depth {depth} ignored the DC coefficient"
                    );
                    let limit = (1u16 << depth) - 1;
                    assert!(
                        y.iter()
                            .chain(u.iter())
                            .chain(v.iter())
                            .all(|sample| *sample <= limit),
                        "depth {depth} sample past the limit"
                    );
                }
                Picture::Pcm => {
                    let (expect_y, expect_u, expect_v) = pcm_planes(depth);
                    assert_eq!(y, expect_y, "depth {depth} pcm luma");
                    assert_eq!(u, expect_u, "depth {depth} pcm cb");
                    assert_eq!(v, expect_v, "depth {depth} pcm cr");
                }
            }
            let Some(ffmpeg) = ffmpeg.as_ref() else {
                continue;
            };
            let raw = ffmpeg_raw(ffmpeg, depth, &annex);
            if matches!(depth, 11 | 13) {
                assert!(
                    raw.is_none(),
                    "ffmpeg decoded {depth}-bit, which it refuses"
                );
                continue;
            }
            let Some(raw) = raw else {
                panic!("ffmpeg did not decode {depth}-bit 4:2:0");
            };
            let (fy, fu, fv) = split_raw(&raw);
            assert_eq!(y, fy, "depth {depth} {:?} luma", picture_name(picture));
            assert_eq!(u, fu, "depth {depth} {:?} cb", picture_name(picture));
            assert_eq!(v, fv, "depth {depth} {:?} cr", picture_name(picture));
        }
    }
}

#[test]
fn a_depth_past_14_or_a_mismatch_is_refused() {
    let mismatch = annex_with(2, 0, Picture::Flat);
    let err = decode_err(&mismatch);
    assert!(
        matches!(err, CodecError::Unsupported { what: "bit depth" }),
        "mismatched depths: {err:?}"
    );
    let past = annex_with(7, 7, Picture::Flat);
    let err = decode_err(&past);
    assert!(
        matches!(err, CodecError::Unsupported { what: "bit depth" }),
        "15-bit: {err:?}"
    );
}

fn picture_name(picture: Picture) -> &'static str {
    match picture {
        Picture::Flat => "flat",
        Picture::Dc => "dc",
        Picture::Pcm => "pcm",
    }
}

fn planes(picture: &VideoOutput, depth: u8) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let Some(high) = picture.yuv420_high.as_ref() else {
        panic!("depth {depth} dropped the wide samples");
    };
    assert_eq!(high.bit_depth(), depth);
    assert!(picture.yuv420p10.is_none());
    (high.y().to_vec(), high.u().to_vec(), high.v().to_vec())
}

fn decode_one(annex: &[u8], depth: u32) -> VideoOutput {
    let mut decoder = open();
    let mut frames = decoder
        .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
        .unwrap_or_else(|err| panic!("depth {depth}: {err}"));
    frames.extend(
        decoder
            .drain()
            .unwrap_or_else(|err| panic!("drain {depth}: {err}")),
    );
    assert_eq!(frames.len(), 1, "depth {depth} picture count");
    frames.pop().expect("picture")
}

fn decode_err(annex: &[u8]) -> CodecError {
    let mut decoder = open();
    decoder
        .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
        .expect_err("refused depth")
}

fn open() -> H264Decoder {
    let mut params = CodecParams::video(CodecId::H264, WIDTH as u32, HEIGHT as u32);
    params.nal_length_size = 0;
    H264Decoder::open(&params).expect("open")
}

fn ffmpeg_raw(ffmpeg: &PathBuf, depth: u32, annex: &[u8]) -> Option<Vec<u8>> {
    let path = std::env::temp_dir().join(format!(
        "cutlass-h264-depth-{depth}-{}.h264",
        std::process::id()
    ));
    std::fs::write(&path, annex).expect("annex");
    let pix_fmt = format!("yuv420p{depth}le");
    let raw = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            "-i",
        ])
        .arg(&path)
        .args(["-an", "-f", "rawvideo", "-pix_fmt", &pix_fmt, "-"])
        .stdout(Stdio::piped())
        .output()
        .expect("ffmpeg raw");
    let _ = std::fs::remove_file(&path);
    if !raw.status.success() {
        return None;
    }
    Some(raw.stdout)
}

fn split_raw(raw: &[u8]) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let y_len = WIDTH * HEIGHT;
    let c_len = y_len / 4;
    let words = words(raw);
    assert_eq!(words.len(), y_len + c_len * 2, "ffmpeg frame size");
    let y = words[..y_len].to_vec();
    let u = words[y_len..y_len + c_len].to_vec();
    let v = words[y_len + c_len..].to_vec();
    (y, u, v)
}

fn words(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

fn ffmpeg_bin() -> Option<PathBuf> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let path = PathBuf::from(candidate);
        let Ok(output) = Command::new(candidate).arg("-version").output() else {
            continue;
        };
        if output.status.success() {
            return Some(path);
        }
    }
    eprintln!("skipped: ffmpeg is not installed");
    None
}

fn annex(depth: u32, picture: Picture) -> Vec<u8> {
    annex_with(depth - 8, depth - 8, picture)
}

fn annex_with(luma_minus8: u32, chroma_minus8: u32, picture: Picture) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps(luma_minus8, chroma_minus8));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &slice(picture, luma_minus8 + 8)));
    out
}

fn sps(luma_minus8: u32, chroma_minus8: u32) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(244, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(luma_minus8);
    w.ue(chroma_minus8);
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

fn slice(picture: Picture, depth: u32) -> Vec<u8> {
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
    match picture {
        Picture::Flat => {
            w.ue(3);
            w.ue(0);
            w.se(0);
            w.bit(true);
        }
        Picture::Dc => {
            w.ue(3);
            w.ue(0);
            w.se(0);
            // TotalCoeff 1, TrailingOnes 1, sign +, total_zeros 0.
            w.bits(0b0101, 4);
        }
        Picture::Pcm => {
            w.ue(25);
            w.align_zeros();
            for index in 0..384 {
                w.bits(pcm_sample(index, depth), depth);
            }
        }
    }
    w.trailing();
    w.bytes
}

fn pcm_planes(depth: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let y = (0..256)
        .map(|index| pcm_sample(index, depth) as u16)
        .collect();
    let u = (256..320)
        .map(|index| pcm_sample(index, depth) as u16)
        .collect();
    let v = (320..384)
        .map(|index| pcm_sample(index, depth) as u16)
        .collect();
    (y, u, v)
}

fn pcm_sample(index: u32, depth: u32) -> u32 {
    let mask = (1u32 << depth) - 1;
    (index.wrapping_mul(17).wrapping_add(3)) & mask
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

    fn align_zeros(&mut self) {
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
