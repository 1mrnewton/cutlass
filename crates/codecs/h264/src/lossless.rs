//! QP′Y-zero transform bypass, clause 8.5.
//!
//! FFmpeg's software decoder is the oracle. A vertical or horizontal intra block
//! also runs the intra residual process in clause 8.5.15. The comparison skips
//! when `ffmpeg` is not installed.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder};

use crate::H264Decoder;
use crate::nal::annex_b_nal;

#[derive(Clone, Copy)]
enum Case {
    /// High, 4:2:0, one luma DC at QP′Y 0. The Hadamard is not applied.
    Dc420,
    /// The bypass flag is set and QP′Y is 26, so the transform still runs.
    Dc420At26,
    /// High 4:4:4 Predictive, 16×32. The lower macroblock is vertical intra.
    Vertical,
    /// High 4:2:2, one chroma DC at QP′Y 0.
    Chroma422,
    /// High 4:4:4 Predictive, one Cb DC at QP′Y 0.
    Chroma444,
}

#[test]
fn qpprime_bypass_matches_ffmpeg() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    };
    for case in [
        Case::Dc420,
        Case::Dc420At26,
        Case::Vertical,
        Case::Chroma422,
        Case::Chroma444,
    ] {
        let annex = annex(case);
        let (width, height, pix) = shape(case);
        let decoded = decode(&annex, width, height, pix);
        let raw = ffmpeg_raw(&ffmpeg, pix, width, height, &annex).unwrap_or_else(|| {
            panic!("ffmpeg did not decode {}", case_name(case));
        });
        same(&format!("{} y", case_name(case)), &decoded.0, &raw.0);
        same(&format!("{} u", case_name(case)), &decoded.1, &raw.1);
        same(&format!("{} v", case_name(case)), &decoded.2, &raw.2);
    }
}

#[test]
fn x264_lossless_matches_ffmpeg() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    };
    let Some(annex) = x264_lossless(&ffmpeg) else {
        panic!("ffmpeg could not encode a lossless stream");
    };
    let units = access_units(&annex);
    assert!(units.len() > 1, "lossless stream produced one picture");
    let mut params = CodecParams::video(CodecId::H264, 64, 64);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    let mut frames = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        let pictures = decoder
            .push(&Packet::video(
                unit.clone(),
                index as i64,
                1,
                25,
                index == 0,
            ))
            .unwrap_or_else(|err| panic!("lossless access unit {index}: {err}"));
        frames.extend(pictures);
    }
    frames.extend(decoder.drain().expect("drain"));
    let raw = ffmpeg_file(&ffmpeg, "yuv444p", &annex).expect("oracle");
    let frame_len = 64 * 64 * 3;
    assert_eq!(raw.len(), frames.len() * frame_len);
    for (index, picture) in frames.iter().enumerate() {
        let start = index * frame_len;
        let Some(frame) = picture.yuv444.as_ref() else {
            panic!("lossless picture {index} dropped its planes");
        };
        let (y, u, v) = split_444(64, 64, &raw[start..start + frame_len]);
        same(&format!("lossless {index} y"), frame.y(), &y);
        same(&format!("lossless {index} u"), frame.u(), &u);
        same(&format!("lossless {index} v"), frame.v(), &v);
    }
}

fn shape(case: Case) -> (u32, u32, &'static str) {
    match case {
        Case::Vertical => (16, 32, "yuv420p"),
        Case::Chroma422 => (16, 16, "yuv422p"),
        Case::Chroma444 => (16, 16, "yuv444p"),
        Case::Dc420 | Case::Dc420At26 => (16, 16, "yuv420p"),
    }
}

fn case_name(case: Case) -> &'static str {
    match case {
        Case::Dc420 => "dc420",
        Case::Dc420At26 => "dc420-qp26",
        Case::Vertical => "vertical",
        Case::Chroma422 => "chroma422",
        Case::Chroma444 => "chroma444",
    }
}

fn decode(annex: &[u8], width: u32, height: u32, pix: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut params = CodecParams::video(CodecId::H264, width, height);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    let mut pictures = decoder
        .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
        .expect("decode");
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 1);
    let picture = &pictures[0];
    match pix {
        "yuv422p" => {
            let Some(frame) = picture.yuv422.as_ref() else {
                panic!("4:2:2 dropped its planes");
            };
            (frame.y().to_vec(), frame.u().to_vec(), frame.v().to_vec())
        }
        "yuv444p" => {
            let Some(frame) = picture.yuv444.as_ref() else {
                panic!("4:4:4 dropped its planes");
            };
            (frame.y().to_vec(), frame.u().to_vec(), frame.v().to_vec())
        }
        _ => (
            picture.picture.y().to_vec(),
            picture.picture.u().to_vec(),
            picture.picture.v().to_vec(),
        ),
    }
}

fn annex(case: Case) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps(case));
    out.extend(annex_b_nal(3, 8, &pps(case)));
    out.extend(annex_b_nal(3, 5, &slice(case)));
    out
}

fn sps(case: Case) -> Vec<u8> {
    let (profile, chroma, height_minus) = match case {
        Case::Chroma422 => (122u32, 2u32, 0u32),
        Case::Chroma444 => (244, 3, 0),
        Case::Vertical => (244, 1, 1),
        Case::Dc420 | Case::Dc420At26 => (100, 1, 0),
    };
    let mut w = Writer::new();
    w.bits(profile, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(chroma);
    if chroma == 3 {
        w.bit(false);
    }
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(height_minus);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn pps(case: Case) -> Vec<u8> {
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
    w.se(if matches!(case, Case::Dc420At26) {
        0
    } else {
        -26
    });
    w.se(0);
    w.se(0);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn slice(case: Case) -> Vec<u8> {
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
    match case {
        Case::Dc420 | Case::Dc420At26 => write_i16(&mut w, 1, true, false),
        Case::Chroma422 => write_i16(&mut w, 2, false, true),
        Case::Chroma444 => write_i16(&mut w, 3, false, true),
        Case::Vertical => {
            write_i16(&mut w, 1, false, false);
            w.ue(1);
            w.ue(0);
            w.se(0);
            w.bits(0b0101, 4);
        }
    }
    w.trailing();
    w.bytes
}

fn write_i16(w: &mut Writer, chroma: u32, luma_dc: bool, chroma_dc: bool) {
    if chroma_dc && chroma == 2 {
        w.ue(7);
        w.ue(0);
    } else {
        w.ue(3);
        if chroma == 2 || chroma == 1 {
            w.ue(0);
        }
    }
    w.se(0);
    let token = |writer: &mut Writer, one: bool| {
        if one {
            writer.bits(0b0101, 4);
        } else {
            writer.bit(true);
        }
    };
    token(w, luma_dc);
    if chroma == 3 {
        token(w, chroma_dc);
        w.bit(true);
    } else if chroma_dc {
        w.bits(0b0101, 4);
        w.bit(true);
    }
}

fn same(name: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{name} length");
    if let Some(index) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{name}[{index}] = {} want {}", got[index], want[index]);
    }
}

fn split_420(width: usize, height: usize, raw: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y_len = width * height;
    let c_len = width / 2 * height / 2;
    (
        raw[..y_len].to_vec(),
        raw[y_len..y_len + c_len].to_vec(),
        raw[y_len + c_len..].to_vec(),
    )
}

fn split_422(width: usize, height: usize, raw: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y_len = width * height;
    let c_len = width / 2 * height;
    (
        raw[..y_len].to_vec(),
        raw[y_len..y_len + c_len].to_vec(),
        raw[y_len + c_len..].to_vec(),
    )
}

fn split_444(width: usize, height: usize, raw: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n = width * height;
    (
        raw[..n].to_vec(),
        raw[n..n * 2].to_vec(),
        raw[n * 2..].to_vec(),
    )
}

fn ffmpeg_raw(
    ffmpeg: &PathBuf,
    pix: &str,
    width: u32,
    height: u32,
    annex: &[u8],
) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let bytes = ffmpeg_file(ffmpeg, pix, annex).ok()?;
    let width = width as usize;
    let height = height as usize;
    Some(match pix {
        "yuv422p" => split_422(width, height, &bytes),
        "yuv444p" => split_444(width, height, &bytes),
        _ => split_420(width, height, &bytes),
    })
}

fn ffmpeg_file(ffmpeg: &PathBuf, pix: &str, annex: &[u8]) -> Result<Vec<u8>, String> {
    static TEMP: AtomicU64 = AtomicU64::new(0);
    let id = TEMP.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("cutlass-h264-bypass-{id}.h264"));
    std::fs::write(&path, annex).map_err(|err| err.to_string())?;
    let output = Command::new(ffmpeg)
        .args([
            "-y",
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
        .args(["-f", "rawvideo", "-pix_fmt", pix, "-"])
        .stdin(Stdio::null())
        .output();
    let _ = std::fs::remove_file(&path);
    let output = output.map_err(|err| err.to_string())?;
    if !output.status.success() || output.stdout.is_empty() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    Ok(output.stdout)
}

fn x264_lossless(ffmpeg: &PathBuf) -> Option<Vec<u8>> {
    let path =
        std::env::temp_dir().join(format!("cutlass-h264-lossless-{}.h264", std::process::id()));
    let status = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=25:duration=0.2",
            "-an",
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv444p",
            "-g",
            "8",
            "-bf",
            "0",
            "-f",
            "h264",
        ])
        .arg(&path)
        .status()
        .ok()?;
    if !status.success() {
        let _ = std::fs::remove_file(&path);
        return None;
    }
    let annex = std::fs::read(&path).ok();
    let _ = std::fs::remove_file(&path);
    annex
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

fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut index = 0usize;
    while index + 3 < data.len() {
        if data[index] == 0 && data[index + 1] == 0 && data[index + 2] == 1 {
            let header = if index > 0 && data[index - 1] == 0 {
                index - 1
            } else {
                index
            };
            starts.push((header, index + 3));
            index += 3;
        } else {
            index += 1;
        }
    }
    let mut units = Vec::new();
    let mut current = Vec::new();
    let mut saw_vcl = false;
    for (n, &(header, body)) in starts.iter().enumerate() {
        let end = starts
            .get(n + 1)
            .map(|(next, _)| *next)
            .unwrap_or(data.len());
        if body >= data.len() || header >= end {
            continue;
        }
        let kind = data[body] & 31;
        let vcl = kind == 1 || kind == 5;
        let starts_picture = vcl && data.get(body + 1).is_some_and(|byte| byte & 0x80 != 0);
        if (kind == 9 || (starts_picture && saw_vcl)) && !current.is_empty() {
            units.push(std::mem::take(&mut current));
            saw_vcl = false;
        }
        current.extend_from_slice(&data[header..end]);
        saw_vcl |= vcl;
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
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
