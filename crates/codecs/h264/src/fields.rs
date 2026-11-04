//! Field pictures and macroblock-adaptive frame-field.
//!
//! Complementary fields share one `frame_num` and weave into a frame.
//! An adaptive frame stores each pair as two frame macroblocks or as a
//! top field and a bottom field. FFmpeg's software decoder is the oracle.

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

#[test]
fn x264_mbaff_matches_ffmpeg() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    };
    for (name, source, params) in [
        (
            "cavlc",
            "testsrc2=size=64x64:rate=25:duration=0.08",
            "tff=1:bframes=0:keyint=1:cabac=0",
        ),
        (
            "cabac",
            "testsrc2=size=64x64:rate=25:duration=0.08",
            "tff=1:bframes=0:keyint=1:cabac=1",
        ),
        (
            "predicted",
            "testsrc2=size=64x64:rate=25:duration=0.08",
            "tff=1:bframes=0:keyint=2:cabac=0",
        ),
        (
            "stripes",
            "color=c=gray:s=64x64:r=25,geq=lum='if(eq(mod(Y,2),0),235,16)':cb=128:cr=128",
            "tff=1:bframes=0:keyint=1:cabac=0",
        ),
        (
            "stripes cabac",
            "color=c=gray:s=64x64:r=25,geq=lum='if(eq(mod(Y,2),0),235,16)':cb=128:cr=128",
            "tff=1:bframes=0:keyint=1:cabac=1",
        ),
        (
            "mixed",
            "color=c=gray:s=64x64:r=25,geq=lum='if(lt(X,32),128,if(eq(mod(Y,2),0),235,16))':cb=128:cr=128",
            "tff=1:bframes=0:keyint=1:cabac=1",
        ),
    ] {
        let Some(annex) = x264_mbaff(&ffmpeg, source, params) else {
            panic!("x264 did not encode the {name} adaptive frame");
        };
        let raw = ffmpeg_raw_size(&ffmpeg, &annex, 64, 64).unwrap_or_else(|err| {
            panic!("ffmpeg did not decode the {name} adaptive frame: {err}");
        });
        let decoded = decode_units(&annex, 64, 64);
        let frame = 64 * 64 * 3 / 2;
        assert_eq!(decoded.len(), raw.len() / frame, "{name} picture count");
        for (index, picture) in decoded.iter().enumerate() {
            let start = index * frame;
            let (y, u, v) = split_420_at(&raw[start..start + frame], 64, 64);
            same(&format!("x264 {name} {index} y"), &picture.0, &y);
            same(&format!("x264 {name} {index} u"), &picture.1, &u);
            same(&format!("x264 {name} {index} v"), &picture.2, &v);
        }
    }
}

#[test]
fn mbaff_pairs_match_ffmpeg() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    };
    for (name, annex) in [
        ("field pair", access_mbaff(&[field_pair()])),
        ("frame pair", access_mbaff(&[frame_pair()])),
    ] {
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

fn access_mbaff(slices: &[Vec<u8>]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps_mbaff());
    out.extend(annex_b_nal(3, 8, &pps()));
    for slice in slices {
        out.extend(annex_b_nal(3, 5, slice));
    }
    out
}

fn sps_mbaff() -> Vec<u8> {
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
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn field_pair() -> Vec<u8> {
    let mut w = Writer::new();
    header(&mut w, None, 0);
    w.bit(true);
    intra16(&mut w);
    w.ue(25);
    w.align();
    w.bytes.extend(std::iter::repeat_n(40, 256));
    w.bytes.extend(std::iter::repeat_n(200, 64));
    w.bytes.extend(std::iter::repeat_n(220, 64));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn frame_pair() -> Vec<u8> {
    let mut w = Writer::new();
    header(&mut w, None, 0);
    w.bit(false);
    intra16(&mut w);
    intra16(&mut w);
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

fn decode_units(annex: &[u8], width: u32, height: u32) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut params = CodecParams::video(CodecId::H264, width, height);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    let mut pictures = Vec::new();
    for (n, unit) in access_units(annex).iter().enumerate() {
        pictures.extend(
            decoder
                .push(&Packet::video(unit.clone(), n as i64, 1, 25, n == 0))
                .expect("decode"),
        );
    }
    pictures.extend(decoder.drain().expect("drain"));
    pictures
        .iter()
        .map(|picture| {
            (
                picture.picture.y().to_vec(),
                picture.picture.u().to_vec(),
                picture.picture.v().to_vec(),
            )
        })
        .collect()
}

fn x264_mbaff(ffmpeg: &std::path::Path, source: &str, params: &str) -> Option<Vec<u8>> {
    let path = std::env::temp_dir().join(format!("cutlass-mbaff-{}.h264", std::process::id()));
    let status = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            source,
            "-an",
            "-c:v",
            "libx264",
            "-flags",
            "+ilme+ildct",
            "-x264-params",
            params,
            "-frames:v",
            "2",
            "-pix_fmt",
            "yuv420p",
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

fn ffmpeg_raw_size(
    ffmpeg: &std::path::Path,
    annex: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, String> {
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
        .map_err(|err| err.to_string())?;
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(annex)
        .map_err(|err| err.to_string())?;
    drop(child.stdin.take());
    let output = child.wait_with_output().map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let frame = width * height * 3 / 2;
    if output.stdout.len() < frame {
        return Err(format!("short decode {}", output.stdout.len()));
    }
    Ok(output.stdout)
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
            starts.push(header);
            index += 3;
        } else {
            index += 1;
        }
    }
    let mut units = Vec::new();
    let mut current = Vec::new();
    let mut saw = false;
    for (n, &header) in starts.iter().enumerate() {
        let end = starts.get(n + 1).copied().unwrap_or(data.len());
        let body = header + if data[header + 2] == 1 { 3 } else { 4 };
        let kind = data[body] & 31;
        let vcl = kind == 1 || kind == 5;
        let starts_picture = vcl && data.get(body + 1).is_some_and(|byte| byte & 0x80 != 0);
        if (kind == 9 || (starts_picture && saw)) && !current.is_empty() {
            units.push(std::mem::take(&mut current));
            saw = false;
        }
        current.extend_from_slice(&data[header..end]);
        saw |= vcl;
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
}

fn split_420_at(raw: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = width * height;
    let c = y / 4;
    (
        raw[..y].to_vec(),
        raw[y..y + c].to_vec(),
        raw[y + c..y + 2 * c].to_vec(),
    )
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
