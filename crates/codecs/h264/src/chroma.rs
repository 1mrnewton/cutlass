//! 4:2:2 and 4:4:4 pictures at 8 through 14 bits.
//!
//! libx264 supplies the inter, CABAC, and deblock pictures at 8 and 10 bits.
//! The other depths are written here. FFmpeg's software decoder is the picture
//! oracle at 8, 9, 10, 12, and 14 bits. It refuses 11 and 13, and the
//! recommendation is the oracle there. A comparison skips when `ffmpeg` is
//! not installed.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};

use crate::H264Decoder;
use crate::nal::annex_b_nal;

const WIDTH: usize = 16;
const HEIGHT: usize = 16;

#[derive(Clone, Copy)]
enum Format {
    C422,
    C444,
}

#[derive(Clone, Copy)]
enum Picture {
    Flat,
    LumaDc,
    ChromaDc,
    Pcm,
}

#[test]
fn hand_built_422_and_444_match_the_oracle() {
    let ffmpeg = ffmpeg_bin();
    for format in [Format::C422, Format::C444] {
        for depth in [8u32, 9, 10, 11, 12, 13, 14] {
            for picture in [
                Picture::Flat,
                Picture::LumaDc,
                Picture::ChromaDc,
                Picture::Pcm,
            ] {
                let annex = annex(format, depth, picture);
                let decoded = decode_one(&annex, format, depth);
                let (y, u, v) = planes(&decoded, format, depth);
                let (y_len, c_len) = lengths(format);
                assert_eq!(y.len(), y_len);
                assert_eq!(u.len(), c_len);
                assert_eq!(v.len(), c_len);
                let mid = 1u16 << (depth - 1);
                match picture {
                    Picture::Flat => {
                        assert!(y.iter().all(|sample| *sample == mid));
                        assert!(u.iter().all(|sample| *sample == mid));
                        assert!(v.iter().all(|sample| *sample == mid));
                    }
                    Picture::LumaDc => {
                        assert!(y.iter().any(|sample| *sample != mid));
                    }
                    Picture::ChromaDc => {
                        assert!(
                            u.iter().any(|sample| *sample != mid),
                            "{:?} depth {depth} ignored the chroma DC",
                            format_name(format)
                        );
                    }
                    Picture::Pcm => {
                        let (expect_y, expect_u, expect_v) = pcm_planes(format, depth);
                        assert_eq!(y, expect_y);
                        assert_eq!(u, expect_u);
                        assert_eq!(v, expect_v);
                    }
                }
                let Some(ffmpeg) = ffmpeg.as_ref() else {
                    continue;
                };
                let raw = ffmpeg_raw(ffmpeg, format, depth, &annex);
                if matches!(depth, 11 | 13) {
                    assert!(raw.is_none(), "ffmpeg decoded {depth}-bit");
                    continue;
                }
                let Some(raw) = raw else {
                    panic!(
                        "ffmpeg did not decode {:?} {depth}-bit",
                        format_name(format)
                    );
                };
                let (fy, fu, fv) = split_raw(format, depth, &raw);
                assert_eq!(
                    y,
                    fy,
                    "{:?} {depth} {:?} luma",
                    format_name(format),
                    picture_name(picture)
                );
                assert_eq!(
                    u,
                    fu,
                    "{:?} {depth} {:?} cb",
                    format_name(format),
                    picture_name(picture)
                );
                assert_eq!(
                    v,
                    fv,
                    "{:?} {depth} {:?} cr",
                    format_name(format),
                    picture_name(picture)
                );
            }
        }
    }
}

#[test]
fn x264_422_and_444_match_ffmpeg() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    };
    for (profile, pix, format, depth) in [
        ("high422", "yuv422p", Format::C422, 8u32),
        ("high444", "yuv444p", Format::C444, 8),
        ("high422", "yuv422p10le", Format::C422, 10),
        ("high444", "yuv444p10le", Format::C444, 10),
    ] {
        let Some(annex) = x264(&ffmpeg, profile, pix) else {
            if depth == 8 {
                panic!("ffmpeg could not encode {profile}");
            }
            eprintln!("skipped: this ffmpeg cannot encode {profile} {pix}");
            continue;
        };
        let units = access_units(&annex);
        assert!(!units.is_empty());
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
                .unwrap_or_else(|err| panic!("{profile} access unit {index}: {err}"));
            frames.extend(pictures);
        }
        frames.extend(decoder.drain().expect("drain"));
        let raw = ffmpeg_file(&ffmpeg, pix, &annex).unwrap_or_else(|err| {
            panic!("{profile} oracle failed: {err}");
        });
        let frame_len = raw_len(format, depth, 64, 64);
        assert_eq!(raw.len(), frames.len() * frame_len, "{profile} frame count");
        assert!(frames.len() > 1, "{profile} produced one picture");
        for (index, picture) in frames.iter().enumerate() {
            let start = index * frame_len;
            let (y, u, v) = planes(picture, format, depth);
            let (fy, fu, fv) = split_sized(format, depth, 64, 64, &raw[start..start + frame_len]);
            assert_eq!(y, fy, "{profile} picture {index} luma");
            assert_eq!(u, fu, "{profile} picture {index} cb");
            assert_eq!(v, fv, "{profile} picture {index} cr");
        }
    }
}

#[test]
fn monochrome_chroma_format_is_refused() {
    let annex = annex_chroma0();
    let mut params = CodecParams::video(CodecId::H264, WIDTH as u32, HEIGHT as u32);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    let err = decoder
        .push(&Packet::video(annex, 0, 1, 30, true))
        .expect_err("monochrome");
    assert!(
        matches!(
            err,
            CodecError::Unsupported {
                what: "chroma format"
            }
        ),
        "{err:?}"
    );
}

fn planes(picture: &VideoOutput, format: Format, depth: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    assert!(picture.yuv420p10.is_none());
    assert!(picture.yuv420_high.is_none());
    match (format, depth) {
        (Format::C422, 8) => {
            let Some(frame) = picture.yuv422.as_ref() else {
                panic!("8-bit 4:2:2 dropped its planes");
            };
            assert!(picture.yuv422_high.is_none());
            assert!(picture.yuv444.is_none());
            (
                frame.y().iter().map(|sample| u16::from(*sample)).collect(),
                frame.u().iter().map(|sample| u16::from(*sample)).collect(),
                frame.v().iter().map(|sample| u16::from(*sample)).collect(),
            )
        }
        (Format::C444, 8) => {
            let Some(frame) = picture.yuv444.as_ref() else {
                panic!("8-bit 4:4:4 dropped its planes");
            };
            assert!(picture.yuv444_high.is_none());
            assert!(picture.yuv422.is_none());
            (
                frame.y().iter().map(|sample| u16::from(*sample)).collect(),
                frame.u().iter().map(|sample| u16::from(*sample)).collect(),
                frame.v().iter().map(|sample| u16::from(*sample)).collect(),
            )
        }
        (Format::C422, _) => {
            let Some(frame) = picture.yuv422_high.as_ref() else {
                panic!("{depth}-bit 4:2:2 dropped its planes");
            };
            assert_eq!(frame.bit_depth(), depth as u8);
            assert!(picture.yuv422.is_none());
            (frame.y().to_vec(), frame.u().to_vec(), frame.v().to_vec())
        }
        (Format::C444, _) => {
            let Some(frame) = picture.yuv444_high.as_ref() else {
                panic!("{depth}-bit 4:4:4 dropped its planes");
            };
            assert_eq!(frame.bit_depth(), depth as u8);
            assert!(picture.yuv444.is_none());
            (frame.y().to_vec(), frame.u().to_vec(), frame.v().to_vec())
        }
    }
}

fn lengths(format: Format) -> (usize, usize) {
    let y = WIDTH * HEIGHT;
    let c = match format {
        Format::C422 => WIDTH / 2 * HEIGHT,
        Format::C444 => WIDTH * HEIGHT,
    };
    (y, c)
}

fn decode_one(annex: &[u8], format: Format, depth: u32) -> VideoOutput {
    let mut params = CodecParams::video(CodecId::H264, WIDTH as u32, HEIGHT as u32);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    let mut frames = decoder
        .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
        .unwrap_or_else(|err| panic!("{:?} {depth}: {err}", format_name(format)));
    frames.extend(decoder.drain().expect("drain"));
    assert_eq!(frames.len(), 1);
    frames.pop().expect("picture")
}

fn ffmpeg_raw(ffmpeg: &PathBuf, format: Format, depth: u32, annex: &[u8]) -> Option<Vec<u8>> {
    ffmpeg_file(ffmpeg, &pix_fmt(format, depth), annex).ok()
}

fn ffmpeg_file(ffmpeg: &PathBuf, pix: &str, annex: &[u8]) -> Result<Vec<u8>, String> {
    let n = TEMP.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "cutlass-h264-chroma-oracle-{}-{}-{}.h264",
        pix,
        std::process::id(),
        n
    ));
    std::fs::write(&path, annex).expect("annex");
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
        .args(["-an", "-f", "rawvideo", "-pix_fmt", pix, "-"])
        .stdout(Stdio::piped())
        .output()
        .expect("ffmpeg raw");
    let _ = std::fs::remove_file(&path);
    if !raw.status.success() {
        return Err(String::from_utf8_lossy(&raw.stderr).into_owned());
    }
    Ok(raw.stdout)
}

static TEMP: AtomicU64 = AtomicU64::new(0);

fn split_raw(format: Format, depth: u32, raw: &[u8]) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    split_sized(format, depth, WIDTH, HEIGHT, raw)
}

fn split_sized(
    format: Format,
    depth: u32,
    width: usize,
    height: usize,
    raw: &[u8],
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let y_len = width * height;
    let c_len = match format {
        Format::C422 => width / 2 * height,
        Format::C444 => width * height,
    };
    if depth == 8 {
        assert_eq!(raw.len(), y_len + c_len * 2);
        let y = raw[..y_len]
            .iter()
            .map(|sample| u16::from(*sample))
            .collect();
        let u = raw[y_len..y_len + c_len]
            .iter()
            .map(|sample| u16::from(*sample))
            .collect();
        let v = raw[y_len + c_len..]
            .iter()
            .map(|sample| u16::from(*sample))
            .collect();
        return (y, u, v);
    }
    let words = words(raw);
    assert_eq!(words.len(), y_len + c_len * 2);
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

fn raw_len(format: Format, depth: u32, width: usize, height: usize) -> usize {
    let y = width * height;
    let c = match format {
        Format::C422 => width / 2 * height,
        Format::C444 => width * height,
    };
    let samples = y + c * 2;
    if depth == 8 { samples } else { samples * 2 }
}

fn pix_fmt(format: Format, depth: u32) -> String {
    let family = match format {
        Format::C422 => "yuv422p",
        Format::C444 => "yuv444p",
    };
    if depth == 8 {
        family.to_string()
    } else {
        format!("{family}{depth}le")
    }
}

fn x264(ffmpeg: &PathBuf, profile: &str, pix: &str) -> Option<Vec<u8>> {
    let path = std::env::temp_dir().join(format!(
        "cutlass-h264-{profile}-{pix}-{}.h264",
        std::process::id()
    ));
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
            "-profile:v",
            profile,
            "-pix_fmt",
            pix,
            "-bf",
            "2",
            "-g",
            "8",
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
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let Ok(output) = Command::new(candidate).arg("-version").output() else {
            continue;
        };
        if output.status.success() {
            return Some(PathBuf::from(candidate));
        }
    }
    eprintln!("skipped: ffmpeg is not installed");
    None
}

fn format_name(format: Format) -> &'static str {
    match format {
        Format::C422 => "4:2:2",
        Format::C444 => "4:4:4",
    }
}

fn picture_name(picture: Picture) -> &'static str {
    match picture {
        Picture::Flat => "flat",
        Picture::LumaDc => "luma-dc",
        Picture::ChromaDc => "chroma-dc",
        Picture::Pcm => "pcm",
    }
}

fn annex(format: Format, depth: u32, picture: Picture) -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &sps(format, depth - 8));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &slice(format, picture, depth)));
    out
}

fn annex_chroma0() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(100, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(0);
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
    let mut out = annex_b_nal(3, 7, &w.bytes);
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &slice(Format::C422, Picture::Flat, 8)));
    out
}

fn sps(format: Format, minus8: u32) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(
        if matches!(format, Format::C422) {
            122
        } else {
            244
        },
        8,
    );
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(if matches!(format, Format::C422) { 2 } else { 3 });
    if matches!(format, Format::C444) {
        w.bit(false);
    }
    w.ue(minus8);
    w.ue(minus8);
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

fn slice(format: Format, picture: Picture, depth: u32) -> Vec<u8> {
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
        Picture::Flat => write_i16(&mut w, format, false, false),
        Picture::LumaDc => write_i16(&mut w, format, true, false),
        Picture::ChromaDc => write_i16(&mut w, format, false, true),
        Picture::Pcm => {
            w.ue(25);
            w.align_zeros();
            let count = if matches!(format, Format::C422) {
                512
            } else {
                768
            };
            for index in 0..count {
                w.bits(pcm_sample(index, depth), depth);
            }
        }
    }
    w.trailing();
    w.bytes
}

fn write_i16(w: &mut Writer, format: Format, luma_dc: bool, chroma_dc: bool) {
    if chroma_dc && matches!(format, Format::C422) {
        w.ue(7);
        w.ue(0);
    } else {
        w.ue(3);
        if matches!(format, Format::C422) {
            w.ue(0);
        }
    }
    w.se(0);
    if luma_dc {
        w.bits(0b0101, 4);
    } else {
        w.bit(true);
    }
    if matches!(format, Format::C444) {
        if chroma_dc {
            w.bits(0b0101, 4);
        } else {
            w.bit(true);
        }
        w.bit(true);
    } else if chroma_dc {
        w.bits(0b0101, 4);
        w.bit(true);
    }
}

fn pcm_planes(format: Format, depth: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let (y_len, c_len) = lengths(format);
    let y = (0..y_len)
        .map(|index| pcm_sample(index as u32, depth) as u16)
        .collect();
    let u = (y_len..y_len + c_len)
        .map(|index| pcm_sample(index as u32, depth) as u16)
        .collect();
    let v = (y_len + c_len..y_len + c_len * 2)
        .map(|index| pcm_sample(index as u32, depth) as u16)
        .collect();
    (y, u, v)
}

fn pcm_sample(index: u32, depth: u32) -> u32 {
    let mask = (1u32 << depth) - 1;
    (index.wrapping_mul(17).wrapping_add(3)) & mask
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
