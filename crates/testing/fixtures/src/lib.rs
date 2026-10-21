//! Loads the shared media fixtures for codec tests.
//!
//! Fixtures live in `fixtures/` at the repository root.
//! `scripts/make-video-fixtures.py` generates them, and `fixtures/README.md`
//! records where each one came from. A missing or malformed fixture panics,
//! since that is a broken checkout, not a test result.

#![deny(unsafe_code)]

use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use cutlass_core::{ColorRange, Packet, PixelFormat, Timestamp, VideoCodec, VideoStreamConfig};

/// A compressed stream plus what a correct decoder outputs for it.
#[derive(Debug, Clone)]
pub struct VideoFixture {
    pub name: String,
    pub config: VideoStreamConfig,
    /// In decode order.
    pub packets: Vec<Packet>,
    pub pixel_format: PixelFormat,
    pub color_range: ColorRange,
    /// The reference decoder's output, in presentation order.
    pub frames: Vec<ExpectedFrame>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedFrame {
    pub pts: Timestamp,
    /// [`Crc32`] of the visible bytes of each plane, row by row, planes in
    /// order. See [`visible_row_bytes`].
    pub crc32: u32,
}

/// Loads `fixtures/video/<name>`.
pub fn video(name: &str) -> VideoFixture {
    let dir = fixtures_dir().join("video").join(name);
    let manifest = read_to_string(&dir.join("stream.txt"));
    let config_record = read(&dir.join("config.bin"));
    let packet_data = read(&dir.join("packets.bin"));

    let mut fields = Fields::default();
    let mut packets = Vec::new();
    let mut frames = Vec::new();
    let mut offset = 0;
    for (number, line) in manifest.lines().enumerate() {
        if line.starts_with('#') {
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        let context = || format!("{name}/stream.txt line {}", number + 1);
        match words.as_slice() {
            [] => {}
            ["codec", codec] => fields.codec = Some(parse_codec(codec, &context)),
            ["width", value] => fields.width = Some(parse(value, &context)),
            ["height", value] => fields.height = Some(parse(value, &context)),
            ["timescale", value] => fields.timescale = Some(parse(value, &context)),
            ["pixel_format", value] => {
                fields.pixel_format = Some(parse_pixel_format(value, &context))
            }
            ["color_range", value] => fields.color_range = Some(parse_color_range(value, &context)),
            ["packet", pts, dts, duration, flag, size] => {
                let timescale = fields.timescale(&context);
                let size: usize = parse(size, &context);
                let data = packet_data
                    .get(offset..offset + size)
                    .unwrap_or_else(|| panic!("{}: packet runs past packets.bin", context()))
                    .to_vec();
                offset += size;
                packets.push(Packet {
                    data,
                    pts: Timestamp::new(parse(pts, &context), timescale),
                    dts: parse_optional(dts, &context).map(|dts| Timestamp::new(dts, timescale)),
                    duration: parse_optional(duration, &context)
                        .map(|duration| Timestamp::new(duration, timescale)),
                    keyframe: *flag == "K",
                });
            }
            ["frame", pts, crc32] => {
                let timescale = fields.timescale(&context);
                frames.push(ExpectedFrame {
                    pts: Timestamp::new(parse(pts, &context), timescale),
                    crc32: u32::from_str_radix(crc32, 16)
                        .unwrap_or_else(|_| panic!("{}: bad crc32 {crc32}", context())),
                });
            }
            _ => panic!("{}: unrecognized line {line:?}", context()),
        }
    }
    assert_eq!(
        offset,
        packet_data.len(),
        "{name}: packets.bin has bytes no packet uses"
    );

    let missing = |field: &str| -> ! { panic!("{name}/stream.txt has no {field}") };
    VideoFixture {
        name: name.to_owned(),
        config: VideoStreamConfig {
            codec: fields.codec.unwrap_or_else(|| missing("codec")),
            width: fields.width.unwrap_or_else(|| missing("width")),
            height: fields.height.unwrap_or_else(|| missing("height")),
            config_record,
        },
        packets,
        pixel_format: fields
            .pixel_format
            .unwrap_or_else(|| missing("pixel_format")),
        color_range: fields.color_range.unwrap_or_else(|| missing("color_range")),
        frames,
    }
}

/// Bytes of picture data in one row of a plane, without row padding. `width`
/// is the plane width in plane pixels.
pub fn visible_row_bytes(format: PixelFormat, plane: usize, width: usize) -> usize {
    let bytes_per_sample = match format {
        PixelFormat::Nv12 => 1,
        PixelFormat::P010 => 2,
    };
    let samples_per_pixel = if plane == 0 { 1 } else { 2 };
    width * samples_per_pixel * bytes_per_sample
}

/// CRC-32 with the zlib polynomial, so it matches Python's `zlib.crc32`.
#[derive(Debug, Clone, Copy)]
pub struct Crc32(u32);

impl Crc32 {
    pub fn new() -> Self {
        Self(!0)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u32::from(byte);
            for _ in 0..8 {
                let mask = (self.0 & 1).wrapping_neg();
                self.0 = (self.0 >> 1) ^ (0xedb8_8320 & mask);
            }
        }
    }

    pub fn finish(self) -> u32 {
        !self.0
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
struct Fields {
    codec: Option<VideoCodec>,
    width: Option<u32>,
    height: Option<u32>,
    timescale: Option<NonZeroU32>,
    pixel_format: Option<PixelFormat>,
    color_range: Option<ColorRange>,
}

impl Fields {
    fn timescale(&self, context: &dyn Fn() -> String) -> NonZeroU32 {
        self.timescale.unwrap_or_else(|| {
            panic!(
                "{}: timescale must come before packets and frames",
                context()
            )
        })
    }
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures")
}

fn read(path: &Path) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

fn read_to_string(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

fn parse<T: std::str::FromStr>(value: &str, context: &dyn Fn() -> String) -> T {
    value
        .parse()
        .unwrap_or_else(|_| panic!("{}: can't parse {value:?}", context()))
}

fn parse_optional(value: &str, context: &dyn Fn() -> String) -> Option<i64> {
    (value != "-").then(|| parse(value, context))
}

fn parse_codec(value: &str, context: &dyn Fn() -> String) -> VideoCodec {
    match value {
        "h264" => VideoCodec::H264,
        "hevc" => VideoCodec::Hevc,
        "vp8" => VideoCodec::Vp8,
        "vp9" => VideoCodec::Vp9,
        "av1" => VideoCodec::Av1,
        "mpeg2" => VideoCodec::Mpeg2,
        "mpeg4" => VideoCodec::Mpeg4Part2,
        "mjpeg" => VideoCodec::Mjpeg,
        "prores" => VideoCodec::ProRes,
        _ => panic!("{}: unknown codec {value:?}", context()),
    }
}

fn parse_pixel_format(value: &str, context: &dyn Fn() -> String) -> PixelFormat {
    match value {
        "nv12" => PixelFormat::Nv12,
        "p010" => PixelFormat::P010,
        _ => panic!("{}: unknown pixel format {value:?}", context()),
    }
}

fn parse_color_range(value: &str, context: &dyn Fn() -> String) -> ColorRange {
    match value {
        "limited" => ColorRange::Limited,
        "full" => ColorRange::Full,
        _ => panic!("{}: unknown color range {value:?}", context()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_zlib() {
        let mut crc = Crc32::new();
        crc.update(b"123456789");
        assert_eq!(crc.finish(), 0xcbf4_3926);
    }

    #[test]
    fn crc32_of_split_input_matches_whole() {
        let mut split = Crc32::new();
        split.update(b"1234");
        split.update(b"56789");
        assert_eq!(split.finish(), 0xcbf4_3926);
    }

    #[test]
    fn p010_chroma_rows_hold_two_samples_of_two_bytes_per_pixel() {
        assert_eq!(visible_row_bytes(PixelFormat::P010, 1, 64), 256);
    }
}
