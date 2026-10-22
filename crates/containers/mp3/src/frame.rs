//! One MPEG audio frame header.
//!
//! ISO/IEC 11172-3 and 13818-3. MPEG-2.5 uses the MPEG-2 tables at the lower
//! rates. A free bitrate has no frame size, so this reader refuses it.

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Frame {
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
    pub(crate) samples: i64,
    pub(crate) len: usize,
    version: Version,
    layer: Layer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Version {
    Mpeg1,
    Mpeg2,
    Mpeg25,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layer {
    I,
    Ii,
    Iii,
}

pub(crate) fn header(data: &[u8], at: usize) -> Result<Frame, Error> {
    if data.len().saturating_sub(at) < 4 {
        return Err(Error::Truncated {
            at: u64::try_from(at).map_err(|_| Error::TooLarge)?,
            need: 4,
        });
    }
    let bytes = [data[at], data[at + 1], data[at + 2], data[at + 3]];
    if bytes[0] != 0xFF || bytes[1] & 0xE0 != 0xE0 {
        return Err(Error::Inconsistent("the frame header is not audio"));
    }
    let version = match (bytes[1] >> 3) & 0x03 {
        0 => Version::Mpeg25,
        1 => return Err(Error::Inconsistent("the frame header is not audio")),
        2 => Version::Mpeg2,
        _ => Version::Mpeg1,
    };
    let layer = match (bytes[1] >> 1) & 0x03 {
        1 => Layer::Iii,
        2 => Layer::Ii,
        3 => Layer::I,
        _ => return Err(Error::Inconsistent("the frame header is not audio")),
    };
    let bitrate_id = usize::from(bytes[2] >> 4);
    let bitrate = match bitrate_id {
        0 => return Err(Error::Inconsistent("a frame has no bitrate")),
        15 => return Err(Error::Inconsistent("the frame header is not audio")),
        index => u32::from(bitrate_kbps(version, layer)[index - 1]) * 1000,
    };
    let rate_id = usize::from((bytes[2] >> 2) & 0x03);
    if rate_id == 3 {
        return Err(Error::Inconsistent("the frame header is not audio"));
    }
    let sample_rate = sample_rate(version)[rate_id];
    let padding = u32::from((bytes[2] >> 1) & 0x01);
    let channels = if bytes[3] >> 6 == 0x03 { 1 } else { 2 };
    let samples = samples(version, layer);
    let len = frame_len(version, layer, bitrate, sample_rate, padding)?;
    Ok(Frame {
        sample_rate,
        channels,
        samples,
        len,
        version,
        layer,
    })
}

pub(crate) fn same_stream(first: &Frame, next: &Frame) -> bool {
    first.version == next.version
        && first.layer == next.layer
        && first.sample_rate == next.sample_rate
        && first.channels == next.channels
}

fn samples(version: Version, layer: Layer) -> i64 {
    match layer {
        Layer::I => 384,
        Layer::Ii => 1152,
        Layer::Iii => match version {
            Version::Mpeg1 => 1152,
            Version::Mpeg2 | Version::Mpeg25 => 576,
        },
    }
}

fn frame_len(
    version: Version,
    layer: Layer,
    bitrate: u32,
    sample_rate: u32,
    padding: u32,
) -> Result<usize, Error> {
    let (coefficient, slot): (u64, u64) = match (version, layer) {
        (_, Layer::I) => (12, 4),
        (Version::Mpeg1, _) | (_, Layer::Ii) => (144, 1),
        (_, Layer::Iii) => (72, 1),
    };
    let base = coefficient * u64::from(bitrate) / u64::from(sample_rate);
    let len = base
        .checked_mul(slot)
        .and_then(|len| len.checked_add(u64::from(padding) * slot))
        .ok_or(Error::TooLarge)?;
    let len = usize::try_from(len).map_err(|_| Error::TooLarge)?;
    if len < 4 {
        return Err(Error::Inconsistent("a frame is too small"));
    }
    Ok(len)
}

fn sample_rate(version: Version) -> &'static [u32; 3] {
    match version {
        Version::Mpeg1 => &[44_100, 48_000, 32_000],
        Version::Mpeg2 => &[22_050, 24_000, 16_000],
        Version::Mpeg25 => &[11_025, 12_000, 8_000],
    }
}

fn bitrate_kbps(version: Version, layer: Layer) -> &'static [u16; 14] {
    match (version, layer) {
        (Version::Mpeg1, Layer::I) => &[
            32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
        ],
        (Version::Mpeg1, Layer::Ii) => &[
            32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
        ],
        (Version::Mpeg1, Layer::Iii) => &[
            32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
        ],
        (_, Layer::I) => &[
            32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
        ],
        _ => &[8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_known_mpeg1_layer3_frame_is_417_bytes() {
        let frame = header(&[0xFF, 0xFB, 0x90, 0x00], 0).expect("header");
        assert_eq!(frame.sample_rate, 44_100);
        assert_eq!(frame.channels, 2);
        assert_eq!(frame.samples, 1152);
        assert_eq!(frame.len, 417);
    }

    #[test]
    fn a_known_mpeg2_layer3_frame_is_208_bytes() {
        let frame = header(&[0xFF, 0xF3, 0x80, 0x00], 0).expect("header");
        assert_eq!(frame.sample_rate, 22_050);
        assert_eq!(frame.samples, 576);
        assert_eq!(frame.len, 208);
    }

    #[test]
    fn a_known_mpeg2_layer2_frame_keeps_1152_samples() {
        let frame = header(&[0xFF, 0xF5, 0x80, 0x00], 0).expect("header");
        assert_eq!(frame.samples, 1152);
        assert_eq!(frame.len, 417);
    }

    #[test]
    fn a_known_layer1_frame_is_136_bytes() {
        let frame = header(&[0xFF, 0xFF, 0x40, 0x00], 0).expect("header");
        assert_eq!(frame.samples, 384);
        assert_eq!(frame.len, 136);
    }
}
