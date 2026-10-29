//! MPEG audio frame length inside a WAV data chunk.
//!
//! The header layout is ISO/IEC 11172-3 and ISO/IEC 13818-3. This module
//! does not decode samples. Free format (bitrate index 0) is refused.

use crate::limits::MAX_MPEG_RESYNC;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameHead {
    pub len: u64,
    pub samples: u64,
}

pub(crate) fn parse_header(header: &[u8; 4]) -> Option<FrameHead> {
    let bits = u32::from_be_bytes(*header);
    if bits >> 21 != 0x7FF {
        return None;
    }
    let version = (bits >> 19) & 0b11;
    let layer = (bits >> 17) & 0b11;
    if version == 0b01 || layer == 0 {
        return None;
    }
    let bitrate_index = (bits >> 12) & 0x0F;
    let rate_index = (bits >> 10) & 0b11;
    let padding = (bits >> 9) & 1;
    if bitrate_index == 0 || bitrate_index == 0x0F || rate_index == 0b11 {
        return None;
    }
    let sample_rate = sample_rate(version, rate_index)?;
    let bitrate = bitrate_kbps(version, layer, bitrate_index)?;
    let bitrate_bps = u64::from(bitrate) * 1_000;
    let rate = u64::from(sample_rate);
    let len = frame_bytes(version, layer, bitrate_bps, rate, padding)?;
    let samples = samples_per_frame(version, layer)?;
    Some(FrameHead { len, samples })
}

/// Finds the next header at `offset` or within [`MAX_MPEG_RESYNC`] bytes after it.
pub(crate) fn find_frame(data: &[u8], offset: usize) -> Option<(usize, FrameHead)> {
    let end = offset.saturating_add(usize::try_from(MAX_MPEG_RESYNC).unwrap_or(usize::MAX));
    let last = end.min(data.len());
    let mut at = offset;
    while at + 4 <= last && at + 4 <= data.len() {
        let mut header = [0; 4];
        header.copy_from_slice(data.get(at..at + 4)?);
        if let Some(frame) = parse_header(&header) {
            let len = usize::try_from(frame.len).ok()?;
            if len >= 4 && at + len <= data.len() {
                return Some((at, frame));
            }
        }
        at += 1;
    }
    None
}

fn frame_bytes(version: u32, layer: u32, bitrate: u64, rate: u64, padding: u32) -> Option<u64> {
    if rate == 0 {
        return None;
    }
    let pad = u64::from(padding);
    let len = if layer == 0b11 {
        (12 * bitrate / rate + pad).checked_mul(4)?
    } else if version == 0b11 {
        144 * bitrate / rate + pad
    } else {
        72 * bitrate / rate + pad
    };
    if len < 4 { None } else { Some(len) }
}

fn samples_per_frame(version: u32, layer: u32) -> Option<u64> {
    match layer {
        0b11 => Some(384),
        0b10 => Some(1152),
        0b01 if version == 0b11 => Some(1152),
        0b01 => Some(576),
        _ => None,
    }
}

fn sample_rate(version: u32, index: u32) -> Option<u32> {
    let base = match index {
        0 => 44_100,
        1 => 48_000,
        2 => 32_000,
        _ => return None,
    };
    match version {
        0b11 => Some(base),
        0b10 => Some(base / 2),
        0b00 => Some(base / 4),
        _ => None,
    }
}

fn bitrate_kbps(version: u32, layer: u32, index: u32) -> Option<u32> {
    let index = usize::try_from(index).ok()?;
    let table: &[u32] = if version == 0b11 {
        match layer {
            0b11 => &[
                0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
            ],
            0b10 => &[
                0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
            ],
            0b01 => &[
                0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
            ],
            _ => return None,
        }
    } else if layer == 0b11 {
        &[
            0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
        ]
    } else {
        &[0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
    };
    let kbps = *table.get(index)?;
    if kbps == 0 { None } else { Some(kbps) }
}

#[cfg(test)]
mod tests {
    use super::parse_header;

    #[test]
    fn a_128k_mpeg1_layer3_frame_is_417_bytes() {
        let mut bits = 0u32;
        bits |= 0x7FF << 21;
        bits |= 0b11 << 19;
        bits |= 0b01 << 17;
        bits |= 1 << 16;
        bits |= 9 << 12;
        let frame = parse_header(&bits.to_be_bytes()).expect("header");
        assert_eq!(frame.len, 417);
        assert_eq!(frame.samples, 1152);
    }
}
