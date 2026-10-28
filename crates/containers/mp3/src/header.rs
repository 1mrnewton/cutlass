//! MPEG audio frame header.
//!
//! Layout and the Layer III byte count follow ISO/IEC 11172-3:1993 clause
//! 2.4.1.3 and ISO/IEC 13818-3:1998 clause 2.4.2.3. MPEG-2.5 sample rates are
//! the Fraunhofer extension: 8 kHz, 11.025 kHz, and 12 kHz use the MPEG-2
//! Layer III frame rules with the base rate divided by four.
//!
//! This module does not decode side information or samples.

use crate::limits::MAX_FRAME_BYTES;

/// MPEG audio version bits in the frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpegVersion {
    /// Bits `11`. ISO/IEC 11172-3. 32, 44.1, and 48 kHz.
    Mpeg1,
    /// Bits `10`. ISO/IEC 13818-3. 16, 22.05, and 24 kHz.
    Mpeg2,
    /// Bits `00`. Fraunhofer MPEG-2.5. 8, 11.025, and 12 kHz.
    Mpeg25,
}

/// A parsed 32-bit MPEG audio header.
///
/// `frame_bytes` is `None` for free format (bitrate index 0). ISO/IEC 11172-3
/// defines that case as a fixed bitrate that is not in the table, so the
/// stride is the distance to the next sync.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: MpegVersion,
    pub layer: u8,
    /// True when a 16-bit CRC follows the header.
    pub crc: bool,
    /// Kilobits per second. Zero is free format.
    pub bitrate_kbps: u32,
    pub sample_rate: u32,
    /// One when the padding slot is present.
    pub padding: u32,
    pub channels: u16,
    pub samples_per_frame: u32,
    pub frame_bytes: Option<u32>,
    pub raw: u32,
}

/// ISO/IEC 11172-3 Table of bitrates, in kbit/s, indexed by `[lsf][layer - 1][index]`.
///
/// Index 0 is free format. The MPEG-2 and MPEG-2.5 rows are the lower
/// sampling-frequency table from ISO/IEC 13818-3.
const BITRATE_KBPS: [[[u16; 15]; 3]; 2] = [
    [
        [
            0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
        ],
        [
            0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
        ],
        [
            0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
        ],
    ],
    [
        [
            0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
        ],
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    ],
];

const SAMPLE_RATES: [u32; 3] = [44_100, 48_000, 32_000];

/// Parse one big-endian MPEG audio header.
///
/// Returns `None` when the sync, version, layer, bitrate index, or sample-rate
/// index is reserved or inconsistent.
#[must_use]
pub fn parse_header(raw: u32) -> Option<Header> {
    // Eleven sync bits. Bit 20 is a version bit, including for MPEG-2.5.
    if raw & 0xFFE0_0000 != 0xFFE0_0000 {
        return None;
    }
    let version = match (raw >> 19) & 0b11 {
        0b11 => MpegVersion::Mpeg1,
        0b10 => MpegVersion::Mpeg2,
        0b00 => MpegVersion::Mpeg25,
        _ => return None,
    };
    let layer_bits = (raw >> 17) & 0b11;
    let layer = match layer_bits {
        0b11 => 1,
        0b10 => 2,
        0b01 => 3,
        _ => return None,
    };
    let bitrate_index = ((raw >> 12) & 0xF) as usize;
    if bitrate_index == 0xF {
        return None;
    }
    let rate_index = ((raw >> 10) & 0b11) as usize;
    if rate_index == 0b11 {
        return None;
    }
    let lsf = version != MpegVersion::Mpeg1;
    let bitrate_kbps = u32::from(BITRATE_KBPS[usize::from(lsf)][layer - 1][bitrate_index]);
    let shift = match version {
        MpegVersion::Mpeg1 => 0,
        MpegVersion::Mpeg2 => 1,
        MpegVersion::Mpeg25 => 2,
    };
    let sample_rate = SAMPLE_RATES[rate_index] >> shift;
    let padding = (raw >> 9) & 1;
    let channels = if (raw >> 6) & 0b11 == 0b11 { 1 } else { 2 };
    let crc = (raw >> 16) & 1 == 0;
    let samples_per_frame = samples_per_frame(version, layer);
    let frame_bytes = if bitrate_kbps == 0 {
        None
    } else {
        Some(frame_bytes_for(
            layer,
            bitrate_kbps,
            sample_rate,
            padding,
            lsf,
        )?)
    };
    Some(Header {
        version,
        layer: u8::try_from(layer).ok()?,
        crc,
        bitrate_kbps,
        sample_rate,
        padding,
        channels,
        samples_per_frame,
        frame_bytes,
        raw,
    })
}

/// Layer III side-information length, in bytes, excluding the 4-byte header
/// and excluding a CRC.
///
/// ISO/IEC 11172-3: 32 bytes stereo, 17 mono. ISO/IEC 13818-3: 17 stereo, 9 mono.
/// MPEG-2.5 uses the MPEG-2 lengths.
#[must_use]
pub fn side_info_len(header: &Header) -> usize {
    let mono = header.channels == 1;
    match header.version {
        MpegVersion::Mpeg1 => {
            if mono {
                17
            } else {
                32
            }
        }
        MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => {
            if mono {
                9
            } else {
                17
            }
        }
    }
}

/// Two headers belong to one elementary stream when version, layer, rate, and
/// channel count match. Bitrate and padding may change.
#[must_use]
pub fn compatible(a: &Header, b: &Header) -> bool {
    a.version == b.version
        && a.layer == b.layer
        && a.sample_rate == b.sample_rate
        && a.channels == b.channels
}

/// Map a sample rate to the header's version and frequency index.
#[must_use]
pub fn classify_rate(sample_rate: u32) -> Option<(MpegVersion, u8)> {
    for (index, base) in SAMPLE_RATES.iter().copied().enumerate() {
        let index = u8::try_from(index).ok()?;
        if sample_rate == base {
            return Some((MpegVersion::Mpeg1, index));
        }
        if sample_rate == base / 2 {
            return Some((MpegVersion::Mpeg2, index));
        }
        if sample_rate == base / 4 {
            return Some((MpegVersion::Mpeg25, index));
        }
    }
    None
}

/// Bitrate index for a Layer III kilobit rate. `None` when it is not in the table.
#[must_use]
pub fn layer3_bitrate_index(version: MpegVersion, bitrate_kbps: u32) -> Option<u8> {
    let lsf = usize::from(version != MpegVersion::Mpeg1);
    let row = &BITRATE_KBPS[lsf][2];
    let kbps = u16::try_from(bitrate_kbps).ok()?;
    let index = row.iter().position(|rate| *rate == kbps)?;
    u8::try_from(index).ok()
}

/// Build a Layer III header and its frame length.
///
/// `channels` is 1 or 2. The protection bit is clear (no CRC). Channel mode
/// is mono or stereo.
pub fn encode_layer3(
    version: MpegVersion,
    bitrate_kbps: u32,
    sample_rate: u32,
    padding: bool,
    channels: u16,
) -> Option<(u32, u32)> {
    let (rate_version, rate_index) = classify_rate(sample_rate)?;
    if rate_version != version || !(channels == 1 || channels == 2) {
        return None;
    }
    let bitrate_index = layer3_bitrate_index(version, bitrate_kbps)?;
    if bitrate_index == 0 {
        return None;
    }
    let version_bits = match version {
        MpegVersion::Mpeg1 => 0b11,
        MpegVersion::Mpeg2 => 0b10,
        MpegVersion::Mpeg25 => 0b00,
    };
    let mode = if channels == 1 { 0b11 } else { 0b00 };
    let raw = (0x7FF << 21)
        | (version_bits << 19)
        | (0b01 << 17)
        | (1 << 16)
        | (u32::from(bitrate_index) << 12)
        | (u32::from(rate_index) << 10)
        | (u32::from(padding) << 9)
        | (mode << 6);
    let header = parse_header(raw)?;
    let size = header.frame_bytes?;
    Some((raw, size))
}

fn samples_per_frame(version: MpegVersion, layer: usize) -> u32 {
    match layer {
        1 => 384,
        2 => 1152,
        _ => {
            if version == MpegVersion::Mpeg1 {
                1152
            } else {
                576
            }
        }
    }
}

/// Byte length of one frame, including the four-byte header.
///
/// Layer I: `(12 * bitrate / sample_rate + padding) * 4`.
/// Layer II and MPEG-1 Layer III: `144 * bitrate / sample_rate + padding`.
/// MPEG-2 and MPEG-2.5 Layer III: `72 * bitrate / sample_rate + padding`.
fn frame_bytes_for(
    layer: usize,
    bitrate_kbps: u32,
    sample_rate: u32,
    padding: u32,
    lsf: bool,
) -> Option<u32> {
    let rate = u64::from(sample_rate);
    let kbps = u64::from(bitrate_kbps);
    let size = match layer {
        1 => ((kbps * 12_000) / rate + u64::from(padding)) * 4,
        2 => (kbps * 144_000) / rate + u64::from(padding),
        _ => (kbps * 144_000) / (rate << u32::from(lsf)) + u64::from(padding),
    };
    let size = u32::try_from(size).ok()?;
    if !(4..=MAX_FRAME_BYTES).contains(&size) {
        return None;
    }
    Some(size)
}

/// One Layer III frame with the frame index stored at bytes 5 and 6.
#[cfg(test)]
#[must_use]
pub(crate) fn synthetic_frame(
    version: MpegVersion,
    bitrate_kbps: u32,
    sample_rate: u32,
    channels: u16,
    index: u16,
) -> Vec<u8> {
    let (raw, size) =
        encode_layer3(version, bitrate_kbps, sample_rate, false, channels).expect("synthetic");
    let mut frame = vec![0u8; size as usize];
    frame[..4].copy_from_slice(&raw.to_be_bytes());
    frame[5] = u8::try_from(index >> 8).unwrap_or(0);
    frame[6] = u8::try_from(index & 0x00FF).unwrap_or(0);
    frame
}

#[cfg(test)]
pub(crate) fn frame_index(frame: &[u8]) -> u16 {
    (u16::from(frame[5]) << 8) | u16::from(frame[6])
}

#[cfg(test)]
mod tests {
    use super::{MpegVersion, encode_layer3, parse_header, side_info_len};

    #[test]
    fn mpeg1_128k_44100_is_417_bytes_without_padding() {
        let (raw, size) = encode_layer3(MpegVersion::Mpeg1, 128, 44_100, false, 2).expect("hdr");
        assert_eq!(size, 417);
        let header = parse_header(raw).expect("parse");
        assert_eq!(header.frame_bytes, Some(417));
        assert_eq!(header.sample_rate, 44_100);
        assert_eq!(header.channels, 2);
        assert_eq!(header.samples_per_frame, 1152);
        assert_eq!(header.layer, 3);
        assert!(!header.crc);
        assert_eq!(side_info_len(&header), 32);
    }

    #[test]
    fn padding_adds_one_byte_and_48000_divides_evenly() {
        let (_, padded) = encode_layer3(MpegVersion::Mpeg1, 128, 44_100, true, 2).expect("pad");
        assert_eq!(padded, 418);
        let (_, even) = encode_layer3(MpegVersion::Mpeg1, 128, 48_000, false, 2).expect("48k");
        assert_eq!(even, 384);
    }

    #[test]
    fn mpeg2_and_mpeg25_use_the_shorter_layer3_frame() {
        let (_, mpeg2) = encode_layer3(MpegVersion::Mpeg2, 64, 22_050, false, 1).expect("m2");
        // 64 * 144000 / (22050 * 2) = 208.
        assert_eq!(mpeg2, 208);
        let header = parse_header(
            encode_layer3(MpegVersion::Mpeg2, 64, 22_050, false, 1)
                .expect("m2")
                .0,
        )
        .expect("parse");
        assert_eq!(header.samples_per_frame, 576);
        assert_eq!(side_info_len(&header), 9);

        let (_, mpeg25) = encode_layer3(MpegVersion::Mpeg25, 32, 11_025, false, 2).expect("m25");
        // 32 * 144000 / (11025 * 2) = 208.
        assert_eq!(mpeg25, 208);
        let header = parse_header(
            encode_layer3(MpegVersion::Mpeg25, 32, 11_025, false, 2)
                .expect("m25")
                .0,
        )
        .expect("parse");
        assert_eq!(header.sample_rate, 11_025);
        assert_eq!(header.samples_per_frame, 576);
    }

    #[test]
    fn reserved_version_bitrate_and_rate_are_not_headers() {
        assert!(parse_header(0).is_none());
        // Sync, reserved version bits `01`, Layer III, 128 kbps, 44.1 kHz.
        assert!(parse_header(0xFFEB_9000).is_none());
        // Sync, MPEG-1, layer III, bitrate index 15, rate index 0.
        assert!(parse_header(0xFFFB_F000).is_none());
        // Sync, MPEG-1, layer III, bitrate 128, sample-rate index 3.
        let (raw, _) = encode_layer3(MpegVersion::Mpeg1, 128, 44_100, false, 2).expect("hdr");
        let bad_rate = (raw & !0x0000_0C00) | 0x0000_0C00;
        assert!(parse_header(bad_rate).is_none());
    }

    #[test]
    fn free_format_has_no_size_in_the_header() {
        let (raw, _) = encode_layer3(MpegVersion::Mpeg1, 128, 44_100, false, 2).expect("hdr");
        let free = raw & !0x0000_F000;
        let header = parse_header(free).expect("free");
        assert_eq!(header.bitrate_kbps, 0);
        assert!(header.frame_bytes.is_none());
        assert_eq!(header.layer, 3);
    }

    #[test]
    fn frame_length_matches_the_iso_formula() {
        let cases = [
            (
                MpegVersion::Mpeg1,
                320_u32,
                32_000_u32,
                false,
                2_u16,
                1440_u32,
            ),
            (MpegVersion::Mpeg1, 32, 48_000, false, 1, 96),
            (MpegVersion::Mpeg2, 8, 24_000, false, 2, 24),
            (MpegVersion::Mpeg2, 160, 16_000, true, 1, 721),
        ];
        for (version, kbps, rate, padding, channels, expect) in cases {
            let (_, size) = encode_layer3(version, kbps, rate, padding, channels).expect("enc");
            let lsf = version != MpegVersion::Mpeg1;
            let denom = u64::from(rate) << u32::from(lsf);
            let formula =
                u32::try_from(u64::from(kbps) * 144_000 / denom).expect("fit") + u32::from(padding);
            assert_eq!(size, expect);
            assert_eq!(size, formula);
        }
    }
}
