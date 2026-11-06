//! The 32-bit MPEG audio frame header.
//!
//! ISO/IEC 11172-3:1993 clause 2.4.1.3 and 2.4.2.3; ISO/IEC 13818-3:1998
//! clause 2.4.2.3 for the lower sampling frequencies. MPEG-2.5 is the
//! Fraunhofer extension: version bits `00`, 8, 11.025 and 12 kHz, with the
//! MPEG-2 Layer III frame rules.
//!
//! `cutlass-mp3-container` frames the file with its own header reader. The
//! two agree on validity, rate, channel count, bitrate, and frame length for
//! every header; `tests/container.rs` checks that through the demuxer.

use cutlass_decode::CodecError;

/// MPEG audio version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Version {
    /// ISO/IEC 11172-3: 32, 44.1 and 48 kHz, two granules per frame.
    Mpeg1,
    /// ISO/IEC 13818-3: 16, 22.05 and 24 kHz, one granule per frame.
    Mpeg2,
    /// Fraunhofer MPEG-2.5: 8, 11.025 and 12 kHz, one granule per frame.
    Mpeg25,
}

impl Version {
    /// Lower sampling frequency syntax (one granule, 8-bit `main_data_begin`).
    #[must_use]
    pub fn lsf(self) -> bool {
        self != Self::Mpeg1
    }

    pub(crate) fn bits(self) -> u32 {
        match self {
            Self::Mpeg1 => 0b11,
            Self::Mpeg2 => 0b10,
            Self::Mpeg25 => 0b00,
        }
    }
}

/// Channel mode field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Stereo,
    JointStereo,
    DualChannel,
    Mono,
}

impl Mode {
    pub(crate) fn bits(self) -> u32 {
        match self {
            Self::Stereo => 0,
            Self::JointStereo => 1,
            Self::DualChannel => 2,
            Self::Mono => 3,
        }
    }
}

/// One parsed frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub version: Version,
    pub layer: u8,
    /// A 16-bit CRC follows the header (`protection_bit` is 0).
    pub protected: bool,
    pub bitrate_index: u8,
    /// Kilobits per second; zero is free format.
    pub bitrate_kbps: u32,
    /// Index into the 44.1/48/32 kHz family, before the version divides it.
    pub rate_index: u8,
    pub sample_rate: u32,
    pub padding: bool,
    pub private: bool,
    pub mode: Mode,
    /// Joint-stereo `mode_extension`: bit 1 is M/S, bit 0 is intensity.
    pub mode_extension: u8,
    pub copyright: bool,
    pub original: bool,
    pub emphasis: u8,
}

/// ISO/IEC 11172-3 Table 3-B.1 (Layer III row) and ISO/IEC 13818-3 Table
/// 2.4.2.3 (lower sampling frequencies), kilobits per second.
const BITRATES: [[u16; 15]; 2] = [
    [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
];

/// MPEG-1 sampling frequencies by index. MPEG-2 halves them, MPEG-2.5
/// quarters them.
const BASE_RATES: [u32; 3] = [44_100, 48_000, 32_000];

/// Largest Layer III frame this crate accepts, in bytes.
///
/// A legal MPEG-1 frame at 320 kbit/s and 32 kHz is 1441 bytes. Free format
/// is capped at the same 4096 bytes `cutlass-mp3-container` uses.
pub const MAX_FRAME_BYTES: usize = 4096;

impl FrameHeader {
    /// Parse four big-endian bytes.
    ///
    /// A bad sync, the reserved version, the reserved layer, bitrate index
    /// 15, sample-rate index 3, and emphasis `10` are
    /// [`CodecError::Malformed`]. Layer I and II parse; the decoder then
    /// refuses them.
    pub fn parse(bytes: [u8; 4]) -> Result<Self, CodecError> {
        let raw = u32::from_be_bytes(bytes);
        let bad = |at_bit: u64, what: &'static str| CodecError::Malformed { at_bit, what };
        if raw >> 21 != 0x7FF {
            return Err(bad(0, "frame sync"));
        }
        let version = match (raw >> 19) & 3 {
            0b11 => Version::Mpeg1,
            0b10 => Version::Mpeg2,
            0b00 => Version::Mpeg25,
            _ => return Err(bad(11, "version")),
        };
        let layer = match (raw >> 17) & 3 {
            0b11 => 1,
            0b10 => 2,
            0b01 => 3,
            _ => return Err(bad(13, "layer")),
        };
        let protected = (raw >> 16) & 1 == 0;
        let bitrate_index = ((raw >> 12) & 0xF) as u8;
        if bitrate_index == 15 {
            return Err(bad(16, "bitrate index"));
        }
        let rate_index = ((raw >> 10) & 3) as u8;
        if rate_index == 3 {
            return Err(bad(20, "sampling frequency"));
        }
        let shift = match version {
            Version::Mpeg1 => 0,
            Version::Mpeg2 => 1,
            Version::Mpeg25 => 2,
        };
        let sample_rate = BASE_RATES[usize::from(rate_index)] >> shift;
        let bitrate_kbps = if layer == 3 {
            u32::from(BITRATES[usize::from(version.lsf())][usize::from(bitrate_index)])
        } else {
            0
        };
        let mode = match (raw >> 6) & 3 {
            0 => Mode::Stereo,
            1 => Mode::JointStereo,
            2 => Mode::DualChannel,
            _ => Mode::Mono,
        };
        let emphasis = (raw & 3) as u8;
        if emphasis == 2 {
            return Err(bad(30, "emphasis"));
        }
        Ok(Self {
            version,
            layer,
            protected,
            bitrate_index,
            bitrate_kbps,
            rate_index,
            sample_rate,
            padding: (raw >> 9) & 1 == 1,
            private: (raw >> 8) & 1 == 1,
            mode,
            mode_extension: ((raw >> 4) & 3) as u8,
            copyright: (raw >> 3) & 1 == 1,
            original: (raw >> 2) & 1 == 1,
            emphasis,
        })
    }

    /// The header as four big-endian bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 4] {
        let raw = (0x7FFu32 << 21)
            | (self.version.bits() << 19)
            | (u32::from(4 - self.layer) << 17)
            | (u32::from(!self.protected) << 16)
            | (u32::from(self.bitrate_index) << 12)
            | (u32::from(self.rate_index) << 10)
            | (u32::from(self.padding) << 9)
            | (u32::from(self.private) << 8)
            | (self.mode.bits() << 6)
            | (u32::from(self.mode_extension & 3) << 4)
            | (u32::from(self.copyright) << 3)
            | (u32::from(self.original) << 2)
            | u32::from(self.emphasis & 3);
        raw.to_be_bytes()
    }

    #[must_use]
    pub fn channels(&self) -> usize {
        if self.mode == Mode::Mono { 1 } else { 2 }
    }

    /// Granules per frame: two for MPEG-1, one otherwise.
    #[must_use]
    pub fn granules(&self) -> usize {
        if self.version.lsf() { 1 } else { 2 }
    }

    /// PCM samples per channel in one Layer III frame.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.granules() * 576
    }

    /// Side-information bytes after the header and CRC.
    #[must_use]
    pub fn side_info_bytes(&self) -> usize {
        match (self.version.lsf(), self.channels()) {
            (false, 1) => 17,
            (false, _) => 32,
            (true, 1) => 9,
            (true, _) => 17,
        }
    }

    /// Index into the nine scalefactor-band tables: 44.1, 48, 32, 22.05,
    /// 24, 16, 11.025, 12, 8 kHz.
    #[must_use]
    pub fn rate_table(&self) -> usize {
        let family = match self.version {
            Version::Mpeg1 => 0,
            Version::Mpeg2 => 3,
            Version::Mpeg25 => 6,
        };
        family + usize::from(self.rate_index)
    }

    /// Bytes in this frame, header included. `None` for free format.
    ///
    /// MPEG-1 Layer III: `144000 * kbps / rate + padding`. MPEG-2 and
    /// MPEG-2.5: `72000 * kbps / rate + padding`.
    #[must_use]
    pub fn frame_bytes(&self) -> Option<usize> {
        if self.layer != 3 || self.bitrate_kbps == 0 {
            return None;
        }
        let numerator = if self.version.lsf() { 72_000 } else { 144_000 };
        let size = numerator * self.bitrate_kbps / self.sample_rate + u32::from(self.padding);
        usize::try_from(size).ok()
    }

    /// Two headers belong to one stream when version, layer, rate, and
    /// channel count match. Bitrate, padding, and joint-stereo modes may change.
    #[must_use]
    pub fn same_stream(&self, other: &Self) -> bool {
        self.version == other.version
            && self.layer == other.layer
            && self.sample_rate == other.sample_rate
            && self.channels() == other.channels()
    }
}

/// Version and frequency index for a Layer III sample rate.
#[must_use]
pub fn classify_rate(sample_rate: u32) -> Option<(Version, u8)> {
    for (index, base) in BASE_RATES.iter().copied().enumerate() {
        let index = u8::try_from(index).ok()?;
        if sample_rate == base {
            return Some((Version::Mpeg1, index));
        }
        if sample_rate == base / 2 {
            return Some((Version::Mpeg2, index));
        }
        if sample_rate == base / 4 {
            return Some((Version::Mpeg25, index));
        }
    }
    None
}

/// The Layer III bitrates of a version, index 1 through 14, in kbit/s.
#[must_use]
pub fn bitrates(version: Version) -> [u32; 14] {
    let row = &BITRATES[usize::from(version.lsf())];
    let mut out = [0u32; 14];
    for (slot, kbps) in out.iter_mut().zip(row[1..].iter()) {
        *slot = u32::from(*kbps);
    }
    out
}

/// Bitrate index of a Layer III rate, or `None` when the table lacks it.
#[must_use]
pub fn bitrate_index(version: Version, kbps: u32) -> Option<u8> {
    let row = &BITRATES[usize::from(version.lsf())];
    let position = row
        .iter()
        .skip(1)
        .position(|rate| u32::from(*rate) == kbps)?;
    u8::try_from(position + 1).ok()
}

#[cfg(test)]
mod tests {
    use super::{FrameHeader, Mode, Version, bitrate_index, classify_rate};

    fn header(version: Version, kbps: u32, rate: u32, padding: bool, mode: Mode) -> FrameHeader {
        let (rate_version, rate_index) = classify_rate(rate).expect("rate");
        assert_eq!(rate_version, version);
        FrameHeader {
            version,
            layer: 3,
            protected: false,
            bitrate_index: bitrate_index(version, kbps).expect("bitrate"),
            bitrate_kbps: kbps,
            rate_index,
            sample_rate: rate,
            padding,
            private: false,
            mode,
            mode_extension: 0,
            copyright: false,
            original: true,
            emphasis: 0,
        }
    }

    #[test]
    fn a_header_round_trips_through_its_bytes() {
        let cases = [
            (Version::Mpeg1, 128, 44_100, true, Mode::JointStereo),
            (Version::Mpeg1, 320, 32_000, false, Mode::Stereo),
            (Version::Mpeg2, 8, 24_000, false, Mode::Mono),
            (Version::Mpeg25, 64, 8_000, true, Mode::DualChannel),
        ];
        for (version, kbps, rate, padding, mode) in cases {
            let header = header(version, kbps, rate, padding, mode);
            let parsed = FrameHeader::parse(header.to_bytes()).expect("parse");
            assert_eq!(parsed, header);
        }
    }

    #[test]
    fn frame_bytes_follow_the_iso_formula() {
        let h = header(Version::Mpeg1, 128, 44_100, false, Mode::Stereo);
        assert_eq!(h.frame_bytes(), Some(417));
        assert_eq!(h.side_info_bytes(), 32);
        assert_eq!(h.samples(), 1152);
        let h = header(Version::Mpeg1, 320, 32_000, true, Mode::Stereo);
        assert_eq!(h.frame_bytes(), Some(1441));
        let h = header(Version::Mpeg2, 64, 22_050, false, Mode::Mono);
        assert_eq!(h.frame_bytes(), Some(208));
        assert_eq!(h.side_info_bytes(), 9);
        assert_eq!(h.samples(), 576);
        let h = header(Version::Mpeg25, 8, 8_000, false, Mode::Mono);
        assert_eq!(h.frame_bytes(), Some(72));
        assert_eq!(h.rate_table(), 8);
    }

    #[test]
    fn reserved_fields_are_malformed() {
        let good = header(Version::Mpeg1, 128, 44_100, false, Mode::Stereo).to_bytes();
        let raw = u32::from_be_bytes(good);
        for bad in [
            raw & !(1 << 21),             // sync
            (raw & !(3 << 19)) | 1 << 19, // reserved version
            raw & !(3 << 17),             // reserved layer
            raw | (0xF << 12),            // bitrate 15
            raw | (3 << 10),              // rate 3
            (raw & !3) | 2,               // emphasis 10
        ] {
            assert!(FrameHeader::parse(bad.to_be_bytes()).is_err(), "{bad:08x}");
        }
    }
}
