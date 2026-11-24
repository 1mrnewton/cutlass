//! Supplemental enhancement information (clause 7.3.5 and Annex D): the
//! messages that describe how a picture is shown, not how it decodes.
//!
//! Read here: picture timing (1, for `pic_struct`), user data registered
//! by Recommendation ITU-T T.35 (4, HDR10+ among others), mastering
//! display colour volume (137), content light level (144) and alternative
//! transfer characteristics (147). Every other message is skipped.
//!
//! No message is needed to decode a picture, so one that is cut short or
//! runs past its NAL unit is dropped, with the messages after it, and is
//! not an error.

use cutlass_decode::{ContentLight, ItuT35, MasteringDisplay};

use crate::ps::{Sps, transfer_of};

const PIC_TIMING: u32 = 1;
const T35: u32 = 4;
const MASTERING_DISPLAY: u32 = 137;
const CONTENT_LIGHT: u32 = 144;
const ALTERNATIVE_TRANSFER: u32 = 147;

/// Most T.35 messages kept for one picture.
pub const MAX_T35: usize = 32;

/// `pic_struct`, `source_scan_type` and `duplicate_flag` of a picture
/// timing SEI message (clause D.3.3, Table D.2): whether the picture is a
/// frame or a field, which field, and how many times it is shown.
///
/// `pic_struct` 0 is a progressive frame; 1 and 2 a top and a bottom
/// field; 3 and 4 a frame shown top or bottom field first; 5 and 6 those
/// with the first field repeated; 7 and 8 a frame shown twice and three
/// times; 9 to 12 a field paired with the field before or after it.
/// `source_scan_type` 0 is interlaced, 1 progressive, 2 unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PicTiming {
    pub pic_struct: u8,
    pub source_scan_type: u8,
    pub duplicate: bool,
}

/// The high dynamic range description of a coded video sequence. Each
/// part lasts from the message that carries it to the end of the sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hdr {
    pub mastering: Option<MasteringDisplay>,
    pub light: Option<ContentLight>,
    /// `preferred_transfer_characteristics`, when it names a transfer.
    pub transfer: Option<u8>,
}

impl Hdr {
    /// Take what the messages of one access unit carry.
    pub fn update(&mut self, sei: &Sei) {
        self.mastering = sei.hdr.mastering.or(self.mastering);
        self.light = sei.hdr.light.or(self.light);
        self.transfer = sei.hdr.transfer.or(self.transfer);
    }

    /// These values, and `other`'s where there is none.
    pub fn or(self, other: Self) -> Self {
        Self {
            mastering: self.mastering.or(other.mastering),
            light: self.light.or(other.light),
            transfer: self.transfer.or(other.transfer),
        }
    }
}

/// What the SEI NAL units before the slices of one picture carry.
#[derive(Clone, Debug, Default)]
pub struct Sei {
    pub hdr: Hdr,
    /// In stream order.
    pub t35: Vec<ItuT35>,
    /// The first byte of the picture timing message: its meaning depends
    /// on the SPS the picture uses, which the slice header names later.
    timing: Option<u8>,
}

impl Sei {
    /// Read the messages of one SEI NAL unit. A suffix unit (after the
    /// slices) may carry T.35 data; the other messages read here are
    /// prefix messages only.
    pub fn read(&mut self, rbsp: &[u8], suffix: bool) {
        let mut at = 0;
        // More data than the stop bit alone.
        while rbsp.len() > at + 1 || rbsp.get(at).is_some_and(|&b| b != 0x80) {
            let Some(kind) = ff_coded(rbsp, &mut at) else {
                return;
            };
            let Some(size) = ff_coded(rbsp, &mut at) else {
                return;
            };
            let end = at.saturating_add(size as usize);
            let Some(payload) = rbsp.get(at..end) else {
                return;
            };
            at = end;
            self.message(kind, payload, suffix);
        }
    }

    fn message(&mut self, kind: u32, payload: &[u8], suffix: bool) {
        match kind {
            T35 => {
                if self.t35.len() < MAX_T35
                    && let Some(message) = t35(payload)
                {
                    self.t35.push(message);
                }
            }
            _ if suffix => {}
            PIC_TIMING => self.timing = payload.first().copied().or(self.timing),
            MASTERING_DISPLAY => self.hdr.mastering = mastering(payload).or(self.hdr.mastering),
            CONTENT_LIGHT => {
                if let Some(&[a, b, c, d]) = payload.first_chunk::<4>() {
                    self.hdr.light = Some(ContentLight {
                        max_cll: u16::from_be_bytes([a, b]),
                        max_fall: u16::from_be_bytes([c, d]),
                    });
                }
            }
            ALTERNATIVE_TRANSFER => {
                // A code that names no transfer changes nothing.
                if let Some(&code) = payload.first()
                    && transfer_of(u32::from(code)) != cutlass_decode::Transfer::Unspecified
                {
                    self.hdr.transfer = Some(code);
                }
            }
            _ => {}
        }
    }

    /// The picture timing of a picture that uses `sps`: present when the
    /// message came and the VUI says it starts with `pic_struct`.
    pub fn timing(&self, sps: &Sps) -> Option<PicTiming> {
        let byte = self.timing.filter(|_| sps.vui.frame_field_info)?;
        Some(PicTiming {
            pic_struct: byte >> 4,
            source_scan_type: (byte >> 2) & 3,
            duplicate: byte & 2 != 0,
        })
    }
}

/// A payload type or size: bytes of 255 added up, ended by a smaller one.
fn ff_coded(rbsp: &[u8], at: &mut usize) -> Option<u32> {
    let mut value = 0u32;
    loop {
        let byte = *rbsp.get(*at)?;
        *at += 1;
        value = value.checked_add(u32::from(byte))?;
        if byte != 0xff {
            return Some(value);
        }
    }
}

/// User data registered by ITU-T T.35 (clause D.2.6): the country code,
/// its extension after a code of `0xff`, and the bytes after them.
fn t35(payload: &[u8]) -> Option<ItuT35> {
    let (&country_code, rest) = payload.split_first()?;
    let (country_code_extension, rest) = if country_code == 0xff {
        let (&extension, rest) = rest.split_first()?;
        (Some(extension), rest)
    } else {
        (None, rest)
    };
    Some(ItuT35 {
        country_code,
        country_code_extension,
        payload: rest.to_vec(),
    })
}

/// Mastering display colour volume (clause D.2.28). The message codes
/// chromaticity in steps of 0.00002 and luminance in steps of 0.0001
/// candelas per square metre, green first, then blue, then red; the
/// result is in the fixed-point units of [`MasteringDisplay`], red first,
/// each value rounded to the nearest step.
fn mastering(payload: &[u8]) -> Option<MasteringDisplay> {
    let bytes = payload.first_chunk::<24>()?;
    let word = |i: usize| u64::from(u16::from_be_bytes([bytes[2 * i], bytes[2 * i + 1]]));
    let long = |at: usize| {
        u64::from(u32::from_be_bytes([
            bytes[at],
            bytes[at + 1],
            bytes[at + 2],
            bytes[at + 3],
        ]))
    };
    let scaled = |value: u64, to: u64, from: u64| (value * to + from / 2) / from;
    let chroma =
        |i: usize| -> u16 { u16::try_from(scaled(word(i), 1 << 16, 50_000)).unwrap_or(u16::MAX) };
    let point = |i: usize| [chroma(2 * i), chroma(2 * i + 1)];
    let luminance = |at: usize, one: u64| -> u32 {
        u32::try_from(scaled(long(at), one, 10_000)).unwrap_or(u32::MAX)
    };
    Some(MasteringDisplay {
        primaries: [point(2), point(0), point(1)],
        white_point: point(3),
        max_luminance: luminance(16, 1 << 8),
        min_luminance: luminance(20, 1 << 14),
    })
}

#[cfg(test)]
mod tests {
    use super::{Hdr, Sei};
    use cutlass_decode::{ContentLight, ItuT35, MasteringDisplay};

    /// One SEI message: type, size, payload.
    fn message(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![kind, payload.len() as u8];
        out.extend_from_slice(payload);
        out
    }

    fn read(rbsp: &[u8], suffix: bool) -> Sei {
        let mut sei = Sei::default();
        sei.read(rbsp, suffix);
        sei
    }

    /// BT.2020 primaries and D65 at 1000 and 0.005 candelas per square
    /// metre, as x265 writes `master-display`.
    fn display() -> Vec<u8> {
        let mut payload = Vec::new();
        for value in [8500u16, 39850, 6550, 2300, 35400, 14600, 15635, 16450] {
            payload.extend_from_slice(&value.to_be_bytes());
        }
        payload.extend_from_slice(&10_000_000u32.to_be_bytes());
        payload.extend_from_slice(&50u32.to_be_bytes());
        payload
    }

    #[test]
    fn the_hdr_messages_of_one_unit_are_read() {
        let mut rbsp = message(137, &display());
        rbsp.extend(message(144, &[0x03, 0xe8, 0x01, 0x90]));
        rbsp.extend(message(147, &[18]));
        rbsp.push(0x80);
        let sei = read(&rbsp, false);
        let want = MasteringDisplay {
            // Red, green, blue: 0.708 0.292, 0.170 0.797, 0.131 0.046.
            primaries: [[46399, 19137], [11141, 52232], [8585, 3015]],
            white_point: [20493, 21561],
            max_luminance: 1000 << 8,
            min_luminance: 82,
        };
        assert_eq!(sei.hdr.mastering, Some(want));
        assert_eq!(
            sei.hdr.light,
            Some(ContentLight {
                max_cll: 1000,
                max_fall: 400
            })
        );
        assert_eq!(sei.hdr.transfer, Some(18));
        assert!(sei.t35.is_empty());
    }

    #[test]
    fn the_coded_chromaticity_comes_back_from_the_fixed_point_value() {
        // 0.16 fixed point is finer than the message's steps of 0.00002.
        let sei = read(&message(137, &display()), false);
        let display = sei.hdr.mastering.expect("display");
        let back = |v: u16| (u64::from(v) * 50_000 + (1 << 15)) >> 16;
        assert_eq!(back(display.primaries[0][0]), 35400);
        assert_eq!(back(display.primaries[1][1]), 39850);
        assert_eq!(back(display.white_point[0]), 15635);
        assert_eq!(back(display.white_point[1]), 16450);
        assert_eq!(
            (u64::from(display.min_luminance) * 10_000 + (1 << 13)) >> 14,
            50
        );
    }

    #[test]
    fn t35_keeps_the_country_code_and_the_bytes_after_it() {
        // HDR10+: United States, then the provider code and its data.
        let mut rbsp = message(4, &[0xb5, 0x00, 0x3c, 0x00, 0x01, 0x04, 0x00]);
        rbsp.extend(message(4, &[0xff, 0x12, 0x34]));
        rbsp.push(0x80);
        for suffix in [false, true] {
            let sei = read(&rbsp, suffix);
            assert_eq!(
                sei.t35,
                [
                    ItuT35 {
                        country_code: 0xb5,
                        country_code_extension: None,
                        payload: vec![0x00, 0x3c, 0x00, 0x01, 0x04, 0x00],
                    },
                    ItuT35 {
                        country_code: 0xff,
                        country_code_extension: Some(0x12),
                        payload: vec![0x34],
                    },
                ]
            );
        }
    }

    #[test]
    fn a_suffix_unit_carries_no_hdr_description() {
        let mut rbsp = message(144, &[0x03, 0xe8, 0x01, 0x90]);
        rbsp.extend(message(147, &[18]));
        rbsp.push(0x80);
        assert_eq!(read(&rbsp, true).hdr, Hdr::default());
    }

    #[test]
    fn a_long_type_and_size_are_sums_of_bytes() {
        // Type 255 + 5 with a size of 255 + 1 bytes, then light level.
        let mut rbsp = vec![0xff, 0x05, 0xff, 0x01];
        rbsp.extend(std::iter::repeat_n(0xaa, 256));
        rbsp.extend(message(144, &[0, 1, 0, 2]));
        rbsp.push(0x80);
        let sei = read(&rbsp, false);
        assert_eq!(
            sei.hdr.light,
            Some(ContentLight {
                max_cll: 1,
                max_fall: 2
            })
        );
    }

    #[test]
    fn short_and_cut_messages_are_dropped() {
        // Light level in two bytes, a display in ten, an empty transfer,
        // an empty T.35 message and one that ends at its extension.
        let mut rbsp = message(144, &[0x03, 0xe8]);
        rbsp.extend(message(137, &[0; 10]));
        rbsp.extend(message(147, &[]));
        rbsp.extend(message(4, &[]));
        rbsp.extend(message(4, &[0xff]));
        rbsp.push(0x80);
        let sei = read(&rbsp, false);
        assert_eq!(sei.hdr, Hdr::default());
        assert!(sei.t35.is_empty());

        // A message that says it is longer than its NAL unit ends the
        // unit: what came before it stays.
        let mut rbsp = message(147, &[16]);
        rbsp.extend([144, 200, 1, 2, 3]);
        let sei = read(&rbsp, false);
        assert_eq!(sei.hdr.transfer, Some(16));
        assert_eq!(sei.hdr.light, None);

        // A type or a size that never ends.
        for rbsp in [&[][..], &[0xff; 9], &[4, 0xff, 0xff]] {
            let sei = read(rbsp, false);
            assert_eq!(sei.hdr, Hdr::default());
            assert!(sei.t35.is_empty());
        }
    }

    #[test]
    fn a_transfer_code_that_names_nothing_is_ignored() {
        for code in [0u8, 2, 3, 19, 255] {
            let sei = read(&message(147, &[code]), false);
            assert_eq!(sei.hdr.transfer, None, "{code}");
        }
    }

    #[test]
    fn random_bytes_do_not_panic() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        for round in 0..4000 {
            let mut rbsp = Vec::new();
            for _ in 0..(round % 70) {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                rbsp.push([4, 137, 144, 147, 1, 0xff, 0x80, 24, 0][(seed % 9) as usize]);
            }
            let sei = read(&rbsp, round % 2 == 0);
            assert!(sei.t35.len() <= super::MAX_T35);
        }
    }

    #[test]
    fn later_messages_replace_earlier_ones_and_missing_ones_keep_them() {
        let mut hdr = Hdr::default();
        hdr.update(&read(&message(144, &[0, 10, 0, 5]), false));
        hdr.update(&read(&message(147, &[18]), false));
        assert_eq!(hdr.transfer, Some(18));
        assert_eq!(hdr.light.map(|l| l.max_cll), Some(10));
        hdr.update(&read(&message(144, &[0, 20, 0, 5]), false));
        assert_eq!(hdr.light.map(|l| l.max_cll), Some(20));
        let fallback = Hdr {
            transfer: Some(16),
            mastering: Some(MasteringDisplay::default()),
            light: None,
        };
        let both = hdr.or(fallback);
        assert_eq!(both.transfer, Some(18));
        assert_eq!(both.mastering, Some(MasteringDisplay::default()));
    }
}
