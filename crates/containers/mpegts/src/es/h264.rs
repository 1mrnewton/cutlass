//! H.264 access units (ITU-T H.264 clause 7.4.1.2.3).
//!
//! A new picture starts at a slice whose `first_mb_in_slice` is not past
//! the one before, or whose `frame_num`, field or IDR state changed. Two
//! complementary fields of one frame stay in one access unit, the way the
//! decoder takes them. A keyframe is an IDR picture or one after a
//! recovery-point SEI message, as FFmpeg's parser marks them.

use cutlass_codec::CodecId;

use super::Facts;
use super::bits::{Bits, unescape};
use super::video::{Class, Syntax};
use crate::limits::MAX_PARAMS_BYTES;

const SLICE: u8 = 1;
const PARTITION_A: u8 = 2;
const IDR: u8 = 5;
const SEI: u8 = 6;
const SPS: u8 = 7;
const PPS: u8 = 8;
const AUD: u8 = 9;
const RECOVERY_POINT: u32 = 6;

/// The longest parameter set read.
const MAX_SET_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sps {
    id: u32,
    log2_max_frame_num: u32,
    frame_mbs_only: bool,
    separate_colour_plane: bool,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Picture {
    first_mb: u32,
    frame_num: Option<u32>,
    /// `Some(bottom)` for a field, `None` for a frame or when unknown.
    field: Option<bool>,
    idr: bool,
}

#[derive(Default)]
pub struct H264 {
    sps: Vec<(Sps, Vec<u8>)>,
    /// `(pps_id, sps_id, unit)`.
    pps: Vec<(u32, u32, Vec<u8>)>,
    last: Option<Picture>,
    /// The picture in progress is a field without its pair yet:
    /// `(frame_num, bottom, idr)`.
    lone_field: Option<(Option<u32>, bool, bool)>,
    rbsp: Vec<u8>,
}

impl H264 {
    fn sps_of(&self, pps_id: u32) -> Option<Sps> {
        let (_, sps_id, _) = self.pps.iter().find(|(id, _, _)| *id == pps_id)?;
        self.sps
            .iter()
            .map(|(sps, _)| *sps)
            .find(|sps| sps.id == *sps_id)
    }

    /// `first_mb_in_slice`, `frame_num` and the field of a slice header
    /// in `rbsp`. The last two are `None` when the PPS or SPS is not known.
    /// `None` when the head ended early and more of it may come.
    fn slice_fields(&self, complete: bool) -> Option<(u32, Option<u32>, Option<bool>)> {
        let mut bits = Bits::new(&self.rbsp);
        let short = |found: Option<u32>| {
            if complete {
                Some(found)
            } else {
                found.map(Some)
            }
        };
        let Some(first_mb) = short(bits.ue())? else {
            return Some((0, None, None));
        };
        if short(bits.ue())?.is_none() {
            return Some((first_mb, None, None));
        }
        let Some(pps_id) = short(bits.ue())? else {
            return Some((first_mb, None, None));
        };
        let Some(sps) = self.sps_of(pps_id) else {
            return Some((first_mb, None, None));
        };
        if sps.separate_colour_plane {
            short(bits.bits(2))?;
        }
        let frame_num = short(bits.bits(sps.log2_max_frame_num))?;
        let mut field = None;
        if !sps.frame_mbs_only && short(bits.bits(1))? == Some(1) {
            field = short(bits.bits(1))?.map(|bottom| bottom == 1);
        }
        Some((first_mb, frame_num, field))
    }

    fn slice(&mut self, nal_type: u8, head: &[u8], complete: bool) -> Class {
        unescape(head, head.len(), &mut self.rbsp);
        let Some((first_mb, frame_num, field)) = self.slice_fields(complete) else {
            return Class::NeedMore;
        };
        let idr = nal_type == IDR;
        let picture = Picture {
            first_mb,
            frame_num,
            field,
            idr,
        };
        let first = self.last.is_none_or(|last| {
            first_mb <= last.first_mb
                || last.frame_num != frame_num
                || last.field != field
                || last.idr != idr
        });
        self.last = Some(picture);
        let mut pairs = false;
        if first {
            pairs = match (self.lone_field, field) {
                (Some((lone_num, lone_bottom, lone_idr)), Some(bottom)) => {
                    lone_num == frame_num && lone_bottom != bottom && !(idr && !lone_idr)
                }
                _ => false,
            };
            self.lone_field = match field {
                Some(bottom) if !pairs => Some((frame_num, bottom, idr)),
                _ => None,
            };
        }
        Class::Picture {
            first,
            pairs,
            key: idr,
        }
    }

    fn keep_sps(&mut self, unit: &[u8]) {
        unescape(unit, MAX_SET_BYTES, &mut self.rbsp);
        let Some(sps) = parse_sps(self.rbsp.get(1..).unwrap_or_default()) else {
            return;
        };
        self.sps.retain(|(old, _)| old.id != sps.id);
        self.sps.push((sps, unit.to_vec()));
        self.trim();
    }

    fn keep_pps(&mut self, unit: &[u8]) {
        unescape(unit, 16, &mut self.rbsp);
        let mut bits = Bits::new(self.rbsp.get(1..).unwrap_or_default());
        let (Some(pps_id), Some(sps_id)) = (bits.ue(), bits.ue()) else {
            return;
        };
        if pps_id > 255 || sps_id > 31 {
            return;
        }
        self.pps.retain(|(old, _, _)| *old != pps_id);
        self.pps.push((pps_id, sps_id, unit.to_vec()));
        self.trim();
    }

    /// Keeps the parameter sets inside [`MAX_PARAMS_BYTES`], dropping the
    /// oldest.
    fn trim(&mut self) {
        loop {
            let total = self.sps.iter().map(|(_, unit)| unit.len()).sum::<usize>()
                + self
                    .pps
                    .iter()
                    .map(|(_, _, unit)| unit.len())
                    .sum::<usize>();
            if total <= MAX_PARAMS_BYTES {
                return;
            }
            if self.pps.len() > 1 {
                self.pps.remove(0);
            } else if self.sps.len() > 1 {
                self.sps.remove(0);
            } else {
                self.pps.clear();
                self.sps.clear();
                return;
            }
        }
    }
}

impl Syntax for H264 {
    const HEAD: usize = 64;

    fn classify(&mut self, head: &[u8], complete: bool) -> Class {
        let Some(&nal) = head.first() else {
            return if complete {
                Class::Inside { whole: false }
            } else {
                Class::NeedMore
            };
        };
        let nal_type = nal & 0x1F;
        match nal_type {
            SLICE | PARTITION_A | IDR => self.slice(nal_type, &head[1..], complete),
            SPS | PPS => Class::Opens {
                key: false,
                whole: true,
            },
            SEI if !complete && head.len() < Self::HEAD => Class::NeedMore,
            SEI => Class::Opens {
                key: has_recovery_point(&head[1..]),
                whole: false,
            },
            AUD | 13..=18 => Class::Opens {
                key: false,
                whole: false,
            },
            _ => Class::Inside { whole: false },
        }
    }

    fn whole(&mut self, unit: &[u8]) {
        match unit.first().map(|nal| nal & 0x1F) {
            Some(SPS) => self.keep_sps(unit),
            Some(PPS) => self.keep_pps(unit),
            _ => {}
        }
    }

    fn facts(&self) -> Option<Facts> {
        let (sps, _) = self.sps.last()?;
        let mut extradata = Vec::new();
        for (_, unit) in &self.sps {
            extradata.extend_from_slice(&[0, 0, 0, 1]);
            extradata.extend_from_slice(unit);
        }
        for (_, _, unit) in &self.pps {
            extradata.extend_from_slice(&[0, 0, 0, 1]);
            extradata.extend_from_slice(unit);
        }
        Some(Facts::Video {
            codec: CodecId::H264,
            width: sps.width,
            height: sps.height,
            extradata,
        })
    }

    fn reset(&mut self) {
        self.last = None;
        self.lone_field = None;
    }
}

/// Whether the SEI messages that fit in `payload` hold a recovery point.
fn has_recovery_point(payload: &[u8]) -> bool {
    let mut at = 0;
    loop {
        let Some((kind, after)) = sei_number(payload, at) else {
            return false;
        };
        if kind == RECOVERY_POINT {
            return true;
        }
        let Some((size, body)) = sei_number(payload, after) else {
            return false;
        };
        let Some(next) = body.checked_add(size as usize) else {
            return false;
        };
        // The trailing bits end the messages.
        if payload.get(next).is_none_or(|&byte| byte == 0x80) {
            return false;
        }
        at = next;
    }
}

/// A SEI payload type or size: `0xFF` bytes each add 255.
fn sei_number(bytes: &[u8], mut at: usize) -> Option<(u32, usize)> {
    let mut value = 0_u32;
    loop {
        let byte = *bytes.get(at)?;
        at += 1;
        value = value.checked_add(u32::from(byte))?;
        if byte != 0xFF {
            return Some((value, at));
        }
    }
}

/// Profiles whose SPS carries the chroma and bit-depth fields.
const fn has_chroma_fields(profile: u32) -> bool {
    matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    )
}

fn skip_scaling_list(bits: &mut Bits<'_>, size: u32) -> Option<()> {
    let mut last = 8_i32;
    let mut next = 8_i32;
    for _ in 0..size {
        if next != 0 {
            let delta = bits.se()?;
            next = (last + delta).rem_euclid(256);
        }
        if next != 0 {
            last = next;
        }
    }
    Some(())
}

/// Reads the fields framing needs from an SPS RBSP after its NAL header
/// (clause 7.3.2.1.1).
fn parse_sps(rbsp: &[u8]) -> Option<Sps> {
    let mut bits = Bits::new(rbsp);
    let profile = bits.bits(8)?;
    bits.skip(16)?;
    let id = bits.ue()?;
    if id > 31 {
        return None;
    }
    let mut chroma_format = 1;
    let mut separate_colour_plane = false;
    if has_chroma_fields(profile) {
        chroma_format = bits.ue()?;
        if chroma_format > 3 {
            return None;
        }
        if chroma_format == 3 {
            separate_colour_plane = bits.bit()?;
        }
        bits.ue()?;
        bits.ue()?;
        bits.skip(1)?;
        if bits.bit()? {
            let lists = if chroma_format == 3 { 12 } else { 8 };
            for index in 0..lists {
                if bits.bit()? {
                    skip_scaling_list(&mut bits, if index < 6 { 16 } else { 64 })?;
                }
            }
        }
    }
    let log2_max_frame_num = bits.ue()?.checked_add(4)?;
    if log2_max_frame_num > 16 {
        return None;
    }
    match bits.ue()? {
        0 => {
            bits.ue()?;
        }
        1 => {
            bits.skip(1)?;
            bits.se()?;
            bits.se()?;
            let cycle = bits.ue()?;
            if cycle > 255 {
                return None;
            }
            for _ in 0..cycle {
                bits.se()?;
            }
        }
        2 => {}
        _ => return None,
    }
    bits.ue()?;
    bits.skip(1)?;
    let width_mbs = bits.ue()?.checked_add(1)?;
    let height_units = bits.ue()?.checked_add(1)?;
    let frame_mbs_only = bits.bit()?;
    if !frame_mbs_only {
        bits.skip(1)?;
    }
    bits.skip(1)?;
    let crop = if bits.bit()? {
        [bits.ue()?, bits.ue()?, bits.ue()?, bits.ue()?]
    } else {
        [0; 4]
    };
    let chroma_array = if separate_colour_plane {
        0
    } else {
        chroma_format
    };
    let (unit_x, unit_y) = match chroma_array {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    };
    let field_factor = if frame_mbs_only { 1 } else { 2 };
    let width = width_mbs.checked_mul(16)?;
    let height = height_units.checked_mul(16)?.checked_mul(field_factor)?;
    let crop_x = (crop[0].checked_add(crop[1])?).checked_mul(unit_x)?;
    let crop_y = (crop[2].checked_add(crop[3])?).checked_mul(unit_y * field_factor)?;
    let (width, height) = match (width.checked_sub(crop_x), height.checked_sub(crop_y)) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
        _ => (width, height),
    };
    Some(Sps {
        id,
        log2_max_frame_num,
        frame_mbs_only,
        separate_colour_plane,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::{has_recovery_point, parse_sps};

    #[test]
    fn a_1080p_high_profile_sps_reads_its_cropped_size() {
        // The SPS libx264 writes for High profile 1920x1080 at 25 fps
        // (ffmpeg 8.1.1): 120x68 macroblocks cropped by 8 rows.
        let sps = [
            0x64, 0x00, 0x28, 0xAC, 0xD9, 0x40, 0x78, 0x02, 0x27, 0xE5, 0xC0, 0x44, 0x00, 0x00,
            0x03, 0x00, 0x04, 0x00, 0x00, 0x03, 0x00, 0xC8, 0x3C, 0x60, 0xC6, 0x58,
        ];
        let mut rbsp = Vec::new();
        super::unescape(&sps, 64, &mut rbsp);
        let parsed = parse_sps(&rbsp).expect("sps");
        assert_eq!((parsed.width, parsed.height), (1920, 1080));
        assert!(parsed.frame_mbs_only);
    }

    #[test]
    fn a_recovery_point_is_found_after_another_message() {
        // User data unregistered (5) of 2 bytes, then a recovery point (6).
        assert!(has_recovery_point(&[5, 2, 0xAA, 0xBB, 6, 1, 0x84, 0x80]));
        assert!(!has_recovery_point(&[5, 2, 0xAA, 0xBB, 0x80]));
        assert!(!has_recovery_point(&[0xFF, 0xFF]));
    }
}
