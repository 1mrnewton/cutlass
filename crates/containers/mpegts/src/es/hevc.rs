//! HEVC access units (ITU-T H.265 clause 7.4.2.4.4).
//!
//! A picture starts at a slice segment with `first_slice_segment_in_pic_flag`
//! set. Units of layers above 0 stay inside the access unit of their base
//! layer. A keyframe is an IRAP picture (unit types 16 to 23), as FFmpeg's
//! parser marks them.

use cutlass_codec::CodecId;

use super::Facts;
use super::bits::{Bits, unescape};
use super::video::{Class, Syntax};
use crate::limits::MAX_PARAMS_BYTES;

const VPS: u8 = 32;
const SPS: u8 = 33;
const PPS: u8 = 34;
const AUD: u8 = 35;
const PREFIX_SEI: u8 = 39;

const MAX_SET_BYTES: usize = 4096;

#[derive(Default)]
pub struct Hevc {
    /// `(unit type, id, unit)` of each parameter set.
    sets: Vec<(u8, u32, Vec<u8>)>,
    size: Option<(u32, u32)>,
    rbsp: Vec<u8>,
}

impl Hevc {
    fn keep(&mut self, kind: u8, unit: &[u8]) {
        unescape(unit, MAX_SET_BYTES, &mut self.rbsp);
        let body = self.rbsp.get(2..).unwrap_or_default();
        let id = match kind {
            VPS => body.first().map(|byte| u32::from(byte >> 4)),
            SPS => match parse_sps(body) {
                Some((id, width, height)) => {
                    self.size = Some((width, height));
                    Some(id)
                }
                None => None,
            },
            _ => Bits::new(body).ue(),
        };
        let Some(id) = id else {
            return;
        };
        self.sets
            .retain(|(old, old_id, _)| !(*old == kind && *old_id == id));
        self.sets.push((kind, id, unit.to_vec()));
        while self
            .sets
            .iter()
            .map(|(_, _, unit)| unit.len())
            .sum::<usize>()
            > MAX_PARAMS_BYTES
        {
            self.sets.remove(0);
        }
    }
}

impl Syntax for Hevc {
    const HEAD: usize = 3;

    fn classify(&mut self, head: &[u8], complete: bool) -> Class {
        let [first, second, rest @ ..] = head else {
            return if complete {
                Class::Inside { whole: false }
            } else {
                Class::NeedMore
            };
        };
        let kind = (first >> 1) & 0x3F;
        let layer = ((first & 1) << 5) | (second >> 3);
        if layer != 0 {
            return Class::Inside { whole: false };
        }
        match kind {
            0..=31 => {
                if kind > 21 {
                    // Reserved VCL types: kept with the picture.
                    return Class::Inside { whole: false };
                }
                let Some(&flags) = rest.first() else {
                    return if complete {
                        Class::Inside { whole: false }
                    } else {
                        Class::NeedMore
                    };
                };
                Class::Picture {
                    first: flags & 0x80 != 0,
                    pairs: false,
                    key: (16..=23).contains(&kind),
                }
            }
            VPS | SPS | PPS => Class::Opens {
                key: false,
                whole: true,
            },
            AUD | PREFIX_SEI | 41..=44 | 48..=55 => Class::Opens {
                key: false,
                whole: false,
            },
            _ => Class::Inside { whole: false },
        }
    }

    fn whole(&mut self, unit: &[u8]) {
        if let Some(&first) = unit.first() {
            let kind = (first >> 1) & 0x3F;
            if matches!(kind, VPS | SPS | PPS) {
                self.keep(kind, unit);
            }
        }
    }

    fn facts(&self) -> Option<Facts> {
        let (width, height) = self.size?;
        let mut extradata = Vec::new();
        for wanted in [VPS, SPS, PPS] {
            for (_, _, unit) in self.sets.iter().filter(|(kind, _, _)| *kind == wanted) {
                extradata.extend_from_slice(&[0, 0, 0, 1]);
                extradata.extend_from_slice(unit);
            }
        }
        Some(Facts::Video {
            codec: CodecId::Hevc,
            width,
            height,
            extradata,
        })
    }

    fn reset(&mut self) {}
}

/// Skips `profile_tier_level(1, max_sub_layers_minus1)` (clause 7.3.3).
fn skip_profile_tier_level(bits: &mut Bits<'_>, max_sub_layers_minus1: u32) -> Option<()> {
    bits.skip(96)?;
    let mut present = [(false, false); 8];
    for slot in present.iter_mut().take(max_sub_layers_minus1 as usize) {
        *slot = (bits.bit()?, bits.bit()?);
    }
    if max_sub_layers_minus1 > 0 {
        bits.skip(2 * (8 - max_sub_layers_minus1 as usize))?;
    }
    for &(profile, level) in present.iter().take(max_sub_layers_minus1 as usize) {
        if profile {
            bits.skip(88)?;
        }
        if level {
            bits.skip(8)?;
        }
    }
    Some(())
}

/// The id and the cropped picture size of an SPS RBSP after its unit
/// header (clause 7.3.2.2.1).
fn parse_sps(rbsp: &[u8]) -> Option<(u32, u32, u32)> {
    parse_sps_fields(rbsp).map(|(id, width, height, _)| (id, width, height))
}

/// `sps_max_num_reorder_pics` of the highest sub-layer, from an SPS RBSP
/// after its two-byte NAL header (clause 7.3.2.2).
#[must_use]
pub fn reorder_pictures(rbsp: &[u8]) -> Option<u32> {
    parse_sps_fields(rbsp).and_then(|(.., reorder)| reorder)
}

fn parse_sps_fields(rbsp: &[u8]) -> Option<(u32, u32, u32, Option<u32>)> {
    let mut bits = Bits::new(rbsp);
    bits.skip(4)?;
    let max_sub_layers_minus1 = bits.bits(3)?;
    if max_sub_layers_minus1 > 6 {
        return None;
    }
    bits.skip(1)?;
    skip_profile_tier_level(&mut bits, max_sub_layers_minus1)?;
    let id = bits.ue()?;
    if id > 15 {
        return None;
    }
    let chroma_format = bits.ue()?;
    if chroma_format > 3 {
        return None;
    }
    let separate_colour_plane = chroma_format == 3 && bits.bit()?;
    let width = bits.ue()?;
    let height = bits.ue()?;
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
    let crop_x = crop[0].checked_add(crop[1])?.checked_mul(unit_x)?;
    let crop_y = crop[2].checked_add(crop[3])?.checked_mul(unit_y)?;
    let reorder = sub_layer_reorder(&mut bits, max_sub_layers_minus1);
    match (width.checked_sub(crop_x), height.checked_sub(crop_y)) {
        (Some(w), Some(h)) if w > 0 && h > 0 => Some((id, w, h, reorder)),
        _ => (width > 0 && height > 0).then_some((id, width, height, reorder)),
    }
}

/// From the bit depths on: the highest sub-layer's reorder count.
fn sub_layer_reorder(bits: &mut Bits<'_>, max_sub_layers_minus1: u32) -> Option<u32> {
    bits.ue()?;
    bits.ue()?;
    bits.ue()?;
    let all = bits.bit()?;
    let first = if all { 0 } else { max_sub_layers_minus1 };
    let mut reorder = 0;
    for _ in first..=max_sub_layers_minus1 {
        bits.ue()?;
        reorder = bits.ue()?;
        bits.ue()?;
    }
    (reorder <= 16).then_some(reorder)
}

#[cfg(test)]
mod tests {
    use super::{Hevc, parse_sps, reorder_pictures};
    use crate::es::bits::unescape;
    use crate::es::video::{Class, Syntax};

    #[test]
    fn slices_and_parameter_sets_are_told_apart() {
        let mut hevc = Hevc::default();
        // IDR_W_RADL (19), first slice segment.
        assert_eq!(
            hevc.classify(&[19 << 1, 1, 0x80], false),
            Class::Picture {
                first: true,
                pairs: false,
                key: true
            }
        );
        // TRAIL_R (1), a later slice segment.
        assert_eq!(
            hevc.classify(&[1 << 1, 1, 0x00], false),
            Class::Picture {
                first: false,
                pairs: false,
                key: false
            }
        );
        assert_eq!(
            hevc.classify(&[33 << 1, 1, 0], false),
            Class::Opens {
                key: false,
                whole: true
            }
        );
        // A slice of layer 1 stays with its base layer.
        assert_eq!(
            hevc.classify(&[1 << 1, 0x09, 0x80], false),
            Class::Inside { whole: false }
        );
    }

    #[test]
    fn a_1080p_sps_reads_its_cropped_size() {
        // The SPS libx265 writes for Main profile 1920x1080 (ffmpeg 8.1.1),
        // after its two-byte unit header.
        let sps = [
            0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00,
            0x00, 0x03, 0x00, 0x78, 0xA0, 0x03, 0xC0, 0x80, 0x10, 0xE5, 0x96, 0x56, 0x69, 0x24,
            0xCA, 0xF0, 0x16, 0x80, 0x80, 0x00, 0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x0C, 0x84,
        ];
        let mut rbsp = Vec::new();
        unescape(&sps[2..], 4096, &mut rbsp);
        assert_eq!(parse_sps(&rbsp), Some((0, 1920, 1080)));
    }

    #[test]
    fn the_sps_states_the_reorder_depth() {
        // A libx265 SPS from FFmpeg 8.1.1 at 320x240 with `bframes=3`;
        // FFmpeg's `trace_headers` reads sps_max_num_reorder_pics 2.
        let sps = [
            0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00,
            0x00, 0x03, 0x00, 0x3C, 0xA0, 0x0A, 0x08, 0x0F, 0x16, 0x59, 0x59, 0x64, 0x93, 0x2B,
            0xC0, 0x5A, 0x02, 0x00, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x03, 0x00, 0x32, 0x10,
        ];
        let mut rbsp = Vec::new();
        unescape(&sps, sps.len(), &mut rbsp);
        assert_eq!(reorder_pictures(&rbsp[2..]), Some(2));
    }
}
