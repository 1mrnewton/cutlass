//! Display order for a decoder that hands pictures back in decode order.
//!
//! VideoToolbox returns H.264 and HEVC pictures as it decodes them. A stream
//! with B-frames then runs backward in time. [`Reorder`] holds as many
//! pictures as the stream may reorder and releases the earliest one each time
//! that bound is passed. The bound is read from the sequence parameter set.
//! An H.264 set without a bitstream restriction uses the level's picture
//! buffer, as the standard does. A set that does not parse uses
//! [`MAX_DEPTH`].

/// The most pictures any H.264 or HEVC stream may hold for reorder.
pub const MAX_DEPTH: usize = 16;

/// Pictures held until the stream's reorder depth is passed.
#[derive(Debug)]
pub struct Reorder<T> {
    depth: usize,
    held: Vec<(i64, T)>,
}

impl<T> Reorder<T> {
    #[must_use]
    pub fn new(depth: usize) -> Self {
        Self {
            depth: depth.min(MAX_DEPTH),
            held: Vec::new(),
        }
    }

    /// Hold `item`, then release the earliest pictures above the depth.
    pub fn push(&mut self, pts: i64, item: T, out: &mut Vec<T>) {
        self.held.push((pts, item));
        while self.held.len() > self.depth {
            out.push(self.take_first());
        }
    }

    /// Release every held picture, earliest first.
    pub fn flush(&mut self, out: &mut Vec<T>) {
        while !self.held.is_empty() {
            out.push(self.take_first());
        }
    }

    /// Hand back every held picture without ordering, for the caller to drop.
    pub fn clear(&mut self) -> Vec<T> {
        self.held.drain(..).map(|(_, item)| item).collect()
    }

    fn take_first(&mut self) -> T {
        let mut first = 0;
        for (index, (pts, _)) in self.held.iter().enumerate() {
            if *pts < self.held[first].0 {
                first = index;
            }
        }
        self.held.remove(first).1
    }
}

/// How many pictures an H.264 stream may hold back, from its SPS NAL unit.
#[must_use]
pub fn h264_depth(sps: &[u8]) -> Option<usize> {
    let rbsp = unescape(sps.get(1..)?);
    let mut bits = Bits::new(&rbsp);
    let profile = bits.u(8)?;
    let constraints = bits.u(8)?;
    let level = bits.u(8)?;
    bits.ue()?;
    if matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        let chroma = bits.ue()?;
        if chroma == 3 {
            bits.u(1)?;
        }
        bits.ue()?;
        bits.ue()?;
        bits.u(1)?;
        if bits.u(1)? == 1 {
            let lists = if chroma == 3 { 12 } else { 8 };
            for index in 0..lists {
                if bits.u(1)? == 1 {
                    skip_scaling_list(&mut bits, if index < 6 { 16 } else { 64 })?;
                }
            }
        }
    }
    bits.ue()?;
    let order_type = bits.ue()?;
    match order_type {
        0 => {
            bits.ue()?;
        }
        1 => {
            bits.u(1)?;
            bits.se()?;
            bits.se()?;
            for _ in 0..bits.ue()? {
                bits.se()?;
            }
        }
        _ => {}
    }
    bits.ue()?;
    bits.u(1)?;
    let width_mbs = bits.ue()?.checked_add(1)?;
    let map_units = bits.ue()?.checked_add(1)?;
    let frame_only = bits.u(1)?;
    let height_mbs = map_units.checked_mul(2 - frame_only)?;
    // Picture order type 2 is output order equal to decode order.
    if order_type == 2 {
        return Some(0);
    }
    let fallback = level_depth(level, constraints, width_mbs, height_mbs);
    if frame_only == 0 {
        bits.u(1)?;
    }
    bits.u(1)?;
    if bits.u(1)? == 1 {
        for _ in 0..4 {
            bits.ue()?;
        }
    }
    if bits.u(1)? == 0 {
        return Some(fallback);
    }
    Some(vui_depth(&mut bits)?.unwrap_or(fallback))
}

/// `max_num_reorder_frames` from the VUI, or `None` when it carries no
/// bitstream restriction.
fn vui_depth(bits: &mut Bits<'_>) -> Option<Option<usize>> {
    if bits.u(1)? == 1 && bits.u(8)? == 255 {
        bits.u(16)?;
        bits.u(16)?;
    }
    if bits.u(1)? == 1 {
        bits.u(1)?;
    }
    if bits.u(1)? == 1 {
        bits.u(4)?;
        if bits.u(1)? == 1 {
            bits.u(24)?;
        }
    }
    if bits.u(1)? == 1 {
        bits.ue()?;
        bits.ue()?;
    }
    if bits.u(1)? == 1 {
        bits.u(32)?;
        bits.u(32)?;
        bits.u(1)?;
    }
    let nal_hrd = bits.u(1)?;
    if nal_hrd == 1 {
        skip_hrd(bits)?;
    }
    let vcl_hrd = bits.u(1)?;
    if vcl_hrd == 1 {
        skip_hrd(bits)?;
    }
    if nal_hrd == 1 || vcl_hrd == 1 {
        bits.u(1)?;
    }
    bits.u(1)?;
    if bits.u(1)? == 0 {
        return Some(None);
    }
    // Motion vectors over picture edges, then bytes per picture, bits per
    // macroblock, and the two motion vector lengths.
    bits.u(1)?;
    for _ in 0..4 {
        bits.ue()?;
    }
    let depth = bits.ue()?;
    Some(Some(usize::try_from(depth).ok()?))
}

fn skip_hrd(bits: &mut Bits<'_>) -> Option<()> {
    let count = bits.ue()?.checked_add(1)?;
    bits.u(8)?;
    for _ in 0..count.min(32) {
        bits.ue()?;
        bits.ue()?;
        bits.u(1)?;
    }
    bits.u(20)?;
    Some(())
}

fn skip_scaling_list(bits: &mut Bits<'_>, size: usize) -> Option<()> {
    let mut last = 8i64;
    let mut next = 8i64;
    for _ in 0..size {
        if next != 0 {
            next = (last + bits.se()? + 256) % 256;
        }
        if next != 0 {
            last = next;
        }
    }
    Some(())
}

/// The picture buffer a level allows, which bounds reorder without a VUI.
/// Intra-only profiles, flagged by constraint set 3 on high profiles, hold
/// nothing; the caller reaches here only for those with B-frames possible.
fn level_depth(level: u32, constraints: u32, width_mbs: u32, height_mbs: u32) -> usize {
    let level_1b = level == 11 && constraints & 0x10 != 0;
    let max_mbs: u32 = match level {
        _ if level_1b => 396,
        9 | 10 => 396,
        11 => 900,
        12 | 13 | 20 => 2_376,
        21 => 4_752,
        22 | 30 => 8_100,
        31 => 18_000,
        32 => 20_480,
        40 | 41 => 32_768,
        42 => 34_816,
        50 => 110_400,
        51 | 52 => 184_320,
        _ => 696_320,
    };
    let frame = width_mbs.saturating_mul(height_mbs).max(1);
    usize::try_from(max_mbs / frame)
        .unwrap_or(MAX_DEPTH)
        .min(MAX_DEPTH)
}

/// How many pictures an HEVC stream may hold back, from its SPS NAL unit.
#[must_use]
pub fn hevc_depth(sps: &[u8]) -> Option<usize> {
    let rbsp = unescape(sps.get(2..)?);
    let mut bits = Bits::new(&rbsp);
    bits.u(4)?;
    let sub_layers = bits.u(3)?;
    bits.u(1)?;
    skip_profile_tier_level(&mut bits, sub_layers)?;
    bits.ue()?;
    if bits.ue()? == 3 {
        bits.u(1)?;
    }
    bits.ue()?;
    bits.ue()?;
    if bits.u(1)? == 1 {
        for _ in 0..4 {
            bits.ue()?;
        }
    }
    bits.ue()?;
    bits.ue()?;
    bits.ue()?;
    let every_layer = bits.u(1)? == 1;
    let first = if every_layer { 0 } else { sub_layers };
    let mut depth = 0;
    for _ in first..=sub_layers {
        bits.ue()?;
        depth = bits.ue()?;
        bits.ue()?;
    }
    usize::try_from(depth).ok()
}

fn skip_profile_tier_level(bits: &mut Bits<'_>, sub_layers: u32) -> Option<()> {
    // General profile space through level: 2 + 1 + 5 + 32 + 48 + 8 bits.
    bits.skip(96)?;
    let mut profile = Vec::new();
    let mut level = Vec::new();
    for _ in 0..sub_layers {
        profile.push(bits.u(1)?);
        level.push(bits.u(1)?);
    }
    if sub_layers > 0 {
        for _ in sub_layers..8 {
            bits.u(2)?;
        }
    }
    for (profile, level) in profile.into_iter().zip(level) {
        if profile == 1 {
            bits.skip(88)?;
        }
        if level == 1 {
            bits.u(8)?;
        }
    }
    Some(())
}

/// Drop emulation prevention bytes: `00 00 03` reads as `00 00`.
fn unescape(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0;
    for &byte in data {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn bit(&mut self) -> Option<u32> {
        let byte = *self.data.get(self.at / 8)?;
        let bit = (byte >> (7 - self.at % 8)) & 1;
        self.at += 1;
        Some(u32::from(bit))
    }

    fn u(&mut self, count: u32) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..count.min(32) {
            value = (value << 1) | self.bit()?;
        }
        Some(value)
    }

    fn skip(&mut self, count: usize) -> Option<()> {
        let end = self.at.checked_add(count)?;
        if end > self.data.len() * 8 {
            return None;
        }
        self.at = end;
        Some(())
    }

    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        let rest = self.u(zeros)?;
        (1u32 << zeros).checked_sub(1)?.checked_add(rest)
    }

    fn se(&mut self) -> Option<i64> {
        let code = i64::from(self.ue()?);
        Some(if code % 2 == 1 {
            (code + 1) / 2
        } else {
            -(code / 2)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_DEPTH, Reorder, h264_depth, hevc_depth, unescape};

    #[test]
    fn decode_order_leaves_in_display_order() {
        let mut reorder = Reorder::new(2);
        let mut out = Vec::new();
        for pts in [0, 3, 1, 2, 6, 4, 5] {
            reorder.push(pts, pts, &mut out);
        }
        reorder.flush(&mut out);
        assert_eq!(out, [0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn depth_zero_passes_each_picture_through() {
        let mut reorder = Reorder::new(0);
        let mut out = Vec::new();
        reorder.push(5, "a", &mut out);
        assert_eq!(out, ["a"]);
        assert!(reorder.clear().is_empty());
    }

    #[test]
    fn depth_is_capped() {
        let mut reorder = Reorder::new(1_000);
        let mut out = Vec::new();
        for pts in 0..=16 {
            reorder.push(pts, pts, &mut out);
        }
        assert_eq!(out, [0]);
        assert_eq!(reorder.clear().len(), MAX_DEPTH);
    }

    #[test]
    fn emulation_prevention_is_removed() {
        assert_eq!(unescape(&[0, 0, 3, 1, 0, 0, 3]), [0, 0, 1, 0, 0]);
    }

    /// x264 High, 640x480, three B-frames.
    const X264_HIGH: [u8; 25] = [
        0x67, 0x64, 0x00, 0x1e, 0xac, 0xd9, 0x40, 0xa0, 0x3d, 0xb0, 0x11, 0x00, 0x00, 0x03, 0x00,
        0x01, 0x00, 0x00, 0x03, 0x00, 0x3c, 0x0f, 0x16, 0x2d, 0x96,
    ];
    /// x264 Baseline, 640x480.
    const X264_BASELINE: [u8; 24] = [
        0x67, 0x42, 0xc0, 0x1e, 0xd9, 0x00, 0xa0, 0x3d, 0xb0, 0x11, 0x00, 0x00, 0x03, 0x00, 0x01,
        0x00, 0x00, 0x03, 0x00, 0x3c, 0x0f, 0x16, 0x2e, 0x48,
    ];
    /// iPhone High 3.1, 848x608, no bitstream restriction.
    const IPHONE: [u8; 15] = [
        0x27, 0x64, 0x00, 0x1f, 0xac, 0x56, 0x50, 0x35, 0x04, 0xda, 0x6a, 0x02, 0x02, 0x02, 0x04,
    ];
    /// x265 Main, 640x480, four B-frames.
    const X265: [u8; 42] = [
        0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x5a, 0xa0, 0x05, 0x02, 0x01, 0xe1, 0x65, 0x95, 0x9a, 0x49, 0x32, 0xbc, 0x05,
        0xa0, 0x20, 0x00, 0x00, 0x03, 0x00, 0x20, 0x00, 0x00, 0x03, 0x03, 0xc1,
    ];
    /// macOS screen recording, HEVC Main, 3200x2400.
    const SCREEN: [u8; 35] = [
        0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0xb0, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x99, 0xa0, 0x01, 0x90, 0x20, 0x02, 0x58, 0x58, 0x81, 0x5e, 0xe4, 0x59, 0x54,
        0xdc, 0x04, 0x04, 0x04, 0x02,
    ];

    #[test]
    fn an_h264_vui_names_its_reorder_depth() {
        assert_eq!(h264_depth(&X264_HIGH), Some(2));
        assert_eq!(h264_depth(&X264_BASELINE), Some(0));
        assert!(h264_depth(&[0x67]).is_none());
    }

    #[test]
    fn an_h264_sps_without_a_restriction_holds_the_level_buffer() {
        // Level 3.1 allows 18000 macroblocks; 53 x 38 fit eight times.
        assert_eq!(h264_depth(&IPHONE), Some(8));
    }

    #[test]
    fn an_hevc_sps_names_its_reorder_depth() {
        assert_eq!(hevc_depth(&X265), Some(2));
        assert_eq!(hevc_depth(&SCREEN), Some(2));
    }

    #[test]
    fn a_short_hevc_sps_does_not_parse() {
        assert!(hevc_depth(&[0x42, 0x01, 0x01]).is_none());
    }
}
