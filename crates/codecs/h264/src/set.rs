//! Sequence and picture parameter sets, clauses 7.3.2.1 and 7.3.2.2.
//!
//! Interlaced pictures, other chroma formats, and other bit depths are refused.

use cutlass_decode::{CodecError, ColorMatrix, ColorPrimaries, ColorRange, Transfer};

use crate::bits::Bits;

pub const MAX_SPS: usize = 32;
pub const MAX_PPS: usize = 256;
const MAX_REFS: u32 = 16;

#[derive(Clone, Debug)]
pub struct Sps {
    pub id: u8,
    pub log2_max_frame_num: u8,
    pub poc_type: u8,
    pub log2_max_poc_lsb: u8,
    pub delta_pic_order_always_zero: bool,
    pub offset_non_ref: i32,
    pub offset_top_bottom: i32,
    pub offset_for_ref: Vec<i32>,
    pub max_num_ref_frames: u8,
    pub gaps_allowed: bool,
    pub width: u32,
    pub height: u32,
    pub crop_left: u32,
    pub crop_right: u32,
    pub crop_top: u32,
    pub crop_bottom: u32,
    pub direct_8x8_inference: bool,
    pub mb_width: usize,
    pub mb_height: usize,
    pub weight4: [[u8; 16]; 6],
    pub weight8: [[u8; 64]; 6],
    pub range: ColorRange,
    pub primaries: ColorPrimaries,
    pub transfer: Transfer,
    pub matrix: ColorMatrix,
    pub max_reorder: u32,
}

#[derive(Clone, Debug)]
pub struct Pps {
    pub id: u8,
    pub sps_id: u8,
    pub cabac: bool,
    pub bottom_field_pic_order: bool,
    pub num_ref_idx_l0: u8,
    pub num_ref_idx_l1: u8,
    pub weighted_pred: bool,
    pub weighted_bipred: u8,
    pub init_qp: i32,
    pub chroma_qp_offset: i32,
    pub second_chroma_qp_offset: i32,
    pub deblocking_control: bool,
    pub constrained_intra: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8: bool,
    pub weight4: [[u8; 16]; 6],
    pub weight8: [[u8; 64]; 6],
}

impl Sps {
    pub fn visible(&self) -> Result<(u32, u32), CodecError> {
        let crop_x = self
            .crop_left
            .saturating_add(self.crop_right)
            .saturating_mul(2);
        let crop_y = self
            .crop_top
            .saturating_add(self.crop_bottom)
            .saturating_mul(2);
        if crop_x >= self.width || crop_y >= self.height {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "frame crop",
            });
        }
        let width = self.width - crop_x;
        let height = self.height - crop_y;
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "visible size",
            });
        }
        Ok((width, height))
    }
}

pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, CodecError> {
    let mut bits = Bits::new(rbsp);
    let profile = bits.u(8)? as u8;
    let _constraint = bits.u(8)? as u8;
    let _level = bits.u(8)? as u8;
    let id = bits.ue()?;
    if id >= MAX_SPS as u32 {
        return Err(CodecError::Malformed {
            at_bit: bits.pos() as u64,
            what: "sps id",
        });
    }
    let mut chroma_format = 1u8;
    let mut weight4 = flat4();
    let mut weight8 = flat8();
    if matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        chroma_format = bits.ue()? as u8;
        if chroma_format == 3 && bits.bit()? {
            return Err(CodecError::Unsupported {
                what: "separate colour plane",
            });
        }
        let bit_depth_luma = bits.ue()?;
        let bit_depth_chroma = bits.ue()?;
        if bit_depth_luma != 0 || bit_depth_chroma != 0 {
            return Err(CodecError::Unsupported { what: "bit depth" });
        }
        if bits.bit()? {
            return Err(CodecError::Unsupported {
                what: "qpprime transform bypass",
            });
        }
        if bits.bit()? {
            (weight4, weight8) = scaling_lists(&mut bits, true)?;
        }
    }
    if chroma_format != 1 {
        return Err(CodecError::Unsupported {
            what: "chroma format",
        });
    }
    if !matches!(profile, 66 | 77 | 88 | 100 | 110 | 122 | 244) {
        return Err(CodecError::Unsupported { what: "profile" });
    }
    let log2_max_frame_num = (bits.ue()? + 4) as u8;
    if !(4..=16).contains(&log2_max_frame_num) {
        return Err(bits.malformed("log2_max_frame_num"));
    }
    let poc_type = bits.ue()? as u8;
    if poc_type > 2 {
        return Err(bits.malformed("pic_order_cnt_type"));
    }
    let mut log2_max_poc_lsb = 0u8;
    let mut delta_pic_order_always_zero = false;
    let mut offset_non_ref = 0i32;
    let mut offset_top_bottom = 0i32;
    let mut offset_for_ref = Vec::new();
    if poc_type == 0 {
        log2_max_poc_lsb = (bits.ue()? + 4) as u8;
        if !(4..=16).contains(&log2_max_poc_lsb) {
            return Err(bits.malformed("log2_max_pic_order_cnt_lsb"));
        }
    } else if poc_type == 1 {
        delta_pic_order_always_zero = bits.bit()?;
        offset_non_ref = bits.se()?;
        offset_top_bottom = bits.se()?;
        let cycles = bits.ue()?;
        if cycles > 256 {
            return Err(CodecError::Limit {
                what: "pic_order_cnt cycle",
            });
        }
        offset_for_ref.reserve(cycles as usize);
        for _ in 0..cycles {
            offset_for_ref.push(bits.se()?);
        }
    }
    let max_num_ref_frames = bits.ue()?;
    if max_num_ref_frames > MAX_REFS {
        return Err(CodecError::Limit {
            what: "reference frames",
        });
    }
    let gaps_allowed = bits.bit()?;
    let width_mbs = bits.ue()?.saturating_add(1);
    let height_map = bits.ue()?.saturating_add(1);
    if width_mbs > 256 || height_map > 256 {
        return Err(CodecError::Limit {
            what: "picture size",
        });
    }
    let frame_mbs_only = bits.bit()?;
    if !frame_mbs_only {
        return Err(CodecError::Unsupported {
            what: "interlaced picture",
        });
    }
    let width = width_mbs.saturating_mul(16);
    let height = height_map.saturating_mul(16);
    if width > 4096 || height > 4096 || width.saturating_mul(height) > 8_388_608 {
        return Err(CodecError::Limit { what: "coded size" });
    }
    let direct_8x8_inference = bits.bit()?;
    let mut crop_left = 0u32;
    let mut crop_right = 0u32;
    let mut crop_top = 0u32;
    let mut crop_bottom = 0u32;
    if bits.bit()? {
        crop_left = bits.ue()?;
        crop_right = bits.ue()?;
        crop_top = bits.ue()?;
        crop_bottom = bits.ue()?;
    }
    let vui = if bits.bit()? {
        parse_vui(&mut bits, max_num_ref_frames)?
    } else {
        Vui::unspecified(max_num_ref_frames)
    };
    let _ = bits.rbsp_trailing();
    let sps = Sps {
        id: id as u8,
        log2_max_frame_num,
        poc_type,
        log2_max_poc_lsb,
        delta_pic_order_always_zero,
        offset_non_ref,
        offset_top_bottom,
        offset_for_ref,
        max_num_ref_frames: max_num_ref_frames as u8,
        gaps_allowed,
        width,
        height,
        crop_left,
        crop_right,
        crop_top,
        crop_bottom,
        direct_8x8_inference,
        mb_width: width_mbs as usize,
        mb_height: height_map as usize,
        weight4,
        weight8,
        range: vui.range,
        primaries: vui.primaries,
        transfer: vui.transfer,
        matrix: vui.matrix,
        max_reorder: vui.max_reorder,
    };
    sps.visible()?;
    Ok(sps)
}

struct Vui {
    range: ColorRange,
    primaries: ColorPrimaries,
    transfer: Transfer,
    matrix: ColorMatrix,
    max_reorder: u32,
}

impl Vui {
    fn unspecified(max_reorder: u32) -> Self {
        Self {
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
            matrix: ColorMatrix::Unspecified,
            max_reorder,
        }
    }
}

fn parse_vui(bits: &mut Bits<'_>, max_reorder: u32) -> Result<Vui, CodecError> {
    let mut vui = Vui::unspecified(max_reorder);
    if bits.bit()? {
        let aspect = bits.u(8)?;
        if aspect == 255 {
            let _ = bits.u(16)?;
            let _ = bits.u(16)?;
        }
    }
    if bits.bit()? {
        let _ = bits.bit()?;
    }
    if bits.bit()? {
        let _ = bits.u(3)?;
        vui.range = if bits.bit()? {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
        if bits.bit()? {
            vui.primaries = colour(bits.u(8)?);
            vui.transfer = transfer_of(bits.u(8)?);
            vui.matrix = matrix_of(bits.u(8)?);
        }
    }
    if bits.bit()? {
        let _ = bits.ue()?;
        let _ = bits.ue()?;
    }
    if bits.bit()? {
        let _num_units = bits.u(32)?;
        let _time_scale = bits.u(32)?;
        let _ = bits.bit()?;
    }
    let nal_hrd = bits.bit()?;
    if nal_hrd {
        hrd(bits)?;
    }
    let vcl_hrd = bits.bit()?;
    if vcl_hrd {
        hrd(bits)?;
    }
    if nal_hrd || vcl_hrd {
        let _ = bits.bit()?;
    }
    let _ = bits.bit()?;
    if bits.bit()? {
        let _ = bits.bit()?;
        let _ = bits.ue()?;
        let _ = bits.ue()?;
        let _ = bits.ue()?;
        let _ = bits.ue()?;
        vui.max_reorder = bits.ue()?;
        let _ = bits.ue()?;
    }
    Ok(vui)
}

fn hrd(bits: &mut Bits<'_>) -> Result<(), CodecError> {
    let cpb = bits.ue()? + 1;
    if cpb > 32 {
        return Err(CodecError::Limit { what: "cpb count" });
    }
    let bit_rate_scale = bits.u(4)?;
    let cpb_size_scale = bits.u(4)?;
    let _ = (bit_rate_scale, cpb_size_scale);
    for _ in 0..cpb {
        let _ = bits.ue()?;
        let _ = bits.ue()?;
        let _ = bits.bit()?;
    }
    let _ = bits.u(5)?;
    let _ = bits.u(5)?;
    let _ = bits.u(5)?;
    let _ = bits.u(5)?;
    Ok(())
}

fn colour(idc: u32) -> ColorPrimaries {
    match idc {
        1 => ColorPrimaries::Bt709,
        5 | 6 => ColorPrimaries::Bt601,
        9 => ColorPrimaries::Bt2020,
        _ => ColorPrimaries::Unspecified,
    }
}

fn transfer_of(idc: u32) -> Transfer {
    match idc {
        1 | 6 => Transfer::Bt709,
        4 | 5 => Transfer::Bt601,
        14..=16 => Transfer::Bt2020,
        _ => Transfer::Unspecified,
    }
}

fn matrix_of(idc: u32) -> ColorMatrix {
    match idc {
        1 => ColorMatrix::Bt709,
        5 | 6 => ColorMatrix::Bt601,
        9 | 10 => ColorMatrix::Bt2020,
        _ => ColorMatrix::Unspecified,
    }
}

pub fn parse_pps(rbsp: &[u8], sps_of: &[Option<Sps>]) -> Result<Pps, CodecError> {
    let mut bits = Bits::new(rbsp);
    let id = bits.ue()?;
    let sps_id = bits.ue()?;
    if id >= MAX_PPS as u32 || sps_id >= MAX_SPS as u32 {
        return Err(bits.malformed("pps id"));
    }
    let Some(sps) = sps_of[sps_id as usize].as_ref() else {
        return Err(bits.malformed("pps without sps"));
    };
    let cabac = bits.bit()?;
    let bottom = bits.bit()?;
    let groups = bits.ue()?;
    if groups != 0 {
        return Err(CodecError::Unsupported {
            what: "slice groups",
        });
    }
    let num_ref_idx_l0 = (bits.ue()? + 1) as u8;
    let num_ref_idx_l1 = (bits.ue()? + 1) as u8;
    if num_ref_idx_l0 > 16 || num_ref_idx_l1 > 16 {
        return Err(CodecError::Limit { what: "ref idx" });
    }
    let weighted_pred = bits.bit()?;
    let weighted_bipred = bits.u(2)? as u8;
    let init_qp = bits.se()? + 26;
    let _ = bits.se()?;
    let chroma_qp_offset = bits.se()?;
    let deblocking_control = bits.bit()?;
    let constrained_intra = bits.bit()?;
    let redundant = bits.bit()?;
    let mut transform_8x8 = false;
    let mut weight4 = sps.weight4;
    let mut weight8 = sps.weight8;
    let mut second = chroma_qp_offset;
    if bits.more_rbsp() {
        transform_8x8 = bits.bit()?;
        if bits.bit()? {
            (weight4, weight8) = scaling_lists(&mut bits, transform_8x8)?;
        }
        if bits.more_rbsp() {
            second = bits.se()?;
        }
    }
    let _ = bits.rbsp_trailing();
    Ok(Pps {
        id: id as u8,
        sps_id: sps_id as u8,
        cabac,
        bottom_field_pic_order: bottom,
        num_ref_idx_l0,
        num_ref_idx_l1,
        weighted_pred,
        weighted_bipred,
        init_qp,
        chroma_qp_offset,
        second_chroma_qp_offset: second,
        deblocking_control,
        constrained_intra,
        redundant_pic_cnt_present: redundant,
        transform_8x8,
        weight4,
        weight8,
    })
}

type Scaling = ([[u8; 16]; 6], [[u8; 64]; 6]);

fn scaling_lists(bits: &mut Bits<'_>, with8: bool) -> Result<Scaling, CodecError> {
    let mut w4 = flat4();
    let mut w8 = flat8();
    for i in 0..6 {
        if bits.bit()? {
            let mut list = [0u8; 16];
            read_list(bits, &mut list, 16)?;
            if list.iter().all(|v| *v == 0) {
                list = if i < 3 {
                    DEFAULT4_INTRA
                } else {
                    DEFAULT4_INTER
                };
            }
            w4[i] = inverse4(&list);
        } else if i == 0 {
            w4[i] = inverse4(&DEFAULT4_INTRA);
        } else if i == 3 {
            w4[i] = inverse4(&DEFAULT4_INTER);
        } else {
            w4[i] = w4[i - 1];
        }
    }
    if with8 {
        for i in 0..2 {
            if bits.bit()? {
                let mut list = [0u8; 64];
                read_list(bits, &mut list, 64)?;
                if list.iter().all(|v| *v == 0) {
                    list = if i == 0 {
                        DEFAULT8_INTRA
                    } else {
                        DEFAULT8_INTER
                    };
                }
                w8[i] = inverse8(&list);
            } else if i == 0 {
                w8[i] = inverse8(&DEFAULT8_INTRA);
            } else {
                w8[i] = inverse8(&DEFAULT8_INTER);
            }
            w8[i + 2] = w8[i];
            w8[i + 4] = w8[i];
        }
    }
    Ok((w4, w8))
}

fn read_list(bits: &mut Bits<'_>, list: &mut [u8], size: usize) -> Result<(), CodecError> {
    let mut last = 8i32;
    let mut next = 8i32;
    let mut use_default = false;
    for j in 0..size {
        if next != 0 {
            let delta = bits.se()?;
            next = (last + delta).rem_euclid(256);
            if j == 0 && next == 0 {
                use_default = true;
            }
        }
        if use_default {
            list.fill(0);
            return Ok(());
        }
        let value = if next == 0 { last } else { next };
        last = value;
        list[j] = value as u8;
    }
    Ok(())
}

fn inverse4(scan: &[u8; 16]) -> [u8; 16] {
    let mut out = [16u8; 16];
    for (idx, &pos) in ZIGZAG4.iter().enumerate() {
        out[pos] = scan[idx];
    }
    out
}

fn inverse8(scan: &[u8; 64]) -> [u8; 64] {
    let mut out = [16u8; 64];
    for (idx, &pos) in ZIGZAG8.iter().enumerate() {
        out[pos] = scan[idx];
    }
    out
}

fn flat4() -> [[u8; 16]; 6] {
    [[16; 16]; 6]
}

fn flat8() -> [[u8; 64]; 6] {
    [[16; 64]; 6]
}

pub const ZIGZAG4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

pub const ZIGZAG8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const DEFAULT4_INTRA: [u8; 16] = [
    6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42,
];
const DEFAULT4_INTER: [u8; 16] = [
    10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34,
];
const DEFAULT8_INTRA: [u8; 64] = [
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23, 23, 23, 23, 23, 23, 25, 25, 25,
    25, 25, 25, 25, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42,
];
const DEFAULT8_INTER: [u8; 64] = [
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21, 21, 21, 21, 21, 21, 22, 22, 22,
    22, 22, 22, 22, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35,
];
