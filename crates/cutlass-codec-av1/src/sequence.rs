//! Sequence header OBU (specification section 5.5 / markdown 6.4).

use crate::bits::BitReader;
use crate::error::DecodeError;
use crate::limits::DecoderLimits;
use cutlass_core::{ColorPrimaries, ColorRange, ColorSpace, MatrixCoefficients, TransferFunction};

pub const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;
pub const SELECT_INTEGER_MV: u8 = 2;

/// Chroma sample position as signaled. `Reserved` is rejected at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSamplePosition {
    Unknown,
    Vertical,
    Colocated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceHeader {
    pub profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info_present: bool,
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    pub num_ticks_per_picture_minus_1: u32,
    pub decoder_model_info_present: bool,
    pub buffer_delay_length_minus_1: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_time_length_minus_1: u8,
    pub frame_presentation_time_length_minus_1: u8,
    pub initial_display_delay_present: bool,
    pub operating_points_cnt_minus_1: u8,
    pub operating_point_idc: Vec<u16>,
    pub seq_level_idx: Vec<u8>,
    pub seq_tier: Vec<u8>,
    pub decoder_model_present_for_this_op: Vec<bool>,
    pub decoder_buffer_delay: Vec<u32>,
    pub encoder_buffer_delay: Vec<u32>,
    pub low_delay_mode_flag: Vec<bool>,
    pub initial_display_delay_present_for_this_op: Vec<bool>,
    pub initial_display_delay_minus_1: Vec<u8>,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
    pub frame_id_numbers_present: bool,
    pub delta_frame_id_length_minus_2: u8,
    pub additional_frame_id_length_minus_1: u8,
    pub frame_width_bits: u8,
    pub frame_height_bits: u8,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_force_screen_content_tools: u8,
    pub seq_force_integer_mv: u8,
    pub order_hint_bits: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub film_grain_params_present: bool,
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub color: ColorSpace,
    pub color_primaries_raw: u8,
    pub transfer_raw: u8,
    pub matrix_raw: u8,
    pub chroma_sample_position: ChromaSamplePosition,
    pub separate_uv_delta_q: bool,
    /// Index chosen by [`crate::DecoderConfig::operating_point`].
    pub operating_point: u8,
    pub operating_point_idc_selected: u16,
}

impl SequenceHeader {
    pub fn id_len(&self) -> u32 {
        if self.frame_id_numbers_present {
            u32::from(self.additional_frame_id_length_minus_1)
                + u32::from(self.delta_frame_id_length_minus_2)
                + 3
        } else {
            0
        }
    }

    pub fn num_planes(&self) -> usize {
        if self.mono_chrome { 1 } else { 3 }
    }

    /// Same sequence tools and geometry. A repeated header must match.
    pub fn same_sequence(&self, other: &Self) -> bool {
        self.profile == other.profile
            && self.max_frame_width == other.max_frame_width
            && self.max_frame_height == other.max_frame_height
            && self.bit_depth == other.bit_depth
            && self.mono_chrome == other.mono_chrome
            && self.subsampling_x == other.subsampling_x
            && self.subsampling_y == other.subsampling_y
            && self.order_hint_bits == other.order_hint_bits
            && self.frame_id_numbers_present == other.frame_id_numbers_present
            && self.operating_point_idc == other.operating_point_idc
            && self.enable_order_hint == other.enable_order_hint
            && self.use_128x128_superblock == other.use_128x128_superblock
            && self.film_grain_params_present == other.film_grain_params_present
    }
}

pub fn parse(
    payload: &[u8],
    limits: &DecoderLimits,
    operating_point: u8,
) -> Result<SequenceHeader, DecodeError> {
    let mut b = BitReader::new(payload);
    let profile = b.f(3)? as u8;
    if profile > 2 {
        return Err(DecodeError::malformed("seq_profile is reserved"));
    }
    let still_picture = b.f(1)? == 1;
    let reduced = b.f(1)? == 1;
    if reduced && !still_picture {
        return Err(DecodeError::malformed(
            "reduced_still_picture_header requires still_picture",
        ));
    }

    let mut timing_info_present = false;
    let mut num_units_in_display_tick = 1u32;
    let mut time_scale = 1u32;
    let mut equal_picture_interval = false;
    let mut num_ticks_per_picture_minus_1 = 0u32;
    let mut decoder_model_info_present = false;
    let mut buffer_delay_length_minus_1 = 0u8;
    let mut num_units_in_decoding_tick = 1u32;
    let mut buffer_removal_time_length_minus_1 = 0u8;
    let mut frame_presentation_time_length_minus_1 = 0u8;
    let mut initial_display_delay_present = false;
    let op_count;
    let mut operating_point_idc = Vec::new();
    let mut seq_level_idx = Vec::new();
    let mut seq_tier = Vec::new();
    let mut decoder_model_present = Vec::new();
    let mut decoder_buffer_delay = Vec::new();
    let mut encoder_buffer_delay = Vec::new();
    let mut low_delay_mode_flag = Vec::new();
    let mut initial_display_present_op = Vec::new();
    let mut initial_display_delay_minus_1 = Vec::new();

    if reduced {
        let level = b.f(5)? as u8;
        op_count = 0u8;
        operating_point_idc.push(0);
        seq_level_idx.push(level);
        seq_tier.push(0);
        decoder_model_present.push(false);
        decoder_buffer_delay.push(0);
        encoder_buffer_delay.push(0);
        low_delay_mode_flag.push(false);
        initial_display_present_op.push(false);
        initial_display_delay_minus_1.push(0);
    } else {
        timing_info_present = b.f(1)? == 1;
        if timing_info_present {
            num_units_in_display_tick = b.f(32)?;
            time_scale = b.f(32)?;
            if num_units_in_display_tick == 0 || time_scale == 0 {
                return Err(DecodeError::malformed("timing_info is zero"));
            }
            equal_picture_interval = b.f(1)? == 1;
            if equal_picture_interval {
                num_ticks_per_picture_minus_1 = b.uvlc()?;
            }
            decoder_model_info_present = b.f(1)? == 1;
            if decoder_model_info_present {
                buffer_delay_length_minus_1 = b.f(5)? as u8;
                num_units_in_decoding_tick = b.f(32)?;
                buffer_removal_time_length_minus_1 = b.f(5)? as u8;
                frame_presentation_time_length_minus_1 = b.f(5)? as u8;
            }
        }
        initial_display_delay_present = b.f(1)? == 1;
        op_count = b.f(5)? as u8;
        for i in 0..=op_count {
            let idc = b.f(12)? as u16;
            if operating_point_idc.contains(&idc) {
                return Err(DecodeError::malformed("duplicate operating_point_idc"));
            }
            operating_point_idc.push(idc);
            let level = b.f(5)? as u8;
            seq_level_idx.push(level);
            let tier = if level > 7 { b.f(1)? as u8 } else { 0 };
            seq_tier.push(tier);
            let model = if decoder_model_info_present {
                b.f(1)? == 1
            } else {
                false
            };
            decoder_model_present.push(model);
            if model {
                let n = u32::from(buffer_delay_length_minus_1) + 1;
                decoder_buffer_delay.push(b.f(n)?);
                encoder_buffer_delay.push(b.f(n)?);
                low_delay_mode_flag.push(b.f(1)? == 1);
            } else {
                decoder_buffer_delay.push(0);
                encoder_buffer_delay.push(0);
                low_delay_mode_flag.push(false);
            }
            if initial_display_delay_present {
                let present = b.f(1)? == 1;
                initial_display_present_op.push(present);
                if present {
                    initial_display_delay_minus_1.push(b.f(4)? as u8);
                } else {
                    initial_display_delay_minus_1.push(0);
                }
            } else {
                initial_display_present_op.push(false);
                initial_display_delay_minus_1.push(0);
            }
            let _ = i;
        }
    }

    if operating_point as usize > op_count as usize {
        return Err(DecodeError::malformed(
            "operating_point is outside the sequence header",
        ));
    }
    let selected = operating_point_idc[operating_point as usize];

    let frame_width_bits = b.f(4)? as u8 + 1;
    let frame_height_bits = b.f(4)? as u8 + 1;
    let max_frame_width = b.f(u32::from(frame_width_bits))? + 1;
    let max_frame_height = b.f(u32::from(frame_height_bits))? + 1;
    if max_frame_width > limits.max_dimension || max_frame_height > limits.max_dimension {
        return Err(DecodeError::limit(
            "sequence max dimensions exceed max_dimension",
        ));
    }
    let pixels = u64::from(max_frame_width) * u64::from(max_frame_height);
    if pixels > limits.max_pixels {
        return Err(DecodeError::limit("sequence max pixels exceed max_pixels"));
    }

    let frame_id_numbers_present = if reduced { false } else { b.f(1)? == 1 };
    let (delta_frame_id_length_minus_2, additional_frame_id_length_minus_1) =
        if frame_id_numbers_present {
            (b.f(4)? as u8, b.f(3)? as u8)
        } else {
            (0, 0)
        };
    if frame_id_numbers_present {
        let id_len = u32::from(additional_frame_id_length_minus_1)
            + u32::from(delta_frame_id_length_minus_2)
            + 3;
        if id_len > 16 {
            return Err(DecodeError::malformed("frame id length exceeds 16"));
        }
    }

    let use_128 = b.f(1)? == 1;
    let enable_filter_intra = b.f(1)? == 1;
    let enable_intra_edge_filter = b.f(1)? == 1;
    let (
        enable_interintra_compound,
        enable_masked_compound,
        enable_warped_motion,
        enable_dual_filter,
        enable_order_hint,
        enable_jnt_comp,
        enable_ref_frame_mvs,
        seq_force_screen_content_tools,
        seq_force_integer_mv,
        order_hint_bits,
    ) = if reduced {
        (
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            SELECT_SCREEN_CONTENT_TOOLS,
            SELECT_INTEGER_MV,
            0,
        )
    } else {
        let enable_interintra_compound = b.f(1)? == 1;
        let enable_masked_compound = b.f(1)? == 1;
        let enable_warped_motion = b.f(1)? == 1;
        let enable_dual_filter = b.f(1)? == 1;
        let enable_order_hint = b.f(1)? == 1;
        let (enable_jnt_comp, enable_ref_frame_mvs) = if enable_order_hint {
            (b.f(1)? == 1, b.f(1)? == 1)
        } else {
            (false, false)
        };
        let choose_screen = b.f(1)? == 1;
        let seq_force_screen_content_tools = if choose_screen {
            SELECT_SCREEN_CONTENT_TOOLS
        } else {
            b.f(1)? as u8
        };
        let seq_force_integer_mv = if seq_force_screen_content_tools > 0 {
            let choose = b.f(1)? == 1;
            if choose {
                SELECT_INTEGER_MV
            } else {
                b.f(1)? as u8
            }
        } else {
            SELECT_INTEGER_MV
        };
        let order_hint_bits = if enable_order_hint {
            b.f(3)? as u8 + 1
        } else {
            0
        };
        (
            enable_interintra_compound,
            enable_masked_compound,
            enable_warped_motion,
            enable_dual_filter,
            enable_order_hint,
            enable_jnt_comp,
            enable_ref_frame_mvs,
            seq_force_screen_content_tools,
            seq_force_integer_mv,
            order_hint_bits,
        )
    };

    let enable_superres = b.f(1)? == 1;
    let enable_cdef = b.f(1)? == 1;
    let enable_restoration = b.f(1)? == 1;
    let color = parse_color(&mut b, profile)?;
    let film_grain_params_present = b.f(1)? == 1;

    let bits_used = b.position();
    let payload_bits = payload.len() * 8;
    if payload_bits < bits_used {
        return Err(DecodeError::truncated("sequence header"));
    }
    b.trailing_bits((payload_bits - bits_used) as i32)?;

    Ok(SequenceHeader {
        profile,
        still_picture,
        reduced_still_picture_header: reduced,
        timing_info_present,
        num_units_in_display_tick,
        time_scale,
        equal_picture_interval,
        num_ticks_per_picture_minus_1,
        decoder_model_info_present,
        buffer_delay_length_minus_1,
        num_units_in_decoding_tick,
        buffer_removal_time_length_minus_1,
        frame_presentation_time_length_minus_1,
        initial_display_delay_present,
        operating_points_cnt_minus_1: op_count,
        operating_point_idc,
        seq_level_idx,
        seq_tier,
        decoder_model_present_for_this_op: decoder_model_present,
        decoder_buffer_delay,
        encoder_buffer_delay,
        low_delay_mode_flag,
        initial_display_delay_present_for_this_op: initial_display_present_op,
        initial_display_delay_minus_1,
        max_frame_width,
        max_frame_height,
        frame_id_numbers_present,
        delta_frame_id_length_minus_2,
        additional_frame_id_length_minus_1,
        frame_width_bits,
        frame_height_bits,
        use_128x128_superblock: use_128,
        enable_filter_intra,
        enable_intra_edge_filter,
        enable_interintra_compound,
        enable_masked_compound,
        enable_warped_motion,
        enable_dual_filter,
        enable_order_hint,
        enable_jnt_comp,
        enable_ref_frame_mvs,
        seq_force_screen_content_tools,
        seq_force_integer_mv,
        order_hint_bits,
        enable_superres,
        enable_cdef,
        enable_restoration,
        film_grain_params_present,
        bit_depth: color.bit_depth,
        mono_chrome: color.mono,
        subsampling_x: color.subsampling_x,
        subsampling_y: color.subsampling_y,
        color: color.space,
        color_primaries_raw: color.cp,
        transfer_raw: color.tc,
        matrix_raw: color.mc,
        chroma_sample_position: color.csp,
        separate_uv_delta_q: color.separate_uv_delta_q,
        operating_point,
        operating_point_idc_selected: selected,
    })
}

struct ColorParsed {
    bit_depth: u8,
    mono: bool,
    subsampling_x: u8,
    subsampling_y: u8,
    space: ColorSpace,
    cp: u8,
    tc: u8,
    mc: u8,
    csp: ChromaSamplePosition,
    separate_uv_delta_q: bool,
}

fn parse_color(b: &mut BitReader<'_>, profile: u8) -> Result<ColorParsed, DecodeError> {
    let high_bitdepth = b.f(1)? == 1;
    let bit_depth = if profile == 2 && high_bitdepth {
        if b.f(1)? == 1 { 12 } else { 10 }
    } else {
        if high_bitdepth { 10 } else { 8 }
    };
    let mono = if profile == 1 { false } else { b.f(1)? == 1 };
    if mono && profile == 1 {
        return Err(DecodeError::malformed("profile 1 cannot be monochrome"));
    }
    let described = b.f(1)? == 1;
    let (cp, tc, mc) = if described {
        (b.f(8)? as u8, b.f(8)? as u8, b.f(8)? as u8)
    } else {
        (2, 2, 2) // CP/TC/MC unspecified
    };
    let (range, subsampling_x, subsampling_y, csp) = if mono {
        let range = b.f(1)? == 1;
        (range, 1u8, 1u8, ChromaSamplePosition::Unknown)
    } else if cp == 1 && tc == 13 && mc == 0 {
        (true, 0, 0, ChromaSamplePosition::Unknown)
    } else {
        let range = b.f(1)? == 1;
        let (sx, sy) = if profile == 0 {
            (1, 1)
        } else if profile == 1 {
            (0, 0)
        } else if bit_depth == 12 {
            let sx = b.f(1)? as u8;
            let sy = if sx == 1 { b.f(1)? as u8 } else { 0 };
            (sx, sy)
        } else {
            (1, 0)
        };
        let csp = if sx == 1 && sy == 1 {
            match b.f(2)? {
                0 => ChromaSamplePosition::Unknown,
                1 => ChromaSamplePosition::Vertical,
                2 => ChromaSamplePosition::Colocated,
                _ => return Err(DecodeError::malformed("chroma_sample_position reserved")),
            }
        } else {
            ChromaSamplePosition::Unknown
        };
        (range, sx, sy, csp)
    };
    let separate_uv_delta_q = if mono { false } else { b.f(1)? == 1 };

    if mc == 0 && (subsampling_x != 0 || subsampling_y != 0) {
        return Err(DecodeError::malformed("identity matrix requires 4:4:4"));
    }
    validate_profile_layout(profile, bit_depth, mono, subsampling_x, subsampling_y)?;

    let space = ColorSpace {
        primaries: map_primaries(cp),
        transfer: map_transfer(tc),
        matrix: map_matrix(mc),
        range: if range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        },
    };
    Ok(ColorParsed {
        bit_depth,
        mono,
        subsampling_x,
        subsampling_y,
        space,
        cp,
        tc,
        mc,
        csp,
        separate_uv_delta_q,
    })
}

fn validate_profile_layout(
    profile: u8,
    bit_depth: u8,
    mono: bool,
    sx: u8,
    sy: u8,
) -> Result<(), DecodeError> {
    let layout_ok = if mono {
        matches!(profile, 0 | 2)
    } else {
        matches!(
            (profile, bit_depth, sx, sy),
            (0, 8 | 10, 1, 1)
                | (1, 8 | 10, 0, 0)
                | (2, 8 | 10, 1, 0)
                | (2, 12, 1, 1)
                | (2, 12, 1, 0)
                | (2, 12, 0, 0)
        )
    };
    if layout_ok {
        Ok(())
    } else {
        Err(DecodeError::malformed(
            "profile / bit depth / subsampling combination is not legal",
        ))
    }
}

fn map_primaries(cp: u8) -> ColorPrimaries {
    match cp {
        1 => ColorPrimaries::Bt709,
        5 => ColorPrimaries::Bt601_625,
        6 => ColorPrimaries::Bt601_525,
        9 => ColorPrimaries::Bt2020,
        12 => ColorPrimaries::DisplayP3,
        _ => ColorPrimaries::Unspecified,
    }
}

fn map_transfer(tc: u8) -> TransferFunction {
    match tc {
        1 | 6 | 14 | 15 => TransferFunction::Bt709,
        13 => TransferFunction::Srgb,
        16 => TransferFunction::Pq,
        18 => TransferFunction::Hlg,
        _ => TransferFunction::Unspecified,
    }
}

fn map_matrix(mc: u8) -> MatrixCoefficients {
    match mc {
        0 => MatrixCoefficients::Identity,
        1 => MatrixCoefficients::Bt709,
        5 | 6 => MatrixCoefficients::Bt601,
        9 => MatrixCoefficients::Bt2020Ncl,
        _ => MatrixCoefficients::Unspecified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push(bits: &mut Vec<u8>, bit_len: &mut usize, value: u32, n: usize) {
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte = *bit_len / 8;
            if byte == bits.len() {
                bits.push(0);
            }
            if bit == 1 {
                bits[byte] |= 1 << (7 - (*bit_len % 8));
            }
            *bit_len += 1;
        }
    }

    /// Reduced still-picture main profile, 8-bit 4:2:0, 16x16, tools off.
    #[test]
    fn reduced_still_main_8bit() {
        let mut bytes = Vec::new();
        let mut n = 0usize;
        push(&mut bytes, &mut n, 0, 3); // profile
        push(&mut bytes, &mut n, 1, 1); // still
        push(&mut bytes, &mut n, 1, 1); // reduced
        push(&mut bytes, &mut n, 0, 5); // level
        push(&mut bytes, &mut n, 3, 4); // width bits minus 1 -> 4
        push(&mut bytes, &mut n, 3, 4); // height
        push(&mut bytes, &mut n, 15, 4); // width-1
        push(&mut bytes, &mut n, 15, 4);
        push(&mut bytes, &mut n, 0, 1); // 128 sb
        push(&mut bytes, &mut n, 0, 1); // filter intra
        push(&mut bytes, &mut n, 0, 1); // intra edge
        push(&mut bytes, &mut n, 0, 1); // superres
        push(&mut bytes, &mut n, 0, 1); // cdef
        push(&mut bytes, &mut n, 0, 1); // restoration
        push(&mut bytes, &mut n, 0, 1); // high bitdepth
        push(&mut bytes, &mut n, 0, 1); // mono
        push(&mut bytes, &mut n, 0, 1); // color description
        push(&mut bytes, &mut n, 0, 1); // range limited
        push(&mut bytes, &mut n, 0, 2); // csp unknown
        push(&mut bytes, &mut n, 0, 1); // separate uv
        push(&mut bytes, &mut n, 0, 1); // film grain
        // trailing one
        push(&mut bytes, &mut n, 1, 1);
        while !n.is_multiple_of(8) {
            push(&mut bytes, &mut n, 0, 1);
        }
        let seq = parse(&bytes, &DecoderLimits::default(), 0).unwrap();
        assert_eq!(seq.max_frame_width, 16);
        assert_eq!(seq.bit_depth, 8);
        assert!(!seq.mono_chrome);
        assert_eq!((seq.subsampling_x, seq.subsampling_y), (1, 1));
        assert!(seq.reduced_still_picture_header);
        assert_eq!(seq.color.matrix, MatrixCoefficients::Unspecified);
    }
}
