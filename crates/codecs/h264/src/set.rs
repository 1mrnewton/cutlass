//! Sequence and picture parameter sets, clauses 7.3.2.1 and 7.3.2.2.
//!
//! Monochrome is refused. 4:2:0, 4:2:2, and 4:4:4 are accepted, including
//! separate colour planes, field pictures, macroblock-adaptive frame-field,
//! QP′Y-zero transform bypass, and slice groups of every map type.
//! A bit depth outside 8 through 14, or a luma depth that differs from chroma,
//! is refused. A picture that fits no level in Annex A is refused.

use cutlass_decode::{CodecError, ColorMatrix, ColorPrimaries, ColorRange, SampleAspect, Transfer};

use crate::bits::Bits;

pub const MAX_SPS: usize = 32;
pub const MAX_PPS: usize = 256;
const MAX_REFS: u32 = 16;

#[derive(Clone, Debug)]
pub struct Sps {
    pub id: u8,
    /// `level_idc`. Ten times the level number, or 9 for level 1b.
    pub level_idc: u8,
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
    /// `frame_mbs_only_flag`. False selects field or picture-adaptive coding.
    pub frame_mbs_only: bool,
    /// `mb_adaptive_frame_field_flag`. A frame picture is pairs of macroblocks.
    pub mbaff: bool,
    pub mb_width: usize,
    pub mb_height: usize,
    pub weight4: [[u8; 16]; 6],
    pub weight8: [[u8; 64]; 6],
    pub range: ColorRange,
    pub primaries: ColorPrimaries,
    pub transfer: Transfer,
    pub matrix: ColorMatrix,
    pub max_reorder: u32,
    pub bit_depth: u8,
    /// `chroma_format_idc`: 1 is 4:2:0, 2 is 4:2:2, 3 is 4:4:4.
    pub chroma: u8,
    /// `qpprime_y_zero_transform_bypass_flag`.
    pub bypass: bool,
    /// `separate_colour_plane_flag`. Each plane is coded with monochrome syntax.
    pub separate: bool,
    /// `pic_struct_present_flag`.
    pub pic_struct_present: bool,
    /// `num_units_in_tick`. Zero when timing is absent.
    pub num_units: u32,
    /// `time_scale`. Zero when timing is absent.
    pub time_scale: u32,
    /// `fixed_frame_rate_flag`. False when timing is absent.
    pub fixed_frame_rate: bool,
    /// Annex E sample aspect. Unspecified is width 0, height 1.
    pub sample_aspect: SampleAspect,
    /// Bit length of `cpb_removal_delay`. Zero when the HRD is absent.
    pub cpb_delay_bits: u8,
    /// Bit length of `dpb_output_delay`. Zero when the HRD is absent.
    pub dpb_delay_bits: u8,
    /// `time_offset_length`.
    pub time_offset_bits: u8,
    /// `slice_header_restriction_flag`. True for an Annex A SPS, so an
    /// extension slice omits `store_ref_base_pic_flag` and `scan_idx`.
    pub slice_header_restriction: bool,
    /// `inter_layer_deblocking_filter_control_present_flag`.
    pub inter_layer_deblock: bool,
    /// `extended_spatial_scalability_idc`. 0 is the same size as the reference.
    pub svc_spatial: u8,
    /// Picture-level spatial offsets and chroma phases. Each offset is the
    /// coded value; the luma sample offset is twice that value.
    pub svc_scale: SvcScale,
    /// `seq_tcoeff_level_prediction_flag`.
    pub seq_tcoeff: bool,
    /// `adaptive_tcoeff_level_prediction_flag`.
    pub adaptive_tcoeff: bool,
    /// Annex H `view_id` list. Empty for an Annex A or SVC sequence.
    /// Index 0 is the base view.
    pub views: Vec<u16>,
    /// Annex I `depth_view_present_flag`, one entry per `views` index.
    pub depth_views: Vec<bool>,
    /// Annex I `texture_view_present_flag`, one entry per `views` index.
    pub texture_views: Vec<bool>,
    /// Profile 139. The 3D-AVC extension was present.
    pub avc_3d: bool,
    /// `3dv_acquisition_idc`. 0 has no depth ranges. 1 and 2 store them.
    pub acquisition: u8,
    /// `ZNear` for each depth view, in acquisition order. Empty when absent.
    pub depth_near: Vec<i32>,
    /// `ZFar` for each depth view, in acquisition order.
    pub depth_far: Vec<i32>,
    /// `slice_header_prediction_flag`. A 3D slice copies header fields.
    pub header_prediction: bool,
    /// `seq_view_synthesis_flag`. A texture partition can copy an interview picture.
    pub view_synthesis: bool,
    /// `DisparityScale` for depth-view pair `(source, reference)`, row-major.
    /// Empty means every entry is zero.
    pub disparity_scale: Vec<i32>,
    /// `DisparityOffset`, same layout as `disparity_scale`.
    pub disparity_offset: Vec<i32>,
    /// `reduced_resolution_flag`. Depth pictures are half the texture size.
    pub reduced_depth: bool,
    /// Depth picture width in macroblocks. Equal to the texture width when
    /// `reduced_depth` is clear.
    pub depth_mb_width: usize,
    /// Depth picture height in macroblocks.
    pub depth_mb_height: usize,
    /// `depth_hor_mult_minus1 + 1`. 2 when reduced resolution is absent.
    pub depth_hor_mult: u32,
    /// `depth_ver_mult_minus1 + 1`.
    pub depth_ver_mult: u32,
    /// `depth_hor_rsh`. 0 when reduced resolution is absent.
    pub depth_hor_rsh: u32,
    /// `depth_ver_rsh`.
    pub depth_ver_rsh: u32,
    /// `alc_sps_enable_flag`. A texture P macroblock may carry adaptive luminance compensation.
    pub alc: bool,
    /// `enable_rle_skip_flag`. A CABAC B slice may code a run of skipped macroblocks.
    pub rle: bool,
    /// Annex H interview reference views, one entry per `views` index.
    /// View 0 is empty. An Annex A or SVC sequence has no entries.
    pub interview: Vec<Interview>,
}

/// Picture-level extended spatial scalability. Offsets are `se` syntax values.
#[derive(Clone, Debug, Default)]
pub struct SvcScale {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
    /// `chroma_phase_x_plus1_flag` minus 1.
    pub chroma_x: i8,
    /// `chroma_phase_y_plus1` minus 1.
    pub chroma_y: i8,
    /// `seq_ref_layer_chroma_phase_x_plus1_flag` minus 1.
    pub ref_chroma_x: i8,
    /// `seq_ref_layer_chroma_phase_y_plus1` minus 1.
    pub ref_chroma_y: i8,
}

/// Interview reference views for one view. Annex H.
#[derive(Clone, Debug, Default)]
pub struct Interview {
    pub anchor_l0: Vec<u16>,
    pub anchor_l1: Vec<u16>,
    pub non_anchor_l0: Vec<u16>,
    pub non_anchor_l1: Vec<u16>,
}

/// `num_slice_groups_minus1` and the map parameters that follow it.
/// `minus1` of zero is one group covering the picture.
#[derive(Clone, Debug)]
pub struct SliceGroups {
    pub minus1: u8,
    pub map_type: u8,
    pub run: Vec<u32>,
    pub top_left: Vec<u32>,
    pub bottom_right: Vec<u32>,
    pub direction: bool,
    pub rate: u32,
    pub explicit: Vec<u8>,
}

impl SliceGroups {
    pub(crate) fn none() -> Self {
        Self {
            minus1: 0,
            map_type: 0,
            run: Vec::new(),
            top_left: Vec::new(),
            bottom_right: Vec::new(),
            direction: false,
            rate: 1,
            explicit: Vec::new(),
        }
    }
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
    /// `26 + pic_init_qs_minus26`. QSY starts here.
    pub init_qs: i32,
    pub chroma_qp_offset: i32,
    pub second_chroma_qp_offset: i32,
    pub deblocking_control: bool,
    pub constrained_intra: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8: bool,
    pub weight4: [[u8; 16]; 6],
    pub weight8: [[u8; 64]; 6],
    pub groups: SliceGroups,
}

impl Sps {
    /// CropUnitX and CropUnitY from Table 6-1. Monochrome is not stored.
    pub fn crop_unit(&self) -> (u32, u32) {
        let frame = u32::from(self.frame_mbs_only);
        match self.chroma {
            3 => (1, 2 - frame),
            2 => (2, 2 - frame),
            _ => (2, (2 - frame) * 2),
        }
    }

    pub fn visible(&self) -> Result<(u32, u32), CodecError> {
        let (unit_x, unit_y) = self.crop_unit();
        let crop_x = self
            .crop_left
            .saturating_add(self.crop_right)
            .saturating_mul(unit_x);
        let crop_y = self
            .crop_top
            .saturating_add(self.crop_bottom)
            .saturating_mul(unit_y);
        if crop_x >= self.width || crop_y >= self.height {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "frame crop",
            });
        }
        let width = self.width - crop_x;
        let height = self.height - crop_y;
        if (unit_x == 2 && !width.is_multiple_of(2)) || (unit_y == 2 && !height.is_multiple_of(2)) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "visible size",
            });
        }
        Ok((width, height))
    }

    /// The coded picture fits the level named in the SPS, or a higher defined level.
    fn within_level(&self) -> bool {
        let Ok(width_mbs) = u32::try_from(self.mb_width) else {
            return false;
        };
        let Ok(height_mbs) = u32::try_from(self.mb_height) else {
            return false;
        };
        let named = level_max_fs(self.level_idc)
            .is_some_and(|max_fs| macroblocks_fit(width_mbs, height_mbs, max_fs));
        named || frame_within_levels(width_mbs, height_mbs)
    }
}

struct Level {
    idc: u8,
    max_fs: u32,
}

/// Table A-1 `MaxFS`, in macroblocks.
///
/// `level_idc` 9 is level 1b. Level 1b is also `level_idc` 11 with
/// `constraint_set3_flag` on Baseline, Main, and Extended, and that row's
/// MaxFS is 99, the same as level 1. The `level_idc` 11 row here is level 1.1.
const LEVELS: &[Level] = &[
    Level {
        idc: 10,
        max_fs: 99,
    },
    Level { idc: 9, max_fs: 99 },
    Level {
        idc: 11,
        max_fs: 396,
    },
    Level {
        idc: 12,
        max_fs: 396,
    },
    Level {
        idc: 13,
        max_fs: 396,
    },
    Level {
        idc: 20,
        max_fs: 396,
    },
    Level {
        idc: 21,
        max_fs: 792,
    },
    Level {
        idc: 22,
        max_fs: 1_620,
    },
    Level {
        idc: 30,
        max_fs: 1_620,
    },
    Level {
        idc: 31,
        max_fs: 3_600,
    },
    Level {
        idc: 32,
        max_fs: 5_120,
    },
    Level {
        idc: 40,
        max_fs: 8_192,
    },
    Level {
        idc: 41,
        max_fs: 8_192,
    },
    Level {
        idc: 42,
        max_fs: 8_704,
    },
    Level {
        idc: 50,
        max_fs: 22_080,
    },
    Level {
        idc: 51,
        max_fs: 36_864,
    },
    Level {
        idc: 52,
        max_fs: 36_864,
    },
    Level {
        idc: 60,
        max_fs: 139_264,
    },
    Level {
        idc: 61,
        max_fs: 139_264,
    },
    Level {
        idc: 62,
        max_fs: 139_264,
    },
];

fn greatest_max_fs() -> u32 {
    LEVELS.iter().map(|level| level.max_fs).max().unwrap_or(0)
}

/// `MaxFS` for `level_idc` when level 1b's constraint flag is clear.
pub(crate) fn level_max_fs(level_idc: u8) -> Option<u32> {
    LEVELS
        .iter()
        .find(|level| level.idc == level_idc)
        .map(|level| level.max_fs)
}

/// Annex A.3.1 e–g against one level's `MaxFS`.
pub(crate) fn macroblocks_fit(width_mbs: u32, height_mbs: u32, max_fs: u32) -> bool {
    let Some(area) = width_mbs.checked_mul(height_mbs) else {
        return false;
    };
    if area == 0 || area > max_fs {
        return false;
    }
    let Some(bound) = max_fs.checked_mul(8) else {
        return false;
    };
    let Some(wide) = width_mbs.checked_mul(width_mbs) else {
        return false;
    };
    let Some(high) = height_mbs.checked_mul(height_mbs) else {
        return false;
    };
    wide <= bound && high <= bound
}

/// The picture fits at least one level in Table A-1.
pub(crate) fn frame_within_levels(width_mbs: u32, height_mbs: u32) -> bool {
    macroblocks_fit(width_mbs, height_mbs, greatest_max_fs())
}

#[cfg(test)]
fn floor_sqrt(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    loop {
        let next = x.saturating_add(n / x) / 2;
        if next >= x {
            return x;
        }
        x = next;
    }
}

/// Longest coded edge, in samples, that Table A-1 allows.
#[cfg(test)]
pub(crate) fn max_coded_edge() -> u32 {
    floor_sqrt(greatest_max_fs().saturating_mul(8)).saturating_mul(16)
}

/// Greatest `MaxFS` in Table A-1.
#[cfg(test)]
pub(crate) fn max_coded_mbs() -> u32 {
    greatest_max_fs()
}

pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, CodecError> {
    parse_sequence(rbsp, false)
}

/// Subset SPS, NAL type 15. Profiles 83 and 86 carry the SVC extension.
/// Any other profile is `subset sps` and is left for the caller to skip.
pub(crate) fn parse_subset_sps(rbsp: &[u8]) -> Result<Sps, CodecError> {
    parse_sequence(rbsp, true)
}

fn parse_sequence(rbsp: &[u8], subset: bool) -> Result<Sps, CodecError> {
    let mut bits = Bits::new(rbsp);
    let profile = bits.u(8)? as u8;
    let _constraint = bits.u(8)? as u8;
    let level_idc = bits.u(8)? as u8;
    let id = bits.ue()?;
    if id >= MAX_SPS as u32 {
        return Err(CodecError::Malformed {
            at_bit: bits.pos() as u64,
            what: "sps id",
        });
    }
    let mut chroma_format = 1u8;
    let mut bit_depth = 8u8;
    let mut separate = false;
    let mut bypass = false;
    let mut weight4 = flat4();
    let mut weight8 = flat8();
    if matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        chroma_format = bits.ue()? as u8;
        if chroma_format == 3 && bits.bit()? {
            separate = true;
        }
        let bit_depth_luma = bits.ue()?;
        let bit_depth_chroma = bits.ue()?;
        if bit_depth_luma != bit_depth_chroma || bit_depth_luma > 6 {
            return Err(CodecError::Unsupported { what: "bit depth" });
        }
        bit_depth = (8 + bit_depth_luma) as u8;
        bypass = bits.bit()?;
        if bits.bit()? {
            (weight4, weight8) = scaling_lists(&mut bits, true, chroma_format == 3)?;
        }
    }
    if chroma_format == 0 || chroma_format > 3 {
        return Err(CodecError::Unsupported {
            what: "chroma format",
        });
    }
    let scalable = matches!(profile, 83 | 86);
    let multiview = matches!(profile, 118 | 128);
    let depth = matches!(profile, 135 | 138);
    let three_d = profile == 139;
    if subset {
        if !scalable && !multiview && !depth && !three_d {
            return Err(CodecError::Unsupported { what: "subset sps" });
        }
    } else if !matches!(profile, 66 | 77 | 88 | 100 | 110 | 122 | 244) {
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
    // Reject an edge no level allows before reading the rest of the SPS.
    if !frame_within_levels(width_mbs, 1) || !frame_within_levels(1, height_map) {
        return Err(CodecError::Limit {
            what: "picture size",
        });
    }
    let frame_mbs_only = bits.bit()?;
    let mbaff = if frame_mbs_only { false } else { bits.bit()? };
    let height_mbs = if frame_mbs_only {
        height_map
    } else {
        height_map.saturating_mul(2)
    };
    let width = width_mbs.saturating_mul(16);
    let height = height_mbs.saturating_mul(16);
    let direct_8x8_inference = bits.bit()?;
    if !frame_mbs_only && !direct_8x8_inference {
        return Err(bits.malformed("direct 8x8 inference"));
    }
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
    let plain = SvcSyntax {
        inter_layer_deblock: false,
        spatial: 0,
        seq_tcoeff: false,
        adaptive_tcoeff: false,
        restriction: true,
        scale: SvcScale::default(),
    };
    let mut avc_3d = false;
    let mut acquisition = 0u8;
    let mut depth_near = Vec::new();
    let mut depth_far = Vec::new();
    let mut header_prediction = false;
    let mut view_synthesis = false;
    let mut disparity_scale = Vec::new();
    let mut disparity_offset = Vec::new();
    let mut reduced_depth = false;
    let mut depth_mb_width = usize::try_from(width_mbs).unwrap_or(0);
    let mut depth_mb_height = usize::try_from(height_mbs).unwrap_or(0);
    let mut depth_hor_mult = 2u32;
    let mut depth_ver_mult = 2u32;
    let mut depth_hor_rsh = 0u32;
    let mut depth_ver_rsh = 0u32;
    let mut alc = false;
    let mut rle = false;
    let mut texture_views = Vec::new();
    let (svc, views, depth_views, interview) = if subset && scalable {
        let syntax = svc_extension(&mut bits, if separate { 0 } else { chroma_format })?;
        if bits.bit()? {
            return Err(CodecError::Unsupported { what: "svc vui" });
        }
        (syntax, Vec::new(), Vec::new(), Vec::new())
    } else if subset && multiview {
        if !bits.bit()? {
            return Err(bits.malformed("mvc"));
        }
        let (views, interview) = mvc_extension(&mut bits)?;
        if bits.bit()? {
            return Err(CodecError::Unsupported { what: "mvc vui" });
        }
        let depth_views = vec![false; views.len()];
        texture_views = vec![true; views.len()];
        (plain, views, depth_views, interview)
    } else if subset && three_d {
        if !bits.bit()? {
            return Err(bits.malformed("mvcd"));
        }
        let mvcd = mvcd_extension(&mut bits)?;
        if bits.bit()? || bits.bit()? {
            return Err(CodecError::Unsupported { what: "mvcd vui" });
        }
        let axis = three_d_extension(
            &mut bits,
            &mvcd.views,
            &mvcd.depth_views,
            &mvcd.texture_views,
            (width_mbs, height_mbs),
            frame_mbs_only,
        )?;
        acquisition = axis.acquisition;
        depth_near = axis.near;
        depth_far = axis.far;
        header_prediction = axis.header_prediction;
        view_synthesis = axis.view_synthesis;
        alc = axis.alc;
        rle = axis.rle;
        disparity_scale = axis.disparity_scale;
        disparity_offset = axis.disparity_offset;
        if axis.reduced {
            reduced_depth = true;
            depth_mb_width = axis.depth_mb_width;
            depth_mb_height = axis.depth_mb_height;
            depth_hor_mult = axis.hor_mult;
            depth_ver_mult = axis.ver_mult;
            depth_hor_rsh = axis.hor_rsh;
            depth_ver_rsh = axis.ver_rsh;
        }
        avc_3d = true;
        texture_views = mvcd.texture_views;
        let views = mvcd.views;
        let depth_views = mvcd.depth_views;
        let paired = depth_views
            .iter()
            .zip(&texture_views)
            .skip(1)
            .all(|(depth, texture)| *depth && *texture);
        let interview = if paired { mvcd.interview } else { Vec::new() };
        (plain, views, depth_views, interview)
    } else if subset {
        if !bits.bit()? {
            return Err(bits.malformed("mvcd"));
        }
        let mvcd = mvcd_extension(&mut bits)?;
        if bits.bit()? || bits.bit()? {
            return Err(CodecError::Unsupported { what: "mvcd vui" });
        }
        texture_views = mvcd.texture_views;
        let views = mvcd.views;
        let depth_views = mvcd.depth_views;
        (plain, views, depth_views, Vec::new())
    } else {
        (plain, Vec::new(), Vec::new(), Vec::new())
    };
    if subset && bits.bit()? {
        while bits.more_rbsp() {
            let _ = bits.bit()?;
        }
    }
    let _ = bits.rbsp_trailing();
    let sps = Sps {
        id: id as u8,
        level_idc,
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
        frame_mbs_only,
        mbaff,
        mb_width: width_mbs as usize,
        mb_height: height_mbs as usize,
        weight4,
        weight8,
        range: vui.range,
        primaries: vui.primaries,
        transfer: vui.transfer,
        matrix: vui.matrix,
        max_reorder: vui.max_reorder,
        bit_depth,
        chroma: chroma_format,
        bypass,
        separate,
        pic_struct_present: vui.pic_struct_present,
        num_units: vui.num_units,
        time_scale: vui.time_scale,
        fixed_frame_rate: vui.fixed_frame_rate,
        sample_aspect: vui.sample_aspect,
        cpb_delay_bits: vui.cpb_delay_bits,
        dpb_delay_bits: vui.dpb_delay_bits,
        time_offset_bits: vui.time_offset_bits,
        slice_header_restriction: svc.restriction,
        inter_layer_deblock: svc.inter_layer_deblock,
        svc_spatial: svc.spatial,
        svc_scale: svc.scale,
        seq_tcoeff: svc.seq_tcoeff,
        adaptive_tcoeff: svc.adaptive_tcoeff,
        views,
        depth_views,
        texture_views,
        avc_3d,
        acquisition,
        depth_near,
        depth_far,
        header_prediction,
        view_synthesis,
        disparity_scale,
        disparity_offset,
        reduced_depth,
        depth_mb_width,
        depth_mb_height,
        depth_hor_mult,
        depth_ver_mult,
        depth_hor_rsh,
        depth_ver_rsh,
        alc,
        rle,
        interview,
    };
    if !sps.within_level() {
        return Err(CodecError::Limit {
            what: "picture size",
        });
    }
    sps.visible()?;
    Ok(sps)
}

struct SvcSyntax {
    inter_layer_deblock: bool,
    spatial: u8,
    seq_tcoeff: bool,
    adaptive_tcoeff: bool,
    restriction: bool,
    scale: SvcScale,
}

/// `seq_parameter_set_svc_extension`.
fn svc_extension(bits: &mut Bits<'_>, chroma_array: u8) -> Result<SvcSyntax, CodecError> {
    let inter_layer_deblock = bits.bit()?;
    let spatial = match bits.u(2)? {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => {
            return Err(bits.malformed("extended spatial scalability"));
        }
    };
    let mut chroma_x = -1i8;
    let mut chroma_y = 0i8;
    if chroma_array == 1 || chroma_array == 2 {
        chroma_x = if bits.bit()? { 0 } else { -1 };
    }
    if chroma_array == 1 {
        chroma_y = chroma_phase_y(bits)?;
    }
    let mut scale = SvcScale {
        chroma_x,
        chroma_y,
        ref_chroma_x: chroma_x,
        ref_chroma_y: chroma_y,
        ..SvcScale::default()
    };
    if spatial == 1 {
        if chroma_array > 0 {
            scale.ref_chroma_x = if bits.bit()? { 0 } else { -1 };
            scale.ref_chroma_y = chroma_phase_y(bits)?;
        }
        scale.left = bits.se()?;
        scale.top = bits.se()?;
        scale.right = bits.se()?;
        scale.bottom = bits.se()?;
    }
    let seq_tcoeff = bits.bit()?;
    let adaptive_tcoeff = if seq_tcoeff { bits.bit()? } else { false };
    let restriction = bits.bit()?;
    Ok(SvcSyntax {
        inter_layer_deblock,
        spatial,
        seq_tcoeff,
        adaptive_tcoeff,
        restriction,
        scale,
    })
}

/// `chroma_phase_y_plus1` is 0, 1, or 2. The phase is that value minus 1.
fn chroma_phase_y(bits: &mut Bits<'_>) -> Result<i8, CodecError> {
    let plus1 = bits.u(2)?;
    if plus1 > 2 {
        return Err(bits.malformed("chroma phase"));
    }
    Ok(plus1 as i8 - 1)
}

fn mvc_limit(bits: &mut Bits<'_>, max: u32) -> Result<u32, CodecError> {
    let value = bits.ue()?;
    if value > max {
        return Err(CodecError::Limit { what: "mvc" });
    }
    Ok(value)
}

/// `seq_parameter_set_mvc_extension`. The view list and its interview
/// reference views are kept.
fn mvc_extension(bits: &mut Bits<'_>) -> Result<(Vec<u16>, Vec<Interview>), CodecError> {
    let views_minus1 = mvc_limit(bits, 1023)?;
    let mut views = Vec::new();
    for _ in 0..=views_minus1 {
        let id = mvc_limit(bits, 1023)?;
        let Ok(id) = u16::try_from(id) else {
            return Err(CodecError::Limit { what: "mvc" });
        };
        views.push(id);
    }
    let mut interview = vec![Interview::default()];
    for _ in 1..=views_minus1 {
        interview.push(Interview {
            anchor_l0: read_view_refs(bits)?,
            anchor_l1: read_view_refs(bits)?,
            non_anchor_l0: Vec::new(),
            non_anchor_l1: Vec::new(),
        });
    }
    for index in 1..=views_minus1 {
        let Some(refs) = interview.get_mut(index as usize) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "mvc",
            });
        };
        refs.non_anchor_l0 = read_view_refs(bits)?;
        refs.non_anchor_l1 = read_view_refs(bits)?;
    }
    let levels = mvc_limit(bits, 63)?;
    for _ in 0..=levels {
        let _ = bits.u(8)?;
        let ops = mvc_limit(bits, 1023)?;
        for _ in 0..=ops {
            let _ = bits.u(3)?;
            let targets = mvc_limit(bits, 1023)?;
            for _ in 0..=targets {
                let _ = mvc_limit(bits, 1023)?;
            }
            let _ = mvc_limit(bits, 1023)?;
        }
    }
    Ok((views, interview))
}

fn read_view_refs(bits: &mut Bits<'_>) -> Result<Vec<u16>, CodecError> {
    let count = mvc_limit(bits, 15)?;
    let mut refs = Vec::new();
    for _ in 0..count {
        let id = mvc_limit(bits, 1023)?;
        let Ok(id) = u16::try_from(id) else {
            return Err(CodecError::Limit { what: "mvc" });
        };
        refs.push(id);
    }
    Ok(refs)
}

/// `seq_parameter_set_mvcd_extension`. Depth-view ids and their interview
/// lists are kept. The level operation points are consumed.
struct MvcdViews {
    views: Vec<u16>,
    depth_views: Vec<bool>,
    texture_views: Vec<bool>,
    interview: Vec<Interview>,
}

fn mvcd_extension(bits: &mut Bits<'_>) -> Result<MvcdViews, CodecError> {
    let views_minus1 = mvc_limit(bits, 1023)?;
    let mut views = Vec::new();
    let mut depth_views = Vec::new();
    let mut texture_views = Vec::new();
    let mut interview = Vec::new();
    for _ in 0..=views_minus1 {
        let id = mvc_limit(bits, 1023)?;
        let Ok(id) = u16::try_from(id) else {
            return Err(CodecError::Limit { what: "mvc" });
        };
        let depth = bits.bit()?;
        let texture = bits.bit()?;
        if !depth && !texture {
            return Err(bits.malformed("mvcd"));
        }
        views.push(id);
        depth_views.push(depth);
        texture_views.push(texture);
        interview.push(Interview::default());
    }
    for index in 1..=views_minus1 {
        let depth = depth_views.get(index as usize).copied().unwrap_or(false);
        if depth {
            let Some(refs) = interview.get_mut(index as usize) else {
                return Err(bits.malformed("mvcd"));
            };
            refs.anchor_l0 = read_view_refs(bits)?;
            refs.anchor_l1 = read_view_refs(bits)?;
        }
    }
    for index in 1..=views_minus1 {
        let depth = depth_views.get(index as usize).copied().unwrap_or(false);
        if depth {
            let Some(refs) = interview.get_mut(index as usize) else {
                return Err(bits.malformed("mvcd"));
            };
            refs.non_anchor_l0 = read_view_refs(bits)?;
            refs.non_anchor_l1 = read_view_refs(bits)?;
        }
    }
    let levels = mvc_limit(bits, 63)?;
    for _ in 0..=levels {
        let _ = bits.u(8)?;
        let ops = mvc_limit(bits, 1023)?;
        for _ in 0..=ops {
            let _ = bits.u(3)?;
            let targets = mvc_limit(bits, 1023)?;
            for _ in 0..=targets {
                let _ = mvc_limit(bits, 1023)?;
                let _ = bits.bit()?;
                let _ = bits.bit()?;
            }
            let _ = mvc_limit(bits, 1023)?;
            let _ = mvc_limit(bits, 1023)?;
        }
    }
    Ok(MvcdViews {
        views,
        depth_views,
        texture_views,
        interview,
    })
}

struct DepthAxis {
    acquisition: u8,
    near: Vec<i32>,
    far: Vec<i32>,
    header_prediction: bool,
    view_synthesis: bool,
    disparity_scale: Vec<i32>,
    disparity_offset: Vec<i32>,
    reduced: bool,
    depth_mb_width: usize,
    depth_mb_height: usize,
    hor_mult: u32,
    ver_mult: u32,
    hor_rsh: u32,
    ver_rsh: u32,
    alc: bool,
    rle: bool,
}

struct ReducedDepth {
    mb_width: usize,
    mb_height: usize,
    hor_mult: u32,
    ver_mult: u32,
    hor_rsh: u32,
    ver_rsh: u32,
}

/// `reduced_resolution_flag` 1. The depth picture is half the texture picture
/// on both axes. The multipliers and shifts feed J.8.2.1.1.
fn reduced_depth(bits: &mut Bits<'_>, coded: (u32, u32)) -> Result<ReducedDepth, CodecError> {
    let mb_width = depth_mbs(bits)?;
    let mb_height = depth_mbs(bits)?;
    let hor_mult = depth_factor(bits, 1023)?.saturating_add(1);
    let ver_mult = depth_factor(bits, 1023)?.saturating_add(1);
    let hor_rsh = depth_factor(bits, 31)?;
    let ver_rsh = depth_factor(bits, 31)?;
    let (texture_w, texture_h) = coded;
    if mb_width.saturating_mul(2) != usize::try_from(texture_w).unwrap_or(usize::MAX)
        || mb_height.saturating_mul(2) != usize::try_from(texture_h).unwrap_or(usize::MAX)
    {
        return Err(bits.malformed("3d"));
    }
    Ok(ReducedDepth {
        mb_width,
        mb_height,
        hor_mult,
        ver_mult,
        hor_rsh,
        ver_rsh,
    })
}

fn depth_mbs(bits: &mut Bits<'_>) -> Result<usize, CodecError> {
    let minus = bits.ue()?;
    let count = minus.checked_add(1).ok_or_else(|| bits.malformed("3d"))?;
    usize::try_from(count).map_err(|_| bits.malformed("3d"))
}

fn depth_factor(bits: &mut Bits<'_>, limit: u32) -> Result<u32, CodecError> {
    let value = bits.ue()?;
    if value > limit {
        return Err(bits.malformed("3d"));
    }
    Ok(value)
}

/// `seq_parameter_set_3davc_extension`. Header prediction, view synthesis, and a
/// half-resolution depth picture are stored. Acquisition 1 and 2 store integer
/// depth ranges and the disparity pair. Adaptive luminance compensation and
/// run-length skip are stored. Cropping and a grid stay refused.
fn three_d_extension(
    bits: &mut Bits<'_>,
    views: &[u16],
    depth_views: &[bool],
    texture_views: &[bool],
    coded: (u32, u32),
    frame_only: bool,
) -> Result<DepthAxis, CodecError> {
    let depth_ids: Vec<u16> = views
        .iter()
        .zip(depth_views)
        .filter_map(|(id, depth)| depth.then_some(*id))
        .collect();
    let mut axis = DepthAxis {
        acquisition: 0,
        near: Vec::new(),
        far: Vec::new(),
        header_prediction: false,
        view_synthesis: false,
        disparity_scale: Vec::new(),
        disparity_offset: Vec::new(),
        reduced: false,
        depth_mb_width: 0,
        depth_mb_height: 0,
        hor_mult: 2,
        ver_mult: 2,
        hor_rsh: 0,
        ver_rsh: 0,
        alc: false,
        rle: false,
    };
    if !depth_ids.is_empty() {
        let acquisition = bits.ue()?;
        if acquisition > 2 {
            return Err(CodecError::Unsupported { what: "3d" });
        }
        axis.acquisition = acquisition as u8;
        for id in &depth_ids {
            let coded = mvc_limit(bits, 1023)?;
            if u16::try_from(coded).ok() != Some(*id) {
                return Err(bits.malformed("3d"));
            }
        }
        if acquisition != 0 {
            let (near, far) = depth_ranges(bits, depth_ids.len())?;
            let (scale, offset) = view_synthesis(bits, depth_ids.len())?;
            axis.near = near;
            axis.far = far;
            axis.disparity_scale = scale;
            axis.disparity_offset = offset;
        }
        if bits.bit()? {
            if !frame_only {
                return Err(CodecError::Unsupported { what: "3d" });
            }
            let depth = reduced_depth(bits, coded)?;
            axis.reduced = true;
            axis.depth_mb_width = depth.mb_width;
            axis.depth_mb_height = depth.mb_height;
            axis.hor_mult = depth.hor_mult;
            axis.ver_mult = depth.ver_mult;
            axis.hor_rsh = depth.hor_rsh;
            axis.ver_rsh = depth.ver_rsh;
        }
        if bits.bit()? || bits.ue()? != 0 {
            return Err(CodecError::Unsupported { what: "3d" });
        }
        axis.header_prediction = bits.bit()?;
        axis.view_synthesis = bits.bit()?;
    }
    axis.alc = bits.bit()?;
    axis.rle = bits.bit()?;
    let paired = depth_views
        .iter()
        .zip(texture_views)
        .skip(1)
        .all(|(depth, texture)| *depth && *texture);
    if !paired {
        for _ in 0..2 {
            for texture in texture_views.iter().skip(1) {
                if *texture {
                    let _ = read_view_refs(bits)?;
                    let _ = read_view_refs(bits)?;
                }
            }
        }
    }
    Ok(axis)
}

/// One depth parameter set, NAL type 16. `pred_direction` 2 stores absolute ranges.
/// A view-synthesis pair replaces the sequence disparity, and a nonlinear model
/// replaces the identity `NdrInverse` table.
#[derive(Clone)]
pub(crate) struct DepthSet {
    pub id: u8,
    pub near: Vec<i32>,
    pub far: Vec<i32>,
    pub disparity_scale: Vec<i32>,
    pub disparity_offset: Vec<i32>,
    pub ndr: [u8; 256],
}

pub(crate) fn parse_dps(rbsp: &[u8]) -> Result<DepthSet, CodecError> {
    let mut bits = Bits::new(rbsp);
    let id = bits.ue()?;
    if id == 0 || id > 63 {
        return Err(bits.malformed("3d"));
    }
    if bits.ue()? != 2 {
        return Err(CodecError::Unsupported { what: "3d" });
    }
    let views = bits.ue()?.saturating_add(1) as usize;
    if views > 1024 {
        return Err(CodecError::Limit { what: "3d" });
    }
    let (near, far) = depth_ranges(&mut bits, views)?;
    let (disparity_scale, disparity_offset) = if bits.bit()? {
        view_synthesis(&mut bits, views)?
    } else {
        (Vec::new(), Vec::new())
    };
    if bits.bit()? {
        return Err(CodecError::Unsupported { what: "3d" });
    }
    let count = usize::try_from(bits.ue()?).unwrap_or(usize::MAX);
    if count > 62 {
        return Err(CodecError::Unsupported { what: "3d" });
    }
    let mut model = Vec::with_capacity(count);
    for _ in 0..count {
        let value = bits.ue()?;
        if value > 65_535 {
            return Err(CodecError::Unsupported { what: "3d" });
        }
        model.push(value);
    }
    let ndr = ndr_inverse(&model)?;
    bits.rbsp_trailing()?;
    Ok(DepthSet {
        id: id as u8,
        near,
        far,
        disparity_scale,
        disparity_offset,
        ndr,
    })
}

/// J-11. `model` is `nonlinear_depth_representation_model[1..=num]`. An empty
/// model is the identity table.
fn ndr_inverse(model: &[u32]) -> Result<[u8; 256], CodecError> {
    let num = model.len();
    let mut nodes = vec![0u64; num.saturating_add(2)];
    for (index, value) in model.iter().copied().enumerate() {
        if let Some(slot) = nodes.get_mut(index.saturating_add(1)) {
            *slot = u64::from(value);
        }
    }
    let mut inverse = identity_ndr();
    let segments = num.saturating_add(1);
    for k in 0..=num {
        let pos1 = (255 * k) / segments;
        let pos2 = (255 * k.saturating_add(1)) / segments;
        let dev1 = i64::try_from(nodes.get(k).copied().unwrap_or(0)).unwrap_or(0);
        let dev2 = i64::try_from(nodes.get(k.saturating_add(1)).copied().unwrap_or(0)).unwrap_or(0);
        let x1 = i64::try_from(pos1).unwrap_or(0) - dev1;
        let y1 = i64::try_from(pos1).unwrap_or(0) + dev1;
        let x2 = i64::try_from(pos2).unwrap_or(0) - dev2;
        let y2 = i64::try_from(pos2).unwrap_or(0) + dev2;
        if x2 < x1 || x2 < 0 || x1 > 255 {
            continue;
        }
        let den = x2 - x1;
        if den == 0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "3d",
            });
        }
        let start = usize::try_from(x1.max(0)).unwrap_or(0);
        let end = usize::try_from(x2.min(255)).unwrap_or(0);
        for (x, slot) in inverse.iter_mut().enumerate().take(end + 1).skip(start) {
            let sample = i64::try_from(x).unwrap_or(0);
            let value = round_div((sample - x1) * (y2 - y1), den) + y1;
            *slot = u8::try_from(value.clamp(0, 255)).unwrap_or(0);
        }
    }
    Ok(inverse)
}

pub(crate) fn identity_ndr() -> [u8; 256] {
    let mut inverse = [0u8; 256];
    for (index, value) in inverse.iter_mut().enumerate() {
        *value = u8::try_from(index).unwrap_or(0);
    }
    inverse
}

/// `Round` of an integer ratio. The denominator is positive.
fn round_div(num: i64, den: i64) -> i64 {
    if num >= 0 {
        num.saturating_add(den / 2) / den
    } else {
        -(num.saturating_neg().saturating_add(den / 2) / den)
    }
}

/// `depth_ranges` with `predDirection` 2. A clear flag has no value to copy.
fn depth_ranges(bits: &mut Bits<'_>, views: usize) -> Result<(Vec<i32>, Vec<i32>), CodecError> {
    let near_flag = bits.bit()?;
    let far_flag = bits.bit()?;
    if !near_flag || !far_flag {
        return Err(CodecError::Unsupported { what: "3d" });
    }
    Ok((depth_axis(bits, views)?, depth_axis(bits, views)?))
}

fn depth_axis(bits: &mut Bits<'_>, views: usize) -> Result<Vec<i32>, CodecError> {
    let equal = if views > 1 { bits.bit()? } else { true };
    let count = if equal { 1 } else { views };
    let mut values = Vec::with_capacity(count);
    let mut man_len = 0u32;
    for index in 0..count {
        if index == 0 {
            man_len = bits.u(5)?.saturating_add(1);
        }
        let sign = bits.bit()?;
        let exp = bits.u(7)?;
        let mantissa = if man_len == 0 { 0 } else { bits.u(man_len)? };
        values.push(depth_value(sign, exp, mantissa, man_len)?);
    }
    if equal {
        let Some(first) = values.first().copied() else {
            return Err(CodecError::Unsupported { what: "3d" });
        };
        values.resize(views, first);
    }
    Ok(values)
}

/// J.8.4, kept only when the depth is an integer.
fn depth_value(sign: bool, exp: u32, mantissa: u32, man_len: u32) -> Result<i32, CodecError> {
    if exp == 0 || exp >= 127 || man_len > 30 {
        return Err(CodecError::Unsupported { what: "3d" });
    }
    let shift = exp as i32 - 31;
    let base = (1i64 << man_len) + i64::from(mantissa);
    let magnitude = if shift >= man_len as i32 {
        let up = shift - man_len as i32;
        if up > 30 {
            return Err(CodecError::Unsupported { what: "3d" });
        }
        base << up
    } else {
        let down = man_len as i32 - shift;
        let mask = (1i64 << down) - 1;
        if base & mask != 0 {
            return Err(CodecError::Unsupported { what: "3d" });
        }
        base >> down
    };
    let value = if sign { -magnitude } else { magnitude };
    i32::try_from(value).map_err(|_| CodecError::Unsupported { what: "3d" })
}

/// `vsp_param`. One depth view writes no bits. A pair stores the four differences.
fn view_synthesis(bits: &mut Bits<'_>, views: usize) -> Result<(Vec<i32>, Vec<i32>), CodecError> {
    let mut scale = vec![0; views.saturating_mul(views)];
    let mut offset = vec![0; views.saturating_mul(views)];
    for i in 0..views {
        for j in 0..i {
            let wji = signed_ue(bits)?;
            let oji = signed_ue(bits)?;
            let wij = signed_ue(bits)?;
            let oij = signed_ue(bits)?;
            let forward = j.saturating_mul(views).saturating_add(i);
            let back = i.saturating_mul(views).saturating_add(j);
            if let Some(slot) = scale.get_mut(forward) {
                *slot = wji;
            }
            if let Some(slot) = offset.get_mut(forward) {
                *slot = oji;
            }
            if let Some(slot) = scale.get_mut(back) {
                *slot = wij.saturating_sub(wji);
            }
            if let Some(slot) = offset.get_mut(back) {
                *slot = oij.saturating_sub(oji);
            }
        }
    }
    Ok((scale, offset))
}

fn signed_ue(bits: &mut Bits<'_>) -> Result<i32, CodecError> {
    let value = bits.ue()?;
    i32::try_from(value).map_err(|_| CodecError::Unsupported { what: "3d" })
}

struct Vui {
    range: ColorRange,
    primaries: ColorPrimaries,
    transfer: Transfer,
    matrix: ColorMatrix,
    max_reorder: u32,
    pic_struct_present: bool,
    num_units: u32,
    time_scale: u32,
    fixed_frame_rate: bool,
    sample_aspect: SampleAspect,
    cpb_delay_bits: u8,
    dpb_delay_bits: u8,
    time_offset_bits: u8,
}

impl Vui {
    fn unspecified(max_reorder: u32) -> Self {
        Self {
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
            matrix: ColorMatrix::Unspecified,
            max_reorder,
            pic_struct_present: false,
            num_units: 0,
            time_scale: 0,
            fixed_frame_rate: false,
            sample_aspect: SampleAspect::UNSPECIFIED,
            cpb_delay_bits: 0,
            dpb_delay_bits: 0,
            time_offset_bits: 0,
        }
    }
}

fn parse_vui(bits: &mut Bits<'_>, max_reorder: u32) -> Result<Vui, CodecError> {
    let mut vui = Vui::unspecified(max_reorder);
    if bits.bit()? {
        vui.sample_aspect = sample_aspect(bits)?;
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
        vui.num_units = bits.u(32)?;
        vui.time_scale = bits.u(32)?;
        vui.fixed_frame_rate = bits.bit()?;
    }
    let nal_hrd = bits.bit()?;
    if nal_hrd {
        let lengths = hrd(bits)?;
        vui.cpb_delay_bits = lengths.0;
        vui.dpb_delay_bits = lengths.1;
        vui.time_offset_bits = lengths.2;
    }
    let vcl_hrd = bits.bit()?;
    if vcl_hrd {
        let lengths = hrd(bits)?;
        if nal_hrd && (vui.cpb_delay_bits, vui.dpb_delay_bits, vui.time_offset_bits) != lengths {
            return Err(bits.malformed("hrd"));
        }
        vui.cpb_delay_bits = lengths.0;
        vui.dpb_delay_bits = lengths.1;
        vui.time_offset_bits = lengths.2;
    }
    if nal_hrd || vcl_hrd {
        let _ = bits.bit()?;
    }
    vui.pic_struct_present = bits.bit()?;
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

fn hrd(bits: &mut Bits<'_>) -> Result<(u8, u8, u8), CodecError> {
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
    let _initial = bits.u(5)?;
    let cpb_bits = bits.u(5)? + 1;
    let dpb_bits = bits.u(5)? + 1;
    let offset_bits = bits.u(5)?;
    Ok((cpb_bits as u8, dpb_bits as u8, offset_bits as u8))
}

const ASPECT: [SampleAspect; 17] = [
    SampleAspect::UNSPECIFIED,
    SampleAspect {
        width: 1,
        height: 1,
    },
    SampleAspect {
        width: 12,
        height: 11,
    },
    SampleAspect {
        width: 10,
        height: 11,
    },
    SampleAspect {
        width: 16,
        height: 11,
    },
    SampleAspect {
        width: 40,
        height: 33,
    },
    SampleAspect {
        width: 24,
        height: 11,
    },
    SampleAspect {
        width: 20,
        height: 11,
    },
    SampleAspect {
        width: 32,
        height: 11,
    },
    SampleAspect {
        width: 80,
        height: 33,
    },
    SampleAspect {
        width: 18,
        height: 11,
    },
    SampleAspect {
        width: 15,
        height: 11,
    },
    SampleAspect {
        width: 64,
        height: 33,
    },
    SampleAspect {
        width: 160,
        height: 99,
    },
    SampleAspect {
        width: 4,
        height: 3,
    },
    SampleAspect {
        width: 3,
        height: 2,
    },
    SampleAspect {
        width: 2,
        height: 1,
    },
];

fn sample_aspect(bits: &mut Bits<'_>) -> Result<SampleAspect, CodecError> {
    let idc = bits.u(8)?;
    if idc == 255 {
        let width = bits.u(16)?;
        let height = bits.u(16)?;
        return Ok(SampleAspect::ratio(width, height));
    }
    let Some(index) = usize::try_from(idc).ok() else {
        return Ok(SampleAspect::UNSPECIFIED);
    };
    Ok(ASPECT
        .get(index)
        .copied()
        .unwrap_or(SampleAspect::UNSPECIFIED))
}

fn colour(idc: u32) -> ColorPrimaries {
    match idc {
        1 => ColorPrimaries::Bt709,
        4 => ColorPrimaries::Bt470m,
        5 | 6 => ColorPrimaries::Bt601,
        7 => ColorPrimaries::Smpte240m,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        10 => ColorPrimaries::Smpte428,
        11 => ColorPrimaries::Smpte431,
        12 => ColorPrimaries::Smpte432,
        22 => ColorPrimaries::Ebu3213,
        _ => ColorPrimaries::Unspecified,
    }
}

fn transfer_of(idc: u32) -> Transfer {
    match idc {
        1 | 6 => Transfer::Bt709,
        4 | 5 => Transfer::Bt601,
        7 => Transfer::Smpte240m,
        8 => Transfer::Linear,
        9 => Transfer::Log100,
        10 => Transfer::LogSqrt,
        11 => Transfer::Iec61966_2_4,
        12 => Transfer::Bt1361,
        13 => Transfer::Srgb,
        14 | 15 => Transfer::Bt2020,
        16 => Transfer::Pq,
        17 => Transfer::Smpte428,
        18 => Transfer::Hlg,
        _ => Transfer::Unspecified,
    }
}

fn matrix_of(idc: u32) -> ColorMatrix {
    match idc {
        0 => ColorMatrix::Identity,
        1 => ColorMatrix::Bt709,
        4 => ColorMatrix::Fcc,
        5 | 6 => ColorMatrix::Bt601,
        7 => ColorMatrix::Smpte240m,
        8 => ColorMatrix::Ycgco,
        9 | 10 => ColorMatrix::Bt2020,
        11 => ColorMatrix::Smpte2085,
        12 => ColorMatrix::ChromaDerived,
        13 => ColorMatrix::ChromaDerivedConstant,
        14 => ColorMatrix::Ictcp,
        15 => ColorMatrix::IptC2,
        16 => ColorMatrix::YcgcoRe,
        17 => ColorMatrix::YcgcoRo,
        _ => ColorMatrix::Unspecified,
    }
}

/// A picture parameter set names an id. An Annex A slice resolves it in
/// `sps_of`. A slice extension resolves it in `subset` when the Annex A
/// table has no set with that id.
pub(crate) fn parse_pps_in(
    rbsp: &[u8],
    sps_of: &[Option<Sps>],
    subset: &[Option<Sps>],
) -> Result<Pps, CodecError> {
    let mut bits = Bits::new(rbsp);
    let id = bits.ue()?;
    let sps_id = bits.ue()?;
    if id >= MAX_PPS as u32 || sps_id >= MAX_SPS as u32 {
        return Err(bits.malformed("pps id"));
    }
    let Some(sps) = sps_of
        .get(sps_id as usize)
        .and_then(|sps| sps.as_ref())
        .or_else(|| subset.get(sps_id as usize).and_then(|sps| sps.as_ref()))
    else {
        return Err(bits.malformed("pps without sps"));
    };
    let cabac = bits.bit()?;
    let bottom = bits.bit()?;
    let group_count = bits.ue()?;
    if group_count > 7 {
        return Err(bits.malformed("slice groups"));
    }
    let groups = if group_count == 0 {
        SliceGroups::none()
    } else {
        read_groups(&mut bits, sps, group_count as u8)?
    };
    let num_ref_idx_l0 = (bits.ue()? + 1) as u8;
    let num_ref_idx_l1 = (bits.ue()? + 1) as u8;
    if num_ref_idx_l0 > 16 || num_ref_idx_l1 > 16 {
        return Err(CodecError::Limit { what: "ref idx" });
    }
    let weighted_pred = bits.bit()?;
    let weighted_bipred = bits.u(2)? as u8;
    let qp_bd = 6 * i32::from(sps.bit_depth.saturating_sub(8));
    let init_minus = bits.se()?;
    if !(-(26 + qp_bd)..=25).contains(&init_minus) {
        return Err(bits.malformed("init qp"));
    }
    let init_qp = init_minus + 26;
    let qs_minus = bits.se()?;
    if !(-26..=25).contains(&qs_minus) {
        return Err(bits.malformed("init qs"));
    }
    let init_qs = qs_minus + 26;
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
            (weight4, weight8) = scaling_lists(&mut bits, transform_8x8, sps.chroma == 3)?;
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
        init_qs,
        chroma_qp_offset,
        second_chroma_qp_offset: second,
        deblocking_control,
        constrained_intra,
        redundant_pic_cnt_present: redundant,
        transform_8x8,
        weight4,
        weight8,
        groups,
    })
}

fn read_groups(bits: &mut Bits<'_>, sps: &Sps, minus1: u8) -> Result<SliceGroups, CodecError> {
    let map_type = bits.ue()?;
    if map_type > 6 {
        return Err(bits.malformed("slice group map"));
    }
    let map_type = map_type as u8;
    let units = crate::fmo::map_units(sps.mb_width, sps.mb_height, sps.frame_mbs_only);
    if units == 0 {
        return Err(bits.malformed("slice group map"));
    }
    let mut groups = SliceGroups::none();
    groups.minus1 = minus1;
    groups.map_type = map_type;
    let count = usize::from(minus1) + 1;
    match map_type {
        0 => {
            groups.run.reserve(count);
            for _ in 0..count {
                groups.run.push(bits.ue()?);
            }
        }
        1 => {}
        2 => {
            let width = sps.mb_width as u32;
            for _ in 0..usize::from(minus1) {
                let top = bits.ue()?;
                let bottom = bits.ue()?;
                if !rectangle(top, bottom, width, units) {
                    return Err(bits.malformed("slice group map"));
                }
                groups.top_left.push(top);
                groups.bottom_right.push(bottom);
            }
        }
        3..=5 => {
            if minus1 != 1 {
                return Err(bits.malformed("slice group map"));
            }
            groups.direction = bits.bit()?;
            let rate_minus = bits.ue()?;
            if rate_minus >= units {
                return Err(bits.malformed("slice group change"));
            }
            groups.rate = rate_minus + 1;
        }
        6 => {
            let declared = bits.ue()?;
            if declared != units - 1 {
                return Err(bits.malformed("slice group id"));
            }
            let width = crate::fmo::ceil_log2(u32::from(minus1) + 1);
            groups.explicit.reserve(units as usize);
            for _ in 0..units {
                let id = bits.u(width)?;
                if id > u32::from(minus1) {
                    return Err(bits.malformed("slice group id"));
                }
                groups.explicit.push(id as u8);
            }
        }
        _ => {
            return Err(bits.malformed("slice group map"));
        }
    }
    Ok(groups)
}

fn rectangle(top: u32, bottom: u32, width: u32, units: u32) -> bool {
    if width == 0 || top >= units || bottom >= units {
        return false;
    }
    let x0 = top % width;
    let y0 = top / width;
    let x1 = bottom % width;
    let y1 = bottom / width;
    x1 >= x0 && y1 >= y0
}

type Scaling = ([[u8; 16]; 6], [[u8; 64]; 6]);

fn scaling_lists(bits: &mut Bits<'_>, with8: bool, chroma444: bool) -> Result<Scaling, CodecError> {
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
        // 4:4:4 carries six 8×8 lists: intra Y, inter Y, then Cb and Cr for each.
        // 4:2:0 and 4:2:2 carry the two luma lists and copy them onto chroma.
        let count: usize = if chroma444 { 6 } else { 2 };
        for i in 0..count {
            if bits.bit()? {
                let mut list = [0u8; 64];
                read_list(bits, &mut list, 64)?;
                if list.iter().all(|v| *v == 0) {
                    list = if i.is_multiple_of(2) {
                        DEFAULT8_INTRA
                    } else {
                        DEFAULT8_INTER
                    };
                }
                w8[i] = inverse8(&list);
            } else if i == 0 {
                w8[i] = inverse8(&DEFAULT8_INTRA);
            } else if i == 1 {
                w8[i] = inverse8(&DEFAULT8_INTER);
            } else {
                w8[i] = w8[i - 2];
            }
            if !chroma444 {
                w8[i + 2] = w8[i];
                w8[i + 4] = w8[i];
            }
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

/// Clause 8.5.6, the field scan: where each coefficient of a 4×4 block goes in
/// a field macroblock or a field picture, as `row * 4 + column`. Coefficients
/// run down the columns first, because a field's rows are twice as far apart.
pub const FIELD4: [usize; 16] = [0, 4, 1, 8, 12, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

/// Clause 8.5.7, the 8×8 field scan, as `row * 8 + column`.
pub const FIELD8: [usize; 64] = [
    0, 8, 16, 1, 9, 24, 32, 17, 2, 25, 40, 48, 56, 33, 10, 3, 18, 41, 49, 57, 26, 11, 4, 19, 34,
    42, 50, 58, 27, 12, 5, 20, 35, 43, 51, 59, 28, 13, 6, 21, 36, 44, 52, 60, 29, 14, 22, 37, 45,
    53, 61, 30, 7, 15, 38, 46, 54, 62, 23, 31, 39, 47, 55, 63,
];

/// The 4×4 scan of a block: zig-zag for a frame macroblock, the field scan
/// for a field macroblock or a macroblock of a field picture.
#[inline]
pub fn scan4(field: bool) -> &'static [usize; 16] {
    if field { &FIELD4 } else { &ZIGZAG4 }
}

/// The 8×8 scan of a block, chosen as in [`scan4`].
#[inline]
pub fn scan8(field: bool) -> &'static [usize; 64] {
    if field { &FIELD8 } else { &ZIGZAG8 }
}

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

#[cfg(test)]
mod scans {
    use super::{FIELD4, FIELD8, ZIGZAG4, scan4, scan8};

    #[test]
    fn a_field_scan_visits_every_coefficient_once() {
        let mut seen4 = FIELD4;
        seen4.sort_unstable();
        assert!(seen4.iter().copied().eq(0..16));
        let mut seen8 = FIELD8;
        seen8.sort_unstable();
        assert!(seen8.iter().copied().eq(0..64));
    }

    /// Clause 8.5.6: after the first coefficient a field scan goes down the
    /// first column, where the zig-zag goes across the first row.
    #[test]
    fn a_field_scan_goes_down_the_column_first() {
        assert_eq!(FIELD4[..5], [0, 4, 1, 8, 12]);
        assert_eq!(FIELD8[..5], [0, 8, 16, 1, 9]);
        assert_eq!(ZIGZAG4[..3], [0, 1, 4]);
        assert_eq!(scan4(true), &FIELD4);
        assert_eq!(scan4(false), &ZIGZAG4);
        assert_eq!(scan8(true), &FIELD8);
    }
}
