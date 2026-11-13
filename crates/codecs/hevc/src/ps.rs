//! Sequence and picture parameter sets (clauses 7.3.2.2, 7.3.2.3, 7.3.4,
//! 7.3.7, and Annex E), and the tile scan of clause 6.5.1.

use cutlass_decode::{CodecError, ColorMatrix, ColorPrimaries, ColorRange, SampleAspect, Transfer};

use crate::bits::Bits;

/// Largest luma edge, the edge of a level 6.2 picture held to an 8:1 aspect.
pub const MAX_EDGE: u32 = 16_888;
/// Most luma samples in a picture: `MaxLumaPs` of level 6.2 (Table A.8).
pub const MAX_LUMA_SAMPLES: u64 = 35_651_584;
pub const MAX_DPB: usize = 16;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StRps {
    /// `DeltaPocS0`, closest first, all negative.
    pub negative: Vec<i32>,
    pub negative_used: Vec<bool>,
    /// `DeltaPocS1`, closest first, all positive.
    pub positive: Vec<i32>,
    pub positive_used: Vec<bool>,
}

impl StRps {
    fn count(&self) -> usize {
        self.negative.len() + self.positive.len()
    }
}

/// Scaling factors in raster order: `[sizeId][matrixId]`, 4×4 for size 0 and
/// 8×8 for the rest, with the DC of 16×16 and 32×32.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalingList {
    pub lists: [[[u8; 64]; 6]; 4],
    pub dc: [[u8; 6]; 2],
}

impl ScalingList {
    pub fn default_lists() -> Self {
        let mut lists = [[[16u8; 64]; 6]; 4];
        for size in lists.iter_mut().skip(1) {
            for (matrix, list) in size.iter_mut().enumerate() {
                *list = if matrix < 3 {
                    DEFAULT_INTRA
                } else {
                    DEFAULT_INTER
                };
            }
        }
        Self {
            lists,
            dc: [[16; 6]; 2],
        }
    }
}

const DEFAULT_INTRA: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 21, 24, 16, 16, 16, 16, 17, 19, 22, 25, 16, 16, 17, 18, 20, 22, 25, 29,
    16, 16, 18, 21, 24, 27, 31, 36, 17, 17, 20, 24, 30, 35, 41, 47, 18, 19, 22, 27, 35, 44, 54, 65,
    21, 22, 25, 31, 41, 54, 70, 88, 24, 25, 29, 36, 47, 65, 88, 115,
];

const DEFAULT_INTER: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 20, 24, 16, 16, 16, 17, 18, 20, 24, 25, 16, 16, 17, 18, 20, 24, 25, 28,
    16, 17, 18, 20, 24, 25, 28, 33, 17, 18, 20, 24, 25, 28, 33, 41, 18, 20, 24, 25, 28, 33, 41, 54,
    20, 24, 25, 28, 33, 41, 54, 71, 24, 25, 28, 33, 41, 54, 71, 91,
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sps {
    pub id: u8,
    pub max_sub_layers: u8,
    pub chroma_format_idc: u8,
    pub width: u32,
    pub height: u32,
    /// Conformance window in luma samples: left, right, top, bottom.
    pub crop: [u32; 4],
    pub bit_depth: u8,
    pub bit_depth_chroma: u8,
    pub log2_max_poc_lsb: u8,
    pub max_dec_pic_buffering: [u8; 8],
    pub max_num_reorder: [u8; 8],
    pub max_latency_increase_plus1: [u32; 8],
    pub log2_min_cb: u8,
    pub log2_ctb: u8,
    pub log2_min_tb: u8,
    pub log2_max_tb: u8,
    pub max_th_depth_inter: u8,
    pub max_th_depth_intra: u8,
    pub scaling_list: Option<ScalingList>,
    pub amp: bool,
    pub sao: bool,
    pub pcm: Option<Pcm>,
    pub st_rps: Vec<StRps>,
    pub long_term_present: bool,
    pub lt_ref_pic_poc_lsb: Vec<u32>,
    pub lt_used_by_curr: Vec<bool>,
    pub temporal_mvp: bool,
    pub strong_intra_smoothing: bool,
    pub vui: Vui,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pcm {
    pub bit_depth: u8,
    pub bit_depth_chroma: u8,
    pub log2_min: u8,
    pub log2_max: u8,
    pub loop_filter_disabled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vui {
    pub range: ColorRange,
    pub primaries: ColorPrimaries,
    pub transfer: Transfer,
    pub matrix: ColorMatrix,
    pub sample_aspect: SampleAspect,
    pub num_units: u32,
    pub time_scale: u32,
    pub fixed_rate: bool,
}

impl Vui {
    fn unspecified() -> Self {
        Self {
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
            matrix: ColorMatrix::Unspecified,
            sample_aspect: SampleAspect::UNSPECIFIED,
            num_units: 0,
            time_scale: 0,
            fixed_rate: false,
        }
    }
}

impl Sps {
    pub fn ctb_size(&self) -> u32 {
        1 << self.log2_ctb
    }

    pub fn ctb_width(&self) -> u32 {
        self.width.div_ceil(self.ctb_size())
    }

    pub fn ctb_height(&self) -> u32 {
        self.height.div_ceil(self.ctb_size())
    }

    pub fn qp_bd_offset(&self) -> i32 {
        6 * (i32::from(self.bit_depth) - 8)
    }

    pub fn qp_bd_offset_c(&self) -> i32 {
        6 * (i32::from(self.bit_depth_chroma) - 8)
    }
}

pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, CodecError> {
    let mut bits = Bits::new(rbsp);
    let _vps_id = bits.u(4)?;
    let max_sub_layers = bits.u(3)? as u8 + 1;
    if max_sub_layers > 7 {
        return Err(bits.malformed("sps_max_sub_layers_minus1"));
    }
    let _nesting = bits.bit()?;
    profile_tier_level(&mut bits, max_sub_layers)?;
    let id = bits.ue_max(15, "sps_seq_parameter_set_id")? as u8;
    let chroma_format_idc = bits.ue_max(3, "chroma_format_idc")? as u8;
    if chroma_format_idc == 3 && bits.bit()? {
        return Err(CodecError::Unsupported {
            what: "separate colour planes",
        });
    }
    let width = bits.ue()?;
    let height = bits.ue()?;
    if width == 0 || height == 0 {
        return Err(bits.malformed("picture size"));
    }
    if width > MAX_EDGE
        || height > MAX_EDGE
        || u64::from(width) * u64::from(height) > MAX_LUMA_SAMPLES
    {
        return Err(CodecError::Limit {
            what: "picture size",
        });
    }
    let mut crop = [0u32; 4];
    if bits.bit()? {
        let (sub_w, sub_h) = match chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        for (n, edge) in crop.iter_mut().enumerate() {
            let unit = if n < 2 { sub_w } else { sub_h };
            *edge = bits.ue()?.saturating_mul(unit);
        }
        if crop[0].saturating_add(crop[1]) >= width || crop[2].saturating_add(crop[3]) >= height {
            return Err(bits.malformed("conformance window"));
        }
    }
    let bit_depth = bits.ue_max(8, "bit_depth_luma_minus8")? as u8 + 8;
    let bit_depth_chroma = bits.ue_max(8, "bit_depth_chroma_minus8")? as u8 + 8;
    let log2_max_poc_lsb = bits.ue_max(12, "log2_max_pic_order_cnt_lsb_minus4")? as u8 + 4;
    let ordering_all = bits.bit()?;
    let mut max_dec_pic_buffering = [1u8; 8];
    let mut max_num_reorder = [0u8; 8];
    let mut max_latency_increase_plus1 = [0u32; 8];
    let first = if ordering_all { 0 } else { max_sub_layers - 1 };
    for i in first..max_sub_layers {
        let i = usize::from(i);
        let dec = bits.ue_max(MAX_DPB as u32 - 1, "sps_max_dec_pic_buffering_minus1")? + 1;
        let reorder = bits.ue_max(dec - 1, "sps_max_num_reorder_pics")?;
        max_dec_pic_buffering[i] = dec as u8;
        max_num_reorder[i] = reorder as u8;
        max_latency_increase_plus1[i] = bits.ue()?;
    }
    for i in 0..usize::from(first) {
        let last = usize::from(first);
        max_dec_pic_buffering[i] = max_dec_pic_buffering[last];
        max_num_reorder[i] = max_num_reorder[last];
        max_latency_increase_plus1[i] = max_latency_increase_plus1[last];
    }
    let log2_min_cb = bits.ue_max(3, "log2_min_luma_coding_block_size_minus3")? as u8 + 3;
    let log2_ctb = log2_min_cb + bits.ue_max(3, "log2_diff_max_min_luma_coding_block_size")? as u8;
    let log2_min_tb = bits.ue_max(3, "log2_min_luma_transform_block_size_minus2")? as u8 + 2;
    let log2_max_tb =
        log2_min_tb + bits.ue_max(3, "log2_diff_max_min_luma_transform_block_size")? as u8;
    if !(4..=6).contains(&log2_ctb) || log2_min_tb >= log2_min_cb || log2_max_tb > log2_ctb.min(5) {
        return Err(bits.malformed("block sizes"));
    }
    if width % (1 << log2_min_cb) != 0 || height % (1 << log2_min_cb) != 0 {
        return Err(bits.malformed("picture size is not a multiple of the coding block"));
    }
    let depth_max = u32::from(log2_ctb - log2_min_tb);
    let max_th_depth_inter = bits.ue_max(depth_max, "max_transform_hierarchy_depth_inter")? as u8;
    let max_th_depth_intra = bits.ue_max(depth_max, "max_transform_hierarchy_depth_intra")? as u8;
    let scaling_list = if bits.bit()? {
        let mut list = ScalingList::default_lists();
        if bits.bit()? {
            scaling_list_data(&mut bits, &mut list)?;
        }
        Some(list)
    } else {
        None
    };
    let amp = bits.bit()?;
    let sao = bits.bit()?;
    let pcm = if bits.bit()? {
        let pcm_depth = bits.u(4)? as u8 + 1;
        let pcm_depth_c = bits.u(4)? as u8 + 1;
        let log2_min = bits.ue_max(2, "log2_min_pcm_luma_coding_block_size_minus3")? as u8 + 3;
        let log2_max =
            log2_min + bits.ue_max(2, "log2_diff_max_min_pcm_luma_coding_block_size")? as u8;
        if pcm_depth > bit_depth || pcm_depth_c > bit_depth_chroma || log2_max > log2_ctb.min(5) {
            return Err(bits.malformed("pcm parameters"));
        }
        Some(Pcm {
            bit_depth: pcm_depth,
            bit_depth_chroma: pcm_depth_c,
            log2_min,
            log2_max,
            loop_filter_disabled: bits.bit()?,
        })
    } else {
        None
    };
    let num_st = bits.ue_max(64, "num_short_term_ref_pic_sets")? as usize;
    let mut st_rps: Vec<StRps> = Vec::with_capacity(num_st);
    for index in 0..num_st {
        let rps = parse_st_rps(&mut bits, &st_rps, index, num_st, false)?;
        st_rps.push(rps);
    }
    let long_term_present = bits.bit()?;
    let mut lt_ref_pic_poc_lsb = Vec::new();
    let mut lt_used_by_curr = Vec::new();
    if long_term_present {
        let count = bits.ue_max(32, "num_long_term_ref_pics_sps")?;
        for _ in 0..count {
            lt_ref_pic_poc_lsb.push(bits.u(u32::from(log2_max_poc_lsb))?);
            lt_used_by_curr.push(bits.bit()?);
        }
    }
    let temporal_mvp = bits.bit()?;
    let strong_intra_smoothing = bits.bit()?;
    let vui = if bits.bit()? {
        parse_vui(&mut bits, max_sub_layers)?
    } else {
        Vui::unspecified()
    };
    if bits.bit()? {
        let range = bits.bit()?;
        let multilayer = bits.bit()?;
        let three_d = bits.bit()?;
        let scc = bits.bit()?;
        let _four = bits.u(4)?;
        if range {
            for _ in 0..9 {
                if bits.bit()? {
                    return Err(CodecError::Unsupported {
                        what: "range extension coding tools",
                    });
                }
            }
        }
        if multilayer || three_d || scc {
            return Err(CodecError::Unsupported {
                what: "sps extension",
            });
        }
    }
    if chroma_format_idc != 1 {
        return Err(CodecError::Unsupported {
            what: "chroma format other than 4:2:0",
        });
    }
    if bit_depth != bit_depth_chroma || bit_depth > 12 {
        return Err(CodecError::Unsupported { what: "bit depth" });
    }
    Ok(Sps {
        id,
        max_sub_layers,
        chroma_format_idc,
        width,
        height,
        crop,
        bit_depth,
        bit_depth_chroma,
        log2_max_poc_lsb,
        max_dec_pic_buffering,
        max_num_reorder,
        max_latency_increase_plus1,
        log2_min_cb,
        log2_ctb,
        log2_min_tb,
        log2_max_tb,
        max_th_depth_inter,
        max_th_depth_intra,
        scaling_list,
        amp,
        sao,
        pcm,
        st_rps,
        long_term_present,
        lt_ref_pic_poc_lsb,
        lt_used_by_curr,
        temporal_mvp,
        strong_intra_smoothing,
        vui,
    })
}

fn general_profile(bits: &mut Bits<'_>) -> Result<(), CodecError> {
    // profile_space, tier, idc, 32 compatibility flags, 4 source flags,
    // 43 constraint bits, and one more flag: 88 bits.
    bits.skip(88)
}

fn profile_tier_level(bits: &mut Bits<'_>, max_sub_layers: u8) -> Result<(), CodecError> {
    general_profile(bits)?;
    let _level = bits.u(8)?;
    let subs = usize::from(max_sub_layers - 1);
    let mut profile_present = [false; 8];
    let mut level_present = [false; 8];
    for i in 0..subs {
        profile_present[i] = bits.bit()?;
        level_present[i] = bits.bit()?;
    }
    if subs > 0 {
        bits.skip(2 * (8 - subs))?;
    }
    for i in 0..subs {
        if profile_present[i] {
            general_profile(bits)?;
        }
        if level_present[i] {
            bits.skip(8)?;
        }
    }
    Ok(())
}

/// `st_ref_pic_set(stRpsIdx)` of clause 7.3.7, with the inter prediction of
/// clause 7.4.8.
pub fn parse_st_rps(
    bits: &mut Bits<'_>,
    sets: &[StRps],
    index: usize,
    num_sets: usize,
    in_slice: bool,
) -> Result<StRps, CodecError> {
    let predict = index != 0 && bits.bit()?;
    if predict {
        let delta_idx = if in_slice {
            bits.ue_max(index as u32 - 1, "delta_idx_minus1")? as usize + 1
        } else {
            1
        };
        let Some(reference) = index.checked_sub(delta_idx).and_then(|at| sets.get(at)) else {
            return Err(bits.malformed("delta_idx_minus1"));
        };
        let sign = bits.bit()?;
        let abs = bits.ue_max(32_767, "abs_delta_rps_minus1")? as i32 + 1;
        let delta_rps = if sign { -abs } else { abs };
        let total = reference.count();
        let mut used = [false; 33];
        let mut use_delta = [true; 33];
        for j in 0..=total {
            used[j] = bits.bit()?;
            if !used[j] {
                use_delta[j] = bits.bit()?;
            }
        }
        let neg_ref = reference.negative.len();
        let mut rps = StRps::default();
        for j in (0..reference.positive.len()).rev() {
            let d = reference.positive[j] + delta_rps;
            if d < 0 && use_delta[neg_ref + j] {
                rps.negative.push(d);
                rps.negative_used.push(used[neg_ref + j]);
            }
        }
        if delta_rps < 0 && use_delta[total] {
            rps.negative.push(delta_rps);
            rps.negative_used.push(used[total]);
        }
        for j in 0..neg_ref {
            let d = reference.negative[j] + delta_rps;
            if d < 0 && use_delta[j] {
                rps.negative.push(d);
                rps.negative_used.push(used[j]);
            }
        }
        for j in (0..neg_ref).rev() {
            let d = reference.negative[j] + delta_rps;
            if d > 0 && use_delta[j] {
                rps.positive.push(d);
                rps.positive_used.push(used[j]);
            }
        }
        if delta_rps > 0 && use_delta[total] {
            rps.positive.push(delta_rps);
            rps.positive_used.push(used[total]);
        }
        for j in 0..reference.positive.len() {
            let d = reference.positive[j] + delta_rps;
            if d > 0 && use_delta[neg_ref + j] {
                rps.positive.push(d);
                rps.positive_used.push(used[neg_ref + j]);
            }
        }
        if rps.count() > MAX_DPB {
            return Err(bits.malformed("short-term reference picture set"));
        }
        let _ = num_sets;
        Ok(rps)
    } else {
        let negatives = bits.ue_max(MAX_DPB as u32, "num_negative_pics")? as usize;
        let positives = bits.ue_max(MAX_DPB as u32, "num_positive_pics")? as usize;
        if negatives + positives > MAX_DPB {
            return Err(bits.malformed("short-term reference picture set"));
        }
        let mut rps = StRps::default();
        let mut poc = 0i32;
        for _ in 0..negatives {
            poc -= bits.ue_max(32_767, "delta_poc_s0_minus1")? as i32 + 1;
            rps.negative.push(poc);
            rps.negative_used.push(bits.bit()?);
        }
        poc = 0;
        for _ in 0..positives {
            poc += bits.ue_max(32_767, "delta_poc_s1_minus1")? as i32 + 1;
            rps.positive.push(poc);
            rps.positive_used.push(bits.bit()?);
        }
        Ok(rps)
    }
}

/// Diagonal scan positions of a 4×4 and an 8×8 block, raster index.
fn diagonal(size: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(size * size);
    let mut x = 0i32;
    let mut y = 0i32;
    let n = size as i32;
    while out.len() < size * size {
        while y >= 0 {
            if x < n && y < n {
                out.push((y * n + x) as usize);
            }
            y -= 1;
            x += 1;
        }
        y = x;
        x = 0;
    }
    out
}

fn scaling_list_data(bits: &mut Bits<'_>, list: &mut ScalingList) -> Result<(), CodecError> {
    let scan4 = diagonal(4);
    let scan8 = diagonal(8);
    for size in 0..4usize {
        let step = if size == 3 { 3 } else { 1 };
        let mut matrix = 0usize;
        while matrix < 6 {
            if !bits.bit()? {
                let delta = bits.ue_max(5, "scaling_list_pred_matrix_id_delta")? as usize * step;
                if delta > matrix {
                    return Err(bits.malformed("scaling_list_pred_matrix_id_delta"));
                }
                if delta == 0 {
                    list.lists[size][matrix] = if size == 0 {
                        [16; 64]
                    } else if matrix < 3 {
                        DEFAULT_INTRA
                    } else {
                        DEFAULT_INTER
                    };
                    if size > 1 {
                        list.dc[size - 2][matrix] = 16;
                    }
                } else {
                    list.lists[size][matrix] = list.lists[size][matrix - delta];
                    if size > 1 {
                        list.dc[size - 2][matrix] = list.dc[size - 2][matrix - delta];
                    }
                }
            } else {
                let mut next = 8i32;
                let count = if size == 0 { 16 } else { 64 };
                if size > 1 {
                    let dc = bits.se_range(-7, 247, "scaling_list_dc_coef_minus8")? + 8;
                    next = dc;
                    list.dc[size - 2][matrix] = dc as u8;
                }
                let scan = if size == 0 { &scan4 } else { &scan8 };
                for &pos in scan.iter().take(count) {
                    let delta = bits.se_range(-128, 127, "scaling_list_delta_coef")?;
                    next = (next + delta + 256).rem_euclid(256);
                    list.lists[size][matrix][pos] = next as u8;
                }
            }
            matrix += step;
        }
    }
    Ok(())
}

fn parse_vui(bits: &mut Bits<'_>, max_sub_layers: u8) -> Result<Vui, CodecError> {
    let mut vui = Vui::unspecified();
    if bits.bit()? {
        let idc = bits.u(8)?;
        vui.sample_aspect = if idc == 255 {
            SampleAspect::ratio(bits.u(16)?, bits.u(16)?)
        } else {
            aspect(idc)
        };
    }
    if bits.bit()? {
        let _overscan_appropriate = bits.bit()?;
    }
    if bits.bit()? {
        let _format = bits.u(3)?;
        vui.range = if bits.bit()? {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
        if bits.bit()? {
            vui.primaries = primaries_of(bits.u(8)?);
            vui.transfer = transfer_of(bits.u(8)?);
            vui.matrix = matrix_of(bits.u(8)?);
        }
    }
    if bits.bit()? {
        let _top = bits.ue()?;
        let _bottom = bits.ue()?;
    }
    let _neutral_chroma = bits.bit()?;
    let _field_seq = bits.bit()?;
    let _frame_field_info = bits.bit()?;
    if bits.bit()? {
        for _ in 0..4 {
            let _offset = bits.ue()?;
        }
    }
    if bits.bit()? {
        vui.num_units = bits.u(32)?;
        vui.time_scale = bits.u(32)?;
        vui.fixed_rate = true;
        if bits.bit()? {
            let _ticks = bits.ue()?;
        }
        if bits.bit()? {
            hrd_parameters(bits, true, max_sub_layers)?;
        }
    }
    if bits.bit()? {
        let _tiles_fixed = bits.bit()?;
        let _mv_over_boundaries = bits.bit()?;
        let _restricted_lists = bits.bit()?;
        for _ in 0..5 {
            let _ = bits.ue()?;
        }
    }
    Ok(vui)
}

fn hrd_parameters(bits: &mut Bits<'_>, common: bool, max_sub_layers: u8) -> Result<(), CodecError> {
    let mut nal = false;
    let mut vcl = false;
    let mut sub_pic = false;
    if common {
        nal = bits.bit()?;
        vcl = bits.bit()?;
        if nal || vcl {
            sub_pic = bits.bit()?;
            if sub_pic {
                bits.skip(8 + 5 + 1 + 5)?;
            }
            bits.skip(8)?;
            if sub_pic {
                bits.skip(4)?;
            }
            bits.skip(15)?;
        }
    }
    for _ in 0..max_sub_layers {
        let general = bits.bit()?;
        let within = if general { true } else { bits.bit()? };
        let mut low_delay = false;
        if within {
            let _duration = bits.ue()?;
        } else {
            low_delay = bits.bit()?;
        }
        let mut cpb = 1;
        if !low_delay {
            cpb = bits.ue_max(31, "cpb_cnt_minus1")? + 1;
        }
        for present in [nal, vcl] {
            if present {
                for _ in 0..cpb {
                    let _ = bits.ue()?;
                    let _ = bits.ue()?;
                    if sub_pic {
                        let _ = bits.ue()?;
                        let _ = bits.ue()?;
                    }
                    let _cbr = bits.bit()?;
                }
            }
        }
    }
    Ok(())
}

fn aspect(idc: u32) -> SampleAspect {
    const TABLE: [(u32, u32); 17] = [
        (0, 1),
        (1, 1),
        (12, 11),
        (10, 11),
        (16, 11),
        (40, 33),
        (24, 11),
        (20, 11),
        (32, 11),
        (80, 33),
        (18, 11),
        (15, 11),
        (64, 33),
        (160, 99),
        (4, 3),
        (3, 2),
        (2, 1),
    ];
    TABLE
        .get(idc as usize)
        .map_or(SampleAspect::UNSPECIFIED, |&(w, h)| {
            SampleAspect::ratio(w, h)
        })
}

fn primaries_of(idc: u32) -> ColorPrimaries {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pps {
    pub id: u8,
    pub sps_id: u8,
    pub dependent_slices: bool,
    pub output_flag_present: bool,
    pub extra_slice_header_bits: u8,
    pub sign_data_hiding: bool,
    pub cabac_init_present: bool,
    pub num_ref_idx_default: [u8; 2],
    pub init_qp: i32,
    pub constrained_intra_pred: bool,
    pub transform_skip: bool,
    pub cu_qp_delta: bool,
    pub diff_cu_qp_delta_depth: u8,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub slice_chroma_qp_offsets: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass: bool,
    pub tiles: bool,
    pub entropy_sync: bool,
    pub tile_columns: Vec<u32>,
    pub tile_rows: Vec<u32>,
    pub uniform_spacing: bool,
    pub loop_filter_across_tiles: bool,
    pub loop_filter_across_slices: bool,
    pub deblocking_override_enabled: bool,
    pub deblocking_disabled: bool,
    pub beta_offset: i32,
    pub tc_offset: i32,
    pub scaling_list: Option<ScalingList>,
    pub lists_modification: bool,
    pub log2_parallel_merge_level: u8,
    pub slice_header_extension: bool,
}

pub fn parse_pps(rbsp: &[u8]) -> Result<Pps, CodecError> {
    let mut bits = Bits::new(rbsp);
    let id = bits.ue_max(63, "pps_pic_parameter_set_id")? as u8;
    let sps_id = bits.ue_max(15, "pps_seq_parameter_set_id")? as u8;
    let dependent_slices = bits.bit()?;
    let output_flag_present = bits.bit()?;
    let extra_slice_header_bits = bits.u(3)? as u8;
    let sign_data_hiding = bits.bit()?;
    let cabac_init_present = bits.bit()?;
    let l0 = bits.ue_max(14, "num_ref_idx_l0_default_active_minus1")? as u8 + 1;
    let l1 = bits.ue_max(14, "num_ref_idx_l1_default_active_minus1")? as u8 + 1;
    let init_qp = 26 + bits.se_range(-62, 25, "init_qp_minus26")?;
    let constrained_intra_pred = bits.bit()?;
    let transform_skip = bits.bit()?;
    let cu_qp_delta = bits.bit()?;
    let diff_cu_qp_delta_depth = if cu_qp_delta {
        bits.ue_max(3, "diff_cu_qp_delta_depth")? as u8
    } else {
        0
    };
    let cb_qp_offset = bits.se_range(-12, 12, "pps_cb_qp_offset")?;
    let cr_qp_offset = bits.se_range(-12, 12, "pps_cr_qp_offset")?;
    let slice_chroma_qp_offsets = bits.bit()?;
    let weighted_pred = bits.bit()?;
    let weighted_bipred = bits.bit()?;
    let transquant_bypass = bits.bit()?;
    let tiles = bits.bit()?;
    let entropy_sync = bits.bit()?;
    let mut tile_columns = Vec::new();
    let mut tile_rows = Vec::new();
    let mut uniform_spacing = true;
    let mut loop_filter_across_tiles = true;
    if tiles {
        let columns = bits.ue_max(MAX_EDGE / 16, "num_tile_columns_minus1")? + 1;
        let rows = bits.ue_max(MAX_EDGE / 16, "num_tile_rows_minus1")? + 1;
        uniform_spacing = bits.bit()?;
        if uniform_spacing {
            tile_columns = vec![0; columns as usize];
            tile_rows = vec![0; rows as usize];
        } else {
            for _ in 1..columns {
                tile_columns.push(bits.ue_max(MAX_EDGE, "column_width_minus1")? + 1);
            }
            tile_columns.push(0);
            for _ in 1..rows {
                tile_rows.push(bits.ue_max(MAX_EDGE, "row_height_minus1")? + 1);
            }
            tile_rows.push(0);
        }
        loop_filter_across_tiles = bits.bit()?;
    }
    let loop_filter_across_slices = bits.bit()?;
    let mut deblocking_override_enabled = false;
    let mut deblocking_disabled = false;
    let mut beta_offset = 0;
    let mut tc_offset = 0;
    if bits.bit()? {
        deblocking_override_enabled = bits.bit()?;
        deblocking_disabled = bits.bit()?;
        if !deblocking_disabled {
            beta_offset = 2 * bits.se_range(-6, 6, "pps_beta_offset_div2")?;
            tc_offset = 2 * bits.se_range(-6, 6, "pps_tc_offset_div2")?;
        }
    }
    let scaling_list = if bits.bit()? {
        let mut list = ScalingList::default_lists();
        scaling_list_data(&mut bits, &mut list)?;
        Some(list)
    } else {
        None
    };
    let lists_modification = bits.bit()?;
    let log2_parallel_merge_level = bits.ue_max(4, "log2_parallel_merge_level_minus2")? as u8 + 2;
    let slice_header_extension = bits.bit()?;
    if bits.bit()? {
        let range = bits.bit()?;
        let multilayer = bits.bit()?;
        let three_d = bits.bit()?;
        let scc = bits.bit()?;
        let _four = bits.u(4)?;
        if range {
            if transform_skip && bits.ue()? != 0 {
                return Err(CodecError::Unsupported {
                    what: "transform skip larger than 4x4",
                });
            }
            if bits.bit()? || bits.bit()? {
                return Err(CodecError::Unsupported {
                    what: "range extension picture tools",
                });
            }
            if bits.ue()? != 0 || bits.ue()? != 0 {
                return Err(CodecError::Unsupported {
                    what: "sao offset scale",
                });
            }
        }
        if multilayer || three_d || scc {
            return Err(CodecError::Unsupported {
                what: "pps extension",
            });
        }
    }
    Ok(Pps {
        id,
        sps_id,
        dependent_slices,
        output_flag_present,
        extra_slice_header_bits,
        sign_data_hiding,
        cabac_init_present,
        num_ref_idx_default: [l0, l1],
        init_qp,
        constrained_intra_pred,
        transform_skip,
        cu_qp_delta,
        diff_cu_qp_delta_depth,
        cb_qp_offset,
        cr_qp_offset,
        slice_chroma_qp_offsets,
        weighted_pred,
        weighted_bipred,
        transquant_bypass,
        tiles,
        entropy_sync,
        tile_columns,
        tile_rows,
        uniform_spacing,
        loop_filter_across_tiles,
        loop_filter_across_slices,
        deblocking_override_enabled,
        deblocking_disabled,
        beta_offset,
        tc_offset,
        scaling_list,
        lists_modification,
        log2_parallel_merge_level,
        slice_header_extension,
    })
}

/// The CTB scan of one SPS and PPS pair (clause 6.5.1).
#[derive(Clone, Debug)]
pub struct Layout {
    pub ctb_width: u32,
    pub ctb_height: u32,
    pub log2_ctb: u8,
    pub rs_to_ts: Vec<u32>,
    pub ts_to_rs: Vec<u32>,
    /// Tile index of each CTB, raster order.
    pub tile_of: Vec<u16>,
    /// Column boundaries in CTBs, `columns + 1` entries.
    pub col_bd: Vec<u32>,
    /// Z-scan order of the minimum transform blocks inside a CTB, by row
    /// then column (clause 6.5.2), and their count per edge.
    pub zorder: Vec<u16>,
    pub zorder_edge: usize,
    /// `Log2MinTrafoSize`.
    pub log2_min_tb: u8,
}

impl Layout {
    pub fn new(sps: &Sps, pps: &Pps) -> Result<Self, CodecError> {
        let ctb_width = sps.ctb_width();
        let ctb_height = sps.ctb_height();
        let widths = spacing(
            &pps.tile_columns,
            ctb_width,
            pps.uniform_spacing || !pps.tiles,
        )?;
        let heights = spacing(
            &pps.tile_rows,
            ctb_height,
            pps.uniform_spacing || !pps.tiles,
        )?;
        let mut col_bd = vec![0u32];
        for w in &widths {
            col_bd.push(col_bd[col_bd.len() - 1] + w);
        }
        let mut row_bd = vec![0u32];
        for h in &heights {
            row_bd.push(row_bd[row_bd.len() - 1] + h);
        }
        let total = (ctb_width * ctb_height) as usize;
        let mut rs_to_ts = vec![0u32; total];
        let mut ts_to_rs = vec![0u32; total];
        let mut tile_of = vec![0u16; total];
        let mut ts = 0u32;
        let mut tile = 0u16;
        for tr in 0..heights.len() {
            for tc in 0..widths.len() {
                for y in row_bd[tr]..row_bd[tr + 1] {
                    for x in col_bd[tc]..col_bd[tc + 1] {
                        let rs = (y * ctb_width + x) as usize;
                        rs_to_ts[rs] = ts;
                        ts_to_rs[ts as usize] = rs as u32;
                        tile_of[rs] = tile;
                        ts += 1;
                    }
                }
                tile = tile.saturating_add(1);
            }
        }
        let edge_log2 = u32::from(sps.log2_ctb.saturating_sub(sps.log2_min_tb));
        let edge = 1usize << edge_log2;
        let mut zorder = vec![0u16; edge * edge];
        for (i, z) in zorder.iter_mut().enumerate() {
            let (tx, ty) = (i % edge, i / edge);
            let mut morton = 0usize;
            for bit in 0..edge_log2 {
                morton |= ((tx >> bit) & 1) << (2 * bit);
                morton |= ((ty >> bit) & 1) << (2 * bit + 1);
            }
            *z = morton as u16;
        }
        Ok(Self {
            ctb_width,
            ctb_height,
            log2_ctb: sps.log2_ctb,
            rs_to_ts,
            ts_to_rs,
            tile_of,
            col_bd,
            zorder,
            zorder_edge: edge,
            log2_min_tb: sps.log2_min_tb,
        })
    }

    pub fn ctb_count(&self) -> u32 {
        self.ctb_width * self.ctb_height
    }

    /// First CTB column of the tile column holding column `x`.
    pub fn tile_col_start(&self, x: u32) -> u32 {
        let at = self.col_bd.partition_point(|&bd| bd <= x);
        self.col_bd[at.saturating_sub(1)]
    }
}

fn spacing(coded: &[u32], total: u32, uniform: bool) -> Result<Vec<u32>, CodecError> {
    let count = coded.len().max(1) as u32;
    if count > total {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "tile count",
        });
    }
    if uniform {
        return Ok((0..count)
            .map(|i| ((i + 1) * total) / count - (i * total) / count)
            .collect());
    }
    let mut out: Vec<u32> = coded[..coded.len() - 1].to_vec();
    let sum: u32 = out.iter().fold(0u32, |acc, &w| acc.saturating_add(w));
    if sum >= total {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "tile sizes",
        });
    }
    out.push(total - sum);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{Bits, StRps, diagonal, parse_st_rps};

    #[test]
    fn diagonal_scan_runs_up_right() {
        assert_eq!(
            diagonal(4),
            vec![0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15]
        );
    }

    #[test]
    fn predicted_rps_adds_the_delta_and_sorts() {
        // Reference set: -1 (used), 1 (used).
        let reference = StRps {
            negative: vec![-1],
            negative_used: vec![true],
            positive: vec![1],
            positive_used: vec![true],
        };
        // inter_ref_pic_set_prediction_flag 1, sign 1, abs_delta_rps_minus1 0
        // (delta -1), then used flags 1, 1, 1 for -2, 0, and -1.
        // 1, 1, 1 (ue 0), 1, 1, 1 -> bits: 1 1 1 1 1 1
        let data = [0b1111_1110];
        let mut bits = Bits::new(&data);
        let rps = parse_st_rps(&mut bits, &[reference], 1, 2, false).unwrap();
        assert_eq!(rps.negative, vec![-1, -2]);
        assert!(rps.positive.is_empty());
    }
}
