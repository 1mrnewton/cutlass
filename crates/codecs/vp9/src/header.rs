//! Uncompressed and compressed VP9 headers.
//!
//! The order matches libvpx `read_uncompressed_header` and
//! `read_compressed_header`, including the probability reset that runs
//! before the loop filter and segmentation are read.

#![allow(dead_code)]

use cutlass_decode::CodecError;

use crate::bits::{BitReader, BoolDecoder};
use crate::geom::{
    self, ALTREF_FRAME, COMPOUND_REFERENCE, GOLDEN_FRAME, LAST_FRAME, ONLY_4X4, REFERENCE_SELECT,
    SINGLE_REFERENCE, SWITCHABLE,
};
use crate::prob::{self, FrameContext};
use crate::tables::{AC_QLOOKUP, DC_QLOOKUP};

const MAX_PROB: u8 = 255;
const REF_SLOTS: usize = 8;
const REFS_PER_FRAME: usize = 3;

#[derive(Clone, Default)]
pub struct SegState {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal: bool,
    pub update_data: bool,
    pub abs_delta: bool,
    pub tree_probs: [u8; 7],
    pub pred_probs: [u8; 3],
    pub feature_on: [[bool; 4]; 8],
    pub feature_data: [[i16; 4]; 8],
}

/// Decoder memory a header is allowed to read. Nothing here changes until
/// the frame that produced the header has been reconstructed.
#[derive(Clone)]
pub struct Persist {
    pub contexts: [FrameContext; 4],
    pub key_frame: bool,
    pub intra_only: bool,
    pub width: u32,
    pub height: u32,
    pub last_show_frame: bool,
    pub last_width: u32,
    pub last_height: u32,
    pub need_resync: bool,
    pub ref_ok: [bool; REF_SLOTS],
    pub ref_w: [u32; REF_SLOTS],
    pub ref_h: [u32; REF_SLOTS],
    pub seg: SegState,
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
    pub sign_bias: [u8; 4],
}

impl Persist {
    pub fn new() -> Self {
        Self {
            contexts: std::array::from_fn(|_| FrameContext::unready()),
            key_frame: true,
            intra_only: false,
            width: 0,
            height: 0,
            last_show_frame: false,
            last_width: 0,
            last_height: 0,
            need_resync: true,
            ref_ok: [false; REF_SLOTS],
            ref_w: [0; REF_SLOTS],
            ref_h: [0; REF_SLOTS],
            seg: SegState::default(),
            ref_deltas: [0; 4],
            mode_deltas: [0; 2],
            sign_bias: [0; 4],
        }
    }
}

impl Default for Persist {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct Header {
    pub profile: u8,
    pub show_existing: bool,
    pub frame_to_show: u8,
    pub key_frame: bool,
    pub show_frame: bool,
    pub error_resilient: bool,
    pub intra_only: bool,
    pub reset_frame_context: u8,
    pub refresh_flags: u8,
    pub color_space: u8,
    pub color_range: u8,
    pub width: u32,
    pub height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub show_width: u32,
    pub show_height: u32,
    pub ref_index: [u8; REFS_PER_FRAME],
    pub ref_size: [(u32, u32); REFS_PER_FRAME],
    pub allow_hp: bool,
    pub interp: u8,
    pub refresh_context: bool,
    pub frame_parallel: bool,
    pub context_idx: u8,
    pub filter_level: u8,
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub delta_update: bool,
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
    pub base_qindex: u8,
    pub y_dc_delta: i8,
    pub uv_dc_delta: i8,
    pub uv_ac_delta: i8,
    pub lossless: bool,
    pub y_dq: [[i16; 2]; 8],
    pub uv_dq: [[i16; 2]; 8],
    pub seg: SegState,
    pub log2_tile_cols: u8,
    pub log2_tile_rows: u8,
    pub tx_mode: u8,
    pub reference_mode: u8,
    pub comp_fixed: u8,
    pub comp_var: [u8; 2],
    pub sign_bias: [u8; 4],
    pub use_prev_mvs: bool,
    pub need_resync: bool,
    pub header_bytes: usize,
    pub compressed_bytes: usize,
}

pub struct Parsed<'a> {
    pub header: Header,
    pub fc: FrameContext,
    pub contexts: [FrameContext; 4],
    pub tiles: &'a [u8],
}

pub fn parse<'a>(state: &Persist, data: &'a [u8]) -> Result<Parsed<'a>, CodecError> {
    let mut bits = BitReader::new(data);
    if bits.read(2)? != 2 {
        return Err(bad(&bits, "frame marker"));
    }
    let profile = read_profile(&mut bits)?;
    if profile != 0 {
        return Err(CodecError::Unsupported {
            what: "vp9 profile",
        });
    }
    if bits.read(1)? == 1 {
        return show_existing(state, &mut bits, data);
    }
    let key_frame = bits.read(1)? == 0;
    let show_frame = bits.read(1)? == 1;
    let error_resilient = bits.read(1)? == 1;
    let mut header = empty_header(profile, key_frame, show_frame, error_resilient);
    let mut contexts = state.contexts.clone();
    let mut seg = state.seg.clone();
    let mut ref_deltas = state.ref_deltas;
    let mut mode_deltas = state.mode_deltas;
    let mut sign_bias = state.sign_bias;
    let mut need_resync = state.need_resync;

    if key_frame {
        read_key_setup(&mut bits, &mut header)?;
        need_resync = false;
    } else {
        read_inter_setup(
            &mut bits,
            &mut header,
            &mut sign_bias,
            &mut need_resync,
            state,
        )?;
    }
    header.need_resync = need_resync;

    read_context_flags(&mut bits, &mut header)?;
    if header.intra() || error_resilient {
        reset_past(
            &mut contexts,
            &mut seg,
            &mut ref_deltas,
            &mut mode_deltas,
            &mut sign_bias,
            &mut header,
        );
    }
    let (level, sharp) =
        read_loop_filter(&mut bits, &mut ref_deltas, &mut mode_deltas, &mut header)?;
    header.filter_level = level;
    header.sharpness = sharp;
    read_quant(&mut bits, &mut header)?;
    read_segmentation(&mut bits, &mut seg)?;
    header.seg = seg;
    fill_dequant(&mut header);
    read_tiles(&mut bits, &mut header)?;
    let compressed = usize::try_from(bits.read(16)?).unwrap_or(0);
    if compressed == 0 {
        return Err(bad(&bits, "header size"));
    }
    let start = bits.bytes_read();
    let end = start.checked_add(compressed).ok_or(CodecError::Limit {
        what: "compressed header",
    })?;
    if data.len() < end {
        return Err(CodecError::Truncated {
            at_bit: bits.position() as u64,
            what: "compressed header",
        });
    }
    header.header_bytes = start;
    header.compressed_bytes = compressed;
    header.ref_deltas = ref_deltas;
    header.mode_deltas = mode_deltas;
    header.sign_bias = sign_bias;
    header.use_prev_mvs = use_prev(state, &header);
    let mut fc = load_context(&contexts, header.context_idx)?;
    read_compressed(&mut header, &mut fc, &data[start..end])?;
    Ok(Parsed {
        header,
        fc,
        contexts,
        tiles: &data[end..],
    })
}

fn show_existing<'a>(
    state: &Persist,
    bits: &mut BitReader<'_>,
    data: &'a [u8],
) -> Result<Parsed<'a>, CodecError> {
    let slot = u8::try_from(bits.read(3)?).unwrap_or(0);
    let index = usize::from(slot);
    if !state.ref_ok.get(index).copied().unwrap_or(false) {
        return Err(bad(bits, "shown reference"));
    }
    let mut header = empty_header(0, state.key_frame, true, false);
    header.show_existing = true;
    header.frame_to_show = slot;
    header.refresh_flags = 0;
    header.width = state.width;
    header.height = state.height;
    header.show_width = state.ref_w[index];
    header.show_height = state.ref_h[index];
    header.render_width = header.show_width;
    header.render_height = header.show_height;
    header.filter_level = 0;
    header.intra_only = state.intra_only;
    header.need_resync = state.need_resync;
    header.sign_bias = state.sign_bias;
    header.seg = state.seg.clone();
    header.ref_deltas = state.ref_deltas;
    header.mode_deltas = state.mode_deltas;
    let consumed = if bits.bytes_read() == 0 {
        1
    } else {
        bits.bytes_read()
    };
    let end = consumed.min(data.len());
    Ok(Parsed {
        header,
        fc: FrameContext::defaults(),
        contexts: state.contexts.clone(),
        tiles: &data[end..],
    })
}

fn read_key_setup(bits: &mut BitReader<'_>, header: &mut Header) -> Result<(), CodecError> {
    read_sync(bits)?;
    read_color(bits, header)?;
    header.refresh_flags = 0xff;
    read_frame_size(bits, header)?;
    Ok(())
}

fn read_inter_setup(
    bits: &mut BitReader<'_>,
    header: &mut Header,
    sign_bias: &mut [u8; 4],
    need_resync: &mut bool,
    state: &Persist,
) -> Result<(), CodecError> {
    header.intra_only = if header.show_frame {
        false
    } else {
        bits.read(1)? == 1
    };
    header.reset_frame_context = if header.error_resilient {
        0
    } else {
        u8::try_from(bits.read(2)?).unwrap_or(0)
    };
    if header.intra_only {
        read_intra_only(bits, header)?;
        *need_resync = false;
        return Ok(());
    }
    if *need_resync {
        return Err(bad(bits, "keyframe required"));
    }
    header.refresh_flags = u8::try_from(bits.read(8)?).unwrap_or(0);
    for i in 0..REFS_PER_FRAME {
        let slot = u8::try_from(bits.read(3)?).unwrap_or(0);
        let index = usize::from(slot);
        if !state.ref_ok.get(index).copied().unwrap_or(false) {
            return Err(bad(bits, "reference index"));
        }
        header.ref_index[i] = slot;
        header.ref_size[i] = (state.ref_w[index], state.ref_h[index]);
        let bias_at = usize::from(LAST_FRAME) + i;
        sign_bias[bias_at] = u8::try_from(bits.read(1)?).unwrap_or(0);
    }
    read_size_with_refs(bits, header)?;
    header.allow_hp = bits.read(1)? == 1;
    header.interp = read_interp(bits)?;
    Ok(())
}

fn read_intra_only(bits: &mut BitReader<'_>, header: &mut Header) -> Result<(), CodecError> {
    read_sync(bits)?;
    header.color_space = 1;
    header.color_range = 0;
    header.refresh_flags = u8::try_from(bits.read(8)?).unwrap_or(0);
    read_frame_size(bits, header)
}

fn read_context_flags(bits: &mut BitReader<'_>, header: &mut Header) -> Result<(), CodecError> {
    if header.error_resilient {
        header.refresh_context = false;
        header.frame_parallel = true;
    } else {
        header.refresh_context = bits.read(1)? == 1;
        header.frame_parallel = bits.read(1)? == 1;
    }
    header.context_idx = u8::try_from(bits.read(2)?).unwrap_or(0);
    Ok(())
}

fn reset_past(
    contexts: &mut [FrameContext; 4],
    seg: &mut SegState,
    ref_deltas: &mut [i8; 4],
    mode_deltas: &mut [i8; 2],
    sign_bias: &mut [u8; 4],
    header: &mut Header,
) {
    clear_features(seg);
    seg.abs_delta = false;
    *ref_deltas = [1, 0, -1, -1];
    *mode_deltas = [0, 0];
    let fresh = FrameContext::defaults();
    let reset_all = header.key_frame || header.error_resilient || header.reset_frame_context == 3;
    if reset_all {
        for ctx in contexts.iter_mut() {
            *ctx = fresh.clone();
        }
    } else if header.reset_frame_context == 2 {
        contexts[usize::from(header.context_idx)] = fresh;
    }
    *sign_bias = [0; 4];
    header.context_idx = 0;
}

fn read_loop_filter(
    bits: &mut BitReader<'_>,
    ref_deltas: &mut [i8; 4],
    mode_deltas: &mut [i8; 2],
    header: &mut Header,
) -> Result<(u8, u8), CodecError> {
    let level = u8::try_from(bits.read(6)?).unwrap_or(0);
    let sharp = u8::try_from(bits.read(3)?).unwrap_or(0);
    header.delta_enabled = bits.read(1)? == 1;
    header.delta_update = false;
    if header.delta_enabled {
        header.delta_update = bits.read(1)? == 1;
        if header.delta_update {
            read_delta_list(bits, ref_deltas)?;
            read_delta_list(bits, mode_deltas)?;
        }
    }
    Ok((level, sharp))
}

fn read_delta_list(bits: &mut BitReader<'_>, slots: &mut [i8]) -> Result<(), CodecError> {
    for slot in slots {
        if bits.read(1)? == 1 {
            *slot = read_signed(bits, 6)?;
        }
    }
    Ok(())
}

fn read_quant(bits: &mut BitReader<'_>, header: &mut Header) -> Result<(), CodecError> {
    header.base_qindex = u8::try_from(bits.read(8)?).unwrap_or(0);
    header.y_dc_delta = read_delta_q(bits)?;
    header.uv_dc_delta = read_delta_q(bits)?;
    header.uv_ac_delta = read_delta_q(bits)?;
    header.lossless = header.base_qindex == 0
        && header.y_dc_delta == 0
        && header.uv_dc_delta == 0
        && header.uv_ac_delta == 0;
    Ok(())
}

fn read_delta_q(bits: &mut BitReader<'_>) -> Result<i8, CodecError> {
    if bits.read(1)? == 0 {
        Ok(0)
    } else {
        read_signed(bits, 4)
    }
}

fn read_segmentation(bits: &mut BitReader<'_>, seg: &mut SegState) -> Result<(), CodecError> {
    seg.update_map = false;
    seg.update_data = false;
    seg.enabled = bits.read(1)? == 1;
    if !seg.enabled {
        return Ok(());
    }
    seg.update_map = bits.read(1)? == 1;
    if seg.update_map {
        read_prob_list(bits, &mut seg.tree_probs)?;
        seg.temporal = bits.read(1)? == 1;
        if seg.temporal {
            read_prob_list(bits, &mut seg.pred_probs)?;
        } else {
            seg.pred_probs = [MAX_PROB; 3];
        }
    }
    seg.update_data = bits.read(1)? == 1;
    if seg.update_data {
        seg.abs_delta = bits.read(1)? == 1;
        clear_features(seg);
        read_features(bits, seg)?;
    }
    Ok(())
}

fn read_prob_list(bits: &mut BitReader<'_>, probs: &mut [u8]) -> Result<(), CodecError> {
    for prob in probs {
        *prob = if bits.read(1)? == 1 {
            u8::try_from(bits.read(8)?).unwrap_or(MAX_PROB)
        } else {
            MAX_PROB
        };
    }
    Ok(())
}

fn read_features(bits: &mut BitReader<'_>, seg: &mut SegState) -> Result<(), CodecError> {
    for seg_id in 0..8 {
        for feature in 0..4 {
            if bits.read(1)? == 0 {
                continue;
            }
            seg.feature_on[seg_id][feature] = true;
            let max = feature_max(feature);
            let mut data = i32::try_from(unsigned_max(bits, max)?).unwrap_or(0);
            if feature < 2 && bits.read(1)? == 1 {
                data = -data;
            }
            seg.feature_data[seg_id][feature] = i16::try_from(data).unwrap_or(0);
        }
    }
    Ok(())
}

fn read_tiles(bits: &mut BitReader<'_>, header: &mut Header) -> Result<(), CodecError> {
    let mi_cols = geom::mi_count(header.width);
    let (min_log2, max_log2) = tile_limits(mi_cols);
    let mut cols = min_log2;
    let mut ones = max_log2.saturating_sub(min_log2);
    while ones > 0 && bits.read(1)? == 1 {
        cols += 1;
        ones -= 1;
    }
    if cols > 6 {
        return Err(bad(bits, "tile columns"));
    }
    let mut rows = bits.read(1)?;
    if rows == 1 {
        rows += bits.read(1)?;
    }
    header.log2_tile_cols = u8::try_from(cols).unwrap_or(0);
    header.log2_tile_rows = u8::try_from(rows).unwrap_or(0);
    Ok(())
}

fn read_compressed(
    header: &mut Header,
    fc: &mut FrameContext,
    data: &[u8],
) -> Result<(), CodecError> {
    let mut reader = BoolDecoder::new(data)?;
    header.tx_mode = if header.lossless {
        ONLY_4X4
    } else {
        read_tx_mode(&mut reader)?
    };
    if header.tx_mode == geom::TX_MODE_SELECT {
        read_tx_probs(fc, &mut reader)?;
    }
    read_coef_probs(fc, header.tx_mode, &mut reader)?;
    for prob in &mut fc.skip {
        prob::diff_update(&mut reader, prob)?;
    }
    if !header.intra() {
        read_inter_probs(header, fc, &mut reader)?;
    }
    reader.finish()
}

fn read_tx_mode(reader: &mut BoolDecoder<'_>) -> Result<u8, CodecError> {
    let mut mode = reader.read_literal(2)?;
    if mode == u32::from(geom::ALLOW_32X32) {
        mode += reader.read(128)?;
    }
    Ok(u8::try_from(mode).unwrap_or(ONLY_4X4))
}

fn read_tx_probs(fc: &mut FrameContext, reader: &mut BoolDecoder<'_>) -> Result<(), CodecError> {
    update_rows(&mut fc.tx8, reader)?;
    update_rows(&mut fc.tx16, reader)?;
    update_rows(&mut fc.tx32, reader)?;
    Ok(())
}

fn update_rows<const N: usize>(
    rows: &mut [[u8; N]; 2],
    reader: &mut BoolDecoder<'_>,
) -> Result<(), CodecError> {
    for row in rows {
        for prob in row {
            prob::diff_update(reader, prob)?;
        }
    }
    Ok(())
}

fn read_coef_probs(
    fc: &mut FrameContext,
    tx_mode: u8,
    reader: &mut BoolDecoder<'_>,
) -> Result<(), CodecError> {
    let max_tx = usize::from(geom::tx_mode_biggest(tx_mode));
    for tx in 0..=max_tx {
        if reader.read(128)? == 0 {
            continue;
        }
        update_coef_tx(fc, tx, reader)?;
    }
    Ok(())
}

fn update_coef_tx(
    fc: &mut FrameContext,
    tx: usize,
    reader: &mut BoolDecoder<'_>,
) -> Result<(), CodecError> {
    for plane in 0..2 {
        for reference in 0..2 {
            update_coef_band(&mut fc.coef[tx][plane][reference], reader)?;
        }
    }
    Ok(())
}

fn update_coef_band(
    bands: &mut [[[u8; 3]; 6]; 6],
    reader: &mut BoolDecoder<'_>,
) -> Result<(), CodecError> {
    for (band, contexts) in bands.iter_mut().enumerate() {
        let kept = if band == 0 { 3 } else { 6 };
        for ctx in contexts.iter_mut().take(kept) {
            for prob in ctx {
                prob::diff_update(reader, prob)?;
            }
        }
    }
    Ok(())
}

fn read_inter_probs(
    header: &mut Header,
    fc: &mut FrameContext,
    reader: &mut BoolDecoder<'_>,
) -> Result<(), CodecError> {
    for row in &mut fc.inter_mode {
        for prob in row {
            prob::diff_update(reader, prob)?;
        }
    }
    if header.interp == SWITCHABLE {
        for row in &mut fc.interp {
            for prob in row {
                prob::diff_update(reader, prob)?;
            }
        }
    }
    for prob in &mut fc.intra_inter {
        prob::diff_update(reader, prob)?;
    }
    header.reference_mode = read_reference_mode(header, reader)?;
    if header.reference_mode != SINGLE_REFERENCE {
        setup_compound(header);
    }
    read_reference_probs(header, fc, reader)?;
    for row in &mut fc.y_mode {
        for prob in row {
            prob::diff_update(reader, prob)?;
        }
    }
    for row in &mut fc.partition {
        for prob in row {
            prob::diff_update(reader, prob)?;
        }
    }
    read_mv_probs(fc, header.allow_hp, reader)
}

fn read_reference_mode(header: &Header, reader: &mut BoolDecoder<'_>) -> Result<u8, CodecError> {
    if !compound_allowed(&header.sign_bias) {
        return Ok(SINGLE_REFERENCE);
    }
    if reader.read(128)? == 0 {
        return Ok(SINGLE_REFERENCE);
    }
    if reader.read(128)? == 1 {
        Ok(REFERENCE_SELECT)
    } else {
        Ok(COMPOUND_REFERENCE)
    }
}

fn read_reference_probs(
    header: &Header,
    fc: &mut FrameContext,
    reader: &mut BoolDecoder<'_>,
) -> Result<(), CodecError> {
    if header.reference_mode == REFERENCE_SELECT {
        for prob in &mut fc.comp_inter {
            prob::diff_update(reader, prob)?;
        }
    }
    if header.reference_mode != COMPOUND_REFERENCE {
        for row in &mut fc.single_ref {
            for prob in row {
                prob::diff_update(reader, prob)?;
            }
        }
    }
    if header.reference_mode != SINGLE_REFERENCE {
        for prob in &mut fc.comp_ref {
            prob::diff_update(reader, prob)?;
        }
    }
    Ok(())
}

fn read_mv_probs(
    fc: &mut FrameContext,
    allow_hp: bool,
    reader: &mut BoolDecoder<'_>,
) -> Result<(), CodecError> {
    prob::update_mv(reader, &mut fc.mv.joints)?;
    for comp in &mut fc.mv.comp {
        prob::update_mv(reader, std::slice::from_mut(&mut comp.sign))?;
        prob::update_mv(reader, &mut comp.classes)?;
        prob::update_mv(reader, std::slice::from_mut(&mut comp.class0))?;
        prob::update_mv(reader, &mut comp.bits)?;
    }
    for comp in &mut fc.mv.comp {
        for row in &mut comp.class0_fp {
            prob::update_mv(reader, row)?;
        }
        prob::update_mv(reader, &mut comp.fp)?;
    }
    if allow_hp {
        for comp in &mut fc.mv.comp {
            prob::update_mv(reader, std::slice::from_mut(&mut comp.class0_hp))?;
            prob::update_mv(reader, std::slice::from_mut(&mut comp.hp))?;
        }
    }
    Ok(())
}

fn setup_compound(header: &mut Header) {
    let bias = &header.sign_bias;
    if bias[usize::from(LAST_FRAME)] == bias[usize::from(GOLDEN_FRAME)] {
        header.comp_fixed = ALTREF_FRAME;
        header.comp_var = [LAST_FRAME, GOLDEN_FRAME];
    } else if bias[usize::from(LAST_FRAME)] == bias[usize::from(ALTREF_FRAME)] {
        header.comp_fixed = GOLDEN_FRAME;
        header.comp_var = [LAST_FRAME, ALTREF_FRAME];
    } else {
        header.comp_fixed = LAST_FRAME;
        header.comp_var = [GOLDEN_FRAME, ALTREF_FRAME];
    }
}

fn compound_allowed(bias: &[u8; 4]) -> bool {
    bias[usize::from(GOLDEN_FRAME)] != bias[usize::from(LAST_FRAME)]
        || bias[usize::from(ALTREF_FRAME)] != bias[usize::from(LAST_FRAME)]
}

fn load_context(contexts: &[FrameContext; 4], index: u8) -> Result<FrameContext, CodecError> {
    let ctx = &contexts[usize::from(index)];
    if !ctx.initialized {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "entropy context",
        });
    }
    Ok(ctx.clone())
}

fn fill_dequant(header: &mut Header) {
    let base = i32::from(header.base_qindex);
    for seg_id in 0..8 {
        let q = seg_qindex(&header.seg, seg_id, base);
        header.y_dq[seg_id][0] = lookup(&DC_QLOOKUP, q, i32::from(header.y_dc_delta));
        header.y_dq[seg_id][1] = lookup(&AC_QLOOKUP, q, 0);
        header.uv_dq[seg_id][0] = lookup(&DC_QLOOKUP, q, i32::from(header.uv_dc_delta));
        header.uv_dq[seg_id][1] = lookup(&AC_QLOOKUP, q, i32::from(header.uv_ac_delta));
    }
}

fn seg_qindex(seg: &SegState, seg_id: usize, base: i32) -> i32 {
    if seg.enabled && seg.feature_on[seg_id][0] {
        let data = i32::from(seg.feature_data[seg_id][0]);
        let q = if seg.abs_delta { data } else { base + data };
        q.clamp(0, 255)
    } else {
        base.clamp(0, 255)
    }
}

fn lookup(table: &[i16; 256], qindex: i32, delta: i32) -> i16 {
    let index = (qindex + delta).clamp(0, 255);
    table[usize::try_from(index).unwrap_or(0)]
}

fn use_prev(state: &Persist, header: &Header) -> bool {
    !header.error_resilient
        && header.width == state.last_width
        && header.height == state.last_height
        && !state.intra_only
        && state.last_show_frame
        && !state.key_frame
}

fn read_profile(bits: &mut BitReader<'_>) -> Result<u8, CodecError> {
    let mut profile = bits.read(1)?;
    profile |= bits.read(1)? << 1;
    if profile > 2 {
        profile += bits.read(1)?;
    }
    Ok(u8::try_from(profile).unwrap_or(0))
}

fn read_sync(bits: &mut BitReader<'_>) -> Result<(), CodecError> {
    if bits.read(8)? == 0x49 && bits.read(8)? == 0x83 && bits.read(8)? == 0x42 {
        Ok(())
    } else {
        Err(bad(bits, "sync code"))
    }
}

fn read_color(bits: &mut BitReader<'_>, header: &mut Header) -> Result<(), CodecError> {
    header.color_space = u8::try_from(bits.read(3)?).unwrap_or(0);
    if header.color_space == 7 {
        return Err(bad(bits, "profile 0 rgb"));
    }
    header.color_range = u8::try_from(bits.read(1)?).unwrap_or(0);
    Ok(())
}

fn read_frame_size(bits: &mut BitReader<'_>, header: &mut Header) -> Result<(), CodecError> {
    let width = bits.read(16)?.saturating_add(1);
    let height = bits.read(16)?.saturating_add(1);
    check_edge(width)?;
    check_edge(height)?;
    header.width = width;
    header.height = height;
    header.show_width = width;
    header.show_height = height;
    if bits.read(1)? == 1 {
        let render_w = bits.read(16)?.saturating_add(1);
        let render_h = bits.read(16)?.saturating_add(1);
        check_edge(render_w)?;
        check_edge(render_h)?;
        header.render_width = render_w;
        header.render_height = render_h;
    } else {
        header.render_width = width;
        header.render_height = height;
    }
    Ok(())
}

fn read_size_with_refs(bits: &mut BitReader<'_>, header: &mut Header) -> Result<(), CodecError> {
    let mut found = false;
    for i in 0..REFS_PER_FRAME {
        if bits.read(1)? == 1 {
            header.width = header.ref_size[i].0;
            header.height = header.ref_size[i].1;
            found = true;
            break;
        }
    }
    if !found {
        let width = bits.read(16)?.saturating_add(1);
        let height = bits.read(16)?.saturating_add(1);
        header.width = width;
        header.height = height;
    }
    check_edge(header.width)?;
    check_edge(header.height)?;
    let mut valid = false;
    for size in header.ref_size {
        valid |= valid_ref_size(size.0, size.1, header.width, header.height);
    }
    if !valid {
        return Err(bad(bits, "reference size"));
    }
    header.show_width = header.width;
    header.show_height = header.height;
    if bits.read(1)? == 1 {
        header.render_width = bits.read(16)?.saturating_add(1);
        header.render_height = bits.read(16)?.saturating_add(1);
        check_edge(header.render_width)?;
        check_edge(header.render_height)?;
    } else {
        header.render_width = header.width;
        header.render_height = header.height;
    }
    Ok(())
}

fn read_interp(bits: &mut BitReader<'_>) -> Result<u8, CodecError> {
    if bits.read(1)? == 1 {
        return Ok(SWITCHABLE);
    }
    const MAP: [u8; 4] = [geom::SMOOTH, geom::EIGHTTAP, geom::SHARP, geom::BILINEAR];
    let index = usize::try_from(bits.read(2)?).unwrap_or(0);
    Ok(MAP[index])
}

fn valid_ref_size(ref_w: u32, ref_h: u32, width: u32, height: u32) -> bool {
    width.saturating_mul(2) >= ref_w
        && height.saturating_mul(2) >= ref_h
        && width <= ref_w.saturating_mul(16)
        && height <= ref_h.saturating_mul(16)
}

fn check_edge(edge: u32) -> Result<(), CodecError> {
    if edge == 0 || edge > geom::MAX_EDGE {
        Err(CodecError::Limit {
            what: "vp9 frame edge",
        })
    } else {
        Ok(())
    }
}

fn tile_limits(mi_cols: usize) -> (u32, u32) {
    let sb64 = mi_cols.div_ceil(8);
    let mut min_log2 = 0u32;
    while (64u32 << min_log2) < u32::try_from(sb64).unwrap_or(u32::MAX) {
        min_log2 += 1;
    }
    let mut max_log2 = 1u32;
    while sb64 >> max_log2 >= 4 {
        max_log2 += 1;
    }
    (min_log2, max_log2.saturating_sub(1))
}

fn unsigned_bits(max: u32) -> u32 {
    if max == 0 {
        0
    } else {
        32 - max.leading_zeros()
    }
}

fn unsigned_max(bits: &mut BitReader<'_>, max: u32) -> Result<u32, CodecError> {
    let value = bits.read(unsigned_bits(max))?;
    Ok(value.min(max))
}

fn feature_max(feature: usize) -> u32 {
    match feature {
        0 => 255,
        1 => 63,
        2 => 3,
        _ => 0,
    }
}

fn clear_features(seg: &mut SegState) {
    seg.feature_on = [[false; 4]; 8];
    seg.feature_data = [[0; 4]; 8];
}

fn read_signed(bits: &mut BitReader<'_>, width: u32) -> Result<i8, CodecError> {
    let mag = i32::try_from(bits.read(width)?).unwrap_or(0);
    let value = if bits.read(1)? == 1 { -mag } else { mag };
    i8::try_from(value).map_err(|_| bad(bits, "signed literal"))
}

fn empty_header(profile: u8, key_frame: bool, show_frame: bool, error_resilient: bool) -> Header {
    Header {
        profile,
        show_existing: false,
        frame_to_show: 0,
        key_frame,
        show_frame,
        error_resilient,
        intra_only: false,
        reset_frame_context: 0,
        refresh_flags: 0,
        color_space: 0,
        color_range: 0,
        width: 0,
        height: 0,
        render_width: 0,
        render_height: 0,
        show_width: 0,
        show_height: 0,
        ref_index: [0; REFS_PER_FRAME],
        ref_size: [(0, 0); REFS_PER_FRAME],
        allow_hp: false,
        interp: geom::EIGHTTAP,
        refresh_context: false,
        frame_parallel: false,
        context_idx: 0,
        filter_level: 0,
        sharpness: 0,
        delta_enabled: false,
        delta_update: false,
        ref_deltas: [0; 4],
        mode_deltas: [0; 2],
        base_qindex: 0,
        y_dc_delta: 0,
        uv_dc_delta: 0,
        uv_ac_delta: 0,
        lossless: false,
        y_dq: [[0; 2]; 8],
        uv_dq: [[0; 2]; 8],
        seg: SegState::default(),
        log2_tile_cols: 0,
        log2_tile_rows: 0,
        tx_mode: ONLY_4X4,
        reference_mode: SINGLE_REFERENCE,
        comp_fixed: LAST_FRAME,
        comp_var: [GOLDEN_FRAME, ALTREF_FRAME],
        sign_bias: [0; 4],
        use_prev_mvs: false,
        need_resync: true,
        header_bytes: 0,
        compressed_bytes: 0,
    }
}

fn bad(bits: &BitReader<'_>, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: bits.position() as u64,
        what,
    }
}

impl Header {
    pub(crate) fn is_intra(&self) -> bool {
        self.key_frame || self.intra_only
    }

    fn intra(&self) -> bool {
        self.is_intra()
    }
}

#[cfg(test)]
mod tests {
    use super::{Persist, parse};
    use crate::tables::COEF_PROBS;

    /// libvpx-vp9 lossless 16×16 keyframe, IVF payload only.
    const RED_16: [u8; 42] = [
        0x82, 0x49, 0x83, 0x42, 0x00, 0x00, 0xf0, 0x00, 0xf6, 0x00, 0x38, 0x24, 0x1c, 0x18, 0x00,
        0x00, 0x00, 0x20, 0x00, 0x00, 0x10, 0xbf, 0xff, 0xff, 0xc8, 0xd7, 0x80, 0x31, 0xff, 0xff,
        0xff, 0xec, 0xfc, 0x87, 0x0f, 0xff, 0xff, 0xff, 0x6c, 0x8c, 0x00, 0x00,
    ];

    #[test]
    fn lossless_keyframe_header_matches_libvpx() {
        let parsed = parse(&Persist::new(), &RED_16).expect("header");
        let header = &parsed.header;
        assert_eq!(header.profile, 0);
        assert!(header.key_frame);
        assert!(header.show_frame);
        assert!(!header.error_resilient);
        assert_eq!((header.width, header.height), (16, 16));
        assert_eq!(header.color_space, 0);
        assert_eq!(header.color_range, 0);
        assert_eq!(header.base_qindex, 0);
        assert!(header.lossless);
        assert_eq!(header.y_dq[0], [4, 4]);
        assert_eq!(header.filter_level, 0);
        assert!(header.delta_enabled);
        assert_eq!(header.ref_deltas, [1, 0, -1, -1]);
        assert_eq!(header.header_bytes, 18);
        assert_eq!(header.compressed_bytes, 2);
        assert_eq!(header.tx_mode, 0);
        assert!(header.refresh_context);
        assert!(header.frame_parallel);
        assert_eq!(parsed.tiles.first().copied(), Some(0x10));
        assert_eq!(parsed.fc.coef[0][0][0][0][0], COEF_PROBS[0][0][0][0][0]);
        assert!(parsed.contexts[0].initialized);
    }

    fn must_fail(bytes: &[u8]) -> String {
        match parse(&Persist::new(), bytes) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("parser accepted {bytes:?}"),
        }
    }

    #[test]
    fn profile_one_is_unsupported() {
        let err = must_fail(&[0xa0]);
        assert!(err.contains("profile"), "{err}");
    }

    #[test]
    fn bad_marker_and_sync_are_malformed() {
        let marker = must_fail(&[0x00]);
        assert!(marker.contains("marker"), "{marker}");
        let sync = must_fail(&[0x82, 0x00, 0x00, 0x00]);
        assert!(sync.contains("sync"), "{sync}");
    }

    #[test]
    fn empty_packet_is_truncated() {
        let err = must_fail(&[]);
        assert!(err.contains("truncated"), "{err}");
    }
}
