//! Frame-level decoding: OBU dispatch, reference frames, tiles, filters and output.

use std::sync::Arc;

use cutlass_decode::CodecError;

use crate::dsp::Pixel;
use crate::dsp::fg::{FilmGrainParams, apply_film_grain};
use crate::dsp::lr::LrScratch;

use super::cdf::CdfContext;
use super::frame::{FrameState, LrUnitParams, Picture, RefData};
use super::msac::Msac;
use super::mvpred::motion_field_estimation;
use super::obu::{
    FrameHeader, GrainParams, HeaderContext, KEY_FRAME, OBU_FRAME, OBU_FRAME_HEADER,
    OBU_REDUNDANT_FRAME_HEADER, OBU_SEQUENCE_HEADER, OBU_TILE_GROUP, OBU_TILE_LIST,
    PRIMARY_REF_NONE, RefHeaderState, SequenceHeader, parse_frame_header, parse_sequence_header,
    relative_dist, split_obus,
};
use super::postfilter;
use super::recon::Scratch;
use super::tile::{RefView, RowHook, TileDec, malformed};
use super::types::{INTRA_FRAME, MiInfo, Mv};

type R<T> = Result<T, CodecError>;

/// A decoded picture of either sample size.
#[derive(Clone)]
pub enum AnyPicture {
    U8(Picture<u8>),
    U16(Picture<u16>),
}

/// A sample type that can be viewed inside an [`AnyPicture`].
pub trait PicturePixel: Pixel {
    fn view(p: &AnyPicture) -> Option<&Picture<Self>>;
    fn wrap(p: Picture<Self>) -> AnyPicture;
}

impl PicturePixel for u8 {
    fn view(p: &AnyPicture) -> Option<&Picture<u8>> {
        match p {
            AnyPicture::U8(x) => Some(x),
            AnyPicture::U16(_) => None,
        }
    }
    fn wrap(p: Picture<u8>) -> AnyPicture {
        AnyPicture::U8(p)
    }
}

impl PicturePixel for u16 {
    fn view(p: &AnyPicture) -> Option<&Picture<u16>> {
        match p {
            AnyPicture::U16(x) => Some(x),
            AnyPicture::U8(_) => None,
        }
    }
    fn wrap(p: Picture<u16>) -> AnyPicture {
        AnyPicture::U16(p)
    }
}

/// One shown frame.
#[derive(Clone)]
pub struct DecodedFrame {
    pub picture: Arc<AnyPicture>,
    pub width: usize,
    pub height: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub bit_depth: u32,
    pub mono: bool,
    pub sub_x: bool,
    pub sub_y: bool,
    pub color_primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
    pub full_range: bool,
}

#[derive(Clone)]
struct Slot {
    pic: Arc<AnyPicture>,
    data: Arc<RefData>,
    cdf: Arc<CdfContext>,
}

/// The decoder state that persists between temporal units.
pub struct Av1Core {
    seq: Option<SequenceHeader>,
    ref_hdr: [RefHeaderState; 8],
    slots: [Option<Slot>; 8],
    /// Frame in progress (header seen, tiles pending).
    pending: Option<Pending>,
    lr_scratch: LrScratch,
    /// The in-loop filter thread's copy of the frame state, kept for reuse.
    filter_fs: Option<FrameState>,
    /// Apply film grain to shown pictures.
    pub apply_grain: bool,
    pool8: Vec<Picture<u8>>,
    pool16: Vec<Picture<u16>>,
    /// Pictures handed out (shown or still referenced) that return to the
    /// pools once nothing else holds them.
    outstanding: Vec<Arc<AnyPicture>>,
    /// Largest picture (upscaled width x height) accepted; larger is `Limit`.
    /// The default is the largest picture any AV1 level allows.
    pub max_pixels: u64,
    /// Threads the in-loop filters (deblocking, CDEF, loop restoration) may
    /// use; 1 runs them on the calling thread. Defaults to
    /// `std::thread::available_parallelism()`.
    pub threads: usize,
}

struct Pending {
    hdr: FrameHeader,
    tiles: Vec<(usize, Vec<u8>)>,
}

impl Default for Av1Core {
    fn default() -> Self {
        Self::new()
    }
}

impl Av1Core {
    pub fn new() -> Self {
        Self {
            seq: None,
            ref_hdr: Default::default(),
            slots: Default::default(),
            pending: None,
            lr_scratch: LrScratch::new(),
            filter_fs: None,
            apply_grain: true,
            pool8: Vec::new(),
            pool16: Vec::new(),
            outstanding: Vec::new(),
            max_pixels: 35_651_584,
            threads: std::thread::available_parallelism().map_or(1, usize::from),
        }
    }

    pub fn reset(&mut self) {
        let seq = self.seq.take();
        let (grain, max, threads) = (self.apply_grain, self.max_pixels, self.threads);
        *self = Self::new();
        self.seq = seq;
        self.apply_grain = grain;
        self.max_pixels = max;
        self.threads = threads;
    }

    /// The header state of every reference slot.
    pub fn ref_headers(&self) -> &[RefHeaderState; 8] {
        &self.ref_hdr
    }

    /// The picture held by a reference slot.
    pub fn slot_picture(&self, slot: usize) -> Option<Arc<AnyPicture>> {
        self.slots.get(slot)?.as_ref().map(|s| s.pic.clone())
    }

    /// The CDFs saved with a reference slot.
    pub fn slot_cdf(&self, slot: usize) -> Option<Arc<CdfContext>> {
        self.slots.get(slot)?.as_ref().map(|s| s.cdf.clone())
    }

    /// Motion field and segmentation data saved with a reference slot.
    pub fn slot_data(&self, slot: usize) -> Option<Arc<RefData>> {
        self.slots.get(slot)?.as_ref().map(|s| s.data.clone())
    }

    pub fn sequence(&self) -> Option<&SequenceHeader> {
        self.seq.as_ref()
    }

    /// Decodes one temporal unit and returns the frames it shows. A unit that
    /// fails leaves the reference state as it was before the call.
    /// Decodes one Annex B (length-delimited) temporal unit.
    pub fn decode_annexb(&mut self, data: &[u8]) -> R<Vec<DecodedFrame>> {
        let low = super::obu::annexb_to_low_overhead(data)?;
        self.decode(&low)
    }

    pub fn decode(&mut self, data: &[u8]) -> R<Vec<DecodedFrame>> {
        let saved = (self.seq.clone(), self.ref_hdr.clone(), self.slots.clone());
        let r = self.decode_unit(data);
        if r.is_err() {
            self.seq = saved.0;
            self.ref_hdr = saved.1;
            self.slots = saved.2;
            self.pending = None;
        }
        r
    }

    fn decode_unit(&mut self, data: &[u8]) -> R<Vec<DecodedFrame>> {
        let mut out = Vec::new();
        for obu in split_obus(data)? {
            match obu.obu_type {
                OBU_SEQUENCE_HEADER => {
                    let s = parse_sequence_header(obu.payload)?;
                    if s.max_frame_width > 16384 || s.max_frame_height > 16384 {
                        return Err(CodecError::Limit { what: "frame size" });
                    }
                    self.seq = Some(s);
                }
                OBU_FRAME_HEADER | OBU_REDUNDANT_FRAME_HEADER | OBU_FRAME => {
                    if obu.obu_type == OBU_REDUNDANT_FRAME_HEADER && self.pending.is_some() {
                        continue;
                    }
                    if obu.obu_type == OBU_FRAME_HEADER && self.pending.is_some() {
                        // frame_header_copy
                        continue;
                    }
                    let Some(seq) = self.seq.clone() else {
                        return Err(malformed("frame before sequence header"));
                    };
                    if let Some(op) = seq.operating_points.first()
                        && op.idc != 0
                        && obu.has_extension
                    {
                        let in_t = (op.idc >> obu.temporal_id) & 1;
                        let in_s = (op.idc >> (obu.spatial_id + 8)) & 1;
                        if in_t == 0 || in_s == 0 {
                            continue;
                        }
                    }
                    let ctx = HeaderContext {
                        seq: &seq,
                        refs: &self.ref_hdr,
                        temporal_id: obu.temporal_id,
                        spatial_id: obu.spatial_id,
                    };
                    let hdr = parse_frame_header(obu.payload, &ctx)?;
                    if hdr.show_existing_frame {
                        if let Some(f) = self.show_existing(&seq, &hdr)? {
                            out.push(f);
                        }
                        continue;
                    }
                    if u64::from(hdr.upscaled_width) * u64::from(hdr.frame_height) > self.max_pixels
                    {
                        return Err(CodecError::Limit { what: "frame size" });
                    }
                    let mut p = Pending {
                        hdr,
                        tiles: Vec::new(),
                    };
                    if obu.obu_type == OBU_FRAME {
                        let rest = &obu.payload[p.hdr.header_bytes..];
                        self.add_tile_group(&mut p, rest)?;
                    }
                    let done = p.tiles.len() == p.hdr.tile_info.cols * p.hdr.tile_info.rows;
                    if done {
                        if let Some(f) = self.finish_frame(&seq, p)? {
                            out.push(f);
                        }
                    } else {
                        self.pending = Some(p);
                    }
                }
                OBU_TILE_GROUP => {
                    let Some(mut p) = self.pending.take() else {
                        return Err(malformed("tile group without frame header"));
                    };
                    let Some(seq) = self.seq.clone() else {
                        return Err(malformed("frame before sequence header"));
                    };
                    self.add_tile_group(&mut p, obu.payload)?;
                    if p.tiles.len() == p.hdr.tile_info.cols * p.hdr.tile_info.rows {
                        if let Some(f) = self.finish_frame(&seq, p)? {
                            out.push(f);
                        }
                    } else {
                        self.pending = Some(p);
                    }
                }
                OBU_TILE_LIST => {
                    return Err(CodecError::Unsupported {
                        what: "large scale tile list",
                    });
                }
                _ => {}
            }
        }
        Ok(out)
    }

    fn add_tile_group(&mut self, p: &mut Pending, data: &[u8]) -> R<()> {
        let ti = &p.hdr.tile_info;
        let num_tiles = ti.cols * ti.rows;
        let mut r = super::bits::BitReader::new(data);
        let mut start_end_present = false;
        if num_tiles > 1 {
            start_end_present = r.flag()?;
        }
        let (tg_start, tg_end) = if num_tiles == 1 || !start_end_present {
            (0, num_tiles - 1)
        } else {
            let bits = ti.cols_log2 + ti.rows_log2;
            (r.f(bits)? as usize, r.f(bits)? as usize)
        };
        r.byte_alignment();
        let mut pos = r.position() / 8;
        if tg_end >= num_tiles || tg_start > tg_end || tg_start != p.tiles.len() {
            return Err(malformed("tile group range"));
        }
        for tile in tg_start..=tg_end {
            let size = if tile == tg_end {
                data.len() - pos
            } else {
                let n = ti.tile_size_bytes as usize;
                if pos + n > data.len() {
                    return Err(CodecError::Truncated {
                        at_bit: (pos * 8) as u64,
                        what: "tile size",
                    });
                }
                let mut v = 0usize;
                for i in 0..n {
                    v |= usize::from(data[pos + i]) << (8 * i);
                }
                pos += n;
                v + 1
            };
            if pos + size > data.len() {
                return Err(CodecError::Truncated {
                    at_bit: (pos * 8) as u64,
                    what: "tile data",
                });
            }
            p.tiles.push((tile, data[pos..pos + size].to_vec()));
            pos += size;
        }
        Ok(())
    }

    fn show_existing(
        &mut self,
        seq: &SequenceHeader,
        hdr: &FrameHeader,
    ) -> R<Option<DecodedFrame>> {
        let idx = hdr.frame_to_show_map_idx as usize;
        let Some(slot) = self.slots[idx].clone() else {
            return Err(malformed("show_existing_frame of empty slot"));
        };
        let rh = self.ref_hdr[idx].clone();
        let mut grain = rh.grain.clone();
        grain.grain_seed = hdr.grain.grain_seed;
        if seq.film_grain_params_present {
            grain = hdr.grain.clone();
        }
        let frame = self.output(seq, &slot.pic, &rh, &grain);
        if rh.frame_type == KEY_FRAME {
            // Reference frame loading process (7.21) then update all slots.
            let mut new_hdr = rh.clone();
            new_hdr.showable = false;
            for i in 0..8 {
                self.ref_hdr[i] = new_hdr.clone();
                self.slots[i] = Some(slot.clone());
            }
        }
        Ok(Some(frame))
    }

    fn output(
        &self,
        seq: &SequenceHeader,
        pic: &Arc<AnyPicture>,
        rh: &RefHeaderState,
        grain: &GrainParams,
    ) -> DecodedFrame {
        let w = rh.upscaled_width as usize;
        let h = rh.frame_height as usize;
        let mono = seq.mono_chrome;
        let picture = if self.apply_grain && grain.apply_grain {
            let p = to_fg_params(grain);
            Arc::new(match &**pic {
                AnyPicture::U8(src) => {
                    AnyPicture::U8(grain_apply(src, &p, w, h, mono, seq.matrix_coefficients))
                }
                AnyPicture::U16(src) => {
                    AnyPicture::U16(grain_apply(src, &p, w, h, mono, seq.matrix_coefficients))
                }
            })
        } else {
            Arc::clone(pic)
        };
        DecodedFrame {
            picture,
            width: w,
            height: h,
            render_width: rh.render_width as usize,
            render_height: rh.render_height as usize,
            bit_depth: rh.bit_depth,
            mono,
            sub_x: rh.subsampling_x,
            sub_y: rh.subsampling_y,
            color_primaries: seq.color_primaries,
            transfer: seq.transfer_characteristics,
            matrix: seq.matrix_coefficients,
            full_range: seq.color_range,
        }
    }

    fn finish_frame(&mut self, seq: &SequenceHeader, p: Pending) -> R<Option<DecodedFrame>> {
        let hdr = p.hdr;
        // Initial CDFs.
        let mut cdf: Box<CdfContext> = if hdr.primary_ref_frame == PRIMARY_REF_NONE {
            CdfContext::new(hdr.quant.base_q_idx)
        } else {
            let slot_idx = hdr.ref_frame_idx[hdr.primary_ref_frame as usize] as usize;
            let Some(slot) = &self.slots[slot_idx] else {
                return Err(malformed("primary reference frame"));
            };
            let mut c = Box::new((*slot.cdf).clone());
            c.clear_counts();
            c
        };
        if hdr.primary_ref_frame == PRIMARY_REF_NONE {
            // already initialized with defaults
        } else {
            // load_cdfs keeps coefficient CDFs from the reference.
        }
        if !hdr.frame_is_intra {
            // Every reference must share the sequence format and obey the
            // scaling limits of 7.11.3.3.
            for i in 0..7 {
                let s = hdr.ref_frame_idx[i] as usize;
                let rh = &self.ref_hdr[s];
                let fw = hdr.frame_width;
                let fh = hdr.frame_height;
                if self.slots[s].is_none()
                    || rh.bit_depth != seq.bit_depth
                    || rh.subsampling_x != seq.subsampling_x
                    || rh.subsampling_y != seq.subsampling_y
                    || 2 * fw < rh.upscaled_width
                    || 2 * fh < rh.frame_height
                    || fw > 16 * rh.upscaled_width
                    || fh > 16 * rh.frame_height
                {
                    return Err(malformed("reference frame format"));
                }
            }
        }
        self.reclaim_pictures();
        let refs_data: [Option<Arc<RefData>>; 8] =
            std::array::from_fn(|i| self.slots[i].as_ref().map(|s| s.data.clone()));
        let mut fs = FrameState::new(seq, &hdr);
        if hdr.use_ref_frame_mvs {
            fs.motion_field = motion_field_estimation(seq, &hdr, &refs_data);
        }
        if hdr.primary_ref_frame != PRIMARY_REF_NONE {
            let slot_idx = hdr.ref_frame_idx[hdr.primary_ref_frame as usize] as usize;
            if let Some(rd) = &refs_data[slot_idx]
                && rd.mi_rows == fs.mi_rows
                && rd.mi_cols == fs.mi_cols
            {
                fs.prev_segment_ids = Some(rd.segment_ids.clone());
            }
        }
        let (pic, saved_cdf) = if seq.bit_depth == 8 {
            let refs: [Option<&Picture<u8>>; 8] = std::array::from_fn(|i| {
                if i == 0 {
                    return None;
                }
                let slot = hdr.ref_frame_idx[i - 1] as usize;
                match self.slots[slot].as_ref().map(|s| &*s.pic) {
                    Some(AnyPicture::U8(p)) if !hdr.frame_is_intra => Some(p),
                    _ => None,
                }
            });
            let (pic, c) = decode_frame::<u8>(
                seq,
                &hdr,
                &mut fs,
                &p.tiles,
                &mut cdf,
                refs,
                &self.ref_hdr,
                &mut self.lr_scratch,
                &mut self.filter_fs,
                &mut self.pool8,
                self.threads,
            )?;
            (AnyPicture::U8(pic), c)
        } else {
            let refs: [Option<&Picture<u16>>; 8] = std::array::from_fn(|i| {
                if i == 0 {
                    return None;
                }
                let slot = hdr.ref_frame_idx[i - 1] as usize;
                match self.slots[slot].as_ref().map(|s| &*s.pic) {
                    Some(AnyPicture::U16(p)) if !hdr.frame_is_intra => Some(p),
                    _ => None,
                }
            });
            let (pic, c) = decode_frame::<u16>(
                seq,
                &hdr,
                &mut fs,
                &p.tiles,
                &mut cdf,
                refs,
                &self.ref_hdr,
                &mut self.lr_scratch,
                &mut self.filter_fs,
                &mut self.pool16,
                self.threads,
            )?;
            (AnyPicture::U16(pic), c)
        };
        // Reference frame update process (7.20).
        let mut final_cdf = if hdr.disable_frame_end_update_cdf {
            cdf
        } else {
            saved_cdf
        };
        final_cdf.clear_counts();
        let rh = RefHeaderState {
            valid: true,
            frame_id: hdr.current_frame_id,
            upscaled_width: hdr.upscaled_width,
            frame_width: hdr.frame_width,
            frame_height: hdr.frame_height,
            render_width: hdr.render_width,
            render_height: hdr.render_height,
            mi_cols: hdr.mi_cols,
            mi_rows: hdr.mi_rows,
            frame_type: hdr.frame_type,
            order_hint: hdr.order_hint,
            saved_order_hints: hdr.order_hints,
            gm_params: hdr.gm_params,
            loop_filter_ref_deltas: hdr.loop_filter.ref_deltas,
            loop_filter_mode_deltas: hdr.loop_filter.mode_deltas,
            feature_enabled: hdr.segmentation.feature_enabled,
            feature_data: hdr.segmentation.feature_data,
            grain: hdr.grain.clone(),
            bit_depth: seq.bit_depth,
            subsampling_x: seq.subsampling_x,
            subsampling_y: seq.subsampling_y,
            showable: hdr.showable_frame,
        };
        let pic = Arc::new(pic);
        let shown = if hdr.show_frame {
            Some(self.output(seq, &pic, &rh, &hdr.grain))
        } else {
            None
        };
        if hdr.refresh_frame_flags != 0 {
            let data = Arc::new(ref_data(seq, &hdr, &fs));
            let slot = Slot {
                pic,
                data,
                cdf: Arc::from(final_cdf),
            };
            let mut old = Vec::new();
            for i in 0..8 {
                if (hdr.refresh_frame_flags >> i) & 1 != 0 {
                    self.ref_hdr[i] = rh.clone();
                    if let Some(o) = self.slots[i].replace(slot.clone()) {
                        old.push(o.pic);
                    }
                }
            }
            drop(slot);
            for p in old {
                self.release_picture(p);
            }
        } else {
            self.release_picture(pic);
        }
        Ok(shown)
    }

    /// Recycles `p` if nothing else holds it, else remembers it until the
    /// holders (the caller's frames, other slots) let go.
    fn release_picture(&mut self, p: Arc<AnyPicture>) {
        match Arc::try_unwrap(p) {
            Ok(AnyPicture::U8(p)) => recycle(&mut self.pool8, p),
            Ok(AnyPicture::U16(p)) => recycle(&mut self.pool16, p),
            Err(p) => {
                if !self.outstanding.iter().any(|o| Arc::ptr_eq(o, &p)) {
                    if self.outstanding.len() >= 16 {
                        self.outstanding.remove(0);
                    }
                    self.outstanding.push(p);
                }
            }
        }
    }

    /// Moves outstanding pictures nobody else holds back to the pools.
    fn reclaim_pictures(&mut self) {
        let mut i = 0;
        while i < self.outstanding.len() {
            if Arc::strong_count(&self.outstanding[i]) == 1 {
                let p = self.outstanding.swap_remove(i);
                self.release_picture(p);
            } else {
                i += 1;
            }
        }
    }
}

fn ref_data(seq: &SequenceHeader, hdr: &FrameHeader, fs: &FrameState) -> RefData {
    let w8 = fs.mi_cols >> 1;
    let h8 = fs.mi_rows >> 1;
    let mut mf_ref = vec![-1i8; w8 * h8];
    let mut mf_mv = vec![Mv::ZERO; w8 * h8];
    if !hdr.frame_is_intra {
        for y8 in 0..h8 {
            for x8 in 0..w8 {
                let m = fs.mi_at(2 * y8 + 1, 2 * x8 + 1);
                for list in 0..2 {
                    let r = m.ref_frame[list];
                    if r > INTRA_FRAME {
                        let ref_idx = hdr.ref_frame_idx[r as usize - 1] as usize;
                        let dist = relative_dist(seq, hdr.ref_order_hint[ref_idx], hdr.order_hint);
                        if dist < 0 {
                            let mv = m.mv[list];
                            if i32::from(mv.row).abs() < (1 << 12)
                                && i32::from(mv.col).abs() < (1 << 12)
                            {
                                mf_ref[y8 * w8 + x8] = r;
                                mf_mv[y8 * w8 + x8] = mv;
                            }
                        }
                    }
                }
            }
        }
    }
    RefData {
        mi_cols: fs.mi_cols,
        mi_rows: fs.mi_rows,
        frame_type: hdr.frame_type,
        mf_ref,
        mf_mv,
        segment_ids: fs.segment_ids.clone(),
        saved_order_hints: hdr.order_hints,
    }
}

fn to_fg_params(g: &GrainParams) -> FilmGrainParams {
    let mut p = FilmGrainParams {
        apply_grain: g.apply_grain,
        grain_seed: g.grain_seed,
        update_grain: g.update_grain,
        num_y_points: g.num_y_points,
        chroma_scaling_from_luma: g.chroma_scaling_from_luma,
        num_cb_points: g.num_cb_points,
        num_cr_points: g.num_cr_points,
        grain_scaling_minus_8: g.grain_scaling_minus_8,
        ar_coeff_lag: g.ar_coeff_lag,
        ar_coeffs_y_plus_128: g.ar_coeffs_y_plus_128,
        ar_coeffs_cb_plus_128: g.ar_coeffs_cb_plus_128,
        ar_coeffs_cr_plus_128: g.ar_coeffs_cr_plus_128,
        ar_coeff_shift_minus_6: g.ar_coeff_shift_minus_6,
        grain_scale_shift: g.grain_scale_shift,
        cb_mult: g.cb_mult,
        cb_luma_mult: g.cb_luma_mult,
        cb_offset: g.cb_offset,
        cr_mult: g.cr_mult,
        cr_luma_mult: g.cr_luma_mult,
        cr_offset: g.cr_offset,
        overlap_flag: g.overlap_flag,
        clip_to_restricted_range: g.clip_to_restricted_range,
        ..FilmGrainParams::default()
    };
    p.point_y_value.copy_from_slice(&g.point_y_value[..14]);
    p.point_y_scaling.copy_from_slice(&g.point_y_scaling[..14]);
    p.point_cb_value.copy_from_slice(&g.point_cb_value[..10]);
    p.point_cb_scaling
        .copy_from_slice(&g.point_cb_scaling[..10]);
    p.point_cr_value.copy_from_slice(&g.point_cr_value[..10]);
    p.point_cr_scaling
        .copy_from_slice(&g.point_cr_scaling[..10]);
    p
}

fn grain_apply<P: Pixel>(
    src: &Picture<P>,
    p: &FilmGrainParams,
    w: usize,
    h: usize,
    mono: bool,
    matrix: u8,
) -> Picture<P> {
    let mut dst = src.clone();
    let empty: [P; 0] = [];
    let (su, sv, suv_stride) = if mono {
        (&empty[..], &empty[..], 0)
    } else {
        (
            &src.planes[1].data[..],
            &src.planes[2].data[..],
            src.planes[1].stride,
        )
    };
    let (dy, rest) = dst.planes.split_at_mut(1);
    let ys = dy[0].stride;
    if mono {
        let mut e1: [P; 0] = [];
        let mut e2: [P; 0] = [];
        apply_film_grain(
            p,
            &src.planes[0].data,
            su,
            sv,
            src.planes[0].stride,
            suv_stride,
            &mut dy[0].data,
            &mut e1,
            &mut e2,
            ys,
            0,
            w,
            h,
            src.sub_x,
            src.sub_y,
            true,
            matrix,
            src.bd,
        );
    } else {
        let (du, dv) = rest.split_at_mut(1);
        let uvs = du[0].stride;
        apply_film_grain(
            p,
            &src.planes[0].data,
            su,
            sv,
            src.planes[0].stride,
            suv_stride,
            &mut dy[0].data,
            &mut du[0].data,
            &mut dv[0].data,
            ys,
            uvs,
            w,
            h,
            src.sub_x,
            src.sub_y,
            false,
            matrix,
            src.bd,
        );
    }
    dst
}

/// Takes a picture of the given format from the pool, or allocates one.
/// Pooled pictures keep old samples; the decoder writes every sample it reads.
fn take_picture<P: Pixel>(
    pool: &mut Vec<Picture<P>>,
    w: usize,
    h: usize,
    num_planes: usize,
    sub_x: bool,
    sub_y: bool,
    bd: u32,
) -> Picture<P> {
    let fresh_geometry = |w: usize, h: usize| ((w + 127 + 160) & !63, (h + 127 + 160) & !63);
    let (aw, ah) = fresh_geometry(w, h);
    if let Some(i) = pool.iter().position(|p| {
        p.planes.len() == num_planes
            && p.sub_x == sub_x
            && p.sub_y == sub_y
            && p.bd == bd
            && p.planes[0].stride == aw
            && p.planes[0].alloc_h == ah
    }) {
        let mut p = pool.swap_remove(i);
        for (k, pl) in p.planes.iter_mut().enumerate() {
            let (sx, sy) = if k > 0 {
                (sub_x, sub_y)
            } else {
                (false, false)
            };
            pl.w = (w + usize::from(sx)) >> u32::from(sx);
            pl.h = (h + usize::from(sy)) >> u32::from(sy);
        }
        return p;
    }
    Picture::new(w, h, num_planes, sub_x, sub_y, bd)
}

/// A pooled copy of `src`.
fn copy_picture<P: Pixel>(pool: &mut Vec<Picture<P>>, src: &Picture<P>) -> Picture<P> {
    let mut d = take_picture(
        pool,
        src.planes[0].w,
        src.planes[0].h,
        src.planes.len(),
        src.sub_x,
        src.sub_y,
        src.bd,
    );
    for (dp, sp) in d.planes.iter_mut().zip(&src.planes) {
        if dp.data.len() == sp.data.len() && dp.stride == sp.stride {
            dp.data.copy_from_slice(&sp.data);
            dp.w = sp.w;
            dp.h = sp.h;
        } else {
            *dp = sp.clone();
        }
    }
    d
}

const POOL_MAX: usize = 6;

fn recycle<P: Pixel>(pool: &mut Vec<Picture<P>>, p: Picture<P>) {
    if pool.len() < POOL_MAX {
        pool.push(p);
    }
}

/// Decodes the tiles of a frame into `cur`, calling `on_row(fs, cur, r0,
/// r1)` after each superblock row of each tile (luma 4x4 rows `r0..r1`).
/// Returns the CDFs saved from the context tile.
fn decode_tiles<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    tiles: &[(usize, Vec<u8>)],
    cdf: &mut CdfContext,
    refs: &[Option<&Picture<P>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    cur: &mut Picture<P>,
    on_row: &mut RowHook<'_, P>,
) -> R<Box<CdfContext>> {
    let num_planes = seq.num_planes();
    let mut saved: Option<Box<CdfContext>> = None;
    let sb128 = seq.use_128x128_superblock;
    let ti = &hdr.tile_info;
    let mut scratch = Scratch::<P>::new();
    let n_above = fs.mi_stride + 64;
    let mut above_level: [Vec<u8>; 3] = std::array::from_fn(|_| vec![0u8; n_above]);
    let mut above_dc: [Vec<u8>; 3] = std::array::from_fn(|_| vec![0u8; n_above]);
    let mut above_seg_pred = vec![0u8; n_above];
    let mut above_palette: [Vec<[u16; 8]>; 2] = std::array::from_fn(|_| vec![[0u16; 8]; n_above]);
    let mut color_map_y = vec![0u8; 128 * 128];
    let mut color_map_uv = vec![0u8; 128 * 128];
    for (tile_num, data) in tiles {
        let tile_row = tile_num / ti.cols;
        let tile_col = tile_num % ti.cols;
        let mut tile_cdf = Box::new(cdf.clone());
        let ref_views: [Option<RefView<'_, P>>; 8] = std::array::from_fn(|i| {
            if i == 0 {
                return None;
            }
            let slot = hdr.ref_frame_idx[i - 1] as usize;
            refs[i].map(|pic| RefView {
                pic,
                upscaled_width: ref_hdr[slot].upscaled_width as usize,
                frame_height: ref_hdr[slot].frame_height as usize,
                frame_width: ref_hdr[slot].frame_width as usize,
            })
        });
        let mut td = TileDec {
            seq,
            hdr,
            fs: &mut *fs,
            cur: &mut *cur,
            refs: ref_views,
            msac: Msac::new(data, hdr.disable_cdf_update),
            cdf: &mut tile_cdf,
            mi_row_start: ti.mi_row_starts[tile_row] as usize,
            mi_row_end: ti.mi_row_starts[tile_row + 1] as usize,
            mi_col_start: ti.mi_col_starts[tile_col] as usize,
            mi_col_end: ti.mi_col_starts[tile_col + 1] as usize,
            sb_size: if sb128 { 15 } else { 12 },
            sb_size4: if sb128 { 32 } else { 16 },
            bd: seq.bit_depth,
            sub_x: seq.subsampling_x,
            sub_y: seq.subsampling_y,
            num_planes,
            above_level: std::mem::take(&mut above_level),
            above_dc: std::mem::take(&mut above_dc),
            above_seg_pred: std::mem::take(&mut above_seg_pred),
            left_level: [[0; 32]; 3],
            left_dc: [[0; 32]; 3],
            left_seg_pred: [0; 32],
            above_palette: std::mem::take(&mut above_palette),
            left_palette: [[[0; 8]; 32]; 2],
            delta_lf: [0; 4],
            current_q_index: i32::from(hdr.quant.base_q_idx),
            read_deltas: false,
            ref_lr_wiener: [[[0; 3]; 2]; 3],
            ref_sgr_xqd: [[0; 2]; 3],
            block_decoded: [[[false; 35]; 35]; 3],
            b: Default::default(),
            pb: Default::default(),
            plan: Default::default(),
            src: None,
            enc: Default::default(),
            mode: Default::default(),
            me_scratch: Default::default(),
            plan_q: 0,
            cfl_choice: (0, 0),
            intra_choice: (0, None),
            est_hint: (0, 0),
            cur_filters: [0, 0],
            plan_delta_lf: [0; 4],
            color_map_y: std::mem::take(&mut color_map_y),
            color_map_uv: std::mem::take(&mut color_map_uv),
            scratch: std::mem::replace(&mut scratch, Scratch::empty()),
        };
        let res = td.decode_tile_rows(&mut *on_row);
        above_level = std::mem::take(&mut td.above_level);
        above_dc = std::mem::take(&mut td.above_dc);
        above_seg_pred = std::mem::take(&mut td.above_seg_pred);
        above_palette = std::mem::take(&mut td.above_palette);
        color_map_y = std::mem::take(&mut td.color_map_y);
        color_map_uv = std::mem::take(&mut td.color_map_uv);
        scratch = std::mem::replace(&mut td.scratch, Scratch::empty());
        drop(td);
        res?;
        if *tile_num == ti.context_update_tile_id as usize {
            saved = Some(tile_cdf);
        }
    }
    Ok(saved.unwrap_or_else(|| Box::new(cdf.clone())))
}

/// One superblock row of the frame being decoded, copied for
/// [`filter_rows_worker`]: luma 4x4 rows `r0..r1`, their samples, mode info,
/// loop filter transform sizes and CDEF indices.
#[derive(Default)]
struct RowMsg<P> {
    r0: usize,
    r1: usize,
    last: bool,
    /// Per plane: first sample row and the rows (whole strides).
    px: [(usize, Vec<P>); 3],
    /// First 4x4 row copied and the mode info / transform size rows (the
    /// last row reaches the end of the arrays).
    mi_row0: usize,
    mi: Vec<MiInfo>,
    lf_tx: [Vec<u8>; 3],
    cdef_row0: usize,
    cdef: Vec<i8>,
    /// Loop restoration units of every plane (when the frame uses them).
    lr: [Vec<LrUnitParams>; 3],
}

impl<P: Pixel> RowMsg<P> {
    fn fill(
        &mut self,
        fs: &FrameState,
        cur: &Picture<P>,
        sb4: usize,
        r0: usize,
        r1: usize,
        last: bool,
        sub_y: bool,
    ) {
        self.r0 = r0;
        self.r1 = r1;
        self.last = last;
        for (p, pl) in cur.planes.iter().enumerate().take(3) {
            let sy = usize::from(p > 0 && sub_y);
            let y0 = (4 * r0) >> sy;
            let y1 = if last {
                pl.alloc_h
            } else {
                ((4 * r1) >> sy).min(pl.alloc_h)
            };
            let (row0, v) = &mut self.px[p];
            *row0 = y0;
            v.clear();
            v.extend_from_slice(&pl.data[y0 * pl.stride..y1 * pl.stride]);
        }
        let st = fs.mi_stride;
        let end = if last { fs.mi.len() } else { r1 * st };
        self.mi_row0 = r0;
        self.mi.clear();
        self.mi.extend_from_slice(&fs.mi[r0 * st..end]);
        // The transform size maps are indexed in each plane's 4x4 units.
        for (p, (d, s)) in self.lf_tx.iter_mut().zip(&fs.lf_tx).enumerate() {
            let sy = usize::from(p > 0 && sub_y);
            let e = if last { s.len() } else { (r1 >> sy) * st };
            d.clear();
            d.extend_from_slice(&s[(r0 >> sy) * st..e]);
        }
        let cs = fs.cdef_stride;
        let c0 = r0 >> 4;
        let c1 = if last {
            fs.cdef_idx.len()
        } else {
            ((r0 + sb4) >> 4) * cs
        };
        self.cdef_row0 = c0;
        self.cdef.clear();
        self.cdef
            .extend_from_slice(&fs.cdef_idx[(c0 * cs).min(c1)..c1.min(fs.cdef_idx.len())]);
        for (d, s) in self.lr.iter_mut().zip(&fs.lr) {
            d.clear();
            d.extend_from_slice(s);
        }
    }

    /// Writes the copied rows into `fs` and `pic`.
    fn apply(&self, fs: &mut FrameState, pic: &mut Picture<P>, sub_y: bool) {
        for (pl, (row0, v)) in pic.planes.iter_mut().zip(&self.px) {
            let o = row0 * pl.stride;
            pl.data[o..o + v.len()].copy_from_slice(v);
        }
        let o = self.mi_row0 * fs.mi_stride;
        fs.mi[o..o + self.mi.len()].copy_from_slice(&self.mi);
        for (p, (d, s)) in fs.lf_tx.iter_mut().zip(&self.lf_tx).enumerate() {
            let o = (self.mi_row0 >> usize::from(p > 0 && sub_y)) * fs.mi_stride;
            d[o..o + s.len()].copy_from_slice(s);
        }
        let o = self.cdef_row0 * fs.cdef_stride;
        fs.cdef_idx[o..o + self.cdef.len()].copy_from_slice(&self.cdef);
        for (d, s) in fs.lr.iter_mut().zip(&self.lr) {
            d.clear();
            d.extend_from_slice(s);
        }
    }
}

/// Copies the sample rows of luma 4x4 rows `r0..r1` (to the bottom of the
/// allocation when `last`) from `src` to `dst`.
fn copy_rows<P: Pixel>(
    src: &Picture<P>,
    dst: &mut Picture<P>,
    r0: usize,
    r1: usize,
    last: bool,
    sub_y: bool,
) {
    for (p, (d, s)) in dst.planes.iter_mut().zip(&src.planes).enumerate() {
        let sy = usize::from(p > 0 && sub_y);
        let y0 = (4 * r0) >> sy;
        let y1 = if last {
            s.alloc_h
        } else {
            ((4 * r1) >> sy).min(s.alloc_h)
        };
        let r = y0 * s.stride..y1 * s.stride;
        d.data[r.clone()].copy_from_slice(&s.data[r]);
    }
}

/// Where [`filter_rows_worker`] writes: the deblocked picture, and the CDEF
/// and loop restoration outputs of the stages that run.
struct FilterOut<'a, P> {
    deb: &'a mut Picture<P>,
    cdef: Option<&'a mut Picture<P>>,
    lr: Option<&'a mut Picture<P>>,
    lr_scratch: &'a mut LrScratch,
}

/// Runs CDEF and loop restoration on luma 4x4 rows `a..b` (to the bottom
/// of the frame when `last`), whose deblocking and that of the rows below
/// them that these stages read is done.
fn filter_ready_rows<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    out: &mut FilterOut<'_, P>,
    pads: &mut [Vec<u16>; 3],
    a: usize,
    b: usize,
    last: bool,
) {
    let sub_y = seq.subsampling_y;
    if let Some(dst) = out.cdef.as_deref_mut() {
        copy_rows(out.deb, dst, a, b, last, sub_y);
        postfilter::cdef_sb_rows(seq, hdr, fs, out.deb, dst, a, b, pads);
    }
    if let Some(lr) = out.lr.as_deref_mut() {
        let src = out.cdef.as_deref().unwrap_or(out.deb);
        copy_rows(src, lr, a, b, last, sub_y);
        let units = postfilter::lr_units(seq, fs);
        let stripes = a / 16..if last { usize::MAX } else { b / 16 };
        postfilter::loop_restoration_stripes(
            seq,
            hdr,
            fs,
            src,
            out.deb,
            lr,
            &units,
            out.lr_scratch,
            stripes,
        );
    }
}

/// Deblocks (when `lf_on`) the superblock rows arriving on `rx` into
/// `out.deb`, and runs CDEF and loop restoration one superblock row behind,
/// as each row's CDEF and loop restoration read deblocked samples of the
/// next. Spent messages go back on `back` for reuse.
fn filter_rows_worker<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    mut out: FilterOut<'_, P>,
    lf_on: bool,
    rx: std::sync::mpsc::Receiver<RowMsg<P>>,
    back: std::sync::mpsc::Sender<RowMsg<P>>,
) {
    let mut pads = postfilter::cdef_pads();
    let mut pending: Option<(usize, usize)> = None;
    for m in rx {
        m.apply(fs, out.deb, seq.subsampling_y);
        if lf_on {
            postfilter::loop_filter_rows(seq, hdr, fs, out.deb, m.r0, m.r1, m.last);
        }
        if let Some((a, b)) = pending.take() {
            filter_ready_rows(seq, hdr, fs, &mut out, &mut pads, a, b, false);
        }
        if m.last {
            filter_ready_rows(seq, hdr, fs, &mut out, &mut pads, m.r0, m.r1, true);
        } else {
            pending = Some((m.r0, m.r1));
        }
        let _ = back.send(m);
    }
}

/// Decodes all tiles of a frame and runs the in-loop filters. Returns the
/// final picture and the CDFs saved from the context tile.
fn decode_frame<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    tiles: &[(usize, Vec<u8>)],
    cdf: &mut CdfContext,
    refs: [Option<&Picture<P>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    lr_scratch: &mut LrScratch,
    filter_fs: &mut Option<FrameState>,
    pool: &mut Vec<Picture<P>>,
    threads: usize,
) -> R<(Picture<P>, Box<CdfContext>)> {
    let num_planes = seq.num_planes();
    let new_pic = |pool: &mut Vec<Picture<P>>| {
        take_picture::<P>(
            pool,
            hdr.frame_width as usize,
            hdr.frame_height as usize,
            num_planes,
            seq.subsampling_x,
            seq.subsampling_y,
            seq.bit_depth,
        )
    };
    let mut cur = new_pic(pool);
    let lf_on = hdr.loop_filter.level[0] != 0 || hdr.loop_filter.level[1] != 0;
    let cdef_on = seq.enable_cdef
        && !hdr.coded_lossless
        && !hdr.allow_intrabc
        && (hdr.cdef.bits > 0
            || hdr.cdef.y_pri[0] != 0
            || hdr.cdef.y_sec[0] != 0
            || hdr.cdef.uv_pri[0] != 0
            || hdr.cdef.uv_sec[0] != 0);
    // With one tile column, superblock rows complete in order: deblocking
    // and CDEF then run on a second thread one superblock row behind.
    let pipeline = threads > 1
        && hdr.tile_info.cols == 1
        && !hdr.use_superres
        && (lf_on || cdef_on || hdr.lr.uses_lr);
    let mut lr_pic = None;
    let (saved, cur, cdef_pic) = if pipeline {
        let mut deb = new_pic(pool);
        let mut cdef_dst = if cdef_on { Some(new_pic(pool)) } else { None };
        let mut lr_dst = if hdr.lr.uses_lr {
            Some(new_pic(pool))
        } else {
            None
        };
        let mut pf_fs = match filter_fs.take() {
            Some(f) if f.fits_filter_shell(fs) => f.reshape_filter_shell(fs),
            _ => fs.filter_shell(),
        };
        let saved = std::thread::scope(|s| {
            let (tx, rx) = std::sync::mpsc::channel::<RowMsg<P>>();
            let (back_tx, back_rx) = std::sync::mpsc::channel::<RowMsg<P>>();
            let out = FilterOut {
                deb: &mut deb,
                cdef: cdef_dst.as_mut(),
                lr: lr_dst.as_mut(),
                lr_scratch: &mut *lr_scratch,
            };
            let pf_fs = &mut pf_fs;
            let worker = s.spawn(move || {
                filter_rows_worker(seq, hdr, pf_fs, out, lf_on, rx, back_tx);
            });
            let sb4 = if seq.use_128x128_superblock { 32 } else { 16 };
            let res = decode_tiles(
                seq,
                hdr,
                fs,
                tiles,
                cdf,
                &refs,
                ref_hdr,
                &mut cur,
                &mut |fs: &FrameState, cur: &Picture<P>, r0: usize, r1: usize| {
                    let mut m = back_rx.try_recv().unwrap_or_default();
                    m.fill(fs, cur, sb4, r0, r1, r1 >= fs.mi_rows, seq.subsampling_y);
                    // A send only fails once the worker is gone (it panicked):
                    // the join below reports that.
                    let _ = tx.send(m);
                },
            );
            drop(tx);
            let joined = worker.join();
            match (res, joined) {
                (Err(e), _) => Err(e),
                (Ok(_), Err(_)) => Err(malformed("in-loop filter thread")),
                (Ok(c), Ok(())) => Ok(c),
            }
        })?;
        recycle(pool, cur);
        *filter_fs = Some(pf_fs);
        lr_pic = lr_dst;
        (saved, deb, cdef_dst)
    } else {
        let saved = decode_tiles(
            seq,
            hdr,
            fs,
            tiles,
            cdf,
            &refs,
            ref_hdr,
            &mut cur,
            &mut |_: &FrameState, _: &Picture<P>, _: usize, _: usize| {},
        )?;
        // decode_frame_wrapup: in-loop filters.
        postfilter::loop_filter(seq, hdr, fs, &mut cur, threads);
        let cdef_pic = if cdef_on {
            let mut d = copy_picture(pool, &cur);
            postfilter::cdef(seq, hdr, fs, &cur, &mut d, threads);
            Some(d)
        } else {
            None
        };
        (saved, cur, cdef_pic)
    };
    let (up_cdef, up_deblocked) = if hdr.use_superres {
        let c = postfilter::superres(seq, hdr, cdef_pic.as_ref().unwrap_or(&cur));
        let d = if hdr.lr.uses_lr {
            Some(postfilter::superres(seq, hdr, &cur))
        } else {
            None
        };
        recycle(pool, cur);
        if let Some(p) = cdef_pic {
            recycle(pool, p);
        }
        (c, d)
    } else {
        match cdef_pic {
            Some(c) => (c, Some(cur)),
            None => (cur, None),
        }
    };
    let mut final_pic = if let Some(out) = lr_pic {
        recycle(pool, up_cdef);
        if let Some(p) = up_deblocked {
            recycle(pool, p);
        }
        out
    } else if hdr.lr.uses_lr {
        let mut out = copy_picture(pool, &up_cdef);
        {
            let deblocked = up_deblocked.as_ref().unwrap_or(&up_cdef);
            postfilter::loop_restoration(
                seq, hdr, fs, &up_cdef, deblocked, &mut out, lr_scratch, threads,
            );
        }
        recycle(pool, up_cdef);
        if let Some(p) = up_deblocked {
            recycle(pool, p);
        }
        out
    } else {
        if let Some(p) = up_deblocked {
            recycle(pool, p);
        }
        up_cdef
    };
    // Visible size.
    for (p, pl) in final_pic.planes.iter_mut().enumerate() {
        let (sx, sy) = if p > 0 {
            (
                usize::from(seq.subsampling_x),
                usize::from(seq.subsampling_y),
            )
        } else {
            (0, 0)
        };
        pl.w = (hdr.upscaled_width as usize + sx) >> sx;
        pl.h = (hdr.frame_height as usize + sy) >> sy;
    }
    Ok((final_pic, saved))
}
