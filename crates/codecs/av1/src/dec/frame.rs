//! Frame buffers and per-frame decode state.

use crate::dsp::Pixel;

use super::obu::{FrameHeader, SequenceHeader};
use super::types::{MiInfo, Mv};

#[derive(Clone)]
pub struct Plane<P> {
    pub data: Vec<P>,
    pub stride: usize,
    /// Visible size of this plane.
    pub w: usize,
    pub h: usize,
    /// Allocated rows.
    pub alloc_h: usize,
}

impl<P: Pixel> Plane<P> {
    pub fn new(w: usize, h: usize, alloc_w: usize, alloc_h: usize) -> Self {
        let stride = alloc_w.max(w);
        let alloc_h = alloc_h.max(h);
        Self {
            data: vec![P::default(); stride * alloc_h],
            stride,
            w,
            h,
            alloc_h,
        }
    }
}

/// One decoded picture (up to three planes).
#[derive(Clone)]
pub struct Picture<P> {
    pub planes: Vec<Plane<P>>,
    pub bd: u32,
    pub sub_x: bool,
    pub sub_y: bool,
}

impl<P: Pixel> Picture<P> {
    /// Allocates a picture whose planes hold the visible size `w x h` and
    /// enough room for whole superblocks.
    pub fn new(w: usize, h: usize, num_planes: usize, sub_x: bool, sub_y: bool, bd: u32) -> Self {
        let aw = (w + 127 + 160) & !63;
        let ah = (h + 127 + 160) & !63;
        let mut planes = Vec::with_capacity(num_planes);
        for p in 0..num_planes {
            let (sx, sy) = if p > 0 {
                (sub_x, sub_y)
            } else {
                (false, false)
            };
            let pw = (w + usize::from(sx)) >> u32::from(sx);
            let ph = (h + usize::from(sy)) >> u32::from(sy);
            planes.push(Plane::new(pw, ph, aw >> u32::from(sx), ah >> u32::from(sy)));
        }
        Self {
            planes,
            bd,
            sub_x,
            sub_y,
        }
    }
}

/// Loop restoration parameters of one unit.
#[derive(Clone, Copy, Debug, Default)]
pub struct LrUnitParams {
    pub kind: u8,
    pub wiener: [[i8; 3]; 2],
    pub sgr_set: u8,
    pub sgr_xqd: [i8; 2],
}

/// What a reference slot keeps besides its picture (spec 7.20).
#[derive(Clone)]
pub struct RefData {
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub frame_type: u8,
    /// MfRefFrames / MfMvs at 8x8 granularity (the odd 4x4 positions).
    pub mf_ref: Vec<i8>,
    pub mf_mv: Vec<Mv>,
    pub segment_ids: Vec<u8>,
    pub saved_order_hints: [u32; 8],
}

/// Mutable state of the frame being decoded.
pub struct FrameState {
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub mi_stride: usize,
    pub mi: Vec<MiInfo>,
    pub tx_types: Vec<u8>,
    pub inter_tx_sizes: Vec<u8>,
    /// LoopfilterTxSizes per plane, indexed in that plane's 4x4 units with `mi_stride`.
    pub lf_tx: [Vec<u8>; 3],
    pub segment_ids: Vec<u8>,
    pub prev_segment_ids: Option<Vec<u8>>,
    /// cdef_idx per 64x64 block, -1 = unset.
    pub cdef_idx: Vec<i8>,
    pub cdef_stride: usize,
    pub lr: [Vec<LrUnitParams>; 3],
    pub lr_cols: [usize; 3],
    pub lr_rows: [usize; 3],
    /// MotionFieldMvs[ref][y8][x8] for ref = 1..=7 (index 0 unused).
    pub motion_field: Vec<Vec<Mv>>,
}

impl FrameState {
    pub fn new(seq: &SequenceHeader, hdr: &FrameHeader) -> Self {
        let mi_cols = hdr.mi_cols as usize;
        let mi_rows = hdr.mi_rows as usize;
        // Room for a whole superblock past the frame edge.
        let mi_stride = (mi_cols + 32) & !31;
        let rows = (mi_rows + 32) & !31;
        let n = mi_stride * rows;
        let cdef_stride = mi_stride / 16 + 1;
        let mut lr: [Vec<LrUnitParams>; 3] = Default::default();
        let mut lr_cols = [0; 3];
        let mut lr_rows = [0; 3];
        for p in 0..seq.num_planes() {
            if hdr.lr.frame_type[p] != 0 {
                let (sx, sy) = if p > 0 {
                    (u32::from(seq.subsampling_x), u32::from(seq.subsampling_y))
                } else {
                    (0, 0)
                };
                let size = hdr.lr.unit_size[p] as usize;
                let count = |unit: usize, frame: usize| ((frame + (unit >> 1)) / unit).max(1);
                let rw = (hdr.upscaled_width as usize + ((1 << sx) >> 1)) >> sx;
                let rh = (hdr.frame_height as usize + ((1 << sy) >> 1)) >> sy;
                lr_cols[p] = count(size, rw);
                lr_rows[p] = count(size, rh);
                lr[p] = vec![LrUnitParams::default(); lr_cols[p] * lr_rows[p]];
            }
        }
        Self {
            mi_cols,
            mi_rows,
            mi_stride,
            mi: vec![MiInfo::default(); n],
            tx_types: vec![0; n],
            inter_tx_sizes: vec![0; n],
            lf_tx: [vec![0; n], vec![0; n], vec![0; n]],
            segment_ids: vec![0; n],
            prev_segment_ids: None,
            cdef_idx: vec![-1; cdef_stride * (rows / 16 + 1)],
            cdef_stride,
            lr,
            lr_cols,
            lr_rows,
            motion_field: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn mi_at(&self, row: usize, col: usize) -> &MiInfo {
        &self.mi[row * self.mi_stride + col]
    }
}
