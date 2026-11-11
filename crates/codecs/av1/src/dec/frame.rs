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
    /// The luma stride and allocated height of a picture of visible size
    /// `w x h`.
    pub fn geometry(w: usize, h: usize) -> (usize, usize) {
        let mut aw = (w + 127 + 160) & !63;
        // A row a multiple of 1024 samples long puts the rows of a block
        // in the same few cache sets (4K luma would be 4096): 64 more
        // spread them.
        if aw.is_multiple_of(1024) {
            aw += 64;
        }
        (aw, (h + 127 + 160) & !63)
    }

    /// Allocates a picture whose planes hold the visible size `w x h` and
    /// enough room for whole superblocks.
    pub fn new(w: usize, h: usize, num_planes: usize, sub_x: bool, sub_y: bool, bd: u32) -> Self {
        let (aw, ah) = Self::geometry(w, h);
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

/// Rows of 8x8 positions in a band of a frame's saved motion: the rows a
/// 64x64 superblock row covers, and the rows motion field estimation
/// projects together (7.9).
pub const MF_BAND_ROWS: usize = 8;

/// MfRefFrames and MfMvs of one band: whole rows of 8x8 positions (the
/// odd 4x4 positions).
#[derive(Clone, Default)]
pub struct MfBand {
    pub refs: Vec<i8>,
    pub mvs: Vec<Mv>,
}

/// What a reference slot keeps besides its picture (spec 7.20).
///
/// The frame's size and order hints are known from its header. Its saved
/// motion arrives band by band while the frame is parsed, and its segment
/// ids when the parse ends: a reader waits for the band or the ids it
/// asks for. Whoever makes one must [`close`](Self::close) it, or its
/// readers wait for ever.
#[derive(Clone)]
pub struct RefData {
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub frame_type: u8,
    pub saved_order_hints: [u32; 8],
    mf: Vec<std::sync::OnceLock<MfBand>>,
    segment_ids: std::sync::OnceLock<Vec<u8>>,
}

impl RefData {
    /// The data of a frame not parsed yet.
    pub fn new(
        mi_cols: usize,
        mi_rows: usize,
        frame_type: u8,
        saved_order_hints: [u32; 8],
    ) -> Self {
        let bands = (mi_rows >> 1).div_ceil(MF_BAND_ROWS);
        Self {
            mi_cols,
            mi_rows,
            frame_type,
            saved_order_hints,
            mf: (0..bands).map(|_| std::sync::OnceLock::new()).collect(),
            segment_ids: std::sync::OnceLock::new(),
        }
    }

    /// Number of motion bands.
    pub fn bands(&self) -> usize {
        self.mf.len()
    }

    /// Rows of 8x8 positions in band `k`.
    pub fn band_rows(&self, k: usize) -> usize {
        (self.mi_rows >> 1)
            .saturating_sub(k * MF_BAND_ROWS)
            .min(MF_BAND_ROWS)
    }

    /// Whether band `k` has its motion (or there is no such band).
    pub fn has_band(&self, k: usize) -> bool {
        self.mf.get(k).is_none_or(|b| b.get().is_some())
    }

    /// Gives band `k` its motion. Only the first call for a band counts.
    pub fn set_band(&self, k: usize, band: MfBand) {
        if let Some(b) = self.mf.get(k) {
            let _ = b.set(band);
        }
    }

    /// The motion of band `k`, once the frame's parse has got there.
    /// `None` past the last band.
    pub fn band(&self, k: usize) -> Option<&MfBand> {
        self.mf.get(k).map(std::sync::OnceLock::wait)
    }

    /// Gives the frame its segment ids (the first call counts).
    pub fn set_segment_ids(&self, ids: Vec<u8>) {
        let _ = self.segment_ids.set(ids);
    }

    /// The segment ids, indexed like [`FrameState::segment_ids`], once
    /// the frame's parse has ended. Empty when that parse failed.
    pub fn segment_ids(&self) -> &[u8] {
        self.segment_ids.wait()
    }

    /// Ends the data: bands still without motion get none, and so do
    /// the segment ids. A frame that failed leaves its readers this.
    pub fn close(&self) {
        let w8 = self.mi_cols >> 1;
        for (k, b) in self.mf.iter().enumerate() {
            let n = w8 * self.band_rows(k);
            let _ = b.set(MfBand {
                refs: vec![-1; n],
                mvs: vec![Mv::ZERO; n],
            });
        }
        let _ = self.segment_ids.set(Vec::new());
    }
}

/// The loop restoration units of each plane (all unset), with their
/// columns and rows.
#[allow(clippy::type_complexity)]
fn lr_units(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
) -> ([Vec<LrUnitParams>; 3], [usize; 3], [usize; 3]) {
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
    (lr, lr_cols, lr_rows)
}

/// An entry of [`FrameState::lf_tx`]: the width and height in 4x4 units of
/// the transform block that holds the unit, as base-2 logarithms (0 to 5)
/// in the low three bits and the three above them.
///
/// A skipped inter block gives its own size instead, of the plane: its
/// loop filter edges are its own sides only (7.14.2), and this changes no
/// filter size, as such a block's transforms are its own size unless it
/// is 128 samples wide or high, and every size from 16 samples gives the
/// same filter.
#[inline]
pub fn lf_dims(w_log2: u32, h_log2: u32) -> u8 {
    (w_log2 | h_log2 << 3) as u8
}

/// The width in 4x4 units of an [`FrameState::lf_tx`] entry.
#[inline(always)]
pub fn lf_w4(v: u8) -> usize {
    1 << (v & 7)
}

/// The height in 4x4 units of an [`FrameState::lf_tx`] entry.
#[inline(always)]
pub fn lf_h4(v: u8) -> usize {
    1 << ((v >> 3) & 7)
}

/// Mutable state of the frame being decoded.
pub struct FrameState {
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub mi_stride: usize,
    pub mi: Vec<MiInfo>,
    /// The loop filter deltas of each 4x4, laid out like `mi`: sized and
    /// written only for frames with delta_lf_present.
    pub delta_lf: Vec<[i8; 4]>,
    /// uv_mode of the last intra block with chroma at each 4x4.
    pub uv_modes: Vec<u8>,
    pub tx_types: Vec<u8>,
    pub inter_tx_sizes: Vec<u8>,
    /// LoopfilterTxSizes per plane, indexed in that plane's 4x4 units
    /// with `mi_stride`, as [`lf_dims`] writes them.
    pub lf_tx: [Vec<u8>; 3],
    pub segment_ids: Vec<u8>,
    /// The data of the primary reference frame, when it has the size of
    /// this frame (its segment ids predict this frame's).
    pub prev_segment_ids: Option<std::sync::Arc<RefData>>,
    /// cdef_idx per 64x64 block, -1 = unset.
    pub cdef_idx: Vec<i8>,
    pub cdef_stride: usize,
    pub lr: [Vec<LrUnitParams>; 3],
    pub lr_cols: [usize; 3],
    pub lr_rows: [usize; 3],
    /// MotionFieldMvs[ref][y8][x8] for ref = 1..=7 (index 0 unused).
    pub motion_field: Vec<super::mvpred::MfEntry>,
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
        let (lr, lr_cols, lr_rows) = lr_units(seq, hdr);
        Self {
            mi_cols,
            mi_rows,
            mi_stride,
            mi: vec![MiInfo::default(); n],
            delta_lf: Vec::new(),
            uv_modes: vec![0; n],
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

    /// [`Self::new`], reusing the arrays of `old` (the state of an earlier
    /// frame) when the frame has the same size in 4x4 units. Every 4x4 of
    /// the frame is written by its block before it is read, and the 4x4s
    /// past the frame edges are never written, so only the CDEF indices,
    /// loop restoration units and per-frame extras start afresh.
    pub fn reuse(old: Option<Self>, seq: &SequenceHeader, hdr: &FrameHeader) -> Self {
        match old {
            Some(mut fs)
                if fs.mi_cols == hdr.mi_cols as usize && fs.mi_rows == hdr.mi_rows as usize =>
            {
                let (lr, lr_cols, lr_rows) = lr_units(seq, hdr);
                fs.cdef_idx.fill(-1);
                fs.lr = lr;
                fs.lr_cols = lr_cols;
                fs.lr_rows = lr_rows;
                fs.prev_segment_ids = None;
                fs.motion_field.clear();
                fs
            }
            _ => Self::new(seq, hdr),
        }
    }
    /// A state of the same size holding only what reconstruction keeps of
    /// the blocks it replays (their mode info), to be filled row by row.
    pub fn recon_shell(&self) -> Self {
        Self {
            mi: vec![MiInfo::default(); self.mi.len()],
            uv_modes: vec![0; self.uv_modes.len()],
            ..self.empty_shell()
        }
    }

    /// A state of the same size holding only what the in-loop filters read
    /// besides their block info (loop filter transform sizes, CDEF
    /// indices, loop restoration units), to be filled row by row.
    pub fn filter_shell(&self) -> Self {
        Self {
            lf_tx: std::array::from_fn(|p| vec![0; self.lf_tx[p].len()]),
            cdef_idx: vec![-1; self.cdef_idx.len()],
            ..self.empty_shell()
        }
    }

    /// A state with the dimensions of this one and no arrays.
    fn empty_shell(&self) -> Self {
        Self {
            mi_cols: self.mi_cols,
            mi_rows: self.mi_rows,
            mi_stride: self.mi_stride,
            mi: Vec::new(),
            delta_lf: Vec::new(),
            uv_modes: Vec::new(),
            tx_types: Vec::new(),
            inter_tx_sizes: Vec::new(),
            lf_tx: Default::default(),
            segment_ids: Vec::new(),
            prev_segment_ids: None,
            cdef_idx: Vec::new(),
            cdef_stride: self.cdef_stride,
            lr: Default::default(),
            lr_cols: self.lr_cols,
            lr_rows: self.lr_rows,
            motion_field: Vec::new(),
        }
    }

    /// Whether `self`, a [`Self::recon_shell`], has the array sizes of a
    /// reconstruction shell of `fs` (its contents are rewritten row by
    /// row).
    pub fn fits_recon_shell(&self, fs: &FrameState) -> bool {
        self.mi_stride == fs.mi_stride
            && self.mi.len() == fs.mi.len()
            && self.uv_modes.len() == fs.uv_modes.len()
    }

    /// Whether `self`, a [`Self::filter_shell`], has the array sizes of a
    /// filter shell of `fs` (its contents are rewritten row by row).
    pub fn fits_filter_shell(&self, fs: &FrameState) -> bool {
        self.mi_stride == fs.mi_stride
            && self.cdef_idx.len() == fs.cdef_idx.len()
            && self
                .lf_tx
                .iter()
                .zip(&fs.lf_tx)
                .all(|(a, b)| a.len() == b.len())
    }

    /// Takes on the dimensions of `fs`, whose shell it fits.
    pub fn reshape_shell(mut self, fs: &FrameState) -> Self {
        self.mi_cols = fs.mi_cols;
        self.mi_rows = fs.mi_rows;
        self.cdef_stride = fs.cdef_stride;
        self.lr_cols = fs.lr_cols;
        self.lr_rows = fs.lr_rows;
        self
    }

    #[inline(always)]
    pub fn mi_at(&self, row: usize, col: usize) -> &MiInfo {
        &self.mi[row * self.mi_stride + col]
    }
}

#[cfg(test)]
mod ref_data_tests {
    use super::{MF_BAND_ROWS, MfBand, RefData};
    use crate::dec::types::Mv;

    /// 40 rows of 4x4 units: 20 rows of 8x8 in bands of 8, 8 and 4.
    fn data() -> RefData {
        RefData::new(16, 40, 1, [0; 8])
    }

    #[test]
    fn bands_cover_the_rows_of_8x8_positions() {
        let rd = data();
        assert_eq!(rd.bands(), 3);
        assert_eq!(
            [rd.band_rows(0), rd.band_rows(1), rd.band_rows(2)],
            [MF_BAND_ROWS, MF_BAND_ROWS, 4]
        );
        assert_eq!(rd.band_rows(3), 0);
        assert!(rd.band(3).is_none(), "no band past the last");
        assert!(rd.has_band(3), "nothing to wait for there");
    }

    #[test]
    fn a_reader_waits_for_the_band_it_asks_for() {
        let rd = std::sync::Arc::new(data());
        rd.set_band(
            0,
            MfBand {
                refs: vec![1; 64],
                mvs: vec![Mv::ZERO; 64],
            },
        );
        // Band 0 is there while band 1 is not.
        assert!(rd.has_band(0) && !rd.has_band(1));
        assert_eq!(rd.band(0).map(|b| b.refs[0]), Some(1));
        let reader = {
            let rd = rd.clone();
            std::thread::spawn(move || rd.band(1).map(|b| b.refs[0]))
        };
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!reader.is_finished(), "band 1 has no motion yet");
        rd.set_band(
            1,
            MfBand {
                refs: vec![3; 64],
                mvs: vec![Mv::ZERO; 64],
            },
        );
        assert_eq!(reader.join().ok().flatten(), Some(3));
    }

    #[test]
    fn only_the_first_motion_of_a_band_counts() {
        let rd = data();
        for r in [2, 5] {
            rd.set_band(
                2,
                MfBand {
                    refs: vec![r; 32],
                    mvs: vec![Mv::ZERO; 32],
                },
            );
        }
        assert_eq!(rd.band(2).map(|b| b.refs[0]), Some(2));
    }

    #[test]
    fn closing_leaves_readers_no_motion_and_no_ids() {
        let rd = std::sync::Arc::new(data());
        rd.set_band(
            0,
            MfBand {
                refs: vec![4; 64],
                mvs: vec![Mv::ZERO; 64],
            },
        );
        let reader = {
            let rd = rd.clone();
            std::thread::spawn(move || (rd.band(2).map(|b| b.refs.clone()), rd.segment_ids().len()))
        };
        rd.close();
        let (last, ids) = reader.join().unwrap_or_default();
        // 4 rows of 8 positions, all without a reference.
        assert_eq!(last, Some(vec![-1; 32]));
        assert_eq!(ids, 0);
        assert_eq!(rd.band(0).map(|b| b.refs[0]), Some(4), "saved motion stays");
        // Ids that come after the close are too late.
        rd.set_segment_ids(vec![1; 8]);
        assert!(rd.segment_ids().is_empty());
    }
}
