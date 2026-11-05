//! The picture under reconstruction and the decoded pictures it references.

use std::sync::Arc;

use cutlass_decode::CodecError;

use crate::cabac::Contexts;
use crate::ps::{Layout, Pps, Sps};
use crate::slice::SliceHeader;

/// A motion vector in quarter luma samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    pub x: i16,
    pub y: i16,
}

/// Motion of one 4×4 luma block. `pred` bit 0 is list 0, bit 1 list 1;
/// zero is intra or not yet decoded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MvField {
    pub mv: [Mv; 2],
    pub ref_idx: [i8; 2],
    pub pred: u8,
}

impl MvField {
    pub fn uses(&self, list: usize) -> bool {
        self.pred & (1 << list) != 0
    }
}

/// Motion of a 16×16 block of a decoded picture, kept for temporal motion
/// vector prediction (clause 8.5.3.2.8). The reference is named by its
/// order count and whether it was a long-term reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColMv {
    pub mv: [Mv; 2],
    pub poc: [i32; 2],
    pub long_term: [bool; 2],
    pub pred: u8,
}

/// A decoded picture, immutable once it enters the DPB.
#[derive(Debug)]
pub struct Decoded {
    pub width: usize,
    pub height: usize,
    pub planes: [Vec<u16>; 3],
    pub col: Vec<ColMv>,
    pub col_stride: usize,
    pub poc: i32,
}

impl Decoded {
    /// A picture that the bitstream named and did not send (clause 8.3.3):
    /// mid-grey, with no motion.
    pub fn missing(width: usize, height: usize, bit_depth: u8, poc: i32) -> Self {
        let grey = 1u16 << (bit_depth - 1);
        Self {
            width,
            height,
            planes: [
                vec![grey; width * height],
                vec![grey; (width / 2) * (height / 2)],
                vec![grey; (width / 2) * (height / 2)],
            ],
            col: Vec::new(),
            col_stride: 0,
            poc,
        }
    }

    pub fn col_at(&self, x: usize, y: usize) -> ColMv {
        if self.col_stride == 0 {
            return ColMv::default();
        }
        self.col
            .get((y >> 4) * self.col_stride + (x >> 4))
            .copied()
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
pub struct RefPic {
    pub pic: Arc<Decoded>,
    pub poc: i32,
    pub long_term: bool,
    /// Identity of the DPB picture, for deblocking's same-picture test.
    pub uid: u64,
}

/// One independent or dependent slice segment of the picture.
#[derive(Clone, Debug)]
pub struct SliceCtx {
    pub header: SliceHeader,
    pub refs: [Vec<RefPic>; 2],
    /// `SliceAddrRs`: the first CTB of the independent slice.
    pub slice_addr: u32,
    pub qp: i32,
    /// Every reference precedes the current picture (`NoBackwardPredFlag`).
    pub no_backward_pred: bool,
}

/// SAO parameters of one CTB and component (clause 7.4.9.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SaoParams {
    /// 0 off, 1 band, 2 edge.
    pub kind: u8,
    /// Band position or edge class.
    pub class: u8,
    pub offsets: [i16; 4],
}

pub const MODE_INTRA: u8 = 1;
pub const MODE_INTER: u8 = 2;
pub const FLAG_SKIP: u8 = 4;
/// PCM with the loop filter off, or transquant bypass: no deblocking or SAO.
pub const FLAG_KEEP: u8 = 8;
pub const FLAG_PCM: u8 = 16;
pub const FLAG_CBF: u8 = 32;

/// Reconstruction state of the picture being decoded.
pub struct Picture {
    pub sps: Arc<Sps>,
    pub pps: Arc<Pps>,
    pub layout: Arc<Layout>,
    pub width: usize,
    pub height: usize,
    pub planes: [Vec<u16>; 3],
    /// Width of the 4×4 grid.
    pub w4: usize,
    pub mvf: Vec<MvField>,
    /// `MODE_*` and `FLAG_*` of each 4×4.
    pub info: Vec<u8>,
    pub intra_mode: Vec<u8>,
    pub qp: Vec<i8>,
    pub depth: Vec<u8>,
    /// Boundary strength of vertical edges: x on the 8 grid, 4 rows each.
    pub bs_ver: Vec<u8>,
    /// Boundary strength of horizontal edges: y on the 8 grid, 4 columns each.
    pub bs_hor: Vec<u8>,
    /// Slice segment index of each CTB, raster order.
    pub ctb_slice: Vec<u16>,
    /// `SliceAddrRs` of each decoded CTB.
    pub ctb_slice_addr: Vec<u32>,
    pub sao: Vec<[SaoParams; 3]>,
    pub slices: Vec<SliceCtx>,
    pub poc: i32,
    pub ctbs_done: u32,
    /// Contexts and QP at the end of the last slice segment, for a
    /// dependent slice segment.
    pub dependent_ctx: Contexts,
    pub dependent_qp: i32,
    /// Contexts stored after the second CTB of each row, by CTB.
    pub wpp_ctx: Vec<Option<Box<Contexts>>>,
}

impl Picture {
    pub fn new(
        sps: Arc<Sps>,
        pps: Arc<Pps>,
        layout: Arc<Layout>,
        poc: i32,
    ) -> Result<Self, CodecError> {
        let width = sps.width as usize;
        let height = sps.height as usize;
        let luma = width.checked_mul(height).ok_or(CodecError::Limit {
            what: "picture size",
        })?;
        let w4 = width / 4;
        let h4 = height / 4;
        let ctbs = layout.ctb_count() as usize;
        Ok(Self {
            width,
            height,
            planes: [vec![0; luma], vec![0; luma / 4], vec![0; luma / 4]],
            w4,
            mvf: vec![MvField::default(); w4 * h4],
            info: vec![0; w4 * h4],
            intra_mode: vec![1; w4 * h4],
            qp: vec![0; w4 * h4],
            depth: vec![0; w4 * h4],
            bs_ver: vec![0; (width / 8 + 1) * h4],
            bs_hor: vec![0; (height / 8 + 1) * w4],
            ctb_slice: vec![u16::MAX; ctbs],
            ctb_slice_addr: vec![u32::MAX; ctbs],
            sao: vec![[SaoParams::default(); 3]; ctbs],
            slices: Vec::new(),
            poc,
            ctbs_done: 0,
            dependent_ctx: [0; crate::cabac::CONTEXTS],
            dependent_qp: 0,
            wpp_ctx: if pps.entropy_sync {
                vec![None; ctbs]
            } else {
                Vec::new()
            },
            sps,
            pps,
            layout,
        })
    }

    #[inline]
    pub fn at4(&self, x: usize, y: usize) -> usize {
        (y >> 2) * self.w4 + (x >> 2)
    }

    #[inline]
    pub fn ctb_rs(&self, x: usize, y: usize) -> usize {
        let log2 = self.layout.log2_ctb;
        (y >> log2) * self.layout.ctb_width as usize + (x >> log2)
    }

    /// `MinTbAddrZs` of a luma position (clause 6.5.2), computed from the
    /// tile scan and the z-order inside the CTB.
    #[inline]
    pub fn zs(&self, x: usize, y: usize) -> u64 {
        let log2_ctb = u32::from(self.layout.log2_ctb);
        let log2_tb = u32::from(self.sps.log2_min_tb);
        let rs = self.ctb_rs(x, y);
        let ts = u64::from(self.layout.rs_to_ts[rs]);
        let mask = (1usize << log2_ctb) - 1;
        let tx = ((x & mask) >> log2_tb) as u64;
        let ty = ((y & mask) >> log2_tb) as u64;
        let mut morton = 0u64;
        for bit in 0..(log2_ctb - log2_tb) {
            morton |= ((tx >> bit) & 1) << (2 * bit);
            morton |= ((ty >> bit) & 1) << (2 * bit + 1);
        }
        (ts << (2 * (log2_ctb - log2_tb))) | morton
    }

    /// Z-scan availability (clause 6.4.1).
    #[inline]
    pub fn available(&self, x: usize, y: usize, nx: isize, ny: isize) -> bool {
        if nx < 0 || ny < 0 || nx as usize >= self.width || ny as usize >= self.height {
            return false;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        let nb_ctb = self.ctb_rs(nx, ny);
        let cur_ctb = self.ctb_rs(x, y);
        if self.ctb_slice_addr[nb_ctb] != self.ctb_slice_addr[cur_ctb]
            || self.layout.tile_of[nb_ctb] != self.layout.tile_of[cur_ctb]
        {
            return false;
        }
        if nb_ctb != cur_ctb {
            return self.layout.rs_to_ts[nb_ctb] < self.layout.rs_to_ts[cur_ctb];
        }
        self.zs(nx, ny) <= self.zs(x, y)
    }

    pub fn slice_of(&self, x: usize, y: usize) -> Option<&SliceCtx> {
        let index = *self.ctb_slice.get(self.ctb_rs(x, y))?;
        self.slices.get(usize::from(index))
    }
}
