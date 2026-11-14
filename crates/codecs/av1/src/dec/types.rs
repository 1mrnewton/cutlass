//! Shared decoder types and small spec helpers.

use crate::spec::tables::{
    MI_HEIGHT_LOG2, MI_WIDTH_LOG2, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, SUBSAMPLED_SIZE,
    TX_HEIGHT, TX_HEIGHT_LOG2, TX_WIDTH, TX_WIDTH_LOG2,
};

pub const INTRA_FRAME: i8 = 0;
pub const NONE: i8 = -1;
pub const LAST_FRAME: i8 = 1;
pub const LAST2_FRAME: i8 = 2;
pub const LAST3_FRAME: i8 = 3;
pub const GOLDEN_FRAME: i8 = 4;
pub const BWDREF_FRAME: i8 = 5;
pub const ALTREF2_FRAME: i8 = 6;
pub const ALTREF_FRAME: i8 = 7;
/// Marks a 4x4 unit that has not been decoded in this frame yet.
pub const NOT_WRITTEN: i8 = -2;

// Block sizes.
pub const BLOCK_4X4: u8 = 0;
pub const BLOCK_8X8: u8 = 3;
pub const BLOCK_16X16: u8 = 6;
pub const BLOCK_32X32: u8 = 9;
pub const BLOCK_64X64: u8 = 12;
pub const BLOCK_128X128: u8 = 15;
pub const BLOCK_INVALID: u8 = 22;

// Intra modes.
pub const DC_PRED: u8 = 0;
pub const V_PRED: u8 = 1;
pub const H_PRED: u8 = 2;
pub const D157_PRED: u8 = 6;
pub const D67_PRED: u8 = 8;
pub const SMOOTH_PRED: u8 = 9;
pub const PAETH_PRED: u8 = 12;
pub const UV_CFL_PRED: u8 = 13;
// Inter modes.
pub const NEARESTMV: u8 = 14;
pub const NEARMV: u8 = 15;
pub const GLOBALMV: u8 = 16;
pub const NEWMV: u8 = 17;
pub const NEAREST_NEARESTMV: u8 = 18;
pub const NEAR_NEARMV: u8 = 19;
pub const NEAREST_NEWMV: u8 = 20;
pub const NEW_NEARESTMV: u8 = 21;
pub const NEAR_NEWMV: u8 = 22;
pub const NEW_NEARMV: u8 = 23;
pub const GLOBAL_GLOBALMV: u8 = 24;
pub const NEW_NEWMV: u8 = 25;

// Motion modes.
pub const SIMPLE: u8 = 0;
pub const OBMC: u8 = 1;
pub const LOCALWARP: u8 = 2;

// Compound types.
pub const COMPOUND_WEDGE: u8 = 0;
pub const COMPOUND_DIFFWTD: u8 = 1;
pub const COMPOUND_AVERAGE: u8 = 2;
pub const COMPOUND_INTRA: u8 = 3;
pub const COMPOUND_DISTANCE: u8 = 4;

// Transform types.
pub const DCT_DCT: u8 = 0;
pub const IDTX: u8 = 9;
pub const V_DCT: u8 = 10;
pub const H_DCT: u8 = 11;
pub const V_ADST: u8 = 12;
pub const H_ADST: u8 = 13;
pub const V_FLIPADST: u8 = 14;
pub const H_FLIPADST: u8 = 15;

pub const TX_4X4: u8 = 0;
pub const TX_8X8: u8 = 1;
pub const TX_16X16: u8 = 2;
pub const TX_32X32: u8 = 3;
pub const TX_64X64: u8 = 4;
pub const TX_16X32: u8 = 9;
pub const TX_32X16: u8 = 10;
pub const TX_16X64: u8 = 17;
pub const TX_64X16: u8 = 18;

#[derive(Clone, Copy, Debug, Default, Eq)]
pub struct Mv {
    pub row: i16,
    pub col: i16,
}

impl Mv {
    pub const ZERO: Mv = Mv { row: 0, col: 0 };
    pub const INVALID: Mv = Mv {
        row: i16::MIN,
        col: i16::MIN,
    };
    pub fn new(row: i32, col: i32) -> Self {
        Mv {
            row: row as i16,
            col: col as i16,
        }
    }

    /// Both components in one word, to compare them at once.
    #[inline(always)]
    fn bits(self) -> u32 {
        u32::from(self.row as u16) | (u32::from(self.col as u16) << 16)
    }
}

impl PartialEq for Mv {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.bits() == other.bits()
    }
}

/// Mode info kept for every 4x4 luma unit of the frame: 16 bytes, so a
/// block writes one vector to each of its units. The small fields are
/// packed; [`MiFields`] has them apart, to build one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MiInfo {
    pub ref_frame: [i8; 2],
    pub mv: [Mv; 2],
    pub bsize: u8,
    pub y_mode: u8,
    /// is_inter, skip, skip_mode, comp_group_idx, compound_idx (bits 0 to
    /// 4) and motion_mode (bits 5 and 6).
    flags: u8,
    /// The luma palette size (bits 0 to 3) and the chroma one.
    palette: u8,
    /// The two interpolation filters, 3 bits each.
    interp: u8,
    /// tx_size (bits 0 to 4) and segment_id.
    tx_seg: u8,
}

/// The fields of a [`MiInfo`], unpacked.
#[derive(Clone, Copy, Debug)]
pub struct MiFields {
    pub ref_frame: [i8; 2],
    pub mv: [Mv; 2],
    pub bsize: u8,
    pub y_mode: u8,
    pub is_inter: bool,
    pub skip: bool,
    pub skip_mode: bool,
    pub segment_id: u8,
    pub tx_size: u8,
    pub palette_size: [u8; 2],
    pub comp_group_idx: u8,
    pub compound_idx: u8,
    pub interp_filter: [u8; 2],
    pub motion_mode: u8,
}

impl Default for MiFields {
    fn default() -> Self {
        Self {
            ref_frame: [NOT_WRITTEN, NONE],
            mv: [Mv::ZERO; 2],
            bsize: 0,
            y_mode: 0,
            is_inter: false,
            skip: false,
            skip_mode: false,
            segment_id: 0,
            tx_size: 0,
            palette_size: [0; 2],
            comp_group_idx: 0,
            compound_idx: 0,
            interp_filter: [0; 2],
            motion_mode: 0,
        }
    }
}

impl From<MiFields> for MiInfo {
    #[inline(always)]
    fn from(f: MiFields) -> Self {
        Self {
            ref_frame: f.ref_frame,
            mv: f.mv,
            bsize: f.bsize,
            y_mode: f.y_mode,
            flags: u8::from(f.is_inter)
                | u8::from(f.skip) << 1
                | u8::from(f.skip_mode) << 2
                | (f.comp_group_idx & 1) << 3
                | (f.compound_idx & 1) << 4
                | (f.motion_mode & 3) << 5,
            palette: (f.palette_size[0] & 15) | f.palette_size[1] << 4,
            interp: (f.interp_filter[0] & 7) | (f.interp_filter[1] & 7) << 3,
            tx_seg: (f.tx_size & 31) | f.segment_id << 5,
        }
    }
}

impl Default for MiInfo {
    fn default() -> Self {
        MiFields::default().into()
    }
}

impl MiInfo {
    #[inline(always)]
    pub fn is_inter(&self) -> bool {
        self.flags & 1 != 0
    }
    #[inline(always)]
    pub fn skip(&self) -> bool {
        self.flags & 2 != 0
    }
    #[inline(always)]
    pub fn skip_mode(&self) -> bool {
        self.flags & 4 != 0
    }
    #[inline(always)]
    pub fn comp_group_idx(&self) -> u8 {
        (self.flags >> 3) & 1
    }
    #[inline(always)]
    pub fn compound_idx(&self) -> u8 {
        (self.flags >> 4) & 1
    }
    #[inline(always)]
    pub fn motion_mode(&self) -> u8 {
        (self.flags >> 5) & 3
    }
    #[inline(always)]
    pub fn palette_size(&self) -> [u8; 2] {
        [self.palette & 15, self.palette >> 4]
    }
    #[inline(always)]
    pub fn interp_filter(&self) -> [u8; 2] {
        [self.interp & 7, (self.interp >> 3) & 7]
    }
    #[inline(always)]
    pub fn tx_size(&self) -> u8 {
        self.tx_seg & 31
    }
    #[inline(always)]
    pub fn segment_id(&self) -> u8 {
        self.tx_seg >> 5
    }
}

const _: () = assert!(size_of::<MiInfo>() == 16);

#[inline(always)]
pub fn bw4(bsize: u8) -> usize {
    NUM_4X4_BLOCKS_WIDE[bsize as usize] as usize
}
#[inline(always)]
pub fn bh4(bsize: u8) -> usize {
    NUM_4X4_BLOCKS_HIGH[bsize as usize] as usize
}
#[inline(always)]
pub fn block_width(bsize: u8) -> usize {
    bw4(bsize) * 4
}
#[inline(always)]
pub fn block_height(bsize: u8) -> usize {
    bh4(bsize) * 4
}
#[inline(always)]
pub fn mi_width_log2(bsize: u8) -> u32 {
    MI_WIDTH_LOG2[bsize as usize] as u32
}
#[inline(always)]
pub fn mi_height_log2(bsize: u8) -> u32 {
    MI_HEIGHT_LOG2[bsize as usize] as u32
}
#[inline(always)]
pub fn tx_w(tx: u8) -> usize {
    TX_WIDTH[tx as usize] as usize
}
#[inline(always)]
pub fn tx_h(tx: u8) -> usize {
    TX_HEIGHT[tx as usize] as usize
}
#[inline(always)]
pub fn tx_w_log2(tx: u8) -> u32 {
    TX_WIDTH_LOG2[tx as usize] as u32
}
#[inline(always)]
pub fn tx_h_log2(tx: u8) -> u32 {
    TX_HEIGHT_LOG2[tx as usize] as u32
}

pub fn plane_residual_size(bsize: u8, sub_x: bool, sub_y: bool) -> u8 {
    SUBSAMPLED_SIZE[bsize as usize][usize::from(sub_x)][usize::from(sub_y)]
}

pub fn find_tx_size(w: usize, h: usize) -> u8 {
    for t in 0..19u8 {
        if tx_w(t) == w && tx_h(t) == h {
            return t;
        }
    }
    0
}

pub fn has_newmv(mode: u8) -> bool {
    matches!(
        mode,
        NEWMV | NEW_NEWMV | NEAR_NEWMV | NEW_NEARMV | NEAREST_NEWMV | NEW_NEARESTMV
    )
}

pub fn is_directional(mode: u8) -> bool {
    (V_PRED..=D67_PRED).contains(&mode)
}
