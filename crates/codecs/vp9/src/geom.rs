#![allow(dead_code)]
//! Block geometry for profile 0.
//!
//! Sizes, partition children, and transform limits follow libvpx
//! `vp9_common_data.c`. Partition context bytes are the libvpx lookup, not
//! the `15 >> b_width_log2` formula in the specification text.

pub const BLOCK_4X4: u8 = 0;
pub const BLOCK_4X8: u8 = 1;
pub const BLOCK_8X4: u8 = 2;
pub const BLOCK_8X8: u8 = 3;
pub const BLOCK_8X16: u8 = 4;
pub const BLOCK_16X8: u8 = 5;
pub const BLOCK_16X16: u8 = 6;
pub const BLOCK_16X32: u8 = 7;
pub const BLOCK_32X16: u8 = 8;
pub const BLOCK_32X32: u8 = 9;
pub const BLOCK_32X64: u8 = 10;
pub const BLOCK_64X32: u8 = 11;
pub const BLOCK_64X64: u8 = 12;
pub const BLOCK_INVALID: u8 = 13;

pub const PARTITION_NONE: u8 = 0;
pub const PARTITION_HORZ: u8 = 1;
pub const PARTITION_VERT: u8 = 2;
pub const PARTITION_SPLIT: u8 = 3;

pub const TX_4X4: u8 = 0;
pub const TX_8X8: u8 = 1;
pub const TX_16X16: u8 = 2;
pub const TX_32X32: u8 = 3;

pub const ONLY_4X4: u8 = 0;
pub const ALLOW_8X8: u8 = 1;
pub const ALLOW_16X16: u8 = 2;
pub const ALLOW_32X32: u8 = 3;
pub const TX_MODE_SELECT: u8 = 4;

pub const DC: u8 = 0;
pub const V: u8 = 1;
pub const H: u8 = 2;
pub const D45: u8 = 3;
pub const D135: u8 = 4;
pub const D117: u8 = 5;
pub const D153: u8 = 6;
pub const D207: u8 = 7;
pub const D63: u8 = 8;
pub const TM: u8 = 9;
pub const NEAREST: u8 = 10;
pub const NEAR: u8 = 11;
pub const ZERO: u8 = 12;
pub const NEW: u8 = 13;

pub const INTRA_FRAME: u8 = 0;
pub const LAST_FRAME: u8 = 1;
pub const GOLDEN_FRAME: u8 = 2;
pub const ALTREF_FRAME: u8 = 3;

pub const SINGLE_REFERENCE: u8 = 0;
pub const COMPOUND_REFERENCE: u8 = 1;
pub const REFERENCE_SELECT: u8 = 2;

pub const EIGHTTAP: u8 = 0;
pub const SMOOTH: u8 = 1;
pub const SHARP: u8 = 2;
pub const BILINEAR: u8 = 3;
pub const SWITCHABLE: u8 = 4;

pub const DCT_DCT: u8 = 0;
pub const ADST_DCT: u8 = 1;
pub const DCT_ADST: u8 = 2;
pub const ADST_ADST: u8 = 3;

pub const MI_SIZE: usize = 8;
pub const MAX_EDGE: u32 = 4096;
pub const BORDER: usize = 32;

const BLOCK_W: [u32; 13] = [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64];
const BLOCK_H: [u32; 13] = [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64];
const MI_W: [u8; 13] = [1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8];
const MI_H: [u8; 13] = [1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8];
const MI_LOG2: [u8; 13] = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3];
const SIZE_GROUP: [u8; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3];
const MAX_TX: [u8; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3];
const PART_ABOVE: [u8; 13] = [15, 15, 14, 14, 14, 12, 12, 12, 8, 8, 8, 0, 0];
const PART_LEFT: [u8; 13] = [15, 14, 15, 14, 12, 14, 12, 8, 12, 8, 0, 8, 0];

const SUBSIZE: [[u8; 13]; 4] = [
    [
        BLOCK_4X4,
        BLOCK_4X8,
        BLOCK_8X4,
        BLOCK_8X8,
        BLOCK_8X16,
        BLOCK_16X8,
        BLOCK_16X16,
        BLOCK_16X32,
        BLOCK_32X16,
        BLOCK_32X32,
        BLOCK_32X64,
        BLOCK_64X32,
        BLOCK_64X64,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X4,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_64X32,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_4X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X32,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X64,
    ],
    [
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_4X4,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_8X8,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_16X16,
        BLOCK_INVALID,
        BLOCK_INVALID,
        BLOCK_32X32,
    ],
];

/// 4:2:0 chroma transform for a luma block and luma transform.
const UV_TX_420: [[u8; 4]; 13] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 1, 2, 2],
    [0, 1, 2, 2],
    [0, 1, 2, 2],
    [0, 1, 2, 3],
];

pub fn block_wh(bsize: u8) -> (u32, u32) {
    (BLOCK_W[bsize as usize], BLOCK_H[bsize as usize])
}

pub fn mi_wh(bsize: u8) -> (usize, usize) {
    (
        usize::from(MI_W[bsize as usize]),
        usize::from(MI_H[bsize as usize]),
    )
}

pub fn mi_width_log2(bsize: u8) -> u32 {
    u32::from(MI_LOG2[bsize as usize])
}

pub fn size_group(bsize: u8) -> usize {
    usize::from(SIZE_GROUP[bsize as usize])
}

pub fn max_tx(bsize: u8) -> u8 {
    MAX_TX[bsize as usize]
}

pub fn subsize(partition: u8, bsize: u8) -> u8 {
    SUBSIZE[partition as usize][bsize as usize]
}

pub fn partition_context_byte(bsize: u8) -> (u8, u8) {
    (PART_ABOVE[bsize as usize], PART_LEFT[bsize as usize])
}

pub fn uv_tx_420(bsize: u8, tx: u8) -> u8 {
    UV_TX_420[bsize as usize][tx as usize]
}

pub fn tx_mode_biggest(mode: u8) -> u8 {
    if mode >= ALLOW_32X32 { TX_32X32 } else { mode }
}

pub fn intra_tx_type(mode: u8) -> u8 {
    match mode {
        V | D117 | D63 => ADST_DCT,
        H | D153 | D207 => DCT_ADST,
        D135 | TM => ADST_ADST,
        _ => DCT_DCT,
    }
}

pub fn align8(value: u32) -> u32 {
    value.div_ceil(8) * 8
}

pub fn mi_count(edge: u32) -> usize {
    usize::try_from(align8(edge) / 8).unwrap_or(0)
}
