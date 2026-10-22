//! Block geometry. Sizes follow the specification's `BLOCK_*` order.
//!
//! `Partition_Subsize` for `NONE` / `HORZ` / `VERT` / `SPLIT` is the square-block
//! portion of the table in the additional-tables annex. Rectangular partitions
//! are rejected by the caller until that row of the table is exercised.

pub const MI_SIZE: u32 = 4;

pub const BLOCK_4X4: usize = 0;
pub const BLOCK_4X8: usize = 1;
pub const BLOCK_8X4: usize = 2;
pub const BLOCK_8X8: usize = 3;
pub const BLOCK_8X16: usize = 4;
pub const BLOCK_16X8: usize = 5;
pub const BLOCK_16X16: usize = 6;
pub const BLOCK_16X32: usize = 7;
pub const BLOCK_32X16: usize = 8;
pub const BLOCK_32X32: usize = 9;
pub const BLOCK_32X64: usize = 10;
pub const BLOCK_64X32: usize = 11;
pub const BLOCK_64X64: usize = 12;
pub const BLOCK_64X128: usize = 13;
pub const BLOCK_128X64: usize = 14;
pub const BLOCK_128X128: usize = 15;

pub const PARTITION_NONE: u8 = 0;
pub const PARTITION_HORZ: u8 = 1;
pub const PARTITION_VERT: u8 = 2;
pub const PARTITION_SPLIT: u8 = 3;

pub const DC_PRED: u8 = 0;
pub const V_PRED: u8 = 1;
pub const H_PRED: u8 = 2;
#[allow(dead_code)]
pub const UV_CFL_PRED: u8 = 13;

/// `Intra_Mode_Context`.
pub const INTRA_MODE_CONTEXT: [u8; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

const BLOCK_WIDTH: [u32; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLOCK_HEIGHT: [u32; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

pub fn block_width(bsize: usize) -> Option<u32> {
    BLOCK_WIDTH.get(bsize).copied()
}

pub fn block_height(bsize: usize) -> Option<u32> {
    BLOCK_HEIGHT.get(bsize).copied()
}

/// `Mi_Width_Log2`: log2 of the width in 4x4 units.
pub fn mi_width_log2(bsize: usize) -> Option<u32> {
    let w = block_width(bsize)?;
    Some(w.trailing_zeros() - 2)
}

pub fn mi_height_log2(bsize: usize) -> Option<u32> {
    let h = block_height(bsize)?;
    Some(h.trailing_zeros() - 2)
}

pub fn num_4x4_wide(bsize: usize) -> Option<u32> {
    Some(block_width(bsize)? / MI_SIZE)
}

pub fn superblock_size(use_128: bool) -> usize {
    if use_128 { BLOCK_128X128 } else { BLOCK_64X64 }
}

/// Sub-block size for the four partition types that square blocks use in the
/// forced-split and uniform cases. `None` means the combination is invalid.
pub fn partition_subsize(partition: u8, bsize: usize) -> Option<usize> {
    let square = matches!(
        bsize,
        BLOCK_4X4 | BLOCK_8X8 | BLOCK_16X16 | BLOCK_32X32 | BLOCK_64X64 | BLOCK_128X128
    );
    if !square {
        return None;
    }
    match partition {
        PARTITION_NONE => Some(bsize),
        PARTITION_HORZ => match bsize {
            BLOCK_8X8 => Some(BLOCK_8X4),
            BLOCK_16X16 => Some(BLOCK_16X8),
            BLOCK_32X32 => Some(BLOCK_32X16),
            BLOCK_64X64 => Some(BLOCK_64X32),
            BLOCK_128X128 => Some(BLOCK_128X64),
            _ => None,
        },
        PARTITION_VERT => match bsize {
            BLOCK_8X8 => Some(BLOCK_4X8),
            BLOCK_16X16 => Some(BLOCK_8X16),
            BLOCK_32X32 => Some(BLOCK_16X32),
            BLOCK_64X64 => Some(BLOCK_32X64),
            BLOCK_128X128 => Some(BLOCK_64X128),
            _ => None,
        },
        PARTITION_SPLIT => match bsize {
            BLOCK_8X8 => Some(BLOCK_4X4),
            BLOCK_16X16 => Some(BLOCK_8X8),
            BLOCK_32X32 => Some(BLOCK_16X16),
            BLOCK_64X64 => Some(BLOCK_32X32),
            BLOCK_128X128 => Some(BLOCK_64X64),
            _ => None,
        },
        _ => None,
    }
}
