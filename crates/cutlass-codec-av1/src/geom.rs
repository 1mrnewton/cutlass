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

/// `Subsampled_Size[bsize][sub_x][sub_y]`. `None` is `BLOCK_INVALID`.
pub fn subsampled_block_size(bsize: usize, sub_x: u8, sub_y: u8) -> Option<usize> {
    const INVALID: u8 = 255;
    const TABLE: [[[u8; 2]; 2]; 22] = [
        [[0, 0], [0, 0]],
        [[1, 0], [INVALID, 0]],
        [[2, INVALID], [0, 0]],
        [[3, 2], [1, 0]],
        [[4, 3], [INVALID, 1]],
        [[5, INVALID], [3, 2]],
        [[6, 5], [4, 3]],
        [[7, 6], [INVALID, 4]],
        [[8, INVALID], [6, 5]],
        [[9, 8], [7, 6]],
        [[10, 9], [INVALID, 7]],
        [[11, INVALID], [9, 8]],
        [[12, 11], [10, 9]],
        [[13, 12], [INVALID, 10]],
        [[14, INVALID], [12, 11]],
        [[15, 14], [13, 12]],
        [[16, 1], [INVALID, 1]],
        [[17, INVALID], [2, 2]],
        [[18, 4], [INVALID, 16]],
        [[19, INVALID], [5, 17]],
        [[20, 7], [INVALID, 18]],
        [[21, INVALID], [8, 19]],
    ];
    if sub_x > 1 || sub_y > 1 {
        return None;
    }
    let value = TABLE.get(bsize)?[sub_x as usize][sub_y as usize];
    if value == INVALID {
        None
    } else {
        Some(value as usize)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsampled_420_sizes_match_the_spec_rows() {
        assert_eq!(subsampled_block_size(BLOCK_16X16, 1, 1), Some(BLOCK_8X8));
        assert_eq!(subsampled_block_size(BLOCK_8X8, 1, 1), Some(BLOCK_4X4));
        assert_eq!(subsampled_block_size(BLOCK_4X8, 1, 0), None);
        assert_eq!(subsampled_block_size(16, 0, 1), Some(BLOCK_4X8));
    }
}
