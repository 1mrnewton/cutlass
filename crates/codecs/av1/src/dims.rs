//! Block and transform sizes from the AV1 specification tables.

pub const TX_4X4: usize = 0;
pub const TX_8X8: usize = 1;
pub const TX_16X16: usize = 2;
pub const TX_32X32: usize = 3;
pub const TX_64X64: usize = 4;
pub const TX_4X8: usize = 5;
pub const TX_8X4: usize = 6;
pub const TX_8X16: usize = 7;
pub const TX_16X8: usize = 8;
pub const TX_16X32: usize = 9;
pub const TX_32X16: usize = 10;
pub const TX_32X64: usize = 11;
pub const TX_64X32: usize = 12;
pub const TX_4X16: usize = 13;
pub const TX_16X4: usize = 14;
pub const TX_8X32: usize = 15;
pub const TX_32X8: usize = 16;
pub const TX_16X64: usize = 17;
pub const TX_64X16: usize = 18;

pub const DCT_DCT: u8 = 0;
pub const ADST_DCT: u8 = 1;
pub const DCT_ADST: u8 = 2;
pub const ADST_ADST: u8 = 3;
pub const FLIPADST_DCT: u8 = 4;
pub const DCT_FLIPADST: u8 = 5;
pub const FLIPADST_FLIPADST: u8 = 6;
pub const ADST_FLIPADST: u8 = 7;
pub const FLIPADST_ADST: u8 = 8;
pub const IDTX: u8 = 9;
pub const V_DCT: u8 = 10;
pub const H_DCT: u8 = 11;
pub const V_ADST: u8 = 12;
pub const H_ADST: u8 = 13;
pub const V_FLIPADST: u8 = 14;
pub const H_FLIPADST: u8 = 15;

pub const DC_PRED: u8 = 0;
pub const V_PRED: u8 = 1;
pub const D67_PRED: u8 = 8;
pub const SMOOTH_PRED: u8 = 9;
pub const PAETH_PRED: u8 = 12;
pub const UV_CFL_PRED: u8 = 13;

pub const PARTITION_NONE: u32 = 0;
pub const PARTITION_HORZ: u32 = 1;
pub const PARTITION_VERT: u32 = 2;
pub const PARTITION_SPLIT: u32 = 3;
pub const PARTITION_HORZ_A: u32 = 4;
pub const PARTITION_HORZ_B: u32 = 5;
pub const PARTITION_VERT_A: u32 = 6;
pub const PARTITION_VERT_B: u32 = 7;
pub const PARTITION_HORZ_4: u32 = 8;
pub const PARTITION_VERT_4: u32 = 9;

pub const BLOCK_8X8: usize = 3;
pub const BLOCK_128X128: usize = 15;

pub const MI_WIDTH_LOG2: [usize; 22] = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 0, 2, 1, 3, 2, 4,
];
pub const MI_HEIGHT_LOG2: [usize; 22] = [
    0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 2, 0, 3, 1, 4, 2,
];
pub const NUM_4X4_WIDE: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
pub const NUM_4X4_HIGH: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

pub const TX_WIDTH: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
pub const TX_HEIGHT: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
pub const TX_WIDTH_LOG2: [u32; 19] = [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6];
pub const TX_HEIGHT_LOG2: [u32; 19] = [2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 4, 2, 5, 3, 6, 4];
pub const TX_SIZE_SQR: [usize; 19] = [0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2];
pub const TX_SIZE_SQR_UP: [usize; 19] = [0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4];
pub const MAX_TX_SIZE_RECT: [usize; 22] = [
    0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18,
];
pub const MAX_TX_DEPTH: [usize; 22] = [
    0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4,
];
pub const SPLIT_TX_SIZE: [usize; 19] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];
pub const TRANSFORM_ROW_SHIFT: [u32; 19] =
    [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];
pub const ADJUSTED_TX_SIZE: [usize; 19] = [
    0, 1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 3, 3, 13, 14, 15, 16, 9, 10,
];

pub const MODE_TO_TXFM: [u8; 14] = [0, 1, 2, 0, 3, 1, 2, 2, 1, 3, 1, 2, 3, 0];

const INTRA_TX_INV1: [u8; 7] = [IDTX, DCT_DCT, V_DCT, H_DCT, ADST_ADST, ADST_DCT, DCT_ADST];
const INTRA_TX_INV2: [u8; 5] = [IDTX, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST];

pub const COEFF_BASE_CTX_OFFSET: [u8; 19 * 25] = [
    0, 1, 6, 6, 0, 1, 6, 6, 21, 0, 6, 6, 21, 21, 0, 6, 21, 21, 21, 0, 0, 0, 0, 0, 0, 0, 1, 6, 6,
    21, 1, 6, 6, 21, 21, 6, 6, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 0, 1, 6, 6, 21,
    1, 6, 6, 21, 21, 6, 6, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 0, 1, 6, 6, 21, 1, 6,
    6, 21, 21, 6, 6, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 0, 1, 6, 6, 21, 1, 6, 6,
    21, 21, 6, 6, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 0, 11, 11, 11, 0, 11, 11, 11,
    11, 0, 6, 6, 21, 21, 0, 6, 21, 21, 21, 0, 21, 21, 21, 21, 0, 0, 16, 6, 6, 21, 16, 16, 6, 21,
    21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 6, 6, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 0, 16, 6, 6, 21, 16, 16, 6, 21,
    21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 0, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 6, 6, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 0, 16, 6, 6, 21, 16, 16, 6,
    21, 21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 0, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 6, 6, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 0, 16, 6, 6, 21, 16, 16,
    6, 21, 21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 0, 11, 11, 11, 0, 11,
    11, 11, 11, 0, 6, 6, 21, 21, 0, 6, 21, 21, 21, 0, 21, 21, 21, 21, 0, 0, 16, 6, 6, 21, 16, 16,
    6, 21, 21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 6, 6, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 0, 16, 6, 6, 21, 16, 16,
    6, 21, 21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 0, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 6, 6, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 0, 16, 6, 6, 21, 16,
    16, 6, 21, 21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21, 16, 16, 21, 21, 21,
];

#[must_use]
pub fn block_width(size: usize) -> usize {
    NUM_4X4_WIDE[size] * 4
}

#[must_use]
pub fn block_height(size: usize) -> usize {
    NUM_4X4_HIGH[size] * 4
}

#[must_use]
pub fn intra_tx_type(set: usize, symbol: u32) -> Option<u8> {
    let table: &[u8] = if set == 1 {
        &INTRA_TX_INV1
    } else {
        &INTRA_TX_INV2
    };
    table.get(symbol as usize).copied()
}

/// Sub-block size for none, horizontal, vertical, and split.
#[must_use]
pub fn partition_subsize(partition: u32, bsize: usize) -> Option<usize> {
    const NONE: [usize; 6] = [0, 3, 6, 9, 12, 15];
    const HORZ: [usize; 6] = [0, 2, 5, 8, 11, 14];
    const VERT: [usize; 6] = [0, 1, 4, 7, 10, 13];
    const SPLIT: [usize; 6] = [0, 0, 3, 6, 9, 12];
    let bsl = MI_WIDTH_LOG2[bsize];
    let square = match partition {
        PARTITION_NONE => &NONE,
        PARTITION_HORZ | PARTITION_HORZ_A | PARTITION_HORZ_B => &HORZ,
        PARTITION_VERT | PARTITION_VERT_A | PARTITION_VERT_B => &VERT,
        PARTITION_SPLIT => &SPLIT,
        PARTITION_HORZ_4 => return [None, None, Some(17), Some(19), Some(21), None][bsl],
        PARTITION_VERT_4 => return [None, None, Some(16), Some(18), Some(20), None][bsl],
        _ => return None,
    };
    square.get(bsl).copied()
}

/// 4:2:0 residual block size. `sub` is 1 when the plane is chroma.
#[must_use]
pub fn subsampled_size(bsize: usize, sub_x: bool, sub_y: bool) -> usize {
    const TABLE: [[usize; 4]; 22] = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 2, 1, 0],
        [4, 3, 0, 1],
        [5, 0, 3, 2],
        [6, 5, 4, 3],
        [7, 6, 0, 4],
        [8, 0, 6, 5],
        [9, 8, 7, 6],
        [10, 9, 0, 7],
        [11, 0, 9, 8],
        [12, 11, 10, 9],
        [13, 12, 0, 10],
        [14, 0, 12, 11],
        [15, 14, 13, 12],
        [16, 4, 0, 4],
        [17, 0, 2, 2],
        [18, 4, 0, 16],
        [19, 0, 5, 17],
        [20, 7, 0, 18],
        [21, 0, 8, 19],
    ];
    let index = usize::from(sub_x) * 2 + usize::from(sub_y);
    TABLE[bsize][index]
}

pub(crate) fn rectangular_transforms_are_numbered() -> bool {
    [
        TX_4X8, TX_8X4, TX_8X16, TX_16X8, TX_4X16, TX_16X4, TX_8X32, TX_32X8,
    ]
    .windows(2)
    .all(|pair| pair[0] < pair[1])
}

#[cfg(test)]
mod tests {
    use super::{
        TX_4X4, TX_4X8, TX_4X16, TX_8X4, TX_8X8, TX_8X16, TX_8X32, TX_16X4, TX_16X8, TX_16X16,
        TX_16X32, TX_16X64, TX_32X8, TX_32X16, TX_32X32, TX_32X64, TX_64X16, TX_64X32, TX_64X64,
    };

    #[test]
    fn transform_ids_follow_the_spec_order() {
        assert_eq!(
            [
                TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64, TX_4X8, TX_8X4, TX_8X16, TX_16X8,
                TX_16X32, TX_32X16, TX_32X64, TX_64X32, TX_4X16, TX_16X4, TX_8X32, TX_32X8,
                TX_16X64, TX_64X16,
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
            ]
        );
    }
}
