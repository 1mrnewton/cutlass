//! Inverse transform, intra prediction, motion compensation, and deblocking.
//! Clauses 8.3, 8.4, 8.5, and 8.7. The encoder is not specified by the recommendation.

const NORM4: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

const NORM8: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

const ALPHA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];

const BETA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

const TC0: [[u8; 52]; 3] = [
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 10, 11, 12, 13, 15, 17,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 23, 25,
    ],
];

const CHROMA_QP: [i32; 52] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39,
    39, 39,
];

pub fn chroma_qp(luma_qp: i32, offset: i32) -> i32 {
    CHROMA_QP[(luma_qp + offset).clamp(0, 51) as usize]
}

/// QP′C, which includes `qp_bd`, and the deblock index with that offset removed.
///
/// `luma_qp` is SliceQPY. For an 8-bit picture `qp_bd` is 0 and the quantiser
/// index is [`chroma_qp`].
pub fn chroma_qps(luma_qp: i32, offset: i32, qp_bd: i32) -> (i32, i32) {
    let max = 51 + qp_bd;
    let index = (luma_qp + qp_bd + offset).clamp(0, max);
    let quant = if qp_bd > 0 && index < qp_bd {
        index
    } else {
        CHROMA_QP[(index - qp_bd) as usize] + qp_bd
    };
    (quant, quant - qp_bd)
}

fn norm4(i: usize, j: usize) -> usize {
    if i.is_multiple_of(2) && j.is_multiple_of(2) {
        0
    } else if i % 2 == 1 && j % 2 == 1 {
        1
    } else {
        2
    }
}

pub(crate) fn class8(i: usize, j: usize) -> usize {
    norm8(i, j)
}

fn norm8(i: usize, j: usize) -> usize {
    let (a, b) = (i % 4, j % 4);
    if (a, b) == (0, 0) {
        0
    } else if i % 2 == 1 && j % 2 == 1 {
        1
    } else if (a, b) == (2, 2) {
        2
    } else if (a, j % 2) == (0, 1) || (i % 2, b) == (1, 0) {
        3
    } else if (a, b) == (0, 2) || (a, b) == (2, 0) {
        4
    } else {
        5
    }
}

/// `LevelScale4x4(qp % 6, row, col) * weight`, equation 8-313 with a scaling list.
pub(crate) fn level_scale4(qp: i32, row: usize, col: usize, weight: i32) -> i32 {
    weight * NORM4[(qp.rem_euclid(6)) as usize][norm4(row, col)]
}

pub fn scale4(coeff: i32, qp: i32, i: usize, j: usize, weight: i32) -> i32 {
    let ls = weight * NORM4[(qp.rem_euclid(6)) as usize][norm4(i, j)];
    let q = qp / 6;
    let v = i64::from(coeff) * i64::from(ls);
    let scaled = if qp >= 24 {
        v << (q - 4)
    } else {
        (v + (1i64 << (3 - q))) >> (4 - q)
    };
    scaled as i32
}

pub fn scale8(coeff: i32, qp: i32, i: usize, j: usize, weight: i32) -> i32 {
    let ls = weight * NORM8[(qp.rem_euclid(6)) as usize][norm8(i, j)];
    let q = qp / 6;
    let v = i64::from(coeff) * i64::from(ls);
    let scaled = if qp >= 36 {
        v << (q - 6)
    } else {
        (v + (1i64 << (5 - q))) >> (6 - q)
    };
    scaled as i32
}

pub fn idct4(block: &mut [i32; 16]) {
    // Only the DC coefficient: both passes spread it over the block as it
    // is, so every sample is the same rounded value.
    if block[1..] == [0; 15] {
        *block = [(block[0] + 32) >> 6; 16];
        return;
    }
    idct4_full(block);
}

fn idct4_full(block: &mut [i32; 16]) {
    let mut mid = [0i32; 16];
    for i in 0..4 {
        let d0 = block[i * 4];
        let d1 = block[i * 4 + 1];
        let d2 = block[i * 4 + 2];
        let d3 = block[i * 4 + 3];
        let e0 = d0 + d2;
        let e1 = d0 - d2;
        let e2 = (d1 >> 1) - d3;
        let e3 = d1 + (d3 >> 1);
        mid[i * 4] = e0 + e3;
        mid[i * 4 + 1] = e1 + e2;
        mid[i * 4 + 2] = e1 - e2;
        mid[i * 4 + 3] = e0 - e3;
    }
    for j in 0..4 {
        let d0 = mid[j];
        let d1 = mid[4 + j];
        let d2 = mid[8 + j];
        let d3 = mid[12 + j];
        let e0 = d0 + d2;
        let e1 = d0 - d2;
        let e2 = (d1 >> 1) - d3;
        let e3 = d1 + (d3 >> 1);
        block[j] = (e0 + e3 + 32) >> 6;
        block[4 + j] = (e1 + e2 + 32) >> 6;
        block[8 + j] = (e1 - e2 + 32) >> 6;
        block[12 + j] = (e0 - e3 + 32) >> 6;
    }
}

pub fn luma_dc(coeff: &[i32; 16], qp: i32, weight00: i32) -> [i32; 16] {
    let mut temp = [0i32; 16];
    for i in 0..4 {
        let z0 = coeff[i * 4] + coeff[i * 4 + 1];
        let z1 = coeff[i * 4] - coeff[i * 4 + 1];
        let z2 = coeff[i * 4 + 2] - coeff[i * 4 + 3];
        let z3 = coeff[i * 4 + 2] + coeff[i * 4 + 3];
        temp[i * 4] = z0 + z3;
        temp[i * 4 + 1] = z0 - z3;
        temp[i * 4 + 2] = z1 - z2;
        temp[i * 4 + 3] = z1 + z2;
    }
    let mut out = [0i32; 16];
    let ls = weight00 * NORM4[(qp.rem_euclid(6)) as usize][0];
    let q = qp / 6;
    for i in 0..4 {
        let z0 = temp[i] + temp[8 + i];
        let z1 = temp[i] - temp[8 + i];
        let z2 = temp[4 + i] - temp[12 + i];
        let z3 = temp[4 + i] + temp[12 + i];
        let f = [z0 + z3, z1 + z2, z1 - z2, z0 - z3];
        for (row, value) in f.iter().enumerate() {
            let v = i64::from(*value) * i64::from(ls);
            let scaled = if qp >= 36 {
                v << (q - 6)
            } else {
                (v + (1i64 << (5 - q))) >> (6 - q)
            };
            out[row * 4 + i] = scaled as i32;
        }
    }
    out
}

pub fn chroma_dc(c00: i32, c01: i32, c10: i32, c11: i32, qp: i32, weight00: i32) -> [i32; 4] {
    let f = [
        c00 + c01 + c10 + c11,
        c00 - c01 + c10 - c11,
        c00 + c01 - c10 - c11,
        c00 - c01 - c10 + c11,
    ];
    let ls = weight00 * NORM4[(qp.rem_euclid(6)) as usize][0];
    let q = qp / 6;
    let mut out = [0i32; 4];
    for (slot, value) in f.iter().enumerate() {
        let v = i64::from(*value) * i64::from(ls);
        out[slot] = ((v << q) >> 5) as i32;
    }
    out
}

/// Inverse 2×4 Hadamard and scale for a 4:2:2 chroma DC block.
///
/// `raster` is row-major, index `y * 2 + x`. The scale matches the Intra_16x16
/// DC scale: round when `qp < 36`, otherwise shift.
pub fn chroma_dc422(raster: &[i32; 8], qp: i32, weight: i32) -> [i32; 8] {
    let mut temp = [0i64; 8];
    for y in 0..4 {
        let left = i64::from(raster[y * 2]);
        let right = i64::from(raster[y * 2 + 1]);
        temp[y * 2] = left + right;
        temp[y * 2 + 1] = left - right;
    }
    let ls = i64::from(weight) * i64::from(NORM4[(qp.rem_euclid(6)) as usize][0]);
    let q = qp / 6;
    let mut out = [0i32; 8];
    for x in 0..2 {
        let z0 = temp[x] + temp[4 + x];
        let z1 = temp[x] - temp[4 + x];
        let z2 = temp[2 + x] - temp[6 + x];
        let z3 = temp[2 + x] + temp[6 + x];
        let mixed = [z0 + z3, z1 + z2, z1 - z2, z0 - z3];
        for (y, value) in mixed.iter().enumerate() {
            let v = value * ls;
            let scaled = if qp >= 36 {
                v << (q - 6)
            } else {
                (v + (1i64 << (5 - q))) >> (6 - q)
            };
            out[y * 2 + x] = scaled as i32;
        }
    }
    out
}

pub fn idct8(block: &mut [i32; 64]) {
    // Only the DC coefficient, as in [`idct4`].
    if block[1..] == [0; 63] {
        *block = [(block[0] + 32) >> 6; 64];
        return;
    }
    idct8_full(block);
}

fn idct8_full(block: &mut [i32; 64]) {
    let mut mid = [0i32; 64];
    for i in 0..8 {
        idct8_1d(&block[i * 8..i * 8 + 8], &mut mid[i * 8..i * 8 + 8]);
    }
    let mut cols = [0i32; 8];
    let mut out = [0i32; 8];
    for j in 0..8 {
        for i in 0..8 {
            cols[i] = mid[i * 8 + j];
        }
        idct8_1d(&cols, &mut out);
        for i in 0..8 {
            block[i * 8 + j] = (out[i] + 32) >> 6;
        }
    }
}

fn idct8_1d(d: &[i32], f: &mut [i32]) {
    let a0 = d[0] + d[4];
    let a2 = d[0] - d[4];
    let a4 = (d[2] >> 1) - d[6];
    let a6 = (d[6] >> 1) + d[2];
    let b0 = a0 + a6;
    let b2 = a2 + a4;
    let b4 = a2 - a4;
    let b6 = a0 - a6;
    let a1 = -d[3] + d[5] - d[7] - (d[7] >> 1);
    let a3 = d[1] + d[7] - d[3] - (d[3] >> 1);
    let a5 = -d[1] + d[7] + d[5] + (d[5] >> 1);
    let a7 = d[3] + d[5] + d[1] + (d[1] >> 1);
    let b1 = (a7 >> 2) + a1;
    let b3 = a3 + (a5 >> 2);
    let b5 = (a3 >> 2) - a5;
    let b7 = a7 - (a1 >> 2);
    f[0] = b0 + b7;
    f[7] = b0 - b7;
    f[1] = b2 + b5;
    f[6] = b2 - b5;
    f[2] = b4 + b3;
    f[5] = b4 - b3;
    f[3] = b6 + b1;
    f[4] = b6 - b1;
}

/// One sample width. `u8` is 8-bit. [`Depth`] is 9 through 14 bits.
pub trait Sample: Copy + Default + Into<i32> {
    const LIMIT: i32;
    const MID: i32;
    const SHIFT: i32;
    const QP_BD: i32;
    const BITS: u32;

    fn clip(value: i32) -> Self;

    fn mid() -> Self {
        Self::clip(Self::MID)
    }

    /// A PCM code of [`Self::BITS`] bits.
    fn from_code(value: u32) -> Self;

    /// The sample in an 8-bit plane. A wider sample keeps the top 8 bits.
    fn pack_eight(self) -> u8;

    /// The sample in a `u16` plane. 8-bit has no wide plane.
    fn pack_wide(self) -> Option<u16>;
}

impl Sample for u8 {
    const LIMIT: i32 = 255;
    const MID: i32 = 128;
    const SHIFT: i32 = 0;
    const QP_BD: i32 = 0;
    const BITS: u32 = 8;

    fn clip(value: i32) -> Self {
        value.clamp(0, Self::LIMIT) as u8
    }

    fn from_code(value: u32) -> Self {
        value as u8
    }

    fn pack_eight(self) -> u8 {
        self
    }

    fn pack_wide(self) -> Option<u16> {
        None
    }
}

/// One sample of `BITS` bits, stored in the low `BITS` of a `u16`.
///
/// `BITS` is 9, 10, 11, 12, 13, or 14. [`Ten`] is the 10-bit width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Depth<const BITS: u32>(pub u16);

/// One 10-bit sample.
pub type Ten = Depth<10>;

impl<const BITS: u32> From<Depth<BITS>> for i32 {
    fn from(value: Depth<BITS>) -> Self {
        i32::from(value.0)
    }
}

impl<const BITS: u32> Sample for Depth<BITS> {
    const LIMIT: i32 = (1i32 << BITS) - 1;
    const MID: i32 = 1i32 << (BITS - 1);
    const SHIFT: i32 = BITS as i32 - 8;
    const QP_BD: i32 = 6 * (BITS as i32 - 8);
    const BITS: u32 = BITS;

    fn clip(value: i32) -> Self {
        Self(value.clamp(0, Self::LIMIT) as u16)
    }

    fn from_code(value: u32) -> Self {
        Self((value & ((1u32 << BITS) - 1)) as u16)
    }

    fn pack_eight(self) -> u8 {
        (self.0 >> (BITS - 8)) as u8
    }

    fn pack_wide(self) -> Option<u16> {
        Some(self.0)
    }
}

fn pel<S: Sample>(value: S) -> i32 {
    value.into()
}

pub fn add_block<S: Sample>(
    plane: &mut [S],
    stride: usize,
    x0: usize,
    y0: usize,
    block: &[i32],
    n: usize,
    step: usize,
) {
    if n == 0 || block.iter().all(|&value| value == 0) {
        return;
    }
    for (y, residual) in block.chunks_exact(n).take(n).enumerate() {
        let at = (y0 + y * step) * stride + x0;
        for (sample, &value) in plane[at..at + n].iter_mut().zip(residual) {
            *sample = S::clip(pel(*sample) + value);
        }
    }
}

/// `above` is 8 samples, `left` is 4, `corner` is p[-1,-1]. `above_ok[4..]` may be copied from above[3].
pub fn pred4<S: Sample>(
    mode: u8,
    above: &[S; 8],
    left: &[S; 4],
    corner: S,
    above_ok: bool,
    left_ok: bool,
) -> [S; 16] {
    let mut out = [S::default(); 16];
    let p = |x: i32, y: i32| -> i32 {
        if y == -1 && x == -1 {
            pel(corner)
        } else if y == -1 {
            let index = x.clamp(0, above.len() as i32 - 1) as usize;
            pel(above[index])
        } else {
            let index = y.clamp(0, left.len() as i32 - 1) as usize;
            pel(left[index])
        }
    };
    for y in 0..4 {
        for x in 0..4 {
            let sample = match mode {
                0 if above_ok => p(x, -1),
                1 if left_ok => p(-1, y),
                2 => dc4(above, left, above_ok, left_ok),
                3 if above_ok => {
                    if x == 3 && y == 3 {
                        (p(6, -1) + 3 * p(7, -1) + 2) >> 2
                    } else {
                        (p(x + y, -1) + 2 * p(x + y + 1, -1) + p(x + y + 2, -1) + 2) >> 2
                    }
                }
                4 if above_ok && left_ok => {
                    if x > y {
                        (p(x - y - 2, -1) + 2 * p(x - y - 1, -1) + p(x - y, -1) + 2) >> 2
                    } else if x < y {
                        (p(-1, y - x - 2) + 2 * p(-1, y - x - 1) + p(-1, y - x) + 2) >> 2
                    } else {
                        (p(0, -1) + 2 * p(-1, -1) + p(-1, 0) + 2) >> 2
                    }
                }
                5 if above_ok && left_ok => {
                    let z = 2 * x - y;
                    if matches!(z, 0 | 2 | 4 | 6) {
                        (p(x - (y >> 1) - 1, -1) + p(x - (y >> 1), -1) + 1) >> 1
                    } else if matches!(z, 1 | 3 | 5) {
                        (p(x - (y >> 1) - 2, -1)
                            + 2 * p(x - (y >> 1) - 1, -1)
                            + p(x - (y >> 1), -1)
                            + 2)
                            >> 2
                    } else if z == -1 {
                        (p(-1, 0) + 2 * p(-1, -1) + p(0, -1) + 2) >> 2
                    } else {
                        (p(-1, y - 1) + 2 * p(-1, y - 2) + p(-1, y - 3) + 2) >> 2
                    }
                }
                6 if above_ok && left_ok => {
                    let z = 2 * y - x;
                    if matches!(z, 0 | 2 | 4 | 6) {
                        (p(-1, y - (x >> 1) - 1) + p(-1, y - (x >> 1)) + 1) >> 1
                    } else if matches!(z, 1 | 3 | 5) {
                        (p(-1, y - (x >> 1) - 2)
                            + 2 * p(-1, y - (x >> 1) - 1)
                            + p(-1, y - (x >> 1))
                            + 2)
                            >> 2
                    } else if z == -1 {
                        (p(-1, 0) + 2 * p(-1, -1) + p(0, -1) + 2) >> 2
                    } else {
                        (p(x - 1, -1) + 2 * p(x - 2, -1) + p(x - 3, -1) + 2) >> 2
                    }
                }
                7 if above_ok => {
                    if y == 0 || y == 2 {
                        (p(x + (y >> 1), -1) + p(x + (y >> 1) + 1, -1) + 1) >> 1
                    } else {
                        (p(x + (y >> 1), -1)
                            + 2 * p(x + (y >> 1) + 1, -1)
                            + p(x + (y >> 1) + 2, -1)
                            + 2)
                            >> 2
                    }
                }
                8 if left_ok => {
                    let z = x + 2 * y;
                    if matches!(z, 0 | 2 | 4) {
                        (p(-1, y + (x >> 1)) + p(-1, y + (x >> 1) + 1) + 1) >> 1
                    } else if matches!(z, 1 | 3) {
                        (p(-1, y + (x >> 1))
                            + 2 * p(-1, y + (x >> 1) + 1)
                            + p(-1, y + (x >> 1) + 2)
                            + 2)
                            >> 2
                    } else if z == 5 {
                        (p(-1, 2) + 3 * p(-1, 3) + 2) >> 2
                    } else {
                        p(-1, 3)
                    }
                }
                _ => dc4(above, left, above_ok, left_ok),
            };
            out[(y as usize) * 4 + x as usize] = S::clip(sample);
        }
    }
    out
}

fn dc4<S: Sample>(above: &[S; 8], left: &[S; 4], above_ok: bool, left_ok: bool) -> i32 {
    if above_ok && left_ok {
        (pel(above[0])
            + pel(above[1])
            + pel(above[2])
            + pel(above[3])
            + pel(left[0])
            + pel(left[1])
            + pel(left[2])
            + pel(left[3])
            + 4)
            >> 3
    } else if left_ok {
        (pel(left[0]) + pel(left[1]) + pel(left[2]) + pel(left[3]) + 2) >> 2
    } else if above_ok {
        (pel(above[0]) + pel(above[1]) + pel(above[2]) + pel(above[3]) + 2) >> 2
    } else {
        S::MID
    }
}

pub fn pred16<S: Sample>(
    mode: u8,
    above: &[S],
    left: &[S],
    above_ok: bool,
    left_ok: bool,
    corner: S,
) -> [S; 256] {
    let mut out = [S::mid(); 256];
    match mode {
        0 if above_ok => {
            for y in 0..16 {
                let Some(row) = above.get(..16) else {
                    return out;
                };
                out[y * 16..y * 16 + 16].copy_from_slice(row);
            }
        }
        1 if left_ok => {
            for y in 0..16 {
                out[y * 16..y * 16 + 16].fill(left.get(y).copied().unwrap_or(S::mid()));
            }
        }
        2 => {
            let value = if above_ok && left_ok {
                let sum = above.iter().take(16).map(|v| pel(*v)).sum::<i32>()
                    + left.iter().take(16).map(|v| pel(*v)).sum::<i32>();
                S::clip((sum + 16) >> 5)
            } else if left_ok {
                let sum = left.iter().take(16).map(|v| pel(*v)).sum::<i32>();
                S::clip((sum + 8) >> 4)
            } else if above_ok {
                let sum = above.iter().take(16).map(|v| pel(*v)).sum::<i32>();
                S::clip((sum + 8) >> 4)
            } else {
                S::mid()
            };
            out.fill(value);
        }
        3 if above_ok && left_ok => plane16(&mut out, above, left, corner),
        _ => {}
    }
    out
}

/// Equations 8-128 and 8-129. The last term on each side is the corner sample p[−1, −1].
fn plane_gradient<S: Sample>(samples: &[S], corner: S, half: usize) -> i32 {
    let mut sum = 0i32;
    let last = half - 1;
    for step in 0..half {
        let weight = i32::try_from(step + 1).unwrap_or(0);
        let right = pel(samples.get(half + step).copied().unwrap_or(S::default()));
        let left = if step == last {
            pel(corner)
        } else {
            pel(samples
                .get(half - 2 - step)
                .copied()
                .unwrap_or(S::default()))
        };
        sum += weight * (right - left);
    }
    sum
}

fn plane16<S: Sample>(out: &mut [S; 256], above: &[S], left: &[S], corner: S) {
    let h = plane_gradient(above, corner, 8);
    let v = plane_gradient(left, corner, 8);
    let a = 16 * (pel(above[15]) + pel(left[15]));
    let b = (5 * h + 32) >> 6;
    let c = (5 * v + 32) >> 6;
    for y in 0..16 {
        for x in 0..16 {
            out[y * 16 + x] = S::clip((a + b * (x as i32 - 7) + c * (y as i32 - 7) + 16) >> 5);
        }
    }
}

pub fn pred_chroma<S: Sample>(
    mode: u8,
    above: &[S],
    left: &[S],
    above_ok: bool,
    left_ok: bool,
    corner: S,
) -> [S; 64] {
    let mut out = [S::mid(); 64];
    match mode {
        1 if left_ok => {
            for y in 0..8 {
                out[y * 8..y * 8 + 8].fill(left[y]);
            }
        }
        2 if above_ok => {
            for y in 0..8 {
                out[y * 8..y * 8 + 8].copy_from_slice(&above[..8]);
            }
        }
        3 if above_ok && left_ok => {
            let h = plane_gradient(above, corner, 4);
            let v = plane_gradient(left, corner, 4);
            let a = 16 * (pel(above[7]) + pel(left[7]));
            let b = (17 * h + 16) >> 5;
            let c = (17 * v + 16) >> 5;
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] =
                        S::clip((a + b * (x as i32 - 3) + c * (y as i32 - 3) + 16) >> 5);
                }
            }
        }
        _ => {
            // DC, four 4x4 regions.
            let dc = |sum: i32, n: i32| S::clip((sum + (n / 2)) / n);
            let top_left = if above_ok && left_ok {
                dc(
                    pel(above[0])
                        + pel(above[1])
                        + pel(above[2])
                        + pel(above[3])
                        + pel(left[0])
                        + pel(left[1])
                        + pel(left[2])
                        + pel(left[3]),
                    8,
                )
            } else if above_ok {
                dc(
                    pel(above[0]) + pel(above[1]) + pel(above[2]) + pel(above[3]),
                    4,
                )
            } else if left_ok {
                dc(pel(left[0]) + pel(left[1]) + pel(left[2]) + pel(left[3]), 4)
            } else {
                S::mid()
            };
            let top_right = if above_ok {
                dc(
                    pel(above[4]) + pel(above[5]) + pel(above[6]) + pel(above[7]),
                    4,
                )
            } else {
                top_left
            };
            let bot_left = if left_ok {
                dc(pel(left[4]) + pel(left[5]) + pel(left[6]) + pel(left[7]), 4)
            } else {
                top_left
            };
            let bot_right = if above_ok && left_ok {
                dc(
                    pel(above[4])
                        + pel(above[5])
                        + pel(above[6])
                        + pel(above[7])
                        + pel(left[4])
                        + pel(left[5])
                        + pel(left[6])
                        + pel(left[7]),
                    8,
                )
            } else if above_ok {
                top_right
            } else if left_ok {
                bot_left
            } else {
                S::mid()
            };
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = match (x < 4, y < 4) {
                        (true, true) => top_left,
                        (false, true) => top_right,
                        (true, false) => bot_left,
                        (false, false) => bot_right,
                    };
                }
            }
        }
    }
    out
}

/// 8×16 chroma intra prediction for 4:2:2. Modes are DC, horizontal, vertical, and plane.
pub fn pred_chroma422<S: Sample>(
    mode: u8,
    above: &[S],
    left: &[S],
    above_ok: bool,
    left_ok: bool,
    corner: S,
) -> [S; 128] {
    let mut out = [S::mid(); 128];
    match mode {
        1 if left_ok => {
            for y in 0..16 {
                out[y * 8..y * 8 + 8].fill(left[y]);
            }
        }
        2 if above_ok => {
            for y in 0..16 {
                out[y * 8..y * 8 + 8].copy_from_slice(&above[..8]);
            }
        }
        3 if above_ok && left_ok => plane8x16(&mut out, above, left, corner),
        _ => dc8x16(&mut out, above, left, above_ok, left_ok),
    }
    out
}

fn dc8x16<S: Sample>(out: &mut [S; 128], above: &[S], left: &[S], above_ok: bool, left_ok: bool) {
    let group = |samples: &[S], at: usize| -> i32 {
        pel(samples[at]) + pel(samples[at + 1]) + pel(samples[at + 2]) + pel(samples[at + 3])
    };
    let avg = |sum: i32, n: i32| S::clip((sum + (n / 2)) / n);
    if above_ok && left_ok {
        let top = group(above, 4);
        let rows = [
            (avg(group(left, 0) + group(above, 0), 8), avg(top, 4)),
            (avg(group(left, 4), 4), avg(top + group(left, 4), 8)),
            (avg(group(left, 8), 4), avg(top + group(left, 8), 8)),
            (avg(group(left, 12), 4), avg(top + group(left, 12), 8)),
        ];
        for (band, (left_dc, right_dc)) in rows.iter().enumerate() {
            for y in 0..4 {
                let row = (band * 4 + y) * 8;
                out[row..row + 4].fill(*left_dc);
                out[row + 4..row + 8].fill(*right_dc);
            }
        }
        return;
    }
    if above_ok {
        let left_dc = avg(group(above, 0), 4);
        let right_dc = avg(group(above, 4), 4);
        for y in 0..16 {
            let row = y * 8;
            out[row..row + 4].fill(left_dc);
            out[row + 4..row + 8].fill(right_dc);
        }
        return;
    }
    if left_ok {
        for band in 0..4 {
            let value = avg(group(left, band * 4), 4);
            for y in 0..4 {
                out[(band * 4 + y) * 8..(band * 4 + y) * 8 + 8].fill(value);
            }
        }
    }
}

fn plane8x16<S: Sample>(out: &mut [S; 128], above: &[S], left: &[S], corner: S) {
    let h = (pel(above[4]) - pel(above[2]))
        + 2 * (pel(above[5]) - pel(above[1]))
        + 3 * (pel(above[6]) - pel(above[0]))
        + 4 * (pel(above[7]) - pel(corner));
    let mut v = pel(left[8]) - pel(left[6]);
    let down = [
        pel(left[9]),
        pel(left[10]),
        pel(left[11]),
        pel(left[12]),
        pel(left[13]),
        pel(left[14]),
        pel(left[15]),
    ];
    let up = [
        pel(left[5]),
        pel(left[4]),
        pel(left[3]),
        pel(left[2]),
        pel(left[1]),
        pel(left[0]),
        pel(corner),
    ];
    for (k, (low, high)) in up.into_iter().zip(down).enumerate() {
        v += (k as i32 + 2) * (high - low);
    }
    let h = (17 * h + 16) >> 5;
    let v = (5 * v + 32) >> 6;
    let mut a = 16 * (pel(left[15]) + pel(above[7]) + 1) - 7 * v - 3 * h;
    for y in 0..16 {
        let b = a;
        a += v;
        for x in 0..8 {
            out[y * 8 + x] = S::clip((b + x as i32 * h) >> 5);
        }
    }
}

pub fn sample<S: Sample>(plane: &[S], stride: usize, w: i32, h: i32, x: i32, y: i32) -> i32 {
    let x = x.clamp(0, w - 1) as usize;
    let y = y.clamp(0, h - 1) as usize;
    pel(plane[y * stride + x])
}

pub(crate) fn tap<S: Sample>(
    plane: &[S],
    stride: usize,
    w: i32,
    h: i32,
    x: i32,
    y: i32,
    horizontal: bool,
) -> i32 {
    let at = |d: i32| {
        if horizontal {
            sample(plane, stride, w, h, x + d, y)
        } else {
            sample(plane, stride, w, h, x, y + d)
        }
    };
    at(-2) - 5 * at(-1) + 20 * at(0) + 20 * at(1) - 5 * at(2) + at(3)
}

pub struct Pel {
    pub x: i32,
    pub y: i32,
    pub x_frac: i32,
    pub y_frac: i32,
}

/// Quarter-sample luma. `x` and `y` are the integer part, fractions are 0..3.
pub fn qpel<S: Sample>(plane: &[S], stride: usize, w: i32, h: i32, at: Pel) -> S {
    let Pel {
        x,
        y,
        x_frac,
        y_frac,
    } = at;
    let full = |dx: i32, dy: i32| sample(plane, stride, w, h, x + dx, y + dy);
    let half_h =
        |dx: i32, dy: i32| S::clip((tap(plane, stride, w, h, x + dx, y + dy, true) + 16) >> 5);
    let half_v =
        |dx: i32, dy: i32| S::clip((tap(plane, stride, w, h, x + dx, y + dy, false) + 16) >> 5);
    let half_h_raw = |dx: i32, dy: i32| tap(plane, stride, w, h, x + dx, y + dy, true);
    let diag = |dx: i32| {
        let mut acc = 0i32;
        let weights = [-2, -1, 0, 1, 2, 3];
        let coef = [1, -5, 20, 20, -5, 1];
        for (k, &dy) in weights.iter().enumerate() {
            acc += coef[k] * half_h_raw(dx, dy);
        }
        S::clip((acc + 512) >> 10)
    };
    let avg = |a: S, b: S| (pel(a) + pel(b) + 1) >> 1;
    let value = match (x_frac, y_frac) {
        (0, 0) => full(0, 0),
        (1, 0) => avg(S::clip(full(0, 0)), half_h(0, 0)),
        (2, 0) => pel(half_h(0, 0)),
        (3, 0) => avg(half_h(0, 0), S::clip(full(1, 0))),
        (0, 1) => avg(S::clip(full(0, 0)), half_v(0, 0)),
        (0, 2) => pel(half_v(0, 0)),
        (0, 3) => avg(half_v(0, 0), S::clip(full(0, 1))),
        (2, 2) => pel(diag(0)),
        (2, 1) => avg(half_h(0, 0), diag(0)),
        (1, 2) => avg(half_v(0, 0), diag(0)),
        (2, 3) => avg(diag(0), half_h(0, 1)),
        (3, 2) => avg(diag(0), half_v(1, 0)),
        (1, 1) => avg(half_h(0, 0), half_v(0, 0)),
        (3, 1) => avg(half_h(0, 0), half_v(1, 0)),
        (1, 3) => avg(half_v(0, 0), half_h(0, 1)),
        _ => avg(half_v(1, 0), half_h(0, 1)),
    };
    S::clip(value)
}

pub fn chroma_mc<S: Sample>(plane: &[S], stride: usize, w: i32, h: i32, at: Pel) -> S {
    let Pel {
        x,
        y,
        x_frac,
        y_frac,
    } = at;
    let a = sample(plane, stride, w, h, x, y);
    let b = sample(plane, stride, w, h, x + 1, y);
    let c = sample(plane, stride, w, h, x, y + 1);
    let d = sample(plane, stride, w, h, x + 1, y + 1);
    let xf = x_frac.clamp(0, 7);
    let yf = y_frac.clamp(0, 7);
    S::clip(
        ((8 - xf) * (8 - yf) * a + xf * (8 - yf) * b + (8 - xf) * yf * c + xf * yf * d + 32) >> 6,
    )
}

/// A rectangle whose motion vector is the same for every sample.
#[derive(Clone, Copy)]
pub struct BlockMc {
    pub x: i32,
    pub y: i32,
    pub w: usize,
    pub h: usize,
    pub x_frac: i32,
    pub y_frac: i32,
}

/// Luma motion compensation for one partition. Matches [`qpel`] sample for sample.
pub fn put_luma_block<S: Sample>(
    dst: &mut [S],
    dst_stride: usize,
    src: &[S],
    src_stride: usize,
    pic: (i32, i32),
    block: BlockMc,
) {
    let BlockMc {
        x,
        y,
        w,
        h,
        x_frac,
        y_frac,
    } = block;
    if w == 0 || h == 0 || w > 16 || h > 16 || dst.len() < (h - 1) * dst_stride + w {
        return;
    }
    let (pic_w, pic_h) = pic;
    if !luma_window_inside(x, y, w, h, pic_w, pic_h) {
        put_luma_extended(dst, dst_stride, src, src_stride, pic, block);
        return;
    }
    let size = (w, h);
    let at = |dx, dy| src_at(src, src_stride, x + dx, y + dy);
    match (x_frac, y_frac) {
        (0, 0) => copy_block(dst, dst_stride, at(0, 0), size),
        (2, 0) => half_h_block(dst, dst_stride, at(0, 0), size),
        (0, 2) => half_v_block(dst, dst_stride, at(0, 0), size),
        (2, 2) => diag_block(dst, dst_stride, at(0, 0), size),
        (1, 0) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            copy_block(a, w, at(0, 0), size);
            half_h_block(b, w, at(0, 0), size);
        }),
        (3, 0) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            half_h_block(a, w, at(0, 0), size);
            copy_block(b, w, at(1, 0), size);
        }),
        (0, 1) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            copy_block(a, w, at(0, 0), size);
            half_v_block(b, w, at(0, 0), size);
        }),
        (0, 3) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            half_v_block(a, w, at(0, 0), size);
            copy_block(b, w, at(0, 1), size);
        }),
        (2, 1) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            half_h_block(a, w, at(0, 0), size);
            diag_block(b, w, at(0, 0), size);
        }),
        (1, 2) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            half_v_block(a, w, at(0, 0), size);
            diag_block(b, w, at(0, 0), size);
        }),
        (2, 3) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            diag_block(a, w, at(0, 0), size);
            half_h_block(b, w, at(0, 1), size);
        }),
        (3, 2) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            diag_block(a, w, at(0, 0), size);
            half_v_block(b, w, at(1, 0), size);
        }),
        (1, 1) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            half_h_block(a, w, at(0, 0), size);
            half_v_block(b, w, at(0, 0), size);
        }),
        (3, 1) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            half_h_block(a, w, at(0, 0), size);
            half_v_block(b, w, at(1, 0), size);
        }),
        (1, 3) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            half_v_block(a, w, at(0, 0), size);
            half_h_block(b, w, at(0, 1), size);
        }),
        (3, 3) => avg_blocks(dst, dst_stride, w, h, |a, b| {
            half_v_block(a, w, at(1, 0), size);
            half_h_block(b, w, at(0, 1), size);
        }),
        _ => put_luma_scalar(dst, dst_stride, src, src_stride, pic, block),
    }
}

/// Chroma motion compensation for one partition. Matches [`chroma_mc`].
pub fn put_chroma_block<S: Sample>(
    dst: &mut [S],
    dst_stride: usize,
    src: &[S],
    src_stride: usize,
    pic: (i32, i32),
    block: BlockMc,
) {
    let BlockMc {
        x,
        y,
        w,
        h,
        x_frac,
        y_frac,
    } = block;
    if w == 0 || h == 0 || dst.len() < (h - 1) * dst_stride + w {
        return;
    }
    let (pic_w, pic_h) = pic;
    let xf = x_frac.clamp(0, 7);
    let yf = y_frac.clamp(0, 7);
    if x < 0
        || y < 0
        || x + i32::try_from(w).unwrap_or(i32::MAX) >= pic_w
        || y + i32::try_from(h).unwrap_or(i32::MAX) >= pic_h
    {
        put_chroma_extended(dst, dst_stride, src, src_stride, pic, block);
        return;
    }
    let x0 = usize::try_from(x).unwrap_or(0);
    let y0 = usize::try_from(y).unwrap_or(0);
    if xf == 0 && yf == 0 {
        copy_block(dst, dst_stride, src_at(src, src_stride, x, y), (w, h));
        return;
    }
    if yf == 0 {
        let wa = 8 - xf;
        for row in 0..h {
            let start = (y0 + row) * src_stride + x0;
            let span = &src[start..start + w + 1];
            let out = &mut dst[row * dst_stride..row * dst_stride + w];
            for (sample, pair) in out.iter_mut().zip(span.windows(2)) {
                let value = wa * pel(pair[0]) + xf * pel(pair[1]);
                *sample = S::clip((value + 4) >> 3);
            }
        }
        return;
    }
    if xf == 0 {
        let wa = 8 - yf;
        for row in 0..h {
            let start = (y0 + row) * src_stride + x0;
            let row0 = &src[start..start + w];
            let row1 = &src[start + src_stride..start + src_stride + w];
            let out = &mut dst[row * dst_stride..row * dst_stride + w];
            for (sample, (&left, &right)) in out.iter_mut().zip(row0.iter().zip(row1)) {
                let value = wa * pel(left) + yf * pel(right);
                *sample = S::clip((value + 4) >> 3);
            }
        }
        return;
    }
    let wa = (8 - xf) * (8 - yf);
    let wb = xf * (8 - yf);
    let wc = (8 - xf) * yf;
    let wd = xf * yf;
    for row in 0..h {
        let start = (y0 + row) * src_stride + x0;
        let row0 = &src[start..start + w + 1];
        let row1 = &src[start + src_stride..start + src_stride + w + 1];
        let out = &mut dst[row * dst_stride..row * dst_stride + w];
        for (sample, (left, right)) in out.iter_mut().zip(row0.windows(2).zip(row1.windows(2))) {
            let value = wa * pel(left[0])
                + wb * pel(left[1])
                + wc * pel(right[0])
                + wd * pel(right[1])
                + 32;
            *sample = S::clip(value >> 6);
        }
    }
}

fn luma_window_inside(x: i32, y: i32, w: usize, h: usize, pic_w: i32, pic_h: i32) -> bool {
    let bw = i32::try_from(w).unwrap_or(i32::MAX);
    let bh = i32::try_from(h).unwrap_or(i32::MAX);
    x >= 2 && y >= 2 && x + bw + 3 <= pic_w && y + bh + 3 <= pic_h
}

fn put_chroma_extended<S: Sample>(
    dst: &mut [S],
    dst_stride: usize,
    src: &[S],
    src_stride: usize,
    pic: (i32, i32),
    block: BlockMc,
) {
    let (pic_w, pic_h) = pic;
    let width = block.w + 1;
    let height = block.h + 1;
    let mut window = [S::default(); 17 * 17];
    for row in 0..height {
        for col in 0..width {
            let sx = block.x + i32::try_from(col).unwrap_or(0);
            let sy = block.y + i32::try_from(row).unwrap_or(0);
            window[row * width + col] = S::clip(sample(src, src_stride, pic_w, pic_h, sx, sy));
        }
    }
    let wide = i32::try_from(width).unwrap_or(0);
    let high = i32::try_from(height).unwrap_or(0);
    put_chroma_block(
        dst,
        dst_stride,
        &window,
        width,
        (wide, high),
        BlockMc {
            x: 0,
            y: 0,
            w: block.w,
            h: block.h,
            x_frac: block.x_frac,
            y_frac: block.y_frac,
        },
    );
}

fn put_luma_extended<S: Sample>(
    dst: &mut [S],
    dst_stride: usize,
    src: &[S],
    src_stride: usize,
    pic: (i32, i32),
    block: BlockMc,
) {
    let (pic_w, pic_h) = pic;
    let width = block.w + 5;
    let height = block.h + 5;
    let mut window = [S::default(); 21 * 21];
    for row in 0..height {
        for col in 0..width {
            let sx = block.x - 2 + i32::try_from(col).unwrap_or(0);
            let sy = block.y - 2 + i32::try_from(row).unwrap_or(0);
            window[row * width + col] = S::clip(sample(src, src_stride, pic_w, pic_h, sx, sy));
        }
    }
    let wide = i32::try_from(width).unwrap_or(0);
    let high = i32::try_from(height).unwrap_or(0);
    put_luma_block(
        dst,
        dst_stride,
        &window,
        width,
        (wide, high),
        BlockMc {
            x: 2,
            y: 2,
            w: block.w,
            h: block.h,
            x_frac: block.x_frac,
            y_frac: block.y_frac,
        },
    );
}

fn put_luma_scalar<S: Sample>(
    dst: &mut [S],
    dst_stride: usize,
    src: &[S],
    src_stride: usize,
    pic: (i32, i32),
    block: BlockMc,
) {
    let (pic_w, pic_h) = pic;
    for row in 0..block.h {
        for col in 0..block.w {
            dst[row * dst_stride + col] = qpel(
                src,
                src_stride,
                pic_w,
                pic_h,
                Pel {
                    x: block.x + i32::try_from(col).unwrap_or(0),
                    y: block.y + i32::try_from(row).unwrap_or(0),
                    x_frac: block.x_frac,
                    y_frac: block.y_frac,
                },
            );
        }
    }
}

fn avg_blocks<S: Sample>(
    dst: &mut [S],
    dst_stride: usize,
    w: usize,
    h: usize,
    fill: impl FnOnce(&mut [S], &mut [S]),
) {
    let mut left = [S::default(); 256];
    let mut right = [S::default(); 256];
    fill(&mut left, &mut right);
    for row in 0..h {
        let out = &mut dst[row * dst_stride..row * dst_stride + w];
        let a = &left[row * w..row * w + w];
        let b = &right[row * w..row * w + w];
        for (sample, (&left_px, &right_px)) in out.iter_mut().zip(a.iter().zip(b)) {
            *sample = avg(left_px, right_px);
        }
    }
}

struct Src<'a, S: Sample> {
    samples: &'a [S],
    stride: usize,
    x: usize,
    y: usize,
}

fn src_at<S: Sample>(samples: &[S], stride: usize, x: i32, y: i32) -> Src<'_, S> {
    Src {
        samples,
        stride,
        x: usize::try_from(x).unwrap_or(0),
        y: usize::try_from(y).unwrap_or(0),
    }
}

fn copy_block<S: Sample>(dst: &mut [S], dst_stride: usize, src: Src<'_, S>, size: (usize, usize)) {
    let (w, h) = size;
    for row in 0..h {
        let from = (src.y + row) * src.stride + src.x;
        dst[row * dst_stride..row * dst_stride + w].copy_from_slice(&src.samples[from..from + w]);
    }
}

fn half_h_block<S: Sample>(
    dst: &mut [S],
    dst_stride: usize,
    src: Src<'_, S>,
    size: (usize, usize),
) {
    let (w, h) = size;
    for row in 0..h {
        let start = (src.y + row) * src.stride + src.x - 2;
        let taps = &src.samples[start..start + w + 5];
        let out = &mut dst[row * dst_stride..row * dst_stride + w];
        for (col, sample) in out.iter_mut().enumerate() {
            *sample = S::clip((fir6_at(taps, col) + 16) >> 5);
        }
    }
}

fn half_v_block<S: Sample>(
    dst: &mut [S],
    dst_stride: usize,
    src: Src<'_, S>,
    size: (usize, usize),
) {
    let (w, h) = size;
    for row in 0..h {
        let mut rows = [&[][..]; 6];
        for (k, samples) in rows.iter_mut().enumerate() {
            let sy = src.y + row + k - 2;
            let start = sy * src.stride + src.x;
            *samples = &src.samples[start..start + w];
        }
        let out = &mut dst[row * dst_stride..row * dst_stride + w];
        for (col, sample) in out.iter_mut().enumerate() {
            let acc = pel(rows[0][col]) - 5 * pel(rows[1][col])
                + 20 * pel(rows[2][col])
                + 20 * pel(rows[3][col])
                - 5 * pel(rows[4][col])
                + pel(rows[5][col]);
            *sample = S::clip((acc + 16) >> 5);
        }
    }
}

fn diag_block<S: Sample>(dst: &mut [S], dst_stride: usize, src: Src<'_, S>, size: (usize, usize)) {
    let (w, h) = size;
    let mut tmp = [0i32; 21 * 16];
    for vy in 0..h + 5 {
        let start = (src.y + vy - 2) * src.stride + src.x - 2;
        let taps = &src.samples[start..start + w + 5];
        for col in 0..w {
            tmp[vy * 16 + col] = fir6_at(taps, col);
        }
    }
    for row in 0..h {
        let out = &mut dst[row * dst_stride..row * dst_stride + w];
        for (col, sample) in out.iter_mut().enumerate() {
            let acc = tmp[row * 16 + col] - 5 * tmp[(row + 1) * 16 + col]
                + 20 * tmp[(row + 2) * 16 + col]
                + 20 * tmp[(row + 3) * 16 + col]
                - 5 * tmp[(row + 4) * 16 + col]
                + tmp[(row + 5) * 16 + col];
            *sample = S::clip((acc + 512) >> 10);
        }
    }
}

fn fir6_at<S: Sample>(taps: &[S], col: usize) -> i32 {
    pel(taps[col]) - 5 * pel(taps[col + 1]) + 20 * pel(taps[col + 2]) + 20 * pel(taps[col + 3])
        - 5 * pel(taps[col + 4])
        + pel(taps[col + 5])
}

pub fn weight_uni<S: Sample>(sample: S, w: i32, o: i32, denom: i32) -> S {
    let v = if denom >= 1 {
        ((pel(sample) * w + (1 << (denom - 1))) >> denom) + o
    } else {
        pel(sample) * w + o
    };
    S::clip(v)
}

#[derive(Clone, Copy)]
pub struct BiWeight {
    pub w0: i32,
    pub w1: i32,
    pub o0: i32,
    pub o1: i32,
    pub denom: i32,
}

#[inline(always)]
pub fn weight_bi<S: Sample>(a: S, b: S, weight: BiWeight) -> S {
    let BiWeight {
        w0,
        w1,
        o0,
        o1,
        denom,
    } = weight;
    let shift = denom + 1;
    let value = ((pel(a) * w0 + pel(b) * w1 + (1 << (shift - 1))) >> shift) + ((o0 + o1 + 1) >> 1);
    S::clip(value)
}

/// One row of bipred weights. The weights stay outside the pixel loop so the
/// compiler can widen and multiply several samples at once.
pub fn weight_row<S: Sample>(dst: &mut [S], a: &[S], b: &[S], weight: BiWeight) {
    let n = dst.len();
    if a.len() < n || b.len() < n {
        return;
    }
    let a = &a[..n];
    let b = &b[..n];
    let shift = weight.denom + 1;
    if !(1..31).contains(&shift) {
        for i in 0..n {
            dst[i] = weight_bi(a[i], b[i], weight);
        }
        return;
    }
    let round = 1i32 << (shift - 1);
    let off = (weight.o0 + weight.o1 + 1) >> 1;
    let w0 = weight.w0;
    let w1 = weight.w1;
    for (sample, (&left, &right)) in dst.iter_mut().zip(a.iter().zip(b)) {
        let value = ((pel(left) * w0 + pel(right) * w1 + round) >> shift) + off;
        *sample = S::clip(value);
    }
}

#[inline(always)]
pub fn avg<S: Sample>(a: S, b: S) -> S {
    S::clip((pel(a) + pel(b) + 1) >> 1)
}

#[derive(Clone, Copy)]
pub struct Edge {
    pub vertical: bool,
    pub bs: u8,
    pub qpav: i32,
    pub offset_a: i32,
    pub offset_b: i32,
    pub chroma: bool,
    pub len: usize,
    /// Frame rows between samples. Field macroblocks step by two.
    pub row_step: usize,
}

pub fn filter_edge<S: Sample>(plane: &mut [S], stride: usize, x: usize, y: usize, edge: Edge) {
    let Edge {
        vertical,
        bs,
        qpav,
        offset_a,
        offset_b,
        chroma,
        len,
        row_step,
    } = edge;
    if bs == 0 || len == 0 {
        return;
    }
    let index_a = (qpav + offset_a).clamp(0, 51) as usize;
    let index_b = (qpav + offset_b).clamp(0, 51) as usize;
    let run = FilterRun {
        len,
        bs,
        alpha: i32::from(ALPHA[index_a]) << S::SHIFT,
        beta: i32::from(BETA[index_b]) << S::SHIFT,
        index_a,
        chroma,
    };
    let at = y * stride + x;
    if vertical {
        filter_span::<S, true>(plane, stride, row_step, at, run);
    } else {
        filter_span::<S, false>(plane, stride, row_step, at, run);
    }
}

struct FilterRun {
    len: usize,
    bs: u8,
    alpha: i32,
    beta: i32,
    index_a: usize,
    chroma: bool,
}

fn filter_span<S: Sample, const VERTICAL: bool>(
    plane: &mut [S],
    stride: usize,
    row_step: usize,
    at: usize,
    run: FilterRun,
) {
    let len = run.len;
    let vert = stride * row_step;
    let tc0 = if run.bs < 4 {
        i32::from(TC0[usize::from(run.bs) - 1][run.index_a]) << S::SHIFT
    } else {
        0
    };
    if VERTICAL {
        // The edge runs down the picture: each line across it is eight
        // samples in a row.
        let mut at = at;
        for _ in 0..len {
            let Some(line) = plane.get_mut(at - 4..at + 4) else {
                return;
            };
            if let Ok(line) = <&mut [S; 8]>::try_from(line) {
                filter_line(line, &run, tc0);
            }
            at += vert;
        }
        return;
    }
    // The edge runs along a row: the eight rows around it, four above and
    // four below, each `len` long.
    let first = at - 4 * vert;
    let Some(window) = plane.get_mut(first..first + 7 * vert + len) else {
        return;
    };
    let (r0, rest) = window.split_at_mut(vert);
    let (r1, rest) = rest.split_at_mut(vert);
    let (r2, rest) = rest.split_at_mut(vert);
    let (r3, rest) = rest.split_at_mut(vert);
    let (r4, rest) = rest.split_at_mut(vert);
    let (r5, rest) = rest.split_at_mut(vert);
    let (r6, r7) = rest.split_at_mut(vert);
    let (r0, r1, r2, r3) = (&r0[..len], &mut r1[..len], &mut r2[..len], &mut r3[..len]);
    let (r4, r5, r6, r7) = (&mut r4[..len], &mut r5[..len], &mut r6[..len], &r7[..len]);
    for i in 0..len {
        let mut line = [r0[i], r1[i], r2[i], r3[i], r4[i], r5[i], r6[i], r7[i]];
        if filter_line(&mut line, &run, tc0) {
            r1[i] = line[1];
            r2[i] = line[2];
            r3[i] = line[3];
            r4[i] = line[4];
            r5[i] = line[5];
            r6[i] = line[6];
        }
    }
}

/// Clause 8.7.2.3 and 8.7.2.4 on one line of samples across an edge:
/// p3, p2, p1, p0, then q0, q1, q2, q3. Says whether the line was filtered.
/// `tc0` is the table value for a strength below 4.
#[inline(always)]
fn filter_line<S: Sample>(line: &mut [S; 8], run: &FilterRun, tc0: i32) -> bool {
    let (alpha, beta, chroma) = (run.alpha, run.beta, run.chroma);
    let p1 = pel(line[2]);
    let p0 = pel(line[3]);
    let q0 = pel(line[4]);
    let q1 = pel(line[5]);
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return false;
    }
    let p3 = pel(line[0]);
    let p2 = pel(line[1]);
    let q2 = pel(line[6]);
    let q3 = pel(line[7]);
    if run.bs < 4 {
        let ap = (p2 - p0).abs();
        let aq = (q2 - q0).abs();
        let tc = if chroma {
            tc0 + 1
        } else {
            tc0 + i32::from(ap < beta) + i32::from(aq < beta)
        };
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        line[3] = S::clip(p0 + delta);
        line[4] = S::clip(q0 - delta);
        if !chroma && ap < beta {
            let d = ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0);
            line[2] = S::clip(p1 + d);
        }
        if !chroma && aq < beta {
            let d = ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0);
            line[5] = S::clip(q1 + d);
        }
    } else {
        let strong_p = !chroma && (p2 - p0).abs() < beta && (p0 - q0).abs() < ((alpha >> 2) + 2);
        let strong_q = !chroma && (q2 - q0).abs() < beta && (p0 - q0).abs() < ((alpha >> 2) + 2);
        if strong_p {
            line[3] = S::clip((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            line[2] = S::clip((p2 + p1 + p0 + q0 + 2) >> 2);
            line[1] = S::clip((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
        } else {
            line[3] = S::clip((2 * p1 + p0 + q1 + 2) >> 2);
        }
        if strong_q {
            line[4] = S::clip((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            line[5] = S::clip((p0 + q0 + q1 + q2 + 2) >> 2);
            line[6] = S::clip((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3);
        } else {
            line[4] = S::clip((2 * q1 + q0 + p1 + 2) >> 2);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{
        BlockMc, Depth, Pel, Sample, chroma_mc, chroma_qps, pel, pred_chroma, pred16,
        put_chroma_block, put_luma_block, qpel,
    };

    #[test]
    fn ten_bit_chroma_qp_keeps_the_low_indices() {
        assert_eq!(chroma_qps(20, 0, 0), (chroma_qp_at(20), chroma_qp_at(20)));
        assert_eq!(chroma_qps(-6, 0, 12), (6, -6));
        assert_eq!(chroma_qps(26, 0, 12), (38, 26));
        assert_eq!(chroma_qps(40, 0, 12), (48, 36));
        assert_eq!(chroma_qps(51, 0, 12), (51, 39));
        // QpBdOffsetY is 6, 18, 24, 30, and 36 at 9, 11, 12, 13, and 14 bits.
        assert_eq!(chroma_qps(26, 0, 6), (32, 26));
        assert_eq!(chroma_qps(-6, 0, 6), (0, -6));
        assert_eq!(chroma_qps(26, 0, 18), (44, 26));
        assert_eq!(chroma_qps(26, 0, 24), (50, 26));
        assert_eq!(chroma_qps(26, 0, 30), (56, 26));
        assert_eq!(chroma_qps(26, 0, 36), (62, 26));
        assert_eq!(chroma_qps(51, 0, 36), (75, 39));
    }

    #[test]
    fn a_dc_only_block_takes_the_same_values_as_the_full_transform() {
        for dc in [
            -4000, -65, -64, -33, -32, -31, -1, 0, 1, 31, 32, 33, 64, 100, 4000,
        ] {
            let mut four = [0i32; 16];
            four[0] = dc;
            let mut full = four;
            super::idct4(&mut four);
            super::idct4_full(&mut full);
            assert_eq!(four, full, "4x4 dc {dc}");
            let mut eight = [0i32; 64];
            eight[0] = dc;
            let mut full = eight;
            super::idct8(&mut eight);
            super::idct8_full(&mut full);
            assert_eq!(eight, full, "8x8 dc {dc}");
        }
    }

    #[test]
    fn each_depth_clips_at_its_own_limit() {
        assert_eq!(pel(Depth::<9>::clip(10_000)), 511);
        assert_eq!(pel(Depth::<9>::mid()), 256);
        assert_eq!(Depth::<9>::QP_BD, 6);
        assert_eq!(pel(Depth::<11>::clip(10_000)), 2047);
        assert_eq!(pel(Depth::<11>::mid()), 1024);
        assert_eq!(Depth::<11>::QP_BD, 18);
        assert_eq!(pel(Depth::<12>::clip(10_000)), 4095);
        assert_eq!(pel(Depth::<12>::mid()), 2048);
        assert_eq!(pel(Depth::<13>::clip(10_000)), 8191);
        assert_eq!(pel(Depth::<13>::mid()), 4096);
        assert_eq!(pel(Depth::<14>::clip(20_000)), 16383);
        assert_eq!(pel(Depth::<14>::clip(10_000)), 10_000);
        assert_eq!(pel(Depth::<14>::mid()), 8192);
        assert_eq!(Depth::<14>::QP_BD, 36);
        assert_eq!(Depth::<10>::from_code(1000).pack_eight(), 250);
        assert_eq!(Depth::<9>::from_code(600).pack_wide(), Some(88));
    }

    fn chroma_qp_at(luma: i32) -> i32 {
        super::chroma_qp(luma, 0)
    }

    /// Equations 8-124 through 8-129. The corner sample is p[−1, −1], not p[0, −1].
    #[test]
    fn intra16_plane_uses_the_corner_sample() {
        let above: Vec<u8> = (0..16).collect();
        let left = [0u8; 16];
        let pred = pred16(3, &above, &left, true, true, 100);
        assert_eq!(pred[0], 28);
        assert_eq!(pred[7 * 16 + 7], 8);
        let chroma_above: Vec<u8> = (0..8).collect();
        let chroma_left = [0u8; 8];
        let chroma = pred_chroma(3, &chroma_above, &chroma_left, true, true, 100);
        assert_eq!(chroma[0], 41);
    }

    #[test]
    fn plane_uses_the_corner_in_the_outer_term() {
        let above: Vec<u8> = (1..=16).collect();
        let left = above.clone();
        let luma = pred16(3, &above, &left, true, true, 0);
        assert_eq!(luma[0], 2);
        assert_eq!(luma[7 * 16 + 7], 16);

        let chroma_above = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let chroma_left = [8u8, 16, 24, 32, 40, 48, 56, 64];
        let chroma = pred_chroma(3, &chroma_above, &chroma_left, true, true, 0);
        assert_eq!(chroma[0], 18);
        assert_eq!(chroma[3 * 8 + 3], 72);
        assert_eq!(chroma[7 * 8 + 7], 144);
    }

    #[test]
    fn block_motion_matches_the_sample_filter() {
        let width = 48i32;
        let height = 40i32;
        let mut plane = vec![0u8; (width * height) as usize];
        for (index, sample) in plane.iter_mut().enumerate() {
            *sample = u8::try_from((index * 17 + index / 48) % 256).unwrap_or(0);
        }
        for frac_x in 0..4 {
            for frac_y in 0..4 {
                for (x, y, w, h) in [(0, 0, 8, 8), (2, 3, 4, 8), (8, 8, 16, 16), (30, 22, 8, 4)] {
                    let mut got = vec![0u8; w * h];
                    put_luma_block(
                        &mut got,
                        w,
                        &plane,
                        width as usize,
                        (width, height),
                        BlockMc {
                            x,
                            y,
                            w,
                            h,
                            x_frac: frac_x,
                            y_frac: frac_y,
                        },
                    );
                    for row in 0..h {
                        for col in 0..w {
                            let want = qpel(
                                &plane,
                                width as usize,
                                width,
                                height,
                                Pel {
                                    x: x + i32::try_from(col).unwrap_or(0),
                                    y: y + i32::try_from(row).unwrap_or(0),
                                    x_frac: frac_x,
                                    y_frac: frac_y,
                                },
                            );
                            assert_eq!(got[row * w + col], want, "{frac_x},{frac_y} at {x},{y}");
                        }
                    }
                }
            }
        }
        let cw = width / 2;
        let ch = height / 2;
        for frac_x in 0..8 {
            for frac_y in 0..8 {
                let mut got = vec![0u8; 4 * 4];
                put_chroma_block(
                    &mut got,
                    4,
                    &plane,
                    cw as usize,
                    (cw, ch),
                    BlockMc {
                        x: 3,
                        y: 2,
                        w: 4,
                        h: 4,
                        x_frac: frac_x,
                        y_frac: frac_y,
                    },
                );
                for row in 0..4usize {
                    for col in 0..4usize {
                        let want = chroma_mc(
                            &plane,
                            cw as usize,
                            cw,
                            ch,
                            Pel {
                                x: 3 + i32::try_from(col).unwrap_or(0),
                                y: 2 + i32::try_from(row).unwrap_or(0),
                                x_frac: frac_x,
                                y_frac: frac_y,
                            },
                        );
                        assert_eq!(got[row * 4 + col], want);
                    }
                }
            }
        }
    }
}
