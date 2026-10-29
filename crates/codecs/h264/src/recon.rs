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

pub fn idct8(block: &mut [i32; 64]) {
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

pub fn clip_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

pub fn add_block(plane: &mut [u8], stride: usize, x0: usize, y0: usize, block: &[i32], n: usize) {
    for y in 0..n {
        for x in 0..n {
            let sample = i32::from(plane[(y0 + y) * stride + x0 + x]) + block[y * n + x];
            plane[(y0 + y) * stride + x0 + x] = clip_u8(sample);
        }
    }
}

/// `above` is 8 samples, `left` is 4, `corner` is p[-1,-1]. `above_ok[4..]` may be copied from above[3].
pub fn pred4(
    mode: u8,
    above: &[u8; 8],
    left: &[u8; 4],
    corner: u8,
    above_ok: bool,
    left_ok: bool,
) -> [u8; 16] {
    let mut out = [0u8; 16];
    let p = |x: i32, y: i32| -> i32 {
        if y == -1 && x == -1 {
            i32::from(corner)
        } else if y == -1 {
            let index = x.clamp(0, above.len() as i32 - 1) as usize;
            i32::from(above[index])
        } else {
            let index = y.clamp(0, left.len() as i32 - 1) as usize;
            i32::from(left[index])
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
            out[(y as usize) * 4 + x as usize] = clip_u8(sample);
        }
    }
    out
}

fn dc4(above: &[u8; 8], left: &[u8; 4], above_ok: bool, left_ok: bool) -> i32 {
    if above_ok && left_ok {
        (i32::from(above[0])
            + i32::from(above[1])
            + i32::from(above[2])
            + i32::from(above[3])
            + i32::from(left[0])
            + i32::from(left[1])
            + i32::from(left[2])
            + i32::from(left[3])
            + 4)
            >> 3
    } else if left_ok {
        (i32::from(left[0]) + i32::from(left[1]) + i32::from(left[2]) + i32::from(left[3]) + 2) >> 2
    } else if above_ok {
        (i32::from(above[0]) + i32::from(above[1]) + i32::from(above[2]) + i32::from(above[3]) + 2)
            >> 2
    } else {
        128
    }
}

pub fn pred16(mode: u8, above: &[u8], left: &[u8], above_ok: bool, left_ok: bool) -> [u8; 256] {
    let mut out = [128u8; 256];
    match mode {
        0 if above_ok => {
            for y in 0..16 {
                out[y * 16..y * 16 + 16].copy_from_slice(&above[..16]);
            }
        }
        1 if left_ok => {
            for y in 0..16 {
                out[y * 16..y * 16 + 16].fill(left[y]);
            }
        }
        2 => {
            let value = if above_ok && left_ok {
                let sum = above[..16].iter().map(|v| i32::from(*v)).sum::<i32>()
                    + left[..16].iter().map(|v| i32::from(*v)).sum::<i32>();
                ((sum + 16) >> 5) as u8
            } else if left_ok {
                let sum = left[..16].iter().map(|v| i32::from(*v)).sum::<i32>();
                ((sum + 8) >> 4) as u8
            } else if above_ok {
                let sum = above[..16].iter().map(|v| i32::from(*v)).sum::<i32>();
                ((sum + 8) >> 4) as u8
            } else {
                128
            };
            out.fill(value);
        }
        3 if above_ok && left_ok => plane16(&mut out, above, left),
        _ => {}
    }
    out
}

fn plane16(out: &mut [u8; 256], above: &[u8], left: &[u8]) {
    let mut h = 0i32;
    let mut v = 0i32;
    for i in 1..=8 {
        let above_r = i32::from(
            *above
                .get((8 + i) as usize)
                .unwrap_or(&above[above.len() - 1]),
        );
        let above_l = i32::from(*above.get((8 - i) as usize).unwrap_or(&above[0]));
        let left_b = i32::from(*left.get((8 + i) as usize).unwrap_or(&left[left.len() - 1]));
        let left_t = i32::from(*left.get((8 - i) as usize).unwrap_or(&left[0]));
        h += i * (above_r - above_l);
        v += i * (left_b - left_t);
    }
    let a = 16 * (i32::from(above[15]) + i32::from(left[15]));
    let b = (5 * h + 32) >> 6;
    let c = (5 * v + 32) >> 6;
    for y in 0..16 {
        for x in 0..16 {
            out[y * 16 + x] = clip_u8((a + b * (x as i32 - 7) + c * (y as i32 - 7) + 16) >> 5);
        }
    }
}

pub fn pred_chroma(mode: u8, above: &[u8], left: &[u8], above_ok: bool, left_ok: bool) -> [u8; 64] {
    let mut out = [128u8; 64];
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
            let mut h = 0i32;
            let mut v = 0i32;
            for i in 1..=4 {
                let above_r = i32::from(
                    *above
                        .get((4 + i) as usize)
                        .unwrap_or(&above[above.len() - 1]),
                );
                let above_l = i32::from(*above.get((4 - i) as usize).unwrap_or(&above[0]));
                let left_b =
                    i32::from(*left.get((4 + i) as usize).unwrap_or(&left[left.len() - 1]));
                let left_t = i32::from(*left.get((4 - i) as usize).unwrap_or(&left[0]));
                h += i * (above_r - above_l);
                v += i * (left_b - left_t);
            }
            let a = 16 * (i32::from(above[7]) + i32::from(left[7]));
            let b = (17 * h + 16) >> 5;
            let c = (17 * v + 16) >> 5;
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] =
                        clip_u8((a + b * (x as i32 - 3) + c * (y as i32 - 3) + 16) >> 5);
                }
            }
        }
        _ => {
            // DC, four 4x4 regions.
            let dc = |sum: i32, n: i32| ((sum + (n / 2)) / n) as u8;
            let top_left = if above_ok && left_ok {
                dc(
                    i32::from(above[0])
                        + i32::from(above[1])
                        + i32::from(above[2])
                        + i32::from(above[3])
                        + i32::from(left[0])
                        + i32::from(left[1])
                        + i32::from(left[2])
                        + i32::from(left[3]),
                    8,
                )
            } else if above_ok {
                dc(
                    i32::from(above[0])
                        + i32::from(above[1])
                        + i32::from(above[2])
                        + i32::from(above[3]),
                    4,
                )
            } else if left_ok {
                dc(
                    i32::from(left[0])
                        + i32::from(left[1])
                        + i32::from(left[2])
                        + i32::from(left[3]),
                    4,
                )
            } else {
                128
            };
            let top_right = if above_ok {
                dc(
                    i32::from(above[4])
                        + i32::from(above[5])
                        + i32::from(above[6])
                        + i32::from(above[7]),
                    4,
                )
            } else {
                top_left
            };
            let bot_left = if left_ok {
                dc(
                    i32::from(left[4])
                        + i32::from(left[5])
                        + i32::from(left[6])
                        + i32::from(left[7]),
                    4,
                )
            } else {
                top_left
            };
            let bot_right = if above_ok && left_ok {
                dc(
                    i32::from(above[4])
                        + i32::from(above[5])
                        + i32::from(above[6])
                        + i32::from(above[7])
                        + i32::from(left[4])
                        + i32::from(left[5])
                        + i32::from(left[6])
                        + i32::from(left[7]),
                    8,
                )
            } else if above_ok {
                top_right
            } else if left_ok {
                bot_left
            } else {
                128
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

pub fn sample(plane: &[u8], stride: usize, w: i32, h: i32, x: i32, y: i32) -> i32 {
    let x = x.clamp(0, w - 1) as usize;
    let y = y.clamp(0, h - 1) as usize;
    i32::from(plane[y * stride + x])
}

fn tap(plane: &[u8], stride: usize, w: i32, h: i32, x: i32, y: i32, horizontal: bool) -> i32 {
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
pub fn qpel(plane: &[u8], stride: usize, w: i32, h: i32, at: Pel) -> u8 {
    let Pel {
        x,
        y,
        x_frac,
        y_frac,
    } = at;
    let full = |dx: i32, dy: i32| sample(plane, stride, w, h, x + dx, y + dy);
    let half_h =
        |dx: i32, dy: i32| clip_u8((tap(plane, stride, w, h, x + dx, y + dy, true) + 16) >> 5);
    let half_v =
        |dx: i32, dy: i32| clip_u8((tap(plane, stride, w, h, x + dx, y + dy, false) + 16) >> 5);
    let half_h_raw = |dx: i32, dy: i32| tap(plane, stride, w, h, x + dx, y + dy, true);
    let diag = |dx: i32| {
        let mut acc = 0i32;
        let weights = [-2, -1, 0, 1, 2, 3];
        let coef = [1, -5, 20, 20, -5, 1];
        for (k, &dy) in weights.iter().enumerate() {
            acc += coef[k] * half_h_raw(dx, dy);
        }
        clip_u8((acc + 512) >> 10)
    };
    let avg = |a: u8, b: u8| (i32::from(a) + i32::from(b) + 1) >> 1;
    let value = match (x_frac, y_frac) {
        (0, 0) => full(0, 0),
        (1, 0) => avg(clip_u8(full(0, 0)), half_h(0, 0)),
        (2, 0) => i32::from(half_h(0, 0)),
        (3, 0) => avg(half_h(0, 0), clip_u8(full(1, 0))),
        (0, 1) => avg(clip_u8(full(0, 0)), half_v(0, 0)),
        (0, 2) => i32::from(half_v(0, 0)),
        (0, 3) => avg(half_v(0, 0), clip_u8(full(0, 1))),
        (2, 2) => i32::from(diag(0)),
        (2, 1) => avg(half_h(0, 0), diag(0)),
        (1, 2) => avg(half_v(0, 0), diag(0)),
        (2, 3) => avg(diag(0), half_h(0, 1)),
        (3, 2) => avg(diag(0), half_v(1, 0)),
        (1, 1) => avg(half_h(0, 0), half_v(0, 0)),
        (3, 1) => avg(half_h(0, 0), half_v(1, 0)),
        (1, 3) => avg(half_v(0, 0), half_h(0, 1)),
        _ => avg(half_v(1, 0), half_h(0, 1)),
    };
    clip_u8(value)
}

pub fn chroma_mc(plane: &[u8], stride: usize, w: i32, h: i32, at: Pel) -> u8 {
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
    clip_u8(
        ((8 - xf) * (8 - yf) * a + xf * (8 - yf) * b + (8 - xf) * yf * c + xf * yf * d + 32) >> 6,
    )
}

pub fn weight_uni(sample: u8, w: i32, o: i32, denom: i32) -> u8 {
    let v = if denom >= 1 {
        ((i32::from(sample) * w + (1 << (denom - 1))) >> denom) + o
    } else {
        i32::from(sample) * w + o
    };
    clip_u8(v)
}

pub struct BiWeight {
    pub w0: i32,
    pub w1: i32,
    pub o0: i32,
    pub o1: i32,
    pub denom: i32,
}

pub fn weight_bi(a: u8, b: u8, weight: BiWeight) -> u8 {
    let BiWeight {
        w0,
        w1,
        o0,
        o1,
        denom,
    } = weight;
    let shift = denom + 1;
    let value = ((i32::from(a) * w0 + i32::from(b) * w1 + (1 << (shift - 1))) >> shift)
        + ((o0 + o1 + 1) >> 1);
    clip_u8(value)
}

pub fn avg(a: u8, b: u8) -> u8 {
    clip_u8((i32::from(a) + i32::from(b) + 1) >> 1)
}

pub struct Edge {
    pub vertical: bool,
    pub bs: u8,
    pub qpav: i32,
    pub offset_a: i32,
    pub offset_b: i32,
    pub chroma: bool,
    pub len: usize,
}

pub fn filter_edge(plane: &mut [u8], stride: usize, x: usize, y: usize, edge: Edge) {
    let Edge {
        vertical,
        bs,
        qpav,
        offset_a,
        offset_b,
        chroma,
        len,
    } = edge;
    if bs == 0 || len == 0 {
        return;
    }
    let index_a = (qpav + offset_a).clamp(0, 51) as usize;
    let index_b = (qpav + offset_b).clamp(0, 51) as usize;
    let alpha = i32::from(ALPHA[index_a]);
    let beta = i32::from(BETA[index_b]);
    for i in 0..len {
        let (px, py) = if vertical { (x, y + i) } else { (x + i, y) };
        let load = |dx: i32, dy: i32| -> i32 {
            let sx = px as i32 + if vertical { dx } else { 0 };
            let sy = py as i32 + if vertical { 0 } else { dy };
            i32::from(plane[(sy as usize) * stride + sx as usize])
        };
        let store = |dx: i32, dy: i32, value: i32, plane: &mut [u8]| {
            let sx = px as i32 + if vertical { dx } else { 0 };
            let sy = py as i32 + if vertical { 0 } else { dy };
            plane[(sy as usize) * stride + sx as usize] = clip_u8(value);
        };
        let p0 = load(-1, -1);
        let q0 = load(0, 0);
        let p1 = load(-2, -2);
        let q1 = load(1, 1);
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let p2 = load(-3, -3);
        let q2 = load(2, 2);
        let p3 = load(-4, -4);
        let q3 = load(3, 3);
        if bs < 4 {
            let tc0 = i32::from(TC0[bs as usize - 1][index_a]);
            let ap = (p2 - p0).abs();
            let aq = (q2 - q0).abs();
            let tc = if chroma {
                tc0 + 1
            } else {
                tc0 + i32::from(ap < beta) + i32::from(aq < beta)
            };
            let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            store(-1, -1, p0 + delta, plane);
            store(0, 0, q0 - delta, plane);
            if !chroma && ap < beta {
                let d = ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0);
                store(-2, -2, p1 + d, plane);
            }
            if !chroma && aq < beta {
                let d = ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0);
                store(1, 1, q1 + d, plane);
            }
        } else {
            let strong_p =
                !chroma && (p2 - p0).abs() < beta && (p0 - q0).abs() < ((alpha >> 2) + 2);
            let strong_q =
                !chroma && (q2 - q0).abs() < beta && (p0 - q0).abs() < ((alpha >> 2) + 2);
            if strong_p {
                store(-1, -1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3, plane);
                store(-2, -2, (p2 + p1 + p0 + q0 + 2) >> 2, plane);
                store(-3, -3, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3, plane);
            } else {
                store(-1, -1, (2 * p1 + p0 + q1 + 2) >> 2, plane);
            }
            if strong_q {
                store(0, 0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3, plane);
                store(1, 1, (p0 + q0 + q1 + q2 + 2) >> 2, plane);
                store(2, 2, (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3, plane);
            } else {
                store(0, 0, (2 * q1 + q0 + p1 + 2) >> 2, plane);
            }
        }
    }
}
