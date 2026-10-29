//! Inverse transforms from section 7.13 of the AV1 specification.

use crate::dims::{
    ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_ADST,
    FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, V_ADST, V_DCT, V_FLIPADST,
};

const COS128: [i32; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
];

fn round2(x: i64, n: u32) -> i64 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

fn brev(bits: u32, x: usize) -> usize {
    if bits == 0 || bits >= usize::BITS {
        return 0;
    }
    let width = usize::BITS - bits;
    (x & ((1usize << bits) - 1)).reverse_bits() >> width
}

fn cos128(angle: i32) -> i32 {
    let angle2 = angle & 255;
    if (0..=64).contains(&angle2) {
        COS128[angle2 as usize]
    } else if (65..=128).contains(&angle2) {
        -COS128[(128 - angle2) as usize]
    } else if (129..=192).contains(&angle2) {
        -COS128[(angle2 - 128) as usize]
    } else {
        COS128[(256 - angle2) as usize]
    }
}

fn butterfly(values: &mut [i64], a: usize, b: usize, angle: i32, flip: bool) {
    let cosine = i64::from(cos128(angle));
    let sine = i64::from(cos128(angle - 64));
    let left = values[a];
    let right = values[b];
    values[a] = round2(left * cosine - right * sine, 12);
    values[b] = round2(left * sine + right * cosine, 12);
    if flip {
        values.swap(a, b);
    }
}

fn hadamard(values: &mut [i64], a: usize, b: usize, flip: bool, range: i32) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let left = values[a];
    let right = values[b];
    let limit = 1i64 << (range - 1);
    values[a] = (left + right).clamp(-limit, limit - 1);
    values[b] = (left - right).clamp(-limit, limit - 1);
}

fn permute_dct(values: &mut [i64], n: u32) {
    let len = 1usize << n;
    if len > values.len() {
        return;
    }
    if len <= 64 {
        let mut copy = [0i64; 64];
        copy[..len].copy_from_slice(&values[..len]);
        for (i, slot) in values.iter_mut().enumerate().take(len) {
            *slot = copy[brev(n, i)];
        }
        return;
    }
    let copy = values[..len].to_vec();
    for (i, slot) in values.iter_mut().enumerate().take(len) {
        *slot = copy[brev(n, i)];
    }
}

fn inverse_dct(values: &mut [i64], n: u32, range: i32) {
    permute_dct(values, n);
    dct_first(values, n, range);
    dct_middle(values, n, range);
    dct_last(values, n, range);
}

fn dct_first(values: &mut [i64], n: u32, range: i32) {
    if n == 6 {
        for i in 0..16 {
            butterfly(values, 32 + i, 63 - i, 63 - 4 * brev(4, i) as i32, false);
        }
    }
    if n >= 5 {
        for i in 0..8 {
            butterfly(
                values,
                16 + i,
                31 - i,
                6 + ((brev(3, 7 - i) as i32) << 3),
                false,
            );
        }
    }
    if n == 6 {
        for i in 0..16 {
            hadamard(values, 32 + i * 2, 33 + i * 2, i & 1 == 1, range);
        }
    }
    if n >= 4 {
        for i in 0..4 {
            butterfly(
                values,
                8 + i,
                15 - i,
                12 + ((brev(2, 3 - i) as i32) << 4),
                false,
            );
        }
    }
    if n >= 5 {
        for i in 0..8 {
            hadamard(values, 16 + 2 * i, 17 + 2 * i, i & 1 == 1, range);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..2 {
                butterfly(
                    values,
                    62 - i * 4 - j,
                    33 + i * 4 + j,
                    60 - 16 * brev(2, i) as i32 + 64 * j as i32,
                    true,
                );
            }
        }
    }
    if n >= 3 {
        for i in 0..2 {
            butterfly(values, 4 + i, 7 - i, 56 - 32 * i as i32, false);
        }
    }
}

fn dct_middle(values: &mut [i64], n: u32, range: i32) {
    if n >= 4 {
        for i in 0..4 {
            hadamard(values, 8 + 2 * i, 9 + 2 * i, i & 1 == 1, range);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..2 {
                butterfly(
                    values,
                    30 - 4 * i - j,
                    17 + 4 * i + j,
                    24 + ((j as i32) << 6) + ((1 - i as i32) << 5),
                    true,
                );
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            for j in 0..2 {
                hadamard(values, 32 + i * 4 + j, 35 + i * 4 - j, i & 1 == 1, range);
            }
        }
    }
    for i in 0..2 {
        butterfly(values, 2 * i, 2 * i + 1, 32 + 16 * i as i32, i == 0);
    }
    if n >= 3 {
        for i in 0..2 {
            hadamard(values, 4 + 2 * i, 5 + 2 * i, i == 1, range);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            butterfly(values, 14 - i, 9 + i, 48 + 64 * i as i32, true);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            for j in 0..2 {
                hadamard(values, 16 + 4 * i + j, 19 + 4 * i - j, i & 1 == 1, range);
            }
        }
    }
    if n == 6 {
        for i in 0..2 {
            for j in 0..4 {
                let angle = 56 - i as i32 * 32 + i32::from(j > 1) * 64;
                butterfly(values, 61 - i * 8 - j, 34 + i * 8 + j, angle, true);
            }
        }
    }
    for i in 0..2 {
        hadamard(values, i, 3 - i, false, range);
    }
    if n >= 3 {
        butterfly(values, 6, 5, 32, true);
    }
}

fn dct_last(values: &mut [i64], n: u32, range: i32) {
    if n >= 4 {
        for i in 0..2 {
            for j in 0..2 {
                hadamard(values, 8 + 4 * i + j, 11 + 4 * i - j, i == 1, range);
            }
        }
    }
    if n >= 5 {
        for i in 0..4 {
            butterfly(values, 29 - i, 18 + i, 48 + i32::from(i > 1) * 64, true);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..4 {
                hadamard(values, 32 + 8 * i + j, 39 + 8 * i - j, i & 1 == 1, range);
            }
        }
    }
    if n >= 3 {
        for i in 0..4 {
            hadamard(values, i, 7 - i, false, range);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            butterfly(values, 13 - i, 10 + i, 32, true);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..4 {
                hadamard(values, 16 + i * 8 + j, 23 + i * 8 - j, i == 1, range);
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            butterfly(values, 59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
        }
    }
    if n >= 4 {
        for i in 0..8 {
            hadamard(values, i, 15 - i, false, range);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            butterfly(values, 27 - i, 20 + i, 32, true);
        }
    }
    if n == 6 {
        for i in 0..8 {
            hadamard(values, 32 + i, 47 - i, false, range);
            hadamard(values, 48 + i, 63 - i, true, range);
        }
    }
    if n >= 5 {
        for i in 0..16 {
            hadamard(values, i, 31 - i, false, range);
        }
    }
    if n == 6 {
        for i in 0..8 {
            butterfly(values, 55 - i, 40 + i, 32, true);
        }
        for i in 0..32 {
            hadamard(values, i, 63 - i, false, range);
        }
    }
}

fn adst_input(values: &mut [i64], n: u32) {
    let len = 1usize << n;
    let copy = values[..len].to_vec();
    for (i, slot) in values.iter_mut().enumerate().take(len) {
        let idx = if i & 1 == 1 { i - 1 } else { len - i - 1 };
        *slot = copy[idx];
    }
}

fn adst_output(values: &mut [i64], n: u32) {
    let len = 1usize << n;
    let copy = values[..len].to_vec();
    for (i, slot) in values.iter_mut().enumerate().take(len) {
        let a = (i >> 3) & 1;
        let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
        let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
        let d = (i & 1) ^ ((i >> 1) & 1);
        let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
        let sample = copy[idx];
        *slot = if i & 1 == 1 { -sample } else { sample };
    }
}

fn inverse_adst4(values: &mut [i64]) {
    let t = [values[0], values[1], values[2], values[3]];
    let mut s = [0i64; 7];
    s[0] = 1321 * t[0];
    s[1] = 2482 * t[0];
    s[2] = 3344 * t[1];
    s[3] = 3803 * t[2];
    s[4] = 1321 * t[2];
    s[5] = 2482 * t[3];
    s[6] = 3803 * t[3];
    let b7 = (t[0] - t[2]) + t[3];
    s[0] += s[3];
    s[1] -= s[4];
    s[3] = s[2];
    s[2] = 3344 * b7;
    s[0] += s[5];
    s[1] -= s[6];
    let mut x = [s[0] + s[3], s[1] + s[3], s[2], s[0] + s[1]];
    x[3] -= s[3];
    for i in 0..4 {
        values[i] = round2(x[i], 12);
    }
}

fn inverse_adst8(values: &mut [i64], range: i32) {
    adst_input(values, 3);
    for i in 0..4 {
        butterfly(values, 2 * i, 2 * i + 1, 60 - 16 * i as i32, true);
    }
    for i in 0..4 {
        hadamard(values, i, 4 + i, false, range);
    }
    for i in 0..2 {
        butterfly(values, 4 + 3 * i, 5 + i, 48 - 32 * i as i32, true);
    }
    for j in 0..2 {
        for i in 0..2 {
            hadamard(values, 4 * j + i, 2 + 4 * j + i, false, range);
        }
    }
    for i in 0..2 {
        butterfly(values, 2 + 4 * i, 3 + 4 * i, 32, true);
    }
    adst_output(values, 3);
}

fn inverse_adst16(values: &mut [i64], range: i32) {
    adst_input(values, 4);
    for i in 0..8 {
        butterfly(values, 2 * i, 2 * i + 1, 62 - 8 * i as i32, true);
    }
    for i in 0..8 {
        hadamard(values, i, 8 + i, false, range);
    }
    for i in 0..2 {
        butterfly(values, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i32, true);
        butterfly(values, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i32, true);
    }
    for j in 0..2 {
        for i in 0..4 {
            hadamard(values, 8 * j + i, 4 + 8 * j + i, false, range);
        }
    }
    for j in 0..2 {
        for i in 0..2 {
            butterfly(
                values,
                4 + 8 * j + 3 * i,
                5 + 8 * j + i,
                48 - 32 * i as i32,
                true,
            );
        }
    }
    for j in 0..4 {
        for i in 0..2 {
            hadamard(values, 4 * j + i, 2 + 4 * j + i, false, range);
        }
    }
    for i in 0..4 {
        butterfly(values, 2 + 4 * i, 3 + 4 * i, 32, true);
    }
    adst_output(values, 4);
}

fn inverse_adst(values: &mut [i64], n: u32, range: i32) {
    if n == 2 {
        inverse_adst4(values);
    } else if n == 3 {
        inverse_adst8(values, range);
    } else {
        inverse_adst16(values, range);
    }
}

fn inverse_identity(values: &mut [i64], n: u32) {
    let len = 1usize << n;
    match n {
        2 => {
            for slot in values.iter_mut().take(len) {
                *slot = round2(*slot * 5793, 12);
            }
        }
        3 => {
            for slot in values.iter_mut().take(len) {
                *slot *= 2;
            }
        }
        4 => {
            for slot in values.iter_mut().take(len) {
                *slot = round2(*slot * 11586, 12);
            }
        }
        _ => {
            for slot in values.iter_mut().take(len) {
                *slot *= 4;
            }
        }
    }
}

fn row_kind(tx_type: u8) -> u8 {
    match tx_type {
        DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT => 0,
        DCT_ADST | ADST_ADST | DCT_FLIPADST | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST
        | H_ADST | H_FLIPADST => 1,
        _ => 2,
    }
}

fn col_kind(tx_type: u8) -> u8 {
    match tx_type {
        DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT => 0,
        ADST_DCT | ADST_ADST | FLIPADST_DCT | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST
        | V_ADST | V_FLIPADST => 1,
        _ => 2,
    }
}

fn apply_1d(values: &mut [i64], n: u32, kind: u8, range: i32) {
    if kind == 0 {
        inverse_dct(values, n, range);
    } else if kind == 1 {
        inverse_adst(values, n, range);
    } else {
        inverse_identity(values, n);
    }
}

/// Working storage for one inverse transform.
pub struct InverseScratch<'a> {
    pub residual: &'a mut [i64],
    pub row: &'a mut [i64],
    pub out: &'a mut [i32],
}

/// Inverse 2D transform. `coeff` is row-major and `width` by `height`.
pub fn inverse_2d(
    coeff: &[i32],
    width: usize,
    height: usize,
    tx_type: u8,
    row_shift: u32,
) -> Vec<i32> {
    let n = width.saturating_mul(height);
    let mut residual = vec![0i64; n];
    let mut row = vec![0i64; width.max(height)];
    let mut out = vec![0i32; n];
    inverse_2d_into(
        coeff,
        &mut InverseScratch {
            residual: &mut residual,
            row: &mut row,
            out: &mut out,
        },
        width,
        height,
        tx_type,
        row_shift,
    );
    out
}

/// Same transform as [`inverse_2d`], writing into caller storage.
pub fn inverse_2d_into(
    coeff: &[i32],
    scratch: &mut InverseScratch<'_>,
    width: usize,
    height: usize,
    tx_type: u8,
    row_shift: u32,
) {
    let n = width.saturating_mul(height);
    let row_n = width.max(height);
    if n == 0 || scratch.residual.len() < n || scratch.row.len() < row_n || scratch.out.len() < n {
        for slot in scratch.out.iter_mut().take(n) {
            *slot = 0;
        }
        return;
    }
    let log_w = width.trailing_zeros();
    let log_h = height.trailing_zeros();
    let rect = log_w.abs_diff(log_h) == 1;
    for i in 0..height {
        for (j, slot) in scratch.row.iter_mut().enumerate().take(width) {
            let sample = if i < 32 && j < 32 {
                coeff.get(i * width + j).copied().unwrap_or(0)
            } else {
                0
            };
            *slot = if rect {
                round2(i64::from(sample) * 2896, 12)
            } else {
                i64::from(sample)
            };
        }
        apply_1d(&mut scratch.row[..row_n], log_w, row_kind(tx_type), 16);
        for j in 0..width {
            scratch.residual[i * width + j] = round2(scratch.row[j], row_shift);
        }
    }
    for sample in &mut scratch.residual[..n] {
        *sample = (*sample).clamp(-(1 << 15), (1 << 15) - 1);
    }
    for j in 0..width {
        for i in 0..height {
            scratch.row[i] = scratch.residual[i * width + j];
        }
        apply_1d(&mut scratch.row[..row_n], log_h, col_kind(tx_type), 16);
        for i in 0..height {
            scratch.residual[i * width + j] = round2(scratch.row[i], 4);
        }
    }
    for (slot, sample) in scratch.out.iter_mut().zip(scratch.residual.iter()).take(n) {
        *slot = i32::try_from(*sample).unwrap_or(0);
    }
}
