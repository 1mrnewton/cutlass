//! The CDEF kernels as the specification writes them.

use super::{CENTRE, Filter, UNAVAILABLE, WINDOW};

/// `Div_Table` of 7.15.2.
const DIV_TABLE: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];

/// The CDEF direction process (7.15.2) on samples of `bd` bits.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn direction(px: &[[u16; 8]; 8], bd: u32) -> (u8, i32) {
    let mut partial = [[0i32; 15]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let x = i32::from(px[i][j] >> (bd - 8)) - 128;
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }
    let sq = |v: i32| v * v;
    let mut cost = [0i32; 8];
    cost[2] = partial[2][..8].iter().map(|&v| sq(v)).sum();
    cost[6] = partial[6][..8].iter().map(|&v| sq(v)).sum();
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    for d in [0, 4] {
        for i in 0..7 {
            cost[d] += (sq(partial[d][i]) + sq(partial[d][14 - i])) * DIV_TABLE[i + 1];
        }
        cost[d] += sq(partial[d][7]) * DIV_TABLE[8];
    }
    for d in [1, 3, 5, 7] {
        for j in 0..5 {
            cost[d] += sq(partial[d][3 + j]);
        }
        cost[d] *= DIV_TABLE[8];
        for j in 0..3 {
            cost[d] += (sq(partial[d][j]) + sq(partial[d][10 - j])) * DIV_TABLE[2 * j + 2];
        }
    }
    best(&cost)
}

/// The direction with the largest cost (the first of equals), and how far
/// it stands above the direction at a right angle to it.
pub(super) fn best(cost: &[i32; 8]) -> (u8, i32) {
    let mut best_cost = 0;
    let mut dir = 0usize;
    for (d, &c) in cost.iter().enumerate() {
        if c > best_cost {
            best_cost = c;
            dir = d;
        }
    }
    (dir as u8, (best_cost - cost[(dir + 4) & 7]) >> 10)
}

/// `constrain` of 7.15.3.
fn constrain(diff: i32, threshold: i32, shift: u32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let ad = diff.abs();
    let v = ad.min((threshold - (ad >> shift)).max(0));
    if diff < 0 { -v } else { v }
}

/// The CDEF filter process (7.15.3), one sample at a time.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn filter(win: &[u16; WINDOW], f: &Filter, rows: usize) -> [[u16; 8]; 8] {
    let mut out = [[0u16; 8]; 8];
    for (i, row) in out.iter_mut().enumerate().take(rows) {
        for (j, o) in row.iter_mut().enumerate() {
            let base = i * super::STRIDE + j;
            let x = i32::from(win[base + CENTRE]);
            let (mut sum, mut max, mut min) = (0i32, x, x);
            let mut tap = |at: usize, weight: i32, strength: u16, shift: u32| {
                let at = at.min(2 * CENTRE);
                for at in [at, 2 * CENTRE - at] {
                    let p = win[base + at];
                    if p != UNAVAILABLE {
                        let p = i32::from(p);
                        sum += weight * constrain(p - x, i32::from(strength), shift);
                        max = max.max(p);
                        min = min.min(p);
                    }
                }
            };
            for k in 0..2 {
                tap(f.pri_at[k], i32::from(f.pri_taps[k]), f.pri, f.pri_shift);
                tap(f.sec_at[k], 2, f.sec, f.sec_shift);
                tap(f.sec_at[2 + k], 1, f.sec, f.sec_shift);
            }
            let y = x + ((8 + sum - i32::from(sum < 0)) >> 4);
            *o = y.clamp(min, max) as u16;
        }
    }
    out
}
