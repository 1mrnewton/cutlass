//! The CDEF kernels as the specification writes them.

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
