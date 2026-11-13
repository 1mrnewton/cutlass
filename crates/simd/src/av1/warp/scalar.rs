//! The warp one sample at a time, as the specification writes it.

use super::Filters;

/// The vertical sums of the block, before their rounding.
fn sums(rows: [&[u8; 16]; 15], f: &Filters<'_>) -> [[i32; 8]; 8] {
    let [sx4, alpha, beta] = f.h;
    let [sy4, gamma, delta] = f.v;
    let inter: [[i32; 8]; 15] = core::array::from_fn(|r| {
        let i1 = r as i32 - 7;
        core::array::from_fn(|c| {
            let i2 = c as i32 - 4;
            let taps = f.at(sx4 + alpha * i2 + beta * i1);
            let s: i32 = (0..8)
                .map(|t| i32::from(taps[t]) * i32::from(rows[r][c + t]))
                .sum();
            (s + 4) >> 3
        })
    });
    core::array::from_fn(|r| {
        let i1 = r as i32 - 4;
        core::array::from_fn(|c| {
            let i2 = c as i32 - 4;
            let taps = f.at(sy4 + gamma * i2 + delta * i1);
            (0..8).map(|t| i32::from(taps[t]) * inter[r + t][c]).sum()
        })
    })
}

/// [`super::put8x8`].
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn put8x8(rows: [&[u8; 16]; 15], f: &Filters<'_>) -> [[u8; 8]; 8] {
    sums(rows, f).map(|row| row.map(|s| ((s + 1024) >> 11).clamp(0, 255) as u8))
}

/// [`super::prep8x8`].
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn prep8x8(rows: [&[u8; 16]; 15], f: &Filters<'_>) -> [[i16; 8]; 8] {
    sums(rows, f).map(|row| row.map(|s| ((s + 64) >> 7) as i16))
}
