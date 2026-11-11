//! The edge codes one unit at a time, as the specification decides them.

use super::{EdgeUnits, Edges};

/// One unit of [`super::edge_codes`]: lane `l` of `u`.
pub(crate) fn edge_code(u: &EdgeUnits, l: usize, edges: Edges, luma: bool) -> u16 {
    let (d, across, lvl_across, on) = match edges {
        Edges::Left { first } => {
            let d = u.dims[l] & 7;
            let col = first.wrapping_add(l as u8);
            let (a, la) = if l == 0 {
                (u.across[15], u.lvl_across[15])
            } else {
                (u.dims[l - 1], u.lvl[l - 1])
            };
            (d, a & 7, la, col & ((1u8 << d) - 1) == 0)
        }
        Edges::Top { zeros } => {
            let d = (u.dims[l] >> 3) & 7;
            (d, (u.across[l] >> 3) & 7, u.lvl_across[l], d <= zeros)
        }
    };
    let m = d.min(across);
    let size: u16 = match (luma, m) {
        (true, 0) | (false, 0) => 4,
        (true, 1) => 8,
        (true, _) => 16,
        (false, _) => 6,
    };
    let lvl = if u.lvl[l] != 0 { u.lvl[l] } else { lvl_across };
    if on && lvl != 0 {
        size << 8 | u16::from(lvl)
    } else {
        0
    }
}

#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
pub(super) fn edge_codes(u: &EdgeUnits, edges: Edges, luma: bool) -> ([u16; 16], u16) {
    let codes: [u16; 16] = core::array::from_fn(|l| edge_code(u, l, edges, luma));
    let mask = codes
        .iter()
        .enumerate()
        .fold(0, |m, (l, &c)| m | u16::from(c != 0) << l);
    (codes, mask)
}

#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
pub(super) fn levels(records: &[[u8; 8]; 16], i: usize) -> [u8; 16] {
    core::array::from_fn(|k| records[k][i])
}

#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
pub(super) fn levels_odd(records: &[[u8; 8]; 32], i: usize) -> [u8; 16] {
    core::array::from_fn(|k| records[2 * k + 1][i])
}
