//! Bit costs of modes and motion vectors, in 1/256 bit.

use crate::bool_coder::cost;
use crate::modes::{ABOVE_4X4, LEFT_4X4, Mv, ZERO_4X4, sub_mv_context};
use crate::tables::{
    B_MODE_PROBS, B_MODE_TREE, KF_B_MODE_PROBS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, KF_Y_MODE_TREE,
    MV_REF_TREE, SMALL_MV_TREE, SPLIT_MV_PROBS, SPLIT_MV_TREE, SUB_MV_REF_PROBS, SUB_MV_REF_TREE,
    UV_MODE_TREE, Y_MODE_TREE,
};

/// A split part's vector mode, the fourth (a new vector).
pub const NEW_4X4: u8 = 3;

/// The cheapest sub-block vector mode that gives `mv` next to `left` and
/// `above`, and its cost without any new vector.
#[must_use]
pub fn split_mode(left: Mv, above: Mv, mv: Mv) -> (u8, u32) {
    let probs = &SUB_MV_REF_PROBS[sub_mv_context(left, above)];
    let mut best = (NEW_4X4, tree_cost(&SUB_MV_REF_TREE, probs, NEW_4X4));
    for (mode, matches) in [
        (LEFT_4X4, mv == left),
        (ABOVE_4X4, mv == above),
        (ZERO_4X4, mv.is_zero()),
    ] {
        let c = tree_cost(&SUB_MV_REF_TREE, probs, mode);
        if matches && c < best.1 {
            best = (mode, c);
        }
    }
    best
}

/// Cost of each SPLITMV partitioning, 16×8, 8×16, 8×8, and 4×4.
#[must_use]
pub fn split_costs() -> [u32; 4] {
    std::array::from_fn(|p| tree_cost(&SPLIT_MV_TREE, &SPLIT_MV_PROBS, p as u8))
}

/// The branches from the root of `tree` to `symbol`, as (probability
/// index, bit).
#[must_use]
pub fn tree_path(tree: &[i8], symbol: u8) -> Vec<(usize, bool)> {
    fn find(tree: &[i8], node: usize, symbol: u8, path: &mut Vec<(usize, bool)>) -> bool {
        for bit in [false, true] {
            let next = tree[node + usize::from(bit)];
            path.push((node >> 1, bit));
            if next <= 0 {
                if next.unsigned_abs() == symbol {
                    return true;
                }
            } else if find(tree, next as usize, symbol, path) {
                return true;
            }
            path.pop();
        }
        false
    }
    let mut path = Vec::new();
    find(tree, 0, symbol, &mut path);
    path
}

/// Cost of coding `symbol` with `tree` and `probs`.
#[must_use]
pub fn tree_cost(tree: &[i8], probs: &[u8], symbol: u8) -> u32 {
    tree_path(tree, symbol)
        .iter()
        .map(|&(i, bit)| cost(probs.get(i).copied().unwrap_or(128), bit))
        .sum()
}

/// Mode costs for one frame.
#[derive(Clone, Debug)]
pub struct ModeCosts {
    /// Key-frame or inter-frame luma mode costs, DC..B_PRED.
    pub y: [u32; 5],
    pub uv: [u32; 4],
    /// Key frames: `[above][left][mode]`.
    pub kf_b: Box<[[[u32; 10]; 10]; 10]>,
    /// Inter frames.
    pub b: [u32; 10],
}

impl ModeCosts {
    #[must_use]
    pub fn new(key: bool, y_probs: &[u8; 4], uv_probs: &[u8; 3]) -> Self {
        let y = std::array::from_fn(|m| {
            if key {
                tree_cost(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS, m as u8)
            } else {
                tree_cost(&Y_MODE_TREE, y_probs, m as u8)
            }
        });
        let uv = std::array::from_fn(|m| {
            if key {
                tree_cost(&UV_MODE_TREE, &KF_UV_MODE_PROBS, m as u8)
            } else {
                tree_cost(&UV_MODE_TREE, uv_probs, m as u8)
            }
        });
        let mut kf_b = Box::new([[[0u32; 10]; 10]; 10]);
        for a in 0..10 {
            for l in 0..10 {
                for m in 0..10 {
                    kf_b[a][l][m] = tree_cost(&B_MODE_TREE, &KF_B_MODE_PROBS[a][l], m as u8);
                }
            }
        }
        let b = std::array::from_fn(|m| tree_cost(&B_MODE_TREE, &B_MODE_PROBS, m as u8));
        Self { y, uv, kf_b, b }
    }
}

/// Costs of the inter modes under one macroblock's mode probabilities,
/// NEAREST..SPLIT.
#[must_use]
pub fn inter_mode_costs(probs: &[u8; 4]) -> [u32; 5] {
    std::array::from_fn(|m| tree_cost(&MV_REF_TREE, probs, m as u8))
}

/// Largest vector difference a NEWMV can code, in quarter pixels.
pub const MAX_MV_DELTA: i32 = 1023;

/// Costs of every vector component difference, per component.
#[derive(Clone, Debug)]
pub struct MvCosts {
    /// `[component][delta + MAX_MV_DELTA]`.
    pub table: [Vec<u32>; 2],
}

impl MvCosts {
    #[must_use]
    pub fn new(probs: &[[u8; 19]; 2]) -> Self {
        let table = std::array::from_fn(|c| {
            let p = &probs[c];
            (-MAX_MV_DELTA..=MAX_MV_DELTA)
                .map(|d| component_cost(p, d))
                .collect()
        });
        Self { table }
    }

    /// Cost of coding `delta` (row, col) in quarter pixels.
    #[must_use]
    pub fn cost(&self, row: i32, col: i32) -> u32 {
        let at = |c: usize, d: i32| {
            let i = (d.clamp(-MAX_MV_DELTA, MAX_MV_DELTA) + MAX_MV_DELTA) as usize;
            self.table[c][i]
        };
        at(0, row) + at(1, col)
    }
}

fn component_cost(p: &[u8; 19], d: i32) -> u32 {
    let x = d.unsigned_abs();
    let mut total;
    if x < 8 {
        total = cost(p[0], false) + tree_cost(&SMALL_MV_TREE, &p[2..9], x as u8);
    } else {
        total = cost(p[0], true);
        for i in 0..3 {
            total += cost(p[9 + i], (x >> i) & 1 == 1);
        }
        for i in (4..10).rev() {
            total += cost(p[9 + i], (x >> i) & 1 == 1);
        }
        if x & 0xfff0 != 0 {
            total += cost(p[12], (x >> 3) & 1 == 1);
        }
    }
    if x != 0 {
        total += cost(p[1], d < 0);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::{MvCosts, tree_path};
    use crate::tables::{B_MODE_TREE, DEFAULT_MV_PROBS, MV_REF_TREE};

    #[test]
    fn paths_reach_every_leaf() {
        for m in 0..10u8 {
            assert!(!tree_path(&B_MODE_TREE, m).is_empty(), "b mode {m}");
        }
        for m in 0..5u8 {
            assert!(!tree_path(&MV_REF_TREE, m).is_empty(), "mv ref {m}");
        }
        assert_eq!(tree_path(&MV_REF_TREE, 2), vec![(0, false)]);
    }

    #[test]
    fn small_vectors_cost_less() {
        let costs = MvCosts::new(&DEFAULT_MV_PROBS);
        assert!(costs.cost(0, 0) < costs.cost(0, 4));
        assert!(costs.cost(0, 4) < costs.cost(0, 400));
    }
}
