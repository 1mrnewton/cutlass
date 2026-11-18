//! The probabilities other than the coefficients' that a frame may update
//! in its compressed header: skip, tx size, the inter frame's mode,
//! filter, reference, intra mode and partition probabilities, and the
//! motion vector ones. They sit flattened in the order the header codes
//! them, so that a coded symbol can name its probability by position and
//! take the updated value when the tiles are written.

use crate::bits::BoolWriter;
use crate::prob::{self, Counts, FrameContext};

use super::tile::bit_cost;

pub const SKIP: usize = 0;
pub const TX8: usize = SKIP + 3;
pub const TX16: usize = TX8 + 2;
pub const TX32: usize = TX16 + 4;
pub const INTER_MODE: usize = TX32 + 6;
pub const INTERP: usize = INTER_MODE + 21;
pub const INTRA_INTER: usize = INTERP + 8;
pub const COMP_INTER: usize = INTRA_INTER + 4;
pub const SINGLE_REF: usize = COMP_INTER + 5;
pub const COMP_REF: usize = SINGLE_REF + 10;
pub const Y_MODE: usize = COMP_REF + 5;
pub const PARTITION: usize = Y_MODE + 36;
pub const MV_JOINTS: usize = PARTITION + 48;
/// Per component: sign, classes (10), class0, bits (10).
pub const MV_COMP: usize = MV_JOINTS + 3;
pub const MV_COMP_LEN: usize = 22;
/// Per component: class0_fp (2×3), fp (3).
pub const MV_FP: usize = MV_COMP + 2 * MV_COMP_LEN;
pub const MV_FP_LEN: usize = 9;
/// Per component: class0_hp, hp.
pub const MV_HP: usize = MV_FP + 2 * MV_FP_LEN;
pub const LEN: usize = MV_HP + 4;

pub type Flat = [u8; LEN];

/// What the frame's header codes, which decides the probabilities it can
/// update.
#[derive(Clone, Copy)]
pub struct Coded {
    pub tx_select: bool,
    pub inter: bool,
    pub switchable: bool,
    pub allow_hp: bool,
    /// Whether the references' sign biases allow compound prediction (the
    /// header then codes the reference mode).
    pub compound_allowed: bool,
    /// Compound prediction: `None` single only, `Some(true)` chosen per
    /// block, `Some(false)` every block.
    pub compound: Option<bool>,
}

/// `fc`'s updatable probabilities, flattened.
pub fn flatten(fc: &FrameContext) -> Flat {
    let mut f = [0u8; LEN];
    let mut put = |at: usize, v: &[u8]| f[at..at + v.len()].copy_from_slice(v);
    put(SKIP, &fc.skip);
    for c in 0..2 {
        put(TX8 + c, &fc.tx8[c]);
        put(TX16 + 2 * c, &fc.tx16[c]);
        put(TX32 + 3 * c, &fc.tx32[c]);
    }
    for (i, r) in fc.inter_mode.iter().enumerate() {
        put(INTER_MODE + 3 * i, r);
    }
    for (i, r) in fc.interp.iter().enumerate() {
        put(INTERP + 2 * i, r);
    }
    put(INTRA_INTER, &fc.intra_inter);
    put(COMP_INTER, &fc.comp_inter);
    for (i, r) in fc.single_ref.iter().enumerate() {
        put(SINGLE_REF + 2 * i, r);
    }
    put(COMP_REF, &fc.comp_ref);
    for (i, r) in fc.y_mode.iter().enumerate() {
        put(Y_MODE + 9 * i, r);
    }
    for (i, r) in fc.partition.iter().enumerate() {
        put(PARTITION + 3 * i, r);
    }
    put(MV_JOINTS, &fc.mv.joints);
    for (k, c) in fc.mv.comp.iter().enumerate() {
        let b = MV_COMP + k * MV_COMP_LEN;
        put(b, &[c.sign]);
        put(b + 1, &c.classes);
        put(b + 11, &[c.class0]);
        put(b + 12, &c.bits);
        let b = MV_FP + k * MV_FP_LEN;
        put(b, &c.class0_fp[0]);
        put(b + 3, &c.class0_fp[1]);
        put(b + 6, &c.fp);
        put(MV_HP + 2 * k, &[c.class0_hp, c.hp]);
    }
    f
}

/// How a probability is updated in the header.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A sub-exponential code of its difference (libvpx `diff_update`).
    Diff,
    /// Seven bits of the new value (libvpx `update_mv_probs`).
    Mv,
}

/// The ranges the header codes, in its order, with how each is coded.
fn fields(c: &Coded) -> Vec<(usize, usize, Kind)> {
    use Kind::{Diff, Mv};
    let mut v = Vec::new();
    if c.tx_select {
        v.push((TX8, 2, Diff));
        v.push((TX16, 4, Diff));
        v.push((TX32, 6, Diff));
    }
    // The coefficients come here, written by the caller.
    v.push((SKIP, 3, Diff));
    if !c.inter {
        return v;
    }
    v.push((INTER_MODE, 21, Diff));
    if c.switchable {
        v.push((INTERP, 8, Diff));
    }
    v.push((INTRA_INTER, 4, Diff));
    if c.compound == Some(true) {
        v.push((COMP_INTER, 5, Diff));
    }
    if c.compound != Some(false) {
        v.push((SINGLE_REF, 10, Diff));
    }
    if c.compound.is_some() {
        v.push((COMP_REF, 5, Diff));
    }
    v.push((Y_MODE, 36, Diff));
    v.push((PARTITION, 48, Diff));
    v.push((MV_JOINTS, 3, Mv));
    for k in 0..2 {
        v.push((MV_COMP + k * MV_COMP_LEN, MV_COMP_LEN, Mv));
    }
    for k in 0..2 {
        v.push((MV_FP + k * MV_FP_LEN, MV_FP_LEN, Mv));
    }
    if c.allow_hp {
        for k in 0..2 {
            v.push((MV_HP + 2 * k, 2, Mv));
        }
    }
    v
}

/// The counts of each probability's two branches, from the frame's
/// symbol counts.
fn branches(counts: &Counts) -> [[u32; 2]; LEN] {
    let mut b = [[0u32; 2]; LEN];
    let mut tree = |at: usize, tree: &[i8], n: &[u32]| {
        for (sym, &count) in n.iter().enumerate() {
            if count == 0 {
                continue;
            }
            walk(tree, sym as u8, |bit, node| b[at + node][bit] += count);
        }
    };
    for (c, n) in counts.skip.iter().enumerate() {
        tree(SKIP + c, &[0, -1], n);
    }
    for c in 0..2 {
        tree(TX8 + c, &[0, -1], &counts.tx8[c]);
        tree(TX16 + 2 * c, &[0, 2, -1, -2], &counts.tx16[c]);
        tree(TX32 + 3 * c, &[0, 2, -1, 4, -2, -3], &counts.tx32[c]);
    }
    for (i, n) in counts.inter_mode.iter().enumerate() {
        tree(INTER_MODE + 3 * i, &prob::INTER_TREE, n);
    }
    for (i, n) in counts.interp.iter().enumerate() {
        tree(INTERP + 2 * i, &prob::INTERP_TREE, n);
    }
    for (i, n) in counts.intra_inter.iter().enumerate() {
        tree(INTRA_INTER + i, &[0, -1], n);
    }
    for (i, n) in counts.comp_inter.iter().enumerate() {
        tree(COMP_INTER + i, &[0, -1], n);
    }
    for (i, n) in counts.single_ref.iter().enumerate() {
        tree(SINGLE_REF + 2 * i, &[0, -1], &n[0]);
        tree(SINGLE_REF + 2 * i + 1, &[0, -1], &n[1]);
    }
    for (i, n) in counts.comp_ref.iter().enumerate() {
        tree(COMP_REF + i, &[0, -1], n);
    }
    for (i, n) in counts.y_mode.iter().enumerate() {
        tree(Y_MODE + 9 * i, &prob::INTRA_TREE, n);
    }
    for (i, n) in counts.partition.iter().enumerate() {
        tree(PARTITION + 3 * i, &prob::PARTITION_TREE, n);
    }
    tree(MV_JOINTS, &prob::JOINT_TREE, &counts.joints);
    for k in 0..2 {
        let at = MV_COMP + k * MV_COMP_LEN;
        tree(at, &[0, -1], &counts.mv_sign[k]);
        tree(at + 1, &prob::CLASS_TREE, &counts.mv_class[k]);
        tree(at + 11, &[0, -1], &counts.mv_class0[k]);
        for (i, n) in counts.mv_bits[k].iter().enumerate() {
            tree(at + 12 + i, &[0, -1], n);
        }
        let at = MV_FP + k * MV_FP_LEN;
        tree(at, &prob::FP_TREE, &counts.mv_class0_fp[k][0]);
        tree(at + 3, &prob::FP_TREE, &counts.mv_class0_fp[k][1]);
        tree(at + 6, &prob::FP_TREE, &counts.mv_fp[k]);
        tree(MV_HP + 2 * k, &[0, -1], &counts.mv_class0_hp[k]);
        tree(MV_HP + 2 * k + 1, &[0, -1], &counts.mv_hp[k]);
    }
    b
}

/// Walks `symbol`'s path down `tree` (libvpx's layout: entry `i` and
/// `i + 1` are a node's branches, a leaf is `-symbol`), calling `f` with
/// each bit and the index of the node's probability.
pub fn walk(tree: &[i8], symbol: u8, mut f: impl FnMut(usize, usize)) {
    let holds = |node: i8| -> bool {
        let mut stack = [0i8; 16];
        let (mut n, mut cur) = (0, node);
        loop {
            if cur <= 0 {
                if -cur == symbol as i8 {
                    return true;
                }
            } else {
                let at = cur as usize;
                stack[n] = tree[at + 1];
                n += 1;
                cur = tree[at];
                continue;
            }
            if n == 0 {
                return false;
            }
            n -= 1;
            cur = stack[n];
        }
    };
    let mut at = 0usize;
    loop {
        let (l, r) = (tree[at], tree[at + 1]);
        let bit = usize::from(!holds(l));
        f(bit, at >> 1);
        let next = if bit == 0 { l } else { r };
        if next <= 0 {
            return;
        }
        at = next as usize;
    }
}

/// The probabilities after the updates that pay on this frame's counts:
/// each is updated when coding its branches at the new value saves more
/// than the update costs (libvpx `vp9_cond_prob_diff_update` and
/// `update_mv`).
pub fn updates(old: &Flat, counts: &Counts, coded: &Coded) -> Flat {
    let mut new = *old;
    let b = branches(counts);
    let flag = |bit: bool| i64::from(bit_cost(bit, 252));
    for (at, len, kind) in fields(coded) {
        for i in at..at + len {
            let [c0, c1] = b[i];
            let total = u64::from(c0) + u64::from(c1);
            if total == 0 {
                continue;
            }
            let p = ((256 * u64::from(c0) + total / 2) / total).clamp(1, 255) as u8;
            let p = match kind {
                Kind::Diff => p,
                Kind::Mv => (p >> 1) | 1,
            };
            if p == old[i] {
                continue;
            }
            let cost = |q: u8| {
                i64::from(c0) * i64::from(bit_cost(false, q))
                    + i64::from(c1) * i64::from(bit_cost(true, q))
            };
            let update = match kind {
                Kind::Diff => match super::remap(p, old[i]) {
                    Some(v) => i64::from(super::subexp_bits(v)) * 256,
                    None => continue,
                },
                Kind::Mv => 7 * 256,
            };
            if cost(old[i]) - cost(p) - (flag(true) - flag(false)) - update > 0 {
                new[i] = p;
            }
        }
    }
    new
}

/// The header's reference mode, when the sign biases allow compound
/// prediction (single, compound for every block, or chosen per block).
fn ref_mode(w: &mut BoolWriter, coded: &Coded) {
    if !coded.compound_allowed {
        return;
    }
    w.write(u32::from(coded.compound.is_some()), 128);
    if let Some(select) = coded.compound {
        w.write(u32::from(select), 128);
    }
}

/// Writes the updates of the ranges before the coefficients (the tx
/// probabilities) when `before`, else those after.
pub fn write(w: &mut BoolWriter, old: &Flat, new: &Flat, coded: &Coded, before: bool) {
    for (at, len, kind) in fields(coded) {
        let is_tx = matches!(at, TX8 | TX16 | TX32);
        if is_tx != before {
            continue;
        }
        if at == COMP_INTER || (at == SINGLE_REF && coded.compound != Some(true)) {
            // The reference mode comes before its probabilities.
            ref_mode(w, coded);
        }
        for i in at..at + len {
            if new[i] == old[i] {
                w.write(0, 252);
                continue;
            }
            match kind {
                Kind::Diff => match super::remap(new[i], old[i]) {
                    Some(v) => {
                        w.write(1, 252);
                        super::write_subexp(w, v);
                    }
                    None => w.write(0, 252),
                },
                Kind::Mv => {
                    w.write(1, 252);
                    w.write_literal(u32::from(new[i] >> 1), 7);
                }
            }
        }
    }
}
