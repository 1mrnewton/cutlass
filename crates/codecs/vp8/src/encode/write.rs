//! Writing a coded frame: macroblock modes, vectors, and tokens.

use crate::bool_coder::BoolEncoder;
use crate::header::{CoeffProbs, FrameHeader};
use crate::modes::{
    ALTREF, Bounds, INTRA, LAST, MbInfo, Mv, NEARESTMV, NEWMV, SPLITMV, SubInfo, find_near,
    sub_mv_context,
};
use crate::predict::B_PRED;
use crate::tables::{
    B_MODE_PROBS, B_MODE_TREE, KF_B_MODE_PROBS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, KF_Y_MODE_TREE,
    MV_PARTITIONS, MV_REF_TREE, SMALL_MV_TREE, SPLIT_COUNTS, SPLIT_MV_PROBS, SPLIT_MV_TREE,
    SUB_MV_REF_PROBS, SUB_MV_REF_TREE, UV_MODE_TREE, Y_MODE_TREE,
};
use crate::tokens::{Context, TYPE_CHROMA, TYPE_Y_AFTER_Y2, TYPE_Y_WITH_DC, TYPE_Y2, skip_mb};

use super::cost::{NEW_4X4, split_mode, tree_path};
use super::frame::Coded;
use super::residual::{Counts, QBlock, best_prob, branch_cost, count_block, write_block};

/// Writes one vector component difference (Section 17.2).
fn write_component(w: &mut BoolEncoder, p: &[u8; 19], d: i32) {
    let x = d.unsigned_abs();
    if x < 8 {
        w.write(p[0], false);
        w.tree(&SMALL_MV_TREE, &p[2..9], x as u8);
    } else {
        w.write(p[0], true);
        for i in 0..3 {
            w.write(p[9 + i], (x >> i) & 1 == 1);
        }
        for i in (4..10).rev() {
            w.write(p[9 + i], (x >> i) & 1 == 1);
        }
        if x & 0xfff0 != 0 {
            w.write(p[12], (x >> 3) & 1 == 1);
        }
    }
    if x != 0 {
        w.write(p[1], d < 0);
    }
}

fn write_mv(w: &mut BoolEncoder, probs: &[[u8; 19]; 2], d: Mv) {
    write_component(w, &probs[0], d.row);
    write_component(w, &probs[1], d.col);
}

/// Walks the macroblocks the way the decoder reads them and calls `f` with
/// each one and its neighbours.
fn for_each_mb(
    coded: &Coded,
    mbw: usize,
    mbh: usize,
    mut f: impl FnMut(usize, usize, usize, Neighbours<'_>),
) {
    for mby in 0..mbh {
        for mbx in 0..mbw {
            let i = mby * mbw + mbx;
            let n = Neighbours {
                above: (mby > 0).then(|| &coded.mbs[i - mbw]),
                left: (mbx > 0).then(|| &coded.mbs[i - 1]),
                above_left: (mby > 0 && mbx > 0).then(|| &coded.mbs[i - mbw - 1]),
                above_sub: (mby > 0).then(|| &coded.subs[i - mbw]),
                left_sub: (mbx > 0).then(|| &coded.subs[i - 1]),
            };
            f(mbx, mby, i, n);
        }
    }
}

struct Neighbours<'a> {
    above: Option<&'a MbInfo>,
    left: Option<&'a MbInfo>,
    above_left: Option<&'a MbInfo>,
    above_sub: Option<&'a SubInfo>,
    left_sub: Option<&'a SubInfo>,
}

/// Writes every macroblock's modes into the first partition after the
/// header (Sections 11, 16, 17).
pub fn write_modes(w: &mut BoolEncoder, h: &FrameHeader, coded: &Coded, mbw: usize, mbh: usize) {
    let sign_bias = [false, false, h.sign_bias_golden, h.sign_bias_altref];
    for_each_mb(coded, mbw, mbh, |mbx, mby, i, n| {
        let mb = &coded.mbs[i];
        let sub = &coded.subs[i];
        if h.segmentation.update_map {
            let s = mb.segment & 3;
            let p = &h.segmentation.tree_probs;
            w.write(p[0], s >= 2);
            if s >= 2 {
                w.write(p[2], s == 3);
            } else {
                w.write(p[1], s == 1);
            }
        }
        if h.skip_enabled {
            w.write(h.prob_skip, mb.skip);
        }
        if h.tag.key {
            w.tree(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS, mb.y_mode);
            if mb.y_mode == B_PRED {
                for b in 0..16 {
                    let a = if b < 4 {
                        n.above_sub.map_or(0, |s| s.b_modes[b + 12])
                    } else {
                        sub.b_modes[b - 4]
                    };
                    let l = if b & 3 == 0 {
                        n.left_sub.map_or(0, |s| s.b_modes[b + 3])
                    } else {
                        sub.b_modes[b - 1]
                    };
                    w.tree(
                        &B_MODE_TREE,
                        &KF_B_MODE_PROBS[usize::from(a)][usize::from(l)],
                        sub.b_modes[b],
                    );
                }
            }
            w.tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS, mb.uv_mode);
            return;
        }
        w.write(h.prob_intra, mb.ref_frame != INTRA);
        if mb.ref_frame == INTRA {
            w.tree(&Y_MODE_TREE, &h.probs.y_mode, mb.y_mode);
            if mb.y_mode == B_PRED {
                for &m in &sub.b_modes {
                    w.tree(&B_MODE_TREE, &B_MODE_PROBS, m);
                }
            }
            w.tree(&UV_MODE_TREE, &h.probs.uv_mode, mb.uv_mode);
            return;
        }
        w.write(h.prob_last, mb.ref_frame != LAST);
        if mb.ref_frame != LAST {
            w.write(h.prob_golden, mb.ref_frame == ALTREF);
        }
        let near = find_near(n.above, n.left, n.above_left, &sign_bias, mb.ref_frame);
        w.tree(&MV_REF_TREE, &near.probs, mb.y_mode - NEARESTMV);
        let bounds = Bounds::new(mbx, mby, mbw, mbh);
        let best = near.best.clamp(&bounds);
        if mb.y_mode == NEWMV {
            write_mv(
                w,
                &h.probs.mv,
                Mv::new(mb.mv.row - best.row, mb.mv.col - best.col),
            );
        } else if mb.y_mode == SPLITMV {
            let partition = usize::from(mb.partition & 3);
            w.tree(&SPLIT_MV_TREE, &SPLIT_MV_PROBS, mb.partition & 3);
            let map = &MV_PARTITIONS[partition];
            for j in 0..SPLIT_COUNTS[partition] {
                let Some(k) = map.iter().position(|&p| usize::from(p) == j) else {
                    continue;
                };
                let left = if k & 3 == 0 {
                    n.left_sub.map_or(Mv::ZERO, |s| s.mvs[k + 3])
                } else {
                    sub.mvs[k - 1]
                };
                let above = if k < 4 {
                    n.above_sub.map_or(Mv::ZERO, |s| s.mvs[k + 12])
                } else {
                    sub.mvs[k - 4]
                };
                let mv = sub.mvs[k];
                let (mode, _) = split_mode(left, above, mv);
                w.tree(
                    &SUB_MV_REF_TREE,
                    &SUB_MV_REF_PROBS[sub_mv_context(left, above)],
                    mode,
                );
                if mode == NEW_4X4 {
                    write_mv(
                        w,
                        &h.probs.mv,
                        Mv::new(mv.row - best.row, mv.col - best.col),
                    );
                }
            }
        }
    });
}

/// Calls `f(type, ctx, first, block)` for each coded block in decoding
/// order, keeping the token contexts as the decoder does. `row` gets each
/// macroblock row index first.
fn for_each_block(
    coded: &Coded,
    mbw: usize,
    mbh: usize,
    mut row: impl FnMut(usize),
    mut f: impl FnMut(usize, usize, usize, &QBlock),
) {
    let mut above: Vec<Context> = vec![[0; 9]; mbw];
    for mby in 0..mbh {
        row(mby);
        let mut left: Context = [0; 9];
        for (mbx, a) in above.iter_mut().enumerate() {
            let i = mby * mbw + mbx;
            let mb = &coded.mbs[i];
            let blocks = &coded.blocks[i];
            let has_y2 = mb.y_mode != B_PRED && mb.y_mode != SPLITMV;
            if mb.skip {
                skip_mb(&mut left, a, has_y2);
                continue;
            }
            let (y_type, first) = if has_y2 {
                let ctx = usize::from(left[8] + a[8]);
                f(TYPE_Y2, ctx, 0, &blocks[24]);
                let has = u8::from(blocks[24].end > 0);
                left[8] = has;
                a[8] = has;
                (TYPE_Y_AFTER_Y2, 1)
            } else {
                (TYPE_Y_WITH_DC, 0)
            };
            for b in 0..16 {
                let ctx = usize::from(left[b >> 2] + a[b & 3]);
                f(y_type, ctx, first, &blocks[b]);
                let has = u8::from(usize::from(blocks[b].end) > first);
                left[b >> 2] = has;
                a[b & 3] = has;
            }
            for k in 0..8 {
                let (p, r, c) = (k >> 2, (k & 3) >> 1, k & 1);
                let (sl, sa) = (4 + 2 * p + r, 4 + 2 * p + c);
                let ctx = usize::from(left[sl] + a[sa]);
                f(TYPE_CHROMA, ctx, 0, &blocks[16 + k]);
                let has = u8::from(blocks[16 + k].end > 0);
                left[sl] = has;
                a[sa] = has;
            }
        }
    }
}

/// Token branch counts of a coded frame.
#[must_use]
pub fn count_tokens(coded: &Coded, mbw: usize, mbh: usize) -> Box<Counts> {
    let mut counts = Box::new([[[[[0u32; 2]; 11]; 3]; 8]; 4]);
    for_each_block(
        coded,
        mbw,
        mbh,
        |_| {},
        |ty, ctx, first, q| {
            count_block(&mut counts[ty], ctx, first, q);
        },
    );
    counts
}

/// Writes the token partitions.
#[must_use]
pub fn write_tokens(
    coded: &Coded,
    mbw: usize,
    mbh: usize,
    probs: &CoeffProbs,
    partitions: usize,
) -> Vec<Vec<u8>> {
    let mut writers: Vec<BoolEncoder> = (0..partitions).map(|_| BoolEncoder::new()).collect();
    let current = std::cell::Cell::new(0);
    for_each_block(
        coded,
        mbw,
        mbh,
        |mby| current.set(mby % partitions),
        |ty, ctx, first, q| write_block(&mut writers[current.get()], &probs[ty], ctx, first, q),
    );
    writers.into_iter().map(BoolEncoder::finish).collect()
}

/// Chooses a probability array for a tree from symbol counts, keeping the
/// old one unless the new one pays for its `update_bits`.
#[must_use]
pub fn choose_tree_probs<const N: usize>(
    tree: &[i8],
    old: &[u8; N],
    symbols: &[u32],
    update_bits: u32,
) -> Option<[u8; N]> {
    let mut nodes = [[0u32; 2]; N];
    for (symbol, &count) in symbols.iter().enumerate() {
        for (i, bit) in tree_path(tree, symbol as u8) {
            if i < N {
                nodes[i][usize::from(bit)] += count;
            }
        }
    }
    let new: [u8; N] = std::array::from_fn(|i| best_prob(nodes[i]));
    let old_cost: u64 = (0..N).map(|i| branch_cost(old[i], nodes[i])).sum();
    let new_cost: u64 =
        (0..N).map(|i| branch_cost(new[i], nodes[i])).sum::<u64>() + u64::from(update_bits) * 256;
    (new != *old && new_cost < old_cost).then_some(new)
}

/// A flag probability from counts of false and true.
#[must_use]
pub fn flag_prob(falses: u32, trues: u32) -> u8 {
    best_prob([falses, trues])
}
