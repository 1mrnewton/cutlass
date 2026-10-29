#![allow(dead_code)]
//! Frame probabilities, trees, and backward adaptation.
//!
//! Defaults match libvpx. Coefficient probabilities live in [`super::tables`].
//! Adaptation overwrites the working context from the saved context plus
//! counts. Forward updates from the compressed header are not the values
//! that get saved when adaptation runs.

use cutlass_decode::CodecError;

use crate::bits::BoolDecoder;
use crate::tables::{COEF_PROBS, INV_MAP_TABLE};

pub const INTRA_TREE: [i8; 18] = [
    0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7,
];
pub const PARTITION_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub const INTER_TREE: [i8; 6] = [-2, 2, 0, 4, -1, -3];
pub const SEG_TREE: [i8; 14] = [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7];
pub const JOINT_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub const CLASS_TREE: [i8; 20] = [
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
];
pub const FP_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub const INTERP_TREE: [i8; 4] = [0, 2, -1, -2];

const COUNT_FACTOR: [u32; 21] = [
    0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89, 96, 102, 108, 115, 121, 128,
];

#[derive(Clone, Copy)]
pub struct MvComp {
    pub sign: u8,
    pub classes: [u8; 10],
    pub class0: u8,
    pub bits: [u8; 10],
    pub class0_fp: [[u8; 3]; 2],
    pub fp: [u8; 3],
    pub class0_hp: u8,
    pub hp: u8,
}

#[derive(Clone, Copy)]
pub struct MvProbs {
    pub joints: [u8; 3],
    pub comp: [MvComp; 2],
}

#[derive(Clone)]
pub struct FrameContext {
    pub coef: crate::tables::CoefProbs,
    pub skip: [u8; 3],
    pub tx32: [[u8; 3]; 2],
    pub tx16: [[u8; 2]; 2],
    pub tx8: [[u8; 1]; 2],
    pub inter_mode: [[u8; 3]; 7],
    pub interp: [[u8; 2]; 4],
    pub intra_inter: [u8; 4],
    pub comp_inter: [u8; 5],
    pub comp_ref: [u8; 5],
    pub single_ref: [[u8; 2]; 5],
    pub y_mode: [[u8; 9]; 4],
    pub uv_mode: [[u8; 9]; 10],
    pub partition: [[u8; 3]; 16],
    pub mv: MvProbs,
    pub initialized: bool,
}

type TokenCount = [u32; 4];
type CtxCount = [TokenCount; 6];
type BandCount = [CtxCount; 6];
type RefCount = [BandCount; 2];
type PlaneCount = [RefCount; 2];
type EobCtx = [u32; 6];
type EobBand = [EobCtx; 6];
type EobRef = [EobBand; 2];
type EobPlane = [EobRef; 2];

#[derive(Clone)]
pub struct Counts {
    pub coef: [PlaneCount; 4],
    pub eob: [EobPlane; 4],
    pub skip: [[u32; 2]; 3],
    pub tx8: [[u32; 2]; 2],
    pub tx16: [[u32; 3]; 2],
    pub tx32: [[u32; 4]; 2],
    pub inter_mode: [[u32; 4]; 7],
    pub interp: [[u32; 3]; 4],
    pub intra_inter: [[u32; 2]; 4],
    pub comp_inter: [[u32; 2]; 5],
    pub comp_ref: [[u32; 2]; 5],
    pub single_ref: [[[u32; 2]; 2]; 5],
    pub y_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub partition: [[u32; 4]; 16],
    pub joints: [u32; 4],
    pub mv_sign: [[u32; 2]; 2],
    pub mv_class: [[u32; 11]; 2],
    pub mv_class0: [[u32; 2]; 2],
    pub mv_bits: [[[u32; 2]; 10]; 2],
    pub mv_class0_fp: [[[u32; 4]; 2]; 2],
    pub mv_fp: [[u32; 4]; 2],
    pub mv_class0_hp: [[u32; 2]; 2],
    pub mv_hp: [[u32; 2]; 2],
}

impl Counts {
    pub fn new() -> Self {
        Self {
            coef: [[[[[[0; 4]; 6]; 6]; 2]; 2]; 4],
            eob: [[[[[0; 6]; 6]; 2]; 2]; 4],
            skip: [[0; 2]; 3],
            tx8: [[0; 2]; 2],
            tx16: [[0; 3]; 2],
            tx32: [[0; 4]; 2],
            inter_mode: [[0; 4]; 7],
            interp: [[0; 3]; 4],
            intra_inter: [[0; 2]; 4],
            comp_inter: [[0; 2]; 5],
            comp_ref: [[0; 2]; 5],
            single_ref: [[[0; 2]; 2]; 5],
            y_mode: [[0; 10]; 4],
            uv_mode: [[0; 10]; 10],
            partition: [[0; 4]; 16],
            joints: [0; 4],
            mv_sign: [[0; 2]; 2],
            mv_class: [[0; 11]; 2],
            mv_class0: [[0; 2]; 2],
            mv_bits: [[[0; 2]; 10]; 2],
            mv_class0_fp: [[[0; 4]; 2]; 2],
            mv_fp: [[0; 4]; 2],
            mv_class0_hp: [[0; 2]; 2],
            mv_hp: [[0; 2]; 2],
        }
    }
}

impl Default for FrameContext {
    fn default() -> Self {
        Self::defaults()
    }
}

impl FrameContext {
    /// A context that must not be used until a keyframe or a reset installs defaults.
    pub fn unready() -> Self {
        let mut ctx = Self::defaults();
        ctx.initialized = false;
        ctx
    }

    pub fn defaults() -> Self {
        let vertical = MvComp {
            sign: 128,
            classes: [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
            class0: 216,
            bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
            class0_fp: [[128, 128, 64], [96, 112, 64]],
            fp: [64, 96, 64],
            class0_hp: 160,
            hp: 128,
        };
        let horizontal = MvComp {
            sign: 128,
            classes: [216, 128, 176, 160, 176, 176, 192, 198, 198, 208],
            class0: 208,
            bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
            class0_fp: [[128, 128, 64], [96, 112, 64]],
            fp: [64, 96, 64],
            class0_hp: 160,
            hp: 128,
        };
        Self {
            coef: COEF_PROBS,
            skip: [192, 128, 64],
            tx32: [[3, 136, 37], [5, 52, 13]],
            tx16: [[20, 152], [15, 101]],
            tx8: [[100], [66]],
            inter_mode: [
                [2, 173, 34],
                [7, 145, 85],
                [7, 166, 63],
                [7, 94, 66],
                [8, 64, 46],
                [17, 81, 31],
                [25, 29, 30],
            ],
            interp: [[235, 162], [36, 255], [34, 3], [149, 144]],
            intra_inter: [9, 102, 187, 225],
            comp_inter: [239, 183, 119, 96, 41],
            comp_ref: [50, 126, 123, 221, 226],
            single_ref: [[33, 16], [77, 74], [142, 142], [172, 170], [238, 247]],
            y_mode: [
                [65, 32, 18, 144, 162, 194, 41, 51, 98],
                [132, 68, 18, 165, 217, 196, 45, 40, 78],
                [173, 80, 19, 176, 240, 193, 64, 35, 46],
                [221, 135, 38, 194, 248, 121, 96, 85, 29],
            ],
            uv_mode: [
                [120, 7, 76, 176, 208, 126, 28, 54, 103],
                [48, 12, 154, 155, 139, 90, 34, 117, 119],
                [67, 6, 25, 204, 243, 158, 13, 21, 96],
                [97, 5, 44, 131, 176, 139, 48, 68, 97],
                [83, 5, 42, 156, 111, 152, 26, 49, 152],
                [80, 5, 58, 178, 74, 83, 33, 62, 145],
                [86, 5, 32, 154, 192, 168, 14, 22, 163],
                [85, 5, 32, 156, 216, 148, 19, 29, 73],
                [77, 7, 64, 116, 132, 122, 37, 126, 120],
                [101, 21, 107, 181, 192, 103, 19, 67, 125],
            ],
            partition: [
                [199, 122, 141],
                [147, 63, 159],
                [148, 133, 118],
                [121, 104, 114],
                [174, 73, 87],
                [92, 41, 83],
                [82, 99, 50],
                [53, 39, 39],
                [177, 58, 59],
                [68, 26, 63],
                [52, 79, 25],
                [17, 14, 12],
                [222, 34, 30],
                [72, 16, 44],
                [58, 32, 12],
                [10, 7, 6],
            ],
            mv: MvProbs {
                joints: [32, 64, 96],
                comp: [vertical, horizontal],
            },
            initialized: true,
        }
    }
}

pub fn read_tree(r: &mut BoolDecoder<'_>, tree: &[i8], probs: &[u8]) -> Result<u8, CodecError> {
    let mut i = 0i16;
    loop {
        let bit = r.read(u32::from(probs[usize::from((i as u16) >> 1)]))?;
        let next =
            i16::from(tree[usize::try_from(i).unwrap_or(0) + usize::try_from(bit).unwrap_or(0)]);
        if next <= 0 {
            return Ok(u8::try_from(-next).unwrap_or(0));
        }
        i = next;
    }
}

pub fn diff_update(r: &mut BoolDecoder<'_>, prob: &mut u8) -> Result<(), CodecError> {
    if r.read(252)? == 0 {
        return Ok(());
    }
    let del = decode_term_subexp(r)?;
    *prob = inv_remap(del, *prob);
    Ok(())
}

fn decode_term_subexp(r: &mut BoolDecoder<'_>) -> Result<u8, CodecError> {
    if r.read(128)? == 0 {
        return Ok(u8::try_from(r.read_literal(4)?).unwrap_or(0));
    }
    if r.read(128)? == 0 {
        return Ok(u8::try_from(r.read_literal(4)?)
            .unwrap_or(0)
            .saturating_add(16));
    }
    if r.read(128)? == 0 {
        return Ok(u8::try_from(r.read_literal(5)?)
            .unwrap_or(0)
            .saturating_add(32));
    }
    let v = decode_uniform(r)?;
    Ok(v.saturating_add(64))
}

fn decode_uniform(r: &mut BoolDecoder<'_>) -> Result<u8, CodecError> {
    let m = (1u32 << 8) - 191;
    let v = r.read_literal(7)?;
    if v < m {
        Ok(u8::try_from(v).unwrap_or(0))
    } else {
        let extra = r.read(128)?;
        Ok(u8::try_from((v << 1) - m + extra).unwrap_or(0))
    }
}

fn inv_remap(v: u8, m: u8) -> u8 {
    let mapped = u32::from(INV_MAP_TABLE[usize::from(v)]);
    let m = u32::from(m).saturating_sub(1);
    let value = if (m << 1) <= 255 {
        1 + recenter(mapped, m)
    } else {
        255 - recenter(mapped, 254 - m)
    };
    u8::try_from(value).unwrap_or(1)
}

fn recenter(v: u32, m: u32) -> u32 {
    if v > 2 * m {
        return v;
    }
    if v & 1 == 1 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

pub fn update_mv(r: &mut BoolDecoder<'_>, p: &mut [u8]) -> Result<(), CodecError> {
    for slot in p.iter_mut() {
        if r.read(252)? == 1 {
            let v = r.read_literal(7)?;
            *slot = u8::try_from((v << 1) | 1).unwrap_or(1);
        }
    }
    Ok(())
}

fn binary_prob(n0: u32, n1: u32) -> u8 {
    let den = n0.saturating_add(n1);
    if den == 0 {
        return 128;
    }
    clipped_prob(n0, den)
}

fn clipped_prob(num: u32, den: u32) -> u8 {
    let p = (u64::from(num) * 256 + u64::from(den >> 1)) / u64::from(den);
    if p == 0 {
        1
    } else if p > 255 {
        255
    } else {
        u8::try_from(p).unwrap_or(1)
    }
}

fn weighted(pre: u8, next: u8, factor: u32) -> u8 {
    let mixed = u32::from(pre) * (256 - factor) + u32::from(next) * factor;
    u8::try_from((mixed + 128) >> 8).unwrap_or(pre)
}

fn merge_coef(pre: u8, ct0: u32, ct1: u32, sat: u32, factor: u32) -> u8 {
    let prob = binary_prob(ct0, ct1);
    let count = ct0.saturating_add(ct1).min(sat);
    let scale = factor.saturating_mul(count) / sat.max(1);
    weighted(pre, prob, scale)
}

fn merge_mode(pre: u8, ct0: u32, ct1: u32) -> u8 {
    let den = ct0.saturating_add(ct1);
    if den == 0 {
        return pre;
    }
    let count = den.min(20);
    let factor = COUNT_FACTOR[count as usize];
    weighted(pre, clipped_prob(ct0, den), factor)
}

fn tree_merge(tree: &[i8], pre: &[u8], counts: &[u32], out: &mut [u8]) {
    tree_merge_at(0, tree, pre, counts, out);
}

fn tree_merge_at(i: usize, tree: &[i8], pre: &[u8], counts: &[u32], out: &mut [u8]) -> u32 {
    let left = tree_side(tree[i], i16::from(tree[i]), tree, pre, counts, out);
    let right = tree_side(tree[i + 1], i16::from(tree[i + 1]), tree, pre, counts, out);
    out[i >> 1] = merge_mode(pre[i >> 1], left, right);
    left.saturating_add(right)
}

fn tree_side(
    node: i8,
    signed: i16,
    tree: &[i8],
    pre: &[u8],
    counts: &[u32],
    out: &mut [u8],
) -> u32 {
    if signed <= 0 {
        let index = usize::try_from(-signed).unwrap_or(0);
        counts.get(index).copied().unwrap_or(0)
    } else {
        tree_merge_at(usize::from(node as u8), tree, pre, counts, out)
    }
}

/// Coefficient counts are stored per transform size. `tx` selects that size.
pub fn adapt_coef(
    fc: &mut FrameContext,
    pre: &FrameContext,
    counts: &Counts,
    tx: usize,
    key: bool,
    after_key: bool,
) {
    let factor = if key || !after_key { 112 } else { 128 };
    let sat = 24u32;
    let bands = [3usize, 6, 6, 6, 6, 6];
    for plane in 0..2 {
        for reference in 0..2 {
            for (band, &kept) in bands.iter().enumerate() {
                for ctx in 0..kept {
                    let n0 = counts.coef[tx][plane][reference][band][ctx][0];
                    let n1 = counts.coef[tx][plane][reference][band][ctx][1];
                    let n2 = counts.coef[tx][plane][reference][band][ctx][2];
                    let neob = counts.coef[tx][plane][reference][band][ctx][3];
                    let branch = counts.eob[tx][plane][reference][band][ctx];
                    let pairs = [
                        (neob, branch.saturating_sub(neob)),
                        (n0, n1.saturating_add(n2)),
                        (n1, n2),
                    ];
                    for (node, (c0, c1)) in pairs.into_iter().enumerate() {
                        fc.coef[tx][plane][reference][band][ctx][node] = merge_coef(
                            pre.coef[tx][plane][reference][band][ctx][node],
                            c0,
                            c1,
                            sat,
                            factor,
                        );
                    }
                }
            }
        }
    }
}

pub fn adapt_modes(
    fc: &mut FrameContext,
    pre: &FrameContext,
    counts: &Counts,
    select_tx: bool,
    switchable: bool,
) {
    for i in 0..4 {
        fc.intra_inter[i] = merge_mode(
            pre.intra_inter[i],
            counts.intra_inter[i][0],
            counts.intra_inter[i][1],
        );
    }
    for i in 0..5 {
        fc.comp_inter[i] = merge_mode(
            pre.comp_inter[i],
            counts.comp_inter[i][0],
            counts.comp_inter[i][1],
        );
        fc.comp_ref[i] = merge_mode(
            pre.comp_ref[i],
            counts.comp_ref[i][0],
            counts.comp_ref[i][1],
        );
        for j in 0..2 {
            fc.single_ref[i][j] = merge_mode(
                pre.single_ref[i][j],
                counts.single_ref[i][j][0],
                counts.single_ref[i][j][1],
            );
        }
    }
    for i in 0..7 {
        tree_merge(
            &INTER_TREE,
            &pre.inter_mode[i],
            &counts.inter_mode[i],
            &mut fc.inter_mode[i],
        );
    }
    for i in 0..4 {
        tree_merge(
            &INTRA_TREE,
            &pre.y_mode[i],
            &counts.y_mode[i],
            &mut fc.y_mode[i],
        );
    }
    for i in 0..10 {
        tree_merge(
            &INTRA_TREE,
            &pre.uv_mode[i],
            &counts.uv_mode[i],
            &mut fc.uv_mode[i],
        );
    }
    for i in 0..16 {
        tree_merge(
            &PARTITION_TREE,
            &pre.partition[i],
            &counts.partition[i],
            &mut fc.partition[i],
        );
    }
    if switchable {
        for i in 0..4 {
            tree_merge(
                &INTERP_TREE,
                &pre.interp[i],
                &counts.interp[i],
                &mut fc.interp[i],
            );
        }
    }
    if select_tx {
        adapt_tx(fc, pre, counts);
    }
    for i in 0..3 {
        fc.skip[i] = merge_mode(pre.skip[i], counts.skip[i][0], counts.skip[i][1]);
    }
}

fn adapt_tx(fc: &mut FrameContext, pre: &FrameContext, counts: &Counts) {
    for i in 0..2 {
        let c4 = counts.tx8[i][0];
        let c8 = counts.tx8[i][1];
        fc.tx8[i][0] = merge_mode(pre.tx8[i][0], c4, c8);

        let p4 = counts.tx16[i][0];
        let p8 = counts.tx16[i][1];
        let p16 = counts.tx16[i][2];
        fc.tx16[i][0] = merge_mode(pre.tx16[i][0], p4, p8.saturating_add(p16));
        fc.tx16[i][1] = merge_mode(pre.tx16[i][1], p8, p16);

        let q4 = counts.tx32[i][0];
        let q8 = counts.tx32[i][1];
        let q16 = counts.tx32[i][2];
        let q32 = counts.tx32[i][3];
        fc.tx32[i][0] = merge_mode(
            pre.tx32[i][0],
            q4,
            q8.saturating_add(q16).saturating_add(q32),
        );
        fc.tx32[i][1] = merge_mode(pre.tx32[i][1], q8, q16.saturating_add(q32));
        fc.tx32[i][2] = merge_mode(pre.tx32[i][2], q16, q32);
    }
}

pub fn adapt_mv(fc: &mut FrameContext, pre: &FrameContext, counts: &Counts, allow_hp: bool) {
    tree_merge(
        &JOINT_TREE,
        &pre.mv.joints,
        &counts.joints,
        &mut fc.mv.joints,
    );
    for i in 0..2 {
        fc.mv.comp[i].sign = merge_mode(
            pre.mv.comp[i].sign,
            counts.mv_sign[i][0],
            counts.mv_sign[i][1],
        );
        tree_merge(
            &CLASS_TREE,
            &pre.mv.comp[i].classes,
            &counts.mv_class[i],
            &mut fc.mv.comp[i].classes,
        );
        fc.mv.comp[i].class0 = merge_mode(
            pre.mv.comp[i].class0,
            counts.mv_class0[i][0],
            counts.mv_class0[i][1],
        );
        for bit in 0..10 {
            fc.mv.comp[i].bits[bit] = merge_mode(
                pre.mv.comp[i].bits[bit],
                counts.mv_bits[i][bit][0],
                counts.mv_bits[i][bit][1],
            );
        }
        for d in 0..2 {
            tree_merge(
                &FP_TREE,
                &pre.mv.comp[i].class0_fp[d],
                &counts.mv_class0_fp[i][d],
                &mut fc.mv.comp[i].class0_fp[d],
            );
        }
        tree_merge(
            &FP_TREE,
            &pre.mv.comp[i].fp,
            &counts.mv_fp[i],
            &mut fc.mv.comp[i].fp,
        );
        if allow_hp {
            fc.mv.comp[i].class0_hp = merge_mode(
                pre.mv.comp[i].class0_hp,
                counts.mv_class0_hp[i][0],
                counts.mv_class0_hp[i][1],
            );
            fc.mv.comp[i].hp =
                merge_mode(pre.mv.comp[i].hp, counts.mv_hp[i][0], counts.mv_hp[i][1]);
        }
    }
}

pub fn kf_partition() -> [[u8; 3]; 16] {
    [
        [158, 97, 94],
        [93, 24, 99],
        [85, 119, 44],
        [62, 59, 67],
        [149, 53, 53],
        [94, 20, 48],
        [83, 53, 24],
        [52, 18, 18],
        [150, 40, 39],
        [78, 12, 26],
        [67, 33, 11],
        [24, 7, 5],
        [174, 35, 49],
        [68, 11, 27],
        [57, 15, 9],
        [12, 3, 3],
    ]
}
