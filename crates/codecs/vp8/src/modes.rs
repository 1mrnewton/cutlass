//! Macroblock modes and motion vectors (RFC 6386 Sections 11, 16, 17).
//!
//! Motion vectors are kept in quarter luma pixels, the unit they are coded
//! in. The same number is the vector in eighth chroma pixels.

use crate::bool_coder::BoolDecoder;
use crate::predict::{B_DC, B_HE, B_PRED, B_TM, B_VE, DC_PRED, H_PRED, TM_PRED, V_PRED};
use crate::tables::{
    B_MODE_PROBS, B_MODE_TREE, KF_B_MODE_PROBS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, KF_Y_MODE_TREE,
    MODE_CONTEXTS, MV_PARTITIONS, MV_REF_TREE, SMALL_MV_TREE, SPLIT_COUNTS, SPLIT_MV_PROBS,
    SPLIT_MV_TREE, SUB_MV_REF_PROBS, SUB_MV_REF_TREE, UV_MODE_TREE, Y_MODE_TREE,
};

/// Inter modes, numbered after the intra modes.
pub const NEARESTMV: u8 = 5;
pub const NEARMV: u8 = 6;
pub const ZEROMV: u8 = 7;
pub const NEWMV: u8 = 8;
pub const SPLITMV: u8 = 9;

/// Reference frames.
pub const INTRA: u8 = 0;
pub const LAST: u8 = 1;
pub const GOLDEN: u8 = 2;
pub const ALTREF: u8 = 3;

/// Sub-block MV modes.
pub const LEFT_4X4: u8 = 0;
pub const ABOVE_4X4: u8 = 1;
pub const ZERO_4X4: u8 = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Mv {
    pub row: i32,
    pub col: i32,
}

impl Mv {
    pub const ZERO: Self = Self { row: 0, col: 0 };

    #[must_use]
    pub const fn new(row: i32, col: i32) -> Self {
        Self { row, col }
    }

    #[must_use]
    pub fn is_zero(self) -> bool {
        self.row == 0 && self.col == 0
    }

    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.row + other.row, self.col + other.col)
    }

    #[must_use]
    pub fn clamp(self, bounds: &Bounds) -> Self {
        Self::new(
            self.row.clamp(bounds.top, bounds.bottom),
            self.col.clamp(bounds.left, bounds.right),
        )
    }
}

/// How far a macroblock's vector may point: one macroblock past the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub left: i32,
    pub right: i32,
    pub top: i32,
    pub bottom: i32,
}

impl Bounds {
    #[must_use]
    pub fn new(mbx: usize, mby: usize, mbw: usize, mbh: usize) -> Self {
        let i = |v: usize| i32::try_from(v).unwrap_or(i32::MAX / 128);
        Self {
            left: -((i(mbx) + 1) << 6),
            right: (i(mbw) - i(mbx)) << 6,
            top: -((i(mby) + 1) << 6),
            bottom: (i(mbh) - i(mby)) << 6,
        }
    }
}

/// One macroblock's modes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MbInfo {
    pub y_mode: u8,
    pub uv_mode: u8,
    pub ref_frame: u8,
    /// The macroblock's vector (sub-block 15's for SPLITMV).
    pub mv: Mv,
    pub segment: u8,
    /// The coded skip flag.
    pub skip: bool,
    /// Split partitioning for SPLITMV.
    pub partition: u8,
}

/// Per-macroblock sub-block data neighbours read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubInfo {
    pub b_modes: [u8; 16],
    pub mvs: [Mv; 16],
}

impl Default for SubInfo {
    fn default() -> Self {
        Self {
            b_modes: [B_DC; 16],
            mvs: [Mv::ZERO; 16],
        }
    }
}

/// The sub-block mode a whole-block intra mode implies for neighbours.
#[must_use]
pub fn implied_b_mode(y_mode: u8) -> u8 {
    match y_mode {
        V_PRED => B_VE,
        H_PRED => B_HE,
        TM_PRED => B_TM,
        _ => B_DC,
    }
}

/// What `find_near_mvs` produces (Section 16.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Near {
    pub best: Mv,
    pub nearest: Mv,
    pub near: Mv,
    pub probs: [u8; 4],
}

/// Neighbour as `find_near_mvs` sees it; `None` is outside the frame.
pub type Neighbour<'a> = Option<&'a MbInfo>;

/// Section 16.3, with sign bias by reference frame.
#[must_use]
pub fn find_near(
    above: Neighbour<'_>,
    left: Neighbour<'_>,
    above_left: Neighbour<'_>,
    sign_bias: &[bool; 4],
    this_ref: u8,
) -> Near {
    let mut mvs = [Mv::ZERO; 4];
    let mut cnt = [0u8; 4];
    let mut idx = 0usize;
    let biased = |mb: &MbInfo| {
        let mut mv = mb.mv;
        if sign_bias[usize::from(mb.ref_frame & 3)] != sign_bias[usize::from(this_ref & 3)] {
            mv = Mv::new(-mv.row, -mv.col);
        }
        mv
    };
    for (neighbour, weight, first) in [(above, 2, true), (left, 2, false), (above_left, 1, false)] {
        let Some(mb) = neighbour else { continue };
        if mb.ref_frame == INTRA {
            continue;
        }
        if mb.mv.is_zero() {
            cnt[0] += weight;
            continue;
        }
        let mv = biased(mb);
        if first || mv != mvs[idx] {
            idx += 1;
            mvs[idx] = mv;
        }
        cnt[idx] += weight;
    }
    if cnt[3] > 0 && mvs[3] == mvs[1] {
        cnt[1] += 1;
    }
    let split = |mb: Neighbour<'_>| u8::from(mb.is_some_and(|m| m.y_mode == SPLITMV));
    cnt[3] = (split(above) + split(left)) * 2 + split(above_left);
    if cnt[2] > cnt[1] {
        cnt.swap(1, 2);
        mvs.swap(1, 2);
    }
    if cnt[1] >= cnt[0] {
        mvs[0] = mvs[1];
    }
    let prob = |i: usize| MODE_CONTEXTS[usize::from(cnt[i]).min(5)][i];
    Near {
        best: mvs[0],
        nearest: mvs[1],
        near: mvs[2],
        probs: [prob(0), prob(1), prob(2), prob(3)],
    }
}

/// Context for a split sub-block's mode (Section 16.4).
#[must_use]
pub fn sub_mv_context(left: Mv, above: Mv) -> usize {
    let lez = left.is_zero();
    let aez = above.is_zero();
    if left == above {
        if lez { 4 } else { 3 }
    } else if aez {
        2
    } else if lez {
        1
    } else {
        0
    }
}

/// Reads one vector component (Section 17.2).
pub fn read_mv_component(bits: &mut BoolDecoder<'_>, p: &[u8; 19]) -> i32 {
    let mut x: i32;
    if bits.read(p[0]) {
        x = 0;
        for i in 0..3 {
            x += i32::from(bits.read(p[9 + i])) << i;
        }
        for i in (4..10).rev() {
            x += i32::from(bits.read(p[9 + i])) << i;
        }
        if x & 0xfff0 == 0 || bits.read(p[12]) {
            x += 8;
        }
    } else {
        x = i32::from(bits.tree(&SMALL_MV_TREE, &p[2..9]));
    }
    if x != 0 && bits.read(p[1]) { -x } else { x }
}

pub fn read_mv(bits: &mut BoolDecoder<'_>, probs: &[[u8; 19]; 2]) -> Mv {
    let row = read_mv_component(bits, &probs[0]);
    let col = read_mv_component(bits, &probs[1]);
    Mv::new(row, col)
}

/// Inputs to one macroblock's mode parse.
pub struct ModeContext<'a> {
    pub key: bool,
    pub above: Neighbour<'a>,
    pub left: Neighbour<'a>,
    pub above_left: Neighbour<'a>,
    pub above_sub: Option<&'a SubInfo>,
    pub left_sub: Option<&'a SubInfo>,
    pub bounds: Bounds,
    pub sign_bias: [bool; 4],
    pub y_probs: &'a [u8; 4],
    pub uv_probs: &'a [u8; 3],
    pub mv_probs: &'a [[u8; 19]; 2],
    pub prob_intra: u8,
    pub prob_last: u8,
    pub prob_golden: u8,
}

/// Reads one macroblock's modes after its segment and skip flags.
pub fn read_modes(
    bits: &mut BoolDecoder<'_>,
    ctx: &ModeContext<'_>,
    mb: &mut MbInfo,
    sub: &mut SubInfo,
) {
    if ctx.key {
        let y_mode = bits.tree(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS);
        mb.y_mode = y_mode;
        if y_mode == B_PRED {
            for b in 0..16 {
                let above = if b < 4 {
                    ctx.above_sub.map_or(B_DC, |s| s.b_modes[b + 12])
                } else {
                    sub.b_modes[b - 4]
                };
                let left = if b & 3 == 0 {
                    ctx.left_sub.map_or(B_DC, |s| s.b_modes[b + 3])
                } else {
                    sub.b_modes[b - 1]
                };
                sub.b_modes[b] = bits.tree(
                    &B_MODE_TREE,
                    &KF_B_MODE_PROBS[usize::from(above)][usize::from(left)],
                );
            }
        } else {
            sub.b_modes = [implied_b_mode(y_mode); 16];
        }
        mb.uv_mode = bits.tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS);
        mb.ref_frame = INTRA;
        mb.mv = Mv::ZERO;
        sub.mvs = [Mv::ZERO; 16];
        return;
    }
    if !bits.read(ctx.prob_intra) {
        let y_mode = bits.tree(&Y_MODE_TREE, ctx.y_probs);
        mb.y_mode = y_mode;
        if y_mode == B_PRED {
            for mode in &mut sub.b_modes {
                *mode = bits.tree(&B_MODE_TREE, &B_MODE_PROBS);
            }
        } else {
            sub.b_modes = [implied_b_mode(y_mode); 16];
        }
        mb.uv_mode = bits.tree(&UV_MODE_TREE, ctx.uv_probs);
        mb.ref_frame = INTRA;
        mb.mv = Mv::ZERO;
        sub.mvs = [Mv::ZERO; 16];
        return;
    }
    mb.ref_frame = if bits.read(ctx.prob_last) {
        GOLDEN + u8::from(bits.read(ctx.prob_golden))
    } else {
        LAST
    };
    mb.uv_mode = DC_PRED;
    sub.b_modes = [B_DC; 16];
    let near = find_near(
        ctx.above,
        ctx.left,
        ctx.above_left,
        &ctx.sign_bias,
        mb.ref_frame,
    );
    let mode = NEARESTMV + bits.tree(&MV_REF_TREE, &near.probs);
    mb.y_mode = match mode {
        5 => NEARESTMV,
        6 => NEARMV,
        7 => ZEROMV,
        8 => NEWMV,
        _ => SPLITMV,
    };
    match mb.y_mode {
        NEARESTMV => mb.mv = near.nearest.clamp(&ctx.bounds),
        NEARMV => mb.mv = near.near.clamp(&ctx.bounds),
        ZEROMV => mb.mv = Mv::ZERO,
        NEWMV => {
            let best = near.best.clamp(&ctx.bounds);
            mb.mv = read_mv(bits, ctx.mv_probs).add(best);
        }
        _ => {
            let best = near.best.clamp(&ctx.bounds);
            read_split(bits, ctx, best, mb, sub);
            return;
        }
    }
    sub.mvs = [mb.mv; 16];
}

fn read_split(
    bits: &mut BoolDecoder<'_>,
    ctx: &ModeContext<'_>,
    best: Mv,
    mb: &mut MbInfo,
    sub: &mut SubInfo,
) {
    let partition = bits.tree(&SPLIT_MV_TREE, &SPLIT_MV_PROBS);
    mb.partition = partition;
    let map = &MV_PARTITIONS[usize::from(partition & 3)];
    let count = SPLIT_COUNTS[usize::from(partition & 3)];
    for j in 0..count {
        let Some(k) = map.iter().position(|&p| usize::from(p) == j) else {
            continue;
        };
        let left = if k & 3 == 0 {
            ctx.left_sub.map_or(Mv::ZERO, |s| s.mvs[k + 3])
        } else {
            sub.mvs[k - 1]
        };
        let above = if k < 4 {
            ctx.above_sub.map_or(Mv::ZERO, |s| s.mvs[k + 12])
        } else {
            sub.mvs[k - 4]
        };
        let mode = bits.tree(
            &SUB_MV_REF_TREE,
            &SUB_MV_REF_PROBS[sub_mv_context(left, above)],
        );
        let mv = match mode {
            LEFT_4X4 => left,
            ABOVE_4X4 => above,
            ZERO_4X4 => Mv::ZERO,
            // NEW_4X4.
            _ => read_mv(bits, ctx.mv_probs).add(best),
        };
        for (slot, &p) in sub.mvs.iter_mut().zip(map.iter()).skip(k) {
            if usize::from(p) == j {
                *slot = mv;
            }
        }
    }
    mb.mv = sub.mvs[15];
}

#[cfg(test)]
mod tests {
    use super::{GOLDEN, INTRA, LAST, MbInfo, Mv, SPLITMV, find_near, sub_mv_context};

    fn inter(mv: Mv, r: u8) -> MbInfo {
        MbInfo {
            ref_frame: r,
            mv,
            y_mode: 5,
            ..MbInfo::default()
        }
    }

    #[test]
    fn near_mvs_merge_and_swap_like_section_16_3() {
        let bias = [false; 4];
        // All outside or intra: everything zero, zero count zero.
        let near = find_near(None, None, None, &bias, LAST);
        assert_eq!(
            (near.best, near.nearest, near.near),
            (Mv::ZERO, Mv::ZERO, Mv::ZERO)
        );

        let a = inter(Mv::new(4, 8), LAST);
        let l = inter(Mv::new(4, 8), LAST);
        let al = inter(Mv::new(-2, 6), LAST);
        let near = find_near(Some(&a), Some(&l), Some(&al), &bias, LAST);
        assert_eq!(near.nearest, Mv::new(4, 8));
        assert_eq!(near.near, Mv::new(-2, 6));
        assert_eq!(near.best, Mv::new(4, 8));

        // Near outweighs nearest: they swap.
        let a = inter(Mv::new(1, 1), LAST);
        let l = inter(Mv::new(2, 2), LAST);
        let al = inter(Mv::new(2, 2), LAST);
        let near = find_near(Some(&a), Some(&l), Some(&al), &bias, LAST);
        assert_eq!(near.nearest, Mv::new(2, 2));
        assert_eq!(near.near, Mv::new(1, 1));

        // Sign bias flips a golden neighbour's vector for a last-frame block.
        let mut bias = [false; 4];
        bias[usize::from(GOLDEN)] = true;
        let a = inter(Mv::new(4, -8), GOLDEN);
        let near = find_near(Some(&a), None, None, &bias, LAST);
        assert_eq!(near.nearest, Mv::new(-4, 8));

        // Intra neighbours do not count; split neighbours count for mode probs.
        let i = MbInfo {
            ref_frame: INTRA,
            ..MbInfo::default()
        };
        let mut s = inter(Mv::new(4, 4), LAST);
        s.y_mode = SPLITMV;
        let with_split = find_near(Some(&s), Some(&i), None, &[false; 4], LAST);
        let without = find_near(
            Some(&inter(Mv::new(4, 4), LAST)),
            Some(&i),
            None,
            &[false; 4],
            LAST,
        );
        assert_eq!(with_split.nearest, without.nearest);
        assert_ne!(with_split.probs[3], without.probs[3]);
    }

    #[test]
    fn sub_mv_contexts() {
        let z = Mv::ZERO;
        let a = Mv::new(1, 0);
        let b = Mv::new(0, 1);
        assert_eq!(sub_mv_context(z, z), 4);
        assert_eq!(sub_mv_context(a, a), 3);
        assert_eq!(sub_mv_context(a, z), 2);
        assert_eq!(sub_mv_context(z, a), 1);
        assert_eq!(sub_mv_context(a, b), 0);
    }
}
