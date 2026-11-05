//! Signalling costs of inter mode decisions, in bits, from the tile's
//! current CDFs and the contexts the syntax will use (as `read_is_inter`,
//! `read_ref_frames`, `inter_block_mode_info` and `read_mv` compute them).

use crate::dec::coder::Coder;
use crate::dec::tile::TileDec;
use crate::dec::types::{
    ALTREF_FRAME, ALTREF2_FRAME, BWDREF_FRAME, GLOBALMV, GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME,
    LAST2_FRAME, LAST3_FRAME, NEAR_NEARMV, NEAR_NEWMV, NEAREST_NEARESTMV, NEARMV, NEW_NEARMV,
    NEW_NEWMV, NEWMV, NONE, bh4, bw4,
};
use crate::dsp::Pixel;
use crate::spec::tables::COMPOUND_MODE_CTX_MAP;

use super::msac_enc::{bool_cost, symbol_cost};

#[inline]
fn bb(cdf: &[u16; 2], v: bool) -> f64 {
    f64::from(bool_cost(cdf[0], v)) / 256.0
}

#[inline]
fn sb(cdf: &[u16], v: usize) -> f64 {
    f64::from(symbol_cost(cdf, v)) / 256.0
}

impl<P: Pixel, C: Coder> TileDec<'_, P, C> {
    fn nb_refs(&self) -> ([i8; 2], [i8; 2]) {
        let b = &self.b;
        let above = if b.avail_u {
            self.mi(b.mi_row - 1, b.mi_col).ref_frame
        } else {
            [INTRA_FRAME, NONE]
        };
        let left = if b.avail_l {
            self.mi(b.mi_row, b.mi_col - 1).ref_frame
        } else {
            [INTRA_FRAME, NONE]
        };
        (above, left)
    }

    fn nb_ctx(&self, a: &[i8], b2: &[i8], above: [i8; 2], left: [i8; 2]) -> usize {
        let count = |f: i8| -> u32 {
            let mut c = 0;
            if self.b.avail_u {
                c += u32::from(above[0] == f) + u32::from(above[1] == f);
            }
            if self.b.avail_l {
                c += u32::from(left[0] == f) + u32::from(left[1] == f);
            }
            c
        };
        let c0: u32 = a.iter().map(|&f| count(f)).sum();
        let c1: u32 = b2.iter().map(|&f| count(f)).sum();
        match c0.cmp(&c1) {
            std::cmp::Ordering::Less => 0,
            std::cmp::Ordering::Equal => 1,
            std::cmp::Ordering::Greater => 2,
        }
    }

    /// Bits of `is_inter` for the block in `self.b`.
    pub(crate) fn is_inter_bits(&self, inter: bool) -> f64 {
        let (au, al) = (self.b.avail_u, self.b.avail_l);
        let ai = au && self.mi(self.b.mi_row - 1, self.b.mi_col).ref_frame[0] <= INTRA_FRAME;
        let li = al && self.mi(self.b.mi_row, self.b.mi_col - 1).ref_frame[0] <= INTRA_FRAME;
        let ctx = if au && al {
            if li && ai { 3 } else { usize::from(li || ai) }
        } else if au || al {
            2 * usize::from(if au { ai } else { li })
        } else {
            0
        };
        bb(&self.cdf.is_inter[ctx], inter)
    }

    /// Bits of the reference frame syntax for `refs` (single or compound).
    pub(crate) fn ref_bits(&self, refs: [i8; 2]) -> f64 {
        let (above, left) = self.nb_refs();
        let [pr0, pr1] = refs;
        let (au, al) = (self.b.avail_u, self.b.avail_l);
        let above_intra = above[0] <= INTRA_FRAME;
        let left_intra = left[0] <= INTRA_FRAME;
        let above_single = above[1] <= INTRA_FRAME;
        let left_single = left[1] <= INTRA_FRAME;
        let bwd = |f: i8| (BWDREF_FRAME..=ALTREF_FRAME).contains(&f);
        let comp = pr1 > INTRA_FRAME;
        let mut bits = 0.0;
        if self.hdr.reference_select && bw4(self.b.bsize).min(bh4(self.b.bsize)) >= 2 {
            let ctx = if au && al {
                if above_single && left_single {
                    usize::from(bwd(above[0]) ^ bwd(left[0]))
                } else if above_single {
                    2 + usize::from(bwd(above[0]) || above_intra)
                } else if left_single {
                    2 + usize::from(bwd(left[0]) || left_intra)
                } else {
                    4
                }
            } else if au {
                if above_single {
                    usize::from(bwd(above[0]))
                } else {
                    3
                }
            } else if al {
                if left_single {
                    usize::from(bwd(left[0]))
                } else {
                    3
                }
            } else {
                1
            };
            bits += bb(&self.cdf.comp_mode[ctx], comp);
        }
        const FWD: [i8; 4] = [LAST_FRAME, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME];
        const BWD: [i8; 3] = [BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME];
        let cdf = &self.cdf;
        if comp {
            let samedir = |a: i8, b: i8| (a >= BWDREF_FRAME) == (b >= BWDREF_FRAME);
            let above_comp = au && !above_intra && !above_single;
            let left_comp = al && !left_intra && !left_single;
            let above_uni = above_comp && samedir(above[0], above[1]);
            let left_uni = left_comp && samedir(left[0], left[1]);
            let ctx = if au && !above_intra && al && !left_intra {
                let sd = usize::from(samedir(above[0], left[0]));
                if !above_comp && !left_comp {
                    1 + 2 * sd
                } else if !above_comp {
                    if !left_uni { 1 } else { 3 + sd }
                } else if !left_comp {
                    if !above_uni { 1 } else { 3 + sd }
                } else if !above_uni && !left_uni {
                    0
                } else if !above_uni || !left_uni {
                    2
                } else {
                    3 + usize::from((above[0] == BWDREF_FRAME) == (left[0] == BWDREF_FRAME))
                }
            } else if au && al {
                if above_comp {
                    1 + 2 * usize::from(above_uni)
                } else if left_comp {
                    1 + 2 * usize::from(left_uni)
                } else {
                    2
                }
            } else if above_comp {
                4 * usize::from(above_uni)
            } else if left_comp {
                4 * usize::from(left_uni)
            } else {
                2
            };
            let bidir = (pr0 >= BWDREF_FRAME) != (pr1 >= BWDREF_FRAME);
            bits += bb(&cdf.comp_ref_type[ctx], bidir);
            if !bidir {
                let ctx = self.nb_ctx(&FWD, &BWD, above, left);
                bits += bb(&cdf.uni_comp_ref[ctx][0], pr0 == BWDREF_FRAME);
                if pr0 != BWDREF_FRAME {
                    let ctx1 =
                        self.nb_ctx(&[LAST2_FRAME], &[LAST3_FRAME, GOLDEN_FRAME], above, left);
                    bits += bb(&cdf.uni_comp_ref[ctx1][1], pr1 != LAST2_FRAME);
                    if pr1 != LAST2_FRAME {
                        let ctx2 = self.nb_ctx(&[LAST3_FRAME], &[GOLDEN_FRAME], above, left);
                        bits += bb(&cdf.uni_comp_ref[ctx2][2], pr1 == GOLDEN_FRAME);
                    }
                }
            } else {
                let ctx = self.nb_ctx(
                    &[LAST_FRAME, LAST2_FRAME],
                    &[LAST3_FRAME, GOLDEN_FRAME],
                    above,
                    left,
                );
                let far = pr0 == LAST3_FRAME || pr0 == GOLDEN_FRAME;
                bits += bb(&cdf.comp_ref[ctx][0], far);
                if !far {
                    let ctx1 = self.nb_ctx(&[LAST_FRAME], &[LAST2_FRAME], above, left);
                    bits += bb(&cdf.comp_ref[ctx1][1], pr0 == LAST2_FRAME);
                } else {
                    let ctx2 = self.nb_ctx(&[LAST3_FRAME], &[GOLDEN_FRAME], above, left);
                    bits += bb(&cdf.comp_ref[ctx2][2], pr0 == GOLDEN_FRAME);
                }
                let ctx = self.nb_ctx(&[BWDREF_FRAME, ALTREF2_FRAME], &[ALTREF_FRAME], above, left);
                bits += bb(&cdf.comp_bwd_ref[ctx][0], pr1 == ALTREF_FRAME);
                if pr1 != ALTREF_FRAME {
                    let ctx1 = self.nb_ctx(&[BWDREF_FRAME], &[ALTREF2_FRAME], above, left);
                    bits += bb(&cdf.comp_bwd_ref[ctx1][1], pr1 == ALTREF2_FRAME);
                }
            }
        } else {
            let ctx = self.nb_ctx(&FWD, &BWD, above, left);
            let b = pr0 >= BWDREF_FRAME;
            bits += bb(&cdf.single_ref[ctx][0], b);
            if b {
                let ctx2 =
                    self.nb_ctx(&[BWDREF_FRAME, ALTREF2_FRAME], &[ALTREF_FRAME], above, left);
                bits += bb(&cdf.single_ref[ctx2][1], pr0 == ALTREF_FRAME);
                if pr0 != ALTREF_FRAME {
                    let ctx6 = self.nb_ctx(&[BWDREF_FRAME], &[ALTREF2_FRAME], above, left);
                    bits += bb(&cdf.single_ref[ctx6][5], pr0 == ALTREF2_FRAME);
                }
            } else {
                let ctx3 = self.nb_ctx(
                    &[LAST_FRAME, LAST2_FRAME],
                    &[LAST3_FRAME, GOLDEN_FRAME],
                    above,
                    left,
                );
                let far = pr0 == LAST3_FRAME || pr0 == GOLDEN_FRAME;
                bits += bb(&cdf.single_ref[ctx3][2], far);
                if far {
                    let ctx5 = self.nb_ctx(&[LAST3_FRAME], &[GOLDEN_FRAME], above, left);
                    bits += bb(&cdf.single_ref[ctx5][4], pr0 == GOLDEN_FRAME);
                } else {
                    let ctx4 = self.nb_ctx(&[LAST_FRAME], &[LAST2_FRAME], above, left);
                    bits += bb(&cdf.single_ref[ctx4][3], pr0 == LAST2_FRAME);
                }
            }
        }
        bits
    }

    /// Bits of the inter mode and its `ref_mv_idx` (drl) syntax. The motion
    /// vector stack of the block's references must be current (`find_mv_stack`).
    pub(crate) fn mode_bits(&self, compound: bool, mode: u8, idx: usize) -> f64 {
        let b = &self.b;
        let cdf = &self.cdf;
        let mut bits = if compound {
            let ctx = COMPOUND_MODE_CTX_MAP[b.ref_mv_ctx >> 1][b.new_mv_ctx.min(4)] as usize;
            sb(
                &cdf.compound_mode[ctx],
                mode.saturating_sub(NEAREST_NEARESTMV) as usize,
            )
        } else {
            let mut s = bb(&cdf.new_mv[b.new_mv_ctx], mode != NEWMV);
            if mode != NEWMV {
                s += bb(&cdf.zero_mv[b.zero_mv_ctx], mode != GLOBALMV);
                if mode != GLOBALMV {
                    s += bb(&cdf.ref_mv[b.ref_mv_ctx], mode == NEARMV);
                }
            }
            s
        };
        if mode == NEWMV || mode == NEW_NEWMV {
            for i in 0..2 {
                if b.num_mv_found > i + 1 {
                    let ctx = b.drl_ctx_stack[i] as usize;
                    bits += bb(&cdf.drl_mode[ctx], idx > i);
                    if idx <= i {
                        break;
                    }
                }
            }
        } else if matches!(mode, NEARMV | NEAR_NEARMV | NEAR_NEWMV | NEW_NEARMV) {
            for i in 1..3 {
                if b.num_mv_found > i + 1 {
                    let ctx = b.drl_ctx_stack[i] as usize;
                    bits += bb(&cdf.drl_mode[ctx], idx > i);
                    if idx <= i {
                        break;
                    }
                }
            }
        }
        bits
    }

    /// Bits of a motion vector difference (1/8 sample units).
    pub(crate) fn mv_bits(&self, d: (i32, i32)) -> f64 {
        let cdf = &self.cdf;
        let ctx = usize::from(self.b.use_intrabc);
        let joint = usize::from(d.0 != 0) * 2 + usize::from(d.1 != 0);
        let mut bits = sb(&cdf.mv_joint[ctx], joint);
        for (comp, v) in [(0usize, d.0), (1, d.1)] {
            if v == 0 {
                continue;
            }
            let z = (v.unsigned_abs().max(1) - 1) as usize;
            let class = if z >= 2 * 4096 {
                10
            } else if z < 16 {
                0
            } else {
                (31 - ((z >> 3) as u32).leading_zeros()) as usize
            };
            let base = if class == 0 { 0 } else { 2 << (class + 2) };
            let off = z - base;
            let (int, fr, hp) = (off >> 3, (off >> 1) & 3, off & 1);
            bits += bb(&cdf.mv_sign[ctx][comp], v < 0);
            bits += sb(&cdf.mv_class[ctx][comp], class);
            let int_bits = if class == 0 {
                let bit = usize::from(int != 0);
                bb(&cdf.mv_class0_bit[ctx][comp], int != 0)
                    + if self.hdr.force_integer_mv {
                        0.0
                    } else {
                        sb(&cdf.mv_class0_fr[ctx][comp][bit], fr)
                    }
                    + if self.hdr.allow_high_precision_mv {
                        bb(&cdf.mv_class0_hp[ctx][comp], hp != 0)
                    } else {
                        0.0
                    }
            } else {
                let mut s = 0.0;
                for i in 0..class {
                    s += bb(&cdf.mv_bit[ctx][comp][i], (int >> i) & 1 != 0);
                }
                s + if self.hdr.force_integer_mv {
                    0.0
                } else {
                    sb(&cdf.mv_fr[ctx][comp], fr)
                } + if self.hdr.allow_high_precision_mv {
                    bb(&cdf.mv_hp[ctx][comp], hp != 0)
                } else {
                    0.0
                }
            };
            bits += int_bits;
        }
        bits
    }
}
