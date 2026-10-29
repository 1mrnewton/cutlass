//! Inter-frame mode and motion-vector decode.
//!
//! The order matches libvpx `read_inter_frame_mode_info`, `dec_find_mv_refs`,
//! and `assign_mv`. Reference motion from the previous frame is left to the
//! caller: error-resilient frames never use it.

use cutlass_decode::CodecError;

use super::mc::{self, Mv, RefPlane};
use super::{BlockAt, BlockPos, Mi, Scratch};
use crate::bits::BoolDecoder;
use crate::geom::{
    self, ALTREF_FRAME, BLOCK_4X4, BLOCK_4X8, BLOCK_8X4, BLOCK_8X8, COMPOUND_REFERENCE,
    GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME, NEAR, NEAREST, NEW, REFERENCE_SELECT, ZERO,
};
use crate::prob::{self, Counts};

const SENTINEL_FILTER: u8 = 3;
const MV_LOW: i32 = -16_384;
const MV_UPP: i32 = 16_383;

const MODE_COUNTER: [i32; 14] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1];
const COUNTER_CTX: [u8; 19] = [2, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6];

/// Neighbor offsets in mode-info units, libvpx `mv_ref_blocks`.
const MV_REF: [[[i8; 2]; 8]; 13] = [
    near8(),
    near8(),
    near8(),
    near8(),
    [
        [0, -1],
        [-1, 0],
        [1, -1],
        [-1, -1],
        [0, -2],
        [-2, 0],
        [-2, -1],
        [-1, -2],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, 1],
        [-1, -1],
        [-2, 0],
        [0, -2],
        [-1, -2],
        [-2, -1],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, 1],
        [1, -1],
        [-1, -1],
        [-3, 0],
        [0, -3],
        [-3, -3],
    ],
    [
        [0, -1],
        [-1, 0],
        [2, -1],
        [-1, -1],
        [-1, 1],
        [0, -3],
        [-3, 0],
        [-3, -3],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, 2],
        [-1, -1],
        [1, -1],
        [-3, 0],
        [0, -3],
        [-3, -3],
    ],
    [
        [-1, 1],
        [1, -1],
        [-1, 2],
        [2, -1],
        [-1, -1],
        [-3, 0],
        [0, -3],
        [-3, -3],
    ],
    [
        [0, -1],
        [-1, 0],
        [4, -1],
        [-1, 2],
        [-1, -1],
        [0, -3],
        [-3, 0],
        [2, -1],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, 4],
        [2, -1],
        [-1, -1],
        [-3, 0],
        [0, -3],
        [-1, 2],
    ],
    [
        [-1, 3],
        [3, -1],
        [-1, 4],
        [4, -1],
        [-1, -1],
        [-1, 0],
        [0, -1],
        [-1, 6],
    ],
];

const SUBBLOCK: [[u8; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];

const fn near8() -> [[i8; 2]; 8] {
    [
        [-1, 0],
        [0, -1],
        [-1, -1],
        [-2, 0],
        [0, -2],
        [-2, -1],
        [-1, -2],
        [-2, -2],
    ]
}

struct CandidateSearch<'a> {
    list: &'a mut [Mv; 2],
    count: &'a mut usize,
    different: &'a mut bool,
    early: bool,
    done: &'a mut bool,
}

struct MvQuery {
    mode: u8,
    frame: u8,
    bsize: u8,
    mi_row: usize,
    mi_col: usize,
    block: i32,
    bw: usize,
    bh: usize,
}

struct Paint {
    frame: u8,
    plane: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv_row: i32,
    mv_col: i32,
    filter: u8,
    average: bool,
}

impl<'a> Scratch<'a> {
    pub(super) fn read_inter_frame(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        at: BlockAt,
    ) -> Result<Mi, CodecError> {
        let mut mi = Mi::default();
        mi.bsize = at.bsize;
        mi.seg = self.read_segment(reader, at.mi_row, at.mi_col, at.x_mis, at.y_mis)?;
        mi.skip = self.read_skip(reader, at.mi_row, at.mi_col, mi.seg)?;
        let inter = self.read_is_inter(reader, at.mi_row, at.mi_col, mi.seg)?;
        let allow = !mi.skip || !inter;
        mi.tx = self.read_tx(reader, at.bsize, at.mi_row, at.mi_col, allow)?;
        if inter {
            self.read_inter_block(reader, &mut mi, &at)?;
        } else {
            self.read_inter_intra(reader, &mut mi)?;
        }
        Ok(mi)
    }

    pub(super) fn recon_inter(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &Mi,
        pos: BlockPos,
    ) -> Result<(), CodecError> {
        self.predict_inter(mi, pos)?;
        if mi.skip {
            return Ok(());
        }
        for plane in 0..3 {
            self.recon_plane(reader, mi, plane, pos, true)?;
        }
        Ok(())
    }

    fn read_is_inter(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
        seg: u8,
    ) -> Result<bool, CodecError> {
        let seg_i = usize::from(seg);
        if self.header.seg.enabled && self.header.seg.feature_on[seg_i][2] {
            return Ok(self.header.seg.feature_data[seg_i][2] != 0);
        }
        let above = self.above_mi(mi_row, mi_col);
        let left = self.left_mi(mi_row, mi_col);
        let ctx = intra_inter_ctx(above, left);
        let inter = reader.read(u32::from(self.fc.intra_inter[ctx]))? == 1;
        bump2(
            self.collect,
            &mut self.counts.intra_inter,
            ctx,
            usize::from(inter),
        );
        Ok(inter)
    }

    fn read_inter_intra(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &mut Mi,
    ) -> Result<(), CodecError> {
        let group = if mi.bsize < BLOCK_8X8 {
            0
        } else {
            geom::size_group(mi.bsize)
        };
        match mi.bsize {
            BLOCK_4X4 => {
                for slot in &mut mi.bmi {
                    *slot = self.read_y_mode(reader, group)?;
                }
                mi.mode = mi.bmi[3];
            }
            BLOCK_4X8 => {
                let top = self.read_y_mode(reader, group)?;
                mi.bmi[0] = top;
                mi.bmi[2] = top;
                let bottom = self.read_y_mode(reader, group)?;
                mi.bmi[1] = bottom;
                mi.bmi[3] = bottom;
                mi.mode = bottom;
            }
            BLOCK_8X4 => {
                let left = self.read_y_mode(reader, group)?;
                mi.bmi[0] = left;
                mi.bmi[1] = left;
                let right = self.read_y_mode(reader, group)?;
                mi.bmi[2] = right;
                mi.bmi[3] = right;
                mi.mode = right;
            }
            _ => {
                let mode = self.read_y_mode(reader, group)?;
                mi.mode = mode;
                mi.bmi = [mode; 4];
            }
        }
        mi.uv = self.read_uv_mode(reader, mi.mode)?;
        mi.interp = SENTINEL_FILTER;
        mi.refs = [INTRA_FRAME, INTRA_FRAME];
        Ok(())
    }

    fn read_y_mode(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        group: usize,
    ) -> Result<u8, CodecError> {
        let mode = prob::read_tree(reader, &prob::INTRA_TREE, &self.fc.y_mode[group])?;
        let which = usize::from(mode);
        if self.collect {
            self.counts.y_mode[group][which] = self.counts.y_mode[group][which].saturating_add(1);
        }
        Ok(mode)
    }

    fn read_uv_mode(&mut self, reader: &mut BoolDecoder<'_>, y_mode: u8) -> Result<u8, CodecError> {
        let index = usize::from(y_mode);
        let mode = prob::read_tree(reader, &prob::INTRA_TREE, &self.fc.uv_mode[index])?;
        let which = usize::from(mode);
        if self.collect {
            self.counts.uv_mode[index][which] = self.counts.uv_mode[index][which].saturating_add(1);
        }
        Ok(mode)
    }

    fn read_inter_block(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &mut Mi,
        at: &BlockAt,
    ) -> Result<(), CodecError> {
        self.read_refs(reader, mi, at.mi_row, at.mi_col)?;
        let compound = has_second(mi);
        let ctx = self.mode_context(mi.bsize, at.mi_row, at.mi_col)?;
        if self.seg_skip(mi.seg) {
            if mi.bsize < BLOCK_8X8 {
                return Err(super::bad("segment skip"));
            }
            mi.mode = ZERO;
        } else if mi.bsize >= BLOCK_8X8 {
            mi.mode = self.read_inter_mode(reader, ctx)?;
        }
        mi.interp = self.read_block_interp(reader, at.mi_row, at.mi_col)?;
        if mi.bsize < BLOCK_8X8 {
            self.read_sub_mvs(reader, mi, at, ctx, compound)?;
        } else if mi.mode == ZERO {
            mi.mv = [Mv::default(); 2];
        } else {
            self.read_block_mv(reader, mi, at, compound)?;
        }
        Ok(())
    }

    fn seg_skip(&self, seg: u8) -> bool {
        let seg_i = usize::from(seg);
        self.header.seg.enabled && self.header.seg.feature_on[seg_i][3]
    }

    fn read_refs(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &mut Mi,
        mi_row: usize,
        mi_col: usize,
    ) -> Result<(), CodecError> {
        let seg_i = usize::from(mi.seg);
        if self.header.seg.enabled && self.header.seg.feature_on[seg_i][2] {
            mi.refs[0] = u8::try_from(self.header.seg.feature_data[seg_i][2]).unwrap_or(0);
            mi.refs[1] = INTRA_FRAME;
            return Ok(());
        }
        let mode = self.reference_mode(reader, mi_row, mi_col)?;
        if mode == COMPOUND_REFERENCE {
            self.read_compound(reader, mi, mi_row, mi_col)
        } else {
            self.read_single(reader, mi, mi_row, mi_col)
        }
    }

    fn reference_mode(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
    ) -> Result<u8, CodecError> {
        if self.header.reference_mode != REFERENCE_SELECT {
            return Ok(self.header.reference_mode);
        }
        let above = self.above_mi(mi_row, mi_col);
        let left = self.left_mi(mi_row, mi_col);
        let ctx = reference_mode_ctx(above, left, self.header.comp_fixed);
        let mode = u8::try_from(reader.read(u32::from(self.fc.comp_inter[ctx]))?).unwrap_or(0);
        bump2(
            self.collect,
            &mut self.counts.comp_inter,
            ctx,
            usize::from(mode),
        );
        Ok(mode)
    }

    fn read_single(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &mut Mi,
        mi_row: usize,
        mi_col: usize,
    ) -> Result<(), CodecError> {
        let ctx0 = {
            let above = self.above_mi(mi_row, mi_col);
            let left = self.left_mi(mi_row, mi_col);
            single_p1(above, left)
        };
        let bit0 =
            usize::try_from(reader.read(u32::from(self.fc.single_ref[ctx0][0]))?).unwrap_or(0);
        self.bump_single(ctx0, 0, bit0);
        if bit0 == 1 {
            let ctx1 = {
                let above = self.above_mi(mi_row, mi_col);
                let left = self.left_mi(mi_row, mi_col);
                single_p2(above, left)
            };
            let bit1 =
                usize::try_from(reader.read(u32::from(self.fc.single_ref[ctx1][1]))?).unwrap_or(0);
            self.bump_single(ctx1, 1, bit1);
            mi.refs[0] = if bit1 == 1 {
                ALTREF_FRAME
            } else {
                GOLDEN_FRAME
            };
        } else {
            mi.refs[0] = LAST_FRAME;
        }
        mi.refs[1] = INTRA_FRAME;
        Ok(())
    }

    fn read_compound(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &mut Mi,
        mi_row: usize,
        mi_col: usize,
    ) -> Result<(), CodecError> {
        let above = self.above_mi(mi_row, mi_col);
        let left = self.left_mi(mi_row, mi_col);
        let ctx = comp_ref_ctx(above, left, self.header);
        let bit = usize::try_from(reader.read(u32::from(self.fc.comp_ref[ctx]))?).unwrap_or(0);
        bump2(self.collect, &mut self.counts.comp_ref, ctx, bit);
        let fixed = usize::from(self.header.sign_bias[usize::from(self.header.comp_fixed)]);
        mi.refs[fixed] = self.header.comp_fixed;
        mi.refs[fixed ^ 1] = self.header.comp_var[bit];
        Ok(())
    }

    fn read_inter_mode(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        ctx: usize,
    ) -> Result<u8, CodecError> {
        let which = usize::from(prob::read_tree(
            reader,
            &prob::INTER_TREE,
            &self.fc.inter_mode[ctx],
        )?);
        if self.collect {
            self.counts.inter_mode[ctx][which] =
                self.counts.inter_mode[ctx][which].saturating_add(1);
        }
        Ok(NEAREST + u8::try_from(which).unwrap_or(0))
    }

    fn read_block_interp(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
    ) -> Result<u8, CodecError> {
        if self.header.interp != geom::SWITCHABLE {
            return Ok(self.header.interp);
        }
        let left = self.left_mi(mi_row, mi_col).map(|mi| mi.interp);
        let above = self.above_mi(mi_row, mi_col).map(|mi| mi.interp);
        let ctx = interp_ctx(left, above);
        let kind = prob::read_tree(reader, &prob::INTERP_TREE, &self.fc.interp[ctx])?;
        let which = usize::from(kind);
        if self.collect {
            self.counts.interp[ctx][which] = self.counts.interp[ctx][which].saturating_add(1);
        }
        Ok(kind)
    }

    fn read_block_mv(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &mut Mi,
        at: &BlockAt,
        compound: bool,
    ) -> Result<(), CodecError> {
        let mut best = [Mv::default(); 2];
        let nref = 1 + usize::from(compound);
        for (ref_i, slot) in best.iter_mut().enumerate().take(nref) {
            let query = MvQuery {
                mode: mi.mode,
                frame: mi.refs[ref_i],
                bsize: mi.bsize,
                mi_row: at.mi_row,
                mi_col: at.mi_col,
                block: -1,
                bw: at.bw,
                bh: at.bh,
            };
            let (list, count) = self.find_mv(&query)?;
            *slot = lower_precision(list[count - 1], self.header.allow_hp);
        }
        self.assign(reader, mi.mode, &mut mi.mv, &best, &best, compound)
    }

    fn read_sub_mvs(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &mut Mi,
        at: &BlockAt,
        ctx: usize,
        compound: bool,
    ) -> Result<(), CodecError> {
        let (step_w, step_h) = sub_steps(mi.bsize);
        let mut best_ref = [Mv::default(); 2];
        let mut got_new = false;
        let mut last = ZERO;
        for idy in (0..2).step_by(step_h) {
            for idx in (0..2).step_by(step_w) {
                let j = idy * 2 + idx;
                last = self.read_one_sub(
                    reader,
                    mi,
                    at,
                    SubStep {
                        ctx,
                        compound,
                        block: i32::try_from(j).unwrap_or(0),
                        got_new: &mut got_new,
                        best_ref: &mut best_ref,
                    },
                )?;
                if step_h == 2 {
                    mi.sub_mv[j + 2] = mi.sub_mv[j];
                }
                if step_w == 2 {
                    mi.sub_mv[j + 1] = mi.sub_mv[j];
                }
            }
        }
        mi.mode = last;
        mi.mv = mi.sub_mv[3];
        Ok(())
    }

    fn read_one_sub(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &mut Mi,
        at: &BlockAt,
        step: SubStep<'_>,
    ) -> Result<u8, CodecError> {
        let b_mode = self.read_inter_mode(reader, step.ctx)?;
        let mut near = [Mv::default(); 2];
        let nref = 1 + usize::from(step.compound);
        if b_mode == NEAREST || b_mode == NEAR {
            for (ref_i, slot) in near.iter_mut().enumerate().take(nref) {
                *slot = self.sub_candidate(mi, at, b_mode, step.block, ref_i)?;
            }
        } else if b_mode == NEW && !*step.got_new {
            self.cache_new_ref(mi, at, step.best_ref, nref)?;
            *step.got_new = true;
        }
        self.assign(
            reader,
            b_mode,
            &mut mi.sub_mv[usize::try_from(step.block).unwrap_or(0)],
            step.best_ref,
            &near,
            step.compound,
        )?;
        Ok(b_mode)
    }

    fn cache_new_ref(
        &mut self,
        mi: &Mi,
        at: &BlockAt,
        best_ref: &mut [Mv; 2],
        nref: usize,
    ) -> Result<(), CodecError> {
        for (ref_i, slot) in best_ref.iter_mut().enumerate().take(nref) {
            let query = MvQuery {
                mode: NEW,
                frame: mi.refs[ref_i],
                bsize: mi.bsize,
                mi_row: at.mi_row,
                mi_col: at.mi_col,
                block: -1,
                bw: at.bw,
                bh: at.bh,
            };
            let (list, _) = self.find_mv(&query)?;
            *slot = lower_precision(list[0], self.header.allow_hp);
        }
        Ok(())
    }

    fn sub_candidate(
        &mut self,
        mi: &Mi,
        at: &BlockAt,
        b_mode: u8,
        block: i32,
        ref_i: usize,
    ) -> Result<Mv, CodecError> {
        if block == 0 {
            return self.listed_mv(mi, at, b_mode, block, ref_i, true);
        }
        if b_mode == NEAREST {
            let from = if block == 3 { 2 } else { 0 };
            return Ok(mi.sub_mv[from][ref_i]);
        }
        if block == 3 {
            if mi.sub_mv[2][ref_i] != mi.sub_mv[1][ref_i] {
                return Ok(mi.sub_mv[1][ref_i]);
            }
            if mi.sub_mv[2][ref_i] != mi.sub_mv[0][ref_i] {
                return Ok(mi.sub_mv[0][ref_i]);
            }
        }
        self.listed_mv(mi, at, b_mode, block, ref_i, false)
    }

    fn listed_mv(
        &mut self,
        mi: &Mi,
        at: &BlockAt,
        b_mode: u8,
        block: i32,
        ref_i: usize,
        first: bool,
    ) -> Result<Mv, CodecError> {
        let query = MvQuery {
            mode: b_mode,
            frame: mi.refs[ref_i],
            bsize: mi.bsize,
            mi_row: at.mi_row,
            mi_col: at.mi_col,
            block,
            bw: at.bw,
            bh: at.bh,
        };
        let (list, count) = self.find_mv(&query)?;
        if first {
            return Ok(list[count - 1]);
        }
        let mut best = Mv::default();
        let prior = if block == 3 { 2 } else { 0 };
        for candidate in list {
            if mi.sub_mv[prior][ref_i] != candidate {
                best = candidate;
                break;
            }
        }
        Ok(best)
    }

    fn assign(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mode: u8,
        mv: &mut [Mv; 2],
        best_ref: &[Mv; 2],
        near: &[Mv; 2],
        compound: bool,
    ) -> Result<(), CodecError> {
        let nref = 1 + usize::from(compound);
        match mode {
            NEW => {
                for ref_i in 0..nref {
                    let diff = self.read_mv(reader, best_ref[ref_i])?;
                    mv[ref_i] = add_mv(best_ref[ref_i], diff)?;
                }
            }
            NEAR | NEAREST => {
                mv[0] = near[0];
                if compound {
                    mv[1] = near[1];
                }
            }
            ZERO => *mv = [Mv::default(); 2],
            _ => return Err(super::bad("inter mode")),
        }
        Ok(())
    }

    fn read_mv(&mut self, reader: &mut BoolDecoder<'_>, reference: Mv) -> Result<Mv, CodecError> {
        let joint = usize::from(prob::read_tree(
            reader,
            &prob::JOINT_TREE,
            &self.fc.mv.joints,
        )?);
        let use_hp = self.header.allow_hp
            && reference.row.unsigned_abs() < 64
            && reference.col.unsigned_abs() < 64;
        let mut diff = Mv::default();
        if joint == 2 || joint == 3 {
            diff.row = component_i16(self.read_component(reader, 0, use_hp)?)?;
        }
        if joint == 1 || joint == 3 {
            diff.col = component_i16(self.read_component(reader, 1, use_hp)?)?;
        }
        if self.collect {
            count_mv(&mut self.counts, diff);
        }
        Ok(diff)
    }

    fn read_component(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        which: usize,
        use_hp: bool,
    ) -> Result<i32, CodecError> {
        let comp = &self.fc.mv.comp[which];
        let sign = reader.read(u32::from(comp.sign))? == 1;
        let class = usize::from(prob::read_tree(reader, &prob::CLASS_TREE, &comp.classes)?);
        let (mag_base, class_bit) = if class == 0 {
            let bit = i32::try_from(reader.read(u32::from(comp.class0))?).unwrap_or(0);
            (0, bit)
        } else {
            let n = class;
            let mut bits = 0i32;
            for (i, prob) in comp.bits.iter().take(n).enumerate() {
                let bit = i32::try_from(reader.read(u32::from(*prob))?).unwrap_or(0);
                bits |= bit << i;
            }
            (2i32 << (u32::try_from(class).unwrap_or(0) + 2), bits)
        };
        let fp_probs = if class == 0 {
            &comp.class0_fp[usize::try_from(class_bit).unwrap_or(0)]
        } else {
            &comp.fp
        };
        let frac = i32::from(prob::read_tree(reader, &prob::FP_TREE, fp_probs)?);
        let hp_prob = if class == 0 { comp.class0_hp } else { comp.hp };
        let hp = if use_hp {
            i32::try_from(reader.read(u32::from(hp_prob))?).unwrap_or(0)
        } else {
            1
        };
        let mag = mag_base + ((class_bit << 3) | (frac << 1) | hp) + 1;
        Ok(if sign { -mag } else { mag })
    }

    fn find_mv(&self, query: &MvQuery) -> Result<([Mv; 2], usize), CodecError> {
        let mut list = [Mv::default(); 2];
        let mut count = 0usize;
        let mut different = false;
        let early = query.mode != NEAR;
        let mut done = false;
        let mut search = CandidateSearch {
            list: &mut list,
            count: &mut count,
            different: &mut different,
            early,
            done: &mut done,
        };
        let mut index = 0usize;
        if query.block >= 0 {
            for _ in 0..2 {
                self.consider(query, index, true, &mut search);
                index += 1;
                if *search.done {
                    break;
                }
            }
        }
        if !*search.done {
            while index < 8 {
                self.consider(query, index, false, &mut search);
                index += 1;
                if *search.done {
                    break;
                }
            }
        }
        if !done && different {
            for pos in MV_REF[usize::from(query.bsize)] {
                if done {
                    break;
                }
                let Some(candidate) = self.neighbor(query.mi_row, query.mi_col, pos) else {
                    continue;
                };
                for mv in scaled_diff(candidate, query.frame, &self.header.sign_bias) {
                    let Some(mv) = mv else { continue };
                    if push_mv(&mut list, &mut count, mv, early) {
                        done = true;
                        break;
                    }
                }
            }
        }
        if !done {
            count = if query.mode == NEAR { 2 } else { 1 };
        }
        for mv in list.iter_mut().take(count) {
            *mv = clamp_ref(*mv, query, self.mi_rows, self.mi_cols);
        }
        if count == 0 {
            return Err(super::bad("motion vector"));
        }
        Ok((list, count))
    }

    fn consider(&self, query: &MvQuery, index: usize, sub: bool, search: &mut CandidateSearch<'_>) {
        if *search.done {
            return;
        }
        let pos = MV_REF[usize::from(query.bsize)][index];
        let Some(candidate) = self.neighbor(query.mi_row, query.mi_col, pos) else {
            return;
        };
        *search.different = true;
        let found = if sub {
            same_sub(candidate, query.frame, query.block, pos[1])
        } else {
            same_mv(candidate, query.frame)
        };
        if let Some(mv) = found
            && push_mv(search.list, search.count, mv, search.early)
        {
            *search.done = true;
        }
    }

    fn neighbor(&self, mi_row: usize, mi_col: usize, pos: [i8; 2]) -> Option<&Mi> {
        let row = mi_row as isize + isize::from(pos[0]);
        let col = mi_col as isize + isize::from(pos[1]);
        if row < 0
            || col < self.tile_col as isize
            || row >= self.mi_rows as isize
            || col >= self.tile_col_end as isize
        {
            return None;
        }
        let row = usize::try_from(row).ok()?;
        let col = usize::try_from(col).ok()?;
        self.grid.get(row * self.mi_cols + col)
    }

    fn mode_context(&self, bsize: u8, mi_row: usize, mi_col: usize) -> Result<usize, CodecError> {
        let mut sum = 0i32;
        for pos in MV_REF[usize::from(bsize)].iter().take(2) {
            if let Some(mi) = self.neighbor(mi_row, mi_col, *pos) {
                sum += MODE_COUNTER.get(usize::from(mi.mode)).copied().unwrap_or(0);
            }
        }
        let mapped = COUNTER_CTX
            .get(usize::try_from(sum).unwrap_or(0))
            .copied()
            .unwrap_or(9);
        if mapped > 6 {
            return Err(super::bad("mode context"));
        }
        Ok(usize::from(mapped))
    }

    fn predict_inter(&mut self, mi: &Mi, pos: BlockPos) -> Result<(), CodecError> {
        let compound = has_second(mi);
        let nref = 1 + usize::from(compound);
        for ref_i in 0..nref {
            let frame = mi.refs[ref_i];
            let (width, height) = {
                let reference = self.reference(frame)?;
                (reference.width, reference.height)
            };
            if width != self.header.width || height != self.header.height {
                return Err(CodecError::Unsupported {
                    what: "vp9 scaled reference",
                });
            }
            for plane in 0..3 {
                self.predict_plane(mi, pos, plane, ref_i, ref_i > 0)?;
            }
        }
        Ok(())
    }

    fn predict_plane(
        &mut self,
        mi: &Mi,
        pos: BlockPos,
        plane: usize,
        ref_i: usize,
        average: bool,
    ) -> Result<(), CodecError> {
        let ss = u32::from(plane > 0);
        let origin_x = (pos.mi_col << 3) >> ss;
        let origin_y = (pos.mi_row << 3) >> ss;
        if mi.bsize < BLOCK_8X8 {
            let n4_w = (pos.bw << 1) >> ss;
            let n4_h = (pos.bh << 1) >> ss;
            let mut block = 0usize;
            for row in 0..n4_h {
                for col in 0..n4_w {
                    let (mv_row, mv_col) = q4_mv(mi, plane, block, ref_i);
                    block += 1;
                    self.paint(Paint {
                        frame: mi.refs[ref_i],
                        plane,
                        x: origin_x + 4 * col,
                        y: origin_y + 4 * row,
                        w: 4,
                        h: 4,
                        mv_row,
                        mv_col,
                        filter: mi.interp,
                        average,
                    })?;
                }
            }
            return Ok(());
        }
        let (mv_row, mv_col) = q4_mv(mi, plane, 0, ref_i);
        self.paint(Paint {
            frame: mi.refs[ref_i],
            plane,
            x: origin_x,
            y: origin_y,
            w: (pos.bw << 3) >> ss,
            h: (pos.bh << 3) >> ss,
            mv_row,
            mv_col,
            filter: mi.interp,
            average,
        })
    }

    fn paint(&mut self, paint: Paint) -> Result<(), CodecError> {
        let (stride, rows) = self.plane_extent(paint.plane);
        if paint.x >= stride || paint.y >= rows {
            return Ok(());
        }
        let w = paint.w.min(stride - paint.x);
        let h = paint.h.min(rows - paint.y);
        if w == 0 || h == 0 {
            return Ok(());
        }
        let mut pred = vec![0u8; w.saturating_mul(h)];
        {
            let source = self.plane_ref(paint.frame, paint.plane)?;
            mc::render(mc::Job {
                pred: &mut pred,
                pred_stride: w,
                w,
                h,
                origin_x: i32::try_from(paint.x).unwrap_or(0),
                origin_y: i32::try_from(paint.y).unwrap_or(0),
                mv_row: paint.mv_row,
                mv_col: paint.mv_col,
                filter: usize::from(paint.filter),
                reference: source,
            })?;
        }
        self.blit(&paint, w, h, &pred)
    }

    fn plane_extent(&self, plane: usize) -> (usize, usize) {
        if plane == 0 {
            (self.y_stride, self.y.len() / self.y_stride.max(1))
        } else {
            (self.uv_stride, self.u.len() / self.uv_stride.max(1))
        }
    }

    fn plane_ref(&self, frame: u8, plane: usize) -> Result<&RefPlane, CodecError> {
        let reference = self.reference(frame)?;
        Ok(match plane {
            0 => &reference.y,
            1 => &reference.u,
            _ => &reference.v,
        })
    }

    fn reference(&self, frame: u8) -> Result<&mc::Reference, CodecError> {
        if !(1..=3).contains(&frame) {
            return Err(super::bad("reference frame"));
        }
        let slot = usize::from(self.header.ref_index[usize::from(frame - 1)]);
        self.slots
            .frames
            .get(slot)
            .copied()
            .flatten()
            .ok_or(super::bad("reference picture"))
    }

    fn blit(&mut self, paint: &Paint, w: usize, h: usize, pred: &[u8]) -> Result<(), CodecError> {
        let stride = if paint.plane == 0 {
            self.y_stride
        } else {
            self.uv_stride
        };
        let dest = match paint.plane {
            0 => self.y.as_mut_slice(),
            1 => self.u.as_mut_slice(),
            _ => self.v.as_mut_slice(),
        };
        for row in 0..h {
            let src = pred
                .get(row * w..row * w + w)
                .ok_or(super::bad("prediction"))?;
            let at = (paint.y + row) * stride + paint.x;
            let dst = dest.get_mut(at..at + w).ok_or(super::bad("prediction"))?;
            if paint.average {
                for (out, sample) in dst.iter_mut().zip(src) {
                    let mixed = (u16::from(*out) + u16::from(*sample) + 1) >> 1;
                    *out = u8::try_from(mixed).unwrap_or(0);
                }
            } else {
                dst.copy_from_slice(src);
            }
        }
        Ok(())
    }

    fn bump_single(&mut self, ctx: usize, part: usize, bit: usize) {
        if self.collect {
            self.counts.single_ref[ctx][part][bit] =
                self.counts.single_ref[ctx][part][bit].saturating_add(1);
        }
    }
}

struct SubStep<'a> {
    ctx: usize,
    compound: bool,
    block: i32,
    got_new: &'a mut bool,
    best_ref: &'a mut [Mv; 2],
}

fn sub_steps(bsize: u8) -> (usize, usize) {
    match bsize {
        BLOCK_4X4 => (1, 1),
        BLOCK_4X8 => (1, 2),
        BLOCK_8X4 => (2, 1),
        _ => (1, 1),
    }
}

fn has_second(mi: &Mi) -> bool {
    mi.refs[1] > INTRA_FRAME
}

fn is_inter(mi: &Mi) -> bool {
    mi.refs[0] > INTRA_FRAME
}

fn intra_inter_ctx(above: Option<&Mi>, left: Option<&Mi>) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => {
            let above_intra = !is_inter(above);
            let left_intra = !is_inter(left);
            if left_intra && above_intra {
                3
            } else {
                usize::from(left_intra || above_intra)
            }
        }
        (Some(edge), None) | (None, Some(edge)) => 2 * usize::from(!is_inter(edge)),
        (None, None) => 0,
    }
}

fn interp_ctx(left: Option<u8>, above: Option<u8>) -> usize {
    let left = left.unwrap_or(SENTINEL_FILTER);
    let above = above.unwrap_or(SENTINEL_FILTER);
    if left == above || left == SENTINEL_FILTER {
        usize::from(if left == above { left } else { above })
    } else if above == SENTINEL_FILTER {
        usize::from(left)
    } else {
        usize::from(SENTINEL_FILTER)
    }
}

fn reference_mode_ctx(above: Option<&Mi>, left: Option<&Mi>, fixed: u8) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => both_ref_mode(above, left, fixed),
        (Some(edge), None) | (None, Some(edge)) => one_ref_mode(edge, fixed),
        (None, None) => 1,
    }
}

fn both_ref_mode(above: &Mi, left: &Mi, fixed: u8) -> usize {
    if !has_second(above) && !has_second(left) {
        usize::from((above.refs[0] == fixed) ^ (left.refs[0] == fixed))
    } else if !has_second(above) {
        2 + usize::from(above.refs[0] == fixed || !is_inter(above))
    } else if !has_second(left) {
        2 + usize::from(left.refs[0] == fixed || !is_inter(left))
    } else {
        4
    }
}

fn one_ref_mode(edge: &Mi, fixed: u8) -> usize {
    if has_second(edge) {
        3
    } else {
        usize::from(edge.refs[0] == fixed)
    }
}

fn single_p1(above: Option<&Mi>, left: Option<&Mi>) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => single_p1_both(above, left),
        (Some(edge), None) | (None, Some(edge)) => single_one_last(edge),
        (None, None) => 2,
    }
}

fn single_p1_both(above: &Mi, left: &Mi) -> usize {
    let above_intra = !is_inter(above);
    let left_intra = !is_inter(left);
    if above_intra && left_intra {
        2
    } else if above_intra || left_intra {
        single_one_last(if above_intra { left } else { above })
    } else {
        single_p1_inter(above, left)
    }
}

fn single_one_last(edge: &Mi) -> usize {
    if !is_inter(edge) {
        return 2;
    }
    if !has_second(edge) {
        4 * usize::from(edge.refs[0] == LAST_FRAME)
    } else {
        1 + usize::from(edge.refs[0] == LAST_FRAME || edge.refs[1] == LAST_FRAME)
    }
}

fn single_p1_inter(above: &Mi, left: &Mi) -> usize {
    let above_second = has_second(above);
    let left_second = has_second(left);
    if above_second && left_second {
        return 1 + usize::from(mentions(above, LAST_FRAME) || mentions(left, LAST_FRAME));
    }
    if above_second || left_second {
        let single = if above_second { left } else { above };
        let compound = if above_second { above } else { left };
        if single.refs[0] == LAST_FRAME {
            3 + usize::from(mentions(compound, LAST_FRAME))
        } else {
            usize::from(mentions(compound, LAST_FRAME))
        }
    } else {
        2 * usize::from(above.refs[0] == LAST_FRAME) + 2 * usize::from(left.refs[0] == LAST_FRAME)
    }
}

fn single_p2(above: Option<&Mi>, left: Option<&Mi>) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => single_p2_both(above, left),
        (Some(edge), None) | (None, Some(edge)) => single_p2_one(edge),
        (None, None) => 2,
    }
}

fn single_p2_one(edge: &Mi) -> usize {
    if !is_inter(edge) || (edge.refs[0] == LAST_FRAME && !has_second(edge)) {
        2
    } else if !has_second(edge) {
        4 * usize::from(edge.refs[0] == GOLDEN_FRAME)
    } else {
        3 * usize::from(mentions(edge, GOLDEN_FRAME))
    }
}

fn single_p2_both(above: &Mi, left: &Mi) -> usize {
    let above_intra = !is_inter(above);
    let left_intra = !is_inter(left);
    if above_intra && left_intra {
        2
    } else if above_intra || left_intra {
        single_p2_mixed(if above_intra { left } else { above })
    } else {
        single_p2_inter(above, left)
    }
}

fn single_p2_mixed(edge: &Mi) -> usize {
    if !has_second(edge) {
        if edge.refs[0] == LAST_FRAME {
            3
        } else {
            4 * usize::from(edge.refs[0] == GOLDEN_FRAME)
        }
    } else {
        1 + 2 * usize::from(mentions(edge, GOLDEN_FRAME))
    }
}

fn single_p2_inter(above: &Mi, left: &Mi) -> usize {
    let above_second = has_second(above);
    let left_second = has_second(left);
    if above_second && left_second {
        if above.refs == left.refs {
            3 * usize::from(mentions(above, GOLDEN_FRAME) || mentions(left, GOLDEN_FRAME))
        } else {
            2
        }
    } else if above_second || left_second {
        p2_mixed_refs(above, left, above_second)
    } else {
        p2_two_singles(above, left)
    }
}

fn p2_mixed_refs(above: &Mi, left: &Mi, above_second: bool) -> usize {
    let single = if above_second { left } else { above };
    let compound = if above_second { above } else { left };
    if single.refs[0] == GOLDEN_FRAME {
        3 + usize::from(mentions(compound, GOLDEN_FRAME))
    } else if single.refs[0] == ALTREF_FRAME {
        usize::from(mentions(compound, GOLDEN_FRAME))
    } else {
        1 + 2 * usize::from(mentions(compound, GOLDEN_FRAME))
    }
}

fn p2_two_singles(above: &Mi, left: &Mi) -> usize {
    if above.refs[0] == LAST_FRAME && left.refs[0] == LAST_FRAME {
        3
    } else if above.refs[0] == LAST_FRAME || left.refs[0] == LAST_FRAME {
        let other = if above.refs[0] == LAST_FRAME {
            left.refs[0]
        } else {
            above.refs[0]
        };
        4 * usize::from(other == GOLDEN_FRAME)
    } else {
        2 * usize::from(above.refs[0] == GOLDEN_FRAME)
            + 2 * usize::from(left.refs[0] == GOLDEN_FRAME)
    }
}

fn comp_ref_ctx(above: Option<&Mi>, left: Option<&Mi>, header: &crate::header::Header) -> usize {
    let fixed = header.comp_fixed;
    let var0 = header.comp_var[0];
    let var1 = header.comp_var[1];
    let var_at = usize::from(header.sign_bias[usize::from(fixed)] == 0);
    match (above, left) {
        (Some(above), Some(left)) => comp_both(above, left, fixed, var0, var1, var_at),
        (Some(edge), None) | (None, Some(edge)) => comp_one(edge, var1, var_at),
        (None, None) => 2,
    }
}

fn comp_one(edge: &Mi, var: u8, var_at: usize) -> usize {
    if !is_inter(edge) {
        2
    } else if has_second(edge) {
        4 * usize::from(edge.refs[var_at] != var)
    } else {
        3 * usize::from(edge.refs[0] != var)
    }
}

fn comp_both(above: &Mi, left: &Mi, fixed: u8, var0: u8, var1: u8, var_at: usize) -> usize {
    let above_intra = !is_inter(above);
    let left_intra = !is_inter(left);
    if above_intra && left_intra {
        2
    } else if above_intra || left_intra {
        comp_one_edge(if above_intra { left } else { above }, var1, var_at)
    } else {
        comp_inter(above, left, fixed, var0, var1, var_at)
    }
}

fn comp_one_edge(edge: &Mi, var: u8, var_at: usize) -> usize {
    let seen = if has_second(edge) {
        edge.refs[var_at]
    } else {
        edge.refs[0]
    };
    1 + 2 * usize::from(seen != var)
}

fn comp_inter(above: &Mi, left: &Mi, fixed: u8, var0: u8, var1: u8, var_at: usize) -> usize {
    let above_single = !has_second(above);
    let left_single = !has_second(left);
    let above_ref = if above_single {
        above.refs[0]
    } else {
        above.refs[var_at]
    };
    let left_ref = if left_single {
        left.refs[0]
    } else {
        left.refs[var_at]
    };
    if above_ref == left_ref && var1 == above_ref {
        0
    } else if left_single && above_single {
        comp_two_singles(above_ref, left_ref, fixed, var0)
    } else if left_single || above_single {
        comp_single_and_compound(above_ref, left_ref, left_single, var1)
    } else if above_ref == left_ref {
        4
    } else {
        2
    }
}

fn comp_two_singles(above_ref: u8, left_ref: u8, fixed: u8, var0: u8) -> usize {
    let crossed =
        (above_ref == fixed && left_ref == var0) || (left_ref == fixed && above_ref == var0);
    if crossed {
        4
    } else if above_ref == left_ref {
        3
    } else {
        1
    }
}

fn comp_single_and_compound(above_ref: u8, left_ref: u8, left_single: bool, var: u8) -> usize {
    let compound = if left_single { above_ref } else { left_ref };
    let single = if left_single { left_ref } else { above_ref };
    if compound == var && single != var {
        1
    } else if single == var && compound != var {
        2
    } else {
        4
    }
}

fn mentions(mi: &Mi, frame: u8) -> bool {
    mi.refs[0] == frame || mi.refs[1] == frame
}

fn same_mv(mi: &Mi, frame: u8) -> Option<Mv> {
    if mi.refs[0] == frame {
        Some(mi.mv[0])
    } else if mi.refs[1] == frame {
        Some(mi.mv[1])
    } else {
        None
    }
}

fn same_sub(mi: &Mi, frame: u8, block: i32, search_col: i8) -> Option<Mv> {
    let which = if mi.refs[0] == frame {
        0
    } else if mi.refs[1] == frame {
        1
    } else {
        return None;
    };
    Some(sub_mv(mi, which, block, search_col))
}

fn sub_mv(mi: &Mi, which: usize, block: i32, search_col: i8) -> Mv {
    if block >= 0 && mi.bsize < BLOCK_8X8 {
        let index = usize::from(
            SUBBLOCK[usize::try_from(block).unwrap_or(0)][usize::from(search_col == 0)],
        );
        mi.sub_mv[index][which]
    } else {
        mi.mv[which]
    }
}

fn scaled_diff(mi: &Mi, frame: u8, bias: &[u8; 4]) -> [Option<Mv>; 2] {
    if !is_inter(mi) {
        return [None, None];
    }
    let first = (mi.refs[0] != frame).then(|| flip(mi.mv[0], mi.refs[0], frame, bias));
    let second = (has_second(mi) && mi.refs[1] != frame && mi.mv[1] != mi.mv[0])
        .then(|| flip(mi.mv[1], mi.refs[1], frame, bias));
    [first, second]
}

fn flip(mv: Mv, from: u8, to: u8, bias: &[u8; 4]) -> Mv {
    if bias[usize::from(from)] == bias[usize::from(to)] {
        mv
    } else {
        Mv {
            row: mv.row.wrapping_neg(),
            col: mv.col.wrapping_neg(),
        }
    }
}

fn push_mv(list: &mut [Mv; 2], count: &mut usize, mv: Mv, early: bool) -> bool {
    if *count > 0 {
        if list[0] != mv {
            list[*count] = mv;
            *count += 1;
            true
        } else {
            false
        }
    } else {
        list[0] = mv;
        *count = 1;
        early
    }
}

fn clamp_ref(mv: Mv, query: &MvQuery, mi_rows: usize, mi_cols: usize) -> Mv {
    let mi_row = i32::try_from(query.mi_row).unwrap_or(0);
    let mi_col = i32::try_from(query.mi_col).unwrap_or(0);
    let bh = i32::try_from(query.bh).unwrap_or(0);
    let bw = i32::try_from(query.bw).unwrap_or(0);
    let rows = i32::try_from(mi_rows).unwrap_or(0);
    let cols = i32::try_from(mi_cols).unwrap_or(0);
    let top = -(mi_row * 64) - 128;
    let bottom = (rows - bh - mi_row) * 64 + 128;
    let left = -(mi_col * 64) - 128;
    let right = (cols - bw - mi_col) * 64 + 128;
    Mv {
        row: fit_i16(i32::from(mv.row), top, bottom),
        col: fit_i16(i32::from(mv.col), left, right),
    }
}

fn fit_i16(value: i32, min: i32, max: i32) -> i16 {
    let (min, max) = if min <= max { (min, max) } else { (max, min) };
    let clamped = value.clamp(min, max);
    i16::try_from(clamped.clamp(i32::from(i16::MIN), i32::from(i16::MAX))).unwrap_or(0)
}

fn lower_precision(mv: Mv, allow_hp: bool) -> Mv {
    if allow_hp && mv.row.unsigned_abs() < 64 && mv.col.unsigned_abs() < 64 {
        return mv;
    }
    Mv {
        row: drop_hp(mv.row),
        col: drop_hp(mv.col),
    }
}

fn drop_hp(value: i16) -> i16 {
    let wide = i32::from(value);
    if wide & 1 == 0 {
        value
    } else {
        i16::try_from(wide + if wide > 0 { -1 } else { 1 }).unwrap_or(0)
    }
}

fn add_mv(base: Mv, diff: Mv) -> Result<Mv, CodecError> {
    let row = i32::from(base.row) + i32::from(diff.row);
    let col = i32::from(base.col) + i32::from(diff.col);
    if !(MV_LOW < row && row < MV_UPP && MV_LOW < col && col < MV_UPP) {
        return Err(super::bad("motion vector"));
    }
    Ok(Mv {
        row: i16::try_from(row).unwrap_or(0),
        col: i16::try_from(col).unwrap_or(0),
    })
}

fn component_i16(value: i32) -> Result<i16, CodecError> {
    i16::try_from(value).map_err(|_| super::bad("motion vector"))
}

fn q4_mv(mi: &Mi, plane: usize, block: usize, ref_i: usize) -> (i32, i32) {
    if mi.bsize >= BLOCK_8X8 || plane == 0 {
        let mv = if mi.bsize >= BLOCK_8X8 {
            mi.mv[ref_i]
        } else {
            mi.sub_mv[block][ref_i]
        };
        let scale = if plane == 0 { 2 } else { 1 };
        return (i32::from(mv.row) * scale, i32::from(mv.col) * scale);
    }
    let mut row = 0i32;
    let mut col = 0i32;
    for sub in &mi.sub_mv {
        row += i32::from(sub[ref_i].row);
        col += i32::from(sub[ref_i].col);
    }
    (round_q4(row), round_q4(col))
}

fn round_q4(value: i32) -> i32 {
    (if value < 0 { value - 2 } else { value + 2 }) / 4
}

fn bump2(collect: bool, counts: &mut [[u32; 2]], ctx: usize, bit: usize) {
    if collect {
        counts[ctx][bit] = counts[ctx][bit].saturating_add(1);
    }
}

fn count_mv(counts: &mut Counts, diff: Mv) {
    let joint = if diff.row == 0 {
        usize::from(diff.col != 0)
    } else if diff.col == 0 {
        2
    } else {
        3
    };
    counts.joints[joint] = counts.joints[joint].saturating_add(1);
    if joint == 2 || joint == 3 {
        count_component(counts, 0, i32::from(diff.row));
    }
    if joint == 1 || joint == 3 {
        count_component(counts, 1, i32::from(diff.col));
    }
}

fn count_component(counts: &mut Counts, comp: usize, value: i32) {
    let sign = usize::from(value < 0);
    counts.mv_sign[comp][sign] = counts.mv_sign[comp][sign].saturating_add(1);
    let magnitude = value.unsigned_abs().saturating_sub(1);
    let (class, offset) = mv_class(magnitude);
    let class_i = usize::from(class);
    counts.mv_class[comp][class_i] = counts.mv_class[comp][class_i].saturating_add(1);
    let class_bit = offset >> 3;
    let frac = (offset >> 1) & 3;
    let hp = (offset & 1) as usize;
    if class == 0 {
        let bit = class_bit as usize;
        counts.mv_class0[comp][bit] = counts.mv_class0[comp][bit].saturating_add(1);
        counts.mv_class0_fp[comp][bit][frac as usize] =
            counts.mv_class0_fp[comp][bit][frac as usize].saturating_add(1);
        counts.mv_class0_hp[comp][hp] = counts.mv_class0_hp[comp][hp].saturating_add(1);
    } else {
        for i in 0..usize::from(class) {
            let bit = ((class_bit >> i) & 1) as usize;
            counts.mv_bits[comp][i][bit] = counts.mv_bits[comp][i][bit].saturating_add(1);
        }
        counts.mv_fp[comp][frac as usize] = counts.mv_fp[comp][frac as usize].saturating_add(1);
        counts.mv_hp[comp][hp] = counts.mv_hp[comp][hp].saturating_add(1);
    }
}

fn mv_class(magnitude: u32) -> (u8, u32) {
    let class = if magnitude >= 8192 {
        10
    } else if magnitude < 16 {
        0
    } else {
        u8::try_from((magnitude >> 3).ilog2()).unwrap_or(0)
    };
    let base = if class == 0 {
        0
    } else {
        2u32 << (u32::from(class) + 2)
    };
    (class, magnitude - base)
}
