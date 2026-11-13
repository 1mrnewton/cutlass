//! Inter prediction units (clause 7.3.8.6), merge and AMVP motion
//! derivation with temporal prediction (clause 8.5.3.2), and the motion
//! compensation of each prediction block (clause 8.5.3.3).

use cutlass_decode::CodecError;

use crate::cabac;
use crate::ctu::{
    PART_2NXN, PART_2NXND, PART_2NXNU, PART_NLX2N, PART_NRX2N, PART_NX2N, SliceDecoder,
};
use crate::mc::{self, Block, Plane, Wp};
use crate::picture::{ColMv, MODE_INTER, Mv, MvField};
use crate::slice::SliceType;

fn malformed(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

/// Scale a vector by the ratio of two order count distances (equations
/// 8-182 to 8-186).
fn scale(mv: Mv, td: i32, tb: i32) -> Mv {
    let td = td.clamp(-128, 127);
    let tb = tb.clamp(-128, 127);
    if td == 0 {
        return mv;
    }
    let tx = (16384 + (td.abs() >> 1)) / td;
    let factor = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
    let one = |v: i16| -> i16 {
        let product = factor * i32::from(v);
        let magnitude = (product.abs() + 127) >> 8;
        (product.signum() * magnitude).clamp(-32768, 32767) as i16
    };
    Mv {
        x: one(mv.x),
        y: one(mv.y),
    }
}

/// Where the prediction block sits in its coding block.
#[derive(Clone, Copy)]
struct Pb {
    xcb: usize,
    ycb: usize,
    ncb: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    part: usize,
}

impl<'a> SliceDecoder<'a, '_> {
    pub fn prediction_unit(
        &mut self,
        x: usize,
        y: usize,
        (w, h): (usize, usize),
        depth: u32,
        part: usize,
        skip: bool,
    ) -> Result<(), CodecError> {
        let header = &self.pic.slices[self.slice].header;
        let max_merge = usize::from(header.max_merge_cand);
        let num_ref = header.num_ref_idx;
        let mvd_l1_zero = header.mvd_l1_zero;
        let pb = Pb {
            xcb: self.cu.x,
            ycb: self.cu.y,
            ncb: 1 << self.cu.log2,
            x,
            y,
            w,
            h,
            part,
        };
        let merge = skip || self.cabac.decide(cabac::MERGE_FLAG);
        let field = if merge {
            let mut idx = 0;
            if max_merge > 1 && self.cabac.decide(cabac::MERGE_IDX) {
                idx = 1;
                while idx < max_merge - 1 && self.cabac.bypass() {
                    idx += 1;
                }
            }
            if part == 0 && w == pb.ncb && h == pb.ncb {
                self.merge_2nx2n = true;
            }
            self.merge_motion(pb, idx)?
        } else {
            let mut pred = 1u8;
            if self.slice_type == SliceType::B {
                pred = if w + h == 12 {
                    if self.cabac.decide(cabac::INTER_PRED + 4) {
                        2
                    } else {
                        1
                    }
                } else if self.cabac.decide(cabac::INTER_PRED + depth as usize) {
                    3
                } else if self.cabac.decide(cabac::INTER_PRED + 4) {
                    2
                } else {
                    1
                };
            }
            let mut field = MvField {
                mv: [Mv::default(); 2],
                ref_idx: [-1, -1],
                pred,
            };
            let mut mvd = [Mv::default(); 2];
            let mut mvp = [false; 2];
            for list in 0..2 {
                if pred & (1 << list) == 0 {
                    continue;
                }
                let mut idx = 0usize;
                let count = num_ref[list];
                if count > 1 {
                    let max = count - 1;
                    while idx < max.min(2) && self.cabac.decide(cabac::REF_IDX + idx) {
                        idx += 1;
                    }
                    if idx == 2 {
                        while idx < max && self.cabac.bypass() {
                            idx += 1;
                        }
                    }
                }
                field.ref_idx[list] = idx as i8;
                if list == 1 && mvd_l1_zero && pred == 3 {
                    mvd[1] = Mv::default();
                } else {
                    mvd[list] = self.mvd_coding()?;
                }
                mvp[list] = self.cabac.decide(cabac::MVP_FLAG);
            }
            for list in 0..2 {
                if pred & (1 << list) == 0 {
                    continue;
                }
                let predictor = self.amvp(
                    pb,
                    list,
                    field.ref_idx[list] as usize,
                    usize::from(mvp[list]),
                )?;
                field.mv[list] = Mv {
                    x: predictor.x.wrapping_add(mvd[list].x),
                    y: predictor.y.wrapping_add(mvd[list].y),
                };
            }
            field
        };
        let w4 = self.pic.w4;
        for row in (y >> 2)..((y + h) >> 2) {
            let start = row * w4 + (x >> 2);
            self.pic.mvf.fill(start, w >> 2, field);
        }
        self.motion_compensate(x, y, w, h, field)
    }

    fn mvd_coding(&mut self) -> Result<Mv, CodecError> {
        let g0 = [
            self.cabac.decide(cabac::MVD_G0),
            self.cabac.decide(cabac::MVD_G0),
        ];
        let mut g1 = [false; 2];
        for c in 0..2 {
            if g0[c] {
                g1[c] = self.cabac.decide(cabac::MVD_G1);
            }
        }
        let mut out = [0i32; 2];
        for c in 0..2 {
            if !g0[c] {
                continue;
            }
            let mut abs = 1i32;
            if g1[c] {
                let mut value = 0i64;
                let mut k = 1u32;
                while self.cabac.bypass() {
                    value += 1 << k;
                    k += 1;
                    if k > 31 {
                        return Err(malformed("abs_mvd_minus2"));
                    }
                }
                value += i64::from(self.cabac.bypass_bits(k));
                if value > 32_766 {
                    return Err(malformed("abs_mvd_minus2"));
                }
                abs = value as i32 + 2;
            }
            out[c] = if self.cabac.bypass() { -abs } else { abs };
        }
        Ok(Mv {
            x: out[0].clamp(-32768, 32767) as i16,
            y: out[1].clamp(-32768, 32767) as i16,
        })
    }

    /// Availability of a neighbouring prediction block (clause 6.4.2).
    fn pb_available(&self, pb: Pb, nx: isize, ny: isize) -> bool {
        let same_cb = nx >= pb.xcb as isize
            && ny >= pb.ycb as isize
            && (nx as usize) < pb.xcb + pb.ncb
            && (ny as usize) < pb.ycb + pb.ncb;
        let available = if !same_cb {
            self.pic.available(pb.x, pb.y, nx, ny)
        } else {
            !(pb.w << 1 == pb.ncb
                && pb.h << 1 == pb.ncb
                && pb.part == 1
                && pb.ycb + pb.h <= ny as usize
                && pb.xcb + pb.w > nx as usize)
        };
        available && self.pic.info.get(self.pic.at4(nx as usize, ny as usize)) & 3 == MODE_INTER
    }

    fn neighbour(&self, nx: isize, ny: isize) -> MvField {
        self.pic.mv_at(nx as usize, ny as usize)
    }

    /// Merge motion (clauses 8.5.3.2.2 to 8.5.3.2.5).
    fn merge_motion(&self, mut pb: Pb, merge_idx: usize) -> Result<MvField, CodecError> {
        let orig = (pb.w, pb.h);
        let level = u32::from(self.pps.log2_parallel_merge_level);
        if level > 2 && pb.ncb == 8 {
            pb = Pb {
                x: pb.xcb,
                y: pb.ycb,
                w: pb.ncb,
                h: pb.ncb,
                part: 0,
                ..pb
            };
        }
        let part_mode = self.cu.part;
        let (x, y, w, h) = (pb.x as isize, pb.y as isize, pb.w as isize, pb.h as isize);
        let same_region = |nx: isize, ny: isize| -> bool {
            (x >> level) == (nx >> level) && (y >> level) == (ny >> level)
        };
        let mut list: Vec<MvField> = Vec::with_capacity(5);
        let get = |nx: isize, ny: isize| -> Option<MvField> {
            if same_region(nx, ny) || !self.pb_available(pb, nx, ny) {
                None
            } else {
                Some(self.neighbour(nx, ny))
            }
        };
        // Exclusions apply before the pruning comparisons, as in clause
        // 8.5.3.2.3: a pruned B1 still prunes B0 and B2.
        let excluded_a1 = pb.part == 1 && matches!(part_mode, PART_NX2N | PART_NLX2N | PART_NRX2N);
        let excluded_b1 = pb.part == 1 && matches!(part_mode, PART_2NXN | PART_2NXNU | PART_2NXND);
        let a1 = if excluded_a1 {
            None
        } else {
            get(x - 1, y + h - 1)
        };
        if let Some(a) = a1 {
            list.push(a);
        }
        let b1 = if excluded_b1 {
            None
        } else {
            get(x + w - 1, y - 1)
        };
        if let Some(b) = b1
            && a1 != Some(b)
        {
            list.push(b);
        }
        if let Some(b) = get(x + w, y - 1)
            && b1 != Some(b)
        {
            list.push(b);
        }
        if let Some(a) = get(x - 1, y + h)
            && a1 != Some(a)
        {
            list.push(a);
        }
        if list.len() != 4
            && let Some(b) = get(x - 1, y - 1)
            && a1 != Some(b)
            && b1 != Some(b)
        {
            list.push(b);
        }
        let header = &self.pic.slices[self.slice].header;
        let max = usize::from(header.max_merge_cand);
        let is_b = header.slice_type == SliceType::B;
        if merge_idx < list.len() {
            return Ok(self.restrict(list[merge_idx], orig));
        }
        if list.len() < max && header.temporal_mvp {
            let l0 = self.temporal(pb, 0, 0);
            let l1 = if is_b { self.temporal(pb, 1, 0) } else { None };
            if l0.is_some() || l1.is_some() {
                let mut field = MvField {
                    mv: [Mv::default(); 2],
                    ref_idx: [-1, -1],
                    pred: 0,
                };
                if let Some(mv) = l0 {
                    field.mv[0] = mv;
                    field.ref_idx[0] = 0;
                    field.pred |= 1;
                }
                if let Some(mv) = l1 {
                    field.mv[1] = mv;
                    field.ref_idx[1] = 0;
                    field.pred |= 2;
                }
                list.push(field);
            }
        }
        let slice = &self.pic.slices[self.slice];
        if is_b && list.len() > 1 && list.len() < max {
            const ORDER: [(usize, usize); 12] = [
                (0, 1),
                (1, 0),
                (0, 2),
                (2, 0),
                (1, 2),
                (2, 1),
                (0, 3),
                (3, 0),
                (1, 3),
                (3, 1),
                (2, 3),
                (3, 2),
            ];
            let orig_count = list.len();
            for &(i0, i1) in ORDER.iter().take(orig_count * (orig_count - 1)) {
                if list.len() >= max {
                    break;
                }
                let c0 = list[i0];
                let c1 = list[i1];
                if c0.uses(0) && c1.uses(1) {
                    let p0 = slice.refs[0].get(c0.ref_idx[0] as usize).map(|r| r.uid);
                    let p1 = slice.refs[1].get(c1.ref_idx[1] as usize).map(|r| r.uid);
                    if p0 != p1 || c0.mv[0] != c1.mv[1] {
                        list.push(MvField {
                            mv: [c0.mv[0], c1.mv[1]],
                            ref_idx: [c0.ref_idx[0], c1.ref_idx[1]],
                            pred: 3,
                        });
                    }
                }
            }
        }
        let num_ref = if is_b {
            header.num_ref_idx[0].min(header.num_ref_idx[1])
        } else {
            header.num_ref_idx[0]
        };
        let mut zero = 0usize;
        while list.len() < max {
            let idx = if zero < num_ref { zero as i8 } else { 0 };
            list.push(MvField {
                mv: [Mv::default(); 2],
                ref_idx: [idx, if is_b { idx } else { -1 }],
                pred: if is_b { 3 } else { 1 },
            });
            zero += 1;
        }
        let Some(&chosen) = list.get(merge_idx) else {
            return Err(malformed("merge_idx"));
        };
        Ok(self.restrict(chosen, orig))
    }

    /// An 8×4 or 4×8 block predicts from list 0 only (equation 8-130).
    fn restrict(&self, mut field: MvField, (w, h): (usize, usize)) -> MvField {
        if field.pred == 3 && w + h == 12 {
            field.pred = 1;
            field.ref_idx[1] = -1;
            field.mv[1] = Mv::default();
        }
        field
    }

    /// Temporal motion vector prediction (clause 8.5.3.2.8).
    fn temporal(&self, pb: Pb, list: usize, ref_idx: usize) -> Option<Mv> {
        let slice = &self.pic.slices[self.slice];
        let header = &slice.header;
        if !header.temporal_mvp {
            return None;
        }
        let col_list = if header.slice_type == SliceType::B && !header.collocated_from_l0 {
            1
        } else {
            0
        };
        let col_ref = slice.refs[col_list].get(header.collocated_ref_idx)?;
        let col = &col_ref.pic;
        let log2 = self.sps.log2_ctb;
        let xbr = pb.x + pb.w;
        let ybr = pb.y + pb.h;
        if (pb.y >> log2) == (ybr >> log2) && ybr < self.pic.height && xbr < self.pic.width {
            let cand = col.col_at((xbr >> 4) << 4, (ybr >> 4) << 4);
            if let Some(mv) = self.collocated(cand, col.poc, list, ref_idx) {
                return Some(mv);
            }
        }
        let xc = pb.x + (pb.w >> 1);
        let yc = pb.y + (pb.h >> 1);
        let cand = col.col_at((xc >> 4) << 4, (yc >> 4) << 4);
        self.collocated(cand, col.poc, list, ref_idx)
    }

    fn collocated(&self, cand: ColMv, col_poc: i32, list: usize, ref_idx: usize) -> Option<Mv> {
        if cand.pred == 0 {
            return None;
        }
        let slice = &self.pic.slices[self.slice];
        let target = slice.refs[list].get(ref_idx)?;
        let which = if cand.pred & 1 == 0 {
            1
        } else if cand.pred & 2 == 0 {
            0
        } else if slice.no_backward_pred {
            list
        } else {
            usize::from(slice.header.collocated_from_l0)
        };
        if target.long_term != cand.long_term[which] {
            return None;
        }
        let mv = cand.mv[which];
        let col_diff = col_poc - cand.poc[which];
        let cur_diff = self.pic.poc - target.poc;
        if target.long_term || col_diff == cur_diff {
            Some(mv)
        } else {
            Some(scale(mv, col_diff, cur_diff))
        }
    }

    /// Luma motion vector prediction (clauses 8.5.3.2.6 and 8.5.3.2.7).
    fn amvp(&self, pb: Pb, list: usize, ref_idx: usize, mvp_flag: usize) -> Result<Mv, CodecError> {
        let slice = &self.pic.slices[self.slice];
        let Some(target) = slice.refs[list].get(ref_idx) else {
            return Err(malformed("ref_idx"));
        };
        let other = 1 - list;
        let (x, y, w, h) = (pb.x as isize, pb.y as isize, pb.w as isize, pb.h as isize);
        let cur_poc = self.pic.poc;
        let fetch = |nx: isize, ny: isize| -> Option<MvField> {
            if self.pb_available(pb, nx, ny) {
                Some(self.neighbour(nx, ny))
            } else {
                None
            }
        };
        let ref_of = |l: usize, idx: i8| slice.refs[l].get(idx as usize);
        // Same picture: no scaling.
        let same = |m: &MvField| -> Option<Mv> {
            for l in [list, other] {
                if m.uses(l) && ref_of(l, m.ref_idx[l]).is_some_and(|r| r.uid == target.uid) {
                    return Some(m.mv[l]);
                }
            }
            None
        };
        // Same long-term marking, scaled when both are short-term.
        let scaled = |m: &MvField| -> Option<Mv> {
            for l in [list, other] {
                if !m.uses(l) {
                    continue;
                }
                let Some(r) = ref_of(l, m.ref_idx[l]) else {
                    continue;
                };
                if r.long_term != target.long_term {
                    continue;
                }
                if r.long_term || target.long_term {
                    return Some(m.mv[l]);
                }
                return Some(scale(m.mv[l], cur_poc - r.poc, cur_poc - target.poc));
            }
            None
        };
        let a = [fetch(x - 1, y + h), fetch(x - 1, y + h - 1)];
        let is_scaled = a[0].is_some() || a[1].is_some();
        let mut mv_a = a.iter().flatten().find_map(&same);
        if mv_a.is_none() {
            mv_a = a.iter().flatten().find_map(&scaled);
        }
        let b = [
            fetch(x + w, y - 1),
            fetch(x + w - 1, y - 1),
            fetch(x - 1, y - 1),
        ];
        let mut mv_b = b.iter().flatten().find_map(&same);
        if !is_scaled && mv_b.is_some() && mv_a.is_none() {
            mv_a = mv_b;
        }
        if !is_scaled {
            mv_b = b.iter().flatten().find_map(&scaled);
        }
        let mut candidates: Vec<Mv> = Vec::with_capacity(3);
        if let Some(a) = mv_a {
            candidates.push(a);
        }
        if let Some(b) = mv_b
            && mv_a != Some(b)
        {
            candidates.push(b);
        }
        if candidates.len() < 2
            && let Some(t) = self.temporal(pb, list, ref_idx)
        {
            candidates.push(t);
        }
        while candidates.len() < 2 {
            candidates.push(Mv::default());
        }
        Ok(candidates[mvp_flag])
    }

    /// Predict one block from its references into the picture.
    fn motion_compensate(
        &mut self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        field: MvField,
    ) -> Result<(), CodecError> {
        let slice = &self.pic.slices[self.slice];
        let header = &slice.header;
        let explicit = match header.slice_type {
            SliceType::P => self.pps.weighted_pred,
            SliceType::B => self.pps.weighted_bipred,
            SliceType::I => false,
        };
        let bit_depth = u32::from(self.sps.bit_depth);
        let mut refs = [None, None];
        for (list, slot) in refs.iter_mut().enumerate() {
            if field.uses(list) {
                let Some(r) = slice.refs[list].get(field.ref_idx[list] as usize) else {
                    return Err(malformed("reference index"));
                };
                *slot = Some(r.pic.clone());
            }
        }
        let weights = if explicit {
            let Some(table) = header.weights.as_ref() else {
                return Err(malformed("weighted prediction without weights"));
            };
            let mut out = [[Wp {
                weight: 1,
                offset: 0,
                log2_denom: 0,
            }; 2]; 3];
            for list in 0..2 {
                if !field.uses(list) {
                    continue;
                }
                let Some(entry) = table.lists[list].get(field.ref_idx[list] as usize) else {
                    return Err(malformed("weight index"));
                };
                for (c, slot) in out.iter_mut().enumerate() {
                    slot[list] = Wp {
                        weight: entry.weight[c],
                        offset: entry.offset[c],
                        log2_denom: table.log2_denom[usize::from(c > 0)],
                    };
                }
            }
            Some(out)
        } else {
            None
        };
        let refs = &refs;
        cutlass_simd::wide::wide(
            #[inline(always)]
            || self.predict_planes((x, y, w, h), field, refs, weights, bit_depth),
        );
        Ok(())
    }

    /// The prediction of each plane of one block from `refs`, stored.
    #[inline(always)]
    fn predict_planes(
        &mut self,
        (x, y, w, h): (usize, usize, usize, usize),
        field: MvField,
        refs: &[Option<std::sync::Arc<crate::picture::Decoded>>; 2],
        weights: Option<[[Wp; 2]; 3]>,
        bit_depth: u32,
    ) {
        for c in 0..3 {
            let (cx, cy, cw, ch) = if c == 0 {
                (x, y, w, h)
            } else {
                (x / 2, y / 2, w / 2, h / 2)
            };
            let mut used = [false; 2];
            for list in 0..2 {
                let Some(pic) = &refs[list] else {
                    continue;
                };
                let mv = field.mv[list];
                let block = if c == 0 {
                    Block {
                        x: cx as isize + (isize::from(mv.x) >> 2),
                        y: cy as isize + (isize::from(mv.y) >> 2),
                        fx: (mv.x & 3) as usize,
                        fy: (mv.y & 3) as usize,
                        w: cw,
                        h: ch,
                        luma: true,
                    }
                } else {
                    Block {
                        x: cx as isize + (isize::from(mv.x) >> 3),
                        y: cy as isize + (isize::from(mv.y) >> 3),
                        fx: (mv.x & 7) as usize,
                        fy: (mv.y & 7) as usize,
                        w: cw,
                        h: ch,
                        luma: false,
                    }
                };
                // The reference's own size: a damaged stream may name a
                // picture of another size.
                let sub = usize::from(c > 0);
                let plane = Plane {
                    samples: &pic.planes[c],
                    width: pic.width >> sub,
                    height: pic.height >> sub,
                };
                mc::interpolate(
                    &plane,
                    block,
                    bit_depth,
                    &mut self.pred[list][..cw * ch],
                    &mut self.mc,
                );
                used[list] = true;
            }
            let (p0, p1) = self.pred.split_at(1);
            let pred = [
                used[0].then(|| &p0[0][..cw * ch]),
                used[1].then(|| &p1[0][..cw * ch]),
            ];
            let (dst, stride) = self.out.from(c, cx, cy);
            mc::store(dst, stride, cw, ch, bit_depth, pred, weights.map(|w| w[c]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Mv, scale};

    #[test]
    fn scaling_by_equal_distances_keeps_the_vector() {
        let mv = Mv { x: 13, y: -7 };
        assert_eq!(scale(mv, 3, 3), mv);
        assert_eq!(scale(mv, 1, 1), mv);
    }

    #[test]
    fn scaling_doubles_for_twice_the_distance() {
        let mv = Mv { x: 8, y: -8 };
        assert_eq!(scale(mv, 1, 2), Mv { x: 16, y: -16 });
        assert_eq!(scale(mv, 2, -2), Mv { x: -8, y: 8 });
    }
}
