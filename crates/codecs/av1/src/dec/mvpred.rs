//! Motion vector prediction (spec 7.10.2 – 7.10.4) and motion field estimation (7.9).

use crate::dsp::Pixel;

use super::coder::Coder;
use crate::spec::tables::DIV_MULT;

use super::frame::RefData;
use super::obu::{FrameHeader, SequenceHeader, TRANSLATION, relative_dist};
use super::tile::TileDec;
use super::types::{
    GLOBAL_GLOBALMV, GLOBALMV, INTRA_FRAME, MiInfo, Mv, NONE, bh4, block_height, block_width, bw4,
    has_newmv,
};

const REF_CAT_LEVEL: u32 = 640;
const MV_BORDER: i32 = 128;

impl<P: Pixel, C: Coder> TileDec<'_, P, C> {
    fn lower_mv_precision(&self, mv: &mut Mv) {
        if self.hdr.allow_high_precision_mv {
            return;
        }
        for v in [&mut mv.row, &mut mv.col] {
            let x = i32::from(*v);
            *v = if self.hdr.force_integer_mv {
                let a = x.abs();
                let a_int = (a + 3) >> 3;
                if x > 0 {
                    (a_int << 3) as i16
                } else {
                    (-(a_int << 3)) as i16
                }
            } else {
                // An odd component moves one towards zero.
                ((x + i32::from(x < 0)) & !1) as i16
            };
        }
    }

    fn setup_global_mv(&self, ref_list: usize) -> Mv {
        let rf = self.b.ref_frame[ref_list];
        let mut mv = Mv::ZERO;
        if rf > INTRA_FRAME {
            let typ = self.hdr.gm_type[rf as usize];
            let gm = &self.hdr.gm_params[rf as usize];
            if typ == TRANSLATION {
                mv = Mv::new(gm[0] >> 13, gm[1] >> 13);
            } else if typ > TRANSLATION {
                let bw = block_width(self.b.bsize) as i32;
                let bh = block_height(self.b.bsize) as i32;
                let x = self.b.mi_col as i32 * 4 + bw / 2 - 1;
                let y = self.b.mi_row as i32 * 4 + bh / 2 - 1;
                let xc = (gm[2] - (1 << 16)) * x + gm[3] * y + gm[0];
                let yc = gm[4] * x + (gm[5] - (1 << 16)) * y + gm[1];
                if self.hdr.allow_high_precision_mv {
                    mv = Mv::new(round2_signed(yc, 13), round2_signed(xc, 13));
                } else {
                    mv = Mv::new(round2_signed(yc, 14) * 2, round2_signed(xc, 14) * 2);
                }
            }
        }
        self.lower_mv_precision(&mut mv);
        mv
    }

    pub fn find_mv_stack(&mut self, is_compound: bool) {
        let w4 = bw4(self.b.bsize);
        let h4 = bh4(self.b.bsize);
        self.b.num_mv_found = 0;
        self.b.new_mv_count = 0;
        self.b.global_mvs[0] = self.setup_global_mv(0);
        if is_compound {
            self.b.global_mvs[1] = self.setup_global_mv(1);
        }
        let mut found = false;
        self.scan_row(-1, is_compound, &mut found);
        let mut found_above = found;
        found = false;
        self.scan_col(-1, is_compound, &mut found);
        let mut found_left = found;
        found = false;
        if w4.max(h4) <= 16 {
            self.scan_point(-1, w4 as isize, is_compound, &mut found);
        }
        if found {
            found_above = true;
        }
        let close_matches = usize::from(found_above) + usize::from(found_left);
        let num_nearest = self.b.num_mv_found;
        let num_new = self.b.new_mv_count;
        for i in 0..num_nearest {
            self.b.weight_stack[i] += REF_CAT_LEVEL;
        }
        self.b.zero_mv_ctx = 0;
        if self.hdr.use_ref_frame_mvs {
            self.temporal_scan(is_compound);
        }
        found = false;
        self.scan_point(-1, -1, is_compound, &mut found);
        if found {
            found_above = true;
        }
        found = false;
        self.scan_row(-3, is_compound, &mut found);
        if found {
            found_above = true;
        }
        found = false;
        self.scan_col(-3, is_compound, &mut found);
        if found {
            found_left = true;
        }
        found = false;
        if h4 > 1 {
            self.scan_row(-5, is_compound, &mut found);
        }
        if found {
            found_above = true;
        }
        found = false;
        if w4 > 1 {
            self.scan_col(-5, is_compound, &mut found);
        }
        if found {
            found_left = true;
        }
        let total_matches = usize::from(found_above) + usize::from(found_left);
        self.sort_stack(0, num_nearest, is_compound);
        let nf = self.b.num_mv_found;
        self.sort_stack(num_nearest, nf, is_compound);
        if self.b.num_mv_found < 2 {
            self.extra_search(is_compound);
        }
        // Context and clamping (7.10.2.14).
        let nf = self.b.num_mv_found;
        for idx in 0..nf {
            let mut z = 0;
            if idx + 1 < nf {
                let w0 = self.b.weight_stack[idx];
                let w1 = self.b.weight_stack[idx + 1];
                if w0 >= REF_CAT_LEVEL {
                    if w1 < REF_CAT_LEVEL {
                        z = 1;
                    }
                } else {
                    z = 2;
                }
            }
            self.b.drl_ctx_stack[idx] = z;
        }
        let bw = block_width(self.b.bsize) as i32;
        let bh = block_height(self.b.bsize) as i32;
        for list in 0..1 + usize::from(is_compound) {
            for idx in 0..nf {
                let mv = self.b.ref_stack_mv[idx][list];
                let r = self.clamp_mv_row(i32::from(mv.row), MV_BORDER + bh * 8);
                let c = self.clamp_mv_col(i32::from(mv.col), MV_BORDER + bw * 8);
                self.b.ref_stack_mv[idx][list] = Mv::new(r, c);
            }
        }
        if close_matches == 0 {
            self.b.new_mv_ctx = total_matches.min(1);
            self.b.ref_mv_ctx = total_matches;
        } else if close_matches == 1 {
            self.b.new_mv_ctx = 3 - num_new.min(1);
            self.b.ref_mv_ctx = 2 + total_matches;
        } else {
            self.b.new_mv_ctx = 5 - num_new.min(1);
            self.b.ref_mv_ctx = 5;
        }
    }

    fn clamp_mv_row(&self, mv: i32, border: i32) -> i32 {
        let h4 = bh4(self.b.bsize) as i32;
        let to_top = -((self.b.mi_row as i32 * 4) * 8);
        let to_bottom = ((self.fs.mi_rows as i32 - h4 - self.b.mi_row as i32) * 4) * 8;
        mv.clamp(to_top - border, to_bottom + border)
    }

    fn clamp_mv_col(&self, mv: i32, border: i32) -> i32 {
        let w4 = bw4(self.b.bsize) as i32;
        let to_left = -((self.b.mi_col as i32 * 4) * 8);
        let to_right = ((self.fs.mi_cols as i32 - w4 - self.b.mi_col as i32) * 4) * 8;
        mv.clamp(to_left - border, to_right + border)
    }

    /// Whether the luma 4x4 at (`r`, `c`), inside the tile and at most one
    /// row above and one column left or right of the current superblock,
    /// has been decoded in this frame (the spec's "RefFrames has been
    /// written"), from the BlockDecoded flags.
    fn decoded(&self, r: isize, c: isize) -> bool {
        let m = self.sb_size4 as isize - 1;
        let r0 = self.b.mi_row as isize & !m;
        let c0 = self.b.mi_col as isize & !m;
        let y = (r - r0 + 1).clamp(0, 34) as usize;
        let x = (c - c0 + 1).clamp(0, 34) as usize;
        self.block_decoded[0][y][x]
    }

    fn inside(&self, r: isize, c: isize) -> bool {
        c >= self.mi_col_start as isize
            && c < self.mi_col_end as isize
            && r >= self.mi_row_start as isize
            && r < self.mi_row_end as isize
    }

    /// The row scan of 7.10.2.2 (`delta_row` is -1, -3 or -5).
    fn scan_row(&mut self, delta_row: isize, is_compound: bool, found: &mut bool) {
        let (mi_row, mi_col) = (self.b.mi_row, self.b.mi_col);
        let w4 = bw4(self.b.bsize);
        let mut delta_col = 0usize;
        let mut delta_row = delta_row;
        let mut min_len = 1;
        if delta_row < -1 {
            delta_row += (mi_row & 1) as isize;
            delta_col = 1 - (mi_col & 1);
            min_len = 2;
        }
        if w4 >= 16 {
            min_len = 4;
        }
        let r = mi_row as isize + delta_row;
        // The positions inside the tile: rows are checked once, columns
        // only end on the right (they start at the block's).
        if r < self.mi_row_start as isize || r >= self.mi_row_end as isize {
            return;
        }
        let c0 = mi_col + delta_col;
        let end4 = w4
            .min(self.fs.mi_cols - mi_col)
            .min(16)
            .min(self.mi_col_end.saturating_sub(c0));
        let row = r as usize * self.fs.mi_stride + c0;
        let mut i = 0usize;
        while i < end4 {
            let m = self.fs.mi[row + i];
            let len = w4.min(bw4(m.bsize)).max(min_len);
            self.add_ref_mv_candidate(&m, is_compound, len as u32 * 2, found);
            i += len;
        }
    }

    /// The column scan of 7.10.2.3 (`delta_col` is -1, -3 or -5).
    fn scan_col(&mut self, delta_col: isize, is_compound: bool, found: &mut bool) {
        let (mi_row, mi_col) = (self.b.mi_row, self.b.mi_col);
        let h4 = bh4(self.b.bsize);
        let mut delta_row = 0usize;
        let mut delta_col = delta_col;
        let mut min_len = 1;
        if delta_col < -1 {
            delta_row = 1 - (mi_row & 1);
            delta_col += (mi_col & 1) as isize;
            min_len = 2;
        }
        if h4 >= 16 {
            min_len = 4;
        }
        let c = mi_col as isize + delta_col;
        if c < self.mi_col_start as isize || c >= self.mi_col_end as isize {
            return;
        }
        let r0 = mi_row + delta_row;
        let end4 = h4
            .min(self.fs.mi_rows - mi_row)
            .min(16)
            .min(self.mi_row_end.saturating_sub(r0));
        let stride = self.fs.mi_stride;
        let at = r0 * stride + c as usize;
        let mut i = 0usize;
        while i < end4 {
            let m = self.fs.mi[at + i * stride];
            let len = h4.min(bh4(m.bsize)).max(min_len);
            self.add_ref_mv_candidate(&m, is_compound, len as u32 * 2, found);
            i += len;
        }
    }

    fn scan_point(
        &mut self,
        delta_row: isize,
        delta_col: isize,
        is_compound: bool,
        found: &mut bool,
    ) {
        let r = self.b.mi_row as isize + delta_row;
        let c = self.b.mi_col as isize + delta_col;
        if self.inside(r, c) && self.decoded(r, c) {
            let m = *self.mi(r as usize, c as usize);
            self.add_ref_mv_candidate(&m, is_compound, 4, found);
        }
    }

    #[inline(always)]
    fn add_ref_mv_candidate(
        &mut self,
        m: &MiInfo,
        is_compound: bool,
        weight: u32,
        found: &mut bool,
    ) {
        if !m.is_inter {
            return;
        }
        if !is_compound {
            for cand_list in 0..2 {
                if m.ref_frame[cand_list] == self.b.ref_frame[0] {
                    // search stack
                    let large = block_width(m.bsize).min(block_height(m.bsize)) >= 8;
                    let mut cand = if (m.y_mode == GLOBALMV || m.y_mode == GLOBAL_GLOBALMV)
                        && self.hdr.gm_type[self.b.ref_frame[0] as usize] > TRANSLATION
                        && large
                    {
                        self.b.global_mvs[0]
                    } else {
                        m.mv[cand_list]
                    };
                    self.lower_mv_precision(&mut cand);
                    if has_newmv(m.y_mode) {
                        self.b.new_mv_count += 1;
                    }
                    *found = true;
                    let n = self.b.num_mv_found;
                    if let Some(idx) = (0..n).find(|&i| self.b.ref_stack_mv[i][0] == cand) {
                        self.b.weight_stack[idx] += weight;
                    } else if n < 8 {
                        self.b.ref_stack_mv[n][0] = cand;
                        self.b.weight_stack[n] = weight;
                        self.b.num_mv_found += 1;
                    }
                }
            }
        } else if m.ref_frame[0] == self.b.ref_frame[0] && m.ref_frame[1] == self.b.ref_frame[1] {
            let mut cands = m.mv;
            if m.y_mode == GLOBAL_GLOBALMV {
                for l in 0..2 {
                    if self.hdr.gm_type[self.b.ref_frame[l] as usize] > TRANSLATION {
                        cands[l] = self.b.global_mvs[l];
                    }
                }
            }
            for c in cands.iter_mut() {
                self.lower_mv_precision(c);
            }
            *found = true;
            let n = self.b.num_mv_found;
            if let Some(idx) = (0..n).find(|&i| self.b.ref_stack_mv[i] == cands) {
                self.b.weight_stack[idx] += weight;
            } else if n < 8 {
                self.b.ref_stack_mv[n] = cands;
                self.b.weight_stack[n] = weight;
                self.b.num_mv_found += 1;
            }
            if has_newmv(m.y_mode) {
                self.b.new_mv_count += 1;
            }
        }
    }

    fn temporal_scan(&mut self, is_compound: bool) {
        let w4 = bw4(self.b.bsize);
        let h4 = bh4(self.b.bsize);
        let step_w = if w4 >= 16 { 4 } else { 2 };
        let step_h = if h4 >= 16 { 4 } else { 2 };
        let dist = self.b.ref_frame.map(|rf| {
            relative_dist(
                self.seq,
                self.hdr.order_hint,
                self.hdr.order_hints[rf.max(0) as usize],
            )
        });
        // Neighbouring positions often hold the same field entry.
        let mut memo = None;
        // The grid positions below and right of the block's corner: only
        // the bottom and right tile edges bound them.
        let (mi_row, mi_col) = (self.b.mi_row, self.b.mi_col);
        let w8 = self.fs.mi_cols >> 1;
        let mut dr = 0;
        while dr < h4.min(16) {
            let r = (mi_row + dr) | 1;
            if r >= self.mi_row_end {
                break;
            }
            let mut dc = 0;
            while dc < w4.min(16) {
                let c = (mi_col + dc) | 1;
                if c >= self.mi_col_end {
                    break;
                }
                if dr == 0 && dc == 0 {
                    self.b.zero_mv_ctx = 1;
                }
                let entry = self
                    .fs
                    .motion_field
                    .get((r >> 1) * w8 + (c >> 1))
                    .copied()
                    .unwrap_or_default();
                if entry.offset != 0 {
                    self.add_tpl_entry(dr == 0 && dc == 0, entry, is_compound, dist, &mut memo);
                }
                dc += step_w;
            }
            dr += step_h;
        }
        let allow_ext = (2..16).contains(&h4) && (2..16).contains(&w4);
        if allow_ext {
            let pos = [
                (h4 as isize, -2isize),
                (h4 as isize, w4 as isize),
                (h4 as isize - 2, w4 as isize),
            ];
            for (dr, dc) in pos {
                let row = (self.b.mi_row & 15) as isize + dr;
                let col = (self.b.mi_col & 15) as isize + dc;
                if (0..16).contains(&row) && (0..16).contains(&col) {
                    self.add_tpl_ref_mv(dr, dc, is_compound, dist, &mut memo);
                }
            }
        }
    }

    /// `dist`: the distances from this frame to the block's references.
    /// `memo`: the last field entry seen and its candidates (projected,
    /// precision lowered).
    fn add_tpl_ref_mv(
        &mut self,
        dr: isize,
        dc: isize,
        is_compound: bool,
        dist: [i32; 2],
        memo: &mut Option<(MfEntry, [Mv; 2])>,
    ) {
        let r = (self.b.mi_row as isize + dr) | 1;
        let c = (self.b.mi_col as isize + dc) | 1;
        if !self.inside(r, c) {
            return;
        }
        let x8 = (c >> 1) as usize;
        let y8 = (r >> 1) as usize;
        let w8 = self.fs.mi_cols >> 1;
        if dr == 0 && dc == 0 {
            self.b.zero_mv_ctx = 1;
        }
        let entry = self
            .fs
            .motion_field
            .get(y8 * w8 + x8)
            .copied()
            .unwrap_or_default();
        if entry.offset == 0 {
            return;
        }
        self.add_tpl_entry(dr == 0 && dc == 0, entry, is_compound, dist, memo);
    }

    /// The rest of [`Self::add_tpl_ref_mv`] for a field entry `entry` with
    /// a motion vector (`first`: the position is the block's corner).
    #[inline(always)]
    fn add_tpl_entry(
        &mut self,
        first: bool,
        entry: MfEntry,
        is_compound: bool,
        dist: [i32; 2],
        memo: &mut Option<(MfEntry, [Mv; 2])>,
    ) {
        let cands = match *memo {
            Some((e, c)) if e == entry => c,
            _ => {
                let mut c = [field_mv(entry, dist[0]), Mv::ZERO];
                if is_compound {
                    c[1] = field_mv(entry, dist[1]);
                }
                for m in c.iter_mut() {
                    self.lower_mv_precision(m);
                }
                *memo = Some((entry, c));
                c
            }
        };
        if !is_compound {
            let cand = cands[0];
            if first {
                let g = self.b.global_mvs[0];
                self.b.zero_mv_ctx = usize::from(
                    (i32::from(cand.row) - i32::from(g.row)).abs() >= 16
                        || (i32::from(cand.col) - i32::from(g.col)).abs() >= 16,
                );
            }
            let n = self.b.num_mv_found;
            if let Some(idx) = (0..n).find(|&i| self.b.ref_stack_mv[i][0] == cand) {
                self.b.weight_stack[idx] += 2;
            } else if n < 8 {
                self.b.ref_stack_mv[n][0] = cand;
                self.b.weight_stack[n] = 2;
                self.b.num_mv_found += 1;
            }
        } else {
            let [c0, c1] = cands;
            if first {
                let g = self.b.global_mvs;
                let far = |a: Mv, b: Mv| {
                    (i32::from(a.row) - i32::from(b.row)).abs() >= 16
                        || (i32::from(a.col) - i32::from(b.col)).abs() >= 16
                };
                self.b.zero_mv_ctx = usize::from(far(c0, g[0]) || far(c1, g[1]));
            }
            let n = self.b.num_mv_found;
            if let Some(idx) = (0..n).find(|&i| self.b.ref_stack_mv[i] == [c0, c1]) {
                self.b.weight_stack[idx] += 2;
            } else if n < 8 {
                self.b.ref_stack_mv[n] = [c0, c1];
                self.b.weight_stack[n] = 2;
                self.b.num_mv_found += 1;
            }
        }
    }

    fn sort_stack(&mut self, start: usize, end: usize, _is_compound: bool) {
        let mut end = end;
        while end > start {
            let mut new_end = start;
            for idx in start + 1..end {
                if self.b.weight_stack[idx - 1] < self.b.weight_stack[idx] {
                    self.b.weight_stack.swap(idx - 1, idx);
                    self.b.ref_stack_mv.swap(idx - 1, idx);
                    new_end = idx;
                }
            }
            end = new_end;
        }
    }

    fn extra_search(&mut self, is_compound: bool) {
        let mut ref_id: [[Mv; 2]; 2] = [[Mv::ZERO; 2]; 2];
        let mut ref_diff: [[Mv; 2]; 2] = [[Mv::ZERO; 2]; 2];
        let mut id_count = [0usize; 2];
        let mut diff_count = [0usize; 2];
        let w4 = bw4(self.b.bsize)
            .min(16)
            .min(self.fs.mi_cols - self.b.mi_col);
        let h4 = bh4(self.b.bsize)
            .min(16)
            .min(self.fs.mi_rows - self.b.mi_row);
        let num4x4 = w4.min(h4);
        let sign_bias = self.hdr.ref_frame_sign_bias;
        for pass in 0..2 {
            let mut idx = 0;
            while idx < num4x4 && self.b.num_mv_found < 2 {
                let (r, c) = if pass == 0 {
                    (self.b.mi_row as isize - 1, (self.b.mi_col + idx) as isize)
                } else {
                    ((self.b.mi_row + idx) as isize, self.b.mi_col as isize - 1)
                };
                if !self.inside(r, c) {
                    break;
                }
                let m = *self.mi(r as usize, c as usize);
                // add_extra_mv_candidate
                if is_compound {
                    for cand_list in 0..2 {
                        let cand_ref = m.ref_frame[cand_list];
                        if cand_ref > INTRA_FRAME {
                            for list in 0..2 {
                                let mut cand = m.mv[cand_list];
                                if cand_ref == self.b.ref_frame[list] && id_count[list] < 2 {
                                    ref_id[list][id_count[list]] = cand;
                                    id_count[list] += 1;
                                } else if diff_count[list] < 2 {
                                    if sign_bias[cand_ref as usize]
                                        != sign_bias[self.b.ref_frame[list] as usize]
                                    {
                                        cand.row = cand.row.wrapping_neg();
                                        cand.col = cand.col.wrapping_neg();
                                    }
                                    ref_diff[list][diff_count[list]] = cand;
                                    diff_count[list] += 1;
                                }
                            }
                        }
                    }
                } else {
                    for cand_list in 0..2 {
                        let cand_ref = m.ref_frame[cand_list];
                        if cand_ref > INTRA_FRAME {
                            let mut cand = m.mv[cand_list];
                            if sign_bias[cand_ref as usize]
                                != sign_bias[self.b.ref_frame[0] as usize]
                            {
                                cand.row = cand.row.wrapping_neg();
                                cand.col = cand.col.wrapping_neg();
                            }
                            let n = self.b.num_mv_found;
                            if !(0..n).any(|i| self.b.ref_stack_mv[i][0] == cand) {
                                self.b.ref_stack_mv[n][0] = cand;
                                self.b.weight_stack[n] = 2;
                                self.b.num_mv_found += 1;
                            }
                        }
                    }
                }
                idx += if pass == 0 {
                    bw4(m.bsize)
                } else {
                    bh4(m.bsize)
                };
            }
        }
        if is_compound {
            let mut combined = [[Mv::ZERO; 2]; 2];
            for list in 0..2 {
                let mut cc = 0;
                for idx in 0..id_count[list] {
                    combined[cc][list] = ref_id[list][idx];
                    cc += 1;
                }
                let mut idx = 0;
                while idx < diff_count[list] && cc < 2 {
                    combined[cc][list] = ref_diff[list][idx];
                    cc += 1;
                    idx += 1;
                }
                while cc < 2 {
                    combined[cc][list] = self.b.global_mvs[list];
                    cc += 1;
                }
            }
            let n = self.b.num_mv_found;
            if n == 1 {
                if combined[0] == self.b.ref_stack_mv[0] {
                    self.b.ref_stack_mv[n] = combined[1];
                } else {
                    self.b.ref_stack_mv[n] = combined[0];
                }
                self.b.weight_stack[n] = 2;
                self.b.num_mv_found += 1;
            } else {
                for c in combined {
                    let n = self.b.num_mv_found;
                    self.b.ref_stack_mv[n] = c;
                    self.b.weight_stack[n] = 2;
                    self.b.num_mv_found += 1;
                }
            }
        } else {
            for idx in self.b.num_mv_found..2 {
                self.b.ref_stack_mv[idx][0] = self.b.global_mvs[0];
            }
        }
    }

    // ------------------------------------------------------------------
    // Warp samples (7.10.4).

    pub fn find_warp_samples(&mut self) {
        self.b.num_samples = 0;
        self.b.num_samples_scanned = 0;
        let w4 = bw4(self.b.bsize);
        let h4 = bh4(self.b.bsize);
        let mut do_top_left = true;
        let mut do_top_right = true;
        let (r, c) = (self.b.mi_row, self.b.mi_col);
        if self.b.avail_u {
            let src_w = bw4(self.mi(r - 1, c).bsize);
            if w4 <= src_w {
                let col_offset = -((c & (src_w - 1)) as isize);
                if col_offset < 0 {
                    do_top_left = false;
                }
                if col_offset + src_w as isize > w4 as isize {
                    do_top_right = false;
                }
                self.add_sample(-1, 0);
            } else {
                let mut i = 0;
                while i < w4.min(self.fs.mi_cols - c) {
                    let src_w = bw4(self.mi(r - 1, c + i).bsize);
                    let step = w4.min(src_w);
                    self.add_sample(-1, i as isize);
                    i += step;
                }
            }
        }
        if self.b.avail_l {
            let src_h = bh4(self.mi(r, c - 1).bsize);
            if h4 <= src_h {
                let row_offset = -((r & (src_h - 1)) as isize);
                if row_offset < 0 {
                    do_top_left = false;
                }
                self.add_sample(0, -1);
            } else {
                let mut i = 0;
                while i < h4.min(self.fs.mi_rows - r) {
                    let src_h = bh4(self.mi(r + i, c - 1).bsize);
                    let step = h4.min(src_h);
                    self.add_sample(i as isize, -1);
                    i += step;
                }
            }
        }
        if do_top_left {
            self.add_sample(-1, -1);
        }
        if do_top_right && w4.max(h4) <= 16 {
            self.add_sample(-1, w4 as isize);
        }
        if self.b.num_samples == 0 && self.b.num_samples_scanned > 0 {
            self.b.num_samples = 1;
        }
    }

    fn add_sample(&mut self, dr: isize, dc: isize) {
        if self.b.num_samples_scanned >= 8 {
            return;
        }
        let r = self.b.mi_row as isize + dr;
        let c = self.b.mi_col as isize + dc;
        if !self.inside(r, c) {
            return;
        }
        if !self.decoded(r, c) {
            return;
        }
        let m = *self.mi(r as usize, c as usize);
        if m.ref_frame[0] != self.b.ref_frame[0] || m.ref_frame[1] != NONE {
            return;
        }
        let cw4 = bw4(m.bsize);
        let ch4 = bh4(m.bsize);
        let cand_row = r as usize & !(ch4 - 1);
        let cand_col = c as usize & !(cw4 - 1);
        let mid_y = (cand_row * 4 + ch4 * 2) as i32 - 1;
        let mid_x = (cand_col * 4 + cw4 * 2) as i32 - 1;
        let bsz = block_width(self.b.bsize).max(block_height(self.b.bsize)) as i32;
        let threshold = bsz.clamp(16, 112);
        let cmv = self.mi(cand_row, cand_col).mv[0];
        let dr = (i32::from(cmv.row) - i32::from(self.b.mv[0].row)).abs();
        let dc = (i32::from(cmv.col) - i32::from(self.b.mv[0].col)).abs();
        let valid = dr + dc <= threshold;
        let cand = [
            mid_y * 8,
            mid_x * 8,
            mid_y * 8 + i32::from(cmv.row),
            mid_x * 8 + i32::from(cmv.col),
        ];
        self.b.num_samples_scanned += 1;
        if !valid && self.b.num_samples_scanned > 1 {
            return;
        }
        self.b.cand_list[self.b.num_samples] = cand;
        if valid {
            self.b.num_samples += 1;
        }
    }
}

#[inline]
fn round2_signed(x: i32, n: u32) -> i32 {
    crate::dsp::round2_signed(x, n)
}

/// One projected motion field entry: the source motion vector and the
/// distance it covers (`ref_offset`, 0 when no projection landed here).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MfEntry {
    pub mv: Mv,
    pub offset: i8,
}

/// Motion field estimation (7.9). Instead of the spec's per-reference
/// MotionFieldMvs, one entry per 8x8 position keeps the projected source
/// motion vector; `MotionFieldMvs[ref]` is `mv_projection(mv, refToDst, offset)`.
///
/// A projection never leaves its source's group of 8 rows of 8x8 units, so
/// the groups are computed independently, on up to `threads` threads.
pub fn motion_field_estimation(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    refs: &[Option<std::sync::Arc<RefData>>; 8],
    threads: usize,
) -> Vec<MfEntry> {
    motion_field_plan(seq, hdr, refs).run(threads)
}

/// The projections of motion field estimation (7.9) for one frame. Every
/// band of 8 rows of 8x8 positions only receives projections from the same
/// rows of the references, so bands can be computed separately.
pub struct MfPlan {
    passes: Vec<Projection>,
    pub w8: usize,
    pub h8: usize,
}

impl MfPlan {
    /// Number of bands of 8 rows.
    pub fn bands(&self) -> usize {
        if self.passes.is_empty() || self.w8 == 0 {
            0
        } else {
            self.h8.div_ceil(8)
        }
    }

    /// Fills band `k`: `band` holds field rows `8 k..` (whole rows, all
    /// unset).
    pub fn run_band(&self, k: usize, band: &mut [MfEntry]) {
        for pass in &self.passes {
            pass.run(band, 8 * k, self.w8, self.h8);
        }
    }

    /// The whole field, on up to `threads` threads.
    pub fn run(&self, threads: usize) -> Vec<MfEntry> {
        let (w8, h8) = (self.w8, self.h8);
        let mut field = vec![MfEntry::default(); w8 * h8];
        if self.bands() == 0 {
            return field;
        }
        let groups = h8.div_ceil(8);
        let t = threads.clamp(1, groups);
        let per = groups.div_ceil(t) * 8 * w8;
        let passes = &self.passes;
        let run = |y0: usize, band: &mut [MfEntry]| {
            for pass in passes {
                pass.run(band, y0, w8, h8);
            }
        };
        if t == 1 {
            run(0, &mut field);
        } else {
            std::thread::scope(|s| {
                let mut chunks = field.chunks_mut(per).enumerate();
                let first = chunks.next();
                for (i, band) in chunks {
                    s.spawn(move || run(i * per / w8, band));
                }
                if let Some((_, band)) = first {
                    run(0, band);
                }
            });
        }
        field
    }
}

/// The [`MfPlan`] of the frame `hdr`.
pub fn motion_field_plan(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    refs: &[Option<std::sync::Arc<RefData>>; 8],
) -> MfPlan {
    let mi_cols = hdr.mi_cols as usize;
    let mi_rows = hdr.mi_rows as usize;
    let w8 = mi_cols >> 1;
    let h8 = mi_rows >> 1;
    let last_idx = hdr.ref_frame_idx[0] as usize;
    let cur_gold = hdr.order_hints[4];
    let last_alt = refs[last_idx]
        .as_ref()
        .map_or(0, |r| r.saved_order_hints[7]);
    // The projections to run, in order. `project` returns false when the
    // reference cannot be projected (the spec's early return 0).
    let mut passes: Vec<Projection> = Vec::new();
    let mut project = |src: usize, dst_sign: i32| -> bool {
        let src_idx = hdr.ref_frame_idx[src - 1] as usize;
        let Some(rd) = refs[src_idx].clone() else {
            return false;
        };
        if rd.mi_rows != mi_rows
            || rd.mi_cols != mi_cols
            || rd.frame_type == 0
            || rd.frame_type == 2
        {
            return false;
        }
        let ref_to_cur = relative_dist(seq, hdr.order_hints[src], hdr.order_hint);
        if ref_to_cur.abs() > 31 {
            return true;
        }
        let offsets: [i32; 8] = std::array::from_fn(|r| {
            relative_dist(seq, hdr.order_hints[src], rd.saved_order_hints[r])
        });
        passes.push(Projection {
            rd,
            dst_sign,
            ref_to_cur,
            offsets,
        });
        true
    };
    if last_alt != cur_gold {
        project(1, -1);
    }
    let mut ref_stamp = 3 - 2;
    if relative_dist(seq, hdr.order_hints[5], hdr.order_hint) > 0 && project(5, 1) {
        ref_stamp -= 1;
    }
    if relative_dist(seq, hdr.order_hints[6], hdr.order_hint) > 0 && project(6, 1) {
        ref_stamp -= 1;
    }
    if relative_dist(seq, hdr.order_hints[7], hdr.order_hint) > 0 && ref_stamp >= 0 && project(7, 1)
    {
        ref_stamp -= 1;
    }
    if ref_stamp >= 0 {
        project(2, -1);
    }
    MfPlan { passes, w8, h8 }
}

/// One reference's projection in [`motion_field_estimation`].
struct Projection {
    rd: std::sync::Arc<RefData>,
    dst_sign: i32,
    ref_to_cur: i32,
    offsets: [i32; 8],
}

impl Projection {
    /// Projects the source rows of `band`, which holds the field rows from
    /// `y0` on (whole groups of 8 rows).
    fn run(&self, band: &mut [MfEntry], y0: usize, w8: usize, h8: usize) {
        let rows = band.len() / w8;
        let rd = &*self.rd;
        let dst_sign = self.dst_sign;
        let num = self.ref_to_cur * dst_sign;
        for y8 in y0..y0 + rows {
            let row_ref = &rd.mf_ref[y8 * w8..y8 * w8 + w8];
            let row_mv = &rd.mf_mv[y8 * w8..y8 * w8 + w8];
            let mut x8 = 0;
            while x8 < w8 {
                let src_ref = row_ref[x8];
                let ref_offset = if src_ref > INTRA_FRAME {
                    self.offsets[src_ref as usize & 7]
                } else {
                    0
                };
                if ref_offset <= 0 || ref_offset > 31 {
                    x8 += 1;
                    continue;
                }
                // A run of positions with this reference and motion vector
                // shares the projection.
                let mv = row_mv[x8];
                let mut end = x8 + 1;
                while end < w8 && row_ref[end] == src_ref && row_mv[end] == mv {
                    end += 1;
                }
                let proj = mv_projection(mv, num, ref_offset);
                let proj_y = project_pos(y8 as i32, i32::from(proj.row), dst_sign, h8 as i32, 0);
                if let Some(py) = proj_y {
                    let row = &mut band[(py as usize - y0) * w8..][..w8];
                    let dx = dst_sign * project_offset(i32::from(proj.col));
                    let e = MfEntry {
                        mv,
                        offset: ref_offset as i8,
                    };
                    for x in x8..end {
                        let base8 = (x as i32 >> 3) << 3;
                        let px = x as i32 + dx;
                        if px >= 0 && px < w8 as i32 && px >= base8 - 8 && px < base8 + 16 {
                            row[px as usize] = e;
                        }
                    }
                }
                x8 = end;
            }
        }
    }
}

/// MotionFieldMvs[ ref ] at one entry.
pub fn field_mv(e: MfEntry, ref_to_dst: i32) -> Mv {
    if e.offset == 0 {
        return Mv::INVALID;
    }
    mv_projection(e.mv, ref_to_dst, i32::from(e.offset))
}

fn mv_projection(mv: Mv, num: i32, den: i32) -> Mv {
    let den = den.min(31);
    let num = num.clamp(-31, 31);
    let p = |v: i16| {
        let scaled =
            crate::dsp::round2_signed(i32::from(v) * num * DIV_MULT[den as usize] as i32, 14);
        scaled.clamp(-(1 << 14) + 1, (1 << 14) - 1)
    };
    Mv::new(p(mv.row), p(mv.col))
}

/// The offset, in 8x8 units, of a projected motion vector component.
#[inline]
fn project_offset(delta: i32) -> i32 {
    if delta >= 0 {
        delta >> (3 + 1 + 2)
    } else {
        -((-delta) >> (3 + 1 + 2))
    }
}

fn project_pos(v8: i32, delta: i32, dst_sign: i32, max8: i32, max_off8: i32) -> Option<i32> {
    let base8 = (v8 >> 3) << 3;
    let v = v8 + dst_sign * project_offset(delta);
    if v < 0 || v >= max8 || v < base8 - max_off8 || v >= base8 + 8 + max_off8 {
        None
    } else {
        Some(v)
    }
}
