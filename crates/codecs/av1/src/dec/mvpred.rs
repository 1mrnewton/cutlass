//! Motion vector prediction (spec 7.10.2 – 7.10.4) and motion field estimation (7.9).

use crate::dsp::Pixel;
use crate::spec::tables::DIV_MULT;

use super::frame::RefData;
use super::obu::{FrameHeader, SequenceHeader, TRANSLATION, relative_dist};
use super::tile::TileDec;
use super::types::{
    GLOBAL_GLOBALMV, GLOBALMV, INTRA_FRAME, Mv, NONE, NOT_WRITTEN, bh4, block_height, block_width,
    bw4, has_newmv,
};

const REF_CAT_LEVEL: u32 = 640;
const MV_BORDER: i32 = 128;

impl<P: Pixel> TileDec<'_, P> {
    fn lower_mv_precision(&self, mv: &mut Mv) {
        if self.hdr.allow_high_precision_mv {
            return;
        }
        for v in [&mut mv.row, &mut mv.col] {
            let x = i32::from(*v);
            if self.hdr.force_integer_mv {
                let a = x.abs();
                let a_int = (a + 3) >> 3;
                *v = if x > 0 {
                    (a_int << 3) as i16
                } else {
                    (-(a_int << 3)) as i16
                };
            } else if x & 1 != 0 {
                *v = if x > 0 {
                    (x - 1) as i16
                } else {
                    (x + 1) as i16
                };
            }
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

    fn inside(&self, r: isize, c: isize) -> bool {
        c >= self.mi_col_start as isize
            && c < self.mi_col_end as isize
            && r >= self.mi_row_start as isize
            && r < self.mi_row_end as isize
    }

    fn scan_row(&mut self, delta_row: isize, is_compound: bool, found: &mut bool) {
        let w4 = bw4(self.b.bsize);
        let end4 = w4.min(self.fs.mi_cols - self.b.mi_col).min(16);
        let mut delta_col = 0isize;
        let use_step16 = w4 >= 16;
        let mut delta_row = delta_row;
        if delta_row.abs() > 1 {
            delta_row += (self.b.mi_row & 1) as isize;
            delta_col = 1 - (self.b.mi_col & 1) as isize;
        }
        let mut i = 0usize;
        while i < end4 {
            let r = self.b.mi_row as isize + delta_row;
            let c = self.b.mi_col as isize + delta_col + i as isize;
            if !self.inside(r, c) {
                break;
            }
            let mut len = w4.min(bw4(self.mi(r as usize, c as usize).bsize));
            if delta_row.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            let weight = len as u32 * 2;
            self.add_ref_mv_candidate(r as usize, c as usize, is_compound, weight, found);
            i += len;
        }
    }

    fn scan_col(&mut self, delta_col: isize, is_compound: bool, found: &mut bool) {
        let h4 = bh4(self.b.bsize);
        let end4 = h4.min(self.fs.mi_rows - self.b.mi_row).min(16);
        let mut delta_row = 0isize;
        let use_step16 = h4 >= 16;
        let mut delta_col = delta_col;
        if delta_col.abs() > 1 {
            delta_row = 1 - (self.b.mi_row & 1) as isize;
            delta_col += (self.b.mi_col & 1) as isize;
        }
        let mut i = 0usize;
        while i < end4 {
            let r = self.b.mi_row as isize + delta_row + i as isize;
            let c = self.b.mi_col as isize + delta_col;
            if !self.inside(r, c) {
                break;
            }
            let mut len = h4.min(bh4(self.mi(r as usize, c as usize).bsize));
            if delta_col.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            let weight = len as u32 * 2;
            self.add_ref_mv_candidate(r as usize, c as usize, is_compound, weight, found);
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
        if self.inside(r, c) && self.mi(r as usize, c as usize).ref_frame[0] != NOT_WRITTEN {
            self.add_ref_mv_candidate(r as usize, c as usize, is_compound, 4, found);
        }
    }

    fn add_ref_mv_candidate(
        &mut self,
        r: usize,
        c: usize,
        is_compound: bool,
        weight: u32,
        found: &mut bool,
    ) {
        let m = *self.mi(r, c);
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
        let mut dr = 0;
        while dr < h4.min(16) {
            let mut dc = 0;
            while dc < w4.min(16) {
                self.add_tpl_ref_mv(dr as isize, dc as isize, is_compound);
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
                    self.add_tpl_ref_mv(dr, dc, is_compound);
                }
            }
        }
    }

    fn add_tpl_ref_mv(&mut self, dr: isize, dc: isize, is_compound: bool) {
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
        let field = |rf: i8| -> Mv {
            self.fs
                .motion_field
                .get(rf as usize)
                .and_then(|f| f.get(y8 * w8 + x8))
                .copied()
                .unwrap_or(Mv::INVALID)
        };
        if !is_compound {
            let mut cand = field(self.b.ref_frame[0]);
            if cand.row == i16::MIN {
                return;
            }
            self.lower_mv_precision(&mut cand);
            if dr == 0 && dc == 0 {
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
            let mut c0 = field(self.b.ref_frame[0]);
            if c0.row == i16::MIN {
                return;
            }
            let mut c1 = field(self.b.ref_frame[1]);
            if c1.row == i16::MIN {
                return;
            }
            self.lower_mv_precision(&mut c0);
            self.lower_mv_precision(&mut c1);
            if dr == 0 && dc == 0 {
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
        let m = *self.mi(r as usize, c as usize);
        if m.ref_frame[0] == NOT_WRITTEN {
            return;
        }
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

/// Motion field estimation (7.9). Returns MotionFieldMvs indexed by reference frame.
pub fn motion_field_estimation(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    refs: &[Option<std::sync::Arc<RefData>>; 8],
) -> Vec<Vec<Mv>> {
    let mi_cols = hdr.mi_cols as usize;
    let mi_rows = hdr.mi_rows as usize;
    let w8 = mi_cols >> 1;
    let h8 = mi_rows >> 1;
    let mut field: Vec<Vec<Mv>> = (0..8).map(|_| vec![Mv::INVALID; w8 * h8]).collect();
    let last_idx = hdr.ref_frame_idx[0] as usize;
    let cur_gold = hdr.order_hints[4];
    let last_alt = refs[last_idx]
        .as_ref()
        .map_or(0, |r| r.saved_order_hints[7]);
    let project = |src: usize, dst_sign: i32, field: &mut Vec<Vec<Mv>>| -> bool {
        let src_idx = hdr.ref_frame_idx[src - 1] as usize;
        let Some(rd) = refs[src_idx].as_ref() else {
            return false;
        };
        if rd.mi_rows != mi_rows
            || rd.mi_cols != mi_cols
            || rd.frame_type == 0
            || rd.frame_type == 2
        {
            return false;
        }
        let rw8 = mi_cols >> 1;
        for y8 in 0..h8 {
            for x8 in 0..w8 {
                let src_ref = rd.mf_ref[y8 * rw8 + x8];
                if src_ref <= INTRA_FRAME {
                    continue;
                }
                let ref_to_cur = relative_dist(seq, hdr.order_hints[src], hdr.order_hint);
                let ref_offset = relative_dist(
                    seq,
                    hdr.order_hints[src],
                    rd.saved_order_hints[src_ref as usize],
                );
                let pos_valid = ref_to_cur.abs() <= 31 && ref_offset.abs() <= 31 && ref_offset > 0;
                if !pos_valid {
                    continue;
                }
                let mv = rd.mf_mv[y8 * rw8 + x8];
                let proj = mv_projection(mv, ref_to_cur * dst_sign, ref_offset);
                let proj_y = project_pos(y8 as i32, i32::from(proj.row), dst_sign, h8 as i32, 0);
                let proj_x = project_pos(x8 as i32, i32::from(proj.col), dst_sign, w8 as i32, 8);
                if let (Some(py), Some(px)) = (proj_y, proj_x) {
                    for dst in 1..8 {
                        let ref_to_dst = relative_dist(seq, hdr.order_hint, hdr.order_hints[dst]);
                        let p = mv_projection(mv, ref_to_dst, ref_offset);
                        field[dst][py as usize * w8 + px as usize] = p;
                    }
                }
            }
        }
        true
    };
    if last_alt != cur_gold {
        project(1, -1, &mut field);
    }
    let mut ref_stamp = 3 - 2;
    if relative_dist(seq, hdr.order_hints[5], hdr.order_hint) > 0 && project(5, 1, &mut field) {
        ref_stamp -= 1;
    }
    if relative_dist(seq, hdr.order_hints[6], hdr.order_hint) > 0 && project(6, 1, &mut field) {
        ref_stamp -= 1;
    }
    if relative_dist(seq, hdr.order_hints[7], hdr.order_hint) > 0
        && ref_stamp >= 0
        && project(7, 1, &mut field)
    {
        ref_stamp -= 1;
    }
    if ref_stamp >= 0 {
        project(2, -1, &mut field);
    }
    field
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

fn project_pos(v8: i32, delta: i32, dst_sign: i32, max8: i32, max_off8: i32) -> Option<i32> {
    let base8 = (v8 >> 3) << 3;
    let offset8 = if delta >= 0 {
        delta >> (3 + 1 + 2)
    } else {
        -((-delta) >> (3 + 1 + 2))
    };
    let v = v8 + dst_sign * offset8;
    if v < 0 || v >= max8 || v < base8 - max_off8 || v >= base8 + 8 + max_off8 {
        None
    } else {
        Some(v)
    }
}
