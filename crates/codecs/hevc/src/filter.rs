//! In-loop filters: the deblocking filter (clause 8.7.2) and sample
//! adaptive offset (clause 8.7.3), one CTB row at a time.
//!
//! Deblocking a CTB row filters its vertical edges, then its horizontal
//! edges, the one on its top boundary included, which changes the last
//! three lines of the row above. Every vertical edge of a line is then
//! filtered before any horizontal edge that reaches the line, as the
//! picture-wide order of the specification has it. SAO of a row reads the
//! deblocked samples of the row and one line on each side, so it runs once
//! the row below is deblocked.

use std::cell::Cell;
use std::sync::atomic::Ordering::Relaxed;

use cutlass_simd::hevc::{deblock, sao};

use crate::ctu::chroma_qp;
use crate::picture::{FLAG_KEEP, Picture, Work, lock};
use crate::pixel::Pixel;

/// `tC′` of table 8-12, indexed by `Q`.
/// Luma lines at the end of a CTB row that wait for the row below before
/// they are final: the three its deblocking changes, the one SAO reads
/// beside them, rounded up to whole chroma lines on the 8×8 grid.
pub const HOLD: usize = 8;

const TC: [i32; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

/// `β′` of table 8-12, indexed by `Q`.
const BETA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];

/// Decisions of one edge segment.
#[derive(Clone, Copy)]
struct Strength {
    beta: i32,
    tc: i32,
    no_p: bool,
    no_q: bool,
    max: i32,
}

/// The samples on both sides of an edge, by line along the edge (0 to 3)
/// and position across it (-4 to 3, `q0` at 0).
trait Edge {
    fn at(&self, line: usize, i: isize) -> i32;
    fn put(&mut self, line: usize, i: isize, value: i32, max: i32);
}

/// A vertical edge: the four lines are rows of one buffer.
struct Ver<'a, P> {
    s: &'a mut [P],
    /// `q0` of line 0.
    pos: usize,
    stride: usize,
}

impl<P: Pixel> Edge for Ver<'_, P> {
    #[inline(always)]
    fn at(&self, line: usize, i: isize) -> i32 {
        self.s[(self.pos + line * self.stride).wrapping_add_signed(i)].i()
    }

    #[inline(always)]
    fn put(&mut self, line: usize, i: isize, value: i32, max: i32) {
        self.s[(self.pos + line * self.stride).wrapping_add_signed(i)] = P::of(value.clamp(0, max));
    }
}

/// A horizontal edge: `p` ends with the rows above it, `q` starts with the
/// rows below, the four lines are columns from `x`.
struct Hor<'a, P> {
    p: &'a mut [P],
    q: &'a mut [P],
    x: usize,
    stride: usize,
}

impl<P: Pixel> Edge for Hor<'_, P> {
    #[inline(always)]
    fn at(&self, line: usize, i: isize) -> i32 {
        if i >= 0 {
            self.q[i as usize * self.stride + self.x + line].i()
        } else {
            self.p[self.p.len() - (-i) as usize * self.stride + self.x + line].i()
        }
    }

    #[inline(always)]
    fn put(&mut self, line: usize, i: isize, value: i32, max: i32) {
        let v = P::of(value.clamp(0, max));
        if i >= 0 {
            self.q[i as usize * self.stride + self.x + line] = v;
        } else {
            let at = self.p.len() - (-i) as usize * self.stride + self.x + line;
            self.p[at] = v;
        }
    }
}

/// Luma edge filtering of four lines (clauses 8.7.2.5.3 and 8.7.2.5.7).
#[inline(always)]
fn luma_edge<E: Edge>(s: &mut E, st: Strength) {
    let Strength {
        beta,
        tc,
        no_p,
        no_q,
        max,
    } = st;
    let dp = |s: &E, line: usize| (s.at(line, -3) - 2 * s.at(line, -2) + s.at(line, -1)).abs();
    let dq = |s: &E, line: usize| (s.at(line, 2) - 2 * s.at(line, 1) + s.at(line, 0)).abs();
    let dp0 = dp(&*s, 0);
    let dp3 = dp(&*s, 3);
    let dq0 = dq(&*s, 0);
    let dq3 = dq(&*s, 3);
    let d0 = dp0 + dq0;
    let d3 = dp3 + dq3;
    if d0 + d3 >= beta {
        return;
    }
    let strong_line = |s: &E, line: usize, d: i32| -> bool {
        2 * d < (beta >> 2)
            && (s.at(line, -4) - s.at(line, -1)).abs() + (s.at(line, 0) - s.at(line, 3)).abs()
                < (beta >> 3)
            && (s.at(line, -1) - s.at(line, 0)).abs() < ((5 * tc + 1) >> 1)
    };
    if strong_line(&*s, 0, d0) && strong_line(&*s, 3, d3) {
        let tc2 = 2 * tc;
        for line in 0..4 {
            let p3 = s.at(line, -4);
            let p2 = s.at(line, -3);
            let p1 = s.at(line, -2);
            let p0 = s.at(line, -1);
            let q0 = s.at(line, 0);
            let q1 = s.at(line, 1);
            let q2 = s.at(line, 2);
            let q3 = s.at(line, 3);
            if !no_p {
                let v0 = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - tc2, p0 + tc2);
                let v1 = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - tc2, p1 + tc2);
                let v2 = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - tc2, p2 + tc2);
                s.put(line, -1, v0, max);
                s.put(line, -2, v1, max);
                s.put(line, -3, v2, max);
            }
            if !no_q {
                let v0 = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - tc2, q0 + tc2);
                let v1 = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - tc2, q1 + tc2);
                let v2 = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - tc2, q2 + tc2);
                s.put(line, 0, v0, max);
                s.put(line, 1, v1, max);
                s.put(line, 2, v2, max);
            }
        }
        return;
    }
    let side = (beta + (beta >> 1)) >> 3;
    let filter_p1 = dp0 + dp3 < side;
    let filter_q1 = dq0 + dq3 < side;
    let half = tc >> 1;
    for line in 0..4 {
        let p2 = s.at(line, -3);
        let p1 = s.at(line, -2);
        let p0 = s.at(line, -1);
        let q0 = s.at(line, 0);
        let q1 = s.at(line, 1);
        let q2 = s.at(line, 2);
        let mut delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if delta.abs() >= tc * 10 {
            continue;
        }
        delta = delta.clamp(-tc, tc);
        if !no_p {
            s.put(line, -1, p0 + delta, max);
            if filter_p1 {
                let dp = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-half, half);
                s.put(line, -2, p1 + dp, max);
            }
        }
        if !no_q {
            s.put(line, 0, q0 - delta, max);
            if filter_q1 {
                let dq = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-half, half);
                s.put(line, 1, q1 + dq, max);
            }
        }
    }
}

/// Chroma edge filtering of four lines (clause 8.7.2.5.5).
#[inline(always)]
fn chroma_edge<E: Edge>(s: &mut E, st: Strength) {
    let tc = st.tc;
    if tc == 0 {
        return;
    }
    for line in 0..4 {
        let p1 = s.at(line, -2);
        let p0 = s.at(line, -1);
        let q0 = s.at(line, 0);
        let q1 = s.at(line, 1);
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        if !st.no_p {
            s.put(line, -1, p0 + delta, st.max);
        }
        if !st.no_q {
            s.put(line, 0, q0 - delta, st.max);
        }
    }
}

/// Inputs shared by every edge of one direction.
struct EdgeQuery<'m> {
    /// Luma position of `p0` and `q0`.
    p: (usize, usize),
    q: (usize, usize),
    bs: u8,
    /// The CTB last looked up and its slice's deblocking offsets.
    memo: &'m Cell<(usize, i32, i32)>,
}

impl Picture {
    /// Average luma QP, offsets of the slice holding `q0`, and the
    /// do-not-touch flags of both sides.
    #[inline(always)]
    fn edge_terms(&self, query: &EdgeQuery<'_>) -> Option<(i32, i32, i32, bool, bool)> {
        let rs = self.ctb_rs(query.q.0, query.q.1);
        let (beta_offset, tc_offset) = {
            let mut memo = query.memo.get();
            if memo.0 != rs {
                let header = &self.slice_of(query.q.0, query.q.1)?.header;
                memo = (rs, header.beta_offset, header.tc_offset);
                query.memo.set(memo);
            }
            (memo.1, memo.2)
        };
        let ip = self.at4(query.p.0, query.p.1);
        let iq = self.at4(query.q.0, query.q.1);
        let qp_at = |at: usize| i32::from(self.qp.get(at) as i8);
        let qp = (qp_at(ip) + qp_at(iq) + 1) >> 1;
        Some((
            qp,
            beta_offset,
            tc_offset,
            self.info.get(ip) & FLAG_KEEP != 0,
            self.info.get(iq) & FLAG_KEEP != 0,
        ))
    }

    /// The kernel's parameters of the luma segment between `p` and `q`.
    #[inline(always)]
    fn segment(
        &self,
        p: (usize, usize),
        q: (usize, usize),
        bs: u8,
        memo: &Cell<(usize, i32, i32)>,
    ) -> deblock::Segment {
        if bs == 0 {
            return deblock::Segment::default();
        }
        let query = EdgeQuery { p, q, bs, memo };
        self.luma_strength(&query)
            .map(|st| deblock::Segment {
                beta: st.beta as i16,
                tc: st.tc as i16,
                no_p: st.no_p,
                no_q: st.no_q,
            })
            .unwrap_or_default()
    }

    #[inline(always)]
    fn luma_strength(&self, query: &EdgeQuery<'_>) -> Option<Strength> {
        let (qp, beta_offset, tc_offset, no_p, no_q) = self.edge_terms(query)?;
        let shift = u32::from(self.sps.bit_depth) - 8;
        let beta = BETA[(qp + beta_offset).clamp(0, 51) as usize] << shift;
        let tc =
            TC[(qp + 2 * (i32::from(query.bs) - 1) + tc_offset).clamp(0, 53) as usize] << shift;
        Some(Strength {
            beta,
            tc,
            no_p,
            no_q,
            max: (1 << self.sps.bit_depth) - 1,
        })
    }

    fn chroma_strength(&self, query: &EdgeQuery<'_>, offset: i32) -> Option<Strength> {
        let (qp, _, tc_offset, no_p, no_q) = self.edge_terms(query)?;
        let shift = u32::from(self.sps.bit_depth) - 8;
        let qpc = chroma_qp((qp + offset).clamp(0, 57));
        let tc = TC[(qpc + 2 + tc_offset).clamp(0, 53) as usize] << shift;
        Some(Strength {
            beta: 0,
            tc,
            no_p,
            no_q,
            max: (1 << self.sps.bit_depth) - 1,
        })
    }

    /// Deblock CTB row `r`: its vertical edges, then its horizontal ones,
    /// the edge on its top boundary included (clause 8.7.2).
    pub fn deblock_row<P: Pixel>(&self, work: &Work<P>, r: usize) {
        let log2 = work.log2_ctb;
        let top = r << log2;
        let bottom = ((r + 1) << log2).min(self.height);
        if top >= bottom {
            return;
        }
        let mut above = r.checked_sub(1).map(|a| lock(&work.rows[work.slot(a)]));
        let mut band = lock(&work.rows[work.slot(r)]);
        self.deblock_vertical(&mut band, top, bottom);
        self.deblock_horizontal(above.as_deref_mut(), &mut band, top, bottom);
        // The strengths of the row are read: zero again for the picture
        // that takes these maps next, cleared here while in cache.
        let ver = self.width / 8 + 1;
        self.bs_ver
            .clear((top >> 2) * ver..bottom.div_ceil(4) * ver);
        self.bs_hor
            .clear((top >> 3) * self.w4..bottom.div_ceil(8) * self.w4);
    }

    fn deblock_vertical<P: Pixel>(&self, band: &mut [Vec<P>; 3], top: usize, bottom: usize) {
        let width = self.width;
        let ver_stride = width / 8 + 1;
        let cw = width / 2;
        let offsets = [self.pps.cb_qp_offset, self.pps.cr_qp_offset];
        let memo = Cell::new((usize::MAX, 0, 0));
        // Luma of 8 to 11 bits: two segments of four lines at a time.
        let bit_depth = u32::from(self.sps.bit_depth);
        let fast = deblock::accelerated() && (P::bytes(&band[0]).is_some() || bit_depth <= 11);
        if fast {
            for y in (top..bottom).step_by(8) {
                for x in (8..width).step_by(8) {
                    let at = (y >> 2) * ver_stride + (x >> 3);
                    let bs0 = self.bs_ver.get(at);
                    let bs1 = if y + 4 < bottom {
                        self.bs_ver.get(at + ver_stride)
                    } else {
                        0
                    };
                    if bs0 == 0 && bs1 == 0 {
                        continue;
                    }
                    let segs = [
                        self.segment((x - 1, y), (x, y), bs0, &memo),
                        self.segment((x - 1, y + 4), (x, y + 4), bs1, &memo),
                    ];
                    if segs.iter().all(|s| s.beta == 0) {
                        continue;
                    }
                    let at = (y - top, x);
                    if let Some(luma) = P::bytes_mut(&mut band[0]) {
                        lines_across(luma, width, at, segs, deblock::luma8_lines);
                    } else if let Some(luma) = P::words_mut(&mut band[0]) {
                        lines_across(luma, width, at, segs, |l, s| {
                            deblock::luma16_lines(l, s, bit_depth)
                        });
                    }
                }
            }
        }
        // Chroma edges lie on the 16×8 grid of luma.
        let (ystep, xstep) = if fast { (8, 16) } else { (4, 8) };
        for y in (top..bottom).step_by(ystep) {
            for x in (xstep..width).step_by(xstep) {
                let bs = self.bs_ver.get((y >> 2) * ver_stride + (x >> 3));
                if bs == 0 || (fast && !(bs == 2 && x % 16 == 0 && y % 8 == 0)) {
                    continue;
                }
                let query = EdgeQuery {
                    p: (x - 1, y),
                    q: (x, y),
                    bs,
                    memo: &memo,
                };
                if !fast && let Some(st) = self.luma_strength(&query) {
                    let mut edge = Ver {
                        s: &mut band[0],
                        pos: (y - top) * width + x,
                        stride: width,
                    };
                    luma_edge(&mut edge, st);
                }
                if bs == 2 && x % 16 == 0 && y % 8 == 0 {
                    for (c, &offset) in offsets.iter().enumerate() {
                        if let Some(st) = self.chroma_strength(&query, offset) {
                            let mut edge = Ver {
                                s: &mut band[c + 1],
                                pos: (y / 2 - top / 2) * cw + x / 2,
                                stride: cw,
                            };
                            chroma_edge(&mut edge, st);
                        }
                    }
                }
            }
        }
    }

    fn deblock_horizontal<P: Pixel>(
        &self,
        mut above: Option<&mut [Vec<P>; 3]>,
        band: &mut [Vec<P>; 3],
        top: usize,
        bottom: usize,
    ) {
        let width = self.width;
        let cw = width / 2;
        let offsets = [self.pps.cb_qp_offset, self.pps.cr_qp_offset];
        let memo = Cell::new((usize::MAX, 0, 0));
        let bit_depth = u32::from(self.sps.bit_depth);
        let fast = deblock::accelerated() && (P::bytes(&band[0]).is_some() || bit_depth <= 11);
        if fast {
            for y in (top.max(8)..bottom).step_by(8) {
                for x in (0..width).step_by(8) {
                    let row = (y >> 3) * self.w4 + (x >> 2);
                    let (bs0, bs1) = (self.bs_hor.get(row), self.bs_hor.get(row + 1));
                    if bs0 == 0 && bs1 == 0 {
                        continue;
                    }
                    let segs = [
                        self.segment((x, y - 1), (x, y), bs0, &memo),
                        self.segment((x + 4, y - 1), (x + 4, y), bs1, &memo),
                    ];
                    if segs.iter().all(|s| s.beta == 0) {
                        continue;
                    }
                    let (p, q): (&mut [P], &mut [P]) = if y > top {
                        band[0].split_at_mut((y - top) * width)
                    } else if let Some(above) = above.as_deref_mut() {
                        (&mut above[0][..], &mut band[0][..])
                    } else {
                        continue;
                    };
                    if let (Some(p), Some(q)) = (P::bytes_mut(&mut *p), P::bytes_mut(&mut *q)) {
                        rows_across(p, q, width, x, segs, deblock::luma8);
                    } else if let (Some(p), Some(q)) = (P::words_mut(p), P::words_mut(q)) {
                        rows_across(p, q, width, x, segs, |r, s| {
                            deblock::luma16(r, s, bit_depth)
                        });
                    }
                }
            }
        }
        let (ystep, xstep) = if fast { (16, 8) } else { (8, 4) };
        for y in (top.max(8).next_multiple_of(ystep)..bottom).step_by(ystep) {
            for x in (0..width).step_by(xstep) {
                let bs = self.bs_hor.get((y >> 3) * self.w4 + (x >> 2));
                if bs == 0 || (fast && !(bs == 2 && y % 16 == 0 && x % 8 == 0)) {
                    continue;
                }
                let query = EdgeQuery {
                    p: (x, y - 1),
                    q: (x, y),
                    bs,
                    memo: &memo,
                };
                let chroma = bs == 2 && y % 16 == 0 && x % 8 == 0;
                for c in usize::from(fast)..3 {
                    if c > 0 && !chroma {
                        break;
                    }
                    let st = if c == 0 {
                        self.luma_strength(&query)
                    } else {
                        self.chroma_strength(&query, offsets[c - 1])
                    };
                    let Some(st) = st else {
                        continue;
                    };
                    let (stride, x, y, top) = if c == 0 {
                        (width, x, y, top)
                    } else {
                        (cw, x / 2, y / 2, top / 2)
                    };
                    // Rows from `y` on in this band; the rows above either
                    // in this band or, on its top boundary, in the one above.
                    let (p, q) = if y > top {
                        band[c].split_at_mut((y - top) * stride)
                    } else if let Some(above) = above.as_deref_mut() {
                        (&mut above[c][..], &mut band[c][..])
                    } else {
                        continue;
                    };
                    let mut edge = Hor { p, q, x, stride };
                    if c == 0 {
                        luma_edge(&mut edge, st);
                    } else {
                        chroma_edge(&mut edge, st);
                    }
                }
            }
        }
    }

    /// Whether SAO of CTB `cur` may read samples of CTB `nb` (the slice and
    /// tile conditions of clause 8.7.3.2).
    fn sao_may_cross(&self, cur: usize, nb: usize) -> bool {
        if cur == nb {
            return true;
        }
        let layout = &self.layout;
        if !self.pps.loop_filter_across_tiles && layout.tile_of[cur] != layout.tile_of[nb] {
            return false;
        }
        if self.slice_addr(cur) == self.slice_addr(nb) {
            return true;
        }
        let slice = |rs: usize| {
            self.ctb_slice
                .get(rs)
                .and_then(|i| self.slices.get(usize::from(i.load(Relaxed))))
        };
        let (Some(a), Some(b)) = (slice(cur), slice(nb)) else {
            return false;
        };
        if layout.rs_to_ts[nb] < layout.rs_to_ts[cur] {
            a.header.loop_filter_across_slices
        } else {
            b.header.loop_filter_across_slices
        }
    }

    /// Whether any slice of the picture has SAO on.
    pub fn sao_on(&self) -> bool {
        self.sps.sao
            && self
                .slices
                .iter()
                .any(|s| s.header.sao_luma || s.header.sao_chroma)
    }

    /// The luma lines [`Self::sao_row`] makes final for CTB row `r`: those
    /// from [`HOLD`] lines above the row to `HOLD` lines above its end, or
    /// to the picture's end for the last row.
    pub fn sao_lines(&self, work: &Work<impl Pixel>, r: usize) -> (usize, usize) {
        let top = r << work.log2_ctb;
        let from = top.saturating_sub(HOLD);
        let to = if r + 1 >= work.count {
            self.height
        } else {
            (top + (1 << work.log2_ctb) - HOLD).min(self.height)
        };
        (from, to)
    }

    /// The final samples of luma lines [`Self::sao_lines`] of CTB row `r`
    /// into `dst` (those lines of each plane): the deblocked samples, with
    /// sample adaptive offset where it is on (clause 8.7.3). Once row `r`
    /// is deblocked they are all final, and the line past them too: the
    /// deblocking of the row below changes only its last three luma lines
    /// (one chroma line). So the filters finish a row on the thread that
    /// made it, and its lines reach other pictures a row sooner.
    pub fn sao_row<P: Pixel>(&self, work: &Work<P>, r: usize, mut dst: [&mut [P]; 3]) {
        let (from, to) = self.sao_lines(work, r);
        let prev = r.checked_sub(1).map(|a| lock(&work.rows[work.slot(a)]));
        let band = lock(&work.rows[work.slot(r)]);
        let top = r << work.log2_ctb;
        let width = self.width;
        // Component line `y` of the deblocked picture, in row `r` or the
        // row above; empty past them.
        let line = |c: usize, y: usize| -> &[P] {
            let (t, stride) = if c == 0 {
                (top, width)
            } else {
                (top / 2, width / 2)
            };
            if y >= t {
                band[c]
                    .get((y - t) * stride..(y - t + 1) * stride)
                    .unwrap_or(&[])
            } else {
                prev.as_ref().map_or(&[][..], |p| {
                    let above = p[c].len() / stride.max(1);
                    (t - y)
                        .checked_sub(1)
                        .and_then(|back| above.checked_sub(back + 1))
                        .and_then(|at| p[c].get(at * stride..(at + 1) * stride))
                        .unwrap_or(&[])
                })
            }
        };
        if !self.sao_on() {
            for (c, d) in dst.iter_mut().enumerate() {
                let sub = u32::from(c > 0);
                let pw = self.width >> sub;
                for (y, out) in (from >> sub..to >> sub).zip(d.chunks_exact_mut(pw.max(1))) {
                    let l = line(c, y);
                    let n = out.len().min(l.len());
                    out[..n].copy_from_slice(&l[..n]);
                }
            }
            return;
        }
        let ctb_w = self.layout.ctb_width as usize;
        // The samples of CTBs without SAO in a component are copied, in
        // runs of such CTBs; every other CTB writes all of its samples.
        let log2_ctb = work.log2_ctb;
        let ctb_rows = r.checked_sub(1).into_iter().chain([r]);
        for row in ctb_rows.clone() {
            for (c, d) in dst.iter_mut().enumerate() {
                let sub = u32::from(c > 0);
                let pw = self.width >> sub;
                let size = 1usize << (log2_ctb - sub);
                let base = from >> sub;
                let ya = (row * size).max(base);
                let yb = ((row + 1) * size).min(to >> sub);
                let mut rx = 0;
                while rx < ctb_w {
                    if self.sao.get(row * ctb_w + rx)[c].kind != 0 {
                        rx += 1;
                        continue;
                    }
                    let start = rx;
                    while rx < ctb_w && self.sao.get(row * ctb_w + rx)[c].kind == 0 {
                        rx += 1;
                    }
                    let (x0, x1) = (start * size, (rx * size).min(pw));
                    for y in ya..yb {
                        let at = (y - base) * pw;
                        d[at + x0..at + x1].copy_from_slice(&line(c, y)[x0..x1]);
                    }
                }
            }
        }
        // The rows of CTB row `r` lie end to end in its buffer, from its
        // first line: the kernels read them by stride.
        let cur: [&[P]; 3] = [&band[0][..], &band[1][..], &band[2][..]];
        for row in ctb_rows {
            for rx in 0..ctb_w {
                self.sao_ctb(row * ctb_w + rx, &line, &mut dst, (from, to), (cur, top));
            }
        }
    }

    /// SAO of the luma lines `from..to` of CTB `rs` (and the chroma lines
    /// they make) into `dst`, which holds those lines of each plane.
    fn sao_ctb<'l, P: Pixel>(
        &self,
        rs: usize,
        line: &impl Fn(usize, usize) -> &'l [P],
        dst: &mut [&mut [P]; 3],
        (from, to): (usize, usize),
        (cur, cur_top): ([&[P]; 3], usize),
    ) {
        let ctb_w = self.layout.ctb_width as usize;
        let ctb_h = self.layout.ctb_height as usize;
        let log2_ctb = u32::from(self.layout.log2_ctb);
        let bit_depth = u32::from(self.sps.bit_depth);
        let max = (1i32 << bit_depth) - 1;
        let keep_possible =
            self.pps.transquant_bypass || self.sps.pcm.is_some_and(|p| p.loop_filter_disabled);
        let (rx, ry) = (rs % ctb_w, rs / ctb_w);
        let params = self.sao.get(rs);
        if params.iter().all(|p| p.kind == 0) {
            return;
        }
        // May this CTB read each of its eight neighbours? Those above are
        // read only when the band holds the CTB's first line, those below
        // only when it holds its last (the row below may not be decoded
        // yet otherwise).
        let size = 1usize << log2_ctb;
        let (cy0, cy1) = (ry * size, ((ry + 1) * size).min(self.height));
        let reads = [from <= cy0, true, to >= cy1];
        let mut cross = [[false; 3]; 3];
        for (dy, row) in cross.iter_mut().enumerate() {
            for (dx, cell) in row.iter_mut().enumerate() {
                let nx = rx as isize + dx as isize - 1;
                let ny = ry as isize + dy as isize - 1;
                if !reads[dy] || nx < 0 || ny < 0 || nx as usize >= ctb_w || ny as usize >= ctb_h {
                    // Outside the picture or the band: never read (both
                    // paths stop at the picture's edges), so no restriction.
                    *cell = true;
                    continue;
                }
                *cell = self.sao_may_cross(rs, ny as usize * ctb_w + nx as usize);
            }
        }
        for (c, p) in params.iter().enumerate() {
            if p.kind == 0 {
                continue;
            }
            let sub = u32::from(c > 0);
            let pw = self.width >> sub;
            let ph = self.height >> sub;
            let size = 1usize << (log2_ctb - sub);
            let x0 = rx * size;
            let y0 = ry * size;
            let x1 = (x0 + size).min(pw);
            let y1 = (y0 + size).min(ph);
            // The lines of this CTB in the band, and the band's first.
            let (ya, yb) = (y0.max(from >> sub), y1.min(to >> sub));
            if ya >= yb {
                continue;
            }
            let base = from >> sub;
            let region = Region {
                x0,
                y0: ya,
                x1,
                y1: yb,
                pw,
                sub,
                keep_possible,
                top: base,
            };
            let out = &mut *dst[c];
            // Copy the region first unless the path taken writes every
            // sample of it.
            let edge_fast = p.kind == 2 && !keep_possible && cross.iter().flatten().all(|&ok| ok);
            if (p.kind == 1 && keep_possible) || (p.kind == 2 && !edge_fast) {
                for y in ya..yb {
                    let at = (y - base) * pw;
                    out[at + x0..at + x1].copy_from_slice(&line(c, y)[x0..x1]);
                }
            }
            if p.kind == 1 {
                let mut table = [0i32; 32];
                for k in 0..4 {
                    table[(k + usize::from(p.class)) & 31] = i32::from(p.offsets[k]);
                }
                let shift = bit_depth - 5;
                if !keep_possible {
                    for y in ya..yb {
                        let src = &line(c, y)[x0..x1];
                        let o = &mut out[(y - base) * pw + x0..(y - base) * pw + x1];
                        if let (Some(d), Some(s)) = (P::bytes_mut(o), P::bytes(src))
                            && sao::band_row(d, s, p.class, p.offsets.map(|v| v as i8))
                        {
                            continue;
                        }
                        let o = &mut out[(y - base) * pw + x0..(y - base) * pw + x1];
                        if let (Some(d), Some(s)) = (P::words_mut(o), P::words(src))
                            && sao::band_row16(d, s, p.class, p.offsets, bit_depth)
                        {
                            continue;
                        }
                        let o = &mut out[(y - base) * pw + x0..(y - base) * pw + x1];
                        for (o, &s) in o.iter_mut().zip(src) {
                            let v = s.i();
                            *o = P::of((v + table[(v >> shift) as usize & 31]).clamp(0, max));
                        }
                    }
                    continue;
                }
                self.sao_apply(out, &region, |x, y| {
                    let v = line(c, y)[x].i();
                    Some((v + table[(v >> shift) as usize & 31]).clamp(0, max))
                });
            } else {
                let (ax, ay, bx, by): (isize, isize, isize, isize) = match p.class {
                    0 => (-1, 0, 1, 0),
                    1 => (0, -1, 0, 1),
                    2 => (-1, -1, 1, 1),
                    _ => (1, -1, -1, 1),
                };
                let offsets = p.offsets;
                if edge_fast {
                    // Every neighbour may be read: only the picture
                    // edges limit the samples that change, and those
                    // edge samples are copied.
                    let left = usize::from(x0 == 0 && (ax < 0 || bx < 0));
                    let right = usize::from(x1 == pw && (ax > 0 || bx > 0));
                    let top = usize::from(y0 == 0 && (ay < 0 || by < 0));
                    let bottom = usize::from(y1 == ph && (ay > 0 || by > 0));
                    for y in (y0..y0 + top).chain(y1 - bottom..y1) {
                        if !(ya..yb).contains(&y) {
                            continue;
                        }
                        let at = (y - base) * pw;
                        out[at + x0..at + x1].copy_from_slice(&line(c, y)[x0..x1]);
                    }
                    let add = [
                        i32::from(offsets[0]),
                        i32::from(offsets[1]),
                        0,
                        i32::from(offsets[2]),
                        i32::from(offsets[3]),
                    ];
                    let (ra, rb) = ((y0 + top).max(ya), (y1 - bottom).min(yb));
                    let (xa, xb) = (x0 + left, x1 - right);
                    // 8-bit rows whose neighbours lie in this CTB row's
                    // buffer: one kernel call, by stride.
                    let (tc, held) = (cur_top >> sub, cur[c].len() / pw.max(1));
                    let (lo, hi) = (ay.min(by).min(0), ay.max(by).max(0));
                    let ia = ra.max(tc.saturating_add_signed(-lo));
                    let ib = rb.min((tc + held).saturating_add_signed(-hi));
                    let strided = ia < ib
                        && xb > xa
                        && match (
                            P::bytes_mut(&mut out[(ia - base) * pw + xa..]),
                            P::bytes(cur[c]),
                        ) {
                            (Some(d), Some(src)) => sao::edge_strided(
                                (d, pw),
                                (src, pw, (ia - tc) * pw + xa),
                                (ib - ia, xb - xa),
                                (ay * pw as isize + ax, by * pw as isize + bx),
                                offsets.map(|v| v as i8),
                            ),
                            _ => false,
                        };
                    for y in ra..rb {
                        if strided && (ia..ib).contains(&y) {
                            // At most one sample at each end, which the
                            // offsets leave as it is.
                            let (at, lv) = ((y - base) * pw, line(c, y));
                            if left == 1 {
                                out[at + x0] = lv[x0];
                            }
                            if right == 1 {
                                out[at + xb] = lv[xb];
                            }
                            continue;
                        }
                        let la = line(c, y.wrapping_add_signed(ay));
                        let lv = line(c, y);
                        let lb = line(c, y.wrapping_add_signed(by));
                        let o = &mut out[(y - base) * pw..(y - base + 1) * pw];
                        o[x0..xa].copy_from_slice(&lv[x0..xa]);
                        o[xb..x1].copy_from_slice(&lv[xb..x1]);
                        if let (Some(d), Some(v), Some(a), Some(b)) = (
                            P::bytes_mut(&mut o[xa..xb]),
                            P::bytes(&lv[xa..xb]),
                            P::bytes(&la[xa.wrapping_add_signed(ax)..xb.wrapping_add_signed(ax)]),
                            P::bytes(&lb[xa.wrapping_add_signed(bx)..xb.wrapping_add_signed(bx)]),
                        ) && sao::edge_row(d, v, (a, b), offsets.map(|v| v as i8))
                        {
                            continue;
                        }
                        if let (Some(d), Some(v), Some(a), Some(b)) = (
                            P::words_mut(&mut o[xa..xb]),
                            P::words(&lv[xa..xb]),
                            P::words(&la[xa.wrapping_add_signed(ax)..xb.wrapping_add_signed(ax)]),
                            P::words(&lb[xa.wrapping_add_signed(bx)..xb.wrapping_add_signed(bx)]),
                        ) && sao::edge_row16(d, v, (a, b), offsets, bit_depth)
                        {
                            continue;
                        }
                        for x in x0 + left..x1 - right {
                            let v = lv[x].i();
                            let a = la[x.wrapping_add_signed(ax)].i();
                            let b = lb[x.wrapping_add_signed(bx)].i();
                            let edge = (2 + (v - a).signum() + (v - b).signum()) as usize;
                            o[x] = P::of((v + add[edge]).clamp(0, max));
                        }
                    }
                    continue;
                }
                let cell = |v: isize, lo: usize, hi: usize| -> usize {
                    if v < lo as isize {
                        0
                    } else if v >= hi as isize {
                        2
                    } else {
                        1
                    }
                };
                self.sao_apply(out, &region, |x, y| {
                    let (xa, ya) = (x as isize + ax, y as isize + ay);
                    let (xb, yb) = (x as isize + bx, y as isize + by);
                    if xa < 0
                        || ya < 0
                        || xb < 0
                        || yb < 0
                        || xa as usize >= pw
                        || xb as usize >= pw
                        || ya as usize >= ph
                        || yb as usize >= ph
                    {
                        return None;
                    }
                    if !cross[cell(ya, y0, y1)][cell(xa, x0, x1)]
                        || !cross[cell(yb, y0, y1)][cell(xb, x0, x1)]
                    {
                        return None;
                    }
                    let v = line(c, y)[x].i();
                    let a = line(c, ya as usize)[xa as usize].i();
                    let b = line(c, yb as usize)[xb as usize].i();
                    let edge = 2 + (v - a).signum() + (v - b).signum();
                    let k = match edge {
                        0 => 0,
                        1 => 1,
                        3 => 2,
                        4 => 3,
                        _ => return None,
                    };
                    Some((v + i32::from(offsets[k])).clamp(0, max))
                });
            }
        }
    }

    fn sao_apply<P: Pixel>(
        &self,
        dst: &mut [P],
        r: &Region,
        f: impl Fn(usize, usize) -> Option<i32>,
    ) {
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                if r.keep_possible
                    && self.info.get(self.at4(x << r.sub, y << r.sub)) & FLAG_KEEP != 0
                {
                    continue;
                }
                if let Some(v) = f(x, y) {
                    dst[(y - r.top) * r.pw + x] = P::of(v);
                }
            }
        }
    }
}

/// The samples of one CTB in one component plane.
struct Region {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    pw: usize,
    sub: u32,
    keep_possible: bool,
    /// First component row of the band written.
    top: usize,
}

/// The luma kernel on eight lines across a vertical edge at column `x` of
/// band row `row` of `luma` (rows `width` apart): the samples `x - 4` to
/// `x + 3` of each line, filtered by `kernel` and put back.
#[inline(always)]
fn lines_across<T: Copy + Default>(
    luma: &mut [T],
    width: usize,
    (row, x): (usize, usize),
    segs: [deblock::Segment; 2],
    kernel: impl Fn(&mut [[T; 8]; 8], [deblock::Segment; 2]) -> bool,
) {
    let line = |l: usize| (row + l) * width + x - 4;
    let mut lines = [[T::default(); 8]; 8];
    for (l, d) in lines.iter_mut().enumerate() {
        if let Some(r) = luma.get(line(l)..).and_then(|r| r.first_chunk::<8>()) {
            *d = *r;
        }
    }
    if !kernel(&mut lines, segs) {
        return;
    }
    for (l, r) in lines.iter().enumerate() {
        if let Some(d) = luma
            .get_mut(line(l)..)
            .and_then(|d| d.first_chunk_mut::<8>())
        {
            *d = *r;
        }
    }
}

/// The luma kernel on eight columns `x..x + 8` across a horizontal edge:
/// the last four rows of `p` and the first four of `q` (rows `width`
/// apart), filtered by `kernel`; the six rows it may change are put back.
#[inline(always)]
fn rows_across<T: Copy + Default>(
    p: &mut [T],
    q: &mut [T],
    width: usize,
    x: usize,
    segs: [deblock::Segment; 2],
    kernel: impl Fn(&mut [[T; 8]; 8], [deblock::Segment; 2]) -> bool,
) {
    let plen = p.len();
    let at = |i: usize| {
        if i >= 4 {
            (i - 4) * width + x
        } else {
            (plen + x).wrapping_sub((4 - i) * width)
        }
    };
    let mut rows = [[T::default(); 8]; 8];
    for (i, r) in rows.iter_mut().enumerate() {
        let side: &[T] = if i >= 4 { q } else { p };
        if let Some(s) = side.get(at(i)..).and_then(|s| s.first_chunk::<8>()) {
            *r = *s;
        }
    }
    if !kernel(&mut rows, segs) {
        return;
    }
    for (i, r) in rows.iter().enumerate().take(7).skip(1) {
        let side: &mut [T] = if i >= 4 { &mut *q } else { &mut *p };
        if let Some(s) = side.get_mut(at(i)..).and_then(|s| s.first_chunk_mut::<8>()) {
            *s = *r;
        }
    }
}
