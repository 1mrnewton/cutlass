//! In-loop filters: the deblocking filter (clause 8.7.2) and sample
//! adaptive offset (clause 8.7.3), run over a whole decoded picture.

use std::sync::Mutex;
use std::sync::atomic::Ordering::Relaxed;

use cutlass_pool::Pool;

use crate::ctu::chroma_qp;
use crate::picture::{Band, FLAG_KEEP, Picture, lock, split_bands};

/// `tC′` of table 8-12, indexed by `Q`.
const TC: [i32; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

/// `β′` of table 8-12, indexed by `Q`.
const BETA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];

/// Four lines across one edge: `pos` is the first `q0` sample, `across`
/// the step from `p0` to `q0`, `along` the step to the next line.
#[derive(Clone, Copy)]
struct Edge {
    pos: usize,
    across: usize,
    along: usize,
}

/// Decisions of one edge segment.
#[derive(Clone, Copy)]
struct Strength {
    beta: i32,
    tc: i32,
    no_p: bool,
    no_q: bool,
    max: i32,
}

#[inline]
fn at(s: &[u16], e: Edge, line: usize, i: isize) -> i32 {
    let base = e.pos + line * e.along;
    let index = if i >= 0 {
        base + i as usize * e.across
    } else {
        base - (-i) as usize * e.across
    };
    i32::from(s[index])
}

#[inline]
fn put(s: &mut [u16], e: Edge, line: usize, i: isize, value: i32, max: i32) {
    let base = e.pos + line * e.along;
    let index = if i >= 0 {
        base + i as usize * e.across
    } else {
        base - (-i) as usize * e.across
    };
    s[index] = value.clamp(0, max) as u16;
}

/// Luma edge filtering of four lines (clauses 8.7.2.5.3 and 8.7.2.5.7).
fn luma_edge(s: &mut [u16], e: Edge, st: Strength) {
    let Strength {
        beta,
        tc,
        no_p,
        no_q,
        max,
    } = st;
    let dp = |s: &[u16], line: usize| {
        (at(s, e, line, -3) - 2 * at(s, e, line, -2) + at(s, e, line, -1)).abs()
    };
    let dq = |s: &[u16], line: usize| {
        (at(s, e, line, 2) - 2 * at(s, e, line, 1) + at(s, e, line, 0)).abs()
    };
    let dp0 = dp(s, 0);
    let dp3 = dp(s, 3);
    let dq0 = dq(s, 0);
    let dq3 = dq(s, 3);
    let d0 = dp0 + dq0;
    let d3 = dp3 + dq3;
    if d0 + d3 >= beta {
        return;
    }
    let strong_line = |s: &[u16], line: usize, d: i32| -> bool {
        2 * d < (beta >> 2)
            && (at(s, e, line, -4) - at(s, e, line, -1)).abs()
                + (at(s, e, line, 0) - at(s, e, line, 3)).abs()
                < (beta >> 3)
            && (at(s, e, line, -1) - at(s, e, line, 0)).abs() < ((5 * tc + 1) >> 1)
    };
    if strong_line(s, 0, d0) && strong_line(s, 3, d3) {
        let tc2 = 2 * tc;
        for line in 0..4 {
            let p3 = at(s, e, line, -4);
            let p2 = at(s, e, line, -3);
            let p1 = at(s, e, line, -2);
            let p0 = at(s, e, line, -1);
            let q0 = at(s, e, line, 0);
            let q1 = at(s, e, line, 1);
            let q2 = at(s, e, line, 2);
            let q3 = at(s, e, line, 3);
            if !no_p {
                let v0 = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - tc2, p0 + tc2);
                let v1 = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - tc2, p1 + tc2);
                let v2 = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - tc2, p2 + tc2);
                put(s, e, line, -1, v0, max);
                put(s, e, line, -2, v1, max);
                put(s, e, line, -3, v2, max);
            }
            if !no_q {
                let v0 = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - tc2, q0 + tc2);
                let v1 = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - tc2, q1 + tc2);
                let v2 = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - tc2, q2 + tc2);
                put(s, e, line, 0, v0, max);
                put(s, e, line, 1, v1, max);
                put(s, e, line, 2, v2, max);
            }
        }
        return;
    }
    let side = (beta + (beta >> 1)) >> 3;
    let filter_p1 = dp0 + dp3 < side;
    let filter_q1 = dq0 + dq3 < side;
    let half = tc >> 1;
    for line in 0..4 {
        let p2 = at(s, e, line, -3);
        let p1 = at(s, e, line, -2);
        let p0 = at(s, e, line, -1);
        let q0 = at(s, e, line, 0);
        let q1 = at(s, e, line, 1);
        let q2 = at(s, e, line, 2);
        let mut delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if delta.abs() >= tc * 10 {
            continue;
        }
        delta = delta.clamp(-tc, tc);
        if !no_p {
            put(s, e, line, -1, p0 + delta, max);
            if filter_p1 {
                let dp = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-half, half);
                put(s, e, line, -2, p1 + dp, max);
            }
        }
        if !no_q {
            put(s, e, line, 0, q0 - delta, max);
            if filter_q1 {
                let dq = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-half, half);
                put(s, e, line, 1, q1 + dq, max);
            }
        }
    }
}

/// Chroma edge filtering of four lines (clause 8.7.2.5.5).
fn chroma_edge(s: &mut [u16], e: Edge, st: Strength) {
    let tc = st.tc;
    if tc == 0 {
        return;
    }
    for line in 0..4 {
        let p1 = at(s, e, line, -2);
        let p0 = at(s, e, line, -1);
        let q0 = at(s, e, line, 0);
        let q1 = at(s, e, line, 1);
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        if !st.no_p {
            put(s, e, line, -1, p0 + delta, st.max);
        }
        if !st.no_q {
            put(s, e, line, 0, q0 - delta, st.max);
        }
    }
}

/// Inputs shared by every edge of one direction.
struct EdgeQuery {
    /// Luma position of `p0` and `q0`.
    p: (usize, usize),
    q: (usize, usize),
    bs: u8,
}

impl Picture {
    /// Average luma QP, offsets of the slice holding `q0`, and the
    /// do-not-touch flags of both sides.
    fn edge_terms(&self, query: &EdgeQuery) -> Option<(i32, i32, i32, bool, bool)> {
        let header = &self.slice_of(query.q.0, query.q.1)?.header;
        let ip = self.at4(query.p.0, query.p.1);
        let iq = self.at4(query.q.0, query.q.1);
        let qp_at = |at: usize| i32::from(self.qp.get(at) as i8);
        let qp = (qp_at(ip) + qp_at(iq) + 1) >> 1;
        Some((
            qp,
            header.beta_offset,
            header.tc_offset,
            self.info.get(ip) & FLAG_KEEP != 0,
            self.info.get(iq) & FLAG_KEEP != 0,
        ))
    }

    fn luma_strength(&self, query: &EdgeQuery) -> Option<Strength> {
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

    fn chroma_strength(&self, query: &EdgeQuery, offset: i32) -> Option<Strength> {
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

    /// The deblocking filter over the whole picture: every vertical edge,
    /// then every horizontal edge (clause 8.7.2). Bands of rows run on the
    /// pool's workers; no edge of one direction reaches another band.
    pub fn deblock(&self, planes: &mut [Vec<u16>; 3], pool: &Pool) {
        let (width, height) = (self.width, self.height);
        // Vertical edges change only their own rows: any 8-row bands do.
        let step = band_rows(height, pool);
        let mut tops: Vec<usize> = (0..height).step_by(step).collect();
        let bands = split_bands(planes, width, &tops, height);
        pool.slices(bands.len(), |i| {
            let mut band = lock(&bands[i]);
            self.deblock_vertical(&mut band, tops[i], tops.get(i + 1).map_or(height, |&t| t));
        });
        drop(bands);
        // A horizontal edge on row y reads rows y - 4 to y + 3: bands start
        // 4 rows past the 8 grid.
        tops = std::iter::once(0)
            .chain((step + 4..height).step_by(step))
            .collect();
        let bands = split_bands(planes, width, &tops, height);
        pool.slices(bands.len(), |i| {
            let mut band = lock(&bands[i]);
            self.deblock_horizontal(&mut band, tops[i], tops.get(i + 1).map_or(height, |&t| t));
        });
    }

    fn deblock_vertical(&self, band: &mut Band<'_>, top: usize, bottom: usize) {
        let width = self.width;
        let ver_stride = width / 8 + 1;
        let cw = width / 2;
        let offsets = [self.pps.cb_qp_offset, self.pps.cr_qp_offset];
        for y in (top..bottom).step_by(4) {
            for x in (8..width).step_by(8) {
                let bs = self.bs_ver.get((y >> 2) * ver_stride + (x >> 3));
                if bs == 0 {
                    continue;
                }
                let query = EdgeQuery {
                    p: (x - 1, y),
                    q: (x, y),
                    bs,
                };
                if let Some(st) = self.luma_strength(&query) {
                    let edge = Edge {
                        pos: (y - top) * width + x,
                        across: 1,
                        along: width,
                    };
                    luma_edge(band[0], edge, st);
                }
                if bs == 2 && x % 16 == 0 && y % 8 == 0 {
                    for (c, &offset) in offsets.iter().enumerate() {
                        if let Some(st) = self.chroma_strength(&query, offset) {
                            let edge = Edge {
                                pos: (y / 2 - top / 2) * cw + x / 2,
                                across: 1,
                                along: cw,
                            };
                            chroma_edge(band[c + 1], edge, st);
                        }
                    }
                }
            }
        }
    }

    fn deblock_horizontal(&self, band: &mut Band<'_>, top: usize, bottom: usize) {
        let width = self.width;
        let cw = width / 2;
        let offsets = [self.pps.cb_qp_offset, self.pps.cr_qp_offset];
        for y in (top.next_multiple_of(8).max(8)..bottom).step_by(8) {
            for x in (0..width).step_by(4) {
                let bs = self.bs_hor.get((y >> 3) * self.w4 + (x >> 2));
                if bs == 0 {
                    continue;
                }
                let query = EdgeQuery {
                    p: (x, y - 1),
                    q: (x, y),
                    bs,
                };
                if let Some(st) = self.luma_strength(&query) {
                    let edge = Edge {
                        pos: (y - top) * width + x,
                        across: width,
                        along: 1,
                    };
                    luma_edge(band[0], edge, st);
                }
                if bs == 2 && y % 16 == 0 && x % 8 == 0 {
                    for (c, &offset) in offsets.iter().enumerate() {
                        if let Some(st) = self.chroma_strength(&query, offset) {
                            let edge = Edge {
                                pos: (y / 2 - top / 2) * cw + x / 2,
                                across: cw,
                                along: 1,
                            };
                            chroma_edge(band[c + 1], edge, st);
                        }
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

    /// Sample adaptive offset over the whole picture (clause 8.7.3), one
    /// CTB row on each pool job. Every CTB reads the deblocked samples.
    pub fn sao(&self, planes: &mut [Vec<u16>; 3], pool: &Pool) {
        let any = self
            .slices
            .iter()
            .any(|s| s.header.sao_luma || s.header.sao_chroma);
        if !any || !self.sps.sao {
            return;
        }
        let deblocked = copy_planes(planes, pool);
        let ctb_w = self.layout.ctb_width as usize;
        let ctb_h = self.layout.ctb_height as usize;
        let log2_ctb = u32::from(self.layout.log2_ctb);
        let tops: Vec<usize> = (0..ctb_h).map(|ry| ry << log2_ctb).collect();
        let bands = split_bands(planes, self.width, &tops, self.height);
        pool.slices(ctb_h, |ry| {
            let mut band = lock(&bands[ry]);
            for rx in 0..ctb_w {
                self.sao_ctb(ry * ctb_w + rx, &deblocked, &mut band);
            }
        });
    }

    /// SAO of CTB `rs` into `band`, the CTB row that holds it.
    fn sao_ctb(&self, rs: usize, deblocked: &[Vec<u16>; 3], band: &mut Band<'_>) {
        let ctb_w = self.layout.ctb_width as usize;
        let ctb_h = self.layout.ctb_height as usize;
        let log2_ctb = u32::from(self.layout.log2_ctb);
        let bit_depth = u32::from(self.sps.bit_depth);
        let max = (1i32 << bit_depth) - 1;
        let keep_possible =
            self.pps.transquant_bypass || self.sps.pcm.is_some_and(|p| p.loop_filter_disabled);
        let (rx, ry) = (rs % ctb_w, rs / ctb_w);
        let params = self.sao.get(rs);
        // May this CTB read each of its eight neighbours?
        let mut cross = [[false; 3]; 3];
        for (dy, row) in cross.iter_mut().enumerate() {
            for (dx, cell) in row.iter_mut().enumerate() {
                let nx = rx as isize + dx as isize - 1;
                let ny = ry as isize + dy as isize - 1;
                if nx < 0 || ny < 0 || nx as usize >= ctb_w || ny as usize >= ctb_h {
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
            let region = Region {
                x0,
                y0,
                x1,
                y1,
                pw,
                sub,
                keep_possible,
                top: y0,
            };
            let src = &deblocked[c];
            if p.kind == 1 {
                let mut table = [0i32; 32];
                for k in 0..4 {
                    table[(k + usize::from(p.class)) & 31] = i32::from(p.offsets[k]);
                }
                let shift = bit_depth - 5;
                self.sao_apply(band[c], &region, |x, y| {
                    let v = i32::from(src[y * pw + x]);
                    Some((v + table[(v >> shift) as usize]).clamp(0, max))
                });
            } else {
                let (ax, ay, bx, by): (isize, isize, isize, isize) = match p.class {
                    0 => (-1, 0, 1, 0),
                    1 => (0, -1, 0, 1),
                    2 => (-1, -1, 1, 1),
                    _ => (1, -1, -1, 1),
                };
                let offsets = p.offsets;
                if !keep_possible && cross.iter().flatten().all(|&ok| ok) {
                    // Every neighbour may be read: only the picture
                    // edges limit the samples that change.
                    let left = usize::from(x0 == 0 && (ax < 0 || bx < 0));
                    let right = usize::from(x1 == pw && (ax > 0 || bx > 0));
                    let top = usize::from(y0 == 0 && (ay < 0 || by < 0));
                    let bottom = usize::from(y1 == ph && (ay > 0 || by > 0));
                    let add = [
                        i32::from(offsets[0]),
                        i32::from(offsets[1]),
                        0,
                        i32::from(offsets[2]),
                        i32::from(offsets[3]),
                    ];
                    let da = ay * pw as isize + ax;
                    let db = by * pw as isize + bx;
                    let dst = &mut *band[c];
                    for y in y0 + top..y1 - bottom {
                        for x in x0 + left..x1 - right {
                            let at = y * pw + x;
                            let out = (y - y0) * pw + x;
                            let v = i32::from(src[at]);
                            let a = i32::from(src[(at as isize + da) as usize]);
                            let b = i32::from(src[(at as isize + db) as usize]);
                            let edge = (2 + (v - a).signum() + (v - b).signum()) as usize;
                            let delta = add[edge];
                            if delta != 0 {
                                dst[out] = (v + delta).clamp(0, max) as u16;
                            }
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
                self.sao_apply(band[c], &region, |x, y| {
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
                    let v = i32::from(src[y * pw + x]);
                    let a = i32::from(src[ya as usize * pw + xa as usize]);
                    let b = i32::from(src[yb as usize * pw + xb as usize]);
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

    fn sao_apply(&self, dst: &mut [u16], r: &Region, f: impl Fn(usize, usize) -> Option<i32>) {
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                if r.keep_possible
                    && self.info.get(self.at4(x << r.sub, y << r.sub)) & FLAG_KEEP != 0
                {
                    continue;
                }
                if let Some(v) = f(x, y) {
                    dst[(y - r.top) * r.pw + x] = v as u16;
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

/// Rows per band: the picture split about four times per worker, on the 8 grid.
fn band_rows(height: usize, pool: &Pool) -> usize {
    let bands = pool.workers() * 4;
    height.div_ceil(bands).next_multiple_of(8).max(8)
}

/// A copy of the planes, made by the pool's workers.
fn copy_planes(planes: &[Vec<u16>; 3], pool: &Pool) -> [Vec<u16>; 3] {
    const CHUNK: usize = 1 << 16;
    let mut out = [
        vec![0; planes[0].len()],
        vec![0; planes[1].len()],
        vec![0; planes[2].len()],
    ];
    let mut jobs = Vec::new();
    for (dst, src) in out.iter_mut().zip(planes) {
        for (d, s) in dst.chunks_mut(CHUNK).zip(src.chunks(CHUNK)) {
            jobs.push(Mutex::new((d, s)));
        }
    }
    pool.slices(jobs.len(), |i| {
        let mut job = lock(&jobs[i]);
        let (d, s) = &mut *job;
        d.copy_from_slice(s);
    });
    drop(jobs);
    out
}
