//! Temporal dependency model (TPL) of a group of frames.
//!
//! Every 16x16 luma block of the frames still to be coded gets an intra cost
//! (Hadamard SATD of the residual of a simple intra prediction from source
//! samples) and an inter cost (SATD after motion search in each of its
//! references), both measured on the luma plane at half resolution. Walking the frames backwards in coding order, the part of a
//! block's cost that its reference explains (`1 - inter / intra`) flows into
//! the blocks it predicts from, together with what already flowed into the
//! block itself. A frame (or a region of it) through which much of the group
//! is predicted then has a large `(intra + flow) / intra` ratio, its
//! *importance*: spending bits on it pays off in every frame that follows.

use crate::dec::frame::Picture;
use crate::dsp::Pixel;

use super::me::{MeParams, MeScratch, downscale_2x, downscale_2x_dims, motion_search, satd};

/// Analysis block size (luma samples).
pub const BLOCK: usize = 16;
/// The block size on the half-resolution plane.
const HB: usize = BLOCK / 2;

/// A luma plane at half resolution, padded by edge replication to whole
/// blocks plus one block.
struct Half<P> {
    data: Vec<P>,
    stride: usize,
    w: usize,
    h: usize,
}

impl<P: Pixel> Half<P> {
    fn new(pic: &Picture<P>) -> Self {
        let p = &pic.planes[0];
        let (w, h) = downscale_2x_dims(p.w, p.h);
        let stride = w.div_ceil(HB) * HB + HB;
        let rows = h.div_ceil(HB) * HB + HB;
        let mut data = vec![P::default(); stride * rows];
        downscale_2x(&mut data, stride, &p.data, p.stride, p.w, p.h);
        for r in 0..rows {
            let sr = r.min(h - 1);
            if sr != r {
                data.copy_within(sr * stride..sr * stride + w, r * stride);
            }
            let last = data[r * stride + w - 1];
            data[r * stride + w..(r + 1) * stride].fill(last);
        }
        Self { data, stride, w, h }
    }
}

/// One frame of the group: its source and the frames (indices into the
/// group) it is predicted from, all coded before it.
pub struct TplFrame<'a, P: Pixel> {
    pub pic: &'a Picture<P>,
    /// Display index (motion search ranges and candidate scaling use the
    /// display distance).
    pub disp: i64,
    pub refs: Vec<usize>,
    /// False for frames that are already coded (references only).
    pub analyze: bool,
}

/// Importance per frame and per 16x16 block.
pub struct TplStats {
    pub cols: usize,
    pub rows: usize,
    /// `intra` cost of each block of each frame (0 for frames not analyzed).
    pub intra: Vec<Vec<f64>>,
    /// Cost flowing into each block from the frames that predict from it.
    pub flow: Vec<Vec<f64>>,
    /// Sum over each frame's blocks of the better of the intra and inter cost.
    pub best_cost: Vec<f64>,
}

impl TplStats {
    /// `(intra + flow) / intra` of a frame (1 when nothing depends on it).
    pub fn frame_importance(&self, f: usize) -> f64 {
        let i: f64 = self.intra[f].iter().sum();
        let fl: f64 = self.flow[f].iter().sum();
        if i <= 0.0 { 1.0 } else { (i + fl) / i }
    }

    /// Share of a frame's intra cost left after inter prediction (1 when
    /// it predicts nothing from its references).
    pub fn inter_ratio(&self, f: usize) -> f64 {
        let i: f64 = self.intra[f].iter().sum();
        if i <= 0.0 { 1.0 } else { self.best_cost[f] / i }
    }

    /// Importance of the blocks overlapping a `size`-sample square at
    /// (`x`, `y`) of frame `f`.
    pub fn region_importance(&self, f: usize, x: usize, y: usize, size: usize) -> f64 {
        let (mut i, mut fl) = (0.0, 0.0);
        let r1 = (y + size).div_ceil(BLOCK).min(self.rows);
        let c1 = (x + size).div_ceil(BLOCK).min(self.cols);
        for r in y / BLOCK..r1 {
            for c in x / BLOCK..c1 {
                i += self.intra[f][r * self.cols + c];
                fl += self.flow[f][r * self.cols + c];
            }
        }
        if i <= 0.0 { 1.0 } else { (i + fl) / i }
    }
}

/// Intra SATD of the block at (`x`, `y`): the best of DC, vertical,
/// horizontal and Paeth predictions from neighbouring source samples.
fn intra_cost<P: Pixel>(p: &Half<P>, x: usize, y: usize, bd: u32, pred: &mut [P]) -> u32 {
    let s = p.stride;
    let src = &p.data[y * s + x..];
    let mut top = [0i32; HB];
    let mut left = [0i32; HB];
    let mid = 1i32 << (bd - 1);
    let (have_t, have_l) = (y > 0, x > 0);
    for i in 0..HB {
        top[i] = if have_t {
            p.data[(y - 1) * s + x + i].to_i32()
        } else if have_l {
            p.data[y * s + x - 1].to_i32()
        } else {
            mid - 1
        };
        left[i] = if have_l {
            p.data[(y + i) * s + x - 1].to_i32()
        } else if have_t {
            top[0]
        } else {
            mid + 1
        };
    }
    let tl = if have_t && have_l {
        p.data[(y - 1) * s + x - 1].to_i32()
    } else {
        mid
    };
    let dc = {
        let n = (HB * 2) as i32;
        (top.iter().sum::<i32>() + left.iter().sum::<i32>() + n / 2) / n
    };
    let mut best = u32::MAX;
    for mode in 0..4 {
        for i in 0..HB {
            for j in 0..HB {
                let v = match mode {
                    0 => dc,
                    1 => top[j],
                    2 => left[i],
                    _ => {
                        let base = top[j] + left[i] - tl;
                        let (pl, pt, ptl) = (
                            (base - left[i]).abs(),
                            (base - top[j]).abs(),
                            (base - tl).abs(),
                        );
                        if pl <= pt && pl <= ptl {
                            left[i]
                        } else if pt <= ptl {
                            top[j]
                        } else {
                            tl
                        }
                    }
                };
                pred[i * HB + j] = P::from_i32(v);
            }
        }
        best = best.min(satd(src, s, pred, HB, HB, HB));
    }
    best
}

/// Runs the model over `frames`; `order` lists the frames to analyze in
/// coding order.
pub fn analyze<P: Pixel>(frames: &[TplFrame<'_, P>], order: &[usize], bd: u32) -> TplStats {
    let (w, h) = frames
        .first()
        .map_or((0, 0), |f| (f.pic.planes[0].w, f.pic.planes[0].h));
    let cols = w.div_ceil(BLOCK);
    let rows = h.div_ceil(BLOCK);
    let n = frames.len();
    let mut stats = TplStats {
        cols,
        rows,
        intra: vec![vec![0.0; cols * rows]; n],
        flow: vec![vec![0.0; cols * rows]; n],
        best_cost: vec![0.0; n],
    };
    // Per analyzed frame and block: inter cost, reference and motion (1/8 samples).
    let mut best: Vec<Vec<(f64, usize, i32, i32)>> = vec![Vec::new(); n];
    let mut scratch = MeScratch::<P>::default();
    let mut pred = vec![P::default(); HB * HB];
    let half: Vec<Option<Half<P>>> = frames
        .iter()
        .map(|f| (f.pic.planes[0].w > 1 && f.pic.planes[0].h > 1).then(|| Half::new(f.pic)))
        .collect();
    let mut mvs: Vec<(i32, i32)> = vec![(0, 0); cols * rows];
    for &f in order {
        let fr = &frames[f];
        if !fr.analyze {
            continue;
        }
        let Some(y) = half[f].as_ref() else {
            continue;
        };
        let mut fb = vec![(f64::MAX, usize::MAX, 0, 0); cols * rows];
        for br in 0..rows {
            for bc in 0..cols {
                let (x0, y0) = (bc * HB, br * HB);
                let ic = f64::from(intra_cost(y, x0, y0, bd, &mut pred)).max(1.0);
                stats.intra[f][br * cols + bc] = ic;
                fb[br * cols + bc].0 = ic;
            }
        }
        for &r in &fr.refs {
            let Some(rp) = half[r].as_ref() else {
                continue;
            };
            let dist = (fr.disp - frames[r].disp).unsigned_abs().max(1) as i32;
            let params = MeParams {
                max_range: (2 * dist).clamp(4, 16),
                subpel: 1,
                border: 0,
                ..MeParams::default()
            };
            let cost = |r: i32, c: i32| -> u32 { (r.unsigned_abs() + c.unsigned_abs()) >> 3 };
            for br in 0..rows {
                for bc in 0..cols {
                    let (x0, y0) = (bc * HB, br * HB);
                    let mut cands = [(0i32, 0i32); 3];
                    let mut nc = 1;
                    if bc > 0 {
                        cands[nc] = mvs[br * cols + bc - 1];
                        nc += 1;
                    }
                    if br > 0 {
                        cands[nc] = mvs[(br - 1) * cols + bc];
                        nc += 1;
                    }
                    let res = motion_search(
                        &mut scratch,
                        &y.data,
                        y.stride,
                        x0,
                        y0,
                        HB,
                        HB,
                        &rp.data,
                        rp.stride,
                        rp.w,
                        rp.h,
                        &cands[..nc],
                        &params,
                        &cost,
                        bd,
                    );
                    mvs[br * cols + bc] = res.mv;
                    let ic = f64::from(res.dist);
                    let b = &mut fb[br * cols + bc];
                    if ic < b.0 {
                        *b = (ic, r, res.mv.0, res.mv.1);
                    }
                }
            }
        }
        stats.best_cost[f] = fb.iter().map(|b| b.0).sum();
        best[f] = fb;
    }
    // Propagate backwards in coding order.
    for &f in order.iter().rev() {
        if !frames[f].analyze {
            continue;
        }
        for br in 0..rows {
            for bc in 0..cols {
                let k = br * cols + bc;
                let (inter, r, mvr, mvc) = best[f][k];
                if r == usize::MAX {
                    continue;
                }
                let intra = stats.intra[f][k];
                let frac = (1.0 - inter / intra).clamp(0.0, 1.0);
                let amount = (intra + stats.flow[f][k]) * frac;
                if amount <= 0.0 {
                    continue;
                }
                // Spread over the blocks the displaced block overlaps.
                let py = (br * HB) as i64 * 8 + i64::from(mvr);
                let px = (bc * HB) as i64 * 8 + i64::from(mvc);
                let (py, px) = ((py + 4) >> 3, (px + 4) >> 3);
                let b = HB as i64;
                let (gr, gc) = (py.div_euclid(b), px.div_euclid(b));
                let (oy, ox) = (py.rem_euclid(b), px.rem_euclid(b));
                for (dr, hgt) in [(0, b - oy), (1, oy)] {
                    for (dc, wid) in [(0, b - ox), (1, ox)] {
                        let (rr, cc) = (gr + dr, gc + dc);
                        if hgt == 0
                            || wid == 0
                            || rr < 0
                            || cc < 0
                            || rr >= rows as i64
                            || cc >= cols as i64
                        {
                            continue;
                        }
                        let share = (hgt * wid) as f64 / (b * b) as f64;
                        stats.flow[r][rr as usize * cols + cc as usize] += amount * share;
                    }
                }
            }
        }
    }
    stats
}
