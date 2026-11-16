//! A temporal denoiser of the source (libvpx's `noise-sensitivity`,
//! `vp9_denoiser.c`): each 16×16 block of luma is pulled toward the
//! previous denoised picture moved by the block's motion, sample by
//! sample by libvpx's steps, unless the two differ by more than noise
//! does; what comes out is what the encoder codes, and the next picture's
//! reference. Chroma is coded as it comes, as libvpx does.

use super::EncPixel;

/// The block side the denoiser works on.
const B: usize = 16;

/// A block whose motion-compensated mean squared difference is over this
/// (on the 8-bit scale) moved in a way the vector does not follow, or
/// changed: it is left as it is.
const MSE_COPY: u64 = 144;

/// The vector a block's search keeps over the zero vector only when its
/// SAD is under this share of the zero vector's (noise makes any vector
/// look a little better than none).
const ZERO_BIAS: (u64, u64) = (7, 8);

/// A vector at most this long (in samples, either way) is low motion,
/// which libvpx denoises harder.
const LOW_MOTION: i32 = 3;

/// The denoiser's state: the last denoised luma and each block's vector.
pub struct Denoiser<P> {
    prev: Vec<P>,
    mvs: Vec<(i32, i32)>,
    /// libvpx's `increase_denoising`: from `noise-sensitivity` 3.
    strong: bool,
}

impl<P: EncPixel> Denoiser<P> {
    /// A denoiser at libvpx's `noise-sensitivity` `level` (1 or more).
    pub fn new(level: u8) -> Self {
        Self {
            prev: Vec::new(),
            mvs: Vec::new(),
            strong: level >= 3,
        }
    }

    /// Forgets the pictures before (a key frame starts afresh).
    pub fn reset(&mut self) {
        self.prev.clear();
    }

    /// Denoises luma `y` (rows `stride` apart, `w`×`h` visible) of `bd`
    /// bits into a tight plane, which it keeps for the next picture.
    pub fn filter(&mut self, (y, stride): (&[P], usize), (w, h): (usize, usize), bd: u8) -> Vec<P> {
        let mut out: Vec<P> = Vec::with_capacity(w * h);
        for r in 0..h {
            out.extend_from_slice(&y[r * stride..r * stride + w]);
        }
        if self.prev.len() != w * h {
            self.prev.clone_from(&out);
            self.mvs = vec![(0, 0); w.div_ceil(B) * h.div_ceil(B)];
            return out;
        }
        let (bc, br) = (w.div_ceil(B), h.div_ceil(B));
        let shift = u32::from(bd - 8);
        let max = (1i32 << bd) - 1;
        for by in 0..br {
            for bx in 0..bc {
                let (x0, y0) = (bx * B, by * B);
                let (bw, bh) = (B.min(w - x0), B.min(h - y0));
                let mv = self.search(&out, (w, h), (x0, y0, bw, bh), (bx, by));
                self.mvs[by * bc + bx] = mv;
                let px = (x0 as i32 + mv.0) as usize;
                let py = (y0 as i32 + mv.1) as usize;
                // The block's motion-compensated difference.
                let sse = P::sse(&out[y0 * w + x0..], w, &self.prev[py * w + px..], w, bw, bh);
                if (sse >> (2 * shift)) > MSE_COPY * (bw * bh) as u64 {
                    continue;
                }
                let low = mv.0.abs() <= LOW_MOTION && mv.1.abs() <= LOW_MOTION;
                self.block(&mut out, w, (x0, y0, bw, bh), (px, py), low, shift, max);
            }
        }
        self.prev.clone_from(&out);
        out
    }

    /// The block's vector into the last denoised picture, whole samples:
    /// the best of the zero vector, the block's last one and its left and
    /// upper neighbours' this time, refined a sample and then two around.
    fn search(
        &self,
        cur: &[P],
        (w, h): (usize, usize),
        (x0, y0, bw, bh): (usize, usize, usize, usize),
        (bx, by): (usize, usize),
    ) -> (i32, i32) {
        let bc = w.div_ceil(B);
        let sad = |(dx, dy): (i32, i32)| -> Option<u64> {
            let (px, py) = (x0 as i32 + dx, y0 as i32 + dy);
            if px < 0 || py < 0 || px as usize + bw > w || py as usize + bh > h {
                return None;
            }
            let (px, py) = (px as usize, py as usize);
            Some(u64::from(P::sad(
                &cur[y0 * w + x0..],
                w,
                &self.prev[py * w + px..],
                w,
                bw,
                bh,
            )))
        };
        let zero = sad((0, 0)).unwrap_or(u64::MAX);
        let mut best = ((0, 0), zero);
        let mut cands = vec![self.mvs[by * bc + bx]];
        if bx > 0 {
            cands.push(self.mvs[by * bc + bx - 1]);
        }
        if by > 0 {
            cands.push(self.mvs[(by - 1) * bc + bx]);
        }
        for c in cands {
            if let Some(v) = sad(c)
                && v < best.1
            {
                best = (c, v);
            }
        }
        for step in [2, 1] {
            let center = best.0;
            for (dx, dy) in [(-step, 0), (step, 0), (0, -step), (0, step)] {
                let m = (center.0 + dx, center.1 + dy);
                if let Some(v) = sad(m)
                    && v < best.1
                {
                    best = (m, v);
                }
            }
        }
        if best.0 != (0, 0) && best.1 * ZERO_BIAS.1 >= zero * ZERO_BIAS.0 {
            (0, 0)
        } else {
            best.0
        }
    }

    /// libvpx's `vp9_denoiser_filter` on one block: each sample within
    /// 3 (4 strong) of the reference takes the reference's value, the
    /// others move toward it by 3, 4 or 6 (more at low motion); when the
    /// block as a whole moves by more than 2 (3 strong) a sample on
    /// average, the moves are taken back by up to 3, or, past that, the
    /// block is left as it was.
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        out: &mut [P],
        w: usize,
        (x0, y0, bw, bh): (usize, usize, usize, usize),
        (px, py): (usize, usize),
        low: bool,
        shift: u32,
        max: i32,
    ) {
        let strong = self.strong;
        // (libvpx's `shift_inc2`; its `shift_inc1` only moves a bound
        // under the threshold.)
        let inc2 = match (low, strong) {
            (true, true) => 2,
            (true, false) => 1,
            _ => 0,
        };
        let adj = [3 + inc2, 4 + inc2, 6 + inc2];
        let (d1, d2) = (adj[1] - adj[0], adj[2] - adj[1]);
        let thresh = 3 + i32::from(strong);
        let n = (bw * bh) as i64;
        let sum_thresh = n * if strong { 3 } else { 2 };
        let mut den = [0i32; B * B];
        let mut sum = 0i64;
        for r in 0..bh {
            // (A row in fixed lanes, branchless, so that it runs in vector
            // instructions: past the threshold the step is adj[0] to 7,
            // adj[1] to 15, adj[2] past.)
            let (mut sv, mut mv) = ([0i32; B], [0i32; B]);
            for (d, s) in sv.iter_mut().zip(&out[(y0 + r) * w + x0..][..bw]) {
                *d = s.get();
            }
            for (d, m) in mv.iter_mut().zip(&self.prev[(py + r) * w + px..][..bw]) {
                *d = m.get();
            }
            let mut row = [0i32; B];
            let mut rsum = [0i32; B];
            for c in 0..B {
                let (s, m) = (sv[c], mv[c]);
                // (Differences on the 8-bit scale.)
                let diff = (m - s) >> shift;
                let a = diff.abs();
                let k = adj[0] + i32::from(a >= 8) * d1 + i32::from(a >= 16) * d2;
                let take = a <= thresh;
                let step = diff.signum() * k;
                rsum[c] = if take { diff } else { step };
                row[c] = if take {
                    m
                } else {
                    (s + (step << shift)).clamp(0, max)
                };
            }
            sum += i64::from(rsum[..bw].iter().sum::<i32>());
            den[r * bw..][..bw].copy_from_slice(&row[..bw]);
        }
        if sum.abs() > sum_thresh {
            // Take some of it back: up to `delta` a sample toward the
            // source.
            let delta = ((sum.abs() - sum_thresh) / n + 1) as i32;
            if delta >= 4 {
                return;
            }
            for r in 0..bh {
                for c in 0..bw {
                    let s = out[(y0 + r) * w + x0 + c].get();
                    let m = self.prev[(py + r) * w + px + c].get();
                    let diff = (m - s) >> shift;
                    let k = diff.abs().min(delta);
                    let v = &mut den[r * bw + c];
                    if diff > 0 {
                        *v = (*v - (k << shift)).max(0);
                        sum -= i64::from(k);
                    } else if diff < 0 {
                        *v = (*v + (k << shift)).min(max);
                        sum += i64::from(k);
                    }
                }
            }
            if sum.abs() > sum_thresh {
                return;
            }
        }
        for r in 0..bh {
            for c in 0..bw {
                out[(y0 + r) * w + x0 + c] = P::clip(den[r * bw + c], max);
            }
        }
    }
}
